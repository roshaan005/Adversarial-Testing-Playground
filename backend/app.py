from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import io
import urllib.request
import numpy as np

# =========================================
# Setup Flask
# =========================================
app = Flask(__name__)
CORS(app)  # allow frontend requests

# =========================================
# Device Setup
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================
# Load Models
# =========================================
models_dict = {}

# ResNet18
resnet_weights = ResNet18_Weights.DEFAULT
resnet_model = resnet18(weights=resnet_weights).to(device)
resnet_model.eval()
models_dict["resnet18"] = resnet_model

# MobileNetV2
mobilenet_weights = MobileNet_V2_Weights.DEFAULT
mobilenet_model = mobilenet_v2(weights=mobilenet_weights).to(device)
mobilenet_model.eval()
models_dict["mobilenet_v2"] = mobilenet_model

# =========================================
# Load ImageNet labels
# =========================================
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# =========================================
# NumPy-based preprocessing (replaces torchvision.transforms)
# =========================================
# ImageNet normalization constants (same as torchvision)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def center_crop(img: Image.Image, size: int):
    w, h = img.size
    new_w = new_h = size
    left = int(round((w - new_w) / 2.0))
    top = int(round((h - new_h) / 2.0))
    return img.crop((left, top, left + new_w, top + new_h))

def preprocess_image_numpy(pil_image: Image.Image, resize=256, crop_size=224):
    """
    Resize -> center crop -> convert to numpy -> normalize -> convert to torch tensor
    All intermediate steps use numpy operations.
    Returns a torch tensor of shape (1, 3, 224, 224) on the selected device.
    """
    # Resize shortest side to 256 while maintaining aspect ratio (like torchvision.Resize(256))
    w, h = pil_image.size
    if w < h:
        new_w = resize
        new_h = int(h * (resize / w))
    else:
        new_h = resize
        new_w = int(w * (resize / h))
    pil_resized = pil_image.resize((new_w, new_h), resample=Image.BILINEAR)

    # Center crop to 224x224
    pil_cropped = center_crop(pil_resized, crop_size)

    # Convert to numpy array (H, W, C) in uint8
    np_img = np.asarray(pil_cropped).astype(np.float32) / 255.0  # scale to [0,1]

    # If image has alpha or unexpected channels, handle gracefully
    if np_img.ndim == 2:  # grayscale -> replicate channels
        np_img = np.stack([np_img] * 3, axis=-1)
    elif np_img.shape[2] == 4:  # RGBA -> drop alpha
        np_img = np_img[:, :, :3]

    # Normalize (H, W, C) with ImageNet mean/std
    np_img = (np_img - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to (C, H, W)
    np_img = np.transpose(np_img, (2, 0, 1))

    # Convert to torch tensor and add batch dim
    tensor = torch.from_numpy(np_img).unsqueeze(0).to(device)

    return tensor

# =========================================
# Helper Functions
# =========================================
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top1_prob, top1_catid = torch.topk(probs, 1)
    return labels[top1_catid.item()], float(top1_prob.item())

def fgsm_attack(image, epsilon, data_grad):
    # image: tensor with requires_grad result, data_grad: gradient wrt image
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # clamp in normalized space; using same bounds that torchvision normalization can produce
    perturbed_image = torch.clamp(perturbed_image, -2.1179, 2.64)
    return perturbed_image

def pgd_attack(model, image, label, epsilon=0.1, alpha=0.01, iters=10):
    ori_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True

    for _ in range(iters):
        output = model(perturbed_image)
        loss = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1),
            label
        )
        model.zero_grad()
        loss.backward()

        grad_sign = perturbed_image.grad.data.sign()
        perturbed_image = perturbed_image + alpha * grad_sign

        # Project perturbation
        eta = torch.clamp(perturbed_image - ori_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(
            ori_image + eta, min=-2.1179, max=2.64
        ).detach_()
        perturbed_image.requires_grad = True

    return perturbed_image

# =========================================
# API Endpoints
# =========================================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Adversarial Testing Backend is running"})

@app.route("/predict", methods=["POST"])
def run_attack():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image file is required under the 'image' field."}), 400

        file = request.files["image"]
        attack_type = request.form.get("attack", "fgsm")
        epsilon = float(request.form.get("epsilon", 0.1))
        alpha = float(request.form.get("alpha", 0.01))
        iters = int(request.form.get("iterations", 10))
        model_choice = request.form.get("model", "resnet18")  # default to resnet18

        if model_choice not in models_dict:
            return jsonify({"error": f"Unsupported model: {model_choice}"}), 400

        model = models_dict[model_choice]

        # Load image with PIL
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Preprocess using NumPy-based pipeline
        input_tensor = preprocess_image_numpy(img)  # already on device

        # Original prediction
        orig_label, orig_conf = predict(model, input_tensor)

        # Run chosen attack
        if attack_type == "fgsm":
            input_tensor.requires_grad = True
            output = model(input_tensor)
            label_index = torch.argmax(output, 1)
            loss = torch.nn.functional.nll_loss(
                torch.nn.functional.log_softmax(output, dim=1), label_index
            )
            model.zero_grad()
            loss.backward()
            data_grad = input_tensor.grad.data
            perturbed_data = fgsm_attack(input_tensor, epsilon, data_grad)

        elif attack_type == "pgd":
            label_index = torch.argmax(model(input_tensor), 1)
            perturbed_data = pgd_attack(model, input_tensor, label_index, epsilon, alpha, iters)
        else:
            return jsonify({"error": "Unsupported attack"}), 400

        # Adversarial prediction
        adv_label, adv_conf = predict(model, perturbed_data)

        return jsonify({
            "model": model_choice,
            "original": {"label": orig_label, "confidence": round(orig_conf * 100, 2)},
            "adversarial": {"label": adv_label, "confidence": round(adv_conf * 100, 2)},
            "attack": attack_type,
            "epsilon": epsilon,
            "alpha": alpha,
            "iterations": iters
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# Run server
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
