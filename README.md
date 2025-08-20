# Adversarial Testing Playground

A comprehensive web-based playground for testing and analyzing adversarial attacks on machine learning models. This project demonstrates how deep learning models can be fooled by carefully crafted perturbations and provides an interactive interface for researchers and developers to experiment with different attack strategies.

##  What This Project Does

The Adversarial Testing Playground is designed to help researchers, developers, and students understand how vulnerable machine learning models are to adversarial attacks. It provides:

- **Real-time Adversarial Attack Generation**: Upload images and generate adversarial examples using state-of-the-art attack algorithms
- **Model Vulnerability Assessment**: Test how well different models resist various types of attacks
- **Interactive Parameter Tuning**: Experiment with attack parameters to understand their impact
- **Visual Comparison**: Side-by-side comparison of original vs. adversarial images with confidence scores
- **Educational Tool**: Learn about adversarial machine learning through hands-on experimentation

##  Current Features

- **Multiple Attack Methods**: 
  - **FGSM (Fast Gradient Sign Method)**: Single-step attack for quick testing
  - **PGD (Projected Gradient Descent)**: Multi-step iterative attack for stronger perturbations
- **Model Support**: 
  - **ResNet-18**: Deep residual network with 18 layers
  - **MobileNet V2**: Lightweight mobile-optimized network
- **Real-time Results**: Instant comparison of original vs. adversarial predictions
- **Interactive UI**: Modern, responsive interface with real-time parameter adjustment
- **Parameter Control**: Adjust epsilon (perturbation strength), learning rate, and iterations
   <img width="710" height="609" alt="Screenshot 2025-08-20 at 3 08 23 PM" src="https://github.com/user-attachments/assets/ff851ee8-1acc-4f45-8b4c-63ed1291f103" />
  <img width="703" height="616" alt="Screenshot 2025-08-20 at 3 09 09 PM" src="https://github.com/user-attachments/assets/d239bcdf-cea6-4d2d-b28b-b265c58e3aaa" />
<img width="716" height="700" alt="Screenshot 2025-08-20 at 3 10 01 PM" src="https://github.com/user-attachments/assets/51b2dfb6-17fe-420e-90cd-a53130459639" />
<img width="717" height="699" alt="Screenshot 2025-08-20 at 3 10 26 PM" src="https://github.com/user-attachments/assets/545d0a42-f2c3-4be4-951f-3f89d1ec1a9e" />



## Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Adversarial-Testing-Playground
   ```

2. **Install all dependencies**
   ```bash
   npm run install:all
   ```

   This will install both backend Python dependencies and frontend Node.js dependencies.

   > **Note**: If you encounter SSL certificate errors during model download (common on macOS), run:
   > ```bash
   > /Applications/Python\ 3.x/Install\ Certificates.command
   > ```
   > Replace `3.x` with your Python version.

## Running the Application

### Development Mode (Recommended)

Run both backend and frontend simultaneously:
```bash
npm run dev
```

This will start:
- Backend server on `http://localhost:5000`
- Frontend development server on `http://localhost:3000`

### Running Separately

**Backend only:**
```bash
npm run dev:backend
```

**Frontend only:**
```bash
npm run dev:frontend
```

## 🎯 Use Cases & Applications

### Research & Academia
- **Adversarial Machine Learning Research**: Study model vulnerabilities and defense mechanisms
- **Model Robustness Evaluation**: Test how well models resist various attack strategies
- **Educational Tool**: Teach students about ML security and adversarial examples
- **Benchmarking**: Compare different models' resistance to attacks

### Industry & Production
- **Security Auditing**: Evaluate ML systems before deployment
- **Model Validation**: Test model robustness as part of quality assurance
- **Red Team Testing**: Simulate adversarial attacks to improve defenses
- **Compliance**: Meet security requirements for ML systems

### Development & Testing
- **Model Development**: Test models during training and fine-tuning
- **Defense Mechanism Development**: Validate adversarial training and other defenses
- **Performance Testing**: Benchmark attack generation speed and efficiency

## 📖 Usage

1. **Upload an Image**: Click the upload area or drag and drop an image file
2. **Configure Attack Parameters**:
   - Select attack method (FGSM, PGD)
   - Adjust epsilon (perturbation strength)
   - Set iterations (for iterative attacks)
   - Configure learning rate
3. **Run Attack**: Click "Launch Attack" to start the adversarial attack
4. **View Results**: Compare original vs adversarial predictions

## API Endpoints

### POST `/predict`
Runs an adversarial attack on an uploaded image.

**Request:**
- `image`: Image file (multipart/form-data)
- `attack`: Attack type ("fgsm", "pgd")
- `model`: Target model ("resnet18", "mobilenet_v2")
- `epsilon`: Perturbation strength (float)
- `alpha`: Learning rate (float, optional)
- `iterations`: Number of iterations (int, optional)

**Response:**
```json
{
  "model": "resnet18",
  "original": {
    "label": "golden retriever",
    "confidence": 95.2
  },
  "adversarial": {
    "label": "labrador retriever", 
    "confidence": 87.3
  },
  "attack": "fgsm",
  "epsilon": 0.03,
  "alpha": 0.01,
  "iterations": 10
}
```

## 🏗️ Technical Architecture

### System Overview
The playground follows a client-server architecture where:
- **Frontend**: React-based UI for user interaction and result visualization
- **Backend**: Flask API server handling ML model inference and attack generation
- **Communication**: RESTful API with multipart form data for image uploads

### How It Works
1. **Image Upload**: User uploads an image through the web interface
2. **Model Selection**: User chooses a target model (ResNet-18 or MobileNet V2)
3. **Attack Configuration**: User sets attack parameters (epsilon, iterations, learning rate)
4. **Attack Execution**: Backend processes the image through the selected model and attack algorithm
5. **Result Analysis**: System compares original vs. adversarial predictions and displays results

### Attack Algorithms Explained
- **FGSM**: Calculates gradients of loss with respect to input, then adds perturbation in the direction of gradient sign
- **PGD**: Iterative version of FGSM with projection to ensure perturbations stay within epsilon ball

## 📁 Project Structure

```
Adversarial-Testing-Playground/
├── backend/
│   ├── app.py              # Flask backend server with attack implementations
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   └── playground.tsx  # Main playground component
│   │   └── lib/
│   │       └── api.ts      # API service functions
│   ├── package.json        # Frontend dependencies
│   └── ...
├── package.json            # Root package.json with scripts
├── start.sh               # Startup script for both servers
└── README.md
```

## 🚀 Future Enhancements & Roadmap

### Phase 1: Enhanced Model Support
- **Additional Pre-trained Models**:
  - VGG-16/19: Very deep convolutional networks
  - Inception v3: Inception architecture for better feature extraction
  - EfficientNet: Efficient scaling methods
  - Vision Transformer (ViT): Attention-based architectures
  - DenseNet: Dense connectivity patterns
- **Custom Model Upload**: Allow users to upload their own trained models
- **Model Comparison**: Side-by-side testing of multiple models on the same image

### Phase 2: Advanced Attack Methods
- **Carlini & Wagner (C&W)**: L2 norm attack for high-quality perturbations
- **DeepFool**: Minimal perturbation attack for research purposes
- **JSMA (Jacobian-based Saliency Map Attack)**: Pixel-wise targeted attacks
- **One-Pixel Attack**: Minimal pixel manipulation attacks
- **Universal Adversarial Perturbations**: Generate perturbations that fool multiple images
- **Physical World Attacks**: Simulate real-world adversarial examples

### Phase 3: Comprehensive Analytics & Metrics
- **Attack Success Metrics**:
  - Success rate across different epsilon values
  - Confidence drop analysis
  - Perturbation magnitude statistics
- **Image Quality Metrics**:
  - L2 and L∞ norm calculations
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Human perception studies
- **Model Robustness Analysis**:
  - Adversarial training evaluation
  - Defense mechanism testing
  - Transferability analysis between models
- **Performance Benchmarks**:
  - Attack generation time
  - Memory usage optimization
  - GPU vs CPU performance comparison

### Phase 4: Advanced Features
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Video Attacks**: Apply attacks to video streams
- **3D Model Attacks**: Extend to 3D point clouds and meshes
- **Text Adversarial Examples**: NLP model attacks
- **Audio Adversarial Examples**: Speech recognition model attacks
- **Collaborative Research**: Share attack results and findings

### Phase 5: Research & Educational Tools
- **Attack Visualization**: Interactive perturbation heatmaps
- **Gradient Analysis**: Visualize gradients during attack generation
- **Defense Mechanisms**: Implement and test various defense strategies
- **Tutorial System**: Step-by-step guides for different attack types
- **Research Paper Integration**: Link to relevant academic papers
- **Community Features**: User submissions and leaderboards

## 🛠️ Technologies Used

- **Backend**: Flask, PyTorch, TorchVision
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Communication**: Axios for HTTP requests
- **Machine Learning**: PyTorch ecosystem for model inference and attack generation

## 🐛 Troubleshooting

### Common Issues

**Backend Connection Failed**
- Ensure backend is running on port 5000
- Check if port 5000 is available and not blocked by firewall
- Verify CORS configuration in backend

**Model Loading Issues**
- Ensure sufficient RAM (models require ~2GB)
- Check PyTorch installation and compatibility
- Verify CUDA compatibility if using GPU

**Attack Generation Fails**
- Check image format (JPEG/PNG supported)
- Verify attack parameters are within valid ranges
- Ensure model is properly loaded

**SSL Certificate Issues (macOS)**
- If you encounter SSL errors when downloading models, run:
  ```bash
  /Applications/Python\ 3.x/Install\ Certificates.command
  ```
- Or install certificates manually for your Python version

### Performance Tips

- **GPU Usage**: Ensure CUDA is available for faster processing
- **Model Selection**: Use smaller models (ResNet-18) for faster attacks
- **Parameter Tuning**: Start with smaller epsilon values
- **Batch Processing**: Consider implementing batch processing for multiple images

## 🤝 Contributing

We welcome contributions! Here are some ways you can help:

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Feature Requests**: Suggest new attack methods, models, or UI improvements
3. **Code Contributions**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes
   - Test thoroughly
   - Submit a pull request
4. **Documentation**: Improve README, add code comments, or create tutorials
5. **Research**: Implement new attack methods or defense mechanisms

### Development Setup
- Follow the installation instructions above
- Use the provided npm scripts for development
- Ensure both frontend and backend are running for full functionality

## 📚 References & Resources

- [Adversarial Examples in the Physical World](https://arxiv.org/abs/1607.02533) - FGSM paper
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) - PGD paper
- [PyTorch Adversarial Training](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) - Official tutorial
- [CleverHans](https://github.com/cleverhans-lab/cleverhans) - Adversarial example library

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- ImageNet for the pre-trained models
- Research community for adversarial attack algorithms
- React and Flask communities for the web frameworks
