import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

export interface AttackRequest {
  image: File;
  attack: string;
  model: string;
  epsilon: number;
  alpha?: number;
  iterations?: number;
}

export interface AttackResponse {
  original: {
    label: string;
    confidence: number;
  };
  adversarial: {
    label: string;
    confidence: number;
  };
  model: string;
  attack: string;
  epsilon: number;
  alpha: number;
  iterations: number;
}

export const api = {
  async runAttack(request: AttackRequest): Promise<AttackResponse> {
    const formData = new FormData();
    formData.append('image', request.image);
    formData.append('attack', request.attack);
    formData.append('model', request.model);
    formData.append('epsilon', request.epsilon.toString());
    
    if (request.alpha) {
      formData.append('alpha', request.alpha.toString());
    }
    
    if (request.iterations) {
      formData.append('iterations', request.iterations.toString());
    }

    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  // Helper function to convert base64 to file
  base64ToFile(base64String: string, filename: string): File {
    const arr = base64String.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    
    return new File([u8arr], filename, { type: mime });
  }
};
