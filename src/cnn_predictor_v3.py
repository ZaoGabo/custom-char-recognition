"""
CNN v3 Predictor

Inference wrapper for the CNN v3 ResNet-like model trained on Colab.
"""
import torch
import numpy
import json
from pathlib import Path
from typing import Tuple, List

from src.cnn_model_v3 import CharCNN_v3
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS


class CNNPredictor_v3:
    
    def __init__(self, model_dir: str = 'models/cnn_modelo_v3'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config_path = self.model_dir / 'config_v3.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            num_classes = self.config['num_classes']
            dropout_rate = self.config.get('dropout_rate', 0.3)
        else:
            print(f"⚠️ Config not found at {config_path}. Using defaults...")
            num_classes = 62  # EMNIST ByClass
            dropout_rate = 0.3

        model_path = self.model_dir / 'best_model_v3.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        self.model = CharCNN_v3(num_classes=num_classes, dropout_rate=dropout_rate)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = LabelMap(CUSTOM_LABELS)
        
        print(f"CNN v3 loaded from {model_dir}")
        print(f"  Device: {self.device}")
        print(f"  Accuracy: {self.config.get('best_test_acc', 'N/A')}")
    
    def predict(self, imagen):  # type: ignore
        """
        Predict character from image.
        
        Args:
            imagen: (28, 28) normalized array [0, 1]
        
        Returns:
            (character, probability, top5_predictions)
        """
        # Convert directly to PyTorch tensor to bypass pydantic validation
        if not torch.is_tensor(imagen):
            tensor = torch.from_numpy(numpy.asarray(imagen, dtype=numpy.float32))
        else:
            tensor = imagen.float()
        
        # Ensure correct shape (B, C, H, W)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dim
        
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        probs_np = probs.cpu().numpy()[0]
        
        top5_indices = numpy.argsort(probs_np)[::-1][:5]
        top5 = [
            (self.label_map.get_label(int(idx)), float(probs_np[idx]))
            for idx in top5_indices
        ]
        
        return top5[0][0], top5[0][1], top5
    
    def predict_proba(self, imagen):  # type: ignore
        """Return probability distribution over all classes."""
        imagen = numpy.asarray(imagen, dtype=numpy.float32)
        
        if len(imagen.shape) == 2:
            imagen = imagen[numpy.newaxis, numpy.newaxis, ...]
        elif len(imagen.shape) == 3:
            imagen = imagen[numpy.newaxis, ...]
        
        tensor = torch.from_numpy(imagen)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()[0]


def cargar_cnn_predictor_v3(model_dir: str = 'models/cnn_modelo_v3') -> CNNPredictor_v3:
    return CNNPredictor_v3(model_dir)


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    
    print("Testing CNN v3 Predictor")
    
    predictor = cargar_cnn_predictor_v3()
    
    # Generate random test
    print("\nGenerating random test...")
    test_img = np.random.rand(28, 28)
    char, prob, top5 = predictor.predict(test_img)
    
    print(f"Prediction: {char} ({prob*100:.1f}%)")
    print("Top 5:")
    for c, p in top5:
        print(f"  {c}: {p*100:.1f}%")
