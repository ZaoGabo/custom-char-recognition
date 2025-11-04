"""
CNN v2 Finetuned Predictor

Inference wrapper for the fine-tuned CNN v2 model.
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

from src.cnn_model_v2 import CharCNN_v2
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS


class CNNPredictor_v2_Finetuned:
    
    def __init__(self, model_dir: str = 'models/cnn_modelo_v2_finetuned'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        config_path = self.model_dir / 'config_finetuned.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        num_classes = self.config['num_classes']
        dropout_rate = self.config.get('dropout_rate', 0.5)
        
        self.model = CharCNN_v2(num_classes=num_classes, dropout_rate=dropout_rate)
        
        model_path = self.model_dir / 'best_model_finetuned.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = LabelMap(CUSTOM_LABELS)
        
        print(f"CNN v2 Finetuned loaded from {model_dir}")
        print(f"  Validation accuracy: {self.config.get('best_val_acc', 0)*100:.2f}%")
        print(f"  Epochs trained: {self.config.get('epochs_trained', 0)}")
        print(f"  Device: {self.device}")
    
    
    def predict(self, imagen: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict character from image.
        
        Args:
            imagen: (28, 28) normalized array [0, 1]
        
        Returns:
            (character, probability, top5_predictions)
        """
        if len(imagen.shape) == 2:
            imagen = imagen[np.newaxis, np.newaxis, ...]
        elif len(imagen.shape) == 3:
            imagen = imagen[np.newaxis, ...]
        
        tensor = torch.from_numpy(imagen.astype(np.float32))
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        probs_np = probs.cpu().numpy()[0]
        
        top5_indices = np.argsort(probs_np)[::-1][:5]
        top5 = [
            (self.label_map.get_label(int(idx)), float(probs_np[idx]))
            for idx in top5_indices
        ]
        
        return top5[0][0], top5[0][1], top5
    
    def predict_proba(self, imagen: np.ndarray) -> np.ndarray:
        """Return probability distribution over all classes."""
        if len(imagen.shape) == 2:
            imagen = imagen[np.newaxis, np.newaxis, ...]
        elif len(imagen.shape) == 3:
            imagen = imagen[np.newaxis, ...]
        
        tensor = torch.from_numpy(imagen.astype(np.float32))
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()[0]


def cargar_cnn_predictor_v2_finetuned(model_dir: str = 'models/cnn_modelo_v2_finetuned') -> CNNPredictor_v2_Finetuned:
    return CNNPredictor_v2_Finetuned(model_dir)


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    
    print("Testing CNN v2 Finetuned Predictor")
    
    predictor = cargar_cnn_predictor_v2_finetuned()
    
    try:
        from src.data_loader import DataLoader
        loader = DataLoader('data/raw', limit_per_class=10)
        X, y = loader.cargar_datos()
        
        print(f"\nTesting with {len(X)} samples...")
        
        correct = 0
        for i in range(min(100, len(X))):
            pred_char, pred_prob, _ = predictor.predict(X[i])
            true_label = predictor.label_map.get_label(y[i])
            
            if pred_char == true_label:
                correct += 1
        
        accuracy = correct / min(100, len(X))
        print(f"Test accuracy: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"Test failed: {e}")
        
        print("\nGenerating random test...")
        test_img = np.random.rand(28, 28)
        char, prob, top5 = predictor.predict(test_img)
        
        print(f"Prediction: {char} ({prob*100:.1f}%)")
        print("Top 5:")
        for c, p in top5:
            print(f"  {c}: {p*100:.1f}%")

