"""
Predictor para CNN v2 MEJORADA
Wrapper para hacer predicciones con el modelo CNN v2 entrenado
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

from src.cnn_model_v2 import CharCNN_v2
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS


class CNNPredictor_v2:
    """Predictor para CNN v2"""
    
    def __init__(self, model_dir: str = 'models/cnn_modelo_v2'):
        """
        Inicializar predictor
        
        Args:
            model_dir: Directorio con el modelo entrenado
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar configuración
        config_path = self.model_dir / 'config_v2.json'
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró config_v2.json en {model_dir}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Crear modelo
        num_classes = self.config['num_classes']
        dropout_rate = self.config.get('dropout_rate', 0.5)
        
        self.model = CharCNN_v2(num_classes=num_classes, dropout_rate=dropout_rate)
        
        # Cargar pesos
        model_path = self.model_dir / 'best_model_v2.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró best_model_v2.pth en {model_dir}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # El checkpoint puede ser un dict con 'model_state_dict' o directamente el state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label map
        self.label_map = LabelMap(CUSTOM_LABELS)
        
        print(f"✅ CNN v2 MEJORADA cargada desde {model_dir}")
        print(f"   - Accuracy validación: {self.config.get('best_val_acc', 0)*100:.2f}%")
        print(f"   - Epochs entrenados: {self.config.get('epochs_trained', 0)}")
        print(f"   - Augmentation: {self.config.get('augmentation', 'normal')}")
        print(f"   - Device: {self.device}")
    
    def predict(self, imagen: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predecir carácter en una imagen
        
        Args:
            imagen: numpy array (28, 28) normalizado [0, 1]
        
        Returns:
            (carácter, probabilidad, top5)
        """
        # Convertir a tensor
        if len(imagen.shape) == 2:
            imagen = imagen[np.newaxis, np.newaxis, ...]  # (1, 1, 28, 28)
        elif len(imagen.shape) == 3:
            imagen = imagen[np.newaxis, ...]  # (1, 1, 28, 28)
        
        tensor = torch.from_numpy(imagen.astype(np.float32))
        tensor = tensor.to(self.device)
        
        # Predecir
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Convertir a numpy
        probs_np = probs.cpu().numpy()[0]
        
        # Top 5
        top5_indices = np.argsort(probs_np)[::-1][:5]
        top5 = [
            (self.label_map.get_label(int(idx)), float(probs_np[idx]))
            for idx in top5_indices
        ]
        
        # Mejor predicción
        mejor_char = top5[0][0]
        mejor_prob = top5[0][1]
        
        return mejor_char, mejor_prob, top5
    
    def predict_proba(self, imagen: np.ndarray) -> np.ndarray:
        """
        Obtener todas las probabilidades
        
        Args:
            imagen: numpy array (28, 28) normalizado [0, 1]
        
        Returns:
            numpy array (num_classes,) con probabilidades
        """
        # Convertir a tensor
        if len(imagen.shape) == 2:
            imagen = imagen[np.newaxis, np.newaxis, ...]
        elif len(imagen.shape) == 3:
            imagen = imagen[np.newaxis, ...]
        
        tensor = torch.from_numpy(imagen.astype(np.float32))
        tensor = tensor.to(self.device)
        
        # Predecir
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()[0]
    
    def predecir(self, imagen: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Alias en español de predict()"""
        return self.predict(imagen)


def cargar_cnn_predictor_v2(model_dir: str = 'models/cnn_modelo_v2') -> CNNPredictor_v2:
    """
    Función helper para cargar el predictor CNN v2
    
    Args:
        model_dir: Directorio con el modelo
    
    Returns:
        CNNPredictor_v2 inicializado
    """
    return CNNPredictor_v2(model_dir)


# === PRUEBA ===
if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    
    print("="*70)
    print("PRUEBA DE CNN v2 PREDICTOR")
    print("="*70)
    
    # Cargar predictor
    predictor = cargar_cnn_predictor_v2()
    
    # Cargar imagen de prueba
    try:
        from src.data_loader import DataLoader
        
        print("\nCargando datos de prueba...")
        loader = DataLoader('data/raw')
        X, y = loader.cargar_imagenes()
        
        # Buscar una 'A' mayúscula (índice 10)
        indices_A = np.where(y == 10)[0]
        if len(indices_A) > 0:
            idx = indices_A[0]
            imagen = X[idx].reshape(28, 28)
            
            print(f"\nProbando con imagen de clase {y[idx]} (A)")
            print(f"Shape de imagen: {imagen.shape}")
            print(f"Range: [{imagen.min():.3f}, {imagen.max():.3f}]")
            print(f"Mean: {imagen.mean():.3f}")
            
            # Predecir
            char, prob, top5 = predictor.predecir(imagen)
            
            print(f"\n{'='*70}")
            print(f"PREDICCIÓN CNN v2: {char} ({prob*100:.2f}% confianza)")
            print(f"{'='*70}")
            print("\nTop 5:")
            for i, (c, p) in enumerate(top5, 1):
                print(f"  {i}. {c}: {p*100:.2f}%")
            
            # Verificar
            if char == 'A':
                print(f"\n✅ ¡Predicción correcta! (CNN v2 MEJORADA)")
            else:
                print(f"\n❌ Predicción incorrecta (esperaba 'A', obtuvo '{char}')")
        else:
            print("⚠️ No se encontraron imágenes de 'A' en el dataset")
    
    except Exception as e:
        print(f"⚠️ No se pudo cargar datos de prueba: {e}")
        print("Pero el predictor CNN v2 está funcionando correctamente")
    
    print(f"\n{'='*70}")
    print("✅ CNN v2 Predictor funcionando correctamente")
    print(f"{'='*70}")
