"""
Predictor CNN v3 SUPER

Clase envoltorio para inferencia usando el modelo CNN v3 SUPER (ResNet).
Optimizado para documentos escaneados. Usa TorchScript para mejor portabilidad.
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

from src.label_map import LabelMap
from src.config import CUSTOM_LABELS
from src.utils.logger import app_logger as logger


class CNNPredictor_v3_SUPER:
    """
    Maneja la carga y predicción con el modelo CNN v3 SUPER.
    """
    
    def __init__(self, model_dir: str = 'models/cnn_modelo_v3_super'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar configuración
        config_path = self.model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            logger.warning(f"Configuracion no encontrada en {config_path}. Usando valores por defecto.")
            self.config = {'num_classes': 62, 'model_version': 'v3_super'}

        # Cargar modelo TorchScript
        model_path = self.model_dir / 'model.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        logger.info(f"Cargando modelo TorchScript desde {model_path}...")
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        
        self.label_map = LabelMap(CUSTOM_LABELS)
        
        logger.info(f"CNN v3 SUPER cargado desde {model_dir}")
        logger.info(f"Dispositivo: {self.device}")
    
    def predict(self, imagen: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice el caracter de una imagen.
        
        Args:
            imagen: Array numpy (28, 28) normalizado [0, 1]
        
        Returns:
            Tupla (caracter, probabilidad, top5_predicciones)
        """
        # Convertir a tensor PyTorch
        if not torch.is_tensor(imagen):
            tensor = torch.from_numpy(np.asarray(imagen, dtype=np.float32))
        else:
            tensor = imagen.float()
        
        # Asegurar dimensiones correctas (B, C, H, W)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.to(self.device)
        
        # Inferencia
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        probs_np = probs.cpu().numpy()[0]
        
        # Obtener top 5
        top5_indices = np.argsort(probs_np)[::-1][:5]
        top5 = [
            (self.label_map.get_label(int(idx)), float(probs_np[idx]))
            for idx in top5_indices
        ]
        
        return top5[0][0], top5[0][1], top5
