"""
Predictor CNN v2 Finetuned

Clase envoltorio para inferencia usando el modelo CNN v2 re-entrenado.
Optimizado para imágenes de canvas (dibujadas a mano).
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

from src.cnn_model_v2 import CharCNN_v2
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS
from src.utils.logger import app_logger as logger


class CNNPredictor_v2_Finetuned:
    """
    Maneja la carga y predicción con el modelo CNN v2 Finetuned.
    """
    
    def __init__(self, model_dir: str = 'models/cnn_modelo_v2_finetuned'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar configuración si existe
        config_path = self.model_dir / 'config_finetuned.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            num_classes = self.config['num_classes']
            dropout_rate = self.config.get('dropout_rate', 0.5)
        else:
            logger.warning(f"Configuracion no encontrada en {config_path}. Infiriendo valores.")
            self.config = {}
            num_classes = None  # Se inferirá después
            dropout_rate = 0.5

        # Verificar archivo de pesos
        model_path = self.model_dir / 'best_model_finetuned.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Archivo de pesos no encontrado: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Inferir numero de clases si es necesario
        if num_classes is None:
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
            if 'fc4.weight' in state_dict:
                num_classes = state_dict['fc4.weight'].shape[0]
                logger.info(f"Clases inferidas: {num_classes}")
            else:
                raise ValueError("No se pudo inferir num_classes del checkpoint")

        # Inicializar arquitectura
        self.model = CharCNN_v2(num_classes=num_classes, dropout_rate=dropout_rate)
        
        # Cargar pesos
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = LabelMap(CUSTOM_LABELS)
        
        logger.info(f"CNN v2 Finetuned cargado desde {model_dir} en {self.device}")
    
    
    def predict(self, imagen: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice el caracter de una imagen.
        
        Args:
            imagen: Array numpy (28, 28) normalizado [0, 1]
        
        Returns:
            Tupla (caracter, probabilidad, top5_predicciones)
        """
        # Ajustar dimensiones para PyTorch (Batch, Channel, Height, Width)
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
        
        # Obtener top 5
        top5_indices = np.argsort(probs_np)[::-1][:5]
        top5 = [
            (self.label_map.get_label(int(idx)), float(probs_np[idx]))
            for idx in top5_indices
        ]
        
        # Retornar mejor predicción y alternativas
        return top5[0][0], top5[0][1], top5
