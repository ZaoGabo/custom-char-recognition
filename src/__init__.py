"""
Paquete principal del sistema de reconocimiento de caracteres personalizados.
"""

from .config import *
from .label_map import LabelMap, DEFAULT_LABEL_MAP
from .data_loader import DataLoader
# Removed problematic wildcard import: from .utils import *

# CNN v2 Finetuned - Modelo de producción
from .cnn_model_v2 import CharCNN_v2
from .cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned

__version__ = "2.0.0"
__author__ = "Sistema de Reconocimiento de Caracteres"