"""
Paquete principal del sistema de reconocimiento de caracteres personalizados.
"""

from .config import *
from .label_map import LabelMap, DEFAULT_LABEL_MAP
from .data_loader import DataLoader
from .network import NeuralNetwork
from .utils import *

__version__ = "1.0.0"
__author__ = "Sistema de Reconocimiento de Caracteres"