"""
Constantes centralizadas del proyecto.
Evita código quemado y facilita mantenimiento.
"""
from enum import Enum
from typing import Tuple

class ModelVersion(Enum):
    """Versiones de modelos disponibles"""
    V2_FINETUNED = "v2_finetuned"
    V3 = "v3"
    V3_SUPER = "v3_super"

class Environment(Enum):
    """Ambientes de ejecución"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

# Configuración de imágenes
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (28, 28)
MAX_IMAGE_SIZE_MB: int = 5
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg']

# Configuración de API
DEFAULT_API_PORT: int = 8000
DEFAULT_API_HOST: str = "0.0.0.0"
API_TIMEOUT_SECONDS: int = 30
MAX_BATCH_SIZE: int = 64

# Configuración de texto OCR
MAX_TEXT_LENGTH: int = 1000
MIN_CHARACTER_CONFIDENCE: float = 0.3

# Rutas de modelos (relativas a project root)
MODELS_DIR = "models"
MODEL_PATHS = {
    ModelVersion.V2_FINETUNED: f"{MODELS_DIR}/cnn_modelo_v2_finetuned",
    ModelVersion.V3: f"{MODELS_DIR}/cnn_modelo_v3",
}

# Configuración de logging
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"

# Performance
DEFAULT_NUM_WORKERS: int = 4
ENABLE_CUDA_DEFAULT: bool = True
