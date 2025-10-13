"""
Configuracion global del proyecto, cargada desde config.yml.
"""

import yaml
from pathlib import Path

def _cargar_config():
    ruta_config = Path(__file__).parent.parent / "config.yml"
    if not ruta_config.exists():
        raise FileNotFoundError(f"El archivo de configuración no se encontró en {ruta_config}")
    
    with open(ruta_config, 'r') as f:
        return yaml.safe_load(f)

_config = _cargar_config()

NETWORK_CONFIG = _config.get('network_config', {})
DATA_CONFIG = _config.get('data_config', {})
AUGMENTATION_CONFIG = _config.get('augmentation_config', {})
PATHS = _config.get('paths', {})
LOGGING_CONFIG = _config.get('logging_config', {})
CUSTOM_LABELS = _config.get('custom_labels', [])

# Convertir tamano_imagen a tupla
if 'tamano_imagen' in DATA_CONFIG and isinstance(DATA_CONFIG['tamano_imagen'], list):
    DATA_CONFIG['tamano_imagen'] = tuple(DATA_CONFIG['tamano_imagen'])
