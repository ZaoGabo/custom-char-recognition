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
if 'lr_scheduler_config' in _config:
    NETWORK_CONFIG['lr_scheduler_config'] = _config['lr_scheduler_config']
    
DATA_CONFIG = _config.get('data_config', {})
AUGMENTATION_CONFIG = _config.get('augmentation_config', {})
ADVANCED_AUGMENTATION_CONFIG = _config.get('advanced_augmentation', {})
PATHS = _config.get('paths', {})
LOGGING_CONFIG = _config.get('logging_config', {})
CUSTOM_LABELS = _config.get('custom_labels', [])

# Convertir tamano_imagen a tupla
if 'tamano_imagen' in DATA_CONFIG and isinstance(DATA_CONFIG['tamano_imagen'], list):
    DATA_CONFIG['tamano_imagen'] = tuple(DATA_CONFIG['tamano_imagen'])

if 'gauss_noise' in ADVANCED_AUGMENTATION_CONFIG:
    gauss_cfg = ADVANCED_AUGMENTATION_CONFIG['gauss_noise']
    for key in ('std_range', 'mean_range', 'var_limit'):
        value = gauss_cfg.get(key)
        if isinstance(value, list) and len(value) == 2:
            gauss_cfg[key] = tuple(value)

if 'coarse_dropout' in ADVANCED_AUGMENTATION_CONFIG:
    dropout_cfg = ADVANCED_AUGMENTATION_CONFIG['coarse_dropout']
    for key in ('num_holes_range', 'hole_height_range', 'hole_width_range'):
        value = dropout_cfg.get(key)
        if isinstance(value, list) and len(value) == 2:
            dropout_cfg[key] = tuple(value)

if 'affine' not in ADVANCED_AUGMENTATION_CONFIG and 'shift_scale_rotate' in ADVANCED_AUGMENTATION_CONFIG:
    ADVANCED_AUGMENTATION_CONFIG['affine'] = dict(ADVANCED_AUGMENTATION_CONFIG['shift_scale_rotate'])

if 'affine' in ADVANCED_AUGMENTATION_CONFIG:
    affine_cfg = ADVANCED_AUGMENTATION_CONFIG['affine']
    for key in ('translate_percent', 'scale', 'rotate', 'shear'):
        value = affine_cfg.get(key)
        if isinstance(value, list) and len(value) == 2:
            affine_cfg[key] = tuple(value)
