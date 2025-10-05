"""
Configuracion global del proyecto de reconocimiento de caracteres personalizados.
"""

# Parametros de la red neuronal
NETWORK_CONFIG = {
    'capas': [784, 512, 256, 128, 52],
    'activaciones': ['relu', 'relu', 'relu', 'softmax'],
    'tasa_aprendizaje': 0.001,
    'lambda_l2': 0.0005,
    'dropout_rate': 0.2,
    'epocas': 120,
    'tamano_lote': 64,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    # Compatibilidad con scripts antiguos
    'entrada_neuronas': 784,
    'oculta_neuronas': 128,
    'salida_neuronas': 52,
    'funcion_activacion': 'sigmoide',
}

# Parametros de preprocesamiento
DATA_CONFIG = {
    'tamano_imagen': (28, 28),
    'normalizar': True,
    'usar_augmentacion': True,
    'division_entrenamiento': 0.8,
    'division_validacion': 0.1,
    'division_prueba': 0.1,
    'semilla': 42
}

# Parametros de aumentacion
AUGMENTATION_CONFIG = {
    'rango_rotacion': 15,
    'desplazamiento_horizontal': 0.1,
    'desplazamiento_vertical': 0.1,
    'rango_zoom': 0.1,
    'voltear_horizontal': False,
    'voltear_vertical': False
}

# Rutas de archivos
PATHS = {
    'datos_crudos': 'data/raw/',
    'datos_procesados': 'data/processed/',
    'modelos': 'models/',
    'salidas': 'output/',
    'registros': 'output/logs/'
}

# Configuracion de logging
LOGGING_CONFIG = {
    'nivel': 'INFO',
    'formato': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'archivo': 'output/logs/entrenamiento.log'
}
CUSTOM_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

