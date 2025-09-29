"""
Configuración global del proyecto de reconocimiento de caracteres personalizados.
"""

# Parámetros de la red neuronal
NETWORK_CONFIG = {
    'entrada_neuronas': 784,         # 28x28 píxeles
    'oculta_neuronas': 128,          # Puedes ajustar según rendimiento
    'salida_neuronas': 52,           # Número de clases (A-Z, a-z)
    'tasa_aprendizaje': 0.001,
    'epocas': 100,
    'tamano_lote': 32,
    'funcion_activacion': 'sigmoide'  # Actualmente fija en la clase
}

# Parámetros de preprocesamiento
DATA_CONFIG = {
    'tamano_imagen': (28, 28),
    'normalizar': True,
    'usar_augmentacion': True,
    'division_entrenamiento': 0.8,
    'division_validacion': 0.1,
    'division_prueba': 0.1,
    'semilla': 42  # Para reproducibilidad
}

# Parámetros de augmentación
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

# Configuración de logging
LOGGING_CONFIG = {
    'nivel': 'INFO',
    'formato': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'archivo': 'output/logs/entrenamiento.log'
}
CUSTOM_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
