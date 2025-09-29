
import os
import pickle
import numpy as np
from .network import NeuralNetwork as RedNeuronal
from .data_loader import DataLoader
from .config import PATHS, DATA_CONFIG, NETWORK_CONFIG
from .label_map import DEFAULT_LABEL_MAP
from .scripts.generar_imagenes_sinteticas import generar_imagenes_sinteticas

# --- Paso 1: Verificar y poblar datos si es necesario ---
ruta_raw = PATHS['datos_crudos']
modelo_path = os.path.join(PATHS['modelos'], "modelo_entrenado.pkl")

# Verificar si hay carpetas con imágenes
carpetas_con_imagenes = 0
total_imagenes = 0

for item in os.listdir(ruta_raw):
    item_path = os.path.join(ruta_raw, item)
    if os.path.isdir(item_path) and not item.startswith('.'):
        imagenes = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if len(imagenes) > 0:
            carpetas_con_imagenes += 1
            total_imagenes += len(imagenes)

if total_imagenes == 0:
    print("⚠️ data/raw está vacío. Generando imágenes sintéticas...")
    generar_imagenes_sinteticas()
else:
    print(f"✅ data/raw contiene {total_imagenes} imágenes en {carpetas_con_imagenes} carpetas.")

# --- Paso 2: Cargar datos desde directorios ---
print("📁 Cargando datos desde directorios...")
loader = DataLoader(ruta_datos=ruta_raw, mapa_etiquetas=DEFAULT_LABEL_MAP)
loader.cargar_desde_directorio()

if len(loader.imagenes) == 0:
    print("❌ No se pudieron cargar imágenes. Verificando estructura...")
    # Intentar generar imágenes sintéticas como fallback
    print("🔄 Generando imágenes sintéticas como respaldo...")
    generar_imagenes_sinteticas()
    # Reintentar carga
    loader.cargar_desde_directorio()

loader.preprocesar_imagenes()

# --- Paso 3: Dividir datos ---
X_train, X_val, X_test, y_train, y_val, y_test = loader.dividir_datos()

# --- Paso 4: Augmentación ---
X_train_aug, y_train_aug = loader.aplicar_augmentacion(X_train, y_train)

# --- Paso 5: Entrenamiento ---
if not os.path.exists(modelo_path):
    print("⚙️ Entrenando red neuronal...")
    red = RedNeuronal(
        entrada_neuronas=X_train_aug.shape[1],
        oculta_neuronas=NETWORK_CONFIG['oculta_neuronas'], 
        salida_neuronas=DEFAULT_LABEL_MAP.get_num_classes(),
        tasa_aprendizaje=NETWORK_CONFIG['tasa_aprendizaje']
    )
    
    # Convertir etiquetas a one-hot encoding
    def to_one_hot(labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            one_hot[i, label] = 1.0
        return one_hot
    
    y_train_one_hot = to_one_hot(y_train_aug, DEFAULT_LABEL_MAP.get_num_classes())
    
    # Entrenar por épocas
    print(f"Iniciando entrenamiento con {len(X_train_aug)} imágenes...")
    epocas = NETWORK_CONFIG['epocas']
    
    for epoca in range(epocas):
        # Entrenar con todos los datos
        for i in range(len(X_train_aug)):
            red.entrenar(X_train_aug[i], y_train_one_hot[i])
        
        # Mostrar progreso cada 10 épocas
        if (epoca + 1) % 10 == 0:
            print(f"Época {epoca + 1}/{epocas} completada")
    
    # Guardar modelo
    os.makedirs(PATHS['modelos'], exist_ok=True)
    with open(modelo_path, 'wb') as f:
        pickle.dump(red, f)
    print("✅ Modelo entrenado y guardado en:", modelo_path)
else:
    print("✅ Modelo ya entrenado. No se requiere reentrenamiento.")
