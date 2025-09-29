"""
Entrenar modelo con imágenes reales que tú subas.
"""

import numpy as np
import pickle
import os
from PIL import Image
import string
import sys

# Añadir directorio demo para importar la clase
sys.path.append('demo')
from modelo import RedNeuronalSimple

def cargar_imagenes_reales():
    """Cargar imágenes reales de las carpetas de datos."""
    # Etiquetas: A-Z (mayúsculas) + a-z (minúsculas)
    etiquetas = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    sufijos = ['_upper'] * 26 + ['_lower'] * 26
    
    imagenes = []
    etiquetas_objetivo = []
    
    print("📂 Cargando imágenes reales desde las carpetas...")
    
    data_dir = "data"
    total_imagenes = 0
    
    for i, (caracter, sufijo) in enumerate(zip(etiquetas, sufijos)):
        carpeta = os.path.join(data_dir, f"{caracter}{sufijo}")
        
        if not os.path.exists(carpeta):
            print(f"⚠️  Carpeta no encontrada: {carpeta}")
            continue
            
        # Buscar archivos de imagen
        archivos = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            import glob
            archivos.extend(glob.glob(os.path.join(carpeta, ext)))
            archivos.extend(glob.glob(os.path.join(carpeta, ext.upper())))
        
        if not archivos:
            print(f"⚠️  No hay imágenes en: {carpeta}")
            continue
            
        print(f"   Cargando '{caracter}': {len(archivos)} imágenes")
        
        for archivo in archivos:
            try:
                # Cargar y procesar imagen
                img = Image.open(archivo)
                img = img.convert('L')  # Escala de grises
                img = img.resize((28, 28))  # Redimensionar
                
                img_array = np.array(img).flatten() / 255.0  # Normalizar
                
                # Crear vector objetivo (one-hot)
                objetivo = np.zeros(len(etiquetas))
                objetivo[i] = 1.0
                
                imagenes.append(img_array)
                etiquetas_objetivo.append(objetivo)
                total_imagenes += 1
                
            except Exception as e:
                print(f"❌ Error cargando {archivo}: {e}")
    
    print(f"✅ Total de imágenes reales cargadas: {total_imagenes}")
    
    return np.array(imagenes), np.array(etiquetas_objetivo)

def entrenar_con_imagenes_reales():
    """Entrenar modelo con imágenes reales."""
    print("🚀 Entrenando con imágenes REALES...")
    
    # Cargar imágenes reales
    imagenes, etiquetas = cargar_imagenes_reales()
    
    if len(imagenes) == 0:
        print("❌ No se encontraron imágenes. Asegúrate de tener imágenes en las carpetas data/")
        return None
    
    print(f"📊 Entrenando con {len(imagenes)} imágenes reales")
    
    # Crear modelo
    modelo = RedNeuronalSimple(
        entrada_neuronas=784,
        oculta_neuronas=200,
        salida_neuronas=52,
        tasa_aprendizaje=0.3
    )
    
    # Entrenar
    print("🎯 Entrenando...")
    epocas = 200  # Más épocas para imágenes reales
    
    for epoca in range(epocas):
        # Mezclar datos
        indices = np.random.permutation(len(imagenes))
        
        for idx in indices:
            modelo.entrenar(imagenes[idx], etiquetas[idx])
        
        if (epoca + 1) % 50 == 0:
            print(f"   Época {epoca + 1}/{epocas}")
    
    # Probar
    print("🔍 Evaluando modelo...")
    correctas = 0
    total = len(imagenes)
    
    for i in range(total):
        prediccion = modelo.predecir(imagenes[i])
        pred_idx = np.argmax(prediccion)
        real_idx = np.argmax(etiquetas[i])
        
        if pred_idx == real_idx:
            correctas += 1
    
    precision = (correctas / total) * 100
    print(f"✅ Precisión con imágenes reales: {precision:.2f}%")
    
    # Guardar
    with open("models/modelo_entrenado.pkl", 'wb') as f:
        pickle.dump(modelo, f)
    
    print("✅ Modelo entrenado con datos reales guardado!")
    return modelo

if __name__ == "__main__":
    entrenar_con_imagenes_reales()