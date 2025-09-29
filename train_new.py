"""
Entrenar modelo simple desde cero con compatibilidad total.
"""

import numpy as np
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import string
import random
import sys

# Añadir directorio demo para importar la clase
sys.path.append('demo')
from modelo import RedNeuronalSimple

def generar_imagen_caracter(caracter, tamano=(28, 28)):
    """Generar imagen sintética de un carácter."""
    # Crear imagen en blanco
    img = Image.new('L', tamano, color=0)
    draw = ImageDraw.Draw(img)
    
    # Intentar usar fuente del sistema
    try:
        font_size = random.randint(16, 22)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calcular posición centrada
    bbox = draw.textbbox((0, 0), caracter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (tamano[0] - text_width) // 2
    y = (tamano[1] - text_height) // 2
    
    # Dibujar texto
    draw.text((x, y), caracter, fill=255, font=font)
    
    return img

def crear_datos_entrenamiento():
    """Crear datos de entrenamiento sintéticos."""
    # Etiquetas: A-Z (mayúsculas) + a-z (minúsculas)
    etiquetas = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    
    imagenes = []
    etiquetas_objetivo = []
    
    print("🔧 Generando datos de entrenamiento...")
    
    # Generar 20 imágenes por cada carácter
    for i, caracter in enumerate(etiquetas):
        for _ in range(20):
            # Generar imagen
            img = generar_imagen_caracter(caracter)
            img_array = np.array(img).flatten() / 255.0  # Normalizar
            
            # Crear vector objetivo (one-hot)
            objetivo = np.zeros(len(etiquetas))
            objetivo[i] = 1.0
            
            imagenes.append(img_array)
            etiquetas_objetivo.append(objetivo)
    
    return np.array(imagenes), np.array(etiquetas_objetivo)

def entrenar_modelo_simple():
    """Entrenar modelo simple desde cero."""
    print("🚀 Iniciando entrenamiento de modelo simple...")
    
    # Crear datos
    imagenes, etiquetas = crear_datos_entrenamiento()
    print(f"✅ Datos creados: {imagenes.shape[0]} imágenes")
    
    # Crear modelo
    modelo = RedNeuronalSimple(
        entrada_neuronas=784,  # 28x28
        oculta_neuronas=128,
        salida_neuronas=52,    # A-Z + a-z
        tasa_aprendizaje=0.3
    )
    
    # Entrenar
    print("🎯 Entrenando modelo...")
    epocas = 50
    
    for epoca in range(epocas):
        for i in range(len(imagenes)):
            # Usar método entrenar si existe
            if hasattr(modelo, 'entrenar'):
                modelo.entrenar(imagenes[i], etiquetas[i])
        
        if (epoca + 1) % 10 == 0:
            print(f"   Época {epoca + 1}/{epocas} completada")
    
    # Probar modelo
    print("🔍 Probando modelo...")
    correctas = 0
    total = len(imagenes)
    
    for i in range(total):
        prediccion = modelo.predecir(imagenes[i])
        pred_idx = np.argmax(prediccion)
        real_idx = np.argmax(etiquetas[i])
        
        if pred_idx == real_idx:
            correctas += 1
    
    precision = (correctas / total) * 100
    print(f"✅ Precisión: {precision:.2f}% ({correctas}/{total})")
    
    # Guardar modelo
    print("💾 Guardando modelo...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/modelo_entrenado.pkl", 'wb') as f:
        pickle.dump(modelo, f)
    
    print("✅ Modelo guardado exitosamente!")
    
    return modelo

if __name__ == "__main__":
    entrenar_modelo_simple()