"""
Entrenar modelo con más datos y mejor calidad.
"""

import numpy as np
import pickle
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import string
import random
import sys

# Añadir directorio demo para importar la clase
sys.path.append('demo')
from modelo import RedNeuronalSimple

def generar_imagen_caracter_mejorada(caracter, tamano=(28, 28)):
    """Generar imagen sintética mejorada de un carácter."""
    # Crear imagen en blanco
    img = Image.new('L', (40, 40), color=0)  # Más grande inicialmente
    draw = ImageDraw.Draw(img)
    
    # Usar diferentes fuentes y tamaños
    fonts = []
    font_sizes = [18, 20, 22, 24, 26]
    font_names = ["arial.ttf", "times.ttf", "calibri.ttf", "tahoma.ttf"]
    
    # Intentar cargar diferentes fuentes
    for font_name in font_names:
        try:
            for size in font_sizes:
                fonts.append(ImageFont.truetype(font_name, size))
        except:
            continue
    
    # Si no hay fuentes del sistema, usar fuente por defecto
    if not fonts:
        fonts = [ImageFont.load_default()]
    
    font = random.choice(fonts)
    
    # Calcular posición centrada
    bbox = draw.textbbox((0, 0), caracter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (40 - text_width) // 2
    y = (40 - text_height) // 2
    
    # Dibujar texto
    draw.text((x, y), caracter, fill=255, font=font)
    
    # Aplicar transformaciones aleatorias
    # Rotación ligera
    angle = random.randint(-15, 15)
    img = img.rotate(angle, fillcolor=0)
    
    # Ruido aleatorio
    if random.random() > 0.5:
        # Añadir algo de ruido
        noise = np.random.randint(0, 50, (40, 40))
        img_array = np.array(img)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
    
    # Blur ligero ocasional
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Redimensionar a tamaño final
    img = img.resize(tamano, Image.Resampling.LANCZOS)
    
    return img

def crear_datos_entrenamiento_mejorados():
    """Crear más datos de entrenamiento con mejor calidad."""
    # Etiquetas: A-Z (mayúsculas) + a-z (minúsculas)
    etiquetas = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    
    imagenes = []
    etiquetas_objetivo = []
    
    print("🔧 Generando datos de entrenamiento mejorados...")
    
    # Generar 100 imágenes por cada carácter (5x más datos)
    for i, caracter in enumerate(etiquetas):
        print(f"   Generando '{caracter}': {i+1}/52")
        
        for _ in range(100):
            # Generar imagen mejorada
            img = generar_imagen_caracter_mejorada(caracter)
            img_array = np.array(img).flatten() / 255.0  # Normalizar
            
            # Crear vector objetivo (one-hot)
            objetivo = np.zeros(len(etiquetas))
            objetivo[i] = 1.0
            
            imagenes.append(img_array)
            etiquetas_objetivo.append(objetivo)
    
    return np.array(imagenes), np.array(etiquetas_objetivo)

def entrenar_modelo_mejorado():
    """Entrenar modelo mejorado con más datos."""
    print("🚀 Iniciando entrenamiento de modelo MEJORADO...")
    
    # Crear datos mejorados
    imagenes, etiquetas = crear_datos_entrenamiento_mejorados()
    print(f"✅ Datos creados: {imagenes.shape[0]} imágenes (100 por carácter)")
    
    # Mezclar datos
    indices = np.random.permutation(len(imagenes))
    imagenes = imagenes[indices]
    etiquetas = etiquetas[indices]
    
    # Crear modelo con más neuronas
    modelo = RedNeuronalSimple(
        entrada_neuronas=784,  # 28x28
        oculta_neuronas=256,   # MÁS NEURONAS
        salida_neuronas=52,    # A-Z + a-z
        tasa_aprendizaje=0.2   # Tasa de aprendizaje más baja
    )
    
    # Entrenar por más épocas
    print("🎯 Entrenando modelo mejorado...")
    epocas = 100  # MÁS ÉPOCAS
    
    for epoca in range(epocas):
        # Mezclar datos cada época
        indices = np.random.permutation(len(imagenes))
        imagenes_mezcladas = imagenes[indices]
        etiquetas_mezcladas = etiquetas[indices]
        
        for i in range(len(imagenes_mezcladas)):
            modelo.entrenar(imagenes_mezcladas[i], etiquetas_mezcladas[i])
        
        if (epoca + 1) % 20 == 0:
            print(f"   Época {epoca + 1}/{epocas} completada")
    
    # Probar modelo
    print("🔍 Probando modelo mejorado...")
    correctas = 0
    total = min(1000, len(imagenes))  # Probar con 1000 muestras
    
    for i in range(total):
        prediccion = modelo.predecir(imagenes[i])
        pred_idx = np.argmax(prediccion)
        real_idx = np.argmax(etiquetas[i])
        
        if pred_idx == real_idx:
            correctas += 1
    
    precision = (correctas / total) * 100
    print(f"✅ Precisión MEJORADA: {precision:.2f}% ({correctas}/{total})")
    
    # Guardar modelo
    print("💾 Guardando modelo mejorado...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/modelo_entrenado.pkl", 'wb') as f:
        pickle.dump(modelo, f)
    
    print("✅ Modelo mejorado guardado exitosamente!")
    
    return modelo

if __name__ == "__main__":
    entrenar_modelo_mejorado()