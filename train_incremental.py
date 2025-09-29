"""
Entrenar incrementalmente el modelo existente.
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

def entrenar_incrementalmente():
    """Continuar entrenando el modelo existente."""
    print("🔄 Continuando entrenamiento del modelo existente...")
    
    # Cargar modelo existente
    modelo_path = "models/modelo_entrenado.pkl"
    
    if not os.path.exists(modelo_path):
        print("❌ No se encontró modelo existente. Ejecute train_new.py primero.")
        return
    
    try:
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        print("✅ Modelo existente cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Generar nuevos datos enfocados en caracteres difíciles
    print("🎯 Generando datos adicionales...")
    
    # Caracteres que suelen confundirse
    caracteres_dificiles = ['a', 'o', 'O', '0', 'l', 'I', '1', 't', 'f', 'r', 'n', 'm', 'w', 'v', 'u']
    etiquetas = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    
    imagenes_nuevas = []
    etiquetas_nuevas = []
    
    # Generar más ejemplos de caracteres difíciles
    for caracter in caracteres_dificiles:
        if caracter in etiquetas:
            indice = etiquetas.index(caracter)
            
            print(f"   Generando más ejemplos de '{caracter}'...")
            
            for _ in range(50):  # 50 ejemplos adicionales por carácter difícil
                # Crear imagen con variaciones
                img = Image.new('L', (28, 28), color=0)
                draw = ImageDraw.Draw(img)
                
                # Fuente aleatoria
                try:
                    font_size = random.randint(18, 26)
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Posición con pequeña variación
                bbox = draw.textbbox((0, 0), caracter, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (28 - text_width) // 2 + random.randint(-2, 2)
                y = (28 - text_height) // 2 + random.randint(-2, 2)
                
                # Intensidad variable
                intensidad = random.randint(200, 255)
                draw.text((x, y), caracter, fill=intensidad, font=font)
                
                img_array = np.array(img).flatten() / 255.0
                
                objetivo = np.zeros(len(etiquetas))
                objetivo[indice] = 1.0
                
                imagenes_nuevas.append(img_array)
                etiquetas_nuevas.append(objetivo)
    
    print(f"✅ Generados {len(imagenes_nuevas)} ejemplos adicionales")
    
    # Continuar entrenamiento
    print("🎯 Continuando entrenamiento...")
    
    # Reducir tasa de aprendizaje para ajuste fino
    modelo.tasa_aprendizaje = 0.05
    
    epocas_adicionales = 30
    for epoca in range(epocas_adicionales):
        for i in range(len(imagenes_nuevas)):
            modelo.entrenar(imagenes_nuevas[i], etiquetas_nuevas[i])
        
        if (epoca + 1) % 10 == 0:
            print(f"   Época adicional {epoca + 1}/{epocas_adicionales}")
    
    # Guardar modelo mejorado
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo, f)
    
    print("✅ Modelo mejorado incrementalmente!")
    print("🔄 Reinicie la aplicación web para usar el modelo actualizado")

if __name__ == "__main__":
    entrenar_incrementalmente()