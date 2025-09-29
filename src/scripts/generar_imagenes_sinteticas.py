import os
import random
from PIL import Image, ImageDraw, ImageFont
from ..config import PATHS, CUSTOM_LABELS

def generar_imagenes_sinteticas(cantidad_por_clase=10):
    """
    Genera imágenes sintéticas para cada clase de carácter.
    Crea las imágenes en las carpetas con sufijos _upper/_lower.
    """
    ruta_raw = PATHS['datos_crudos']
    os.makedirs(ruta_raw, exist_ok=True)

    # Fuente por defecto (puedes cambiarla si tienes otras instaladas)
    try:
        fuente = ImageFont.truetype("arial.ttf", 20)
    except:
        fuente = ImageFont.load_default()

    # Mapear cada letra a su carpeta correspondiente
    for letra in CUSTOM_LABELS:
        # Determinar el nombre de la carpeta según si es mayúscula o minúscula
        if letra.isupper():
            carpeta_nombre = f"{letra}_upper"
        else:
            carpeta_nombre = f"{letra}_lower"
            
        carpeta_clase = os.path.join(ruta_raw, carpeta_nombre)
        
        # Crear carpeta si no existe
        if not os.path.exists(carpeta_clase):
            os.makedirs(carpeta_clase, exist_ok=True)

        print(f"Generando {cantidad_por_clase} imágenes para '{letra}' en carpeta '{carpeta_nombre}'...")

        for i in range(cantidad_por_clase):
            # Crear imagen en escala de grises
            img = Image.new('L', (28, 28), color=0)  # Fondo negro
            draw = ImageDraw.Draw(img)

            # Posición aleatoria para variar (centrado con pequeña variación)
            x = random.randint(6, 12)
            y = random.randint(3, 8)
            
            # Dibujar la letra en blanco
            draw.text((x, y), letra, font=fuente, fill=255)

            # Nombre de archivo
            nombre = f"{letra}_{i:03d}.png"
            ruta_imagen = os.path.join(carpeta_clase, nombre)
            img.save(ruta_imagen)

    total_imagenes = cantidad_por_clase * len(CUSTOM_LABELS)
    print(f"✅ Generadas {total_imagenes} imágenes sintéticas ({len(CUSTOM_LABELS)} clases × {cantidad_por_clase} imágenes)")
    print(f"📁 Ubicación: {ruta_raw}")

if __name__ == "__main__":
    generar_imagenes_sinteticas(cantidad_por_clase=15)
