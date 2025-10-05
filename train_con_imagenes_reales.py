"""
Entrenar modelo con imagenes reales que tu subas.
"""

import numpy as np
import pickle
import os
from PIL import Image
import string
import sys
from pathlib import Path

sys.path.append('demo')
from modelo import RedNeuronalSimple

from src.config import PATHS


def cargar_imagenes_reales():
    etiquetas = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    sufijos = ['_upper'] * 26 + ['_lower'] * 26

    imagenes = []
    etiquetas_objetivo = []

    print("Cargando imagenes reales desde las carpetas...")

    data_dir = Path(PATHS['datos_crudos'])
    total_imagenes = 0

    for i, (caracter, sufijo) in enumerate(zip(etiquetas, sufijos)):
        carpeta = data_dir / f"{caracter}{sufijo}"

        if not carpeta.exists():
            print(f"  Carpeta no encontrada: {carpeta}")
            continue

        archivos = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            archivos.extend(carpeta.glob(ext))
            archivos.extend(carpeta.glob(ext.upper()))

        if not archivos:
            print(f"  No hay imagenes en: {carpeta}")
            continue

        print(f"   Cargando '{caracter}': {len(archivos)} imagenes")

        for archivo in archivos:
            try:
                img = Image.open(archivo)
                img = img.convert('L')
                img = img.resize((28, 28))

                img_array = np.array(img).flatten() / 255.0

                objetivo = np.zeros(len(etiquetas))
                objetivo[i] = 1.0

                imagenes.append(img_array)
                etiquetas_objetivo.append(objetivo)
                total_imagenes += 1

            except Exception as e:
                print(f" Error cargando {archivo}: {e}")

    print(f"Total de imagenes reales cargadas: {total_imagenes}")

    return np.array(imagenes), np.array(etiquetas_objetivo)


def entrenar_con_imagenes_reales():
    print("Entrenando con imagenes REALES...")

    imagenes, etiquetas = cargar_imagenes_reales()

    if len(imagenes) == 0:
        print("No se encontraron imagenes. Asegurate de tener imagenes en las carpetas data/raw/")
        return None

    print(f"Entrenando con {len(imagenes)} imagenes reales")

    modelo = RedNeuronalSimple(
        entrada_neuronas=784,
        oculta_neuronas=200,
        salida_neuronas=52,
        tasa_aprendizaje=0.3
    )

    print("Entrenando...")
    epocas = 200

    for epoca in range(epocas):
        indices = np.random.permutation(len(imagenes))

        for idx in indices:
            modelo.entrenar(imagenes[idx], etiquetas[idx])

        if (epoca + 1) % 50 == 0:
            print(f"   Epoca {epoca + 1}/{epocas}")

    print("Evaluando modelo...")
    correctas = 0
    total = len(imagenes)

    for i in range(total):
        prediccion = modelo.predecir(imagenes[i])
        pred_idx = np.argmax(prediccion)
        real_idx = np.argmax(etiquetas[i])

        if pred_idx == real_idx:
            correctas += 1

    precision = (correctas / total) * 100
    print(f"Precision con imagenes reales: {precision:.2f}%")

    modelo_path = Path(PATHS['modelos'])
    modelo_path.mkdir(parents=True, exist_ok=True)
    with open(modelo_path / "modelo_entrenado.pkl", 'wb') as f:
        pickle.dump(modelo, f)

    print("Modelo entrenado con datos reales guardado!")
    return modelo


if __name__ == '__main__':
    entrenar_con_imagenes_reales()
