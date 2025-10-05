"""Entrenamiento rapido del modelo clasico con imagenes reales."""

from __future__ import annotations

import pickle
import string
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.append('demo')
from modelo import RedNeuronalSimple

from src.config import PATHS


def cargar_imagenes_reales() -> Tuple[np.ndarray, np.ndarray]:
    """Cargar imagenes desde ``data/raw`` en formato listo para entrenamiento."""
    etiquetas = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    sufijos = ['_upper'] * 26 + ['_lower'] * 26

    imagenes: list[np.ndarray] = []
    objetivos: list[np.ndarray] = []

    print('Cargando imagenes reales desde las carpetas...')
    data_dir = Path(PATHS['datos_crudos'])

    for indice, (caracter, sufijo) in enumerate(zip(etiquetas, sufijos)):
        carpeta = data_dir / f'{caracter}{sufijo}'
        if not carpeta.exists():
            print(f'  Carpeta no encontrada: {carpeta}')
            continue

        archivos = []
        for patron in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            archivos.extend(carpeta.glob(patron))
            archivos.extend(carpeta.glob(patron.upper()))

        if not archivos:
            print(f'  No hay imagenes en: {carpeta}')
            continue

        print(f"   Cargando '{caracter}': {len(archivos)} imagenes")
        objetivo = np.zeros(len(etiquetas), dtype=np.float32)
        objetivo[indice] = 1.0

        for archivo in archivos:
            try:
                with Image.open(archivo) as img:
                    imagen = img.convert('L').resize((28, 28))
            except (OSError, UnidentifiedImageError) as exc:
                print(f' Error cargando {archivo}: {exc}')
                continue

            imagenes.append(np.asarray(imagen, dtype=np.float32).flatten() / 255.0)
            objetivos.append(objetivo.copy())

    print(f'Total de imagenes reales cargadas: {len(imagenes)}')
    return np.array(imagenes), np.array(objetivos)


def entrenar_con_imagenes_reales() -> RedNeuronalSimple | None:
    """Entrenar ``RedNeuronalSimple`` con las imagenes reales disponibles."""
    print('Entrenando con imagenes REALES...')
    imagenes, etiquetas = cargar_imagenes_reales()

    if imagenes.size == 0:
        print('No se encontraron imagenes. Asegurate de tener datos en data/raw/.')
        return None

    print(f'Entrenando con {len(imagenes)} imagenes reales')
    modelo = RedNeuronalSimple(
        entrada_neuronas=784,
        oculta_neuronas=200,
        salida_neuronas=52,
        tasa_aprendizaje=0.3,
    )

    epocas = 200
    for epoca in range(epocas):
        for indice in np.random.permutation(len(imagenes)):
            modelo.entrenar(imagenes[indice], etiquetas[indice])
        if (epoca + 1) % 50 == 0:
            print(f'   Epoca {epoca + 1}/{epocas}')

    print('Evaluando modelo...')
    predicciones = np.argmax([modelo.predecir(img) for img in imagenes], axis=1)
    reales = np.argmax(etiquetas, axis=1)
    precision = (predicciones == reales).mean() * 100
    print(f'Precision con imagenes reales: {precision:.2f}%')

    modelo_path = Path(PATHS['modelos'])
    modelo_path.mkdir(parents=True, exist_ok=True)
    with (modelo_path / 'modelo_entrenado.pkl').open('wb') as archivo:
        pickle.dump(modelo, archivo)

    print('Modelo entrenado con datos reales guardado!')
    return modelo


if __name__ == '__main__':
    entrenar_con_imagenes_reales()