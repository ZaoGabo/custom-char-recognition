"""
Script para probar el modelo entrenado de reconocimiento de caracteres.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import PATHS, DATA_CONFIG
from src.label_map import DEFAULT_LABEL_MAP
from src.data_loader import DataLoader


def cargar_modelo(modelo_path: Path):
    if not modelo_path.exists():
        raise FileNotFoundError("No se encontro el modelo entrenado. Ejecute `python -m src.trainer --force`.")
    with modelo_path.open('rb') as f:
        return pickle.load(f)


def preprocesar_imagen(modelo, ruta_imagen: Path) -> np.ndarray:
    imagen = Image.open(ruta_imagen).convert('L')
    imagen = imagen.resize(DATA_CONFIG['tamano_imagen'])
    arr = np.array(imagen, dtype=np.float32).flatten() / 255.0
    return arr


def obtener_probabilidades(modelo, entrada: np.ndarray) -> np.ndarray:
    if hasattr(modelo, 'predecir_probabilidades'):
        salida = modelo.predecir_probabilidades(entrada)
        return salida if salida.ndim == 2 else salida.reshape(1, -1)
    salida = modelo.predecir(entrada)
    return salida if salida.ndim == 2 else salida.reshape(1, -1)


def evaluar_modelo(modelo, limite: int = 0):
    loader = DataLoader(PATHS['datos_crudos'], mapa_etiquetas=DEFAULT_LABEL_MAP)
    loader.cargar_desde_directorio()
    loader.preprocesar_imagenes()

    X = loader.imagenes.astype(np.float32)
    y = loader.etiquetas

    if limite > 0:
        X = X[:limite]
        y = y[:limite]

    probs = obtener_probabilidades(modelo, X)
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == y))

    return accuracy, preds, y


def main():
    parser = argparse.ArgumentParser(description="Probar el modelo entrenado de reconocimiento de caracteres")
    parser.add_argument('--limite', type=int, default=0, help='Numero maximo de ejemplos a evaluar (0 = todos)')
    args = parser.parse_args()

    modelo_path = Path(PATHS['modelos']) / "modelo_entrenado.pkl"
    modelo = cargar_modelo(modelo_path)
    accuracy, preds, y = evaluar_modelo(modelo, args.limite)

    print("Resultados de la prueba:")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  muestras: {len(y)}")

    if len(y) > 0:
        errores = np.where(preds != y)[0][:10]
        if len(errores) > 0:
            print("  Ejemplos fallidos (max 10):")
            for idx in errores:
                real = DEFAULT_LABEL_MAP.get_label(int(y[idx]))
                pred = DEFAULT_LABEL_MAP.get_label(int(preds[idx]))
                print(f"    indice {idx}: real={real} pred={pred}")
        else:
            print("  Sin errores en la muestra evaluada")


if __name__ == '__main__':
    main()
