"""Script para evaluar el modelo entrenado contra datos en disco."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import DATA_CONFIG, PATHS
from src.data_loader import DataLoader
from src.label_map import DEFAULT_LABEL_MAP


def cargar_modelo(modelo_path: Path):
    """Cargar el modelo serializado desde ``modelo_path``."""
    if not modelo_path.exists():
        raise FileNotFoundError('No se encontro el modelo entrenado. Ejecute `python -m src.trainer --force`.')
    with modelo_path.open('rb') as archivo:
        import pickle

        return pickle.load(archivo)


def preprocesar_imagen(ruta_imagen: Path) -> np.ndarray:
    """Abrir una imagen, convertirla a escala de grises y normalizarla."""
    try:
        with Image.open(ruta_imagen) as imagen:
            procesada = imagen.convert('L').resize(DATA_CONFIG['tamano_imagen'])
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f'No se pudo procesar {ruta_imagen}') from exc
    return np.asarray(procesada, dtype=np.float32).flatten() / 255.0


def obtener_probabilidades(modelo, entrada: np.ndarray) -> np.ndarray:
    """Obtener probabilidades en formato (1, clases) de ``modelo``."""
    if hasattr(modelo, 'predecir_probabilidades'):
        salida = modelo.predecir_probabilidades(entrada)
    else:
        salida = modelo.predecir(entrada)
    return salida if salida.ndim == 2 else salida.reshape(1, -1)


def evaluar_modelo(modelo, limite: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calcular accuracy del modelo sobre ``data/raw``."""
    loader = DataLoader(PATHS['datos_crudos'], mapa_etiquetas=DEFAULT_LABEL_MAP)
    loader.cargar_desde_directorio()
    loader.preprocesar_imagenes()

    X = loader.imagenes.astype(np.float32)
    y = loader.etiquetas

    if limite > 0:
        X = X[:limite]
        y = y[:limite]

    probabilidades = obtener_probabilidades(modelo, X)
    predicciones = np.argmax(probabilidades, axis=1)
    accuracy = float(np.mean(predicciones == y))

    return accuracy, predicciones, y


def main() -> None:
    """Punto de entrada del script CLI."""
    parser = argparse.ArgumentParser(description='Probar el modelo entrenado de reconocimiento de caracteres')
    parser.add_argument('--limite', type=int, default=0, help='Numero maximo de ejemplos a evaluar (0 = todos)')
    args = parser.parse_args()

    modelo = cargar_modelo(Path(PATHS['modelos']) / 'modelo_entrenado.pkl')
    accuracy, predicciones, etiquetas = evaluar_modelo(modelo, args.limite)

    print('Resultados de la prueba:')
    print(f'  accuracy: {accuracy:.4f}')
    print(f'  muestras: {len(etiquetas)}')

    if len(etiquetas) > 0:
        errores = np.where(predicciones != etiquetas)[0][:10]
        if errores.size > 0:
            print('  Ejemplos fallidos (max 10):')
            for idx in errores:
                real = DEFAULT_LABEL_MAP.get_label(int(etiquetas[idx]))
                pred = DEFAULT_LABEL_MAP.get_label(int(predicciones[idx]))
                print(f'    indice {idx}: real={real} pred={pred}')
        else:
            print('  Sin errores en la muestra evaluada')


if __name__ == '__main__':
    main()