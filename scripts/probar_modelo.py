"""Script para evaluar el modelo entrenado contra datos en disco."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from src.config import DATA_CONFIG, PATHS
from src.data_loader import DataLoader, _leer_imagen_gris
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork


def cargar_modelo(modelo_path: Path):
    """Cargar el modelo serializado desde ``modelo_path``."""
    if not modelo_path.exists():
        raise FileNotFoundError(f'No se encontro el modelo entrenado en {modelo_path}. Ejecute entrenamiento primero.')
    return NeuralNetwork.cargar_modelo(str(modelo_path))


def evaluar_modelo(modelo, limite: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calcular accuracy del modelo sobre ``data/raw``."""
    loader = DataLoader(PATHS['datos_crudos'], mapa_etiquetas=DEFAULT_LABEL_MAP)
    loader.cargar_desde_directorio()

    # Cargar y procesar imágenes
    imagenes = []
    for ruta in loader.rutas_imagenes:
        img = _leer_imagen_gris(ruta, DATA_CONFIG['tamano_imagen'])
        imagenes.append(img.flatten() / 255.0)

    X = np.array(imagenes, dtype=np.float32)
    y = np.array(loader.etiquetas, dtype=np.int32)

    if limite > 0:
        X = X[:limite]
        y = y[:limite]

    probabilidades = modelo.predecir_probabilidades(X)
    predicciones = np.argmax(probabilidades, axis=1)
    accuracy = float(np.mean(predicciones == y))

    return accuracy, predicciones, y


def main() -> None:
    """Punto de entrada del script CLI."""
    parser = argparse.ArgumentParser(description='Probar el modelo entrenado de reconocimiento de caracteres')
    parser.add_argument('--limite', type=int, default=0, help='Numero maximo de ejemplos a evaluar (0 = todos)')
    args = parser.parse_args()

    print('Cargando modelo...')
    modelo = cargar_modelo(Path(PATHS['modelos']) / 'modelo_entrenado')

    print('Evaluando modelo...')
    accuracy, predicciones, etiquetas = evaluar_modelo(modelo, args.limite)

    print('\n' + '='*60)
    print('RESULTADOS DE LA EVALUACION')
    print('='*60)
    print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Muestras evaluadas: {len(etiquetas)}')
    print(f'Aciertos: {int(accuracy * len(etiquetas))}/{len(etiquetas)}')
    print('='*60)

    if len(etiquetas) > 0:
        errores = np.where(predicciones != etiquetas)[0]
        num_errores = len(errores)
        print(f'\nErrores totales: {num_errores}')

        if num_errores > 0:
            print('\nEjemplos de errores (max 10):')
            print('-'*60)
            for idx in errores[:10]:
                real = DEFAULT_LABEL_MAP.get_label(int(etiquetas[idx]))
                pred = DEFAULT_LABEL_MAP.get_label(int(predicciones[idx]))
                print(f'  Muestra {idx:4d}: Real={real:3s} | Predicho={pred:3s}')
            print('-'*60)
        else:
            print('\n✅ ¡Sin errores! El modelo clasificó correctamente todas las muestras.')


if __name__ == '__main__':
    main()