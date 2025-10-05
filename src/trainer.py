import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.config import DATA_CONFIG, NETWORK_CONFIG, PATHS
from src.data_loader import DataLoader
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork
from src.scripts.generar_imagenes_sinteticas import generar_imagenes_sinteticas


def _asegurar_datos() -> Tuple[int, int]:
    ruta_raw = Path(PATHS['datos_crudos'])
    ruta_raw.mkdir(parents=True, exist_ok=True)

    carpetas = [d for d in ruta_raw.iterdir() if d.is_dir() and not d.name.startswith('.')]
    total_imagenes = 0
    carpetas_con_imagenes = 0

    for carpeta in carpetas:
        imagenes = [f for f in carpeta.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
        if imagenes:
            carpetas_con_imagenes += 1
            total_imagenes += len(imagenes)

    if total_imagenes == 0:
        print("data/raw esta vacio. Generando imagenes sinteticas...")
        generar_imagenes_sinteticas()
        return _asegurar_datos()

    print(f"data/raw contiene {total_imagenes} imagenes en {carpetas_con_imagenes} carpetas.")
    return total_imagenes, carpetas_con_imagenes


def _one_hot(labels: np.ndarray, num_clases: int) -> np.ndarray:
    one_hot = np.zeros((labels.shape[0], num_clases), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def preparar_datos(aplicar_augmentacion: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _asegurar_datos()
    loader = DataLoader(ruta_datos=PATHS['datos_crudos'], mapa_etiquetas=DEFAULT_LABEL_MAP)
    loader.cargar_desde_directorio()

    if len(loader.imagenes) == 0:
        raise RuntimeError("No se pudieron cargar imagenes despues de generar datos sinteticos.")

    loader.preprocesar_imagenes()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.dividir_datos()

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if aplicar_augmentacion and DATA_CONFIG.get('usar_augmentacion', False):
        X_train, y_train = loader.aplicar_augmentacion(X_train, y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _construir_modelo(input_dim: int, num_clases: int) -> NeuralNetwork:
    capas_config = NETWORK_CONFIG.get('capas')
    if capas_config:
        capas = list(capas_config)
        capas[0] = input_dim
        capas[-1] = num_clases
    else:
        capas = [input_dim, NETWORK_CONFIG['oculta_neuronas'], num_clases]

    activaciones = NETWORK_CONFIG.get('activaciones')
    if activaciones and len(activaciones) != len(capas) - 1:
        raise ValueError("La lista de activaciones no coincide con la cantidad de capas - 1")

    return NeuralNetwork(
        capas=capas,
        activaciones=activaciones,
        tasa_aprendizaje=NETWORK_CONFIG.get('tasa_aprendizaje', 0.001),
        lambda_l2=NETWORK_CONFIG.get('lambda_l2', 0.0),
        dropout_rate=NETWORK_CONFIG.get('dropout_rate', 0.0),
        beta1=NETWORK_CONFIG.get('beta1', 0.9),
        beta2=NETWORK_CONFIG.get('beta2', 0.999),
        epsilon=NETWORK_CONFIG.get('epsilon', 1e-8),
        semilla=DATA_CONFIG.get('semilla'),
    )


def _evaluar(modelo: NeuralNetwork, X: np.ndarray, y_true_oh: np.ndarray) -> Dict[str, float]:
    y_prob = modelo.predecir_probabilidades(X)
    precision = NeuralNetwork.calcular_precision(y_true_oh, y_prob)
    perdida = modelo._calcular_perdida(y_true_oh, y_prob)  # pylint: disable=protected-access
    return {"loss": float(perdida), "accuracy": float(precision)}


def entrenar_modelo(force: bool = False, verbose: bool = False) -> Tuple[NeuralNetwork, Dict[str, Dict[str, float]]]:
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(aplicar_augmentacion=True)

    num_clases = DEFAULT_LABEL_MAP.get_num_classes()
    y_train_oh = _one_hot(y_train, num_clases)
    y_val_oh = _one_hot(y_val, num_clases)
    y_test_oh = _one_hot(y_test, num_clases)

    modelo_path = Path(PATHS['modelos']) / "modelo_entrenado.pkl"
    modelo_path.parent.mkdir(parents=True, exist_ok=True)

    if modelo_path.exists() and not force:
        print("Modelo existente encontrado. Use --force para reentrenar.")
        with modelo_path.open('rb') as f:
            modelo = pickle.load(f)
    else:
        modelo = _construir_modelo(X_train.shape[1], num_clases)
        print("Entrenando modelo...")
        modelo.fit(
            X_train,
            y_train_oh,
            epocas=NETWORK_CONFIG.get('epocas', 100),
            tamano_lote=NETWORK_CONFIG.get('tamano_lote', 64),
            X_val=X_val,
            Y_val=y_val_oh,
            verbose=verbose,
        )
        with modelo_path.open('wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo guardado en {modelo_path}")

    metricas = {
        'train': _evaluar(modelo, X_train, y_train_oh),
        'val': _evaluar(modelo, X_val, y_val_oh),
        'test': _evaluar(modelo, X_test, y_test_oh),
    }

    y_pred_train = modelo.predecir(X_train)
    y_pred_val = modelo.predecir(X_val)
    y_pred_test = modelo.predecir(X_test)

    metricas['train']['accuracy'] = float(np.mean(y_pred_train == y_train))
    metricas['val']['accuracy'] = float(np.mean(y_pred_val == y_val))
    metricas['test']['accuracy'] = float(np.mean(y_pred_test == y_test))

    return modelo, metricas


def cargar_modelo_guardado() -> NeuralNetwork:
    modelo_path = Path(PATHS['modelos']) / "modelo_entrenado.pkl"
    if not modelo_path.exists():
        raise FileNotFoundError("No se encontro el modelo entrenado. Ejecute `python -m src.trainer --force`.")
    with modelo_path.open('rb') as f:
        return pickle.load(f)


def evaluar_modelo(modelo: NeuralNetwork) -> Dict[str, Dict[str, float]]:
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(aplicar_augmentacion=False)
    num_clases = DEFAULT_LABEL_MAP.get_num_classes()

    metricas = {}
    for nombre, X_split, y_split in (
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test),
    ):
        y_one_hot = _one_hot(y_split, num_clases)
        resultados = _evaluar(modelo, X_split, y_one_hot)
        y_pred = modelo.predecir(X_split)
        resultados['accuracy'] = float(np.mean(y_pred == y_split))
        metricas[nombre] = resultados

    return metricas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de red neuronal para caracteres personalizados")
    parser.add_argument('--force', action='store_true', help='Reentrenar aunque exista un modelo guardado')
    parser.add_argument('--verbose', action='store_true', help='Mostrar progreso detallado de entrenamiento')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, metricas = entrenar_modelo(force=args.force, verbose=args.verbose)
    print("Resultados:")
    for split, valores in metricas.items():
        loss = valores['loss']
        acc = valores['accuracy']
        print(f"  {split}: loss={loss:.4f} accuracy={acc:.4f}")


if __name__ == '__main__':
    main()
