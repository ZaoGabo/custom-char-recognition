"""Entrenamiento y evaluacion del modelo de reconocimiento de caracteres."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import DATA_CONFIG, NETWORK_CONFIG, PATHS
from src.data_loader import DataLoader
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork
from src.scripts.generar_imagenes_sinteticas import generar_imagenes_sinteticas


def _asegurar_datos() -> None:
    """Verificar que existan datos en ``data/raw`` y generarlos si es necesario."""
    ruta_raw = Path(PATHS['datos_crudos'])
    ruta_raw.mkdir(parents=True, exist_ok=True)
    if not any(ruta_raw.iterdir()):
        print('data/raw esta vacio. Generando imagenes sinteticas...')
        generar_imagenes_sinteticas()

def _one_hot(labels: np.ndarray, num_clases: int) -> np.ndarray:
    """Codificar un vector de etiquetas en formato one-hot."""
    one_hot = np.zeros((labels.shape[0], num_clases), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot

def _construir_modelo(input_dim: int, num_clases: int) -> NeuralNetwork:
    """Instanciar ``NeuralNetwork`` con los parametros del archivo de configuracion."""
    capas_config = NETWORK_CONFIG.get('capas')
    capas = list(capas_config) if capas_config else [NETWORK_CONFIG['oculta_neuronas']]
    capas = [input_dim, *capas, num_clases]

    activaciones = NETWORK_CONFIG.get('activaciones')
    if activaciones and len(activaciones) != len(capas) - 1:
        raise ValueError('La lista de activaciones no coincide con la cantidad de capas - 1')

    return NeuralNetwork(
        capas=capas,
        activaciones=activaciones,
        tasa_aprendizaje=NETWORK_CONFIG.get('tasa_aprendizaje', 0.001),
        lambda_l2=NETWORK_CONFIG.get('lambda_l2', 0.0),
        dropout_rate=NETWORK_CONFIG.get('dropout_rate', 0.0),
        semilla=DATA_CONFIG.get('semilla'),
    )

def evaluar_modelo(modelo: NeuralNetwork, loader: DataLoader, rutas_val: List[str], y_val_idx: List[int]) -> Dict[str, float]:
    """Evaluar el modelo en el conjunto de validación."""
    num_clases = modelo.capas[-1]
    tamano_lote = NETWORK_CONFIG.get('tamano_lote', 64)
    img_height, img_width = DATA_CONFIG['tamano_imagen']
    
    val_gen = loader.generar_lotes(rutas_val, y_val_idx, tamano_lote, (img_height, img_width))
    pasos_val = len(rutas_val) // tamano_lote
    
    perdida_total = 0
    correctos = 0
    total = 0

    for _ in range(pasos_val):
        X_lote, y_lote = next(val_gen)
        y_lote_oh = _one_hot(y_lote, num_clases)
        
        y_pred_prob = modelo.predecir_probabilidades(X_lote)
        perdida_total += modelo.calcular_perdida(y_lote_oh, y_pred_prob)
        
        y_pred_idx = np.argmax(y_pred_prob, axis=1)
        correctos += np.sum(y_pred_idx == y_lote)
        total += len(y_lote)

    return {
        'loss': perdida_total / pasos_val if pasos_val > 0 else 0,
        'accuracy': correctos / total if total > 0 else 0
    }

def entrenar_modelo(force: bool = False, verbose: bool = False) -> None:
    """Flujo completo de entrenamiento y evaluacion."""
    _asegurar_datos()
    
    loader = DataLoader(ruta_datos=PATHS['datos_crudos'])
    loader.cargar_desde_directorio()
    
    rutas_train, rutas_val, y_train_idx, y_val_idx = loader.dividir_datos()
    
    num_clases = DEFAULT_LABEL_MAP.get_num_classes()
    img_height, img_width = DATA_CONFIG['tamano_imagen']
    input_dim = img_height * img_width

    modelo_dir = Path(PATHS['modelos']) / 'modelo_entrenado'
    if modelo_dir.exists() and not force:
        print('Modelo existente encontrado. Use --force para reentrenar.')
        return

    modelo = _construir_modelo(input_dim, num_clases)
    
    train_gen = loader.generar_lotes(
        rutas_train, y_train_idx, 
        NETWORK_CONFIG['tamano_lote'], 
        (img_height, img_width), 
        augment=True
    )
    
    epocas = NETWORK_CONFIG.get('epocas', 100)
    pasos_por_epoca = len(rutas_train) // NETWORK_CONFIG['tamano_lote']

    for epoca in range(epocas):
        perdida_epoca = 0
        for _ in range(pasos_por_epoca):
            X_lote, y_lote = next(train_gen)
            y_lote_oh = _one_hot(y_lote, num_clases)
            
            caches, masks = modelo._forward(X_lote, training=True)
            grads_W, grads_b = modelo._backward(caches, masks, X_lote, y_lote_oh)
            modelo._actualizar_parametros(grads_W, grads_b)
            
            perdida_epoca += modelo.calcular_perdida(y_lote_oh, caches[f'A{modelo.num_capas - 1}'].T)
        
        if verbose:
            print(f"Epoca {epoca+1}/{epocas} - Perdida de entrenamiento: {perdida_epoca / pasos_por_epoca:.4f}")

    modelo.guardar_modelo(str(modelo_dir))
    print(f'Modelo guardado en {modelo_dir}')
    
    print("\nEvaluando modelo en conjunto de validación...")
    metricas = evaluar_modelo(modelo, loader, rutas_val, y_val_idx)
    print(f"Perdida de validación: {metricas['loss']:.4f}")
    print(f"Precisión de validación: {metricas['accuracy']:.4f}")

def parse_args() -> argparse.Namespace:
    """Crear el parser de argumentos para la CLI."""
    parser = argparse.ArgumentParser(description='Entrenamiento de red neuronal')
    parser.add_argument('--force', action='store_true', help='Reentrenar aunque exista un modelo.')
    parser.add_argument('--verbose', action='store_true', help='Mostrar progreso.')
    return parser.parse_args()

def main() -> None:
    """Ejecutar entrenamiento desde la linea de comandos."""
    args = parse_args()
    entrenar_modelo(force=args.force, verbose=args.verbose)

if __name__ == '__main__':
    main()
