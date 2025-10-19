"""Script para realizar validacion cruzada (K-Fold Cross-Validation) del modelo.

Este script entrena y evalua el modelo usando K-Fold cross-validation para obtener
una estimacion mas robusta del rendimiento del modelo.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

from src.config import DATA_CONFIG, NETWORK_CONFIG, PATHS
from src.data_loader import DataLoader
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork
from src.scripts.generar_imagenes_sinteticas import generar_imagenes_sinteticas


def _one_hot(labels: np.ndarray, num_clases: int) -> np.ndarray:
    """Codificar un vector de etiquetas en formato one-hot."""
    one_hot = np.zeros((labels.shape[0], num_clases), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def _construir_modelo(input_dim: int, num_clases: int) -> NeuralNetwork:
    """Instanciar NeuralNetwork con los parametros del archivo de configuracion."""
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
        use_batch_norm=NETWORK_CONFIG.get('use_batch_norm', False),
        semilla=DATA_CONFIG.get('semilla'),
    )


def _actualizar_tasa_aprendizaje(epoca: int, lr_inicial: float) -> float:
    """Actualizar la tasa de aprendizaje segun el scheduler configurado."""
    scheduler_config = NETWORK_CONFIG.get('lr_scheduler_config')

    if not scheduler_config or scheduler_config.get('tipo') != 'step_decay':
        return lr_inicial

    tasa_decaimento = scheduler_config.get('tasa_decaimento', 0.1)
    epocas_decaimento = scheduler_config.get('epocas_decaimento', 50)

    if epocas_decaimento <= 0:
        return lr_inicial

    nueva_lr = lr_inicial * (tasa_decaimento ** (epoca // epocas_decaimento))
    return nueva_lr


def crear_folds(
    rutas: List[str],
    etiquetas: List[int],
    k: int = 5,
    semilla: int = 42
) -> List[Tuple[List[str], List[int], List[str], List[int]]]:
    """
    Dividir datos en K folds para cross-validation.

    Returns:
        Lista de tuplas (rutas_train, y_train, rutas_val, y_val) para cada fold
    """
    rng = np.random.default_rng(semilla)

    # Convertir a arrays numpy
    rutas_arr = np.array(rutas)
    etiquetas_arr = np.array(etiquetas)

    # Obtener indices y mezclar
    indices = np.arange(len(rutas))
    rng.shuffle(indices)

    # Dividir indices en K folds
    fold_size = len(indices) // k
    folds = []

    for i in range(k):
        # Indices de validacion para este fold
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else len(indices)
        val_indices = indices[val_start:val_end]

        # Indices de entrenamiento (todos menos validacion)
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        # Crear fold
        rutas_train = rutas_arr[train_indices].tolist()
        y_train = etiquetas_arr[train_indices].tolist()
        rutas_val = rutas_arr[val_indices].tolist()
        y_val = etiquetas_arr[val_indices].tolist()

        folds.append((rutas_train, y_train, rutas_val, y_val))

    return folds


def entrenar_fold(
    fold_num: int,
    rutas_train: List[str],
    y_train: List[int],
    rutas_val: List[str],
    y_val: List[int],
    loader: DataLoader,
    num_clases: int,
    input_dim: int,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Entrenar y evaluar el modelo en un fold especifico.

    Returns:
        Diccionario con metricas de validacion
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num + 1}")
    print(f"{'='*60}")
    print(f"Muestras de entrenamiento: {len(rutas_train)}")
    print(f"Muestras de validacion: {len(rutas_val)}")

    # Crear modelo
    modelo = _construir_modelo(input_dim, num_clases)
    lr_inicial = NETWORK_CONFIG.get('tasa_aprendizaje', 0.001)

    # Configuracion de entrenamiento
    epocas = NETWORK_CONFIG.get('epocas', 100)
    tamano_lote = NETWORK_CONFIG.get('tamano_lote', 32)
    patience = NETWORK_CONFIG.get('early_stopping_patience', 10)
    min_delta = NETWORK_CONFIG.get('early_stopping_min_delta', 0.0)

    img_height, img_width = DATA_CONFIG['tamano_imagen']

    # Generador de entrenamiento
    train_gen = loader.generar_lotes(
        rutas_train, y_train,
        tamano_lote,
        (img_height, img_width),
        augment=True
    )

    pasos_por_epoca = len(rutas_train) // tamano_lote

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    for epoca in range(epocas):
        # Actualizar learning rate
        nueva_lr = _actualizar_tasa_aprendizaje(epoca, lr_inicial)
        modelo.set_tasa_aprendizaje(nueva_lr)

        # Entrenar
        perdida_epoca = 0
        for _ in range(pasos_por_epoca):
            X_lote, y_lote = next(train_gen)
            y_lote_oh = _one_hot(y_lote, num_clases)

            caches, masks, bn_caches = modelo._forward(X_lote, training=True)
            grads_W, grads_b, grads_gamma, grads_beta = modelo._backward(
                caches, masks, bn_caches, X_lote, y_lote_oh
            )
            modelo._actualizar_parametros(grads_W, grads_b, grads_gamma, grads_beta)

            perdida_epoca += modelo.calcular_perdida(y_lote_oh, caches[f'A{modelo.num_capas - 1}'].T)

        # Evaluar en validacion
        val_gen = loader.generar_lotes(rutas_val, y_val, tamano_lote, (img_height, img_width))
        pasos_val = len(rutas_val) // tamano_lote

        perdida_val = 0
        correctos = 0
        total = 0

        for _ in range(pasos_val):
            X_lote, y_lote = next(val_gen)
            y_lote_oh = _one_hot(y_lote, num_clases)

            y_pred_prob = modelo.predecir_probabilidades(X_lote)
            perdida_val += modelo.calcular_perdida(y_lote_oh, y_pred_prob)

            y_pred_idx = np.argmax(y_pred_prob, axis=1)
            correctos += np.sum(y_pred_idx == y_lote)
            total += len(y_lote)

        val_loss = perdida_val / pasos_val if pasos_val > 0 else 0
        val_acc = correctos / total if total > 0 else 0

        if verbose and ((epoca + 1) % 10 == 0 or epoca == 0):
            print(f"Epoca {epoca+1}/{epocas} - "
                  f"Loss train: {perdida_epoca/pasos_por_epoca:.4f} - "
                  f"Loss val: {val_loss:.4f} - "
                  f"Acc val: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Guardar pesos
            best_weights = {
                'pesos': [p.copy() for p in modelo.pesos],
                'sesgos': [s.copy() for s in modelo.sesgos],
            }
            if modelo.use_batch_norm:
                best_weights['gammas'] = [g.copy() for g in modelo.gammas]
                best_weights['betas'] = [b.copy() for b in modelo.betas]
                best_weights['running_means'] = [rm.copy() for rm in modelo.running_means]
                best_weights['running_vars'] = [rv.copy() for rv in modelo.running_vars]
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping en epoca {epoca + 1}")
            break

    # Restaurar mejores pesos
    if best_weights:
        for i, p in enumerate(best_weights['pesos']):
            modelo.pesos[i] = p
        for i, s in enumerate(best_weights['sesgos']):
            modelo.sesgos[i] = s
        if modelo.use_batch_norm:
            for i, g in enumerate(best_weights['gammas']):
                modelo.gammas[i] = g
            for i, b in enumerate(best_weights['betas']):
                modelo.betas[i] = b
            for i, rm in enumerate(best_weights['running_means']):
                modelo.running_means[i] = rm
            for i, rv in enumerate(best_weights['running_vars']):
                modelo.running_vars[i] = rv

    # Evaluacion final en validacion
    val_gen = loader.generar_lotes(rutas_val, y_val, tamano_lote, (img_height, img_width))
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

    final_loss = perdida_total / pasos_val if pasos_val > 0 else 0
    final_acc = correctos / total if total > 0 else 0

    print(f"\nResultados Fold {fold_num + 1}:")
    print(f"  Loss de validacion: {final_loss:.4f}")
    print(f"  Precision de validacion: {final_acc:.4f} ({final_acc*100:.2f}%)")

    return {
        'fold': fold_num + 1,
        'val_loss': final_loss,
        'val_accuracy': final_acc
    }


def cross_validation(k: int = 5, verbose: bool = False) -> Dict:
    """
    Realizar validacion cruzada K-Fold.

    Args:
        k: Numero de folds
        verbose: Mostrar progreso detallado

    Returns:
        Diccionario con resultados de cross-validation
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION {k}-FOLD")
    print(f"{'='*60}\n")

    # Asegurar que existan datos
    ruta_raw = Path(PATHS['datos_crudos'])
    ruta_raw.mkdir(parents=True, exist_ok=True)
    if not any(ruta_raw.iterdir()):
        print('data/raw esta vacio. Generando imagenes sinteticas...')
        generar_imagenes_sinteticas()

    # Cargar datos
    loader = DataLoader(ruta_datos=PATHS['datos_crudos'])
    loader.cargar_desde_directorio()

    num_clases = DEFAULT_LABEL_MAP.get_num_classes()
    img_height, img_width = DATA_CONFIG['tamano_imagen']
    input_dim = img_height * img_width

    # Crear folds
    print(f"Creando {k} folds...")
    folds = crear_folds(
        loader.rutas_imagenes,
        loader.etiquetas,
        k=k,
        semilla=DATA_CONFIG.get('semilla', 42)
    )

    # Entrenar en cada fold
    resultados = []
    for i, (rutas_train, y_train, rutas_val, y_val) in enumerate(folds):
        resultado_fold = entrenar_fold(
            i, rutas_train, y_train, rutas_val, y_val,
            loader, num_clases, input_dim, verbose
        )
        resultados.append(resultado_fold)

    # Calcular estadisticas
    losses = [r['val_loss'] for r in resultados]
    accuracies = [r['val_accuracy'] for r in resultados]

    stats = {
        'k_folds': k,
        'resultados_por_fold': resultados,
        'loss_promedio': float(np.mean(losses)),
        'loss_std': float(np.std(losses)),
        'accuracy_promedio': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'accuracy_min': float(np.min(accuracies)),
        'accuracy_max': float(np.max(accuracies))
    }

    # Mostrar resumen
    print(f"\n{'='*60}")
    print(f"RESUMEN CROSS-VALIDATION {k}-FOLD")
    print(f"{'='*60}")
    print(f"\nPrecision por fold:")
    for r in resultados:
        print(f"  Fold {r['fold']}: {r['val_accuracy']:.4f} ({r['val_accuracy']*100:.2f}%)")

    print(f"\n{'='*60}")
    print(f"ESTADISTICAS FINALES:")
    print(f"{'='*60}")
    print(f"Loss promedio:       {stats['loss_promedio']:.4f} ± {stats['loss_std']:.4f}")
    print(f"Precision promedio:  {stats['accuracy_promedio']:.4f} ({stats['accuracy_promedio']*100:.2f}%)")
    print(f"Desviacion estandar: {stats['accuracy_std']:.4f} ({stats['accuracy_std']*100:.2f}%)")
    print(f"Precision minima:    {stats['accuracy_min']:.4f} ({stats['accuracy_min']*100:.2f}%)")
    print(f"Precision maxima:    {stats['accuracy_max']:.4f} ({stats['accuracy_max']*100:.2f}%)")
    print(f"{'='*60}\n")

    # Guardar resultados
    output_path = Path(PATHS['modelos']) / 'cross_validation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Resultados guardados en: {output_path}")

    return stats


def parse_args() -> argparse.Namespace:
    """Crear el parser de argumentos para la CLI."""
    parser = argparse.ArgumentParser(description='Cross-validation del modelo')
    parser.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Numero de folds para cross-validation (default: 5)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar progreso detallado'
    )
    return parser.parse_args()


def main() -> None:
    """Ejecutar cross-validation desde la linea de comandos."""
    args = parse_args()
    cross_validation(k=args.k_folds, verbose=args.verbose)


if __name__ == '__main__':
    main()

