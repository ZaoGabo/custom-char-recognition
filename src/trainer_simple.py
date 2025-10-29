"""Entrenamiento simplificado sin generadores - carga todo en memoria."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from src.config import DATA_CONFIG, NETWORK_CONFIG, PATHS
from src.data_loader import DataLoader
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork
from src.utils import apply_augmentation, normalize_image


def _cargar_todas_imagenes(rutas: List[str], tamano_imagen: tuple, augment: bool = False) -> np.ndarray:
    """Cargar todas las imágenes en memoria de una vez."""
    print(f"Cargando {len(rutas)} imágenes en memoria...")
    imagenes = []
    
    for i, ruta in enumerate(rutas):
        if (i + 1) % 5000 == 0:
            print(f"  Cargadas {i+1}/{len(rutas)}...")
        
        try:
            with Image.open(ruta) as img_pil:
                imagen = np.array(img_pil.convert("L").resize((tamano_imagen[1], tamano_imagen[0])), dtype=np.uint8)
            
            if augment and DATA_CONFIG.get('usar_augmentacion'):
                imagen = apply_augmentation(imagen)
            
            imagenes.append(imagen)
        except Exception as e:
            print(f"Error cargando {ruta}: {e}")
            # Imagen en blanco si falla
            imagenes.append(np.zeros(tamano_imagen, dtype=np.uint8))
    
    # Normalizar y aplanar
    imagenes_np = np.array(imagenes, dtype=np.float32)
    imagenes_np = normalize_image(imagenes_np)
    imagenes_np = imagenes_np.reshape(imagenes_np.shape[0], -1)
    
    print(f"✅ {len(imagenes)} imágenes cargadas. Shape: {imagenes_np.shape}")
    return imagenes_np


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
        use_batch_norm=NETWORK_CONFIG.get('use_batch_norm', False),
        semilla=DATA_CONFIG.get('semilla'),
    )


def _actualizar_tasa_aprendizaje(epoca: int) -> float:
    """Calcular learning rate con decay."""
    from src.config import NETWORK_CONFIG as cfg
    lr_inicial = cfg.get('tasa_aprendizaje', 0.001)
    scheduler_config = cfg.get('lr_scheduler_config')

    if not scheduler_config or scheduler_config.get('tipo') != 'step_decay':
        return lr_inicial

    tasa_decaimento = scheduler_config.get('tasa_decaimento', 0.1)
    epocas_decaimento = scheduler_config.get('epocas_decaimento', 50)
    
    if epocas_decaimento <= 0:
        return lr_inicial
        
    nueva_lr = lr_inicial * (tasa_decaimento ** (epoca // epocas_decaimento))
    return nueva_lr


def entrenar_modelo_simple(force: bool = False, verbose: bool = False) -> None:
    """Entrenamiento simplificado - carga todo en memoria."""
    
    # Cargar datos
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

    # Cargar TODAS las imágenes en memoria
    print("\n=== CARGANDO DATOS DE ENTRENAMIENTO ===")
    X_train = _cargar_todas_imagenes(rutas_train, (img_height, img_width), augment=True)
    y_train = np.array(y_train_idx)
    y_train_oh = _one_hot(y_train, num_clases)
    
    print("\n=== CARGANDO DATOS DE VALIDACIÓN ===")
    X_val = _cargar_todas_imagenes(rutas_val, (img_height, img_width), augment=False)
    y_val = np.array(y_val_idx)
    y_val_oh = _one_hot(y_val, num_clases)

    # Construir modelo
    print("\n=== CONSTRUYENDO MODELO ===")
    modelo = _construir_modelo(input_dim, num_clases)
    print(f"Arquitectura: {modelo.capas}")
    print(f"Dropout: {modelo.dropout_rate}, L2: {modelo.lambda_l2}, BatchNorm: {modelo.use_batch_norm}")

    # Entrenamiento
    epocas = NETWORK_CONFIG.get('epocas', 100)
    batch_size = NETWORK_CONFIG['tamano_lote']
    patience = NETWORK_CONFIG.get('early_stopping_patience', 10)
    min_delta = NETWORK_CONFIG.get('early_stopping_min_delta', 0.0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = Path(PATHS['modelos']) / 'best_model_temp'
    if best_model_path.exists():
        shutil.rmtree(best_model_path)
    best_model_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== INICIANDO ENTRENAMIENTO ===")
    print(f"Épocas: {epocas}, Batch size: {batch_size}, Early stopping patience: {patience}\n")

    for epoca in range(epocas):
        # Actualizar learning rate
        nueva_lr = _actualizar_tasa_aprendizaje(epoca)
        modelo.set_tasa_aprendizaje(nueva_lr)
        
        # Entrenar una época usando el método fit de la red
        historia = modelo.fit(
            X_train, y_train_oh,
            epocas=1,
            tamano_lote=batch_size,
            barajar=True,
            X_val=X_val,
            Y_val=y_val_oh,
            verbose=False
        )
        
        # Obtener métricas
        train_loss = historia[0]['loss_train']
        val_loss = historia[0]['loss_val']
        val_acc = historia[0]['acc_val']
        
        if verbose:
            print(
                f"Epoca {epoca+1}/{epocas} - "
                f"Perdida de entrenamiento: {train_loss:.4f} - "
                f"Perdida de validación: {val_loss:.4f} - "
                f"Precisión de validación: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            modelo.guardar_modelo(str(best_model_path))
            if verbose:
                print(f'  Mejora encontrada. Guardando mejor modelo en {best_model_path}.')
        else:
            patience_counter += 1
            if verbose:
                print(f'  No hubo mejora. Paciencia: {patience_counter}/{patience}.')

        if patience_counter >= patience:
            print(f'\nDeteniendo entrenamiento (Early Stopping) en la epoca {epoca + 1}.')
            break
    
    # Cargar y guardar el mejor modelo
    if any(best_model_path.iterdir()):
        print(f'\nCargando el mejor modelo con loss de validación: {best_val_loss:.4f}')
        modelo_final = NeuralNetwork.cargar_modelo(str(best_model_path))
        modelo_final.guardar_modelo(str(modelo_dir))
        print(f'Mejor modelo guardado en {modelo_dir}')
        
        # Evaluar
        print('\nEvaluando el mejor modelo en conjunto de validación...')
        Y_val_pred = modelo_final.predecir_probabilidades(X_val)
        val_loss_final = modelo_final.calcular_perdida(y_val_oh, Y_val_pred)
        val_acc_final = NeuralNetwork.calcular_precision(y_val_oh, Y_val_pred)
        print(f'Perdida de validación: {val_loss_final:.4f}')
        print(f'Precisión de validación: {val_acc_final:.4f}')
    else:
        print('\nNo se guardó ningún modelo (no hubo mejoras).')


def main() -> None:
    """Punto de entrada del script."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de reconocimiento de caracteres (SIMPLE)')
    parser.add_argument('--force', action='store_true', help='Forzar reentrenamiento')
    parser.add_argument('--verbose', action='store_true', help='Mostrar progreso detallado')
    args = parser.parse_args()

    entrenar_modelo_simple(force=args.force, verbose=args.verbose)


if __name__ == '__main__':
    main()
