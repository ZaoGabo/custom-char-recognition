"""
Entrenador MINIMALISTA - Solo 10,000 imágenes para diagnóstico
"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path
import argparse

# Asegurar que src esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network import NeuralNetwork
from src.label_map import DEFAULT_LABEL_MAP


def entrenar_mini(force=False, verbose=False):
    """Entrenar con solo 10,000 imágenes para diagnóstico"""
    
    # 1. Cargar configuración
    config_path = Path(__file__).parent.parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. Encontrar SOLO 10,000 imágenes (160 por clase aprox)
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    todas_rutas = []
    todas_labels = []
    
    print("Buscando imágenes...")
    for label_num, label_name in DEFAULT_LABEL_MAP.index_to_label.items():
        carpeta = data_dir / label_name
        if not carpeta.exists():
            continue
        
        imagenes = list(carpeta.glob('*.png'))[:160]  # Solo 160 por clase
        todas_rutas.extend(imagenes)
        todas_labels.extend([label_num] * len(imagenes))
    
    print(f"Total: {len(todas_rutas)} imágenes")
    
    # 3. Convertir rutas a strings
    todas_rutas = [str(p) for p in todas_rutas]
    
    # 4. Mezclar
    indices = np.arange(len(todas_rutas))
    np.random.shuffle(indices)
    todas_rutas = [todas_rutas[i] for i in indices]
    todas_labels = [todas_labels[i] for i in indices]
    
    # 5. Split 80/20
    split_idx = int(0.8 * len(todas_rutas))
    train_rutas = todas_rutas[:split_idx]
    train_labels = todas_labels[:split_idx]
    val_rutas = todas_rutas[split_idx:]
    val_labels = todas_labels[split_idx:]
    
    print(f"Train: {len(train_rutas)}, Val: {len(val_rutas)}")
    
    # 6. CARGAR IMÁGENES - SIN augmentation para ir más rápido
    print("\n=== CARGANDO TRAIN ===")
    X_train = _cargar_simple(train_rutas)
    y_train = np.array(train_labels)
    
    print("\n=== CARGANDO VAL ===")
    X_val = _cargar_simple(val_rutas)
    y_val = np.array(val_labels)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # 7. Crear red neuronal
    capas_ocultas = config['network_config']['capas']
    entrada = 784  # 28x28
    salida = DEFAULT_LABEL_MAP.get_num_classes()
    capas = [entrada] + capas_ocultas + [salida]
    
    modelo = NeuralNetwork(
        capas=capas,
        tasa_aprendizaje=config['network_config']['tasa_aprendizaje'],
        lambda_l2=config['network_config']['lambda_l2'],
        dropout_rate=config['network_config']['dropout_rate'],
        use_batch_norm=config['network_config']['use_batch_norm']
    )
    
    print(f"\n=== RED NEURONAL ===")
    print(f"Arquitectura: {capas}")
    
    # 8. ENTRENAR - Solo 50 épocas para diagnóstico
    print("\n=== ENTRENANDO (50 épocas) ===")
    
    # Convertir y_train y y_val a one-hot
    y_train_oh = np.zeros((len(y_train), DEFAULT_LABEL_MAP.get_num_classes()), dtype=np.float32)
    for i, label in enumerate(y_train):
        y_train_oh[i, label] = 1.0
    
    y_val_oh = np.zeros((len(y_val), DEFAULT_LABEL_MAP.get_num_classes()), dtype=np.float32)
    for i, label in enumerate(y_val):
        y_val_oh[i, label] = 1.0
    
    # Entrenar usando el método fit()
    historia = modelo.fit(
        X=X_train,
        Y=y_train_oh,
        epocas=50,
        tamano_lote=32,
        X_val=X_val,
        Y_val=y_val_oh,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO MINI COMPLETADO")
    if historia:
        mejor_val_acc = max(h.get('acc_val', 0) for h in historia)
        print(f"Mejor precisión validación: {mejor_val_acc:.4f}")
    print(f"{'='*60}\n")


def _cargar_simple(rutas):
    """Carga imágenes SIN augmentation, solo normalización"""
    from PIL import Image
    
    n = len(rutas)
    X = np.zeros((n, 784), dtype=np.float32)
    
    for i, ruta in enumerate(rutas):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n}...")
        
        try:
            with Image.open(ruta) as img:
                img_array = np.array(img.convert('L'), dtype=np.float32)
                img_array = img_array.reshape(784) / 255.0
                X[i] = img_array
        except Exception as e:
            print(f"Error en {ruta}: {e}")
            continue
    
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    entrenar_mini(force=args.force, verbose=args.verbose)


if __name__ == '__main__':
    main()
