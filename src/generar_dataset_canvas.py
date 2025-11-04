"""
Generador de dataset sintético tipo Canvas
Genera imágenes que simulan dibujos del canvas para fine-tuning
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import json
from typing import Tuple, List
from tqdm import tqdm

from src.label_map import LabelMap
from src.config import CUSTOM_LABELS


def generar_imagen_canvas_sintetica(caracter: str, 
                                     canvas_size: int = 280,
                                     stroke_width: int = 25,
                                     variacion: bool = True) -> np.ndarray:
    """
    Generar imagen sintética que simula un dibujo en el canvas
    
    Args:
        caracter: Carácter a dibujar ('A', '5', 'z', etc.)
        canvas_size: Tamaño del canvas temporal
        stroke_width: Grosor del trazo
        variacion: Aplicar variaciones (posición, rotación, escala)
    
    Returns:
        numpy array (28, 28) normalizado [0, 1]
    """
    # Crear canvas grande
    canvas = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(canvas)
    
    # Intentar cargar fuente (si falla, usa default)
    try:
        # Tamaño de fuente base
        font_size = canvas_size // 2
        if variacion:
            # Variación de tamaño ±20%
            font_size = int(font_size * np.random.uniform(0.8, 1.2))
        
        # Intentar varias fuentes
        fuentes_posibles = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
        
        font = None
        for fuente_path in fuentes_posibles:
            if Path(fuente_path).exists():
                font = ImageFont.truetype(fuente_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Calcular posición del texto
    bbox = draw.textbbox((0, 0), caracter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Centrar con variación opcional
    x = (canvas_size - w) // 2
    y = (canvas_size - h) // 2
    
    if variacion:
        # Variación de posición ±15%
        x += int(np.random.uniform(-canvas_size * 0.15, canvas_size * 0.15))
        y += int(np.random.uniform(-canvas_size * 0.15, canvas_size * 0.15))
    
    # Dibujar texto con intensidad variable (simula presión del lápiz)
    intensidad = 255
    if variacion:
        intensidad = int(np.random.uniform(200, 255))
    
    draw.text((x, y), caracter, fill=intensidad, font=font)
    
    # Aplicar transformaciones para simular dibujo a mano
    if variacion:
        # Rotación leve
        angulo = np.random.uniform(-15, 15)
        canvas = canvas.rotate(angulo, fillcolor=0, expand=False)
        
        # Blur variable (simula trazo imperfecto)
        blur_radius = np.random.uniform(0.3, 1.2)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    else:
        # Blur mínimo
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Convertir a array
    canvas_array = np.array(canvas, dtype=np.float32) / 255.0
    
    # Preprocesar igual que en Streamlit
    # 1. Umbralizar
    canvas_bin = (canvas_array > 0.12).astype(np.float32) * canvas_array
    
    # 2. Recortar
    nonzero_rows = np.any(canvas_bin > 0, axis=1)
    nonzero_cols = np.any(canvas_bin > 0, axis=0)
    
    if not np.any(nonzero_rows) or not np.any(nonzero_cols):
        return np.zeros((28, 28), dtype=np.float32)
    
    row_min, row_max = np.where(nonzero_rows)[0][[0, -1]]
    col_min, col_max = np.where(nonzero_cols)[0][[0, -1]]
    
    canvas_crop = canvas_bin[row_min:row_max+1, col_min:col_max+1]
    
    # 3. Redimensionar manteniendo aspect ratio
    h, w = canvas_crop.shape
    if w > h:
        new_w = 20
        new_h = max(1, int(20 * h / w))
    else:
        new_h = 20
        new_w = max(1, int(20 * w / h))
    
    canvas_pil = Image.fromarray((canvas_crop * 255).astype(np.uint8))
    canvas_resize = canvas_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_resize = canvas_resize.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # 4. Centrar en 28x28
    final_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    final_img.paste(canvas_resize, (paste_x, paste_y))
    
    # 5. Convertir y normalizar
    img_array = np.array(final_img, dtype=np.float32) / 255.0
    img_array[img_array < 0.05] = 0.0
    
    return img_array


def generar_dataset_canvas(
    output_dir: str = 'data/canvas_synthetic',
    num_samples_per_class: int = 100,
    seed: int = 42
) -> dict:
    """
    Generar dataset sintético completo tipo canvas
    
    Args:
        output_dir: Directorio de salida
        num_samples_per_class: Muestras por clase
        seed: Semilla aleatoria
    
    Returns:
        dict con estadísticas
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label_map = LabelMap(CUSTOM_LABELS)
    
    print("="*70)
    print("GENERACIÓN DE DATASET SINTÉTICO TIPO CANVAS")
    print("="*70)
    print(f"Clases: {len(CUSTOM_LABELS)}")
    print(f"Muestras por clase: {num_samples_per_class}")
    print(f"Total: {len(CUSTOM_LABELS) * num_samples_per_class}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    X_list = []
    y_list = []
    stats_list = []
    
    for class_idx, caracter in enumerate(tqdm(CUSTOM_LABELS, desc="Generando clases")):
        for sample_idx in range(num_samples_per_class):
            # Generar imagen
            img = generar_imagen_canvas_sintetica(caracter, variacion=True)
            
            # Guardar
            X_list.append(img.flatten())
            y_list.append(class_idx)
            
            # Estadísticas
            stats_list.append({
                'mean': float(np.mean(img)),
                'std': float(np.std(img)),
                'max': float(np.max(img))
            })
    
    # Convertir a arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    # Guardar
    np.save(output_path / 'X_canvas.npy', X)
    np.save(output_path / 'y_canvas.npy', y)
    
    # Estadísticas globales
    stats_global = {
        'num_samples': len(X),
        'num_classes': len(CUSTOM_LABELS),
        'samples_per_class': num_samples_per_class,
        'mean_mean': float(np.mean([s['mean'] for s in stats_list])),
        'mean_std': float(np.mean([s['std'] for s in stats_list])),
        'mean_max': float(np.mean([s['max'] for s in stats_list])),
        'seed': seed
    }
    
    with open(output_path / 'stats.json', 'w') as f:
        json.dump(stats_global, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DATASET GENERADO")
    print(f"{'='*70}")
    print(f"Samples: {stats_global['num_samples']}")
    print(f"Mean promedio: {stats_global['mean_mean']:.3f}")
    print(f"Std promedio: {stats_global['mean_std']:.3f}")
    print(f"Max promedio: {stats_global['mean_max']:.3f}")
    print(f"\nArchivos guardados en: {output_dir}")
    print(f"  - X_canvas.npy: {X.shape}")
    print(f"  - y_canvas.npy: {y.shape}")
    print(f"  - stats.json")
    print(f"{'='*70}")
    
    # Guardar algunas muestras visuales
    save_sample_images(X, y, label_map, output_path / 'samples')
    
    return stats_global


def save_sample_images(X: np.ndarray, y: np.ndarray, 
                       label_map: LabelMap, output_dir: Path, 
                       num_per_class: int = 5):
    """Guardar imágenes de muestra para inspección visual"""
    output_dir.mkdir(exist_ok=True)
    
    for class_idx in range(min(10, len(np.unique(y)))):  # Primeras 10 clases
        indices = np.where(y == class_idx)[0][:num_per_class]
        char = label_map.get_label(class_idx)
        
        # Sanitizar nombre de archivo
        safe_char = char.replace('"', 'dquote').replace('/', 'slash').replace('\\', 'backslash')
        safe_char = safe_char.replace(':', 'colon').replace('*', 'asterisk').replace('?', 'question')
        safe_char = safe_char.replace('<', 'less').replace('>', 'greater').replace('|', 'pipe')
        
        for i, idx in enumerate(indices):
            img = X[idx].reshape(28, 28)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil.save(output_dir / f'class_{class_idx:02d}_{safe_char}_sample_{i}.png')
    
    print(f"\n✅ Muestras visuales guardadas en: {output_dir}")


# === EJECUCIÓN ===
if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    
    # Generar dataset
    stats = generar_dataset_canvas(
        output_dir='data/canvas_synthetic',
        num_samples_per_class=200,  # 200 muestras por clase = 18,800 total
        seed=42
    )
    
    print(f"\n{'='*70}")
    print("SIGUIENTE PASO: FINE-TUNING")
    print(f"{'='*70}")
    print("Usa este dataset para fine-tuning de CNN v2:")
    print("  1. Combinar con EMNIST original (80% EMNIST, 20% Canvas)")
    print("  2. Fine-tune CNN v2 por 20-30 epochs adicionales")
    print("  3. Evaluar mejora en predicciones de canvas real")
    print(f"{'='*70}")
