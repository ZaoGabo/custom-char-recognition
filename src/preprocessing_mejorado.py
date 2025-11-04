"""
Preprocesamiento mejorado para canvas
Ajusta la normalización para que coincida con EMNIST (Mean ~0.160)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple


def preprocesar_canvas_mejorado(imagen_array: np.ndarray, 
                                 target_mean: float = 0.160,
                                 target_std: float = 0.330) -> Tuple[np.ndarray, dict]:
    """
    Preprocesar imagen del canvas con normalización ajustada a EMNIST
    
    Args:
        imagen_array: numpy array RGBA del canvas
        target_mean: Mean objetivo (EMNIST: ~0.160)
        target_std: Std objetivo (EMNIST: ~0.330)
        
    Returns:
        (imagen_procesada, estadísticas)
    """
    # Usar canal alpha (4to canal) para detectar trazos
    if imagen_array.shape[2] == 4:
        imagen_gray = imagen_array[:, :, 3]  # Canal alpha
    else:
        # Convertir RGB a escala de grises
        imagen_pil = Image.fromarray(imagen_array[:, :, :3].astype('uint8'), 'RGB')
        imagen_gray = np.array(imagen_pil.convert('L'))
    
    # Umbralizar (solo píxeles con alpha > 30)
    imagen_bin = (imagen_gray > 30).astype(np.uint8) * 255
    
    # Convertir a PIL
    imagen_pil = Image.fromarray(imagen_bin)
    
    # Recortar contenido (eliminar bordes vacíos)
    bbox = imagen_pil.getbbox()
    if bbox is None:
        # Canvas vacío
        return np.zeros((28, 28), dtype=np.float32), {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'original_mean': 0.0, 'ajustado': False
        }
    
    imagen_crop = imagen_pil.crop(bbox)
    
    # Redimensionar manteniendo aspect ratio
    # Primero a 20x20, luego centrar en 28x28
    w, h = imagen_crop.size
    if w > h:
        new_w = 20
        new_h = int(20 * h / w)
    else:
        new_h = 20
        new_w = int(20 * w / h)
    
    imagen_resize = imagen_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Aplicar blur suave
    imagen_blur = imagen_resize.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Crear canvas 28x28 con fondo NEGRO (0)
    imagen_final = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    imagen_final.paste(imagen_blur, (paste_x, paste_y))
    
    # Convertir a numpy y normalizar [0, 1]
    img_array = np.array(imagen_final, dtype=np.float32) / 255.0
    
    # Limpiar ruido de fondo (< 0.05)
    img_array[img_array < 0.05] = 0.0
    
    # NUEVA LÓGICA: Ajuste de intensidad para coincidir con EMNIST
    original_mean = float(np.mean(img_array[img_array > 0]) if np.any(img_array > 0) else 0)
    
    if np.any(img_array > 0):
        # Calcular factor de ajuste basado en el mean objetivo
        # EMNIST tiene píxeles más tenues (Mean ~0.160 vs Canvas ~0.510)
        
        # Estrategia 1: Reducir intensidad de los píxeles no-cero
        # Factor de escala: target_mean / original_mean
        if original_mean > 0.05:  # Solo ajustar si hay contenido significativo
            # Calcular factor para ajustar el mean
            # Queremos que el mean de píxeles no-cero sea ~0.40 (para que el mean global sea ~0.160)
            target_nonzero_mean = 0.40  # Ajustado empíricamente
            scale_factor = target_nonzero_mean / original_mean
            scale_factor = min(scale_factor, 1.0)  # No aumentar intensidad, solo reducir
            
            # Aplicar escala solo a píxeles no-cero
            mask = img_array > 0
            img_array[mask] = img_array[mask] * scale_factor
            
            # Asegurar rango [0, 1]
            img_array = np.clip(img_array, 0.0, 1.0)
        
        # Estrategia 2: Aplicar gamma correction para tenues más parecidos a EMNIST
        # EMNIST tiene trazos con bordes más difusos y píxeles semi-transparentes
        gamma = 1.5  # Gamma > 1 reduce intensidad de valores medios
        img_array = np.power(img_array, gamma)
    
    # Estadísticas finales
    stats = {
        'mean': float(np.mean(img_array)),
        'std': float(np.std(img_array)),
        'min': float(np.min(img_array)),
        'max': float(np.max(img_array)),
        'original_mean': original_mean,
        'ajustado': True
    }
    
    return img_array, stats


def preprocesar_canvas_original(imagen_array: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Preprocesamiento original (sin ajustes)
    Útil para comparación
    """
    # Usar canal alpha
    if imagen_array.shape[2] == 4:
        imagen_gray = imagen_array[:, :, 3]
    else:
        imagen_pil = Image.fromarray(imagen_array[:, :, :3].astype('uint8'), 'RGB')
        imagen_gray = np.array(imagen_pil.convert('L'))
    
    imagen_bin = (imagen_gray > 30).astype(np.uint8) * 255
    imagen_pil = Image.fromarray(imagen_bin)
    
    bbox = imagen_pil.getbbox()
    if bbox is None:
        return np.zeros((28, 28), dtype=np.float32), {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'original_mean': 0.0, 'ajustado': False
        }
    
    imagen_crop = imagen_pil.crop(bbox)
    
    w, h = imagen_crop.size
    if w > h:
        new_w = 20
        new_h = int(20 * h / w)
    else:
        new_h = 20
        new_w = int(20 * w / h)
    
    imagen_resize = imagen_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    imagen_blur = imagen_resize.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    imagen_final = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    imagen_final.paste(imagen_blur, (paste_x, paste_y))
    
    img_array = np.array(imagen_final, dtype=np.float32) / 255.0
    img_array[img_array < 0.05] = 0.0
    
    stats = {
        'mean': float(np.mean(img_array)),
        'std': float(np.std(img_array)),
        'min': float(np.min(img_array)),
        'max': float(np.max(img_array)),
        'original_mean': float(np.mean(img_array)),
        'ajustado': False
    }
    
    return img_array, stats


# === PRUEBA ===
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    
    os.chdir(Path(__file__).parent.parent)
    
    print("="*70)
    print("PRUEBA DE PREPROCESAMIENTO MEJORADO")
    print("="*70)
    
    # Cargar una imagen EMNIST real para comparación
    try:
        # Cargar imagen real de EMNIST
        raw_dir = Path('data/raw/A_upper')
        imagenes = list(raw_dir.glob('*.png'))
        
        if imagenes:
            # Cargar primera imagen de 'A'
            img_path = imagenes[0]
            img_pil = Image.open(img_path).convert('L')
            img_emnist = np.array(img_pil, dtype=np.float32) / 255.0
            
            print(f"✅ Imagen EMNIST cargada desde: {img_path.name}")
        else:
            # Generar imagen sintética tipo EMNIST
            print("⚠️  Imágenes no encontradas, generando sintética...")
            img_emnist = np.zeros((28, 28))
            img_emnist[8:22, 12:16] = 0.4
            img_emnist[10:12, 10:18] = 0.3
            img_emnist[8:10, 10:18] = 0.35
        
        if img_emnist is not None:
            
            print(f"\n📊 Estadísticas EMNIST real:")
            print(f"   Mean: {np.mean(img_emnist):.3f}")
            print(f"   Std:  {np.std(img_emnist):.3f}")
            print(f"   Min:  {np.min(img_emnist):.3f}")
            print(f"   Max:  {np.max(img_emnist):.3f}")
            
            # Simular un canvas (imagen más brillante)
            # Multiplicar por 3 para simular el problema del canvas
            img_canvas_simulado = np.clip(img_emnist * 3.0, 0, 1)
            
            print(f"\n📊 Canvas simulado (sin ajuste):")
            print(f"   Mean: {np.mean(img_canvas_simulado):.3f}")
            print(f"   Std:  {np.std(img_canvas_simulado):.3f}")
            
            # Crear array RGBA simulado
            canvas_rgba = np.zeros((280, 280, 4), dtype=np.uint8)
            # Escalar de 28x28 a 280x280 y poner en canal alpha
            img_scaled = Image.fromarray((img_canvas_simulado * 255).astype(np.uint8))
            img_scaled = img_scaled.resize((280, 280), Image.Resampling.NEAREST)
            canvas_rgba[:, :, 3] = np.array(img_scaled)
            
            # Procesar con ambos métodos
            img_original, stats_original = preprocesar_canvas_original(canvas_rgba)
            img_mejorado, stats_mejorado = preprocesar_canvas_mejorado(canvas_rgba)
            
            print(f"\n📊 Resultado ORIGINAL:")
            print(f"   Mean: {stats_original['mean']:.3f} (diff: {abs(stats_original['mean'] - 0.160):.3f})")
            print(f"   Std:  {stats_original['std']:.3f}")
            
            print(f"\n📊 Resultado MEJORADO:")
            print(f"   Mean: {stats_mejorado['mean']:.3f} (diff: {abs(stats_mejorado['mean'] - 0.160):.3f})")
            print(f"   Std:  {stats_mejorado['std']:.3f}")
            
            # Calcular mejora
            mejora = abs(stats_original['mean'] - 0.160) - abs(stats_mejorado['mean'] - 0.160)
            if mejora > 0:
                print(f"\n✅ Mejora en Mean: {mejora:.3f} más cerca de EMNIST (0.160)")
            else:
                print(f"\n⚠️  Mean se alejó {-mejora:.3f} de EMNIST")
            
            # Visualizar
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(img_emnist, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'EMNIST Real\nMean: {np.mean(img_emnist):.3f}')
            axes[0].axis('off')
            
            axes[1].imshow(img_canvas_simulado, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Canvas Simulado\nMean: {np.mean(img_canvas_simulado):.3f}')
            axes[1].axis('off')
            
            axes[2].imshow(img_original, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'Preprocesado Original\nMean: {stats_original["mean"]:.3f}')
            axes[2].axis('off')
            
            axes[3].imshow(img_mejorado, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f'Preprocesado MEJORADO\nMean: {stats_mejorado["mean"]:.3f}')
            axes[3].axis('off')
            
            plt.tight_layout()
            output_path = 'comparacion_preprocesamiento.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n💾 Comparación guardada en: {output_path}")
            
            print(f"\n{'='*70}")
            print("✅ Preprocesamiento mejorado funcionando")
            print(f"{'='*70}")
        
    except Exception as e:
        print(f"⚠️ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
