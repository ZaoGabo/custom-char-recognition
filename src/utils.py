"""Funciones auxiliares para el proyecto."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from .config import AUGMENTATION_CONFIG

AUG = {
    'rotation_range': AUGMENTATION_CONFIG.get('rango_rotacion', 0),
    'width_shift_range': AUGMENTATION_CONFIG.get('desplazamiento_horizontal', 0),
    'height_shift_range': AUGMENTATION_CONFIG.get('desplazamiento_vertical', 0),
    'zoom_range': AUGMENTATION_CONFIG.get('rango_zoom', 0),
    'horizontal_flip': AUGMENTATION_CONFIG.get('voltear_horizontal', False),
    'vertical_flip': AUGMENTATION_CONFIG.get('voltear_vertical', False),
    'blur_radius': AUGMENTATION_CONFIG.get('radio_blur', 0),
    'noise_level': AUGMENTATION_CONFIG.get('nivel_ruido', 0),
}


def normalize_image(images: np.ndarray) -> np.ndarray:
    """Normalizar imagenes al rango [0, 1]."""
    return images.astype(np.float32) / 255.0


def denormalize_image(images: np.ndarray) -> np.ndarray:
    """Escalar imagenes normalizadas al rango [0, 255]."""
    return (images * 255).astype(np.uint8)


def apply_augmentation(image: np.ndarray) -> np.ndarray:
    """Aplicar augmentacion agresiva a ``image`` usando configuracion global.

    Incluye rotación, desplazamiento, zoom, flips, blur gaussiano y ruido gaussiano.
    """
    if image.ndim == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        pil_img = Image.fromarray(image)

    # Rotación agresiva
    if AUG['rotation_range'] > 0:
        angle = np.random.uniform(-AUG['rotation_range'], AUG['rotation_range'])
        pil_img = pil_img.rotate(angle, fillcolor=0)

    # Desplazamiento horizontal
    if AUG['width_shift_range'] > 0:
        width, height = pil_img.size
        shift_x = int(np.random.uniform(-AUG['width_shift_range'] * width, AUG['width_shift_range'] * width))
        pil_img = ImageOps.expand(pil_img, border=(abs(shift_x), 0, 0, 0), fill=0)
        crop_box = (shift_x, 0, width + shift_x, height) if shift_x > 0 else (0, 0, width, height)
        pil_img = pil_img.crop(crop_box)

    # Desplazamiento vertical
    if AUG['height_shift_range'] > 0:
        width, height = pil_img.size
        shift_y = int(np.random.uniform(-AUG['height_shift_range'] * height, AUG['height_shift_range'] * height))
        pil_img = ImageOps.expand(pil_img, border=(0, abs(shift_y), 0, 0), fill=0)
        crop_box = (0, shift_y, width, height + shift_y) if shift_y > 0 else (0, 0, width, height)
        pil_img = pil_img.crop(crop_box)

    # Zoom
    if AUG['zoom_range'] > 0:
        zoom = 1 + np.random.uniform(-AUG['zoom_range'], AUG['zoom_range'])
        width, height = pil_img.size
        new_width, new_height = int(width * zoom), int(height * zoom)
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        if zoom > 1:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            pil_img = pil_img.crop((left, top, left + width, top + height))
        else:
            border_x_left = (width - new_width) // 2
            border_x_right = width - new_width - border_x_left
            border_y_top = (height - new_height) // 2
            border_y_bottom = height - new_height - border_y_top
            padding = (border_x_left, border_y_top, border_x_right, border_y_bottom)
            pil_img = ImageOps.expand(pil_img, border=padding, fill=0)

    # Flips horizontales y verticales
    if AUG['horizontal_flip'] and np.random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    if AUG['vertical_flip'] and np.random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

    # Blur gaussiano agresivo
    if AUG['blur_radius'] > 0:
        radius = np.random.uniform(0, AUG['blur_radius'])
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius))

    # Convertir a array para aplicar ruido
    img_array = np.array(pil_img, dtype=np.float32)

    # Ruido gaussiano agresivo
    if AUG['noise_level'] > 0:
        noise = np.random.normal(0, AUG['noise_level'] * 255, img_array.shape)
        img_array += noise
        img_array = np.clip(img_array, 0, 255)

    return img_array.astype(np.uint8)


def plot_images(images, labels, label_map, num_images: int = 16, figsize=(12, 8)) -> None:
    """Visualizar imagenes con sus etiquetas verdaderas."""
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    ejes = axes.ravel()
    for i in range(min(num_images, len(images))):
        img = images[i].reshape(28, 28) if images[i].ndim == 1 else images[i]
        ejes[i].imshow(img, cmap='gray')
        ejes[i].set_title(f'Etiqueta: {label_map.get_label(labels[i])}')
        ejes[i].axis('off')
    plt.tight_layout()
    plt.show()


def save_predictions_plot(images, true_labels, pred_labels, label_map, filepath, num_images: int = 16) -> None:
    """Guardar un grid con predicciones vs etiquetas reales."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    ejes = axes.ravel()
    for i in range(min(num_images, len(images))):
        img = images[i].reshape(28, 28) if images[i].ndim == 1 else images[i]
        true_label = label_map.get_label(true_labels[i])
        pred_label = label_map.get_label(pred_labels[i])
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        ejes[i].imshow(img, cmap='gray')
        ejes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        ejes[i].axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcular la precision clasica (accuracy)."""
    return float(np.mean(y_true == y_pred))


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, label_map) -> None:
    """Imprimir reporte de clasificacion y matriz de confusion."""
    class_names = [label_map.get_label(i) for i in range(label_map.get_num_classes())]
    print('Reporte de Clasificacion:')
    print(classification_report(y_true, y_pred, target_names=class_names))
    print('\nMatriz de Confusion:')
    print(confusion_matrix(y_true, y_pred))
