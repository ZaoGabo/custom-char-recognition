"""
Funciones auxiliares para el proyecto.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
from .config import AUGMENTATION_CONFIG

# Adaptador de claves en español
AUG = {
    'rotation_range': AUGMENTATION_CONFIG['rango_rotacion'],
    'width_shift_range': AUGMENTATION_CONFIG['desplazamiento_horizontal'],
    'height_shift_range': AUGMENTATION_CONFIG['desplazamiento_vertical'],
    'zoom_range': AUGMENTATION_CONFIG['rango_zoom'],
    'horizontal_flip': AUGMENTATION_CONFIG['voltear_horizontal'],
    'vertical_flip': AUGMENTATION_CONFIG['voltear_vertical']
}

def normalize_image(images):
    """Normalizar imágenes al rango [0, 1]."""
    return images.astype(np.float32) / 255.0

def denormalize_image(images):
    """Desnormalizar imágenes al rango [0, 255]."""
    return (images * 255).astype(np.uint8)

def apply_augmentation(image):
    """Aplicar augmentación de datos a una imagen."""
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        pil_img = Image.fromarray(image)

    # Rotación
    if AUG['rotation_range'] > 0:
        angle = np.random.uniform(-AUG['rotation_range'], AUG['rotation_range'])
        pil_img = pil_img.rotate(angle, fillcolor=0)

    # Desplazamiento horizontal
    if AUG['width_shift_range'] > 0:
        width, height = pil_img.size
        shift_x = int(np.random.uniform(-AUG['width_shift_range'] * width, AUG['width_shift_range'] * width))
        pil_img = ImageOps.expand(pil_img, border=(abs(shift_x), 0, 0, 0), fill=0)
        if shift_x > 0:
            pil_img = pil_img.crop((shift_x, 0, width + shift_x, height))
        else:
            pil_img = pil_img.crop((0, 0, width, height))

    # Desplazamiento vertical
    if AUG['height_shift_range'] > 0:
        width, height = pil_img.size
        shift_y = int(np.random.uniform(-AUG['height_shift_range'] * height, AUG['height_shift_range'] * height))
        pil_img = ImageOps.expand(pil_img, border=(0, abs(shift_y), 0, 0), fill=0)
        if shift_y > 0:
            pil_img = pil_img.crop((0, shift_y, width, height + shift_y))
        else:
            pil_img = pil_img.crop((0, 0, width, height))

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
            pil_img = ImageOps.expand(pil_img, border=((width - new_width) // 2, (height - new_height) // 2), fill=0)

    # Flip horizontal
    if AUG['horizontal_flip'] and np.random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    # Flip vertical
    if AUG['vertical_flip'] and np.random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

    return np.array(pil_img)

def plot_images(images, labels, label_map, num_images=16, figsize=(12, 8)):
    """Visualizar imágenes con sus etiquetas."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        img = images[i].reshape(28, 28) if len(images[i].shape) == 1 else images[i]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Etiqueta: {label_map.get_label(labels[i])}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def save_predictions_plot(images, true_labels, pred_labels, label_map, filepath, num_images=16):
    """Guardar gráfico de predicciones."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        img = images[i].reshape(28, 28) if len(images[i].shape) == 1 else images[i]
        true_label = label_map.get_label(true_labels[i])
        pred_label = label_map.get_label(pred_labels[i])
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_accuracy(y_true, y_pred):
    """Calcular precisión (accuracy)."""
    return np.mean(y_true == y_pred)

def print_classification_report(y_true, y_pred, label_map):
    """Imprimir reporte de clasificación."""
    from sklearn.metrics import classification_report, confusion_matrix
    class_names = [label_map.get_label(i) for i in range(label_map.get_num_classes())]

    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
