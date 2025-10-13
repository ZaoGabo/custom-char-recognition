import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.utils import (
    apply_augmentation,
    calculate_accuracy,
    denormalize_image,
    normalize_image,
    plot_images,
    save_predictions_plot,
    print_classification_report
)
from src.label_map import LabelMap

def test_normalize_image():
    """Verificar que la normalizacion escala correctamente a [0, 1]."""
    imagen = np.array([[0, 127.5], [255, 63.75]], dtype=np.float32)
    imagen_normalizada = normalize_image(imagen)
    assert imagen_normalizada.dtype == np.float32
    assert np.allclose(imagen_normalizada, [[0.0, 0.50], [1.0, 0.25]])


def test_denormalize_image():
    """Verificar que la desnormalizacion escala a [0, 255]."""
    imagen_normalizada = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    imagen = denormalize_image(imagen_normalizada)
    assert imagen.dtype == np.uint8
    assert np.array_equal(imagen, [[0, 127], [255, 63]])


def test_calculate_accuracy():
    """Verificar que el calculo de precision sea correcto."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 0])
    assert calculate_accuracy(y_true, y_pred) == 0.8

def test_apply_augmentation():
    """Verificar que la aumentacion de imagen cambie la imagen."""
    imagen = np.random.rand(28, 28) * 255
    imagen_aumentada = apply_augmentation(imagen)
    assert imagen_aumentada.shape == imagen.shape
    assert not np.array_equal(imagen, imagen_aumentada)

@patch('src.utils.plt')
def test_plot_images(mock_plt):
    """Verificar que la funcion de ploteo llame a matplotlib."""
    mock_fig, mock_axes = MagicMock(), MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    imagenes = np.random.rand(16, 28, 28)
    etiquetas = np.arange(16)
    mapa = LabelMap([chr(ord('A') + i) for i in range(16)])
    plot_images(imagenes, etiquetas, mapa)
    
    assert mock_plt.subplots.called
    assert mock_plt.tight_layout.called
    assert mock_plt.show.called

@patch('src.utils.plt')
def test_save_predictions_plot(mock_plt):
    """Verificar que la funcion de guardado de plots llame a matplotlib."""
    mock_fig, mock_axes = MagicMock(), MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    imagenes = np.random.rand(16, 28, 28)
    true_labels = np.arange(16)
    pred_labels = np.arange(16)
    mapa = LabelMap([chr(ord('A') + i) for i in range(16)])
    filepath = "test.png"
    
    save_predictions_plot(imagenes, true_labels, pred_labels, mapa, filepath)
    
    assert mock_plt.subplots.called
    assert mock_plt.savefig.called
    assert mock_plt.close.called

@patch('builtins.print')
@patch('src.utils.classification_report')
@patch('src.utils.confusion_matrix')
def test_print_classification_report(mock_confusion_matrix, mock_classification_report, mock_print):
    """Verificar que el reporte de clasificacion se imprima correctamente."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    mapa = LabelMap(['A', 'B'])
    
    print_classification_report(y_true, y_pred, mapa)
    
    assert mock_classification_report.called
    assert mock_confusion_matrix.called
    assert mock_print.call_count > 0
