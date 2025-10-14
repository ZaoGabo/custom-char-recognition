import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.network import NeuralNetwork
from src.trainer import _one_hot, entrenar_modelo


@pytest.fixture
def mock_data():
    """Fixture para datos de entrenamiento / validacion / prueba simulados."""
    X_train = np.random.rand(10, 4)
    y_train = np.array([0, 1] * 5)
    X_val = np.random.rand(4, 4)
    y_val = np.array([0, 1, 0, 1])
    X_test = np.random.rand(4, 4)
    y_test = np.array([1, 0, 1, 0])
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture
def mock_network():
    """Fixture para una red neuronal simulada."""
    net = MagicMock(spec=NeuralNetwork)

    def predict_probs_dynamic(X):
        num_muestras = X.shape[0]
        num_clases = 2
        probs = np.random.rand(num_muestras, num_clases)
        return probs / probs.sum(axis=1, keepdims=True)

    net.predecir_probabilidades.side_effect = predict_probs_dynamic
    net.calcular_perdida.return_value = 0.5
    net.predecir.side_effect = lambda X: np.argmax(predict_probs_dynamic(X), axis=1)
    return net


def test_one_hot_encoding():
    """Verificar que la codificacion one-hot sea correcta."""
    labels = np.array([0, 2, 1])
    num_clases = 3
    one_hot_labels = _one_hot(labels, num_clases)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.array_equal(one_hot_labels, expected)




@patch('src.trainer.DataLoader')
@patch('src.trainer._construir_modelo')
@patch('src.trainer.Path.exists')
def test_entrenar_modelo_flujo_completo(mock_exists, mock_construir, mock_dataloader, mock_data):
    """Verificar el flujo principal de entrenamiento."""
    # Configurar mocks
    mock_dataloader_instance = mock_dataloader.return_value
    mock_dataloader_instance.dividir_datos.return_value = (['train_path'], ['val_path'], [0], [1])
    
    mock_modelo_instance = MagicMock(spec=NeuralNetwork)
    mock_modelo_instance.capas = [784, 128, 52]  # Simular capas
    mock_construir.return_value = mock_modelo_instance
    
    mock_exists.return_value = False

    with patch('src.trainer.next') as mock_next:
        mock_next.return_value = (np.random.rand(10, 784), np.array([0]*10))
        entrenar_modelo(force=True, verbose=False)

    # Verificar llamadas clave
    mock_dataloader_instance.cargar_desde_directorio.assert_called_once()
    mock_dataloader_instance.dividir_datos.assert_called_once()
    mock_construir.assert_called_once()
    assert mock_modelo_instance.guardar_modelo.call_count >= 1
