import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.network import NeuralNetwork
from src.trainer import _one_hot, _evaluar, entrenar_modelo, preparar_datos


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


def test_evaluar(mock_network):
    """Verificar que la evaluacion calcule metricas correctamente."""
    X = np.random.rand(2, 4)
    y_true_oh = np.array([[1, 0], [0, 1]])
    metricas = _evaluar(mock_network, X, y_true_oh)

    mock_network.predecir_probabilidades.assert_called_once_with(X)

    # Verificar la llamada a calcular_perdida de forma robusta
    mock_network.calcular_perdida.assert_called_once()
    called_args, _ = mock_network.calcular_perdida.call_args
    assert np.array_equal(called_args[0], y_true_oh)
    assert called_args[1].shape == y_true_oh.shape

    assert metricas['loss'] == 0.5

    # El accuracy depende de la salida aleatoria, asi que solo verificamos el tipo y rango
    assert isinstance(metricas['accuracy'], float)
    assert 0.0 <= metricas['accuracy'] <= 1.0


@patch('src.trainer.preparar_datos')
@patch('src.trainer._construir_modelo')
@patch('pickle.dump')
def test_entrenar_modelo_flujo_completo(mock_pickle_dump, mock_construir_modelo, mock_preparar_datos, mock_data, mock_network, tmp_path):
    """
    Verificar el flujo de entrenamiento:
    - Carga de datos
    - Construccion de modelo
    - Entrenamiento (fit)
    - Guardado de modelo
    - Evaluacion
    """
    # Configurar mocks
    mock_preparar_datos.return_value = mock_data
    mock_construir_modelo.return_value = mock_network

    # Sobrescribir la ruta de modelos para usar un directorio temporal
    with patch('src.trainer.PATHS', {'modelos': str(tmp_path)}):
        modelo, metricas = entrenar_modelo(force=True, verbose=False)

        # Verificar llamadas
        mock_preparar_datos.assert_called_once()
        mock_construir_modelo.assert_called_once()
        mock_network.fit.assert_called_once()
        mock_pickle_dump.assert_called_once()

        # Verificar la estructura de las metricas
        assert 'train' in metricas
        assert 'val' in metricas
        assert 'test' in metricas
        assert 'loss' in metricas['test']
        assert 'accuracy' in metricas['test']
        assert modelo == mock_network
