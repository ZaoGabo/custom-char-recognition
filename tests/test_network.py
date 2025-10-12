import numpy as np
import pytest

from src.network import (
    NeuralNetwork,
    _leaky_relu,
    _leaky_relu_derivative,
    _relu,
    _relu_derivative,
    _sigmoid,
    _sigmoid_derivative,
    _softmax,
)


def test_relu():
    """Verificar la funcion de activacion ReLU."""
    x = np.array([-1, 0, 2])
    assert np.array_equal(_relu(x), [0, 0, 2])


def test_relu_derivative():
    """Verificar la derivada de ReLU."""
    x = np.array([-1, 0, 2])
    assert np.array_equal(_relu_derivative(x), [0, 0, 1])


def test_leaky_relu():
    """Verificar la funcion de activacion Leaky ReLU."""
    x = np.array([-2, 0, 2])
    alpha = 0.01
    assert np.allclose(_leaky_relu(x, alpha), [-0.02, 0, 2])


def test_leaky_relu_derivative():
    """Verificar la derivada de Leaky ReLU."""
    x = np.array([-2, 0, 2])
    alpha = 0.01
    assert np.allclose(_leaky_relu_derivative(x, alpha), [alpha, alpha, 1.0])


def test_sigmoid():
    """Verificar la funcion de activacion sigmoide."""
    x = np.array([-1000, 0, 1000])
    # Clip evita que exp under/overflow
    assert np.allclose(_sigmoid(x), [0, 0.5, 1])


def test_sigmoid_derivative():
    """Verificar la derivada de la sigmoide."""
    x = np.array([0])
    assert _sigmoid_derivative(x) == 0.25


def test_softmax():
    """Verificar que la salida de softmax sume 1."""
    x = np.array([[1, 2, 3], [1, 1, 1]])
    softmax_output = _softmax(x)
    assert np.allclose(np.sum(softmax_output, axis=1), [1.0, 1.0])


@pytest.fixture
def red_neuronal():
    """Fixture para una instancia de NeuralNetwork."""
    return NeuralNetwork(capas=[10, 5, 2], semilla=42)


def test_inicializacion_red(red_neuronal):
    """Verificar que las dimensiones de pesos y sesgos sean correctas."""
    assert len(red_neuronal.pesos) == 2
    assert red_neuronal.pesos[0].shape == (5, 10)
    assert red_neuronal.pesos[1].shape == (2, 5)
    assert len(red_neuronal.sesgos) == 2
    assert red_neuronal.sesgos[0].shape == (5, 1)
    assert red_neuronal.sesgos[1].shape == (2, 1)


def test_forward_pass(red_neuronal):
    """Verificar que el forward pass produzca una salida con la dimension correcta."""
    X = np.random.rand(3, 10)  # 3 muestras, 10 caracteristicas
    caches, _ = red_neuronal._forward(X, training=False)
    salida = caches[f'A{red_neuronal.num_capas - 1}']
    assert salida.T.shape == (3, 2)  # 3 muestras, 2 clases


def test_perdida():
    """Verificar el calculo de la perdida cross-entropy."""
    # Red simple para controlar los calculos
    red = NeuralNetwork(capas=[2, 2], semilla=1)
    Y_true = np.array([[1, 0]])
    # Mock de salida de la red para que la perdida sea predecible
    Y_pred = np.array([[0.8, 0.2]])
    perdida = red.calcular_perdida(Y_true, Y_pred)
    assert np.isclose(perdida, -np.log(0.8))


def test_fit_una_epoca(red_neuronal):
    """Verificar que el entrenamiento se ejecute y la perdida disminuya."""
    X = np.random.rand(20, 10)
    Y = np.zeros((20, 2))
    Y[np.arange(20), np.random.randint(0, 2, 20)] = 1  # One-hot

    historia = red_neuronal.fit(X, Y, epocas=2, tamano_lote=5)
    assert len(historia) == 2
    assert historia[1]['loss_train'] < historia[0]['loss_train']


def test_prediccion(red_neuronal):
    """Verificar que los metodos de prediccion devuelvan las formas correctas."""
    X = np.random.rand(4, 10)
    probs = red_neuronal.predecir_probabilidades(X)
    preds = red_neuronal.predecir(X)
    assert probs.shape == (4, 2)
    assert preds.shape == (4,)
