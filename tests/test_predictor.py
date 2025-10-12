import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.predictor import normalizar_entrada, evaluar_red
from src.label_map import LabelMap

@pytest.fixture
def mock_red_predictor():
    """Fixture para una red neuronal simulada para pruebas de prediccion."""
    net = MagicMock()
    # Simular que la red predice la clase 0 para la primera entrada y la 1 para la segunda
    net.predecir_probabilidades.side_effect = [
        np.array([[0.9, 0.1]]),  # Predice 'A'
        np.array([[0.2, 0.8]]),  # Predice 'B'
    ]
    return net

def test_normalizar_entrada():
    """Verificar que la normalizacion de entrada funcione con enteros y flotantes."""
    # Caso 1: Entrada de enteros en [0, 255]
    entrada_int = np.array([0, 127.5, 255])
    resultado_int = normalizar_entrada(entrada_int)
    assert np.allclose(resultado_int, [0.0, 0.5, 1.0])

    # Caso 2: Entrada ya normalizada (flotantes en [0, 1])
    entrada_float = np.array([0.0, 0.5, 1.0])
    resultado_float = normalizar_entrada(entrada_float)
    assert np.allclose(resultado_float, [0.0, 0.5, 1.0])

    # Caso 3: Entrada vacia
    entrada_vacia = np.array([])
    resultado_vacio = normalizar_entrada(entrada_vacia)
    assert resultado_vacio.size == 0

def test_evaluar_red(mock_red_predictor):
    """Verificar que la evaluacion de la red calcule la precision correctamente."""
    # Mock de LabelMap para que coincida con las predicciones
    with patch('src.predictor.DEFAULT_LABEL_MAP', LabelMap(['A', 'B'])):
        # Datos de prueba: [etiqueta_correcta, pixel1, pixel2, ...]
        datos_prueba = [
            "A,0,255",  # Acierto (predice A)
            "A,255,0",  # Fallo (predice B)
        ]

        rendimiento = evaluar_red(mock_red_predictor, datos_prueba)

        assert rendimiento == 0.5
        assert mock_red_predictor.predecir_probabilidades.call_count == 2
