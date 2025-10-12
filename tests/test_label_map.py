import json
import os
import pytest

from src.label_map import LabelMap


@pytest.fixture
def mapa_etiquetas():
    """Fixture para crear un LabelMap de prueba."""
    return LabelMap(['B', 'A', 'C'])


def test_label_map_inicializacion(mapa_etiquetas):
    """Verificar que las etiquetas se ordenen y los mapeos se creen."""
    assert mapa_etiquetas.labels == ['A', 'B', 'C']
    assert mapa_etiquetas.label_to_index == {'A': 0, 'B': 1, 'C': 2}
    assert mapa_etiquetas.index_to_label == {0: 'A', 1: 'B', 2: 'C'}


def test_get_index(mapa_etiquetas):
    """Verificar la obtencion de indices."""
    assert mapa_etiquetas.get_index('B') == 1
    assert mapa_etiquetas.get_index('D') == -1


def test_get_label(mapa_etiquetas):
    """Verificar la obtencion de etiquetas."""
    assert mapa_etiquetas.get_label(2) == 'C'
    assert mapa_etiquetas.get_label(3) == "Unknown"


def test_get_num_classes(mapa_etiquetas):
    """Verificar que el numero de clases sea correcto."""
    assert mapa_etiquetas.get_num_classes() == 3


def test_save_and_load(mapa_etiquetas, tmp_path):
    """Verificar guardado y carga desde JSON."""
    ruta_archivo = os.path.join(tmp_path, "label_map.json")
    mapa_etiquetas.save(ruta_archivo)

    assert os.path.exists(ruta_archivo)

    mapa_cargado = LabelMap.load(ruta_archivo)
    assert mapa_cargado.labels == mapa_etiquetas.labels
    assert mapa_cargado.label_to_index == mapa_etiquetas.label_to_index
