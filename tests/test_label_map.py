import os

import pytest

from src.config import CUSTOM_LABELS
from src.label_map import DEFAULT_LABEL_MAP, LabelMap

# pylint: disable=redefined-outer-name


@pytest.fixture
def label_map_fixture():
    """Fixture para crear un LabelMap de prueba."""
    return LabelMap(['B', 'A', 'C'])


def test_label_map_inicializacion(label_map_fixture):
    """Verificar que se preserve el orden provisto."""
    assert label_map_fixture.labels == ['B', 'A', 'C']
    assert label_map_fixture.label_to_index == {'B': 0, 'A': 1, 'C': 2}
    assert label_map_fixture.index_to_label == {0: 'B', 1: 'A', 2: 'C'}


def test_get_index(label_map_fixture):
    """Verificar la obtencion de indices."""
    assert label_map_fixture.get_index('B') == 0
    assert label_map_fixture.get_index('D') == -1


def test_get_label(label_map_fixture):
    """Verificar la obtencion de etiquetas."""
    assert label_map_fixture.get_label(2) == 'C'
    assert label_map_fixture.get_label(3) == "Unknown"


def test_get_num_classes(label_map_fixture):
    """Verificar que el numero de clases sea correcto."""
    assert label_map_fixture.get_num_classes() == 3


def test_save_and_load(label_map_fixture, tmp_path):
    """Verificar guardado y carga desde JSON."""
    ruta_archivo = os.path.join(tmp_path, "label_map.json")
    label_map_fixture.save(ruta_archivo)

    assert os.path.exists(ruta_archivo)

    mapa_cargado = LabelMap.load(ruta_archivo)
    assert mapa_cargado.labels == label_map_fixture.labels
    assert mapa_cargado.label_to_index == label_map_fixture.label_to_index


def test_label_map_preserva_orden_personalizado():
    etiquetas = ['Z', 'Y', 'X']
    mapa = LabelMap(etiquetas)
    assert mapa.labels == etiquetas
    assert mapa.label_to_index == {'Z': 0, 'Y': 1, 'X': 2}


def test_label_map_por_defecto_coincide_con_config():
    assert DEFAULT_LABEL_MAP.labels == CUSTOM_LABELS
