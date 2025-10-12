import os
import numpy as np
import pytest
from PIL import Image

from src.data_loader import DataLoader
from src.label_map import LabelMap


@pytest.fixture
def dummy_data_dir(tmp_path):
    """Crear un directorio de datos de prueba con imagenes falsas."""
    # Clases y numero de imagenes por clase
    clases = {'A_upper': 2, 'b_lower': 1}
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    for clase, num_imagenes in clases.items():
        clase_dir = data_dir / clase
        clase_dir.mkdir()
        for i in range(num_imagenes):
            # Crear una imagen PNG simple de 1x1
            img = Image.new('L', (1, 1), color=i)
            img.save(clase_dir / f"img_{i}.png")

    return str(data_dir)


@pytest.fixture
def label_map():
    """Fixture para un mapa de etiquetas que coincida con dummy_data_dir."""
    return LabelMap(['A', 'b'])


def test_mapear_carpeta_a_etiqueta(dummy_data_dir):
    """Verificar la logica de conversion de nombres de carpeta."""
    loader = DataLoader(dummy_data_dir)
    assert loader._mapear_carpeta_a_etiqueta('A_upper') == 'A'
    assert loader._mapear_carpeta_a_etiqueta('b_lower') == 'b'
    assert loader._mapear_carpeta_a_etiqueta('C') == 'C'


def test_cargar_desde_directorio(dummy_data_dir, label_map):
    """Verificar que las imagenes y etiquetas se carguen correctamente."""
    loader = DataLoader(ruta_datos=dummy_data_dir, mapa_etiquetas=label_map)
    loader.cargar_desde_directorio(tamano_imagen=(1, 1))

    assert loader.imagenes.shape == (3, 1, 1)
    assert loader.etiquetas.shape == (3,)

    # Las etiquetas deben corresponder a 'A' (indice 0) y 'b' (indice 1)
    # El orden depende del sistema de archivos, asi que contamos
    num_a = np.sum(loader.etiquetas == label_map.get_index('A'))
    num_b = np.sum(loader.etiquetas == label_map.get_index('b'))
    assert num_a == 2
    assert num_b == 1


def test_preprocesar_imagenes(dummy_data_dir, label_map):
    """Verificar que el preprocesamiento aplana y normaliza."""
    loader = DataLoader(ruta_datos=dummy_data_dir, mapa_etiquetas=label_map)
    loader.cargar_desde_directorio(tamano_imagen=(2, 2))

    # Mock de imagenes para que la normalizacion sea predecible
    loader.imagenes = np.full((3, 2, 2), 255, dtype=np.uint8)
    loader.preprocesar_imagenes()

    assert loader.imagenes.shape == (3, 4)  # Aplanadas
    assert np.allclose(loader.imagenes, 1.0)  # Normalizadas


def test_dividir_datos(dummy_data_dir, label_map):
    """Verificar que la division de datos mantenga las proporciones."""
    # Crear mas datos para que la division tenga sentido
    loader = DataLoader(ruta_datos=dummy_data_dir, mapa_etiquetas=label_map)
    loader.imagenes = np.random.rand(100, 4)
    loader.etiquetas = np.random.randint(0, label_map.get_num_classes(), 100)

    X_train, X_val, X_test, y_train, y_val, y_test = loader.dividir_datos(
        proporcion_entrenamiento=0.7,
        proporcion_validacion=0.15
    )

    # La division de scikit-learn puede no ser exacta
    assert X_train.shape[0] == 70
    assert X_val.shape[0] == 18
    assert X_test.shape[0] == 12
    assert y_train.shape[0] == 70
    assert y_val.shape[0] == 18
    assert y_test.shape[0] == 12
