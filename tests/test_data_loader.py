import os
import numpy as np
import pytest
from PIL import Image

from src.data_loader import DataLoader
from src.label_map import LabelMap


@pytest.fixture
def dummy_data_dir(tmp_path):
    """Crear un directorio de datos de prueba con imagenes falsas."""
    clases = {'A_upper': 2, 'b_lower': 1}
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    for clase, num_imagenes in clases.items():
        clase_dir = data_dir / clase
        clase_dir.mkdir()
        for i in range(num_imagenes):
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
    """Verificar que las rutas de imagenes y etiquetas se carguen."""
    loader = DataLoader(ruta_datos=dummy_data_dir, mapa_etiquetas=label_map)
    loader.cargar_desde_directorio()

    assert len(loader.rutas_imagenes) == 3
    assert len(loader.etiquetas) == 3

    num_a = sum(1 for et in loader.etiquetas if et == label_map.get_index('A'))
    num_b = sum(1 for et in loader.etiquetas if et == label_map.get_index('b'))
    assert num_a == 2
    assert num_b == 1


def test_dividir_datos(dummy_data_dir, label_map):
    """Verificar que la division de datos se aplique a las rutas."""
    loader = DataLoader(ruta_datos=dummy_data_dir, mapa_etiquetas=label_map)
    # Simular más datos para una división significativa
    loader.rutas_imagenes = [f"path/{i}" for i in range(100)]
    loader.etiquetas = list(np.random.randint(0, 2, 100))

    rutas_train, rutas_val, y_train, y_val = loader.dividir_datos(
        proporcion_entrenamiento=0.7
    )

    assert len(rutas_train) == 70
    assert len(rutas_val) == 30
    assert len(y_train) == 70
    assert len(y_val) == 30


def test_generar_lotes(dummy_data_dir, label_map):
    """Verificar que el generador de lotes produzca datos con la forma correcta."""
    loader = DataLoader(ruta_datos=dummy_data_dir, mapa_etiquetas=label_map)
    loader.cargar_desde_directorio()

    tamano_lote = 2
    tamano_imagen = (4, 4)
    
    gen = loader.generar_lotes(
        loader.rutas_imagenes, 
        loader.etiquetas, 
        tamano_lote, 
        tamano_imagen
    )
    
    X_lote, y_lote = next(gen)
    
    assert X_lote.shape == (tamano_lote, tamano_imagen[0] * tamano_imagen[1])
    assert y_lote.shape == (tamano_lote,)
    assert X_lote.dtype == np.float32
    assert np.all(X_lote <= 1.0) and np.all(X_lote >= 0.0)
