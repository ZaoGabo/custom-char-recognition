"""Aplicacion web para probar el reconocimiento de caracteres personalizados."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

RUTA_RAIZ = Path(__file__).resolve().parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from src.config import PATHS
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork


@st.cache_resource
def cargar_modelo_entrenado():
    """Cargar el modelo entrenado usando el nuevo formato (JSON + npy)."""
    modelo_dir = RUTA_RAIZ / PATHS['modelos'] / 'modelo_entrenado'
    arquitectura_path = modelo_dir / "arquitectura.json"

    if not arquitectura_path.exists():
        st.error('No se encontro el archivo de arquitectura del modelo.')
        st.info('Ejecute primero: `python -m src.trainer --force`')
        return None

    try:
        modelo = NeuralNetwork.cargar_modelo(str(modelo_dir))
    except (OSError, FileNotFoundError, json.JSONDecodeError) as exc:
        st.error(f'Error cargando el modelo: {exc}')
        return None

    st.success('Modelo cargado exitosamente')
    return modelo


def preprocesar_imagen(imagen_pil: Image.Image) -> np.ndarray:
    """Convertir una imagen PIL en vector normalizado de 784 elementos."""
    imagen = imagen_pil.convert('L').resize((28, 28))
    return np.asarray(imagen, dtype=np.float32).flatten() / 255.0


def _obtener_probabilidades(modelo, entrada: np.ndarray) -> np.ndarray:
    """Obtener una matriz (1, clases) con probabilidades del modelo."""
    salida = modelo.predecir_probabilidades(entrada)
    return salida if salida.ndim == 2 else salida.reshape(1, -1)


def _mostrar_top_predicciones(probabilidades: np.ndarray) -> None:
    """Visualizar las cinco predicciones con mayor probabilidad."""
    vector = probabilidades.flatten()
    top_indices = np.argsort(vector)[-5:][::-1]
    for posicion, indice in enumerate(top_indices, start=1):
        etiqueta = DEFAULT_LABEL_MAP.get_label(int(indice))
        probabilidad = float(vector[indice])
        col_idx, col_bar, col_val = st.columns([1, 2, 1])
        with col_idx:
            st.write(f"**{posicion}. '{etiqueta}'**")
        with col_bar:
            barra = max(0.0, min(1.0, probabilidad))
            st.progress(barra)
        with col_val:
            st.write(f"{probabilidad:.3f}")


def main() -> None:
    """Punto de entrada de la interfaz Streamlit."""
    st.set_page_config(page_title='Reconocimiento de Caracteres', page_icon='NN', layout='centered')
    st.title('Reconocimiento de Caracteres Personalizados')
    
    modelo = cargar_modelo_entrenado()
    if modelo is None:
        st.stop()

    with st.expander('Informacion del Modelo'):
        st.write(f"**Capas:** {modelo.capas}")
        st.write(f"**Funciones de Activacion:** {modelo.activaciones}")

    st.markdown('### Subir Imagen para Prediccion')
    imagen_subida = st.file_uploader(
        'Selecciona una imagen de un caracter:',
        type=['png', 'jpg', 'jpeg', 'bmp'],
    )

    if not imagen_subida:
        return

    try:
        imagen_pil = Image.open(imagen_subida)
    except (OSError, UnidentifiedImageError) as exc:
        st.error(f'No se pudo abrir la imagen: {exc}')
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen_pil, caption='Imagen Original', use_column_width=True)

    entrada = preprocesar_imagen(imagen_pil)
    with col2:
        st.image(entrada.reshape(28, 28), caption='Imagen Procesada (28x28)', use_column_width=True, clamp=True)

    if st.button('Predecir', type='primary'):
        probabilidades = _obtener_probabilidades(modelo, entrada)
        indice_predicho = int(np.argmax(probabilidades))
        confianza = float(probabilidades[0, indice_predicho])
        etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)

        st.metric(label='Caracter Predicho', value=f"'{etiqueta_predicha}'", delta=f'{confianza:.2%}')
        
        st.markdown('#### Top 5 Predicciones')
        _mostrar_top_predicciones(probabilidades)


if __name__ == '__main__':
    main()
