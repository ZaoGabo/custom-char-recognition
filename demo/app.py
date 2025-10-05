"""Aplicacion web para probar el reconocimiento de caracteres personalizados."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

RUTA_RAIZ = Path(__file__).resolve().parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

sys.path.insert(0, str(RUTA_RAIZ / 'src'))

from src.config import PATHS
from src.label_map import DEFAULT_LABEL_MAP


@st.cache_resource
def cargar_modelo():
    """Cargar el modelo entrenado desde ``models/``."""
    modelo_path = RUTA_RAIZ / PATHS['modelos'] / 'modelo_entrenado.pkl'
    if not modelo_path.exists():
        st.error('No se encontro el modelo entrenado.')
        st.info('Ejecute primero: `python -m src.trainer --force`')
        return None

    try:
        with modelo_path.open('rb') as archivo:
            modelo = pickle.load(archivo)
    except (OSError, pickle.UnpicklingError) as exc:  # pragma: no cover
        st.error(f'Error cargando el modelo: {exc}')
        with st.expander('Informacion de debug'):
            st.write(f"**Ruta del modelo:** {modelo_path}")
            st.write(f"**Archivo existe:** {modelo_path.exists()}")
            st.write(f"**Error completo:** {repr(exc)}")
        return None

    st.success('Modelo cargado exitosamente')
    return modelo


def preprocesar_imagen(imagen_pil: Image.Image) -> np.ndarray:
    """Convertir una imagen PIL en vector normalizado de 784 elementos."""
    imagen = imagen_pil.convert('L').resize((28, 28))
    return np.asarray(imagen, dtype=np.float32).flatten() / 255.0


def _obtener_probabilidades(modelo, entrada: np.ndarray) -> np.ndarray:
    """Obtener una matriz (1, clases) con probabilidades del modelo."""
    if hasattr(modelo, 'predecir_probabilidades'):
        salida = modelo.predecir_probabilidades(entrada)
    else:
        salida = modelo.predecir(entrada)
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
            st.progress(probabilidad)
        with col_val:
            st.write(f"{probabilidad:.3f}")


def main() -> None:
    """Punto de entrada de la interfaz Streamlit."""
    st.set_page_config(page_title='Reconocimiento de Caracteres', page_icon='✏', layout='centered')
    st.title('Reconocimiento de Caracteres Personalizados')
    st.markdown('---')
    st.write('Sube una imagen de un caracter para probar el modelo de red neuronal.')

    modelo = cargar_modelo()
    if modelo is None:
        st.stop()

    entrada_dim = getattr(modelo, 'capas', [getattr(modelo, 'entrada_neuronas', 784)])[0]
    with st.expander('Informacion del Modelo'):
        st.write(f"**Neuronas de entrada:** {entrada_dim}")
        st.write(f"**Neuronas de salida:** {DEFAULT_LABEL_MAP.get_num_classes()}")
        st.write(f"**Etiquetas:** {', '.join(DEFAULT_LABEL_MAP.labels[:10])}...")

    st.markdown('### Subir Imagen')
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
        st.markdown('**Imagen Original**')
        st.image(imagen_pil, caption='Imagen subida', use_column_width=True)

    entrada = preprocesar_imagen(imagen_pil)
    with col2:
        st.markdown('**Imagen Procesada (28x28)**')
        st.image(entrada.reshape(28, 28), caption='Imagen procesada', use_column_width=True, clamp=True)

    if st.button('Predecir Caracter', type='primary'):
        try:
            probabilidades = _obtener_probabilidades(modelo, entrada)
        except (ValueError, TypeError) as exc:  # pragma: no cover
            st.error(f'Error durante la prediccion: {exc}')
            return

        indice_predicho = int(np.argmax(probabilidades))
        confianza = float(probabilidades[0, indice_predicho])
        etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)

        st.markdown('### Resultado de la Prediccion')
        info_col, conf_col = st.columns(2)
        with info_col:
            st.metric(label='Caracter Predicho', value=f"'{etiqueta_predicha}'")
        with conf_col:
            st.metric(label='Confianza', value=f'{confianza:.3f}', delta=f'{confianza * 100:.1f}%')
        st.progress(confianza)

        st.markdown('#### Top 5 Predicciones')
        _mostrar_top_predicciones(probabilidades)


if __name__ == '__main__':
    main()