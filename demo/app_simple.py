"""Aplicacion web simplificada para probar el modelo clasico."""

from __future__ import annotations

import pickle
import string
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.append('demo')
from modelo import RedNeuronalSimple

MODELO_PATH = Path(__file__).resolve().parent.parent / 'models' / 'modelo_entrenado.pkl'
ETIQUETAS = list(string.ascii_uppercase) + list(string.ascii_lowercase)


def cargar_modelo() -> RedNeuronalSimple | None:
    """Cargar el modelo `RedNeuronalSimple` pickled."""
    if not MODELO_PATH.exists():
        st.error('No se encontro el modelo entrenado.')
        st.info('Ejecute primero: `python scripts/run_pipeline.py --force`')
        return None

    try:
        sys.modules['__main__'].RedNeuronalSimple = RedNeuronalSimple  # para pickle
        with MODELO_PATH.open('rb') as archivo:
            return pickle.load(archivo)
    except (OSError, pickle.UnpicklingError) as exc:
        st.error(f'Error cargando el modelo: {exc}')
        with st.expander('Informacion de debug'):
            st.write(f"**Ruta del modelo:** {MODELO_PATH}")
            st.write(f"**Archivo existe:** {MODELO_PATH.exists()}")
            st.write(f"**Error completo:** {repr(exc)}")
        return None


def preprocesar_imagen(imagen_pil: Image.Image) -> np.ndarray:
    """Convertir una imagen PIL en vector normalizado para la red clasica."""
    imagen = imagen_pil.convert('L').resize((28, 28))
    return np.asarray(imagen, dtype=np.float32).flatten() / 255.0


def _mostrar_top_predicciones(probabilidades: np.ndarray) -> None:
    """Mostrar top-5 predicciones para el modelo clasico."""
    indices = np.argsort(probabilidades.flatten())[-5:][::-1]
    for posicion, indice in enumerate(indices, start=1):
        etiqueta = ETIQUETAS[indice]
        probabilidad = float(probabilidades[indice])
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(f"**{posicion}. '{etiqueta}'**")
        with col2:
            st.progress(probabilidad)
        with col3:
            st.write(f"{probabilidad:.3f}")


def main() -> None:
    """Punto de entrada de la version simple de la demo."""
    st.set_page_config(page_title='Reconocimiento de Caracteres', page_icon='✏', layout='centered')
    st.title('Reconocimiento de Caracteres Personalizados (Demo simple)')
    st.markdown('---')
    st.write('Sube una imagen para probar el modelo clasico entrenado desde cero.')

    modelo = cargar_modelo()
    if modelo is None:
        st.stop()
    st.success('Modelo cargado exitosamente')

    with st.expander('Informacion del Modelo'):
        st.write(f"**Clases soportadas:** {len(ETIQUETAS)} caracteres")
        st.write('**Arquitectura:** 784 → 128 → 52 neuronas')
        st.write('**Caracteres:** A-Z (mayusculas) y a-z (minusculas)')

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
            salida = modelo.predecir(entrada)
        except (ValueError, TypeError) as exc:
            st.error(f'Error durante la prediccion: {exc}')
            return

        indice_predicho = int(np.argmax(salida))
        confianza = float(salida[indice_predicho])
        etiqueta_predicha = ETIQUETAS[indice_predicho]

        st.markdown('### Resultado de la Prediccion')
        info_col, conf_col = st.columns(2)
        with info_col:
            st.metric(label='Caracter Predicho', value=f"'{etiqueta_predicha}'")
        with conf_col:
            st.metric(label='Confianza', value=f'{confianza:.3f}', delta=f'{confianza * 100:.1f}%')
        st.progress(confianza)

        st.markdown('#### Top 5 Predicciones')
        _mostrar_top_predicciones(salida)

    st.markdown('---')
    st.markdown('### Consejos')
    st.write('- Usa imagenes con buen contraste.')
    st.write('- Centra el caracter en el lienzo.')
    st.write('- Evita fondos complejos o ruidosos.')


if __name__ == '__main__':
    main()