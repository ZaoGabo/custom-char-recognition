"""
Aplicacion web para probar el reconocimiento de caracteres personalizados.
"""

import streamlit as st
import numpy as np
import pickle
import sys
import os
from PIL import Image

# Configurar rutas
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if RUTA_RAIZ not in sys.path:
    sys.path.insert(0, RUTA_RAIZ)

sys.path.insert(0, os.path.join(RUTA_RAIZ, 'src'))

from src.label_map import DEFAULT_LABEL_MAP
from src.config import PATHS


@st.cache_resource
def cargar_modelo():
    modelo_path = os.path.join(RUTA_RAIZ, PATHS['modelos'], "modelo_entrenado.pkl")

    if not os.path.exists(modelo_path):
        st.error("No se encontro el modelo entrenado.")
        st.info("Ejecute primero: `python -m src.trainer --force`")
        return None

    try:
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        st.success("Modelo cargado exitosamente")
        return modelo
    except Exception as e:  # pragma: no cover
        st.error(f"Error cargando el modelo: {str(e)}")
        with st.expander("Informacion de debug"):
            st.write(f"**Ruta del modelo:** {modelo_path}")
            st.write(f"**Archivo existe:** {os.path.exists(modelo_path)}")
            st.write(f"**Error completo:** {repr(e)}")
        return None


def preprocesar_imagen(imagen_pil):
    imagen_pil = imagen_pil.convert('L')
    imagen_pil = imagen_pil.resize((28, 28))
    imagen_np = np.array(imagen_pil).astype(np.float32) / 255.0
    return imagen_np.flatten()


def _obtener_probabilidades(modelo, entrada: np.ndarray) -> np.ndarray:
    if hasattr(modelo, 'predecir_probabilidades'):
        salida = modelo.predecir_probabilidades(entrada)
        return salida if salida.ndim == 2 else salida.reshape(1, -1)
    salida = modelo.predecir(entrada)
    return salida if salida.ndim == 2 else salida.reshape(1, -1)


def main():
    st.set_page_config(
        page_title="Reconocimiento de Caracteres",
        page_icon="lapiz",
        layout="centered"
    )

    st.title("Reconocimiento de Caracteres Personalizados")
    st.markdown("---")
    st.write("Sube una imagen de un caracter para probar el modelo de red neuronal.")

    modelo = cargar_modelo()
    if modelo is None:
        st.stop()

    entrada_dim = getattr(modelo, 'capas', [getattr(modelo, 'entrada_neuronas', 784)])[0]
    with st.expander("Informacion del Modelo"):
        st.write(f"**Neuronas de entrada:** {entrada_dim}")
        st.write(f"**Neuronas de salida:** {DEFAULT_LABEL_MAP.get_num_classes()}")
        st.write(f"**Etiquetas:** {', '.join(DEFAULT_LABEL_MAP.labels[:10])}...")

    st.markdown("### Subir Imagen")
    imagen_subida = st.file_uploader(
        "Selecciona una imagen de un caracter:",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if imagen_subida:
        imagen_pil = Image.open(imagen_subida)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Imagen Original**")
            st.image(imagen_pil, caption="Imagen subida", use_column_width=True)

        with col2:
            entrada = preprocesar_imagen(imagen_pil)
            st.markdown("**Imagen Procesada (28x28)**")
            st.image(entrada.reshape(28, 28), caption="Imagen procesada", use_column_width=True, clamp=True)

        if st.button("Predecir Caracter", type="primary"):
            try:
                salida = _obtener_probabilidades(modelo, entrada)
                indice_predicho = int(np.argmax(salida))
                confianza = float(salida[0, indice_predicho])
                etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)

                st.markdown("### Resultado de la Prediccion")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Caracter Predicho", value=f"'{etiqueta_predicha}'")
                with col2:
                    st.metric(label="Confianza", value=f"{confianza:.3f}", delta=f"{confianza * 100:.1f}%")

                st.progress(confianza)

                st.markdown("#### Top 5 Predicciones")
                probs = salida.flatten()
                top_indices = np.argsort(probs)[-5:][::-1]

                for i, idx in enumerate(top_indices):
                    etiqueta = DEFAULT_LABEL_MAP.get_label(int(idx))
                    probabilidad = float(probs[idx])
                    col_idx, col_bar, col_val = st.columns([1, 2, 1])
                    with col_idx:
                        st.write(f"**{i + 1}. '{etiqueta}'**")
                    with col_bar:
                        st.progress(probabilidad)
                    with col_val:
                        st.write(f"{probabilidad:.3f}")
            except Exception as e:  # pragma: no cover
                st.error(f"Error durante la prediccion: {str(e)}")


if __name__ == '__main__':
    main()
