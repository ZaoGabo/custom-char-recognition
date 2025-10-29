"""Aplicación Streamlit con canvas dibujable para reconocimiento de caracteres."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

RUTA_RAIZ = Path(__file__).resolve().parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from src.config import PATHS
from src.label_map import DEFAULT_LABEL_MAP
from src.network import NeuralNetwork


@st.cache_resource
def cargar_modelo_entrenado():
    """Cargar el modelo entrenado."""
    modelo_dir = RUTA_RAIZ / PATHS['modelos'] / 'modelo_entrenado'
    arquitectura_path = modelo_dir / "arquitectura.json"

    if not arquitectura_path.exists():
        st.error('❌ No se encontró el modelo entrenado.')
        st.info('📝 Ejecuta primero: `python -m src.trainer --force --verbose`')
        return None

    try:
        modelo = NeuralNetwork.cargar_modelo(str(modelo_dir))
        return modelo
    except Exception as exc:
        st.error(f'❌ Error cargando el modelo: {exc}')
        return None


def preprocesar_imagen_canvas(imagen_array: np.ndarray) -> np.ndarray:
    """
    Preprocesar imagen del canvas EXACTAMENTE como EMNIST.
    
    EMNIST tiene: fondo negro (0) y texto blanco (255)
    Canvas dibuja: fondo negro (0) y trazos blancos (255)
    
    Pasos (iguales a EMNIST):
    1. Extraer escala de grises
    2. Encontrar bounding box del contenido
    3. Recortar y centrar con aspect ratio preservado
    4. Padding adaptativo (20% del tamaño)
    5. Resize a 20x20
    6. Centrar en imagen 28x28
    7. Normalizar [0, 1]
    8. Aplicar suavizado gaussiano ligero
    """
    # Extraer escala de grises
    if imagen_array.shape[-1] == 4:
        imagen_rgb = imagen_array[:, :, :3]
        imagen_gray = np.mean(imagen_rgb, axis=2).astype(np.uint8)
    else:
        imagen_gray = imagen_array.astype(np.uint8)

    # Verificar contenido
    if np.max(imagen_gray) < 10:
        return np.zeros(784, dtype=np.float32)

    # Bounding box con threshold bajo (más sensible)
    threshold = 20  # Más sensible a trazos suaves
    mascara = imagen_gray > threshold
    
    if not np.any(mascara):
        return np.zeros(784, dtype=np.float32)
    
    # Encontrar límites
    filas = np.any(mascara, axis=1)
    columnas = np.any(mascara, axis=0)
    indices_filas = np.where(filas)[0]
    indices_columnas = np.where(columnas)[0]
    
    if len(indices_filas) == 0 or len(indices_columnas) == 0:
        return np.zeros(784, dtype=np.float32)
    
    y_min, y_max = indices_filas[0], indices_filas[-1]
    x_min, x_max = indices_columnas[0], indices_columnas[-1]
    
    # Recortar
    imagen_recortada = imagen_gray[y_min:y_max+1, x_min:x_max+1]
    
    # === IGUAL A EMNIST: Preservar aspect ratio ===
    alto, ancho = imagen_recortada.shape
    
    # Calcular tamaño objetivo (20x20 como EMNIST, luego centrar en 28x28)
    tamano_objetivo = 20
    
    # Escalar preservando aspect ratio
    if alto > ancho:
        nuevo_alto = tamano_objetivo
        nuevo_ancho = int(ancho * tamano_objetivo / alto)
    else:
        nuevo_ancho = tamano_objetivo
        nuevo_alto = int(alto * tamano_objetivo / ancho)
    
    # Evitar tamaños 0
    nuevo_alto = max(1, nuevo_alto)
    nuevo_ancho = max(1, nuevo_ancho)
    
    # Redimensionar a tamaño intermedio
    imagen_pil = Image.fromarray(imagen_recortada)
    imagen_escalada = imagen_pil.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    imagen_escalada_np = np.array(imagen_escalada, dtype=np.uint8)
    
    # Centrar en imagen 28x28 (como EMNIST)
    imagen_final = np.zeros((28, 28), dtype=np.uint8)
    offset_y = (28 - nuevo_alto) // 2
    offset_x = (28 - nuevo_ancho) // 2
    imagen_final[offset_y:offset_y+nuevo_alto, offset_x:offset_x+nuevo_ancho] = imagen_escalada_np
    
    # Aplicar Gaussian Blur ligero (como EMNIST tiene variaciones naturales)
    imagen_pil_final = Image.fromarray(imagen_final)
    from PIL import ImageFilter
    imagen_pil_final = imagen_pil_final.filter(ImageFilter.GaussianBlur(radius=0.5))
    imagen_final = np.array(imagen_pil_final, dtype=np.float32)
    
    # Normalizar [0, 1]
    imagen_final = imagen_final / 255.0
    
    # Aplicar threshold suave para limpiar ruido
    imagen_final = np.where(imagen_final > 0.05, imagen_final, 0)

    return imagen_final.flatten()


def mostrar_top_predicciones(probabilidades: np.ndarray, top_k: int = 5) -> None:
    """Visualizar las predicciones con mayor probabilidad."""
    vector = probabilidades.flatten()
    top_indices = np.argsort(vector)[-top_k:][::-1]

    for posicion, indice in enumerate(top_indices, start=1):
        etiqueta = DEFAULT_LABEL_MAP.get_label(int(indice))
        probabilidad = float(vector[indice])

        # Determinar el color según la confianza
        if posicion == 1:
            emoji = "🥇"
            color = "🟢" if probabilidad > 0.7 else "🟡" if probabilidad > 0.4 else "🔴"
        elif posicion == 2:
            emoji = "🥈"
            color = ""
        elif posicion == 3:
            emoji = "🥉"
            color = ""
        else:
            emoji = f"{posicion}."
            color = ""

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.markdown(f"**{emoji} '{etiqueta}'** {color}")
        with col2:
            st.progress(probabilidad)
        with col3:
            st.write(f"{probabilidad*100:.1f}%")


def main() -> None:
    """Punto de entrada de la aplicación."""
    st.set_page_config(
        page_title='Reconocimiento de Caracteres',
        page_icon='✍️',
        layout='wide'
    )

    st.title('✍️ Reconocimiento de Caracteres con IA')
    st.markdown('---')

    # Cargar modelo
    with st.spinner('Cargando modelo...'):
        modelo = cargar_modelo_entrenado()

    if modelo is None:
        st.stop()

    st.success('✅ Modelo cargado exitosamente')

    # Información del modelo
    with st.expander('📊 Información del Modelo'):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Arquitectura:** {modelo.capas}")
            st.write(f"**Activaciones:** {modelo.activaciones}")
        with col2:
            st.write(f"**Clases:** {modelo.capas[-1]} caracteres (A-Z, a-z, 0-9, !@#...)")
            st.write(f"**BatchNorm:** {'✅ Sí' if modelo.use_batch_norm else '❌ No'}")

    st.markdown('---')

    # Layout en columnas
    col_canvas, col_resultado = st.columns([1, 1])

    with col_canvas:
        st.markdown('### 🎨 Dibuja un Carácter')

        # Configuración del canvas
        stroke_width = st.slider('Grosor del trazo:', 1, 30, 15, key='stroke')

        # Canvas dibujable
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparente
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",  # Blanco
            background_color="#000000",  # Fondo negro
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True,
        )

        col_borrar, col_predecir = st.columns(2)
        with col_borrar:
            if st.button('🗑️ Borrar', use_container_width=True):
                st.rerun()

        with col_predecir:
            predecir_btn = st.button('🔍 Predecir', type='primary', use_container_width=True)

    with col_resultado:
        st.markdown('### 📈 Resultados')

        if canvas_result.image_data is not None and predecir_btn:
            # Preprocesar la imagen
            entrada = preprocesar_imagen_canvas(canvas_result.image_data)

            # Verificar si hay contenido dibujado
            if np.sum(entrada) < 0.01:
                st.warning('⚠️ No se detectó ningún dibujo. Por favor, dibuja un carácter.')
            else:
                # Mostrar imagen preprocesada
                with st.expander('🔎 Ver imagen procesada (28x28)', expanded=True):
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.markdown("**Original (recortada)**")
                        img_procesada = entrada.reshape(28, 28)
                        st.image(img_procesada, caption='Imagen enviada al modelo', width=200, clamp=True)
                    with col_img2:
                        st.markdown("**Con threshold**")
                        img_threshold = np.where(img_procesada > 0.3, img_procesada, 0)
                        st.image(img_threshold, caption='Threshold aplicado', width=200, clamp=True)

                # Hacer predicción
                with st.spinner('Analizando...'):
                    probabilidades = modelo.predecir_probabilidades(entrada.reshape(1, -1))

                # Resultado principal
                indice_predicho = int(np.argmax(probabilidades))
                confianza = float(probabilidades[0, indice_predicho])
                etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)

                # Determinar el nivel de confianza
                if confianza > 0.8:
                    confianza_color = "🟢"
                    confianza_texto = "Alta"
                elif confianza > 0.5:
                    confianza_color = "🟡"
                    confianza_texto = "Media"
                else:
                    confianza_color = "🔴"
                    confianza_texto = "Baja"

                st.markdown(f"### {confianza_color} Predicción: **'{etiqueta_predicha}'**")
                st.metric(
                    label="Confianza",
                    value=f"{confianza*100:.1f}%",
                    delta=confianza_texto
                )
                st.progress(confianza)

                st.markdown('---')
                st.markdown('#### 🏆 Top 5 Predicciones')
                mostrar_top_predicciones(probabilidades, top_k=5)

                # Consejos si la confianza es baja
                if confianza < 0.6:
                    st.markdown('---')
                    st.warning(f'''
                    **⚠️ Confianza baja ({confianza*100:.1f}%) - Consejos:**
                    
                    🎯 **Para mejor reconocimiento:**
                    - **Tamaño**: Dibuja el carácter ocupando el 70-80% del canvas
                    - **Centrado**: Centra bien el dibujo (no en las esquinas)
                    - **Grosor**: Usa grosor 12-18 (deslizador arriba)
                    - **Continuidad**: Traza líneas continuas, evita segmentos
                    - **Claridad**: Respeta la forma del carácter (O circular, 0 ovalado)
                    
                    🔤 **Caracteres similares confundibles:**
                    - **0 vs O/o**: El cero (0) es más ovalado/alargado
                    - **1 vs I/l**: El uno (1) puede tener base, la i/I tienen puntos
                    - **5 vs S**: El cinco (5) tiene ángulos, la S es curva
                    - **8 vs B**: El ocho (8) es simétrico, la B tiene línea vertical
                    
                    💡 **Prueba:**
                    1. Borrar y redibujar más grande
                    2. Ver las predicciones top 5 (abajo)
                    3. Ajustar el grosor del trazo
                    ''')
                elif confianza < 0.8:
                    st.info('''
                    💡 **Confianza media - Sugerencias:**
                    - Dibuja más claro y definido
                    - Asegúrate de centrar bien el carácter
                    - Verifica el top 5 para ver alternativas
                    ''')
        else:
            st.info('👈 Dibuja un carácter en el canvas y presiona "Predecir"')

            # Mostrar ejemplos
            st.markdown('#### 📝 Caracteres soportados:')
            st.markdown('**Mayúsculas:** A-Z (26 letras)')
            st.markdown('**Minúsculas:** a-z (26 letras)')
            st.markdown('**Números:** 0-9 (10 dígitos)')
            st.markdown('**Especiales:** ! @ # $ % & * ( ) - + = , . ; : \' " < > / ? | ~ `')

    st.markdown('---')
    st.markdown('### 💡 Consejos de Uso')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('**✏️ Dibujo**\n- Dibuja claro y centrado\n- Usa trazos continuos\n- Llena bien el espacio')
    with col2:
        st.markdown('**🎯 Precisión**\n- Respeta las formas\n- Evita adornos extras\n- Sigue la dirección normal')
    with col3:
        st.markdown('**🔄 Mejora**\n- Prueba diferentes grosores\n- Borra y redibuja si es necesario\n- Observa el top 5 de predicciones')


if __name__ == '__main__':
    main()
