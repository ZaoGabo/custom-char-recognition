"""Aplicación Streamlit SIMPLIFICADA sin PyArrow - Usando Upload de Imágenes."""

from __future__ import annotations

import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw

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


def preprocesar_imagen(imagen_pil: Image.Image) -> np.ndarray:
    """
    Preprocesar imagen para que coincida con el entrenamiento.
    
    Pasos críticos:
    1. Convertir a escala de grises
    2. Invertir colores SI es necesario (detectar automáticamente)
    3. Centrar el contenido
    4. Redimensionar a 28x28
    5. Normalizar [0, 1]
    """
    # Convertir a escala de grises
    imagen_gray = imagen_pil.convert('L')
    
    # Detectar si necesitamos invertir (si el fondo es más claro que el contenido)
    array_temp = np.array(imagen_gray)
    promedio = np.mean(array_temp)
    
    # Si el promedio es alto, significa fondo claro (necesitamos invertir)
    if promedio > 127:
        imagen_gray = ImageOps.invert(imagen_gray)
    
    # Encontrar el bounding box del contenido para centrarlo
    bbox = imagen_gray.getbbox()
    
    if bbox is None:
        # No hay contenido
        return np.zeros(784, dtype=np.float32)
    
    # Recortar al contenido
    imagen_recortada = imagen_gray.crop(bbox)
    
    # Calcular el tamaño del cuadrado con padding del 40%
    ancho, alto = imagen_recortada.size
    tamano_max = max(ancho, alto)
    tamano_nuevo = int(tamano_max * 1.4)
    
    # Crear imagen cuadrada con fondo negro
    imagen_cuadrada = Image.new('L', (tamano_nuevo, tamano_nuevo), color=0)
    
    # Pegar la imagen recortada en el centro
    offset_x = (tamano_nuevo - ancho) // 2
    offset_y = (tamano_nuevo - alto) // 2
    imagen_cuadrada.paste(imagen_recortada, (offset_x, offset_y))
    
    # Redimensionar a 28x28 con antialiasing
    imagen_final = imagen_cuadrada.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convertir a array y normalizar
    array_final = np.array(imagen_final, dtype=np.float32) / 255.0
    
    return array_final.flatten()


def crear_canvas_simple(size=280):
    """Crear un canvas simple usando HTML/CSS."""
    html_code = f"""
    <div style="text-align: center;">
        <canvas id="drawCanvas" width="{size}" height="{size}" 
                style="border: 2px solid #666; background-color: black; cursor: crosshair; border-radius: 8px;">
        </canvas>
        <br><br>
        <button onclick="clearCanvas()" 
                style="padding: 10px 20px; font-size: 16px; margin: 5px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer;">
            🗑️ Borrar Canvas
        </button>
        <button onclick="saveCanvas()" 
                style="padding: 10px 20px; font-size: 16px; margin: 5px; background-color: #0068c9; color: white; border: none; border-radius: 5px; cursor: pointer;">
            💾 Guardar y Predecir
        </button>
    </div>
    
    <script>
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';
        
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', (e) => {{
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {{
                clientX: touch.clientX,
                clientY: touch.clientY
            }});
            canvas.dispatchEvent(mouseEvent);
        }});
        
        canvas.addEventListener('touchmove', (e) => {{
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {{
                clientX: touch.clientX,
                clientY: touch.clientY
            }});
            canvas.dispatchEvent(mouseEvent);
        }});
        
        canvas.addEventListener('touchend', (e) => {{
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {{}});
            canvas.dispatchEvent(mouseEvent);
        }});
        
        function startDrawing(e) {{
            drawing = true;
            draw(e);
        }}
        
        function draw(e) {{
            if (!drawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }}
        
        function stopDrawing() {{
            drawing = false;
            ctx.beginPath();
        }}
        
        function clearCanvas() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }}
        
        function saveCanvas() {{
            const dataURL = canvas.toDataURL('image/png');
            // Crear elemento de descarga temporal
            const link = document.createElement('a');
            link.download = 'canvas_drawing.png';
            link.href = dataURL;
            link.click();
            
            alert('Canvas guardado! Por favor sube la imagen usando el botón "Browse files" arriba.');
        }}
    </script>
    """
    return html_code


def mostrar_top_predicciones(probabilidades: np.ndarray, top_k: int = 5) -> None:
    """Visualizar las predicciones con mayor probabilidad."""
    vector = probabilidades.flatten()
    top_indices = np.argsort(vector)[-top_k:][::-1]
    
    for posicion, indice in enumerate(top_indices, start=1):
        etiqueta = DEFAULT_LABEL_MAP.get_label(int(indice))
        probabilidad = float(vector[indice])
        
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
    st.markdown('**Versión Simplificada - Sin PyArrow**')
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
            st.write(f"**Clases:** {modelo.capas[-1]} caracteres (A-Z, a-z)")
            st.write(f"**BatchNorm:** {'✅ Sí' if modelo.use_batch_norm else '❌ No'}")
    
    st.markdown('---')
    
    # Tabs para diferentes modos
    tab1, tab2 = st.tabs(["📤 Subir Imagen", "🎨 Dibujar (Canvas HTML)"])
    
    with tab1:
        st.markdown('### 📤 Sube una Imagen del Carácter')
        st.info('💡 Sube una imagen de un carácter escrito (preferiblemente blanco sobre fondo negro)')
        
        uploaded_file = st.file_uploader(
            "Selecciona una imagen:",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key='upload'
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('#### 🖼️ Imagen Original')
                imagen_original = Image.open(uploaded_file)
                st.image(imagen_original, use_column_width=True)
            
            with col2:
                st.markdown('#### 🔍 Imagen Procesada (28x28)')
                entrada = preprocesar_imagen(imagen_original)
                img_procesada = entrada.reshape(28, 28)
                st.image(img_procesada, use_column_width=True, clamp=True)
            
            if st.button('🔍 Predecir Carácter', type='primary', key='predict_upload'):
                if np.sum(entrada) < 0.01:
                    st.warning('⚠️ La imagen parece estar vacía o muy oscura.')
                else:
                    with st.spinner('Analizando...'):
                        probabilidades = modelo.predecir_probabilidades(entrada.reshape(1, -1))
                    
                    indice_predicho = int(np.argmax(probabilidades))
                    confianza = float(probabilidades[0, indice_predicho])
                    etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)
                    
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
                    st.metric(label="Confianza", value=f"{confianza*100:.1f}%", delta=confianza_texto)
                    st.progress(confianza)
                    
                    st.markdown('---')
                    st.markdown('#### 🏆 Top 5 Predicciones')
                    mostrar_top_predicciones(probabilidades, top_k=5)
                    
                    if confianza < 0.5:
                        st.markdown('---')
                        st.info('''
                        **💡 Consejos para mejorar:**
                        - Usa imágenes con fondo negro y letra blanca
                        - Centra el carácter en la imagen
                        - Asegúrate de que el carácter sea claro y grande
                        ''')
    
    with tab2:
        st.markdown('### 🎨 Dibuja un Carácter')
        st.warning('⚠️ **Instrucciones:** Dibuja en el canvas negro, luego presiona "💾 Guardar y Predecir", descarga la imagen y súbela en la pestaña "📤 Subir Imagen"')
        
        # Mostrar canvas HTML
        st.components.v1.html(crear_canvas_simple(), height=400)
        
        st.markdown('---')
        st.info('''
        **📝 Cómo usar:**
        1. Dibuja tu carácter en el canvas negro (con trazo blanco)
        2. Presiona "💾 Guardar y Predecir" para descargar la imagen
        3. Ve a la pestaña "📤 Subir Imagen"
        4. Sube la imagen descargada
        5. Presiona "🔍 Predecir Carácter"
        ''')
    
    st.markdown('---')
    st.markdown('### 💡 Consejos Generales')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('**✏️ Formato**\n- Fondo negro, letra blanca\n- Letra centrada\n- Buen contraste')
    with col2:
        st.markdown('**🎯 Calidad**\n- Letra clara y legible\n- Tamaño adecuado\n- Sin ruido extra')
    with col3:
        st.markdown('**📊 Caracteres**\n- A-Z (mayúsculas)\n- a-z (minúsculas)\n- 52 clases total')


if __name__ == '__main__':
    main()

