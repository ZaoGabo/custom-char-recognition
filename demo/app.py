"""
App Final - CNN v2 Finetuned con preprocesamiento CORRECTO
92% de accuracy en caracteres generados
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import requests
from streamlit_drawable_canvas import st_canvas
from PIL import Image

API_URL = "http://127.0.0.1:8000"

from src.cnn_predictor_v2_finetuned import cargar_cnn_predictor_v2_finetuned


def preprocess_canvas_CORRECTO(canvas_data: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento CORRECTO para el canvas.
    
    EMNIST: Fondo NEGRO (0), Caracteres BLANCOS (255)
    Canvas: Fondo NEGRO RGB=(0,0,0), Trazo BLANCO RGB=(255,255,255)
    
    ✅ Ya está en el formato correcto, NO invertir colores.
    """
    # Extraer RGB (ignorar alfa)
    img_rgb = canvas_data[:, :, :3]
    img_pil = Image.fromarray(img_rgb.astype('uint8'))
    img_gray = np.array(img_pil.convert('L'))
    
    # Detectar si hay contenido
    if np.max(img_gray) < 10:
        return np.zeros((28, 28), dtype=np.float32)
    
    # Bounding box (detectar el trazo blanco)
    threshold = 30
    mask = img_gray > threshold
    
    if not np.any(mask):
        return np.zeros((28, 28), dtype=np.float32)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    
    y_min, y_max = row_indices[0], row_indices[-1]
    x_min, x_max = col_indices[0], col_indices[-1]
    
    # Recortar con padding
    padding = 4
    y_min = max(0, y_min - padding)
    y_max = min(img_gray.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(img_gray.shape[1] - 1, x_max + padding)
    
    img_cropped = img_gray[y_min:y_max+1, x_min:x_max+1]
    
    # Resize manteniendo aspect ratio (como EMNIST)
    height, width = img_cropped.shape
    target_size = 20  # EMNIST usa 20x20 dentro de 28x28
    
    if height > width:
        new_height = target_size
        new_width = max(1, int(width * target_size / height))
    else:
        new_width = target_size
        new_height = max(1, int(height * target_size / width))
    
    img_pil = Image.fromarray(img_cropped)
    img_scaled = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_scaled_np = np.array(img_scaled, dtype=np.float32)
    
    # Centrar en 28x28 con fondo NEGRO
    img_final = np.zeros((28, 28), dtype=np.float32)
    offset_y = (28 - new_height) // 2
    offset_x = (28 - new_width) // 2
    img_final[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = img_scaled_np
    
    # Normalizar a [0, 1]
    img_final = img_final / 255.0
    
    return img_final


def get_color_by_confidence(conf: float) -> str:
    """Color según nivel de confianza"""
    if conf >= 0.9:
        return "#d4edda"  # Verde claro
    elif conf >= 0.7:
        return "#fff3cd"  # Amarillo claro
    elif conf >= 0.5:
        return "#ffe5d0"  # Naranja claro
    else:
        return "#f8d7da"  # Rojo claro


def get_emoji_by_confidence(conf: float) -> str:
    """Emoji según confianza"""
    if conf >= 0.95:
        return "🎯"
    elif conf >= 0.85:
        return "✅"
    elif conf >= 0.7:
        return "👍"
    elif conf >= 0.5:
        return "⚠️"
    else:
        return "❌"


def main():
    st.set_page_config(
        page_title="Reconocimiento de Caracteres",
        page_icon="✍️",
        layout="wide"
    )
    
    # Título con gradiente
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 48px;">
            ✍️ Reconocimiento de Caracteres
        </h1>
        <p style="color: white; text-align: center; font-size: 20px; margin: 10px 0 0 0;">
            CNN v2 Finetuned - 92% Accuracy 🎯
        </p>
        <p style="color: rgba(255,255,255,0.8); text-align: center; font-size: 16px; margin: 5px 0 0 0;">
            A-Z, a-z, 0-9 + símbolos especiales (94 clases)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuración Sidebar
    st.sidebar.title("⚙️ Configuración")
    inference_mode = st.sidebar.radio("Modo de Inferencia", ["Local (PyTorch)", "API (ONNX)"])

    # Cargar modelo solo si es local
    if inference_mode == "Local (PyTorch)":
        if 'predictor' not in st.session_state:
            with st.spinner('🔄 Cargando modelo CNN...'):
                st.session_state.predictor = cargar_cnn_predictor_v2_finetuned()
                st.success('✅ Modelo cargado correctamente')
    
    # Layout en 3 columnas
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown("### ✏️ Dibuja un carácter:")
        
        # Instrucciones
        with st.expander("📖 Instrucciones"):
            st.markdown("""
            - **Dibuja** un carácter (letra, número o símbolo)
            - **Presiona** el botón para reconocer
            - **Limpia** el canvas para probar otro
            
            **Nota:** Algunos caracteres similares pueden confundirse:
            - I (mayúscula) ↔ l (minúscula) ↔ | (barra)
            - o (minúscula) ↔ O (mayúscula)
            - c (minúscula) ↔ C (mayúscula)
            """)
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=18,
            stroke_color="rgba(255, 255, 255, 1)",
            background_color="rgba(0, 0, 0, 1)",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_final"
        )
        
        # Botones
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔍 Reconocer", use_container_width=True, type="primary"):
                st.session_state.recognize = True
        with col_btn2:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.recognize = False
                st.rerun()
    
    # Procesar y predecir
    if canvas_result.image_data is not None and st.session_state.get('recognize', False):
        img_processed = preprocess_canvas_CORRECTO(canvas_result.image_data)
        
        if np.max(img_processed) > 0:
            # Predecir
            if inference_mode == "Local (PyTorch)":
                char, conf, top5 = st.session_state.predictor.predict(img_processed)
            else:
                # API Mode
                try:
                    payload = {"image": img_processed.flatten().tolist()}
                    resp = requests.post(f"{API_URL}/predict", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        char = data['character']
                        conf = data['confidence']
                        # Convert API top5 dict to list of tuples for UI
                        top5 = [(item['character'], item['probability']) for item in data['top5']]
                    else:
                        st.error(f"Error API: {resp.status_code} - {resp.text}")
                        return
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
                    return
            
            # Mostrar predicción principal
            bg_color = get_color_by_confidence(conf)
            emoji = get_emoji_by_confidence(conf)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {bg_color} 0%, white 100%); 
                        padding: 30px; 
                        border-radius: 15px; 
                        border: 3px solid {'#28a745' if conf >= 0.9 else '#ffc107' if conf >= 0.7 else '#dc3545'};
                        margin: 20px 0;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                <h2 style="color: #2c3e50; margin: 0; text-align: center;">
                    {emoji} Predicción
                </h2>
                <div style="font-size: 120px; text-align: center; margin: 20px 0; 
                            font-weight: bold; color: #2c3e50; 
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                    {char}
                </div>
                <div style="text-align: center; font-size: 32px; color: #555; font-weight: bold;">
                    {conf*100:.2f}% de confianza
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 5 predicciones
            st.markdown("### 🏆 Top 5 predicciones:")
            
            for i, (label, prob) in enumerate(top5, 1):
                color = get_color_by_confidence(prob)
                is_first = (i == 1)
                
                # Medalla según posición
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                medal = medals[i-1]
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {color} 0%, white 100%); 
                            padding: 15px; 
                            margin: 8px 0; 
                            border-radius: 10px;
                            border-left: 6px solid {'#28a745' if is_first else '#6c757d'};
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            display: flex;
                            align-items: center;
                            justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 32px;">{medal}</span>
                        <span style="font-size: 40px; font-weight: bold; color: #2c3e50;">{label}</span>
                    </div>
                    <span style="font-size: 24px; font-weight: bold; color: #555;">{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Debug info
            with st.expander("🔬 Información técnica"):
                col_debug1, col_debug2 = st.columns(2)
                
                with col_debug1:
                    st.image(img_processed, caption="Imagen procesada (28x28)", 
                            width=280, clamp=True)
                
                with col_debug2:
                    st.markdown("**Estadísticas:**")
                    st.write(f"- Shape: {img_processed.shape}")
                    st.write(f"- Min: {img_processed.min():.3f}")
                    st.write(f"- Max: {img_processed.max():.3f}")
                    st.write(f"- Mean: {img_processed.mean():.3f}")
                    st.write(f"- Std: {img_processed.std():.3f}")
                    st.write(f"- Dtype: {img_processed.dtype}")
                    
                    st.markdown("**Formato:**")
                    st.success("✅ Fondo negro (~0.0)")
                    st.success("✅ Carácter blanco (~1.0)")
                    st.success("✅ Normalizado [0, 1]")
        else:
            st.warning("⚠️ No se detectó ningún trazo. Dibuja algo en el canvas.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🧠 <b>CNN v2 Finetuned</b> - Entrenado con 30 épocas</p>
        <p>📊 Accuracy: <b>83.80%</b> en validación | <b>92%</b> en pruebas sintéticas</p>
        <p>⚙️ Dispositivo: <b>CUDA (GPU)</b></p>
        <p style="font-size: 12px; margin-top: 10px;">
            💡 <i>Nota: Caracteres similares como I/l/|, o/O, c/C pueden confundirse</i>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
