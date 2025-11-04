"""
Character Recognition Web Application

Production-ready Streamlit application for handwritten character recognition
using a fine-tuned convolutional neural network (CNN v2).

Model: CNN v2 Finetuned (83.80% validation accuracy)
Classes: 94 characters (A-Z, a-z, 0-9, symbols)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned


st.set_page_config(
    page_title="Character Recognition",
    page_icon="🔤",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Load CNN model. Cached for performance."""
    return CNNPredictor_v2_Finetuned()


def preprocess_canvas(canvas_data: np.ndarray) -> np.ndarray:
    """
    Preprocess canvas drawing for model inference.
    
    Args:
        canvas_data: RGBA image from canvas (H, W, 4)
    
    Returns:
        Normalized grayscale image (28, 28) in range [0, 1]
    """
    img_rgb = canvas_data[:, :, :3]
    
    img_pil = Image.fromarray(img_rgb.astype('uint8'))
    img_gray = img_pil.convert('L')
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img_resized)
    img_normalized = img_array.astype(np.float32) / 255.0
    
    return img_normalized


def is_canvas_empty(canvas_data: np.ndarray) -> bool:
    """
    Check if canvas contains any drawing.
    
    Args:
        canvas_data: RGBA image from canvas
    
    Returns:
        True if canvas is empty, False otherwise
    """
    if canvas_data is None:
        return True
    
    alpha_channel = canvas_data[:, :, 3]
    drawn_pixels = np.sum(alpha_channel > 200)
    
    return drawn_pixels < 50


def main():
    st.title("Character Recognition System")
    st.markdown("""
    Draw a character in the canvas below and click **Predict** to recognize it.
    
    **Supported characters:** A-Z (uppercase), a-z (lowercase), 0-9, and symbols
    """)
    
    model = load_model()
    
    st.markdown("### Draw Here")
    
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("Predict", use_container_width=True, type="primary")
    
    if predict_button:
        if canvas_result.image_data is None:
            st.warning("Canvas is empty. Please draw a character first.")
            return
        
        if is_canvas_empty(canvas_result.image_data):
            st.warning("No drawing detected. Please draw a character.")
            return
        
        img_processed = preprocess_canvas(canvas_result.image_data)
        
        char, prob, top5 = model.predict(img_processed)
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 80px; margin: 0;'>{char}</h1>
            <p style='font-size: 24px; color: #666;'>{prob*100:.1f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View Top 5 Predictions"):
            for i, (label, confidence) in enumerate(top5, 1):
                st.progress(confidence, text=f"{i}. **{label}** — {confidence*100:.1f}%")
        
        with st.expander("View Preprocessed Image"):
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.image(img_processed, width=140, caption="28×28 grayscale", use_container_width=False)
            
            st.markdown(f"""
            **Image statistics:**
            - Mean: {img_processed.mean():.3f}
            - Std: {img_processed.std():.3f}
            - Min: {img_processed.min():.3f}
            - Max: {img_processed.max():.3f}
            """)


if __name__ == "__main__":
    main()
