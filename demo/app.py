"""
Aplicación web para probar el reconocimiento de caracteres personalizados.
"""

import streamlit as st
import numpy as np
import pickle
import sys
import os
from PIL import Image, ImageOps

# Configurar rutas
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if RUTA_RAIZ not in sys.path:
    sys.path.insert(0, RUTA_RAIZ)

# Importación directa de archivos individuales para evitar problemas circulares
sys.path.insert(0, os.path.join(RUTA_RAIZ, 'src'))

# Importar directamente lo necesario
from label_map import DEFAULT_LABEL_MAP
from config import PATHS

# Cargar modelo entrenado
@st.cache_resource
def cargar_modelo():
    modelo_path = os.path.join(RUTA_RAIZ, PATHS['modelos'], "modelo_entrenado.pkl")
    
    if not os.path.exists(modelo_path):
        st.error("❌ No se encontró el modelo entrenado.")
        st.info("Ejecute primero: `python src/trainer.py`")
        return None
    
    try:
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        st.success("✅ Modelo cargado exitosamente")
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        
        # Mostrar información adicional de debug
        with st.expander("🔧 Información de Debug"):
            st.write(f"**Ruta del modelo:** {modelo_path}")
            st.write(f"**Archivo existe:** {os.path.exists(modelo_path)}")
            st.write(f"**Error completo:** {repr(e)}")
        
        return None

# Preprocesar imagen
def preprocesar_imagen(imagen_pil):
    """Preprocesar imagen para el modelo."""
    # Convertir a escala de grises
    imagen_pil = imagen_pil.convert('L')
    
    # Redimensionar a 28x28
    imagen_pil = imagen_pil.resize((28, 28))
    
    # Convertir a numpy y normalizar
    imagen_np = np.array(imagen_pil).astype(np.float32)
    imagen_np = imagen_np / 255.0  # Normalizar a [0, 1]
    
    return imagen_np.flatten()

def main():
    """Función principal de la aplicación."""
    # Configurar página
    st.set_page_config(
        page_title="Reconocimiento de Caracteres", 
        page_icon="🔤",
        layout="centered"
    )
    
    st.title("🔤 Reconocimiento de Caracteres Personalizados")
    st.markdown("---")
    st.write("Sube una imagen de un carácter para probar el modelo de red neuronal.")
    
    # Cargar modelo
    modelo = cargar_modelo()
    if modelo is None:
        st.stop()
    
    # Mostrar información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write(f"**Neuronas de entrada:** {modelo.entrada_neuronas}")
        st.write(f"**Neuronas ocultas:** {modelo.oculta_neuronas}")
        st.write(f"**Neuronas de salida:** {modelo.salida_neuronas}")
        st.write(f"**Clases soportadas:** {DEFAULT_LABEL_MAP.get_num_classes()}")
        st.write(f"**Etiquetas:** {', '.join(DEFAULT_LABEL_MAP.labels[:10])}...")
    
    # Subir imagen
    st.markdown("### 📷 Subir Imagen")
    imagen_subida = st.file_uploader(
        "Selecciona una imagen de un carácter:", 
        type=["png", "jpg", "jpeg", "bmp"]
    )
    
    if imagen_subida:
        # Mostrar imagen original
        imagen_pil = Image.open(imagen_subida)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Imagen Original**")
            st.image(imagen_pil, caption="Imagen subida", use_column_width=True)
        
        with col2:
            # Preprocesar imagen
            entrada = preprocesar_imagen(imagen_pil)
            
            # Mostrar imagen procesada
            st.markdown("**Imagen Procesada (28x28)**")
            imagen_procesada = entrada.reshape(28, 28)
            st.image(imagen_procesada, caption="Imagen procesada", use_column_width=True, clamp=True)
        
        # Hacer predicción
        if st.button("🔍 Predecir Carácter", type="primary"):
            try:
                # Obtener predicción del modelo
                salida = modelo.predecir(entrada)
                
                # Encontrar la clase con mayor probabilidad
                indice_predicho = np.argmax(salida)
                confianza = salida[indice_predicho]
                etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)
                
                # Mostrar resultado
                st.markdown("### 🎯 Resultado de la Predicción")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Carácter Predicho", 
                        value=f"'{etiqueta_predicha}'"
                    )
                
                with col2:
                    st.metric(
                        label="Confianza", 
                        value=f"{float(confianza):.3f}",
                        delta=f"{float(confianza)*100:.1f}%"
                    )
                
                # Mostrar barra de confianza
                st.progress(float(confianza))
                
                # Mostrar top 5 predicciones
                st.markdown("#### 📊 Top 5 Predicciones")
                
                # Obtener top 5 índices
                top_indices = np.argsort(salida.flatten())[-5:][::-1]
                
                for i, idx in enumerate(top_indices):
                    etiqueta = DEFAULT_LABEL_MAP.get_label(idx)
                    probabilidad = float(salida[idx])
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.write(f"**{i+1}. '{etiqueta}'**")
                    
                    with col2:
                        st.progress(probabilidad)
                    
                    with col3:
                        st.write(f"{probabilidad:.3f}")
                
            except Exception as e:
                st.error(f"❌ Error durante la predicción: {str(e)}")
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 💡 Consejos:")
    st.write("- Use imágenes claras con buen contraste")
    st.write("- Los caracteres deben estar centrados")
    st.write("- Funciona mejor con fondos simples")
    st.write("- El modelo puede distinguir entre mayúsculas y minúsculas")
    
    # Estadísticas del modelo
    with st.sidebar:
        st.markdown("## 📊 Estadísticas del Sistema")
        st.write(f"**Total de clases:** {DEFAULT_LABEL_MAP.get_num_classes()}")
        st.write("**Arquitectura:** 784 → 128 → 52")
        st.write("**Entrenamiento:** 100 épocas")
        st.write("**Dataset:** Imágenes sintéticas")
        
        st.markdown("---")
        st.markdown("**Desarrollado con:**")
        st.write("- Red neuronal desde cero")
        st.write("- Streamlit para la interfaz")
        st.write("- NumPy para computación")

if __name__ == "__main__":
    main()
    salida = modelo.predecir(entrada)
    indice = np.argmax(salida)
    etiqueta = mapa_etiquetas.get_label(indice)

    st.success(f"✅ Carácter reconocido: **{etiqueta}**")
    st.bar_chart(salida.flatten())
