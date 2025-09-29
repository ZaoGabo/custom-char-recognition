"""
Aplicación web simple para probar el reconocimiento de caracteres.
"""

import streamlit as st
import numpy as np
import pickle
import os
import sys
from PIL import Image
import string

# Importar la clase del modelo
from modelo import RedNeuronalSimple

# Configuración
MODELO_PATH = "../models/modelo_entrenado.pkl"

# Mapeo de etiquetas (A-Z mayúsculas, a-z minúsculas)
ETIQUETAS = list(string.ascii_uppercase) + list(string.ascii_lowercase)

def cargar_modelo():
    """Cargar modelo entrenado."""
    modelo_path = os.path.join(os.path.dirname(__file__), MODELO_PATH)
    
    if not os.path.exists(modelo_path):
        st.error("❌ No se encontró el modelo entrenado.")
        st.info("Ejecute primero: `python train_simple.py`")
        return None
    
    try:
        # Asegurarse de que la clase esté disponible
        sys.modules['__main__'].RedNeuronalSimple = RedNeuronalSimple
        
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        
        # Mostrar información adicional de debug
        with st.expander("🔧 Información de Debug"):
            st.write(f"**Ruta del modelo:** {modelo_path}")
            st.write(f"**Archivo existe:** {os.path.exists(modelo_path)}")
            st.write(f"**Error completo:** {repr(e)}")
            
        return None

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
    
    st.success("✅ Modelo cargado exitosamente")
    
    # Mostrar información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write(f"**Clases soportadas:** {len(ETIQUETAS)} caracteres")
        st.write(f"**Arquitectura:** 784 → 128 → 52 neuronas")
        st.write("**Caracteres:** A-Z (mayúsculas) y a-z (minúsculas)")
    
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
        
        # Preprocesar imagen
        entrada = preprocesar_imagen(imagen_pil)
        
        with col2:
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
                etiqueta_predicha = ETIQUETAS[indice_predicho]
                
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
                    etiqueta = ETIQUETAS[idx]
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
                st.write("Detalles del error:")
                st.code(str(e))
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 💡 Consejos:")
    st.write("- Use imágenes claras con buen contraste")
    st.write("- Los caracteres deben estar centrados")
    st.write("- Funciona mejor con fondos simples")
    st.write("- El modelo puede distinguir entre mayúsculas y minúsculas")

if __name__ == "__main__":
    main()