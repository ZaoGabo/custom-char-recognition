"""
Script para probar el modelo entrenado de reconocimiento de caracteres.
"""

import os
import pickle
import numpy as np
from PIL import Image
import sys

# Agregar src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

from config import PATHS, DATA_CONFIG
from label_map import DEFAULT_LABEL_MAP

def cargar_modelo():
    """Cargar el modelo entrenado."""
    modelo_path = os.path.join(PATHS['modelos'], "modelo_entrenado.pkl")
    
    if not os.path.exists(modelo_path):
        print("❌ No se encontró el modelo entrenado.")
        print("Ejecute primero: python src/trainer.py")
        return None
    
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    
    print("✅ Modelo cargado exitosamente")
    print(f"   - Neuronas de entrada: {modelo.entrada_neuronas}")
    print(f"   - Neuronas ocultas: {modelo.oculta_neuronas}")
    print(f"   - Neuronas de salida: {modelo.salida_neuronas}")
    return modelo

def predecir_imagen(modelo, ruta_imagen):
    """
    Predecir el carácter de una imagen.
    
    Args:
        modelo: Modelo de red neuronal cargado
        ruta_imagen: Ruta a la imagen a predecir
        
    Returns:
        dict: Resultado de la predicción
    """
    try:
        # Cargar y procesar imagen
        img = Image.open(ruta_imagen).convert('L')  # Escala de grises
        img = img.resize(DATA_CONFIG['tamano_imagen'])  # Redimensionar a 28x28
        
        # Normalizar píxeles a [0, 1]
        img_array = np.array(img) / 255.0
        img_flat = img_array.flatten()  # Aplanar a 1D
        
        # Hacer predicción
        salidas = modelo.predecir(img_flat)
        
        # Encontrar la clase con mayor probabilidad
        indice_predicho = np.argmax(salidas)
        confianza = salidas[indice_predicho]
        etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)
        
        return {
            'etiqueta': etiqueta_predicha,
            'indice': indice_predicho,
            'confianza': float(confianza),
            'probabilidades': salidas.flatten()
        }
    
    except Exception as e:
        print(f"❌ Error procesando imagen {ruta_imagen}: {str(e)}")
        return None

def probar_con_imagenes_muestra():
    """Probar el modelo con algunas imágenes de muestra."""
    print("🧪 Probando modelo con imágenes de muestra...")
    
    modelo = cargar_modelo()
    if modelo is None:
        return
    
    # Buscar algunas imágenes de ejemplo
    data_raw = PATHS['datos_crudos']
    imagenes_probadas = 0
    aciertos = 0
    
    for carpeta in os.listdir(data_raw)[:10]:  # Solo las primeras 10 carpetas
        carpeta_path = os.path.join(data_raw, carpeta)
        if not os.path.isdir(carpeta_path):
            continue
            
        # Obtener la etiqueta real de la carpeta
        if carpeta.endswith('_upper'):
            etiqueta_real = carpeta[0].upper()
        elif carpeta.endswith('_lower'):
            etiqueta_real = carpeta[0].lower()
        else:
            etiqueta_real = carpeta
        
        # Tomar la primera imagen de la carpeta
        archivos = [f for f in os.listdir(carpeta_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not archivos:
            continue
            
        imagen_path = os.path.join(carpeta_path, archivos[0])
        resultado = predecir_imagen(modelo, imagen_path)
        
        if resultado:
            imagenes_probadas += 1
            es_correcto = resultado['etiqueta'] == etiqueta_real
            if es_correcto:
                aciertos += 1
            
            print(f"📁 {carpeta} → Esperado: '{etiqueta_real}' | Predicho: '{resultado['etiqueta']}' | Confianza: {resultado['confianza']:.3f} | {'✅' if es_correcto else '❌'}")
    
    if imagenes_probadas > 0:
        accuracy = aciertos / imagenes_probadas
        print(f"\\n📊 Resultados de la prueba:")
        print(f"   - Imágenes probadas: {imagenes_probadas}")
        print(f"   - Aciertos: {aciertos}")
        print(f"   - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

def mostrar_distribución_clases():
    """Mostrar la distribución de clases en los datos."""
    print("\\n📊 Distribución de clases en los datos:")
    
    data_raw = PATHS['datos_crudos']
    conteos = {}
    
    for carpeta in os.listdir(data_raw):
        carpeta_path = os.path.join(data_raw, carpeta)
        if not os.path.isdir(carpeta_path) or carpeta.startswith('.'):
            continue
            
        # Contar imágenes
        archivos = [f for f in os.listdir(carpeta_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Obtener etiqueta
        if carpeta.endswith('_upper'):
            etiqueta = carpeta[0].upper()
        elif carpeta.endswith('_lower'):
            etiqueta = carpeta[0].lower()
        else:
            etiqueta = carpeta
            
        conteos[etiqueta] = len(archivos)
    
    # Mostrar conteos organizados
    print("\\nMayúsculas:")
    for letra in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if letra in conteos:
            print(f"   {letra}: {conteos[letra]} imágenes")
    
    print("\\nMinúsculas:")
    for letra in 'abcdefghijklmnopqrstuvwxyz':
        if letra in conteos:
            print(f"   {letra}: {conteos[letra]} imágenes")
    
    total = sum(conteos.values())
    print(f"\\nTotal: {total} imágenes en {len(conteos)} clases")

def main():
    """Función principal de prueba."""
    print("=" * 60)
    print("🧪 PRUEBA DEL MODELO DE RECONOCIMIENTO DE CARACTERES")
    print("=" * 60)
    
    # Mostrar información del dataset
    mostrar_distribución_clases()
    
    print("\\n" + "=" * 60)
    
    # Probar modelo
    probar_con_imagenes_muestra()
    
    print("\\n" + "=" * 60)
    print("✅ Prueba completada")
    print("💡 Para usar la aplicación web: streamlit run demo/app.py")

if __name__ == "__main__":
    main()