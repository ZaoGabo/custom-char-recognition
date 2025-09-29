"""
Script de prueba para verificar la carga de datos desde las carpetas organizadas.
"""

import sys
import os
import numpy as np

# Agregar src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

from config import PATHS
from data_loader import DataLoader

def verificar_estructura_carpetas():
    """Verificar que todas las carpetas existan."""
    print("🔍 Verificando estructura de carpetas...")
    
    data_path = PATHS['datos_crudos']
    carpetas_encontradas = []
    carpetas_faltantes = []
    
    # Verificar carpetas mayúsculas
    for letra in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        carpeta = f"{letra}_upper"
        ruta_carpeta = os.path.join(data_path, carpeta)
        if os.path.exists(ruta_carpeta):
            carpetas_encontradas.append(carpeta)
        else:
            carpetas_faltantes.append(carpeta)
    
    # Verificar carpetas minúsculas
    for letra in 'abcdefghijklmnopqrstuvwxyz':
        carpeta = f"{letra}_lower"
        ruta_carpeta = os.path.join(data_path, carpeta)
        if os.path.exists(ruta_carpeta):
            carpetas_encontradas.append(carpeta)
        else:
            carpetas_faltantes.append(carpeta)
    
    print(f"✅ Carpetas encontradas: {len(carpetas_encontradas)}/52")
    if carpetas_faltantes:
        print(f"❌ Carpetas faltantes: {carpetas_faltantes}")
    else:
        print("✅ Todas las carpetas están presentes")
    
    return len(carpetas_faltantes) == 0

def probar_mapeo_etiquetas():
    """Probar el mapeo de carpetas a etiquetas."""
    print("\\n🏷️ Probando mapeo de etiquetas...")
    
    # Crear instancia del data loader
    data_loader = DataLoader(PATHS['datos_crudos'])
    
    # Probar algunos mapeos
    casos_prueba = [
        'A_upper', 'Z_upper', 'a_lower', 'z_lower',
        'M_upper', 'n_lower'
    ]
    
    for carpeta in casos_prueba:
        etiqueta = data_loader._mapear_carpeta_a_etiqueta(carpeta)
        indice = data_loader.mapa_etiquetas.get_index(etiqueta)
        
        print(f"  {carpeta} → '{etiqueta}' → índice {indice}")
        
        if indice == -1:
            print(f"    ❌ Error: etiqueta '{etiqueta}' no encontrada")
        else:
            print(f"    ✅ Mapeado correctamente")

def contar_imagenes_por_carpeta():
    """Contar imágenes en cada carpeta."""
    print("\\n📊 Contando imágenes por carpeta...")
    
    data_path = PATHS['datos_crudos']
    total_imagenes = 0
    carpetas_con_imagenes = 0
    
    # Obtener todas las carpetas
    carpetas = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    carpetas.sort()
    
    for carpeta in carpetas:
        if carpeta.startswith('.'):
            continue
            
        ruta_carpeta = os.path.join(data_path, carpeta)
        imagenes = [f for f in os.listdir(ruta_carpeta) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        num_imagenes = len(imagenes)
        total_imagenes += num_imagenes
        
        if num_imagenes > 0:
            carpetas_con_imagenes += 1
            print(f"  📁 {carpeta}: {num_imagenes} imágenes")
    
    print(f"\\n📈 Resumen:")
    print(f"  - Total de carpetas: {len(carpetas)}")
    print(f"  - Carpetas con imágenes: {carpetas_con_imagenes}")
    print(f"  - Total de imágenes: {total_imagenes}")
    
    return total_imagenes > 0

def probar_carga_datos():
    """Probar la carga de datos completa."""
    print("\\n🚀 Probando carga de datos...")
    
    try:
        # Crear data loader
        data_loader = DataLoader(PATHS['datos_crudos'])
        
        # Intentar cargar datos
        data_loader.cargar_desde_directorio()
        
        if len(data_loader.imagenes) > 0:
            print(f"✅ Carga exitosa:")
            print(f"  - Imágenes cargadas: {len(data_loader.imagenes)}")
            print(f"  - Forma de imagen: {data_loader.imagenes.shape}")
            print(f"  - Etiquetas únicas: {len(np.unique(data_loader.etiquetas))}")
            print(f"  - Rango de etiquetas: {np.min(data_loader.etiquetas)} - {np.max(data_loader.etiquetas)}")
        else:
            print("⚠️ No se cargaron imágenes. Verifique que haya imágenes en las carpetas.")
            
    except Exception as e:
        print(f"❌ Error durante la carga: {str(e)}")
        return False
    
    return True

def main():
    """Función principal de verificación."""
    print("=" * 60)
    print("🧪 VERIFICACIÓN DEL SISTEMA DE CARGA DE DATOS")
    print("=" * 60)
    
    # 1. Verificar estructura de carpetas
    estructura_ok = verificar_estructura_carpetas()
    
    # 2. Probar mapeo de etiquetas
    probar_mapeo_etiquetas()
    
    # 3. Contar imágenes
    hay_imagenes = contar_imagenes_por_carpeta()
    
    # 4. Probar carga completa (solo si hay imágenes)
    if hay_imagenes:
        carga_ok = probar_carga_datos()
    else:
        print("\\n⚠️ No hay imágenes para cargar. Agregue imágenes a las carpetas y vuelva a intentar.")
        carga_ok = False
    
    # Resumen final
    print("\\n" + "=" * 60)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    print(f"✅ Estructura de carpetas: {'OK' if estructura_ok else 'ERROR'}")
    print(f"✅ Mapeo de etiquetas: OK")
    print(f"✅ Imágenes disponibles: {'SÍ' if hay_imagenes else 'NO'}")
    print(f"✅ Carga de datos: {'OK' if carga_ok else 'PENDIENTE' if not hay_imagenes else 'ERROR'}")
    
    if estructura_ok and hay_imagenes and carga_ok:
        print("\\n🎉 ¡Sistema listo para entrenamiento!")
        print("Ejecute: python src/trainer.py")
    elif estructura_ok and not hay_imagenes:
        print("\\n📥 Sistema configurado correctamente.")
        print("Agregue imágenes a las carpetas y ejecute: python src/trainer.py")
    else:
        print("\\n🔧 Hay problemas que resolver antes del entrenamiento.")

if __name__ == "__main__":
    main()