"""
Script de prueba para verificar la carga de datos desde las carpetas organizadas.
"""

import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.config import PATHS
from src.data_loader import DataLoader


def verificar_estructura_carpetas():
    print("Verificando estructura de carpetas...")

    data_path = PATHS['datos_crudos']
    carpetas_encontradas = []
    carpetas_faltantes = []

    for letra in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        carpeta = f"{letra}_upper"
        ruta_carpeta = os.path.join(data_path, carpeta)
        if os.path.exists(ruta_carpeta):
            carpetas_encontradas.append(carpeta)
        else:
            carpetas_faltantes.append(carpeta)

    for letra in 'abcdefghijklmnopqrstuvwxyz':
        carpeta = f"{letra}_lower"
        ruta_carpeta = os.path.join(data_path, carpeta)
        if os.path.exists(ruta_carpeta):
            carpetas_encontradas.append(carpeta)
        else:
            carpetas_faltantes.append(carpeta)

    print(f"Carpetas encontradas: {len(carpetas_encontradas)}/52")
    if carpetas_faltantes:
        print(f"Carpetas faltantes: {carpetas_faltantes}")
    else:
        print("Todas las carpetas estan presentes")

    return len(carpetas_faltantes) == 0


def probar_mapeo_etiquetas():
    print("\nProbando mapeo de etiquetas...")

    data_loader = DataLoader(PATHS['datos_crudos'])

    casos_prueba = [
        'A_upper', 'Z_upper', 'a_lower', 'z_lower',
        'M_upper', 'n_lower'
    ]

    for carpeta in casos_prueba:
        etiqueta = data_loader._mapear_carpeta_a_etiqueta(carpeta)
        indice = data_loader.mapa_etiquetas.get_index(etiqueta)

        print(f"  {carpeta} -> '{etiqueta}' -> indice {indice}")
        if indice == -1:
            print(f"    Error: etiqueta '{etiqueta}' no encontrada")
        else:
            print("    Mapeado correctamente")


def contar_imagenes_por_carpeta():
    print("\nContando imagenes por carpeta...")

    data_path = PATHS['datos_crudos']
    total_imagenes = 0
    carpetas_con_imagenes = 0

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
            print(f"  {carpeta}: {num_imagenes} imagenes")

    print("\nResumen:")
    print(f"  Total de carpetas: {len(carpetas)}")
    print(f"  Carpetas con imagenes: {carpetas_con_imagenes}")
    print(f"  Total de imagenes: {total_imagenes}")

    return total_imagenes > 0


def probar_carga_datos():
    print("\nProbando carga de datos...")

    try:
        data_loader = DataLoader(PATHS['datos_crudos'])
        data_loader.cargar_desde_directorio()

        if len(data_loader.imagenes) > 0:
            print("Carga exitosa:")
            print(f"  Imagenes cargadas: {len(data_loader.imagenes)}")
            print(f"  Forma de imagen: {data_loader.imagenes.shape}")
            print(f"  Etiquetas unicas: {len(np.unique(data_loader.etiquetas))}")
        else:
            print("No se cargaron imagenes. Verifique que haya imagenes en las carpetas.")

    except Exception as e:
        print(f"Error durante la carga: {str(e)}")
        return False

    return True


def main():
    print("=" * 60)
    print("VERIFICACION DEL SISTEMA DE CARGA DE DATOS")
    print("=" * 60)

    estructura_ok = verificar_estructura_carpetas()
    probar_mapeo_etiquetas()
    hay_imagenes = contar_imagenes_por_carpeta()

    if hay_imagenes:
        carga_ok = probar_carga_datos()
    else:
        print("\nNo hay imagenes para cargar. Agregue imagenes a las carpetas y vuelva a intentar.")
        carga_ok = False

    print("\n" + "=" * 60)
    print("RESUMEN DE VERIFICACION")
    print("=" * 60)
    print(f"Estructura de carpetas: {'OK' if estructura_ok else 'ERROR'}")
    print("Mapeo de etiquetas: OK")
    print(f"Imagenes disponibles: {'SI' if hay_imagenes else 'NO'}")
    print(f"Carga de datos: {'OK' if carga_ok else 'PENDIENTE' if not hay_imagenes else 'ERROR'}")

    if estructura_ok and hay_imagenes and carga_ok:
        print("\nSistema listo para entrenamiento!")
        print("Ejecute: python -m src.trainer --force")
    elif estructura_ok and not hay_imagenes:
        print("\nSistema configurado correctamente.")
        print("Agregue imagenes a las carpetas y ejecute: python -m src.trainer --force")
    else:
        print("\nHay problemas que resolver antes del entrenamiento.")


if __name__ == '__main__':
    main()
