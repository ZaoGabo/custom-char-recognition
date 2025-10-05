"""Verificacion rapida de la estructura de datos del proyecto."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import PATHS
from src.data_loader import DataLoader


def verificar_estructura_carpetas() -> bool:
    """Comprobar que existan las 52 carpetas esperadas en ``data/raw``."""
    print('Verificando estructura de carpetas...')
    data_path = Path(PATHS['datos_crudos'])
    carpetas_encontradas = []
    carpetas_faltantes = []

    for letra in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        carpeta = data_path / f'{letra}_upper'
        if carpeta.exists():
            carpetas_encontradas.append(carpeta.name)
        else:
            carpetas_faltantes.append(carpeta.name)

    for letra in 'abcdefghijklmnopqrstuvwxyz':
        carpeta = data_path / f'{letra}_lower'
        if carpeta.exists():
            carpetas_encontradas.append(carpeta.name)
        else:
            carpetas_faltantes.append(carpeta.name)

    print(f'Carpetas encontradas: {len(carpetas_encontradas)}/52')
    if carpetas_faltantes:
        print(f'Carpetas faltantes: {carpetas_faltantes}')
    else:
        print('Todas las carpetas estan presentes')
    return not carpetas_faltantes


def probar_mapeo_etiquetas() -> None:
    """Mostrar algunos ejemplos de mapeo carpeta -> indice."""
    print('\nProbando mapeo de etiquetas...')
    data_loader = DataLoader(PATHS['datos_crudos'])
    for carpeta in ['A_upper', 'Z_upper', 'a_lower', 'z_lower', 'M_upper', 'n_lower']:
        etiqueta = data_loader._mapear_carpeta_a_etiqueta(carpeta)
        indice = data_loader.mapa_etiquetas.get_index(etiqueta)
        print(f"  {carpeta} -> '{etiqueta}' -> indice {indice}")


def contar_imagenes_por_carpeta() -> Tuple[int, int]:
    """Contar la cantidad de imagenes existentes por carpeta."""
    print('\nContando imagenes por carpeta...')
    data_path = Path(PATHS['datos_crudos'])
    total_imagenes = 0
    carpetas_con_imagenes = 0

    for carpeta in sorted(p for p in data_path.iterdir() if p.is_dir()):
        if carpeta.name.startswith('.'):
            continue
        imagenes = [f for f in carpeta.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
        if imagenes:
            carpetas_con_imagenes += 1
            total_imagenes += len(imagenes)
            print(f'  {carpeta.name}: {len(imagenes)} imagenes')

    print('\nResumen:')
    print(f'  Total de carpetas: {sum(1 for _ in data_path.iterdir() if _.is_dir())}')
    print(f'  Carpetas con imagenes: {carpetas_con_imagenes}')
    print(f'  Total de imagenes: {total_imagenes}')
    return total_imagenes, carpetas_con_imagenes


def probar_carga_datos() -> bool:
    """Cargar los datos usando ``DataLoader`` y mostrar detalles basicos."""
    print('\nProbando carga de datos...')
    loader = DataLoader(PATHS['datos_crudos'])
    loader.cargar_desde_directorio()
    if len(loader.imagenes) > 0:
        print('Carga exitosa:')
        print(f'  Imagenes cargadas: {len(loader.imagenes)}')
        print(f'  Forma de imagen: {loader.imagenes.shape}')
        print(f'  Etiquetas unicas: {len(np.unique(loader.etiquetas))}')
        return True
    print('No se cargaron imagenes. Verifique que haya datos disponibles.')
    return False


def main() -> None:
    """Ejecutar toda la verificacion desde la linea de comandos."""
    print('=' * 60)
    print('VERIFICACION DEL SISTEMA DE CARGA DE DATOS')
    print('=' * 60)

    estructura_ok = verificar_estructura_carpetas()
    probar_mapeo_etiquetas()
    total_imagenes, carpetas_con_imagenes = contar_imagenes_por_carpeta()
    hay_imagenes = total_imagenes > 0

    carga_ok = probar_carga_datos() if hay_imagenes else False

    print('\n' + '=' * 60)
    print('RESUMEN DE VERIFICACION')
    print('=' * 60)
    estado_estructura = 'OK' if estructura_ok else 'ERROR'
    print(f'Estructura de carpetas: {estado_estructura}')
    print('Mapeo de etiquetas: OK')
    disponibilidad = 'SI' if hay_imagenes else 'NO'
    print(f'Imagenes disponibles: {disponibilidad}')
    estado_carga = 'OK' if carga_ok else ('PENDIENTE' if not hay_imagenes else 'ERROR')
    print(f'Carga de datos: {estado_carga}')

    if estructura_ok and hay_imagenes and carga_ok:
        print('\nSistema listo para entrenamiento!')
        print('Ejecute: python -m src.trainer --force')
    elif estructura_ok and not hay_imagenes:
        print('\nSistema configurado correctamente.')
        print('Agregue imagenes a las carpetas y ejecute: python -m src.trainer --force')
    else:
        print('\nHay problemas que resolver antes del entrenamiento.')


if __name__ == '__main__':
    main()