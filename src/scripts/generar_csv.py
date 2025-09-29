# src/scripts/generar_csv.py

import os
import pandas as pd
from ..config import PATHS

def generar_csv_desde_raw():
    ruta_raw = PATHS['datos_crudos']
    salida_csv = os.path.join(PATHS['datos_procesados'], "dataset.csv")
    registros = []

    for clase in os.listdir(ruta_raw):
        ruta_clase = os.path.join(ruta_raw, clase)
        if not os.path.isdir(ruta_clase):
            continue

        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_relativa = os.path.join(clase, archivo)
                registros.append({'image_path': ruta_relativa, 'label': clase})

    os.makedirs(PATHS['datos_procesados'], exist_ok=True)
    df = pd.DataFrame(registros)

    if df.empty:
        print("⚠️ No se encontraron imágenes válidas en data/raw/")
    else:
        df.to_csv(salida_csv, index=False)
        print(f"✅ CSV generado con {len(df)} registros.")
