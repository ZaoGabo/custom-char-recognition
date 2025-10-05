"""Generacion de imagenes sinteticas para entrenamiento."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from ..config import CUSTOM_LABELS, PATHS


def _obtener_fuente() -> ImageFont.ImageFont:
    """Obtener una fuente TrueType si esta disponible, o la fuente por defecto."""
    try:
        return ImageFont.truetype('arial.ttf', 20)
    except OSError:  # pragma: no cover - depende del SO
        return ImageFont.load_default()


def _nombre_carpeta(letra: str) -> str:
    return f"{letra}_upper" if letra.isupper() else f"{letra}_lower"


def generar_imagenes_sinteticas(cantidad_por_clase: int = 10) -> None:
    """Crear ``cantidad_por_clase`` imagenes por letra en ``data/raw``."""
    ruta_raw = Path(PATHS['datos_crudos'])
    ruta_raw.mkdir(parents=True, exist_ok=True)

    fuente = _obtener_fuente()

    for letra in CUSTOM_LABELS:
        carpeta = ruta_raw / _nombre_carpeta(letra)
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"Generando {cantidad_por_clase} imagenes para '{letra}' en '{carpeta.name}'...")

        for indice in range(cantidad_por_clase):
            imagen = Image.new('L', (28, 28), color=0)
            draw = ImageDraw.Draw(imagen)
            posicion = (random.randint(6, 12), random.randint(3, 8))
            draw.text(posicion, letra, font=fuente, fill=255)
            imagen.save(carpeta / f"{letra}_{indice:03d}.png")

    total = cantidad_por_clase * len(CUSTOM_LABELS)
    print(f"Generadas {total} imagenes sinteticas ({len(CUSTOM_LABELS)} clases).")
    print(f"Ubicacion: {ruta_raw}")


if __name__ == '__main__':
    generar_imagenes_sinteticas(cantidad_por_clase=15)