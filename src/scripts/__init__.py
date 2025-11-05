"""Scripts auxiliares para el sistema de reconocimiento de caracteres."""

try:
    from .generar_imagenes_sinteticas import generar_imagenes_sinteticas  # type: ignore
except ImportError:  # pragma: no cover
    generar_imagenes_sinteticas = None

from .generar_csv import generar_csv_desde_raw

__all__ = [
    "generar_imagenes_sinteticas",
    "generar_csv_desde_raw",
]