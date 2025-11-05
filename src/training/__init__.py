"""Utilidades de entrenamiento para modelos CNN."""

from .pipeline import entrenar_modelo, _one_hot, _actualizar_tasa_aprendizaje

__all__ = [
    "entrenar_modelo",
    "_one_hot",
    "_actualizar_tasa_aprendizaje",
]
