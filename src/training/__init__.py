"""Utilidades de entrenamiento para modelos CNN."""

from .pipeline import entrenar_modelo, _one_hot, _actualizar_tasa_aprendizaje
from .robust_trainer import (
    RecoverableTrainingError,
    RobustTrainer,
    RobustTrainerConfig,
    TrainingResult,
)

__all__ = [
    "entrenar_modelo",
    "_one_hot",
    "_actualizar_tasa_aprendizaje",
    "RobustTrainer",
    "RobustTrainerConfig",
    "TrainingResult",
    "RecoverableTrainingError",
]
