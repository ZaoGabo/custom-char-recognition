"""
Script de Entrenamiento Robusto (Fine-Tuning)

Entrena el modelo CNN v2 reutilizando el pipeline robusto.
Soporta reintentos automaticos, checkpointing y recuperacion de errores.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Asegurar que podemos importar src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.robust_trainer import (
    RecoverableTrainingError,
    RobustTrainer,
    RobustTrainerConfig,
    TrainingResult,
)
from src.utils.logger import app_logger as logger


def _configurar_trampas_signal() -> None:
    def _handler(sig, frame):
        del sig, frame
        logger.warning("Señaly recibida. Esperando finalizar epoca actual...")

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)  # Windows only


def entrenar_con_reintentos(
    *,
    max_reintentos: int,
    force: bool,
    verbose: bool,
    device: Optional[str],
    epochs: Optional[int],
    data_dir: Optional[str],
    model_dir_name: str,
    trainer_config: RobustTrainerConfig,
) -> Optional[TrainingResult]:
    
    for intento in range(1, max_reintentos + 1):
        try:
            logger.info("=" * 60)
            logger.info(f"INTENTO DE ENTRENAMIENTO {intento}/{max_reintentos}")
            logger.info("=" * 60)

            trainer = RobustTrainer(
                model_dir_name=model_dir_name,
                data_dir=data_dir,
                device=device,
                config=trainer_config,
            )

            result = trainer.train(
                force=force,
                resume=trainer_config.resume,
                max_epochs=epochs,
                verbose=verbose,
            )

            logger.info("=" * 60)
            logger.info("[OK] ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            if result.best_metric_name and result.best_metric_value is not None:
                logger.info(f"Mejor {result.best_metric_name}: {result.best_metric_value:.4f}")
            logger.info(f"Checkpoint final: {result.best_model_path}")
            logger.info("=" * 60)
            return result

        except KeyboardInterrupt:
            logger.warning("Interrupcion por usuario.")
            if intento < max_reintentos:
                logger.info("Reintentando en 5 segundos...")
                time.sleep(5)
                continue
            return None

        except RecoverableTrainingError as exc:
            logger.error(f"ERROR RECUPERABLE: {exc}")
            if intento < max_reintentos:
                logger.info("Reintentando en 10 segundos...")
                time.sleep(10)
                continue
            return None

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"ERROR INESPERADO: {exc}")
            if intento < max_reintentos:
                logger.info("Reintentando en 10 segundos...")
                time.sleep(10)
                continue
            raise

    return None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning robusto para CNN v2")
    parser.add_argument("--max-retries", type=int, default=5, help="Reintentos maximos ante fallos")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda o mps")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe epochs de config")
    parser.add_argument("--data-dir", type=str, default=None, help="Ruta datos procesados")
    parser.add_argument("--model-dir-name", type=str, default="cnn_modelo_v2_finetuned", help="Carpeta destino")
    parser.add_argument("--no-force", action="store_true", help="No sobreescribir modelos")
    parser.add_argument("--verbose", action="store_true", help="Mostrar logs detallados")
    parser.add_argument("--patience", type=int, default=8, help="Paciencia early stopping")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--max-checkpoints", type=int, default=5, help="Checkpoints a guardar")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configurar_trampas_signal()

    trainer_config = RobustTrainerConfig(
        patience=max(args.patience, 0),
        min_delta=1e-4,
        gradient_clip_norm=None if args.grad_clip <= 0 else args.grad_clip,
        max_checkpoints=max(args.max_checkpoints, 0),
        resume=True,
        scheduler_type=None,
        scheduler_min_lr=None,
        num_workers=0,
    )

    result = entrenar_con_reintentos(
        max_reintentos=args.max_retries,
        force=not args.no_force,
        verbose=args.verbose,
        device=args.device,
        epochs=args.epochs,
        data_dir=args.data_dir,
        model_dir_name=args.model_dir_name,
        trainer_config=trainer_config,
    )

    return 0 if result else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("ERROR FATAL en script de entrenamiento")
        sys.exit(1)
