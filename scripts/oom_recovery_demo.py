"""Demo para validar la recuperación ante OOM con RobustTrainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entrenar_finetune_robusto import entrenar_con_reintentos
from src.training.robust_trainer import RobustTrainer, RobustTrainerConfig, TrainingResult


def _patch_trainer_for_oom() -> None:
    """Parchea ``RobustTrainer`` para forzar un OOM en la primera época."""

    original_run_epoch = RobustTrainer._run_epoch
    state = {"oom_triggered": False}

    def _run_epoch_with_oom(
        self,
        *,
        epoch,
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler,
        criterion,
        history,
        verbose,
    ):
        if device.type != "cuda":
            return original_run_epoch(
                self,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                history=history,
                verbose=verbose,
            )

        if not state["oom_triggered"]:
            state["oom_triggered"] = True
            if verbose:
                print("[OOM demo] Forzando OOM en GPU en la primera época...")
            buffers = []
            try:
                while True:
                    buffers.append(
                        torch.empty((256, 1024, 1024), dtype=torch.float32, device=device)
                    )
            except torch.cuda.OutOfMemoryError:
                if verbose:
                    print("[OOM demo] OOM capturado, propagando...")
                raise
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if verbose:
                        print("[OOM demo] RuntimeError OOM capturado, propagando...")
                    raise torch.cuda.OutOfMemoryError(str(exc))
                raise
            finally:
                del buffers

        return original_run_epoch(
            self,
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            history=history,
            verbose=verbose,
        )

    RobustTrainer._run_epoch = _run_epoch_with_oom  # type: ignore[assignment]


def run_demo(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere una GPU CUDA para la demo de OOM.")

    _patch_trainer_for_oom()

    config = RobustTrainerConfig(
        patience=1,
        max_checkpoints=1,
        resume=False,
        batch_size=args.batch_size,
    )

    result = entrenar_con_reintentos(
        max_reintentos=2,
        force=True,
        verbose=args.verbose,
        device="cuda",
        epochs=args.epochs,
        data_dir=args.data_dir,
        model_dir_name=args.model_dir_name,
        trainer_config=config,
    )

    if isinstance(result, TrainingResult):
        print("RESULT_OK", True)
        print("Best metric:", result.best_metric_name, result.best_metric_value)
        print("Checkpoint final:", Path(result.best_model_path))
    else:
        print("RESULT_OK", False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo de recuperación ante un OOM real usando RobustTrainer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Número de épocas a ejecutar después del OOM",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Tamaño de batch para la corrida de demostración",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Ruta de datos procesados (usa config.yml por defecto)",
    )
    parser.add_argument(
        "--model-dir-name",
        type=str,
        default="cnn_modelo_v2_oom_demo",
        help="Carpeta de modelos para la demo",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar métricas por época",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_demo(_parse_args())
