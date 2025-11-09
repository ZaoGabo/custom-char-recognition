"""Entrenador robusto que reutiliza el pipeline CNN v2."""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.robust_trainer import (
    RecoverableTrainingError,
    RobustTrainer,
    RobustTrainerConfig,
    TrainingResult,
)


def _configurar_trampas_signal() -> None:
    def _handler(sig, frame):
        del sig, frame
        print("\nSe recibio una senal; se esperara a que finalice la epoca actual...")

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]


def _imprimir_header(args: argparse.Namespace) -> None:
    print("=" * 80)
    print(" ENTRENADOR ROBUSTO - CNN v2 (pipeline)")
    print("=" * 80)
    print(f"  Modelo destino: {args.model_dir_name}")
    print(f"  Datos: {args.data_dir or 'paths.datos_procesados'}")
    print(f"  Epochs: {args.epochs or 'config.yml'}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Early stopping patience: {args.patience}")
    grad_clip = "off" if args.grad_clip <= 0 else f"{args.grad_clip}"  # evita ruido en logs
    print(f"  Grad clip: {grad_clip}")
    scheduler = args.scheduler or 'config.yml'
    print(f"  Scheduler: {scheduler}")
    print(f"  Resume: {'yes' if not args.no_resume else 'no'}")
    print(f"  Checkpoints maximos: {args.max_checkpoints}")
    print("  Reintentos automaticos habilitados")
    print("=" * 80)
    print()


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
            print(f"\n{'=' * 80}")
            print(f"INTENTO {intento}/{max_reintentos}")
            print(f"{'=' * 80}\n")

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

            print(f"\n{'=' * 80}")
            print("  ENTRENAMIENTO COMPLETADO")
            print(f"  Modelos guardados en: {result.output_dir}")
            if result.best_metric_name and result.best_metric_value is not None:
                print(
                    f"  Mejor {result.best_metric_name}: {result.best_metric_value:.4f}"
                )
            print(f"  Epocas ejecutadas: {result.epochs_trained}")
            print(f"  Checkpoint final: {result.best_model_path}")
            print(f"{'=' * 80}\n")
            return result

        except KeyboardInterrupt:
            print(f"\n{'!' * 80}")
            print(f"  Interrupcion detectada en intento {intento}")
            print(f"{'!' * 80}")
            if intento < max_reintentos:
                print("  Reintentando en 5 segundos...")
                time.sleep(5)
                continue
            print(f"  Maximo de reintentos alcanzado ({max_reintentos})")
            return None

        except RecoverableTrainingError as exc:
            print(f"\n{'!' * 80}")
            print(f"  ERROR RECUPERABLE en intento {intento}:")
            print(f"    {exc}")
            print(f"{'!' * 80}")
            if intento < max_reintentos:
                print("  Reintentando en 10 segundos...")
                time.sleep(10)
                continue
            print(f"  Maximo de reintentos alcanzado ({max_reintentos})")
            return None

        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n{'!' * 80}")
            print(f"  ERROR INESPERADO en intento {intento}:")
            print(f"    {type(exc).__name__}: {exc}")
            print(f"{'!' * 80}")
            if intento < max_reintentos:
                print("  Reintentando en 10 segundos...")
                time.sleep(10)
                continue
            print(f"  Maximo de reintentos alcanzado ({max_reintentos})")
            raise

    return None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning robusto utilizando el pipeline CNN v2")
    parser.add_argument("--max-retries", type=int, default=5, help="numero de reintentos ante fallos imprevistos")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda o mps")
    parser.add_argument("--epochs", type=int, default=None, help="sobrescribe epochs de config.yml")
    parser.add_argument("--data-dir", type=str, default=None, help="ruta alternativa con datos procesados para fine-tuning")
    parser.add_argument("--model-dir-name", type=str, default="cnn_modelo_v2_finetuned", help="carpeta dentro de models/ para guardar pesos")
    parser.add_argument("--no-force", action="store_true", help="no sobreescribir modelos existentes")
    parser.add_argument("--verbose", action="store_true", help="mostrar metricas por epoca")
    parser.add_argument("--patience", type=int, default=8, help="epocas sin mejora antes de detener")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="norma maxima para clipping de gradientes (0 desactiva)")
    parser.add_argument("--max-checkpoints", type=int, default=5, help="maximo de checkpoints a conservar")
    parser.add_argument("--no-resume", action="store_true", help="no reanudar desde checkpoints previos")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "step_decay", "none"],
        default=None,
        help="scheduler de tasa de aprendizaje (por defecto usa config.yml)",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4, help="mejora minima para resetear paciencia")
    parser.add_argument("--num-workers", type=int, default=0, help="workers para DataLoader")
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=None,
        help="eta_min para CosineAnnealingLR (si aplica)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configurar_trampas_signal()
    _imprimir_header(args)

    scheduler_type = args.scheduler
    if scheduler_type == "none":
        scheduler_type = None

    trainer_config = RobustTrainerConfig(
        patience=max(args.patience, 0),
        min_delta=max(args.min_delta, 0.0),
        gradient_clip_norm=None if args.grad_clip <= 0 else args.grad_clip,
        max_checkpoints=max(args.max_checkpoints, 0),
        resume=not args.no_resume,
        scheduler_type=scheduler_type,
        scheduler_min_lr=args.scheduler_min_lr,
        num_workers=max(args.num_workers, 0),
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

    print()
    if result:
        print("=" * 80)
        print("  PROCESO COMPLETADO")
        print(f"  Modelo guardado en models/{args.model_dir_name}/")
        if result.best_metric_name and result.best_metric_value is not None:
            print(f"  Mejor {result.best_metric_name}: {result.best_metric_value:.4f}")
        print(f"  Historia: {result.history_path}")
        print("=" * 80)
        return 0

    print("=" * 80)
    print("  PROCESO INCOMPLETO")
    print("  El entrenamiento no pudo completarse")
    print("=" * 80)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n{'=' * 80}")
        print("  ERROR FATAL")
        print(f"    {type(exc).__name__}: {exc}")
        print(f"{'=' * 80}\n")
        sys.exit(1)
