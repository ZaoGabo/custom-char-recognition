"""Entrenador robusto que reutiliza el pipeline CNN v2."""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.pipeline import entrenar_modelo


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
    print("  Reintentos automaticos habilitados")
    print("=" * 80)
    print()


def entrenar_con_reintentos(
    max_reintentos: int,
    force: bool,
    verbose: bool,
    device: Optional[str],
    epochs: Optional[int],
    data_dir: Optional[str],
    model_dir_name: str,
) -> bool:
    for intento in range(1, max_reintentos + 1):
        try:
            print(f"\n{'=' * 80}")
            print(f"INTENTO {intento}/{max_reintentos}")
            print(f"{'=' * 80}\n")

            output_dir = entrenar_modelo(
                force=force,
                verbose=verbose,
                device=device,
                max_epochs=epochs,
                model_dir_name=model_dir_name,
                data_dir=data_dir,
            )

            print(f"\n{'=' * 80}")
            print("  ENTRENAMIENTO COMPLETADO")
            print(f"  Modelos guardados en: {output_dir}")
            print(f"{'=' * 80}\n")
            return True

        except KeyboardInterrupt:
            print(f"\n{'!' * 80}")
            print(f"  Interrupcion detectada en intento {intento}")
            print(f"{'!' * 80}")
            if intento < max_reintentos:
                print("  Reintentando en 5 segundos...")
                time.sleep(5)
                continue
            print(f"  Maximo de reintentos alcanzado ({max_reintentos})")
            return False

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

    return False


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning robusto utilizando el pipeline CNN v2")
    parser.add_argument("--max-retries", type=int, default=5, help="numero de reintentos ante fallos imprevistos")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda o mps")
    parser.add_argument("--epochs", type=int, default=None, help="sobrescribe epochs de config.yml")
    parser.add_argument("--data-dir", type=str, default=None, help="ruta alternativa con datos procesados para fine-tuning")
    parser.add_argument("--model-dir-name", type=str, default="cnn_modelo_v2_finetuned", help="carpeta dentro de models/ para guardar pesos")
    parser.add_argument("--no-force", action="store_true", help="no sobreescribir modelos existentes")
    parser.add_argument("--verbose", action="store_true", help="mostrar metricas por epoca")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configurar_trampas_signal()
    _imprimir_header(args)

    success = entrenar_con_reintentos(
        max_reintentos=args.max_retries,
        force=not args.no_force,
        verbose=args.verbose,
        device=args.device,
        epochs=args.epochs,
        data_dir=args.data_dir,
        model_dir_name=args.model_dir_name,
    )

    print()
    if success:
        print("=" * 80)
        print("  PROCESO COMPLETADO")
        print(f"  Modelo guardado en models/{args.model_dir_name}/")
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
