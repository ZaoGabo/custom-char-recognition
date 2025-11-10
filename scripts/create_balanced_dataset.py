"""CLI para generar un dataset balanceado mediante augmentaciones ligeras."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import PATHS
from src.data.balanced_augmentation import BalancedAugmentation

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0


def _ensure_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileExistsError(f"El directorio destino ya existe: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _save_augmented_image(array: np.ndarray, path: Path) -> None:
    img = np.clip(array, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)


def _gather_class_samples(class_dir: Path) -> Tuple[List[Path], List[np.ndarray]]:
    image_paths: List[Path] = []
    tensors: List[np.ndarray] = []
    for path in sorted(class_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        image_paths.append(path)
        tensors.append(_load_grayscale(path))
    return image_paths, tensors


def _default_output(root: Path, strategy: str, target: int) -> Path:
    return root / f"balanced_{strategy}_{target}"


def create_balanced_dataset(
    source_dir: Path,
    output_dir: Path,
    target_per_class: int,
    strategy: str,
    seed: int | None,
    copy_original: bool,
) -> Dict[str, Dict[str, int]]:
    augmentor = BalancedAugmentation(strategy=strategy, seed=seed)
    summary: Dict[str, Dict[str, int]] = {}

    class_dirs = [p for p in sorted(source_dir.iterdir()) if p.is_dir()]
    iterator: Iterable[Path] = tqdm(class_dirs, desc="Clases", unit="clase") if class_dirs else []

    for class_dir in iterator:
        image_paths, tensors = _gather_class_samples(class_dir)
        generated = augmentor.augment_class(tensors, target_per_class)

        class_output = output_dir / class_dir.name
        class_output.mkdir(parents=True, exist_ok=True)

        if copy_original:
            for original_path in image_paths:
                destination = class_output / original_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original_path, destination)

        next_index = 0
        for array in generated:
            filename = f"aug_{next_index:05d}.png"
            destination = class_output / filename
            while destination.exists():
                next_index += 1
                filename = f"aug_{next_index:05d}.png"
                destination = class_output / filename
            _save_augmented_image(array, destination)
            next_index += 1

        summary[class_dir.name] = {
            "original": len(image_paths),
            "generadas": len(generated),
            "total": len(image_paths) + len(generated),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar dataset balanceado a partir de imágenes existentes.")
    default_source = Path(PATHS.get("datos_crudos", "data/raw"))

    parser.add_argument("--source-dir", type=Path, default=default_source, help="Directorio con subcarpetas por clase.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directorio destino para escribir el dataset balanceado.")
    parser.add_argument("--target-per-class", type=int, default=800, help="Número objetivo de imágenes por clase.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=sorted(BalancedAugmentation.PRESETS.keys()),
        default="medium",
        help="Preset de intensidad de augmentación.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para el generador aleatorio.")
    parser.add_argument("--copy-original", action="store_true", help="Copiar también las imágenes originales al destino.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir el directorio destino si ya existe.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        root_processed = Path(PATHS.get("datos_procesados", "data/processed"))
        output_dir = _default_output(root_processed, args.strategy, args.target_per_class)

    _ensure_dir(output_dir, overwrite=args.overwrite)

    summary = create_balanced_dataset(
        source_dir=args.source_dir,
        output_dir=output_dir,
        target_per_class=args.target_per_class,
        strategy=args.strategy,
        seed=args.seed,
        copy_original=args.copy_original,
    )

    metadata = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "target_per_class": args.target_per_class,
        "strategy": args.strategy,
        "seed": args.seed,
        "copy_original": args.copy_original,
        "resumen": summary,
    }

    metadata_path = output_dir / "summary.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Dataset balanceado generado en: {output_dir}")
    print(f"Resumen guardado en: {metadata_path}")


if __name__ == "__main__":
    main()
