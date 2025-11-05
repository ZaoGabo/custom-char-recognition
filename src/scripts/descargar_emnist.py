"""Descarga el dataset EMNIST y lo organiza en carpetas por clase."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import EMNIST

LOGGER = logging.getLogger(__name__)


_SPECIAL_CHAR_DIRS: Dict[str, str] = {
    '!': 'exclamation',
    '@': 'at',
    '#': 'hash',
    '$': 'dollar',
    '%': 'percent',
    '&': 'ampersand',
    '*': 'asterisk',
    '(': 'lparen',
    ')': 'rparen',
    '-': 'minus',
    '_': 'underscore',
    '+': 'plus',
    '=': 'equals',
    '[': 'lbracket',
    ']': 'rbracket',
    '{': 'lbrace',
    '}': 'rbrace',
    ';': 'semicolon',
    ':': 'colon',
    "'": 'quote',
    '"': 'dquote',
    ',': 'comma',
    '.': 'period',
    '<': 'less',
    '>': 'greater',
    '/': 'slash',
    '?': 'question',
    '|': 'pipe',
    '~': 'tilde',
    '`': 'backtick',
}


def _char_to_dir_name(char: str) -> Optional[str]:
    if char.isdigit():
        return f"{char}_digit"
    if char.isalpha():
        if char.isupper():
            return f"{char}_upper"
        return f"{char}_lower"
    return _SPECIAL_CHAR_DIRS.get(char)


def _fix_orientation(image_array: np.ndarray) -> np.ndarray:
    rotated = np.rot90(image_array, k=1)
    return np.fliplr(rotated)


def _save_sample(output_dir: Path, dir_name: str, char: str, index: int, image_array: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_array, mode='L')
    filename = output_dir / f"emnist_{char}_{index:05d}.png"
    image.save(filename)


EMNIST_URL_OVERRIDE = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def _ensure_emnist_url_override() -> None:
    if getattr(EMNIST, "url", None) != EMNIST_URL_OVERRIDE:
        LOGGER.debug("Actualizando URL de descarga de EMNIST a %s", EMNIST_URL_OVERRIDE)
        EMNIST.url = EMNIST_URL_OVERRIDE


def descargar_emnist(
    split: str,
    output_root: Path,
    download_root: Path,
    max_per_class: Optional[int],
    dry_run: bool,
    skip_existing: bool,
) -> Dict[str, int]:
    _ensure_emnist_url_override()
    dataset = EMNIST(root=str(download_root), split=split, download=True)
    counts: Dict[str, int] = defaultdict(int)
    existing_cache: Dict[str, int] = {}

    samples = zip(dataset.data, dataset.targets)
    total_samples = len(dataset.data)

    for image_tensor, label_tensor in tqdm(samples, total=total_samples, desc=f"Procesando {split}"):
        label_index = int(label_tensor)
        char = dataset.classes[label_index]
        dir_name = _char_to_dir_name(char)
        if dir_name is None:
            continue

        dir_path = output_root / dir_name
        if dir_name not in existing_cache:
            existing = 0
            if dir_path.exists():
                existing = len(list(dir_path.glob('*.png')))
            existing_cache[dir_name] = existing
            counts[dir_name] = existing
        else:
            existing = existing_cache[dir_name]

        if skip_existing and existing > 0:
            continue

        if max_per_class is not None and counts[dir_name] >= max_per_class:
            continue

        if dry_run:
            counts[dir_name] += 1
            existing_cache[dir_name] = counts[dir_name]
            continue

        image_array = image_tensor.numpy()
        image_array = _fix_orientation(image_array)
        index = counts[dir_name]
        _save_sample(dir_path, dir_name, char, index, image_array)
        counts[dir_name] += 1
        existing_cache[dir_name] = counts[dir_name]

    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga y exporta EMNIST a carpetas por clase")
    parser.add_argument('--split', type=str, default='byclass', choices=['byclass', 'balanced', 'letters', 'digits', 'bymerge'], help='Split de EMNIST a descargar')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Directorio donde se guardaran las imagenes exportadas')
    parser.add_argument('--download-dir', type=str, default='data/emnist_download', help='Carpeta donde se almacenara el archivo original de EMNIST')
    parser.add_argument('--max-per-class', type=int, default=None, help='Limite de muestras por clase (para crear subconjuntos)')
    parser.add_argument('--dry-run', action='store_true', help='Solo muestra cuantas imagenes se procesarian, sin escribir en disco')
    parser.add_argument('--skip-existing', action='store_true', help='No volver a generar clases que ya tienen imagenes')
    parser.add_argument('--log-level', type=str, default='INFO', help='Nivel de logging (DEBUG, INFO, WARNING, ERROR)')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_root = Path(args.output_dir)
    download_root = Path(args.download_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Descargando split '%s' de EMNIST", args.split)
    counts = descargar_emnist(
        split=args.split,
        output_root=output_root,
        download_root=download_root,
        max_per_class=args.max_per_class,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )

    total_exported = sum(counts.values())
    clases = len([c for c in counts.values() if c > 0])
    LOGGER.info("Proceso completado: %d imagenes en %d clases", total_exported, clases)


if __name__ == '__main__':
    main()
