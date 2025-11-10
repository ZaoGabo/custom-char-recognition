"""Balanced augmentation utilities for character datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for augmentation intensity."""

    noise_std: float = 0.12
    shift_px: int = 2
    scale_variation: float = 0.08
    rotation_deg: float = 12.0


class CanvasStyleAugmentation:
    """Apply lightweight augmentations emulating canvas drawing artifacts."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = image.astype(np.float32, copy=True)
        img = self._apply_shift(img, rng)
        img = self._apply_scale(img, rng)
        img = self._apply_rotation(img, rng)
        img = self._apply_noise(img, rng)
        img = np.clip(img, 0.0, 1.0)
        return img

    def _apply_shift(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        max_shift = max(self.config.shift_px, 0)
        if max_shift == 0:
            return image
        shift_y = int(rng.integers(-max_shift, max_shift + 1))
        shift_x = int(rng.integers(-max_shift, max_shift + 1))
        shifted = np.roll(image, shift_y, axis=0)
        shifted = np.roll(shifted, shift_x, axis=1)
        if shift_y > 0:
            shifted[:shift_y, :] = 0.0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0.0
        if shift_x > 0:
            shifted[:, :shift_x] = 0.0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0.0
        return shifted

    def _apply_scale(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        variation = max(self.config.scale_variation, 0.0)
        if variation <= 0.0:
            return image
        factor = float(1.0 + rng.uniform(-variation, variation))
        if np.isclose(factor, 1.0):
            return image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        height, width = image.shape[:2]
        new_h = max(1, int(round(height * factor)))
        new_w = max(1, int(round(width * factor)))
        resized = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if new_h > height or new_w > width:
            top = max((new_h - height) // 2, 0)
            left = max((new_w - width) // 2, 0)
            bottom = min(top + height, new_h)
            right = min(left + width, new_w)
            resized = resized.crop((left, top, right, bottom))
        canvas = Image.new("L", (width, height), color=0)
        offset_x = max((width - resized.size[0]) // 2, 0)
        offset_y = max((height - resized.size[1]) // 2, 0)
        canvas.paste(resized, (offset_x, offset_y))
        return np.asarray(canvas, dtype=np.float32) / 255.0

    def _apply_rotation(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        max_deg = self.config.rotation_deg
        if max_deg <= 0:
            return image
        angle = rng.uniform(-max_deg, max_deg)
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        rotated = pil_image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        return np.asarray(rotated, dtype=np.float32) / 255.0

    def _apply_noise(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = max(self.config.noise_std, 0.0)
        if std <= 0:
            return image
        noise = rng.normal(0.0, std, size=image.shape).astype(np.float32)
        return image + noise


class BalancedAugmentation:
    """Generate additional samples to equalize per-class counts."""

    PRESETS = {
        "light": AugmentationConfig(noise_std=0.06, shift_px=1, scale_variation=0.04, rotation_deg=6.0),
        "medium": AugmentationConfig(),
        "aggressive": AugmentationConfig(noise_std=0.18, shift_px=3, scale_variation=0.12, rotation_deg=18.0),
    }

    def __init__(
        self,
        *,
        strategy: str = "medium",
        seed: int | None = 42,
    ) -> None:
        if strategy not in self.PRESETS:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        self.config = self.PRESETS[strategy]
        self.rng = np.random.default_rng(seed)
        self.canvas_aug = CanvasStyleAugmentation(self.config)

    def augment_class(
        self,
        samples: Sequence[np.ndarray],
        target_count: int,
    ) -> List[np.ndarray]:
        if not samples:
            return []
        base = list(samples)
        generated: List[np.ndarray] = []
        while len(base) + len(generated) < target_count:
            avatar = base[self.rng.integers(0, len(base))]
            augmented = self.canvas_aug.apply(avatar, self.rng)
            generated.append(augmented.astype(np.float32))
        return generated

    def balance_dataset(
        self,
        images_by_label: Iterable[Tuple[int, Sequence[np.ndarray]]],
        target_per_class: int,
    ) -> List[Tuple[int, np.ndarray]]:
        balanced: List[Tuple[int, np.ndarray]] = []
        for label, images in images_by_label:
            augmented = self.augment_class(images, target_per_class)
            balanced.extend((label, img) for img in augmented)
        return balanced