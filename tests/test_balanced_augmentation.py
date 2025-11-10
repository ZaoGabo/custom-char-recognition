import numpy as np
import pytest

from src.data.balanced_augmentation import BalancedAugmentation


@pytest.fixture(scope="module")
def base_samples() -> list[np.ndarray]:
    rng = np.random.default_rng(123)
    sample = rng.random((28, 28), dtype=np.float32)
    return [sample for _ in range(3)]


def test_augment_class_reaches_target(base_samples: list[np.ndarray]) -> None:
    augmentor = BalancedAugmentation(strategy="light", seed=7)
    target = 8

    generated = augmentor.augment_class(base_samples, target)

    assert len(generated) == max(0, target - len(base_samples))
    assert all(0.0 <= float(img.min()) for img in generated)
    assert all(float(img.max()) <= 1.0 for img in generated)


def test_balance_dataset_returns_expected_pairs(base_samples: list[np.ndarray]) -> None:
    augmentor = BalancedAugmentation(strategy="medium", seed=11)
    target = 5
    labels_with_samples = [(0, base_samples[:2]), (1, base_samples[:1])]

    augmented_pairs = augmentor.balance_dataset(labels_with_samples, target)

    expected = sum(max(0, target - len(items)) for _, items in labels_with_samples)
    assert len(augmented_pairs) == expected
    assert {label for label, _ in augmented_pairs} <= {0, 1}
    for _, image in augmented_pairs:
        assert image.shape == base_samples[0].shape
        assert np.issubdtype(image.dtype, np.floating)
        assert image.min() >= 0.0 and image.max() <= 1.0
