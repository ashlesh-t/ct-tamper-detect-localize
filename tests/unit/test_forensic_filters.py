import numpy as np
import pytest

from pipeline.util.forensic_filters import (
    generate_ela_map,
    generate_noise_residual,
    generate_fft_energy_map,
    apply_lung_window,
)


@pytest.fixture
def image_norm():
    rng = np.random.default_rng(42)
    return rng.random((64, 64), dtype=np.float64).astype(np.float32)


def test_generate_ela_map_shape_and_range(image_norm):
    ela = generate_ela_map(image_norm)
    assert ela.shape == image_norm.shape
    assert ela.dtype == np.float32 or ela.dtype == np.float64
    assert ela.min() >= 0.0
    assert ela.max() <= 1.0


def test_generate_noise_residual_shape_and_range(image_norm):
    residual = generate_noise_residual(image_norm)
    assert residual.shape == image_norm.shape
    assert residual.min() >= 0.0
    assert residual.max() <= 1.0


def test_generate_fft_energy_map_shape_and_range(image_norm):
    energy = generate_fft_energy_map(image_norm)
    assert energy.shape == image_norm.shape
    assert energy.min() >= 0.0
    assert energy.max() <= 1.0


def test_apply_lung_window_clips_and_normalizes():
    image = np.array([[-2000, -600, 150], [1500, 3000, -1350]], dtype=np.float32)
    windowed = apply_lung_window(image, level=-600, width=1500)
    assert windowed.shape == image.shape
    assert windowed.min() >= 0.0
    assert windowed.max() <= 1.0
    # Center of the window should map close to 0.5
    center_only = apply_lung_window(np.array([[-600.0]]), level=-600, width=1500)
    assert center_only[0, 0] == pytest.approx(0.5, abs=1e-6)
