import numpy as np
import pytest

from pipeline.preProces.preProcess import global_preprocess, CTMultiChannelPreprocessor, ensure_3ch


@pytest.fixture
def raw_slice():
    rng = np.random.default_rng(0)
    return rng.uniform(-1000, 2000, size=(128, 128)).astype(np.float32)


def test_global_preprocess_shape_dtype_range(raw_slice):
    out = global_preprocess(raw_slice, target_img_size=64)
    assert out.shape == (64, 64)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_global_preprocess_disables_steps(raw_slice):
    out = global_preprocess(raw_slice, target_img_size=32, do_clahe=False, do_gamma=False, do_sharpen=False)
    assert out.shape == (32, 32)
    assert np.isfinite(out).all()


def test_ct_multi_channel_preprocessor_produces_three_channels(raw_slice):
    preprocessor = CTMultiChannelPreprocessor(target_size=64)
    channels = preprocessor.preprocess_single_slice(raw_slice)
    assert set(channels.keys()) == {"CT", "ROI", "FFT"}
    for name, arr in channels.items():
        assert arr.shape == (64, 64), f"{name} channel has unexpected shape"
        assert np.isfinite(arr).all()
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0 + 1e-5


def test_ensure_3ch_handles_none():
    out = ensure_3ch(None, img_size=32)
    assert out.shape == (32, 32, 3)


def test_ensure_3ch_stacks_2d():
    arr = np.ones((16, 16), dtype=np.float32)
    out = ensure_3ch(arr)
    assert out.shape == (16, 16, 3)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_ensure_3ch_truncates_extra_channels():
    arr = np.ones((16, 16, 5), dtype=np.float32)
    out = ensure_3ch(arr)
    assert out.shape == (16, 16, 3)
