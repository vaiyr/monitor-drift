"""Tests for activation extraction utilities."""

import numpy as np

from src.extract import load_activations, save_activations


def test_save_load_activations(tmp_path):
    rng = np.random.default_rng(42)
    acts = {
        5: rng.standard_normal((10, 64)).astype(np.float32),
        9: rng.standard_normal((10, 64)).astype(np.float32),
    }
    path = str(tmp_path / "test.npz")
    save_activations(acts, path)
    loaded = load_activations(path)
    assert set(loaded.keys()) == {5, 9}
    np.testing.assert_allclose(loaded[5], acts[5])
    np.testing.assert_allclose(loaded[9], acts[9])
