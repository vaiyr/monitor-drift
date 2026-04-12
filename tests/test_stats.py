"""Tests for bootstrap CI and permutation test."""

import numpy as np

from src.probe import train_linear_probe
from src.stats import bootstrap_auroc_ci, permutation_test_auroc


def test_bootstrap_ci_reasonable(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    ci_low, ci_high = bootstrap_auroc_ci(probe, X, y, n_bootstrap=100)
    assert ci_low < ci_high
    assert ci_high - ci_low < 0.3


def test_permutation_test_significant(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    p = permutation_test_auroc(probe, X, y, n_permutations=100)
    assert p < 0.05


def test_permutation_test_random_labels():
    """Random labels should not be significant."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 64))
    y = rng.integers(0, 2, 100)
    probe = train_linear_probe(X, y)
    p = permutation_test_auroc(probe, X, y, n_permutations=100)
    # With random labels, p-value should generally not be extreme
    # but randomness means it occasionally can be low
    assert p > 0.005
