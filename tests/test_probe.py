"""Tests for probe training and evaluation."""

import numpy as np

from src.probe import (
    cosine_similarity,
    estimate_confidence_direction,
    evaluate_probe,
    get_probe_direction,
    get_probe_probs,
    regress_out_directions,
    select_best_layer,
    train_linear_probe,
)
from src.stats import bootstrap_auroc_ci, permutation_test_auroc


def test_train_linear_probe(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    auroc = evaluate_probe(probe, X, y)
    assert auroc > 0.9, f"Expected high AUROC on training data, got {auroc}"


def test_train_probe_with_groups(sample_activations_with_groups):
    X, y, groups = sample_activations_with_groups
    probe = train_linear_probe(X, y, groups=groups)
    auroc = evaluate_probe(probe, X, y)
    assert auroc > 0.8


def test_get_probe_probs(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    probs = get_probe_probs(probe, X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_get_probe_direction(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    direction = get_probe_direction(probe)
    assert direction.shape == (X.shape[1],)
    assert np.linalg.norm(direction) > 0


def test_bootstrap_auroc_ci(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    ci_low, ci_high = bootstrap_auroc_ci(probe, X, y, n_bootstrap=100)
    assert ci_low < ci_high
    assert ci_low > 0.5
    assert ci_high <= 1.0


def test_permutation_test(sample_activations):
    X, y = sample_activations
    probe = train_linear_probe(X, y)
    p_value = permutation_test_auroc(probe, X, y, n_permutations=100)
    assert 0 <= p_value <= 1
    assert p_value < 0.05


def test_select_best_layer(sample_activations):
    X, y = sample_activations
    activations = {0: X, 1: X * 0.5, 2: X * 2.0}
    best = select_best_layer(activations, y, cv_folds=3)
    assert best in activations


def test_select_best_layer_with_groups(sample_activations_with_groups):
    X, y, groups = sample_activations_with_groups
    activations = {0: X, 1: X * 0.5}
    best = select_best_layer(activations, y, groups=groups, cv_folds=3)
    assert best in activations


def test_estimate_confidence_direction(sample_activations):
    X, y = sample_activations
    rng = np.random.default_rng(42)
    log_probs = rng.standard_normal(len(X))
    direction = estimate_confidence_direction(X, log_probs)
    assert direction.shape == (X.shape[1],)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-6


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-10
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-10


def test_regress_out_directions(sample_activations):
    X, y = sample_activations
    rng = np.random.default_rng(42)
    directions = rng.standard_normal((3, X.shape[1]))
    X_res = regress_out_directions(X, directions)
    assert X_res.shape == X.shape
    for d in directions:
        proj = X_res @ d
        assert np.abs(proj).mean() < 0.1


def test_regress_out_preserves_signal(sample_activations):
    X, y = sample_activations
    rng = np.random.default_rng(123)
    random_dirs = rng.standard_normal((2, X.shape[1]))
    X_res = regress_out_directions(X, random_dirs)

    probe = train_linear_probe(X_res, y)
    auroc = evaluate_probe(probe, X_res, y)
    assert auroc > 0.8
