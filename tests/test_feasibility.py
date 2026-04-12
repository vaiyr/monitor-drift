"""Tests for feasibility gate."""

import numpy as np

from src.feasibility import (
    compute_problem_stats,
    label_generation,
    run_feasibility_gate,
    stratify_problems,
)


def test_label_generation():
    assert label_generation(2, 2, 2, 2) == "general"
    assert label_generation(2, 2, 1, 2) == "shortcut"
    assert label_generation(1, 2, 0, 2) == "failing"


def test_compute_problem_stats(sample_generations):
    stats = compute_problem_stats(sample_generations)
    assert len(stats) > 0
    for s in stats:
        assert 0 <= s.visible_pass_rate <= 1
        assert 0 <= s.hidden_pass_rate <= 1


def test_stratify_problems(sample_generations):
    stats = compute_problem_stats(sample_generations)
    stats = stratify_problems(stats)
    categories = {s.category for s in stats}
    # Should have at least some categorized problems
    assert len(categories) > 0


def test_run_feasibility_gate(sample_activations_with_groups):
    X, y, groups = sample_activations_with_groups
    acts = {5: X[:, :32], 9: X, 13: X[:, :32]}  # different "layers"
    result = run_feasibility_gate(acts, y, groups)
    assert result.probe_auroc > 0.5
    assert result.best_layer in acts
    assert isinstance(result.gate_passed, bool)


def test_feasibility_gate_with_confidence(sample_activations_with_groups):
    X, y, groups = sample_activations_with_groups
    rng = np.random.default_rng(42)
    conf_dir = rng.standard_normal(X.shape[1])
    conf_dir = conf_dir / np.linalg.norm(conf_dir)

    acts = {9: X}
    result = run_feasibility_gate(acts, y, groups, confidence_direction=conf_dir)
    assert 0 <= result.confidence_alignment <= 1
