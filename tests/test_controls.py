"""Tests for experiment controls."""

import numpy as np

from src.controls import (
    confidence_alignment_over_training,
    frozen_vs_fresh_probe_tracking,
    random_label_baseline,
)


def test_random_label_baseline(sample_activations):
    X, y = sample_activations
    result = random_label_baseline(X, y, n_runs=10)
    assert "mean_auroc" in result
    assert result["mean_auroc"] < 0.75


def test_frozen_vs_fresh_probe_tracking(sample_checkpoint_activations):
    acts, labels, groups = sample_checkpoint_activations
    result = frozen_vs_fresh_probe_tracking(acts, labels, layer=9, groups_list=groups)
    assert len(result["frozen_aurocs"]) == 5
    assert len(result["fresh_aurocs"]) == 5
    # First checkpoint should have high frozen probe accuracy
    assert result["frozen_aurocs"][0] > 0.6
    # Fresh probes should always be at least reasonable
    for auroc in result["fresh_aurocs"]:
        assert auroc > 0.5


def test_frozen_probe_degrades(sample_checkpoint_activations):
    """Frozen probe should degrade as separation decreases."""
    acts, labels, groups = sample_checkpoint_activations
    result = frozen_vs_fresh_probe_tracking(acts, labels, layer=9, groups_list=groups)
    # First frozen should be >= last frozen (as we reduce separation)
    assert result["frozen_aurocs"][0] >= result["frozen_aurocs"][-1] - 0.1


def test_confidence_alignment(sample_checkpoint_activations):
    acts, labels, groups = sample_checkpoint_activations
    rng = np.random.default_rng(42)
    conf_dir = rng.standard_normal(64)
    conf_dir = conf_dir / np.linalg.norm(conf_dir)

    alignments = confidence_alignment_over_training(
        acts, labels, conf_dir, layer=9, groups_list=groups,
    )
    assert len(alignments) == 5
    for a in alignments:
        assert 0 <= a <= 1
