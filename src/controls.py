"""Controls for the RL feature dynamics experiment.

1. Problem-stratified cross-validation (built into probe.py via group-KFold)
2. Random labels baseline
3. Confidence direction alignment tracking
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.probe import (
    cosine_similarity,
    get_probe_direction,
    get_probe_probs,
    train_linear_probe,
)

def random_label_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = 100,
) -> dict:
    """Train probes on shuffled labels. Should yield ~0.5 AUROC."""
    rng = np.random.default_rng(42)
    aurocs = []
    for _ in range(n_runs):
        y_shuffled = rng.permutation(y)
        probe = train_linear_probe(X, y_shuffled)
        probs = get_probe_probs(probe, X)
        if len(np.unique(y)) >= 2:
            aurocs.append(roc_auc_score(y, probs))

    return {
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "min_auroc": float(np.min(aurocs)),
        "max_auroc": float(np.max(aurocs)),
    }


def confidence_alignment_over_training(
    checkpoint_activations: list[dict[int, np.ndarray]],
    checkpoint_labels: list[np.ndarray],
    confidence_direction: np.ndarray,
    layer: int,
    groups_list: list[np.ndarray] | None = None,
) -> list[float]:
    """Track cosine similarity between probe direction and confidence direction
    across training checkpoints.

    Returns list of alignment values (one per checkpoint).
    """
    alignments = []
    for i, (acts, labels) in enumerate(zip(checkpoint_activations, checkpoint_labels)):
        X = acts[layer]
        groups = groups_list[i] if groups_list else None
        probe = train_linear_probe(X, labels, groups=groups)
        probe_dir = get_probe_direction(probe)
        alignment = abs(cosine_similarity(probe_dir, confidence_direction))
        alignments.append(alignment)
    return alignments


def frozen_vs_fresh_probe_tracking(
    checkpoint_activations: list[dict[int, np.ndarray]],
    checkpoint_labels: list[np.ndarray],
    layer: int,
    groups_list: list[np.ndarray] | None = None,
) -> dict:
    """Track frozen probe and fresh probe AUROC across checkpoints.

    Trains a frozen probe at checkpoint 0, applies it at all subsequent checkpoints.
    Also trains fresh probes at each checkpoint.

    Returns dict with 'frozen_aurocs' and 'fresh_aurocs' lists.
    """
    if not checkpoint_activations:
        return {"frozen_aurocs": [], "fresh_aurocs": []}

    # Train frozen probe at checkpoint 0
    X0 = checkpoint_activations[0][layer]
    y0 = checkpoint_labels[0]
    g0 = groups_list[0] if groups_list else None
    frozen_probe = train_linear_probe(X0, y0, groups=g0)

    frozen_aurocs = []
    fresh_aurocs = []

    for i, (acts, labels) in enumerate(zip(checkpoint_activations, checkpoint_labels)):
        X = acts[layer]
        y = labels

        # Frozen probe
        try:
            frozen_auroc = float(roc_auc_score(y, get_probe_probs(frozen_probe, X)))
        except ValueError:
            frozen_auroc = 0.5
        frozen_aurocs.append(frozen_auroc)

        # Fresh probe
        groups = groups_list[i] if groups_list else None
        fresh_probe = train_linear_probe(X, y, groups=groups)
        try:
            fresh_auroc = float(roc_auc_score(y, get_probe_probs(fresh_probe, X)))
        except ValueError:
            fresh_auroc = 0.5
        fresh_aurocs.append(fresh_auroc)

    return {
        "frozen_aurocs": frozen_aurocs,
        "fresh_aurocs": fresh_aurocs,
    }
