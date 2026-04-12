"""Bootstrap CIs and permutation tests for probe evaluation."""

from __future__ import annotations


import numpy as np
from sklearn.metrics import roc_auc_score

from src.probe import get_probe_probs
from src.registry import BOOTSTRAP_RESAMPLES


def bootstrap_auroc_ci(
    probe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(42)
    probs = get_probe_probs(probe, X_test)
    aurocs = []
    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_test[idx]
        p_b = probs[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, p_b))

    alpha = (1 - ci) / 2
    return float(np.percentile(aurocs, alpha * 100)), float(np.percentile(aurocs, (1 - alpha) * 100))


def permutation_test_auroc(
    probe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_permutations: int = 1000,
) -> float:
    """Permutation test p-value for AUROC > 0.5."""
    rng = np.random.default_rng(42)
    probs = get_probe_probs(probe, X_test)
    observed = roc_auc_score(y_test, probs)

    count_ge = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_test)
        if len(np.unique(y_perm)) < 2:
            continue
        perm_auroc = roc_auc_score(y_perm, probs)
        if perm_auroc >= observed:
            count_ge += 1

    return (count_ge + 1) / (n_permutations + 1)
