"""Linear probe training, evaluation, and confidence direction estimation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray | None = None,
    cv_folds: int = 5,
) -> Pipeline:
    """Train linear probe with cross-validated regularization.

    Args:
        groups: problem IDs for group-stratified CV. If provided, uses GroupKFold
                so no problem appears in both train and test folds.
    """
    if groups is not None:
        n_groups = len(set(groups))
        n_splits = min(cv_folds, n_groups)
        cv = GroupKFold(n_splits=n_splits)
        cv_iter = list(cv.split(X_train, y_train, groups))
    else:
        cv_iter = cv_folds

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=cv_iter, scoring="roc_auc",
            max_iter=1000, solver="lbfgs",
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_probe(probe, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Compute AUROC for a probe."""
    probs = probe.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


def get_probe_probs(probe, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities from a probe."""
    return probe.predict_proba(X)[:, 1]


def get_probe_direction(probe) -> np.ndarray:
    """Extract the weight vector (decision direction) from a trained probe."""
    scaler = probe.named_steps["scaler"]
    clf = probe.named_steps["clf"]
    # Unscale coefficients: w_original = w_scaled / scale
    return (clf.coef_[0] / scaler.scale_).astype(np.float64)


def select_best_layer(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    cv_folds: int = 5,
) -> int:
    """Select best layer via cross-validated AUROC."""
    best_layer = -1
    best_score = -1.0

    for layer_idx, X in sorted(activations.items()):
        if groups is not None:
            n_groups = len(set(groups))
            n_splits = min(cv_folds, n_groups)
            outer_cv = GroupKFold(n_splits=n_splits)
        else:
            outer_cv = cv_folds

        # Inner CV for LogisticRegressionCV uses simple k-fold (no groups)
        # Outer CV for cross_val_score uses GroupKFold if groups provided
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(Cs=10, cv=3, scoring="roc_auc", max_iter=1000)),
        ])
        scores = cross_val_score(pipe, X, labels, cv=outer_cv, scoring="roc_auc",
                                 groups=groups if groups is not None else None)
        mean_score = scores.mean()
        print(f"  layer {layer_idx}: AUROC = {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_layer = layer_idx

    print(f"  -> best layer: {best_layer} (AUROC = {best_score:.4f})")
    return best_layer


def estimate_confidence_direction(
    activations: np.ndarray,
    log_probs: np.ndarray,
) -> np.ndarray:
    """Estimate the 'solution confidence' direction via linear regression.

    Args:
        activations: (n_samples, hidden_dim) from correct general solutions
        log_probs: (n_samples,) mean log-probability of each solution's tokens

    Returns:
        (hidden_dim,) normalized direction vector
    """
    X = activations - activations.mean(axis=0)
    y = log_probs - log_probs.mean()
    # OLS: w = (X^T X)^{-1} X^T y
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return np.zeros(X.shape[1])
    return w / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def diff_of_means_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means direction: mu_pos - mu_neg.

    RepE / Arditi-style. Returns a vector in the original activation basis
    (no whitening, no unit-normalization). Points from negative class toward
    positive class.
    """
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(X.shape[1])
    return (pos.mean(axis=0) - neg.mean(axis=0)).astype(np.float64)


def auroc_from_direction(
    direction: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Score samples by dot product with direction and compute AUROC.

    This gives an extraction-method-agnostic way to compare DoM / LAT / LR
    directions on the same footing: no threshold fitting, no scaler, just
    the raw projection as the score.
    """
    scores = X @ direction
    return float(roc_auc_score(y, scores))


def regress_out_directions(X: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Project out specified directions from activation matrix.

    Args:
        X: (n_samples, hidden_dim) activation matrix
        directions: (n_directions, hidden_dim) direction vectors to remove

    Returns:
        X_residual: (n_samples, hidden_dim) with directions projected out
    """
    Q, _ = np.linalg.qr(directions.T)
    return X - X @ Q @ Q.T
