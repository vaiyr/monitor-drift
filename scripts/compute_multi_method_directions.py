"""Multi-method direction trajectories across checkpoints.

Computes LR / Difference-of-Means / LAT directions at each of the 6
checkpoints, using cached layer-11 residual stream activations, and reports:

  * Per-method fresh-direction AUROC trajectory
  * Cross-checkpoint cosine matrix per method (vs base)
  * Cross-method cosine at each checkpoint
  * LOO validation of DoM and LAT at step_5

Reads activations from results/paper/activations_cc/step_{N}/
{shortcut,general}.npz, as downloaded from the Modal volume.

Writes results/paper/multi_method_directions.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.probe import (
    auroc_from_direction,
    cosine_similarity,
    diff_of_means_direction,
)

REPO = Path(__file__).resolve().parents[1]
ACT_DIR = REPO / "results" / "paper" / "activations_cc"
OUT_PATH = REPO / "results" / "paper" / "multi_method_directions.json"

PROBE_LAYER = 11
N_CHECKPOINTS = 6
CHECKPOINT_NAMES = ["base", "round_0", "round_1", "round_2", "round_3", "round_4"]


def load_checkpoint(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load layer-11 activations, labels, and group IDs for one checkpoint.

    Returns:
        X:       (n_samples, hidden_dim)
        y:       (n_samples,) with 1=shortcut, 0=general
        groups:  (n_samples,) problem IDs (may be int or str)
    """
    step_dir = ACT_DIR / f"step_{step}"
    s_npz = np.load(step_dir / "shortcut.npz")
    g_npz = np.load(step_dir / "general.npz")
    s_key = f"layer_{PROBE_LAYER}"
    X_s = s_npz[s_key]
    X_g = g_npz[s_key]

    s_pids_path = step_dir / "shortcut_pids.npy"
    g_pids_path = step_dir / "general_pids.npy"
    if s_pids_path.exists() and g_pids_path.exists():
        s_pids = np.load(s_pids_path, allow_pickle=True)
        g_pids = np.load(g_pids_path, allow_pickle=True)
    else:
        # Fallback: synthetic groups (one per sample), no LOO meaning
        s_pids = np.arange(len(X_s))
        g_pids = np.arange(len(X_g)) + 10_000

    X = np.concatenate([X_s, X_g], axis=0)
    y = np.concatenate([np.ones(len(X_s)), np.zeros(len(X_g))]).astype(int)
    groups = np.concatenate([s_pids, g_pids])
    return X, y, groups


def lr_direction(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic LR probe with explicit C and seed.

    Unlike src.probe.train_linear_probe which uses LogisticRegressionCV (which
    in turn picks C via group-k-fold and is sensitive to fold order), this fit
    is a simple StandardScaler + LogisticRegression at a fixed C and seed, so
    two calls on identical data return identical coefficients. This is what
    lets us measure solver variance vs. true feature rotation.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, solver="lbfgs", max_iter=2000, random_state=seed,
        )),
    ])
    pipe.fit(X, y)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    return (clf.coef_[0] / scaler.scale_).astype(np.float64)


def lat_paired_direction(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    min_pair_size: int = 1,
) -> np.ndarray:
    """LAT with per-problem pairing.

    RepE's LAT assumes each paired difference (h_pos - h_neg) comes from a
    matched stimulus pair. The shortcut/general distinction here has per-
    problem pairs: for every problem with at least one shortcut and one general
    solution, we can form a per-problem contrastive difference using the
    per-problem class means. Collecting these per-problem differences and
    taking the top PC gives a RepE-faithful direction.
    """
    unique_groups = np.unique(groups)
    pair_diffs = []
    for g in unique_groups:
        mask = groups == g
        yg = y[mask]
        Xg = X[mask]
        pos = Xg[yg == 1]
        neg = Xg[yg == 0]
        if len(pos) < min_pair_size or len(neg) < min_pair_size:
            continue
        pair_diffs.append(pos.mean(axis=0) - neg.mean(axis=0))
    if len(pair_diffs) < 2:
        return np.zeros(X.shape[1])
    D = np.stack(pair_diffs, axis=0)
    # RepE: center the per-pair differences, then top PC via SVD.
    D_centered = D - D.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(D_centered, full_matrices=False)
    direction = vt[0].astype(np.float64)
    # Sign alignment using mean over the uncentered differences
    if (D.mean(axis=0) @ direction) < 0:
        direction = -direction
    return direction


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically downsample the majority class to match the minority.

    Matches the writeup's 'balanced shortcut/general classes' protocol.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    pos_sel = rng.permutation(pos_idx)[:n]
    neg_sel = rng.permutation(neg_idx)[:n]
    sel = np.concatenate([pos_sel, neg_sel])
    sel.sort()
    return X[sel], y[sel], groups[sel]


def groupkfold_auroc_from_direction_method(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Cross-validated AUROC for a direction-extraction method.

    For LR/DoM/LAT, fit the direction on train folds, score the test fold by
    dot product, and compute AUROC from the pooled scores. Matches the
    protocol of §4.3 of the writeup.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    all_scores = np.zeros(len(X))
    for train_idx, test_idx in cv.split(X, y, groups):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        g_tr = groups[train_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        d = direction_fit(method, X_tr, y_tr, g_tr)
        if np.linalg.norm(d) < 1e-12:
            continue
        all_scores[test_idx] = X[test_idx] @ d
    return float(roc_auc_score(y, all_scores))


def direction_fit(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    if name == "lr":
        return lr_direction(X, y)
    if name == "dom":
        return diff_of_means_direction(X, y)
    if name == "lat":
        return lat_paired_direction(X, y, groups)
    raise ValueError(name)


def loo_validation(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = 3,
) -> dict:
    """Leave-one-problem-out validation for DoM/LAT/LR.

    For each unique group with at least min_group_size samples and at least
    one positive and one negative sample held out, refit the direction on
    the other groups and compute AUROC on the held-out group.
    """
    unique_groups = np.unique(groups)
    fold_aurocs = []
    fold_sizes = []
    fold_skipped_labels = 0
    for g in unique_groups:
        test_mask = groups == g
        if test_mask.sum() < min_group_size:
            continue
        y_test = y[test_mask]
        if len(np.unique(y_test)) < 2:
            fold_skipped_labels += 1
            continue
        train_mask = ~test_mask
        X_train = X[train_mask]
        y_train = y[train_mask]
        g_train = groups[train_mask]
        if len(np.unique(y_train)) < 2:
            continue
        d = direction_fit(name, X_train, y_train, g_train)
        if np.linalg.norm(d) < 1e-12:
            continue
        try:
            auroc = auroc_from_direction(d, X[test_mask], y_test)
        except ValueError:
            continue
        fold_aurocs.append(auroc)
        fold_sizes.append(int(test_mask.sum()))
    if not fold_aurocs:
        return {"n_folds": 0, "mean": None, "median": None, "min": None, "std": None}
    arr = np.array(fold_aurocs)
    return {
        "n_folds": len(fold_aurocs),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "skipped_single_label": int(fold_skipped_labels),
    }


def main() -> None:
    print(f"[multi-method] loading activations from {ACT_DIR}")
    checkpoints = []
    for step in range(N_CHECKPOINTS):
        X_raw, y_raw, g_raw = load_checkpoint(step)
        X, y, groups = balance_classes(X_raw, y_raw, g_raw, seed=0)
        print(
            f"  step_{step} ({CHECKPOINT_NAMES[step]}):"
            f" n_raw={len(X_raw)} -> balanced n={len(X)}"
            f" dim={X.shape[1]}"
            f" pos={int(y.sum())} neg={int((1 - y).sum())}"
            f" groups={len(np.unique(groups))}"
        )
        checkpoints.append((X, y, groups))

    methods = ["lr", "dom", "lat"]
    # LR solver-variance sanity check at step_0: re-fit LR 5 times at different
    # seeds on the same (balanced) data and report pairwise cosines. If LR is
    # stable, these should be ≈ 1.0; if solver instability is large, they'll be
    # << 1.0 and the 'rotation' finding is confounded by solver variance.
    X0, y0, _g0 = checkpoints[0]
    lr_seeds = [lr_direction(X0, y0, seed=s) for s in range(5)]
    lr_self_cosines = [
        cosine_similarity(lr_seeds[i], lr_seeds[j])
        for i in range(len(lr_seeds))
        for j in range(i + 1, len(lr_seeds))
    ]
    lr_self_cosine_mean = float(np.mean(lr_self_cosines))
    lr_self_cosine_min = float(np.min(lr_self_cosines))
    print(
        f"\n[multi-method] LR solver variance at step_0: mean cos={lr_self_cosine_mean:.4f}"
        f" min cos={lr_self_cosine_min:.4f} across 5 seeds"
    )

    # Fit directions at each checkpoint (default seed=0 for LR)
    directions: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    for step in range(N_CHECKPOINTS):
        X, y, groups = checkpoints[step]
        for m in methods:
            d = direction_fit(m, X, y, groups)
            directions[m].append(d)
            print(
                f"  fit {m} @ step_{step}:"
                f" ||d||={np.linalg.norm(d):.4f}"
            )

    # Fresh-direction AUROC trajectory per method (in-sample, using the full
    # checkpoint data — same metric as the frozen/fresh AUROC in §4.3).
    in_sample_auroc: dict[str, list[float]] = {m: [] for m in methods}
    for step in range(N_CHECKPOINTS):
        X, y, _ = checkpoints[step]
        for m in methods:
            in_sample_auroc[m].append(auroc_from_direction(directions[m][step], X, y))

    # Group-k-fold CV AUROC per method — matches the writeup's §4.3 protocol
    # (which uses LogisticRegressionCV with GroupKFold). Runs 5 folds by
    # problem ID, scores on the held-out folds, and computes a pooled AUROC.
    gkf_auroc: dict[str, list[float]] = {m: [] for m in methods}
    for step in range(N_CHECKPOINTS):
        X, y, groups = checkpoints[step]
        for m in methods:
            gkf_auroc[m].append(
                groupkfold_auroc_from_direction_method(m, X, y, groups, n_splits=5)
            )

    # Frozen-direction AUROC trajectory: freeze the step_0 direction per method
    # and measure its AUROC at every subsequent checkpoint. This is the
    # apples-to-apples version of §4.3's frozen AUROC table.
    frozen_auroc: dict[str, list[float]] = {m: [] for m in methods}
    for m in methods:
        d0 = directions[m][0]
        for step in range(N_CHECKPOINTS):
            X, y, _ = checkpoints[step]
            frozen_auroc[m].append(auroc_from_direction(d0, X, y))

    # Cosine trajectory: cos(step_0 direction, step_k direction) per method
    cos_vs_base: dict[str, list[float]] = {m: [] for m in methods}
    for m in methods:
        d0 = directions[m][0]
        for step in range(N_CHECKPOINTS):
            cos_vs_base[m].append(cosine_similarity(d0, directions[m][step]))

    # Cross-method cosine at each checkpoint (LR vs DoM, LR vs LAT, DoM vs LAT)
    cross_method_cosine: list[dict[str, float]] = []
    for step in range(N_CHECKPOINTS):
        cross_method_cosine.append({
            "lr_vs_dom": cosine_similarity(
                directions["lr"][step], directions["dom"][step]
            ),
            "lr_vs_lat": cosine_similarity(
                directions["lr"][step], directions["lat"][step]
            ),
            "dom_vs_lat": cosine_similarity(
                directions["dom"][step], directions["lat"][step]
            ),
        })

    # LOO validation at step_5 for DoM and LAT (LR LOO already exists in §3.3)
    X5, y5, g5 = checkpoints[5]
    loo_results = {
        "lr": loo_validation("lr", X5, y5, g5),
        "dom": loo_validation("dom", X5, y5, g5),
        "lat": loo_validation("lat", X5, y5, g5),
    }

    result = {
        "probe_layer": PROBE_LAYER,
        "checkpoint_names": CHECKPOINT_NAMES,
        "balanced_classes": True,
        "in_sample_auroc": in_sample_auroc,
        "gkf_auroc": gkf_auroc,
        "frozen_auroc": frozen_auroc,
        "cos_vs_base": cos_vs_base,
        "cross_method_cosine": cross_method_cosine,
        "loo_step_5": loo_results,
        "direction_norms": {
            m: [float(np.linalg.norm(directions[m][s])) for s in range(N_CHECKPOINTS)]
            for m in methods
        },
        "lr_solver_variance_step_0": {
            "n_seeds": len(lr_seeds),
            "pairwise_cosines": [float(c) for c in lr_self_cosines],
            "mean_cos": lr_self_cosine_mean,
            "min_cos": lr_self_cosine_min,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[multi-method] wrote {OUT_PATH}")

    # Pretty summary
    print("\n== In-sample AUROC (fresh) ==")
    header = "ckpt".ljust(10) + "".join(m.upper().ljust(10) for m in methods)
    print(header)
    for step, name in enumerate(CHECKPOINT_NAMES):
        row = name.ljust(10)
        for m in methods:
            row += f"{in_sample_auroc[m][step]:.4f}".ljust(10)
        print(row)

    print("\n== GroupKFold-CV AUROC (matches writeup §4.3 'fresh AUROC' protocol) ==")
    print(header)
    for step, name in enumerate(CHECKPOINT_NAMES):
        row = name.ljust(10)
        for m in methods:
            row += f"{gkf_auroc[m][step]:.4f}".ljust(10)
        print(row)

    print("\n== Frozen-at-step_0 AUROC ==")
    print(header)
    for step, name in enumerate(CHECKPOINT_NAMES):
        row = name.ljust(10)
        for m in methods:
            row += f"{frozen_auroc[m][step]:.4f}".ljust(10)
        print(row)

    print("\n== cos(step_0, step_k) ==")
    print(header)
    for step, name in enumerate(CHECKPOINT_NAMES):
        row = name.ljust(10)
        for m in methods:
            row += f"{cos_vs_base[m][step]:+.4f}".ljust(10)
        print(row)

    print("\n== Cross-method cosine at each checkpoint ==")
    print("ckpt".ljust(10) + "LR·DoM".ljust(12) + "LR·LAT".ljust(12) + "DoM·LAT".ljust(12))
    for step, name in enumerate(CHECKPOINT_NAMES):
        c = cross_method_cosine[step]
        print(
            name.ljust(10)
            + f"{c['lr_vs_dom']:+.4f}".ljust(12)
            + f"{c['lr_vs_lat']:+.4f}".ljust(12)
            + f"{c['dom_vs_lat']:+.4f}".ljust(12)
        )

    print("\n== LOO validation @ step_5 ==")
    print("method".ljust(10) + "n_folds".ljust(10) + "mean".ljust(10) + "median".ljust(10) + "min".ljust(10))
    for m in methods:
        res = loo_results[m]
        mean = res["mean"]
        median = res["median"]
        mn = res["min"]
        mean_s = f"{mean:.4f}" if mean is not None else "n/a"
        median_s = f"{median:.4f}" if median is not None else "n/a"
        min_s = f"{mn:.4f}" if mn is not None else "n/a"
        print(
            m.ljust(10)
            + str(res["n_folds"]).ljust(10)
            + mean_s.ljust(10)
            + median_s.ljust(10)
            + min_s.ljust(10)
        )


if __name__ == "__main__":
    main()
