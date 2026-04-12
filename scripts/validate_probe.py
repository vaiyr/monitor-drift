"""Priority-0 validation: problem-identity confound check for the layer-11 probe.

Runs three checks on the base-checkpoint activations (step_0):

  1. Drop the two dominant shortcut-producing problems (92 and 795) and retrain.
     The AUROC should remain high (>0.85) if the feature is not driven by those
     two problems alone.
  2. Leave-one-problem-out (LOO-PO) across all shortcut-producing problems. Train
     on all but one shortcut problem, test on the held-out one + all general.
     Wild variance here would suggest problem-identity leakage.
  3. Train on 15 shortcut problems, test on the held-out 2. Tests generalization.

Writes results as JSON to /results/probes/validation.json on the Modal volume.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.extract import load_activations
from src.probe import evaluate_probe, train_linear_probe


def run_validation(
    volume_path: str = "/results",
    target_layer: int = 11,
    dominant_pids: tuple[str, ...] = ("92", "795"),
) -> dict:
    """Execute the three validation checks. Returns a dict of results."""
    step_dir = Path(volume_path) / "activations" / "step_0"
    print(f"[validate] loading activations from {step_dir}")

    s_acts_all = load_activations(str(step_dir / "shortcut.npz"))
    g_acts_all = load_activations(str(step_dir / "general.npz"))

    if target_layer not in s_acts_all or target_layer not in g_acts_all:
        raise KeyError(
            f"layer {target_layer} not in activation files "
            f"(have: shortcut={sorted(s_acts_all)}, general={sorted(g_acts_all)})"
        )

    s_acts = s_acts_all[target_layer]
    g_acts = g_acts_all[target_layer]
    s_pids = np.load(str(step_dir / "shortcut_pids.npy"), allow_pickle=True).astype(str)
    g_pids = np.load(str(step_dir / "general_pids.npy"), allow_pickle=True).astype(str)

    X = np.vstack([s_acts, g_acts])
    y = np.concatenate([np.ones(len(s_acts), dtype=int), np.zeros(len(g_acts), dtype=int)])
    pids = np.concatenate([s_pids, g_pids])

    n_shortcut = int(len(s_acts))
    n_general = int(len(g_acts))
    shortcut_problem_ids = sorted(set(s_pids.tolist()))
    pid_counts = Counter(s_pids.tolist())

    print(f"[validate] n_shortcut={n_shortcut}, n_general={n_general}")
    print(f"[validate] {len(shortcut_problem_ids)} distinct shortcut problems")
    print(f"[validate] shortcut counts per problem: {dict(pid_counts)}")

    results: dict[str, Any] = {
        "target_layer": target_layer,
        "n_shortcut": n_shortcut,
        "n_general": n_general,
        "n_shortcut_problems": len(shortcut_problem_ids),
        "shortcut_counts_per_problem": dict(pid_counts),
        "dominant_pids_requested": list(dominant_pids),
    }

    # Baseline: the original probe's in-sample AUROC for sanity comparison.
    base_probe = train_linear_probe(X, y, groups=pids)
    base_auroc = evaluate_probe(base_probe, X, y)
    print(f"[validate] baseline in-sample AUROC (all data): {base_auroc:.4f}")
    results["baseline_insample_auroc"] = float(base_auroc)

    # --- Check 1: drop the dominant problems entirely ---
    dominant_present = [p for p in dominant_pids if p in set(pids.tolist())]
    results["dominant_pids_present"] = dominant_present
    mask_drop = ~np.isin(pids, list(dominant_pids))
    n_drop_remaining = int(mask_drop.sum())
    n_shortcut_after = int(((y == 1) & mask_drop).sum())
    n_general_after = int(((y == 0) & mask_drop).sum())
    n_short_probs_after = len(
        set(pids[mask_drop & (y == 1)].tolist())
    )

    print(
        f"[validate] check 1: dropping {dominant_present} → "
        f"{n_drop_remaining} samples ({n_shortcut_after} shortcut, "
        f"{n_general_after} general, {n_short_probs_after} shortcut problems)"
    )

    if n_shortcut_after >= 5 and n_general_after >= 5 and n_short_probs_after >= 2:
        X_d, y_d, pids_d = X[mask_drop], y[mask_drop], pids[mask_drop]
        probe_dropped = train_linear_probe(X_d, y_d, groups=pids_d)
        auroc_dropped = evaluate_probe(probe_dropped, X_d, y_d)
        print(f"[validate] check 1 AUROC (drop dominant, in-sample): {auroc_dropped:.4f}")
        results["check1_drop_dominant"] = {
            "auroc": float(auroc_dropped),
            "n_shortcut": n_shortcut_after,
            "n_general": n_general_after,
            "n_shortcut_problems": n_short_probs_after,
        }
    else:
        print("[validate] check 1: skipping (not enough data after drop)")
        results["check1_drop_dominant"] = {"auroc": None, "reason": "insufficient data"}

    # --- Check 2: leave-one-shortcut-problem-out CV ---
    # Train on everything except one shortcut problem; test on held-out shortcut
    # samples plus a class-matched slice of general samples.
    loo_aurocs: dict[str, float] = {}
    loo_details: dict[str, dict] = {}
    for held_out in shortcut_problem_ids:
        held_mask = pids == held_out
        held_short_mask = held_mask & (y == 1)
        n_held_short = int(held_short_mask.sum())
        if n_held_short == 0:
            continue

        train_mask = ~held_mask
        # Train only if there is still at least one other shortcut problem.
        train_short_groups = set(pids[train_mask & (y == 1)].tolist())
        if len(train_short_groups) < 2:
            continue

        X_tr, y_tr, g_tr = X[train_mask], y[train_mask], pids[train_mask]
        try:
            probe = train_linear_probe(X_tr, y_tr, groups=g_tr)
        except Exception as e:
            print(f"[validate] LOO {held_out}: train failed ({e})")
            continue

        # Test set = held-out shortcut samples + all general samples.
        test_mask = held_short_mask | (y == 0)
        X_te, y_te = X[test_mask], y[test_mask]
        if len(set(y_te.tolist())) < 2:
            continue

        try:
            auroc = evaluate_probe(probe, X_te, y_te)
        except Exception as e:
            print(f"[validate] LOO {held_out}: eval failed ({e})")
            continue
        loo_aurocs[str(held_out)] = float(auroc)
        loo_details[str(held_out)] = {
            "auroc": float(auroc),
            "n_held_shortcut": n_held_short,
            "n_general_test": int((y == 0).sum()),
        }
        print(
            f"[validate] LOO {held_out}: AUROC={auroc:.4f} "
            f"(held {n_held_short} shortcut samples)"
        )

    if loo_aurocs:
        vals = np.array(list(loo_aurocs.values()))
        results["check2_loo"] = {
            "per_problem": loo_details,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "n_folds": int(len(vals)),
        }
        print(
            f"[validate] LOO summary: min={vals.min():.4f}, "
            f"mean={vals.mean():.4f}, max={vals.max():.4f}, "
            f"std={vals.std():.4f} (n={len(vals)})"
        )
    else:
        results["check2_loo"] = {"per_problem": {}, "reason": "no folds runnable"}

    # --- Check 3: train on 15 problems (drop dominant), test on the 2 dominant ---
    # Only runs if at least one of the dominant problems is present.
    if dominant_present:
        train_mask = ~np.isin(pids, list(dominant_pids))
        train_short_groups = set(pids[train_mask & (y == 1)].tolist())
        if len(train_short_groups) >= 2:
            X_tr, y_tr, g_tr = X[train_mask], y[train_mask], pids[train_mask]
            probe = train_linear_probe(X_tr, y_tr, groups=g_tr)

            # Evaluate on held-out dominant shortcut samples + all general
            test_mask = np.isin(pids, dominant_present) & (y == 1) | (y == 0)
            X_te, y_te = X[test_mask], y[test_mask]
            n_dom_short = int((np.isin(pids, dominant_present) & (y == 1)).sum())
            if len(set(y_te.tolist())) >= 2:
                auroc = evaluate_probe(probe, X_te, y_te)
                print(
                    f"[validate] check 3: train on 15 / test on dominant "
                    f"({n_dom_short} shortcut) AUROC={auroc:.4f}"
                )
                results["check3_dominant_generalization"] = {
                    "auroc": float(auroc),
                    "n_dominant_shortcut_test": n_dom_short,
                    "train_short_groups": sorted(train_short_groups),
                }
            else:
                results["check3_dominant_generalization"] = {
                    "reason": "test set has only one class"
                }
        else:
            results["check3_dominant_generalization"] = {
                "reason": "not enough non-dominant shortcut groups to train"
            }
    else:
        results["check3_dominant_generalization"] = {
            "reason": "no dominant pids present in data"
        }

    # --- Interpretation ---
    interpretation = []
    c1 = results["check1_drop_dominant"].get("auroc")
    if c1 is not None:
        c1 = float(c1)
        if c1 >= 0.85:
            interpretation.append(f"Check 1 PASS: drop-dominant AUROC {c1:.3f} ≥ 0.85")
        else:
            interpretation.append(f"Check 1 FAIL: drop-dominant AUROC {c1:.3f} < 0.85")

    if "min" in results.get("check2_loo", {}):
        mn = float(results["check2_loo"]["min"])
        mean = float(results["check2_loo"]["mean"])
        if mn >= 0.7:
            interpretation.append(f"Check 2 PASS: LOO min {mn:.3f} ≥ 0.7 (mean {mean:.3f})")
        else:
            interpretation.append(f"Check 2 FAIL: LOO min {mn:.3f} < 0.7 (mean {mean:.3f})")

    if "auroc" in results.get("check3_dominant_generalization", {}):
        c3 = float(results["check3_dominant_generalization"]["auroc"])
        if c3 >= 0.75:
            interpretation.append(f"Check 3 PASS: dominant generalization AUROC {c3:.3f}")
        else:
            interpretation.append(f"Check 3 FAIL: dominant generalization AUROC {c3:.3f}")

    results["interpretation"] = interpretation
    for line in interpretation:
        print(f"[validate] {line}")

    return results


def run_step_probe_validation(
    volume_path: str = "/results",
    target_layer: int = 11,
    step: int = 5,
    activations_subdir: str = "activations_cc",
    max_loo_folds: int = 60,
) -> dict:
    """Validate a freshly-trained probe at a specific checkpoint step.

    Loads activations from /<activations_subdir>/step_<step>/ and runs the
    same drop-dominant + LOO checks as run_feasibility_probe_validation, but
    on already-extracted activations (no re-extraction needed).

    Used to rule out that the fresh probe's high AUROC is due to new
    problem-identity leakage emerging under training.
    """
    step_dir = Path(volume_path) / activations_subdir / f"step_{step}"
    print(f"[validate-step] loading activations from {step_dir}")

    s_acts_all = load_activations(str(step_dir / "shortcut.npz"))
    g_acts_all = load_activations(str(step_dir / "general.npz"))
    if target_layer not in s_acts_all or target_layer not in g_acts_all:
        raise KeyError(
            f"layer {target_layer} missing "
            f"(shortcut={sorted(s_acts_all)}, general={sorted(g_acts_all)})"
        )

    s_acts = s_acts_all[target_layer]
    g_acts = g_acts_all[target_layer]
    s_pids = np.load(str(step_dir / "shortcut_pids.npy"), allow_pickle=True).astype(str)
    g_pids = np.load(str(step_dir / "general_pids.npy"), allow_pickle=True).astype(str)

    X = np.vstack([s_acts, g_acts])
    y = np.concatenate([np.ones(len(s_acts), dtype=int),
                        np.zeros(len(g_acts), dtype=int)])
    pids = np.concatenate([s_pids, g_pids])

    n_shortcut_total = int((y == 1).sum())
    n_general_total = int((y == 0).sum())
    shortcut_problem_ids = sorted(set(pids[y == 1].tolist()))
    pid_counts = Counter(pids[y == 1].tolist())

    print(f"[validate-step] step={step} layer={target_layer}: "
          f"{n_shortcut_total} shortcut, {n_general_total} general "
          f"over {len(shortcut_problem_ids)} shortcut problems")

    # Baseline: train on all data, compute in-sample AUROC (should be very high)
    base_probe = train_linear_probe(X, y, groups=pids)
    base_auroc = evaluate_probe(base_probe, X, y)
    print(f"[validate-step] baseline in-sample AUROC: {base_auroc:.4f}")

    # Dominant pids = top-2 by shortcut count
    dominant_pids = tuple(p for p, _ in pid_counts.most_common(2))

    results: dict[str, Any] = {
        "step": step,
        "target_layer": target_layer,
        "n_shortcut": n_shortcut_total,
        "n_general": n_general_total,
        "n_shortcut_problems": len(shortcut_problem_ids),
        "dominant_pids": list(dominant_pids),
        "top_shortcut_counts": dict(pid_counts.most_common(10)),
        "baseline_insample_auroc": float(base_auroc),
    }

    # Check 1: drop dominant
    mask_drop = ~np.isin(pids, list(dominant_pids))
    X_d, y_d, pids_d = X[mask_drop], y[mask_drop], pids[mask_drop]
    if (y_d == 1).sum() >= 5 and len(set(pids_d[y_d == 1].tolist())) >= 2:
        probe_d = train_linear_probe(X_d, y_d, groups=pids_d)
        auroc_d = evaluate_probe(probe_d, X_d, y_d)
        print(f"[validate-step] check 1: drop {list(dominant_pids)} → "
              f"AUROC={auroc_d:.4f}")
        results["check1_drop_dominant"] = {
            "auroc": float(auroc_d),
            "n_shortcut": int((y_d == 1).sum()),
            "n_general": int((y_d == 0).sum()),
            "n_shortcut_problems": len(set(pids_d[y_d == 1].tolist())),
        }

    # Check 2: LOO, capped at max_loo_folds by sample count
    loo_candidates = shortcut_problem_ids
    if len(loo_candidates) > max_loo_folds:
        loo_candidates = sorted(
            shortcut_problem_ids,
            key=lambda p: -pid_counts.get(p, 0),
        )[:max_loo_folds]

    loo_aurocs: dict[str, float] = {}
    loo_details: dict[str, dict] = {}
    for held_out in loo_candidates:
        held_mask = pids == held_out
        held_short_mask = held_mask & (y == 1)
        n_held_short = int(held_short_mask.sum())
        if n_held_short == 0:
            continue
        train_mask = ~held_mask
        train_short_groups = set(pids[train_mask & (y == 1)].tolist())
        if len(train_short_groups) < 2:
            continue
        probe = train_linear_probe(
            X[train_mask], y[train_mask], groups=pids[train_mask]
        )
        test_mask = held_short_mask | (y == 0)
        X_te, y_te = X[test_mask], y[test_mask]
        if len(set(y_te.tolist())) < 2:
            continue
        auroc = evaluate_probe(probe, X_te, y_te)
        loo_aurocs[str(held_out)] = float(auroc)
        loo_details[str(held_out)] = {
            "auroc": float(auroc),
            "n_held_shortcut": n_held_short,
        }

    if loo_aurocs:
        vals = np.array(list(loo_aurocs.values()))
        results["check2_loo"] = {
            "per_problem": loo_details,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "n_folds": int(len(vals)),
        }
        print(f"[validate-step] LOO: min={vals.min():.4f}, "
              f"mean={vals.mean():.4f}, median={np.median(vals):.4f}, "
              f"max={vals.max():.4f}, std={vals.std():.4f}, n={len(vals)}")

    return results


def save_results(results: dict[str, Any], volume_path: str = "/results",
                 filename: str = "validation.json") -> Path:
    out_path = Path(volume_path) / "probes" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[validate] wrote {out_path}")
    return out_path


def run_feasibility_probe_validation(
    volume_path: str = "/results",
    target_layer: int = 11,
    dominant_pids: tuple[str, ...] = ("92", "795"),
    n_general_cap: int = 500,
    feasibility_subdir: str = "feasibility",
    max_loo_folds: int = 60,
) -> dict:
    """Validate the *feasibility* probe.

    Re-extracts activations from the feasibility generations, reproduces the
    layer-11 GroupKFold probe, then runs drop-dominant + LOO + train-on-15
    validations. Works for any feasibility_subdir (e.g. "feasibility_cc"
    for CodeContests).
    """
    import gc

    import torch

    from src.extract import extract_activations, load_model_for_extraction
    from src.registry import BASE_MODEL
    from src.types import Generation

    gen_path = Path(volume_path) / feasibility_subdir / "generations.jsonl"
    print(f"[validate-feas] loading {gen_path}")
    gens: list[Generation] = []
    with gen_path.open() as f:
        for line in f:
            if line.strip():
                gens.append(Generation(**json.loads(line)))

    shortcuts = [g for g in gens if g.label == "shortcut"]
    generals = [g for g in gens if g.label == "general"]
    print(f"[validate-feas] {len(shortcuts)} shortcut, {len(generals)} general")

    # Cap general samples so extraction finishes in reasonable time.
    generals = generals[:n_general_cap]

    texts = [g.code for g in shortcuts] + [g.code for g in generals]
    y = np.array([1] * len(shortcuts) + [0] * len(generals), dtype=int)
    pids = np.array(
        [g.problem_id for g in shortcuts] + [g.problem_id for g in generals],
        dtype=object,
    ).astype(str)

    # --- Extract activations at target_layer only ---
    print(f"[validate-feas] loading base model and extracting activations "
          f"(n={len(texts)}, layer={target_layer})")
    model, tokenizer = load_model_for_extraction(BASE_MODEL, adapter_path=None)
    acts = extract_activations(
        model, tokenizer, texts, target_layers=[target_layer], batch_size=4,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if target_layer not in acts:
        raise RuntimeError(f"extract_activations returned no layer {target_layer}")
    X = acts[target_layer]
    assert len(X) == len(y), f"mismatch: X={len(X)}, y={len(y)}"

    n_shortcut_total = int((y == 1).sum())
    n_general_total = int((y == 0).sum())
    shortcut_problem_ids = sorted(set(pids[y == 1].tolist()))
    pid_counts = Counter(pids[y == 1].tolist())
    print(f"[validate-feas] {len(shortcut_problem_ids)} shortcut problems: {dict(pid_counts)}")

    # --- Baseline: reproduce feasibility probe in-sample AUROC ---
    base_probe = train_linear_probe(X, y, groups=pids)
    base_auroc = evaluate_probe(base_probe, X, y)
    print(f"[validate-feas] reproduced in-sample AUROC: {base_auroc:.4f}")

    results: dict[str, Any] = {
        "target_layer": target_layer,
        "n_shortcut": n_shortcut_total,
        "n_general": n_general_total,
        "n_shortcut_problems": len(shortcut_problem_ids),
        "shortcut_counts_per_problem": dict(pid_counts),
        "dominant_pids_requested": list(dominant_pids),
        "baseline_insample_auroc": float(base_auroc),
    }

    # --- Check 1: drop dominant problems, refit probe ---
    mask_drop = ~np.isin(pids, list(dominant_pids))
    X_d, y_d, pids_d = X[mask_drop], y[mask_drop], pids[mask_drop]
    n_drop_short_probs = len(set(pids_d[y_d == 1].tolist()))
    if (y_d == 1).sum() >= 5 and (y_d == 0).sum() >= 5 and n_drop_short_probs >= 2:
        probe_d = train_linear_probe(X_d, y_d, groups=pids_d)
        auroc_d = evaluate_probe(probe_d, X_d, y_d)
        print(f"[validate-feas] check 1 (drop {list(dominant_pids)}): AUROC={auroc_d:.4f} "
              f"on {int((y_d == 1).sum())} short / {int((y_d == 0).sum())} gen "
              f"over {n_drop_short_probs} shortcut problems")
        results["check1_drop_dominant"] = {
            "auroc": float(auroc_d),
            "n_shortcut": int((y_d == 1).sum()),
            "n_general": int((y_d == 0).sum()),
            "n_shortcut_problems": n_drop_short_probs,
        }
    else:
        results["check1_drop_dominant"] = {"auroc": None, "reason": "insufficient data"}

    # --- Check 2: leave-one-shortcut-problem-out CV ---
    # If there are more shortcut problems than max_loo_folds, sample the ones
    # with the most samples (those are the ones we can get a stable per-fold
    # estimate on).
    loo_candidates = shortcut_problem_ids
    if len(loo_candidates) > max_loo_folds:
        loo_candidates = sorted(
            shortcut_problem_ids,
            key=lambda p: -pid_counts.get(p, 0),
        )[:max_loo_folds]
        print(f"[validate-feas] capping LOO to top-{max_loo_folds} shortcut problems "
              f"by sample count")

    loo_aurocs: dict[str, float] = {}
    loo_details: dict[str, dict] = {}
    for held_out in loo_candidates:
        held_mask = pids == held_out
        held_short_mask = held_mask & (y == 1)
        n_held_short = int(held_short_mask.sum())
        if n_held_short == 0:
            continue

        train_mask = ~held_mask
        train_short_groups = set(pids[train_mask & (y == 1)].tolist())
        if len(train_short_groups) < 2:
            continue

        try:
            probe = train_linear_probe(
                X[train_mask], y[train_mask], groups=pids[train_mask]
            )
        except Exception as e:
            print(f"[validate-feas] LOO {held_out}: train failed ({e})")
            continue

        test_mask = held_short_mask | (y == 0)
        X_te, y_te = X[test_mask], y[test_mask]
        if len(set(y_te.tolist())) < 2:
            continue

        auroc = evaluate_probe(probe, X_te, y_te)
        loo_aurocs[str(held_out)] = float(auroc)
        loo_details[str(held_out)] = {
            "auroc": float(auroc),
            "n_held_shortcut": n_held_short,
            "n_general_test": int((y == 0).sum()),
        }
        print(f"[validate-feas] LOO {held_out}: AUROC={auroc:.4f} "
              f"(held {n_held_short} shortcut samples)")

    if loo_aurocs:
        vals = np.array(list(loo_aurocs.values()))
        results["check2_loo"] = {
            "per_problem": loo_details,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "n_folds": int(len(vals)),
        }
        print(f"[validate-feas] LOO summary: min={vals.min():.4f}, "
              f"mean={vals.mean():.4f}, max={vals.max():.4f}, "
              f"std={vals.std():.4f} (n={len(vals)})")
    else:
        results["check2_loo"] = {"per_problem": {}, "reason": "no folds runnable"}

    # --- Check 3: train on 15 non-dominant problems, test on the 2 dominant ---
    dominant_present = [p for p in dominant_pids if p in set(pids.tolist())]
    if dominant_present:
        train_mask = ~np.isin(pids, list(dominant_pids))
        train_short_groups = set(pids[train_mask & (y == 1)].tolist())
        if len(train_short_groups) >= 2:
            probe = train_linear_probe(
                X[train_mask], y[train_mask], groups=pids[train_mask],
            )
            test_mask = (np.isin(pids, dominant_present) & (y == 1)) | (y == 0)
            X_te, y_te = X[test_mask], y[test_mask]
            n_dom_short = int((np.isin(pids, dominant_present) & (y == 1)).sum())
            if len(set(y_te.tolist())) >= 2:
                auroc = evaluate_probe(probe, X_te, y_te)
                print(f"[validate-feas] check 3 (train on {len(train_short_groups)} "
                      f"non-dominant / test on {n_dom_short} dominant shortcut + "
                      f"{int((y==0).sum())} general): AUROC={auroc:.4f}")
                results["check3_dominant_generalization"] = {
                    "auroc": float(auroc),
                    "n_dominant_shortcut_test": n_dom_short,
                    "n_train_groups": len(train_short_groups),
                }
            else:
                results["check3_dominant_generalization"] = {
                    "reason": "test set has only one class"
                }
        else:
            results["check3_dominant_generalization"] = {
                "reason": "not enough non-dominant shortcut groups"
            }
    else:
        results["check3_dominant_generalization"] = {
            "reason": "no dominant pids present"
        }

    # --- Interpretation ---
    interpretation = []
    c1 = results["check1_drop_dominant"].get("auroc")
    if c1 is not None:
        c1 = float(c1)
        tag = "PASS" if c1 >= 0.85 else "FAIL"
        interpretation.append(f"Check 1 {tag}: drop-dominant AUROC {c1:.3f} vs 0.85 threshold")

    if "min" in results.get("check2_loo", {}):
        mn = float(results["check2_loo"]["min"])
        mean = float(results["check2_loo"]["mean"])
        tag = "PASS" if mn >= 0.7 else "FAIL"
        interpretation.append(
            f"Check 2 {tag}: LOO min {mn:.3f} vs 0.7 (mean {mean:.3f}, n={results['check2_loo']['n_folds']})"
        )

    if "auroc" in results.get("check3_dominant_generalization", {}):
        c3 = float(results["check3_dominant_generalization"]["auroc"])
        tag = "PASS" if c3 >= 0.75 else "FAIL"
        interpretation.append(f"Check 3 {tag}: dominant generalization AUROC {c3:.3f} vs 0.75")

    results["interpretation"] = interpretation
    for line in interpretation:
        print(f"[validate-feas] {line}")

    return results


if __name__ == "__main__":
    # Local execution path (if activations are mounted locally). Prefer invoking
    # via deploy/app.py::validate_probe for Modal execution.
    res = run_validation()
    save_results(res)
