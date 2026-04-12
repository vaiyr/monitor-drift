"""Feasibility gate: probe go/no-go and problem stratification."""

from __future__ import annotations

import numpy as np

from src.registry import (
    FEASIBILITY_PROBE_THRESHOLD,
    SHORTCUT_NECESSARY_HIDDEN_THRESHOLD,
    SHORTCUT_NECESSARY_VISIBLE_THRESHOLD,
)
from src.types import FeasibilityResult, Generation, ProblemStats, SolutionLabel


def label_generation(visible_passed: int, visible_total: int,
                     hidden_passed: int, hidden_total: int) -> SolutionLabel:
    """Label a generation as shortcut, general, or failing."""
    if visible_passed < visible_total:
        return "failing"
    if hidden_passed == hidden_total:
        return "general"
    return "shortcut"


def compute_problem_stats(generations: list[Generation]) -> list[ProblemStats]:
    """Compute per-problem pass rate statistics from generations."""
    from collections import defaultdict
    by_problem: dict[str, list[Generation]] = defaultdict(list)
    for g in generations:
        by_problem[g.problem_id].append(g)

    stats = []
    for pid, gens in by_problem.items():
        n = len(gens)
        vis_pass = sum(1 for g in gens if g.visible_passed == g.visible_total)
        hid_pass = sum(1 for g in gens if g.hidden_passed == g.hidden_total)
        stats.append(ProblemStats(
            problem_id=pid,
            visible_pass_rate=vis_pass / n if n else 0,
            hidden_pass_rate=hid_pass / n if n else 0,
            n_samples=n,
        ))
    return stats


def stratify_problems(stats: list[ProblemStats]) -> list[ProblemStats]:
    """Classify problems as shortcut-necessary, shortcut-optional, or unclear.

    shortcut-necessary: hidden pass rate < 5%, visible pass rate > 20%
    shortcut-optional: both visible and hidden pass rates > 20%
    unclear: everything else
    """
    for s in stats:
        if (s.hidden_pass_rate < SHORTCUT_NECESSARY_HIDDEN_THRESHOLD
                and s.visible_pass_rate > SHORTCUT_NECESSARY_VISIBLE_THRESHOLD):
            s.category = "shortcut_necessary"
        elif (s.hidden_pass_rate > SHORTCUT_NECESSARY_VISIBLE_THRESHOLD
              and s.visible_pass_rate > SHORTCUT_NECESSARY_VISIBLE_THRESHOLD):
            s.category = "shortcut_optional"
        else:
            s.category = "unclear"
    return stats


def run_feasibility_gate(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    confidence_direction: np.ndarray | None = None,
) -> FeasibilityResult:
    """Run the probe go/no-go gate.

    Args:
        activations: layer -> (n_samples, hidden_dim)
        labels: (n_samples,) binary labels (1=shortcut, 0=general)
        groups: (n_samples,) problem IDs for group-stratified CV
        confidence_direction: if provided, compute alignment with probe direction

    Returns:
        FeasibilityResult with gate_passed flag
    """
    from src.probe import (
        cosine_similarity,
        evaluate_probe,
        get_probe_direction,
        select_best_layer,
        train_linear_probe,
    )

    if not activations:
        print("[feasibility] no activations extracted")
        return FeasibilityResult(
            probe_auroc=0.0, best_layer=0, confidence_alignment=0.0,
            n_shortcut_necessary=0, n_shortcut_optional=0, gate_passed=False,
        )

    best_layer = select_best_layer(activations, labels, groups=groups, cv_folds=5)
    if best_layer < 0:
        best_layer = sorted(activations.keys())[0]
    X = activations[best_layer]
    probe = train_linear_probe(X, labels, groups=groups)
    auroc = evaluate_probe(probe, X, labels)

    alignment = 0.0
    if confidence_direction is not None:
        probe_dir = get_probe_direction(probe)
        alignment = abs(cosine_similarity(probe_dir, confidence_direction))

    gate_passed = auroc >= FEASIBILITY_PROBE_THRESHOLD

    print(f"[feasibility] AUROC={auroc:.4f}, layer={best_layer}, "
          f"confidence_alignment={alignment:.4f}, gate={'PASSED' if gate_passed else 'FAILED'}")

    return FeasibilityResult(
        probe_auroc=auroc,
        best_layer=best_layer,
        confidence_alignment=alignment,
        n_shortcut_necessary=0,
        n_shortcut_optional=0,
        gate_passed=gate_passed,
    )
