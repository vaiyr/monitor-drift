"""Generate the figures that accompany the WRITEUP.md preprint.

Reads the raw JSON artifacts under results/paper/raw/ and the bootstrap
output under results/paper/ci_and_permutation.json, writes PDF figures to
results/paper/figures/.

Figures produced:

  fig1_probe_trajectory.pdf
    Frozen vs fresh AUROC across 6 checkpoints, plus cos(frozen, fresh) as a
    twin-axis overlay. The load-bearing plot for §4.3.

  fig2_dose_response.pdf
    Dose-response curves at round 2 and round 4 on the same axes. Shows
    monotonic behavior at both checkpoints, round 2 more responsive, and
    the round-4 α=5 main-trajectory outlier as a separate marker alongside
    the rerun. The load-bearing plot for §4.5.

  fig3_strategy_stability.pdf
    Iterative fraction of produced shortcuts across 6 checkpoints with
    bootstrap 95% CIs. Companion to §4.4.

  fig4_frozen_alpha5_trajectory.pdf
    The α=5 single-dose trajectory with bootstrap CIs on each point, for
    §4.5.1. Annotates the round-4 α=5 rerun as a correction marker.

Run with:
  python scripts/make_figures.py

No compute required. Finishes in under 10 seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "paper" / "raw"
OUT = REPO / "results" / "paper" / "figures"
CI = REPO / "results" / "paper" / "ci_and_permutation.json"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str):
    return json.loads((RAW / name).read_text())


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.autolayout": True,
})

FROZEN_COLOR = "#2E86AB"
FRESH_COLOR = "#E63946"
COSINE_COLOR = "#6B7280"
ROUND2_COLOR = "#2A9D8F"
ROUND4_COLOR = "#E76F51"


def fig1_probe_trajectory():
    data = json.loads(CI.read_text())
    frozen = data["probe_trajectory"]["frozen_aurocs"]
    fresh = data["probe_trajectory"]["fresh_aurocs"]
    cosines = list(data["cosine_trajectory"]["cosines"].values())
    rb = data["probe_trajectory"]["random_baseline"]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    x = list(range(6))
    x_labels = ["base", "round 0", "round 1", "round 2", "round 3", "round 4"]

    # AUROC lines
    ax1.plot(x, frozen, "o-", color=FROZEN_COLOR, label="frozen probe AUROC",
             lw=2, markersize=7)
    ax1.plot(x, fresh, "s-", color=FRESH_COLOR, label="fresh probe AUROC",
             lw=2, markersize=7)

    # Random baseline band
    rb_mean, rb_std = rb["mean_auroc"], rb["std_auroc"]
    ax1.axhspan(rb_mean - 2 * rb_std, rb_mean + 2 * rb_std,
                color="gray", alpha=0.2, label="random-label baseline (±2σ)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=15)
    ax1.set_ylabel("AUROC", color="black")
    ax1.set_ylim(0.45, 1.03)
    ax1.set_title(
        "Frozen probe AUROC degrades under training while fresh probe\n"
        "finds a rotated direction with higher AUROC (cos ≈ 0.37 sustained)"
    )

    # Cosine on a second axis
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, cosines, "D--", color=COSINE_COLOR,
             label="cos(frozen, fresh)", lw=1.5, markersize=6, alpha=0.9)
    ax2.set_ylabel("cosine similarity", color=COSINE_COLOR)
    ax2.tick_params(axis="y", colors=COSINE_COLOR)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower left", framealpha=0.9)

    path = OUT / "fig1_probe_trajectory.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig2_dose_response():
    r2 = json.loads(CI.read_text()).get("frozen_alpha_sweep_round2", [])
    r4 = json.loads(CI.read_text()).get("frozen_alpha_sweep_round4", [])
    rerun = json.loads(CI.read_text()).get("round_4_alpha5_reproducibility", {})

    def extract(sweep):
        alphas = [r["alpha"] for r in sweep]
        rel = [r["delta"]["rel_point"] * 100 for r in sweep]
        lo = [r["delta"]["rel_ci_lo"] * 100 for r in sweep]
        hi = [r["delta"]["rel_ci_hi"] * 100 for r in sweep]
        return alphas, rel, lo, hi

    a2, y2, lo2, hi2 = extract(r2)
    a4, y4, lo4, hi4 = extract(r4)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0, color="black", lw=0.8)

    # Round 2
    ax.errorbar(a2, y2,
                yerr=[np.array(y2) - np.array(lo2), np.array(hi2) - np.array(y2)],
                fmt="o-", color=ROUND2_COLOR, lw=2, capsize=4,
                label="round 2 (full α-sweep)", markersize=7)

    # Round 4 (exclude main-trajectory α=5 outlier; keep the other sweep points)
    a4_clean, y4_clean, lo4_clean, hi4_clean = [], [], [], []
    for a, y, lo, h, row in zip(a4, y4, lo4, hi4, r4):
        # Skip the main-trajectory α=5 outlier — will annotate separately
        if row.get("source") == "main_trajectory":
            continue
        a4_clean.append(a)
        y4_clean.append(y)
        lo4_clean.append(lo)
        hi4_clean.append(h)
    a4_clean = np.array(a4_clean)
    y4_clean = np.array(y4_clean)
    lo4_clean = np.array(lo4_clean)
    hi4_clean = np.array(hi4_clean)
    ax.errorbar(a4_clean, y4_clean,
                yerr=[y4_clean - lo4_clean, hi4_clean - y4_clean],
                fmt="s-", color=ROUND4_COLOR, lw=2, capsize=4,
                label="round 4 (α=5 value is rerun, §4.5.3)", markersize=7)

    # Plot the main-trajectory α=5 outlier as a distinct marker
    main_delta = rerun.get("main_trajectory_abs_delta")
    if main_delta is not None:
        # Convert abs to rel using the round_4 baseline average
        # (approximate: use 0.428 from main trajectory)
        main_rel = (main_delta / 0.428) * 100
        ax.plot(5.0, main_rel, "x", color="black", markersize=10,
                mew=2, label="round 4 α=5 main trajectory (outlier, §4.5.3)")
        ax.annotate("single-point\nnoise artifact\n(rerun → −9.8%)",
                    xy=(5.0, main_rel), xytext=(6.2, main_rel + 3),
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("steering coefficient α")
    ax.set_ylabel("relative Δ shortcut rate (%)")
    ax.set_title(
        "Frozen direction retains causal leverage at both checkpoints\n"
        "(monotonic dose-response; round 2 more responsive than round 4)"
    )
    ax.set_xticks([2, 5, 7, 10])
    ax.legend(loc="lower left")
    ax.set_ylim(-35, 15)

    path = OUT / "fig2_dose_response.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig3_strategy_stability():
    data = json.loads(CI.read_text())
    rows = data.get("strategy_trajectory", [])
    if not rows:
        return
    x = list(range(len(rows)))
    x_labels = [r["step"].replace("step_", "").replace("_", " ") for r in rows]
    points = [r["iterative_fraction"]["point"] for r in rows]
    lo = [r["iterative_fraction"]["ci_lo"] for r in rows]
    hi = [r["iterative_fraction"]["ci_hi"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.axhline(points[0], color="gray", ls="--", lw=0.8,
               label=f"base fraction = {points[0]:.3f}")
    ax.errorbar(x, points,
                yerr=[np.array(points) - np.array(lo), np.array(hi) - np.array(points)],
                fmt="o-", color=ROUND2_COLOR, lw=2, capsize=4, markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15)
    ax.set_ylabel("iterative fraction of shortcuts")
    ax.set_ylim(0.55, 0.80)
    p = data.get("strategy_stability_test", {}).get("p_two_sided", 1.0)
    ax.set_title(
        f"Shortcut strategy distribution is stable across training\n"
        f"(base vs round 4 permutation p = {p:.2f}; ±4pp band, no trend)"
    )
    ax.legend(loc="upper right")

    path = OUT / "fig3_strategy_stability.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig4_frozen_alpha5_trajectory():
    data = json.loads(CI.read_text())
    traj = data["frozen_alpha5_trajectory"]
    rerun = data.get("round_4_alpha5_reproducibility", {})
    ushape = data.get("u_shape_test", {})

    x = list(range(len(traj)))
    x_labels = [r["step"].replace("round_", "r") for r in traj]
    rel = [r["delta"]["rel_point"] * 100 for r in traj]
    lo = [r["delta"]["rel_ci_lo"] * 100 for r in traj]
    hi = [r["delta"]["rel_ci_hi"] * 100 for r in traj]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axhline(0, color="black", lw=0.8)
    ax.errorbar(x, rel,
                yerr=[np.array(rel) - np.array(lo), np.array(hi) - np.array(rel)],
                fmt="o-", color=FROZEN_COLOR, lw=2, capsize=4, markersize=7,
                label="α=5, main trajectory")

    # Mark the round-4 rerun on the same axes
    rerun_rel_abs = rerun.get("rerun_abs_delta")
    if rerun_rel_abs is not None:
        # absolute → relative using rerun baseline 0.469
        rerun_rel = (rerun_rel_abs / 0.469) * 100
        ax.plot(5, rerun_rel, "D", color=FRESH_COLOR, markersize=10,
                label="round 4 α=5 rerun (§4.5.3)")
        ax.annotate(
            f"rerun: {rerun_rel:.1f}%",
            xy=(5, rerun_rel), xytext=(4.2, rerun_rel - 6),
            fontsize=9, color=FRESH_COLOR,
            arrowprops=dict(arrowstyle="->", color=FRESH_COLOR, lw=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("training checkpoint")
    ax.set_ylabel("relative Δ shortcut rate at α=5 (%)")
    p = ushape.get("p_one_sided_lower", 1.0)
    ax.set_title(
        "α=5 single-dose trajectory with bootstrap 95% CIs\n"
        f"(U-shape not distinguishable from noise, p ≈ {p:.2f})"
    )
    ax.legend(loc="upper right")
    ax.set_ylim(-25, 20)

    path = OUT / "fig4_frozen_alpha5_trajectory.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    fig1_probe_trajectory()
    fig2_dose_response()
    fig3_strategy_stability()
    fig4_frozen_alpha5_trajectory()
    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()
