"""Figure 4: directional ablation by extraction method at round 4."""
import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style, savefig, BLUE, VERMILLION, GREEN, ORANGE, PURPLE, GREY

apply_style()

# Frozen (fit at base, ablated at r4)
frozen = [
    ("LogisticRegression-CV",   -0.9,  -10.2,  9.3,  PURPLE),
    ("Deterministic LR ($C{=}1$)",  2.4,  -7.4, 13.1,  ORANGE),
    ("Difference-of-means",     -7.2, -14.8,  1.6,  GREEN),
]
# Fresh (fit at r4, ablated at r4)
fresh = [
    ("LogisticRegression-CV",    1.2,  -8.1, 10.8,  PURPLE),
    ("Deterministic LR ($C{=}1$)",  -4.8, -12.6,  2.9,  ORANGE),
    ("Difference-of-means",    -36.5, -42.5, -30.2, GREEN),
]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)

def plot_bars(ax, rows, title):
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    los_v = [r[1] - r[2] for r in rows]
    his_v = [r[3] - r[1] for r in rows]
    ci_lo = [r[2] for r in rows]
    ci_hi = [r[3] for r in rows]
    colors = [r[4] for r in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, means, yerr=[los_v, his_v], color=colors, edgecolor="black",
                  lw=0.6, capsize=4, error_kw={"lw": 0.9})
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=8.5)
    ax.set_title(title)
    # Place numeric labels BEYOND the error bar caps so nothing overlaps.
    for bar, m, lo, hi, col in zip(bars, means, ci_lo, ci_hi, colors):
        if m < 0:
            y = lo - 2.0
            va = "top"
        else:
            y = hi + 2.0
            va = "bottom"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{m:+.1f}%",
                ha="center", va=va, fontsize=9, fontweight="bold",
                color=col)

plot_bars(axL, frozen, "(a) Frozen direction (fit at base)")
plot_bars(axR, fresh,  "(b) Fresh direction (refit at round 4)")
axL.set_ylabel(r"$\Delta$ shortcut rate under ablation (relative, %)")
axR.set_ylim(-52, 24); axL.set_ylim(-52, 24)

fig.suptitle("Projecting out the direction at round 4: LR variants fail the necessity test; DoM does not",
             y=1.02, fontsize=10.5, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig4_ablation")
