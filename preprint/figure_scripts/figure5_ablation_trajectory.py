"""Figure 5: frozen-direction ablation trajectory across rounds 0, 2, 4."""
import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style, savefig, GREEN, ORANGE, PURPLE

apply_style()

# Rel Δ (%) with 95% CIs, n=960 per condition, binomial-bootstrap.
# Rows: (round_label, CV-LR, det-LR, DoM) as (mean, lo, hi) tuples.
rounds = ["round 0", "round 2", "round 4"]
cv_lr = [(-0.23, -9.48, +10.05), (+0.90, -8.25, +11.25), (-0.92, -10.38, +9.64)]
det_lr = [(-3.86, -13.02, +6.12), (-6.78, -15.25, +2.25), (+2.42, -7.26, +13.37)]
dom = [(-0.97, -10.76, +10.13), (-10.53, -18.90, -1.38), (-7.18, -16.55, +3.04)]

fig, ax = plt.subplots(figsize=(6.8, 3.6))
x = np.arange(len(rounds))
dx = 0.22

for offset, rows, color, label in [
    (-dx, cv_lr, PURPLE, "CV-LR (frozen)"),
    (0.0, det_lr, ORANGE, "det.\\ LR (frozen)"),
    (+dx, dom, GREEN, "DoM (frozen)"),
]:
    means = np.array([r[0] for r in rows])
    los = means - np.array([r[1] for r in rows])
    his = np.array([r[2] for r in rows]) - means
    ax.errorbar(x + offset, means, yerr=[los, his], fmt="o", color=color,
                capsize=3, lw=1.1, label=label, markersize=6, elinewidth=1.0)

ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(rounds)
ax.set_ylabel(r"$\Delta$ shortcut rate under ablation (relative, %)")
ax.set_title("Frozen-direction ablation trajectory: CV-LR fails from round 0")
ax.legend(loc="lower left", frameon=False)
ax.set_ylim(-22, 16)

fig.tight_layout()
savefig(fig, "fig5_ablation_trajectory")
