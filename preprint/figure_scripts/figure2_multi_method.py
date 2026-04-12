"""Figure 2: method-sensitivity of the rotation trajectory.
Single panel — the frozen/fresh AUROC story is now in Fig 1(a)."""
import json, os
import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style, savefig, VERMILLION, GREEN, PURPLE, LIGHT

apply_style()

here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "..", "..", "results", "paper",
                                   "multi_method_directions.json")))
rounds = np.arange(6)
labels = ["base", "r0", "r1", "r2", "r3", "r4"]

fig, ax = plt.subplots(figsize=(6.6, 3.4))

cvlr_traj = [1.000, 0.778, 0.718, 0.367, 0.380, 0.399]
ax.axhline(1.0, color=LIGHT, lw=0.8, ls="--", zorder=0)
ax.plot(rounds, cvlr_traj, "-D", color=PURPLE, lw=1.7, ms=5,
        label="LogisticRegressionCV")
ax.plot(rounds, data["cos_vs_base"]["lr"], "-o", color=VERMILLION, lw=1.7, ms=5,
        label="Deterministic LR ($C{=}1$)")
ax.plot(rounds, data["cos_vs_base"]["dom"], "-s", color=GREEN, lw=1.7, ms=5,
        label="Difference-of-means")
ax.set_xticks(rounds); ax.set_xticklabels(labels)
ax.set_xlabel("SFT round")
ax.set_ylabel(r"cos$(d^{\mathrm{base}}, d^{(k)})$")
ax.set_ylim(0.25, 1.05)
ax.set_title("Three extraction methods, three rotation trajectories")
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
ax.annotate("CV picks a new $C$ at r2,\nlooks like a phase transition",
            xy=(2, 0.367), xytext=(3.1, 0.52),
            fontsize=8, color=PURPLE, ha="center",
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8,
                            connectionstyle="arc3,rad=-0.2"))

fig.tight_layout()
savefig(fig, "fig2_multi_method")
