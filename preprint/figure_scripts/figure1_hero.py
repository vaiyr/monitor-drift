"""Figure 1 (hero): three panels.
(a) direction rotation + frozen-vs-fresh AUROC gap overlaid;
(b) geometric schematic cartoon of rotation vs. decision boundary;
(c) same direction, opposite verdicts (additive vs. ablation).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
import numpy as np
from _style import apply_style, savefig, BLUE, VERMILLION, GREEN, GREY, LIGHT

apply_style()

rounds = np.arange(6)
cos_lr = [1.0, 0.347, 0.361, 0.363, 0.380, 0.357]
cos_dom = [1.0, 0.825, 0.664, 0.767, 0.773, 0.762]
frozen_auroc_lr = [0.980, 0.923, 0.893, 0.908, 0.916, 0.889]
fresh_auroc_lr = [0.980, 0.973, 0.924, 1.000, 1.000, 1.000]

fig = plt.figure(figsize=(13.6, 3.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.35)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# ---------------- Panel A: rotation + AUROC gap ----------------
axA.axhline(1.0, color=LIGHT, lw=0.8, ls="--", zorder=0)
axA.fill_between(rounds, frozen_auroc_lr, fresh_auroc_lr,
                 color=BLUE, alpha=0.10, zorder=0,
                 label="fresh vs. frozen AUROC gap")
axA.plot(rounds, cos_lr, "-o", color=VERMILLION, lw=1.8, ms=5,
         label=r"cos rotation (det. LR)")
axA.plot(rounds, cos_dom, "-s", color=GREEN, lw=1.8, ms=5,
         label=r"cos rotation (diff.-of-means)")
axA.plot(rounds, frozen_auroc_lr, "-^", color=BLUE, lw=1.8, ms=5,
         label="frozen-probe AUROC")
axA.plot(rounds, fresh_auroc_lr, "--v", color=BLUE, lw=1.4, ms=5,
         alpha=0.75, label="fresh-probe AUROC")
axA.set_xticks(rounds)
axA.set_xticklabels(["base", "r0", "r1", "r2", "r3", "r4"])
axA.set_xlabel("SFT round")
axA.set_ylabel("cosine / AUROC")
axA.set_ylim(0.22, 1.08)
axA.set_title("(a) Direction rotates; fresh probe recovers, frozen does not")
# Compact legend below the plot, horizontal
axA.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55),
           frameon=False, fontsize=7.6, ncol=2, handlelength=1.8,
           columnspacing=1.2, labelspacing=0.35)
axA.annotate("70\u00b0 rotation\nin one step",
             xy=(1, 0.347), xytext=(2.1, 0.28),
             fontsize=8.5, color=VERMILLION,
             arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=0.9))

# ---------------- Panel B: geometric schematic ----------------
axB.set_xlim(-1.35, 1.35)
axB.set_ylim(-1.35, 1.35)
axB.set_aspect("equal")
axB.axis("off")
axB.set_title("(b) Why the dissociation happens")
rng = np.random.default_rng(0)
general_center = np.array([-0.18, -0.70])
shortcut_center = np.array([0.18,  0.70])
general = rng.normal(loc=general_center, scale=(0.22, 0.12), size=(90, 2))
shortcut = rng.normal(loc=shortcut_center, scale=(0.22, 0.12), size=(90, 2))
axB.add_patch(Ellipse(xy=general_center, width=0.78, height=0.42,
                     facecolor=GREEN, alpha=0.18, edgecolor=GREEN, lw=1.2))
axB.add_patch(Ellipse(xy=shortcut_center, width=0.78, height=0.42,
                     facecolor=VERMILLION, alpha=0.18,
                     edgecolor=VERMILLION, lw=1.2))
axB.scatter(general[:, 0], general[:, 1], s=12, color=GREEN, alpha=0.75,
            edgecolors="none")
axB.scatter(shortcut[:, 0], shortcut[:, 1], s=12, color=VERMILLION, alpha=0.75,
            edgecolors="none")
# Class labels placed to the right of the clusters, clear of both arrows
axB.text(0.75, 0.70, "shortcut", fontsize=9.5, color=VERMILLION,
         fontweight="bold", va="center", ha="left")
axB.text(-0.75, -0.70, "general", fontsize=9.5, color=GREEN,
         fontweight="bold", va="center", ha="right")
# New direction (d^{(k)}): aligned with class separation (nearly vertical)
new_vec = np.array([0.17, 1.00])
axB.add_patch(FancyArrowPatch((-new_vec[0], -new_vec[1]), tuple(new_vec),
                              arrowstyle="->", color=BLUE, lw=2.2,
                              mutation_scale=13))
axB.text(new_vec[0] + 0.10, new_vec[1] + 0.02, r"$d^{(k)}$",
         fontsize=11, color=BLUE, fontweight="bold")
# Old direction (d^{base}): rotated ~70 deg off the separation axis
theta = np.deg2rad(70)
c, s = np.cos(theta), np.sin(theta)
old_vec = np.array([[c, -s], [s, c]]) @ new_vec
axB.add_patch(FancyArrowPatch((-old_vec[0], -old_vec[1]), tuple(old_vec),
                              arrowstyle="->", color=GREY, lw=2.0,
                              mutation_scale=12, linestyle="--"))
axB.text(old_vec[0] - 0.30, old_vec[1] + 0.10, r"$d^{\mathrm{base}}$",
         fontsize=11, color=GREY, fontweight="bold")
# Rotation arc
ang_new = np.arctan2(new_vec[1], new_vec[0])
ang_old = np.arctan2(old_vec[1], old_vec[0])
arc = np.linspace(ang_new, ang_old, 30)
r = 0.38
axB.plot(r * np.cos(arc), r * np.sin(arc), color=VERMILLION, lw=1.0, ls=":")
axB.text(0.46, 0.25, r"$\approx 70^{\circ}$", fontsize=9, color=VERMILLION)
axB.text(0, -1.25,
         "the classes separate along "
         r"$d^{(k)}$; $d^{\mathrm{base}}$ misses the gap."
         "\n"
         r"a big push along $d^{\mathrm{base}}$ still crosses; ablating it removes nothing.",
         ha="center", va="top", fontsize=7.8, color=GREY)

# ---------------- Panel C: same direction, opposite verdicts ----------------
labels = ["additive steering\n(\u03b1 = 10)", "directional\nablation"]
means = [-17.1, -0.9]
ci_lo = [-25.8, -10.2]
ci_hi = [-7.4, 9.3]
err_lo = [m - lo for m, lo in zip(means, ci_lo)]
err_hi = [hi - m for m, hi in zip(means, ci_hi)]
colors = [VERMILLION, BLUE]
bars = axC.bar(labels, means, yerr=[err_lo, err_hi], color=colors,
               capsize=5, edgecolor="black", lw=0.6, error_kw={"lw": 1.0})
axC.axhline(0, color="black", lw=0.8, ls="--")
axC.set_ylabel(r"$\Delta$ shortcut rate (relative, %)")
axC.set_ylim(-34, 18)
axC.set_title("(c) Same direction. Opposite verdicts.")
# Place value labels OUTSIDE the CI whiskers so nothing overlaps.
label_positions = [(-25.8 - 1.5, "top"), (9.3 + 1.5, "bottom")]
for bar, m, (ypos, va) in zip(bars, means, label_positions):
    axC.text(bar.get_x() + bar.get_width() / 2, ypos, f"{m:+.1f}%",
             ha="center", va=va, fontsize=10, fontweight="bold",
             color=bar.get_facecolor())
axC.text(0.5, -32, "round 4, frozen CV-LR direction, $n=960$",
         ha="center", fontsize=8, color=GREY, transform=axC.transData)

fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.28, wspace=0.35)
savefig(fig, "fig1_hero")
