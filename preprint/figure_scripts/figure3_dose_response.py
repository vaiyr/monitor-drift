"""Figure 3: additive-steering dose-response at round 2 and round 4."""
import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style, savefig, BLUE, VERMILLION, GREY, LIGHT

apply_style()

alphas = np.array([2, 5, 7, 10])
r2_mean = np.array([-6.6, -12.6, -17.8, -24.5])
r2_lo = np.array([-15.1, -21.1, -26.0, -32.3])
r2_hi = np.array([3.0, -3.3, -8.9, -16.0])

r4_mean = np.array([-5.6, -9.8, -12.1, -17.1])
r4_lo = np.array([-14.7, -18.4, -21.2, -25.8])
r4_hi = np.array([4.7, -0.2, -2.0, -7.4])

# Random-direction control at alpha=10, round 4 (WI3): 3 seeds
rand_alphas = np.array([10, 10, 10])
rand_points = np.array([-5.8, 14.7, -12.0])

fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.axhline(0, color="black", lw=0.6, zorder=0)
ax.fill_between(alphas, r2_lo, r2_hi, color=BLUE, alpha=0.15)
ax.plot(alphas, r2_mean, "-o", color=BLUE, lw=1.8, ms=6, label="Round 2 (frozen CV-LR)")

ax.fill_between(alphas, r4_lo, r4_hi, color=VERMILLION, alpha=0.15)
ax.plot(alphas, r4_mean, "-s", color=VERMILLION, lw=1.8, ms=6, label="Round 4 (frozen CV-LR)")

# Random ctrl seeds: jitter x for visibility
jitter = np.array([-0.2, 0.0, 0.2])
ax.scatter(rand_alphas + jitter, rand_points, marker="x", color=GREY, s=50, lw=1.5,
           label="Random dir., $\\alpha{=}10$, round 4 (3 seeds)", zorder=5)

ax.set_xlabel(r"Steering coefficient $\alpha$")
ax.set_ylabel(r"$\Delta$ shortcut rate (relative, %)")
ax.set_title("Additive steering: monotonic dose-response at both checkpoints")
ax.set_xticks([2, 5, 7, 10])
ax.legend(loc="lower left", frameon=False)
ax.set_ylim(-36, 22)

fig.tight_layout()
savefig(fig, "fig3_dose_response")
