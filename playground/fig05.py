"""
Figure 5 — Asymmetric strategy: differential drift weighting shifts the boundary.

Each row is one prototype-separation regime (EASY top, REALISTIC bottom).
Within each row, three strategy variants: balanced ($w_L = w_R$),
asymmetric ($w_R > w_L$), and single-archetype ($w_L = 0$, threshold-only).

The framework's prediction: in the easy regime, the boundary tilt induced
by asymmetric weighting still yields confident decisions. In the realistic
regime, the same tilt has much larger behavioural impact because
discrimination is already marginal — Go-bias / asymmetric-lapse effects
in Go/NoGo are predicted to be most pronounced on hard discriminations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

import framework_common as fc
from framework_common import (
    OUTDIR, K, START, RAMP, ou_traj,
    C_L, C_R, C_L_LIGHT, C_R_LIGHT,
)


def _build_means_and_bundles(end_L, end_R, seed_offset=0):
    mean_L = START + RAMP[:, None] * (end_L - START)
    mean_R = START + RAMP[:, None] * (end_R - START)
    trajs_L = np.stack([mean_L + ou_traj(fc.N, seed=seed_offset + 100 + k)
                        for k in range(K)])
    trajs_R = np.stack([mean_R + ou_traj(fc.N, seed=seed_offset + 200 + k)
                        for k in range(K)])
    return mean_L, mean_R, trajs_L, trajs_R


def _draw_strategy_panel(ax, mode, mean_L, mean_R, trajs_L, trajs_R,
                         theta_L_ref, theta_R_ref, title):
    xs = np.linspace(-4.0, 4.0, 250)
    ys = np.linspace(-2.5, 3.0, 200)
    XX, YY = np.meshgrid(xs, ys)
    THETA = np.arctan2(YY, XX)
    s_L = 0.5 * (1 + np.cos(THETA - theta_L_ref))
    s_R = 0.5 * (1 + np.cos(THETA - theta_R_ref))
    if mode == "balanced":
        D = 0.5 * s_R - 0.5 * s_L
    elif mode == "asym":
        D = 0.8 * s_R - 0.2 * s_L
    else:  # single
        D = s_R - 0.5
    ax.contourf(XX, YY, D, levels=[-10, 0, 10],
                colors=[C_L_LIGHT, C_R_LIGHT], alpha=0.5)

    # Bundle context
    for k in range(K):
        ax.plot(trajs_L[k, :, 0], trajs_L[k, :, 1],
                color=C_L, alpha=0.15, lw=0.5)
        ax.plot(trajs_R[k, :, 0], trajs_R[k, :, 1],
                color=C_R, alpha=0.15, lw=0.5)
    ax.plot(mean_L[:, 0], mean_L[:, 1], color=C_L, lw=2.0,
            alpha=0.35 if mode == "single" else 0.95)
    ax.plot(mean_R[:, 0], mean_R[:, 1], color=C_R, lw=2.0)
    ax.scatter(*mean_L[-1], c=C_L, s=130, marker="*",
               edgecolor="k", lw=0.9,
               alpha=0.4 if mode == "single" else 1.0)
    ax.scatter(*mean_R[-1], c=C_R, s=130, marker="*",
               edgecolor="k", lw=0.9)
    ax.scatter(0, 0, c="k", s=35, marker="x", lw=2)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 3.0)
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.18)


fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)

regimes = [
    (fc.END_L_EASY, fc.END_R_EASY, 0, "EASY"),
    (fc.END_L, fc.END_R, 9000, "REALISTIC"),
]
strategies = [
    ("balanced", "balanced  ($w_L = w_R$)"),
    ("asym", "asymmetric  ($w_R = 0.8$)"),
    ("single", "single archetype"),
]
for row_idx, (end_L, end_R, seed_off, regime_lbl) in enumerate(regimes):
    mean_L, mean_R, trajs_L, trajs_R = _build_means_and_bundles(
        end_L, end_R, seed_offset=seed_off)
    theta_L_ref = np.arctan2(mean_L[-1, 1], mean_L[-1, 0])
    theta_R_ref = np.arctan2(mean_R[-1, 1], mean_R[-1, 0])
    for col_idx, (mode, mode_lbl) in enumerate(strategies):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        _draw_strategy_panel(ax, mode, mean_L, mean_R, trajs_L, trajs_R,
                             theta_L_ref, theta_R_ref,
                             f"{regime_lbl} · {mode_lbl}")

fig.suptitle("Figure 5 — Asymmetric strategies × prototype-separation regime",
             fontsize=11, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/05_asymmetric_strategy.png")
print("Figure 5 saved.")
plt.close()
