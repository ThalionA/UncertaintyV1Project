"""
Figure 3 — Pointwise SI(t) traces by difficulty, in two prototype-separation regimes.

Each row is one regime. Each row has three SI(t) panels (easy-L / ambig /
easy-R) plus a joint (drift, diffusion) signature scatter.

The framework's prediction: SI(t) magnitude grades with trial difficulty,
ambiguous trials have low drift and high diffusion. *Crucially*, SI
magnitudes are large in the easy regime (well-separated prototypes give
high |cosine| differences) and small in the realistic regime (close
prototypes mean small cosine differences). This is the empirical signature
of the difficulty × framework-distinctiveness interaction (Prediction 14).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import framework_common as fc
from framework_common import (
    OUTDIR, T, N, RAMP, START, ou_traj,
    C_L, C_R, C_AMB, si_normalized,
)


N_TRIALS = 40


def _build_regime(end_L, end_R, seed_offset=0):
    """Build mean trajectories + SI batches for L/R/ambig trials."""
    mean_L = START + RAMP[:, None] * (end_L - START)
    mean_R = START + RAMP[:, None] * (end_R - START)
    end_amb = 0.5 * (end_L + end_R)
    mean_amb = START + RAMP[:, None] * (end_amb - START)

    def _trials(mean_traj, sigma, tau, seed_base):
        return np.stack([mean_traj + ou_traj(N, sigma=sigma, tau=tau,
                                             seed=seed_offset + seed_base + k)
                         for k in range(N_TRIALS)])

    trials_L = _trials(mean_L, 0.20, 0.10, 1000)
    trials_R = _trials(mean_R, 0.20, 0.10, 2000)
    trials_amb = _trials(mean_amb, 0.55, 0.08, 3000)

    SI_L = np.stack([si_normalized(trials_L[k], mean_L, mean_R)
                     for k in range(N_TRIALS)])
    SI_R = np.stack([si_normalized(trials_R[k], mean_L, mean_R)
                     for k in range(N_TRIALS)])
    SI_amb = np.stack([si_normalized(trials_amb[k], mean_L, mean_R)
                       for k in range(N_TRIALS)])
    return SI_L, SI_R, SI_amb


def _draw_row(fig, gs_row, SI_L, SI_R, SI_amb, regime_label):
    panels = [
        (gs_row[0], SI_L, C_L, "easy L"),
        (gs_row[1], SI_amb, C_AMB, "ambig"),
        (gs_row[2], SI_R, C_R, "easy R"),
    ]
    for cell, SI, color, lbl in panels:
        ax = fig.add_subplot(cell)
        for k in range(min(20, SI.shape[0])):
            ax.plot(T, SI[k], color=color, alpha=0.18, lw=0.6)
        mu = SI.mean(axis=0); sd = SI.std(axis=0)
        ax.plot(T, mu, color=color, lw=2.0, zorder=5)
        ax.fill_between(T, mu - sd, mu + sd, color=color, alpha=0.22)
        ax.axhline(0, color="k", lw=0.4, ls=":", alpha=0.5)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("SI(t)", fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{regime_label} · {lbl}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)

    # Joint signature panel
    ax = fig.add_subplot(gs_row[3])
    for SI, color, lbl in [(SI_L, C_L, "easy L"),
                           (SI_amb, C_AMB, "ambig"),
                           (SI_R, C_R, "easy R")]:
        drifts = SI.mean(axis=1)
        diffs = SI.std(axis=1)
        ax.scatter(drifts, diffs, c=color, alpha=0.6, s=20,
                   edgecolor="white", lw=0.5, label=lbl)
    ax.axvline(0, color="k", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel(r"drift $\mathbb{E}_t[SI]$", fontsize=8)
    ax.set_ylabel(r"diffusion $\sqrt{\mathrm{Var}_t[SI]}$", fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"{regime_label} · joint signature", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.18)


fig = plt.figure(figsize=(16, 9.0))
gs = GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.35,
              width_ratios=[1, 1, 1, 1])

regimes = [
    (fc.END_L_EASY, fc.END_R_EASY, 0, "EASY"),
    (fc.END_L, fc.END_R, 9999, "REALISTIC"),
]
for row_idx, (end_L, end_R, seed_off, label) in enumerate(regimes):
    SI_L, SI_R, SI_amb = _build_regime(end_L, end_R, seed_offset=seed_off)
    _draw_row(fig, [gs[row_idx, j] for j in range(4)],
              SI_L, SI_R, SI_amb, label)

fig.suptitle("Figure 3 — SI(t) by difficulty, in two prototype-separation regimes\n"
             "(realistic regime → smaller SI magnitudes → framework-distinctive "
             "predictions become diagnostic)",
             fontsize=11, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/03_SI_traces_by_difficulty.png")
print("Figure 3 saved.")
plt.close()
