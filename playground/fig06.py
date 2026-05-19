"""
Figure 6 — Bayesian framing in two prototype-separation regimes.

Each row is one regime. Each row has:
  A. vMF likelihoods on the unit circle for L vs R prototypes
     (note the overlap in the realistic regime — that's the framework's
     bundle-width-matters regime).
  B. Posterior trajectory: accumulator X(t) = log P(R)/P(L), starting from
     three priors (L-biased / uniform / R-biased), bounded ±B.

The framework's prediction: in the easy regime, the vMF likelihoods barely
overlap and the posterior log-ratio swings to its bound regardless of
prior; in the realistic regime, the likelihoods overlap substantially
and the prior has a much larger impact on the final posterior — which is
the dissociable signature for testing where the prior physically lives.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import i0

import framework_common as fc
from framework_common import OUTDIR, T, RAMP, START, C_L, C_R


def _vmf_on_circle(ax, end_L, end_R, regime_label):
    mean_L = START + RAMP[-1] * (end_L - START)
    mean_R = START + RAMP[-1] * (end_R - START)
    mu_L = np.arctan2(mean_L[1], mean_L[0])
    mu_R = np.arctan2(mean_R[1], mean_R[0])
    kappa = 4.0
    th = np.linspace(-np.pi, np.pi, 1000)
    vmL = np.exp(kappa * np.cos(th - mu_L)) / (2 * np.pi * i0(kappa))
    vmR = np.exp(kappa * np.cos(th - mu_R)) / (2 * np.pi * i0(kappa))
    scale = 0.5 / max(vmL.max(), vmR.max())
    # Unit circle
    ax.plot(np.cos(th), np.sin(th), color="0.55", lw=0.7, ls="--")
    # vMF density as outward offset
    rL = 1.0 + scale * vmL
    rR = 1.0 + scale * vmR
    ax.fill(np.concatenate([np.cos(th), np.cos(th)[::-1]]),
            np.concatenate([np.sin(th), (rL * np.sin(th))[::-1]]),
            color=C_L, alpha=0.0)  # filler — actually plotting below
    # Plot the outer boundary lines for each density
    ax.plot(rL * np.cos(th), rL * np.sin(th), color=C_L, lw=1.8,
            label=f"$P(\\theta\\,|\\,L)$  κ={kappa:.0f}")
    ax.plot(rR * np.cos(th), rR * np.sin(th), color=C_R, lw=1.8,
            label=f"$P(\\theta\\,|\\,R)$  κ={kappa:.0f}")
    # Bundle means
    ax.scatter(np.cos(mu_L), np.sin(mu_L), c=C_L, s=120, marker="*",
               edgecolor="k", lw=0.8, zorder=5)
    ax.scatter(np.cos(mu_R), np.sin(mu_R), c=C_R, s=120, marker="*",
               edgecolor="k", lw=0.8, zorder=5)
    sep = np.degrees(np.arccos(np.cos(mu_L - mu_R)))
    ax.set_title(f"A · vMF likelihoods  —  {regime_label}\n"
                 f"angular separation ≈ {sep:.0f}°", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.legend(fontsize=7, loc="lower left")


def _accumulator_with_prior(ax, drift_per_step, regime_label):
    """Plot 3 accumulator traces with L-biased, uniform, R-biased starts."""
    rng = np.random.default_rng(42)
    n_steps = 200
    dt_acc = 0.01
    sigma = 0.5
    bound = 2.0
    priors = [
        (-0.6, "L-biased prior", "#c44d52"),
        (0.0, "uniform prior", "0.4"),
        (0.6, "R-biased prior", "#3b6e9c"),
    ]
    for X0, lbl, color in priors:
        X = np.empty(n_steps)
        X[0] = X0
        for i in range(1, n_steps):
            X[i] = X[i - 1] + drift_per_step + sigma * np.sqrt(dt_acc) * rng.standard_normal()
            # Clip at bound
            if abs(X[i]) >= bound:
                X[i:] = np.sign(X[i]) * bound
                break
        t = np.arange(n_steps) * dt_acc
        ax.plot(t, X, color=color, lw=1.6, label=lbl)
    ax.axhline(bound, color="k", lw=1.2)
    ax.axhline(-bound, color="k", lw=1.2)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlim(0, n_steps * dt_acc)
    ax.set_ylim(-bound - 0.3, bound + 0.3)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel(r"log posterior $\log P(R)/P(L)$", fontsize=8)
    ax.set_title(f"B · posterior accumulator  —  {regime_label}\n"
                 f"drift per step = {drift_per_step:.3f}",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)


fig = plt.figure(figsize=(12.5, 9.0))
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)

regimes = [
    (fc.END_L_EASY, fc.END_R_EASY, "EASY", 0.030),
    (fc.END_L, fc.END_R, "REALISTIC", 0.010),  # smaller drift in realistic
]
for row_idx, (end_L, end_R, label, drift) in enumerate(regimes):
    ax_A = fig.add_subplot(gs[row_idx, 0])
    _vmf_on_circle(ax_A, end_L, end_R, label)
    ax_B = fig.add_subplot(gs[row_idx, 1])
    _accumulator_with_prior(ax_B, drift_per_step=drift, regime_label=label)

fig.suptitle("Figure 6 — Bayesian framing in two prototype-separation regimes\n"
             "(realistic regime → vMF likelihoods overlap → prior has larger impact)",
             fontsize=11, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/06_bayesian_framing.png")
print("Figure 6 saved.")
plt.close()
