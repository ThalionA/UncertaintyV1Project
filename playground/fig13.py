"""Figure 13 — κ over training (Prediction 12).

Bundle concentration κ should grow over training as the V1 → response
synaptic weights consolidate the average pattern. Framework predicts a
canonical sigmoidal-ish learning curve. Failure mode: no learning
trajectory (κ flat at its initial random value because plasticity isn't
happening, or is happening on something else).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import framework_common as fc


def main():
    rng = np.random.default_rng(31)
    n_trials = 1500
    trial_idx = np.arange(n_trials)

    # Framework: κ grows over training with a saturating learning curve.
    # Plus a few noisy "animals" to show the inter-animal variability.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Figure 13 — Bundle concentration κ over training",
                 fontsize=11)

    n_animals = 5
    ax = axes[0, 0]
    for i in range(n_animals):
        kappa_asymp = rng.uniform(20, 50)
        tau_anim = rng.uniform(250, 600)
        kappa_t = kappa_asymp * (1 - np.exp(-trial_idx / tau_anim))
        kappa_t += 1.5 * rng.standard_normal(n_trials)  # measurement noise
        ax.plot(trial_idx, kappa_t, alpha=0.7, lw=1.0,
                label=f"animal {i+1}  τ ≈ {tau_anim:.0f} trials")
    ax.set_xlabel("trial number", fontsize=9)
    ax.set_ylabel("recovered κ", fontsize=9)
    ax.set_title("Framework prediction: saturating learning curve",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)

    # Failure mode: no learning, κ flat at baseline noise.
    ax = axes[0, 1]
    for i in range(n_animals):
        kappa_flat = 3.0 + 1.5 * rng.standard_normal(n_trials)
        ax.plot(trial_idx, kappa_flat, alpha=0.7, lw=1.0,
                label=f"animal {i+1}  no learning")
    ax.set_xlabel("trial number", fontsize=9)
    ax.set_ylabel("recovered κ", fontsize=9)
    ax.set_title("Failure mode: no learning (κ flat at noise level)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 55)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"{fc.OUTDIR}/13_kappa_over_training.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 13 saved.")


if __name__ == "__main__":
    main()
