"""Figure 11 — Cross-modal DDM consistency.

Prediction 8: DDM parameters {v, s, s_v} estimated from neural SI(t)
statistics should agree with those fit to behavioural choice + RT data.
Framework predicts diagonal alignment in the (behavioural-fit,
neural-fit) plane. Failure mode: no agreement — the neural SI doesn't
carry the same DDM signal as behaviour.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import framework_common as fc


def main():
    rng = np.random.default_rng(31)
    n_animals = 12

    # Generate synthetic ground-truth parameters per animal.
    v_true = rng.uniform(0.3, 3.0, n_animals)
    s_true = rng.uniform(0.5, 1.5, n_animals)
    sv_true = rng.uniform(0.1, 1.0, n_animals)

    # Framework: behavioural and neural recovery both noisy estimates of truth.
    v_behav = v_true + 0.20 * rng.standard_normal(n_animals)
    v_neural = v_true + 0.20 * rng.standard_normal(n_animals)
    s_behav = s_true + 0.10 * rng.standard_normal(n_animals)
    s_neural = s_true + 0.10 * rng.standard_normal(n_animals)
    sv_behav = sv_true + 0.10 * rng.standard_normal(n_animals)
    sv_neural = sv_true + 0.10 * rng.standard_normal(n_animals)

    # Failure: neural fit is unrelated to ground truth — random.
    v_neural_null = rng.uniform(0.3, 3.0, n_animals)
    s_neural_null = rng.uniform(0.5, 1.5, n_animals)
    sv_neural_null = rng.uniform(0.1, 1.0, n_animals)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), squeeze=False)
    fig.suptitle("Figure 11 — Cross-modal DDM recovery (behavioural vs neural fits)",
                 fontsize=11, y=0.995)

    def _scatter(ax, x, y, label):
        ax.scatter(x, y, s=70, c="steelblue", edgecolors="black", zorder=3)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=0.6, alpha=0.5, label="identity")
        r, p = pearsonr(x, y)
        ax.set_title(f"{label}\nPearson r = {r:.2f}, p = {p:.3f}",
                     fontsize=9)
        ax.set_xlabel(f"{label} from behaviour", fontsize=8)
        ax.set_ylabel(f"{label} from neural SI", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    # Framework: top row
    _scatter(axes[0, 0], v_behav, v_neural, "$v$ (drift)")
    _scatter(axes[0, 1], s_behav, s_neural, "$s$ (diffusion)")
    _scatter(axes[0, 2], sv_behav, sv_neural, "$s_v$ (drift variability)")
    # Failure: bottom row
    _scatter(axes[1, 0], v_behav, v_neural_null, "$v$ (failure)")
    _scatter(axes[1, 1], s_behav, s_neural_null, "$s$ (failure)")
    _scatter(axes[1, 2], sv_behav, sv_neural_null, "$s_v$ (failure)")

    # Row labels
    axes[0, 0].text(-0.25, 0.5, "framework\nprediction",
                    transform=axes[0, 0].transAxes, fontsize=11,
                    rotation=90, va="center", ha="center", fontweight="bold")
    axes[1, 0].text(-0.25, 0.5, "failure mode\n(neural ≠ behaviour)",
                    transform=axes[1, 0].transAxes, fontsize=11,
                    rotation=90, va="center", ha="center", fontweight="bold")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06,
                        wspace=0.35, hspace=0.45)
    out = f"{fc.OUTDIR}/11_cross_modal_ddm.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 11 saved.")


if __name__ == "__main__":
    main()
