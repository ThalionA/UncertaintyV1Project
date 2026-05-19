"""Figure 10 — RT distributions under the DDM mapping.

Prediction 7: drift rate sets mean RT, drift variability $s_v$ shifts mass
into the right tail. Framework predicts the canonical fat-tail-for-
ambiguous shape; the failure mode is an exponential RT distribution that
doesn't change with trial difficulty (i.e. RT is determined by something
other than evidence accumulation).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import framework_common as fc


def _ddm_rt(v, s, sv, B=1.0, dt=0.005, t_max=2.5, n_trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    rts = np.empty(n_trials)
    choices = np.empty(n_trials, dtype=int)
    for i in range(n_trials):
        v_i = v + sv * rng.standard_normal()
        x = 0.0
        t = 0.0
        while abs(x) < B and t < t_max:
            x += v_i * dt + s * np.sqrt(dt) * rng.standard_normal()
            t += dt
        rts[i] = t
        choices[i] = 1 if x >= 0 else -1
    return rts, choices


def _exp_rt(rate, n_trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / rate, n_trials)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Figure 10 — RT distributions: framework (DDM) vs failure (no difficulty effect)",
                 fontsize=11)

    # Framework: drift-rate-dependent RT distributions.
    ax = axes[0, 0]
    conditions = [
        ("easy", 3.0, 1.0, 0.3, "#2ca02c"),
        ("medium", 1.5, 1.0, 0.5, "#1f77b4"),
        ("ambiguous", 0.4, 1.0, 0.8, "#9467bd"),
    ]
    for label, v, s, sv, color in conditions:
        rts, _ = _ddm_rt(v, s, sv)
        ax.hist(rts, bins=70, density=True, alpha=0.55, color=color,
                label=f"{label}  v={v:.1f}, $s_v$={sv:.1f}")
    ax.set_xlabel("RT (s)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("Framework prediction (DDM)\n"
                 "low drift → slow RT + fat right tail",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 2.5)

    # Failure: RT distributions identical regardless of difficulty.
    ax = axes[0, 1]
    for label, _v, _s, _sv, color in conditions:
        rts = _exp_rt(rate=1.2, seed=hash(label) & 0xFFFF)
        ax.hist(rts, bins=70, density=True, alpha=0.55, color=color,
                label=f"{label}  (no difficulty dependence)")
    ax.set_xlabel("RT (s)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("Failure mode\n"
                 "RT independent of trial difficulty",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 2.5)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"{fc.OUTDIR}/10_rt_distributions.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 10 saved.")


if __name__ == "__main__":
    main()
