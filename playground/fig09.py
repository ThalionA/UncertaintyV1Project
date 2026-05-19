"""Figure 9 — Trial-mean SI vs confidence proxy.

Prediction 6: |SI| should be monotonically related to confidence,
measured via decision-posterior entropy (low entropy = confident) or
pre-RZ velocity (extremes = confident; middle = uncertain).

Left panel: framework prediction — high |SI| trials cluster at the
confident end of the proxy. Right panel: failure mode — the relationship
is decoupled (no correlation between SI and confidence).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import framework_common as fc


def _entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    rng = np.random.default_rng(2026)
    n = 300

    # Trial drift rates from a wide range — easy vs ambiguous trials.
    drifts = rng.uniform(-3.0, 3.0, n)
    # Decision posterior P(Go) generated from a logistic of drift + noise.
    p_go = 1.0 / (1.0 + np.exp(-1.2 * drifts + 0.3 * rng.standard_normal(n)))
    entropy = _entropy(p_go)

    # Framework: SI tracks drift with noise; confidence tracks entropy.
    si = np.tanh(drifts / 2.0) + 0.10 * rng.standard_normal(n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Figure 9 — Trial-mean SI vs IO decision-posterior entropy",
                 fontsize=11)

    # Framework prediction: clean negative relationship between |SI| and H.
    ax = axes[0, 0]
    ax.scatter(si, entropy, s=15, c="steelblue", alpha=0.55, edgecolors="none")
    r, p = pearsonr(np.abs(si), entropy)
    ax.set_title(f"Framework prediction:  $|SI|$ negatively related to $H$\n"
                 f"Pearson r(|SI|, H) = {r:.2f}, p = {p:.1g}", fontsize=10)
    ax.set_xlabel("trial-mean SI", fontsize=9)
    ax.set_ylabel("IO decision-posterior entropy $H$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.axvline(0, color="gray", lw=0.5, ls=":", alpha=0.4)

    # Failure mode: SI shuffled relative to entropy.
    si_shuffled = rng.permutation(si)
    ax = axes[0, 1]
    ax.scatter(si_shuffled, entropy, s=15, c="darkgray", alpha=0.55,
               edgecolors="none")
    r_null, p_null = pearsonr(np.abs(si_shuffled), entropy)
    ax.set_title(f"Failure mode (SI ↔ confidence decoupled)\n"
                 f"Pearson r(|SI|, H) = {r_null:.2f}, p = {p_null:.2f}",
                 fontsize=10)
    ax.set_xlabel("trial-mean SI (shuffled)", fontsize=9)
    ax.set_ylabel("IO decision-posterior entropy $H$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.axvline(0, color="gray", lw=0.5, ls=":", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{fc.OUTDIR}/09_si_vs_confidence.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 9 saved.")


if __name__ == "__main__":
    main()
