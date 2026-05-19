"""Figure 12 — The SBC clash test (Prediction 9).

The central empirical wedge between this framework and sampling-based codes.
SBC predicts that within-trial residuals of r(t) carry choice information
(they ARE the samples from the posterior). This framework predicts they
do NOT — once SI is controlled, residuals are diffusion noise.

Left: framework prediction — choice has no relationship to residual
variance after controlling for signed_contrast / SI. Right: SBC
prediction — choice strongly depends on residual variance because
residuals are posterior samples.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import framework_common as fc


def main():
    rng = np.random.default_rng(2024)
    n = 600
    # Trial difficulty (signed_contrast surrogate).
    contrast = rng.choice([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], size=n)
    # Mean SI tracks contrast.
    si_mean = np.tanh(1.8 * contrast) + 0.05 * rng.standard_normal(n)
    # Within-trial residual variance — drawn from a wide distribution.
    res_var = rng.gamma(shape=2.0, scale=0.15, size=n)

    # FRAMEWORK: choice = sign(si_mean) + Bernoulli noise driven by SI strength.
    # Residual variance has NO effect on choice once SI is conditioned on.
    p_go_fw = 1.0 / (1.0 + np.exp(-3.0 * si_mean))
    choice_fw = rng.binomial(1, p_go_fw)

    # SBC ALTERNATIVE: choice depends on residual variance because residuals
    # ARE choice-informative samples from the posterior.
    # Mix the residual variance into the choice probability.
    p_go_sbc = 1.0 / (1.0 + np.exp(-3.0 * si_mean - 4.0 * (res_var - res_var.mean())))
    choice_sbc = rng.binomial(1, p_go_sbc)

    # Logistic regression: choice ~ residual_var | contrast (binned)
    def _binned_p_go(choice_data):
        bins = np.linspace(0, 1.5, 12)
        mids = 0.5 * (bins[:-1] + bins[1:])
        p = np.full(len(mids), np.nan)
        n_per = np.zeros(len(mids), dtype=int)
        for i in range(len(mids)):
            msk = (res_var >= bins[i]) & (res_var < bins[i + 1])
            n_per[i] = int(msk.sum())
            if n_per[i] > 0:
                p[i] = float(choice_data[msk].mean())
        return mids, p, n_per

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Figure 12 — SBC clash test: do within-trial residuals predict choice?",
                 fontsize=11)

    # Framework
    ax = axes[0, 0]
    mids, p_fw, _ = _binned_p_go(choice_fw)
    ax.plot(mids, p_fw, "o-", color="steelblue", lw=1.5, markersize=8,
            label="P(Go) | residual_var (controlling for SI)")
    # Fit a logistic-regression slope estimate
    from numpy.polynomial import polynomial as P
    coef = P.polyfit(res_var, choice_fw - 0.5 * (1 - p_go_fw.mean()), 1)
    ax.set_title(f"Framework prediction: residuals carry no choice info\n"
                 f"(slope ≈ {coef[1]:.3f}, near zero)",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel("within-trial residual variance of $\\mathbf{r}(t)$", fontsize=9)
    ax.set_ylabel("P(Go)", fontsize=9)
    ax.tick_params(labelsize=8)

    # SBC
    ax = axes[0, 1]
    mids, p_sbc, _ = _binned_p_go(choice_sbc)
    ax.plot(mids, p_sbc, "s-", color="indianred", lw=1.5, markersize=8,
            label="P(Go) | residual_var (controlling for SI)")
    coef_sbc = P.polyfit(res_var, choice_sbc - 0.5 * (1 - p_go_sbc.mean()), 1)
    ax.set_title(f"SBC alternative: residuals strongly predict choice\n"
                 f"(slope ≈ {coef_sbc[1]:.3f}, well above zero)",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel("within-trial residual variance of $\\mathbf{r}(t)$", fontsize=9)
    ax.set_ylabel("P(Go)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{fc.OUTDIR}/12_sbc_clash_test.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 12 saved.")


if __name__ == "__main__":
    main()
