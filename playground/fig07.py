"""Figure 7 — Bundle width sweep: variants diverge as bundles get broader.

Tests Prediction 2 from the empirical-status table: the four SI variants
(prototype / expected exemplar / max exemplar / vMF) converge to identical
output on tight bundles and diverge as bundle width grows. The framework's
prediction: divergence is real and grows with bundle width. The failure
mode: variants stay identical regardless of bundle width (i.e. the bundle
representation is empirically vacuous).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ive

import framework_common as fc


def _kappa_banerjee(R_bar, p):
    R = np.clip(np.asarray(R_bar, dtype=float), 0.0, 1.0 - 1e-6)
    return R * (p - R ** 2) / (1.0 - R ** 2)


def _log_C_p(kappa, p):
    nu = p / 2.0 - 1.0
    k = np.clip(np.asarray(kappa, dtype=float), 1e-12, None)
    log_I = np.log(ive(nu, k) + 1e-300) + k
    return nu * np.log(k) - (p / 2.0) * np.log(2.0 * np.pi) - log_I


def _si_norm(sR, sL, eps=1e-9):
    sR_p = (1.0 + sR) / 2.0
    sL_p = (1.0 + sL) / 2.0
    return (sR_p - sL_p) / (sR_p + sL_p + eps)


def _gen_bundles(bundle_spread, n_per_side=14, seed=0):
    """Two trajectory bundles with controllable angular spread."""
    rng = np.random.default_rng(seed)
    bundles = {}
    for side, end in [("L", fc.END_L), ("R", fc.END_R)]:
        mean = fc.START + fc.RAMP[:, None] * (end - fc.START)
        b = np.empty((n_per_side, fc.N, 2))
        for k in range(n_per_side):
            angular_offset = bundle_spread * rng.standard_normal()
            cos_t, sin_t = np.cos(angular_offset), np.sin(angular_offset)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            b[k] = (mean - fc.START) @ rot.T + fc.START
            b[k] += fc.ou_traj(fc.N, sigma=0.05, seed=seed * 1000 + k)
        bundles[side] = b
    return bundles


def _si_variants(r_t, bundles):
    """Return dict {variant_name: SI(t)} for the four methods."""
    out = {}
    # Prototype
    mean_L = bundles["L"].mean(axis=0)
    mean_R = bundles["R"].mean(axis=0)
    sL = fc.cosine_sim(r_t, mean_L)
    sR = fc.cosine_sim(r_t, mean_R)
    out["prototype"] = _si_norm(sR, sL)
    # Expected exemplar
    sL_k = np.stack([fc.cosine_sim(r_t, bundles["L"][k])
                     for k in range(bundles["L"].shape[0])])
    sR_k = np.stack([fc.cosine_sim(r_t, bundles["R"][k])
                     for k in range(bundles["R"].shape[0])])
    out["expected_exemplar"] = _si_norm(sR_k.mean(axis=0), sL_k.mean(axis=0))
    out["max_exemplar"] = _si_norm(sR_k.max(axis=0), sL_k.max(axis=0))
    # vMF
    p_dim = 2
    K = bundles["L"].shape[0]
    log_p = {}
    for side in ("L", "R"):
        Ek_hat = bundles[side] / (np.linalg.norm(bundles[side], axis=2,
                                                 keepdims=True) + 1e-9)
        resultant = Ek_hat.sum(axis=0)
        R_len = np.linalg.norm(resultant, axis=1)
        R_bar = R_len / K
        mu = resultant / (R_len[:, None] + 1e-9)
        kappa = _kappa_banerjee(R_bar, p=p_dim)
        r_hat = r_t / (np.linalg.norm(r_t, axis=1, keepdims=True) + 1e-9)
        dot = np.einsum("ij,ij->i", r_hat, mu)
        log_p[side] = kappa * dot + _log_C_p(kappa, p=p_dim)
    out["vmf"] = np.tanh(0.5 * (log_p["R"] - log_p["L"]))
    return out


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    fig.suptitle("Figure 7 — Bundle-width sweep: variants converge on tight bundles, "
                 "diverge on broad",
                 fontsize=11, y=0.99)
    spreads = [0.00, 0.05, 0.15, 0.35]
    method_colors = {
        "prototype": "#1f77b4",
        "expected_exemplar": "#2ca02c",
        "max_exemplar": "#ff7f0e",
        "vmf": "#d62728",
    }

    # ----- top row: framework prediction
    for ax, spread in zip(axes[0], spreads):
        bundles = _gen_bundles(spread, seed=42)
        # Use an "easy R" trial as r(t).
        rng = np.random.default_rng(7)
        r_t = fc.MEAN_R + 0.06 * rng.standard_normal((fc.N, 2))
        si = _si_variants(r_t, bundles)
        for method, trace in si.items():
            ax.plot(fc.T, trace, color=method_colors[method], lw=1.5,
                    label=method)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlabel("time within trial (s)", fontsize=8)
        ax.set_ylabel("SI(t)", fontsize=8)
        ax.set_title(f"bundle angular spread = {spread:.2f} rad", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-1.05, 1.05)

    # ----- bottom row: failure-mode (bundle representation is vacuous —
    # every variant gives the same SI regardless of spread, because the
    # bundle dimension is being ignored)
    for ax, spread in zip(axes[1], spreads):
        bundles = _gen_bundles(spread, seed=42)
        rng = np.random.default_rng(7)
        r_t = fc.MEAN_R + 0.06 * rng.standard_normal((fc.N, 2))
        # In the failure mode, every method reduces to the prototype answer.
        proto_si = _si_variants(r_t, _gen_bundles(0.0, seed=42))["prototype"]
        for method in method_colors:
            ax.plot(fc.T, proto_si, color=method_colors[method], lw=1.5,
                    alpha=0.8)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlabel("time within trial (s)", fontsize=8)
        ax.set_ylabel("SI(t)", fontsize=8)
        ax.set_title(f"failure-mode at spread = {spread:.2f}\n(variants collapse anyway)",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-1.05, 1.05)

    # Row labels
    axes[0, 0].text(-0.30, 0.5, "framework\nprediction",
                    transform=axes[0, 0].transAxes, fontsize=11,
                    rotation=90, va="center", ha="center", fontweight="bold")
    axes[1, 0].text(-0.30, 0.5, "failure mode\n(no bundle effect)",
                    transform=axes[1, 0].transAxes, fontsize=11,
                    rotation=90, va="center", ha="center", fontweight="bold")

    # Legend
    handles = [plt.Line2D([0], [0], color=c, lw=1.5, label=m)
               for m, c in method_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.10,
                        wspace=0.30, hspace=0.50)
    out = f"{fc.OUTDIR}/07_bundle_width_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 7 saved.")


if __name__ == "__main__":
    main()
