"""IO-HMM posteriors broken down by latent state — all mice, native support.

Per mouse: state occupancy (hard and gamma-weighted), how sharply each state
posterior is concentrated, and the state-conditional condition-mean posteriors,
so a mouse whose MARGINAL looks unlike the others can be read as a mixture of
states rather than a different model. Every width statistic is on the native
72-bin circular support (doubled-angle resultant R, dimensionless), never folded.

Usage: python diagnostics/io_hmm_posteriors_by_state.py [--pkl PATH]
Figures -> figures/io_hmm_by_state/  (png+svg via figsave.save_fig)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from figsave import save_fig            # noqa: E402
import io_hmm_data                      # noqa: E402
from utils import load_vr_export        # noqa: E402

REPO = _HERE.parent.parent
GRID = io_hmm_data.GRID_DEG_IO          # 72 x 2.5 deg, [0,180)
OUT = _HERE.parent / "figures" / "io_hmm_by_state"


def resultant(p):
    """Doubled-angle resultant length, per row of (n,72). 0 = uniform, 1 = delta."""
    a = np.deg2rad(2 * GRID)
    return np.hypot((p * np.cos(a)).sum(1), (p * np.sin(a)).sum(1))


def load(pkl):
    mice = io_hmm_data.load_io_hmm_pkl(str(pkl), allow_partial=False)
    out = {}
    for m in sorted(mice):
        e = mice[m]; K = int(e["n_states"])
        by = np.asarray(io_hmm_data._get_posterior_field(e, "PS_stim_G_tr_by_state"), float)  # (K,72,n)
        by = np.transpose(by, (0, 2, 1))                                                     # (K,n,72)
        by /= by.sum(2, keepdims=True)
        marg = np.asarray(io_hmm_data._get_posterior_field(e, "PS_stim_G_tr"), float).T
        marg /= marg.sum(1, keepdims=True)
        gam = np.asarray(io_hmm_data._get_posterior_field(e, "gamma"), float)
        d = e["data"]
        _, _, _, _, trials = load_vr_export(m)
        idx, _ = io_hmm_data.align_trials_to_export(d, trials, mouse_id=m)
        out[m] = dict(K=K, by=by[:, idx], marg=marg[idx], gamma=gam[idx],
                      hard=gam[idx].argmax(1),
                      ori=np.asarray(trials["orientation"]).ravel().astype(float),
                      ct=np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3),
                      disp=np.asarray(trials["dispersion"]).ravel().astype(float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(REPO / "data" / "fitted_data_and_posteriors_hmm.pkl"))
    a = ap.parse_args()
    D = load(a.pkl); OUT.mkdir(parents=True, exist_ok=True)
    mice = sorted(D); Kmax = max(D[m]["K"] for m in mice)
    cols = plt.cm.tab10(np.arange(Kmax))

    # ---- fig 1: occupancy + per-state concentration + marginal concentration ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    for i, m in enumerate(mice):
        g = D[m]["gamma"]; occ = g.mean(0)
        bottom = 0
        for z in range(D[m]["K"]):
            ax.bar(i, occ[z], bottom=bottom, color=cols[z], edgecolor="k", lw=0.5)
            bottom += occ[z]
    ax.set_xticks(range(len(mice))); ax.set_xticklabels([f"m{m}\nK={D[m]['K']}" for m in mice])
    ax.set_ylabel("gamma-weighted occupancy"); ax.set_title("state occupancy (mean posterior prob)", fontsize=9)
    ax = axes[1]
    for i, m in enumerate(mice):
        for z in range(D[m]["K"]):
            R = resultant(D[m]["by"][z]); w = D[m]["gamma"][:, z]
            ax.errorbar(i + (z - (D[m]["K"] - 1) / 2) * 0.22, np.average(R, weights=w),
                        yerr=np.sqrt(np.average((R - np.average(R, weights=w)) ** 2, weights=w)),
                        fmt="o", color=cols[z], ms=6, capsize=2)
        ax.plot(i, resultant(D[m]["marg"]).mean(), "k_", ms=14, mew=2)
    ax.set_xticks(range(len(mice))); ax.set_xticklabels([f"m{m}" for m in mice])
    ax.set_ylabel("resultant R (0 uniform, 1 delta)")
    ax.set_title("concentration of P(stim | state) per state (colour)\nblack tick = marginal", fontsize=9)
    ax = axes[2]
    for i, m in enumerate(mice):
        R = resultant(D[m]["marg"]); ct = D[m]["ct"]
        for j, c in enumerate(sorted(np.unique(ct))):
            ax.plot(i + (j - 1.5) * 0.18, R[ct == c].mean(), "s", color=plt.cm.magma(0.15 + 0.25 * j), ms=6)
    ax.set_xticks(range(len(mice))); ax.set_xticklabels([f"m{m}" for m in mice])
    ax.set_ylabel("resultant R of the MARGINAL"); ax.set_title("marginal concentration by contrast\n(0.01, 0.25, 0.5, 1 light->dark)", fontsize=9)
    for ax in axes: ax.set_ylim(bottom=0)
    fig.suptitle("IO-HMM latent states — occupancy and how sharp each state's stimulus posterior is (native 72-bin support)", fontsize=10)
    fig.tight_layout(); save_fig(fig, OUT, "fig1_state_occupancy_concentration")

    # ---- fig 2: state-conditional condition means, one row per mouse, contrast 1 & 0.01 ----
    for c_show in (1.0, 0.01):
        fig, axes = plt.subplots(len(mice), 4, figsize=(13, 2.1 * len(mice)), sharex=True, sharey="row")
        oris = [0.0, 45.0, 60.0, 90.0]
        for r, m in enumerate(mice):
            for k, o in enumerate(oris):
                ax = axes[r, k]; msk = (D[m]["ct"] == c_show) & (D[m]["ori"] == o)
                if msk.sum() == 0: ax.set_axis_off(); continue
                for z in range(D[m]["K"]):
                    w = D[m]["gamma"][msk, z]
                    if w.sum() < 1e-6: continue
                    ax.plot(GRID, np.average(D[m]["by"][z][msk], axis=0, weights=w) / 2.5,
                            color=cols[z], lw=1.3, label=f"state {z} (Σγ={w.sum():.0f})")
                ax.plot(GRID, D[m]["marg"][msk].mean(0) / 2.5, "k--", lw=1.0, label="marginal")
                ax.axvline(o, color="0.5", lw=0.7, ls=":"); ax.set_xlim(0, 180)
                ax.set_xticks([0, 45, 90, 135, 180]); ax.tick_params(labelsize=6)
                if r == 0: ax.set_title(f"ori {o:.0f}", fontsize=8)
                if k == 0: ax.set_ylabel(f"m{m} K={D[m]['K']}\nprob/deg", fontsize=8)
                if r == 0 and k == 0: ax.legend(fontsize=5.5, frameon=False)
                if r == len(mice) - 1: ax.set_xlabel("orientation (deg)", fontsize=8)
        fig.suptitle(f"State-conditional condition-mean posteriors P(stim | state), contrast {c_show:g} — "
                     "gamma-weighted within condition; dashed = marginal", fontsize=10)
        fig.tight_layout(); save_fig(fig, OUT, f"fig2_state_condmeans_ctr{c_show:g}".replace(".", "p"))

    # ---- stdout summary ----
    print(f"{'mouse':6s} {'K':>2s} {'occupancy (hard)':22s} {'R per state':28s} {'R marginal':>10s}")
    for m in mice:
        occ = np.bincount(D[m]["hard"], minlength=D[m]["K"])
        Rz = [np.average(resultant(D[m]["by"][z]), weights=D[m]["gamma"][:, z]) for z in range(D[m]["K"])]
        print(f"m{m:<5d} {D[m]['K']:>2d} {str(occ.tolist()):22s} {str([round(x,3) for x in Rz]):28s} {resultant(D[m]['marg']).mean():10.3f}")


if __name__ == "__main__":
    main()
