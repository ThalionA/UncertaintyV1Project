# -*- coding: utf-8 -*-
"""Sharpen/broaden and location tests for the five losses, on the REAL
perceptual (IO-target) posteriors — not synthetic Gaussian bumps.

For every real IO perceptual posterior P (the decoder target, `Dist['spat']
['target']` in `loss_comparison_v1`, 6 mice) we ask how each production loss
responds when P is perturbed, and what each loss recovers when fit to P:

  (A) Independent sweeps — sharpen/broaden P by a temperature (P^(1/T), T<1
      sharper, T>1 broader) and shift P in orientation; score each loss
      (candidate, P) and average over real posteriors. Shows the width
      asymmetry and the shared location signal on REAL posterior shapes.
  (B) Joint landscape — the 2-D loss surface over (location shift × sharpness)
      per loss, averaged over real posteriors.
  (C) Recovery — gradient-descend a free softmax under each loss from an
      over-confident spike at P's mode, using that mouse's REAL (rank-~6) PCA
      basis, and read back the recovered width/location; plus example overlays
      of real posteriors with each loss's fit. KL/CE/JS recover the real
      posterior; Projection-based & Wasserstein collapse to spikes (the real
      basis only constrains ~6 directions, leaving the rest free).

Companion: `subspace_error_realdata.py` (per-PC decoded−target error spectrum).
All conclusions live in the titles/captions — no text boxes over the data.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  locsharp_sweeps.png       (A) loss vs sharpness and vs location shift
  locsharp_landscape.png    (B) 2-D loss heatmap per loss
  locsharp_recovery.png     (C) recovered width & location vs the real target
  locsharp_examples.png     (C) real posteriors + each loss's fit

Run:  cd nn_decoder && OMP_NUM_THREADS=1 \
        python diagnostics/location_sharpness_grid.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NN_DECODER = str(Path(__file__).resolve().parent.parent)
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

import peakiness_style as ps                                          # noqa: E402
from pca_loss import pca_distance                                    # noqa: E402
from nn_classifier import (                                          # noqa: E402
    KL_calc, JS_calc, Wasserstein_calc_1D, cross_entropy,
)
from loss_smoothness_demo import optimise_single, EPS                # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

LOSSES = ["PCA", "CE", "KL", "JS", "Wasserstein"]
LCOL = {"PCA": ps.PCA_EVAR, "CE": ps.CE, "KL": ps.KL, "JS": ps.JS,
        "Wasserstein": ps.WASSERSTEIN}
N_CATS = 91
THETA = np.arange(N_CATS, dtype=float)          # orientation grid (bins, 0–90°)
LOSS_ORDER_FIT = ["PCA", "CE", "KL", "JS", "Wasserstein"]


# ---------------------------------------------------------------------------
def load_real(results_root, run, split, n_sample):
    """Pool real IO-target posteriors (+ that mouse's PCA basis) across mice.

    The target is loss-invariant, so it's read from one present cell per mouse.
    Returns a list of (P (91,), pcs (K,91), evar (K,)) sampled evenly."""
    pool = []
    for loss in ("KL", "JS", "CE", "PCA", "Wasserstein"):
        slug = f"Q_{loss}_half_100ms" + ("_all" if loss == "PCA" else "")
        f = Path(results_root) / run / slug / f"{split}.mat"
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get("results")
        if not isinstance(res, dict):
            continue
        for mk in sorted(res):
            D = res[mk]["Dist"]
            tg = np.asarray(D["spat"]["target"], float)
            pcs = np.asarray(D["pcs"], float)
            evar = np.asarray(D["explained_var"], float)
            for i in range(tg.shape[0]):
                p = tg[i]
                if np.isfinite(p).all() and p.sum() > 0:
                    pool.append((p / p.sum(), pcs, evar))
        break                                    # one cell suffices (loss-invariant)
    if not pool:
        raise SystemExit(f"no real targets under {run}/. rsync first.")
    idx = np.linspace(0, len(pool) - 1, min(n_sample, len(pool))).astype(int)
    return [pool[i] for i in idx]


def losses5(pred, target, pcs, evar):
    """Five production losses for one (pred, target) pair, that mouse's basis."""
    p = torch.tensor(pred, dtype=torch.float32).unsqueeze(0)
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    return {
        "PCA": float(pca_distance(pred[None, :], target[None, :], pcs, evar)[0]),
        "CE": float(cross_entropy(p, t)[0]),
        "KL": float(KL_calc(p, t)[0]),
        "JS": float(JS_calc(p, t)[0]),
        "Wasserstein": float(Wasserstein_calc_1D(p, t)[0]),
    }


def lin_mean(p):
    return float((p * THETA).sum())


def lin_std(p):
    mu = lin_mean(p)
    return float(np.sqrt(max((p * THETA ** 2).sum() - mu ** 2, 0.0)))


def temper(p, T):
    """Sharpen (T<1) / broaden (T>1) a posterior by a temperature power."""
    q = np.power(np.clip(p, 0, None), 1.0 / T)
    s = q.sum()
    return q / s if s > 0 else p.copy()


def shift(p, d):
    """Translate a posterior by d bins (truncate off-edge mass, renormalise)."""
    out = np.zeros_like(p)
    if d >= 0:
        out[d:] = p[:N_CATS - d]
    else:
        out[:d] = p[-d:]
    s = out.sum()
    return out / s if s > 0 else p.copy()


def _mm(v):
    return (v - v.min()) / (v.max() - v.min() + EPS)


# ======================================================================
# (A) Independent sweeps on real posteriors
# ======================================================================
def fig_sweeps(samples, out_root):
    Ts = np.geomspace(0.45, 2.4, 19)            # sharper -> broader
    shifts = np.arange(-14, 15, 2)
    wloss = {k: np.zeros(len(Ts)) for k in LOSSES}
    sloss = {k: np.zeros(len(shifts)) for k in LOSSES}
    wwidth = np.zeros(len(Ts))
    base_width = 0.0
    for P, pcs, evar in samples:
        base_width += lin_std(P)
        for ti, T in enumerate(Ts):
            cand = temper(P, T)
            wwidth[ti] += lin_std(cand)
            L = losses5(cand, P, pcs, evar)
            for k in LOSSES:
                wloss[k][ti] += L[k]
        for di, d in enumerate(shifts):
            L = losses5(shift(P, d), P, pcs, evar)
            for k in LOSSES:
                sloss[k][di] += L[k]
    n = len(samples)
    wwidth /= n
    base_width /= n

    ps.apply()
    fig, (axW, axS) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for k in LOSSES:
        axW.plot(wwidth, _mm(wloss[k]), color=LCOL[k], lw=2, marker="o", ms=3,
                 label=ps.loss_label(k))
    axW.axvline(base_width, color="0.4", ls="--", lw=1.2)
    axW.set_xlabel("candidate width — linear std (bins)   ← sharper · broader →")
    axW.set_ylabel("loss (per-loss min→max normalised)")
    axW.set_title("Sharpen / broaden the real posterior (location fixed)\n"
                  "KL/JS/CE penalise sharpening > broadening; "
                  "Projection-based & Wasserstein ≈ symmetric")
    axW.legend(frameon=False, fontsize=8, loc="upper center")

    for k in LOSSES:
        axS.plot(shifts, _mm(sloss[k]), color=LCOL[k], lw=2, marker="s", ms=3,
                 label=ps.loss_label(k))
    axS.set_xlabel("location shift (bins off the real posterior)")
    axS.set_ylabel("loss (per-loss min→max normalised)")
    axS.set_title("Shift location (sharpness fixed)\n"
                  "every loss rises together — location is the shared signal")
    axS.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Sharpen/broaden & location on the real IO posteriors "
                 f"({n} posteriors, 6 mice; mean width {base_width:.1f} bins)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ps.save_fig(fig, Path(out_root), "locsharp_sweeps", layout=None)
    print(f"  sweeps: {n} real posteriors, base width {base_width:.1f} bins")


# ======================================================================
# (B) Joint landscape on real posteriors
# ======================================================================
def fig_landscape(samples, out_root):
    Ts = np.geomspace(0.5, 2.2, 13)
    shifts = np.arange(-14, 15, 2)
    grids = {k: np.zeros((len(Ts), len(shifts))) for k in LOSSES}
    width_y = np.zeros(len(Ts))
    base_width = 0.0
    for P, pcs, evar in samples:
        base_width += lin_std(P)
        for ti, T in enumerate(Ts):
            tp = temper(P, T)
            width_y[ti] += lin_std(tp)
            for di, d in enumerate(shifts):
                L = losses5(shift(tp, d), P, pcs, evar)
                for k in LOSSES:
                    grids[k][ti, di] += L[k]
    n = len(samples)
    width_y /= n
    base_width /= n

    ps.apply()
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    ext = [shifts[0], shifts[-1], width_y[0], width_y[-1]]
    for ax, k in zip(axes, LOSSES):
        g = grids[k]
        gn = (g - g.min()) / (g.max() - g.min() + EPS)
        im = ax.imshow(gn, origin="lower", aspect="auto", extent=ext,
                       cmap="viridis", vmin=0, vmax=1)
        ax.plot(0, base_width, "*", color="white", ms=13, mec="k", mew=0.6)
        ax.set_title(ps.loss_label(k), color=LCOL[k], fontweight="bold")
        ax.set_xlabel("location shift (bins)")
    axes[0].set_ylabel("candidate width (bins)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cbar.set_label("loss (per-loss min→max)")
    fig.suptitle("Joint loss landscape over (location × sharpness) on the real "
                 "posteriors; white star = the true target", y=1.04)
    ps.save_fig(fig, Path(out_root), "locsharp_landscape", layout=None)


# ======================================================================
# (C) Recovery on real posteriors — what each loss reconstructs
# ======================================================================
def _spike_logits(P):
    """Over-confident spike at the real posterior's mode."""
    init = np.full(N_CATS, -6.0, dtype=np.float32)
    init[int(np.argmax(P))] = 6.0
    return torch.tensor(init)


def fig_recovery(samples, out_root, steps=5000):
    # span the width range: sort sampled posteriors by width, pick a spread.
    # We score the fit by PEAKINESS (max-prob), not 2nd-moment width: the
    # over-sharpening is a high-frequency SPIKE that barely changes the variance
    # (the projection-based fit keeps a roughly-correct broad pedestal AND adds a
    # spike), so a width/std metric is blind to it — exactly as the loss is.
    samples = sorted(samples, key=lambda s: lin_std(s[0]))
    idx = np.linspace(0, len(samples) - 1, 16).astype(int)
    chosen = [samples[i] for i in idx]
    tgt_pk, tgt_l = [], []
    rec_pk = {k: [] for k in LOSSES}
    rec_l = {k: [] for k in LOSSES}
    for P, pcs, evar in chosen:
        tgt_pk.append(float(P.max())); tgt_l.append(lin_mean(P))
        pcs_t = torch.tensor(pcs, dtype=torch.float32)
        evar_t = torch.tensor(evar, dtype=torch.float32)
        init = _spike_logits(P)
        for k in LOSSES:
            fit, _, _ = optimise_single(P, k, pcs_t, evar_t, init, steps=steps)
            rec_pk[k].append(float(fit.max())); rec_l[k].append(lin_mean(fit))
    tgt_pk = np.array(tgt_pk); tgt_l = np.array(tgt_l)

    ps.apply()
    fig, (axP, axL) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    lim = max(tgt_pk.max(), max(max(rec_pk[k]) for k in LOSSES)) * 1.08
    axP.plot([0, lim], [0, lim], color="0.5", ls="--", lw=1.3, label="identity (calibrated)")
    for k in LOSSES:
        axP.plot(tgt_pk, rec_pk[k], color=LCOL[k], lw=1.6, marker="o", ms=5,
                 label=ps.loss_label(k))
    axP.set_xlim(0, lim); axP.set_ylim(0, lim)
    axP.set_xlabel("real posterior peakiness (max-prob)")
    axP.set_ylabel("recovered peakiness (max-prob)")
    axP.set_title("Sharpness recovery — KL/CE/JS & Wasserstein match the real "
                  "posterior;\nonly the projection-based loss over-sharpens (~5× "
                  "on broad targets)")
    axP.legend(frameon=False, fontsize=8, loc="upper left")

    axL.plot([min(tgt_l), max(tgt_l)], [min(tgt_l), max(tgt_l)],
             color="0.5", ls="--", lw=1.3, label="identity")
    for k in LOSSES:
        axL.plot(tgt_l, rec_l[k], color=LCOL[k], lw=1.6, marker="s", ms=5,
                 label=ps.loss_label(k))
    axL.set_xlabel("real posterior location — mean (bin)")
    axL.set_ylabel("recovered location (bin)")
    axL.set_title("Location recovery — every loss tracks the real posterior")
    axL.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("What each loss reconstructs from the real IO posterior "
                 f"(free fit from an over-confident spike, {steps} steps)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ps.save_fig(fig, Path(out_root), "locsharp_recovery", layout=None)
    lo = tgt_pk < np.median(tgt_pk)               # the broad (low-peakiness) targets
    print("  recovery, broad targets — recovered/real peakiness:")
    for k in LOSSES:
        r = np.array(rec_pk[k])[lo] / (tgt_pk[lo] + EPS)
        print(f"    {k:12s} {r.mean():.2f}x")


def fig_examples(samples, out_root, steps=5000):
    # pick a narrow, a mid, a broad and the most bimodal real posterior
    by_w = sorted(samples, key=lambda s: lin_std(s[0]))
    def bimodality(P):                            # crude: 2nd-mode mass / peak
        idx = int(np.argmax(P)); m = P.copy()
        m[max(0, idx - 6):idx + 7] = 0
        return m.max() / (P.max() + EPS)
    examples = [("narrowest", by_w[0]),
                ("median width", by_w[len(by_w) // 2]),
                ("broadest", by_w[-1]),
                ("most bimodal", max(samples, key=lambda s: bimodality(s[0])))]

    ps.apply()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
    for ax, (name, (P, pcs, evar)) in zip(axes, examples):
        ax.fill_between(THETA, P, color="0.82", lw=0, zorder=0, label="IO target")
        pcs_t = torch.tensor(pcs, dtype=torch.float32)
        evar_t = torch.tensor(evar, dtype=torch.float32)
        init = _spike_logits(P)
        for k in LOSSES:
            fit, _, _ = optimise_single(P, k, pcs_t, evar_t, init, steps=steps)
            ax.plot(THETA, fit, color=LCOL[k], lw=1.5, label=ps.loss_label(k))
        ax.set_title(f"{name}  (width {lin_std(P):.0f} bins)", fontsize=9)
        ax.set_xlabel("orientation (bin)")
    axes[0].set_ylabel("probability")
    axes[0].legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
    fig.suptitle("Free fit to real IO posteriors: KL/CE/JS & Wasserstein recover "
                 "the shape; only the projection-based loss spikes", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ps.save_fig(fig, Path(out_root), "locsharp_examples", layout=None)


def main(results_root, run, split, out_root, n_sweep):
    samples = load_real(results_root, run, split, n_sweep)
    print(f"loaded {len(samples)} real IO posteriors from {run}")
    fig_sweeps(samples, out_root)
    fig_landscape(samples, out_root)
    fig_recovery(samples, out_root)
    fig_examples(samples, out_root)
    print(f"\nDone. {Path(out_root).resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="loss_comparison_v1")
    p.add_argument("--split", default="stratified_balanced")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-root", default="figures/peakiness_scatter")
    p.add_argument("--n-sweep", type=int, default=90,
                   help="number of real posteriors to average the sweeps over")
    a = p.parse_args()
    main(a.results_root, a.run, a.split, a.out_root, a.n_sweep)
