# -*- coding: utf-8 -*-
"""Sharpen/broaden and location tests for the five losses, on the REAL
perceptual (IO-target) posteriors and the REAL trained decoders.

Two questions, both on real data — no synthetic bumps, no free-fitting:

  (A) How does each loss respond when a real IO posterior is sharpened/broadened
      or shifted?  For every real posterior P we sharpen/broaden it by raising it
      to a power (P^(1/T): T<1 sharper, T>1 broader) and shift it in orientation,
      then score each production loss (candidate, P) and average over ALL real
      posteriors (all trials, 6 mice).  → the loss geometry along each axis,
      plus the joint (location × sharpness) landscape.
  (B) What do the trained decoders actually do?  For example trials we overlay
      the real IO target with each loss-trained decoder's real decoded posterior
      (`Dist['spat']['decoded']`).  The projection-based decoder over-sharpens
      (spikes) on broad/bimodal targets; CE/KL/JS track the shape.

Companion: `subspace_error_realdata.py` (per-PC decoded−target error spectrum).
All conclusions live in the titles/captions — no text boxes over the data.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  locsharp_sweeps.png       (A) loss vs sharpness (power) and vs location shift
  locsharp_landscape.png    (A) 2-D loss heatmap per loss
  locsharp_examples.png     (B) real decoded posteriors vs the IO target

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

LOSSES = ["PCA", "CE", "KL", "JS", "Wasserstein"]
LCOL = {"PCA": ps.PCA_EVAR, "CE": ps.CE, "KL": ps.KL, "JS": ps.JS,
        "Wasserstein": ps.WASSERSTEIN}
N_CATS = 91
THETA = np.arange(N_CATS, dtype=float)          # orientation grid (bins, 0–90°)
EPS = 1e-12


# ---------------------------------------------------------------------------
def load_data(results_root, run, split):
    """Per mouse: the IO target (loss-invariant), its PCA basis, and each
    loss-trained decoder's real decoded posterior."""
    mice = {}
    for loss in LOSSES:
        slug = f"Q_{loss}_half_100ms" + ("_all" if loss == "PCA" else "")
        f = Path(results_root) / run / slug / f"{split}.mat"
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get("results")
        if not isinstance(res, dict):
            continue
        for mk in sorted(res):
            D = res[mk]["Dist"]
            md = mice.setdefault(mk, {"decoded": {}})
            if "target" not in md:
                md["target"] = np.asarray(D["spat"]["target"], float)
                md["pcs"] = np.asarray(D["pcs"], float)
                md["evar"] = np.asarray(D["explained_var"], float)
            md["decoded"][loss] = np.asarray(D["spat"]["decoded"], float)
    if not mice:
        raise SystemExit(f"no cells under {run}/. rsync first.")
    return mice


def all_targets(mice):
    """Every real IO posterior (normalised) paired with its mouse's basis."""
    out = []
    for md in mice.values():
        tg, pcs, evar = md["target"], md["pcs"], md["evar"]
        for i in range(tg.shape[0]):
            p = tg[i]
            if np.isfinite(p).all() and p.sum() > 0:
                out.append((p / p.sum(), pcs, evar))
    return out


def losses5(pred, target, pcs, evar):
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
    """Sharpen (T<1) / broaden (T>1) by raising the posterior to the power 1/T."""
    q = np.power(np.clip(p, 0, None), 1.0 / T)
    s = q.sum()
    return q / s if s > 0 else p.copy()


def shift(p, d):
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
# (A) Independent sweeps on real posteriors — loss response
# ======================================================================
def fig_sweeps(targets, out_root):
    Ts = np.geomspace(0.45, 2.4, 19)            # sharper -> broader (power 1/T)
    shifts = np.arange(-14, 15, 2)
    wloss = {k: np.zeros(len(Ts)) for k in LOSSES}
    sloss = {k: np.zeros(len(shifts)) for k in LOSSES}
    wwidth = np.zeros(len(Ts))
    base_width = 0.0
    for P, pcs, evar in targets:
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
    n = len(targets)
    wwidth /= n
    base_width /= n

    ps.apply()
    fig, (axW, axS) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for k in LOSSES:
        axW.plot(wwidth, _mm(wloss[k]), color=LCOL[k], lw=2, marker="o", ms=3,
                 label=ps.loss_label(k))
    axW.axvline(base_width, color="0.4", ls="--", lw=1.2)
    axW.set_xlabel("width of the powered posterior  P^(1/T)  (bins)   "
                   "← sharper · broader →")
    axW.set_ylabel("loss (per-loss min→max normalised)")
    axW.set_title("Sharpen / broaden by raising to a power (location fixed)\n"
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

    fig.suptitle(f"Sharpen/broaden & location on ALL real IO posteriors "
                 f"({n} trials, 6 mice; mean width {base_width:.1f} bins)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ps.save_fig(fig, Path(out_root), "locsharp_sweeps", layout=None)
    print(f"  sweeps: ALL {n} real posteriors, mean width {base_width:.1f} bins")


# ======================================================================
# (A) Joint landscape on real posteriors
# ======================================================================
def fig_landscape(targets, out_root):
    Ts = np.geomspace(0.3, 4.2, 9)
    shifts = np.arange(-12, 13, 4)
    grids = {k: np.zeros((len(Ts), len(shifts))) for k in LOSSES}
    width_y = np.zeros(len(Ts))
    base_width = 0.0
    for P, pcs, evar in targets:
        base_width += lin_std(P)
        for ti, T in enumerate(Ts):
            tp = temper(P, T)
            width_y[ti] += lin_std(tp)
            for di, d in enumerate(shifts):
                L = losses5(shift(tp, d), P, pcs, evar)
                for k in LOSSES:
                    grids[k][ti, di] += L[k]
    n = len(targets)
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
    axes[0].set_ylabel("width of P^(1/T) (bins)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cbar.set_label("loss (per-loss min→max)")
    fig.suptitle(f"Joint loss landscape over (location × sharpness) on all {n} "
                 "real posteriors; white star = the true target", y=1.04)
    ps.save_fig(fig, Path(out_root), "locsharp_landscape", layout=None)


# ======================================================================
# (B) Real decoded posteriors vs the IO target — what the decoders do
# ======================================================================
def fig_examples(mice, out_root, mouse="mouse_0"):
    md = mice[mouse]
    tg = md["target"]
    norm = np.array([tg[i] / (tg[i].sum() + EPS) for i in range(tg.shape[0])])
    widths = np.array([lin_std(norm[i]) for i in range(norm.shape[0])])

    def bimodality(p):
        i = int(np.argmax(p)); m = p.copy()
        m[max(0, i - 6):i + 7] = 0
        return m.max() / (p.max() + EPS)
    bim = np.array([bimodality(norm[i]) for i in range(norm.shape[0])])
    order = np.argsort(widths)
    picks = [("narrowest", order[0]),
             ("median width", order[len(order) // 2]),
             ("broadest", order[-1]),
             ("most bimodal", int(np.argmax(bim)))]

    ps.apply()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
    for ax, (name, i) in zip(axes, picks):
        ax.fill_between(THETA, norm[i], color="0.82", lw=0, zorder=0, label="IO target")
        for k in LOSSES:
            dec = md["decoded"].get(k)
            if dec is None:
                continue
            ax.plot(THETA, dec[i], color=LCOL[k], lw=1.5, label=ps.loss_label(k))
        ax.set_title(f"{name}  (width {widths[i]:.0f} bins)", fontsize=9)
        ax.set_xlabel("orientation (bin)")
    axes[0].set_ylabel("probability")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Real trained decoders vs the IO target: the projection-based & "
                 "Wasserstein decoders over-sharpen (jagged spikes); CE/KL/JS "
                 f"track the shape ({mouse})", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ps.save_fig(fig, Path(out_root), "locsharp_examples", layout=None)
    print(f"  examples: real decoded posteriors, {mouse}, trials "
          f"{[int(i) for _, i in picks]}")


def main(results_root, run, split, out_root):
    mice = load_data(results_root, run, split)
    targets = all_targets(mice)
    print(f"loaded {len(targets)} real IO posteriors from {len(mice)} mice")
    fig_sweeps(targets, out_root)
    fig_landscape(targets, out_root)
    fig_examples(mice, out_root)
    print(f"\nDone. {Path(out_root).resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="loss_comparison_v1")
    p.add_argument("--split", default="stratified_balanced")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-root", default="figures/peakiness_scatter")
    a = p.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
