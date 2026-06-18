# -*- coding: utf-8 -*-
"""Systematic location × sharpness probe of the five losses.

The companion to the hand-picked ``loss_smoothness_demo`` gallery (fig5/fig9):
instead of a few example targets, we sweep a target's **peak location** and its
**sharpness (width)** — independently and jointly — and ask, for each of the five
production losses (PCA, CE, KL, JS, Wasserstein):

  (A) Independent loss surfaces — how does each loss change as a candidate
      posterior is made sharper/broader (location fixed) or shifted off the
      target (width fixed)?  [no fitting; the loss *landscape*]
  (B) Joint loss landscape — the 2-D loss surface over (location offset × width)
      per loss.  PCA/Wasserstein show a valley that runs flat along the width
      axis (they only "see" location); CE/KL/JS show a basin localised in both.
  (C) Direct-fit recovery — gradient-descend a free softmax posterior under each
      loss (from an over-confident spike) onto targets across a location×width
      grid, and read back the *recovered* location and width.  PCA/Wasserstein
      collapse to a sharp spike at every target width (recovered width flat-low);
      CE/KL/JS track the target width.  Location is recovered by every loss.

All losses are the **production** functions from ``nn_classifier`` / ``pca_loss``
(CE is the trainer's cross-entropy branch, ≡ KL up to the constant H(target), so
its fit tracks KL — shown for completeness). The PCA basis is fit exactly as
``run_experiment.fit_pca_basis`` does (condition-averaged broad bumps), reusing
``loss_smoothness_demo.fit_basis``.

The per-PC view of *where on the real decoder* PCA's mismatch lives is the
companion ``subspace_error_realdata.py`` (per-PC decoded−target error spectrum).

Outputs (PNG+SVG) under ``figures/peakiness_scatter/``:
  locsharp_independent_sweeps.png   (A) loss vs width and loss vs shift, 5 losses
  locsharp_joint_landscape.png      (B) 2-D loss heatmap per loss
  locsharp_fit_recovery.png         (C) recovered width & location vs target
  locsharp_metrics.csv              numeric outputs

Run:  cd nn_decoder && OMP_NUM_THREADS=1 python diagnostics/location_sharpness_grid.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
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
# reuse the synthetic-target + direct-fit machinery already validated there
from loss_smoothness_demo import (                                   # noqa: E402
    bump, circ_std_bins, circ_dist, entropy_np, fit_basis,
    optimise_single, N_CATS, EPS,
)

torch.manual_seed(0)
np.random.seed(0)

LOSSES = ["PCA", "CE", "KL", "JS", "Wasserstein"]
LCOL = {"PCA": ps.PCA_EVAR, "CE": ps.CE, "KL": ps.KL, "JS": ps.JS,
        "Wasserstein": ps.WASSERSTEIN}
CENTER = N_CATS // 2          # 45 — fixed reference location
SIGMA_TARGET = 12.0           # broad reference target width (bins)


# ---------------------------------------------------------------------------
def losses5(pred, target, pcs, evar):
    """All five production losses for one (pred, target) pair of length N_CATS."""
    p = torch.tensor(pred, dtype=torch.float32).unsqueeze(0)
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    return {
        "PCA": float(pca_distance(pred[None, :], target[None, :], pcs, evar)[0]),
        "CE": float(cross_entropy(p, t)[0]),
        "KL": float(KL_calc(p, t)[0]),
        "JS": float(JS_calc(p, t)[0]),
        "Wasserstein": float(Wasserstein_calc_1D(p, t)[0]),
    }


def circ_mean_bin(p, n=N_CATS):
    """Circular mean location (in bins) of a distribution on the ring."""
    ang = 2 * np.pi * np.arange(n) / n
    m = np.arctan2((p * np.sin(ang)).sum(), (p * np.cos(ang)).sum())
    return (m % (2 * np.pi)) * n / (2 * np.pi)


# ======================================================================
# (A) Independent loss surfaces — sharpen/broaden and shift, all 5 losses
# ======================================================================
def fig_independent(pcs, evar, out_root, rows):
    target = bump(CENTER, SIGMA_TARGET)

    sig_cands = np.linspace(2, 30, 29)             # sharpen -> broaden
    shift_cands = np.arange(0, 28)                  # shift off target (bins)

    width = {k: [] for k in LOSSES}
    for s in sig_cands:
        L = losses5(bump(CENTER, s), target, pcs, evar)
        for k in LOSSES:
            width[k].append(L[k])
    shift = {k: [] for k in LOSSES}
    for d in shift_cands:
        L = losses5(bump(CENTER + d, SIGMA_TARGET), target, pcs, evar)
        for k in LOSSES:
            shift[k].append(L[k])
    width = {k: np.array(v) for k, v in width.items()}
    shift = {k: np.array(v) for k, v in shift.items()}

    for i, s in enumerate(sig_cands):
        rows.append({"fig": "independent", "axis": "width", "level": float(s),
                     **{k: float(width[k][i]) for k in LOSSES}})
    for i, d in enumerate(shift_cands):
        rows.append({"fig": "independent", "axis": "shift", "level": int(d),
                     **{k: float(shift[k][i]) for k in LOSSES}})

    # min-max normalise each loss to [0,1] over the plotted range. This is the
    # honest comparison: it cancels additive constants, so CE (= KL + H(target),
    # a constant offset) maps EXACTLY onto KL — they share a gradient, hence a
    # landscape. Dividing by max alone would leave CE's constant in and make it
    # look spuriously flatter than KL.
    def _mm(v):
        return (v - v.min()) / (v.max() - v.min() + EPS)

    ps.apply()
    fig, (axW, axS) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for k in LOSSES:
        axW.plot(sig_cands, _mm(width[k]), color=LCOL[k], lw=2, marker="o", ms=3,
                 label=ps.loss_label(k))
    axW.axvline(SIGMA_TARGET, color="0.4", ls="--", lw=1.2)
    axW.text(SIGMA_TARGET + 0.4, 0.9, "target\nwidth", fontsize=8, color="0.3")
    axW.axvspan(2, SIGMA_TARGET, color="orange", alpha=0.06)
    axW.set_xlabel("candidate width  σ_cand (bins)   ← sharper · broader →")
    axW.set_ylabel("loss (per-loss min→max normalised)")
    axW.set_title("Sharpen / broaden (clean Gaussian candidate, location fixed)\n"
                  "KL/JS/CE punish too-sharp; Projection-based & Wasserstein ~symmetric")
    axW.legend(frameon=False, fontsize=8, loc="lower right")
    # raw too-sharp / too-broad asymmetry ratio (the restoring-force asymmetry).
    # CE's *value* ratio is compressed by its +H(target) constant, but its
    # gradient — hence fit — is identical to KL's; reported as ≡KL.
    Ls = losses5(bump(CENTER, 2), target, pcs, evar)
    Lb = losses5(bump(CENTER, 24), target, pcs, evar)
    lines = ["too-sharp / too-broad:"]
    for k in LOSSES:
        tag = f"{Ls[k]/(Lb[k]+EPS):4.1f}×" + ("  (≡KL grad)" if k == "CE" else "")
        lines.append(f"  {ps.loss_label(k):16s}{tag}")
    axW.text(0.5, 0.97, "\n".join(lines), transform=axW.transAxes, fontsize=7,
             ha="center", va="top", family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    for k in LOSSES:
        axS.plot(shift_cands, _mm(shift[k]), color=LCOL[k], lw=2, marker="s",
                 ms=3, label=ps.loss_label(k))
    axS.set_xlabel("candidate peak shift (bins off target)")
    axS.set_ylabel("loss (per-loss min→max normalised)")
    axS.set_title("Shift location (width fixed)\n"
                  "every loss rises together — location is the shared signal")
    axS.legend(frameon=False, fontsize=8)

    fig.suptitle("Loss landscape along the two error axes — all five losses "
                 "(clean Gaussian candidates, no fitting)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ps.save_fig(fig, Path(out_root), "locsharp_independent_sweeps", layout=None)
    # offset-removed asymmetry (sharp vs broad), constant cancels → CE = KL
    nb = {k: _mm(width[k]) for k in LOSSES}
    i_s, i_b = int(np.argmin(np.abs(sig_cands - 2))), int(np.argmin(np.abs(sig_cands - 24)))
    print("  normalised loss [too-sharp | too-broad]: " +
          "  ".join(f"{k} {nb[k][i_s]:.2f}/{nb[k][i_b]:.2f}" for k in LOSSES))


# ======================================================================
# (B) Joint loss landscape — 2-D (shift × width) heatmap per loss
# ======================================================================
def fig_joint(pcs, evar, out_root, rows):
    target = bump(CENTER, SIGMA_TARGET)
    shifts = np.arange(-22, 23, 2)              # location offset (bins)
    sigmas = np.linspace(2, 30, 29)             # candidate width (bins)

    grids = {k: np.zeros((len(sigmas), len(shifts))) for k in LOSSES}
    for i, s in enumerate(sigmas):
        for j, d in enumerate(shifts):
            L = losses5(bump(CENTER + d, s), target, pcs, evar)
            for k in LOSSES:
                grids[k][i, j] = L[k]
    for k in LOSSES:
        for i, s in enumerate(sigmas):
            for j, d in enumerate(shifts):
                rows.append({"fig": "joint", "loss": k, "sigma": float(s),
                             "shift": int(d), "loss_val": float(grids[k][i, j])})

    ps.apply()
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    ext = [shifts[0], shifts[-1], sigmas[0], sigmas[-1]]
    for ax, k in zip(axes, LOSSES):
        g = grids[k]
        gn = (g - g.min()) / (g.max() - g.min() + EPS)   # per-loss min-max
        im = ax.imshow(gn, origin="lower", aspect="auto", extent=ext,
                       cmap="viridis", vmin=0, vmax=1)
        # true optimum and the flat-valley readout
        ax.plot(0, SIGMA_TARGET, "*", color="white", ms=13, mec="k", mew=0.6)
        ax.axhline(SIGMA_TARGET, color="white", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(k, color=LCOL[k], fontweight="bold")
        ax.set_xlabel("peak shift (bins)")
    axes[0].set_ylabel("candidate width σ (bins)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cbar.set_label("loss (per-loss min→max)")
    fig.suptitle("Joint loss landscape over (location × coarse width); white "
                 "star = true target.  Vertical asymmetry is the tell — CE/KL/JS "
                 "punish too-sharp (bright bottom) ≫ too-broad; Projection-based "
                 "& Wasserstein more symmetric (CE & KL identical — same gradient)", y=1.04)
    ps.save_fig(fig, Path(out_root), "locsharp_joint_landscape", layout=None)

    # quantify the "flat along width" claim: loss range across width at the
    # correct location, relative to the loss range across shift at target width
    print("\n  width-blindness (Δloss across width at Δ=0) / (Δloss across "
          "shift at σ_t), per loss:")
    j0 = np.argmin(np.abs(shifts - 0))
    i_t = np.argmin(np.abs(sigmas - SIGMA_TARGET))
    for k in LOSSES:
        g = grids[k]
        gn = (g - g.min()) / (g.max() - g.min() + EPS)
        wspan = gn[:, j0].max() - gn[:, j0].min()        # across width @ correct loc
        sspan = gn[i_t, :].max() - gn[i_t, :].min()       # across shift @ target width
        print(f"    {k:11s} width-span={wspan:.3f}  shift-span={sspan:.3f}  "
              f"ratio={wspan/(sspan+EPS):.2f}")
        rows.append({"fig": "joint_summary", "loss": k,
                     "width_span_norm": float(wspan),
                     "shift_span_norm": float(sspan)})


# ======================================================================
# (C) Direct-fit recovery — recovered width & location vs target
# ======================================================================
def fig_recovery(pcs, evar, out_root, rows, steps=8000):
    pcs_t = torch.tensor(pcs)
    evar_t = torch.tensor(evar)

    # --- width recovery: target width varies, location fixed at CENTER ---
    target_sigmas = np.array([3, 5, 7, 9, 12, 15, 18, 22], float)
    init_sharp = torch.tensor(np.log(bump(CENTER, 1.2) + EPS), dtype=torch.float32)
    rec_w = {k: [] for k in LOSSES}
    tgt_w = []
    for st in target_sigmas:
        target = bump(CENTER, st)
        tgt_w.append(circ_std_bins(target))
        for k in LOSSES:
            fit, _, _ = optimise_single(target, k, pcs_t, evar_t, init_sharp,
                                        steps=steps)
            rec_w[k].append(circ_std_bins(fit))
    tgt_w = np.array(tgt_w)

    # --- location recovery: target location varies, width fixed; init from a
    #     UNIFORM prior (zero logits) so the loss is free to place the peak —
    #     a clean location control. (A sharp off-target init instead confounds
    #     this with PCA/Wasserstein's weak force to *relocate* a committed spike,
    #     a separate effect.) ---
    target_locs = np.array([15, 25, 35, 45, 55, 65, 75], float)
    init_unif = torch.zeros(N_CATS, dtype=torch.float32)
    rec_l = {k: [] for k in LOSSES}
    for cl in target_locs:
        target = bump(cl, 8.0)
        for k in LOSSES:
            fit, _, _ = optimise_single(target, k, pcs_t, evar_t, init_unif,
                                        steps=steps)
            rec_l[k].append(circ_mean_bin(fit))

    for i, st in enumerate(target_sigmas):
        rows.append({"fig": "recovery", "axis": "width",
                     "target_sigma": float(st), "target_width": float(tgt_w[i]),
                     **{f"rec_{k}": float(rec_w[k][i]) for k in LOSSES}})
    for i, cl in enumerate(target_locs):
        rows.append({"fig": "recovery", "axis": "location",
                     "target_loc": float(cl),
                     **{f"rec_{k}": float(rec_l[k][i]) for k in LOSSES}})

    ps.apply()
    fig, (axW, axL) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    lim = max(tgt_w.max(), max(max(rec_w[k]) for k in LOSSES)) * 1.08
    axW.plot([0, lim], [0, lim], color="0.5", ls="--", lw=1.3, label="identity (perfect)")
    for k in LOSSES:
        axW.plot(tgt_w, rec_w[k], color=LCOL[k], lw=2, marker="o", ms=5, label=ps.loss_label(k))
    axW.set_xlim(0, lim); axW.set_ylim(0, lim)
    axW.set_xlabel("target width — circular std (bins)")
    axW.set_ylabel("recovered width (bins)")
    axW.set_title("Width recovery", fontsize=11)
    axW.legend(frameon=False, fontsize=8, loc="upper left")
    axW.text(0.96, 0.04, "CE/KL/JS recover the target width;\nProjection-based & Wasserstein "
             "collapse to spiky fits —\nflat in the high-frequency 'fine-width'\n"
             "subspace (still collapsed at 50k steps)",
             transform=axW.transAxes, fontsize=8, ha="right", va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    axL.plot([10, 80], [10, 80], color="0.5", ls="--", lw=1.3, label="identity")
    for k in LOSSES:
        axL.plot(target_locs, rec_l[k], color=LCOL[k], lw=2, marker="s", ms=5, label=ps.loss_label(k))
    axL.set_xlabel("target peak location (bin)")
    axL.set_ylabel("recovered location (bin)")
    axL.set_title("Location recovery (uniform init)", fontsize=11)
    axL.legend(frameon=False, fontsize=8, loc="upper left")
    axL.text(0.96, 0.04, "every loss tracks the target\nlocation — the shared signal",
             transform=axL.transAxes, fontsize=8, ha="right", va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    fig.suptitle("What each loss recovers when location & sharpness are swept "
                 f"(direct fit from an over-confident spike, {steps} steps)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ps.save_fig(fig, Path(out_root), "locsharp_fit_recovery", layout=None)

    print("\n  recovered width at target σ=18 (bins; target width "
          f"≈{tgt_w[-2]:.1f}):")
    for k in LOSSES:
        print(f"    {k:11s} {rec_w[k][-2]:.2f}")


# ======================================================================
def write_csv(rows, out_root):
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    path = Path(out_root) / "locsharp_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def main(out_root, broad_sigma=9.0):
    os.makedirs(out_root, exist_ok=True)
    pcs, evar, _ = fit_basis(broad_sigma)
    print(f"PCA basis: {len(evar)} PCs; top-2 evar = {evar[:2]} "
          f"(sum {evar[:2].sum():.3f})")
    rows = []
    fig_independent(pcs, evar, out_root, rows)
    fig_joint(pcs, evar, out_root, rows)
    fig_recovery(pcs, evar, out_root, rows)
    path = write_csv(rows, out_root)
    print(f"\nWrote 3 figures + {path.name} to {out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", default="figures/peakiness_scatter")
    p.add_argument("--broad-sigma", type=float, default=9.0,
                   help="width (bins) of the broad bumps the PCA basis is fit on")
    args = p.parse_args()
    main(args.out_root, broad_sigma=args.broad_sigma)
