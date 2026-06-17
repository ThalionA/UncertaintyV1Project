# -*- coding: utf-8 -*-
"""Item 4 — how similar are the temporal decoder's per-time-bin posteriors, in
LOCATION and WIDTH, across the five losses and across λ_entropy?

The temporal (SBC) decoder emits T=10 per-bin posteriors whose average is the
trial posterior. "Sampling" would mean the 10 bins are DIFFERENT draws — peaks at
spread-out locations, each bin sharp — so the bins are *dissimilar*. The null is
"every bin ≈ the time-average" — the bins are near-identical copies. We quantify
bin-to-bin (dis)similarity directly, per trial, on the orientation grid θ (0–90°):

  per bin t:  μ_t = mean_θ,   σ_t = width (circular-ish std) of bin t
  LOCATION dispersion = std_t(μ_t)   — how much the bin peaks move (deg)
  WIDTH    dispersion = std_t(σ_t)   — how much the bin widths differ (deg)
  mean per-bin width  = mean_t(σ_t)  — how sharp each bin is (peakiness context)

averaged over trials, then over mice (mean ± sem, n mice). Small dispersion =
bins are near-identical (no sampling); large = bins genuinely differ. Swept
against λ_H to see whether the entropy penalty manufactures bin-to-bin variety.

Two figures (PNG+SVG):
  temporal_bin_similarity.png   location- & width-dispersion + mean per-bin
                                width vs λ_H, per loss (+ IO-target width refs)
  temporal_bin_examples.png     example trial's 10 per-bin posteriors per loss ×
                                λ_H — TWIN y-axis (broad target/avg on the left,
                                peaky per-bin posteriors on the right) so the
                                peaky bins are shown at full height, UNCLIPPED.

Reads the `lambdaH_sweep_entlam<λ>` runs (temporal arch, `decoded_samp`).

Run:  cd nn_decoder && OMP_NUM_THREADS=1 \
        python diagnostics/temporal_bin_similarity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps                                          # noqa: E402
from cross_loss_eval import _agg                                     # noqa: E402
from diagnostics.lambda_h_sampling_test import _moments, THETA       # noqa: E402
from diagnostics.lambda_h_temporal_sweep import _lambda_of, _find_cell  # noqa: E402

LOSSES = ["PCA", "CE", "KL", "JS", "Wasserstein"]
LCOL = {"PCA": ps.PCA_EVAR, "CE": ps.CE, "KL": ps.KL, "JS": ps.JS,
        "Wasserstein": ps.WASSERSTEIN}


def collect(results_root, prefix, split):
    """{lambda: {loss: {metric: [per-mouse means]}}} for bin-similarity metrics."""
    runs = sorted(Path(results_root).glob(f"{prefix}_entlam*"),
                  key=lambda p: (_lambda_of(p.name)
                                 if _lambda_of(p.name) is not None else 1e9))
    data = {}
    for rd in runs:
        lam = _lambda_of(rd.name)
        if lam is None:
            continue
        for loss in LOSSES:
            f = _find_cell(rd, loss, split)
            if f is None:
                continue
            res = sio.loadmat(str(f), simplify_cells=True).get("results")
            if not isinstance(res, dict):
                continue
            rec = {k: [] for k in ("loc_disp", "wid_disp", "mean_wid", "tgt_wid")}
            for mk in sorted(res):
                D = res[mk]["Dist"]["temp"]
                ds = np.asarray(D["decoded_samp"], float)        # (n, 91, T)
                tgt = np.asarray(D["target"], float)             # (n, 91)
                mu_t, var_t = _moments(ds, 1)                    # (n, T) each
                sd_t = np.sqrt(np.clip(var_t, 0, None))          # (n, T) per-bin width
                rec["loc_disp"].append(float(np.nanmean(np.nanstd(mu_t, axis=1))))
                rec["wid_disp"].append(float(np.nanmean(np.nanstd(sd_t, axis=1))))
                rec["mean_wid"].append(float(np.nanmean(np.nanmean(sd_t, axis=1))))
                _, tgt_var = _moments(tgt, 1)
                rec["tgt_wid"].append(float(np.nanmean(np.sqrt(np.clip(tgt_var, 0, None)))))
            data.setdefault(lam, {})[loss] = rec
    return data


def fig_quant(data, out_root):
    ps.apply()
    lams = sorted(data)
    x = np.arange(len(lams))
    xlabels = [f"{l:g}" for l in lams]
    tgt_wid = np.nanmean([_agg(data[l][L]["tgt_wid"])[0] for l in lams for L in data[l]])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    panels = [
        ("loc_disp", "between-bin LOCATION dispersion  std$_t(\\mu_t)$  (deg)",
         "Do the bin peaks MOVE?\n(0 = bins share a location)"),
        ("wid_disp", "between-bin WIDTH dispersion  std$_t(\\sigma_t)$  (deg)",
         "Do the bin widths DIFFER?\n(0 = bins share a width)"),
        ("mean_wid", "mean per-bin width  mean$_t(\\sigma_t)$  (deg)",
         "How sharp is each bin?\n(well below target = peaky)"),
    ]
    for ax, (key, ylab, title) in zip(axes, panels):
        for loss in LOSSES:
            ys = [_agg(data[l].get(loss, {}).get(key, []))[0] for l in lams]
            es = [_agg(data[l].get(loss, {}).get(key, []))[1] for l in lams]
            ax.errorbar(x, ys, yerr=es, color=LCOL[loss], lw=2, marker="o", ms=4,
                        capsize=2, label=loss)
        if key in ("loc_disp", "mean_wid"):
            ax.axhline(tgt_wid, color="k", ls=":", lw=1.4,
                       label="IO target width" if key == "loc_disp" else None)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha="right")
        ax.set_xlabel(r"entropy penalty $\lambda_H$")
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7, loc="best")
    ps.label_panels(axes)
    fig.suptitle("Temporal-bin similarity in location & width across λ_H "
                 f"(IO target width ≈ {tgt_wid:.0f}°; bins are near-identical "
                 "copies unless these rise)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ps.save_fig(fig, Path(out_root), "temporal_bin_similarity", layout=None)

    print(f"IO target width ≈ {tgt_wid:.1f} deg")
    print(f"{'loss':12s} {'λ_H':>6s} | {'loc_disp':>8s} {'wid_disp':>8s} {'mean_wid':>8s}")
    for loss in LOSSES:
        for lam in lams:
            ld = _agg(data[lam].get(loss, {}).get("loc_disp", []))[0]
            wd = _agg(data[lam].get(loss, {}).get("wid_disp", []))[0]
            mw = _agg(data[lam].get(loss, {}).get("mean_wid", []))[0]
            print(f"  {loss:10s} {lam:6.3f} | {ld:8.1f} {wd:8.1f} {mw:8.1f}")


def fig_examples(results_root, prefix, split, out_root, mouse="mouse_0",
                 lam_cols=(0.0, 0.01, 0.1)):
    """Twin-axis gallery: broad target+avg on the LEFT axis (so they stay
    visible), the 10 peaky per-bin posteriors on the RIGHT axis (full height,
    UNCLIPPED). Rows = losses, cols = λ_H."""
    ps.apply()
    runs = {(_lambda_of(p.name)): p
            for p in Path(results_root).glob(f"{prefix}_entlam*")
            if _lambda_of(p.name) is not None}
    cols = [l for l in lam_cols if l in runs]
    if not cols:                                # fall back to whatever exists
        cols = sorted(runs)[:3]
    if not cols:
        print("examples: no λ_H runs found — skipped."); return

    # pick one broad-target example trial from the PCA cell of the lowest-λ run
    fref = _find_cell(runs[min(runs)], "PCA", split)
    res = sio.loadmat(str(fref), simplify_cells=True)["results"]
    tgt0 = np.asarray(res[mouse]["Dist"]["temp"]["target"], float)
    sel = int(np.nanargmax([_moments(tgt0[i:i + 1], 1)[1][0]
                            for i in range(tgt0.shape[0])]))   # widest target

    nR, nC = len(LOSSES), len(cols)
    fig, axes = plt.subplots(nR, nC, figsize=(2.6 * nC + 1.4, 1.45 * nR + 1.0),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for r, loss in enumerate(LOSSES):
        for c, lam in enumerate(cols):
            ax = axes[r, c]
            f = _find_cell(runs[lam], loss, split)
            if f is None:
                ax.axis("off"); continue
            D = sio.loadmat(str(f), simplify_cells=True)["results"][mouse]["Dist"]["temp"]
            ds = np.asarray(D["decoded_samp"], float)[sel]    # (91, T)
            dec = np.asarray(D["decoded"], float)[sel]         # (91,)
            tg = np.asarray(D["target"], float)[sel]           # (91,)
            # LEFT axis: target (grey) + time-average (black), scaled to target
            ax.fill_between(THETA, tg, color="0.85", lw=0, zorder=0)
            ax.plot(THETA, dec, color="k", lw=1.6, zorder=3)
            ax.set_ylim(0, max(np.nanmax(tg), np.nanmax(dec)) * 1.25)
            ax.set_yticks([])
            # RIGHT axis (twin): the 10 per-bin posteriors at FULL height
            axR = ax.twinx()
            T = ds.shape[1]
            for t in range(T):
                axR.plot(THETA, ds[:, t], color=plt.cm.viridis(t / max(T - 1, 1)),
                         lw=0.8, alpha=0.85)
            axR.set_ylim(0, np.nanmax(ds) * 1.05)
            axR.set_yticks([])
            if r == 0:
                ax.set_title(f"λ_H = {lam:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel(loss, fontsize=9, color=LCOL[loss])
            if r == nR - 1:
                ax.set_xlabel("orientation (deg)", fontsize=8)

    handles = [Line2D([0], [0], color="0.7", lw=6, label="IO target (left axis)"),
               Line2D([0], [0], color="k", lw=1.6, label="time-average (left axis)"),
               Line2D([0], [0], color=plt.cm.viridis(0.5), lw=1.2,
                      label="10 per-bin posteriors (right axis, unclipped)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-bin posteriors vs the time-average & target — TWIN axes so "
                 f"peaky bins are unclipped ({mouse}, widest-target trial #{sel})",
                 y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    ps.save_fig(fig, Path(out_root), "temporal_bin_examples", layout=None)
    print(f"\nexamples: trial #{sel} (widest target), {mouse}, "
          f"λ_H cols {', '.join(f'{l:g}' for l in cols)}")


def main(results_root, prefix, split, out_root):
    data = collect(results_root, prefix, split)
    if not data:
        raise SystemExit(f"no {prefix}_entlam* runs under {results_root}.")
    fig_quant(data, out_root)
    fig_examples(results_root, prefix, split, out_root)
    print(f"\nDone. {Path(out_root).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", default="lambdaH_sweep")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-root", default="figures/peakiness_scatter")
    a = ap.parse_args()
    main(a.results_root, a.prefix, a.split, a.out_root)
