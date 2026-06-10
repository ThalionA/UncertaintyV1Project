# -*- coding: utf-8 -*-
"""Real-data uncertainty scaling: does the over-sharpening grow with uncertainty?
(toy fig 8b analogue, on real V1 — 2026-06-10, Theo's item.)

The toy shows PCA's over-sharpening grows with input noise / location uncertainty.
The real-data version uses the per-trial STIMULUS uncertainty saved in each .mat
(`trials.dispersion`, `trials.contrast`) — independent of the decoded/target, so
there is no mechanical confound — and asks whether the decoded peakiness ignores
it. As a stimulus gets more uncertain (higher dispersion / lower contrast) the
ideal observer broadens (its max-prob drops); a calibrated decoder should follow,
an over-sharpening one should not.

loss_comparison_v1, spatial decoder (loss alone), 5 losses, 6 mice pooled. Per
trial: decoded max-prob (peakiness), IO-target max-prob (the ideal), and the
over-sharpening = decoded − target max-prob.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  uncertainty_scaling_realdata.png

Usage:  python diagnostics/uncertainty_scaling_realdata.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def collect(results_root, run, split, arch='spat'):
    """pooled per-trial arrays per loss: decoded max-prob, target max-prob,
    dispersion, contrast. Target arrays are loss-invariant (read from PCA cell)."""
    out = {}
    for loss in LOSSES:
        f = Path(results_root) / run / _slug(loss) / f'{split}.mat'
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get('results')
        if not isinstance(res, dict):
            continue
        dec_mp, tgt_mp, disp, con = [], [], [], []
        for mk in sorted(res):
            D = res[mk]['Dist'][arch]
            tr = res[mk]['trials']
            dec_mp.append(np.asarray(D['decoded'], float).max(1))
            tgt_mp.append(np.asarray(D['target'], float).max(1))
            disp.append(np.asarray(tr['dispersion'], float))
            con.append(np.asarray(tr['contrast'], float))
        out[loss] = {'dec': np.concatenate(dec_mp), 'tgt': np.concatenate(tgt_mp),
                     'disp': np.concatenate(disp), 'con': np.concatenate(con)}
    return out


def _bins(x, nmax=7):
    """Unique levels if few (discrete stimulus design), else quantile-bin edges.
    Returns (centers, assign) where assign[i] is the bin index of x[i]."""
    u = np.unique(x[np.isfinite(x)])
    if len(u) <= nmax:
        return u, np.searchsorted(u, x)
    q = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, nmax + 1))
    q[-1] += 1e-9
    assign = np.clip(np.digitize(x, q) - 1, 0, nmax - 1)
    centers = np.array([x[(assign == b)].mean() if (assign == b).any() else np.nan
                        for b in range(nmax)])
    return centers, assign


def _binned_mean(x, y, nmax=7):
    centers, assign = _bins(x, nmax)
    m = np.array([np.nanmean(y[assign == b]) if (assign == b).any() else np.nan
                  for b in range(len(centers))])
    s = np.array([np.nanstd(y[assign == b], ddof=1) / np.sqrt(max((assign == b).sum(), 1))
                  if (assign == b).sum() > 1 else 0.0 for b in range(len(centers))])
    return centers, m, s


def main(results_root, run, split, out_root):
    ps.apply()
    data = collect(results_root, run, split)
    if not data:
        raise SystemExit('no loss cells found.')
    ntot = len(next(iter(data.values()))['dec'])
    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1))

    # (a) peakiness vs stimulus DISPERSION, with the IO target reference
    # (b) peakiness vs stimulus CONTRAST
    for ax, key, xlab in ((axes[0], 'disp', 'stimulus dispersion (→ more uncertain)'),
                          (axes[1], 'con', 'stimulus contrast (→ less uncertain)')):
        for loss in LOSSES:
            if loss not in data:
                continue
            cx, m, s = _binned_mean(data[loss][key], data[loss]['dec'])
            ax.errorbar(cx, m, yerr=s, color=LCOL[loss], lw=2, marker='o', ms=4,
                        capsize=2, label=loss)
        # IO target peakiness (loss-invariant) — the ideal each decoder should track
        ref = data['PCA'] if 'PCA' in data else next(iter(data.values()))
        cx, mt, _ = _binned_mean(ref[key], ref['tgt'])
        ax.plot(cx, mt, color='k', ls='--', lw=1.6, marker='s', ms=3, label='IO target')
        ax.set_xlabel(xlab); ax.set_ylabel('decoded peakiness (mean max-prob)')
        if ax is axes[0]:
            ax.legend(fontsize=7.5, loc='best')

    # (c) over-confidence RATIO (decoded / target max-prob, per dispersion bin) —
    #     the RELATIVE over-sharpening. Max-prob is bounded, so the absolute gap
    #     compresses, but the ratio (× the ideal) grows as the IO broadens — the
    #     toy fig-8b result, in the scale-free metric.
    ax = axes[2]
    ref = data['PCA'] if 'PCA' in data else next(iter(data.values()))
    cxr, mt_ref, _ = _binned_mean(ref['disp'], ref['tgt'])
    for loss in LOSSES:
        if loss not in data:
            continue
        cx, md, _ = _binned_mean(data[loss]['disp'], data[loss]['dec'])
        ax.plot(cx, md / mt_ref, color=LCOL[loss], lw=2, marker='o', ms=4, label=loss)
    ax.axhline(1.0, color='0.5', ls=':', lw=1, label='= IO target')
    ax.set_xlabel('stimulus dispersion (→ more uncertain)')
    ax.set_ylabel('over-confidence ratio  (decoded / target max-prob)')
    ax.set_title('Relative over-sharpening grows with uncertainty')

    ps.label_panels(axes)
    fig.suptitle(f'Real V1: decoded peakiness vs stimulus uncertainty '
                 f'(spatial, 6 mice, {ntot} trials)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'uncertainty_scaling_realdata')

    # numeric: over-confidence ratio (decoded/target) low→high dispersion per loss
    cxr, mt_ref, _ = _binned_mean(ref['disp'], ref['tgt'])
    print('over-confidence ratio decoded/target (low→high dispersion):')
    for loss in LOSSES:
        if loss not in data:
            continue
        cx, md, _ = _binned_mean(data[loss]['disp'], data[loss]['dec'])
        r = md / mt_ref
        good = np.isfinite(r)
        print(f'  {loss:12s} {r[good][0]:.1f}× (low) → {r[good][-1]:.1f}× (high disp)')
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
