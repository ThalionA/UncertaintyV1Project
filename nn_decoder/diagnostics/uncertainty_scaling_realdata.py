# -*- coding: utf-8 -*-
"""Real-data uncertainty scaling: does the over-sharpening grow with uncertainty?
(toy fig 8b analogue, on real V1 — 2026-06-10, Theo's item; orientation axis
added 2026-06-13 per the 2026-06-10 meeting "check peakiness with orientation".)

The toy shows PCA's over-sharpening grows with input noise / location uncertainty.
The real-data version uses the per-trial STIMULUS descriptors saved in each .mat
(`trials.dispersion`, `trials.contrast`, `trials.orientation`) — independent of the
decoded/target, so there is no mechanical confound — and asks whether the decoded
peakiness ignores the stimulus-driven uncertainty. As a stimulus gets more uncertain
(higher dispersion / lower contrast) the ideal observer broadens (its max-prob drops);
a calibrated decoder should follow, an over-sharpening one should not.

Orientation is a LINEAR 0–90° axis (0 and 90° are the task references / prior modes),
so the IO is most confident near 0/90° and broadest near the 45° category boundary
(where the posterior is bimodal). The orientation panels therefore test whether the
decoder tracks this boundary-driven uncertainty, or over-sharpens worst exactly where
the IO is least certain.

loss_comparison_v1, spatial decoder (loss alone), 5 losses, 6 mice pooled. Per
trial: decoded max-prob (peakiness), IO-target max-prob (the ideal), and the
over-confidence ratio = decoded / target max-prob.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  uncertainty_scaling_realdata.png   (2×3: peakiness & ratio vs disp / con / ori)

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

# Per-stimulus-variable: (key, axis label, n bins). Orientation is a discrete
# 9-level 0–90° design, so show every level; disp/contrast are 4-level.
VARS = [('disp', 'stimulus dispersion (→ more uncertain)', 7),
        ('con',  'stimulus contrast (→ less uncertain)', 7),
        ('ori',  'stimulus orientation (deg, 0/90 = references)', 9)]


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _trial_field(tr, key, n):
    """Per-trial stimulus descriptor, nan-filled if the field is absent."""
    v = tr.get(key) if hasattr(tr, 'get') else (tr[key] if key in tr else None)
    if v is None:
        return np.full(n, np.nan)
    v = np.asarray(v, float).ravel()
    return v if v.size == n else np.full(n, np.nan)


def collect(results_root, run, split, arch='spat'):
    """pooled per-trial arrays per loss: decoded max-prob, target max-prob,
    dispersion, contrast, orientation. Target arrays are loss-invariant (PCA cell)."""
    out = {}
    for loss in LOSSES:
        f = Path(results_root) / run / _slug(loss) / f'{split}.mat'
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get('results')
        if not isinstance(res, dict):
            continue
        dec_mp, tgt_mp, disp, con, ori = [], [], [], [], []
        for mk in sorted(res):
            D = res[mk]['Dist'][arch]
            tr = res[mk]['trials']
            dec = np.asarray(D['decoded'], float)
            n = dec.shape[0]
            dec_mp.append(dec.max(1))
            tgt_mp.append(np.asarray(D['target'], float).max(1))
            disp.append(_trial_field(tr, 'dispersion', n))
            con.append(_trial_field(tr, 'contrast', n))
            ori.append(_trial_field(tr, 'orientation', n))
        out[loss] = {'dec': np.concatenate(dec_mp), 'tgt': np.concatenate(tgt_mp),
                     'disp': np.concatenate(disp), 'con': np.concatenate(con),
                     'ori': np.concatenate(ori)}
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
    ref = data['PCA'] if 'PCA' in data else next(iter(data.values()))

    fig, axes = plt.subplots(2, 3, figsize=ps.figsize(3, 2))

    # Row 0 — decoded peakiness vs each stimulus variable, with the IO target
    #         reference (loss-invariant) that a calibrated decoder should track.
    # Row 1 — over-confidence ratio (decoded / IO-target max-prob): >1 = the decoder
    #         is sharper than the ideal; tracks WHERE each loss over-commits.
    for j, (key, xlab, nb) in enumerate(VARS):
        have = np.isfinite(ref[key]).any()
        ax0, ax1 = axes[0, j], axes[1, j]
        if not have:
            for ax in (ax0, ax1):
                ax.text(0.5, 0.5, f'{key}\nnot available', ha='center', va='center',
                        transform=ax.transAxes)
                ax.axis('off')
            continue

        for loss in LOSSES:
            if loss not in data:
                continue
            cx, m, s = _binned_mean(data[loss][key], data[loss]['dec'], nb)
            ax0.errorbar(cx, m, yerr=s, color=LCOL[loss], lw=2, marker='o', ms=4,
                         capsize=2, label=ps.loss_label(loss))
        cx, mt, _ = _binned_mean(ref[key], ref['tgt'], nb)
        ax0.plot(cx, mt, color='k', ls='--', lw=1.6, marker='s', ms=3, label='IO target')
        ax0.set_xlabel(xlab); ax0.set_ylabel('decoded peakiness (mean max-prob)')
        if j == 0:
            ax0.legend(fontsize=7.5, loc='best')

        cxr, mt_ref, _ = _binned_mean(ref[key], ref['tgt'], nb)
        for loss in LOSSES:
            if loss not in data:
                continue
            cx, md, _ = _binned_mean(data[loss][key], data[loss]['dec'], nb)
            ax1.plot(cx, md / mt_ref, color=LCOL[loss], lw=2, marker='o', ms=4, label=ps.loss_label(loss))
        ax1.axhline(1.0, color='0.5', ls=':', lw=1)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel('over-confidence ratio\n(decoded / target max-prob)')

    ps.label_panels(axes.ravel())
    fig.suptitle('Real V1: decoded peakiness vs stimulus dispersion / contrast / '
                 f'orientation  (spatial, 6 mice, {ntot} trials)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'uncertainty_scaling_realdata')

    # numeric: over-confidence ratio (decoded/target) low→high level, per variable.
    for key, xlab, nb in VARS:
        if not np.isfinite(ref[key]).any():
            print(f'{key}: not available'); continue
        cxr, mt_ref, _ = _binned_mean(ref[key], ref['tgt'], nb)
        lo, hi = cxr[0], cxr[-1]
        print(f'\nover-confidence ratio decoded/target vs {key} '
              f'(level {lo:.3g} → {hi:.3g}):')
        for loss in LOSSES:
            if loss not in data:
                continue
            cx, md, _ = _binned_mean(data[loss][key], data[loss]['dec'], nb)
            r = md / mt_ref
            g = np.isfinite(r)
            print(f'  {loss:12s} {r[g][0]:.2f}× → {r[g][-1]:.2f}×   '
                  f'(min {np.nanmin(r):.2f} at {cx[np.nanargmin(r)]:.3g}, '
                  f'max {np.nanmax(r):.2f} at {cx[np.nanargmax(r)]:.3g})')
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
