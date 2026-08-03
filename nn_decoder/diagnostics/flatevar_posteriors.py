# -*- coding: utf-8 -*-
"""Example fitted posteriors for the surviving `flatevar_v1` decoders — including
the individual temporal-bin posteriors.

Two figures.

  fig7  OVERLAY — for each decoder, the IO target against the spatial and temporal
        decoded posteriors, on the same example trials for every decoder so the
        comparison is like for like.
  fig8  TEMPORAL BINS — the per-bin posteriors the temporal decoder actually emits
        (`decoded_samp`, 91 orientations x 10 time bins) as a heatmap, with the IO
        target and the bin-averaged decode overlaid. This is what the Jensen average
        is averaging over, and it is invisible in fig7.

TRIAL SELECTION (2026-07-29 meeting ask: "select trials by diff from temp bins").
Trials are ranked by how much the per-bin posterior CHANGES across time bins —
mean total-variation distance between consecutive bins — and the most dynamic are
shown. Selection is computed once, on a reference decoder (KL, the calibrated one),
and the SAME trial indices are then used for every decoder. Selecting per decoder
would let each one pick the trials that flatter it.

Outputs (PNG+SVG) under figures/flatevar/.
Usage:
    python diagnostics/flatevar_posteriors.py
    python diagnostics/flatevar_posteriors.py --mouse mouse_1 --n-trials 5
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flatevar_assessment import _path  # noqa: E402

# Which decoders to show, in story order. Kept short on purpose — a gallery with
# every cell is unreadable, and these four carry the whole contrast.
SHOW = [
    ('evar  (H=8, wd 0)',  ('RAW', 'hpsweep_v2/lam0p003_drop0_acttanh_h8_pat0_vf0p2_wd0_shp0/'
                            'Q_PCA_half_100ms_all'), ps.PCA_EVAR),
    ('FLAT  (H=8, wd 0)',  ('flatevar_v1', 'A_flat_wd0'),     ps.FLAT_EVAR),
    ('FLAT  (linear, wd 0)', ('flatevar_v1', 'B_flat_lin_wd0'), ps.FLAT_EVAR),
    ('KL  (H=8)',          ('flatevar_v1', 'R_reference_kl'), ps.KL),
]
REF_FOR_SELECTION = ('flatevar_v1', 'R_reference_kl')


def _load(results_root, spec):
    p = _path(results_root, spec)
    if p is None:
        return None
    return sio.loadmat(p, simplify_cells=True)['results']


def _pick_trials(res, mouse, n):
    """Most temporally dynamic trials: mean TV distance between consecutive bins."""
    samp = np.asarray(res[mouse]['Dist']['temp']['decoded_samp'], float)
    if samp.ndim != 3 or samp.shape[2] < 2:
        raise SystemExit('decoded_samp has no time-bin axis — cannot rank by dynamism.')
    # samp is (trials, orientations, bins); TV = 0.5 * sum|p_t - p_{t+1}|
    tv = 0.5 * np.abs(np.diff(samp, axis=2)).sum(axis=1)      # (trials, bins-1)
    score = np.nanmean(tv, axis=1)
    ok = np.isfinite(score)
    idx = np.argsort(-np.where(ok, score, -np.inf))[:n]
    return np.sort(idx), score


def fig_overlay(results_root, out_root, mouse, trials, cells):
    ps.apply()
    nrow, ncol = len(cells), len(trials)
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow),
                             sharex=True, squeeze=False)
    theta = np.linspace(0, 90, 91)
    for ri, (lab, res, colr) in enumerate(cells):
        D = res[mouse]['Dist']
        for ci, t in enumerate(trials):
            ax = axes[ri][ci]
            tgt = np.asarray(D['spat']['target'], float)[t]
            ax.fill_between(theta, 0, tgt, color='0.75', lw=0, zorder=1,
                            label='IO target' if (ri == 0 and ci == 0) else None)
            ax.plot(theta, np.asarray(D['spat']['decoded'], float)[t], lw=1.4,
                    color=ps.SPATIAL, zorder=3,
                    label='spatial' if (ri == 0 and ci == 0) else None)
            ax.plot(theta, np.asarray(D['temp']['decoded'], float)[t], lw=1.4,
                    color=ps.TEMPORAL, zorder=3,
                    label='temporal' if (ri == 0 and ci == 0) else None)
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f'trial {t}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            if ri == nrow - 1:
                ax.set_xlabel('orientation (deg)', fontsize=7.5)
    axes[0][0].legend(fontsize=6, frameon=True, loc='upper right')
    ps.label_panels(axes.ravel())
    fig.suptitle(f'Fitted posteriors, {mouse} — same trials for every decoder, chosen as the most '
                 'temporally dynamic (largest change across time bins). y-scales are free.',
                 y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'flatevar_fig7_example_posteriors')


def fig_temporal_bins(results_root, out_root, mouse, trials, cells):
    """The per-bin posteriors, as heatmaps — meeting ask 2026-07-29 #4."""
    ps.apply()
    cells = [c for c in cells]
    nrow, ncol = len(cells), len(trials)
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow),
                             squeeze=False)
    for ri, (lab, res, colr) in enumerate(cells):
        D = res[mouse]['Dist']
        samp = np.asarray(D['temp']['decoded_samp'], float)
        tgt_all = np.asarray(D['temp']['target'], float)
        for ci, t in enumerate(trials):
            ax = axes[ri][ci]
            bins = samp[t]                              # (orientations, time bins)
            # Shared colour scale within a row so bins are comparable across trials
            vmax = np.nanpercentile(samp[trials], 99.5)
            im = ax.imshow(bins, aspect='auto', origin='lower', cmap='magma',
                           vmin=0, vmax=vmax,
                           extent=[0, bins.shape[1], 0, 90])
            # IO target as a line on the same orientation axis
            tp = tgt_all[t]
            ax.plot(bins.shape[1] * tp / max(tp.max(), 1e-12) * 0.28,
                    np.linspace(0, 90, tp.size), color='w', lw=1.2, alpha=0.9,
                    label='IO target' if (ri == 0 and ci == 0) else None)
            ax.set_xticks(np.arange(0, bins.shape[1] + 1, 3))
            if ri == 0:
                ax.set_title(f'trial {t}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{lab}\norientation (deg)', fontsize=7)
            else:
                ax.set_yticklabels([])
            if ri == nrow - 1:
                ax.set_xlabel('time bin', fontsize=7.5)
            if ci == ncol - 1:
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=5)
    axes[0][0].legend(fontsize=6, frameon=True, loc='upper right')
    ps.label_panels(axes.ravel())
    fig.suptitle(f'Individual temporal-bin posteriors, {mouse} — what the Jensen average averages '
                 'over. White line = IO target (scaled). Colour scale shared within a row.',
                 y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'flatevar_fig8_temporal_bins')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/flatevar')
    ap.add_argument('--mouse', default=None, help='e.g. mouse_0 (default: first)')
    ap.add_argument('--n-trials', type=int, default=4)
    a = ap.parse_args()

    cells = []
    for lab, spec, colr in SHOW:
        r = _load(a.results_root, spec)
        if r is None:
            print(f"  [skip] {lab}: not downloaded")
            continue
        cells.append((lab, r, colr))
    if not cells:
        raise SystemExit('no cells available')

    ref = _load(a.results_root, REF_FOR_SELECTION)
    mice = sorted(k for k in ref if isinstance(ref[k], dict)
                  and isinstance(ref[k].get('Dist'), dict))
    # Only animals present in EVERY shown cell, so the same trials exist throughout.
    for _, r, _c in cells:
        mice = [m for m in mice if m in r]
    mouse = a.mouse or mice[0]
    if mouse not in mice:
        raise SystemExit(f'{mouse} not in every cell; available: {mice}')

    trials, score = _pick_trials(ref, mouse, a.n_trials)
    print(f"  {mouse}: showing trials {list(trials)} "
          f"(temporal-dynamism score {np.round(score[trials], 3)}; "
          f"cohort median {np.nanmedian(score):.3f})")

    fig_overlay(a.results_root, a.out_root, mouse, trials, cells)
    fig_temporal_bins(a.results_root, a.out_root, mouse, trials, cells)


if __name__ == '__main__':
    main()
