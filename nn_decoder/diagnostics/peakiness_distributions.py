# -*- coding: utf-8 -*-
"""Bulk-vs-tail peakiness distributions, spatial vs temporal, all losses
(2026-06-05, answering Máté's slide-28 reading).

Máté's question: is the SPATIAL PCA decoder's peakiness a genuine shift of the
distribution's BULK relative to the information-theoretic losses, or mostly a long
right TAIL with the bulk in the right place? This plots the full per-trial
distributions (pooled over 6 mice, Q / half / 100 ms) so the median/IQR — not just
the mean — are visible, two ways:

  * max(P): the spatial PCA bulk is clearly shifted (median ~3x the calibrated
    losses and the target);
  * 1 − H(P)/log N: the gap looks SMALLER, because the spatial posteriors are
    jagged/MULTIMODAL — a sharp spike plus secondary bumps keeps entropy moderate
    even when max(P) is high. That is exactly why max(P) is the right read for the
    spatial decoder, and likely why the entropy panel reads as "bulk overlaps".

Panel C contrasts the per-loss spatial-vs-temporal medians: PCA (and Wasserstein)
have spat != temp (both over-peaky, by different amounts), while CE/KL/JS have
spat ~= temp ~= target (both calibrated).

Output (PNG+SVG) under figures/peakiness_scatter/:
  bulk_vs_tail.png

Usage:  python diagnostics/peakiness_distributions.py
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

NCAT = 91
CELLS = {'PCA': 'Q_PCA_half_100ms_all', 'CE': 'Q_CE_half_100ms',
         'KL': 'Q_KL_half_100ms', 'JS': 'Q_JS_half_100ms',
         'Wasserstein': 'Q_Wasserstein_half_100ms'}
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}


def _H(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def peak(p, metric):
    return p.max(-1) if metric == 'maxP' else 1.0 - _H(p) / np.log(NCAT)


def load_all(results_root, run, split):
    """pooled[loss][arch] = per-trial decoded; targets pooled separately."""
    pooled, tgt = {}, []
    for loss, slug in CELLS.items():
        f = Path(results_root) / run / slug / f'{split}.mat'
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True)['results']
        pooled[loss] = {'spat': [], 'temp': []}
        for mk in sorted(res):
            D = res[mk]['Dist']
            for a in ('spat', 'temp'):
                pooled[loss][a].append(np.asarray(D[a]['decoded'], float))
            if loss == 'PCA':
                tgt.append(np.asarray(D['spat']['target'], float))
        for a in ('spat', 'temp'):
            pooled[loss][a] = np.concatenate(pooled[loss][a], axis=0)
    return pooled, np.concatenate(tgt, axis=0)


def main(results_root, run, split, out_root):
    ps.apply()
    pooled, tgt = load_all(results_root, run, split)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panels A,B: spatial distributions under max(P) and 1-H/logN
    for ax, metric, lab in ((axes[0], 'maxP', 'spatial peakiness   max(P)'),
                            (axes[1], 'normH', 'spatial peakiness   1 − H/log N')):
        tg = peak(tgt, metric)
        hi = np.percentile(peak(pooled['PCA']['spat'], metric), 99)
        bins = np.linspace(0, hi, 46)
        for loss in ('CE', 'KL', 'JS', 'Wasserstein', 'PCA'):
            v = peak(pooled[loss]['spat'], metric)
            lw = 2.6 if loss == 'PCA' else 1.6
            ax.hist(v, bins=bins, density=True, histtype='step', color=LCOL[loss],
                    lw=lw, label=f'{loss} (med {np.median(v):.3f})')
            ax.axvline(np.median(v), color=LCOL[loss], ls='-', lw=1.0, alpha=0.35)
        ax.axvline(tg.mean(), color='k', ls=':', lw=1.4, label=f'IO target ({tg.mean():.3f})')
        ax.set_xlabel(lab); ax.set_ylabel('density'); ax.set_xlim(0, hi)
        ax.legend(fontsize=7.5)
    cal = np.median(np.concatenate([peak(pooled[l]['spat'], 'maxP') for l in ('CE', 'KL', 'JS')]))
    pca = np.median(peak(pooled['PCA']['spat'], 'maxP'))
    axes[0].set_title(f'BULK is shifted, not just the tail:\n'
                      f'PCA median {pca:.3f} = {pca/cal:.1f}× the CE/KL/JS median {cal:.3f}',
                      fontsize=10.5, loc='left')
    axes[1].set_title('Entropy understates it — spatial posteriors are\n'
                      'jagged/multimodal, so H stays up while max(P) spikes',
                      fontsize=10.5, loc='left')

    # Panel C: per-loss spatial vs temporal median max(P)
    ax = axes[2]
    losses = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
    x = np.arange(len(losses))
    msp = [np.median(peak(pooled[l]['spat'], 'maxP')) for l in losses]
    mtp = [np.median(peak(pooled[l]['temp'], 'maxP')) for l in losses]
    ax.bar(x - 0.2, msp, 0.38, color=[LCOL[l] for l in losses], alpha=0.55, label='spatial (PPC)')
    ax.bar(x + 0.2, mtp, 0.38, color=[LCOL[l] for l in losses], alpha=0.9, hatch='//',
           label='temporal (SBC)')
    ax.axhline(peak(tgt, 'maxP').mean(), color='k', ls=':', lw=1.4, label='IO target')
    ax.set_xticks(x); ax.set_xticklabels(losses, rotation=20, fontsize=9)
    ax.set_ylabel('median max(P)')
    ax.set_title('spat≠temp under PCA/Wass = both over-peaky\n'
                 'by different amounts; CE/KL/JS: both ≈ target', fontsize=10.5, loc='left')
    ax.legend(fontsize=8)

    fig.suptitle('Spatial PCA peakiness: where the distribution actually sits '
                 f'(Q half 100ms, 6 mice pooled, {pooled["PCA"]["spat"].shape[0]} trials)',
                 y=1.03, fontsize=12.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'bulk_vs_tail')

    # numeric readout
    print('median max(P)   spat / temp   (target '
          f'{peak(tgt,"maxP").mean():.3f}):')
    for l in losses:
        print(f'  {l:12s} {np.median(peak(pooled[l]["spat"],"maxP")):.3f} / '
              f'{np.median(peak(pooled[l]["temp"],"maxP")):.3f}')
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
