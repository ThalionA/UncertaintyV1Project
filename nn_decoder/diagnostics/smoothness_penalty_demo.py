# -*- coding: utf-8 -*-
"""Output-smoothness penalty (method 2): does it tame PCA's over-sharpening?
(2026-06-16 — the loss-side fix, vs the smooth=0 baseline.)

`smooth_lambda` adds Σᵢ(pᵢ₊₁−pᵢ)² (Dirichlet energy of the decoded posterior) to the
training loss, penalising exactly the high-frequency spikes the PCA loss leaves free.
Compares a smooth run to its smooth=0 baseline (default: smoothtest_smooth0p1 vs noreg,
both PCA, patience 0, 200 ep) on peakiness + KL(decoded‖target), with example posteriors.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  smoothness_penalty_demo.png

Usage:  python diagnostics/smoothness_penalty_demo.py
        python diagnostics/smoothness_penalty_demo.py --base loss_comparison_v1 --smooth smoothsweep_smooth0p1
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
from cross_loss_eval import _eval_one, _agg  # noqa: E402

ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
THETA = np.arange(91.0)


def _cell(run, split):
    return f'results/{run}/Q_PCA_half_100ms_all/{split}.mat'


def _stats(run, arch, split):
    res = sio.loadmat(_cell(run, split), simplify_cells=True)['results']
    pk, kl, tg = [], [], []
    for mk in sorted(res):
        D = res[mk]['Dist']
        dec, t = np.asarray(D[arch]['decoded'], float), np.asarray(D[arch]['target'], float)
        pk.append(float(np.nanmean(dec.max(1)))); tg.append(float(np.nanmean(t.max(1))))
        kl.append(_eval_one(dec, t, 'KL', D.get('pcs'), D.get('explained_var')))
    return _agg(pk)[0], _agg(kl)[0], _agg(tg)[0]


def main(base, smooth, split, out_root, mouse='mouse_0', ncol=4):
    ps.apply()
    fig, axes = plt.subplots(len(ARCHS), ncol,
                             figsize=ps.figsize(ncol, len(ARCHS), panel_w=1.9, panel_h=1.5), sharex=True)
    axes = np.atleast_2d(axes)
    for r, (arch, alabel) in enumerate(ARCHS):
        rb = sio.loadmat(_cell(base, split), simplify_cells=True)['results'][mouse]['Dist'][arch]
        rs = sio.loadmat(_cell(smooth, split), simplify_cells=True)['results'][mouse]['Dist'][arch]
        tgt = np.asarray(rb['target'], float)
        db, dsm = np.asarray(rb['decoded'], float), np.asarray(rs['decoded'], float)
        sel = np.argsort(tgt.max(1))[np.linspace(0, tgt.shape[0] - 1, ncol).round().astype(int)]
        for c, tr in enumerate(sel):
            ax = axes[r, c]
            ax.fill_between(THETA, tgt[tr], color='0.85', lw=0, zorder=0)
            ax.plot(THETA, db[tr], color=ps.PCA_EVAR, lw=1.1, label='PCA (smooth=0)')
            ax.plot(THETA, dsm[tr], color='#1b7837', lw=1.5, label='PCA + smoothness')
            ps.cap_posterior_ylim(ax, float(tgt[tr].max()), mult=3.0)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'target mp {tgt[tr].max():.3f}', fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f'{alabel}\nPCA posterior', fontsize=8)
            if r == len(ARCHS) - 1:
                ax.set_xlabel('orientation (deg)', fontsize=8)
    axes[0, 0].legend(fontsize=6.5, loc='upper right')
    fig.suptitle(f'Output-smoothness penalty tames PCA over-sharpening '
                 f'({smooth} vs {base}, {mouse})', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'smoothness_penalty_demo')

    print(f"{'arch':9s} | {'peakiness base→smooth (tgt)':>30s} | {'KL base→smooth':>16s}")
    for arch, alabel in ARCHS:
        p0, k0, tg = _stats(base, arch, split); p1, k1, _ = _stats(smooth, arch, split)
        print(f"{alabel:9s} |  {p0:.3f} → {p1:.3f}  (tgt {tg:.3f})       {k0:.2f} → {k1:.2f}")
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base', default='noreg')
    ap.add_argument('--smooth', default='smoothtest_smooth0p1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.base, a.smooth, a.split, a.out_root)
