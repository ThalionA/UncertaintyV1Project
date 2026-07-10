# -*- coding: utf-8 -*-
"""Is the overfitting a capacity-vs-data problem? Overfitting (val/train fit-loss ratio)
vs the parameters-per-training-trial ratio, pooled over mice AND hidden widths.

Motivation: every loss overfits heavily (val/train ≫ 1) because the net has ~5–8k
parameters and only ~350 training trials per mouse (11–24× overparameterised). If that
ratio is the cause — not the loss — then the overfitting should scale with params/trial
across BOTH sources of its variation: mice (neuron count / trial count differ) and hidden
width (params ∝ H). Pooling the width sweep {4,8,16,32,64} × 6 mice spans params/trial
~1.5–50, a 30× range, so the two collapse onto one trend if capacity-vs-data is the story.

params = N·H + H + H·C + C  (one-hidden-layer MLP, C=91); trials = per-mouse training-set
size (n_full − n_test from the .mat). val/train is the final-epoch fit-loss ratio.

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  overfit_vs_capacity.png
Usage:  python diagnostics/overfit_vs_capacity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

RUN_PARENT = 'hpsweep_wide'
LOSS_SLUG = {'PCA': 'Q_PCA_half_100ms_all', 'KL': 'Q_KL_half_100ms',
             'JS': 'Q_JS_half_100ms', 'Wasserstein': 'Q_Wasserstein_half_100ms'}
LOSSES = ['PCA', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
WIDTHS = [4, 8, 16, 32, 64]
C_CATS = 91
BASE_H = 32          # the per-mouse baseline (the literal "6 mice" test), ringed in the plot


def cell(H):
    return f"lam0p003_drop0_acttanh_h{H}_pat0_vf0p2"


def n_params(N, H):
    return N * H + H + H * C_CATS + C_CATS


def mouse_dims(results_root):
    """{mouse_idx: (N, n_train)} from the baseline PCA cell (constant across cells:
    same split, same neurons). n_train = n_full − n_test."""
    d = {}
    root = Path(results_root) / RUN_PARENT / cell(BASE_H) / LOSS_SLUG['PCA']
    res = sio.loadmat(str(root / 'stratified_balanced.mat'), simplify_cells=True).get('results', {})
    for mk in res:
        idx = int(str(mk).split('_')[-1])
        ck = torch.load(str(root / 'checkpoints' / f'{mk}_stratified_balanced.pt'),
                        map_location='cpu', weights_only=False)['spat']
        N = int(ck['model_params']['input_size'])
        n_test = int(np.asarray(res[mk]['Dist']['spat']['decoded']).shape[0])
        n_full = int(np.asarray(res[mk]['Dist']['spat']['full_decoded']).shape[0])
        d[idx] = (N, max(n_full - n_test, 1))
    return d


def ratio_by_mouse(results_root, loss, H, arch):
    ck_dir = Path(results_root) / RUN_PARENT / cell(H) / LOSS_SLUG[loss] / 'checkpoints'
    out = {}
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        idx = int(pt.stem.split('_')[1])
        h = (torch.load(str(pt), map_location='cpu', weights_only=False).get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if t and v and t[-1] > 0:
            out[idx] = v[-1] / t[-1]
    return out


def main(results_root, out_root):
    ps.apply()
    dims = mouse_dims(results_root)
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharex=True, sharey=True)
    for ax, (arch, alab) in zip(axes, ARCHS):
        allx, ally = [], []
        for loss in LOSSES:
            xs, ys, x32, y32 = [], [], [], []
            for H in WIDTHS:
                r = ratio_by_mouse(results_root, loss, H, arch)
                for idx, rat in r.items():
                    if idx not in dims:
                        continue
                    N, ntr = dims[idx]
                    ppt = n_params(N, H) / ntr
                    xs.append(ppt); ys.append(rat); allx.append(ppt); ally.append(rat)
                    if H == BASE_H:
                        x32.append(ppt); y32.append(rat)
            if xs:
                ax.scatter(xs, ys, s=22, color=LCOL[loss], alpha=0.7, lw=0, label=ps.loss_label(loss))
                ax.scatter(x32, y32, s=46, facecolors='none', edgecolors=LCOL[loss], lw=1.3)  # H=32 ringed
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.axhline(1.0, color='0.5', lw=1.0, ls=':')
        if len(allx) > 3:
            rho, p = spearmanr(allx, ally)
            ax.text(0.04, 0.96, f'Spearman ρ = {rho:.2f}\n(p = {p:.1e}, n = {len(allx)})',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
        ax.set_xlabel('parameters / training trial')
        ax.set_ylabel('val / train fit-loss (overfitting)')
        ax.set_title(alab, fontsize=10)
        if arch == 'spat':
            ax.legend(fontsize=7, loc='lower right', frameon=True, title='(ring = H=32 baseline)',
                      title_fontsize=6.5)
    ps.label_panels(axes)
    fig.suptitle('Overfitting scales with parameters-per-trial — pooled over 6 mice × 5 hidden widths '
                 '(each point = one mouse×width×loss)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'overfit_vs_capacity')

    # numeric: per-mouse baseline (H=32) params/trial, and the pooled correlation.
    print('per-mouse baseline (H=32):')
    for idx in sorted(dims):
        N, ntr = dims[idx]
        print(f'  mouse {idx}: N={N:3d}  n_train={ntr:4d}  params={n_params(N, 32):5d}  '
              f'params/trial={n_params(N, 32)/ntr:5.1f}')
    print(f'\nDone -> {Path(out_root).resolve()}/overfit_vs_capacity.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
