# -*- coding: utf-8 -*-
"""Is the overfitting a capacity-vs-data problem? Overfitting (val/train fit-loss ratio) vs
parameters-per-training-trial, pooled over mice AND hidden widths.

In v1 (H=32 base) every loss overfit heavily because the net carried ~5–8k parameters against
only ~350 training trials per mouse (11–24x overparameterised). Pooling the width sweep × 6
mice spans a wide params/trial range, so if capacity-vs-data is the cause the points collapse
onto ONE rising trend regardless of loss, mouse or width — which is what v1 showed
(Spearman ρ = 0.52 spatial / 0.59 temporal, p < 1e-9).

v2 re-bases at H=8 and re-centres the width axis on {2..32}, so the same plot should sit
further LEFT and LOWER (less overparameterised, less overfitting) while keeping the trend.

params = N·H + H + H·C + C  (one-hidden-layer MLP, C=91); trials = per-mouse training-set size.
Targets either sweep via `--sweep` (see `hpsweep_spec.py`).

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  overfit_vs_capacity_<sweep>.png
Usage:  python diagnostics/overfit_vs_capacity.py --sweep v2
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402

LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
C_CATS = 91


def n_params(N, H):
    return N * H + H + H * C_CATS + C_CATS


def mouse_dims(results_root, spec):
    """{mouse_idx: (N, n_train)} from the baseline PCA cell (constant across cells)."""
    d = {}
    root = Path(results_root) / spec['parent'] / S.baseline_cell(spec) / S.LOSS_SLUG['PCA']
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


def ratio_by_mouse(results_root, spec, loss, H, arch):
    ck_dir = (Path(results_root) / spec['parent'] / S.cell_for(spec, 'width', H)
              / S.LOSS_SLUG[loss] / 'checkpoints')
    out = {}
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        idx = int(pt.stem.split('_')[1])
        h = (torch.load(str(pt), map_location='cpu', weights_only=False).get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if t and v and t[-1] > 0:
            out[idx] = v[-1] / t[-1]
    return out


def main(results_root, out_root, sweep):
    ps.apply()
    spec = S.SPECS[sweep]
    widths = spec['axes']['width']['vals']
    base_h = spec['base']['width']
    dims = mouse_dims(results_root, spec)
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharex=True, sharey=True)
    for ax, (arch, alab) in zip(axes, ARCHS):
        allx, ally = [], []
        for loss in S.LOSSES:
            xs, ys, xb, yb = [], [], [], []
            for H in widths:
                for idx, rat in ratio_by_mouse(results_root, spec, loss, H, arch).items():
                    if idx not in dims:
                        continue
                    N, ntr = dims[idx]
                    ppt = n_params(N, H) / ntr
                    xs.append(ppt); ys.append(rat); allx.append(ppt); ally.append(rat)
                    if H == base_h:
                        xb.append(ppt); yb.append(rat)
            if xs:
                ax.scatter(xs, ys, s=22, color=LCOL[loss], alpha=0.7, lw=0, label=ps.loss_label(loss))
                ax.scatter(xb, yb, s=46, facecolors='none', edgecolors=LCOL[loss], lw=1.3)
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
            ax.legend(fontsize=7, loc='lower right', frameon=True,
                      title=f'(ring = H={base_h} base)', title_fontsize=6.5)
    ps.label_panels(axes)
    fig.suptitle(f'[{sweep}] Overfitting scales with parameters-per-trial — pooled over 6 mice × '
                 f'{len(widths)} widths (each point = one mouse×width×loss)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'overfit_vs_capacity_{sweep}')
    print(f'per-mouse baseline (H={base_h}):')
    for idx in sorted(dims):
        N, ntr = dims[idx]
        print(f'  mouse {idx}: N={N:3d} n_train={ntr:4d} params={n_params(N, base_h):5d} '
              f'params/trial={n_params(N, base_h)/ntr:5.1f}')
    print(f'Done -> {Path(out_root).resolve()}/overfit_vs_capacity_{sweep}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.sweep)
