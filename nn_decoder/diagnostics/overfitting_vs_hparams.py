# -*- coding: utf-8 -*-
"""Train–val overfitting vs each swept hyperparameter — the fit-loss counterpart of
`peakiness_vs_hparams.py`, in the same layout (rows = arch, cols = axis, one line per
loss, shared y).

Overfitting is read as the final **val/train fit-loss ratio** (dimensionless; 1 = train
and val equal = no overfitting, >1 = val worse = overfitting). The ratio, rather than the
raw val−train gap, is used because the raw fit-loss scales differ ~100× across losses and
would not share a y-axis; the ratio is the analogue of peakiness's bounded max-prob.

This is the REAL decoders (spat/temp). Its relation to the over-sharpening is the point:
peakiness (`peakiness_vs_hparams.py`) is the over-sharpening the loss work is about, and
it is largely DECOUPLED from this fit-loss overfitting (the over-sharpening lives in the
loss-blind shape subspace — see PCA-Peakiness-Mechanism §8).

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  overfitting_vs_hparams.png
Usage:  python diagnostics/overfitting_vs_hparams.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
FIELD = 'fit_loss'
AXES = {
    'lambda_H':   dict(tok='lam',  vals=[0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], xlabel='entropy λ_H', xtype='index'),
    'dropout':    dict(tok='drop', vals=[0, 0.1, 0.25, 0.5, 0.75, 0.9],    xlabel='dropout p',   xtype='lin'),
    'width':      dict(tok='h',    vals=[4, 8, 16, 32, 64],                xlabel='hidden width', xtype='log2'),
    'activation': dict(tok='act',  vals=['tanh', 'relu', 'gelu'],          xlabel='activation',  xtype='index'),
    'patience':   dict(tok='pat',  vals=[0, 10, 20, 40],                   xlabel='patience',    xtype='lin'),
}
DEFAULT_AXES = ['lambda_H', 'dropout', 'width', 'activation', 'patience']
BASELINE = dict(lam='0p003', drop='0', act='tanh', h='32', pat='0', vf='0p2')


def _g(x):
    return f"{x:g}".replace('.', 'p')


def cell_for(axis, v):
    tok = dict(BASELINE)
    tok[AXES[axis]['tok']] = v if isinstance(v, str) else _g(v)
    return (f"lam{tok['lam']}_drop{tok['drop']}_act{tok['act']}"
            f"_h{tok['h']}_pat{tok['pat']}_vf{tok['vf']}")


def _overfit_ratio(ck_dir, arch):
    """final val/train fit-loss ratio per mouse -> mean, sem (>1 = overfitting)."""
    rs = []
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        node = torch.load(str(pt), map_location='cpu', weights_only=False).get(arch)
        h = (node or {}).get('history') or {}
        t, v = h.get(f'train_{FIELD}'), h.get(f'val_{FIELD}')
        if t and v and t[-1] > 0:
            rs.append(v[-1] / t[-1])
    if not rs:
        return None, None
    rs = np.array(rs, float)
    return rs.mean(), rs.std() / np.sqrt(len(rs))


def _xpos(cfg):
    return list(range(len(cfg['vals']))) if cfg['xtype'] == 'index' else list(cfg['vals'])


def main(results_root, out_root, axes):
    ps.apply()
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    print("overfitting (val/train fit-loss ratio)")
    for r, (arch, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], AXES[axis]
            xall = _xpos(cfg)
            for loss in LOSSES:
                xs, ys, es = [], [], []
                for x, v in zip(xall, cfg['vals']):
                    ck = Path(results_root) / RUN_PARENT / cell_for(axis, v) / LOSS_SLUG[loss] / 'checkpoints'
                    m, s = _overfit_ratio(ck, arch)
                    if m is not None:
                        xs.append(x); ys.append(m); es.append(s if s is not None else 0.0)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=LCOL[loss], lw=1.6, marker='o', ms=4,
                                capsize=2, label=ps.loss_label(loss))
            ax.set_yscale('log')                          # ratio spans ~2 orders (KL-spatial ≈60)
            if cfg['xtype'] == 'log2':
                ax.set_xscale('log', base=2); ax.set_xticks(cfg['vals']); ax.set_xticklabels(cfg['vals'])
            elif cfg['xtype'] == 'index':
                ax.set_xticks(xall); ax.set_xticklabels(cfg['vals'], fontsize=7)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\nval/train fit-loss')
            if r == 0:
                ax.set_title(axis, fontsize=9)
            ax.axhline(1.0, color='0.4', lw=1.1, ls=':')
    axgrid[0][0].plot([], [], color='0.4', lw=1.1, ls=':', label='no overfitting (=1)')
    handles, lbls = axgrid[0][0].get_legend_handles_labels()
    axgrid[0][0].legend(handles, lbls, fontsize=6.5, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle('Train–val overfitting (val/train fit-loss ratio, final) vs each hyperparameter '
                 '(shared y; 1 = no overfitting, dotted). Real decoders, mean±sem, 6 mice', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'overfitting_vs_hparams')
    print(f'Done -> {Path(out_root).resolve()}/overfitting_vs_hparams.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--axes', nargs='+', default=DEFAULT_AXES, choices=list(AXES))
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.axes)
