# -*- coding: utf-8 -*-
"""Train–val gap vs each swept hyperparameter, for the real decoder and its shuffle control.

For one loss, sweeps every one-at-a-time axis in `hpsweep_wide` and plots the final
train–val fit-loss gap (val − train = overfitting), mean±sem over mice, for the REAL
decoder and the SHUFFLE control. Question: which knobs shrink the gap — and do they act
more on the null (pure noise-overfitting) than on the real decoder (whose gap we've shown
is a static offset)?

Axes (all vs the all-defaults baseline): entropy λ_H, dropout, hidden width, activation,
early-stop patience. Categorical/uneven axes (activation, λ_H) are index-plotted; width is
log2; dropout/patience linear.

Layout: rows = arch (spatial, temporal), cols = axis, with a SHARED y-axis across all
panels so magnitudes are directly comparable.

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  gap_vs_hparams_<loss>.png
Usage:
  python diagnostics/shuffle_gap_vs_reg.py --loss KL
  python diagnostics/shuffle_gap_vs_reg.py --loss PCA --axes dropout width patience
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

RUN_PARENT = 'hpsweep_wide'
LOSS_SLUG = {'PCA': 'Q_PCA_half_100ms_all', 'KL': 'Q_KL_half_100ms',
             'JS': 'Q_JS_half_100ms', 'Wasserstein': 'Q_Wasserstein_half_100ms'}
ARCHS = [('spat', 'spat_shf', 'spatial'), ('temp', 'temp_shf', 'temporal')]
REAL_C, SHUF_C = '#2166ac', '#b2182b'
FIELD = 'fit_loss'

# axis -> slug token, values, x-label, x-scaling ('lin' | 'log2' | 'index')
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


def _slug_tok(v):
    return v if isinstance(v, str) else _g(v)


def cell_for(axis, v):
    tok = dict(BASELINE)
    tok[AXES[axis]['tok']] = _slug_tok(v)
    return (f"lam{tok['lam']}_drop{tok['drop']}_act{tok['act']}"
            f"_h{tok['h']}_pat{tok['pat']}_vf{tok['vf']}")


def _final_gap(ck_dir, arch):
    """final (val − train) fit-loss per mouse -> mean, sem. Last recorded epoch per mouse
    so early-stopped patience cells are handled correctly."""
    tr, va = [], []
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        node = torch.load(str(pt), map_location='cpu', weights_only=False).get(arch)
        h = (node or {}).get('history') or {}
        t, v = h.get(f'train_{FIELD}'), h.get(f'val_{FIELD}')
        if t and v:
            tr.append(t[-1]); va.append(v[-1])
    if not tr:
        return None
    g = np.array(va) - np.array(tr)
    return g.mean(), g.std() / np.sqrt(len(g))


def _xpos(cfg):
    return list(range(len(cfg['vals']))) if cfg['xtype'] == 'index' else list(cfg['vals'])


def main(results_root, out_root, loss, axes):
    ps.apply()
    slug = LOSS_SLUG[loss]
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    print(f"loss={loss}  axes={axes}")
    for r, (real, shuf, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], AXES[axis]
            xall = _xpos(cfg)
            for who, arch, col in [('real', real, REAL_C), ('shuffle', shuf, SHUF_C)]:
                xs, ys, es = [], [], []
                for x, v in zip(xall, cfg['vals']):
                    ck = Path(results_root) / RUN_PARENT / cell_for(axis, v) / slug / 'checkpoints'
                    g = _final_gap(ck, arch)
                    if g:
                        xs.append(x); ys.append(g[0]); es.append(g[1])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=col, lw=1.5, marker='o', ms=4,
                                capsize=2, label=who)
            if cfg['xtype'] == 'log2':
                ax.set_xscale('log', base=2); ax.set_xticks(cfg['vals']); ax.set_xticklabels(cfg['vals'])
            elif cfg['xtype'] == 'index':
                ax.set_xticks(xall); ax.set_xticklabels(cfg['vals'], fontsize=7)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\nval−train gap')
            if r == 0:
                ax.set_title(axis, fontsize=9)
            ax.axhline(0, color='0.6', lw=0.7, ls=':')
    axgrid[0][0].legend(handles=[Line2D([0], [0], color=REAL_C, lw=2, marker='o', label='real'),
                                 Line2D([0], [0], color=SHUF_C, lw=2, marker='o', label='shuffle')],
                        fontsize=7, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'{ps.loss_label(loss)} — train–val gap vs each swept hyperparameter '
                 f'(shared y; mean±sem, 6 mice)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'gap_vs_hparams_{loss}')
    print(f'Done -> {Path(out_root).resolve()}/gap_vs_hparams_{loss}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--loss', default='KL', choices=list(LOSS_SLUG))
    ap.add_argument('--axes', nargs='+', default=DEFAULT_AXES, choices=list(AXES))
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.loss, a.axes)
