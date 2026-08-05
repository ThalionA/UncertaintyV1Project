# -*- coding: utf-8 -*-
"""Train / validation fit-loss curves per configuration (2026-08-04, Theo's request).

Rows = architecture (spatial, temporal), columns = configuration. Each panel shows
one light line per mouse: TRAIN dashed, VAL solid, with a dot at
``history['best_epoch']`` — the epoch whose weights were RESTORED and used for every
reported number (`nn_classifier.fit_model` line ~676). Training continues for
`patience` further epochs after that dot before stopping, which is exactly why the
overfitting ratio must be read at the dot and not at the end of the line (the
2026-08-04 fix).

The val curve here is the INTERNAL carve from the training half (used for early
stopping AND restart selection), not the held-out test half — so it is optimistically
biased, and the test loss sits above it. That gap is the selection optimism, not a bug.

Both axes are the cell's own training loss, so absolute values are comparable WITHIN
a weighting family but not across (evar losses are ~50x larger than flat/MSE).

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_trainval_curves.py [--set main|pc-flat|pc-evar] [--lam 0]
"""

from __future__ import annotations

import argparse
import glob
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from projflat_report import have, _slug  # noqa: E402
from projflat_config_axes import configs_for  # noqa: E402

ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]


def curves(results_root, cell, arch):
    """[(train, val, best_epoch)] per mouse."""
    ck = _slug(results_root, cell) / 'checkpoints'
    out = []
    for pt in sorted(ck.glob('mouse_*_stratified_balanced.pt')):
        h = (torch.load(str(pt), map_location='cpu',
                        weights_only=False).get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if t and v:
            out.append((np.array(t, float), np.array(v, float), h.get('best_epoch')))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--set', dest='cfgset', default='main',
                    choices=['main', 'pc-flat', 'pc-evar'])
    ap.add_argument('--lam', default='0')
    a = ap.parse_args()

    cfgs = configs_for(a.lam, a.cfgset)
    cfgs = [(l, c) for l, c in cfgs if have(a.results_root, c)]
    if not cfgs:
        raise SystemExit('no cells available')

    ps.apply()
    fig, axes = plt.subplots(len(ARCHS), len(cfgs),
                             figsize=ps.figsize(len(cfgs), len(ARCHS)), squeeze=False)
    for ri, (arch, alab) in enumerate(ARCHS):
        for ci, (lab, cell) in enumerate(cfgs):
            ax = axes[ri][ci]
            for t, v, b in curves(a.results_root, cell, arch):
                ep = np.arange(len(t))
                ax.plot(ep, t, color=ps.SPATIAL if arch == 'spat' else ps.TEMPORAL,
                        lw=0.9, ls='--', alpha=0.45)
                ax.plot(ep, v, color=ps.SPATIAL if arch == 'spat' else ps.TEMPORAL,
                        lw=1.1, alpha=0.85)
                if b is not None and 0 <= int(b) < len(v):
                    ax.plot(int(b), v[int(b)], 'o', ms=4, mfc='w',
                            mec='k', mew=1.0, zorder=5)
            ax.set_yscale('log')
            ax.grid(alpha=0.2, lw=0.5)
            if ri == 0:
                ax.set_title(lab.replace('\n', ', '), fontsize=8)
            if ri == len(ARCHS) - 1:
                ax.set_xlabel('epoch', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{alab}\nfit-loss (own training loss)', fontsize=8)
    axes[0][0].legend(handles=[
        Line2D([0], [0], color='0.3', ls='--', lw=1.0),
        Line2D([0], [0], color='0.3', lw=1.2),
        Line2D([0], [0], color='k', marker='o', ls='none', ms=4, mfc='w')],
        labels=['train', 'validation (internal carve)', 'best epoch (weights restored)'],
        fontsize=6, frameon=True)
    fig.suptitle('Train / validation fit-loss per configuration, one line per mouse. The dot is the '
                 'RESTORED best epoch — training runs `patience` further epochs past it, which is why '
                 'the overfitting ratio must be read at the dot, not the line end.', y=1.02, fontsize=8)
    fig.tight_layout()
    suffix = ('' if a.cfgset == 'main' else f'_{a.cfgset}') + \
             ('' if a.lam == '0' else f'_lam{a.lam}')
    ps.save_fig(fig, Path(a.out_root), f'projflat_cfg_trainval{suffix}')


if __name__ == '__main__':
    main()
