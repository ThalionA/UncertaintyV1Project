# -*- coding: utf-8 -*-
"""Train–val gap vs each swept hyperparameter, for the real decoder AND its shuffle control.

Companion to `overfitting_vs_hparams.py` (which plots the real decoder's val/train *ratio*):
this one plots the raw val−train fit-loss gap for the real decoder and for the shuffle-trained
control side by side. The question it answers: which knobs shrink the gap, and do they act more
on the null (which purely memorises scrambled labels) than on the real decoder (whose gap is a
static offset)?

Layout: rows = arch (spatial, temporal), cols = axis, SHARED y so magnitudes are comparable.
Sweep spec (parent run, baseline cell, axis grids) comes from `hpsweep_spec.py` — `--sweep {v1,v2}`.

NB the gap is in each loss's own units, so only compare within a loss, not across.

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  gap_vs_hparams_<sweep>_<loss>.png
Usage:  python diagnostics/shuffle_gap_vs_reg.py --sweep v2 --loss KL
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402

ARCHS = [('spat', 'spat_shf', 'spatial'), ('temp', 'temp_shf', 'temporal')]
REAL_C, SHUF_C = '#2166ac', '#b2182b'
FIELD = 'fit_loss'


def _final_gap(ck_dir, arch):
    """final (val − train) fit-loss per mouse -> mean, sem. Uses each mouse's LAST
    recorded epoch so early-stopped cells are handled correctly. sem uses ddof=1 to
    match the canonical aggregators (cross_loss_eval._agg)."""
    tr, va = [], []
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        node = torch.load(str(pt), map_location='cpu', weights_only=False).get(arch)
        h = (node or {}).get('history') or {}
        t, v = h.get(f'train_{FIELD}'), h.get(f'val_{FIELD}')
        if t and v:
            tr.append(t[-1]); va.append(v[-1])
    if not tr:
        return None
    g = np.asarray(va, float) - np.asarray(tr, float)
    sem = float(g.std(ddof=1) / np.sqrt(g.size)) if g.size > 1 else 0.0
    return g.mean(), sem


def main(results_root, out_root, sweep, loss, axes):
    ps.apply()
    spec = S.SPECS[sweep]
    axes = [a for a in axes if loss in S.axis_losses(spec, a)]
    if not axes:
        raise SystemExit(f"no swept axes apply to loss={loss} in sweep {sweep}.")
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    n_series = 0
    print(f"train-val gap (val - train {FIELD}) — sweep={sweep} loss={loss}")
    for r, (real, shuf, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], spec['axes'][axis]
            for who, arch, col in [('real', real, REAL_C), ('shuffle', shuf, SHUF_C)]:
                xs, ys, es = [], [], []
                for x, v in zip(S.xpos(cfg), cfg['vals']):
                    ck = (Path(results_root) / spec['parent'] / S.cell_for(spec, axis, v)
                          / S.LOSS_SLUG[loss] / 'checkpoints')
                    g = _final_gap(ck, arch)
                    if g:
                        xs.append(x); ys.append(g[0]); es.append(g[1])
                if xs:
                    n_series += 1
                    ax.errorbar(xs, ys, yerr=es, color=col, lw=1.6, marker='o', ms=4,
                                capsize=2, label=who)
            S.apply_xaxis(ax, cfg)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\nval−train gap')
            if r == 0:
                ax.set_title(axis, fontsize=9)
            ax.axhline(0, color='0.6', lw=0.7, ls=':')
    if n_series == 0:
        raise SystemExit(f"no cells loaded under {spec['parent']}/ for loss={loss} — "
                         "rsync the run down first (refusing to save an empty figure).")
    axgrid[0][0].legend(handles=[Line2D([0], [0], color=REAL_C, lw=2, marker='o', label='real'),
                                 Line2D([0], [0], color=SHUF_C, lw=2, marker='o', label='shuffle')],
                        fontsize=7, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'[{sweep}] {ps.loss_label(loss)} — train–val gap vs each hyperparameter '
                 f'(shared y; mean±sem over mice). Does regularisation shrink the null’s overfitting?',
                 y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'gap_vs_hparams_{sweep}_{loss}')
    print(f'Done -> {Path(out_root).resolve()}/gap_vs_hparams_{sweep}_{loss}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    ap.add_argument('--loss', default='KL', choices=list(S.LOSS_SLUG))
    ap.add_argument('--axes', nargs='+', default=None)
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.sweep, a.loss, a.axes or S.DEFAULT_AXES[a.sweep])
