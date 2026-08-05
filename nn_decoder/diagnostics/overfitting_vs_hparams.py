# -*- coding: utf-8 -*-
"""Train–val overfitting vs each swept hyperparameter — the fit-loss counterpart of
`peakiness_vs_hparams.py`, same layout (rows = arch, cols = axis, one line per loss, shared y).

Overfitting = final **val/train fit-loss ratio** (dimensionless; 1 = no overfitting). The
ratio rather than the raw val−train gap, because the raw fit-loss scales differ ~100x across
losses and would not share a y-axis. Log y (the ratio spans ~2 orders).

The point: this is VARIANCE, and it is largely decoupled from the over-sharpening (which is
BIAS — see `peakiness_vs_hparams.py`). In v1, KL overfit most and PCA least — the inverse of
the peakiness ranking — and the whole thing tracked params-per-trial (`overfit_vs_capacity.py`).
v2 re-bases at H=8 (~5x params/trial vs ~18x), so these ratios should drop across the board,
and the new **weight_decay** axis is another lever on it.

Targets either sweep via `--sweep` (see `hpsweep_spec.py`).
Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  overfitting_vs_hparams_<sweep>.png
Usage:  python diagnostics/overfitting_vs_hparams.py --sweep v2
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402

LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
FIELD = 'fit_loss'


def _overfit_ratio(ck_dir, arch, at='best'):
    """val/train fit-loss ratio per mouse -> mean, sem (>1 = overfitting).

    ``at='best'`` (default) reads the ratio at ``history['best_epoch']`` — the
    epoch whose weights were actually RESTORED and used to produce the saved
    posteriors (``nn_classifier.fit_model`` line ~676 does
    ``model.load_state_dict(_es_best_state)``). ``at='last'`` reads the final
    epoch, which under early stopping is ``best_epoch + patience`` and therefore
    describes a DIFFERENT (further-overfit) model than the one being scored.

    This mismatch inflated the reported ratio by 14-68% on the projflat_v1 cells
    (e.g. h8 flat spatial 3.32 -> 4.85), though the ordering across configs was
    preserved. Found 2026-08-04.

    Runs without early stopping (patience=0) have no ``best_epoch`` key, so they
    fall back to the final epoch and are bit-for-bit unchanged — every existing
    hpsweep figure is unaffected.
    """
    rs = []
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        node = torch.load(str(pt), map_location='cpu', weights_only=False).get(arch)
        h = (node or {}).get('history') or {}
        t, v = h.get(f'train_{FIELD}'), h.get(f'val_{FIELD}')
        if not (t and v):
            continue
        i = -1
        if at == 'best':
            b = h.get('best_epoch')
            if b is not None and 0 <= int(b) < len(t):
                i = int(b)
        if t[i] > 0:
            rs.append(v[i] / t[i])
    if not rs:
        return None, None
    rs = np.array(rs, float)
    # ddof=1 to match the canonical aggregators (cross_loss_eval._agg,
    # plot_overfit_vs_width._msem). ddof=0 understates the SEM by sqrt(n/(n-1))
    # = 9.5% at n=6, which made these error bars inconsistent with sibling figures.
    sem = rs.std(ddof=1) / np.sqrt(rs.size) if rs.size > 1 else 0.0
    return rs.mean(), float(sem)


def main(results_root, out_root, sweep, axes):
    ps.apply()
    spec = S.SPECS[sweep]
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    print(f"overfitting (val/train fit-loss ratio) — sweep={sweep}")
    n_series = 0
    for r, (arch, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], spec['axes'][axis]
            xall = S.xpos(cfg)
            for loss in S.axis_losses(spec, axis):
                xs, ys, es = [], [], []
                for x, v in zip(xall, cfg['vals']):
                    ck = (Path(results_root) / spec['parent'] / S.cell_for(spec, axis, v)
                          / S.LOSS_SLUG[loss] / 'checkpoints')
                    m, s = _overfit_ratio(ck, arch)
                    if m is not None:
                        xs.append(x); ys.append(m); es.append(s if s is not None else 0.0)
                if xs:
                    n_series += 1
                    ax.errorbar(xs, ys, yerr=es, color=LCOL[loss], lw=1.6, marker='o', ms=4,
                                capsize=2, label=ps.loss_label(loss))
            ax.set_yscale('log')
            S.apply_xaxis(ax, cfg)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\nval/train fit-loss')
            if r == 0:
                ax.set_title(axis + (' (PCA)' if cfg['losses'] else ''), fontsize=9)
            ax.axhline(1.0, color='0.4', lw=1.1, ls=':')
    if n_series == 0:
        raise SystemExit(f"no cells loaded under {spec['parent']}/ for axes {axes} — "
                         "rsync the run down first (refusing to save an empty figure).")
    axgrid[0][0].plot([], [], color='0.4', lw=1.1, ls=':', label='no overfitting (=1)')
    h, l = axgrid[0][0].get_legend_handles_labels()
    axgrid[0][0].legend(h, l, fontsize=6.5, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'[{sweep}] Train–val overfitting (val/train fit-loss, final) vs each hyperparameter '
                 f'(shared log y; 1 = none). Real decoders, mean±sem, 6 mice', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'overfitting_vs_hparams_{sweep}')
    print(f'Done -> {Path(out_root).resolve()}/overfitting_vs_hparams_{sweep}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    ap.add_argument('--axes', nargs='+', default=None)
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.sweep, a.axes or S.DEFAULT_AXES[a.sweep])
