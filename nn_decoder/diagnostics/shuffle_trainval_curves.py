# -*- coding: utf-8 -*-
"""Train vs val curves for the SHUFFLE control decoders, vs the real ones.

The shuffle controls (`spat_shf`/`temp_shf`, trained on label-shuffled data) had their
per-epoch history discarded until 2026-06-18; `run_hyperparam_sweep.py` now saves it.
Real decoders learn generalisable signal (val tracks train, a static gap); the shuffle
controls memorise scrambled labels (val diverges upward = overfitting-to-noise).

Normalisation = **predict-mean** (2026-06-18 meeting #8): each curve is divided by that
mouse's predict-mean fit-loss — the loss of outputting the marginal-mean target on every
trial, the strictest "no trial-info" null (`predict_mean_baseline.py`). So **1.0 is the
chance line (#7)**: <1 beats predict-mean, >1 is worse. This puts every loss on one
comparable axis (raw fit-loss scales differ ~100× across losses) and states the null
explicitly, as asked.

Two views:
  * per-loss (`single`): spatial|temporal, RAW fit-loss on a shared y, with the
    predict-mean chance line drawn in absolute units;
  * cross-loss (`grid`): arch x loss, predict-mean-normalised, ALL panels shared y,
    chance line at 1.0.

Outputs (PNG+SVG) under figures/hpsweep_shuffle/.
Usage:
  python diagnostics/shuffle_trainval_curves.py                 # grid + per-loss
  python diagnostics/shuffle_trainval_curves.py --mode grid
  python diagnostics/shuffle_trainval_curves.py --mode single --loss KL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one  # noqa: E402  (same loss maths as training)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402  (single source of parent/baseline/slugs)

LOSS_SLUG = S.LOSS_SLUG
LOSS_ORDER = S.LOSSES
ARCHS = [('spat', 'spat_shf', 'spatial'), ('temp', 'temp_shf', 'temporal')]
# Parent run dir; rebound from --sweep in main() (default v2 — v1 used the
# H=32 base later shown to be ~18x overparameterised).
RUN_PARENT = S.SPECS['v2']['parent']
REAL_C, SHUF_C = '#2166ac', '#b2182b'                    # blue real, red shuffle
CHANCE_C = '0.35'
FIELD = 'fit_loss'


def _stack_by_mouse(ck_dir: Path, arch: str, which: str) -> dict:
    out = {}
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        try:
            idx = int(pt.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        node = torch.load(str(pt), map_location='cpu', weights_only=False).get(arch)
        arr = ((node or {}).get('history') or {}).get(f'{which}_{FIELD}')
        if arr:
            out[idx] = np.asarray(arr, float)
    return out


def _predict_mean_by_mouse(slug_dir: Path, loss: str, arch: str) -> dict:
    """{mouse_idx: predict-mean fit-loss} — loss of outputting the marginal-mean target
    on every trial (the strictest null), computed from the cell's .mat via the same loss
    maths as training (`_eval_one`)."""
    f = Path(slug_dir) / 'stratified_balanced.mat'
    out = {}
    if not f.is_file():
        return out
    res = sio.loadmat(str(f), simplify_cells=True).get('results')
    if not isinstance(res, dict):
        return out
    for mk in res:
        if not str(mk).startswith('mouse_'):
            continue
        try:
            idx = int(str(mk).split('_')[-1])
        except ValueError:
            continue
        D = res[mk].get('Dist') if isinstance(res[mk], dict) else None
        if not (isinstance(D, dict) and arch in D):
            continue
        tgt = np.asarray(D[arch]['target'], float)
        if tgt.ndim != 2:
            continue
        ok = np.isfinite(tgt).all(1)
        if not ok.any():
            continue
        # LEAVE-ONE-OUT marginal mean (see predict_mean_baseline): the plain
        # test-set mean fits the null on the trials it scores, biasing ratios
        # low by O(1/n). 2026-07 audit.
        n_ok = int(ok.sum())
        tot = tgt[ok].sum(axis=0)
        pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
        if n_ok > 1:
            pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
        v = _eval_one(pm, tgt, loss, D.get('pcs'), D.get('explained_var'))
        if np.isfinite(v) and v > 0:
            out[idx] = float(v)
    return out


def _curves(ck_dir, arch, norm='raw', slug_dir=None, loss=None):
    """mean±sem over mice of (train, val). norm='pm' divides each mouse by its
    predict-mean fit-loss; 'raw' leaves absolute. Returns (mt,st,mv,sv,n)."""
    tr, va = _stack_by_mouse(ck_dir, arch, 'train'), _stack_by_mouse(ck_dir, arch, 'val')
    idxs = sorted(set(tr) & set(va))
    ref = None
    if norm == 'pm':
        pm = _predict_mean_by_mouse(slug_dir, loss, arch)
        idxs = [i for i in idxs if i in pm]
        ref = {i: pm[i] for i in idxs}
    if not idxs:
        return None
    L = min(min(len(tr[i]) for i in idxs), min(len(va[i]) for i in idxs))
    T = np.stack([tr[i][:L] for i in idxs]); V = np.stack([va[i][:L] for i in idxs])
    if ref is not None:
        r = np.array([ref[i] for i in idxs])[:, None]
        T, V = T / r, V / r
    n = T.shape[0]
    # ddof=1 to match the canonical aggregators (cross_loss_eval._agg); ddof=0
    # understates the SEM by 9.5% at n=6. 2026-07 audit.
    d = 1 if n > 1 else 0
    return T.mean(0), T.std(0, ddof=d) / np.sqrt(n), V.mean(0), V.std(0, ddof=d) / np.sqrt(n), n


def _band(ax, m, s, color, ls, label=None):
    ep = np.arange(len(m))
    ax.plot(ep, m, color=color, lw=1.7, ls=ls, label=label)
    ax.fill_between(ep, m - s, m + s, color=color, alpha=0.14, lw=0)


def _legend(ax, chance_label):
    ax.legend(handles=[Line2D([0], [0], color=REAL_C, lw=2, label='real decoder'),
                       Line2D([0], [0], color=SHUF_C, lw=2, label='shuffle control'),
                       Line2D([0], [0], color='0.5', lw=2, ls='-', label='train'),
                       Line2D([0], [0], color='0.5', lw=2, ls='--', label='val'),
                       Line2D([0], [0], color=CHANCE_C, lw=1.2, ls=':', label=chance_label)],
              fontsize=6.5, loc='best', frameon=True)


def _cells_losses(results_root, cell):
    base = Path(results_root) / RUN_PARENT / cell
    return [l for l in LOSS_ORDER if (base / LOSS_SLUG[l] / 'checkpoints').is_dir()]


def _pm_mean(slug_dir, loss, arch):
    pm = _predict_mean_by_mouse(slug_dir, loss, arch)
    return float(np.mean(list(pm.values()))) if pm else np.nan


def plot_single(results_root, out_root, loss, cell):
    """spatial|temporal, RAW fit-loss on a shared y, with the predict-mean chance line."""
    ps.apply()
    slug_dir = Path(results_root) / RUN_PARENT / cell / LOSS_SLUG[loss]
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharey=True, sharex=True)
    for ax, (real, shuf, alab) in zip(axes, ARCHS):
        for arch, col in [(real, REAL_C), (shuf, SHUF_C)]:
            c = _curves(slug_dir / 'checkpoints', arch)
            if c:
                _band(ax, c[0], c[1], col, '-'); _band(ax, c[2], c[3], col, '--')
        pm = _pm_mean(slug_dir, loss, real)               # chance line, absolute units
        if np.isfinite(pm):
            ax.axhline(pm, color=CHANCE_C, lw=1.2, ls=':')
        ax.set_xlabel('epoch'); ax.set_title(alab, fontsize=10)
    axes[0].set_ylabel(f'{ps.loss_label(loss)} fit-loss')
    _legend(axes[0], 'predict-mean')
    ps.label_panels(axes)
    fig.suptitle(f'{ps.loss_label(loss)} — real vs shuffle train/val (shared y; predict-mean '
                 f'= chance, dotted; mean±sem, 6 mice)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'shuffle_trainval_{loss}_{cell}')


def plot_grid(results_root, out_root, cell):
    """arch x loss, predict-mean-normalised, ALL panels shared y (chance = 1.0)."""
    ps.apply()
    losses = _cells_losses(results_root, cell)
    if not losses:
        raise SystemExit(f'no losses found in {cell}')
    nR, nC = len(ARCHS), len(losses)
    fig, axes = plt.subplots(nR, nC, figsize=ps.figsize(nC, nR),
                             sharey=True, sharex=True, squeeze=False)
    print(f"{'loss':6s} {'arch':9s} {'val/PM real':>12s} {'shuffle':>9s}  n")
    for r, (real, shuf, alab) in enumerate(ARCHS):
        for cix, loss in enumerate(losses):
            ax = axes[r][cix]
            slug_dir = Path(results_root) / RUN_PARENT / cell / LOSS_SLUG[loss]
            vals = {}
            for arch, col in [(real, REAL_C), (shuf, SHUF_C)]:
                c = _curves(slug_dir / 'checkpoints', arch, norm='pm', slug_dir=slug_dir, loss=loss)
                if c:
                    _band(ax, c[0], c[1], col, '-'); _band(ax, c[2], c[3], col, '--')
                    vals[col] = c[2][-1]; n = c[4]
            ax.axhline(1.0, color=CHANCE_C, lw=1.0, ls=':')
            if r == 0:
                ax.set_title(ps.loss_label(loss), fontsize=10)
            if r == nR - 1:
                ax.set_xlabel('epoch')
            if cix == 0:
                ax.set_ylabel(f'{alab}\nfit-loss ÷ predict-mean')
            print(f"{loss:6s} {alab:9s} {vals.get(REAL_C, float('nan')):12.3f} "
                  f"{vals.get(SHUF_C, float('nan')):9.3f}  {n}")
    _legend(axes[0][0], 'predict-mean (chance)')
    ps.label_panels(axes.ravel())
    fig.suptitle('Real vs shuffle train/val ÷ predict-mean (shared y; 1.0 = chance/predict-mean, '
                 'dotted). Shuffle val climbs above 1 = worse than the null', y=1.01, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'shuffle_trainval_grid_{cell}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--mode', default='both', choices=['both', 'grid', 'single'])
    ap.add_argument('--loss', default='PCA', choices=list(LOSS_SLUG))
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    ap.add_argument('--cell', default=None, help='defaults to the sweep baseline cell')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    global RUN_PARENT
    spec = S.SPECS[a.sweep]
    RUN_PARENT = spec['parent']
    if a.cell is None:
        a.cell = S.baseline_cell(spec)
    if not (Path(a.results_root) / RUN_PARENT / a.cell).is_dir():
        raise SystemExit(f"cell {RUN_PARENT}/{a.cell} not on disk — rsync the run down "
                         "first (refusing to save an empty figure).")
    if a.list:
        for l in _cells_losses(a.results_root, a.cell):
            print(f'  {a.cell}  {l}')
        return
    if a.mode in ('both', 'grid'):
        plot_grid(a.results_root, a.out_root, a.cell)
    if a.mode == 'single':
        plot_single(a.results_root, a.out_root, a.loss, a.cell)
    elif a.mode == 'both':
        for l in _cells_losses(a.results_root, a.cell):
            plot_single(a.results_root, a.out_root, l, a.cell)
    print(f'\nDone -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
