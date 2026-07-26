# -*- coding: utf-8 -*-
"""Sweep specs shared by the hpsweep analysis diagnostics.

One place for the parent run-name, baseline configuration, cell-directory naming and
per-axis grids of each sweep, so the plotters take `--sweep {v1,v2}` instead of each
hardcoding one:

  v1 = `hpsweep_wide`  — H=32 base (later shown to be ~18x overparameterised).
  v2 = `hpsweep_v2`    — H=8 base, width re-centred {2..32}, plus two new axes:
                         weight_decay, and shape_lambda (PCA-only — the loss-side
                         over-sharpening cure, lambda_Brier = shape/100).

Cell dirs encode the NON-loss axes only (the loss is a slug inside), so the 4 losses
share a cell dir — see the dir-count gotcha in PROJECT_LOG.
"""

from __future__ import annotations

LOSS_SLUG = {'PCA': 'Q_PCA_half_100ms_all', 'KL': 'Q_KL_half_100ms',
             'JS': 'Q_JS_half_100ms', 'Wasserstein': 'Q_Wasserstein_half_100ms'}
LOSSES = ['PCA', 'KL', 'JS', 'Wasserstein']


def _g(x) -> str:
    return f"{x:g}".replace('.', 'p').replace('-', 'm')


def _ax(key, vals, xlabel, xtype, losses=None, extra=None):
    """xtype: 'lin' | 'log2' | 'index' (categorical/uneven). losses=None -> all losses.
    extra: additional baseline overrides this axis is swept at (e.g. val_fraction is
    only meaningful with early stopping active, so it was swept at patience=20)."""
    return dict(key=key, vals=vals, xlabel=xlabel, xtype=xtype, losses=losses,
                extra=extra or {})


SPECS = {
    'v1': dict(
        parent='hpsweep_wide',
        base=dict(lam=3e-3, drop=0.0, act='tanh', width=32, pat=0, vf=0.2),
        axes={
            'lambda_H':   _ax('lam',   [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], 'entropy λ_H', 'index'),
            'dropout':    _ax('drop',  [0, 0.1, 0.25, 0.5, 0.75, 0.9], 'dropout p', 'lin'),
            'width':      _ax('width', [4, 8, 16, 32, 64], 'hidden width', 'log2'),
            'activation': _ax('act',   ['tanh', 'relu', 'gelu'], 'activation', 'index'),
            'patience':   _ax('pat',   [0, 10, 20, 40], 'patience', 'lin'),
            # Swept at patience=20 (inert at patience 0, where nothing early-stops).
            'val_fraction': _ax('vf', [0.1, 0.2, 0.3], 'val fraction  (@pat 20)', 'index',
                                extra={'pat': 20}),
        },
    ),
    'v2': dict(
        parent='hpsweep_v2',
        base=dict(lam=3e-3, drop=0.0, act='tanh', width=8, pat=0, vf=0.2, wd=1e-4, shape=0),
        axes={
            'lambda_H':     _ax('lam',   [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], 'entropy λ_H', 'index'),
            'dropout':      _ax('drop',  [0, 0.1, 0.25, 0.5, 0.75, 0.9], 'dropout p', 'lin'),
            'width':        _ax('width', [2, 4, 8, 16, 32], 'hidden width', 'log2'),
            'activation':   _ax('act',   ['tanh', 'relu', 'gelu'], 'activation', 'index'),
            'patience':     _ax('pat',   [0, 10, 20, 40], 'patience', 'lin'),
            'weight_decay': _ax('wd',    [0, 1e-4, 1e-3, 1e-2, 1e-1], 'weight decay', 'index'),
            # PCA-only: the loss-side cure (PCA + (shape/100)·Brier).
            'shape_lambda': _ax('shape', [0, 1, 3, 10, 30], 'shape_lambda  (100·λ_Brier)',
                                'index', losses=['PCA']),
            # Swept at patience=20 (inert at patience 0, where nothing early-stops).
            'val_fraction': _ax('vf', [0.1, 0.2, 0.3], 'val fraction  (@pat 20)', 'index',
                                extra={'pat': 20}),
        },
    ),
}

# Axes drawn by default. `val_fraction` is deliberately excluded: it is swept at a
# different patience from the baseline, so it is not a clean one-at-a-time column —
# request it explicitly with `--axes ... val_fraction` (it IS addressable, which is
# the point; those cells previously had no consumer at all).
DEFAULT_AXES = {
    'v1': ['lambda_H', 'dropout', 'width', 'activation', 'patience'],
    'v2': ['lambda_H', 'dropout', 'width', 'activation', 'patience',
           'weight_decay', 'shape_lambda'],
}


def cell_name(spec: dict, overrides: dict | None = None) -> str:
    t = {**spec['base'], **(overrides or {})}
    parts = [f"lam{_g(t['lam'])}", f"drop{_g(t['drop'])}", f"act{t['act']}",
             f"h{t['width']}", f"pat{t['pat']}", f"vf{_g(t['vf'])}"]
    if 'wd' in spec['base']:                       # v2 encodes the two extra axes
        parts += [f"wd{_g(t['wd'])}", f"shp{_g(t['shape'])}"]
    return '_'.join(parts)


def cell_for(spec: dict, axis: str, value) -> str:
    """Cell dir for one axis moved off baseline (plus any per-axis `extra` overrides
    the axis was swept at, e.g. val_fraction @ patience=20)."""
    cfg = spec['axes'][axis]
    return cell_name(spec, {**cfg.get('extra', {}), cfg['key']: value})


def baseline_cell(spec: dict) -> str:
    return cell_name(spec)


def axis_losses(spec: dict, axis: str) -> list:
    """Losses this axis is defined for (shape_lambda is PCA-only)."""
    return spec['axes'][axis]['losses'] or LOSSES


def xpos(cfg: dict) -> list:
    """x positions: categorical/uneven axes are index-plotted."""
    return list(range(len(cfg['vals']))) if cfg['xtype'] == 'index' else list(cfg['vals'])


def apply_xaxis(ax, cfg) -> None:
    if cfg['xtype'] == 'log2':
        ax.set_xscale('log', base=2); ax.set_xticks(cfg['vals']); ax.set_xticklabels(cfg['vals'])
    elif cfg['xtype'] == 'index':
        ax.set_xticks(xpos(cfg)); ax.set_xticklabels(cfg['vals'], fontsize=7)
