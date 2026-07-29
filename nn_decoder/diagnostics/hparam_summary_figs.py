# -*- coding: utf-8 -*-
"""Four summary figures over the hyperparameter sweep, plotted to one consistent standard.

Standard applied throughout: minimal titles (what is plotted, no exposition), explicit x and y
labels, a legend on every figure, mean +/- SEM over the 6 mice on every point, and IDENTICAL
axis limits for the spatial and temporal rows so the two architectures are directly comparable.

  fig1  peakiness vs each hyperparameter
  fig2  normalised loss (performance vs chance) vs each hyperparameter
  fig3  peakiness vs normalised loss (how the two relate)
  fig4  overfitting (val/train fit-loss) vs each hyperparameter

Data: `hpsweep_v2` for all axes. The hidden-width axis additionally carries WIDTH = 0, the
no-hidden-layer (linear) decoder, taken from `prodfix_v1` C_nohidden_*. Those two runs differ
only in the restart-selection rule; at the shared H=8 baseline they agree to <=4% on PCA and are
identical for KL/JS, so the splice is safe. Wasserstein has no width-0 cell and is simply absent
from that point.

Every axis is index-plotted with its literal values as tick labels — required because width now
starts at 0, where a log scale is undefined.

Outputs (PNG+SVG) under figures/hparam_summary/.
Usage:  python diagnostics/hparam_summary_figs.py
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
from story_figures import _results, peaky, tgt_peak, normloss  # noqa: E402

LOSSES = ['PCA', 'KL', 'JS', 'Wasserstein']
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
NOHIDDEN = {'PCA': 'C_nohidden_pca', 'KL': 'C_nohidden_kl', 'JS': 'C_nohidden_js'}
N_CATS = 91

# axis key -> (label, values). Width carries 0 = no hidden layer.
AXES = [
    ('lambda_H',     'entropy λ_H',   [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
    ('dropout',      'dropout p',     [0, 0.1, 0.25, 0.5, 0.75, 0.9]),
    ('width',        'hidden width',  [0, 2, 4, 8, 16, 32]),
    ('activation',   'activation',    ['tanh', 'relu', 'gelu']),
    ('patience',     'patience',      [0, 10, 20, 40]),
    ('weight_decay', 'weight decay',  [0, 1e-4, 1e-3, 1e-2, 1e-1]),
    ('shape_lambda', 'shape λ (×100)', [0, 1, 3, 10, 30]),
]


def _cell_ref(axis, val, loss):
    """(run, cell, slug_loss) for one axis value, or None if that combination does not exist."""
    spec = S.SPECS['v2']
    if axis == 'width' and val == 0:                 # linear decoder, from prodfix_v1
        cell = NOHIDDEN.get(loss)
        return None if cell is None else ('prodfix_v1', cell, None)
    if axis == 'shape_lambda' and loss != 'PCA':     # PCA-loss only knob
        return None
    return ('hpsweep_v2', S.cell_for(spec, axis, val), loss)


def _overfit(run, cell, slug_loss, arch):
    from story_figures import _slug_dir
    ck = _slug_dir('results', run, cell, loss=slug_loss) / 'checkpoints'
    rs = []
    for pt in sorted(ck.glob('mouse_*_stratified_balanced.pt')):
        h = (torch.load(str(pt), map_location='cpu', weights_only=False).get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if t and v and t[-1] > 0:
            rs.append(v[-1] / t[-1])
    return np.array(rs, float)


def _ms(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return None, None
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


def collect(metric):
    """{(axis, loss, arch): [(xpos, mean, sem)]} for one metric."""
    out = {}
    for axis, _lab, vals in AXES:
        for loss in LOSSES:
            for arch, _a in ARCHS:
                pts = []
                for x, v in enumerate(vals):
                    ref = _cell_ref(axis, v, loss)
                    if ref is None:
                        continue
                    run, cell, slug = ref
                    try:
                        if metric == 'overfit':
                            m, s = _ms(_overfit(run, cell, slug, arch))
                        else:
                            res = _results('results', run, cell, loss=slug)
                            m, s = _ms(peaky(res, arch) if metric == 'peaky'
                                       else normloss(res, arch))
                    except SystemExit:
                        continue
                    if m is not None:
                        pts.append((x, m, s))
                if pts:
                    out[(axis, loss, arch)] = pts
    return out


def _panel_grid(metric, ylabel, fname, out_root, logy=False, ref=None, ref_lab=None):
    ps.apply()
    data = collect(metric)
    if not data:
        raise SystemExit('no cells loaded — rsync hpsweep_v2 and prodfix_v1 down first.')
    n = len(AXES)
    fig, ax = plt.subplots(2, n, figsize=ps.figsize(n, 2), sharey=True, squeeze=False)
    # Shared limits are set from the plotted data (plus the reference line), not left to
    # autoscale: with sharey + a log axis matplotlib otherwise pads to decades that hold
    # no data. Both rows keep IDENTICAL limits so the architectures stay comparable.
    allv = [m for pts in data.values() for _x, m, _s in pts if np.isfinite(m)]
    alle = [m + (s or 0) for pts in data.values() for _x, m, s in pts if np.isfinite(m)]
    lo, hi = min(allv + ([ref] if ref is not None else [])), max(alle + ([ref] if ref is not None else []))
    for r, (arch, alab) in enumerate(ARCHS):
        for c, (axis, xlab, vals) in enumerate(AXES):
            a = ax[r][c]
            for loss in LOSSES:
                pts = data.get((axis, loss, arch))
                if not pts:
                    continue
                x, m, s = zip(*pts)
                a.errorbar(x, m, yerr=s, color=ps.color(loss), lw=1.5, marker='o', ms=3.5,
                           capsize=2, label=ps.loss_label(loss) if (r == 0 and c == 0) else None)
            if ref is not None:
                a.axhline(ref, color='k', ls=':', lw=1.2)
            if logy:
                a.set_yscale('log')
            a.set_xticks(range(len(vals)))
            a.set_xticklabels([f'{v:g}' if not isinstance(v, str) else v for v in vals],
                              fontsize=6.5, rotation=45 if isinstance(vals[0], str) else 0)
            if r == 1:
                a.set_xlabel(xlab, fontsize=8)
            if c == 0:
                a.set_ylabel(f'{alab}\n{ylabel}', fontsize=8)
            if r == 0:
                a.set_title(axis.replace('_', ' '), fontsize=8)
    if logy:
        ax[0][0].set_ylim(lo * 0.7, hi * 1.5)
    else:
        pad = 0.08 * (hi - lo)
        ax[0][0].set_ylim(max(0.0, lo - pad), hi + pad)
    h, l = ax[0][0].get_legend_handles_labels()
    if ref is not None:
        h.append(Line2D([0], [0], color='k', ls=':', lw=1.2)); l.append(ref_lab)
    ax[0][0].legend(h, l, fontsize=6, frameon=True)
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), fname)


def fig_relation(out_root):
    """fig3 — peakiness vs normalised loss, identical limits on both panels."""
    ps.apply()
    pk, nl = collect('peaky'), collect('normloss')
    fig, ax = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharex=True, sharey=True)
    io = tgt_peak(_results('results', 'hpsweep_v2', S.baseline_cell(S.SPECS['v2']), loss='PCA'),
                  'spat').mean()
    for c, (arch, alab) in enumerate(ARCHS):
        for loss in LOSSES:
            xs, ys = [], []
            for key in pk:
                if key[1] != loss or key[2] != arch:
                    continue
                d_pk = {p[0]: p[1] for p in pk[key]}
                d_nl = {p[0]: p[1] for p in nl.get(key, [])}
                for k in set(d_pk) & set(d_nl):
                    xs.append(d_pk[k]); ys.append(d_nl[k])
            if xs:
                ax[c].scatter(xs, ys, s=26, color=ps.color(loss), alpha=0.75, lw=0,
                              label=ps.loss_label(loss) if c == 0 else None)
        ax[c].axvline(io, color='k', ls=':', lw=1.2)
        ax[c].axhline(1.0, color='0.45', ls='--', lw=1.2)
        ax[c].set_xscale('log'); ax[c].set_yscale('log')
        ax[c].set_xlabel('decoded peakiness (max-prob)', fontsize=8)
        ax[c].set_title(alab, fontsize=9)
    ax[0].set_ylabel('normalised loss ÷ predict-mean', fontsize=8)
    h, l = ax[0].get_legend_handles_labels()
    h += [Line2D([0], [0], color='k', ls=':', lw=1.2),
          Line2D([0], [0], color='0.45', ls='--', lw=1.2)]
    l += ['IO target', 'chance']
    ax[0].legend(h, l, fontsize=6.5, frameon=True)
    ps.label_panels(ax)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'hp_fig3_peakiness_vs_performance')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/hparam_summary')
    a = ap.parse_args()
    io = tgt_peak(_results('results', 'hpsweep_v2', S.baseline_cell(S.SPECS['v2']), loss='PCA'),
                  'spat').mean()
    _panel_grid('peaky', 'decoded peakiness', 'hp_fig1_peakiness', a.out_root,
                ref=io, ref_lab='IO target')
    _panel_grid('normloss', 'normalised loss', 'hp_fig2_performance', a.out_root,
                logy=True, ref=1.0, ref_lab='chance')
    fig_relation(a.out_root)
    _panel_grid('overfit', 'val / train fit-loss', 'hp_fig4_overfitting', a.out_root,
                logy=True, ref=1.0, ref_lab='no overfitting')
    print(f'Done -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
