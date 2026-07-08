# -*- coding: utf-8 -*-
"""Decoded peakiness vs each swept hyperparameter — the over-sharpening the loss work
is actually about (not the fit-loss train–val gap, which is blind to it; see
`PCA-Peakiness-Mechanism.md` §8 and GOTCHAS).

Peakiness = across-trial mean decoded max-probability (IO target ≈ 0.059; higher =
over-confident). For each `hpsweep_wide` axis (λ_H, dropout, width, activation,
patience) it plots peakiness vs the axis value, one line per loss, with the IO-target
reference. Question: which generic knob moves PCA's over-sharpening toward target — and
do the calibrated losses (KL/JS) stay put? (The docs predict width/early-stop *cap* it,
dropout/λ_H/activation don't, and the real fix is loss-side — λ·Brier / smooth_lambda —
which is NOT an axis here.)

Layout: rows = arch (spatial, temporal), cols = axis, SHARED y so magnitudes compare.
Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  peakiness_vs_hparams.png
Usage:  python diagnostics/peakiness_vs_hparams.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from plot_overfit_vs_width import load_peak, _msem  # noqa: E402

RUN_PARENT = 'hpsweep_wide'
LOSSES = ['PCA', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
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


def _xpos(cfg):
    return list(range(len(cfg['vals']))) if cfg['xtype'] == 'index' else list(cfg['vals'])


def main(results_root, out_root, split, axes):
    ps.apply()
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    io_all = []
    print(f"peakiness (decoded max-prob), split={split}")
    for r, (arch, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], AXES[axis]
            xall = _xpos(cfg)
            for loss in LOSSES:
                xs, ys, es = [], [], []
                for x, v in zip(xall, cfg['vals']):
                    run = f"{RUN_PARENT}/{cell_for(axis, v)}"
                    pk, tgt = load_peak(results_root, run, 'Q', loss, 'half', 100, split, arch)
                    m, s = _msem(pk)
                    if m is not None:
                        xs.append(x); ys.append(m); es.append(s if s is not None else 0.0)
                        if tgt is not None:
                            io_all.append(tgt)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=LCOL[loss], lw=1.6, marker='o', ms=4,
                                capsize=2, label=ps.loss_label(loss))
            if cfg['xtype'] == 'log2':
                ax.set_xscale('log', base=2); ax.set_xticks(cfg['vals']); ax.set_xticklabels(cfg['vals'])
            elif cfg['xtype'] == 'index':
                ax.set_xticks(xall); ax.set_xticklabels(cfg['vals'], fontsize=7)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\ndecoded max-prob')
            if r == 0:
                ax.set_title(axis, fontsize=9)
    io = float(np.mean(io_all)) if io_all else np.nan
    for ax in axgrid.ravel():
        if np.isfinite(io):
            ax.axhline(io, color='k', ls=':', lw=1.3)
    if LOSSES:
        axgrid[0][0].plot([], [], 'k:', lw=1.3, label='IO target')
    handles, lbls = axgrid[0][0].get_legend_handles_labels()
    axgrid[0][0].legend(handles, lbls, fontsize=6.5, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'Decoded peakiness vs each hyperparameter (shared y; IO target ≈ {io:.3f}, dotted). '
                 f'Which knob tames PCA over-sharpening?', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'peakiness_vs_hparams')
    print(f'IO-target peakiness ≈ {io:.3f}')
    print(f'Done -> {Path(out_root).resolve()}/peakiness_vs_hparams.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--axes', nargs='+', default=DEFAULT_AXES, choices=list(AXES))
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.split, a.axes)
