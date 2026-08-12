# -*- coding: utf-8 -*-
"""Does the decoder's over-sharpening come from the RANK bottleneck or from the
NON-LINEARITY? The rr8 cells let you separate them (2026-08-05 meeting item).

Three architectures, matched hyperparameters (raw input, lambda_H 0, dropout 0, wd 0):

    lin   hidden_sizes=[]              ONE full-rank map, MORE parameters
    rr8   [8] + identity activation    rank-<=8 affine logit map, NO non-linearity
    h8    [8] + tanh                   rank-<=8 AND a non-linearity

so the two steps along the x axis isolate one factor each:

    lin -> rr8   adds the RANK bottleneck   (nothing else changes)
    rr8 -> h8    adds the TANH              (same width, same parameter count)

Each panel draws one thin line per mouse across the three architectures, plus the
mean in bold. The per-mouse lines are the point: a mean can move while individual
animals disagree, and that is exactly what happens for EVAR/temporal, where the
rank step is carried by two of six mice. Panel titles carry the n/6 consistency of
each step so the reader never has to take the mean on trust.

Rows are the two axes that actually separate these cells:
  peakiness   decoded peak / IO target peak   (1.0 = on target, >1 = over-sharpened)
  KL / pm     held-out KL over the leave-one-out predict-mean null (<1 beats chance)
The projection loss is deliberately NOT a row: every cell scores ~0.46-0.66 on it,
i.e. the training metric is nearly blind to a 6x difference in peakiness. That
disagreement is itself the finding, and it is printed to stdout.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_rank_vs_nonlinearity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import peakiness_style as ps                                    # noqa: E402
from projflat_report import _common_basis, measures, have       # noqa: E402

import projflat_cells as pcells                                 # noqa: E402

ARCHS = [('lin', 'lin\n(full rank)'), ('rr8', 'rr8\n(rank 8, no NL)'), ('h8', 'h8\n(rank 8 + tanh)')]
# Weighting blocks from the one shared table, so this figure can never disagree with
# the bar/scatter figures about which nine cells are "the headline set".
WEIGHTINGS = pcells.by_arch()
ROWS = [('ratio', 'peakiness\n(decoded peak / target peak)', 1.0),
        ('kl',    'KL / predict-mean\n(<1 beats chance)',    1.0)]


def _collect(results_root, cellmap, arch, key, basis):
    """(n_mice, 3) array of `key` for lin/rr8/h8, or None if a cell is missing."""
    cols = []
    for tok, _ in ARCHS:
        cell = cellmap[tok]
        if not have(results_root, cell):
            return None
        cols.append(np.asarray(measures(results_root, cell, arch, basis)[key], float))
    n = min(len(c) for c in cols)
    return np.column_stack([c[:n] for c in cols])


def plot_arch(results_root, out_root, arch):
    basis = _common_basis(results_root, arch)
    ps.apply()
    nrow, ncol = len(ROWS), len(WEIGHTINGS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.7 * nrow), squeeze=False)
    x = np.arange(3)
    for j, (wlab, cellmap) in enumerate(WEIGHTINGS):
        for i, (key, ylab, ref) in enumerate(ROWS):
            ax = axes[i][j]
            M = _collect(results_root, cellmap, arch, key, basis)
            if M is None:
                ax.text(0.5, 0.5, 'cells missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='0.5')
                ax.set_xticks(x); ax.set_xticklabels([l for _, l in ARCHS], fontsize=7)
                continue
            for row in M:                                  # one thin line per mouse
                ax.plot(x, row, '-o', color='0.62', lw=0.9, ms=3.2,
                        alpha=0.85, zorder=2)
            ax.plot(x, M.mean(0), '-o', color=ps.PCA_EVAR, lw=2.4, ms=7,
                    zorder=3, label='mean')
            ax.axhline(ref, color='k', ls='--', lw=0.9, alpha=0.55, zorder=1)
            d_rank, d_nl = M[:, 0] - M[:, 1], M[:, 1] - M[:, 2]
            ax.set_xticks(x); ax.set_xticklabels([l for _, l in ARCHS], fontsize=7)
            ax.set_xlim(-0.35, 2.35)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(f'{wlab}   rank step {int((d_rank > 0).sum())}/{len(d_rank)} · '
                         f'tanh step {int((d_nl > 0).sum())}/{len(d_nl)}', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, frameon=True, loc='best')
    # y > 1 puts the title above the axes box; save_fig crops tight, so it is kept.
    # (A suptitle at y<=1.0 collides with the top row's own panel titles here.)
    fig.tight_layout()
    fig.suptitle(
        f'{arch.upper()} — is the over-sharpening the RANK bottleneck or the NON-LINEARITY?\n'
        'lin->rr8 adds the rank bottleneck alone; rr8->h8 adds the tanh alone.   '
        'Grey = individual mice (n=6), orange = mean.',
        fontsize=9.5, y=1.055)
    stem = f'projflat_rank_vs_nonlinearity_{arch}'
    ps.save_fig(fig, Path(out_root), stem)
    return stem


def report(results_root):
    """The cross-metric disagreement, printed: projection loss barely moves while
    peakiness moves severalfold across the same cells."""
    print('\nCross-metric disagreement (mean over 6 mice) — the projection loss is the '
          'TRAINING metric for the flat/EVAR cells:')
    print(f"  {'arch':5s} {'weighting':10s} {'lin':>22s} {'rr8':>22s} {'h8':>22s}")
    for arch in ('spat', 'temp'):
        basis = _common_basis(results_root, arch)
        for wlab, cellmap in WEIGHTINGS:
            cells = [cellmap[t] for t, _ in ARCHS]
            if not all(have(results_root, c) for c in cells):
                continue
            out = []
            for c in cells:
                m = measures(results_root, c, arch, basis)
                out.append(f"pk {np.nanmean(m['ratio']):5.2f} proj {np.nanmean(m['proj']):.2f}")
            print(f"  {arch:5s} {wlab:10s} " + ' '.join(f'{o:>22s}' for o in out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()
    for arch in ('spat', 'temp'):
        stem = plot_arch(a.results_root, a.out_root, arch)
        print(f'  {stem}')
    report(a.results_root)
    print(f'\nDone. {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
