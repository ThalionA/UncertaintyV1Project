# -*- coding: utf-8 -*-
"""Overfitting axis for the flatevar family — val/train fit-loss ratio.

The third of the three axes (peakiness = bias, normalised loss = performance,
this = variance). Ratio at the final epoch; 1.0 means train and held-out fit
equally well, higher means the fit does not generalise.

READ THIS TOGETHER WITH THE OTHER TWO AXES. A decoder annihilated by weight
decay has val/train = 1.00 exactly, because a uniform output fits train and val
equally badly — perfect apparent regularisation, and a corpse. Cells flagged by
`flatevar_assessment` as collapsed/suppressed are marked here for that reason;
low overfitting is only good news if the decoder is alive.

Outputs (PNG+SVG) under figures/flatevar/.
Usage:  python diagnostics/flatevar_overfitting.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from overfitting_vs_hparams import _overfit_ratio  # noqa: E402
from flatevar_assessment import _path  # noqa: E402

# (label, spec, colour, is_dead) — dead flagged from the assessment run
CELLS = [
    ('FLAT wd=0\n(H=8)',        ('flatevar_v1', 'A_flat_wd0'),      ps.FLAT_EVAR, False),
    ('FLAT wd=0\n(linear)',     ('flatevar_v1', 'B_flat_lin_wd0'),  ps.FLAT_EVAR, False),
    ('FLAT wd=1e-4\n(H=8)',     ('flatevar_v1', 'A_flat_base'),     ps.FLAT_EVAR, True),
    ('evar wd=1e-4\n(H=8)',     ('flatevar_v1', 'R_evar_base'),     ps.PCA_EVAR,  False),
    ('KL wd=1e-4',              ('flatevar_v1', 'R_reference_kl'),  ps.KL,        False),
    ('JS wd=1e-4',              ('flatevar_v1', 'R_reference_js'),  ps.JS,        False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/flatevar')
    a = ap.parse_args()

    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharey=True)
    x = np.arange(len(CELLS))
    for ci, (arch, alab) in enumerate((('spat', 'spatial'), ('temp', 'temporal'))):
        ax = axes[ci]
        for xi, (lab, spec, colr, dead) in zip(x, CELLS):
            p = _path(a.results_root, spec)
            if p is None:
                continue
            ck = Path(p).parent / 'checkpoints'
            m, s = _overfit_ratio(ck, arch)
            if m is None:
                continue
            ax.errorbar(xi, m, yerr=s, fmt='D', ms=7, color=colr, mec='k', mew=0.6,
                        capsize=4, lw=2.0, alpha=0.35 if dead else 1.0, zorder=3)
            if dead:
                ax.plot(xi, m, 'x', ms=11, mew=2.5, color='k', zorder=5)
        ax.axhline(1.0, color='k', ls=':', lw=1.4, zorder=1)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in CELLS], fontsize=7)
        ax.set_xlim(-0.6, len(CELLS) - 0.4)
        ax.set_title(alab, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        if ci == 0:
            ax.set_ylabel('val / train fit-loss\n(dotted = no overfitting)', fontsize=8)
            ax.legend(handles=[Line2D([0], [0], color='k', marker='x', ls='none',
                                      ms=9, mew=2.5)],
                      labels=['annihilated: ratio 1.0 is a corpse,\nnot regularisation'],
                      fontsize=6.5, frameon=True, loc='best')
    ps.label_panels(axes)
    fig.suptitle('Overfitting (val ÷ train fit-loss, final epoch, 200 ep with NO early stopping), '
                 'n=6 mice. The flat-weighted spatial decoder at wd=0 overfits hard; the temporal '
                 'one does not.', y=1.02, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'flatevar_fig9_overfitting')


if __name__ == '__main__':
    main()
