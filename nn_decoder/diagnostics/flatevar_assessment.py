# -*- coding: utf-8 -*-
"""Two-axis assessment of every `flatevar_v1` cell that survived the weight-decay
annihilation, as a single figure.

House rule: a decoder is judged TWICE, never once.
  * over-sharpening = decoded peakiness / IO target      (bias; 1.0 = on target)
  * normalised loss = held-out KL(decoded||target)
                      / leave-one-out predict-mean       (< 1 beats chance)
A decoder that lands the first and fails the second is a lobotomy, not a cure.

Every point is one mouse (n=6), because the robust statement in this project is
sign consistency across animals, not the group mean — so the reader should be
able to count the animals rather than take a mean on trust.

The wd-matched evar baseline is imported from `hpsweep_v2`, which is the only run
containing an evar-weighted PCA cell at weight_decay=0 (the flat cells only
survive at wd=0, so comparing them against the wd=1e-4 baseline confounds the
weighting with the decay). Caveat carried in the axis label: hpsweep_v2 predates
the 2026-07-16 restart-selection fix.

Usage:  python diagnostics/flatevar_assessment.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

SPLIT = 'stratified_balanced'
HPSWEEP_WD0 = ('hpsweep_v2/lam0p003_drop0_acttanh_h8_pat0_vf0p2_wd0_shp0/'
               'Q_PCA_half_100ms_all')

# (label, results-relative path or flatevar cell, colour, marker)
ROWS = [
    ('KL  (H=8, wd 1e-4)',        ('flatevar_v1', 'R_reference_kl'),  ps.KL),
    ('JS  (H=8, wd 1e-4)',        ('flatevar_v1', 'R_reference_js'),  ps.JS),
    ('FLAT  (H=8, wd 0)',         ('flatevar_v1', 'A_flat_wd0'),      ps.FLAT_EVAR),
    ('FLAT  (linear, wd 0)',      ('flatevar_v1', 'B_flat_lin_wd0'),  ps.FLAT_EVAR),
    ('evar  (H=8, wd 0)  matched', ('RAW', HPSWEEP_WD0),              ps.PCA_EVAR),
    ('evar  (H=8, wd 1e-4)',      ('flatevar_v1', 'R_evar_base'),     ps.PCA_EVAR),
    ('evar + input PCA k=16',     ('flatevar_v1', 'C_evar_npc16'),    ps.PCA_EVAR),
]


def _path(results_root, spec):
    kind, val = spec
    if kind == 'RAW':
        return f'{results_root}/{val}/{SPLIT}.mat'
    hits = glob.glob(f'{results_root}/{kind}/{val}/*/{SPLIT}.mat')
    return hits[0] if hits else None


def load(path, arch):
    r = sio.loadmat(path, simplify_cells=True)['results']
    mice = sorted(k for k in r if isinstance(r[k], dict)
                  and isinstance(r[k].get('Dist'), dict))
    pk = np.array([np.asarray(r[m]['Dist'][arch]['decoded'], float).max(1).mean()
                   for m in mice])
    tg = np.array([np.asarray(r[m]['Dist'][arch]['target'], float).max(1).mean()
                   for m in mice])
    nl = np.array(_norm_by_mouse(r, arch)[('KL', 'pm')], float)
    return pk / tg, nl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/flatevar')
    a = ap.parse_args()

    ps.apply()
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), sharey=True)
    y = np.arange(len(ROWS))[::-1]          # first row at the top

    for ri, (arch, alab) in enumerate((('spat', 'spatial'), ('temp', 'temporal'))):
        for ci, metric in enumerate(('bias', 'perf')):
            ax = axes[ri][ci]
            for yi, (lab, spec, colr) in zip(y, ROWS):
                p = _path(a.results_root, spec)
                if p is None:
                    continue
                ratio, nl = load(p, arch)
                v = ratio if metric == 'bias' else nl
                # one open circle per mouse, jittered on y so ties stay visible
                jit = np.linspace(-0.17, 0.17, v.size)
                ax.plot(v, yi + jit, 'o', ms=3.5, mfc='none', mec=colr,
                        mew=1.0, alpha=0.85, ls='none')
                m = v.mean()
                sem = v.std(ddof=1) / np.sqrt(v.size)
                ax.plot(m, yi, 'D', ms=6.5, color=colr, mec='k', mew=0.6, zorder=4)
                ax.plot([m - sem, m + sem], [yi, yi], '-', color=colr, lw=2.2,
                        zorder=3)
            ax.axvline(1.0, color='k', ls=':', lw=1.4, zorder=1)
            ax.set_xscale('log')
            ax.set_yticks(y)
            ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.5)
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
            ax.grid(axis='x', alpha=0.25, lw=0.5)
            if ri == 0:
                ax.set_title('over-sharpening\n(peakiness / IO target)' if metric == 'bias'
                             else 'performance\n(loss / predict-mean)', fontsize=9)
            if ri == 1:
                ax.set_xlabel('dotted line = ON TARGET' if metric == 'bias'
                              else 'dotted line = CHANCE  (left is better)', fontsize=8)
            if ci == 0:
                ax.text(-0.62, 0.5, alab, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=11, fontweight='bold')

    axes[0][1].legend(handles=[
        Line2D([0], [0], marker='o', mfc='none', mec='0.35', ls='none', ms=4),
        Line2D([0], [0], marker='D', color='0.35', ls='none', ms=6),
    ], labels=['one mouse', 'mean ± sem'], fontsize=7, frameon=True, loc='lower right')

    ps.label_panels(axes.ravel())
    fig.suptitle('flatevar_v1 — every surviving decoder judged on BOTH axes, n=6 mice. '
                 'Cells at weight_decay > 0 under flat weighting were annihilated and are '
                 'excluded.', y=1.03, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'flatevar_fig5_assessment')


if __name__ == '__main__':
    main()
