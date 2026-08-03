# -*- coding: utf-8 -*-
"""Spatial vs temporal, head to head, for every `flatevar_v1` decoder that survived
the weight-decay annihilation.

Paired within mouse: each animal contributes one (spatial, temporal) pair per
decoder, so Delta = spatial - temporal is a within-animal contrast and the n=6
paired test is the generalisation claim. Trial-level tests across animals would be
pseudoreplication (GOTCHAS) and are not done here.

All three axes, because they dissociate:
  * over-sharpening  = peakiness / IO target
  * normalised loss under KL          (calibrated metric)
  * normalised loss under PROJECTION  (common evar weighting for every decoder)

Sign convention: Delta < 0 means SPATIAL is lower. On the loss rows lower is
better, so Delta < 0 = spatial wins. On the bias row the target is 1.0, not 0, so
a lower ratio is only better if it does not undershoot — read it with the
assessment figure, which shows the absolute levels.

Reported per decoder: the paired-t p and the number of mice sharing the sign.
The sign count is the robust statement at n=6 (sign-test floor p=0.031); several
uncorrected tests are run, so no single p should be read on its own.

Outputs (PNG+SVG) under figures/flatevar/.
Usage:  python diagnostics/flatevar_spat_vs_temp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flatevar_assessment import ROWS, _path, _common_basis, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/flatevar')
    a = ap.parse_args()

    basis = {arch: _common_basis(a.results_root, arch) for arch in ('spat', 'temp')}
    data = {}
    for lab, spec, colr in ROWS:
        p = _path(a.results_root, spec)
        if p is None:
            continue
        s = load(p, 'spat', basis['spat'])
        t = load(p, 'temp', basis['temp'])
        if min(len(s[0]), len(t[0])) == 0:
            continue
        n = min(len(s[0]), len(t[0]))
        data[lab] = (colr, [(s[i][:n], t[i][:n]) for i in range(3)])

    ps.apply()
    fig, axes = plt.subplots(3, 1, figsize=ps.figsize(2, 3), sharex=True)
    labs = [r[0] for r in ROWS if r[0] in data]
    x = np.arange(len(labs))
    METRICS = [('over-sharpening\nΔ (spatial − temporal)', 0),
               ('normalised loss under KL\nΔ (spatial − temporal)', 1),
               ('normalised loss under PROJECTION\nΔ (spatial − temporal)', 2)]

    for ri, (ylab, mi) in enumerate(METRICS):
        ax = axes[ri]
        for xi, lab in zip(x, labs):
            colr, per_metric = data[lab]
            sv, tv = per_metric[mi]
            d = sv - tv
            jit = np.linspace(-0.17, 0.17, d.size)
            ax.plot(xi + jit, d, 'o', ms=3.5, mfc='none', mec=colr, mew=1.0,
                    alpha=0.85, ls='none', zorder=2)
            m = d.mean()
            sem = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
            ax.errorbar(xi, m, yerr=sem, fmt='D', ms=6.5, color=colr, mec='k',
                        mew=0.6, capsize=4, lw=2.0, zorder=4)
            # annotate sign consistency + paired p (the robust statement is the count)
            if d.size > 1:
                _, pv = ttest_rel(sv, tv)
                n_same = int((d < 0).sum()) if m < 0 else int((d > 0).sum())
                ax.annotate(f'{n_same}/{d.size}\np={pv:.3f}',
                            (xi, m), textcoords='offset points',
                            xytext=(0, 13 if m >= 0 else -26), ha='center',
                            fontsize=5.6, color='0.25')
        ax.axhline(0.0, color='k', ls=':', lw=1.4, zorder=1)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=38, ha='right', fontsize=7)
        ax.set_xlim(-0.6, len(labs) - 0.4)
        if ri == 0:
            ax.legend(handles=[
                Line2D([0], [0], marker='o', mfc='none', mec='0.35', ls='none', ms=4),
                Line2D([0], [0], marker='D', color='0.35', ls='none', ms=6)],
                labels=['one mouse', 'mean ± sem'], fontsize=6.5, frameon=True,
                loc='best')

    ps.label_panels(axes)
    fig.suptitle('Spatial vs temporal, paired within mouse (n=6). Below 0 = spatial lower; on the '
                 'loss rows that means spatial wins. Annotation = mice sharing the sign, and the '
                 'paired-t p (uncorrected — read the sign count).', y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'flatevar_fig6_spat_vs_temp')


if __name__ == '__main__':
    main()
