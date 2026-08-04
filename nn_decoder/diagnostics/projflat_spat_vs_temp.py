# -*- coding: utf-8 -*-
"""Spatial vs temporal PERFORMANCE across every swept parameter in projflat_v1,
scored under BOTH KL and the projection loss (2026-08-04, Theo's request).

The projection column is essential and is the point of "quantify this": the
projection metric is what the decoder trains on, and it is BLIND to the
over-sharpening that KL exposes — so the spatial-vs-temporal gap can look
completely different under the two metrics. Showing only one hides that.

Layout: rows = scoring metric (KL, projection on a common evar weighting), columns
= the swept parameter. In every panel, two lines — spatial vs temporal — as
normalised loss / leave-one-out predict-mean (< 1 beats chance; lower is better).
All flat/MSE, H=8 (the meeting's chosen loss); linear tracks it (see fig1).

  input dim   raw / pc3 / pc5 / pc10   (un-regularised corner)
  lambda_H    0 / 3e-3 / 1e-2          (raw, dropout 0, wd 0) — temporal-only knob
  dropout     0 / 0.25 / 0.5           (raw, lambda_H 0, wd 0)
  weight decay 0 / 1e-6 / 1e-5         (raw, lambda_H 0, dropout 0)

n=6 mice, mean +/- sem; the paired spatial-temporal difference and its sign count
are annotated (the robust statement at n=6 is the sign count, not the p).

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_spat_vs_temp.py
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
from projflat_report import (_res, have, _common_basis, _norm_by_mouse,  # noqa: E402
                             _proj_common, _mice, _tok)

ARCHS = [('spat', 'spatial', ps.SPATIAL), ('temp', 'temporal', ps.TEMPORAL)]

# (column title, x tick labels, [cell per x])
COLUMNS = [
    ('input dimensionality', ['raw', 'pc3', 'pc5', 'pc10'],
     ['h8_raw_l0_d0_w0', 'h8_pc3_l0_d0_w0', 'h8_pc5_l0_d0_w0', 'h8_pc10_l0_d0_w0']),
    ('entropy penalty lambda_H', ['0', '3e-3', '1e-2'],
     ['h8_raw_l0_d0_w0', 'h8_raw_l0p003_d0_w0', 'h8_raw_l0p01_d0_w0']),
    ('dropout', ['0', '0.25', '0.5'],
     ['h8_raw_l0_d0_w0', 'h8_raw_l0_d0p25_w0', 'h8_raw_l0_d0p5_w0']),
    ('weight decay', ['0', '1e-6', '1e-5'],
     ['h8_raw_l0_d0_w0', 'h8_raw_l0_d0_w1em6', 'h8_raw_l0_d0_w1em5']),
]


def _norm(results_root, cell, arch, metric, basis):
    """Per-mouse normalised loss (<1 beats chance) under metric ∈ {KL, PCA}."""
    r = _res(results_root, cell)
    if metric == 'KL':
        return np.array(_norm_by_mouse(r, arch)[('KL', 'pm')], float)
    return _proj_common(r, arch, basis)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()

    ps.apply()
    basis = {arch: _common_basis(a.results_root, arch) for arch, _, _ in ARCHS}
    metrics = [('KL', 'normalised loss under KL\n(calibrated)'),
               ('PCA', 'normalised loss under PROJECTION\n(the training metric)')]
    fig, axes = plt.subplots(len(metrics), len(COLUMNS),
                             figsize=ps.figsize(len(COLUMNS), len(metrics)), squeeze=False)
    for ri, (metric, ylab) in enumerate(metrics):
        for ci, (title, ticks, cells) in enumerate(COLUMNS):
            ax = axes[ri][ci]
            missing = [c for c in cells if not have(a.results_root, c)]
            if missing:
                ax.text(0.5, 0.5, f'missing:\n{len(missing)} cells', ha='center',
                        va='center', transform=ax.transAxes, fontsize=7)
                continue
            x = np.arange(len(cells))
            per_arch = {}
            for arch, alab, colr in ARCHS:
                ms, ss = [], []
                for c in cells:
                    v = _norm(a.results_root, c, arch, metric, basis[arch])
                    ms.append(v.mean()); ss.append(v.std(ddof=1) / np.sqrt(v.size))
                per_arch[arch] = [_norm(a.results_root, c, arch, metric, basis[arch]) for c in cells]
                ax.errorbar(x, ms, yerr=ss, marker='o', ms=5, lw=1.6, color=colr,
                            capsize=3, label=alab if (ri == 0 and ci == 0) else None)
            # annotate the spatial-temporal paired difference + sign count at each x
            for xi, c in enumerate(cells):
                sp = _norm(a.results_root, c, 'spat', metric, basis['spat'])
                te = _norm(a.results_root, c, 'temp', metric, basis['temp'])
                d = sp - te
                n_temp = int((te < sp).sum())          # temporal better (lower loss)
                _, p = ttest_rel(sp, te) if sp.size > 1 else (np.nan, np.nan)
                ytop = max(np.nanmean(sp), np.nanmean(te))
                ax.annotate(f'{n_temp}/{sp.size}\np={p:.2f}', (xi, ytop),
                            textcoords='offset points', xytext=(0, 6), ha='center',
                            fontsize=5.3, color='0.3')
            ax.axhline(1.0, color='k', ls=':', lw=1.4, zorder=0)
            ax.set_yscale('log')
            ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=7)
            ax.grid(axis='y', alpha=0.25, lw=0.5)
            if ri == 0:
                ax.set_title(title, fontsize=9)
            if ri == len(metrics) - 1:
                ax.set_xlabel(title, fontsize=8)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=8)
    axes[0][0].legend(handles=[Line2D([0], [0], color=ps.SPATIAL, marker='o', lw=1.6),
                               Line2D([0], [0], color=ps.TEMPORAL, marker='o', lw=1.6),
                               Line2D([0], [0], color='k', ls=':', lw=1.4)],
                      labels=['spatial', 'temporal', 'chance'], fontsize=6.5, frameon=True)
    ps.label_panels(axes.ravel())
    fig.suptitle('projflat_v1 — spatial vs temporal PERFORMANCE across each parameter, flat/MSE H=8, n=6 mice. '
                 'Both metrics: KL (calibrated) and the projection loss (the training metric, blind to '
                 'over-sharpening). Annotation = mice where temporal wins, and the paired p (uncorrected).',
                 y=1.02, fontsize=8.3)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'projflat_fig8_spat_vs_temp')


if __name__ == '__main__':
    main()
