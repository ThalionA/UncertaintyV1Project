# -*- coding: utf-8 -*-
"""Spatial vs temporal across the lambda_H ladder (io_hmm_v3), KL metric, predict-mean norm.

Four calibrated groups (KL/JS x h8/h4) plus pca_h8 as the contrast where lambda actually moves
the temporal decoder. lambda_H acts on the TEMPORAL decoder only — the spatial cells are exact
replicates across the ladder; this script VERIFIES that bit-equality per group and says
"replicates" in the legend only once it holds. Expectation from the wide sweep: lambda is inert
for KL/JS (<=4% temporal range), so the point of the figure is to SHOW that flatness.

ONE FIGURE, rows = group, cols = {across animals (per-mouse means, paired over 6 mice),
within each animal (per-trial paired Delta temp - spat)}. x = lambda_H, categorical
{0, 1e-4, 3e-4, 1e-3, 3e-3}. Thin lines = mice, bold = mean +- SEM. Stats on the figure:
per-lambda paired t across mice (stars, n=6) and a mixedlm (per-trial loss ~ arch + (1|mouse))
at lambda=0 only — pooling lambdas would replicate the identical spatial trials 5x
(pseudoreplication), so the mixed model stays within one cell.

No shuffle normalisation here — architecturally biased for spat-vs-temp (log 2026-07-29).

Reuses spat_temp_best_cell.load() (imported, not copied) once per (group, lambda) cell.

Outputs (PNG+SVG) under figures/io_hmm_wide/spat_temp/: spat_temp_by_lambda_KL_pm
Usage:  python diagnostics/spat_temp_by_lambda.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spat_temp_best_cell import load as load_cell, mixedlm_arch  # noqa: E402  (the wheel — do not copy)

RUN = 'io_hmm_v3'
METRIC = 'KL'
NORM = 'pm'
NORM_LABEL = 'normalised loss (÷ predict-mean)'
LAMBDAS = ['0', '1e-4', '3e-4', '1e-3', '3e-3']
GROUPS = [('kl', 'h8'), ('kl', 'h4'), ('js', 'h8'), ('js', 'h4'), ('pca', 'h8')]
EXCLUDE = 2


def _cell(loss, arch, lam):
    return f'{loss}_{arch}_lh{lam}'


def load_group(results_root, loss, arch):
    """{lam: ({(mouse, decoder, norm): per-trial}, mice)} for one loss x width group."""
    return {lam: load_cell(results_root, METRIC, run=RUN,
                           cell=_cell(loss, arch, lam), loss=None)
            for lam in LAMBDAS}


def verify_spatial_replicates(group_data, mice):
    """True iff every mouse's raw spatial per-trial loss vector is bit-equal across lambdas."""
    ref = group_data[LAMBDAS[0]][0]
    for lam in LAMBDAS[1:]:
        cur = group_data[lam][0]
        for m in mice:
            if not np.array_equal(ref[(m, 'spat', 'raw')], cur[(m, 'spat', 'raw')]):
                return False
    return True


def _stars(p):
    return '**' if p < 0.01 else ('*' if p < 0.05 else '')


def figure(all_groups, out_root):
    ps.apply()
    x = np.arange(len(LAMBDAS))
    fig, ax = plt.subplots(len(GROUPS), 2,
                           figsize=ps.figsize(2, len(GROUPS)), sharex='col')
    for r, (loss, arch) in enumerate(GROUPS):
        gd, mice, replicates = all_groups[(loss, arch)]
        glab = f'{loss.upper()} {arch}' + ('  (contrast)' if loss == 'pca' else '')

        # per-mouse means per lambda, and per-trial paired deltas
        sp = np.array([[gd[lam][0][(m, 'spat', NORM)].mean() for lam in LAMBDAS]
                       for m in mice])                       # (n_mice, n_lam)
        tp = np.array([[gd[lam][0][(m, 'temp', NORM)].mean() for lam in LAMBDAS]
                       for m in mice])
        dmu, dsem = np.zeros_like(sp), np.zeros_like(sp)
        for i, m in enumerate(mice):
            for j, lam in enumerate(LAMBDAS):
                s = gd[lam][0][(m, 'spat', NORM)]; t = gd[lam][0][(m, 'temp', NORM)]
                n = min(s.size, t.size); d = t[:n] - s[:n]
                dmu[i, j] = d.mean(); dsem[i, j] = d.std(ddof=1) / np.sqrt(d.size)

        # ---- col 0: across animals ------------------------------------------
        a = ax[r][0]
        for i in range(len(mice)):
            a.plot(x, sp[i], '-', color=ps.SPATIAL, lw=0.8, alpha=0.45, zorder=2)
            a.plot(x, tp[i], '-', color=ps.TEMPORAL, lw=0.8, alpha=0.45, zorder=2)
        for arr, c in ((sp, ps.SPATIAL), (tp, ps.TEMPORAL)):
            a.errorbar(x, arr.mean(0), yerr=arr.std(0, ddof=1) / np.sqrt(arr.shape[0]),
                       color=c, lw=2.2, marker='o', ms=5, capsize=3, zorder=5)
        ps.chance_line(a, 1.0, label=None)
        tmean = tp.mean(0)
        rng = 100.0 * (tmean.max() - tmean.min()) / tmean.mean()
        a.text(0.03, 0.97, f'temporal range {rng:.1f}% of mean', transform=a.transAxes,
               fontsize=6.5, va='top', ha='left',
               bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
        a.set_ylabel(f'{glab}\n{NORM_LABEL}', fontsize=8)
        if r == 0:
            a.set_title(f'across animals (n={len(mice)} mice)', fontsize=9)

        # ---- col 1: within each animal --------------------------------------
        b = ax[r][1]
        for i in range(len(mice)):
            b.plot(x, dmu[i], '-', color='0.62', lw=0.8, alpha=0.7, zorder=2)
        gmu = dmu.mean(0); gsem = dmu.std(0, ddof=1) / np.sqrt(dmu.shape[0])
        b.errorbar(x, gmu, yerr=gsem, color='k', lw=2.2, marker='o', ms=5,
                   capsize=3, zorder=5)
        b.axhline(0, color='k', lw=1.0)
        p6 = np.array([sstats.ttest_rel(tp[:, j], sp[:, j]).pvalue
                       for j in range(len(LAMBDAS))])
        ylo, yhi = b.get_ylim(); yoff = 0.04 * (yhi - ylo)
        for j, p in enumerate(p6):
            s = _stars(p)
            if s:
                b.text(x[j], gmu[j] + gsem[j] + yoff, s, ha='center', va='bottom',
                       fontsize=8)
        # mixedlm at lambda=0 only (pooling lambdas would 5x-replicate the
        # identical spatial trials — pseudoreplication)
        try:
            rows = [pd.DataFrame({'loss': gd['0'][0][(m, dec, NORM)],
                                  'arch': dec, 'mouse': m})
                    for m in mice for dec in ('spat', 'temp')]
            # powell: the default (bfgs) fails its gradient check on these
            # per-trial KL/pm losses; all optimisers agree on beta/SE/p to 4 dp
            # and powell converges warning-free with a positive group variance
            c, se, p, warned = mixedlm_arch(pd.concat(rows, ignore_index=True),
                                            method='powell')
            b.text(0.03, 0.03,
                   f'mixedlm @λ=0: β={c:+.3f}\n(SE {se:.3f}), p={p:.3f}'
                   + (' [not converged]' if warned else ''),
                   transform=b.transAxes, fontsize=6, va='bottom', ha='left',
                   bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
        except Exception:
            pass
        b.set_ylabel('Δ (temporal − spatial)', fontsize=8)
        if r == 0:
            b.set_title(f'within each animal (n={len(mice)} mice,\n'
                        'per-trial paired; * p<0.05, ** p<0.01 across mice)', fontsize=9)

    for c in (0, 1):
        ax[-1][c].set_xticks(x)
        ax[-1][c].set_xticklabels(LAMBDAS, fontsize=8)
        ax[-1][c].set_xlabel('λ_H (temporal decoder only)', fontsize=8)
    replicates_all = all(all_groups[g][2] for g in GROUPS)
    ax[0][0].legend(handles=[
        Line2D([0], [0], color=ps.SPATIAL, lw=2.2, marker='o', ms=5,
               label='spatial' + (' (replicates across λ)' if replicates_all else '')),
        Line2D([0], [0], color=ps.TEMPORAL, lw=2.2, marker='o', ms=5, label='temporal'),
        Line2D([0], [0], color='0.5', ls=':', lw=1.2, label='predict-mean null')],
        fontsize=6, frameon=True, framealpha=0.9, loc='lower right')
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'spat_temp_by_lambda_{METRIC}_{NORM}')


def main(results_root, out_root):
    print(f'run: {RUN}   metric: {METRIC}   norm: {NORM_LABEL}')
    print('lambda_H acts on the temporal decoder only — spatial cells should be exact '
          'replicates across the ladder (verified below).\n')
    all_groups = {}
    for loss, arch in GROUPS:
        gd = load_group(results_root, loss, arch)
        mice = gd[LAMBDAS[0]][1]
        for lam in LAMBDAS[1:]:
            assert gd[lam][1] == mice, f'{loss}_{arch}: mouse set differs at lh{lam}'
        rep = verify_spatial_replicates(gd, mice)
        all_groups[(loss, arch)] = (gd, mice, rep)
        print(f'== {loss.upper()} {arch}  (spatial replicates across λ: '
              f'{"VERIFIED bit-equal" if rep else "FAILED — NOT bit-equal!"}) ==')
        keep = [i for i, m in enumerate(mice) if m != EXCLUDE]
        for lam in LAMBDAS:
            sp = np.array([gd[lam][0][(m, 'spat', NORM)].mean() for m in mice])
            tp = np.array([gd[lam][0][(m, 'temp', NORM)].mean() for m in mice])
            d = tp - sp
            p6 = sstats.ttest_rel(tp, sp).pvalue
            p5 = sstats.ttest_rel(tp[keep], sp[keep]).pvalue
            print(f'  λ={lam:5s}  spat {sp.mean():7.4f}  temp {tp.mean():7.4f}  '
                  f'Δ {d.mean():+7.4f} ± {d.std(ddof=1) / np.sqrt(d.size):.4f}   '
                  f'p(n=6)={p6:.4f}  p(n=5, excl M{EXCLUDE})={p5:.4f}')
        tmean = np.array([np.mean([gd[lam][0][(m, 'temp', NORM)].mean() for m in mice])
                          for lam in LAMBDAS])
        print(f'  temporal mean range across λ: '
              f'{100.0 * (tmean.max() - tmean.min()) / tmean.mean():.1f}% of mean\n')
    figure(all_groups, out_root)
    print(f'Done -> {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/io_hmm_wide/spat_temp')
    ap.add_argument('--run', default=RUN)
    a = ap.parse_args()
    RUN = a.run
    main(a.results_root, a.out_root)
