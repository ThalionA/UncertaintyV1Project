# -*- coding: utf-8 -*-
"""Report for the `prodfix_v1` decision run (Q/half/100 ms, H=8, 6 mice, restart
selection on validation, seed 0).

Three figures, each judged on peakiness AND chance-normalised loss — because a fix that
lands peakiness without beating chance is a lobotomy, not a cure (the weight_decay
lesson). Normalised loss = held-out KL(decoded || IO target) / leave-one-out predict-mean;
< 1 beats chance.

  fig1  ARM C — over-sharpening with ZERO hidden units. The decisive bias-vs-capacity
        test: a linear decoder (multinomial logistic regression) has no hidden layer to
        overfit with, so if the projection loss still over-sharpens there, the
        over-sharpening cannot be capacity-driven overfitting.
  fig2  ARM A — the production fix. shape_lambda vs smooth_lambda ladders, peakiness on
        top and normalised loss below, so a peakiness-only "cure" is visible as such.
  fig3  the two calibrated generalists, per mouse (KL vs JS).

Outputs (PNG+SVG) under figures/prodfix/.
Usage:  python diagnostics/prodfix_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

RUN = 'prodfix_v1'
SPLIT = 'stratified_balanced'
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS}


def _cell_dir(results_root, cell):
    d = Path(results_root) / RUN / cell
    if not d.is_dir():
        raise SystemExit(f"missing cell {d} — rsync the prodfix_v1 run down first.")
    subs = [p for p in d.iterdir() if p.is_dir()]
    if not subs:
        raise SystemExit(f"{d} has no slug directory.")
    return subs[0]                      # slug is discovered, not assumed (residual!)


def _res(results_root, cell):
    return sio.loadmat(str(_cell_dir(results_root, cell) / f'{SPLIT}.mat'),
                       simplify_cells=True)['results']


def peaky(results_root, cell, arch):
    r = _res(results_root, cell)
    return np.array([np.asarray(r[m]['Dist'][arch]['decoded'], float).max(1).mean()
                     for m in sorted(r)])


def target_peak(results_root, cell, arch):
    r = _res(results_root, cell)
    return np.array([np.asarray(r[m]['Dist'][arch]['target'], float).max(1).mean()
                     for m in sorted(r)])


def normloss(results_root, cell, arch):
    return np.array(_norm_by_mouse(_res(results_root, cell), arch)[('KL', 'pm')], float)


def _msem(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


# ----------------------------------------------------------------- fig 1: Arm C
def fig_arm_c(results_root, out_root):
    ps.apply()
    groups = [('A_baseline_pca', 'C_nohidden_pca', 'PCA'),
              ('A_reference_kl', 'C_nohidden_kl', 'KL'),
              ('A_reference_js', 'C_nohidden_js', 'JS')]
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharey=True)
    for ax, (arch, alab) in zip(axes, ARCHS):
        tgt = target_peak(results_root, 'A_baseline_pca', arch).mean()
        for j, (mlp, lin, loss) in enumerate(groups):
            for k, (cell, hatch, lbl) in enumerate([(mlp, '', 'H=8'), (lin, '//', 'no hidden layer')]):
                m, s = _msem(peaky(results_root, cell, arch))
                ax.bar(j + (k - 0.5) * 0.36, m, 0.34, yerr=s, capsize=3,
                       color=LCOL[loss], alpha=0.95 if k == 0 else 0.55, hatch=hatch,
                       edgecolor='k', linewidth=0.5,
                       label=lbl if j == 0 else None)
        ax.axhline(tgt, color='k', ls=':', lw=1.4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([ps.loss_label(g[2]) for g in groups], fontsize=8)
        ax.set_title(alab, fontsize=10)
        if arch == 'spat':
            ax.set_ylabel('decoded peakiness (max-prob)')
            h, l = ax.get_legend_handles_labels()
            ax.legend(h + [Line2D([0], [0], color='k', ls=':', lw=1.4)],
                      l + ['IO target'], fontsize=7, frameon=True)
    ps.label_panels(axes)
    fig.suptitle('Removing the ENTIRE hidden layer barely dents the over-sharpening — so it is not '
                 'capacity-driven overfitting (6 mice, mean±sem)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'prodfix_fig1_nohidden')


# ----------------------------------------------------------------- fig 2: Arm A
def fig_arm_a(results_root, out_root):
    ps.apply()
    LAD = {'shape_lambda (PCA + λ·Brier)':
           [('A_baseline_pca', 0.0), ('A_shape3', 0.03), ('A_shape10', 0.1), ('A_shape30', 0.3)],
           'smooth_lambda (output smoothness)':
           [('A_baseline_pca', 0.0), ('A_smooth0p1', 0.1), ('A_smooth0p3', 0.3), ('A_smooth1', 1.0)]}
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), squeeze=False)
    for c, (lab, ladder) in enumerate(LAD.items()):
        for arch, alab in ARCHS:
            col = ps.ARCH[arch]
            xs = list(range(len(ladder)))
            pk = [_msem(peaky(results_root, cell, arch)) for cell, _ in ladder]
            nl = [_msem(normloss(results_root, cell, arch)) for cell, _ in ladder]
            axes[0][c].errorbar(xs, [m for m, _ in pk], yerr=[s for _, s in pk],
                                color=col, lw=1.8, marker='o', ms=4, capsize=2, label=alab)
            axes[1][c].errorbar(xs, [m for m, _ in nl], yerr=[s for _, s in nl],
                                color=col, lw=1.8, marker='o', ms=4, capsize=2, label=alab)
        for r in (0, 1):
            axes[r][c].set_xticks(range(len(ladder)))
            axes[r][c].set_xticklabels([f'{v:g}' for _, v in ladder])
        axes[0][c].set_title(lab, fontsize=9)
        axes[1][c].set_xlabel('λ')
    tgt = target_peak(results_root, 'A_baseline_pca', 'spat').mean()
    for c in (0, 1):
        axes[0][c].axhline(tgt, color='k', ls=':', lw=1.3)
        axes[1][c].axhline(1.0, color='k', ls=':', lw=1.3)
        axes[1][c].set_yscale('log')
    axes[0][0].set_ylabel('decoded peakiness\n(dotted = IO target)')
    axes[1][0].set_ylabel('normalised loss ÷ predict-mean\n(dotted = chance; <1 beats it)')
    axes[0][0].legend(fontsize=7, frameon=True)
    ps.label_panels(axes.ravel())
    fig.suptitle('Both knobs land peakiness on target (top) — but only shape_lambda also beats chance '
                 '(bottom). smooth_lambda is a partial lobotomy.', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'prodfix_fig2_production_fix')


# ----------------------------------------------------------------- fig 3: KL vs JS
def fig_kl_js(results_root, out_root):
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    for ax, (arch, alab) in zip(axes, ARCHS):
        kl, js = normloss(results_root, 'A_reference_kl', arch), normloss(results_root, 'A_reference_js', arch)
        for i, (a, b) in enumerate(zip(kl, js)):
            ax.plot([0, 1], [a, b], '-o', color='0.6', ms=4, lw=1)
        ax.plot([0, 1], [kl.mean(), js.mean()], '-o', color=ps.KL, ms=8, lw=2.5, zorder=5)
        t, p = ttest_rel(js, kl)
        ax.axhline(1.0, color='k', ls=':', lw=1.3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['KL', 'JS'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f'{alab}   (paired t p={p:.3f}, {int((js < kl).sum())}/6 mice favour JS)',
                     fontsize=9)
        if arch == 'spat':
            ax.set_ylabel('normalised loss ÷ predict-mean\n(lower is better; dotted = chance)')
    ps.label_panels(axes)
    fig.suptitle('The two calibrated generalists, per mouse — JS matches KL’s calibration and '
                 'scores at least as well', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'prodfix_fig3_kl_vs_js')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/prodfix')
    a = ap.parse_args()
    fig_arm_c(a.results_root, a.out_root)
    fig_arm_a(a.results_root, a.out_root)
    fig_kl_js(a.results_root, a.out_root)
    print(f'Done -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
