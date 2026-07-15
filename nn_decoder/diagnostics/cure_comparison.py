# -*- coding: utf-8 -*-
"""The v2 headline: two knobs that both drive PCA peakiness down, but only one makes a
GOOD decoder.

For the PCA over-sharpening, `shape_lambda` (the loss-side fix, PCA + (λ=shape/100)·Brier)
and strong `weight_decay` BOTH collapse the decoded peakiness — so on peakiness alone they
look equivalent. But the chance-normalised KL-skill (÷ predict-mean; <1 beats chance)
separates them cleanly:

  * shape_lambda lands peakiness ON the IO target AND drives skill BELOW 1 (beats chance) —
    a genuine calibration cure;
  * weight_decay overshoots peakiness PAST the target to ≈1/91 (uniform) and its skill
    plateaus at ≈1.6 (the uniform decoder) — never beating chance. It trades over-sharpening
    for under-fitting: a dead decoder.

2×2: rows = {decoded peakiness, KL-skill}, cols = {shape_lambda, weight_decay}; spatial vs
temporal PCA decoders, mean±sem over 6 mice (hpsweep_v2).

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  cure_comparison.png
Usage:  python diagnostics/cure_comparison.py
"""

from __future__ import annotations

import argparse
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
from plot_overfit_vs_width import load_peak, _msem  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402
from performance_vs_hparams import _skills  # noqa: E402

SPEC = S.SPECS['v2']
COLS = [('shape_lambda', 'shape_lambda  (100·λ_Brier)'), ('weight_decay', 'weight decay')]
ARCH_C = {'spat': '#2166ac', 'temp': '#b2182b'}
ALAB = {'spat': 'spatial', 'temp': 'temporal'}


def main(results_root, out_root):
    ps.apply()
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), squeeze=False)
    io_peak = []
    for c, (axis, xlabel) in enumerate(COLS):
        cfg = SPEC['axes'][axis]
        xall, vals = S.xpos(cfg), cfg['vals']
        for arch in ('spat', 'temp'):
            pk_m, pk_s, sk_m, sk_s, xs = [], [], [], [], []
            for x, v in zip(xall, vals):
                run = f"{SPEC['parent']}/{S.cell_for(SPEC, axis, v)}"
                pk, tgt = load_peak(results_root, run, 'Q', 'PCA', 'half', 100, 'stratified_balanced', arch)
                m, s = _msem(pk)
                if m is None:
                    continue
                mat = Path(results_root) / run / S.LOSS_SLUG['PCA'] / 'stratified_balanced.mat'
                res = sio.loadmat(str(mat), simplify_cells=True).get('results', {})
                skm, sks = _msem(_skills(res, arch, 'pm'))
                xs.append(x); pk_m.append(m); pk_s.append(s or 0)
                sk_m.append(skm); sk_s.append(sks or 0)
                if tgt is not None:
                    io_peak.append(tgt)
            axes[0][c].errorbar(xs, pk_m, yerr=pk_s, color=ARCH_C[arch], lw=1.8, marker='o', ms=4, capsize=2)
            axes[1][c].errorbar(xs, sk_m, yerr=sk_s, color=ARCH_C[arch], lw=1.8, marker='o', ms=4, capsize=2)
        for r in (0, 1):
            S.apply_xaxis(axes[r][c], cfg)
            axes[r][c].set_xlabel(xlabel if r == 1 else '')
        axes[0][c].set_title(xlabel.split('  ')[0], fontsize=10)
    io = float(np.mean(io_peak)) if io_peak else 0.059
    for c in (0, 1):
        axes[0][c].axhline(io, color='k', ls=':', lw=1.3)          # IO target
        axes[1][c].axhline(1.0, color='k', ls=':', lw=1.3)         # chance
        axes[1][c].set_yscale('log')
    axes[0][0].set_ylabel('decoded peakiness\n(max-prob)')
    axes[1][0].set_ylabel('KL-skill  ÷ predict-mean\n(<1 beats chance)')
    handles = [Line2D([0], [0], color=ARCH_C['spat'], lw=2, marker='o', label='spatial'),
               Line2D([0], [0], color=ARCH_C['temp'], lw=2, marker='o', label='temporal'),
               Line2D([0], [0], color='k', lw=1.3, ls=':', label='IO target / chance')]
    axes[0][0].legend(handles=handles, fontsize=7, loc='best', frameon=True)
    ps.label_panels(axes.ravel())
    fig.suptitle('PCA over-sharpening: shape_lambda CURES it (skill beats chance), weight_decay only '
                 'LOBOTOMISES it (peakiness→uniform, skill stuck > chance) — hpsweep_v2, 6 mice', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'cure_comparison')
    print(f'IO target peakiness ≈ {io:.3f}; chance skill = 1.0')
    print(f'Done -> {Path(out_root).resolve()}/cure_comparison.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
