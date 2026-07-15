# -*- coding: utf-8 -*-
"""The v2 headline: two knobs that both drive PCA peakiness down, but only one makes a
GOOD decoder — and the projection loss can't tell.

For PCA over-sharpening, `shape_lambda` (loss-side fix, PCA + (λ=shape/100)·Brier) and strong
`weight_decay` BOTH collapse the decoded peakiness, so on peakiness alone they look equivalent.
The chance-normalised loss separates them — but only under the CALIBRATED (KL) metric:
  * shape_lambda lands peakiness ON the IO target AND drives KL normalised loss BELOW 1
    (beats chance) — a genuine cure;
  * weight_decay overshoots peakiness PAST target to ≈1/91 (uniform) and its KL normalised
    loss plateaus ≈1.6 (the uniform decoder) — never beating chance. A dead decoder.
  * the **projection-based** normalised loss (the PCA training metric) is width-blind: it stays
    ~flat and beats chance for both, so it CANNOT see the difference — the reason KL is needed.

Rows = {decoded peakiness, KL normalised loss, projection-based normalised loss};
cols = {shape_lambda, weight_decay}; spatial vs temporal PCA, mean±sem over 6 mice (hpsweep_v2).
`--null {pm,shf}` chooses the chance floor (predict-mean default; shuffle also produced).

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  cure_comparison_<null>.png
Usage:  python diagnostics/cure_comparison.py --null pm
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
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

SPEC = S.SPECS['v2']
COLS = [('shape_lambda', 'shape_lambda  (100·λ_Brier)'), ('weight_decay', 'weight decay')]
ARCH_C = {'spat': '#2166ac', 'temp': '#b2182b'}
NULL_LABEL = {'pm': 'predict-mean', 'shf': 'shuffle'}


def main(results_root, out_root, null):
    ps.apply()
    nlab = NULL_LABEL[null]
    # rows: peakiness, KL normalised loss, projection normalised loss
    fig, axes = plt.subplots(3, 2, figsize=ps.figsize(2, 3), squeeze=False)
    io_peak = []
    for c, (axis, xlabel) in enumerate(COLS):
        cfg = SPEC['axes'][axis]
        for arch in ('spat', 'temp'):
            xs, pk, kln, pjn = [], [], [], []
            for x, v in zip(S.xpos(cfg), cfg['vals']):
                run = f"{SPEC['parent']}/{S.cell_for(SPEC, axis, v)}"
                p, tgt = load_peak(results_root, run, 'Q', 'PCA', 'half', 100, 'stratified_balanced', arch)
                m, _ = _msem(p)
                if m is None:
                    continue
                mat = Path(results_root) / run / S.LOSS_SLUG['PCA'] / 'stratified_balanced.mat'
                res = sio.loadmat(str(mat), simplify_cells=True).get('results', {})
                nb = _norm_by_mouse(res, arch)
                kl, _ = _msem(nb[('KL', null)]); pj, _ = _msem(nb[('PCA', null)])
                xs.append(x); pk.append(m); kln.append(kl); pjn.append(pj)
                if tgt is not None:
                    io_peak.append(tgt)
            axes[0][c].plot(xs, pk, color=ARCH_C[arch], lw=1.8, marker='o', ms=4)
            axes[1][c].plot(xs, kln, color=ARCH_C[arch], lw=1.8, marker='o', ms=4)
            axes[2][c].plot(xs, pjn, color=ARCH_C[arch], lw=1.8, marker='o', ms=4)
        for r in range(3):
            S.apply_xaxis(axes[r][c], cfg)
            axes[r][c].set_xlabel(xlabel if r == 2 else '')
        axes[0][c].set_title(xlabel.split('  ')[0], fontsize=10)
    io = float(np.mean(io_peak)) if io_peak else 0.059
    for c in (0, 1):
        axes[0][c].axhline(io, color='k', ls=':', lw=1.3)
        for r in (1, 2):
            axes[r][c].axhline(1.0, color='k', ls=':', lw=1.3); axes[r][c].set_yscale('log')
    axes[0][0].set_ylabel('decoded peakiness\n(max-prob)')
    axes[1][0].set_ylabel(f'KL loss ÷ {nlab}\n(<1 beats chance)')
    axes[2][0].set_ylabel(f'Projection loss ÷ {nlab}\n(blind → ~flat)')
    handles = [Line2D([0], [0], color=ARCH_C['spat'], lw=2, marker='o', label='spatial'),
               Line2D([0], [0], color=ARCH_C['temp'], lw=2, marker='o', label='temporal'),
               Line2D([0], [0], color='k', lw=1.3, ls=':', label='IO target / chance')]
    axes[0][0].legend(handles=handles, fontsize=7, loc='best', frameon=True)
    ps.label_panels(axes.ravel())
    fig.suptitle(f'PCA over-sharpening (÷ {nlab}): shape_lambda CURES it (KL loss beats chance), weight_decay '
                 f'only LOBOTOMISES it (→uniform). Projection loss is BLIND to both — hpsweep_v2, 6 mice',
                 y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'cure_comparison_{null}')
    print(f'IO target ≈ {io:.3f}; chance = 1.0 (÷{nlab})')
    print(f'Done -> {Path(out_root).resolve()}/cure_comparison_{null}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--null', default='pm', choices=['pm', 'shf'])
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.null)
