# -*- coding: utf-8 -*-
"""Decoded peakiness vs each swept hyperparameter — the over-sharpening the loss work is
actually about (the fit-loss train–val gap is BLIND to it; see PCA-Peakiness-Mechanism §8).

Peakiness = across-trial mean decoded max-probability (IO target ≈ 0.059; higher =
over-confident). One line per loss, rows = arch, cols = axis, SHARED y so magnitudes compare.

Targets either sweep via `--sweep` (see `hpsweep_spec.py`):
  v1 (`hpsweep_wide`, H=32 base) — generic knobs fail to fix PCA over-sharpening; only
      early-stopping caps it (and only to ~4.5x target); KL/JS sit on target throughout.
  v2 (`hpsweep_v2`, H=8 base) — adds the **weight_decay** axis and the PCA-only
      **shape_lambda** axis (the loss-side cure: PCA + (shape/100)·Brier), so this run
      should show the peakiness actually LANDING on target as shape_lambda rises.

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  peakiness_vs_hparams_<sweep>.png
Usage:  python diagnostics/peakiness_vs_hparams.py --sweep v2
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402

LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]


def main(results_root, out_root, split, sweep, axes):
    ps.apply()
    spec = S.SPECS[sweep]
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    io_all = []
    print(f"peakiness (decoded max-prob) — sweep={sweep} parent={spec['parent']}")
    for r, (arch, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], spec['axes'][axis]
            xall = S.xpos(cfg)
            for loss in S.axis_losses(spec, axis):          # shape_lambda -> PCA only
                xs, ys, es = [], [], []
                for x, v in zip(xall, cfg['vals']):
                    run = f"{spec['parent']}/{S.cell_for(spec, axis, v)}"
                    pk, tgt = load_peak(results_root, run, 'Q', loss, 'half', 100, split, arch)
                    m, s = _msem(pk)
                    if m is not None:
                        xs.append(x); ys.append(m); es.append(s if s is not None else 0.0)
                        if tgt is not None:
                            io_all.append(tgt)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=LCOL[loss], lw=1.6, marker='o', ms=4,
                                capsize=2, label=ps.loss_label(loss))
            S.apply_xaxis(ax, cfg)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\ndecoded max-prob')
            if r == 0:
                ax.set_title(axis + (' (PCA)' if cfg['losses'] else ''), fontsize=9)
    io = float(np.mean(io_all)) if io_all else np.nan
    for ax in axgrid.ravel():
        if np.isfinite(io):
            ax.axhline(io, color='k', ls=':', lw=1.3)
    axgrid[0][0].plot([], [], 'k:', lw=1.3, label='IO target')
    h, l = axgrid[0][0].get_legend_handles_labels()
    axgrid[0][0].legend(h, l, fontsize=6.5, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'[{sweep}] Decoded peakiness vs each hyperparameter (shared y; IO target ≈ '
                 f'{io:.3f}, dotted). Which knob tames the over-sharpening?', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'peakiness_vs_hparams_{sweep}')
    print(f'IO-target peakiness ≈ {io:.3f}')
    print(f'Done -> {Path(out_root).resolve()}/peakiness_vs_hparams_{sweep}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    ap.add_argument('--axes', nargs='+', default=None)
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.split, a.sweep, a.axes or S.DEFAULT_AXES[a.sweep])
