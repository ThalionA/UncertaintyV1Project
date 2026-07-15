# -*- coding: utf-8 -*-
"""Actual decoding performance — **normalised loss** vs each swept hyperparameter.

Peakiness (bias) and the train–val gap (variance) don't say whether the decoder is any GOOD.
This scores the held-out decoded posteriors under a common metric and normalises to a chance
floor:

  normalised loss = loss(decoded‖target) / loss(null‖target)   (< 1 beats chance, 1 = chance)

Both metrics are always shown (per the standing request), because they disagree — that IS the
point (PCA-Peakiness-Mechanism §8):
  * **KL** — calibrated; exposes over-sharpening (an over-confident decoder scores badly).
  * **Projection-based** (the PCA training loss) — width-blind; scores over-confident decoders
    the same as calibrated ones, so it stays ~flat across conditions.

And both chance nulls:
  * **predict-mean** — the marginal-mean target every trial (the optimal constant; strictest).
  * **shuffle** — the shuffle-trained control decoder (Dist['<arch>_shf']).

One figure per (metric × null); same layout as `peakiness_vs_hparams.py` (rows = arch, cols =
axis, one line per loss, shared log y, dotted chance line at 1). One `.mat` pass fills all four.

Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  performance_vs_hparams_<sweep>_<metric>_<null>.png
Usage:  python diagnostics/performance_vs_hparams.py --sweep v2
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one  # noqa: E402  (same loss maths as training)
from plot_overfit_vs_width import _msem  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402

LCOL = {'PCA': ps.PCA_EVAR, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
METRICS = ['KL', 'PCA']            # calibrated + projection-based (always both)
NULLS = ['pm', 'shf']              # predict-mean + shuffle (always both)
NULL_LABEL = {'pm': 'predict-mean', 'shf': 'shuffle'}


def _norm_by_mouse(res, arch):
    """{(metric, null): [per-mouse normalised loss]} for one cell/arch — one .mat pass."""
    out = {(m, n): [] for m in METRICS for n in NULLS}
    for mk in res:
        if not (isinstance(res[mk], dict) and isinstance(res[mk].get('Dist'), dict)):
            continue
        D = res[mk]['Dist']
        if arch not in D:
            continue
        pcs, evar = D.get('pcs'), D.get('explained_var')
        dec = np.asarray(D[arch]['decoded'], float)
        tgt = np.asarray(D[arch]['target'], float)
        ok = np.isfinite(tgt).all(1)
        if not ok.any():
            continue
        pm = np.tile(np.nanmean(tgt[ok], 0, keepdims=True), (tgt.shape[0], 1))
        shf = arch + '_shf'
        sdec = np.asarray(D[shf]['decoded'], float) if shf in D else None
        stgt = np.asarray(D[shf].get('target', tgt), float) if shf in D else None
        for metric in METRICS:
            num = _eval_one(dec, tgt, metric, pcs, evar)
            den = {'pm': _eval_one(pm, tgt, metric, pcs, evar),
                   'shf': (_eval_one(sdec, stgt, metric, pcs, evar) if sdec is not None else np.nan)}
            for n in NULLS:
                if np.isfinite(num) and np.isfinite(den[n]) and den[n] > 0:
                    out[(metric, n)].append(num / den[n])
    return out


def collect(results_root, spec, axes):
    """data[(metric,null)][axis][loss][arch] = [(xpos, mean, sem)]."""
    data = {(m, n): {ax: {l: {'spat': [], 'temp': []} for l in S.axis_losses(spec, ax)}
                     for ax in axes} for m in METRICS for n in NULLS}
    for axis in axes:
        cfg = spec['axes'][axis]
        for loss in S.axis_losses(spec, axis):
            for x, v in zip(S.xpos(cfg), cfg['vals']):
                mat = (Path(results_root) / spec['parent'] / S.cell_for(spec, axis, v)
                       / S.LOSS_SLUG[loss] / 'stratified_balanced.mat')
                if not mat.is_file():
                    continue
                res = sio.loadmat(str(mat), simplify_cells=True).get('results')
                if not isinstance(res, dict):
                    continue
                for arch, _ in ARCHS:
                    per = _norm_by_mouse(res, arch)
                    for key in per:
                        m, s = _msem(per[key])
                        if m is not None:
                            data[key][axis][loss][arch].append((x, m, s))
    return data


def _plot(spec, axes, data, metric, null, sweep, out_root):
    ps.apply()
    mlab, nlab = ps.loss_label(metric), NULL_LABEL[null]
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    for r, (arch, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], spec['axes'][axis]
            for loss, per_arch in data[(metric, null)][axis].items():
                pts = per_arch[arch]
                if pts:
                    xs, ys, es = zip(*pts)
                    ax.errorbar(xs, ys, yerr=es, color=LCOL[loss], lw=1.6, marker='o', ms=4,
                                capsize=2, label=ps.loss_label(loss))
            ax.set_yscale('log')
            ax.axhline(1.0, color='0.4', lw=1.1, ls=':')
            S.apply_xaxis(ax, cfg)
            if r == len(ARCHS) - 1:
                ax.set_xlabel(cfg['xlabel'])
            if c == 0:
                ax.set_ylabel(f'{alab}\n{mlab} loss ÷ {nlab}')
            if r == 0:
                ax.set_title(axis + (' (PCA)' if cfg['losses'] else ''), fontsize=9)
    axgrid[0][0].plot([], [], color='0.4', lw=1.1, ls=':', label=f'chance ({nlab})')
    h, l = axgrid[0][0].get_legend_handles_labels()
    axgrid[0][0].legend(h, l, fontsize=6.5, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'[{sweep}] {mlab} normalised loss (÷ {nlab}; <1 beats chance, log y) vs each '
                 f'hyperparameter. {"Calibrated: exposes over-sharpening" if metric=="KL" else "Projection-based: width-blind (~flat = the point)"}',
                 y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'performance_vs_hparams_{sweep}_{metric}_{null}')
    print(f'  -> performance_vs_hparams_{sweep}_{metric}_{null}.png')


def main(results_root, out_root, sweep, axes):
    spec = S.SPECS[sweep]
    data = collect(results_root, spec, axes)
    print(f"normalised loss — sweep={sweep} (metrics {METRICS} x nulls {NULLS})")
    for metric in METRICS:
        for null in NULLS:
            _plot(spec, axes, data, metric, null, sweep, out_root)
    print(f'Done -> {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    ap.add_argument('--axes', nargs='+', default=None)
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.sweep, a.axes or S.DEFAULT_AXES[a.sweep])
