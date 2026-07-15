# -*- coding: utf-8 -*-
"""Actual decoding PERFORMANCE (chance-normalised skill) vs each swept hyperparameter.

Peakiness (bias) and the train–val gap (variance) don't say whether the decoder is any
GOOD. This scores the held-out decoded posteriors under a **calibrated** metric — forward
KL(decoded‖IO target), which (unlike the PCA training loss) is not fooled by over-sharpening
— normalised to a chance floor:

  skill = KL(decoded‖target) / KL(null‖target)      (< 1 beats chance, 1 = chance, > 1 worse)

  --null pm   : predict-mean = the marginal-mean target every trial = the OPTIMAL constant
                under forward KL (strictest null; `predict_mean_baseline.py`).  [default]
  --null shf  : the shuffle-trained control decoder (Dist['<arch>_shf']).

Same layout as `peakiness_vs_hparams.py` (rows = arch, cols = axis, one line per loss,
shared y, dotted chance line at 1). This is the metric that reveals, e.g., that weight_decay
which drove peakiness to uniform is actually WORSE than chance (a dead decoder), while
shape_lambda that lands peakiness on target also beats chance (a real cure).

Targets either sweep via `--sweep` (see `hpsweep_spec.py`).
Outputs (PNG+SVG) under figures/hpsweep_shuffle/:  performance_vs_hparams_<sweep>_<null>.png
Usage:  python diagnostics/performance_vs_hparams.py --sweep v2 --null pm
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
METRIC = 'KL'                                        # calibrated scorer (exposes over-sharpening)


def _skills(res, arch, null):
    """per-mouse skill = KL(decoded‖target) / KL(null‖target)."""
    out = []
    for mk in res:
        if not (isinstance(res[mk], dict) and isinstance(res[mk].get('Dist'), dict)):
            continue
        D = res[mk]['Dist']
        if arch not in D:
            continue
        pcs, evar = D.get('pcs'), D.get('explained_var')
        dec = np.asarray(D[arch]['decoded'], float)
        tgt = np.asarray(D[arch]['target'], float)
        kl = _eval_one(dec, tgt, METRIC, pcs, evar)
        if null == 'pm':
            ok = np.isfinite(tgt).all(1)
            if not ok.any():
                continue
            pm = np.tile(np.nanmean(tgt[ok], 0, keepdims=True), (tgt.shape[0], 1))
            kn = _eval_one(pm, tgt, METRIC, pcs, evar)
        else:                                        # shuffle-trained control
            shf = arch + '_shf'
            if shf not in D:
                continue
            kn = _eval_one(np.asarray(D[shf]['decoded'], float),
                           np.asarray(D[shf].get('target', tgt), float), METRIC, pcs, evar)
        if np.isfinite(kl) and np.isfinite(kn) and kn > 0:
            out.append(kl / kn)
    return out


def collect(results_root, spec, axes, null):
    data = {}
    for axis in axes:
        cfg = spec['axes'][axis]
        data[axis] = {loss: {'spat': [], 'temp': []} for loss in S.axis_losses(spec, axis)}
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
                    m, s = _msem(_skills(res, arch, null))
                    if m is not None:
                        data[axis][loss][arch].append((x, m, s))
    return data


def main(results_root, out_root, sweep, axes, null):
    ps.apply()
    spec = S.SPECS[sweep]
    data = collect(results_root, spec, axes, null)
    nlabel = 'predict-mean' if null == 'pm' else 'shuffle'
    fig, axgrid = plt.subplots(len(ARCHS), len(axes),
                               figsize=ps.figsize(len(axes), len(ARCHS)),
                               sharey=True, squeeze=False)
    print(f"KL-skill (÷{nlabel}; <1 beats chance) — sweep={sweep}")
    for r, (arch, alab) in enumerate(ARCHS):
        for c, axis in enumerate(axes):
            ax, cfg = axgrid[r][c], spec['axes'][axis]
            for loss, per_arch in data[axis].items():
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
                ax.set_ylabel(f'{alab}\nKL-skill (÷{nlabel})')
            if r == 0:
                ax.set_title(axis + (' (PCA)' if cfg['losses'] else ''), fontsize=9)
    axgrid[0][0].plot([], [], color='0.4', lw=1.1, ls=':', label=f'chance ({nlabel})')
    h, l = axgrid[0][0].get_legend_handles_labels()
    axgrid[0][0].legend(h, l, fontsize=6.5, loc='best', frameon=True)
    ps.label_panels(axgrid.ravel())
    fig.suptitle(f'[{sweep}] Decoding performance: KL-skill (÷{nlabel}; <1 beats chance, log y) vs each '
                 f'hyperparameter. The metric that says which decoder is actually GOOD', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'performance_vs_hparams_{sweep}_{null}')
    print(f'Done -> {Path(out_root).resolve()}/performance_vs_hparams_{sweep}_{null}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hpsweep_shuffle')
    ap.add_argument('--sweep', default='v2', choices=list(S.SPECS))
    ap.add_argument('--null', default='pm', choices=['pm', 'shf'])
    ap.add_argument('--axes', nargs='+', default=None)
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.sweep, a.axes or S.DEFAULT_AXES[a.sweep], a.null)
