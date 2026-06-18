# -*- coding: utf-8 -*-
"""Dropout vs early stopping as the regulariser (2026-06-10 meeting).

Early stopping already exists (`Config.patience`); dropout was added
(`Config.dropout` → MLP). Compared on the matched grid (Q/half/100ms, 6 mice):

  none        — no regulariser   (`noreg`,              patience 0, dropout 0, 200 ep)
  early-stop  — early stopping    (`loss_comparison_v1`, patience 15, dropout 0)
  drop <p>    — dropout only      (`dropreg_drop<p>`,    patience 0, dropout p, 200 ep)

`noreg` vs `dropreg` isolates dropout cleanly (same 100% data / 200 epochs, only
dropout differs); `early-stop` is the reference regulariser.

Metrics — both **regime-independent** (this matters): decoded **peakiness** (max-prob,
vs the IO-target reference) and **raw KL(decoded, IO target)** (lower = closer to the
broad target = better calibrated). NB the shuffle-normalised KL-*skill* is NOT used
here: the shuffle is retrained under each condition's regime (200 ep vs early-stopped),
so its scale shifts across conditions and skill becomes non-comparable — raw KL avoids
that. The train–val *gap* (Máté's literal overfitting ask) is unavailable: the codebase
carves a val split only when patience>0 (val ⟺ early-stopping), so patience-0 runs have
no val curve — needs a `monitor_val` knob to decouple. See report.

Outputs (PNG+SVG) under figures/peakiness_scatter/:  dropout_vs_earlystop.png

Usage:  python diagnostics/dropout_vs_earlystop.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from plot_overfit_vs_width import load_peak, _slug, _msem  # noqa: E402
from cross_loss_eval import _eval_one  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]


def _drop_val(name):
    m = re.search(r'drop([0-9p]+)', name)
    return float(m.group(1).replace('p', '.')) if m else None


def discover_conditions(results_root):
    root = Path(results_root)
    conds = []
    if (root / 'noreg').is_dir():
        conds.append(('noreg', 'none'))
    if (root / 'loss_comparison_v1').is_dir():
        conds.append(('loss_comparison_v1', 'early-stop'))
    drops = [(d.name, _drop_val(d.name)) for d in root.glob('dropreg_drop*') if d.is_dir()]
    for name, p in sorted(drops, key=lambda t: (t[1] if t[1] is not None else 0)):
        conds.append((name, f'drop {p:g}'))
    return conds


def load_rawloss(results_root, run, target, loss, window, bin_ms, split, arch, metric='KL'):
    """Per-mouse mean raw loss(decoded, IO target) under `metric` — regime-independent.
    metric='KL' = calibration; metric='PCA' = the location-weighted PCA yardstick, which is
    blind to posterior width and so cannot see the over-sharpening — the point of showing it
    next to KL (same decoders, two metrics)."""
    f = Path(results_root) / run / _slug(target, loss, window, bin_ms) / f'{split}.mat'
    if not f.is_file():
        return []
    res = sio.loadmat(str(f), simplify_cells=True).get('results')
    if not isinstance(res, dict):
        return []
    out = []
    for mk in sorted(res):
        D = res[mk].get('Dist') if isinstance(res[mk], dict) else None
        if not (isinstance(D, dict) and arch in D):
            continue
        v = _eval_one(np.asarray(D[arch]['decoded'], float),
                      np.asarray(D[arch]['target'], float), metric,
                      D.get('pcs'), D.get('explained_var'))
        if np.isfinite(v):
            out.append(v)
    return out


def main(results_root, split, out_root, target='Q', window='half', bin_ms=100):
    ps.apply()
    conds = discover_conditions(results_root)
    if not any(lab == 'none' or lab.startswith('drop') for _, lab in conds):
        raise SystemExit('dropout comparison needs noreg / dropreg_drop* runs — rsync them down first.')
    x = np.arange(len(conds))
    labels = [lab for _, lab in conds]

    M = {a: {l: {'peak': [], 'kl': [], 'pca': []} for l in LOSSES} for a, _ in ARCHS}
    io_peak = []
    for run, _lab in conds:
        for arch, _ in ARCHS:
            for loss in LOSSES:
                pk, tgt = load_peak(results_root, run, target, loss, window, bin_ms, split, arch)
                kl = load_rawloss(results_root, run, target, loss, window, bin_ms, split, arch, 'KL')
                pca = load_rawloss(results_root, run, target, loss, window, bin_ms, split, arch, 'PCA')
                M[arch][loss]['peak'].append(_msem(pk))
                M[arch][loss]['kl'].append(_msem(kl))
                M[arch][loss]['pca'].append(_msem(pca))
                if tgt is not None:
                    io_peak.append(tgt)
    io_peak = float(np.mean(io_peak)) if io_peak else np.nan

    METRICS = [('peak', 'decoded peakiness (max-prob)'),
               ('kl', 'KL(decoded ‖ IO target) — calibration'),
               ('pca', 'Projection-based loss — location-only (width-blind)')]
    fig, axes = plt.subplots(2, 3, figsize=ps.figsize(3, 2), sharex=True)
    for r, (arch, alabel) in enumerate(ARCHS):
        for c, (key, mlabel) in enumerate(METRICS):
            ax = axes[r, c]
            for loss in LOSSES:
                ys = [m if m is not None else np.nan for m, _ in M[arch][loss][key]]
                es = [s if s is not None else np.nan for _, s in M[arch][loss][key]]
                ax.errorbar(x, ys, yerr=es, color=LCOL[loss], lw=2, marker='o', ms=4,
                            capsize=2, label=ps.loss_label(loss))
            if key == 'peak' and np.isfinite(io_peak):
                ax.axhline(io_peak, color='k', ls=':', lw=1.4, label='IO target')
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7.5)
            ax.set_ylabel(f'{alabel}\n{mlabel}' if c == 0 else mlabel, fontsize=8)
            if r == 0:
                ax.set_title(mlabel, fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='best')
    ps.label_panels(axes.ravel())
    fig.suptitle('Dropout vs early stopping — over-sharpening & calibration, per loss '
                 '(Q/half/100ms, 6 mice; train–val gap needs monitor_val)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'dropout_vs_earlystop')

    print(f'conditions: {labels}   (IO-target peakiness ≈ {io_peak:.3f})')
    for arch, alabel in ARCHS:
        print(f'\n[{alabel}]')
        for key, mlabel in METRICS:
            print(f'  {mlabel}:')
            for loss in LOSSES:
                vals = [m for m, _ in M[arch][loss][key]]
                print(f'    {loss:12s} ' + '  '.join(
                    f'{v:.3f}' if (v is not None and np.isfinite(v)) else '  nan' for v in vals))
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.split, a.out_root)
