# -*- coding: utf-8 -*-
"""Why is the linear + flat/MSE + raw SPATIAL decoder "worse than chance"?
It is not — on the typical trial it is ~2x BETTER. The mean is destroyed by a
small tail of catastrophic trials (2026-08-04, Theo's challenge).

MSE squares errors, so a handful of confidently-wrong held-out predictions can
dominate the average while the median trial is fine. The linear raw decoder is the
most exposed configuration in the run: 5,915 input weights fit to ~376 training
trials (15.7 params/trial), with no hidden layer to smooth the output and — for the
SPATIAL arch — no Jensen average over time bins to damp a misplaced peak.

  a  per-trial MSE / predict-mean, distribution (log x). Dotted = chance; the
     median markers show the typical trial is well below it.
  b  mean vs median of that ratio — the gap IS the tail.
  c  cumulative share of total squared error by trial rank: for linear raw spatial
     the worst 1% of trials carry ~28% of all the error.

Read together: "worse than chance" here is a statement about the MEAN under a
squared-error metric with an over-parameterised model, not about typical
performance. Both the hidden layer (520 params) and input PCA (273 params at k=3)
remove the tail; the temporal Jensen average halves it.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_tail_diagnosis.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

CELLS = [
    ('linear raw, spatial',  'lin_raw_l0_d0_w0',  'spat', ps.PCA_EVAR),
    ('linear raw, temporal', 'lin_raw_l0_d0_w0',  'temp', ps.SPATIAL),
    ('8 hidden raw, spatial', 'h8_raw_l0_d0_w0',  'spat', ps.FLAT_EVAR),
    ('linear 3 PCs, spatial', 'lin_pc3_l0_d0_w0', 'spat', ps.KL),
]


def ratios(results_root, cell, arch):
    """Per-trial MSE / (that mouse's mean predict-mean MSE), pooled over mice."""
    p = glob.glob(f'{results_root}/projflat_v1/{cell}/*/stratified_balanced.mat')[0]
    r = sio.loadmat(p, simplify_cells=True)['results']
    out = []
    for m in sorted(r):
        if not (isinstance(r[m], dict) and isinstance(r[m].get('Dist'), dict)):
            continue
        D = r[m]['Dist']
        dec = np.asarray(D[arch]['decoded'], float)
        tgt = np.asarray(D[arch]['target'], float)
        ok = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
        dec, tgt = dec[ok], tgt[ok]
        n = len(tgt); tot = tgt.sum(0)
        pm = (tot[None, :] - tgt) / (n - 1)
        out.append(((dec - tgt) ** 2).mean(1) / ((pm - tgt) ** 2).mean(1).mean())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()

    data = {lab: ratios(a.results_root, c, ar) for lab, c, ar, _ in CELLS}
    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1))

    # (a) distributions
    ax = axes[0]
    bins = np.logspace(-2.2, 1.6, 60)
    for (lab, _c, _ar, colr) in CELLS:
        v = data[lab]
        ax.hist(v, bins=bins, histtype='step', lw=1.5, color=colr, density=True,
                label=lab)
        ax.plot(np.median(v), 0, 'v', ms=6, color=colr, clip_on=False, zorder=5)
    ax.axvline(1.0, color='k', ls=':', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('per-trial MSE ÷ predict-mean', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.set_title('(a) most trials are well below chance\n(▼ = median)', fontsize=8.5)
    ax.legend(fontsize=5.8, frameon=True)

    # (b) mean vs median
    ax = axes[1]
    x = np.arange(len(CELLS))
    w = 0.38
    for xi, (lab, _c, _ar, colr) in enumerate(CELLS):
        v = data[lab]
        ax.bar(xi - w / 2, np.median(v), w, color=colr, edgecolor='k', lw=0.5,
               label='median' if xi == 0 else None)
        ax.bar(xi + w / 2, v.mean(), w, color=colr, edgecolor='k', lw=0.5,
               hatch='///', alpha=0.75, label='mean' if xi == 0 else None)
    ax.axhline(1.0, color='k', ls=':', lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0].replace(', ', '\n') for c in CELLS], fontsize=6.5)
    ax.set_ylabel('MSE ÷ predict-mean', fontsize=8)
    ax.set_title('(b) the mean sits above chance,\nthe median far below — the gap is the tail',
                 fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=True)

    # (c) cumulative share of total error
    ax = axes[2]
    for (lab, _c, _ar, colr) in CELLS:
        v = np.sort(data[lab])[::-1]
        frac_trials = np.arange(1, v.size + 1) / v.size * 100
        ax.plot(frac_trials, np.cumsum(v) / v.sum() * 100, color=colr, lw=1.6,
                label=lab)
    ax.axvline(1.0, color='0.4', ls='--', lw=1.2)
    ax.text(1.15, 8, 'worst 1%\nof trials', fontsize=6, color='0.35')
    ax.set_xscale('log')
    ax.set_xlabel('% of trials (worst first, log)', fontsize=8)
    ax.set_ylabel('% of total squared error', fontsize=8)
    ax.set_title('(c) a 1% tail carries ~28% of all the error\n(linear raw spatial)',
                 fontsize=8.5)
    ax.legend(fontsize=5.8, frameon=True, loc='lower right')

    fig.suptitle('“Worse than chance” for linear + flat/MSE + raw SPATIAL is a MEAN artefact of a squared-error '
                 'metric on an over-parameterised model\n(5,915 weights / 376 training trials). The median trial '
                 'is ~2x better than chance. A hidden layer, input PCA, or the temporal average all remove the tail.',
                 y=1.04, fontsize=8)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'projflat_fig9_tail_diagnosis')

    print(f"{'config':24s}{'median':>9s}{'mean':>8s}{'p99':>8s}{'top1%share':>12s}")
    for lab, _c, _ar, _col in CELLS:
        v = data[lab]
        srt = np.sort(v)[::-1]
        share = srt[:max(1, int(.01 * v.size))].sum() / v.sum() * 100
        print(f'{lab:24s}{np.median(v):9.2f}{v.mean():8.2f}'
              f'{np.percentile(v, 99):8.2f}{share:11.0f}%')


if __name__ == '__main__':
    main()
