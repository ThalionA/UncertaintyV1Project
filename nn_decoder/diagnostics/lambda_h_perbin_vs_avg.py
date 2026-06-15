# -*- coding: utf-8 -*-
"""λ_H sweep, the RIGHT question (2026-06-10 meeting, corrected framing): can the
temporal/SBC decoder produce **peaky individual time-bins** (committed "samples")
while its **time-averaged** posterior stays **broad and calibrated** (≈ the IO
target) — even under the information-theoretic losses (CE/KL) that promote a broad
overall distribution?

The entropy penalty λ_H acts on the per-bin (instantaneous) distributions, so it
*should* sharpen the bins; the fit-loss (CE/KL) constrains the time-AVERAGE to be
broad. The SBC ideal is the two coexisting: low per-bin entropy (peaky samples) +
high time-averaged entropy ≈ IO target (broad posterior). That requires the peaky
bins to sit at VARIED locations across time, so they average back to a broad shape.

Each `.mat` saves `Dist['temp']['decoded_samp']` (n, 91, 10) = the per-bin
distributions (10 bins), and `['decoded']` (n, 91) = their time-average (verified:
`decoded_samp.mean(bins) == decoded`). We measure, per λ_H × loss (6 mice):
  - per-bin entropy  H(per-bin dist), mean over bins & trials   ("are samples peaky?")
  - time-avg entropy H(decoded)                                  ("is output calibrated?")
  - IO-target entropy H(target)                                  (the broad reference)
  - sampling spread  = time-avg − per-bin                        ("are bins distinct samples?")

Reads the `lambdaH_sweep_entlam<λ>` runs. Outputs (PNG+SVG) under
figures/peakiness_scatter/:  lambda_h_perbin_vs_avg.png

Usage:  python diagnostics/lambda_h_perbin_vs_avg.py
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
from cross_loss_eval import _agg  # noqa: E402
from diagnostics.lambda_h_temporal_sweep import _lambda_of, _find_cell  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}


def _entropy(p, axis):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis)


def collect(results_root, prefix, split):
    """{lambda: {loss: {'perbin':[per mouse], 'avg':[...], 'tgt':[...]}}}."""
    runs = sorted(Path(results_root).glob(f'{prefix}_entlam*'),
                  key=lambda p: (_lambda_of(p.name) if _lambda_of(p.name) is not None else 1e9))
    data = {}
    for rd in runs:
        lam = _lambda_of(rd.name)
        if lam is None:
            continue
        for loss in LOSSES:
            f = _find_cell(rd, loss, split)
            if f is None:
                continue
            res = sio.loadmat(str(f), simplify_cells=True).get('results')
            if not isinstance(res, dict):
                continue
            pb, av, tg = [], [], []
            for mk in sorted(res):
                D = res[mk]['Dist']['temp']
                ds = np.asarray(D['decoded_samp'], float)   # (n, 91, 10) cats axis=1, bins axis=2
                dec = np.asarray(D['decoded'], float)        # (n, 91)
                t = np.asarray(D['target'], float)
                pb.append(float(np.nanmean(_entropy(ds, 1))))   # over cats, mean bins+trials
                av.append(float(np.nanmean(_entropy(dec, 1))))
                tg.append(float(np.nanmean(_entropy(t, 1))))
            data.setdefault(lam, {})[loss] = {'perbin': pb, 'avg': av, 'tgt': tg}
    return data


def main(results_root, prefix, split, out_root):
    ps.apply()
    data = collect(results_root, prefix, split)
    if not data:
        raise SystemExit(f'no {prefix}_entlam* runs found under {results_root}.')
    lams = sorted(data)
    x = np.arange(len(lams))
    xlabels = [f'{l:g}' for l in lams]
    tgt_H = np.nanmean([_agg(data[l][L]['tgt'])[0] for l in lams for L in data[l]])

    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1), sharex=True)

    def _line(ax, field):
        for loss in LOSSES:
            ys, es = [], []
            for lam in lams:
                m, s = _agg(data[lam].get(loss, {}).get(field, []))
                ys.append(m); es.append(s)
            ax.errorbar(x, ys, yerr=es, color=LCOL[loss], lw=2, marker='o', ms=4,
                        capsize=2, label=loss)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right')
        ax.set_xlabel(r'entropy penalty $\lambda_H$')

    # (a) per-bin (instantaneous) entropy — "are the samples peaky?"
    _line(axes[0], 'perbin')
    axes[0].axhline(tgt_H, color='k', ls=':', lw=1.5, label='IO target')
    axes[0].set_ylabel('per-bin entropy  H  (nats)')
    axes[0].set_title('Are individual bins peaky?\n(low = peaky samples)', fontsize=8.5)
    axes[0].legend(fontsize=7, loc='lower left')

    # (b) time-averaged posterior entropy — "is the output calibrated?"
    _line(axes[1], 'avg')
    axes[1].axhline(tgt_H, color='k', ls=':', lw=1.5)
    axes[1].set_ylabel('time-averaged entropy  H  (nats)')
    axes[1].set_title('Is the average calibrated?\n(≈ IO target = broad/correct)', fontsize=8.5)

    # (c) sampling spread = time-avg − per-bin — "are the bins distinct samples?"
    for loss in LOSSES:
        ys = []
        for lam in lams:
            a = _agg(data[lam].get(loss, {}).get('avg', []))[0]
            b = _agg(data[lam].get(loss, {}).get('perbin', []))[0]
            ys.append(a - b)
        axes[2].plot(x, ys, color=LCOL[loss], lw=2, marker='o', ms=4, label=loss)
    axes[2].set_xticks(x); axes[2].set_xticklabels(xlabels, rotation=20, ha='right')
    axes[2].set_xlabel(r'entropy penalty $\lambda_H$')
    axes[2].set_ylabel('sampling spread  H(avg) − H(bin)')
    axes[2].set_title('Are the bins distinct samples?\n(small/flat = bins ≈ the average)', fontsize=8.5)

    ps.label_panels(axes)
    fig.suptitle('Peaky per-bin samples vs broad calibrated average across λ_H '
                 '(temporal, 6 mice, Q/half/100ms)', y=1.03)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'lambda_h_perbin_vs_avg')

    print(f'IO-target entropy ≈ {tgt_H:.2f} nats (max ln(91)={np.log(91):.2f})')
    print(f"{'loss':12s} {'λ_H':>6s} | {'per-bin H':>9s} {'avg H':>7s} {'spread':>7s}")
    for loss in LOSSES:
        for lam in lams:
            pb = _agg(data[lam].get(loss, {}).get('perbin', []))[0]
            av = _agg(data[lam].get(loss, {}).get('avg', []))[0]
            print(f'  {loss:10s} {lam:6.3f} | {pb:9.2f} {av:7.2f} {av-pb:7.2f}')
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', default='lambdaH_sweep')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.prefix, a.split, a.out_root)
