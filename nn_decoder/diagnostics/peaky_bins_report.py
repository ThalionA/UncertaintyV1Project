# -*- coding: utf-8 -*-
"""Did we get PEAKY BINS with a CALIBRATED AVERAGE? Report for the shapefix_v1 probe.

The target state (a sampling code): each individual time bin is a sharp commitment while the
average over bins recovers the broad ideal-observer posterior. So the three quantities must move
independently, and all three are judged together:

  1. PER-BIN peakiness      Dist['temp']['decoded_samp'] (n_trials, 91, T) -> should RISE above target
  2. TRIAL-AVERAGE peakiness Dist['temp']['decoded']     -> should STAY at the IO target
  3. normalised loss under BOTH the projection metric and KL -> must stay below chance

A cell that sharpens the bins by wrecking the average is not a success, which is why the ratio
per-bin / average is reported explicitly: it is the "how much sampling is there" number, and it is
1.0 when the decoder emits the same posterior in every bin.

Also reports the Jensen gap mean_t KL(p_t || p_bar) — an independent read on bin disagreement that
does not depend on peakiness — and the spatial arch for reference (it has no bins and no lambda_H).

Outputs (PNG+SVG) under figures/hparam_summary/: peaky_bins_report
Usage:  python diagnostics/peaky_bins_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps
from decoder_metrics import kl_rows  # noqa: E402  (canonical KL — audit D1)  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from story_figures import _results, peaky, tgt_peak, normloss  # noqa: E402

RUN = 'shapefix_v1'
CELLS = [('lamH0_drop0', 0.0, ''), ('lamH0p003_drop0', 3e-3, ''), ('lamH0p01_drop0', 1e-2, ''),
         ('lamH0p03_drop0', 3e-2, ''), ('lamH0p1_drop0', 1e-1, ''),
         ('lamH0p03_drop0p5', 3e-2, ' +drop0.5')]


def per_bin_stats(res):
    """per-mouse: (per-bin peakiness, trial-average peakiness, Jensen gap)."""
    pb, av, jg = [], [], []
    for mk in sorted(res):
        if not str(mk).startswith('mouse_'):
            continue
        D = res[mk]['Dist']
        samp = np.asarray(D['temp']['decoded_samp'], float)          # (n, 91, T)
        avg = np.asarray(D['temp']['decoded'], float)
        if samp.ndim != 3 or samp.size == 0:
            continue
        pb.append(samp.max(1).mean())
        av.append(avg.max(1).mean())
        pbar = samp.mean(-1)
        jg.append(np.mean([kl_rows(np.tile(pbar[i], (samp.shape[-1], 1)), samp[i].T).mean()
                           for i in range(samp.shape[0])]))
    return np.array(pb), np.array(av), np.array(jg)


def _ms(v):
    v = np.asarray(v, float)
    return (v.mean(), v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else (v.mean(), 0.0)


def main(results_root, out_root):
    ps.apply()
    rows, io = [], None
    for cell, lam, note in CELLS:
        try:
            res = _results(results_root, RUN, cell)
        except SystemExit:
            continue
        pb, av, jg = per_bin_stats(res)
        if pb.size == 0:
            continue
        io = tgt_peak(res, 'temp').mean()
        rows.append(dict(cell=cell, lam=lam, note=note, pb=pb, av=av, jg=jg,
                         nlP_t=normloss(res, 'temp', 'PCA'), nlK_t=normloss(res, 'temp', 'KL'),
                         nlP_s=normloss(res, 'spat', 'PCA'), nlK_s=normloss(res, 'spat', 'KL'),
                         pk_s=peaky(res, 'spat')))
    if not rows:
        raise SystemExit(f'no {RUN} cells on disk yet — the run may still be going.')

    x = np.arange(len(rows))
    lab = [f"{r['lam']:g}{r['note']}" for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=ps.figsize(3, 1))

    # (a) peakiness: per-bin vs trial-average, against the IO target
    for key, col, nm in [('pb', '#7570b3', 'temporal per-bin'), ('av', '#4a4a4a', 'temporal average'),
                         ('pk_s', '#d95f02', 'spatial')]:
        m = [_ms(r[key])[0] for r in rows]; s = [_ms(r[key])[1] for r in rows]
        ax[0].errorbar(x, m, yerr=s, color=col, lw=1.6, marker='o', ms=4, capsize=2, label=nm)
    ax[0].axhline(io, color='k', ls=':', lw=1.3)
    ax[0].set_ylabel('decoded peakiness (max-prob)', fontsize=8)
    ax[0].set_title('peakiness', fontsize=9)

    # (b) the sampling ratio
    m = [_ms(r['pb'] / r['av'])[0] for r in rows]; s = [_ms(r['pb'] / r['av'])[1] for r in rows]
    ax[1].errorbar(x, m, yerr=s, color='#7570b3', lw=1.6, marker='o', ms=4, capsize=2,
                   label='per-bin ÷ average')
    ax[1].axhline(1.0, color='k', ls=':', lw=1.3)
    ax[1].set_ylabel('per-bin ÷ trial-average peakiness', fontsize=8)
    ax[1].set_title('bin sharpening', fontsize=9)

    # (c) performance, both metrics, both arches
    for key, col, ls, nm in [('nlP_t', '#7570b3', '-', 'temporal, projection'),
                             ('nlK_t', '#7570b3', '--', 'temporal, KL'),
                             ('nlP_s', '#d95f02', '-', 'spatial, projection'),
                             ('nlK_s', '#d95f02', '--', 'spatial, KL')]:
        m = [_ms(r[key])[0] for r in rows]; s = [_ms(r[key])[1] for r in rows]
        ax[2].errorbar(x, m, yerr=s, color=col, ls=ls, lw=1.6, marker='o', ms=4, capsize=2, label=nm)
    ax[2].axhline(1.0, color='k', ls=':', lw=1.3)
    ax[2].set_yscale('log')
    ax[2].set_ylabel('normalised loss', fontsize=8)
    ax[2].set_title('performance', fontsize=9)

    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(lab, rotation=30, ha='right', fontsize=7)
        a_.set_xlabel('entropy λ_H', fontsize=8)
        a_.legend(fontsize=6, frameon=True)
    ps.label_panels(ax)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'peaky_bins_report')

    print(f'shape_lambda = 30 (λ_Brier 0.3), early stopping, REP 3.  IO target = {io:.5f}\n')
    print(f"{'λ_H':>12s}{'per-bin':>10s}{'average':>10s}{'ratio':>8s}{'Jensen':>9s}"
          f"{'projT':>8s}{'KL_T':>7s}{'projS':>8s}{'KL_S':>7s}")
    for r, l in zip(rows, lab):
        print(f"{l:>12s}{_ms(r['pb'])[0]:10.4f}{_ms(r['av'])[0]:10.4f}"
              f"{_ms(r['pb']/r['av'])[0]:8.2f}{_ms(r['jg'])[0]:9.3f}"
              f"{_ms(r['nlP_t'])[0]:8.2f}{_ms(r['nlK_t'])[0]:7.2f}"
              f"{_ms(r['nlP_s'])[0]:8.2f}{_ms(r['nlK_s'])[0]:7.2f}")
    print(f'\nDone -> {Path(out_root).resolve()}/peaky_bins_report.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hparam_summary')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
