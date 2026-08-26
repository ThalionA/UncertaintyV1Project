# -*- coding: utf-8 -*-
"""Does the temporal advantage exceed what the ARCHITECTURE confers for free?

The temporal (SBC) decoder Jensen-averages 10 per-bin posteriors; averaging helps
even a decoder with no information, so "temporal beats spatial" needs a control
that holds the architecture fixed and removes the information. The shuffle
decoders are exactly that: same architecture, trained on shuffled targets, scored
against their own shuffled target (the cross_loss_eval convention).

  Delta_real     = temporal - spatial normalised loss   (the effect)
  Delta_shuffle  = same for the shuffle decoders        (the architectural bias)
  Delta_real - Delta_shuffle < 0  =>  the advantage exceeds the architecture

This is the control the 2026-07-29 entry demanded: it found the shuffle null
architecturally biased (temporal shuffle 1.10-1.13x LOWER than spatial) and
concluded "do not use the shuffle null for spatial-vs-temporal" — as a DIVISOR.
Used as a CONTRAST (this figure) it is the right tool: it measures the bias
rather than dividing by it.

Losses are per-trial KL, each divided by that arch's own leave-one-out
predict-mean null (a per-mouse constant), so "1" is the trivial predictor.

Usage: python diagnostics/spat_temp_shuffle_control.py [--cell kl_h8_lh0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sstats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps                       # noqa: E402
from nn_classifier import fit_loss_per_trial       # noqa: E402

ARMS = [('io_hmm_v3', 'IO-HMM targets (72 x 2.5 deg)'),
        ('io_hmm_v3_exportref', 'old export Q (91 x 1 deg)')]
EXCLUDE = 2     # mouse 2, flagged red throughout this project


def _kl(dec, tgt):
    return np.asarray(fit_loss_per_trial(torch.tensor(dec), torch.tensor(tgt),
                                         'KL', None, None).detach(), float)


def arm_values(run, cell, loss='KL'):
    """{arch: per-mouse normalised loss} for the real and shuffle decoders."""
    d = next(Path(f'results/{run}/{cell}').glob(f'Q_{loss}_half_100ms*'))
    R = loadmat(str(d / 'stratified_balanced.mat'), squeeze_me=True,
                struct_as_record=False)['results']
    mice = sorted(int(k.split('_')[-1]) for k in R._fieldnames if k.startswith('mouse_'))
    out = {a: [] for a in ('spat', 'temp', 'spat_shf', 'temp_shf')}
    for m in mice:
        D = getattr(R, f'mouse_{m}').Dist
        for arch in out:
            a = getattr(D, arch)
            dec = np.asarray(a.decoded, float); tgt = np.asarray(a.target, float)
            ok = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
            n, tot = int(ok.sum()), tgt[ok].sum(0)
            pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
            pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)          # leave-one-out
            out[arch].append(_kl(dec, tgt)[ok].mean() / _kl(pm, tgt)[ok].mean())
    return {k: np.array(v) for k, v in out.items()}, mice


def figure(data, mice, cell, out_root):
    ps.apply()
    fig, ax = plt.subplots(len(ARMS), 2, figsize=ps.figsize(2, len(ARMS)))
    for r, (run, alab) in enumerate(ARMS):
        v = data[run]
        real, shuf = v['temp'] - v['spat'], v['temp_shf'] - v['spat_shf']

        # -- left: the four absolute values ---------------------------------
        a = ax[r][0]
        for i, m in enumerate(mice):
            c = '#cb181d' if m == EXCLUDE else '0.7'
            a.plot([0, 1], [v['spat'][i], v['temp'][i]], '-o', ms=3.5, lw=1.0, color=c)
            a.plot([2, 3], [v['spat_shf'][i], v['temp_shf'][i]], '-o', ms=3.5, lw=1.0, color=c)
        for x0, key in ((0, 'spat'), (1, 'temp'), (2, 'spat_shf'), (3, 'temp_shf')):
            y = v[key]
            a.errorbar([x0], [y.mean()], yerr=[y.std(ddof=1) / np.sqrt(y.size)],
                       color=ps.SPATIAL if 'spat' in key else ps.TEMPORAL,
                       marker='o', ms=7, lw=2.2, capsize=4, zorder=5)
        a.axhline(1.0, color='0.4', ls=':', lw=1.1)
        a.set_xticks([0, 1, 2, 3])
        a.set_xticklabels(['spatial', 'temporal', 'spatial', 'temporal'], fontsize=7)
        a.text(0.5, -0.20, 'real decoders', ha='center', transform=a.get_xaxis_transform(),
               fontsize=7.5, style='italic')
        a.text(2.5, -0.20, 'SHUFFLE (no information)', ha='center',
               transform=a.get_xaxis_transform(), fontsize=7.5, style='italic')
        a.set_xlim(-0.4, 3.4)
        a.set_ylabel(f'{alab}\nnormalised loss (÷ predict-mean)', fontsize=7.5)
        if r == 0:
            a.set_title('absolute performance', fontsize=9)

        # -- right: the contrast that matters --------------------------------
        b = ax[r][1]
        for i, m in enumerate(mice):
            b.plot([0, 1], [real[i], shuf[i]], '-o', ms=4, lw=1.2,
                   color='#cb181d' if m == EXCLUDE else '0.62',
                   zorder=4 if m == EXCLUDE else 2)
        for x0, y in ((0, real), (1, shuf)):
            b.errorbar([x0], [y.mean()], yerr=[y.std(ddof=1) / np.sqrt(y.size)],
                       color='k', marker='o', ms=7, lw=2.4, capsize=4, zorder=5)
        b.axhline(0, color='k', lw=1.0)
        t_r = sstats.ttest_1samp(real, 0); t_s = sstats.ttest_1samp(shuf, 0)
        t_d = sstats.ttest_rel(real, shuf)
        b.text(0.03, 0.03,
               f'real   Δ={real.mean():+.3f}  t(5)={t_r.statistic:+.2f}, p={t_r.pvalue:.3f}\n'
               f'shuffle Δ={shuf.mean():+.3f}  t(5)={t_s.statistic:+.2f}, p={t_s.pvalue:.3f}\n'
               f'real−shuffle {np.mean(real-shuf):+.3f}  t(5)={t_d.statistic:+.2f}, '
               f'p={t_d.pvalue:.3f}   ({int((real < shuf).sum())}/6 mice)',
               transform=b.transAxes, fontsize=6, va='bottom', ha='left',
               bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
        b.set_xticks([0, 1])
        b.set_xticklabels(['real\n(the effect)', 'shuffle\n(architecture only)'], fontsize=7.5)
        b.set_xlim(-0.35, 1.35)
        b.set_ylabel('Δ (temporal − spatial)', fontsize=7.5)
        if r == 0:
            b.set_title('effect vs architectural bias', fontsize=9)
    ax[0][0].legend(handles=[
        Line2D([0], [0], color='0.7', lw=1.0, marker='o', ms=3.5, label='mouse'),
        Line2D([0], [0], color='#cb181d', lw=1.2, marker='o', ms=3.5, label=f'mouse {EXCLUDE}'),
        Line2D([0], [0], color='k', lw=2.2, marker='o', ms=6, label='mean ± SEM')],
        fontsize=6, frameon=True, loc='best')
    fig.suptitle(f'Does the temporal advantage exceed the architecture?  {cell}, KL, '
                 'n = 6 mice\nshuffle decoders = same architecture trained on shuffled '
                 'targets (no information); below 0 = temporal better', fontsize=9.5)
    ps.label_panels(ax.ravel())
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ps.save_fig(fig, Path(out_root), f'spat_temp_shuffle_control_{cell}')


def main(cell, out_root):
    data, mice = {}, None
    for run, _ in ARMS:
        data[run], mice = arm_values(run, cell)
    figure(data, mice, cell, out_root)
    for run, alab in ARMS:
        v = data[run]
        real, shuf = v['temp'] - v['spat'], v['temp_shf'] - v['spat_shf']
        print(f'{alab:34s} real {real.mean():+.4f} (p={sstats.ttest_1samp(real,0).pvalue:.4f})  '
              f'shuffle {shuf.mean():+.4f} (p={sstats.ttest_1samp(shuf,0).pvalue:.4f})  '
              f'real−shuffle {np.mean(real-shuf):+.4f} '
              f'(p={sstats.ttest_rel(real,shuf).pvalue:.4f}, {int((real<shuf).sum())}/6)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cell', default='kl_h8_lh0')
    ap.add_argument('--out-root', default='figures/io_hmm_wide/spat_temp')
    a = ap.parse_args()
    main(a.cell, a.out_root)
