# -*- coding: utf-8 -*-
"""Per-trial peakiness: temporal (SBC) vs spatial (PPC), under the PCA loss
(2026-06-05, Máté's request).

Máté, reading the max(P)/H(P) distributions, noted that the SPATIAL PCA decoder's
bulk sits where the calibrated losses do, with a long peaky TAIL — so maybe the
"peaky" example trials are just unlucky tail samples. His diagnostic: a scatter of
per-trial peakiness, temporal (y) vs spatial (x), each dot a trial, colour-coded by
the per-trial PCA-loss difference (spatial − temporal). "Representative" calibrated-
spatial trials live top-left (high temporal peakiness, low spatial peakiness) with a
hot colour (spatial PCA loss > temporal).

Peakiness is shown two ways (`--metric`): max(P), and 1 − H(P)/log N (N=91).

We pool every held-out trial across the 6 mice of the matched PCA run
(`loss_comparison_v1`, Q / half / 100 ms). The colour is the raw per-trial PCA-loss
difference on each mouse's own basis (symmetric robust colour limits).

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  scatter_<metric>.png        the scatter + marginal histograms + y=x + target lines
  representative_examples.png  example posteriors (spatial vs temporal vs target) for
                               a few top-left "representative" trials + a tail trial

Usage:  python diagnostics/peakiness_scatter_spat_temp.py
        python diagnostics/peakiness_scatter_spat_temp.py --metric normH
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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from pca_loss import pca_distance  # noqa: E402

RUN = 'loss_comparison_v1'
SLUG = 'Q_PCA_half_100ms_all'
NCAT = 91


def _H(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def peak(p, metric):
    return p.max(-1) if metric == 'maxP' else 1.0 - _H(p) / np.log(NCAT)


def load_pool(results_root, split):
    """Pool every test trial across mice. Returns a dict of arrays (one entry per
    trial) for spatial/temporal decoded posteriors, target, per-trial PCA loss,
    and the originating mouse + within-mouse trial index."""
    res = sio.loadmat(f'{results_root}/{RUN}/{SLUG}/{split}.mat',
                      simplify_cells=True)['results']
    out = {k: [] for k in ('mouse', 'trial', 'dec_s', 'dec_t', 'tgt',
                           'loss_s', 'loss_t')}
    for mk in sorted(res):
        D = res[mk]['Dist']
        pcs, evar = np.asarray(D['pcs'], float), np.asarray(D['explained_var'], float)
        ds = np.asarray(D['spat']['decoded'], float)
        dt = np.asarray(D['temp']['decoded'], float)
        tg = np.asarray(D['spat']['target'], float)         # spat/temp targets identical
        ls = pca_distance(ds, tg, pcs, evar)
        lt = pca_distance(dt, tg, pcs, evar)
        n = ds.shape[0]
        out['mouse'] += [int(mk.split('_')[1])] * n
        out['trial'] += list(range(n))
        out['dec_s'].append(ds); out['dec_t'].append(dt); out['tgt'].append(tg)
        out['loss_s'].append(ls); out['loss_t'].append(lt)
    for k in ('dec_s', 'dec_t', 'tgt', 'loss_s', 'loss_t'):
        out[k] = np.concatenate(out[k], axis=0)
    out['mouse'] = np.asarray(out['mouse']); out['trial'] = np.asarray(out['trial'])
    return out


def fig_scatter(P, metric, out_dir):
    xs = peak(P['dec_s'], metric)              # spatial peakiness
    ys = peak(P['dec_t'], metric)              # temporal peakiness
    tgt = peak(P['tgt'], metric)
    dloss = P['loss_s'] - P['loss_t']          # hot = spatial fits WORSE on PCA loss
    tgt_mean = float(tgt.mean())
    lim = np.percentile(np.abs(dloss), 90)

    fig = plt.figure(figsize=(9.2, 8.4))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    hi = max(xs.max(), ys.max()) * 1.03
    sc = ax.scatter(xs, ys, c=dloss, cmap='coolwarm', vmin=-lim, vmax=lim,
                    s=10, alpha=0.6, lw=0)
    ax.plot([0, hi], [0, hi], color='0.4', ls='--', lw=1.2, zorder=1)
    ax.axvline(tgt_mean, color='k', ls=':', lw=1.1)
    ax.axhline(tgt_mean, color='k', ls=':', lw=1.1)
    ax.text(tgt_mean, hi * 0.99, ' IO target', fontsize=8, va='top', color='k')

    # medians as crosshair
    mx, my = np.median(xs), np.median(ys)
    ax.axvline(mx, color=ps.PCA_EVAR, ls='-', lw=1.0, alpha=0.5)
    ax.axhline(my, color=ps.PCA_EVAR, ls='-', lw=1.0, alpha=0.5)

    # representative "top-left" trials: low spatial peakiness, high temporal,
    # spatial loss > temporal (hot). Score and pick a spread.
    def pct(v):
        return (v.argsort().argsort() + 0.5) / len(v)
    score = (1 - pct(xs)) + pct(ys) + (dloss > 0).astype(float) * 0.5
    cand = np.where((xs < mx) & (ys > my) & (dloss > 0))[0]
    if len(cand):
        cand = cand[np.argsort(-score[cand])]
        # spread picks across the temporal range
        reps = cand[np.linspace(0, len(cand) - 1, min(5, len(cand))).astype(int)]
        ax.scatter(xs[reps], ys[reps], s=120, facecolors='none',
                   edgecolors='k', lw=1.6, zorder=5)
        for r in reps:
            ax.annotate(f'm{P["mouse"][r]}·{P["trial"][r]}', (xs[r], ys[r]),
                        textcoords='offset points', xytext=(6, 4), fontsize=7)
    else:
        reps = np.array([], int)

    ax.set_xlabel(f'spatial (PPC) peakiness  [{("max P" if metric=="maxP" else "1−H/logN")}]')
    ax.set_ylabel(f'temporal (SBC) peakiness  [{("max P" if metric=="maxP" else "1−H/logN")}]')
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    cb = fig.colorbar(sc, ax=axr, fraction=0.5, pad=0.05)
    cb.set_label('PCA loss:  spatial − temporal   (hot = spatial fits worse)', fontsize=8.5)

    nb = 50
    axt.hist(xs, bins=np.linspace(0, hi, nb), color=ps.FLAT_EVAR, alpha=0.8)
    axt.axvline(mx, color=ps.PCA_EVAR, lw=1.2); axt.axvline(tgt_mean, color='k', ls=':', lw=1)
    axr.hist(ys, bins=np.linspace(0, hi, nb), color=ps.PCA_EVAR, alpha=0.7,
             orientation='horizontal')
    axr.axhline(my, color=ps.PCA_EVAR, lw=1.2); axr.axhline(tgt_mean, color='k', ls=':', lw=1)
    for a in (axt.get_xaxis(), axr.get_yaxis()):
        a.set_visible(False)
    axt.set_yticks([]); axr.set_xticks([])
    axt.set_title(f'Per-trial peakiness — temporal vs spatial, PCA loss '
                  f'(Q half 100ms, {len(xs)} trials, 6 mice)\n'
                  f'spatial median {mx:.3f} · temporal median {my:.3f} · IO target {tgt_mean:.3f}',
                  fontsize=11, loc='left')
    ps.save_fig(fig, out_dir, f'scatter_{metric}')
    return reps


def fig_examples(P, reps, metric, out_dir):
    if not len(reps):
        print('  [examples] no representative trials found'); return
    x = np.arange(NCAT)
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, reps):
        ps.target_band(ax, x, P['tgt'][r], label='IO target')
        ax.plot(x, P['dec_s'][r], color=ps.FLAT_EVAR, lw=1.6, label='spatial (PPC)')
        ax.plot(x, P['dec_t'][r], color=ps.PCA_EVAR, lw=1.6, label='temporal (SBC)')
        ax.set_xlabel('orientation bin')
        ax.set_title(f'm{P["mouse"][r]}·trial {P["trial"][r]}\n'
                     f'spat maxP {P["dec_s"][r].max():.2f} / temp {P["dec_t"][r].max():.2f}',
                     fontsize=8.5)
    axes[0].legend(fontsize=7.5); axes[0].set_ylabel('probability')
    fig.suptitle('Representative top-left trials: spatial is calibrated, temporal is peaky',
                 y=1.04, fontsize=12)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, 'representative_examples')


def main(results_root, split, metric, out_root):
    ps.apply()
    P = load_pool(results_root, split)
    out_dir = Path(out_root)
    reps = fig_scatter(P, metric, out_dir)
    fig_examples(P, reps, metric, out_dir)
    # quick numeric readout (both metrics)
    for m in ('maxP', 'normH'):
        xs, ys, tg = peak(P['dec_s'], m), peak(P['dec_t'], m), peak(P['tgt'], m)
        print(f'  [{m}] spatial: mean {xs.mean():.3f} median {np.median(xs):.3f} | '
              f'temporal: mean {ys.mean():.3f} median {np.median(ys):.3f} | '
              f'target mean {tg.mean():.3f}')
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--metric', default='maxP', choices=('maxP', 'normH'))
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.split, a.metric, a.out_root)
