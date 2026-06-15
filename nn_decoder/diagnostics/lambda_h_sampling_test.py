# -*- coding: utf-8 -*-
"""Does the temporal decoder SAMPLE, or just sharpen? The definitive SBC test on
the λ_H sweep (2026-06-10 meeting, Theo's framing).

Peaky bins (low per-bin entropy) are necessary but not sufficient for sampling — the
peaks must sit at VARIED orientations across the 10 bins so they average to a broad
posterior. The clean, argmax-noise-free way to measure this is the law of total
variance on the time-averaged posterior (θ = orientation 0–90°):

    Var(time-avg posterior) = mean_t Var_within-bin(θ)  +  Var_t( mean_within-bin(θ) )
                            = WITHIN-bin width²          +  BETWEEN-bin sample spread²

  - **between-bin fraction** = between / (within + between): the share of the posterior
    width carried by spread of per-bin peak *locations*. SBC ⇒ high; "bins ≈ the
    average" (no sampling) ⇒ ≈ 0.
  - **sample-location spread** = √between (deg): the absolute spread of per-bin means;
    SBC ⇒ approaches the IO-target width.

Robust to broad bins (uses each bin's mean, not its noisy argmax). Reads the
`lambdaH_sweep_entlam<λ>` runs (temporal arch). Two figures (PNG+SVG):
  lambda_h_sampling_test.png      — between-fraction + sample-spread vs λ_H per loss
  lambda_h_bin_gallery.png        — an example trial's 10 per-bin posteriors, per loss, λ_H 0 vs max

Usage:  python diagnostics/lambda_h_sampling_test.py
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
THETA = np.arange(91.0)        # orientation grid, degrees (0..90)


def _moments(p, cat_axis):
    """mean and variance of θ under distribution p along cat_axis (the 91 grid)."""
    p = np.clip(np.asarray(p, float), 0, None)
    p = p / np.clip(p.sum(cat_axis, keepdims=True), 1e-12, None)
    shape = [1] * p.ndim
    shape[cat_axis] = THETA.size
    th = THETA.reshape(shape)
    mu = (p * th).sum(cat_axis)
    var = (p * th ** 2).sum(cat_axis) - mu ** 2
    return mu, var


def collect(results_root, prefix, split):
    """{lambda: {loss: dict of per-mouse lists}} for the between/within decomposition."""
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
            rec = {k: [] for k in ('frac', 'between_sd', 'within_sd', 'total_sd', 'tgt_sd')}
            for mk in sorted(res):
                D = res[mk]['Dist']['temp']
                ds = np.asarray(D['decoded_samp'], float)    # (n, 91, T)
                tgt = np.asarray(D['target'], float)         # (n, 91)
                mu_t, var_t = _moments(ds, 1)                # (n, T) per-bin mean & var of θ
                within = np.nanmean(var_t, 1)                # (n,) mean within-bin variance
                between = np.nanvar(mu_t, axis=1)            # (n,) variance across bin-means
                total = within + between                     # (n,) == Var(time-avg posterior)
                _, tgt_var = _moments(tgt, 1)                # (n,)
                rec['frac'].append(float(np.nanmean(between / np.clip(total, 1e-9, None))))
                rec['between_sd'].append(float(np.nanmean(np.sqrt(between))))
                rec['within_sd'].append(float(np.nanmean(np.sqrt(within))))
                rec['total_sd'].append(float(np.nanmean(np.sqrt(total))))
                rec['tgt_sd'].append(float(np.nanmean(np.sqrt(np.clip(tgt_var, 0, None)))))
            data.setdefault(lam, {})[loss] = rec
    return data


def fig_sampling(data, out_root):
    ps.apply()
    lams = sorted(data)
    x = np.arange(len(lams))
    xlabels = [f'{l:g}' for l in lams]
    tgt_sd = np.nanmean([_agg(data[l][L]['tgt_sd'])[0] for l in lams for L in data[l]])

    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharex=True)
    # (a) between-bin fraction
    for loss in LOSSES:
        ys = [_agg(data[l].get(loss, {}).get('frac', []))[0] for l in lams]
        es = [_agg(data[l].get(loss, {}).get('frac', []))[1] for l in lams]
        axes[0].errorbar(x, ys, yerr=es, color=LCOL[loss], lw=2, marker='o', ms=4,
                         capsize=2, label=loss)
    axes[0].set_ylabel('between-bin fraction\nVar$_t(\\mu_t)$ / Var(time-avg)')
    axes[0].set_title('Is posterior width from sample SPREAD?\n(high = sampling; ~0 = bins ≈ average)',
                      fontsize=8.5)
    axes[0].legend(fontsize=7, loc='best')
    # (b) absolute sample-location spread vs IO target width
    for loss in LOSSES:
        ys = [_agg(data[l].get(loss, {}).get('between_sd', []))[0] for l in lams]
        axes[1].plot(x, ys, color=LCOL[loss], lw=2, marker='o', ms=4, label=loss)
    axes[1].axhline(tgt_sd, color='k', ls=':', lw=1.5, label='IO target width')
    axes[1].set_ylabel('sample-location spread  √between  (deg)')
    axes[1].set_title('Do sample locations tile the posterior?\n(→ IO target width = full tiling)',
                      fontsize=8.5)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right')
        ax.set_xlabel(r'entropy penalty $\lambda_H$')
    ps.label_panels(axes)
    fig.suptitle('Does the temporal decoder sample? Between- vs within-bin width across λ_H '
                 '(6 mice, Q/half/100ms)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'lambda_h_sampling_test')

    print(f'IO-target width ≈ {tgt_sd:.1f} deg')
    print(f"{'loss':12s} {'λ_H':>6s} | {'between-frac':>11s} {'spread(deg)':>11s} {'within(deg)':>11s}")
    for loss in LOSSES:
        for lam in lams:
            fr = _agg(data[lam].get(loss, {}).get('frac', []))[0]
            bsd = _agg(data[lam].get(loss, {}).get('between_sd', []))[0]
            wsd = _agg(data[lam].get(loss, {}).get('within_sd', []))[0]
            print(f'  {loss:10s} {lam:6.3f} | {fr:11.2f} {bsd:11.1f} {wsd:11.1f}')


def fig_gallery(results_root, prefix, split, out_root, mouse='mouse_0'):
    ps.apply()
    runs = sorted(Path(results_root).glob(f'{prefix}_entlam*'),
                  key=lambda p: (_lambda_of(p.name) if _lambda_of(p.name) is not None else 1e9))
    if len(runs) < 2:
        print('gallery: need ≥2 λ_H runs — skipped.'); return
    lo_run, hi_run = runs[0], runs[-1]
    lo_lam, hi_lam = _lambda_of(lo_run.name), _lambda_of(hi_run.name)

    # pick one broad-target example trial from the PCA cell of the low-λ_H run
    fref = _find_cell(lo_run, 'PCA', split)
    res = sio.loadmat(str(fref), simplify_cells=True)['results']
    tgt0 = np.asarray(res[mouse]['Dist']['temp']['target'], float)
    sel = int(np.nanargmax([_moments(tgt0[i:i+1], 1)[1][0] for i in range(tgt0.shape[0])]))  # widest target

    fig, axes = plt.subplots(len(LOSSES), 2,
                             figsize=ps.figsize(2, len(LOSSES), panel_w=2.4, panel_h=1.3),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for r, loss in enumerate(LOSSES):
        for c, (rd, lam) in enumerate([(lo_run, lo_lam), (hi_run, hi_lam)]):
            ax = axes[r, c]
            f = _find_cell(rd, loss, split)
            if f is None:
                ax.axis('off'); continue
            D = sio.loadmat(str(f), simplify_cells=True)['results'][mouse]['Dist']['temp']
            ds = np.asarray(D['decoded_samp'], float)[sel]   # (91, T)
            dec = np.asarray(D['decoded'], float)[sel]        # (91,)
            tg = np.asarray(D['target'], float)[sel]
            ax.fill_between(THETA, tg, color='0.85', lw=0, zorder=0)
            T = ds.shape[1]
            for t in range(T):                                # the 10 per-bin "samples"
                ax.plot(THETA, ds[:, t], color=plt.cm.viridis(t / max(T - 1, 1)), lw=0.8, alpha=0.8)
            ax.plot(THETA, dec, color='k', lw=1.6, label='time-avg')
            ps.cap_posterior_ylim(ax, float(np.nanmax(tg)), mult=3.0)  # keep the target visible
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'λ_H = {lam:g}', fontsize=9)
            if c == 0:
                ax.set_ylabel(loss, fontsize=8.5, color=LCOL[loss])
            if r == len(LOSSES) - 1:
                ax.set_xlabel('orientation (deg)', fontsize=8)
    fig.suptitle(f'Per-bin distributions (10 bins, viridis) + time-average (black) over the IO target '
                 f'(grey)\n{mouse}, widest-target trial #{sel}: do the bins spread or stack?', y=1.01)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'lambda_h_bin_gallery')
    print(f'\ngallery: trial #{sel} (widest target), {mouse}, λ_H {lo_lam:g} vs {hi_lam:g}')


def main(results_root, prefix, split, out_root):
    data = collect(results_root, prefix, split)
    if not data:
        raise SystemExit(f'no {prefix}_entlam* runs found under {results_root}.')
    fig_sampling(data, out_root)
    fig_gallery(results_root, prefix, split, out_root)
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', default='lambdaH_sweep')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.prefix, a.split, a.out_root)
