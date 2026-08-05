# -*- coding: utf-8 -*-
"""Per-trial spatial-vs-temporal scatter with exemplar trials pulled out and their
TEMPORAL BIN posteriors shown as heatmaps — one figure per configuration
(2026-08-04, Theo's request).

Layout per configuration:
  left, tall   scatter of per-trial loss, spatial (x) vs temporal (y), each
               normalised to that mouse's leave-one-out predict-mean so 1 = chance
               on both axes. Below the diagonal = temporal better. Four exemplar
               trials are circled and numbered.
  top row      for each exemplar: IO target (grey fill) with the spatial and
               temporal decoded posteriors overlaid.
  bottom row   for each exemplar: the temporal decoder's INDIVIDUAL time-bin
               posteriors (`decoded_samp`, 91 orientations x 10 bins) as a heatmap
               — what the Jensen average is averaging over.

Exemplars are chosen to span the scatter, using
  d = log(temporal) - log(spatial)   (which architecture wins, and by how much)
  m = log(temporal) + log(spatial)   (overall difficulty of the trial)
picking: both good (lowest m), spatial much better (largest d), temporal much
better (smallest d), both bad (largest m).

Trials are pooled ACROSS mice after per-mouse normalisation, so an exemplar is
labelled with its mouse. Scored under the cell's OWN training weighting — spatial
and temporal share it within a cell, so the comparison is like-for-like.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_trial_explorer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from projflat_report import _res, _mice, have  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial  # noqa: E402

CONFIGS = [
    ('linear, flat (MSE)',        'lin_raw_l0_d0_w0'),
    ('8 hidden, flat (MSE)',      'h8_raw_l0_d0_w0'),
    ('linear, variance-weighting', 'lin_raw_EVAR'),
    ('8 hidden, variance-weighting', 'h8_raw_EVAR'),
]
THETA = np.linspace(0, 90, 91)


def _t(x):
    return torch.tensor(np.asarray(x, float))


def gather(results_root, cell):
    """Pooled per-trial normalised losses + an index back to (mouse, trial)."""
    r = _res(results_root, cell)
    sp, te, idx = [], [], []
    for m in _mice(r):
        D = r[m]['Dist']
        pcs, evar = D.get('pcs'), D.get('explained_var')
        tgt = np.asarray(D['spat']['target'], float)
        ds = np.asarray(D['spat']['decoded'], float)
        dt = np.asarray(D['temp']['decoded'], float)
        ok = np.isfinite(tgt).all(1) & np.isfinite(ds).all(1) & np.isfinite(dt).all(1)
        n = int(ok.sum()); tot = tgt[ok].sum(0)
        pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
        pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)
        ch = np.nanmean(fit_loss_per_trial(_t(pm[ok]), _t(tgt[ok]), 'PCA',
                                           _t(pcs), _t(evar)).numpy())
        s = fit_loss_per_trial(_t(ds[ok]), _t(tgt[ok]), 'PCA', _t(pcs), _t(evar)).numpy() / ch
        t = fit_loss_per_trial(_t(dt[ok]), _t(tgt[ok]), 'PCA', _t(pcs), _t(evar)).numpy() / ch
        rows = np.where(ok)[0]
        sp.append(s); te.append(t)
        idx += [(m, int(i)) for i in rows]
    return np.concatenate(sp), np.concatenate(te), idx


def _nearest(vals, target, pool=None):
    """Index of the trial whose value is closest to `target` (within `pool`)."""
    cand = np.arange(vals.size) if pool is None else np.asarray(pool)
    return int(cand[np.argmin(np.abs(vals[cand] - target))])


def pick(sp, te):
    """Four representative exemplars spanning the scatter.

    Chosen by PERCENTILE rather than absolute extreme: argmax/argmin lands on
    degenerate outliers (losses of ~0 or ~80) that illustrate nothing. Each pick is
    the trial nearest a chosen percentile, so it is typical of that region.
    """
    eps = 1e-12
    ls, lt = np.log(sp + eps), np.log(te + eps)
    d, m = lt - ls, lt + ls                     # who wins / how hard overall
    mid = np.where(np.abs(d) < np.percentile(np.abs(d), 40))[0]   # near-diagonal
    return [('both good',          _nearest(m, np.percentile(m, 12), mid)),
            ('spatial better',     _nearest(d, np.percentile(d, 97))),
            ('temporal better',    _nearest(d, np.percentile(d, 3))),
            ('both poor',          _nearest(m, np.percentile(m, 93), mid))]


def make_figure(results_root, out_root, title, cell):
    if not have(results_root, cell):
        print(f'  [skip] {cell}')
        return
    sp, te, idx = gather(results_root, cell)
    picks = pick(sp, te)
    r = _res(results_root, cell)

    ps.apply()
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.55, 0.12, 1, 1, 1, 1],
                          hspace=0.42, wspace=0.32)

    # ---- scatter ----
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(sp, te, s=7, c='0.55', alpha=0.45, edgecolor='none', zorder=1)
    lim = [min(sp.min(), te.min()) * 0.7, max(sp.max(), te.max()) * 1.4]
    ax.plot(lim, lim, color='0.3', ls=':', lw=1.4, zorder=2)
    ax.axhline(1.0, color='k', ls='--', lw=0.9, alpha=0.6)
    ax.axvline(1.0, color='k', ls='--', lw=0.9, alpha=0.6)
    for j, (lab, i) in enumerate(picks, 1):
        ax.plot(sp[i], te[i], 'o', ms=13, mfc='none', mec=ps.PCA_EVAR, mew=2, zorder=5)
        ax.annotate(str(j), (sp[i], te[i]), fontsize=9, fontweight='bold',
                    color=ps.PCA_EVAR, xytext=(9, 5), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('spatial loss ÷ predict-mean', fontsize=8)
    ax.set_ylabel('temporal loss ÷ predict-mean', fontsize=8)
    frac = float((te < sp).mean())
    ax.set_title(f'{title}\nper-trial, pooled over 6 mice (n={sp.size}). '
                 f'Below diagonal = temporal better ({frac:.0%} of trials).\n'
                 f'Dashed = chance on each axis.', fontsize=8)

    # ---- exemplars ----
    for j, (lab, i) in enumerate(picks):
        m, tr = idx[i]
        D = r[m]['Dist']
        tgt = np.asarray(D['spat']['target'], float)[tr]
        ds = np.asarray(D['spat']['decoded'], float)[tr]
        dt = np.asarray(D['temp']['decoded'], float)[tr]
        samp = np.asarray(D['temp']['decoded_samp'], float)

        axt = fig.add_subplot(gs[0, 2 + j])
        axt.fill_between(THETA, 0, tgt, color='0.75', lw=0,
                         label='IO target' if j == 0 else None)
        axt.plot(THETA, ds, color=ps.SPATIAL, lw=1.3,
                 label='spatial' if j == 0 else None)
        axt.plot(THETA, dt, color=ps.TEMPORAL, lw=1.3,
                 label='temporal' if j == 0 else None)
        axt.set_yticks([])
        axt.set_title(f'{j + 1}. {lab}\n{m}, trial {tr}\n'
                      f'spat {sp[i]:.2f} · temp {te[i]:.2f}', fontsize=7)
        if j == 0:
            axt.legend(fontsize=5.5, frameon=True)

        axh = fig.add_subplot(gs[1, 2 + j])
        if samp.ndim == 3 and samp.shape[0] > tr:
            bins = samp[tr]
            im = axh.imshow(bins, aspect='auto', origin='lower', cmap='magma',
                            vmin=0, vmax=np.nanpercentile(bins, 99.5),
                            extent=[0, bins.shape[1], 0, 90])
            axh.plot(bins.shape[1] * tgt / max(tgt.max(), 1e-12) * 0.28,
                     THETA, color='w', lw=1.1, alpha=0.9)
            fig.colorbar(im, ax=axh, fraction=0.045, pad=0.03).ax.tick_params(labelsize=5)
        axh.set_xlabel('time bin', fontsize=7)
        if j == 0:
            axh.set_ylabel('orientation (deg)', fontsize=7)
        axh.set_title('temporal bins (white = target)', fontsize=6.5)

    fig.suptitle(f'{title} — spatial vs temporal per trial, with exemplars from four regions of the scatter. '
                 'Bottom row shows what the temporal decoder emits in each 100 ms bin before the Jensen average.',
                 y=1.03, fontsize=8.5)
    stem = 'projflat_trials_' + cell.replace('_l0_d0_w0', '').replace('_raw', '')
    ps.save_fig(fig, Path(out_root), stem)
    print(f'  {stem}: ' + ' | '.join(
        f'{j+1}.{lab} {idx[i][0]}/{idx[i][1]} s={sp[i]:.2f} t={te[i]:.2f}'
        for j, (lab, i) in enumerate(picks)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()
    for title, cell in CONFIGS:
        make_figure(a.results_root, a.out_root, title, cell)


if __name__ == '__main__':
    main()
