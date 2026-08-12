# -*- coding: utf-8 -*-
"""Per-mouse spatial-vs-temporal loss scatter, coloured by how much the temporal
decoder's INDIVIDUAL time bins differ from its FULL (Jensen-averaged) model.

Item 5 from the 2026-08-05 meeting: a square-axis scatter + density, broken out
per individual mouse, with dots coloured by the average difference between the
individual temporal bins and the full temporal model.

Axes (both, per trial, per mouse):
    x = spatial loss  / that mouse's leave-one-out predict-mean   (1 = chance)
    y = temporal loss / that mouse's leave-one-out predict-mean   (1 = chance)
Below the y=x diagonal = the temporal decoder fits that trial better. Losses come
from ``nn_classifier.fit_loss_per_trial(..., 'PCA', pcs, explained_variance)``.

WEIGHTING: each cell is scored under ITS OWN stored projection weighting — the
per-cell ``explained_var`` is the eigenvalue spectrum for the variance-weighting
(EVAR) cells and uniform 1/91 (= MSE over an orthonormal 91-PC basis) for the
``flat_evar=1`` cells. So the metric DIFFERS between the flat and evar configs:
compare spatial vs temporal WITHIN a panel/config, not loss values ACROSS configs.
(Same convention as ``diagnostics/projflat_spat_vs_temp_bymouse.py``, which offers
``--weighting common`` to rescore everything under one evar weighting instead.)
The colour metric is weighting-independent — it is computed on the raw posteriors.

The "full temporal model" posterior is the Jensen average = the arithmetic mean
over the per-bin posteriors (``decoded_samp``); ``decoded`` IS that mean (verified
identical to 1e-8). The colour metric measures, per trial, how far the individual
bins sit from that average:

  --colour jsd   (default) mean_bin KL(bin ‖ average), in BITS. This is the
                 information radius of the bin set = I(bin ; orientation): how many
                 bits of orientation the Jensen average discards by pooling over
                 time. 0 = every bin says the same thing; large = the posterior
                 sweeps across bins. Target-independent (pure posterior spread).
  --colour l1    mean_bin total-variation |bin - average|/2, in [0, 1]. Simpler,
                 bounded spread; also target-independent.
  --colour gain  mean_bin(per-bin loss) - full-model loss, in predict-mean units
                 (signed): how much pooling over time IMPROVES the fit vs a typical
                 single bin. >0 = averaging helps (bins noisy around the truth);
                 ~0 = a single bin was already as good. Uses the IO target.

Usage
-----
    python scatter_spat_temp_by_mouse.py                       # linear/flat, jsd
    python scatter_spat_temp_by_mouse.py --colour gain
    python scatter_spat_temp_by_mouse.py --config all
    python scatter_spat_temp_by_mouse.py --config h8_raw_EVAR --colour l1
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / 'diagnostics'))
import peakiness_style as ps                              # noqa: E402
from nn_classifier import fit_loss_per_trial              # noqa: E402
from projflat_report import _res, _mice, have             # noqa: E402

# The four headline configurations, same as the exemplar explorer.
CONFIGS = {
    'lin_raw_l0_d0_w0': 'linear, flat (MSE)',
    'h8_raw_l0_d0_w0':  '8 hidden, flat (MSE)',
    'lin_raw_EVAR':     'linear, variance-weighting',
    'h8_raw_EVAR':      '8 hidden, variance-weighting',
}

# (cmap, diverging?, unit label) per colour metric.
COLOUR_META = {
    'jsd':  ('viridis', False, 'temporal spread: mean$_{bin}$ KL(bin ‖ avg)  [bits]'),
    'l1':   ('viridis', False, 'temporal spread: mean$_{bin}$ TV(bin, avg)  [0–1]'),
    'gain': ('coolwarm', True, 'pooling gain: mean bin loss − full loss  [÷ predict-mean]'),
}

EPS = 1e-12


def _t(x):
    return torch.tensor(np.asarray(x, float))


def _bin_metric(samp, target, pcs, evar, kind):
    """Per-trial colour value from the temporal per-bin posteriors.

    samp: (n, 91, n_bins) normalised per-bin posteriors. Returns (n,)."""
    samp = np.asarray(samp, float)
    qbar = samp.mean(2)                                   # (n,91) == full model
    qbar = qbar / np.clip(qbar.sum(1, keepdims=True), EPS, None)
    if kind == 'jsd':
        lb = np.log2(samp + EPS) - np.log2(qbar[:, :, None] + EPS)
        kl = np.sum(samp * lb, axis=1)                    # (n, n_bins), bits
        return kl.mean(1)
    if kind == 'l1':
        tv = 0.5 * np.sum(np.abs(samp - qbar[:, :, None]), axis=1)  # (n, n_bins)
        return tv.mean(1)
    if kind == 'gain':
        full = fit_loss_per_trial(_t(qbar), _t(target), 'PCA', _t(pcs), _t(evar)).numpy()
        per_bin = np.stack([
            fit_loss_per_trial(_t(samp[:, :, b]), _t(target), 'PCA', _t(pcs), _t(evar)).numpy()
            for b in range(samp.shape[2])
        ], axis=1)                                        # (n, n_bins)
        return per_bin.mean(1) - full
    raise ValueError(kind)


def gather(results_root, cell, kind):
    """Per-mouse dict: m -> (spat_norm, temp_norm, colour). Losses ÷ that
    mouse's leave-one-out predict-mean (1 = chance); colour per ``kind``."""
    r = _res(results_root, cell)
    out = {}
    for m in _mice(r):
        D = r[m]['Dist']
        pcs, evar = D.get('pcs'), D.get('explained_var')
        tgt = np.asarray(D['spat']['target'], float)
        ds = np.asarray(D['spat']['decoded'], float)
        dt = np.asarray(D['temp']['decoded'], float)
        samp = np.asarray(D['temp']['decoded_samp'], float)
        ok = (np.isfinite(tgt).all(1) & np.isfinite(ds).all(1)
              & np.isfinite(dt).all(1) & np.isfinite(samp).all((1, 2)))
        n = int(ok.sum())
        tot = tgt[ok].sum(0)
        pm = (tot[None, :] - tgt[ok]) / (n - 1)           # LOO predict-mean posterior
        ch = float(np.nanmean(
            fit_loss_per_trial(_t(pm), _t(tgt[ok]), 'PCA', _t(pcs), _t(evar)).numpy()))
        s = fit_loss_per_trial(_t(ds[ok]), _t(tgt[ok]), 'PCA', _t(pcs), _t(evar)).numpy() / ch
        t = fit_loss_per_trial(_t(dt[ok]), _t(tgt[ok]), 'PCA', _t(pcs), _t(evar)).numpy() / ch
        c = _bin_metric(samp[ok], tgt[ok], pcs, evar, kind)
        if kind == 'gain':
            c = c / ch                                    # same currency as the axes
        out[m] = (s, t, c)
    return out


def plot(results_root, out_root, cell, kind):
    if not have(results_root, cell):
        print(f'  [skip] {cell}')
        return
    per_mouse = gather(results_root, cell, kind)
    cmap, diverging, unit = COLOUR_META[kind]

    # Shared colour + axis limits across mice, so the panels are comparable.
    all_c = np.concatenate([c for _, _, c in per_mouse.values()])
    all_s = np.concatenate([s for s, _, _ in per_mouse.values()])
    all_t = np.concatenate([t for _, t, _ in per_mouse.values()])
    if diverging:
        vabs = np.nanpercentile(np.abs(all_c), 99)
        vmin, vmax = -vabs, vabs
        norm = matplotlib.colors.TwoSlopeNorm(0.0, vmin=vmin, vmax=vmax)
    else:
        vmin = float(np.nanpercentile(all_c, 1))
        vmax = float(np.nanpercentile(all_c, 99))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    pos = np.concatenate([all_s[all_s > 0], all_t[all_t > 0]])
    lim = [float(np.nanpercentile(pos, 0.5)) * 0.8,
           float(max(np.nanpercentile(all_s, 99.5), np.nanpercentile(all_t, 99.5))) * 1.25]

    ps.apply()
    mice = list(per_mouse.keys())
    ncol = 3
    nrow = int(np.ceil(len(mice) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.4 * nrow),
                             squeeze=False)
    sc = None
    for k, m in enumerate(mice):
        ax = axes[k // ncol][k % ncol]
        s, t, c = per_mouse[m]
        # faint density backdrop, then the coloured points on top.
        ax.hexbin(s, t, gridsize=28, xscale='log', yscale='log', cmap='Greys',
                  mincnt=1, alpha=0.22, linewidths=0, extent=(np.log10(lim[0]),
                  np.log10(lim[1]), np.log10(lim[0]), np.log10(lim[1])))
        sc = ax.scatter(s, t, c=c, cmap=cmap, norm=norm, s=10, alpha=0.8,
                        linewidths=0, zorder=3)
        ax.plot(lim, lim, color='0.3', ls=':', lw=1.3, zorder=2)
        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5, zorder=1)
        ax.axvline(1.0, color='k', ls='--', lw=0.8, alpha=0.5, zorder=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        frac = float((t < s).mean())
        ax.set_title(f'{m}  (n={s.size}, {frac:.0%} temporal-better)', fontsize=8)
        if k // ncol == nrow - 1:
            ax.set_xlabel('spatial loss ÷ predict-mean', fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel('temporal loss ÷ predict-mean', fontsize=8)
    for k in range(len(mice), nrow * ncol):        # blank any unused cells
        axes[k // ncol][k % ncol].axis('off')

    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(unit, fontsize=8)
    fig.suptitle(f'{CONFIGS.get(cell, cell)} — spatial vs temporal loss per trial, per mouse '
                 f'(square log-log, 1 = chance on each; below diagonal = temporal better).\n'
                 f'Colour = {unit}', fontsize=10, y=1.035)
    stem = f'scatter_spat_temp_by_mouse_{cell}_{kind}'
    ps.save_fig(fig, Path(out_root), stem)
    print(f'  {stem}: colour[{kind}] range {vmin:.3g}..{vmax:.3g}, '
          f'{len(mice)} mice, n={all_s.size}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/scatter_spat_temp_by_mouse')
    ap.add_argument('--config', default='lin_raw_l0_d0_w0',
                    help="cell key, or 'all' for the four headline configs")
    ap.add_argument('--colour', default='jsd', choices=list(COLOUR_META),
                    help="jsd (bits, default) | l1 (total variation) | gain (loss improvement)")
    a = ap.parse_args()
    cells = list(CONFIGS) if a.config == 'all' else [a.config]
    print(f'Per-mouse spat/temp scatter | colour={a.colour} | configs={cells}')
    for cell in cells:
        plot(a.results_root, a.out_root, cell, a.colour)
    print(f'Done. {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
