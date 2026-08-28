# -*- coding: utf-8 -*-
"""Per-mouse spatial-vs-temporal loss scatter, coloured by how much the temporal
decoder's INDIVIDUAL time bins differ from its FULL (Jensen-averaged) model.

Item 5 from the 2026-08-05 meeting: a square-axis scatter + density, broken out
per individual mouse, with dots coloured by the average difference between the
individual temporal bins and the full temporal model.

Two views of the SAME numbers (``--mode scatter|density|both``, default both):
  scatter  one dot per trial, coloured by the spread metric — where each trial sits.
  density  a filled KDE of the whole cloud plus one 50%-of-peak contour per TERTILE
           of the spread metric — whether the cloud SHIFTS as the bins disagree,
           without overplotting. Both KDEs are computed in log10 space to match the
           log-log axes. It also prints ``d = median log10(temp/spat)`` per mouse per
           tertile: movement ALONG the diagonal is just a harder trial, movement
           PERPENDICULAR (d) is the temporal decoder being specifically penalised.

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
from decoder_metrics import bin_divergence_from_mean      # noqa: E402
from projflat_report import _res, _mice, have             # noqa: E402

# The four headline configurations, same as the exemplar explorer.
import projflat_cells as pcells                            # noqa: E402

# The nine headline cells, from the one shared table. rr8 = reduced-rank: a hidden
# layer of 8 with NO non-linearity, so the logit map is affine of rank <= 8; rr8 vs
# lin isolates the rank bottleneck, rr8 vs h8 the tanh. Cells absent from disk are
# skipped by have(), so this is safe on a partial results tree.
CONFIGS = pcells.as_dict()

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
        # the canonical bins-vs-their-own-average TV; this was its first home and
        # it now lives in decoder_metrics beside the other row-wise distribution
        # metrics, so diagnostics/projflat_trial_explorer colours its per-mouse
        # scatter with the SAME number rather than a second copy of this line
        return bin_divergence_from_mean(samp)
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


def _scales(per_mouse, kind):
    """Colour norm + square axis limits shared by every mouse panel of a config,
    so panels (and the two figure modes) are directly comparable."""
    cmap, diverging, unit = COLOUR_META[kind]
    all_c = np.concatenate([c for _, _, c in per_mouse.values()])
    all_s = np.concatenate([s for s, _, _ in per_mouse.values()])
    all_t = np.concatenate([t for _, t, _ in per_mouse.values()])
    if diverging:
        vabs = float(np.nanpercentile(np.abs(all_c), 99))
        vmin, vmax = -vabs, vabs
        norm = matplotlib.colors.TwoSlopeNorm(0.0, vmin=vmin, vmax=vmax)
    else:
        vmin = float(np.nanpercentile(all_c, 1))
        vmax = float(np.nanpercentile(all_c, 99))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    pos = np.concatenate([all_s[all_s > 0], all_t[all_t > 0]])
    lim = [float(np.nanpercentile(pos, 0.5)) * 0.8,
           float(max(np.nanpercentile(all_s, 99.5), np.nanpercentile(all_t, 99.5))) * 1.25]
    return dict(cmap=cmap, unit=unit, norm=norm, vmin=vmin, vmax=vmax, lim=lim,
                all_c=all_c, n=all_s.size)


def _panel_furniture(ax, lim, m, s, t, nrow, ncol, k):
    """Diagonal, chance lines, square log-log axes, per-mouse title."""
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


def _grid(per_mouse):
    mice = list(per_mouse.keys())
    ncol = 3
    nrow = int(np.ceil(len(mice) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.4 * nrow),
                             squeeze=False)
    return fig, axes, mice, nrow, ncol


def plot_scatter(per_mouse, out_root, cell, kind):
    """Per-mouse scatter, one dot per trial, coloured by the bin-spread metric."""
    S = _scales(per_mouse, kind)
    lim = S['lim']
    ps.apply()
    fig, axes, mice, nrow, ncol = _grid(per_mouse)
    sc = None
    for k, m in enumerate(mice):
        ax = axes[k // ncol][k % ncol]
        s, t, c = per_mouse[m]
        ax.hexbin(s, t, gridsize=28, xscale='log', yscale='log', cmap='Greys',
                  mincnt=1, alpha=0.22, linewidths=0,
                  extent=(np.log10(lim[0]), np.log10(lim[1]),
                          np.log10(lim[0]), np.log10(lim[1])))
        sc = ax.scatter(s, t, c=c, cmap=S['cmap'], norm=S['norm'], s=10,
                        alpha=0.8, linewidths=0, zorder=3)
        _panel_furniture(ax, lim, m, s, t, nrow, ncol, k)
    for k in range(len(mice), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(S['unit'], fontsize=8)
    fig.suptitle(f"{CONFIGS.get(cell, cell)} — spatial vs temporal loss per trial, per mouse "
                 f"(square log-log, 1 = chance on each; below diagonal = temporal better).\n"
                 f"Colour = {S['unit']}", fontsize=10, y=1.035)
    stem = f'scatter_spat_temp_by_mouse_{cell}_{kind}'
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}: colour[{kind}] {S['vmin']:.3g}..{S['vmax']:.3g}, "
          f"{len(mice)} mice, n={S['n']}")


def plot_density(per_mouse, out_root, cell, kind, n_groups=3):
    """Per-mouse DENSITY view of the same data: a filled KDE of the overall trial
    cloud, plus one 50%-mass contour per tertile of the bin-spread metric.

    The scatter answers "where does each trial sit"; this answers "does the cloud
    SHIFT as the temporal bins disagree more" without overplotting. Both the KDE
    and the contours are computed in log10 space, matching the log-log axes —
    a linear-space KDE on log axes would distort the shapes.

    Tertile edges are global within a config (not per mouse), so a given contour
    colour means the same spread everywhere in the figure.
    """
    from scipy.stats import gaussian_kde
    S = _scales(per_mouse, kind)
    lim, cm = S['lim'], matplotlib.colormaps[S['cmap']]
    edges = np.nanpercentile(S['all_c'], np.linspace(0, 100, n_groups + 1))
    lo, hi = np.log10(lim[0]), np.log10(lim[1])
    gx, gy = np.mgrid[lo:hi:90j, lo:hi:90j]
    grid = np.vstack([gx.ravel(), gy.ravel()])

    ps.apply()
    fig, axes, mice, nrow, ncol = _grid(per_mouse)
    for k, m in enumerate(mice):
        ax = axes[k // ncol][k % ncol]
        s, t, c = per_mouse[m]
        good = (s > 0) & (t > 0) & np.isfinite(s) & np.isfinite(t) & np.isfinite(c)
        ls, lt, cc = np.log10(s[good]), np.log10(t[good]), c[good]
        # overall density, filled — the backdrop every contour is read against.
        if ls.size > 20 and np.ptp(ls) > 1e-9 and np.ptp(lt) > 1e-9:
            z = gaussian_kde(np.vstack([ls, lt]))(grid).reshape(gx.shape)
            ax.contourf(10 ** gx, 10 ** gy, z, levels=7, cmap='Greys',
                        alpha=0.30, zorder=1)
        # one 50%-of-peak contour per spread tertile.
        for g in range(n_groups):
            g_lo, g_hi = edges[g], edges[g + 1]
            mask = (cc >= g_lo) & (cc <= g_hi if g == n_groups - 1 else cc < g_hi)
            if mask.sum() < 15 or np.ptp(ls[mask]) < 1e-9 or np.ptp(lt[mask]) < 1e-9:
                continue
            try:
                zg = gaussian_kde(np.vstack([ls[mask], lt[mask]]))(grid).reshape(gx.shape)
            except Exception:
                continue
            col = cm(S['norm'](float(np.median(cc[mask]))))
            ax.contour(10 ** gx, 10 ** gy, zg, levels=[zg.max() * 0.5],
                       colors=[col], linewidths=2.0, zorder=3)
            if k == 0:                       # one legend, on the first panel
                ax.plot([], [], color=col, lw=2.0,
                        label=f'{g_lo:.2g}–{g_hi:.2g}')
        _panel_furniture(ax, lim, m, s, t, nrow, ncol, k)
        if k == 0:
            ax.legend(fontsize=6, frameon=True, title='spread tertile',
                      title_fontsize=6, loc='upper left')
    for k in range(len(mice), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    sm = matplotlib.cm.ScalarMappable(norm=S['norm'], cmap=S['cmap'])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(S['unit'], fontsize=8)
    fig.suptitle(f"{CONFIGS.get(cell, cell)} — DENSITY of the same per-trial cloud, per mouse "
                 f"(square log-log, 1 = chance on each; below diagonal = temporal better).\n"
                 f"Grey fill = all trials; coloured line = 50%-of-peak contour per tertile of "
                 f"{S['unit'].split(':')[0]}", fontsize=10, y=1.035)
    stem = f'density_spat_temp_by_mouse_{cell}_{kind}'
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}: {n_groups} tertiles, edges "
          + '/'.join(f'{e:.3g}' for e in edges))


def _shift_table(per_mouse, kind, n_groups=3):
    """Turn the contour shift into a number, per mouse.

    The density contours shift up-and-right with spread, but that is ambiguous:
    moving ALONG the diagonal just means "harder trial" (both decoders worse),
    while moving PERPENDICULAR to it means the temporal decoder is specifically
    penalised. The perpendicular coordinate is

        d = median log10(temporal loss / spatial loss)

    d > 0 = temporal worse. Reported per mouse per spread tertile, then summarised
    ACROSS THE 6 MICE (n=6, the non-pseudoreplicated unit) as mean ± SD of the
    within-mouse high-minus-low difference. Descriptive only — no test at n=6.

    CIRCULARITY CAVEAT: the grouping variable (bin spread) is computed from the
    TEMPORAL decoder's own output, and it also enters the outcome d through the
    temporal loss. So a d-vs-spread relationship is NOT an independent test of
    "does bin disagreement hurt the temporal decoder" — a broader Jensen average
    mechanically changes the temporal loss. Read these as descriptive, and use a
    target-side or stimulus-side grouping variable for the real test."""
    S = _scales(per_mouse, kind)
    edges = np.nanpercentile(S['all_c'], np.linspace(0, 100, n_groups + 1))
    rows, deltas = [], []
    for m, (s, t, c) in per_mouse.items():
        good = (s > 0) & (t > 0) & np.isfinite(s) & np.isfinite(t) & np.isfinite(c)
        ls, lt, cc = np.log10(s[good]), np.log10(t[good]), c[good]
        ds = []
        for g in range(n_groups):
            g_lo, g_hi = edges[g], edges[g + 1]
            mask = (cc >= g_lo) & (cc <= g_hi if g == n_groups - 1 else cc < g_hi)
            ds.append(float(np.median(lt[mask] - ls[mask])) if mask.sum() >= 10 else np.nan)
        rows.append((m, ds))
        if np.isfinite(ds[0]) and np.isfinite(ds[-1]):
            deltas.append(ds[-1] - ds[0])
    print('    d = median log10(temp/spat)  [>0 = temporal worse], by spread tertile:')
    for m, ds in rows:
        print(f'      {m}: ' + '  '.join(f'{d:+.3f}' if np.isfinite(d) else '  n/a'
                                         for d in ds))
    if deltas:
        d = np.array(deltas)
        print(f'    high-minus-low tertile: mean {d.mean():+.3f} ± {d.std(ddof=1):.3f} SD '
              f'over {d.size} mice, {int((d > 0).sum())}/{d.size} positive')


def run_cell(results_root, out_root, cell, kind, mode):
    """Score a config once, then draw whichever views were asked for."""
    if not have(results_root, cell):
        print(f'  [skip] {cell}')
        return
    per_mouse = gather(results_root, cell, kind)
    if mode in ('scatter', 'both'):
        plot_scatter(per_mouse, out_root, cell, kind)
    if mode in ('density', 'both'):
        plot_density(per_mouse, out_root, cell, kind)
        _shift_table(per_mouse, kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/scatter_spat_temp_by_mouse')
    ap.add_argument('--config', default='lin_raw_l0_d0_w0',
                    help="cell key, or 'all' for the four headline configs")
    ap.add_argument('--colour', default='jsd', choices=list(COLOUR_META),
                    help="jsd (bits, default) | l1 (total variation) | gain (loss improvement)")
    ap.add_argument('--mode', default='both', choices=['scatter', 'density', 'both'],
                    help="which view(s) to draw (default both)")
    a = ap.parse_args()
    cells = list(CONFIGS) if a.config == 'all' else [a.config]
    print(f'Per-mouse spat/temp {a.mode} | colour={a.colour} | configs={cells}')
    for cell in cells:
        run_cell(a.results_root, a.out_root, cell, a.colour, a.mode)
    print(f'Done. {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
