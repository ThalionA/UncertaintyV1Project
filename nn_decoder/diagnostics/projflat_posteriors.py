# -*- coding: utf-8 -*-
"""Item #4 of the 2026-07-29 meeting, on the CLEAN projflat_v1 cells: example
fitted posteriors, individual temporal-bin posteriors as heatmaps, and a
spatial-vs-temporal per-trial scatter. Trials selected by how much the posterior
changes across time bins (the "select trials by diff from temp bins" ask).

Decoders shown (H=8, raw input, patience 20):
  evar (over-sharpened) · flat/MSE (the working one) · KL reference.

  fig1  overlay      — IO target + spatial + temporal decode, per decoder x trial.
  fig2  temporal bins — decoded_samp (91 x 10) heatmaps: what the Jensen average
                        averages over.
  fig3  scatter      — per-trial spatial vs temporal loss (KL AND projection),
                        coloured by the trial's across-bin dynamism.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_posteriors.py [--mouse mouse_0] [--n-trials 4]
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flatevar_posteriors import _load, _pick_trials          # noqa: E402  (pure loaders)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial                 # noqa: E402
import torch                                                  # noqa: E402

RUN = 'projflat_v1'
SHOW = [
    ('evar (over-sharp)', 'h8_raw_EVAR',     ps.PCA_EVAR),
    ('flat/MSE (working)', 'h8_raw_l0_d0_w0', ps.FLAT_EVAR),
    ('KL reference',       'h8_raw_KLref',    ps.KL),
]
REF_FOR_SELECTION = (RUN, 'h8_raw_l0_d0_w0')
THETA = np.linspace(0, 90, 91)


def _evar_basis(results_root, mouse):
    """(pcs, evar) from the projflat evar cell — the common projection weighting."""
    r = _load(results_root, (RUN, 'h8_raw_EVAR'))
    D = r[mouse]['Dist']
    return D.get('pcs'), D.get('explained_var')


def fig_overlay(results_root, out_root, mouse, trials, cells):
    ps.apply()
    nrow, ncol = len(cells), len(trials)
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow),
                             sharex=True, squeeze=False)
    for ri, (lab, res, colr) in enumerate(cells):
        D = res[mouse]['Dist']
        for ci, t in enumerate(trials):
            ax = axes[ri][ci]
            tgt = np.asarray(D['spat']['target'], float)[t]
            ax.fill_between(THETA, 0, tgt, color='0.75', lw=0,
                            label='IO target' if (ri == 0 and ci == 0) else None)
            ax.plot(THETA, np.asarray(D['spat']['decoded'], float)[t], lw=1.4,
                    color=ps.SPATIAL, label='spatial' if (ri == 0 and ci == 0) else None)
            ax.plot(THETA, np.asarray(D['temp']['decoded'], float)[t], lw=1.4,
                    color=ps.TEMPORAL, label='temporal' if (ri == 0 and ci == 0) else None)
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f'trial {t}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            if ri == nrow - 1:
                ax.set_xlabel('orientation (deg)', fontsize=7.5)
    axes[0][0].legend(fontsize=6, frameon=True, loc='upper right')
    ps.label_panels(axes.ravel())
    fig.suptitle(f'projflat_v1 fitted posteriors, {mouse} — same trials for every decoder, chosen as the '
                 'most temporally dynamic. Free y-scales.', y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'projflat_fig5_example_posteriors')


def fig_temporal_bins(results_root, out_root, mouse, trials, cells):
    ps.apply()
    nrow, ncol = len(cells), len(trials)
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow), squeeze=False)
    for ri, (lab, res, colr) in enumerate(cells):
        D = res[mouse]['Dist']
        samp = np.asarray(D['temp']['decoded_samp'], float)
        tgt_all = np.asarray(D['temp']['target'], float)
        vmax = np.nanpercentile(samp[trials], 99.5)
        for ci, t in enumerate(trials):
            ax = axes[ri][ci]
            bins = samp[t]                                     # (91, n_bins)
            im = ax.imshow(bins, aspect='auto', origin='lower', cmap='magma',
                           vmin=0, vmax=vmax, extent=[0, bins.shape[1], 0, 90])
            tp = tgt_all[t]
            ax.plot(bins.shape[1] * tp / max(tp.max(), 1e-12) * 0.28,
                    np.linspace(0, 90, tp.size), color='w', lw=1.1, alpha=0.9,
                    label='IO target' if (ri == 0 and ci == 0) else None)
            ax.set_xticks(np.arange(0, bins.shape[1] + 1, 3))
            if ri == 0:
                ax.set_title(f'trial {t}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{lab}\norientation', fontsize=7)
            else:
                ax.set_yticklabels([])
            if ri == nrow - 1:
                ax.set_xlabel('time bin', fontsize=7.5)
            if ci == ncol - 1:
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=5)
    axes[0][0].legend(fontsize=6, frameon=True, loc='upper right')
    ps.label_panels(axes.ravel())
    fig.suptitle(f'projflat_v1 individual temporal-bin posteriors, {mouse} — what the Jensen average averages '
                 'over. White = IO target (scaled). Colour shared within a row.', y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'projflat_fig6_temporal_bins')


def _dynamism(D, arch='temp'):
    """Per-trial mean total-variation distance between consecutive time bins."""
    samp = np.asarray(D[arch]['decoded_samp'], float)         # (trials, 91, bins)
    if samp.ndim != 3 or samp.shape[2] < 2:
        return None
    tv = 0.5 * np.abs(np.diff(samp, axis=2)).sum(axis=1)       # (trials, bins-1)
    return np.nanmean(tv, axis=1)


def fig_scatter(results_root, out_root, mouse):
    """Per-trial spatial vs temporal loss, under KL and projection, for the flat
    decoder — coloured by the trial's across-bin dynamism."""
    ps.apply()
    r = _load(results_root, (RUN, 'h8_raw_l0_d0_w0'))
    D = r[mouse]['Dist']
    pcs, evar = _evar_basis(results_root, mouse)
    dyn = _dynamism(D)
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    for ax, metric, mlab in [(axes[0], 'KL', 'KL(decoded‖target)'),
                             (axes[1], 'PCA', 'projection loss (common evar)')]:
        sp = _per_trial(D, 'spat', metric, pcs, evar)
        te = _per_trial(D, 'temp', metric, pcs, evar)
        ok = np.isfinite(sp) & np.isfinite(te) & (np.isfinite(dyn) if dyn is not None else True)
        c = dyn[ok] if dyn is not None else None
        sc = ax.scatter(sp[ok], te[ok], c=c, cmap='viridis', s=14, alpha=0.8,
                        edgecolor='none')
        lo = float(min(sp[ok].min(), te[ok].min()))
        hi = float(max(sp[ok].max(), te[ok].max()))
        ax.plot([lo, hi], [lo, hi], color='0.4', ls=':', lw=1.4, zorder=0)
        ax.set_xlabel(f'spatial  {mlab}', fontsize=8)
        ax.set_ylabel(f'temporal  {mlab}', fontsize=8)
        frac_temp = float((te[ok] < sp[ok]).mean())
        ax.set_title(f'{metric}: temporal better in {frac_temp:.0%} of trials', fontsize=9)
        if c is not None:
            fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03).set_label(
                'across-bin dynamism', fontsize=7)
    ps.label_panels(axes)
    fig.suptitle(f'projflat_v1 flat/MSE decoder, {mouse}: per-trial spatial vs temporal loss (below the '
                 'diagonal = temporal better). Colour = how much the posterior moves across time bins.',
                 y=1.02, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'projflat_fig7_spat_temp_scatter')


def _per_trial(D, arch, metric, pcs, evar):
    dec = torch.tensor(np.asarray(D[arch]['decoded'], float))
    tgt = torch.tensor(np.asarray(D[arch]['target'], float))
    pt = pcs if metric == 'PCA' else None
    ev = evar if metric == 'PCA' else None
    if pt is not None:
        pt = torch.tensor(np.asarray(pt, float))
        ev = torch.tensor(np.asarray(ev, float))
    return fit_loss_per_trial(dec, tgt, metric, pt, ev).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--mouse', default=None)
    ap.add_argument('--n-trials', type=int, default=4)
    a = ap.parse_args()

    cells = []
    for lab, cell, colr in SHOW:
        try:
            cells.append((lab, _load(a.results_root, (RUN, cell)), colr))
        except Exception:                                     # noqa: BLE001
            print(f"  [skip] {lab}: {cell} not found")
    if not cells:
        raise SystemExit('no projflat cells available')

    ref = _load(a.results_root, REF_FOR_SELECTION)
    mice = sorted(k for k in ref if isinstance(ref[k], dict)
                  and isinstance(ref[k].get('Dist'), dict))
    for _, r, _c in cells:
        mice = [m for m in mice if m in r]
    mouse = a.mouse or mice[0]

    trials, score = _pick_trials(ref, mouse, a.n_trials)
    print(f"  {mouse}: trials {list(trials)} (dynamism {np.round(score[trials], 3)}; "
          f"median {np.nanmedian(score):.3f})")
    fig_overlay(a.results_root, a.out_root, mouse, trials, cells)
    fig_temporal_bins(a.results_root, a.out_root, mouse, trials, cells)
    fig_scatter(a.results_root, a.out_root, mouse)
    print("  wrote projflat_fig5_example_posteriors, _fig6_temporal_bins, _fig7_spat_temp_scatter")


if __name__ == '__main__':
    main()
