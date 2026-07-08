# -*- coding: utf-8 -*-
"""Real-V1 subspace decomposition of the decoded−target error — the toy fig 7
mechanism, now on real data.

The toy model (PCA-Peakiness-Mechanism §5, fig 7) proves the mechanism by
splitting the decoded−target error by principal component: all losses match the
high-evar LOCATION subspace, but PCA's error is ~300× KL's in the low-evar SHAPE
subspace (the loss-blind directions it fills with spiky junk). This is the real-
data analogue on `loss_comparison_v1` (Q/half/100 ms, spatial decoder = loss
alone, 6 mice): for each loss we project each trial's decoded−target difference
onto the target PCA basis and read the mean-squared error per PC, then split into
location (leading PCs to 90% cumulative evar) vs shape (the rest).

The basis is the per-mouse target PCA basis (identical across losses — same IO
targets — so one common basis from the PCA cell; per-PC error is sign-invariant,
being a squared projection).

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  subspace_error_realdata.png

Usage:  python diagnostics/subspace_error_realdata.py
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

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _load(results_root, run, slug, split):
    f = Path(results_root) / run / slug / f'{split}.mat'
    if not f.is_file():
        return None
    return sio.loadmat(str(f), simplify_cells=True).get('results')


def collect(results_root, run, split, arch='spat', basis_loss='PCA'):
    """errs[loss] = (n_mice, K) per-PC mean-sq decoded−target proj error;
    evar (K,) the mean target spectrum; kloc = location-subspace size (≤90% evar)."""
    bres = _load(results_root, run, _slug(basis_loss), split)
    if not isinstance(bres, dict):
        raise SystemExit(f'basis cell {basis_loss} missing under {run}/. rsync first.')
    mice = sorted(bres)
    basis = {mk: (np.asarray(bres[mk]['Dist']['pcs'], float),
                  np.asarray(bres[mk]['Dist']['explained_var'], float)) for mk in mice}
    errs = {}
    for loss in LOSSES:
        res = _load(results_root, run, _slug(loss), split)
        if not isinstance(res, dict):
            print(f'  [skip] {loss}: cell not on disk')
            continue
        per_mouse = []
        for mk in mice:
            if mk not in res:
                continue
            pcs, _ = basis[mk]
            D = res[mk]['Dist'][arch]
            diff = np.asarray(D['decoded'], float) - np.asarray(D['target'], float)
            proj = diff @ pcs.T                      # (n_trials, K)
            per_mouse.append((proj ** 2).mean(0))    # (K,) mean over trials
        if per_mouse:
            errs[loss] = np.stack(per_mouse)
    evar = np.mean([basis[mk][1] for mk in mice], axis=0)
    kloc = int(np.searchsorted(np.cumsum(evar) / evar.sum(), 0.90)) + 1
    return errs, evar, kloc, len(mice)


def main(results_root, run, split, out_root, weight='none'):
    ps.apply()
    errs, evar, kloc, n = collect(results_root, run, split)
    if not errs:
        raise SystemExit('no loss cells found.')
    K = len(evar)
    # weight='evar' → plot the per-PC LOSS contribution evar_k·e_k (what the PCA loss
    # actually weights), instead of the raw error. The trailing (shape) PCs have
    # evar≈0, so PCA's large shape error collapses to ≈the others — a direct picture
    # of the loss being blind to the shape subspace it fills with junk (2026-06-18 #2).
    weighted = (weight == 'evar')
    if weighted:
        errs = {l: errs[l] * evar[None, :] for l in errs}
    err_ylab = ('per-PC loss contribution  (evar × error)' if weighted
                else 'mean-sq decoded−target projection error')
    tot_ylab = 'total loss contribution' if weighted else 'total projection error'
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))

    # (a) per-PC error, log-y, location subspace shaded
    ax = axes[0]
    for loss in LOSSES:
        if loss not in errs:
            continue
        ax.plot(np.arange(K), errs[loss].mean(0), color=LCOL[loss], lw=2, label=ps.loss_label(loss))
    ax.axvspan(-0.5, kloc - 0.5, color='0.85', alpha=0.5, zorder=0)
    ax.set_yscale('log')
    ax.set_xlabel('principal component k   (shaded = location)')
    ax.set_ylabel(err_ylab)
    ax.set_title(f'Per-PC {"loss contribution" if weighted else "error"} (location ≤ PC{kloc})')
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    # (b) location vs shape total error per loss
    ax = axes[1]
    present = [l for l in LOSSES if l in errs]
    x = np.arange(len(present))
    loc = {l: errs[l][:, :kloc].sum(1) for l in present}     # per-mouse totals
    shp = {l: errs[l][:, kloc:].sum(1) for l in present}
    def _ms(v):
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    for j, l in enumerate(present):
        lm, ls = _ms(loc[l]); sm, ss = _ms(shp[l])
        ax.bar(j - 0.2, lm, 0.38, yerr=ls, color=LCOL[l], alpha=0.45, capsize=3,
               label='location' if j == 0 else None)
        ax.bar(j + 0.2, sm, 0.38, yerr=ss, color=LCOL[l], alpha=0.9, hatch='//', capsize=3,
               label='shape' if j == 0 else None)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(ps.loss_labels(present), rotation=15, fontsize=8)
    ax.set_ylabel(tot_ylab)
    ax.set_title('Location vs shape' + (' loss contribution' if weighted else ' error'))
    ax.legend(frameon=False, fontsize=8, loc='best')

    ps.label_panels(axes)
    fig.suptitle(f'Real V1: decoded−target {"loss contribution" if weighted else "error"} by PC subspace'
                 f'{" — evar-weighted (what the loss sees)" if weighted else ""} (spatial, {n} mice)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'subspace_error_realdata' + ('_evarweighted' if weighted else ''))

    # numeric readout (the real-data analogue of the toy's 300× / 67×)
    print(f'\nlocation subspace = PC0..{kloc - 1} (≤90% evar) of {K} PCs')
    print(f'{"loss":12s} {"location":>12s} {"shape":>12s} {"shape/loc":>10s}')
    for l in present:
        lo, sh = loc[l].mean(), shp[l].mean()
        print(f'  {l:10s} {lo:12.3e} {sh:12.3e} {sh/lo:10.2f}')
    if 'PCA' in errs and 'KL' in errs:
        print(f'\n  PCA/KL shape-subspace error ratio : {shp["PCA"].mean()/shp["KL"].mean():.1f}x')
        print(f'  PCA/KL location-subspace ratio    : {loc["PCA"].mean()/loc["KL"].mean():.2f}x  (should be ~1)')
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    ap.add_argument('--weight', choices=['none', 'evar'], default='none',
                    help="evar → plot the per-PC LOSS contribution (evar × error) instead of raw "
                         "error: shows the PCA loss is blind to the shape subspace (2026-06-18 #2). "
                         "Default 'none' leaves the published Fig 9 unchanged.")
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root, weight=a.weight)
