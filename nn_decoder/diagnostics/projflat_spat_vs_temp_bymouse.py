# -*- coding: utf-8 -*-
"""Spatial vs temporal performance under the PROJECTION LOSS, scored under each
decoder's OWN training weighting (2026-08-04, Theo's request).

Matched hyperparameters throughout (raw input, lambda_H 0, dropout 0, wd 0):
  linear (0 hidden)  x  variance-weighting   ->  lin_raw_EVAR
  linear (0 hidden)  x  flat-weighting        ->  lin_raw_l0_d0_w0
  8 hidden units     x  variance-weighting   ->  h8_raw_EVAR
  8 hidden units     x  flat-weighting        ->  h8_raw_l0_d0_w0

Two deliverables:
  (A) per-mouse, one figure per config: two bars/mouse (spatial, temporal), height
      = mean normalised projection loss (per-trial projection distance / the
      mouse's leave-one-out predict-mean; < 1 beats chance), SEM over trials, star
      = WITHIN-mouse paired t (paired by trial; spat & temp decode the same
      trials, verified identical row-for-row).
  (B) across-mice, ALL configs in ONE figure: two bars/config (spatial, temporal),
      height = mean over the 6 mice, SEM over mice, star = paired t OVER THE 6 MICE
      (the population test — the legitimate, non-pseudoreplicated one).

WEIGHTING: each cell is scored under ITS OWN stored projection weighting —
eigenvalue-weighted for the variance-weighting (evar) cells, uniform = MSE for the
flat cells. Consequence: the metric DIFFERS between flat and evar configs, so
compare spatial vs temporal WITHIN a config, not bar heights ACROSS configs.
(`--weighting common` rescoress everything under one evar weighting instead, which
makes configs comparable but is a different question.)

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_spat_vs_temp_bymouse.py            # own weighting
        python diagnostics/projflat_spat_vs_temp_bymouse.py --weighting common
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from projflat_report import _res, _mice, have, _common_basis  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial  # noqa: E402

CONFIGS = [
    ('linear (0 hidden)\nvariance-weighting', 'lin_raw_EVAR',     'lin_evar'),
    ('linear (0 hidden)\nflat-weighting',     'lin_raw_l0_d0_w0', 'lin_flat'),
    ('8 hidden units\nvariance-weighting',    'h8_raw_EVAR',      'h8_evar'),
    ('8 hidden units\nflat-weighting',        'h8_raw_l0_d0_w0',  'h8_flat'),
]


def _t(x):
    return torch.tensor(np.asarray(x, float))


def _pt(D, arch, pcs, evar):
    return fit_loss_per_trial(_t(D[arch]['decoded']), _t(D[arch]['target']),
                              'PCA', _t(pcs), _t(evar)).numpy()


def _chance(D, pcs, evar):
    tgt = np.asarray(D['spat']['target'], float)     # spat & temp share the target
    ok = np.isfinite(tgt).all(1)
    n = int(ok.sum())
    tot = tgt[ok].sum(0)
    pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
    if n > 1:
        pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)
    return float(np.nanmean(fit_loss_per_trial(_t(pm), _t(tgt), 'PCA',
                                               _t(pcs), _t(evar)).numpy()))


def per_mouse(results_root, cell, weighting):
    """{mouse -> (spatial_norm_pertrial, temporal_norm_pertrial)}."""
    r = _res(results_root, cell)
    common = _common_basis(results_root, 'spat') if weighting == 'common' else None
    out = {}
    for m in _mice(r):
        D = r[m]['Dist']
        if weighting == 'own':
            pcs, evar = D.get('pcs'), D.get('explained_var')
        else:
            if m not in common:
                continue
            pcs, evar = common[m]
        sp, te = _pt(D, 'spat', pcs, evar), _pt(D, 'temp', pcs, evar)
        ok = np.isfinite(sp) & np.isfinite(te)
        ch = _chance(D, pcs, evar)
        out[m] = (sp[ok] / ch, te[ok] / ch)
    return out


def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'


# ---------------------------------------------------------- (A) per-mouse figure
def fig_per_mouse(results_root, out_root, title, cell, short, weighting):
    if not have(results_root, cell):
        print(f"  [skip] {short}: {cell} not downloaded")
        return
    ps.apply()
    pm = per_mouse(results_root, cell, weighting)
    mice = sorted(pm)
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(mice))
    w = 0.38
    for xi, m in zip(x, mice):
        sp, te = pm[m]
        _, p = ttest_rel(sp, te)
        for v, off, colr, lab in [(sp, -w / 2, ps.SPATIAL, 'spatial'),
                                  (te, +w / 2, ps.TEMPORAL, 'temporal')]:
            ax.bar(xi + off, v.mean(), w, yerr=v.std(ddof=1) / np.sqrt(v.size),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=lab if xi == 0 else None)
        ax.text(xi, max(sp.mean(), te.mean()) * 1.04 + 0.02, _stars(p),
                ha='center', fontsize=8)
    ax.axhline(1.0, color='k', ls=':', lw=1.4, label='chance')
    ax.set_xticks(x); ax.set_xticklabels([m.replace('mouse_', 'M') for m in mice])
    metric = 'MSE' if 'flat' in short else 'evar-weighted projection'
    ax.set_ylabel(f'normalised {metric} loss\n(/ predict-mean; < 1 beats chance)', fontsize=8)
    ax.set_xlabel('mouse')
    ax.legend(fontsize=7, frameon=True)
    wtag = 'own training weighting' if weighting == 'own' else 'common evar weighting'
    ax.set_title(f'{title.replace(chr(10), ", ")} — projection loss ({wtag}).  '
                 f'Star = within-mouse paired t (n = trials).', fontsize=8)
    fig.tight_layout()
    stem = f'projflat_spmouse_{short}' + ('_own' if weighting == 'own' else '')
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}")


# ------------------------------------------------------ (B) across-mice figure
def fig_across_mice(results_root, out_root, weighting):
    ps.apply()
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(CONFIGS))
    w = 0.38
    for xi, (title, cell, short) in zip(x, CONFIGS):
        if not have(results_root, cell):
            continue
        pm = per_mouse(results_root, cell, weighting)
        mice = sorted(pm)
        sp = np.array([pm[m][0].mean() for m in mice])       # per-mouse means
        te = np.array([pm[m][1].mean() for m in mice])
        _, p = ttest_rel(sp, te)                              # PAIRED OVER MICE (n=6)
        n_temp = int((te < sp).sum())
        for v, off, colr, lab in [(sp, -w / 2, ps.SPATIAL, 'spatial'),
                                  (te, +w / 2, ps.TEMPORAL, 'temporal')]:
            ax.bar(xi + off, v.mean(), w, yerr=v.std(ddof=1) / np.sqrt(v.size),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=lab if xi == 0 else None)
        # per-mouse points overlaid (paired), so the n=6 is visible
        for a_, off in [(sp, -w / 2), (te, +w / 2)]:
            ax.plot(np.full_like(a_, xi + off), a_, 'o', ms=2.5, color='0.25',
                    alpha=0.6, zorder=4)
        top = max(sp.mean(), te.mean())
        ax.text(xi, top * 1.04 + 0.03, f'{_stars(p)}\n{n_temp}/{len(mice)}',
                ha='center', fontsize=7)
    ax.axhline(1.0, color='k', ls=':', lw=1.4, label='chance')
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=7.5)
    ax.set_ylabel('normalised projection loss\n(/ predict-mean; < 1 beats chance)', fontsize=8)
    ax.legend(fontsize=7, frameon=True)
    wtag = 'own training weighting' if weighting == 'own' else 'common evar weighting'
    ax.set_title(f'Spatial vs temporal ACROSS MICE (paired t over n=6), projection loss '
                 f'({wtag}). Star = paired t; n/6 = mice where temporal wins.\n'
                 f'Own weighting: flat configs scored as MSE, evar configs eigenvalue-weighted '
                 f'— compare spatial-vs-temporal WITHIN a config.', fontsize=7.3)
    fig.tight_layout()
    stem = 'projflat_spat_vs_temp_acrossmice' + ('_own' if weighting == 'own' else '_common')
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--weighting', choices=['own', 'common'], default='own')
    a = ap.parse_args()
    print(f"Spatial vs temporal, projection loss, weighting={a.weighting}\n")
    for title, cell, short in CONFIGS:
        fig_per_mouse(a.results_root, a.out_root, title, cell, short, a.weighting)
    fig_across_mice(a.results_root, a.out_root, a.weighting)


if __name__ == '__main__':
    main()
