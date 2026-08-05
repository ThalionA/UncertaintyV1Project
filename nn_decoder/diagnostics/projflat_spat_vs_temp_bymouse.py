# -*- coding: utf-8 -*-
"""Per-mouse spatial vs temporal performance under the PROJECTION LOSS ONLY, with
a within-mouse paired t-test (2026-08-04, Theo's request).

Four SEPARATE figures, matched hyperparameters throughout (raw input, lambda_H 0,
dropout 0, wd 0 — the shared un-regularised corner):

  linear (0 hidden)  x  variance-weighting   ->  lin_raw_EVAR
  linear (0 hidden)  x  flat-weighting        ->  lin_raw_l0_d0_w0
  8 hidden units     x  variance-weighting   ->  h8_raw_EVAR
  8 hidden units     x  flat-weighting        ->  h8_raw_l0_d0_w0

Each figure: one mouse = two bars (spatial, temporal), height = mean normalised
projection loss (per-trial projection distance / the mouse's leave-one-out
predict-mean; < 1 beats chance), error bars = SEM OVER TRIALS. The star is a
paired t-test between spatial and temporal PER MOUSE, paired by trial (spatial and
temporal decode the same test trials — verified identical targets row-for-row).

Scoring is the canonical projection loss under a COMMON evar weighting (from the
evar cell), so the metric is identical across all four figures and only the
TRAINING weighting/architecture differs. (Scoring a flat-trained decoder under its
own uniform weights would be a different metric — not comparable across figures.)

STATS CAVEAT (on every figure): the per-mouse test is n = that mouse's trials, so
it answers "is spatial vs temporal reliable WITHIN this animal" — it is NOT
population evidence, and with hundreds of trials even a tiny difference stars.
Read the BAR HEIGHTS (effect size) first; the population claim is the n=6 paired
test in projflat_fig8. And note: the projection metric is BLIND to over-sharpening,
so spatial and temporal look similar here by design — that similarity is the point.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_spat_vs_temp_bymouse.py
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from projflat_report import _res, _mice, have, _common_basis  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial  # noqa: E402

CONFIGS = [
    ('linear (0 hidden), variance-weighting', 'lin_raw_EVAR',     'projflat_spmouse_lin_evar'),
    ('linear (0 hidden), flat-weighting',     'lin_raw_l0_d0_w0', 'projflat_spmouse_lin_flat'),
    ('8 hidden units, variance-weighting',    'h8_raw_EVAR',      'projflat_spmouse_h8_evar'),
    ('8 hidden units, flat-weighting',        'h8_raw_l0_d0_w0',  'projflat_spmouse_h8_flat'),
]


def _t(x):
    return torch.tensor(np.asarray(x, float))


def _per_trial_proj(D, arch, pcs, evar):
    """Per-trial projection loss under the common (pcs, evar) weighting."""
    return fit_loss_per_trial(_t(D[arch]['decoded']), _t(D[arch]['target']),
                              'PCA', _t(pcs), _t(evar)).numpy()


def _chance(D, arch, pcs, evar):
    """Mean per-trial leave-one-out predict-mean projection loss (the /chance scalar)."""
    tgt = np.asarray(D[arch]['target'], float)
    ok = np.isfinite(tgt).all(1)
    n = int(ok.sum())
    tot = tgt[ok].sum(0)
    pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
    if n > 1:
        pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)
    return float(np.nanmean(fit_loss_per_trial(_t(pm), _t(tgt), 'PCA',
                                               _t(pcs), _t(evar)).numpy()))


def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'


def make_figure(results_root, out_root, title, cell, stem):
    if not have(results_root, cell):
        print(f"  [skip] {stem}: {cell} not downloaded")
        return
    ps.apply()
    r = _res(results_root, cell)
    basis = _common_basis(results_root, 'spat')     # (pcs, evar) per mouse, common metric
    mice = [m for m in _mice(r) if m in basis]
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(mice))
    w = 0.38
    rows = []
    for xi, m in zip(x, mice):
        D = r[m]['Dist']
        pcs, evar = basis[m]
        sp = _per_trial_proj(D, 'spat', pcs, evar)
        te = _per_trial_proj(D, 'temp', pcs, evar)
        ok = np.isfinite(sp) & np.isfinite(te)
        sp, te = sp[ok], te[ok]
        ch = _chance(D, 'spat', pcs, evar)          # spat & temp share the target
        spn, ten = sp / ch, te / ch                 # normalise -> /chance (per-mouse const)
        tstat, p = ttest_rel(sp, te)                # paired over trials (unaffected by scaling)
        for arch, v, off, colr in [('spat', spn, -w / 2, ps.SPATIAL),
                                   ('temp', ten, +w / 2, ps.TEMPORAL)]:
            ax.bar(xi + off, v.mean(), w, yerr=v.std(ddof=1) / np.sqrt(v.size),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=('spatial' if arch == 'spat' else 'temporal') if xi == 0 else None)
        top = max(spn.mean(), ten.mean())
        ax.text(xi, top * 1.05 + 0.02, _stars(p), ha='center', fontsize=8)
        rows.append((m, spn.mean(), ten.mean(), int(sp.size), p))
    ax.axhline(1.0, color='k', ls=':', lw=1.4, label='chance')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('mouse_', 'M') for m in mice])
    ax.set_ylabel('normalised projection loss\n(/ predict-mean; < 1 beats chance)', fontsize=8)
    ax.set_xlabel('mouse')
    ax.legend(fontsize=7, frameon=True)
    ax.set_title(f'{title}\nspatial vs temporal, projection loss only. Star = within-mouse '
                 f'paired t (n = trials). * .05  ** .01  *** .001', fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}: " + "  ".join(
        f"{m.replace('mouse_','M')} s{sm:.2f}/t{tm:.2f} p={p:.1e}" for m, sm, tm, _n, p in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()
    print("Per-mouse spatial vs temporal, PROJECTION LOSS only, within-mouse paired t\n")
    for title, cell, stem in CONFIGS:
        make_figure(a.results_root, a.out_root, title, cell, stem)


if __name__ == '__main__':
    main()
