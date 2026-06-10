# -*- coding: utf-8 -*-
"""evar^α sweep: the *soft* (multiplicative) cousin of the λ·Brier floor (2026-06-10).

The width-matched fix (§7 of the peakiness note) floors the per-PC weights
additively, evar_k → evar_k + λ. This script tests the alternative one-knob
compression evar_k → evar_k**α (renormalised to sum 1): α=1 is plain PCA, α→0
recovers flat-evar, and α<1 lifts the trailing width PCs off ≈0 *multiplicatively*
while keeping the leading location PCs ranked highest. The question is whether
some α lands decoded peakiness on the IO target while beating chance under KL —
i.e. whether the soft knob matches the additive λ=0.1 sweet spot.

Same scoring as `lambda_sweep.py`, on the COMMON true-evar PCA basis (the evar
run's per-mouse pcs/evar) so every variant is directly comparable:
  - decoded peakiness (max-probability) vs the IO target;
  - KL-skill = held-out KL / shuffle KL  (<1 beats chance) — sees the over-confidence;
  - PCA-skill — the training metric, blind across the sweep.
The additive λ=0.1 fix (`wm3_shape10`) is overlaid as a dashed benchmark so the
two fix families can be read off against each other.

Output (PNG+SVG) under figures/loss_variants/:
  evar_alpha_sweep.png

Usage:  python diagnostics/evar_alpha_sweep.py
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
from cross_loss_eval import _eval_one  # noqa: E402

SLUG = 'Q_PCA_half_100ms_all'
# α ladder: evar(α=1) → 0.5 → 0.3 → 0.15 → flat(α=0)
LADDER = [('evar\n(α=1)', 'wm3'), ('α=0.5', 'wm3_alpha0p5'), ('α=0.3', 'wm3_alpha0p3'),
          ('α=0.15', 'wm3_alpha0p15'), ('flat\n(α=0)', 'wm3_flatevar')]
BENCHMARK = ('λ=0.1 (additive)', 'wm3_shape10')   # the §7 sweet spot, for reference
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]


def _load(results_root, run, split):
    f = Path(results_root) / run / SLUG / f'{split}.mat'
    return sio.loadmat(str(f), simplify_cells=True)['results'] if f.is_file() else None


def _variant_stats(res, basis, mice):
    """Per-arch per-mouse maxP / PCA-skill / KL-skill for one loaded run."""
    out = {a: {'maxP': [], 'PCA': [], 'KL': []} for a, _ in ARCHS}
    for mk in mice:
        if mk not in res:
            continue
        pcs, ev = basis[mk]
        for a, _ in ARCHS:
            D = res[mk]['Dist']
            dec = np.asarray(D[a]['decoded'], float)
            tgt = np.asarray(D[a]['target'], float)
            shf = np.asarray(D[a + '_shf']['decoded'], float)
            shf_t = np.asarray(D[a + '_shf']['target'], float)
            out[a]['maxP'].append(dec.max(1).mean())
            for met in ('PCA', 'KL'):
                real = _eval_one(dec, tgt, met, pcs, ev)
                sh = _eval_one(shf, shf_t, met, pcs, ev)
                out[a][met].append(real / sh if sh else np.nan)
    return out


def collect(results_root, split):
    evar = _load(results_root, 'wm3', split)
    basis = {mk: (np.asarray(m['Dist']['pcs'], float),
                  np.asarray(m['Dist']['explained_var'], float))
             for mk, m in evar.items()}
    mice = sorted(evar.keys())
    data = {}
    for lab, run in LADDER:
        res = _load(results_root, run, split)
        if res is not None:
            data[lab] = _variant_stats(res, basis, mice)
    bench = None
    bench_res = _load(results_root, BENCHMARK[1], split)
    if bench_res is not None:
        bench = _variant_stats(bench_res, basis, mice)
    tgt_mp = float(np.mean([np.asarray(evar[mk]['Dist']['spat']['target'], float).max(1).mean()
                            for mk in mice]))
    return data, bench, mice, tgt_mp


def main(results_root, split, out_root):
    ps.apply()
    data, bench, mice, tgt_mp = collect(results_root, split)
    labs = [l for l, _ in LADDER if l in data]
    x = np.arange(len(labs))

    fig, axes = plt.subplots(2, 3, figsize=ps.figsize(3, 2, panel_w=3.6, panel_h=3.0))
    quantities = [('maxP', 'decoded max-probability', tgt_mp, 'IO target'),
                  ('KL', 'KL-skill  (loss / shuffle)', 1.0, 'chance'),
                  ('PCA', 'PCA-skill  (loss / shuffle)', 1.0, 'chance')]

    def _bench_line(ax, q):
        """Overlay the additive λ=0.1 benchmark (spatial+temporal mean) as a band."""
        if bench is None:
            return
        vals = [np.nanmean(bench[a][q]) for a, _ in ARCHS]
        ax.axhspan(min(vals), max(vals), color=ps.SHAPE, alpha=0.13, zorder=0)
        ax.axhline(np.mean(vals), ls='-.', lw=1.2, color=ps.SHAPE, zorder=1,
                   label=BENCHMARK[0])

    # Row 0 — ACROSS mice (mean ± s.e.m.)
    for c, (q, ylab, ref, reflab) in enumerate(quantities):
        ax = axes[0, c]
        _bench_line(ax, q)
        for a, alab in ARCHS:
            M = np.array([data[l][a][q] for l in labs], float)
            m = np.nanmean(M, 1); se = np.nanstd(M, 1, ddof=1) / np.sqrt(M.shape[1])
            ax.errorbar(x, m, yerr=se, color=ps.ARCH[a], marker='o', lw=2, capsize=3,
                        label=alab)
        ax.axhline(ref, ls='--' if q == 'maxP' else ':', lw=1.3,
                   color=ps.TARGET_LINE if q == 'maxP' else ps.CHANCE_GREY, label=reflab)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab)
        if c == 0:
            ax.legend(loc='upper right', fontsize=7.5)

    # Row 1 — WITHIN mice (per-mouse lines)
    for c, (q, ylab, ref, reflab) in enumerate(quantities):
        ax = axes[1, c]
        _bench_line(ax, q)
        for a, alab in ARCHS:
            M = np.array([data[l][a][q] for l in labs], float)
            for j in range(M.shape[1]):
                ax.plot(x, M[:, j], color=ps.ARCH[a], lw=0.9, alpha=0.45)
            ax.plot(x, np.nanmean(M, 1), color=ps.ARCH[a], lw=2.4, label=alab)
        ax.axhline(ref, ls='--' if q == 'maxP' else ':', lw=1.3,
                   color=ps.TARGET_LINE if q == 'maxP' else ps.CHANCE_GREY)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab)

    axes[0, 0].set_title('across mice (mean ± s.e.m.)', fontsize=10, loc='left')
    axes[1, 0].set_title('within mice (one line per mouse)', fontsize=10, loc='left')
    ps.label_panels(axes)
    fig.suptitle('Soft evar weighting (evar$^α$): spatial vs temporal '
                 f'(Q half 100 ms, {len(mice)} mice)', y=1.01)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'evar_alpha_sweep')

    # numeric readout
    print(f'evar^α sweep (target max-prob {tgt_mp:.3f}):')
    for l in labs:
        s = data[l]
        print(f'  {l.splitlines()[0]:9s}  maxP spat/temp {np.mean(s["spat"]["maxP"]):.3f}/'
              f'{np.mean(s["temp"]["maxP"]):.3f}  KL-skill {np.nanmean(s["spat"]["KL"]):.2f}/'
              f'{np.nanmean(s["temp"]["KL"]):.2f}  PCA-skill {np.nanmean(s["spat"]["PCA"]):.2f}/'
              f'{np.nanmean(s["temp"]["PCA"]):.2f}')
    if bench is not None:
        b = bench
        print(f'  {BENCHMARK[0]:9.9s}  maxP spat/temp {np.mean(b["spat"]["maxP"]):.3f}/'
              f'{np.mean(b["temp"]["maxP"]):.3f}  KL-skill {np.nanmean(b["spat"]["KL"]):.2f}/'
              f'{np.nanmean(b["temp"]["KL"]):.2f}  (additive benchmark)')
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_variants')
    a = ap.parse_args()
    main(a.results_root, a.split, a.out_root)
