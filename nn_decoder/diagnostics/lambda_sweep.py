# -*- coding: utf-8 -*-
"""λ-sweep of the width-matched loss: spatial vs temporal at each λ (2026-06-09).

The width-matched loss is PCA_evar + λ·Brier (λ = the literal Brier weight; the
implementation's --shape-lambda config knob is 100·λ, carrying a ×100 scale on
L_PCA). Sweeping λ from 0 (plain evar) → 0.01 → 0.1 → 0.3 → ∞ (flat-evar) traces
a one-parameter family from the peaky,
KL-failing evar decoder to the calibrated flat-evar decoder. This figure shows,
at every λ, the difference between the spatial and temporal decoders, both
ACROSS mice (mean ± s.e.m.) and WITHIN mice (per-mouse lines):

  - decoded peakiness (max-probability) vs the IO target;
  - KL-skill = held-out KL / shuffle KL  (<1 beats chance) — the metric that
    sees the over-confidence;
  - PCA-skill — the training metric, which stays flat/blind across the sweep.

All scored on the COMMON true-evar PCA basis (the evar run's per-mouse pcs/evar)
so the variants are directly comparable.

Output (PNG+SVG) under figures/loss_variants/:
  lambda_sweep.png

Usage:  python diagnostics/lambda_sweep.py
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
# λ ladder: evar(0) → shape1 → shape10 → shape30 → flat(∞)
LADDER = [('evar\n(λ=0)', 'wm3'), ('λ=0.01', 'wm3_shape1'), ('λ=0.1', 'wm3_shape10'),
          ('λ=0.3', 'wm3_shape30'), ('flat\n(λ=∞)', 'wm3_flatevar')]
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]


def _load(results_root, run, split):
    f = Path(results_root) / run / SLUG / f'{split}.mat'
    return sio.loadmat(str(f), simplify_cells=True)['results'] if f.is_file() else None


def collect(results_root, split):
    """Per-(variant, arch): per-mouse maxP and PCA/KL skill on the common basis."""
    evar = _load(results_root, 'wm3', split)
    basis = {mk: (np.asarray(m['Dist']['pcs'], float),
                  np.asarray(m['Dist']['explained_var'], float))
             for mk, m in evar.items()}
    mice = sorted(evar.keys())
    data = {}
    for lab, run in LADDER:
        res = _load(results_root, run, split)
        if res is None:
            continue
        data[lab] = {a: {'maxP': [], 'PCA': [], 'KL': []} for a, _ in ARCHS}
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
                data[lab][a]['maxP'].append(dec.max(1).mean())
                for met in ('PCA', 'KL'):
                    real = _eval_one(dec, tgt, met, pcs, ev)
                    sh = _eval_one(shf, shf_t, met, pcs, ev)
                    data[lab][a][met].append(real / sh if sh else np.nan)
    tgt_mp = float(np.mean([np.asarray(evar[mk]['Dist']['spat']['target'], float).max(1).mean()
                            for mk in mice]))
    return data, mice, tgt_mp


def main(results_root, split, out_root):
    ps.apply()
    data, mice, tgt_mp = collect(results_root, split)
    labs = [l for l, _ in LADDER if l in data]
    x = np.arange(len(labs))

    fig, axes = plt.subplots(2, 3, figsize=ps.figsize(3, 2, panel_w=3.6, panel_h=3.0))
    quantities = [('maxP', 'decoded max-probability', tgt_mp, 'IO target'),
                  ('KL', 'KL-skill  (loss / shuffle)', 1.0, 'chance'),
                  ('PCA', 'PCA-skill  (loss / shuffle)', 1.0, 'chance')]

    # Row 0 — ACROSS mice (mean ± s.e.m.)
    for c, (q, ylab, ref, reflab) in enumerate(quantities):
        ax = axes[0, c]
        for a, alab in ARCHS:
            M = np.array([data[l][a][q] for l in labs], float)   # (nλ, nmice)
            m = np.nanmean(M, 1); se = np.nanstd(M, 1, ddof=1) / np.sqrt(M.shape[1])
            ax.errorbar(x, m, yerr=se, color=ps.ARCH[a], marker='o', lw=2, capsize=3,
                        label=alab)
        ax.axhline(ref, ls='--' if q == 'maxP' else ':', lw=1.3,
                   color=ps.TARGET_LINE if q == 'maxP' else ps.CHANCE_GREY, label=reflab)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab)
        if c == 0:
            ax.legend(loc='upper right')

    # Row 1 — WITHIN mice (per-mouse lines)
    for c, (q, ylab, ref, reflab) in enumerate(quantities):
        ax = axes[1, c]
        for a, alab in ARCHS:
            M = np.array([data[l][a][q] for l in labs], float)   # (nλ, nmice)
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
    fig.suptitle('Width-matched loss across λ: spatial vs temporal '
                 f'(Q half 100 ms, {len(mice)} mice)', y=1.01)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'lambda_sweep')

    # numeric readout
    print(f'λ-sweep (target max-prob {tgt_mp:.3f}):')
    for l in labs:
        s = data[l]
        print(f'  {l.splitlines()[0]:8s}  maxP spat/temp {np.mean(s["spat"]["maxP"]):.3f}/'
              f'{np.mean(s["temp"]["maxP"]):.3f}  KL-skill {np.nanmean(s["spat"]["KL"]):.2f}/'
              f'{np.nanmean(s["temp"]["KL"]):.2f}  PCA-skill {np.nanmean(s["spat"]["PCA"]):.2f}/'
              f'{np.nanmean(s["temp"]["PCA"]):.2f}')
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_variants')
    a = ap.parse_args()
    main(a.results_root, a.split, a.out_root)
