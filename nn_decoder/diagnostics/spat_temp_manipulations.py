# -*- coding: utf-8 -*-
"""Spatial vs temporal, head to head, under every manipulation in the over-sharpening story.

Answers two questions per manipulation:
  ACROSS animals — is the spat-vs-temp difference consistent over the cohort? Paired t over
    mice (n=6), repeated with Mouse 2 excluded (n=5), since M2 is the animal flagged as
    consistently anomalous. The across-animal test is the honest one; trial-level tests are
    pseudoreplication (GOTCHAS).
  WITHIN animal — does every mouse show the same sign? Per-mouse paired lines, M2 marked.

Two metrics, because the project's whole point is that they dissociate:
  * normalised loss = held-out KL(decoded||IO target) / leave-one-out predict-mean
    (< 1 beats chance; lower is better) — PERFORMANCE;
  * decoded peakiness (max-prob, IO target ~0.059) — CALIBRATION.

Values are keyed by mouse ID rather than taken in dict order, so the pairing is correct
even if two cells enumerate mice differently.

Outputs (PNG+SVG) under figures/prodfix/:
  spat_temp_across_animals   — Delta (spatial - temporal) per manipulation, n=6 and n=5
  spat_temp_within_animals   — per-mouse paired lines, one panel per manipulation

Usage:  python diagnostics/spat_temp_manipulations.py [--exclude mouse_2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402
from story_figures import _results, PROD  # noqa: E402  (shared, loss-aware slug resolution)

N_CATS = 91


def _manipulations():
    v2 = S.SPECS['v2']
    # (label, run, cell, loss, cross-regime?)
    return [
        ('baseline',        PROD, 'A_baseline_pca', 'PCA', False),
        ('no hidden layer', PROD, 'C_nohidden_pca', 'PCA', False),
        ('dropout 0.9',     'hpsweep_v2', S.cell_for(v2, 'dropout', 0.9), 'PCA', True),
        ('weight decay .01', 'hpsweep_v2', S.cell_for(v2, 'weight_decay', 0.01), 'PCA', True),
        ('smooth λ=0.3',    PROD, 'A_smooth0p3', 'PCA', False),
        ('smooth λ=1',      PROD, 'A_smooth1', 'PCA', False),
        ('shape λ=0.1',     PROD, 'A_shape10', 'PCA', False),
        ('shape λ=0.3',     PROD, 'A_shape30', 'PCA', False),
        ('KL',              PROD, 'A_reference_kl', 'KL', False),
        ('JS',              PROD, 'A_reference_js', 'JS', False),
    ]


def per_mouse(results_root, run, cell, arch, loss_slug, what):
    """{mouse_id: value} — keyed, so spat/temp pairing can never silently misalign."""
    res = _results(results_root, run, cell, loss=loss_slug)
    out = {}
    for mk in res:
        if not str(mk).startswith('mouse_'):
            continue
        idx = int(str(mk).split('_')[-1])
        D = res[mk].get('Dist')
        if not (isinstance(D, dict) and arch in D):
            continue
        dec = np.asarray(D[arch]['decoded'], float)
        tgt = np.asarray(D[arch]['target'], float)
        if what == 'peaky':
            out[idx] = float(dec.max(1).mean())
            continue
        ok = np.isfinite(tgt).all(1)
        if not ok.any():
            continue
        n_ok = int(ok.sum())
        tot = tgt[ok].sum(axis=0)
        pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
        if n_ok > 1:                       # leave-one-out predict-mean null
            pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
        num = _eval_one(dec, tgt, 'KL', D.get('pcs'), D.get('explained_var'))
        den = _eval_one(pm, tgt, 'KL', D.get('pcs'), D.get('explained_var'))
        if np.isfinite(num) and np.isfinite(den) and den > 0:
            out[idx] = float(num / den)
    return out


def paired(results_root, manip, what, exclude_id=None):
    """(ids, spat, temp) aligned by mouse id, optionally dropping one animal."""
    _lab, run, cell, loss, cross = manip
    slug = loss if cross else None
    sp = per_mouse(results_root, run, cell, 'spat', slug, what)
    tp = per_mouse(results_root, run, cell, 'temp', slug, what)
    ids = sorted(set(sp) & set(tp))
    if exclude_id is not None:
        ids = [i for i in ids if i != exclude_id]
    return ids, np.array([sp[i] for i in ids]), np.array([tp[i] for i in ids])


def _stars(p):
    if not np.isfinite(p):
        return ''
    return '**' if p < 0.01 else ('*' if p < 0.05 else '')


# ------------------------------------------------------------ across animals
def fig_across(results_root, out_root, exclude_id):
    ps.apply()
    M = _manipulations()
    WHAT = [('normloss', 'Δ normalised loss  (spatial − temporal)\n<0: spatial better'),
            ('peaky', 'Δ peakiness  (spatial − temporal)\n<0: spatial less over-confident')]
    fig, axes = plt.subplots(len(WHAT), 1, figsize=ps.figsize(2.6, 2), sharex=True)
    x = np.arange(len(M))
    w = 0.38
    for r, (what, ylab) in enumerate(WHAT):
        ax = axes[r]
        for k, (exc, col, lab) in enumerate([(None, '#4a4a4a', 'all mice (n=6)'),
                                             (exclude_id, '#cb181d', f'M{exclude_id} excluded (n=5)')]):
            ms, ss, st = [], [], []
            for m in M:
                ids, sp, tp = paired(results_root, m, what, exc)
                d = sp - tp
                ms.append(d.mean())
                ss.append(d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0)
                st.append(_stars(sstats.ttest_rel(sp, tp).pvalue) if d.size >= 3 else '')
            bars = ax.bar(x + (k - 0.5) * w, ms, w * 0.95, yerr=ss, capsize=2,
                          color=col, alpha=0.9, edgecolor='k', lw=0.4, label=lab)
            for b, s, mval in zip(bars, st, ms):
                if s:
                    ax.text(b.get_x() + b.get_width() / 2,
                            mval + (1 if mval >= 0 else -1) * (max(map(abs, ms)) * 0.06),
                            s, ha='center', va='bottom' if mval >= 0 else 'top', fontsize=8)
        ax.axhline(0, color='k', lw=1.0)
        ax.set_ylabel(ylab, fontsize=8)
        if r == 0:
            ax.legend(fontsize=7, frameon=True)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([m[0] for m in M], rotation=45, ha='right', fontsize=7.5)
    ps.label_panels(axes)
    fig.suptitle('Spatial vs temporal across animals, per manipulation (paired t over mice; * p<0.05, ** p<0.01, '
                 'uncorrected)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'spat_temp_across_animals')

    # numeric readout
    print(f"{'manipulation':18s}{'metric':10s}{'spat':>8s}{'temp':>8s}{'Δ':>8s}{'p(n=6)':>9s}"
          f"{'Δ(n=5)':>9s}{'p(n=5)':>9s}  sign")
    for m in M:
        for what in ('normloss', 'peaky'):
            ids, sp, tp = paired(results_root, m, what, None)
            _i5, sp5, tp5 = paired(results_root, m, what, exclude_id)
            p6 = sstats.ttest_rel(sp, tp).pvalue
            p5 = sstats.ttest_rel(sp5, tp5).pvalue
            print(f"{m[0]:18s}{what:10s}{sp.mean():8.3f}{tp.mean():8.3f}{(sp-tp).mean():8.3f}"
                  f"{p6:9.3f}{(sp5-tp5).mean():9.3f}{p5:9.3f}  {int((sp<tp).sum())}/6 spat<temp")


# ------------------------------------------------------------- within animal
def fig_within(results_root, out_root, exclude_id):
    ps.apply()
    M = _manipulations()
    ncol = 5
    nrow = int(np.ceil(len(M) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow), sharey=True)
    axes = np.atleast_2d(axes)
    for j, m in enumerate(M):
        ax = axes[j // ncol][j % ncol]
        ids, sp, tp = paired(results_root, m, 'normloss', None)
        for i, s, t in zip(ids, sp, tp):
            is_ex = (i == exclude_id)
            ax.plot([0, 1], [s, t], '-o', ms=4, lw=1.6 if is_ex else 1.0,
                    color='#cb181d' if is_ex else '0.62', zorder=4 if is_ex else 2)
        ax.plot([0, 1], [sp.mean(), tp.mean()], '-o', color='k', ms=7, lw=2.4, zorder=5)
        ax.axhline(1.0, color='k', ls=':', lw=1.2)
        ax.set_yscale('log')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['spat', 'temp'], fontsize=7.5)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(m[0], fontsize=8)
        if j % ncol == 0:
            ax.set_ylabel('normalised loss\n(dotted = chance)', fontsize=8)
    for j in range(len(M), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    axes[0][0].legend(handles=[
        Line2D([0], [0], color='0.62', lw=1.2, marker='o', ms=4, label='mouse'),
        Line2D([0], [0], color='#cb181d', lw=1.8, marker='o', ms=4, label=f'mouse {exclude_id}'),
        Line2D([0], [0], color='k', lw=2.4, marker='o', ms=6, label='mean')],
        fontsize=6, frameon=True, loc='best')
    ps.label_panels(axes.ravel()[:len(M)])
    fig.suptitle('Spatial vs temporal within each animal — one line per mouse, per manipulation '
                 '(normalised loss, lower is better)', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'spat_temp_within_animals')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/prodfix')
    ap.add_argument('--exclude', default=2, type=int, help='mouse id to leave out (default 2)')
    a = ap.parse_args()
    fig_across(a.results_root, a.out_root, a.exclude)
    fig_within(a.results_root, a.out_root, a.exclude)
    print(f'\nDone -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
