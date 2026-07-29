# -*- coding: utf-8 -*-
"""hp_fig5 — how peakiness and performance each relate to OVERFITTING.

One 2x2 figure over the swept cells:
  row 1  x = overfitting (val/train fit-loss)   y = decoded peakiness
  row 2  x = overfitting (val/train fit-loss)   y = KL normalised loss
  col 1  spatial            col 2  temporal

One scatter point per swept cell, coloured by the loss that trained it, built by
intersecting collect('peaky') / collect('nl_KL') / collect('overfit') on the
(axis, loss, arch) key and the x-position index — the same construction as
hparam_summary_figs.fig_relation.

Spearman correlations (overfitting vs each y) are PRINTED, never drawn: pooled over
losses, and within each training loss separately, because a pooled correlation can be
carried entirely by between-loss offsets rather than by overfitting.

Outputs (PNG+SVG) under figures/hparam_summary/ as hp_fig5_overfitting_relations.
Usage:  python diagnostics/hparam_overfitting_relations.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402
from hparam_summary_figs import collect, AXES, LOSSES, ARCHS, _cell_ref  # noqa: E402
from story_figures import _results, tgt_peak  # noqa: E402

CACHE = Path('/private/tmp/claude-501/-Users-theoamvr-Desktop-Experiments-UncertaintyV1/'
             'b40a2bea-25ae-4df2-ac22-7e328c5690c0/scratchpad/hp_overfit_cache.pkl')


def load_metrics(use_cache=True):
    """{'peaky'|'nl_KL'|'overfit': collect(...)}. Cached — collect('overfit') torch-loads
    ~1500 checkpoints."""
    if use_cache and CACHE.exists():
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    d = {k: collect(k) for k in ('peaky', 'nl_PCA', 'nl_KL', 'overfit')}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(d, f)
    return d


def records(D):
    """One record per (axis, loss, arch, x-position) that has all three metrics.

    `cell` is the (run, cell, slug) reference, so the caller can collapse the cells that
    several axes share (every axis contains the baseline value, so the baseline cell is
    re-plotted once per axis).
    """
    axis_vals = {a: v for a, _l, v in AXES}
    recs = []
    for (axis, loss, arch), pts in D['peaky'].items():
        d_pk = {p[0]: (p[1], p[2]) for p in pts}
        d_nl = {p[0]: (p[1], p[2]) for p in D['nl_KL'].get((axis, loss, arch), [])}
        d_np = {p[0]: (p[1], p[2]) for p in D['nl_PCA'].get((axis, loss, arch), [])}
        d_ov = {p[0]: (p[1], p[2]) for p in D['overfit'].get((axis, loss, arch), [])}
        for k in sorted(set(d_pk) & set(d_nl) & set(d_np) & set(d_ov)):
            recs.append(dict(axis=axis, loss=loss, arch=arch, xpos=k,
                             val=axis_vals[axis][k], cell=_cell_ref(axis, axis_vals[axis][k], loss),
                             ov=d_ov[k][0], ov_sem=d_ov[k][1],
                             pk=d_pk[k][0], pk_sem=d_pk[k][1],
                             nl=d_nl[k][0], nl_sem=d_nl[k][1],
                             nlp=d_np[k][0], nlp_sem=d_np[k][1]))
    return recs


# ------------------------------------------------------------------ figure
def figure(recs, io, out_root):
    ps.apply()
    fig, ax = plt.subplots(3, 2, figsize=ps.figsize(2, 3), sharex=True, sharey='row')
    # Projection-based is reported before KL: it is the loss these decoders were trained
    # on, so it is the minimum any performance claim must be shown under.
    ROWS = [('pk', 'decoded peakiness (max-prob)', io, 'IO target', 'k', ':'),
            ('nlp', 'projection normalised loss', 1.0, 'chance', '0.45', '--'),
            ('nl', 'KL normalised loss', 1.0, 'chance', '0.45', '--')]

    def _lim(key, ref):
        """Limits from the plotted means (+ the SEM whisker and the reference line).
        Set explicitly: on a log axis the x-SEM whiskers of points sitting at ratio ~1
        would otherwise drag autoscale down several empty decades."""
        m = np.array([x[key] for x in recs], float)
        e = m + np.array([x[key + '_sem'] for x in recs], float)
        lo = min(np.nanmin(m), ref)
        hi = max(np.nanmax(e), ref)
        return lo * 0.75, hi * 1.4

    for r, (ykey, ylab, ref, _rl, rc, rls) in enumerate(ROWS):
        for c, (arch, alab) in enumerate(ARCHS):
            a = ax[r][c]
            for loss in LOSSES:
                sel = [x for x in recs if x['arch'] == arch and x['loss'] == loss]
                if not sel:
                    continue
                # Every point is a mean over the 6 mice, so it carries its SEM on BOTH
                # axes — drawn faint and behind the markers, since 100+ points would
                # otherwise be unreadable.
                a.errorbar([x['ov'] for x in sel], [x[ykey] for x in sel],
                           xerr=[x['ov_sem'] for x in sel],
                           yerr=[x[ykey + '_sem'] for x in sel],
                           fmt='none', ecolor=ps.color(loss), elinewidth=0.6,
                           alpha=0.35, capsize=0, zorder=1)
                a.scatter([x['ov'] for x in sel], [x[ykey] for x in sel], s=24,
                          color=ps.color(loss), alpha=0.75, lw=0, zorder=2,
                          label=ps.loss_label(loss) if (r == 0 and c == 0) else None)
            a.axhline(ref, color=rc, ls=rls, lw=1.2)
            a.set_xscale('log')
            a.set_yscale('log')
            a.set_xlim(*_lim('ov', 1.0))
            a.set_ylim(*_lim(ykey, ref))
            if r == len(ROWS) - 1:
                a.set_xlabel('overfitting (val / train fit-loss)', fontsize=8)
            if r == 0:
                a.set_title(alab, fontsize=9)
            if c == 0:
                a.set_ylabel(ylab, fontsize=8)
    h, l = ax[0][0].get_legend_handles_labels()
    h += [Line2D([0], [0], color='k', ls=':', lw=1.2),
          Line2D([0], [0], color='0.45', ls='--', lw=1.2)]
    l += ['IO target', 'chance']
    ax[0][0].legend(h, l, fontsize=6.5, frameon=True)
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'hp_fig5_overfitting_relations')


# ------------------------------------------------------------------ stats
def _sp(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(xs[m], ys[m])
    return r, p, int(m.sum())


def _dedup(sel):
    """Collapse cells shared by several axes (the baseline cell recurs on every axis)."""
    seen, out = set(), []
    for x in sel:
        k = (x['cell'], x['arch'])
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def stats(recs):
    for ykey, ylab in [('pk', 'decoded peakiness'), ('nl', 'KL normalised loss')]:
        print(f'\n=== Spearman: overfitting (val/train fit-loss) vs {ylab} ===')
        for arch, alab in ARCHS:
            print(f'  -- {alab} --')
            for tag, sub in [('ALL POINTS   ', lambda s: s), ('UNIQUE CELLS ', _dedup)]:
                sel = sub([x for x in recs if x['arch'] == arch])
                r, p, n = _sp([x['ov'] for x in sel], [x[ykey] for x in sel])
                print(f'    pooled over losses  {tag} rho = {r:+.3f}  p = {p:.2e}  n = {n}')
            for loss in LOSSES:
                sel = _dedup([x for x in recs if x['arch'] == arch and x['loss'] == loss])
                r, p, n = _sp([x['ov'] for x in sel], [x[ykey] for x in sel])
                lab = ps.loss_label(loss)
                if np.isnan(r):
                    print(f'    within {lab:<16s}  n = {n} (too few)')
                else:
                    print(f'    within {lab:<16s}  rho = {r:+.3f}  p = {p:.2e}  n = {n}')


def sensitivity(recs, io):
    """Does the within-loss relation survive dropping the lobotomised cells?

    Heavy regularisation (large dropout / weight decay / lambda_H) drives overfitting DOWN
    and peakiness DOWN at the same time, so a shared driver could manufacture a positive
    within-loss rho without overfitting doing any work. Re-run excluding cells decoded
    flatter than half the IO target.
    """
    print(f'\n=== sensitivity: excluding lobotomised cells (peakiness < {io / 2:.4f}) ===')
    for ykey, ylab in [('pk', 'peakiness'), ('nl', 'KL norm loss')]:
        for arch, alab in ARCHS:
            row = []
            for loss in LOSSES:
                sel = [x for x in _dedup(recs)
                       if x['arch'] == arch and x['loss'] == loss and x['pk'] >= io / 2]
                r, p, n = _sp([x['ov'] for x in sel], [x[ykey] for x in sel])
                row.append(f'{ps.loss_label(loss, short=True)} rho={r:+.3f} p={p:.1e} n={n}')
            print(f'  {ylab:<13s} {alab:<9s} ' + ' | '.join(row))


def ranges(recs):
    print('\n=== ranges (unique cells, both archs pooled) ===')
    for k, lab in [('ov', 'overfitting'), ('pk', 'peakiness'), ('nl', 'KL norm loss')]:
        v = np.array([x[k] for x in _dedup(recs)], float)
        print(f'  {lab:<14s} min {np.nanmin(v):.4g}  median {np.nanmedian(v):.4g}  '
              f'max {np.nanmax(v):.4g}')
    print('\n=== per-loss medians (unique cells) ===')
    for loss in LOSSES:
        for arch, alab in ARCHS:
            sel = _dedup([x for x in recs if x['loss'] == loss and x['arch'] == arch])
            if not sel:
                continue
            print(f'  {ps.loss_label(loss):<16s} {alab:<9s} n={len(sel):>3d}  '
                  f'overfit {np.median([x["ov"] for x in sel]):.3f}  '
                  f'peaky {np.median([x["pk"] for x in sel]):.4f}  '
                  f'nl_KL {np.median([x["nl"] for x in sel]):.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/hparam_summary')
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    D = load_metrics(use_cache=not a.no_cache)
    recs = records(D)
    io = float(tgt_peak(_results('results', 'hpsweep_v2', S.baseline_cell(S.SPECS['v2']),
                                 loss='PCA'), 'spat').mean())
    print(f'IO target peakiness = {io:.4f};  {len(recs)} scatter points '
          f'({len(_dedup(recs))} unique cell x arch)')
    figure(recs, io, a.out_root)
    ranges(recs)
    stats(recs)
    sensitivity(recs, io)
    print(f'\nDone -> {(Path(a.out_root) / "hp_fig5_overfitting_relations.png").resolve()}')


if __name__ == '__main__':
    main()
