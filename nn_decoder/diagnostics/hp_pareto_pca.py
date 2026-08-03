# -*- coding: utf-8 -*-
"""Tradeoff / Pareto view of the swept cells — PROJECTION-LOSS-TRAINED DECODERS ONLY.

hp_fig5 puts all four training losses on one scatter, where the between-loss offsets dominate and
the within-loss structure is unreadable. This drops to the PCA-trained cells (the production loss)
and asks the actual design question: which hyperparameter settings are not dominated?

THE THREE COSTS, all oriented LOWER = BETTER so a Pareto front is meaningful:
  * overfitting            val / train fit-loss, >= 1
  * calibration error      max(peakiness/IO, IO/peakiness), >= 1
  * normalised loss        under the PROJECTION metric (the loss these were trained on, so the
                           minimum any performance claim must be shown under) and under KL

CALIBRATION ERROR, NOT RAW PEAKINESS — this is the one real change from fig5. Peakiness is not a
"lower is better" quantity: the IO target is 0.0594 and missing it in EITHER direction is a
failure, so a Pareto front drawn on raw peakiness is meaningless (it rewards the lobotomised cells
that decode flat). Folding it to the symmetric ratio max(pk/IO, IO/pk) makes "1 = perfect" and the
front real. The raw peakiness and its direction are still printed, and the per-knob figures mark
under-sharpened cells with an open face so the direction is never lost.

OUTPUTS (PNG+SVG) under figures/hp_pareto/
  hp_pareto_pca_overview        3 rows (calibration / projection nl / KL nl) x 2 cols (arch),
                                every PCA cell coloured by which hyperparameter it varies, with
                                the Pareto front stepped over each panel.
  hp_pareto_pca_<hparam>        one per knob: 2x2 planes, both arches, points coloured by the
                                knob's value and joined in order, so the DIRECTION each knob moves
                                you through the tradeoff space is visible against the full cloud.

The Pareto-optimal cells of each panel are PRINTED, which is the actionable output.
Usage:  python diagnostics/hp_pareto_pca.py [--no-cache]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402
from hparam_summary_figs import AXES, ARCHS  # noqa: E402
from hparam_overfitting_relations import load_metrics, records  # noqa: E402
from story_figures import _results, tgt_peak  # noqa: E402

LOSS = 'PCA'
AX_LAB = {a: lab for a, lab, _v in AXES}
AX_COL = dict(zip([a for a, _l, _v in AXES],
                  ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                   '#911eb4', '#008080', '#9a6324']))
# (x key, y key, x label, y label) for every plane drawn
P_OV_NLP = ('ov', 'nlp', 'overfitting (val / train)', 'projection normalised loss')
P_OV_NLK = ('ov', 'nl', 'overfitting (val / train)', 'KL normalised loss')
P_OV_CAL = ('ov', 'cal', 'overfitting (val / train)', 'calibration error')
P_CAL_NLP = ('cal', 'nlp', 'calibration error', 'projection normalised loss')
# The projection normalised loss spans less than half a decade (0.49-1.35) across the whole
# sweep -- that is the finding, not a plotting accident -- so it reads on a LINEAR axis. The
# other three quantities span 1-2 decades and stay logarithmic.
SCALE = {'ov': 'log', 'cal': 'log', 'nl': 'log', 'nlp': 'linear'}
# Marker SHAPE carries the architecture in the per-knob figures, because colour is already
# spent on the knob's value and a second colour encoding is unreadable against it.
MARK = {'spat': 'o', 'temp': 's'}


def pareto(xs, ys):
    """Indices of the non-dominated points (both axes lower = better), sorted by x."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    idx = np.argsort(xs, kind='stable')
    front, best = [], np.inf
    for i in idx:
        if not np.isfinite(xs[i]) or not np.isfinite(ys[i]):
            continue
        if ys[i] < best:
            front.append(int(i))
            best = ys[i]
    return front


def step(ax, xs, ys, front, xlim, ylim, **kw):
    """Stepped Pareto staircase, with the risers clipped to the panel rather than run off to
    infinity — an unclipped staircase drags a log autoscale over several empty decades and
    squashes the data into a band."""
    if len(front) < 2:
        return
    fx = [xs[i] for i in front]
    fy = [ys[i] for i in front]
    px, py = [fx[0]], [ylim[1]]  # riser starts at the top of the panel, not at infinity
    for x_, y_ in zip(fx, fy):
        px += [x_, x_]
        py += [py[-1], y_]
    px.append(xlim[1])
    py.append(py[-1])
    ax.plot(px, py, **kw)


def limits(cells, plane, pad=1.25):
    """Shared limits for one plane over ALL cells and both arches, so every panel in every
    figure of this suite is directly comparable."""
    out = []
    for k in plane[:2]:
        v = np.array([x[k] for x in cells], float)
        v = v[np.isfinite(v) & (v > 0)]
        if SCALE[k] == 'log':
            out.append((v.min() / pad, v.max() * pad))
        else:
            m = (v.max() - v.min()) * (pad - 1.0)
            out.append((max(0.0, v.min() - m), v.max() + m))
    return out[0], out[1]


def _dedup(recs):
    """Collapse cells shared by several axes; keep the axis list so colouring can note it."""
    seen = {}
    for x in recs:
        k = (x['cell'], x['arch'])
        if k not in seen:
            seen[k] = dict(x, axes=[x['axis']])
        else:
            seen[k]['axes'].append(x['axis'])
    return list(seen.values())


def _panel(a, sel, plane, lims, base=None, front=True, color_by=None, cloud=None, marker='o'):
    """Scatter one plane. `color_by` maps a record -> colour; default is by hyperparameter."""
    xk, yk, xl, yl = plane
    xlim, ylim = lims
    a.set_xscale(SCALE[xk]); a.set_yscale(SCALE[yk])
    a.set_xlim(*xlim); a.set_ylim(*ylim)
    if cloud:
        a.scatter([c[xk] for c in cloud], [c[yk] for c in cloud], s=16, color='0.85',
                  lw=0, zorder=1)
    xs = [x[xk] for x in sel]
    ys = [x[yk] for x in sel]
    if front:
        f = pareto(xs, ys)
        step(a, xs, ys, f, xlim, ylim, color='0.25', lw=1.1, ls='-', alpha=0.8, zorder=2)
    cols = [color_by(x) for x in sel] if color_by else [AX_COL[x['axes'][0]] for x in sel]
    a.scatter(xs, ys, s=42 if marker == 's' else 46, c=cols, marker=marker, lw=0.5,
              edgecolors='k', zorder=4)
    if base is not None:
        a.plot(base[xk], base[yk], marker='*', ms=14, color='k', mfc='w', mew=1.2, zorder=6)
    a.axvline(1.0, color='0.6', ls=':', lw=1.0)
    a.axhline(1.0, color='0.6', ls=':', lw=1.0)
    a.set_xlabel(xl, fontsize=8); a.set_ylabel(yl, fontsize=8)


def fig_overview(cells, io, out_root):
    ps.apply()
    fig, ax = plt.subplots(3, 2, figsize=ps.figsize(2, 3), sharex=True, sharey='row')
    planes = [P_OV_CAL, P_OV_NLP, P_OV_NLK]
    for r, plane in enumerate(planes):
        lims = limits(cells, plane)
        for c, (arch, alab) in enumerate(ARCHS):
            sel = [x for x in cells if x['arch'] == arch]
            base = next((x for x in sel if x['is_base']), None)
            _panel(ax[r][c], sel, plane, lims, base=base)
            if r == 0:
                ax[r][c].set_title(alab, fontsize=9)
            if r != len(planes) - 1:
                ax[r][c].set_xlabel('')
            if c == 1:
                ax[r][c].set_ylabel('')
    ax[0][0].legend(handles=(
        [Line2D([0], [0], ls='', marker='o', color=AX_COL[a], ms=5, mec='k', mew=0.4,
                label=AX_LAB[a]) for a, _l, _v in AXES] +
        [Line2D([0], [0], ls='', marker='*', color='k', mfc='w', mew=1.2, ms=10,
                label='baseline cell'),
         Line2D([0], [0], color='0.25', lw=1.1, label='Pareto front')]),
        fontsize=5.5, frameon=True, ncol=2, loc='upper right')
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'hp_pareto_pca_overview')


def fig_hparam(axis, recs, cells, io, out_root):
    """One knob: 2x2 planes, both arches, coloured by the knob's value."""
    ps.apply()
    vals = dict(AXES_V)[axis]
    sel = [x for x in recs if x['axis'] == axis]
    if not sel:
        return
    order = {v: i for i, v in enumerate(vals)}
    cmap = plt.cm.plasma
    def cby(x):
        return cmap(order[x['val']] / max(1, len(vals) - 1) * 0.86)

    fig, ax = plt.subplots(2, 2, figsize=ps.figsize(2, 2))
    for a_, plane in zip(ax.ravel(), [P_OV_NLP, P_OV_NLK, P_OV_CAL, P_CAL_NLP]):
        lims = limits(cells, plane)
        for arch, _alab in ARCHS:
            s = sorted([x for x in sel if x['arch'] == arch], key=lambda x: order[x['val']])
            cloud = [c for c in cells if c['arch'] == arch]
            xk, yk = plane[0], plane[1]
            a_.plot([x[xk] for x in s], [x[yk] for x in s], '-',
                    color='0.45', lw=0.9, alpha=0.7, zorder=3)
            _panel(a_, s, plane, lims, front=False, color_by=cby, cloud=cloud,
                   marker=MARK[arch])
    ax[0][0].legend(handles=(
        [Line2D([0], [0], ls='', marker='o', color=cby(dict(val=v)), ms=5, mec='k', mew=0.4,
                label=f'{v}') for v in vals] +
        [Line2D([0], [0], ls='', marker=MARK[a], color='0.55', mec='k', mew=0.5, ms=6,
                label=al) for a, al in ARCHS] +
        [Line2D([0], [0], ls='', marker='o', color='0.85', ms=5, label='all PCA cells')]),
        fontsize=5.5, frameon=True, ncol=2)
    fig.suptitle(f'{AX_LAB[axis]}', fontsize=10)
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'hp_pareto_pca_{axis}')


AXES_V = [(a, v) for a, _l, v in AXES]


def report(cells, io):
    print(f'\nIO target peakiness = {io:.5f}.  {len(cells)} unique PCA cell x arch '
          f'({len(cells) // 2} cells x 2 arches).')
    for arch, alab in ARCHS:
        sel = [x for x in cells if x['arch'] == arch]
        print(f'\n=== {alab.upper()} — Pareto-optimal cells (lower is better on both axes) ===')
        for plane, nm in [(P_OV_NLP, 'overfitting vs PROJECTION normalised loss'),
                          (P_OV_NLK, 'overfitting vs KL normalised loss'),
                          (P_OV_CAL, 'overfitting vs calibration error'),
                          (P_CAL_NLP, 'calibration error vs PROJECTION normalised loss')]:
            xk, yk = plane[0], plane[1]
            f = pareto([x[xk] for x in sel], [x[yk] for x in sel])
            print(f'  -- {nm} --')
            for i in f:
                x = sel[i]
                print(f'     {x["label"]:<28s} overfit {x["ov"]:5.2f}  calib {x["cal"]:5.2f}'
                      f'  (peaky {x["pk"]:.4f}{" UNDER" if x["pk"] < io else " over"})'
                      f'  projNL {x["nlp"]:.3f}  klNL {x["nl"]:.3f}')
        both = [x for x in sel if x['nlp'] < 1 and x['nl'] < 1]
        print(f'  cells beating chance under BOTH metrics: {len(both)}/{len(sel)}')
        if both:
            b = min(both, key=lambda x: x['cal'])
            print(f'     best-calibrated of those: {b["label"]} '
                  f'(calib {b["cal"]:.2f}, overfit {b["ov"]:.2f}, projNL {b["nlp"]:.3f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/hp_pareto')
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()

    D = load_metrics(use_cache=not a.no_cache)
    recs = [x for x in records(D) if x['loss'] == LOSS]
    io = float(tgt_peak(_results('results', 'hpsweep_v2', S.baseline_cell(S.SPECS['v2']),
                                 loss='PCA'), 'spat').mean())
    base_cell = ('hpsweep_v2', S.baseline_cell(S.SPECS['v2']), 'PCA')
    for x in recs:
        x['cal'] = max(x['pk'] / io, io / x['pk'])
        x['is_base'] = x['cell'] == base_cell
        x['label'] = 'BASELINE' if x['is_base'] else f'{AX_LAB[x["axis"]]} = {x["val"]}'
    cells = _dedup(recs)
    for x in cells:
        if x['is_base']:
            x['label'] = 'BASELINE'

    fig_overview(cells, io, a.out_root)
    for axis, _lab, _v in AXES:
        fig_hparam(axis, recs, cells, io, a.out_root)
    report(cells, io)
    print(f'\nDone -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
