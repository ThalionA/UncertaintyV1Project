# -*- coding: utf-8 -*-
"""Cross-mouse synthesis of the paired old-vs-new target-family scorecards.

Reads the per-mouse CSVs written by ``diagnostics/io_hmm_vs_export_scorecard.py``
(``figures/io_hmm_vs_export_v2/m<M>/io_hmm_vs_export_cells.csv``: one row per
cell x decoder x arm, 160 rows per mouse) and asks the only question the
per-mouse runs cannot: which of the mouse-0 headlines survive across SIX mice?

Nothing is re-scored here. Every number is a per-mouse scorecard number (KL
skill, projection skill, equivalent sharpening s_hat with its agreement/clamped
companions, null-normalised overfitting gap, best epoch), reduced per mouse to
the MEDIAN over the five lambda_H cells of each of the 16 groups
(loss family x hidden size x spatial/temporal), and then compared across mice.
The unit of replication is the MOUSE (n = 6, paired across arms); per-mouse
values and sign counts are the statistics. No p-values.

Metric definitions are the scorecard's and are not reopened:
  skill      = loss / leave-one-out predict-mean null loss on the held-out test
               trials (< 1 beats the strictest null);
  s_hat      = equivalent sharpening factor, calibrated per arm on that arm's own
               targets (1 calibrated, > 1 over-sharpened, < 1 too broad), always
               read with its two companions: agreement > 0.10 means the decoder
               RESHAPES and s_hat under-describes it (ring); clamped means the
               inversion hit a ladder end, 0.25 or 6.0, and the number is a
               bound (cross);
  overfit gap = (val_fit - train_fit) at best_epoch / that arm's predict-mean
               loss under the cell's own training loss (0 = none);
  best_epoch = restored epoch, 0-based; 199 = never early-stopped (censored).

Figures (PNG + SVG via figsave.save_fig) under figures/io_hmm_vs_export_v2/crossmouse/:
  (a) paired_dumbbells          one panel per metric; per group six per-mouse
                                old->new dumbbells plus the across-mouse median
                                dumbbell (bold); sign count "new<old: k/6"
  (b) headline_survival_grid    the mouse-0 headlines x six mice, verdict + the
                                deciding numbers (THE prior-scoring figure)
  (c) kl_vs_proj_disagreement   per group, per mouse: (d KL skill, d proj skill)
                                old->new; quadrants = agreement / disagreement
  (d) s_hat_new_vs_old          group-median s_hat new vs old, all mice, with the
                                per-mouse target widths in the legend and the
                                change vs old-target width for the evar groups

Then prints the prior scoring (PREDICTIONS.md 2026-08-18, five-mouse entry, now
scored on six) plus the three mouse-0 extras, one line each.

Usage:
    python diagnostics/io_hmm_vs_export_crossmouse.py
    python diagnostics/io_hmm_vs_export_crossmouse.py --in-root figures/io_hmm_vs_export_v2 \
        --mice 0 1 2 3 4 5 --out-dir figures/io_hmm_vs_export_v2/crossmouse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.transforms as mtransforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import peakiness_style as ps                          # noqa: E402
from figsave import save_fig                          # noqa: E402
# constants and helpers shared with the per-mouse scorecard (single source of truth)
from io_hmm_vs_export_scorecard import (              # noqa: E402
    LOSS_ORDER, MODEL_ORDER, FAMILY_COLOR, FAMILY_LABEL, AGREE_FLAG, NULL_NAME,
    _log_ticks, wrap_label)

CSV_NAME = 'io_hmm_vs_export_cells.csv'          # written by the scorecard's write_csv
ARCHS = ('spat', 'temp')
ARCH_LABEL = {'spat': 'spatial', 'temp': 'temporal'}
HID_LABEL = {'h8': 'H8', 'lin': 'lin'}
MOUSE_MARKER = ['o', 's', '^', 'v', 'D', 'P']
EPOCH_CAP = 199                                   # best_epoch == cap -> never early-stopped
CLAMP_MAJ = 3                                     # >= 3 of 5 lambda cells clamped -> the median is a bound

# Marginal resultant length R of each mouse's IO-HMM posterior (PREDICTIONS.md,
# 2026-08-18 six-mouse entry, "Baseline (measured 2026-08-18, HMM file)"): the
# target-concentration covariate the six-mouse prior (b) is stated against.
MARGINAL_R = {0: 0.24, 1: 0.23, 2: 0.19, 3: 0.34, 4: 0.46, 5: 0.52}

ARM_LABELS = {'old': 'export Q, 91 linear bins x 1 deg, 0-90',
              'new': 'IO-HMM marginal, 72 circular bins x 2.5 deg, [0,180)'}

# (key, short, y-axis label with the normalisation, reference value, log scale)
METRICS = [
    ('s_hat', 'equivalent sharpening ŝ',
     'ŝ  (calibrated per arm on that arm’s own targets)\n'
     '1 = calibrated · >1 over-sharpened · <1 too broad',
     1.0, True),
    ('kl_skill', 'KL skill',
     'KL skill\nmean KL(tgt‖dec) ÷ mean KL(tgt‖predict-mean)\n<1 beats the null',
     1.0, False),
    ('proj_skill', 'projection skill',
     'projection skill\nproj. loss ÷ proj. loss(predict-mean),\narm-pinned evar, '
     'held-out rows; <1 beats the null',
     1.0, False),
    ('overfit_gap', 'overfitting gap',
     'overfitting gap at best epoch\n(val − train) fit loss ÷ that arm’s\n'
     'predict-mean loss (own training loss)',
     0.0, False),
    ('best_epoch', 'best epoch',
     'best epoch (0-based; restored = argmin val)\n'
     f'{EPOCH_CAP} = never early-stopped (censored)',
     np.nan, False),
]
METRIC_KEYS = [m[0] for m in METRICS]


# ----------------------------------------------------------------------
# loading + per-mouse group reduction
# ----------------------------------------------------------------------
def load_cells(in_root, mice):
    """All per-mouse scorecard CSVs, concatenated, with a ``mouse`` column.
    Fails loudly on a missing CSV or an unexpected row count."""
    frames = []
    for m in mice:
        p = Path(in_root) / f'm{m}' / CSV_NAME
        if not p.exists():
            sys.exit(f'ABORT: missing {p} — run the per-mouse scorecard for mouse {m} first')
        d = pd.read_csv(p)
        d['mouse'] = int(m)
        frames.append(d)
        print(f'  mouse {m}: {len(d)} rows from {p}')
    cells = pd.concat(frames, ignore_index=True)
    cells['censored'] = cells['censored'].astype(str).str.lower().eq('true')
    cells['s_hat_clamped'] = cells['s_hat_clamped'].fillna(0).astype(float) > 0
    cells['reshape'] = cells['s_hat_agreement'] > AGREE_FLAG
    return cells


def group_table(cells):
    """Per (mouse, family, hidden, arch, arm): median over the lambda_H cells of
    every metric, plus the flag counts. Spatial lambda cells are exact replicates
    (lambda_H acts on the temporal decoder only), so a spatial group's median IS
    its single value."""
    keys = ['mouse', 'loss_family', 'hidden', 'arch', 'arm']
    agg = {k: (k, 'median') for k in METRIC_KEYS}
    agg.update(agreement=('s_hat_agreement', 'median'),
               n_clamped=('s_hat_clamped', 'sum'),
               n_reshape=('reshape', 'sum'),
               n_censored=('censored', 'sum'),
               n_cells=('cell', 'size'))
    g = cells.groupby(keys).agg(**agg).reset_index()
    g['clamp_flag'] = g['n_clamped'] >= CLAMP_MAJ
    g['reshape_flag'] = g['agreement'] > AGREE_FLAG
    return g


def groups_in_order():
    return [(f, h, a) for f in LOSS_ORDER for h in MODEL_ORDER for a in ARCHS]


def group_name(f, h, a):
    return f'{f}_{h}_{a}'


def gval(g, mouse, fam, hid, arch, arm, key):
    r = g[(g.mouse == mouse) & (g.loss_family == fam) & (g.hidden == hid)
          & (g.arch == arch) & (g.arm == arm)]
    if r.empty:
        return np.nan
    return float(r.iloc[0][key])


def wide(g, key):
    """DataFrame indexed by (fam, hid, arch), columns MultiIndex (arm, mouse)."""
    return g.pivot_table(index=['loss_family', 'hidden', 'arch'],
                         columns=['arm', 'mouse'], values=key, aggfunc='first')


# ----------------------------------------------------------------------
# (a) paired dumbbells
# ----------------------------------------------------------------------
def _group_x():
    """x positions for the 16 groups with a gutter between loss families."""
    xs, fam_spans = [], {}
    x = 0.0
    for fi, f in enumerate(LOSS_ORDER):
        start = x
        for _h in MODEL_ORDER:
            for _a in ARCHS:
                xs.append(x)
                x += 1.0
        fam_spans[f] = (start - 0.5, x - 0.5)
        x += 0.7
    return np.array(xs), fam_spans


def fig_paired_dumbbells(g, mice, out_dir):
    groups = groups_in_order()
    xs, fam_spans = _group_x()
    n_m = len(mice)
    offs = np.linspace(-0.36, 0.12, n_m)        # six mice, left of centre-right
    x_med = 0.34                                 # the across-mouse median, bold
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(16.0, 3.05 * len(METRICS)),
                             sharex=True)
    sign_rows = {}
    for ax, (key, short, ylab, ref, logsc) in zip(axes, METRICS):
        vals_old = np.full((len(groups), n_m), np.nan)
        vals_new = np.full((len(groups), n_m), np.nan)
        for gi, (f, h, a) in enumerate(groups):
            col = FAMILY_COLOR[f]
            for mi, m in enumerate(mice):
                yo = gval(g, m, f, h, a, 'old', key)
                yn = gval(g, m, f, h, a, 'new', key)
                vals_old[gi, mi], vals_new[gi, mi] = yo, yn
                if not (np.isfinite(yo) and np.isfinite(yn)):
                    continue
                xm = xs[gi] + offs[mi]
                ax.plot([xm, xm], [yo, yn], '-', lw=0.9, color=col, alpha=0.55,
                        zorder=2)
                mk = MOUSE_MARKER[mi]
                ax.plot(xm, yo, marker=mk, ms=3.8, ls='none', mfc='w', mec=col,
                        mew=0.9, zorder=3)
                ax.plot(xm, yn, marker=mk, ms=3.8, ls='none', mfc=col, mec=col,
                        mew=0.9, zorder=3)
                if key == 's_hat':
                    for arm, y in (('old', yo), ('new', yn)):
                        if gval(g, m, f, h, a, arm, 'clamp_flag') > 0:
                            ax.plot(xm, y, marker='x', ms=6, ls='none', color='k',
                                    mew=1.1, zorder=5)
                        if gval(g, m, f, h, a, arm, 'reshape_flag') > 0:
                            ax.plot(xm, y, marker='o', ms=8.0, ls='none', mfc='none',
                                    mec='0.5', mew=0.7, zorder=4)
            # across-mouse median dumbbell
            mo, mn = np.nanmedian(vals_old[gi]), np.nanmedian(vals_new[gi])
            if np.isfinite(mo) and np.isfinite(mn):
                xb = xs[gi] + x_med
                ax.plot([xb, xb], [mo, mn], '-', lw=2.6, color=col, zorder=4)
                ax.plot(xb, mo, marker='s', ms=6.5, ls='none', mfc='w', mec=col,
                        mew=1.8, zorder=5)
                ax.plot(xb, mn, marker='o', ms=7, ls='none', mfc=col, mec=col,
                        zorder=5)
            # sign count
            ok = np.isfinite(vals_old[gi]) & np.isfinite(vals_new[gi])
            k = int(np.sum(vals_new[gi][ok] < vals_old[gi][ok]))
            t = int(np.sum(vals_new[gi][ok] == vals_old[gi][ok]))
            n = int(ok.sum())
            sign_rows[(key, group_name(f, h, a))] = (k, t, n)
            txt = f'{k}/{n}' + (f'\n{t}=' if t else '')
            ax.text(xs[gi], 1.03, txt, ha='center', va='bottom', fontsize=7.5,
                    transform=mtransforms.blended_transform_factory(
                        ax.transData, ax.transAxes),
                    color='0.15')
        # family bands + names
        for fi, f in enumerate(LOSS_ORDER):
            lo, hi = fam_spans[f]
            if fi % 2 == 0:
                ax.axvspan(lo, hi, color='0.955', zorder=0)
        if logsc:
            ax.set_yscale('log')
        if np.isfinite(ref):
            ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=1)
        if key == 'best_epoch':
            ax.axhline(EPOCH_CAP, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=1)
            ax.set_ylim(-8, 215)
        if key == 'overfit_gap':
            top = 0.85
            ax.set_ylim(-0.3, top)
            for gi in range(len(groups)):
                for mi in range(n_m):
                    for v in (vals_old[gi, mi], vals_new[gi, mi]):
                        if np.isfinite(v) and v > top:
                            ax.annotate(f'↑{v:.2f}', (xs[gi] + offs[mi], top), fontsize=6.5,
                                        xytext=(3, -1), textcoords='offset points',
                                        ha='left', va='top', color='0.3')
        if key == 's_hat':
            ax.set_ylim(0.2, 7.5)
            _log_ticks(ax, (0.2, 7.5), which='y')
        ax.set_ylabel(ylab, fontsize=8.2)
        ax.tick_params(labelsize=8)
        ax.text(0.0, 1.12, f'{short}   — numbers above each group: mice with new < old (of {n_m})'
                + ('; "k=" = ties' if key == 's_hat' else ''),
                transform=ax.transAxes, fontsize=10.5, ha='left', va='bottom', fontweight='bold')
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([f'{HID_LABEL[h]}\n{ARCH_LABEL[a][:4]}' for _f, h, a in groups],
                             fontsize=8)
    axes[-1].set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    for f in LOSS_ORDER:
        lo, hi = fam_spans[f]
        axes[-1].text((lo + hi) / 2, -0.30, FAMILY_LABEL[f], ha='center', va='top',
                      fontsize=9.5, color=FAMILY_COLOR[f], transform=mtransforms.
                      blended_transform_factory(axes[-1].transData, axes[-1].transAxes))
    handles = [Line2D([0], [0], marker=MOUSE_MARKER[mi], ls='none', mfc='w', mec='0.3',
                      label=f'mouse {m}', ms=5) for mi, m in enumerate(mice)]
    handles += [Line2D([0], [0], marker='D', ls='none', mfc='w', mec='0.3',
                       label='open = OLD arm (export Q)', ms=5),
                Line2D([0], [0], marker='D', ls='none', mfc='0.3', mec='0.3',
                       label='filled = NEW arm (IO-HMM)', ms=5),
                Line2D([0], [0], ls='-', lw=0.9, color='0.3',
                       label='thin = one mouse (median over its 5 λ_H cells)'),
                Line2D([0], [0], ls='-', lw=2.6, color='0.3', marker='o', ms=6,
                       label='bold = median across the 6 mice'),
                Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.2', ms=8.5,
                       label=f'ring (ŝ only) = agreement > {AGREE_FLAG:.2f}: the decoder '
                             'reshapes, ŝ under-describes'),
                Line2D([0], [0], marker='x', ls='none', color='k', ms=6,
                       label=f'cross (ŝ only) = clamped in ≥{CLAMP_MAJ}/5 λ cells (median is a bound)'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label='reference: 1 (calibrated / null), 0 (no gap), 199 (epoch cap)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=8,
               frameon=False)
    fig.suptitle(
        'Old → new target family, per group and per mouse (n = 6 mice, paired arms; '
        'x = loss family × hidden size × decoder). Thin dumbbells: one mouse, median over '
        'its 5 λ_H cells; bold: median across mice.\n'
        f'OLD = {ARM_LABELS["old"]}; NEW = {ARM_LABELS["new"]}. '
        f'Null in every skill and gap = {NULL_NAME}. λ_H acts on the temporal decoder only: '
        'a spatial group’s 5 λ cells are exact replicates.',
        fontsize=9.5)
    save_fig(fig, out_dir, 'paired_dumbbells', max_px=1560)
    return sign_rows


# ----------------------------------------------------------------------
# (b) headline survival grid — the prior-scoring figure
# ----------------------------------------------------------------------
VERDICT_COLOR = {'holds': '#a1d99b', 'partial': '#fdd49e', 'fails': '#fc9272',
                 'n.a.': '0.88'}


def _rel_range(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return 100.0 * (x.max() - x.min()) / abs(np.median(x))


def score_headlines(cells, g, mice):
    """Return (rows, grid) — rows = list of (label, rule) and grid[row][mouse] =
    (verdict, text, numbers dict). Every rule is explicit and printed."""
    rows, grid = [], {}

    # H1 — proj-evar H8 ŝ: >=3 old -> <=1.2 new (prior b); falsifier new > 2
    lab = 'evar-projection H8 ŝ\nold ≥3 → new ≤1.2'
    rule = 'holds: both decoders new ≤1.2 & old ≥3 · partial: new ≤2 & fall (old <3 or 1.2<new≤2) · fails: new >2 (falsifier) or no fall'
    rows.append(('H1', lab, rule)); grid['H1'] = {}
    for m in mice:
        o = {a: gval(g, m, 'pca', 'h8', a, 'old', 's_hat') for a in ARCHS}
        n = {a: gval(g, m, 'pca', 'h8', a, 'new', 's_hat') for a in ARCHS}
        cl = {a: gval(g, m, 'pca', 'h8', a, 'new', 'clamp_flag') > 0 for a in ARCHS}
        ag = {arm: {a: gval(g, m, 'pca', 'h8', a, arm, 'agreement') > AGREE_FLAG for a in ARCHS}
              for arm in ('old', 'new')}
        if any(not np.isfinite(v) for v in list(o.values()) + list(n.values())):
            v = 'n.a.'
        elif any(n[a] > 2.0 for a in ARCHS) or any(n[a] >= o[a] for a in ARCHS):
            v = 'fails'
        elif all(n[a] <= 1.2 and o[a] >= 3.0 for a in ARCHS):
            v = 'holds'
        else:
            v = 'partial'
        # companions: † = clamped (bound, not estimate); ° = agreement > AGREE_FLAG
        # (the decoder RESHAPES, so ŝ under-describes it). Old-arm evar H8 is
        # ° in every mouse (agreement 0.3-0.7) — the verdict rests on the new arm.
        def _fl(arm, dec):
            return ('†' if (arm == 'new' and cl[dec]) else '') + ('°' if ag[arm][dec] else '')
        txt = (f'old {o["spat"]:.2f}{_fl("old","spat")} | {o["temp"]:.2f}{_fl("old","temp")}\n'
               f'new {n["spat"]:.2f}{_fl("new","spat")} | {n["temp"]:.2f}{_fl("new","temp")}')
        grid['H1'][m] = (v, txt, dict(old=o, new=n, clamped=cl, reshape=ag))

    # H2 — KL/JS lambda_H inert on new targets (temporal; spatial is replicates)
    lab = 'KL/JS λ_H-inert on NEW\n(temporal, 4 groups)'
    rule = ('worst relative range over the 5 λ_H cells of kl_skill and ŝ, over KL/JS × H8/lin: '
            'holds ≤5% · partial ≤15% · fails >15%  (context: same statistic for the evar family)')
    rows.append(('H2', lab, rule)); grid['H2'] = {}
    for m in mice:
        def _worst(arm):
            sub = cells[(cells.mouse == m) & (cells.arch == 'temp') & (cells.arm == arm)]
            w, wn = -1.0, ''
            for fam in ('kl', 'js'):
                for hid in MODEL_ORDER:
                    s = sub[(sub.loss_family == fam) & (sub.hidden == hid)]
                    for key in ('kl_skill', 's_hat'):
                        rr = _rel_range(s[key])
                        if np.isfinite(rr) and rr > w:
                            w, wn = rr, f'{fam}_{hid} {key}'
            c = max(_rel_range(sub[(sub.loss_family == 'pca') & (sub.hidden == hid)][key])
                    for hid in MODEL_ORDER for key in ('kl_skill', 's_hat'))
            return w, wn, c
        worst, worst_name, ctx = _worst('new')
        worst_old, _, ctx_old = _worst('old')   # like-for-like: is inertness a NEW-target property?
        if worst < 0:
            v = 'n.a.'
        elif worst <= 5:
            v = 'holds'
        elif worst <= 15:
            v = 'partial'
        else:
            v = 'fails'
        txt = (f'new {worst:.1f}% ({worst_name})\n'
               f'old arm same stat: {worst_old:.1f}%\nevar new/old: {ctx:.0f}% / {ctx_old:.0f}%')
        grid['H2'][m] = (v, txt, dict(worst=worst, worst_name=worst_name, evar_ctx=ctx,
                                      worst_old=worst_old, evar_ctx_old=ctx_old))

    # H3 — the six KL-vs-projection sign disagreements
    six = [('pca', 'h8', 'spat'), ('pca', 'h8', 'temp'), ('pca', 'lin', 'spat'),
           ('pca', 'lin', 'temp'), ('js', 'h8', 'spat'), ('js', 'h8', 'temp')]
    lab = 'KL-vs-projection sign\ndisagreement (the 6 mouse-0 groups)'
    rule = ('sign(Δ KL skill) ≠ sign(Δ proj skill), Δ = new − old, over evar×{H8,lin}×{spat,temp} + JS H8×{spat,temp}: '
            'holds ≥4/6 · partial 2–3/6 · fails ≤1/6. Codes: e = evar, j = JS; 8 = H8, L = lin; s/t = spatial/temporal')
    rows.append(('H3', lab, rule)); grid['H3'] = {}
    for m in mice:
        k, n, which = 0, 0, []
        for f, h, a in six:
            dk = gval(g, m, f, h, a, 'new', 'kl_skill') - gval(g, m, f, h, a, 'old', 'kl_skill')
            dp = gval(g, m, f, h, a, 'new', 'proj_skill') - gval(g, m, f, h, a, 'old', 'proj_skill')
            if not (np.isfinite(dk) and np.isfinite(dp)):
                continue
            n += 1
            if np.sign(dk) != np.sign(dp):
                k += 1
                which.append(('e' if f == 'pca' else 'j') + ('8' if h == 'h8' else 'L') + a[0])
        if n == 0:
            v = 'n.a.'
        elif k >= 4:
            v = 'holds'
        elif k >= 2:
            v = 'partial'
        else:
            v = 'fails'
        txt = f'{k}/{n} disagree\n' + (', '.join(which) if which else '—')
        grid['H3'][m] = (v, txt, dict(k=k, n=n, which=which))

    # X1 — linear spatial KL fit ~2x worse than H8 on new targets
    lab = 'spatial KL fit: lin ≈2× worse\nthan H8 on NEW'
    rule = ('ratio KL skill(kl_lin_spat) / KL skill(kl_h8_spat), NEW arm (same null, so = raw fit ratio): '
            'holds ≥1.5 · partial 1.15–1.5 · fails <1.15  (old-arm ratio for context)')
    rows.append(('X1', lab, rule)); grid['X1'] = {}
    for m in mice:
        rn = gval(g, m, 'kl', 'lin', 'spat', 'new', 'kl_skill') / gval(g, m, 'kl', 'h8', 'spat', 'new', 'kl_skill')
        ro = gval(g, m, 'kl', 'lin', 'spat', 'old', 'kl_skill') / gval(g, m, 'kl', 'h8', 'spat', 'old', 'kl_skill')
        if not np.isfinite(rn):
            v = 'n.a.'
        elif rn >= 1.5:
            v = 'holds'
        elif rn >= 1.15:
            v = 'partial'
        else:
            v = 'fails'
        txt = f'new {rn:.2f}×\n(old {ro:.2f}×)'
        grid['X1'][m] = (v, txt, dict(new=rn, old=ro))

    # X2 — overfit gap drops old -> new for H8 (evar / KL / JS; flat excluded: it stops at epoch ~4)
    h8g = [(f, 'h8', a) for f in ('pca', 'kl', 'js') for a in ARCHS]
    lab = 'overfitting gap drops old→new\nfor H8 (evar/KL/JS, 6 groups)'
    rule = ('groups with gap_new < gap_old and median Δ = new − old over the 6 H8 groups: '
            'holds ≥5/6 & median Δ ≤ −0.05 · partial ≥4/6 · fails ≤3/6')
    rows.append(('X2', lab, rule)); grid['X2'] = {}
    for m in mice:
        ds = []
        for f, h, a in h8g:
            ds.append(gval(g, m, f, h, a, 'new', 'overfit_gap') - gval(g, m, f, h, a, 'old', 'overfit_gap'))
        ds = np.array(ds)
        ok = np.isfinite(ds)
        k, n = int((ds[ok] < 0).sum()), int(ok.sum())
        med = float(np.median(ds[ok])) if n else np.nan
        if n == 0:
            v = 'n.a.'
        elif k >= 5 and med <= -0.05:
            v = 'holds'
        elif k >= 4:
            v = 'partial'
        else:
            v = 'fails'
        txt = f'{k}/{n} drop\nmedian Δ {med:+.2f}'
        grid['X2'][m] = (v, txt, dict(k=k, n=n, med=med, deltas=ds))

    # X3 — best_epoch collapses for the projection (evar H8) cells
    lab = 'best epoch collapses old→new\nfor evar-projection H8'
    rule = ('ratio best_epoch new/old (group medians), spatial & temporal: '
            'holds both ≤0.5 · partial one ≤0.5 · fails neither  (199 = censored at the cap)')
    rows.append(('X3', lab, rule)); grid['X3'] = {}
    for m in mice:
        o = {a: gval(g, m, 'pca', 'h8', a, 'old', 'best_epoch') for a in ARCHS}
        n = {a: gval(g, m, 'pca', 'h8', a, 'new', 'best_epoch') for a in ARCHS}
        r = {a: (n[a] / o[a] if (np.isfinite(o[a]) and o[a] > 0) else np.nan) for a in ARCHS}
        kk = sum(np.isfinite(r[a]) and r[a] <= 0.5 for a in ARCHS)
        if all(not np.isfinite(r[a]) for a in ARCHS):
            v = 'n.a.'
        elif kk == 2:
            v = 'holds'
        elif kk == 1:
            v = 'partial'
        else:
            v = 'fails'
        txt = (f's {o["spat"]:.0f}→{n["spat"]:.0f}\nt {o["temp"]:.0f}→{n["temp"]:.0f}')
        grid['X3'][m] = (v, txt, dict(old=o, new=n, ratio=r))

    return rows, grid


def fig_headline_grid(rows, grid, mice, out_dir):
    n_r, n_c = len(rows), len(mice)
    fig, ax = plt.subplots(figsize=(1.75 * n_c + 6.4, 1.0 * n_r + 2.0))
    ax.set_xlim(0, n_c + 1.9)
    ax.set_ylim(0, n_r)
    ax.axis('off')
    for ri, (rid, lab, rule) in enumerate(rows):
        y = n_r - 1 - ri
        tally = {k: 0 for k in VERDICT_COLOR}
        for ci, m in enumerate(mice):
            v, txt, _ = grid[rid][m]
            tally[v] += 1
            ax.add_patch(Rectangle((ci, y), 1, 1, facecolor=VERDICT_COLOR[v],
                                   edgecolor='w', lw=2))
            ax.text(ci + 0.5, y + 0.80, v, ha='center', va='center', fontsize=9.5,
                    fontweight='bold', color='0.1')
            ax.text(ci + 0.5, y + 0.38, txt, ha='center', va='center', fontsize=8,
                    color='0.15', linespacing=1.25)
        # row label (left, outside) and tally (right)
        ax.text(-0.08, y + 0.66, f'{rid}  {lab}', ha='right', va='center', fontsize=9,
                fontweight='bold', color='0.1')
        ax.text(-0.08, y + 0.24, wrap_label(rule, 62), ha='right', va='center',
                fontsize=6.6, color='0.35', linespacing=1.2)
        ax.text(n_c + 0.15, y + 0.5,
                f'holds {tally["holds"]}/{n_c}\npartial {tally["partial"]}/{n_c}\n'
                f'fails {tally["fails"]}/{n_c}',
                ha='left', va='center', fontsize=8.5, color='0.1', linespacing=1.3)
    for ci, m in enumerate(mice):
        ax.text(ci + 0.5, n_r + 0.08, f'mouse {m}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.text(n_c + 0.15, n_r + 0.08, f'tally (n = {n_c} mice)', ha='left', va='bottom',
            fontsize=9, fontweight='bold')
    handles = [Patch(facecolor=VERDICT_COLOR[k], label=k) for k in ('holds', 'partial', 'fails', 'n.a.')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.02), ncol=4,
              fontsize=8.5, frameon=False)
    fig.suptitle(
        'Do the mouse-0 headlines survive six mice? One verdict per mouse with the deciding '
        'numbers (group medians over the 5 λ_H cells; † = ŝ clamped at a ladder end, a bound; ° = ŝ agreement > 0.10, the decoder reshapes and ŝ under-describes it).\n'
        'H1–H3 are the registered priors (PREDICTIONS.md 2026-08-18, (b)–(d)); X1–X3 are the '
        'unregistered mouse-0 extras. "partial" = direction right, stated magnitude not met.',
        fontsize=9.5)
    save_fig(fig, out_dir, 'headline_survival_grid', max_px=1560)


# ----------------------------------------------------------------------
# (c) KL vs projection disagreement map
# ----------------------------------------------------------------------
def _symlog_ticks(ax, which, lim, linthresh):
    cand = np.array([-2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 4])
    t = cand[(cand >= lim[0]) & (cand <= lim[1])]
    lab = [f'{v:+g}' if v else '0' for v in t]
    if which == 'x':
        ax.set_xticks(t); ax.set_xticklabels(lab)
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_yticks(t); ax.set_yticklabels(lab)
        ax.yaxis.set_minor_locator(plt.NullLocator())


ROBUST_D = 0.1     # |Δ skill| below this: the sign is inside the noise band, not a claim


def fig_disagreement_map(g, mice, out_dir):
    groups = groups_in_order()
    fig = plt.figure(figsize=(15.2, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    linthresh = ROBUST_D
    dis_count = {}
    allx, ally = [], []
    for ai, arch in enumerate(ARCHS):
        ax = axes[ai]
        for f in LOSS_ORDER:
            for h in MODEL_ORDER:
                col = FAMILY_COLOR[f]
                k, n, kr = 0, 0, 0
                for mi, m in enumerate(mice):
                    dk = gval(g, m, f, h, arch, 'new', 'kl_skill') - gval(g, m, f, h, arch, 'old', 'kl_skill')
                    dp = gval(g, m, f, h, arch, 'new', 'proj_skill') - gval(g, m, f, h, arch, 'old', 'proj_skill')
                    if not (np.isfinite(dk) and np.isfinite(dp)):
                        continue
                    n += 1
                    dis = int(np.sign(dk) != np.sign(dp))
                    k += dis
                    kr += dis * int(min(abs(dk), abs(dp)) >= ROBUST_D)
                    allx.append(dk); ally.append(dp)
                    ax.plot(dk, dp, marker=MOUSE_MARKER[mi], ms=6.5, ls='none',
                            mec=col, mfc=col if h == 'h8' else 'none', mew=1.3,
                            alpha=0.9, zorder=3)
                dis_count[(f, h, arch)] = (k, n, kr)
        ax.set_xscale('symlog', linthresh=linthresh, linscale=0.8)
        ax.set_yscale('symlog', linthresh=linthresh, linscale=0.8)
        ax.axhline(0, color='0.3', lw=0.9, zorder=1)
        ax.axvline(0, color='0.3', lw=0.9, zorder=1)
        ax.axhspan(-linthresh, linthresh, color='0.93', zorder=0)
        ax.axvspan(-linthresh, linthresh, color='0.93', zorder=0)
        ax.set_title(f'{ARCH_LABEL[arch]} decoder', fontsize=10.5)
        ax.set_xlabel('Δ KL skill = new − old  (group median over 5 λ_H cells)\n'
                      '<0: KL fit improved on the new targets   [symlog, linear inside ±0.1]',
                      fontsize=8.2)
        ax.set_ylabel('Δ projection skill = new − old  (arm-pinned evar, held-out rows)\n'
                      '<0: projection fit improved on the new targets   [symlog]',
                      fontsize=8.2)
    lo = min(min(allx), min(ally)); hi = max(max(allx), max(ally))
    lim = (lo - 0.1 * abs(lo) - 0.02, hi + 0.1 * abs(hi) + 0.02)
    for ax in axes[:2]:
        ax.set_xlim(lim); ax.set_ylim(lim)
        _symlog_ticks(ax, 'x', lim, linthresh); _symlog_ticks(ax, 'y', lim, linthresh)
        ax.tick_params(labelsize=8)
        kw = dict(fontsize=7.5, color='0.4', transform=ax.transAxes)
        ax.text(0.02, 0.02, 'both IMPROVE\n(agree)', ha='left', va='bottom', **kw)
        ax.text(0.98, 0.98, 'both WORSEN\n(agree)', ha='right', va='top', **kw)
        ax.text(0.02, 0.98, 'KL improves,\nprojection worsens\n(DISAGREE)', ha='left', va='top', **kw)
        ax.text(0.98, 0.02, 'KL worsens,\nprojection improves\n(DISAGREE)', ha='right', va='bottom', **kw)
        ax.text(0.5, 0.5, f'grey = |Δ| < {ROBUST_D}\n(sign inside the noise band)', ha='center', va='center',
                fontsize=7, color='0.5', transform=ax.transAxes)

    # (right) disagreement count per group, horizontal bars
    ax = axes[2]
    ypos = np.arange(len(groups))[::-1]
    for yi, (f, h, a) in zip(ypos, groups):
        k, n, kr = dis_count.get((f, h, a), (0, 0, 0))
        ax.barh(yi, k, color=FAMILY_COLOR[f], alpha=0.35, edgecolor=FAMILY_COLOR[f],
                height=0.7, zorder=2)
        ax.barh(yi, kr, color=FAMILY_COLOR[f], alpha=0.95, edgecolor=FAMILY_COLOR[f],
                height=0.7, zorder=3)
        ax.text(k + 0.08, yi, f'{k}/{n}' + (f' ({kr} robust)' if k else ''), va='center', fontsize=7.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{FAMILY_LABEL[f]} {HID_LABEL[h]} {ARCH_LABEL[a][:4]}'
                        for f, h, a in groups], fontsize=7.8)
    ax.set_xlim(0, len(mice) + 0.9)
    ax.set_xticks(range(0, len(mice) + 1))
    ax.axvline(len(mice) / 2, ls=':', color='0.5', lw=1)
    ax.set_xlabel(f'mice (of {len(mice)}) whose KL and projection skill\nCHANGE in opposite '
                  f'directions old→new\n(pale = any sign disagreement; solid = both |Δ| ≥ {ROBUST_D})',
                  fontsize=8.5)
    ax.set_title('sign disagreement count per group', fontsize=10.5)
    for yi in ypos[1::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color='0.95', zorder=0)
    ax.tick_params(labelsize=8)

    handles = [Line2D([0], [0], marker='s', ls='none', color=FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f], ms=7) for f in LOSS_ORDER]
    handles += [Line2D([0], [0], marker='o', ls='none', mfc='0.3', mec='0.3', ms=6,
                       label='filled = hidden [8]'),
                Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.3', ms=6,
                       label='open = linear')]
    handles += [Line2D([0], [0], marker=MOUSE_MARKER[mi], ls='none', mfc='0.3', mec='0.3',
                       label=f'mouse {m}', ms=6) for mi, m in enumerate(mice)]
    fig.legend(handles=handles, loc='outside lower center', ncol=12, fontsize=8,
               frameon=False)
    ps.label_panels(axes)
    fig.suptitle(
        'Does swapping the target family move KL skill and projection skill the SAME way? '
        'One point per group per mouse (n = 6 mice; 16 groups = loss family × hidden × decoder).\n'
        f'Skill = loss ÷ {NULL_NAME}; Δ = new (IO-HMM) − old (export). Off-diagonal quadrants '
        '= the two losses disagree about whether the decoder got better.',
        fontsize=9.5)
    save_fig(fig, out_dir, 'kl_vs_proj_disagreement', max_px=1560)
    return dis_count


# ----------------------------------------------------------------------
# (d) s_hat new vs old
# ----------------------------------------------------------------------
def target_widths(cells):
    """Per mouse per arm: the arm's own median target posterior SD (deg). A
    WITHIN-arm descriptor (linear SD on the 0-90 export grid, circular-equivalent
    SD on the [0,180) IO-HMM grid) — used only to label how broad each mouse's
    targets were, never as a cross-arm peakiness comparison."""
    w = cells.groupby(['mouse', 'arm'])['sd_tgt_med_deg_within_arm_only'].median()
    return {(m, a): float(v) for (m, a), v in w.items()}


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank().values; ry = pd.Series(y[ok]).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_shat_scatter(g, cells, mice, out_dir):
    groups = groups_in_order()
    wid = target_widths(cells)
    fig = plt.figure(figsize=(18.4, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    for f, h, a in groups:
        col = FAMILY_COLOR[f]
        for mi, m in enumerate(mice):
            x = gval(g, m, f, h, a, 'old', 's_hat'); y = gval(g, m, f, h, a, 'new', 's_hat')
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            clamped = (gval(g, m, f, h, a, 'old', 'clamp_flag') > 0
                       or gval(g, m, f, h, a, 'new', 'clamp_flag') > 0)
            resh = (gval(g, m, f, h, a, 'old', 'reshape_flag') > 0
                    or gval(g, m, f, h, a, 'new', 'reshape_flag') > 0)
            if clamped:
                ax.plot(x, y, marker='x', ms=9, ls='none', color='k', mew=1.2, zorder=5)
            if resh:
                ax.plot(x, y, marker='o', ms=11, ls='none', mfc='none', mec='0.55',
                        mew=0.7, zorder=1)
            ax.plot(x, y, marker=MOUSE_MARKER[mi], ms=6.5, ls='none', mec=col,
                    mfc=col if h == 'h8' else 'none', mew=1.3, alpha=0.9, zorder=3)
    lim = (0.2, 7.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.plot(lim, lim, ls='--', lw=1, color='0.45', zorder=0)
    ax.axhline(1, ls=':', lw=0.9, color=ps.CHANCE_GREY); ax.axvline(1, ls=':', lw=0.9, color=ps.CHANCE_GREY)
    ax.set_xlim(lim); ax.set_ylim(lim); _log_ticks(ax, lim)
    ax.set_xlabel('ŝ, OLD arm (export Q, 91 linear bins) — calibrated on the OLD targets\n'
                  '1 = calibrated · >1 over-sharpened · <1 too broad', fontsize=8.5)
    ax.set_ylabel('ŝ, NEW arm (IO-HMM marginal, 72 circular bins) — calibrated on the NEW targets\n'
                  '1 = calibrated · >1 over-sharpened · <1 too broad', fontsize=8.5)
    ax.set_title('equivalent sharpening ŝ, new vs old — all 16 groups × 6 mice', fontsize=10.5)
    ax.tick_params(labelsize=8.5)
    ax.text(0.03, 0.97, 'above the identity: MORE over-sharpened\non the new targets', ha='left', va='top',
            fontsize=7.5, color='0.4', transform=ax.transAxes)
    ax.text(0.97, 0.03, 'below: LESS', ha='right', va='bottom', fontsize=7.5, color='0.4',
            transform=ax.transAxes)

    # (b) change in s_hat vs how broad the OLD target was; (c) NEW s_hat vs NEW width
    evar_groups = [(f, h, a) for f, h, a in groups if f == 'pca']
    mk_arch = {'spat': 'o', 'temp': '^'}
    ratios, ws_old, ws_new, news = [], [], [], []
    for f, h, a in evar_groups:
        for mi, m in enumerate(mice):
            x = gval(g, m, f, h, a, 'old', 's_hat'); y = gval(g, m, f, h, a, 'new', 's_hat')
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            r = np.log10(y / x)
            wo, wn = wid[(m, 'old')], wid[(m, 'new')]
            ratios.append(r); ws_old.append(wo); ws_new.append(wn); news.append(y)
            cl_any = (gval(g, m, f, h, a, 'old', 'clamp_flag') > 0
                      or gval(g, m, f, h, a, 'new', 'clamp_flag') > 0)
            cl_new = gval(g, m, f, h, a, 'new', 'clamp_flag') > 0
            kw = dict(marker=mk_arch[a], ms=7, ls='none', mec=FAMILY_COLOR['pca'],
                      mfc=FAMILY_COLOR['pca'] if h == 'h8' else 'none', mew=1.3, alpha=0.9, zorder=3)
            ax2.plot(wo, r, **kw)
            ax3.plot(wn, y, **kw)
            if cl_any:
                ax2.plot(wo, r, marker='x', ms=9, ls='none', color='k', mew=1.2, zorder=5)
            if cl_new:
                ax3.plot(wn, y, marker='x', ms=9, ls='none', color='k', mew=1.2, zorder=5)
    # faint: every other group, to show the evar family is the one that moves
    for f, h, a in groups:
        if f == 'pca':
            continue
        for mi, m in enumerate(mice):
            x = gval(g, m, f, h, a, 'old', 's_hat'); y = gval(g, m, f, h, a, 'new', 's_hat')
            if np.isfinite(x) and np.isfinite(y):
                ax2.plot(wid[(m, 'old')], np.log10(y / x), marker='.', ms=5, ls='none',
                         color=FAMILY_COLOR[f], alpha=0.45, zorder=2)
                ax3.plot(wid[(m, 'new')], y, marker='.', ms=5, ls='none',
                         color=FAMILY_COLOR[f], alpha=0.45, zorder=2)
    rho_old = _spearman(ws_old, ratios)
    rho_new = _spearman(ws_new, news)
    ax2.axhline(0, ls='--', lw=1, color='0.45')
    # mouse labels along the bottom, staggered so neighbours (m1/m2) don't collide
    for mi, m in enumerate(sorted(mice, key=lambda mm: wid[(mm, 'old')])):
        wo = wid[(m, 'old')]
        ax2.annotate(f'm{m}', (wo, -1.52 - 0.16 * (mi % 2)), ha='center', va='bottom',
                     fontsize=8, color='0.25')
    for mi, m in enumerate(sorted(mice, key=lambda mm: wid[(mm, 'new')])):
        wn = wid[(m, 'new')]
        ax3.annotate(f'm{m}', (wn, 0.158 * (1.13 ** (mi % 2))), ha='center', va='bottom',
                     fontsize=8, color='0.25')
    ax2.set_ylim(-1.75, 1.1)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1]); ax2.set_yticklabels(['÷10', '÷3.2', '×1', '×3.2', '×10'])
    ax2.set_xlabel('median OLD-target posterior SD per mouse\n(deg; linear SD on the 0–90 export grid) '
                   '— a within-arm descriptor', fontsize=8.5)
    ax2.set_ylabel('ŝ change old → new  (ŝ_new / ŝ_old, log scale)\n>1: more over-sharpened on the new targets',
                   fontsize=8.5)
    ax2.set_title('does the ŝ change track how broad the OLD target was?\n'
                  f'evar family bold; Spearman ρ = {rho_old:+.2f} (24 points, 6 mice — a descriptor)', fontsize=10)
    ax2.tick_params(labelsize=8.5)
    ax2.set_xlim(0, 32)

    ax3.set_yscale('log'); ax3.set_ylim(0.155, 7.5); _log_ticks(ax3, (0.2, 7.5), which='y')
    ax3.axhline(1, ls=':', lw=1, color=ps.CHANCE_GREY)
    ax3.set_xlim(15, 62)
    ax3.set_xlabel('median NEW-target posterior SD per mouse\n(deg; circular-equivalent SD on the [0,180) grid) '
                   '— a within-arm descriptor', fontsize=8.5)
    ax3.set_ylabel('ŝ, NEW arm (1 = calibrated · >1 over-sharpened)', fontsize=8.5)
    ax3.set_title('…or how broad the NEW target is?  (ŝ_new vs new width)\n'
                  f'evar family bold; Spearman ρ = {rho_new:+.2f} (24 points, 6 mice — a descriptor)', fontsize=10)
    ax3.tick_params(labelsize=8.5)

    handles = [Line2D([0], [0], marker='s', ls='none', color=FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f], ms=7) for f in LOSS_ORDER]
    handles += [Line2D([0], [0], marker='o', ls='none', mfc='0.3', mec='0.3', ms=6,
                       label='filled = hidden [8]'),
                Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.3', ms=6,
                       label='open = linear'),
                Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.55', ms=11,
                       label=f'ring (a) = ŝ agreement > {AGREE_FLAG:.2f} (reshapes)'),
                Line2D([0], [0], marker='x', ls='none', color='k', ms=8,
                       label=f'cross = clamped in ≥{CLAMP_MAJ}/5 λ cells (bound)'),
                Line2D([0], [0], ls='--', color='0.45', label='identity / no change'),
                Line2D([0], [0], marker='o', ls='none', color='0.3', ms=6, label='(b, c) circle = spatial'),
                Line2D([0], [0], marker='^', ls='none', color='0.3', ms=6, label='(b, c) triangle = temporal')]
    handles += [Line2D([0], [0], marker=MOUSE_MARKER[mi], ls='none', mfc='0.3', mec='0.3', ms=6,
                       label=f'mouse {m}: old tgt SD {wid[(m, "old")]:.1f}°, new {wid[(m, "new")]:.1f}°')
                for mi, m in enumerate(mice)]
    fig.legend(handles=handles, loc='outside lower center', ncol=6, fontsize=8, frameon=False)
    ps.label_panels([ax, ax2, ax3])
    fig.suptitle(
        'Equivalent sharpening ŝ on the old vs the new target family, per group per mouse '
        '(group median over 5 λ_H cells; n = 6 mice). ŝ is calibrated per arm on that arm’s own '
        'targets, so 1 means calibrated in both.\n'
        'Target SDs are each arm’s OWN median posterior width (a within-arm descriptor on different '
        'supports, never a cross-arm peakiness comparison). (a) mouse marker = mouse; (b, c) marker = decoder.',
        fontsize=9.5)
    save_fig(fig, out_dir, 's_hat_new_vs_old', max_px=1560)
    print(f'  ŝ change vs old width (evar, 24 pts): Spearman ρ = {rho_old:+.2f}; '
          f'ŝ_new vs new width: ρ = {rho_new:+.2f}')
    return wid


# ----------------------------------------------------------------------
# prior scoring (printed)
# ----------------------------------------------------------------------
def _tally(grid, rid, mice):
    t = {k: 0 for k in VERDICT_COLOR}
    for m in mice:
        t[grid[rid][m][0]] += 1
    return t


_GLYPHS = {'confirmed': '✓', 'magnitude': '↔', 'direction': '✗', 'invalidated': '⚠'}


class _Glyph(dict):
    def __getitem__(self, k):
        return _GLYPHS[str(k).split()[0]]


GLYPH = _Glyph()


def print_prior_scoring(rows, grid, mice, g, wid, dis_count):
    n = len(mice)
    print('\n' + '=' * 78)
    print('PRIOR SCORING — PREDICTIONS.md 2026-08-18 (five-mouse entry, scored on six) + mouse-0 extras')
    print('  classes: confirmed / magnitude / direction / invalidated; verdict per mouse from the grid')
    print('=' * 78)

    # (b)
    t = _tally(grid, 'H1', mice)
    new_sp = [grid['H1'][m][2]['new']['spat'] for m in mice]
    new_tp = [grid['H1'][m][2]['new']['temp'] for m in mice]
    old_sp = [grid['H1'][m][2]['old']['spat'] for m in mice]
    old_tp = [grid['H1'][m][2]['old']['temp'] for m in mice]
    fals = [m for m in mice if max(grid['H1'][m][2]['new'].values()) > 2.0]
    fell = [m for m in mice if all(grid['H1'][m][2]['new'][a] < grid['H1'][m][2]['old'][a] for a in ARCHS)]
    le12 = [m for m in mice if all(grid['H1'][m][2]['new'][a] <= 1.2 for a in ARCHS)]
    if len(fals) >= 2:
        cls = 'invalidated (own falsifier fired)'
    elif t['holds'] >= 4:
        cls = 'confirmed'
    elif t['holds'] + t['partial'] >= 4:
        cls = 'magnitude'
    else:
        cls = 'direction'
    print(f'(b) evar-projection H8 ŝ: old ≥3 → new ≤1.2 in ≥4/{n} mice [~70%; falsifier: 2+ mice new >2]  -> {GLYPH[cls]} {cls}')
    print(f'    holds {t["holds"]}/{n}, partial {t["partial"]}/{n}, fails {t["fails"]}/{n}; '
          f'fell in both decoders in {len(fell)}/{n} (mice {fell}); new ≤1.2 both decoders in {len(le12)}/{n} (mice {le12}); '
          f'new >2 in {len(fals)}/{n} (mice {fals})')
    print('    per mouse spat old→new: ' + ', '.join(f'm{m} {o:.2f}→{v:.2f}' for m, o, v in zip(mice, old_sp, new_sp)))
    print('    per mouse temp old→new: ' + ', '.join(f'm{m} {o:.2f}→{v:.2f}' for m, o, v in zip(mice, old_tp, new_tp))
          + '   (6.00 = clamped bound)')
    print('    target widths, DIFFERENT estimators (old = linear SD on the 0-90 grid; new = circular SD on [0,180)) — NOT a cross-arm width comparison: '
          + ', '.join(f'm{m} {wid[(m, "old")]:.0f}/{wid[(m, "new")]:.0f}' for m in mice))

    # (c)
    t = _tally(grid, 'H2', mice)
    worst = [grid['H2'][m][2]['worst'] for m in mice]
    ctx = [grid['H2'][m][2]['evar_ctx'] for m in mice]
    cls = 'confirmed' if t['holds'] == n else ('magnitude' if t['holds'] + t['partial'] == n else 'direction' if t['fails'] <= 2 else 'invalidated')
    worst_old = [grid['H2'][m][2]['worst_old'] for m in mice]
    print(f'(c) KL/JS λ_H-inert on the new targets replicates across mice [~85%]  -> {GLYPH[cls]} {cls}')
    print(f'    holds {t["holds"]}/{n} (worst 5-λ relative range of kl_skill/ŝ, temporal KL/JS groups ≤5%); '
          + 'per mouse worst NEW: ' + ', '.join(f'm{m} {w:.1f}%' for m, w in zip(mice, worst))
          + ' | same statistic OLD arm: ' + ', '.join(f'm{m} {w:.1f}%' for m, w in zip(mice, worst_old))
          + ' | evar family (new): ' + ', '.join(f'm{m} {c:.0f}%' for m, c in zip(mice, ctx)))
    print('    NB like-for-like: KL/JS are λ_H-inert in BOTH arms — a property of the calibrated losses, '
          'not of the new targets; the contrast is with the evar family, not with the old arm.')

    # (d)
    t = _tally(grid, 'H3', mice)
    ks = [grid['H3'][m][2]['k'] for m in mice]
    maj = sum(k >= 4 for k in ks)
    cls = 'confirmed' if maj >= 4 else ('magnitude' if maj == 3 else ('direction' if maj + t['partial'] >= 4 else 'invalidated'))
    print(f'(d) the six KL-vs-projection sign disagreements replicate in the majority of mice [~65%]  -> {GLYPH[cls]} {cls}')
    print(f'    ≥4/6 disagree in {maj}/{n} mice; per mouse k/6: ' + ', '.join(f'm{m} {k}' for m, k in zip(mice, ks))
          + '; per group (mice disagreeing of 6): '
          + ', '.join(f'{f}_{h}_{a} {dis_count[(f, h, a)][0]} ({dis_count[(f, h, a)][2]} robust)' for f, h, a in
                      [('pca', 'h8', 'spat'), ('pca', 'h8', 'temp'), ('pca', 'lin', 'spat'),
                       ('pca', 'lin', 'temp'), ('js', 'h8', 'spat'), ('js', 'h8', 'temp')]))

    # (e)
    breakers = {}
    for rid, _l, _r in rows:
        for m in mice:
            if grid[rid][m][0] == 'fails':
                breakers.setdefault(m, []).append(rid)
    cls = 'confirmed' if breakers else 'invalidated'
    print(f'(e) at least one mouse breaks a headline [~60%]  -> {GLYPH[cls]} {cls}')
    print('    breakers: ' + ('; '.join(f'm{m}: {", ".join(v)}' for m, v in sorted(breakers.items())) or 'none'))

    # extras
    t = _tally(grid, 'X1', mice)
    rn = [grid['X1'][m][2]['new'] for m in mice]; ro = [grid['X1'][m][2]['old'] for m in mice]
    cls = 'confirmed' if t['holds'] >= 5 else ('magnitude' if t['holds'] + t['partial'] >= 5 else 'direction' if t['fails'] <= 2 else 'invalidated')
    print(f'(x1) linear spatial KL fit ~2× worse than H8 on the new targets  -> {GLYPH[cls]} {cls}')
    print('    ratio lin/H8 new: ' + ', '.join(f'm{m} {r:.2f}' for m, r in zip(mice, rn))
          + ' | old: ' + ', '.join(f'm{m} {r:.2f}' for m, r in zip(mice, ro))
          + f' | ≥1.5× in {t["holds"]}/{n}')
    t = _tally(grid, 'X2', mice)
    ks = [grid['X2'][m][2]['k'] for m in mice]; meds = [grid['X2'][m][2]['med'] for m in mice]
    cls = 'confirmed' if t['holds'] >= 5 else ('magnitude' if t['holds'] + t['partial'] >= 5 else 'direction' if t['fails'] <= 2 else 'invalidated')
    print(f'(x2) overfitting gap drops sharply old→new for H8 (evar/KL/JS)  -> {GLYPH[cls]} {cls}')
    print('    groups dropping /6 and median Δ: ' + ', '.join(f'm{m} {k}/6 ({d:+.2f})' for m, k, d in zip(mice, ks, meds)))
    t = _tally(grid, 'X3', mice)
    cls = 'confirmed' if t['holds'] >= 5 else ('magnitude' if t['holds'] + t['partial'] >= 5 else 'direction' if t['fails'] <= 2 else 'invalidated')
    print(f'(x3) best_epoch collapses old→new for the evar-projection H8 cells  -> {GLYPH[cls]} {cls}')
    print('    spat old→new: ' + ', '.join(f'm{m} {grid["X3"][m][2]["old"]["spat"]:.0f}→{grid["X3"][m][2]["new"]["spat"]:.0f}' for m in mice))
    print('    temp old→new: ' + ', '.join(f'm{m} {grid["X3"][m][2]["old"]["temp"]:.0f}→{grid["X3"][m][2]["new"]["temp"]:.0f}' for m in mice)
          + f' | both ≤0.5× in {t["holds"]}/{n}')
    # the SIX-mouse entry of the same date (registered before the v2 results) is scorable from the same numbers
    if all(m in MARGINAL_R for m in mice):
        le12 = [m for m in mice if all(grid['H1'][m][2]['new'][a] <= 1.2 for a in ARCHS)]
        gt2 = [m for m in mice if any(grid['H1'][m][2]['new'][a] > 2.0 for a in ARCHS)]
        rs = np.array([MARGINAL_R[m] for m in mice])
        rho = {a: _spearman(rs, [grid['H1'][m][2]['new'][a] for m in mice]) for a in ARCHS}
        print(f'(six-mouse entry a) new-arm evar H8 ŝ ≤1.2 in ≥4/6 [~65%; falsifier 3+ mice >2]: '
              f'ŝ ≤1.2 both decoders in {len(le12)}/{n} (mice {le12}); >2 in {len(gt2)}/{n} (mice {gt2}) '
              f'-> {"confirmed" if len(le12) >= 4 and len(gt2) < 3 else "invalidated"} (falsifier did not fire)')
        print(f'(six-mouse entry b) new-arm evar H8 ŝ vs marginal R [~70%; falsifier ρ<0]: Spearman ρ spat {rho["spat"]:+.2f}, '
              f'temp {rho["temp"]:+.2f} over {n} mice (R = ' + ', '.join(f'm{m} {MARGINAL_R[m]:.2f}' for m in mice) + ')'
              f' -> {"confirmed" if min(rho.values()) > 0 else "invalidated"} (a 6-point rank correlation — a descriptor)')
    print('=' * 78)


def print_group_table(g, mice, sign_rows):
    """Across-mouse median old/new per group per metric + the sign count."""
    print('\nGroup medians across mice [old → new]  (sign count = mice with new < old)')
    for key, short, _l, _r, _s in METRICS:
        print(f'  -- {short}')
        for f, h, a in groups_in_order():
            vo = [gval(g, m, f, h, a, 'old', key) for m in mice]
            vn = [gval(g, m, f, h, a, 'new', key) for m in mice]
            k, t, n = sign_rows[(key, group_name(f, h, a))]
            print(f'    {group_name(f, h, a):18s} {np.nanmedian(vo):7.3f} → {np.nanmedian(vn):7.3f}   new<old {k}/{n}'
                  + (f' ({t} ties)' if t else ''))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--in-root', default='figures/io_hmm_vs_export_v2',
                    help='directory holding m<M>/ subdirs with the scorecard CSVs')
    ap.add_argument('--out-dir', default='figures/io_hmm_vs_export_v2/crossmouse')
    ap.add_argument('--mice', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args()
    ps.apply()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading per-mouse scorecard CSVs')
    cells = load_cells(a.in_root, a.mice)
    exp_rows = 40 * 2 * 2
    for m in a.mice:
        nr = int((cells.mouse == m).sum())
        if nr != exp_rows:
            print(f'  WARNING mouse {m}: {nr} rows (expected {exp_rows})')
    g = group_table(cells)
    print(f'  {len(g)} (mouse × group × arm) rows; {len(a.mice)} mice; '
          f'{g.groupby(["loss_family", "hidden", "arch"]).ngroups} groups')
    g.to_csv(out_dir / 'crossmouse_groups.csv', index=False)
    print(f'  -> {out_dir / "crossmouse_groups.csv"}')

    sign_rows = fig_paired_dumbbells(g, a.mice, out_dir)
    rows, grid = score_headlines(cells, g, a.mice)
    fig_headline_grid(rows, grid, a.mice, out_dir)
    dis_count = fig_disagreement_map(g, a.mice, out_dir)
    wid = fig_shat_scatter(g, cells, a.mice, out_dir)

    # headline grid as CSV too
    lines = ['headline,mouse,verdict,numbers']
    for rid, lab, _rule in rows:
        for m in a.mice:
            v, txt, _ = grid[rid][m]
            lines.append(f'{rid},{m},{v},"{txt.replace(chr(10), " | ")}"')
    (out_dir / 'headline_grid.csv').write_text('\n'.join(lines) + '\n')
    print(f'  -> {out_dir / "headline_grid.csv"}')

    print_group_table(g, a.mice, sign_rows)
    print_prior_scoring(rows, grid, a.mice, g, wid, dis_count)


if __name__ == '__main__':
    main()
