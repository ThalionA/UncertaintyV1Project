# -*- coding: utf-8 -*-
"""io_hmm_v3 width sweep — the LOSS-FAMILY figures (lambda_H = 0; h8 and lin).

Reads the settled per-cell scores in figures/io_hmm_wide/cells.csv (written by
diagnostics/io_hmm_wide_extract.py from the scorecard's own functions — nothing
is re-scored here) and draws three figures under figures/io_hmm_wide/:

  loss_family_skill      (a) five loss families on x; KL skill (top row) and
                             projection skill (bottom row); spatial | temporal
                             columns; per-mouse points (h8 filled, lin open) +
                             a median bar per (family, arch); reference at 1.
  calibrated_head_to_head (b) CE vs KL vs JS, paired per mouse: KL skill,
                             projection skill and s_hat at h8 and at h64, for
                             both decoders, plus the per-mouse relative spread
                             max|pairwise diff| / mean across the three (the
                             prior-(d) quantity) against the 5 % reference.
  flat_vs_evar           (c) the two projection weightings — flat vs evar —
                             per mouse at lin and h8, with the KL cell of the
                             same arch as the calibrated reference; KL skill,
                             projection skill, s_hat side by side.
  flat_vs_evar_ladder    (c') the same three families across the full width
                             ladder (lin, rr8, h4 … h64), per-mouse lines.

Prior (d) (PREDICTIONS.md 2026-08-22): ce ~= kl ~= js within ~5 %. Scored on
stdout per mouse x decoder x width x metric as the relative spread above; the
verdict counts mice (n = 6 is the unit) — no p-values.

Conventions: figsave.save_fig (png + svg, PNG <= 1600 px); per-mouse points on
every figure; every axis label states its normalisation; no prose boxes on the
data. Skill ratios and s_hat are drawn on log axes (ratios; 1 is the natural
centre, the evar family spans ~0.8–11).

Usage (from nn_decoder/):
    python diagnostics/io_hmm_wide_loss.py
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import peakiness_style as ps                          # noqa: E402
from figsave import save_fig                          # noqa: E402
from io_hmm_vs_export_scorecard import (              # noqa: E402
    FAMILY_COLOR as _FC, FAMILY_LABEL as _FL, NULL_NAME)

OUT_DIR = _HERE.parent / 'figures' / 'io_hmm_wide'
CSV = OUT_DIR / 'cells.csv'

MICE = tuple(range(6))
MOUSE_MARKER = ['o', 's', '^', 'v', 'D', 'P']        # same assignment as crossmouse
LOSSES = ('pca', 'pcaflat', 'kl', 'js', 'ce')
FAMILY_COLOR = dict(_FC, ce=ps.CE)
FAMILY_LABEL = dict(_FL, ce='CE')
SHORT = {'pca': 'proj.\n(evar)', 'pcaflat': 'proj.\n(flat)', 'kl': 'KL', 'js': 'JS',
         'ce': 'CE'}
DECODERS = ('spat', 'temp')
DEC_LABEL = {'spat': 'spatial decoder', 'temp': 'temporal decoder'}
LADDER = ('lin', 'rr8', 'h4', 'h8', 'h16', 'h32', 'h64')
CALIBRATED = ('ce', 'kl', 'js')
SPREAD_TOL = 0.05                                     # prior (d): "within ~5 %"

# (key, y-axis label carrying the normalisation, reference value)
METRICS = {
    'kl_skill': ('KL skill\nmean KL(tgt‖dec) ÷ mean KL(tgt‖LOO predict-mean)\n'
                 'held-out trials; <1 beats the null', 1.0),
    'proj_skill': ('projection skill\nproj. loss ÷ proj. loss(LOO predict-mean)\n'
                   'evar pinned from js_h8_lh0; <1 beats the null', 1.0),
    's_hat': ('equivalent sharpening ŝ\n(calibration curve built from the targets,\n'
              'one per mouse); 1 = calibrated, >1 over-sharpened', 1.0),
}
METRIC_SHORT = {'kl_skill': 'KL skill', 'proj_skill': 'projection skill',
                's_hat': 'ŝ'}


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load(csv=CSV):
    c = pd.read_csv(csv)
    c = c[c.lambda_H == 0].copy()
    n_exp = len(LOSSES) * len(LADDER) * len(MICE) * len(DECODERS)
    if len(c) != n_exp:
        raise SystemExit(f'ABORT: expected {n_exp} lambda=0 rows, found {len(c)}')
    c['mouse'] = c['mouse'].astype(int)
    return c.set_index(['loss_family', 'arch', 'decoder', 'mouse']).sort_index()


def val(c, loss, arch, dec, key):
    """Per-mouse vector (len 6, mouse order) of one metric."""
    return np.array([c.loc[(loss, arch, dec, m), key] for m in MICE], float)


def clamped(c, loss, arch, dec):
    return np.array([c.loc[(loss, arch, dec, m), 's_hat_clamped'] for m in MICE], int) > 0


LOG_TICKS = np.array([0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
LOG_TICKS_FINE = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                           1.5, 1.75, 2.0, 2.5, 3.0])


def _y_log(ax, ys, ref=1.0):
    """Log y with plain decimal ticks (the scorecard's ``_log_ticks`` stops at 6;
    the evar family reaches 11 here)."""
    ys = np.asarray(ys, float)
    lo, hi = np.nanmin(ys), np.nanmax(ys)
    lo, hi = min(lo, ref) / 1.15, max(hi, ref) * 1.15
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    cand = LOG_TICKS_FINE if hi / lo < 2.2 else LOG_TICKS
    t = cand[(cand >= lo) & (cand <= hi)]
    ax.set_yticks(t)
    ax.set_yticklabels([f'{v:g}' for v in t])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _mouse_points(ax, x, ys, color, filled=True, clamp=None, zorder=3, ms=5.5, jitter=0.0):
    """Six per-mouse markers at x (mouse-specific marker shapes); clamped
    values are overplotted with a black cross (the number is a bound)."""
    for i, (m, y) in enumerate(zip(MICE, ys)):
        xi = x + jitter * (i - 2.5) / 2.5
        ax.plot(xi, y, marker=MOUSE_MARKER[m], ms=ms, ls='none',
                mfc=color if filled else 'none', mec=color, mew=1.2, zorder=zorder)
        if clamp is not None and clamp[i]:
            ax.plot(xi, y, marker='x', ms=ms + 2, ls='none', color='k', mew=1.0,
                    zorder=zorder + 1)


def _median_bar(ax, x, ys, color, w=0.22, lw=2.6, zorder=2):
    med = float(np.median(ys))
    ax.plot([x - w, x + w], [med, med], '-', lw=lw, color=color, zorder=zorder,
            solid_capstyle='butt')
    return med


def _mouse_legend(ax, loc='upper left', **kw):
    handles = [Line2D([0], [0], marker=MOUSE_MARKER[m], ls='none', color='0.35',
                      ms=5.5, label=f'mouse {m}') for m in MICE]
    return ax.legend(handles=handles, loc=loc, ncol=3, handletextpad=0.3,
                     columnspacing=0.8, **kw)


# ----------------------------------------------------------------------
# (a) five loss families, KL skill x projection skill, h8 filled / lin open
# ----------------------------------------------------------------------
def fig_loss_family_skill(c, out_dir):
    keys = ('kl_skill', 'proj_skill')
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True,
                             sharex=True)
    xs = np.arange(len(LOSSES))
    off = {'h8': -0.17, 'lin': 0.17}
    for r, key in enumerate(keys):
        lab, ref = METRICS[key]
        for col, dec in enumerate(DECODERS):
            ax = axes[r, col]
            all_y = []
            for xi, loss in zip(xs, LOSSES):
                colr = FAMILY_COLOR[loss]
                for arch in ('h8', 'lin'):
                    ys = val(c, loss, arch, dec, key)
                    all_y.append(ys)
                    _mouse_points(ax, xi + off[arch], ys, colr, filled=(arch == 'h8'))
                    _median_bar(ax, xi + off[arch], ys, colr)
            ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=1)
            _y_log(ax, np.concatenate(all_y), ref)
            ax.set_xticks(xs)
            ax.set_xticklabels([SHORT[l] for l in LOSSES])
            ax.set_xlim(-0.6, len(LOSSES) - 0.4)
            if col == 0:
                ax.set_ylabel(lab, fontsize=9.5)
            if r == 0:
                ax.set_title(DEC_LABEL[dec])
            if r == 1:
                ax.set_xlabel('training loss family')
            for xi in xs[:-1]:
                ax.axvline(xi + 0.5, color='0.92', lw=0.8, zorder=0)
    ps.label_panels(axes)
    handles = [Line2D([0], [0], marker='o', ls='none', color='0.3', mfc='0.3',
                      ms=6, label='h8 (one tanh layer, width 8) — filled'),
               Line2D([0], [0], marker='o', ls='none', color='0.3', mfc='none',
                      ms=6, mew=1.2, label='lin (single Linear) — open'),
               Line2D([0], [0], ls='-', lw=2.6, color='0.3', label='median over 6 mice'),
               Line2D([0], [0], ls=':', color=ps.CHANCE_GREY,
                      label=f'1 = the null ({NULL_NAME})')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, fontsize=8.5)
    _mouse_legend(axes[0, 1], loc='upper right', fontsize=7.5)
    fig.suptitle('io_hmm_v3, λ_H = 0: skill by training loss family — h8 (filled) and '
                 'lin (open), per mouse + median', fontsize=11.5)
    save_fig(fig, out_dir, 'loss_family_skill', max_px=1560)


# ----------------------------------------------------------------------
# (b) CE vs KL vs JS head-to-head, paired per mouse, at h8 and h64
# ----------------------------------------------------------------------
def rel_spread(c, arch, dec, key):
    """Per-mouse max |pairwise difference| / mean across CE, KL, JS."""
    v = np.stack([val(c, l, arch, dec, key) for l in CALIBRATED])   # 3 x 6
    spread = v.max(0) - v.min(0)
    return spread / v.mean(0)


def fig_calibrated_head_to_head(c, out_dir):
    widths = ('h8', 'h64')
    keys = ('kl_skill', 'proj_skill', 's_hat')
    rows = [(w, d) for w in widths for d in DECODERS]
    fig, axes = plt.subplots(len(rows), len(keys) + 1, figsize=(13.6, 11.8),
                             constrained_layout=True)
    xs = np.arange(len(CALIBRATED))
    spread_rows = []
    for r, (arch, dec) in enumerate(rows):
        for col, key in enumerate(keys):
            ax = axes[r, col]
            lab, ref = METRICS[key]
            vs = np.stack([val(c, l, arch, dec, key) for l in CALIBRATED])
            cl = np.stack([clamped(c, l, arch, dec) for l in CALIBRATED])
            for m in MICE:
                ax.plot(xs, vs[:, m], '-', lw=0.8, color='0.6', zorder=2)
            for xi, loss in zip(xs, CALIBRATED):
                _mouse_points(ax, xi, vs[xi], FAMILY_COLOR[loss], clamp=cl[xi])
                _median_bar(ax, xi, vs[xi], FAMILY_COLOR[loss], w=0.18)
            ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=1)
            _y_log(ax, vs, ref)
            ax.set_xticks(xs)
            ax.set_xticklabels([FAMILY_LABEL[l] for l in CALIBRATED])
            ax.set_xlim(-0.5, len(CALIBRATED) - 0.5)
            ax.set_ylabel(lab if r == 0 else METRIC_SHORT[key], fontsize=8.5)
            ax.set_title(f'{arch} · {DEC_LABEL[dec]} · {METRIC_SHORT[key]}', fontsize=10)
        # right column: relative spread per mouse per metric
        ax = axes[r, -1]
        for k, key in enumerate(keys):
            sp = rel_spread(c, arch, dec, key)
            for m in MICE:
                spread_rows.append(dict(arch=arch, decoder=dec, metric=key, mouse=m,
                                        rel_spread=sp[m]))
            _mouse_points(ax, k, 100 * sp, '0.25', jitter=0.18)
            _median_bar(ax, k, 100 * sp, '0.25', w=0.28)
        ax.axhline(100 * SPREAD_TOL, ls='--', lw=1.0, color=ps.PCA_EVAR, zorder=1)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([METRIC_SHORT[k] for k in keys])
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_ylim(0, max(8.0, 100 * max(1e-9, float(np.nanmax(
            [rel_spread(c, arch, dec, k).max() for k in keys]))) * 1.15))
        ax.set_ylabel('spread across CE / KL / JS\nmax |pairwise diff| ÷ mean  (%)'
                      if r == 0 else 'spread (%)', fontsize=8.5)
        ax.set_title(f'{arch} · {DEC_LABEL[dec]} · spread', fontsize=10)
    handles = [Line2D([0], [0], ls='-', lw=0.8, color='0.6',
                      label='one mouse, paired across the three losses'),
               Line2D([0], [0], ls='-', lw=2.6, color='0.3', label='median over 6 mice'),
               Line2D([0], [0], marker='x', ls='none', color='k', ms=7,
                      label='ŝ inversion clamped (bound)'),
               Line2D([0], [0], ls=':', color=ps.CHANCE_GREY,
                      label='1 = null (skill) / calibrated (ŝ)'),
               Line2D([0], [0], ls='--', color=ps.PCA_EVAR,
                      label=f'prior (d) tolerance: {100 * SPREAD_TOL:.0f} %')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8.5)
    _mouse_legend(axes[0, 0], loc='lower left', fontsize=7)
    fig.suptitle('io_hmm_v3, λ_H = 0: the three calibrated losses head-to-head '
                 '(CE vs KL vs JS), paired per mouse at h8 and h64', fontsize=11.5)
    save_fig(fig, out_dir, 'calibrated_head_to_head', max_px=1560)
    return pd.DataFrame(spread_rows)


# ----------------------------------------------------------------------
# (c) flat vs evar projection (KL of the same arch as reference), lin and h8
# ----------------------------------------------------------------------
TRIO = ('pcaflat', 'pca', 'kl')
TRIO_SHORT = {'pcaflat': 'flat', 'pca': 'evar', 'kl': 'KL'}


def fig_flat_vs_evar(c, out_dir):
    keys = ('kl_skill', 'proj_skill', 's_hat')
    archs = ('lin', 'h8')
    fig, axes = plt.subplots(len(DECODERS), len(keys), figsize=(13.2, 8.2),
                             constrained_layout=True)
    gap = 0.9
    xpos = {}
    for ai, arch in enumerate(archs):
        for ti, loss in enumerate(TRIO):
            xpos[(arch, loss)] = ai * (len(TRIO) + gap) + ti
    for r, dec in enumerate(DECODERS):
        for col, key in enumerate(keys):
            ax = axes[r, col]
            lab, ref = METRICS[key]
            all_y = []
            for arch in archs:
                vs = {l: val(c, l, arch, dec, key) for l in TRIO}
                cl = {l: clamped(c, l, arch, dec) for l in TRIO}
                xx = [xpos[(arch, l)] for l in TRIO]
                for m in MICE:
                    ax.plot(xx[:2], [vs['pcaflat'][m], vs['pca'][m]], '-', lw=0.8,
                            color='0.6', zorder=2)
                for l in TRIO:
                    _mouse_points(ax, xpos[(arch, l)], vs[l], FAMILY_COLOR[l],
                                  filled=(arch == 'h8'), clamp=cl[l])
                    _median_bar(ax, xpos[(arch, l)], vs[l], FAMILY_COLOR[l], w=0.2)
                    all_y.append(vs[l])
                ax.text(np.mean(xx), 1.0, arch, transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=9.5, color='0.25')
            ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=1)
            _y_log(ax, np.concatenate(all_y), ref)
            ticks = [xpos[(a, l)] for a in archs for l in TRIO]
            ax.set_xticks(ticks)
            ax.set_xticklabels([TRIO_SHORT[l] for a in archs for l in TRIO])
            ax.set_xlim(-0.6, max(ticks) + 0.6)
            ax.set_ylabel(lab, fontsize=8.5)
            ax.set_title(f'{DEC_LABEL[dec]} · {METRIC_SHORT[key]}', fontsize=10, pad=16)
            if r == 1 and col == 1:
                ax.set_xlabel('projection weighting (flat · evar); KL = same-arch reference')
    ps.label_panels(axes)
    handles = [Line2D([0], [0], marker='s', ls='none', color=FAMILY_COLOR[l],
                      label=FAMILY_LABEL[l], ms=6) for l in TRIO]
    handles += [Line2D([0], [0], marker='o', ls='none', color='0.3', mfc='none', ms=6,
                       mew=1.2, label='lin — open'),
                Line2D([0], [0], marker='o', ls='none', color='0.3', mfc='0.3', ms=6,
                       label='h8 — filled'),
                Line2D([0], [0], ls='-', lw=0.8, color='0.6',
                       label='one mouse, flat → evar paired'),
                Line2D([0], [0], ls='-', lw=2.6, color='0.3', label='median over 6 mice'),
                Line2D([0], [0], marker='x', ls='none', color='k', ms=7,
                       label='ŝ inversion clamped (bound)'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY,
                       label='1 = null (skill) / calibrated (ŝ)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8.5)
    _mouse_legend(axes[0, 0], loc='upper right', fontsize=7)
    fig.suptitle('io_hmm_v3, λ_H = 0: flat vs evar projection weighting per mouse, '
                 'lin and h8, with KL as the calibrated reference', fontsize=11.5)
    save_fig(fig, out_dir, 'flat_vs_evar', max_px=1560)


def fig_flat_vs_evar_ladder(c, out_dir):
    keys = ('kl_skill', 'proj_skill', 's_hat')
    fig, axes = plt.subplots(len(keys), len(DECODERS), figsize=(11.0, 10.2),
                             constrained_layout=True, sharex=True)
    xs = np.arange(len(LADDER))
    off = {'pcaflat': -0.2, 'pca': 0.0, 'kl': 0.2}
    for r, key in enumerate(keys):
        lab, ref = METRICS[key]
        for col, dec in enumerate(DECODERS):
            ax = axes[r, col]
            all_y = []
            for l in TRIO:
                colr = FAMILY_COLOR[l]
                vs = np.stack([val(c, l, a, dec, key) for a in LADDER])      # 7 x 6
                cl = np.stack([clamped(c, l, a, dec) for a in LADDER])
                all_y.append(vs)
                for m in MICE:
                    ax.plot(xs + off[l], vs[:, m], '-', lw=0.7, color=colr, alpha=0.45,
                            zorder=2)
                for xi in xs:
                    _mouse_points(ax, xi + off[l], vs[xi], colr, clamp=cl[xi], ms=4.2,
                                  zorder=3)
                ax.plot(xs + off[l], np.median(vs, 1), '-', lw=2.4, color=colr,
                        zorder=4)
            ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=1)
            _y_log(ax, np.concatenate(all_y), ref)
            ax.set_xticks(xs)
            ax.set_xticklabels(LADDER)
            ax.set_xlim(-0.6, len(LADDER) - 0.4)
            if col == 0:
                ax.set_ylabel(lab, fontsize=8.5)
            if r == 0:
                ax.set_title(DEC_LABEL[dec])
            if r == len(keys) - 1:
                ax.set_xlabel('architecture (lin = Linear; rr8 = rank-8 linear; '
                              'hN = one tanh layer, width N)')
    ps.label_panels(axes)
    handles = [Line2D([0], [0], ls='-', lw=2.4, color=FAMILY_COLOR[l],
                      label=f'{FAMILY_LABEL[l]} — median (thick), per mouse (thin)')
               for l in TRIO]
    handles += [Line2D([0], [0], marker='x', ls='none', color='k', ms=7,
                       label='ŝ inversion clamped (bound)'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY,
                       label='1 = null (skill) / calibrated (ŝ)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, fontsize=8.5)
    _mouse_legend(axes[1, 0], loc='upper right', fontsize=7)
    fig.suptitle('io_hmm_v3, λ_H = 0: flat vs evar projection vs KL across the width '
                 'ladder, per mouse', fontsize=11.5)
    save_fig(fig, out_dir, 'flat_vs_evar_ladder', max_px=1560)


# ----------------------------------------------------------------------
# stdout: prior (d) scoring + the flat-vs-evar numbers
# ----------------------------------------------------------------------
def print_prior_d(c, spread):
    print('\n=== PRIOR (d): ce ~= kl ~= js within ~5 %  '
          '[relative spread = max |pairwise diff| / mean across the three, per mouse]')
    worst_overall = 0.0
    for arch in ('h8', 'h64'):
        for dec in DECODERS:
            for key in ('kl_skill', 'proj_skill', 's_hat'):
                sp = rel_spread(c, arch, dec, key)
                n_ok = int((sp <= SPREAD_TOL).sum())
                worst_overall = max(worst_overall, sp.max())
                per = ' '.join(f'{100 * s:5.2f}' for s in sp)
                print(f'  {arch:4s} {dec:4s} {METRIC_SHORT[key]:17s} '
                      f'spread % per mouse [{per}]  median {100 * np.median(sp):.2f} %  '
                      f'max {100 * sp.max():.2f} %  within 5 %: {n_ok}/6')
    # pairwise: which pair carries the spread?
    print('\n  pairwise |diff|/mean, median over mice x decoders x {h8,h64} x 3 metrics:')
    for a, b in (('ce', 'kl'), ('ce', 'js'), ('kl', 'js')):
        d = []
        for arch in ('h8', 'h64'):
            for dec in DECODERS:
                for key in ('kl_skill', 'proj_skill', 's_hat'):
                    va, vb = val(c, a, arch, dec, key), val(c, b, arch, dec, key)
                    d.append(np.abs(va - vb) / (0.5 * (va + vb)))
        d = np.concatenate(d)
        print(f'    {a} vs {b}: median {100 * np.median(d):.2f} %, max {100 * d.max():.2f} %')
    # the full ladder, for completeness
    print('\n  across the FULL ladder (all 7 archs x 2 decoders x 3 metrics x 6 mice):')
    allsp = np.concatenate([rel_spread(c, a, d, k) for a in LADDER for d in DECODERS
                            for k in ('kl_skill', 'proj_skill', 's_hat')])
    print(f'    spread median {100 * np.median(allsp):.2f} %, max {100 * allsp.max():.2f} %, '
          f'cells-within-5 %: {int((allsp <= SPREAD_TOL).sum())}/{allsp.size}')
    print('    over 5 % (arch dec metric mouse: spread %, which loss is the outlier):')
    for a in LADDER:
        for d in DECODERS:
            for k in ('kl_skill', 'proj_skill', 's_hat'):
                sp = rel_spread(c, a, d, k)
                v = np.stack([val(c, l, a, d, k) for l in CALIBRATED])
                for m in MICE:
                    if sp[m] > SPREAD_TOL:
                        far = CALIBRATED[int(np.argmax(np.abs(v[:, m] - np.median(v[:, m]))))]
                        print(f'      {a:4s} {d:4s} {METRIC_SHORT[k]:17s} m{m}: '
                              f'{100 * sp[m]:5.1f} %  ({far} off; '
                              + ' '.join(f'{l}={v[i, m]:.3f}' for i, l in enumerate(CALIBRATED))
                              + ')')
    verdict = 'HOLDS' if worst_overall <= SPREAD_TOL else (
        '↔ MAGNITUDE (JS within 10% of CE/KL, not the registered 5%)' if worst_overall <= 2 * SPREAD_TOL
        else 'PARTIAL')
    print(f'\n  VERDICT (d): {verdict} — worst per-mouse spread at h8/h64 = '
          f'{100 * worst_overall:.2f} %')


def print_flat_vs_evar(c):
    print('\n=== (c) flat vs evar projection, per mouse (lin | h8), with KL reference')
    for dec in DECODERS:
        for key in ('kl_skill', 'proj_skill', 's_hat'):
            print(f'  {DEC_LABEL[dec]} · {METRIC_SHORT[key]}')
            for arch in ('lin', 'h8'):
                vf, ve, vk = (val(c, l, arch, dec, key) for l in TRIO)
                cl = clamped(c, 'pca', arch, dec)
                fmt = lambda v: ' '.join(f'{x:5.2f}' for x in v)
                print(f'    {arch:3s} flat [{fmt(vf)}] med {np.median(vf):.2f} | '
                      f'evar [{fmt(ve)}] med {np.median(ve):.2f}'
                      f'{" (clamped: " + str(int(cl.sum())) + " mice)" if cl.any() else ""} | '
                      f'KL [{fmt(vk)}] med {np.median(vk):.2f}')
                rel_fk = np.abs(vf - vk) / (0.5 * (vf + vk))
                sign_e = int((ve > vf).sum())
                print(f'        |flat−KL|/mean per mouse: median {100 * np.median(rel_fk):.1f} %, '
                      f'max {100 * rel_fk.max():.1f} %;  evar > flat in {sign_e}/6 mice; '
                      f'evar/flat median {np.median(ve / vf):.2f}×')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(CSV))
    ap.add_argument('--out', default=str(OUT_DIR))
    a = ap.parse_args()
    ps.apply()
    c = load(Path(a.csv))
    out = Path(a.out)
    fig_loss_family_skill(c, out)
    spread = fig_calibrated_head_to_head(c, out)
    fig_flat_vs_evar(c, out)
    fig_flat_vs_evar_ladder(c, out)
    print_prior_d(c, spread)
    print_flat_vs_evar(c)


if __name__ == '__main__':
    main()
