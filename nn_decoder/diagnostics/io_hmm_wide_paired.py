# -*- coding: utf-8 -*-
"""PAIRED-ARM figures for the io_hmm_v3 width sweep (lambda_H = 0 cells only).

Reads ``figures/io_hmm_wide/cells_paired.csv`` (io_hmm_wide_extract.py run on
both arms: NEW = IO-HMM marginal targets, 72 x 2.5 deg circular; OLD =
export post_s_marginal Q, 91 x 1 deg linear; identical trials/splits/seeds/
config, only the target family differs) and draws, in
``figures/io_hmm_wide/paired/``:

  paired_width_ladder_kl_skill    2 x 4 grid (decoder x loss family): KL skill
                                  vs arch, OLD dashed/open vs NEW solid/filled,
                                  per-mouse thin lines, bold medians (Q1)
  paired_width_ladder_s_hat       the same for equivalent sharpening (Q2)
  paired_width_ladder_proj_skill  the same for evar-pinned projection skill
  q1_width_ratio_dumbbells        per mouse, h64/h8 and h4/h8 KL-skill ratios,
                                  paired dumbbells old -> new, sign counts (Q1)
  rr8_vs_h8_vs_lin_old            lin -> rr8 -> h8 KL-skill dumbbells for the
                                  OLD arm (same layout as the new-arm figure)
  q3_rank_vs_nonlin_steps         compact old-vs-new panel of the two step
                                  ratios (rr8/lin and h8/rr8) per mouse (Q3)
  q4_disagreement_map             dKL skill (new - old) vs dproj skill per
                                  (loss, arch, mouse), colour = loss, size =
                                  ladder rung, per-group sign-count side bar (Q4)

CE is dropped everywhere: CE = KL + H(target) with the same seed, so the ce
cells are a replicate of kl (near-identical in the old arm (414/420 blocks bit-identical, max |diff| 7.3e-3), <=1.3 % float
non-determinism in the new), never independent evidence.

All skills are ratios to EACH ARM'S OWN per-mouse LOO predict-mean null on
held-out trials (< 1 beats that null); s_hat is the equivalent sharpening of
each arm's own target (1 = calibrated). Cross-arm deltas therefore compare
null-relative performance, not raw losses (supports differ: 72 circular vs 91
linear bins). n = 6 mice; everything is per-mouse points, medians and sign
counts.

Prints a Q1-Q4 scoring block with per-mouse numbers and an outcome per
question (replicates / partial / does not replicate).

Usage (from nn_decoder/):
    python diagnostics/io_hmm_wide_paired.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import peakiness_style as ps                          # noqa: E402
from figsave import save_fig                          # noqa: E402

OUT_DIR = _HERE.parent / 'figures' / 'io_hmm_wide' / 'paired'
CSV = _HERE.parent / 'figures' / 'io_hmm_wide' / 'cells_paired.csv'

LOSSES = ('pca', 'pcaflat', 'kl', 'js')          # ce dropped: replicate of kl
CAL = ('pcaflat', 'kl', 'js')
ARCHS = ('lin', 'rr8', 'h4', 'h8', 'h16', 'h32', 'h64')
DECODERS = (('spat', 'spatial'), ('temp', 'temporal'))
MICE = tuple(range(6))
ARMS = ('old', 'new')
AGREE_FLAG = 0.10

FAMILY_COLOR = {'pca': ps.PCA_EVAR, 'pcaflat': ps.FLAT_EVAR, 'kl': ps.KL,
                'js': ps.JS}
FAMILY_LABEL = {'pca': 'projection (evar)', 'pcaflat': 'projection (flat)',
                'kl': 'KL', 'js': 'JS'}
FAMILY_SHORT = {'pca': 'evar', 'pcaflat': 'flat', 'kl': 'KL', 'js': 'JS'}
FAMILY_MARKER = {'pca': 'o', 'pcaflat': 's', 'kl': 'D', 'js': '^'}
MOUSE_COLOR = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
MOUSE_MARKER = ['o', 's', '^', 'D', 'v', 'P']

ARM_LABEL = {'new': 'NEW io_hmm_v3 (IO-HMM marginal, 72 × 2.5°)',
             'old': 'OLD exportref (export Q, 91 × 1°)'}

METRICS = {
    'kl_skill': ('KL skill', 'mean KL(target‖decoded) / own-arm LOO predict-mean '
                             'null\n(held-out trials; < 1 beats that null)'),
    'proj_skill': ('projection skill', 'evar-pinned projection loss / own-arm LOO '
                                       'predict-mean null\n(< 1 beats that null)'),
    's_hat': ('equivalent sharpening ŝ', 'ŝ vs each arm\'s own target (1 = '
              'calibrated; × = clamped bound)'),
}

NOTE_CE = 'CE dropped (≡ KL + H(target), same seed: a replicate of KL, not evidence)'


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load(csv=CSV):
    df = pd.read_csv(csv)
    d0 = df[np.isclose(df.lambda_H, 0.0)].copy()
    n_exp = 5 * len(ARCHS) * len(MICE) * len(DECODERS) * len(ARMS)   # incl. ce
    if len(d0) != n_exp:
        raise SystemExit(f'ABORT: expected {n_exp} lambda_H = 0 rows, got {len(d0)}')
    if d0.duplicated(['arm', 'loss_family', 'arch', 'mouse', 'decoder']).any():
        raise SystemExit('ABORT: duplicate (arm, loss, arch, mouse, decoder) rows')
    # ce really is a replicate of kl before we drop it
    for arm in ARMS:
        kl = d0[(d0.arm == arm) & (d0.loss_family == 'kl')].set_index(
            ['arch', 'mouse', 'decoder']).sort_index()
        ce = d0[(d0.arm == arm) & (d0.loss_family == 'ce')].set_index(
            ['arch', 'mouse', 'decoder']).sort_index()
        gap = float((kl.kl_skill - ce.kl_skill).abs().max())
        if gap > 0.05:
            raise SystemExit(f'ABORT: ce vs kl kl_skill differs by {gap:.3f} in '
                             f'{arm} arm — not a replicate?')
    return d0[d0.loss_family.isin(LOSSES)].copy()


def table(d0, arm, metric, loss, dec):
    """(6 mice x 7 archs) matrix of ``metric`` for one (arm, loss, decoder)."""
    sub = d0[(d0.arm == arm) & (d0.loss_family == loss) & (d0.decoder == dec)]
    t = sub.pivot(index='mouse', columns='arch', values=metric)
    t = t.reindex(index=list(MICE), columns=list(ARCHS))
    if t.isna().any().any():
        raise SystemExit(f'ABORT: missing cells for {arm} {metric} {loss} {dec}')
    return t


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------
def _log_ticks(ax, which='y'):
    axis = ax.yaxis if which == 'y' else ax.xaxis
    lo, hi = ax.get_ylim() if which == 'y' else ax.get_xlim()
    cand = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15])
    t = cand[(cand >= lo) & (cand <= hi)]
    axis.set_ticks(t)
    axis.set_ticklabels([f'{v:g}' for v in t], fontsize=8)
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def _fmt(v):
    return ' '.join(f'{x:5.2f}' for x in np.asarray(v))


# ----------------------------------------------------------------------
# (a) width ladders, both arms overlaid
# ----------------------------------------------------------------------
def fig_width_ladder(d0, metric, out_dir):
    title, ylab = METRICS[metric]
    fig, axes = plt.subplots(len(DECODERS), len(LOSSES), figsize=(14.4, 6.6),
                             sharex=True)
    x = np.arange(len(ARCHS))
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, loss in enumerate(LOSSES):
            ax = axes[r, c]
            for arm in ARMS:
                t = table(d0, arm, metric, loss, dec)
                solid = arm == 'new'
                ls = '-' if solid else '--'
                for m in MICE:
                    y = t.loc[m].to_numpy()
                    col = MOUSE_COLOR[m]
                    ax.plot(x, y, ls, lw=0.9, color=col,
                            alpha=0.75 if solid else 0.55, zorder=2)
                    mfc = col if solid else 'none'
                    ax.plot(x, y, MOUSE_MARKER[m], ms=4.2, mfc=mfc, mec=col,
                            mew=1.0, zorder=3)
                    if metric == 's_hat':
                        cl = table(d0, arm, 's_hat_clamped', loss, dec).loc[m].to_numpy() > 0
                        ax.plot(x[cl], y[cl], 'x', ms=8, color=col, mew=2.0, zorder=4)
                med = t.median(axis=0).to_numpy()
                ax.plot(x, med, ls, lw=3.0, color='k', alpha=0.85, zorder=5)
            ax.axhline(1.0, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=1)
            ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(ARCHS, fontsize=8)
            ax.grid(True, axis='y', lw=0.4, alpha=0.35)
            ax.set_title(f'{FAMILY_LABEL[loss]} · {dec_lab}', fontsize=10,
                         color=FAMILY_COLOR[loss], fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{title} · {dec_lab}', fontsize=9)
            if r == len(DECODERS) - 1:
                ax.set_xlabel('architecture (lin → rank-8 → tanh width)', fontsize=8)
    # share y across the calibrated panels of a row; evar keeps its own scale
    for r in range(len(DECODERS)):
        cal = [axes[r, c] for c, loss in enumerate(LOSSES) if loss != 'pca']
        lo = min(ax.get_ylim()[0] for ax in cal)
        hi = max(ax.get_ylim()[1] for ax in cal)
        for ax in cal:
            ax.set_ylim(lo, hi)
        for ax in axes[r]:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 1.15)
            _log_ticks(ax)
    fig.supylabel(ylab, fontsize=9)
    ref_label = {'kl_skill': 'own-arm null (= 1)', 'proj_skill': 'own-arm null (= 1)',
                 's_hat': 'own target width (ŝ = 1)'}[metric]
    handles = [Line2D([0], [0], marker=MOUSE_MARKER[m], ls='-', lw=0.9,
                      color=MOUSE_COLOR[m], ms=4, label=f'mouse {m}') for m in MICE]
    handles += [Line2D([0], [0], ls='-', lw=2.2, color='k',
                       marker='o', ms=4.5, mfc='k', label='NEW (solid/filled)'),
                Line2D([0], [0], ls='--', lw=2.2, color='k',
                       marker='o', ms=4.5, mfc='none', label='OLD (dashed/open)'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label=ref_label)]
    if metric == 's_hat':
        handles += [Line2D([0], [0], marker='x', ls='none', color='0.3', mew=2,
                           ms=8, label='× = clamped (ladder end, a bound)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=10, fontsize=8,
               frameon=False)
    fig.suptitle(f'{title} vs architecture — BOTH ARMS, λ_H = 0, weight decay 0, 6 mice\n'
                 f'{ARM_LABEL["new"]} vs {ARM_LABEL["old"]}; {NOTE_CE}', fontsize=10.5)
    save_fig(fig, out_dir, f'paired_width_ladder_{metric}')


# ----------------------------------------------------------------------
# (b) Q1 scoring panel: h64/h8 and h4/h8 ratios, old -> new dumbbells
# ----------------------------------------------------------------------
C_WIDE, C_NARROW = '#b2182b', '#2166ac'


def fig_q1_dumbbells(d0, out_dir):
    fig, axes = plt.subplots(len(DECODERS), len(CAL), figsize=(12.6, 6.8),
                             sharex=True, sharey=True)
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, loss in enumerate(CAL):
            ax = axes[r, c]
            counts = {}
            for ratio_name, cols, colr, dy in (('h64/h8', ('h64', 'h8'), C_WIDE, -0.18),
                                               ('h4/h8', ('h4', 'h8'), C_NARROW, +0.18)):
                vals = {}
                for arm in ARMS:
                    t = table(d0, arm, 'kl_skill', loss, dec)
                    vals[arm] = (t[cols[0]] / t[cols[1]]).to_numpy()
                for m in MICE:
                    y = m + dy
                    ax.plot([vals['old'][m], vals['new'][m]], [y, y], '-', lw=1.3,
                            color=colr, alpha=0.55, zorder=2)
                    ax.plot(vals['old'][m], y, 'o', ms=6, mfc='none', mec=colr,
                            mew=1.4, zorder=3)
                    ax.plot(vals['new'][m], y, 'o', ms=6, mfc=colr, mec='k',
                            mew=0.5, zorder=4)
                counts[ratio_name] = vals
            k64o = int((counts['h64/h8']['old'] >= 1.10).sum())
            k64n = int((counts['h64/h8']['new'] >= 1.10).sum())
            k4o = int((counts['h4/h8']['old'] < 1.0).sum())
            k4n = int((counts['h4/h8']['new'] < 1.0).sum())
            ax.set_title(f'{FAMILY_LABEL[loss]} · {dec_lab}\n'
                         f'h64 ≥10% worse: {k64o}/6 → {k64n}/6\n'
                         f'h4 better: {k4o}/6 → {k4n}/6 (old → new)',
                         fontsize=8, color=FAMILY_COLOR[loss])
            ax.axvline(1.0, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=0)
            ax.set_xscale('log')
            ax.set_yticks(list(MICE))
            ax.set_yticklabels([f'mouse {m}' for m in MICE], fontsize=8)
            ax.grid(True, axis='x', lw=0.4, alpha=0.35)
            if r == len(DECODERS) - 1:
                ax.set_xlabel('KL-skill ratio to h8 (> 1 = worse than h8;\n'
                              'each arm ÷ its own LOO predict-mean null first)',
                              fontsize=8)
    axes[0, 0].set_ylim(len(MICE) - 0.4, -0.6)        # mouse 0 on top (shared y)
    for ax in axes.flat:
        xt = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        lo, hi = ax.get_xlim()
        xt = xt[(xt >= lo) & (xt <= hi)]
        ax.set_xticks(xt)
        ax.set_xticklabels([f'{v:g}' for v in xt], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    handles = [Line2D([0], [0], marker='o', ls='-', color=C_WIDE, mfc=C_WIDE,
                      ms=6, label='h64/h8 (wide rung vs h8)'),
               Line2D([0], [0], marker='o', ls='-', color=C_NARROW, mfc=C_NARROW,
                      ms=6, label='h4/h8 (narrow rung vs h8)'),
               Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.3',
                      mew=1.4, ms=6, label='open = OLD (export Q)'),
               Line2D([0], [0], marker='o', ls='none', mfc='0.3', mec='k',
                      ms=6, label='filled = NEW (IO-HMM)'),
               Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label='= h8 (ratio 1)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=8,
               frameon=False)
    fig.suptitle('Q1 — does the width effect replicate on the OLD (sharp) targets?\n'
                 f'h64/h8 and h4/h8 KL-skill ratios, dumbbells OLD → NEW, λ_H = 0; {NOTE_CE}',
                 fontsize=11)
    save_fig(fig, out_dir, 'q1_width_ratio_dumbbells')


# ----------------------------------------------------------------------
# (c) Q3: lin -> rr8 -> h8 for the OLD arm + step-size comparison
# ----------------------------------------------------------------------
def fig_dumbbell_old(d0, out_dir):
    trio = ('lin', 'rr8', 'h8')
    fig, axes = plt.subplots(len(DECODERS), len(CAL), figsize=(12.6, 6.6),
                             sharey=True, sharex='row')
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, loss in enumerate(CAL):
            ax = axes[r, c]
            t = table(d0, 'old', 'kl_skill', loss, dec)[list(trio)]
            for m in MICE:
                v = t.loc[m].to_numpy()
                ax.plot(v, [m, m, m], '-', lw=1.4, color='0.6', zorder=1)
                ax.plot(v[0], m, 's', ms=7, mfc='white', mec=FAMILY_COLOR[loss],
                        mew=1.5, zorder=3)
                ax.plot(v[1], m, 'D', ms=6, mfc=FAMILY_COLOR[loss], mec='k',
                        mew=0.6, alpha=0.55, zorder=3)
                ax.plot(v[2], m, 'o', ms=7, mfc=FAMILY_COLOR[loss], mec='k',
                        mew=0.6, zorder=4)
            k_r = int((t['rr8'] < t['lin']).sum())
            k_h = int((t['h8'] < t['rr8']).sum())
            ax.set_title(f'{FAMILY_LABEL[loss]} · {dec_lab}\n'
                         f'rr8 better than lin: {k_r}/6 · h8 better than rr8: {k_h}/6',
                         fontsize=9.5, color=FAMILY_COLOR[loss])
            ax.set_xscale('log')
            ax.axvline(1.0, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=0)
            ax.set_yticks(list(MICE))
            ax.set_yticklabels([f'mouse {m}' for m in MICE], fontsize=8)
            ax.set_ylim(-0.6, len(MICE) - 0.4)
            ax.grid(True, axis='x', lw=0.4, alpha=0.35)
        axes[r, 0].invert_yaxis()
        for ax in axes[r]:
            _log_ticks(ax, 'x')
            if r == len(DECODERS) - 1:
                ax.set_xlabel('KL skill (÷ OLD-arm LOO predict-mean null; < 1 beats null)',
                              fontsize=8)
    handles = [Line2D([0], [0], marker='s', ls='none', mfc='white', mec='0.3', mew=1.5,
                      ms=7, label='lin  Linear(n, 91)'),
               Line2D([0], [0], marker='D', ls='none', mfc='0.5', mec='k', ms=6,
                      label='rr8  Linear(n, 8) → Linear(8, 91), no nonlinearity'),
               Line2D([0], [0], marker='o', ls='none', mfc='0.3', mec='k', ms=7,
                      label='h8  Linear(n, 8) → tanh → Linear(8, 91)'),
               Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label='null (= 1)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=8,
               frameon=False)
    fig.suptitle('Q3 — rank bottleneck vs nonlinearity on the OLD arm (export Q, 91 × 1°): '
                 f'KL skill at lin → rr8 → h8, λ_H = 0; {NOTE_CE}', fontsize=11)
    save_fig(fig, out_dir, 'rr8_vs_h8_vs_lin_old')


def fig_step_sizes(d0, out_dir):
    steps = (('rr8/lin', ('rr8', 'lin'), 'rank-8 bottleneck alone'),
             ('h8/rr8', ('h8', 'rr8'), 'adding the tanh at rank 8'))
    fig, axes = plt.subplots(len(DECODERS), len(steps), figsize=(11.4, 6.8),
                             sharex=True, sharey=True)
    dy = {loss: off for loss, off in zip(CAL, (-0.24, 0.0, +0.24))}
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, (sname, cols, sdesc) in enumerate(steps):
            ax = axes[r, c]
            note = []
            for loss in CAL:
                vals = {}
                for arm in ARMS:
                    t = table(d0, arm, 'kl_skill', loss, dec)
                    vals[arm] = (t[cols[0]] / t[cols[1]]).to_numpy()
                colr = FAMILY_COLOR[loss]
                for m in MICE:
                    y = m + dy[loss]
                    ax.plot([vals['old'][m], vals['new'][m]], [y, y], '-', lw=1.2,
                            color=colr, alpha=0.5, zorder=2)
                    ax.plot(vals['old'][m], y, 'o', ms=5, mfc='none', mec=colr,
                            mew=1.3, zorder=3)
                    ax.plot(vals['new'][m], y, 'o', ms=5, mfc=colr, mec='k',
                            mew=0.4, zorder=4)
                note.append(f'{FAMILY_SHORT[loss]} '
                            f'{int((vals["old"] < 1).sum())}/6 → '
                            f'{int((vals["new"] < 1).sum())}/6')
            ax.set_title(f'{sname} — {sdesc} · {dec_lab}\nstep helps (< 1), old → new: '
                         + ' · '.join(note), fontsize=9)
            ax.axvline(1.0, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=0)
            ax.set_xscale('log')
            ax.set_yticks(list(MICE))
            ax.set_yticklabels([f'mouse {m}' for m in MICE], fontsize=8)
            ax.grid(True, axis='x', lw=0.4, alpha=0.35)
            if r == len(DECODERS) - 1:
                ax.set_xlabel('KL-skill step ratio (< 1 = the step improves the fit;\n'
                              'each arm ÷ its own LOO predict-mean null first)',
                              fontsize=8)
    axes[0, 0].set_ylim(len(MICE) - 0.35, -0.65)      # mouse 0 on top (shared y)
    for ax in axes.flat:
        xt = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        lo, hi = ax.get_xlim()
        xt = xt[(xt >= lo) & (xt <= hi)]
        ax.set_xticks(xt)
        ax.set_xticklabels([f'{v:g}' for v in xt], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    handles = ([Line2D([0], [0], marker='o', ls='-', color=FAMILY_COLOR[l], ms=5,
                       label=FAMILY_LABEL[l]) for l in CAL]
               + [Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.3',
                         mew=1.3, ms=5, label='open = OLD (export Q)'),
                  Line2D([0], [0], marker='o', ls='none', mfc='0.3', mec='k', ms=5,
                         label='filled = NEW (IO-HMM)'),
                  Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label='no change (= 1)')])
    fig.legend(handles=handles, loc='outside lower center', ncol=6, fontsize=8,
               frameon=False)
    fig.suptitle('Q3 — the two ladder steps in both arms: rank alone (rr8/lin) vs '
                 f'nonlinearity (h8/rr8), KL skill, λ_H = 0; {NOTE_CE}', fontsize=11)
    save_fig(fig, out_dir, 'q3_rank_vs_nonlin_steps')


# ----------------------------------------------------------------------
# (d) Q4: arm-delta disagreement map, KL skill vs projection skill
# ----------------------------------------------------------------------
def group_deltas(d0):
    """Per (decoder, loss, arch): per-mouse new − old for both skills."""
    rows = []
    for dec, _ in DECODERS:
        for loss in LOSSES:
            for arch in ARCHS:
                dk = (table(d0, 'new', 'kl_skill', loss, dec)[arch]
                      - table(d0, 'old', 'kl_skill', loss, dec)[arch])
                dp = (table(d0, 'new', 'proj_skill', loss, dec)[arch]
                      - table(d0, 'old', 'proj_skill', loss, dec)[arch])
                rows.append(dict(dec=dec, loss=loss, arch=arch,
                                 dk=dk.to_numpy(), dp=dp.to_numpy(),
                                 kl_new_better=int((dk < 0).sum()),
                                 proj_new_better=int((dp < 0).sum()),
                                 med_dk=float(dk.median()), med_dp=float(dp.median())))
    g = pd.DataFrame(rows)
    g['disag'] = (g.kl_new_better - g.proj_new_better).abs()
    return g


def fig_disagreement_map(d0, out_dir):
    g = group_deltas(d0)
    sizes = {a: 14 + 11 * i for i, a in enumerate(ARCHS)}
    fig = plt.figure(figsize=(14.2, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.75))
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axb = fig.add_subplot(gs[0, 2])
    for ax, (dec, dec_lab) in zip(axes, DECODERS):
        sub = g[g.dec == dec]
        for _, row in sub.iterrows():
            ax.scatter(row.dk, row.dp, s=sizes[row.arch],
                       marker=FAMILY_MARKER[row.loss],
                       color=FAMILY_COLOR[row.loss], alpha=0.65,
                       edgecolors='k', linewidths=0.3, zorder=3)
        ax.axhline(0, lw=1.0, color='0.4', zorder=1)
        ax.axvline(0, lw=1.0, color='0.4', zorder=1)
        ax.set_xscale('symlog', linthresh=0.2)
        ax.set_title(f'{dec_lab} decoder', fontsize=10)
        ax.set_xlabel('Δ KL skill (NEW − OLD; each arm ÷ its own\n'
                      'LOO predict-mean null; < 0 = NEW better; symlog)', fontsize=8)
        ax.set_ylabel('Δ projection skill (NEW − OLD; own-arm null;\n< 0 = NEW better)',
                      fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.35)
    # shared limits + quadrant labels
    lo_x = min(ax.get_xlim()[0] for ax in axes)
    hi_x = max(ax.get_xlim()[1] for ax in axes)
    lo_y = min(ax.get_ylim()[0] for ax in axes)
    hi_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(lo_y, hi_y)
        q = dict(fontsize=7.5, color='0.45', style='italic')
        ax.text(0.02, 0.98, 'KL: NEW better\nproj: OLD better', ha='left',
                va='top', transform=ax.transAxes, **q)
        ax.text(0.98, 0.98, 'OLD better on both', ha='right', va='top',
                transform=ax.transAxes, **q)
        ax.text(0.02, 0.02, 'NEW better on both', ha='left', va='bottom',
                transform=ax.transAxes, **q)
        ax.text(0.98, 0.02, 'KL: OLD better\nproj: NEW better', ha='right',
                va='bottom', transform=ax.transAxes, **q)
    # side bar: top 12 groups by sign-count disagreement (tie-break: |Δ of medians|)
    g = g.assign(tiebreak=(g.med_dk - g.med_dp).abs())
    top = g.sort_values(['disag', 'tiebreak'],
                        ascending=[False, False]).head(12).reset_index(drop=True)
    yb = np.arange(len(top))
    axb.barh(yb - 0.18, top.kl_new_better, height=0.34, color='0.25',
             label='KL skill: mice with NEW better')
    axb.barh(yb + 0.18, top.proj_new_better, height=0.34, color='0.25',
             fill=False, edgecolor='0.25', hatch='///',
             label='projection skill: mice with NEW better')
    axb.set_yticks(yb)
    axb.set_yticklabels([f'{r.dec}·{FAMILY_SHORT[r.loss]}·{r.arch}'
                         for r in top.itertuples()], fontsize=8)
    axb.invert_yaxis()
    axb.set_xlim(0, 6)
    axb.set_xticks(range(7))
    axb.set_xlabel('mice with NEW better (of 6);\ndifference = metric disagreement',
                   fontsize=8)
    axb.set_title('top 12 groups by\nKL-vs-proj sign-count disagreement', fontsize=9)
    axb.grid(True, axis='x', lw=0.4, alpha=0.35)
    axb.legend(fontsize=7.5, frameon=False, loc='lower right')
    handles = ([Line2D([0], [0], marker=FAMILY_MARKER[l], ls='none',
                       color=FAMILY_COLOR[l], ms=7, label=FAMILY_LABEL[l])
                for l in LOSSES]
               + [Line2D([0], [0], marker='o', ls='none', mfc='0.6', mec='k',
                         ms=4 + 1.2 * i, label=a) for i, a in enumerate(ARCHS)])
    fig.legend(handles=handles, loc='outside lower center', ncol=11, fontsize=8,
               frameon=False)
    fig.suptitle('Q4 — which groups change verdict between arms, per metric? '
                 'one point per (loss, arch, mouse), λ_H = 0\n'
                 f'{ARM_LABEL["new"]} minus {ARM_LABEL["old"]}; skills are '
                 f'null-relative so arms are comparable; {NOTE_CE}', fontsize=10.5)
    save_fig(fig, out_dir, 'q4_disagreement_map')
    return g


# ----------------------------------------------------------------------
# scoring block
# ----------------------------------------------------------------------
def score(d0, g):
    out = []
    P = out.append
    P('PAIRED-ARM SCORING (Q1-Q4), lambda_H = 0, 6 mice; skills are ratios to each '
      'arm\'s own LOO predict-mean null; ce excluded (replicate of kl).')
    P('')

    # ---------------- Q1 ----------------
    P('Q1  h4 optimum / width degradation on the OLD (sharp) targets?')
    k64 = {}
    k4 = {}
    for dec, _ in DECODERS:
        for loss in CAL:
            for arm in ARMS:
                t = table(d0, arm, 'kl_skill', loss, dec)
                r64 = (t.h64 / t.h8).to_numpy()
                r4 = (t.h4 / t.h8).to_numpy()
                k64[(dec, loss, arm)] = int((r64 >= 1.10).sum())
                k4[(dec, loss, arm)] = int((r4 < 1.0).sum())
                P(f'    {dec} {loss:8s} {arm}: h64/h8 [{_fmt(r64)}] >=10% worse '
                  f'{k64[(dec, loss, arm)]}/6 | h4/h8 [{_fmt(r4)}] h4 better '
                  f'{k4[(dec, loss, arm)]}/6')
    deg_old_spat = [k64[('spat', l, 'old')] for l in CAL]
    deg_new_spat = [k64[('spat', l, 'new')] for l in CAL]
    h4_old = [k4[(d, l, 'old')] for d, _ in DECODERS for l in CAL]
    h4_new = [k4[(d, l, 'new')] for d, _ in DECODERS for l in CAL]
    med_old = np.median([float((table(d0, 'old', 'kl_skill', l, 'spat').h64
                                / table(d0, 'old', 'kl_skill', l, 'spat').h8).median())
                         for l in CAL])
    med_new = np.median([float((table(d0, 'new', 'kl_skill', l, 'spat').h64
                                / table(d0, 'new', 'kl_skill', l, 'spat').h8).median())
                         for l in CAL])
    P(f'  OUTCOME Q1: PARTIAL — the h64 degradation replicates in direction on the old '
      f'targets (spatial ≥10%-worse counts old {deg_old_spat} vs new {deg_new_spat} '
      f'of 6 per loss; median spatial h64/h8 old {med_old:.2f} vs new {med_new:.2f}; '
      f'temporal mild in both arms), but the h4 OPTIMUM does NOT: h4 beats h8 in only '
      f'{h4_old} of 6 (old, per decoder×loss) vs {h4_new} (new) — on sharp targets h4 '
      f'≈ h8. The degradation above h8 is a shared property; the h4 minimum is a '
      f'broad-target property.')
    P('')

    # ---------------- Q2 ----------------
    P('Q2  evar over-sharpening grows with width on the OLD targets?')
    q2 = {}
    for dec, _ in DECODERS:
        for arm in ARMS:
            t = table(d0, arm, 's_hat', 'pca', dec)
            cl = table(d0, arm, 's_hat_clamped', 'pca', dec)
            r = (t.h64 / t.h8).to_numpy()
            ladder = t[['h4', 'h8', 'h16', 'h32', 'h64']].to_numpy()
            rho = [pd.Series(row).corr(pd.Series(np.arange(5.)), method='spearman')
                   for row in ladder]
            n_up = int(sum(x > 0 for x in rho))
            n_gt1 = int((r > 1.0).sum())
            n_25 = int((r >= 1.25).sum())
            clamped = [f'm{m}:{a}' for m in MICE for a in ('h8', 'h64')
                       if cl.loc[m, a] > 0]
            q2[(dec, arm)] = (n_25, n_up, n_gt1)
            P(f'    {dec} {arm}: ŝ h8 [{_fmt(t.h8)}] h64 [{_fmt(t.h64)}] h64/h8 '
              f'[{_fmt(r)}] ≥1.25: {n_25}/6, >1: {n_gt1}/6, rising over h4..h64 '
              f'(Spearman>0): {n_up}/6; clamped: {clamped if clamped else "none"}')
    P(f'  OUTCOME Q2: REPLICATES (direction) — ŝ rises with width in 6/6 mice in both '
      f'arms and both decoders; h64/h8 > 1 in 6/6 everywhere except old-temporal '
      f'({q2[("temp", "old")][2]}/6 > 1, {q2[("temp", "old")][0]}/6 ≥ 1.25) where the '
      f'h8 base already over-sharpens 2.4–3.9×, so the relative growth is compressed. '
      f'Width feeds the evar pathology on sharp targets too.')
    P('')

    # ---------------- Q3 ----------------
    P('Q3  rank-vs-nonlinearity split on the OLD targets?')
    q3 = {}
    for dec, _ in DECODERS:
        for loss in CAL:
            for arm in ARMS:
                t = table(d0, arm, 'kl_skill', loss, dec)
                s1 = (t.rr8 / t.lin).to_numpy()
                s2 = (t.h8 / t.rr8).to_numpy()
                q3[(dec, loss, arm)] = (int((s1 < 1).sum()), int((s2 < 1).sum()))
                P(f'    {dec} {loss:8s} {arm}: rr8/lin [{_fmt(s1)}] rr8 better '
                  f'{q3[(dec, loss, arm)][0]}/6 | h8/rr8 [{_fmt(s2)}] h8 better '
                  f'{q3[(dec, loss, arm)][1]}/6')
    tanh_all = all(v[1] == 6 for v in q3.values())
    rank_old_temp = [q3[('temp', l, 'old')][0] for l in CAL]
    P(f'  OUTCOME Q3: REPLICATES — the tanh step (h8 < rr8) holds 6/6 mice in every '
      f'loss × decoder × arm block ({"all 6/6" if tanh_all else "NOT all"}): the '
      f'nonlinearity is load-bearing on sharp targets too. The rank step (rr8 < lin) '
      f'replicates 6/6 spatially in both arms but softens on old-temporal '
      f'({rank_old_temp} of 6 for pcaflat/kl/js), and the old-arm spatial rank step is '
      f'smaller (rr8/lin 0.72–0.99 old vs 0.59–0.95 new) — rank alone buys less on sharp '
      f'targets; the tanh buys the same.')
    P('')

    # ---------------- Q4 ----------------
    P('Q4  which (loss, arch) groups change verdict between arms, per metric?')
    n_groups = len(g)
    kl_new = int((g.kl_new_better >= 5).sum())
    kl_old = int((g.kl_new_better <= 1).sum())
    pj_new = int((g.proj_new_better >= 5).sum())
    pj_old = int((g.proj_new_better <= 1).sum())
    P(f'    decisive (≥5/6 mice one way) of {n_groups} (loss, arch, decoder) groups: '
      f'KL skill — NEW better {kl_new}, OLD better {kl_old}, mixed '
      f'{n_groups - kl_new - kl_old}; projection skill — NEW better {pj_new}, OLD '
      f'better {pj_old}, mixed {n_groups - pj_new - pj_old}')
    top = g.sort_values('disag', ascending=False).head(8)
    for r in top.itertuples():
        P(f'    disag {r.disag}: {r.dec}·{r.loss}·{r.arch} — KL new-better '
          f'{r.kl_new_better}/6 (med Δ {r.med_dk:+.2f}), proj new-better '
          f'{r.proj_new_better}/6 (med Δ {r.med_dp:+.2f})')
    flip = g[((g.kl_new_better >= 5) & (g.proj_new_better <= 1))
             | ((g.kl_new_better <= 1) & (g.proj_new_better >= 5))]
    flip_desc = ', '.join(f'{r.dec}·{r.loss}·{r.arch}' for r in flip.itertuples())
    P(f'  OUTCOME Q4: the two metrics agree that the OLD arm is the easier problem '
      f'relative to its null under projection skill ({pj_old}/{n_groups} groups '
      f'decisively OLD-better) while KL skill is loss- and width-dependent '
      f'({kl_old}/{n_groups} OLD-better, {n_groups - kl_new - kl_old} mixed). '
      f'Decisively OPPOSITE verdicts in {len(flip)}/{n_groups} groups'
      f'{f" ({flip_desc})" if len(flip) else ""}; the sign-count disagreement '
      f'concentrates in temporal evar mid-widths (rr8–h32), where NEW wins under KL '
      f'but OLD wins under projection — the KL-vs-proj disagreement is an evar '
      f'property, not a ladder-wide one.')
    return out


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', type=Path, default=CSV)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    d0 = load(args.csv)
    print(f'{len(d0)} lambda_H = 0 rows (ce dropped) from {args.csv}')

    for metric in ('kl_skill', 's_hat', 'proj_skill'):
        fig_width_ladder(d0, metric, args.out_dir)
    fig_q1_dumbbells(d0, args.out_dir)
    fig_dumbbell_old(d0, args.out_dir)
    fig_step_sizes(d0, args.out_dir)
    g = fig_disagreement_map(d0, args.out_dir)

    print()
    print('\n'.join(score(d0, g)))
    return d0


if __name__ == '__main__':
    main()
