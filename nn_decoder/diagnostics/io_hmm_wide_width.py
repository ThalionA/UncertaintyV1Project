# -*- coding: utf-8 -*-
"""ARCHITECTURE figures for the io_hmm_v3 width sweep (lambda_H = 0 cells only).

Reads ``figures/io_hmm_wide/cells.csv`` (written by ``io_hmm_wide_extract.py``
from the settled scorecard metrics) and draws, for the 7-rung ladder
lin -> rr8 -> h4 -> h8 -> h16 -> h32 -> h64:

  width_ladder_kl_skill    2 x 5 grid (decoder x loss family): KL skill vs arch,
                           per-mouse thin lines + markers, bold median, log y,
                           line at 1 (= the LOO predict-mean null)
  width_ladder_proj_skill  the same under the evar-pinned projection skill
  width_ladder_s_hat       equivalent sharpening (log y, 1 = calibrated);
                           ring = reshapes (agreement > 0.10), cross = clamped
  width_ladder_overfit_gap (val - train fit at best_epoch) / null loss
  rr8_vs_h8_vs_lin         paired per-mouse dumbbells lin -> rr8 -> h8 of KL
                           skill for the calibrated losses, sign counts in the
                           panel title (rank bottleneck vs nonlinearity)
  best_arch_grid           mouse x loss -> best arch under KL skill and under
                           projection skill, per decoder, plus the count of mice
                           for which a width > 8 beats h8 by > 10 %

and prints the scoring of the 2026-08-22 PREDICTIONS.md priors (a)-(d) with
per-mouse numbers; (e) is a lambda question and lives in the lambda script.

All y-axes are ratios to the per-mouse LOO predict-mean null (skill) or to the
target's own width (s_hat); n = 6 mice is the unit, so everything is sign
counts, per-mouse values and medians.

Usage (from nn_decoder/):
    python diagnostics/io_hmm_wide_width.py
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

OUT_DIR = _HERE.parent / 'figures' / 'io_hmm_wide'
CSV = OUT_DIR / 'cells.csv'

LOSSES = ('pca', 'pcaflat', 'kl', 'js', 'ce')
CALIBRATED = ('pcaflat', 'kl', 'js', 'ce')
ARCHS = ('lin', 'rr8', 'h4', 'h8', 'h16', 'h32', 'h64')
WIDE = ('h16', 'h32', 'h64')
DECODERS = (('spat', 'spatial'), ('temp', 'temporal'))
MICE = tuple(range(6))
AGREE_FLAG = 0.10

FAMILY_COLOR = {'pca': ps.PCA_EVAR, 'pcaflat': ps.FLAT_EVAR, 'kl': ps.KL,
                'js': ps.JS, 'ce': ps.CE}
FAMILY_LABEL = {'pca': 'projection (evar)', 'pcaflat': 'projection (flat)',
                'kl': 'KL', 'js': 'JS', 'ce': 'CE'}
# one colour per mouse, stable across every figure in this script
MOUSE_COLOR = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
MOUSE_MARKER = ['o', 's', '^', 'D', 'v', 'P']

METRICS = {
    'kl_skill': ('KL skill', 'mean KL(target‖decoded) / LOO predict-mean null\n'
                             '(held-out trials; < 1 beats null)', True, 1.0),
    'proj_skill': ('projection skill', 'evar-pinned projection loss / LOO '
                                       'predict-mean null\n(< 1 beats null)', True, 1.0),
    's_hat': ('equivalent sharpening ŝ', 'ŝ (1 = target width; ring = reshapes, '
                                          'agreement > 0.10; × = clamped bound)', True, 1.0),
    'overfit_gap': ('overfitting gap', '(val − train fit at best epoch) / null loss\n'
                                       '(own training loss, own evar weighting)', False, 0.0),
}


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load(csv=CSV):
    df = pd.read_csv(csv)
    d0 = df[np.isclose(df.lambda_H, 0.0)].copy()
    n_exp = len(LOSSES) * len(ARCHS) * len(MICE) * len(DECODERS)
    if len(d0) != n_exp:
        raise SystemExit(f'ABORT: expected {n_exp} lambda_H = 0 rows, got {len(d0)}')
    if d0.duplicated(['loss_family', 'arch', 'mouse', 'decoder']).any():
        raise SystemExit('ABORT: duplicate (loss, arch, mouse, decoder) rows')
    return d0


def table(d0, metric, loss, dec):
    """(6 mice x 7 archs) matrix of ``metric`` for one (loss, decoder)."""
    sub = d0[(d0.loss_family == loss) & (d0.decoder == dec)]
    t = sub.pivot(index='mouse', columns='arch', values=metric)
    t = t.reindex(index=list(MICE), columns=list(ARCHS))
    if t.isna().any().any():
        raise SystemExit(f'ABORT: missing cells for {metric} {loss} {dec}')
    return t


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def _log_ticks(ax, which='y'):
    """Label the 1-2-3-5-7 ladder on a log axis instead of bare powers of ten."""
    axis = ax.yaxis if which == 'y' else ax.xaxis
    lo, hi = ax.get_ylim() if which == 'y' else ax.get_xlim()
    cand = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20])
    t = cand[(cand >= lo) & (cand <= hi)]
    axis.set_ticks(t)
    axis.set_ticklabels([f'{v:g}' for v in t], fontsize=8)
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def _ladder_axes():
    fig, axes = plt.subplots(len(DECODERS), len(LOSSES), figsize=(15.6, 6.4),
                             sharex=True)
    return fig, axes


def fig_width_ladder(d0, metric, out_dir):
    title, ylab, logy, ref = METRICS[metric]
    fig, axes = _ladder_axes()
    x = np.arange(len(ARCHS))
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, loss in enumerate(LOSSES):
            ax = axes[r, c]
            t = table(d0, metric, loss, dec)
            flags_r = table(d0, 's_hat_agreement', loss, dec) > AGREE_FLAG
            flags_c = table(d0, 's_hat_clamped', loss, dec) > 0
            for m in MICE:
                y = t.loc[m].to_numpy()
                col = MOUSE_COLOR[m]
                ax.plot(x, y, '-', lw=0.9, color=col, alpha=0.75, zorder=2)
                if metric == 's_hat':
                    rs, cl = flags_r.loc[m].to_numpy(), flags_c.loc[m].to_numpy()
                    plain = ~rs & ~cl
                    ax.plot(x[plain], y[plain], MOUSE_MARKER[m], ms=4.5, color=col,
                            zorder=3)
                    ax.plot(x[rs & ~cl], y[rs & ~cl], 'o', ms=5.5, mfc='none',
                            mec=col, mew=1.2, zorder=3)
                    ax.plot(x[cl], y[cl], 'x', ms=8, color=col, mew=2.0, zorder=4)
                else:
                    ax.plot(x, y, MOUSE_MARKER[m], ms=4, color=col, zorder=3)
            med = t.median(axis=0).to_numpy()
            ax.plot(x, med, '-', lw=3.0, color='k', alpha=0.85, zorder=5)
            ax.axhline(ref, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=1)
            if logy:
                ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(ARCHS, fontsize=8)
            ax.grid(True, axis='y', lw=0.4, alpha=0.35)
            ttl = f'{FAMILY_LABEL[loss]} · {dec_lab}'
            if metric == 'overfit_gap' and loss == 'ce':
                # same decoder as KL, but the CE null = KL null + H(target), so
                # the ratio is in different units and NOT comparable to KL's
                ttl += '\n(÷ CE null = KL null + H(target): not comparable)'
            ax.set_title(ttl, fontsize=10 if '\n' not in ttl else 8.5,
                         color=FAMILY_COLOR[loss], fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{title} · {dec_lab}', fontsize=9)
            if r == len(DECODERS) - 1:
                ax.set_xlabel('architecture (lin → rank-8 → tanh width)', fontsize=8)
    # share y among the four CALIBRATED panels of a row (so a 10 % difference is
    # visible); the evar panel spans an order of magnitude more and keeps its own
    for r in range(len(DECODERS)):
        cal = [axes[r, c] for c, loss in enumerate(LOSSES) if loss != 'pca']
        lo = min(ax.get_ylim()[0] for ax in cal)
        hi = max(ax.get_ylim()[1] for ax in cal)
        for ax in cal:
            ax.set_ylim(lo, hi)
        for ax in axes[r]:
            if logy:
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi * 1.15)      # keep a clamped 6.00 off the frame
                _log_ticks(ax)
    fig.supylabel(ylab, fontsize=9)
    ref_label = {'kl_skill': 'null (= 1)', 'proj_skill': 'null (= 1)',
                 's_hat': 'target width (ŝ = 1)', 'overfit_gap': 'no gap (= 0)'}[metric]
    handles = [Line2D([0], [0], marker=MOUSE_MARKER[m], ls='-', lw=0.9,
                      color=MOUSE_COLOR[m], ms=4, label=f'mouse {m}') for m in MICE]
    handles += [Line2D([0], [0], ls='-', lw=3, color='k', label='median (6 mice)'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label=ref_label)]
    if metric == 's_hat':
        handles += [Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.3',
                           mew=1.2, ms=5.5, label='ring = reshapes (agreement > 0.10)'),
                    Line2D([0], [0], marker='x', ls='none', color='0.3', mew=2, ms=8,
                           label='× = clamped (ladder end, a bound)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=10 if metric == 's_hat' else 8,
               fontsize=8, frameon=False)
    fig.suptitle(f'{title} vs architecture — io_hmm_v3, λ_H = 0, weight decay 0, '
                 '6 mice, IO-HMM marginal targets (72 × 2.5°)', fontsize=11)
    save_fig(fig, out_dir, f'width_ladder_{metric}')


def fig_dumbbell(d0, out_dir, counts):
    trio = ('lin', 'rr8', 'h8')
    fig, axes = plt.subplots(len(DECODERS), len(CALIBRATED), figsize=(13.2, 6.6),
                             sharey=True, sharex='row')
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, loss in enumerate(CALIBRATED):
            ax = axes[r, c]
            t = table(d0, 'kl_skill', loss, dec)[list(trio)]
            for m in MICE:
                y = m
                v = t.loc[m].to_numpy()
                ax.plot(v, [y, y, y], '-', lw=1.4, color='0.6', zorder=1)
                ax.plot(v[0], y, 's', ms=7, mfc='white', mec=FAMILY_COLOR[loss],
                        mew=1.5, zorder=3)
                ax.plot(v[1], y, 'D', ms=6, mfc=FAMILY_COLOR[loss], mec='k',
                        mew=0.6, alpha=0.55, zorder=3)
                ax.plot(v[2], y, 'o', ms=7, mfc=FAMILY_COLOR[loss], mec='k',
                        mew=0.6, zorder=4)
            k = counts[(loss, dec)]
            ax.set_title(f'{FAMILY_LABEL[loss]} · {dec_lab}\n'
                         f'rr8 better than lin: {k["rr8<lin"]}/6 · '
                         f'h8 better than rr8: {k["h8<rr8"]}/6',
                         fontsize=9.5, color=FAMILY_COLOR[loss])
            ax.set_xscale('log')
            ax.axvline(1.0, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=0)
            ax.set_yticks(list(MICE))
            ax.set_yticklabels([f'mouse {m}' for m in MICE], fontsize=8)
            ax.set_ylim(-0.6, len(MICE) - 0.4)
            ax.invert_yaxis()
            ax.grid(True, axis='x', lw=0.4, alpha=0.35)
        for ax in axes[r]:
            _log_ticks(ax, 'x')
            if r == len(DECODERS) - 1:
                ax.set_xlabel('KL skill (÷ LOO predict-mean null; < 1 beats null)',
                              fontsize=8)
    handles = [Line2D([0], [0], marker='s', ls='none', mfc='white', mec='0.3', mew=1.5,
                      ms=7, label='lin  Linear(n, 72)'),
               Line2D([0], [0], marker='D', ls='none', mfc='0.5', mec='k', ms=6,
                      label='rr8  Linear(n, 8) → Linear(8, 72), no nonlinearity'),
               Line2D([0], [0], marker='o', ls='none', mfc='0.3', mec='k', ms=7,
                      label='h8  Linear(n, 8) → tanh → Linear(8, 72)'),
               Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label='null (= 1)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=8,
               frameon=False)
    fig.suptitle('rank bottleneck vs nonlinearity: KL skill at lin → rr8 → h8, '
                 'calibrated losses, λ_H = 0 (CE ≡ KL by construction, same seed)\n'
                 'KL skill = mean KL(target‖decoded) / LOO predict-mean null on '
                 'held-out trials; log axis', fontsize=11)
    save_fig(fig, out_dir, 'rr8_vs_h8_vs_lin')


def best_arch(d0, metric):
    """{(loss, dec): DataFrame mouse -> (best arch, best/h8 - 1)} and, per
    (loss, dec), the number of mice whose best WIDE rung (h16/h32/h64) beats h8 by
    more than 10 %."""
    best, wide_win = {}, {}
    for dec, _ in DECODERS:
        for loss in LOSSES:
            t = table(d0, metric, loss, dec)
            best[(loss, dec)] = pd.DataFrame({
                'arch': t.idxmin(axis=1),
                'margin': t.min(axis=1) / t['h8'] - 1.0})    # ≤ 0 by construction
            ratio = t[list(WIDE)].min(axis=1) / t['h8']      # best wide / h8
            wide_win[(loss, dec)] = int((ratio < 0.9).sum())  # > 10 % better
    return best, wide_win


def fig_best_arch(d0, out_dir, best_by_metric, wide_by_metric):
    rank = {a: i for i, a in enumerate(ARCHS)}
    cmap = plt.get_cmap('viridis', len(ARCHS))
    metrics = ('kl_skill', 'proj_skill')
    fig, axes = plt.subplots(len(DECODERS), len(metrics), figsize=(10.5, 7.2))
    for r, (dec, dec_lab) in enumerate(DECODERS):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            best, wide = best_by_metric[metric], wide_by_metric[metric]
            grid = np.zeros((len(MICE) + 1, len(LOSSES)))
            for j, loss in enumerate(LOSSES):
                b = best[(loss, dec)]
                for i, m in enumerate(MICE):
                    a, mg = b.loc[m, 'arch'], b.loc[m, 'margin']
                    grid[i, j] = rank[a]
                    txt = 'white' if rank[a] < 4 else 'k'
                    ax.text(j, i - 0.13, a, ha='center', va='center', fontsize=11,
                            color=txt, fontweight='bold')
                    ax.text(j, i + 0.25, f'{mg * 100:+.0f}% vs h8', ha='center',
                            va='center', fontsize=7.5, color=txt)
                ax.text(j, len(MICE), f'{wide[(loss, dec)]}/6', ha='center',
                        va='center', fontsize=11, color='k')
            grid[len(MICE), :] = np.nan
            ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5,
                      vmax=len(ARCHS) - 0.5, aspect='auto')
            ax.set_xticks(range(len(LOSSES)))
            ax.set_xticklabels([FAMILY_LABEL[l].replace(' (', '\n(') for l in LOSSES],
                               fontsize=8.5)
            ax.set_yticks(range(len(MICE) + 1))
            ax.set_yticklabels([f'mouse {m}' for m in MICE]
                               + ['mice with a width > 8\n> 10 % better than h8'],
                               fontsize=8.5)
            ax.axhline(len(MICE) - 0.5, color='k', lw=1.0)
            ax.set_title(f'best architecture under {METRICS[metric][0]} · {dec_lab}',
                         fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
    handles = [Line2D([0], [0], marker='s', ls='none', ms=10, color=cmap(i), label=a)
               for i, a in enumerate(ARCHS)]
    fig.legend(handles=handles, loc='outside lower center', ncol=7, fontsize=8.5,
               frameon=False, title='cell colour = best architecture (ladder order)',
               title_fontsize=8.5)
    fig.suptitle('best architecture per mouse × loss family, λ_H = 0 '
                 '(argmin over lin, rr8, h4, h8, h16, h32, h64; skill < 1 beats the null)',
                 fontsize=11)
    save_fig(fig, out_dir, 'best_arch_grid')


# ----------------------------------------------------------------------
# prior scoring
# ----------------------------------------------------------------------
def _fmt(v):
    return ' '.join(f'{x:5.2f}' for x in v)


def _cls(ok, partial=False):
    return '✓ confirmed' if ok else ('↔ partial' if partial else '✗ falsified')


def score_priors(d0):
    out = []
    P = out.append
    P('PRIOR SCORING (PREDICTIONS.md 2026-08-22), lambda_H = 0, 6 mice; ratios are '
      'cell/h8 of the per-mouse metric (< 1 = better than h8 for skills)')
    P('')
    # ---------- (a) flat in width from h8 up, calibrated losses ----------
    P('(a) calibrated losses FLAT in width from h8 up on KL skill (h16/h32/h64 within '
      '~10% of h8); h4 slightly worse. Falsifier: 2 mice with h64 20% better than h8.')
    n_within_all, n_wide_tests = 0, 0
    falsifier_hits = {}
    h4_better = {}
    worse_by_10 = {}
    for loss in CALIBRATED:
        for dec, _ in DECODERS:
            t = table(d0, 'kl_skill', loss, dec)
            rat = t.div(t['h8'], axis=0)
            line = f'  {loss:8s} {dec}: '
            for w in WIDE:
                r = rat[w].to_numpy()
                n_in = int((np.abs(r - 1) <= 0.10).sum())
                n_within_all += n_in
                n_wide_tests += len(r)
                line += f'{w}/h8 [{_fmt(r)}] within10%: {n_in}/6 | '
            falsifier_hits[(loss, dec)] = int((rat['h64'] <= 0.80).sum())
            worse_by_10[(loss, dec)] = int((rat['h64'] >= 1.10).sum())
            h4 = rat['h4'].to_numpy()
            h4_better[(loss, dec)] = int((h4 < 1).sum())
            line += f'h4/h8 [{_fmt(h4)}] h4 better: {h4_better[(loss, dec)]}/6'
            P(line)
    n_fals = sum(falsifier_hits.values())
    P(f'  h64 ≥ 20% BETTER than h8 (falsifier): {n_fals} of {len(falsifier_hits) * 6} '
      f'(loss, decoder, mouse) cases -> falsifier did not fire')
    P(f'  wide rungs within ±10% of h8: {n_within_all}/{n_wide_tests} cases; '
      f'h64 ≥ 10% WORSE than h8: '
      + ', '.join(f'{l}-{d} {k}/6' for (l, d), k in worse_by_10.items()))
    P('  h4 better than h8: ' + ', '.join(f'{l}-{d} {k}/6' for (l, d), k in h4_better.items()))
    kl_spat = table(d0, 'kl_skill', 'kl', 'spat')
    kl_temp = table(d0, 'kl_skill', 'kl', 'temp')
    P(f'  KL-loss h64/h8 medians: spatial {float((kl_spat.h64 / kl_spat.h8).median()):.2f}, '
      f'temporal {float((kl_temp.h64 / kl_temp.h8).median()):.2f}; '
      f'h16/h8: spatial {float((kl_spat.h16 / kl_spat.h8).median()):.2f}, '
      f'temporal {float((kl_temp.h16 / kl_temp.h8).median()):.2f}')
    P('  OUTCOME (a): ↔ partial — no width above 8 helps (falsifier silent, h16 is '
      'within 10% of h8 in most cases) but the ladder is NOT flat: h64 is ≥10% worse '
      'than h8 in 4-5/6 mice for every SPATIAL group but only 2/6 temporal, i.e. skill DEGRADES with '
      'width; and h4 is NOT slightly worse — it equals or beats h8 in most mice.')
    P('')
    # ---------- (b) evar over-sharpening grows with width ----------
    P('(b) evar over-sharpening GROWS with width: s_hat h64 > h8 by ≥ 25% in ≥ 4/6 mice. '
      'Falsifier: flat/decreasing in 4/6.')
    ok_b = True
    for dec, _ in DECODERS:
        t = table(d0, 's_hat', 'pca', dec)
        cl = table(d0, 's_hat_clamped', 'pca', dec)
        r = (t['h64'] / t['h8']).to_numpy()
        n = int((r >= 1.25).sum())
        ok_b &= n >= 4
        # monotonicity across h4..h64: Spearman sign per mouse
        ladder = t[['h4', 'h8', 'h16', 'h32', 'h64']].to_numpy()
        rho = [pd.Series(row).corr(pd.Series(np.arange(5)), method='spearman')
               for row in ladder]
        n_mono = int(sum(x > 0 for x in rho))
        clamped = [f'm{m}:{a}' for m in MICE for a in ('h8', 'h64') if cl.loc[m, a] > 0]
        P(f'  {dec}: s_hat h8 [{_fmt(t["h8"])}] h64 [{_fmt(t["h64"])}] '
          f'h64/h8 [{_fmt(r)}] ≥1.25: {n}/6; rising over h4..h64 (Spearman>0): '
          f'{n_mono}/6; clamped at h8/h64: {clamped if clamped else "none"}')
    P(f'  OUTCOME (b): {_cls(ok_b)} — h64/h8 ≥ 1.25 in 6/6 mice in BOTH decoders '
      '(temporal m5 h64 is a clamped bound of 6.00, so its ratio is a lower bound).')
    P('')
    # ---------- (c) rr8 ~= h8 ----------
    P('(c) rr8 ≈ h8 for calibrated losses (KL skill within 10%, 5/6 mice). '
      'Falsifier: rr8 worse by > 25% in 4+ mice.')
    counts = {}
    fals_c = []
    for loss in CALIBRATED:
        for dec, _ in DECODERS:
            t = table(d0, 'kl_skill', loss, dec)
            r = (t['rr8'] / t['h8']).to_numpy()
            r_lin = (t['rr8'] / t['lin']).to_numpy()
            n_in = int((np.abs(r - 1) <= 0.10).sum())
            n_worse25 = int((r > 1.25).sum())
            k = {'rr8<lin': int((t['rr8'] < t['lin']).sum()),
                 'h8<rr8': int((t['h8'] < t['rr8']).sum())}
            counts[(loss, dec)] = k
            fals_c.append(n_worse25 >= 4)
            P(f'  {loss:8s} {dec}: rr8/h8 [{_fmt(r)}] within10%: {n_in}/6, '
              f'worse>25%: {n_worse25}/6 | rr8/lin [{_fmt(r_lin)}] '
              f'rr8<lin {k["rr8<lin"]}/6, h8<rr8 {k["h8<rr8"]}/6')
    n_groups_fals = sum(fals_c)
    P(f'  falsifier (rr8 worse by >25% in ≥4 mice) fires in {n_groups_fals}/'
      f'{len(fals_c)} (loss, decoder) groups — every spatial group; temporal groups '
      'sit at 9–29% worse (JS 1/6 within 10%, the rest 0/6).')
    P('  OUTCOME (c): ✗ falsified — the tanh IS load-bearing at rank 8: h8 beats rr8 '
      'in 6/6 mice for every calibrated loss and decoder; rr8 recovers only part of '
      'the lin→h8 gain. Rank bottleneck alone is not the mechanism.')
    P('')
    # ---------- (d) ce ~= kl ~= js ----------
    P('(d) CE ≈ KL ≈ JS within ~5% (KL skill).')
    worst_js, worst_ce = 0.0, 0.0
    n_js_in, n_js = 0, 0
    for arch in ARCHS:
        for dec, _ in DECODERS:
            kl = table(d0, 'kl_skill', 'kl', dec)[arch]
            js = table(d0, 'kl_skill', 'js', dec)[arch]
            ce = table(d0, 'kl_skill', 'ce', dec)[arch]
            rj = (js / kl).to_numpy()
            rc = (ce / kl).to_numpy()
            worst_js = max(worst_js, float(np.abs(rj - 1).max()))
            worst_ce = max(worst_ce, float(np.abs(rc - 1).max()))
            n_js_in += int((np.abs(rj - 1) <= 0.05).sum())
            n_js += len(rj)
    kl_all = d0[d0.loss_family == 'kl'].set_index(['arch', 'mouse', 'decoder'])
    ce_all = d0[d0.loss_family == 'ce'].set_index(['arch', 'mouse', 'decoder'])
    same = ((ce_all.kl_skill - kl_all.kl_skill).abs() == 0)
    be_shift = (ce_all.best_epoch - kl_all.best_epoch).abs()
    P(f'  CE/KL: kl_skill identical to the last digit in {int(same.sum())}/{len(same)} '
      f'(arch, decoder, mouse) cells; the other {int((~same).sum())} differ by at most '
      f'{worst_ce * 100:.2f}% with best_epoch shifted by ≤ {int(be_shift.max())} epoch — '
      'CE = KL + H(target) with one seed, so the two families are the same training '
      'problem up to floating-point non-determinism; CE ≈ KL holds by construction, '
      'not as evidence. (overfit_gap is NOT comparable across the pair: its denominator '
      'is the own-loss null, and the CE null carries H(target).)')
    P(f'  JS/KL: max |ratio − 1| = {worst_js * 100:.1f}%; within 5%: {n_js_in}/{n_js} '
      '(arch, decoder, mouse) cases.')
    for dec, _ in DECODERS:
        kl = table(d0, 'kl_skill', 'kl', dec)['h8']
        js = table(d0, 'kl_skill', 'js', dec)['h8']
        P(f'  h8 {dec}: js/kl [{_fmt((js / kl).to_numpy())}]')
    P(f'  OUTCOME (d): {"✓ confirmed" if worst_js <= 0.05 else "↔ magnitude (within 10%, not the registered 5%)"} for JS vs KL '
      f'(worst case {worst_js * 100:.1f}%, the 5% band holds in {n_js_in}/{n_js}); '
      'CE vs KL is an identity, not evidence.')
    P('')
    P('(e) lambda_H — not scored here (this script is the lambda_H = 0 slice).')
    return out, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', type=Path, default=CSV)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    d0 = load(args.csv)
    print(f'{len(d0)} lambda_H = 0 rows from {args.csv}')

    for metric in ('kl_skill', 'proj_skill', 's_hat', 'overfit_gap'):
        fig_width_ladder(d0, metric, args.out_dir)

    lines, counts = score_priors(d0)
    fig_dumbbell(d0, args.out_dir, counts)

    best_by, wide_by = {}, {}
    for metric in ('kl_skill', 'proj_skill'):
        best_by[metric], wide_by[metric] = best_arch(d0, metric)
    fig_best_arch(d0, args.out_dir, best_by, wide_by)

    print()
    print('\n'.join(lines))
    print()
    print('BEST ARCH per mouse (rows = mouse, cols = loss family)')
    for metric in ('kl_skill', 'proj_skill'):
        for dec, dec_lab in DECODERS:
            print(f'  {METRICS[metric][0]} · {dec_lab}:')
            print('    mouse ' + ' '.join(f'{l:>9s}' for l in LOSSES)
                  + '   (best arch, best/h8 − 1)')
            for m in MICE:
                print(f'    {m:5d} ' + ' '.join(
                    f'{best_by[metric][(l, dec)].loc[m, "arch"]:>4s}'
                    f'{best_by[metric][(l, dec)].loc[m, "margin"] * 100:+4.0f}%'
                    for l in LOSSES))
            print('    w>8 beats h8 by >10%: '
                  + ' '.join(f'{wide_by[metric][(l, dec)]:>5d}/6 ' for l in LOSSES))
    tot = {metric: sum(v for (l, d), v in wide_by[metric].items() if l in CALIBRATED)
           for metric in wide_by}
    print(f'  HEADLINE: mice where a width > 8 improves on h8 by > 10% — calibrated '
          f'losses, summed over {len(CALIBRATED)} losses × 2 decoders × 6 mice = '
          f'{len(CALIBRATED) * 12} cases: KL skill {tot["kl_skill"]}, projection skill '
          f'{tot["proj_skill"]}')
    return d0


if __name__ == '__main__':
    main()
