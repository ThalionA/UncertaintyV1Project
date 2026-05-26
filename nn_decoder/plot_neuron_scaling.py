# -*- coding: utf-8 -*-
"""Plot neural-population scaling curves from ``neuron_scaling.py`` output.

Reads the per-mouse CSVs written by ``run_neuron_scaling.py`` (so it
picks up whatever has finished — it does not wait for the aggregate)
and produces, per mouse:

  - ``mouse_<id>_<target>_<split>_detail.png`` — a decoder x ranking
    grid. Each panel compares the random-subsampling curve against that
    ranking's top-first and bottom-first curves, with the target-shuffle
    control as the chance reference.
  - ``mouse_<id>_<target>_<split>_headline.png`` — the plain "how does
    performance scale" view: the random curve for both decoders.

Pure visualisation — no decoder training, no heavy processing (those
live in ``neuron_scaling.py``), per the repo's processing / plotting
split. Figures land in ``nn_decoder/figures/neuron_scaling/``; the raw
data they are built from persists in ``results/neuron_scaling/``.

Usage
-----
    python plot_neuron_scaling.py                       # all mice found
    python plot_neuron_scaling.py --mice 0 3            # specific mice
    python plot_neuron_scaling.py --target Q --split stratified_balanced
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — write PNGs, never open a window
import matplotlib.pyplot as plt

from paths import RESULTS, figures_dir
from neuron_scaling import aggregate_across_mice, normalise_to_full

TARGET_LABELS = {
    'Q': 'Q (perceptual posterior)', 'L': 'L (likelihood)',
    'd': 'd (decision posterior)', 'choice': 'choice',
    'stim_kernel': 'stim kernel', 'stim_cat': 'stim category',
}
# (model key, real/shuffle, human label)
DECODERS = [('spat', 'Spatial decoder (PPC)'),
            ('temp', 'Temporal decoder (SBC)')]
RANKINGS = [('orientation_tuning', 'Orientation tuning'),
            ('weight_magnitude', 'Weight magnitude'),
            ('mean_activity', 'Mean activity')]
RANK_COLOR = {'orientation_tuning': '#1f77b4',
              'weight_magnitude': '#d62728',
              'mean_activity': '#2ca02c'}

Y_LABEL = 'Held-out fit loss  (lower = better)'
X_LABEL = 'Number of decoded neurons'


# ----------------------------------------------------------------------
# Curve extraction
# ----------------------------------------------------------------------

def _full_point(df, model, value_col='fit_loss'):
    """(n_neurons, value) for the full-population fit, or None."""
    sub = df[(df['selection'] == 'full') & (df['model'] == model)]
    if sub.empty:
        return None
    return int(sub['n_neurons'].iloc[0]), float(sub[value_col].iloc[0])


def random_curve(df, model, value_col='fit_loss'):
    """Per-N mean / std / draw-points for the random-subsampling sweep,
    with the full-population point appended as the final N.

    ``value_col`` selects the y-quantity — ``'fit_loss'`` for the raw
    curve or ``'fit_loss_rel_full'`` for the full-normalised curve."""
    sub = df[(df['selection'] == 'random') & (df['model'] == model)]
    g = (sub.groupby('n_neurons')[value_col]
            .agg(['mean', 'std', 'count']).reset_index()
            .sort_values('n_neurons'))
    xs = list(g['n_neurons']); mean = list(g['mean'])
    std = [0.0 if np.isnan(s) else s for s in g['std']]
    fp = _full_point(df, model, value_col)
    if fp is not None and fp[0] not in xs:
        xs.append(fp[0]); mean.append(fp[1]); std.append(0.0)
    order = np.argsort(xs)
    return (np.array(xs)[order], np.array(mean)[order], np.array(std)[order],
            sub[['n_neurons', value_col]])


def targeted_curve(df, selection, direction, model):
    """(n_neurons, fit_loss) for a targeted curve, full point appended."""
    sub = df[(df['selection'] == selection)
             & (df['direction'] == direction)
             & (df['model'] == model)].sort_values('n_neurons')
    xs = list(sub['n_neurons']); ys = list(sub['fit_loss'])
    fp = _full_point(df, model)
    if fp is not None and fp[0] not in xs:
        xs.append(fp[0]); ys.append(fp[1])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_detail(df, mouse, target, split, out_path):
    """decoder (rows) x ranking (cols) grid: random vs top/bottom."""
    fig, axes = plt.subplots(len(DECODERS), len(RANKINGS),
                             figsize=(15, 8.5), sharex=True, sharey='row')
    axes = np.atleast_2d(axes)

    for r, (model, dec_label) in enumerate(DECODERS):
        rx, rmean, rstd, rpts = random_curve(df, model)
        sx, smean, _, _ = random_curve(df, model + '_shf')
        for c, (rank_key, rank_label) in enumerate(RANKINGS):
            ax = axes[r, c]
            colour = RANK_COLOR[rank_key]

            # Random subsampling: mean +/- spread over draws, draws as dots.
            if rx.size:
                ax.fill_between(rx, rmean - rstd, rmean + rstd,
                                color='0.6', alpha=0.25, zorder=1)
                ax.plot(rx, rmean, '-o', color='0.35', lw=1.8, ms=5,
                        label='Random (mean)', zorder=3)
                ax.scatter(rpts['n_neurons'], rpts['fit_loss'], s=14,
                           color='0.55', alpha=0.7, zorder=2,
                           label='Random (draws)')
            # Target-shuffle control — chance reference.
            if sx.size:
                ax.plot(sx, smean, ':', color='0.5', lw=1.5,
                        label='Shuffle control', zorder=2)
            # Targeted: top-first and bottom-first by this ranking.
            tx, ty = targeted_curve(df, rank_key, 'top', model)
            bx, by = targeted_curve(df, rank_key, 'bottom', model)
            if tx.size:
                ax.plot(tx, ty, '-s', color=colour, lw=2.0, ms=6,
                        label='Targeted: top-ranked first', zorder=4)
            if bx.size:
                ax.plot(bx, by, '--^', color=colour, lw=2.0, ms=6,
                        label='Targeted: bottom-ranked first', zorder=4)

            if r == 0:
                ax.set_title(rank_label, fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{dec_label}\n{Y_LABEL}', fontsize=9)
            if r == len(DECODERS) - 1:
                ax.set_xlabel(X_LABEL, fontsize=9)
            ax.grid(alpha=0.25)

    # One shared legend (entries are identical across panels).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Population scaling — Mouse {mouse}  |  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_headline(df, mouse, target, split, out_path):
    """The plain scaling curve: random subsampling, both decoders."""
    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11, 4.6),
                             sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        rx, rmean, rstd, rpts = random_curve(df, model)
        sx, smean, _, _ = random_curve(df, model + '_shf')
        if rx.size:
            ax.fill_between(rx, rmean - rstd, rmean + rstd,
                            color='#1f77b4', alpha=0.18)
            ax.plot(rx, rmean, '-o', color='#1f77b4', lw=2.2, ms=6,
                    label='Random subsampling (mean)')
            ax.scatter(rpts['n_neurons'], rpts['fit_loss'], s=18,
                       color='#1f77b4', alpha=0.55, label='Individual draws')
        if sx.size:
            ax.plot(sx, smean, ':', color='0.5', lw=1.8,
                    label='Shuffle control')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(Y_LABEL, fontsize=10)
    axes[-1].legend(fontsize=9, frameon=False)
    fig.suptitle(
        f'How decoding scales with population size — Mouse {mouse}  |  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Cross-mouse aggregate figures
# ----------------------------------------------------------------------

def _agg_curve(agg, selection, direction, model, full_only_n_mice=None,
               value_col='fit_loss'):
    """(n_neurons, value) from an aggregate frame, sorted by N.

    ``full_only_n_mice`` keeps only rows where every mouse contributed —
    i.e. genuine cross-mouse means rather than partial-coverage points.
    """
    a = agg[(agg['selection'] == selection)
            & (agg['direction'] == direction)
            & (agg['model'] == model)]
    if full_only_n_mice is not None:
        a = a[a['n_mice'] == full_only_n_mice]
    a = a.sort_values('n_neurons')
    return a['n_neurons'].to_numpy(), a[value_col].to_numpy()


def _agg_curve_sem(agg, selection, direction, model, full_only_n_mice=None,
                   value_col='fit_loss'):
    """(n_neurons, value, sem-across-mice) from an aggregate frame.

    SEM uses the SD across mice (recorded by ``aggregate_across_mice`` as
    ``<value_col>_sd_across_mice``) divided by sqrt(n_mice). Returns
    zeros for SEM when the SD column is missing.
    """
    a = agg[(agg['selection'] == selection)
            & (agg['direction'] == direction)
            & (agg['model'] == model)]
    if full_only_n_mice is not None:
        a = a[a['n_mice'] == full_only_n_mice]
    a = a.sort_values('n_neurons')
    xs = a['n_neurons'].to_numpy()
    ys = a[value_col].to_numpy(dtype=float)
    sd_col = f'{value_col}_sd_across_mice'
    if sd_col in a.columns and a['n_mice'].notna().all():
        sd = a[sd_col].to_numpy(dtype=float)
        nm = np.maximum(a['n_mice'].to_numpy(dtype=float), 1.0)
        sem = sd / np.sqrt(nm)
    else:
        sem = np.zeros_like(ys)
    return xs, ys, sem


# ----------------------------------------------------------------------
# Across-draw SEM at the cross-mouse level
# ----------------------------------------------------------------------

def _draw_sem_across_mice(df, model, value_col='fit_loss'):
    """Random-curve mean + combined per-mouse-across-draw SEM, per N.

    Per mouse and per N: mean and SEM = SD/sqrt(n_draws) across the
    random draws. Then combine across mice with trial-count weights:
      mean_agg = Σ w_i m_i / Σ w_i
      SEM_agg  = sqrt(Σ (w_i SEM_i)^2) / Σ w_i      (SE of weighted mean)
    where w_i = n_test_trials of mouse i. This propagates each mouse's
    Monte-Carlo (random subsampling) uncertainty into the aggregate
    band — it does NOT include between-mouse spread (use
    :func:`_agg_curve_sem` for that).

    Returns (xs, ys, sem) sorted by N; xs is the union of Ns where at
    least one mouse contributed at least one draw.
    """
    sub = df[(df['selection'] == 'random') & (df['model'] == model)]
    if sub.empty:
        return np.array([]), np.array([]), np.array([])
    per_mouse = sub.groupby(['mouse', 'n_neurons']).agg(
        mean=(value_col, 'mean'),
        std=(value_col, 'std'),
        cnt=(value_col, 'count'),
        trials=('n_test_trials', 'first'),
    ).reset_index()
    per_mouse['sem'] = (per_mouse['std']
                       / np.sqrt(np.maximum(per_mouse['cnt'], 1)))
    per_mouse['sem'] = per_mouse['sem'].fillna(0.0)

    xs, ys, sems = [], [], []
    for n, g in per_mouse.groupby('n_neurons'):
        w = g['trials'].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        m = g['mean'].to_numpy(dtype=float)
        s = g['sem'].to_numpy(dtype=float)
        xs.append(int(n))
        ys.append(float(np.sum(w * m) / np.sum(w)))
        sems.append(float(np.sqrt(np.sum((w * s) ** 2)) / np.sum(w)))
    order = np.argsort(xs)
    return (np.array(xs)[order], np.array(ys)[order], np.array(sems)[order])


# ----------------------------------------------------------------------
# Relative-N (n_neurons / n_full) helpers
# ----------------------------------------------------------------------

REL_GRID = np.round(np.linspace(0.1, 1.0, 19), 4)


def _full_n_per_mouse(df):
    """{mouse_id: full-population neuron count}."""
    full = df[df['selection'] == 'full']
    if full.empty:
        return {}
    return (full.groupby('mouse')['n_neurons']
                .first().astype(int).to_dict())


def _mouse_curve_xy(df, mid, selection, direction, model,
                    value_col='fit_loss', include_full=True):
    """(xs, ys, n_test_trials) for one mouse's curve.

    Collapses random draws to per-N means. ``include_full`` appends the
    full-population point so the curve ends at the mouse's n_full.
    """
    sub = df[(df['mouse'] == mid)
             & (df['selection'] == selection)
             & (df['direction'] == direction)
             & (df['model'] == model)].sort_values('n_neurons')
    if sub.empty:
        return np.array([]), np.array([]), None
    g = sub.groupby('n_neurons').agg(
        y=(value_col, 'mean'),
        tr=('n_test_trials', 'first'),
    ).reset_index().sort_values('n_neurons')
    xs = g['n_neurons'].to_numpy(dtype=float)
    ys = g['y'].to_numpy(dtype=float)
    tr = int(g['tr'].iloc[0]) if not g.empty else None
    if include_full:
        fp = df[(df['mouse'] == mid)
                & (df['selection'] == 'full')
                & (df['model'] == model)]
        if not fp.empty:
            nfx = float(fp['n_neurons'].iloc[0])
            fpy = float(fp[value_col].iloc[0])
            if nfx not in xs:
                xs = np.append(xs, nfx)
                ys = np.append(ys, fpy)
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order]
            if tr is None:
                tr = int(fp['n_test_trials'].iloc[0])
    return xs, ys, tr


def _aggregate_on_rel_grid(df, selection, direction, model,
                            value_col='fit_loss', rel_grid=REL_GRID):
    """Per-mouse interpolate to ``rel_grid`` (n/n_full), then trial-count
    weighted mean + SEM across mice. Returns (rel_grid, mean, sem).

    Points outside a mouse's observed (rel-N) range are NaN for that
    mouse and excluded from the grid-point average.
    """
    full_n = _full_n_per_mouse(df)
    Y = []
    W = []
    for mid in sorted(df['mouse'].unique()):
        if mid not in full_n:
            continue
        xs, ys, tr = _mouse_curve_xy(
            df, mid, selection, direction, model, value_col=value_col)
        if xs.size == 0:
            continue
        rx = xs / float(full_n[mid])
        order = np.argsort(rx)
        rx, ry = rx[order], ys[order]
        yi = np.interp(rel_grid, rx, ry, left=np.nan, right=np.nan)
        Y.append(yi)
        W.append(float(tr) if tr is not None else 1.0)
    if not Y:
        return rel_grid, np.full_like(rel_grid, np.nan, dtype=float), \
               np.full_like(rel_grid, np.nan, dtype=float)
    Y = np.stack(Y)                                     # (n_mice, n_grid)
    W = np.asarray(W, dtype=float)[:, None]
    mask = np.isfinite(Y).astype(float)
    sw = (W * mask).sum(axis=0)
    mean = np.where(sw > 0, np.nansum(W * Y, axis=0) / np.where(sw > 0, sw, 1.0), np.nan)
    diff = Y - mean[None, :]
    var = np.where(sw > 0, np.nansum((W * mask) * (diff ** 2), axis=0)
                            / np.where(sw > 0, sw, 1.0), np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    n_present = mask.sum(axis=0)
    sem = np.where(n_present > 0, sd / np.sqrt(n_present), np.nan)
    return rel_grid, mean, sem


def _aggregate_paired(df, op, selection='random', direction='na',
                      value_col='fit_loss', include_full=True,
                      model_a='spat', model_b='temp', min_mice=None):
    """Per-mouse model_a OP model_b at each N, then trial-count-weighted
    cross-mouse mean + SEM. ``op`` is ``'diff'`` (a - b) or
    ``'ratio'`` (a / b).

    Pairs the two models on (mouse, selection, direction, n_neurons,
    draw) so the comparison stays within the same trained pair.
    ``include_full=True`` also pulls in the full-population row so the
    final N is present. ``model_a``/``model_b`` default to the spatial
    vs temporal decoders; use ``'spat_shf'``/``'temp_shf'`` for the
    target-shuffle ratio.
    """
    if op not in ('diff', 'ratio'):
        raise ValueError(f"op must be 'diff' or 'ratio'; got {op!r}")
    key = ['mouse', 'selection', 'direction', 'n_neurons', 'draw']
    sel_mask = (df['selection'] == selection) & (df['direction'] == direction)
    if include_full:
        sel_mask |= (df['selection'] == 'full')
    a = (df[(df['model'] == model_a) & sel_mask]
         [key + [value_col, 'n_test_trials']]
         .rename(columns={value_col: 'a'}))
    b = (df[(df['model'] == model_b) & sel_mask]
         [key + [value_col]]
         .rename(columns={value_col: 'b'}))
    paired = pd.merge(a, b, on=key)
    if paired.empty:
        return np.array([]), np.array([]), np.array([])
    if op == 'diff':
        paired['v'] = paired['a'] - paired['b']
    else:
        paired['v'] = paired['a'] / paired['b']
    # per-(mouse, N): mean across draws + n_test_trials
    pm = paired.groupby(['mouse', 'n_neurons']).agg(
        m=('v', 'mean'), tr=('n_test_trials', 'first')).reset_index()
    xs, ys, sems = [], [], []
    for n, g in pm.groupby('n_neurons'):
        w = g['tr'].to_numpy(dtype=float)
        v = g['m'].to_numpy(dtype=float)
        if w.sum() <= 0 or v.size == 0:
            continue
        if min_mice is not None and v.size < min_mice:
            continue
        mu = float(np.sum(w * v) / np.sum(w))
        if v.size > 1:
            wsd = float(np.sqrt(
                np.sum(w * (v - mu) ** 2) / np.sum(w)))
            sem = wsd / np.sqrt(v.size)
        else:
            sem = 0.0
        xs.append(int(n)); ys.append(mu); sems.append(sem)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order], np.array(sems)[order]


def _aggregate_paired_rel(df, op, selection='random', direction='na',
                          value_col='fit_loss', rel_grid=REL_GRID,
                          include_full=True):
    """Same as :func:`_aggregate_paired` but on the relative-N grid.

    Per-mouse: compute spat OP temp at each absolute N, interpolate the
    resulting curve onto ``rel_grid``, then trial-count-weighted mean
    + SEM across mice on the grid. ``include_full`` pulls the
    full-population row in so the curve reaches rel-N = 1.0.
    """
    if op not in ('diff', 'ratio'):
        raise ValueError(f"op must be 'diff' or 'ratio'; got {op!r}")
    key = ['mouse', 'selection', 'direction', 'n_neurons', 'draw']
    sel_mask = (df['selection'] == selection) & (df['direction'] == direction)
    if include_full:
        sel_mask |= (df['selection'] == 'full')
    spat = (df[(df['model'] == 'spat') & sel_mask]
            [key + [value_col, 'n_test_trials']]
            .rename(columns={value_col: 'spat'}))
    temp = (df[(df['model'] == 'temp') & sel_mask]
            [key + [value_col]]
            .rename(columns={value_col: 'temp'}))
    paired = pd.merge(spat, temp, on=key)
    if paired.empty:
        return rel_grid, np.full_like(rel_grid, np.nan, dtype=float), \
               np.full_like(rel_grid, np.nan, dtype=float)
    if op == 'diff':
        paired['v'] = paired['spat'] - paired['temp']
    else:
        paired['v'] = paired['spat'] / paired['temp']
    full_n = _full_n_per_mouse(df)
    Y, W = [], []
    for mid in sorted(paired['mouse'].unique()):
        if mid not in full_n:
            continue
        pm = (paired[paired['mouse'] == mid]
              .groupby('n_neurons').agg(
                  m=('v', 'mean'), tr=('n_test_trials', 'first'))
              .reset_index().sort_values('n_neurons'))
        if pm.empty:
            continue
        xs = pm['n_neurons'].to_numpy(dtype=float)
        ys = pm['m'].to_numpy(dtype=float)
        rx = xs / float(full_n[mid])
        order = np.argsort(rx)
        rx, ry = rx[order], ys[order]
        yi = np.interp(rel_grid, rx, ry, left=np.nan, right=np.nan)
        Y.append(yi)
        W.append(float(pm['tr'].iloc[0]))
    if not Y:
        return rel_grid, np.full_like(rel_grid, np.nan, dtype=float), \
               np.full_like(rel_grid, np.nan, dtype=float)
    Y = np.stack(Y)
    W = np.asarray(W, dtype=float)[:, None]
    mask = np.isfinite(Y).astype(float)
    sw = (W * mask).sum(axis=0)
    mean = np.where(sw > 0, np.nansum(W * Y, axis=0) / np.where(sw > 0, sw, 1.0), np.nan)
    diff = Y - mean[None, :]
    var = np.where(sw > 0, np.nansum((W * mask) * (diff ** 2), axis=0)
                            / np.where(sw > 0, sw, 1.0), np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    n_present = mask.sum(axis=0)
    sem = np.where(n_present > 0, sd / np.sqrt(n_present), np.nan)
    return rel_grid, mean, sem


def _share_ylim(axes):
    """Set every axis in ``axes`` to the union ylim of all of them."""
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    lows, highs = [], []
    for ax in axes:
        lo, hi = ax.get_ylim()
        if np.isfinite(lo) and np.isfinite(hi):
            lows.append(lo); highs.append(hi)
    if not lows:
        return
    lo, hi = min(lows), max(highs)
    for ax in axes:
        ax.set_ylim(lo, hi)


def plot_aggregate_headline(df, agg, target, split, out_path):
    """Per-mouse random curves overlaid with the cross-mouse weighted
    mean and a shaded SEM-across-mice band.

    SEM = SD-across-mice / sqrt(n_mice); only the full-coverage
    n_mice == cohort-size rows are drawn so the band reflects a genuine
    cross-mouse measurement rather than partial coverage.
    """
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11.5, 4.8), sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        for i, mid in enumerate(mice):
            rx, rmean, _, _ = random_curve(df[df['mouse'] == mid], model)
            if rx.size:
                ax.plot(rx, rmean, '-o', color=cmap(i % 10), lw=1.3, ms=4,
                        alpha=0.65, label=f'Mouse {mid}')
        gx, gy, gsem = _agg_curve_sem(
            agg, 'random', 'na', model, full_only_n_mice=n_mice)
        if gx.size:
            ax.fill_between(gx, gy - gsem, gy + gsem,
                            color='k', alpha=0.18, zorder=4,
                            label='Cross-mouse SEM')
            ax.plot(gx, gy, '-o', color='k', lw=2.8, ms=8, zorder=5,
                    label=f'Cross-mouse mean ({basis}-weighted)')
        sx, sy, ssem = _agg_curve_sem(
            agg, 'random', 'na', model + '_shf', full_only_n_mice=n_mice)
        if sx.size:
            ax.fill_between(sx, sy - ssem, sy + ssem,
                            color='0.6', alpha=0.18, zorder=2)
            ax.plot(sx, sy, ':', color='0.4', lw=1.8, label='Shuffle (mean)')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(Y_LABEL, fontsize=10)
    _share_ylim(axes)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(
        f'Population scaling across mice (n={n_mice})  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_aggregate_detail(df, agg, target, split, out_path):
    """ranking x decoder grid of cross-mouse weighted means: random vs
    targeted top/bottom.

    Layout: rows = neuron rankings (orientation tuning, weight
    magnitude, mean activity), columns = decoders (spatial PPC,
    temporal SBC) so the two decoders sit side-by-side at a shared ylim
    for each ranking.

    The random / shuffle curves carry a shaded across-draws SEM band
    (per-mouse SD/sqrt(n_draws) combined across mice with trial-count
    weights). They are clipped at the largest N the targeted curves
    reach in each panel — past that, the random sweep covers extra Ns
    that have no targeted reference, which clutters the comparison.
    The targeted curves are single fits per N (no draws) so they show
    only the cross-mouse SEM band."""
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'

    fig, axes = plt.subplots(len(RANKINGS), len(DECODERS),
                             figsize=(11, 11.5), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for c, (model, dec_label) in enumerate(DECODERS):
        # Random / shuffle: combined per-mouse-across-draw SEM
        # (computed once per decoder, clipped per panel below).
        rx_full, ry_full, rsem_full = _draw_sem_across_mice(df, model)
        sx_full, sy_full, ssem_full = _draw_sem_across_mice(df, model + '_shf')
        for r, (rank_key, rank_label) in enumerate(RANKINGS):
            ax = axes[r, c]
            colour = RANK_COLOR[rank_key]

            tx, ty, tsem = _agg_curve_sem(
                agg, rank_key, 'top', model, full_only_n_mice=n_mice)
            bx, by, bsem = _agg_curve_sem(
                agg, rank_key, 'bottom', model, full_only_n_mice=n_mice)

            # Clip the random/shuffle sweeps at the max N the targeted
            # curves cover. If neither targeted direction has data, fall
            # back to the random sweep's own range.
            xmax_candidates = [int(np.max(arr)) for arr in (tx, bx)
                               if arr.size]
            xmax = max(xmax_candidates) if xmax_candidates else (
                int(np.max(rx_full)) if rx_full.size else None)

            def _clip(xs, ys, sems, xmax):
                if xs.size == 0 or xmax is None:
                    return xs, ys, sems
                m = xs <= xmax
                return xs[m], ys[m], sems[m]

            rx, ry, rsem = _clip(rx_full, ry_full, rsem_full, xmax)
            sx, sy, ssem = _clip(sx_full, sy_full, ssem_full, xmax)

            if rx.size:
                ax.fill_between(rx, ry - rsem, ry + rsem,
                                color='0.5', alpha=0.22, zorder=1,
                                label='Random ±SEM (across draws)')
                ax.plot(rx, ry, '-o', color='0.35', lw=1.8, ms=5,
                        label='Random', zorder=3)
            if sx.size:
                ax.fill_between(sx, sy - ssem, sy + ssem,
                                color='0.7', alpha=0.18, zorder=1)
                ax.plot(sx, sy, ':', color='0.45', lw=1.5,
                        label='Shuffle control', zorder=2)
            if tx.size:
                ax.fill_between(tx, ty - tsem, ty + tsem,
                                color=colour, alpha=0.15, zorder=2)
                ax.plot(tx, ty, '-s', color=colour, lw=2.0, ms=6,
                        label='Targeted: top-ranked first', zorder=4)
            if bx.size:
                ax.fill_between(bx, by - bsem, by + bsem,
                                color=colour, alpha=0.10, zorder=2)
                ax.plot(bx, by, '--^', color=colour, lw=2.0, ms=6,
                        label='Targeted: bottom-ranked first', zorder=4)

            if r == 0:
                ax.set_title(dec_label, fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{rank_label}\n{Y_LABEL}', fontsize=9)
            if r == len(RANKINGS) - 1:
                ax.set_xlabel(X_LABEL, fontsize=9)
            ax.grid(alpha=0.25)

    _share_ylim(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Cross-mouse mean ({basis}-weighted, n={n_mice} mice)  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_aggregate_normalised(df, target, split, out_path):
    """Per-mouse random curves normalised to each mouse's own full-
    population loss (= 1.0), with the cross-mouse weighted mean.

    Dividing out each mouse's full-population loss removes the
    between-mouse absolute-performance offset, so the curves show the
    *shape* of scaling — how much is given up by decoding from fewer
    neurons — on a common scale.
    """
    ndf = normalise_to_full(df)
    agg = aggregate_across_mice(ndf, value_col='fit_loss_rel_full')
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11.5, 4.8), sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        for i, mid in enumerate(mice):
            rx, rmean, _, _ = random_curve(
                ndf[ndf['mouse'] == mid], model, value_col='fit_loss_rel_full')
            if rx.size:
                ax.plot(rx, rmean, '-o', color=cmap(i % 10), lw=1.3, ms=4,
                        alpha=0.65, label=f'Mouse {mid}')
        gx, gy, gsem = _agg_curve_sem(
            agg, 'random', 'na', model, full_only_n_mice=n_mice,
            value_col='fit_loss_rel_full')
        if gx.size:
            ax.fill_between(gx, gy - gsem, gy + gsem,
                            color='k', alpha=0.18, zorder=4,
                            label='Cross-mouse SEM')
            ax.plot(gx, gy, '-o', color='k', lw=2.8, ms=8, zorder=5,
                    label=f'Cross-mouse mean ({basis}-weighted)')
        ax.axhline(1.0, color='0.55', ls='--', lw=1.3, zorder=1,
                   label='Full population (= 1.0)')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Fit loss relative to full population\n'
                       '(1.0 = full;  higher = worse)', fontsize=10)
    _share_ylim(axes)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(
        f'Scaling normalised to full-population loss (n={n_mice} mice)  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Chance-normalised comparison (each decoder / its own shuffle)
# ----------------------------------------------------------------------

def _aggregate_chance_normalised(df, model, value_col='fit_loss',
                                  min_mice=None):
    """Per-mouse fit_loss / fit_loss_shf at each N, then trial-count-weighted
    cross-mouse mean + SEM. ``model`` is ``'spat'`` or ``'temp'``; the
    matching ``'<model>_shf'`` row is used as the per-mouse denominator.

    Pairing happens on (mouse, selection, direction, n_neurons, draw) so
    real and shuffle are paired within the same trained subset.
    """
    key = ['mouse', 'selection', 'direction', 'n_neurons', 'draw']
    real = (df[df['model'] == model][key + [value_col, 'n_test_trials']]
            .rename(columns={value_col: 'real'}))
    shuf = (df[df['model'] == model + '_shf'][key + [value_col]]
            .rename(columns={value_col: 'shuf'}))
    paired = pd.merge(real, shuf, on=key)
    paired = paired[paired['selection'].isin(['random', 'full'])]
    if paired.empty:
        return np.array([]), np.array([]), np.array([])
    paired['v'] = paired['real'] / paired['shuf']
    pm = paired.groupby(['mouse', 'n_neurons']).agg(
        m=('v', 'mean'), tr=('n_test_trials', 'first')).reset_index()
    xs, ys, sems = [], [], []
    for n, g in pm.groupby('n_neurons'):
        w = g['tr'].to_numpy(dtype=float)
        v = g['m'].to_numpy(dtype=float)
        if w.sum() <= 0 or v.size == 0:
            continue
        if min_mice is not None and v.size < min_mice:
            continue
        mu = float(np.sum(w * v) / np.sum(w))
        if v.size > 1:
            wsd = float(np.sqrt(np.sum(w * (v - mu) ** 2) / np.sum(w)))
            sem = wsd / np.sqrt(v.size)
        else:
            sem = 0.0
        xs.append(int(n)); ys.append(mu); sems.append(sem)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order], np.array(sems)[order]


def _aggregate_chance_normalised_rel(df, model, value_col='fit_loss',
                                      rel_grid=REL_GRID):
    """Same as :func:`_aggregate_chance_normalised` but on the
    relative-N grid (n_neurons / n_full)."""
    key = ['mouse', 'selection', 'direction', 'n_neurons', 'draw']
    real = (df[df['model'] == model][key + [value_col, 'n_test_trials']]
            .rename(columns={value_col: 'real'}))
    shuf = (df[df['model'] == model + '_shf'][key + [value_col]]
            .rename(columns={value_col: 'shuf'}))
    paired = pd.merge(real, shuf, on=key)
    paired = paired[paired['selection'].isin(['random', 'full'])]
    full_n = _full_n_per_mouse(df)
    if paired.empty or not full_n:
        return rel_grid, np.full_like(rel_grid, np.nan, dtype=float), \
               np.full_like(rel_grid, np.nan, dtype=float)
    paired['v'] = paired['real'] / paired['shuf']
    Y, W = [], []
    for mid in sorted(paired['mouse'].unique()):
        if mid not in full_n:
            continue
        pm = (paired[paired['mouse'] == mid]
              .groupby('n_neurons').agg(
                  m=('v', 'mean'), tr=('n_test_trials', 'first'))
              .reset_index().sort_values('n_neurons'))
        if pm.empty:
            continue
        rx = pm['n_neurons'].to_numpy(dtype=float) / float(full_n[mid])
        ry = pm['m'].to_numpy(dtype=float)
        order = np.argsort(rx)
        rx, ry = rx[order], ry[order]
        yi = np.interp(rel_grid, rx, ry, left=np.nan, right=np.nan)
        Y.append(yi)
        W.append(float(pm['tr'].iloc[0]))
    if not Y:
        return rel_grid, np.full_like(rel_grid, np.nan, dtype=float), \
               np.full_like(rel_grid, np.nan, dtype=float)
    Y = np.stack(Y); W = np.asarray(W, dtype=float)[:, None]
    mask = np.isfinite(Y).astype(float)
    sw = (W * mask).sum(axis=0)
    mean = np.where(sw > 0, np.nansum(W * Y, axis=0) / np.where(sw > 0, sw, 1.0), np.nan)
    diff = Y - mean[None, :]
    var = np.where(sw > 0, np.nansum((W * mask) * (diff ** 2), axis=0)
                            / np.where(sw > 0, sw, 1.0), np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    n_present = mask.sum(axis=0)
    sem = np.where(n_present > 0, sd / np.sqrt(n_present), np.nan)
    return rel_grid, mean, sem


def plot_chance_normalised_compare(df, target, split, out_path,
                                    relative=False):
    """Each decoder's loss divided by its own shuffle baseline, plotted
    together. This normalises out the spat/temp shuffle asymmetry
    (architectural Jensen effect — see audit/SHUFFLE_ASYMMETRY.md) so
    the comparison reflects only signal extraction.

    Lower values = a larger improvement over the decoder's own chance
    level. If the two curves are similar, the apparent spat<temp gap on
    the raw ratio plot is dominated by the chance asymmetry. If temp is
    still meaningfully below spat after this normalisation, the gap is
    a genuine signal-extraction difference.
    """
    n_mice = len(df['mouse'].unique())
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0))
    xlabel = ('Fraction of full population (n / n_full)'
              if relative else X_LABEL)
    for model in ('spat', 'temp'):
        style = DECODER_STYLE[model]
        if relative:
            xs, ys, sem = _aggregate_chance_normalised_rel(df, model)
        else:
            xs, ys, sem = _aggregate_chance_normalised(df, model,
                                                       min_mice=n_mice)
        ok = np.isfinite(ys) if xs.size else np.array([], dtype=bool)
        if ok.any():
            ax.fill_between(xs[ok], (ys - sem)[ok], (ys + sem)[ok],
                            color=style['color'], alpha=0.20, zorder=2)
            ax.plot(xs[ok], ys[ok], '-', marker=style['marker'],
                    color=style['color'], lw=2.4, ms=6, zorder=4,
                    label=f"{style['label']}  (loss / shuffle)")
    ax.axhline(1.0, color='0.55', ls='--', lw=1.2, zorder=1,
               label='Chance (loss = shuffle)')
    ax.set_title(
        f'Chance-normalised comparison — '
        f'{TARGET_LABELS.get(target, target)}  |  {split}\n'
        f'(n={n_mice} mice; each decoder ÷ its own shuffle)',
        fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('fit_loss / fit_loss_shuffle\n(lower = larger lift over chance)',
                  fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Aggregate detail collapsed to spat/temp ratio
# ----------------------------------------------------------------------

def plot_aggregate_detail_ratio(df, target, split, out_path):
    """One panel per ranking, each showing the spat/temp fit-loss ratio
    for four conditions: shuffle, random, targeted-high, targeted-low.

    Same per-mouse pairing as the simple ratio plot (spat / temp on the
    matching (mouse, selection, direction, n_neurons, draw) row), then
    trial-count-weighted across mice with a SEM band. The shuffle curve
    uses the spat_shf / temp_shf pair. Random/shuffle are clipped at the
    largest N the targeted curves reach in the panel."""
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)

    fig, axes = plt.subplots(1, len(RANKINGS), figsize=(15, 4.8),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    # Random and shuffle ratios are ranking-independent — compute once.
    # min_mice=n_mice keeps only the full-coverage cross-mouse points
    # (matches the absolute-detail plot's policy and avoids the noisy
    # large-N tail where only one mouse contributes).
    r_x, r_y, r_s = _aggregate_paired(
        df, op='ratio', selection='random', direction='na',
        model_a='spat', model_b='temp', min_mice=n_mice)
    s_x, s_y, s_s = _aggregate_paired(
        df, op='ratio', selection='random', direction='na',
        model_a='spat_shf', model_b='temp_shf', min_mice=n_mice)

    for c, (rank_key, rank_label) in enumerate(RANKINGS):
        ax = axes[c]
        colour = RANK_COLOR[rank_key]

        tx, ty, ts = _aggregate_paired(
            df, op='ratio', selection=rank_key, direction='top',
            include_full=False, min_mice=n_mice)
        bx, by, bs = _aggregate_paired(
            df, op='ratio', selection=rank_key, direction='bottom',
            include_full=False, min_mice=n_mice)

        # Clip random/shuffle to the max N the targeted curves cover.
        xmax_candidates = [int(np.max(arr)) for arr in (tx, bx) if arr.size]
        xmax = max(xmax_candidates) if xmax_candidates else (
            int(np.max(r_x)) if r_x.size else None)

        def _clip(xs, ys, sems, xmax):
            if xs.size == 0 or xmax is None:
                return xs, ys, sems
            m = xs <= xmax
            return xs[m], ys[m], sems[m]

        rx, ry, rsem = _clip(r_x, r_y, r_s, xmax)
        sx, sy, ssem = _clip(s_x, s_y, s_s, xmax)

        ax.axhline(1.0, color='0.55', lw=1.2, ls='--', zorder=1,
                   label='spat = temp' if c == 0 else None)
        if sx.size:
            ax.fill_between(sx, sy - ssem, sy + ssem,
                            color='0.7', alpha=0.18, zorder=1)
            ax.plot(sx, sy, ':', color='0.45', lw=1.5, zorder=2,
                    label='Shuffle' if c == 0 else None)
        if rx.size:
            ax.fill_between(rx, ry - rsem, ry + rsem,
                            color='0.5', alpha=0.22, zorder=2)
            ax.plot(rx, ry, '-o', color='0.30', lw=1.8, ms=5, zorder=3,
                    label='Random' if c == 0 else None)
        if tx.size:
            ax.fill_between(tx, ty - ts, ty + ts,
                            color=colour, alpha=0.15, zorder=3)
            ax.plot(tx, ty, '-s', color=colour, lw=2.0, ms=6, zorder=4,
                    label='Targeted: top-ranked first' if c == 0 else None)
        if bx.size:
            ax.fill_between(bx, by - bs, by + bs,
                            color=colour, alpha=0.10, zorder=3)
            ax.plot(bx, by, '--^', color=colour, lw=2.0, ms=6, zorder=4,
                    label='Targeted: bottom-ranked first' if c == 0 else None)

        ax.set_title(rank_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        if c == 0:
            ax.set_ylabel('fit_loss(spat) / fit_loss(temp)\n'
                          '(>1 = temporal wins)', fontsize=10)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f'Spatial / temporal ratio across rankings (n={n_mice} mice)  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Spatial vs temporal: overlay, difference, ratio
# ----------------------------------------------------------------------

DECODER_STYLE = {
    'spat': dict(color='#1f77b4', label='Spatial (PPC)', marker='o'),
    'temp': dict(color='#d62728', label='Temporal (SBC)', marker='s'),
}


def plot_spat_vs_temp_overlay(df, agg, target, split, out_path,
                              relative=False):
    """Spatial and temporal random curves on the same axes (+ shuffle).

    ``relative=True`` plots against n_neurons / n_full (interpolated per
    mouse onto a shared relative grid); otherwise plots against absolute
    n_neurons (the cross-mouse aggregate's native grid). Each decoder's
    curve carries a cross-mouse SEM band — for the relative variant the
    band is computed from per-mouse interpolation; for the absolute
    variant it uses the aggregate's recorded SD-across-mice.
    """
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    xlabel = ('Fraction of full population (n / n_full)'
              if relative else X_LABEL)
    for model in ('spat', 'temp'):
        style = DECODER_STYLE[model]
        if relative:
            xs, ys, sem = _aggregate_on_rel_grid(df, 'random', 'na', model)
        else:
            xs, ys, sem = _agg_curve_sem(
                agg, 'random', 'na', model, full_only_n_mice=n_mice)
        if xs.size and np.any(np.isfinite(ys)):
            ax.fill_between(xs, ys - sem, ys + sem,
                            color=style['color'], alpha=0.18, zorder=2)
            ax.plot(xs, ys, '-', marker=style['marker'],
                    color=style['color'], lw=2.4, ms=6, zorder=4,
                    label=f"{style['label']} (mean ± SEM)")
        # shuffle on the same axes, dotted
        if relative:
            sxs, sys, ssem = _aggregate_on_rel_grid(
                df, 'random', 'na', model + '_shf')
        else:
            sxs, sys, ssem = _agg_curve_sem(
                agg, 'random', 'na', model + '_shf',
                full_only_n_mice=n_mice)
        if sxs.size and np.any(np.isfinite(sys)):
            ax.plot(sxs, sys, ':', color=style['color'], lw=1.5,
                    alpha=0.7, zorder=3,
                    label=f"{style['label']} shuffle")

    ax.set_title(
        f'Spatial vs temporal — '
        f'{TARGET_LABELS.get(target, target)}  |  {split}\n'
        f'({basis}-weighted across n={n_mice} mice)',
        fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(Y_LABEL, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_spat_minus_temp(df, target, split, out_path, relative=False):
    """Per-mouse spat - temp, aggregated across mice with SEM band.

    Positive values mean spatial loses to temporal; negative means
    spatial wins. The zero line is drawn for reference. Pairing happens
    at the (mouse, n_neurons, draw) level so the comparison is
    within-fit. ``relative=True`` uses the n / n_full grid.
    """
    n_mice = len(df['mouse'].unique())
    if relative:
        xs, ys, sem = _aggregate_paired_rel(df, op='diff')
        xlabel = 'Fraction of full population (n / n_full)'
    else:
        xs, ys, sem = _aggregate_paired(df, op='diff')
        xlabel = X_LABEL

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
    ax.axhline(0.0, color='0.55', lw=1.2, ls='--', zorder=1,
               label='spat = temp')
    if xs.size and np.any(np.isfinite(ys)):
        ax.fill_between(xs, ys - sem, ys + sem,
                        color='#6a3d9a', alpha=0.22, zorder=2)
        ax.plot(xs, ys, '-o', color='#6a3d9a', lw=2.2, ms=6, zorder=4,
                label='Mean ± SEM (across mice)')
    ax.set_title(
        f'Spatial − temporal fit loss — '
        f'{TARGET_LABELS.get(target, target)}  |  {split}\n'
        f'(n={n_mice} mice; positive = temporal wins)',
        fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('fit_loss(spat) − fit_loss(temp)', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_spat_over_temp(df, target, split, out_path, relative=False):
    """Per-mouse spat / temp, aggregated across mice with SEM band.

    >1 = temporal wins, <1 = spatial wins. The unity line is drawn for
    reference. Pairing happens at the (mouse, n_neurons, draw) level.
    """
    n_mice = len(df['mouse'].unique())
    if relative:
        xs, ys, sem = _aggregate_paired_rel(df, op='ratio')
        xlabel = 'Fraction of full population (n / n_full)'
    else:
        xs, ys, sem = _aggregate_paired(df, op='ratio')
        xlabel = X_LABEL

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
    ax.axhline(1.0, color='0.55', lw=1.2, ls='--', zorder=1,
               label='spat = temp')
    if xs.size and np.any(np.isfinite(ys)):
        ax.fill_between(xs, ys - sem, ys + sem,
                        color='#33a02c', alpha=0.22, zorder=2)
        ax.plot(xs, ys, '-o', color='#33a02c', lw=2.2, ms=6, zorder=4,
                label='Mean ± SEM (across mice)')
    ax.set_title(
        f'Spatial / temporal fit loss — '
        f'{TARGET_LABELS.get(target, target)}  |  {split}\n'
        f'(n={n_mice} mice;  >1 = temporal wins)',
        fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('fit_loss(spat) / fit_loss(temp)', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Relative-N variants of headline / normalised
# ----------------------------------------------------------------------

def plot_aggregate_headline_relN(df, target, split, out_path):
    """Headline scaling against n_neurons / n_full.

    Per-mouse curves are plotted on the mouse's own relative-N axis
    (each curve ends at 1.0). The cross-mouse mean + SEM band is
    computed by interpolating each mouse onto a shared relative grid
    and trial-count-weighting across mice.
    """
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    full_n = _full_n_per_mouse(df)
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11.5, 4.8), sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        for i, mid in enumerate(mice):
            if mid not in full_n:
                continue
            xs, ys, _ = _mouse_curve_xy(df, mid, 'random', 'na', model)
            if xs.size:
                ax.plot(xs / full_n[mid], ys, '-o', color=cmap(i % 10),
                        lw=1.3, ms=4, alpha=0.65, label=f'Mouse {mid}')
        gx, gy, gsem = _aggregate_on_rel_grid(df, 'random', 'na', model)
        ok = np.isfinite(gy)
        if ok.any():
            ax.fill_between(gx[ok], (gy - gsem)[ok], (gy + gsem)[ok],
                            color='k', alpha=0.18, zorder=4,
                            label='Cross-mouse SEM')
            ax.plot(gx[ok], gy[ok], '-o', color='k', lw=2.8, ms=8,
                    zorder=5, label='Cross-mouse mean (trial_count-weighted)')
        sx, sy_, ssem = _aggregate_on_rel_grid(
            df, 'random', 'na', model + '_shf')
        okk = np.isfinite(sy_)
        if okk.any():
            ax.fill_between(sx[okk], (sy_ - ssem)[okk], (sy_ + ssem)[okk],
                            color='0.6', alpha=0.18, zorder=2)
            ax.plot(sx[okk], sy_[okk], ':', color='0.4', lw=1.8,
                    label='Shuffle (mean)')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Fraction of full population (n / n_full)',
                      fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(Y_LABEL, fontsize=10)
    _share_ylim(axes)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(
        f'Scaling vs relative population size (n={n_mice} mice)  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_aggregate_normalised_relN(df, target, split, out_path):
    """Same as :func:`plot_aggregate_normalised` but on the relative-N
    axis — every curve ends at (1.0, 1.0) so the *shape* of scaling is
    directly comparable across mice on both axes."""
    ndf = normalise_to_full(df)
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    full_n = _full_n_per_mouse(df)
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11.5, 4.8), sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        for i, mid in enumerate(mice):
            if mid not in full_n:
                continue
            xs, ys, _ = _mouse_curve_xy(
                ndf, mid, 'random', 'na', model,
                value_col='fit_loss_rel_full')
            if xs.size:
                ax.plot(xs / full_n[mid], ys, '-o', color=cmap(i % 10),
                        lw=1.3, ms=4, alpha=0.65, label=f'Mouse {mid}')
        gx, gy, gsem = _aggregate_on_rel_grid(
            ndf, 'random', 'na', model, value_col='fit_loss_rel_full')
        ok = np.isfinite(gy)
        if ok.any():
            ax.fill_between(gx[ok], (gy - gsem)[ok], (gy + gsem)[ok],
                            color='k', alpha=0.18, zorder=4,
                            label='Cross-mouse SEM')
            ax.plot(gx[ok], gy[ok], '-o', color='k', lw=2.8, ms=8,
                    zorder=5, label='Cross-mouse mean (trial_count-weighted)')
        ax.axhline(1.0, color='0.55', ls='--', lw=1.3, zorder=1,
                   label='Full population (= 1.0)')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Fraction of full population (n / n_full)',
                      fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Fit loss relative to full population\n'
                       '(1.0 = full;  higher = worse)', fontsize=10)
    _share_ylim(axes)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(
        f'Normalised scaling vs relative population size (n={n_mice})  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def discover_mice(results_dir, target, split, wanted=None):
    """Mouse ids that have a per-mouse CSV on disk."""
    pattern = os.path.join(results_dir, f'mouse_*_{target}_{split}.csv')
    found = {}
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        mid = int(base.split('_')[1])
        if wanted is None or mid in wanted:
            found[mid] = path
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default='Q')
    parser.add_argument('--split', default='stratified_balanced')
    parser.add_argument('--mice', type=int, nargs='+', default=None,
                        help='Mouse ids to plot (default: all found).')
    parser.add_argument('--results-dir', default=None,
                        help='Default: nn_decoder/results/neuron_scaling/.')
    parser.add_argument('--out-dir', default=None,
                        help='Where to write the PNGs. '
                             'Default: nn_decoder/figures/neuron_scaling/.')
    args = parser.parse_args(argv)

    results_dir = (args.results_dir if args.results_dir
                   else str(RESULTS / 'neuron_scaling'))
    mice = discover_mice(results_dir, args.target, args.split,
                         wanted=set(args.mice) if args.mice else None)
    if not mice:
        raise SystemExit(
            f"No per-mouse CSVs found in {results_dir} for "
            f"target={args.target} split={args.split}.")

    out_dir = args.out_dir if args.out_dir else str(figures_dir('neuron_scaling'))
    os.makedirs(out_dir, exist_ok=True)

    written = []
    frames = []
    for mid, path in mice.items():
        df = pd.read_csv(path)
        frames.append(df)
        stem = f'mouse_{mid}_{args.target}_{args.split}'
        written.append(plot_detail(
            df, mid, args.target, args.split,
            os.path.join(out_dir, f'{stem}_detail.png')))
        written.append(plot_headline(
            df, mid, args.target, args.split,
            os.path.join(out_dir, f'{stem}_headline.png')))
        print(f"[mouse {mid}] plotted from {os.path.basename(path)}")

    # --- Cross-mouse aggregate (needs >= 2 mice) ---
    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        agg = aggregate_across_mice(combined)
        agg_csv = os.path.join(results_dir,
                               f'aggregate_{args.target}_{args.split}.csv')
        try:
            agg.to_csv(agg_csv, index=False)
            print(f"[aggregate] {len(agg)} rows "
                  f"({agg['weight_basis'].iloc[0]}-weighted) -> {agg_csv}")
        except OSError as exc:
            print(f"[aggregate] could not write CSV ({exc}); figures still produced")
        stem = f'aggregate_{args.target}_{args.split}'
        written.append(plot_aggregate_headline(
            combined, agg, args.target, args.split,
            os.path.join(out_dir, f'{stem}_headline.png')))
        written.append(plot_aggregate_detail(
            combined, agg, args.target, args.split,
            os.path.join(out_dir, f'{stem}_detail.png')))
        written.append(plot_aggregate_detail_ratio(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_detail_ratio.png')))
        written.append(plot_aggregate_normalised(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_normalised.png')))

        # Spatial vs temporal comparisons (absolute N).
        written.append(plot_spat_vs_temp_overlay(
            combined, agg, args.target, args.split,
            os.path.join(out_dir, f'{stem}_compare.png'),
            relative=False))
        written.append(plot_spat_minus_temp(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_diff.png'),
            relative=False))
        written.append(plot_spat_over_temp(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_ratio.png'),
            relative=False))
        written.append(plot_chance_normalised_compare(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_chance_normalised.png'),
            relative=False))

        # Relative-N (n / n_full) variants.
        written.append(plot_aggregate_headline_relN(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_headline_relN.png')))
        written.append(plot_aggregate_normalised_relN(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_normalised_relN.png')))
        written.append(plot_spat_vs_temp_overlay(
            combined, agg, args.target, args.split,
            os.path.join(out_dir, f'{stem}_compare_relN.png'),
            relative=True))
        written.append(plot_spat_minus_temp(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_diff_relN.png'),
            relative=True))
        written.append(plot_spat_over_temp(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_ratio_relN.png'),
            relative=True))
        written.append(plot_chance_normalised_compare(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_chance_normalised_relN.png'),
            relative=True))
    else:
        print("[aggregate] only one mouse found — skipping cross-mouse figures")

    print(f"\n{len(written)} figure(s) written to {out_dir}")
    for p in written:
        print(f"  {p}")
    return written


if __name__ == '__main__':
    main()
