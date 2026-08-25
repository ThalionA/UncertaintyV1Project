# -*- coding: utf-8 -*-
"""Spatial-vs-temporal decoder divergence: where do the two schemes
disagree most, and what do the posteriors look like there?

For one decoder target (default: ``Q_PCA_half_100ms_all`` — perceptual
posterior) and one split (default: ``stratified_balanced``), this script:

  1. computes per-trial PCA fit-loss for the spatial (``spat`` / spatial)
     and temporal (``temp`` / sampling) decoders, using the project's
     canonical ``pca_distance`` (basis from ``Dist['pcs']`` and
     ``Dist['explained_var']`` — see ``pca_loss.py``);
  2. ranks each mouse's trials by ``delta = loss_spat - loss_temp``
     so the *bottom* decile picks the trials where spat wins by the
     largest margin, the *top* decile picks where temp wins by the
     largest margin;
  3. averages the decoded posteriors and the target posterior within
     each decile, optionally first rolling each row so the *target's
     MAP bin* (argmax of the IO posterior) sits at the centre of the
     axis. The target's argmax is used as the alignment reference
     rather than the stimulus orientation because (a) it is intrinsic
     to the posterior space all three curves share and (b) it puts
     the target curve at 0 by construction, so spat/temp deviations
     from the IO are immediately readable. The IO posterior is often
     bimodal / cardinal-biased and its peak does *not* coincide with
     the stimulus orientation in general; see ``trial_cats`` in
     ``run_experiment.py`` for why ``trial_cats['orientation']`` is
     also not a usable index here (it's the joint Ori-Con-Disp
     condition ID, not the orientation bin);
  4. saves two figures per (target, split): a per-mouse 2 x n_mice
     grid and an across-mouse aggregate with trial-count weighting
     (per the project's hierarchical-aggregation convention). Each
     figure is produced twice — once aligned, once on the raw 0-90 deg
     axis — so location-dependent biases (cardinal anchoring, edge
     effects) and shape-only divergence are both visible.

The pipeline is target-agnostic for any PCA-loss cell (Q, L); non-PCA
cells (``d_MSE_*``, ``stim_cat_CE_*``) are skipped with a one-line
notice because the "average posterior" panels don't carry the same
meaning in those metric spaces.

Spyder::

    from plot_spat_temp_divergence import run_target
    run_target(slug='Q_PCA_half_100ms_all')

CLI::

    python plot_spat_temp_divergence.py \
        --run-name production_full_targets_alltrials_v1 \
        --slug Q_PCA_half_100ms_all \
        --quantile 0.10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
import decoder_plotting_utils as dpu  # noqa: E402
from figsave import save_fig  # noqa: E402


DEFAULT_SPLITS = ('stratified_balanced', 'generalize_contrast', 'generalize_dispersion')


# ---------------------------------------------------------------------
# Per-trial delta loss
# ---------------------------------------------------------------------

def compute_per_trial_delta(dist: dict) -> dict:
    """Per-trial spat-minus-temp PCA loss and the posteriors needed to
    plot it.

    Parameters
    ----------
    dist : dict
        The ``Dist`` sub-dict for one mouse (after ``simplify_cells=True``
        loadmat). Must contain ``'spat'``, ``'temp'``, ``'pcs'``, and
        ``'explained_var'``. Both ``'spat'`` and ``'temp'`` must have
        ``'decoded'`` and ``'target'``.

    Returns
    -------
    dict
        Keys ``delta``, ``loss_spat``, ``loss_temp`` (each shape
        ``(n_trials,)``), and ``decoded_spat``, ``decoded_temp``,
        ``target`` (each shape ``(n_trials, n_cats)``). ``delta`` is
        positive when temp wins (temp loss smaller); negative when spat
        wins. The legends and prints in this module use that sign
        convention consistently.

    Raises
    ------
    AssertionError
        If ``Dist['spat']['target']`` and ``Dist['temp']['target']``
        disagree. They are identical by construction in the production
        .mat tree (both archs are trained on the same per-trial IO
        posterior); the assertion guards against schema drift if the
        run script ever changes.
    """
    spat = dist['spat']
    temp = dist['temp']
    target_spat = np.asarray(spat['target'])
    target_temp = np.asarray(temp['target'])
    assert np.array_equal(target_spat, target_temp), \
        "spat and temp targets must be identical per trial"
    pcs = dist.get('pcs', None)
    evar = dist.get('explained_var', None)

    decoded_spat = np.asarray(spat['decoded'])
    decoded_temp = np.asarray(temp['decoded'])
    loss_spat = dpu.calc_pca_dist(decoded_spat, target_spat, pcs, evar)
    loss_temp = dpu.calc_pca_dist(decoded_temp, target_temp, pcs, evar)
    delta = loss_spat - loss_temp

    return {
        'delta': delta,
        'loss_spat': loss_spat,
        'loss_temp': loss_temp,
        'decoded_spat': decoded_spat,
        'decoded_temp': decoded_temp,
        'target': target_spat,
    }


# ---------------------------------------------------------------------
# Quantile selection
# ---------------------------------------------------------------------

def select_quantiles(delta: np.ndarray, q: float = 0.10):
    """Boolean masks for the top and bottom ``q``-fraction of trials by
    ``delta``.

    ``delta = loss_spat - loss_temp``, so bottom = spat best vs temp,
    top = temp best vs spat. NaN deltas are excluded from both masks.

    Returns
    -------
    (top, bot) : tuple of np.ndarray
        Each shape ``(n_trials,)``, dtype bool.
    """
    delta = np.asarray(delta, dtype=float)
    n = delta.size
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    valid = ~np.isnan(delta)
    n_pick = max(1, int(np.floor(q * valid.sum())))

    order = np.argsort(np.where(valid, delta, np.inf))  # NaNs sink to the end
    bot_idx = order[:n_pick]
    # Top: pick from the valid tail; argsort ascending so take the last n_pick valid.
    valid_sorted_count = int(valid.sum())
    top_idx = order[max(0, valid_sorted_count - n_pick):valid_sorted_count]

    top = np.zeros(n, dtype=bool)
    bot = np.zeros(n, dtype=bool)
    top[top_idx] = True
    bot[bot_idx] = True
    return top, bot


# ---------------------------------------------------------------------
# Alignment + per-quantile averaging
# ---------------------------------------------------------------------

def align_to_truth(posteriors: np.ndarray, true_cat: np.ndarray, n_cats: int) -> np.ndarray:
    """Roll each row of ``posteriors`` so the index in ``true_cat`` lands
    at ``n_cats // 2``. Output has the same shape as input.

    In the production pipeline the alignment reference is
    ``argmax(target, axis=1)`` — the IO posterior's MAP bin. The
    function itself is reference-agnostic; pass whichever per-trial
    index you want at the centre."""
    posteriors = np.asarray(posteriors)
    true_cat = np.asarray(true_cat, dtype=int)
    centre = n_cats // 2
    shifts = centre - true_cat  # roll each row by this many bins
    # Build column indices for fancy indexing once; np.roll per row is
    # ~10x slower for the trial counts we see (300-500).
    cols = np.arange(n_cats)
    rolled_cols = (cols[None, :] - shifts[:, None]) % n_cats
    return posteriors[np.arange(posteriors.shape[0])[:, None], rolled_cols]


def mean_in_quantile(decoded_spat, decoded_temp, target, idx,
                     align: bool, true_cat):
    """Mean decoded-spat / decoded-temp / target posterior across the
    trials selected by ``idx``. Returns three ``(n_cats,)`` arrays.

    If ``idx`` selects no trials the function returns NaN curves —
    plot_aggregate skips those when pooling.
    """
    idx = np.asarray(idx, dtype=bool)
    n_cats = np.asarray(target).shape[1]
    if not idx.any():
        nan = np.full(n_cats, np.nan)
        return nan.copy(), nan.copy(), nan.copy()

    d_spat = np.asarray(decoded_spat)[idx]
    d_temp = np.asarray(decoded_temp)[idx]
    tgt = np.asarray(target)[idx]

    if align:
        c = np.asarray(true_cat)[idx]
        d_spat = align_to_truth(d_spat, c, n_cats=n_cats)
        d_temp = align_to_truth(d_temp, c, n_cats=n_cats)
        tgt = align_to_truth(tgt, c, n_cats=n_cats)

    return d_spat.mean(axis=0), d_temp.mean(axis=0), tgt.mean(axis=0)


# ---------------------------------------------------------------------
# Aggregation across mice
# ---------------------------------------------------------------------

def weighted_pool(curves, weights):
    """Trial-count-weighted average of per-mouse mean curves.

    Curves that are entirely NaN (mouse had an empty quantile) are
    dropped. ``weights`` is parallel to ``curves`` and gives the number
    of trials in that mouse's quantile (per the project's
    hierarchical-aggregation convention — weight by trial count, not by
    mouse).
    """
    curves = [np.asarray(c, dtype=float) for c in curves]
    weights = np.asarray(weights, dtype=float)
    keep = np.array([not np.all(np.isnan(c)) for c in curves])
    if not keep.any():
        return np.full(curves[0].shape, np.nan)
    stacked = np.stack([c for c, k in zip(curves, keep) if k], axis=0)
    w = weights[keep]
    w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
    return (stacked * w[:, None]).sum(axis=0)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

# Three-curve palette: target = neutral grey, spat = warm (orange),
# temp = cool (blue) — matches the colour assignment used in other
# clean-run plots so this figure reads consistently with them.
_C_TARGET = '#444444'
_C_SPAT = '#d95f02'
_C_TEMP = '#1f77b4'


def _xaxis(n_cats: int, align: bool):
    """X-axis values and label for one panel."""
    if align:
        x = np.arange(n_cats) - n_cats // 2
        xlabel = 'orientation offset from target MAP (bins)'
    else:
        x = np.arange(n_cats)
        xlabel = 'orientation bin (0-90 deg)'
    return x, xlabel


def _plot_overlay(ax, x, m_spat, m_temp, m_tgt, *, title=None, show_legend=False):
    ax.plot(x, m_tgt, color=_C_TARGET, lw=1.4, label='target (IO)')
    ax.plot(x, m_spat, color=_C_SPAT, lw=1.4, label='spat decoded')
    ax.plot(x, m_temp, color=_C_TEMP, lw=1.4, label='temp decoded')
    if title:
        ax.set_title(title, fontsize=9)
    if show_legend:
        ax.legend(fontsize=7, frameon=False, loc='upper right')


def plot_per_mouse_grid(per_mouse: list, *, align: bool, n_cats: int,
                         target_label: str, split_label: str, q: float,
                         out_path: Path):
    """2 x n_mice grid. Row 0 = bottom decile (spat wins), row 1 = top
    decile (temp wins). Each panel overlays target / spat_decoded /
    temp_decoded for that mouse, that decile."""
    n_mice = len(per_mouse)
    fig, axes = plt.subplots(2, n_mice, figsize=(2.6 * n_mice, 4.8),
                              sharex=True, sharey=True, squeeze=False)
    x, xlabel = _xaxis(n_cats, align)
    for j, m in enumerate(per_mouse):
        # Row 0: bottom decile (spat << temp loss).
        _plot_overlay(axes[0, j], x, m['bot_mean_spat'], m['bot_mean_temp'], m['bot_mean_tgt'],
                       title=f"{m['mouse_id']}  bot {int(q*100)}% (n={m['n_bot']})  "
                             f"Δ̄={m['bot_mean_delta']:+.2f}",
                       show_legend=(j == 0))
        # Row 1: top decile (spat >> temp loss).
        _plot_overlay(axes[1, j], x, m['top_mean_spat'], m['top_mean_temp'], m['top_mean_tgt'],
                       title=f"{m['mouse_id']}  top {int(q*100)}% (n={m['n_top']})  "
                             f"Δ̄={m['top_mean_delta']:+.2f}")
    # One xlabel under the middle column so per-panel labels don't
    # overlap when n_mice >= 4. The shared x-axis already pins each
    # panel to the same range, so a single label reads correctly.
    mid = n_mice // 2
    axes[-1, mid].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('mean P(orientation)\nbot decile (spat better)')
    axes[1, 0].set_ylabel('mean P(orientation)\ntop decile (temp better)')
    align_tag = 'aligned to target MAP' if align else 'raw orientation axis'
    fig.suptitle(f'spat vs temp divergence by trial — {target_label}, '
                  f'{split_label} — {align_tag}',
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_path.parent, out_path.stem, layout=None)


def plot_target_only_per_mouse(per_mouse: list, *, align: bool, n_cats: int,
                                target_label: str, split_label: str, q: float,
                                out_path: Path):
    """1 x n_mice grid. Each panel overlays the mean IO target posterior
    for the *bot* decile (spat-better trials) and the *top* decile
    (temp-better trials) for that mouse. Diagnostic: if the two curves
    differ for a given mouse, then the trials where spat wins and the
    trials where temp wins come from systematically different parts of
    the IO posterior space — useful context for interpreting the main
    divergence figure.
    """
    n_mice = len(per_mouse)
    fig, axes = plt.subplots(1, n_mice, figsize=(2.6 * n_mice, 2.6),
                              sharex=True, sharey=True, squeeze=False)
    x, xlabel = _xaxis(n_cats, align)
    # Distinct colours from the main figure's spat/temp scheme to avoid
    # confusing the reader: this figure shows targets only.
    c_bot, c_top = '#2ca02c', '#9467bd'  # green / purple

    for j, m in enumerate(per_mouse):
        ax = axes[0, j]
        ax.plot(x, m['bot_mean_tgt'], color=c_bot, lw=1.4,
                label=f"bot {int(q*100)}% (spat better, n={m['n_bot']})")
        ax.plot(x, m['top_mean_tgt'], color=c_top, lw=1.4,
                label=f"top {int(q*100)}% (temp better, n={m['n_top']})")
        ax.set_title(f"{m['mouse_id']}", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, frameon=False, loc='upper right')

    mid = n_mice // 2
    axes[0, mid].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('mean P(orientation)\n— IO target only —')
    align_tag = 'aligned to target MAP' if align else 'raw orientation axis'
    fig.suptitle(f'IO target posterior in bot vs top decile (by Δloss=spat-temp) — '
                  f'{target_label}, {split_label} — {align_tag}',
                  fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, out_path.parent, out_path.stem, layout=None)


def plot_aggregate(per_mouse: list, *, align: bool, n_cats: int,
                    target_label: str, split_label: str, q: float,
                    out_path: Path):
    """1 row x 2 columns: bottom decile, top decile. Thin per-mouse
    lines + bold trial-count-weighted aggregate for each of the three
    curves (target / spat / temp)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0),
                              sharex=True, sharey=True, squeeze=False)
    x, xlabel = _xaxis(n_cats, align)

    for ax, decile in zip(axes[0], ['bot', 'top']):
        weights = [m[f'n_{decile}'] for m in per_mouse]
        # Per-mouse thin lines (faded), one set per curve type.
        for m in per_mouse:
            ax.plot(x, m[f'{decile}_mean_tgt'], color=_C_TARGET, lw=0.6, alpha=0.35)
            ax.plot(x, m[f'{decile}_mean_spat'], color=_C_SPAT, lw=0.6, alpha=0.35)
            ax.plot(x, m[f'{decile}_mean_temp'], color=_C_TEMP, lw=0.6, alpha=0.35)
        # Bold trial-count weighted aggregate.
        agg_tgt = weighted_pool([m[f'{decile}_mean_tgt'] for m in per_mouse], weights)
        agg_spat = weighted_pool([m[f'{decile}_mean_spat'] for m in per_mouse], weights)
        agg_temp = weighted_pool([m[f'{decile}_mean_temp'] for m in per_mouse], weights)
        ax.plot(x, agg_tgt, color=_C_TARGET, lw=2.0, label='target (IO)')
        ax.plot(x, agg_spat, color=_C_SPAT, lw=2.0, label='spat decoded')
        ax.plot(x, agg_temp, color=_C_TEMP, lw=2.0, label='temp decoded')

        decile_label = 'bot' if decile == 'bot' else 'top'
        which_wins = 'spat better' if decile == 'bot' else 'temp better'
        n_tot = int(np.nansum(weights))
        ax.set_title(f'{decile_label} {int(q*100)}% by Δloss — {which_wins} '
                      f'(n_trials={n_tot})', fontsize=10)
        ax.set_xlabel(xlabel)

    axes[0, 0].set_ylabel('mean P(orientation)')
    axes[0, 0].legend(fontsize=8, frameon=False, loc='upper right')
    align_tag = 'aligned to target MAP' if align else 'raw orientation axis'
    fig.suptitle(f'spat vs temp divergence — aggregate (trial-count weighted) — '
                  f'{target_label}, {split_label} — {align_tag}',
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, out_path.parent, out_path.stem, layout=None)


# ---------------------------------------------------------------------
# Per-mouse summary record
# ---------------------------------------------------------------------

def _process_mouse(mouse_id: str, m_data: dict, *, q: float, n_cats: int):
    """Compute everything plot_* needs for one mouse: per-quantile mean
    curves on both axis modes, plus diagnostic stats.

    Alignment reference: ``argmax(target, axis=1)`` — the IO posterior's
    MAP bin per trial. ``trial_cats['orientation']`` is *not* usable for
    this (it's the joint Ori-Con-Disp condition ID, set by
    ``run_experiment.py``), and the stimulus orientation in degrees
    does not coincide with the IO peak either because the IO posterior
    is often cardinal-biased.
    """
    dist = m_data['Dist']
    base = compute_per_trial_delta(dist)
    top, bot = select_quantiles(base['delta'], q=q)
    true_cat = np.argmax(base['target'], axis=1)

    out = {'mouse_id': mouse_id,
           'n_trials': int(base['delta'].size),
           'n_top': int(top.sum()),
           'n_bot': int(bot.sum()),
           'median_delta': float(np.nanmedian(base['delta'])),
           'top_mean_delta': float(np.nanmean(base['delta'][top])) if top.any() else np.nan,
           'bot_mean_delta': float(np.nanmean(base['delta'][bot])) if bot.any() else np.nan,
           'top_mean_loss_spat': float(np.nanmean(base['loss_spat'][top])) if top.any() else np.nan,
           'top_mean_loss_temp': float(np.nanmean(base['loss_temp'][top])) if top.any() else np.nan,
           'bot_mean_loss_spat': float(np.nanmean(base['loss_spat'][bot])) if bot.any() else np.nan,
           'bot_mean_loss_temp': float(np.nanmean(base['loss_temp'][bot])) if bot.any() else np.nan,
           }
    for align_flag, tag in [(False, 'raw'), (True, 'aligned')]:
        for idx, dec in [(top, 'top'), (bot, 'bot')]:
            m_spat, m_temp, m_tgt = mean_in_quantile(
                base['decoded_spat'], base['decoded_temp'], base['target'],
                idx, align=align_flag, true_cat=true_cat)
            suffix = '' if align_flag else '_raw'
            # The plot functions read from <decile>_mean_<curve>[_raw].
            out[f'{dec}_mean_spat{suffix}'] = m_spat
            out[f'{dec}_mean_temp{suffix}'] = m_temp
            out[f'{dec}_mean_tgt{suffix}'] = m_tgt
    return out


def _print_summary(per_mouse: Iterable[dict], target_label: str, split_label: str):
    """Sanity-check table printed before any SVG is written."""
    print(f"\n[{target_label} / {split_label}] per-mouse Δloss = loss_spat - loss_temp")
    print(f"{'mouse':<10} {'n':>4} {'median Δ':>10} {'bot Δ̄':>10} "
          f"{'bot Lspat':>10} {'bot Ltemp':>10} {'top Δ̄':>10} "
          f"{'top Lspat':>10} {'top Ltemp':>10}")
    for m in per_mouse:
        print(f"{m['mouse_id']:<10} {m['n_trials']:>4d} {m['median_delta']:>+10.3f} "
              f"{m['bot_mean_delta']:>+10.3f} {m['bot_mean_loss_spat']:>10.3f} "
              f"{m['bot_mean_loss_temp']:>10.3f} {m['top_mean_delta']:>+10.3f} "
              f"{m['top_mean_loss_spat']:>10.3f} {m['top_mean_loss_temp']:>10.3f}")
    # Sign-convention check: bottom decile should have loss_spat < loss_temp,
    # top decile the reverse, for every mouse.
    bad_bot = [m['mouse_id'] for m in per_mouse
                if m['bot_mean_loss_spat'] >= m['bot_mean_loss_temp']]
    bad_top = [m['mouse_id'] for m in per_mouse
                if m['top_mean_loss_spat'] <= m['top_mean_loss_temp']]
    if bad_bot or bad_top:
        print(f"  WARNING: bot-decile sign flip for {bad_bot}; "
              f"top-decile sign flip for {bad_top}")
    else:
        print("  sign convention OK on all mice")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def _is_pca_slug(slug: str) -> bool:
    """Slug encodes (target, loss, time_window, bin_size_ms[, suffix]).
    The PCA-loss orientation cells are Q_PCA_* and L_PCA_*."""
    return '_PCA_' in slug


def run_target(slug: str = 'Q_PCA_half_100ms_all',
                run_name: str = 'production_full_targets_alltrials_v1',
                splits: Iterable[str] = ('stratified_balanced',),
                q: float = 0.10,
                out_root: str = 'figures/divergence',
                results_root: str | None = None):
    """Top-level driver: for each split in ``splits``, compute per-mouse
    deciles and write four SVGs (per-mouse grid + aggregate, each in
    aligned and raw variants)."""
    if not _is_pca_slug(slug):
        print(f"  [skip] {slug}: PCA-loss specific (divergence panels "
              f"assume 91-bin orientation posterior)")
        return
    if results_root is None:
        results_root = str(paths.RESULTS)
    results = dpu.load_run_tree(run_name, slug, splits=list(splits),
                                  results_root=results_root)
    if not results:
        print(f"  [skip] {slug}: no .mat files loaded")
        return

    out_root_path = Path(out_root) / slug
    target_label = slug

    for split, mat in results.items():
        n_cats = mat['results']['mouse_0']['Dist']['spat']['target'].shape[1]
        per_mouse = [_process_mouse(mid, m, q=q, n_cats=n_cats)
                     for mid, m in mat['results'].items()]
        _print_summary(per_mouse, target_label, split)

        # Strip the suffix-renamed entries so plot_* can read from the
        # canonical names regardless of which axis mode.
        for axis_mode, suffix in [('aligned', ''), ('raw', '_raw')]:
            view = [{**m,
                     'top_mean_spat': m[f'top_mean_spat{suffix}'],
                     'top_mean_temp': m[f'top_mean_temp{suffix}'],
                     'top_mean_tgt':  m[f'top_mean_tgt{suffix}'],
                     'bot_mean_spat': m[f'bot_mean_spat{suffix}'],
                     'bot_mean_temp': m[f'bot_mean_temp{suffix}'],
                     'bot_mean_tgt':  m[f'bot_mean_tgt{suffix}']}
                    for m in per_mouse]
            plot_per_mouse_grid(view,
                                 align=(axis_mode == 'aligned'),
                                 n_cats=n_cats,
                                 target_label=target_label,
                                 split_label=split, q=q,
                                 out_path=out_root_path / split /
                                          f'per_mouse_{axis_mode}.svg')
            plot_aggregate(view,
                            align=(axis_mode == 'aligned'),
                            n_cats=n_cats,
                            target_label=target_label,
                            split_label=split, q=q,
                            out_path=out_root_path / split /
                                     f'aggregate_{axis_mode}.svg')
            plot_target_only_per_mouse(view,
                            align=(axis_mode == 'aligned'),
                            n_cats=n_cats,
                            target_label=target_label,
                            split_label=split, q=q,
                            out_path=out_root_path / split /
                                     f'target_only_per_mouse_{axis_mode}.svg')
        print(f"  wrote 6 PNG+SVG pairs -> {out_root_path / split}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default='production_full_targets_alltrials_v1')
    parser.add_argument('--slug', default='Q_PCA_half_100ms_all')
    parser.add_argument('--splits', nargs='+', default=['stratified_balanced'],
                         help='One or more split names to process.')
    parser.add_argument('--quantile', type=float, default=0.10,
                         help='Tail fraction for top/bot deciles (default 0.10).')
    parser.add_argument('--out-root', default='figures/divergence')
    parser.add_argument('--results-root', default=None,
                         help='Root of the results tree. Defaults to '
                              'nn_decoder/results (via paths.RESULTS).')
    args = parser.parse_args()
    run_target(slug=args.slug,
                run_name=args.run_name,
                splits=args.splits,
                q=args.quantile,
                out_root=args.out_root,
                results_root=args.results_root)
