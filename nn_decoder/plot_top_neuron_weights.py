# -*- coding: utf-8 -*-
"""Find and visualise the highest-weighted neurons per decoder.

For each (target, decoder, mouse) the decoder's first-layer weight
matrix ``W_in`` (shape ``(hidden, n_neurons)``) is loaded from the
saved ``Weights`` field of the fit's ``stratified_balanced.mat``
file. Neurons are ranked by column L2 norm (the decoder's own notion
of importance — same scoring as ``neuron_scaling.rank_by_weight_magnitude``)
and the top fraction (default 10%) is kept.

The actual *activity* of those top neurons is then summarised across
three stimulus dimensions:

  * ``orientation`` — ``abs_from_go`` in [0, 90]; 0 = exactly Go,
    90 = exactly NoGo.
  * ``contrast``    — raw contrast value.
  * ``dispersion``  — raw dispersion value.

Per (target, decoder, mouse) the heatmap and tuning-curve figures
share a single row order (sorted by each neuron's preferred
orientation) across all three condition panels so a row index is a
fixed neuron throughout the figure. x-axis labels show the real
stimulus values that were actually presented (categorical ticks, not
interpolation).

Outputs per (target, decoder, mouse):

  * ``<decoder>/mouse_<id>_heatmap.svg`` — three panels (vs
    orientation / contrast / dispersion), rows = top-neurons,
    cols = real stimulus values, colour = mean response.
  * ``<decoder>/mouse_<id>_curves.svg``  — overlaid per-neuron tuning
    lines + population mean ± SEM, same three panels.

Plus per (target, decoder) an aggregate set that pools top neurons
across mice (each neuron z-scored across its own trials before
pooling, so scales are comparable):

  * ``<decoder>/aggregate_heatmap.svg``
  * ``<decoder>/aggregate_curves.svg``

Cross-decoder comparisons per (target, mouse) and aggregate, in a
sibling ``compare/`` subdirectory:

  * ``mouse_<id>_weight_compare.svg`` — scatter of per-neuron W_in
    column-L2 norms (spat vs temp), four colour groups (neither /
    spat-only / temp-only / both). Jaccard of the two top-sets is
    annotated in the title.
  * ``mouse_<id>_tuning_compare.svg`` — overlaid mean tuning curves
    (top-spat in orange vs top-temp in steelblue) per condition,
    plus an MI histogram on the orientation axis.
  * ``mouse_<id>_category_tuning.svg`` — mean tuning curves per
    neuron category (both / spat-only / temp-only). Tells you
    whether neurons shared by both decoders look different from
    neurons only one decoder picks up.
  * ``aggregate_tuning_compare.svg`` — z-scored pooled aggregate
    of the per-mouse tuning-compare figure. MI panel pools per-mouse
    raw MIs (self-normalising on raw data).
  * ``aggregate_category_tuning.svg`` — z-scored pooled aggregate
    of the per-mouse category figure.

Plus one cross-target figure at the figure-tree root:

  * ``cross_target_jaccard.svg`` — single panel with one bar per
    target (Jaccard(top-spat, top-temp) averaged across mice) and
    per-mouse dots overlaid. Lets you compare decoder agreement
    (spat vs temp) across the different decoded targets.

Output tree::

    figures/top_neuron_weights/
      cross_target_jaccard.svg
      <slug>/
        spat/   mouse_<id>_{heatmap,curves}.svg, aggregate_{heatmap,curves}.svg
        temp/   mouse_<id>_{heatmap,curves}.svg, aggregate_{heatmap,curves}.svg
        compare/
          mouse_<id>_weight_compare.svg
          mouse_<id>_tuning_compare.svg
          mouse_<id>_category_tuning.svg
          aggregate_tuning_compare.svg
          aggregate_category_tuning.svg

CLI::

    python plot_top_neuron_weights.py --run-name production_full_targets_alltrials_v1

Spyder::

    from plot_top_neuron_weights import main
    main()
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
import decoder_plotting_utils as dpu  # noqa: E402
from neuron_scaling import rank_by_weight_magnitude  # noqa: E402


DEFAULT_TARGET_SLUGS = {
    'Q': 'Q_PCA_half_100ms_all',
    'L': 'L_PCA_half_100ms_all',
    'd': 'd_MSE_half_100ms',
}
DECODERS = ('spat', 'temp')
DECODER_LABELS = {'spat': 'Spatial', 'temp': 'Temporal'}
CANONICAL_SPLIT = 'stratified_balanced'
TOP_FRACTION = 0.10
CONDITION_NAMES = ('orientation', 'contrast', 'dispersion')


# ----------------------------------------------------------------------
# Loaders + ranking
# ----------------------------------------------------------------------

def _load_weights(run_name: str, slug: str,
                   results_root: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Return ``{mouse_id: {decoder: W_in}}`` from a slug's
    ``stratified_balanced.mat``. Missing entries are skipped with a
    warning."""
    tree = dpu.load_run_tree(run_name, slug, splits=[CANONICAL_SPLIT],
                              results_root=results_root)
    if CANONICAL_SPLIT not in tree:
        print(f"[warn] {slug}: no {CANONICAL_SPLIT}.mat; skipping")
        return {}
    results = tree[CANONICAL_SPLIT]['results']
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for mouse_key, m_data in results.items():
        if not mouse_key.startswith('mouse_'):
            continue
        mid = int(mouse_key.split('_', 1)[1])
        W = m_data.get('Weights')
        if not isinstance(W, dict):
            print(f"[warn] {slug}/{mouse_key}: no Weights field; skipping")
            continue
        weights_by_dec = {}
        for dec in DECODERS:
            if dec not in W:
                continue
            w_in = np.asarray(W[dec].get('W_in'), dtype=float)
            if w_in.ndim != 2:
                continue
            weights_by_dec[dec] = w_in
        if weights_by_dec:
            out[mid] = weights_by_dec
    return out


def select_top_neurons(W_in: np.ndarray, fraction: float = TOP_FRACTION
                        ) -> np.ndarray:
    """Indices of the top ``fraction`` of neurons by column L2 norm.

    The fraction is rounded up (``ceil``) to a minimum of 1 neuron,
    so even tiny populations return at least one index.
    """
    n_neurons = W_in.shape[1]
    k = max(1, int(np.ceil(n_neurons * fraction)))
    ranking, _scores = rank_by_weight_magnitude(W_in)
    return np.asarray(ranking[:k], dtype=int)


# ----------------------------------------------------------------------
# Activity + tuning
# ----------------------------------------------------------------------

def _load_mouse_activity(mouse_id: int) -> Tuple[np.ndarray, dict]:
    """Per-trial mean response per neuron, plus the trials dict.

    Activity is mean over the T-bin axis so each neuron-by-trial cell
    is a scalar rate — the tuning helpers in neuron_scaling.py use the
    same reduction (``np.nanmean(act, axis=2)``).
    """
    from utils import load_vr_export
    activities_m, _, _, _, trials = load_vr_export(mouse_id)
    # activities_m: (n_neurons, n_trials, n_T)
    per_trial = np.nanmean(activities_m, axis=2)   # (n_neurons, n_trials)
    return per_trial, trials


def _condition_bins(values: np.ndarray, condition: str,
                     n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a continuous condition into discrete levels.

    Returns ``(bin_centres, trial_bin_idx)``. If the raw values already
    have <= n_bins unique levels (typical for contrast / dispersion
    grids), the unique levels are used directly. Otherwise the values
    are digitised into ``n_bins`` equal-width bins.
    """
    finite = np.isfinite(values)
    uniq = np.unique(values[finite])
    if uniq.size <= n_bins:
        centres = uniq
        idx = np.full(values.shape, -1, dtype=int)
        for j, u in enumerate(uniq):
            idx[values == u] = j
        return centres, idx
    edges = np.linspace(uniq.min(), uniq.max(), n_bins + 1)
    idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    idx[~finite] = -1
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, idx


def compute_tuning(per_trial_response: np.ndarray,
                    trials: dict,
                    condition: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-neuron mean response per bin of one condition.

    Parameters
    ----------
    per_trial_response : ndarray, shape (n_neurons, n_trials)
    trials : dict with 'orientation', 'contrast', 'dispersion' keys
    condition : one of 'orientation', 'contrast', 'dispersion'

    Returns
    -------
    centres : ndarray (n_bins,)
        Bin centres (unique levels when possible).
    tuning : ndarray (n_neurons, n_bins)
        Mean response per (neuron, bin); NaN if a bin had no trials.
    """
    vals = np.asarray(trials[condition], dtype=float).reshape(-1)
    centres, idx = _condition_bins(vals, condition)
    n_neurons = per_trial_response.shape[0]
    tuning = np.full((n_neurons, centres.size), np.nan)
    for j in range(centres.size):
        mask = (idx == j)
        if mask.any():
            tuning[:, j] = np.nanmean(per_trial_response[:, mask], axis=1)
    return centres, tuning


def _sort_by_preferred(tuning: np.ndarray) -> np.ndarray:
    """Row order by argmax of each neuron's tuning curve (NaN-safe).
    Stable; ties broken by original index."""
    with np.errstate(invalid='ignore'):
        order_key = np.nan_to_num(tuning, nan=-np.inf).argmax(axis=1)
    return np.argsort(order_key, kind='stable')


def _modulation_index(tuning: np.ndarray) -> np.ndarray:
    """Per-neuron (r_max - r_min) / (r_max + r_min) — bounded in
    [0, 1] for non-negative rates; values near 1 = strongly tuned.
    Same definition used by ``neuron_scaling.rank_by_orientation_tuning``.

    Only valid for non-negative tunings; for z-scored or otherwise
    signed inputs use :func:`_tuning_range` instead.
    """
    r_max = np.nanmax(tuning, axis=1)
    r_min = np.nanmin(tuning, axis=1)
    denom = r_max + r_min
    mi = np.where(np.abs(denom) > 1e-12, (r_max - r_min) / denom, 0.0)
    return np.nan_to_num(mi, nan=0.0)


def _tuning_range(tuning: np.ndarray) -> np.ndarray:
    """Per-neuron ``r_max - r_min`` across the bin axis.

    Works for both raw rates and z-scored values; unlike the
    modulation index it does not assume non-negative input, so it
    stays interpretable when the comparison plot pools z-scored
    neurons across mice. Units match the units of ``tuning`` itself
    (Hz for raw, sigma for z-scored).
    """
    r_max = np.nanmax(tuning, axis=1)
    r_min = np.nanmin(tuning, axis=1)
    rng = r_max - r_min
    return np.nan_to_num(rng, nan=0.0)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _style():
    sns.set_context('talk', font_scale=0.75)
    sns.set_style('ticks')


def _heatmap_panel(ax, centres, tuning, condition_label, order):
    """One condition-vs-neuron heatmap panel.

    ``order`` is an explicit row order (indices into ``tuning``'s
    rows) so every panel in a figure shares the same neuron ordering.
    The x-axis uses categorical ticks at the actual stimulus values
    in ``centres`` — not interpolated.
    """
    tuning = tuning[order]
    im = ax.imshow(tuning, aspect='auto', origin='lower', cmap='viridis',
                    interpolation='nearest')
    ax.set_xticks(np.arange(len(centres)))
    # Format labels: integers stay integers; floats get short form.
    def _fmt(v):
        v = float(v)
        if abs(v - round(v)) < 1e-9:
            return f'{int(round(v))}'
        return f'{v:.3g}'
    ax.set_xticklabels([_fmt(c) for c in centres], rotation=0, fontsize=9)
    ax.set_xlabel(condition_label)
    ax.set_title(f'vs {condition_label}')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def plot_heatmaps(tunings: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    title: str, out_path: Path,
                    sort_by: str = 'orientation'):
    """Three-panel heatmap figure: orientation, contrast, dispersion.

    ``sort_by`` picks which condition's tuning curves define the row
    order; that same order is then applied to all three panels so a
    neuron's row index is consistent across the figure (item 1 in
    the design notes).
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Derive a single shared row order from the chosen condition.
    if sort_by in tunings:
        order = _sort_by_preferred(tunings[sort_by][1])
    else:
        # Fall back to the first available condition.
        first_cond = next(iter(tunings))
        order = _sort_by_preferred(tunings[first_cond][1])
    n_neurons = len(order)

    for ax, cond in zip(axes, CONDITION_NAMES):
        if cond not in tunings:
            ax.set_visible(False)
            continue
        centres, tuning = tunings[cond]
        _heatmap_panel(ax, centres, tuning, cond, order=order)
    # Only the leftmost panel labels the neuron axis (rows are shared).
    axes[0].set_ylabel(f'Neuron (sorted by preferred {sort_by}, '
                        f'n={n_neurons}; rows shared across panels)')
    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_tuning_curves(tunings: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        title: str, out_path: Path):
    """Three-panel overlaid tuning-curve figure: one faint line per
    neuron + the population-mean curve in bold. x-ticks are placed at
    the actual stimulus values."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cond in zip(axes, CONDITION_NAMES):
        if cond not in tunings:
            ax.set_visible(False)
            continue
        centres, tuning = tunings[cond]
        x = np.arange(len(centres))
        # Per-neuron lines.
        for row in tuning:
            ax.plot(x, row, color='steelblue', alpha=0.25, lw=1.0)
        # Population mean + SEM.
        mean = np.nanmean(tuning, axis=0)
        n_eff = np.sum(~np.isnan(tuning), axis=0)
        sem = np.nanstd(tuning, axis=0, ddof=1) / np.sqrt(np.maximum(n_eff, 1))
        ax.errorbar(x, mean, yerr=sem, color='black', lw=2.0,
                     marker='o', capsize=3, label=f'Mean (n={tuning.shape[0]})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{float(c):g}' for c in centres],
                            rotation=0, fontsize=9)
        ax.set_xlabel(cond)
        ax.set_ylabel('Mean response')
        ax.set_title(f'vs {cond}')
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Cross-decoder comparisons
# ----------------------------------------------------------------------

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index of two index sets. NaN on a fully empty union."""
    sa, sb = set(map(int, a)), set(map(int, b))
    union = sa | sb
    if not union:
        return float('nan')
    return len(sa & sb) / len(union)


def _rm_anova_one_way(matrix: np.ndarray) -> Tuple[float, float, int, int]:
    """One-way repeated-measures ANOVA computed from scipy primitives.

    ``matrix`` is ``(n_subjects, k_conditions)`` with no NaNs (drop
    incomplete subjects beforehand). Returns ``(F, p, df_between,
    df_error)`` where ``df_between = k-1`` and ``df_error =
    (n-1)*(k-1)``. NaN F/p returned when df_error < 1.
    """
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float('nan'), float('nan'), max(k - 1, 0), 0
    grand = x.mean()
    subj_means = x.mean(axis=1, keepdims=True)
    cond_means = x.mean(axis=0, keepdims=True)
    ss_subj = k * ((subj_means - grand) ** 2).sum()
    ss_cond = n * ((cond_means - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_subj - ss_cond
    df_between = k - 1
    df_err = (n - 1) * (k - 1)
    if df_err < 1 or ss_err <= 0:
        return float('nan'), float('nan'), df_between, df_err
    F = (ss_cond / df_between) / (ss_err / df_err)
    p = float(scipy_stats.f.sf(F, df_between, df_err))
    return float(F), p, df_between, df_err


def plot_weight_comparison(spat_scores: np.ndarray, temp_scores: np.ndarray,
                            spat_top: np.ndarray, temp_top: np.ndarray,
                            title: str, out_path: Path):
    """Scatter of per-neuron column-L2 norms (spat vs temp), with each
    decoder's top-fraction highlighted. Jaccard of the two top-sets is
    annotated in the title (no longer a separate bar panel — that view
    is captured at higher resolution by the cross-target Jaccard figure).
    """
    _style()
    fig, ax = plt.subplots(figsize=(7, 6))

    all_idx = np.arange(spat_scores.size)
    in_spat = np.isin(all_idx, spat_top)
    in_temp = np.isin(all_idx, temp_top)
    neither = ~(in_spat | in_temp)
    both = in_spat & in_temp
    spat_only = in_spat & ~in_temp
    temp_only = in_temp & ~in_spat

    ax.scatter(spat_scores[neither], temp_scores[neither],
                s=12, color='lightgrey', alpha=0.6, label='neither')
    ax.scatter(spat_scores[spat_only], temp_scores[spat_only],
                s=22, color='darkorange', alpha=0.85, label='top-spat only')
    ax.scatter(spat_scores[temp_only], temp_scores[temp_only],
                s=22, color='steelblue', alpha=0.85, label='top-temp only')
    ax.scatter(spat_scores[both], temp_scores[both],
                s=36, color='crimson', alpha=0.95,
                edgecolor='black', linewidth=0.5, label='both')

    lim = float(np.nanmax(np.concatenate([spat_scores, temp_scores])))
    ax.plot([0, lim], [0, lim], '--', color='black', alpha=0.4, lw=1)
    ax.set_xlabel('Spatial decoder weight L2 norm')
    ax.set_ylabel('Temporal decoder weight L2 norm')
    jacc = _jaccard(spat_top, temp_top)
    ax.set_title(f'Per-neuron W_in column-L2 norm  |  Jaccard(top-spat, top-temp) = {jacc:.3f}')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)

    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_category_tuning(category_tunings: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                          title: str, out_path: Path):
    """Mean tuning curves per neuron-category, three condition panels.

    ``category_tunings`` is ``{category: {condition: (centres, tuning)}}``
    with categories drawn from {'both', 'spat-only', 'temp-only'}.
    Each panel overlays the mean ± SEM curve for whichever categories
    are populated. Answers "do neurons picked up by both decoders
    look different from neurons picked up by only one?".
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cat_color = {'both': 'crimson',
                  'spat-only': 'darkorange',
                  'temp-only': 'steelblue'}
    for ax, cond in zip(axes, CONDITION_NAMES):
        any_plotted = False
        for cat, tunings_by_cond in category_tunings.items():
            if cond not in tunings_by_cond:
                continue
            centres, tuning = tunings_by_cond[cond]
            if tuning.size == 0:
                continue
            x = np.arange(len(centres))
            mean = np.nanmean(tuning, axis=0)
            n_eff = np.sum(~np.isnan(tuning), axis=0)
            sem = np.nanstd(tuning, axis=0, ddof=1) / np.sqrt(
                np.maximum(n_eff, 1))
            ax.errorbar(x, mean, yerr=sem, color=cat_color.get(cat, 'black'),
                         lw=2.0, marker='o', capsize=3,
                         label=f'{cat} (n={tuning.shape[0]})')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{float(c):g}' for c in centres],
                                rotation=0, fontsize=9)
            any_plotted = True
        ax.set_xlabel(cond)
        ax.set_ylabel('Mean response')
        ax.set_title(f'vs {cond}')
        if any_plotted:
            ax.legend(fontsize=8)
    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_cross_target_jaccard(top_idx_by_target_mouse: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                                title: str, out_path: Path,
                                *, alpha: float = 0.05):
    """Per-target spat-vs-temp Jaccard, distribution across mice, with
    repeated-measures statistical comparison.

    For each target, computes ``Jaccard(top-spat, top-temp)`` per
    mouse and displays the distribution as a bar (mean ± SEM) with
    per-mouse dots overlaid. Mice that appear in all targets are
    connected with faint grey lines so the reader can track within-
    mouse changes. A one-way repeated-measures ANOVA across targets
    (subjects = mice present in every target) is annotated in the
    title; pairwise posthoc paired-t comparisons with Bonferroni
    correction are drawn between bar pairs when significant at
    ``alpha`` (default 0.05).

    Parameters
    ----------
    top_idx_by_target_mouse : nested dict
        ``{target: {mouse_id: {decoder: top_idx_array}}}``.
    alpha : float
        Significance threshold for posthoc annotation lines (uncorrected
        cutoff; the displayed p is Bonferroni-corrected).
    """
    _style()
    targets = list(top_idx_by_target_mouse.keys())
    if not targets:
        return

    # Per-(target, mouse) Jaccard table — keyed for line-drawing.
    per_target_per_mouse: Dict[str, Dict[int, float]] = {}
    for target in targets:
        per_mouse = top_idx_by_target_mouse[target]
        d = {}
        for mid, decoder_dict in per_mouse.items():
            if 'spat' not in decoder_dict or 'temp' not in decoder_dict:
                continue
            d[int(mid)] = _jaccard(decoder_dict['spat'], decoder_dict['temp'])
        per_target_per_mouse[target] = d

    # Mice present in *every* target — needed for repeated-measures
    # ANOVA and for the connecting lines.
    common_mice = set.intersection(*[set(d.keys())
                                       for d in per_target_per_mouse.values()]) \
        if per_target_per_mouse else set()
    common_mice = sorted(common_mice)

    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(targets) + 2), 5.5))

    # Bars: mean ± SEM over all available mice per target.
    means, sems, ns = [], [], []
    for t in targets:
        v = np.asarray(list(per_target_per_mouse[t].values()), dtype=float)
        v = v[np.isfinite(v)]
        means.append(float(np.mean(v)) if v.size else np.nan)
        sems.append(float(np.std(v, ddof=1) / np.sqrt(v.size))
                     if v.size > 1 else 0.0)
        ns.append(int(v.size))
    ax.bar(x, means, yerr=sems, capsize=4, color='crimson',
            edgecolor='black', alpha=0.75, width=0.55, zorder=2)

    # Faint connecting lines for each mouse present in all targets.
    for mid in common_mice:
        y = [per_target_per_mouse[t].get(mid, np.nan) for t in targets]
        ax.plot(x, y, color='grey', alpha=0.45, lw=1.0, zorder=3)

    # Per-mouse dots, jittered slightly so coincident values don't stack.
    rng = np.random.RandomState(0)
    for i, t in enumerate(targets):
        vals = list(per_target_per_mouse[t].values())
        if not vals:
            continue
        jit = rng.uniform(-0.05, 0.05, len(vals))
        ax.scatter(x[i] + jit, vals, s=40, color='black',
                    alpha=0.8, zorder=5, edgecolor='white', linewidth=0.5)

    # --- Stats: RM-ANOVA + paired-t posthoc with Bonferroni ---
    stats_title = ''
    if len(common_mice) >= 2 and len(targets) >= 2:
        matrix = np.array([
            [per_target_per_mouse[t][mid] for t in targets]
            for mid in common_mice
        ], dtype=float)
        F, p_anova, df1, df2 = _rm_anova_one_way(matrix)
        stats_title = (f'RM-ANOVA F({df1},{df2}) = {F:.2f}, '
                        f'p = {p_anova:.3g}  (n_mice = {len(common_mice)})')

        # Posthoc paired-t between every target pair, Bonferroni-corrected.
        n_pairs = len(targets) * (len(targets) - 1) // 2
        bonf = max(n_pairs, 1)
        y_top = float(np.nanmax(np.concatenate(
            [list(per_target_per_mouse[t].values()) for t in targets])))
        step = 0.06
        bracket_y = max(y_top + step, 0.7)
        for ai in range(len(targets)):
            for bi in range(ai + 1, len(targets)):
                ta, tb = targets[ai], targets[bi]
                a = np.array([per_target_per_mouse[ta][m]
                               for m in common_mice], dtype=float)
                b = np.array([per_target_per_mouse[tb][m]
                               for m in common_mice], dtype=float)
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 2:
                    continue
                try:
                    _, p_raw = scipy_stats.ttest_rel(a[mask], b[mask])
                except Exception:
                    continue
                p_corr = min(p_raw * bonf, 1.0)
                if p_corr >= alpha:
                    continue
                if p_corr < 0.001: star = '***'
                elif p_corr < 0.01: star = '**'
                else:                star = '*'
                ax.plot([x[ai], x[ai], x[bi], x[bi]],
                         [bracket_y, bracket_y + 0.015,
                          bracket_y + 0.015, bracket_y],
                         lw=1.3, color='black')
                ax.text((x[ai] + x[bi]) / 2, bracket_y + 0.015,
                         f'{star}\n(p={p_corr:.3g})', ha='center', va='bottom',
                         fontsize=8)
                bracket_y += step

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={n})' for t, n in zip(targets, ns)])
    ax.set_ylabel('Jaccard(top-spat, top-temp)')
    ax.set_ylim(-0.02, 1.15)
    ax.axhline(0.0, color='grey', lw=0.8, alpha=0.6)

    n_mice_max = max(ns) if ns else 0
    head = f'Decoder top-set overlap by target  (n_mice up to {n_mice_max})'
    ax.set_title(head + ('\n' + stats_title if stats_title else ''),
                  fontsize=11)
    plt.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_tuning_comparison(spat_tunings: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            temp_tunings: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            title: str, out_path: Path,
                            *,
                            spat_mi: Optional[np.ndarray] = None,
                            temp_mi: Optional[np.ndarray] = None):
    """Overlay mean tuning curves (top-spat vs top-temp neurons), per
    condition, plus a fourth panel showing the per-neuron modulation
    index distribution for each top set across the orientation
    condition (i.e. "how strongly tuned are these neurons?").
    """
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, cond in zip(axes[:3], CONDITION_NAMES):
        if cond not in spat_tunings and cond not in temp_tunings:
            ax.set_visible(False)
            continue
        for label, tunings_set, color in (
                ('top-spat', spat_tunings, 'darkorange'),
                ('top-temp', temp_tunings, 'steelblue')):
            if cond not in tunings_set:
                continue
            centres, tuning = tunings_set[cond]
            x = np.arange(len(centres))
            mean = np.nanmean(tuning, axis=0)
            n_eff = np.sum(~np.isnan(tuning), axis=0)
            sem = np.nanstd(tuning, axis=0, ddof=1) / np.sqrt(
                np.maximum(n_eff, 1))
            ax.errorbar(x, mean, yerr=sem, color=color, lw=2.0,
                         marker='o', capsize=3,
                         label=f'{label} (n={tuning.shape[0]})')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{float(c):g}' for c in centres],
                                rotation=0, fontsize=9)
        ax.set_xlabel(cond)
        ax.set_ylabel('Mean response')
        ax.set_title(f'Mean tuning vs {cond}')
        ax.legend(fontsize=8)

    # Fourth panel: orientation modulation index histogram.
    # MI = (r_max - r_min) / (r_max + r_min) is self-normalising on
    # non-negative rates and bounded in [0, 1] — that's what we want.
    # The aggregate plot used to compute MI on z-scored tunings, which
    # exploded the metric and left the panel empty; instead the
    # caller now pre-computes per-mouse MI on the raw (non-z-scored)
    # tunings and passes the pooled values via spat_mi / temp_mi.
    ax = axes[3]
    if spat_mi is None and 'orientation' in spat_tunings:
        spat_mi = _modulation_index(spat_tunings['orientation'][1])
    if temp_mi is None and 'orientation' in temp_tunings:
        temp_mi = _modulation_index(temp_tunings['orientation'][1])
    spat_mi = np.asarray([]) if spat_mi is None else np.asarray(spat_mi)
    temp_mi = np.asarray([]) if temp_mi is None else np.asarray(temp_mi)
    bins = np.linspace(0, 1, 21)
    if spat_mi.size:
        ax.hist(spat_mi, bins=bins, color='darkorange', alpha=0.55,
                 edgecolor='black', label=f'top-spat (n={spat_mi.size})')
    if temp_mi.size:
        ax.hist(temp_mi, bins=bins, color='steelblue', alpha=0.55,
                 edgecolor='black', label=f'top-temp (n={temp_mi.size})')
    ax.set_xlabel('Orientation modulation index  (r_max − r_min)/(r_max + r_min)')
    ax.set_ylabel('# neurons')
    ax.set_title('Tuning strength distribution')
    ax.legend(fontsize=8)

    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _zscore_per_neuron(response: np.ndarray) -> np.ndarray:
    """Z-score each neuron across its own trials (axis=1)."""
    mu = np.nanmean(response, axis=1, keepdims=True)
    sd = np.nanstd(response, axis=1, ddof=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (response - mu) / sd


def _run_one_target(target: str, slug: str, run_name: str,
                     results_root: str, out_root: Path,
                     top_fraction: float) -> Dict[int, Dict[str, np.ndarray]]:
    """Build all per-target figures and return ``{mouse: {decoder: top_idx}}``
    so the driver can stitch a cross-target Jaccard figure afterwards."""
    print(f"\n== {target} ({slug}) ==")
    weights = _load_weights(run_name, slug, results_root)
    if not weights:
        print(f"  no weights loaded for {slug}; skipping")
        return {}

    slug_dir = out_root / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    # Per-decoder subdir for the per-mouse + aggregate heatmap/curve
    # figures; cross-decoder comparisons land at the slug root.
    dec_dirs = {dec: slug_dir / dec for dec in DECODERS}
    for d in dec_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    compare_dir = slug_dir / 'compare'
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Activity is mouse-level, decoder-independent — load once per mouse.
    activity_cache = {}
    # Cross-mouse pool of z-scored tunings (for aggregate mean curves)
    # and a separate pool of per-mouse raw MI arrays (for the aggregate
    # tuning-strength panel — MI is self-normalising and must run on
    # raw, not z-scored, tunings).
    pooled = {dec: {cond: [] for cond in CONDITION_NAMES}
              for dec in DECODERS}
    pooled_mi = {dec: [] for dec in DECODERS}
    pooled_centres: Dict[str, np.ndarray] = {}
    # Category pools, both z-scored (for mean curves) and raw (for
    # potential future MI extension on categories).
    cat_pooled = {cat: {cond: [] for cond in CONDITION_NAMES}
                   for cat in ('both', 'spat-only', 'temp-only')}
    # Per-mouse top-idx record returned to the driver.
    per_mouse_top: Dict[int, Dict[str, np.ndarray]] = {}

    for mid in sorted(weights.keys()):
        if mid not in activity_cache:
            activity_cache[mid] = _load_mouse_activity(mid)
        per_trial, trials = activity_cache[mid]

        # Per-mouse, per-decoder: ranking, top selection, tunings.
        per_decoder = {}   # decoder -> dict with keys
                           #   'scores' (full pop), 'top_idx', 'tunings'
        for decoder in DECODERS:
            if decoder not in weights[mid]:
                continue
            w_in = weights[mid][decoder]
            if per_trial.shape[0] != w_in.shape[1]:
                print(f"  [warn] mouse_{mid}/{decoder}: activity has "
                       f"{per_trial.shape[0]} neurons but W_in has "
                       f"{w_in.shape[1]}; skipping")
                continue
            _, scores = rank_by_weight_magnitude(w_in)
            top_idx = select_top_neurons(w_in, top_fraction)
            top_response = per_trial[top_idx]
            tunings = {}
            for cond in CONDITION_NAMES:
                centres, tuning = compute_tuning(top_response, trials, cond)
                tunings[cond] = (centres, tuning)
                # Stash z-scored version for the cross-mouse aggregate.
                zresp = _zscore_per_neuron(top_response)
                _, z_tun = compute_tuning(zresp, trials, cond)
                pooled[decoder][cond].append(z_tun)
                pooled_centres.setdefault(cond, centres)
            # Per-mouse MI on the raw orientation tuning (self-normalising;
            # pooled MI lets the aggregate strength panel run on raw data).
            if 'orientation' in tunings:
                pooled_mi[decoder].append(_modulation_index(tunings['orientation'][1]))
            per_decoder[decoder] = {
                'scores': scores, 'top_idx': top_idx, 'tunings': tunings,
                'w_in': w_in, 'per_trial': per_trial,
            }

            title_base = (
                f'{target} | {DECODER_LABELS[decoder]} | mouse_{mid} '
                f'| top {int(top_fraction * 100)}% '
                f'(n={top_idx.size} of {w_in.shape[1]})')
            plot_heatmaps(tunings,
                           title=title_base + '\nactivity heatmaps',
                           out_path=dec_dirs[decoder] / f'mouse_{mid}_heatmap.svg')
            plot_tuning_curves(tunings,
                                title=title_base + '\ntuning curves',
                                out_path=dec_dirs[decoder] / f'mouse_{mid}_curves.svg')

        # Record per-mouse top-idx (used by the driver for cross-target Jaccard).
        if per_decoder:
            per_mouse_top[mid] = {dec: per_decoder[dec]['top_idx']
                                    for dec in per_decoder}

        # Cross-decoder comparison plots — need both decoders present.
        if 'spat' in per_decoder and 'temp' in per_decoder:
            ps, pt = per_decoder['spat'], per_decoder['temp']
            title_w = (f'{target} | mouse_{mid} | top '
                        f'{int(top_fraction * 100)}% per decoder '
                        f'(n_neurons={ps["w_in"].shape[1]})')
            plot_weight_comparison(
                ps['scores'], pt['scores'], ps['top_idx'], pt['top_idx'],
                title=title_w + '\nweight magnitudes & top-set overlap',
                out_path=compare_dir / f'mouse_{mid}_weight_compare.svg')
            title_t = (f'{target} | mouse_{mid} | top '
                        f'{int(top_fraction * 100)}% per decoder '
                        f'(top-spat n={ps["top_idx"].size}, '
                        f'top-temp n={pt["top_idx"].size})')
            plot_tuning_comparison(
                ps['tunings'], pt['tunings'],
                title=title_t + '\ntop-spat vs top-temp tuning profiles',
                out_path=compare_dir / f'mouse_{mid}_tuning_compare.svg')

            # Category split — partition the union of the two top-sets.
            spat_set = set(map(int, ps['top_idx']))
            temp_set = set(map(int, pt['top_idx']))
            cat_idx = {
                'both':       np.array(sorted(spat_set & temp_set), dtype=int),
                'spat-only':  np.array(sorted(spat_set - temp_set), dtype=int),
                'temp-only':  np.array(sorted(temp_set - spat_set), dtype=int),
            }
            cat_tunings_raw = {}
            for cat, idx in cat_idx.items():
                if idx.size == 0:
                    continue
                cat_resp = per_trial[idx]
                t_by_cond = {}
                z_by_cond = {}
                zresp = _zscore_per_neuron(cat_resp)
                for cond in CONDITION_NAMES:
                    c_centres, c_tuning = compute_tuning(cat_resp, trials, cond)
                    t_by_cond[cond] = (c_centres, c_tuning)
                    _, c_z = compute_tuning(zresp, trials, cond)
                    z_by_cond[cond] = c_z
                    cat_pooled[cat][cond].append(c_z)
                cat_tunings_raw[cat] = t_by_cond

            if cat_tunings_raw:
                title_c = (f'{target} | mouse_{mid} | category split '
                            f'(both n={cat_idx["both"].size}, '
                            f'spat-only n={cat_idx["spat-only"].size}, '
                            f'temp-only n={cat_idx["temp-only"].size})')
                plot_category_tuning(
                    cat_tunings_raw,
                    title=title_c + '\nmean tuning per category',
                    out_path=compare_dir / f'mouse_{mid}_category_tuning.svg')

    # ---- Aggregate (pooled across mice) ----
    agg = {dec: {} for dec in DECODERS}
    for decoder in DECODERS:
        for cond, tunings_list in pooled[decoder].items():
            if not tunings_list:
                continue
            centres = pooled_centres[cond]
            stack = [t for t in tunings_list if t.shape[1] == centres.size]
            if not stack:
                continue
            agg[decoder][cond] = (centres, np.vstack(stack))

        if not agg[decoder]:
            continue
        n_pooled = next(iter(agg[decoder].values()))[1].shape[0]
        title_base = (
            f'{target} | {DECODER_LABELS[decoder]} | aggregate | '
            f'pooled top {int(top_fraction * 100)}% '
            f'(n={n_pooled} neurons across mice, z-scored)')
        plot_heatmaps(agg[decoder],
                       title=title_base + '\nactivity heatmaps',
                       out_path=dec_dirs[decoder] / 'aggregate_heatmap.svg')
        plot_tuning_curves(agg[decoder],
                            title=title_base + '\ntuning curves',
                            out_path=dec_dirs[decoder] / 'aggregate_curves.svg')

    # Aggregate cross-decoder tuning comparison. Mean curves are
    # computed from the z-scored aggregate (axis = mean over pooled
    # neurons), but the strength panel uses MI pooled from per-mouse
    # raw tunings — MI is self-normalising on non-negative rates and
    # only makes sense on raw data. The aggregate Jaccard summary
    # plot has been replaced by the cross-target Jaccard figure
    # produced by the driver.
    if agg['spat'] and agg['temp']:
        n_spat_pooled = next(iter(agg['spat'].values()))[1].shape[0]
        n_temp_pooled = next(iter(agg['temp'].values()))[1].shape[0]
        spat_mi_pool = (np.concatenate(pooled_mi['spat'])
                          if pooled_mi['spat'] else None)
        temp_mi_pool = (np.concatenate(pooled_mi['temp'])
                          if pooled_mi['temp'] else None)
        title_t = (f'{target} | aggregate | pooled top '
                    f'{int(top_fraction * 100)}% '
                    f'(top-spat n={n_spat_pooled}, '
                    f'top-temp n={n_temp_pooled})')
        plot_tuning_comparison(
            agg['spat'], agg['temp'],
            title=title_t + '\ntop-spat vs top-temp tuning profiles',
            out_path=compare_dir / 'aggregate_tuning_compare.svg',
            spat_mi=spat_mi_pool, temp_mi=temp_mi_pool)

    # Aggregate category-split tuning (z-scored, pooled across mice).
    agg_cat = {}
    for cat, tunings_by_cond in cat_pooled.items():
        cat_dict = {}
        for cond, tunings_list in tunings_by_cond.items():
            if not tunings_list:
                continue
            centres = pooled_centres.get(cond)
            if centres is None:
                continue
            stack = [t for t in tunings_list if t.shape[1] == centres.size]
            if not stack:
                continue
            cat_dict[cond] = (centres, np.vstack(stack))
        if cat_dict:
            agg_cat[cat] = cat_dict
    if agg_cat:
        n_by_cat = {cat: next(iter(td.values()))[1].shape[0]
                    for cat, td in agg_cat.items()}
        title_c = (f'{target} | aggregate | category split '
                    f'(z-scored, pooled across mice; '
                    + ', '.join(f"{c} n={n_by_cat.get(c, 0)}"
                                  for c in ('both', 'spat-only', 'temp-only'))
                    + ')')
        plot_category_tuning(
            agg_cat,
            title=title_c + '\nmean tuning per category',
            out_path=compare_dir / 'aggregate_category_tuning.svg')

    return per_mouse_top


def main(run_name: str = 'production_full_targets_alltrials_v1',
         results_root: Optional[str] = None,
         out_root: str = 'figures/top_neuron_weights',
         target_slugs: Optional[Dict[str, str]] = None,
         top_fraction: float = TOP_FRACTION):
    """Drive the top-neuron-weight visualisation end-to-end.

    Parameters
    ----------
    run_name : str
        Training RUN_NAME under the results tree.
    results_root : str, optional
        Defaults to ``paths.RESULTS`` (nn_decoder/results/), regardless
        of kernel cwd.
    out_root : str
        Root directory for the figure tree. One subdirectory per slug,
        then one per decoder.
    target_slugs : dict, optional
        Override the {target: slug} mapping. Default covers Q, L, d
        under the all_trials basis.
    top_fraction : float
        Fraction of neurons to keep per (mouse, decoder). Default 0.10.
    """
    if results_root is None:
        results_root = str(paths.RESULTS)
    if target_slugs is None:
        target_slugs = DEFAULT_TARGET_SLUGS

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Top-neuron-weight plots — run {run_name}")
    print(f"  results_root: {results_root}")
    print(f"  out_root    : {out.resolve()}")
    print(f"  top_fraction: {top_fraction}")

    top_by_target: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for target, slug in target_slugs.items():
        per_mouse_top = _run_one_target(
            target, slug, run_name, results_root, out, top_fraction)
        if per_mouse_top:
            top_by_target[target] = per_mouse_top

    if len(top_by_target) >= 2:
        title = (f'Cross-target top-set overlap | top '
                  f'{int(top_fraction * 100)}% per (mouse, decoder)')
        plot_cross_target_jaccard(
            top_by_target,
            title=title + '\nDo decoders pay attention to the same neurons across targets?',
            out_path=out / 'cross_target_jaccard.svg')
        print(f"  cross-target Jaccard -> {out / 'cross_target_jaccard.svg'}")

    print(f"\nDone. Figures under {out.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name',
                         default='production_full_targets_alltrials_v1')
    parser.add_argument('--results-root', default=None,
                         help='Default: nn_decoder/results (via paths.RESULTS).')
    parser.add_argument('--out-root', default='figures/top_neuron_weights')
    parser.add_argument('--top-fraction', type=float, default=TOP_FRACTION)
    args = parser.parse_args()
    main(run_name=args.run_name, results_root=args.results_root,
         out_root=args.out_root, top_fraction=args.top_fraction)
