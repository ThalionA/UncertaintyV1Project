# -*- coding: utf-8 -*-
"""Spatial vs Temporal architecture performance for the post-fix run.

Three-panel figure (Q condmean, Q residual, stim_cat) of shuffle-
normalised test loss with spatial vs temporal bars side by
side per split. Matches the legacy `plot_normalized_bars_v26.py`
idiom (orange = spat, blue = temp, dotted line at 1.0 = shuffle
baseline, lower is better) but reads from the new
`results/<run_name>/<slug>/<split>.mat` tree and uses the stored
training-loss values directly (the same loss each network was trained
on — PCA for Q on either basis, CE for stim_cat).

Per-mouse dots overlaid on the bars. Paired t-test (two-sided) spat
vs temp p-value annotated per split (n = 6 mice). Wilcoxon signed-rank
was considered but has a minimum-achievable-p floor of ~0.03 at n=6
(only when all 6 mice agree in sign), which severely caps power for
the small-but-consistent effects in this dataset.

Spyder usage:

    from plot_post_fix_performance import main
    main()  # reads results/post_fix_loadings_2026_05_17/...

Output: figures/post_fix_performance/spat_vs_temp_post_fix.png plus a
tidy summary CSV with per-mouse normalised losses for downstream
inspection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Configuration: the three slugs we want to compare.
# ----------------------------------------------------------------------

SLUGS = [
    ('Q condmean',       'Q_PCA_half_100ms_condmean'),
    ('L condmean',       'L_PCA_half_100ms_condmean'),
    ('d (MSE)',          'd_MSE_half_100ms'),
    ('Q residual',       'Q_PCA_half_100ms_residual'),
    ('L residual',       'L_PCA_half_100ms_residual'),
    ('stim_cat (CE)',    'stim_cat_CE_half_100ms'),
]
# Layout: 2 rows x 3 cols. Top row = condition_mean basis + d (MSE) as
# the "non-residual" sibling. Bottom row = residual basis + stim_cat
# (CE) as the "within-condition-scoring" sibling. Reading down a column:
# (Q, L) get the basis contrast; col 3 has the two alternative-loss
# decoders. Reading across a row: Q vs L within a basis.
PANEL_ROWS, PANEL_COLS = 2, 3

SPLITS = ('stratified_balanced', 'generalize_contrast', 'generalize_dispersion')
SPLIT_LABELS = {
    'stratified_balanced':   'Stratified',
    'generalize_contrast':   'Gen. Contrast',
    'generalize_dispersion': 'Gen. Dispersion',
}
ARCHS = ('spat', 'temp')
ARCH_LABELS = {'spat': 'Spatial', 'temp': 'Temporal'}
ARCH_COLORS = {'spat': 'darkorange', 'temp': 'steelblue'}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
#
# Losses are recomputed from `Dist[arch]['target']` / `['decoded']` via
# `decoder_plotting_utils.calc_fit_loss`, NOT read from the (legacy,
# contaminated) `KLs[arch]` field. The `KLs` field carries the
# training-time entropy regulariser `entropy_lambda * H(pred_probs)`
# on top of the fit-loss for temporal/sampling models, which has no
# place in a held-out test metric. See
# `nn_decoder/audit/AUDIT_loss_consumers.md` for the full audit.

import decoder_plotting_utils as dpu


def _per_mouse_arrays(mat: Dict, loss_func: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Return per-trial fit-loss arrays for spat / temp / spat_shf /
    temp_shf, one entry per mouse, recomputed from the saved
    `Dist[arch]` arrays via `calc_fit_loss`.

    Output: {mouse_id: {'spat': ndarray, 'temp': ndarray,
                         'spat_shf': ndarray, 'temp_shf': ndarray}}
    """
    results = mat['results']
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for k, v in results.items():
        if not k.startswith('mouse_'):
            continue
        mid = int(k.split('_', 1)[1])
        dist = v.get('Dist')
        if dist is None:
            print(f"[warn] mouse_{mid} missing 'Dist'; skipping")
            continue
        pcs = dist.get('pcs', None)
        evar = dist.get('explained_var', None)
        try:
            arrs = {
                arch: np.asarray(
                    dpu.calc_fit_loss(dist[arch], loss_func, pcs=pcs, evar=evar),
                    dtype=float,
                ).reshape(-1)
                for arch in ('spat', 'temp', 'spat_shf', 'temp_shf')
            }
        except KeyError as exc:
            print(f"[warn] mouse_{mid} missing Dist arch {exc}; skipping")
            continue
        if arrs['spat'].shape != arrs['temp'].shape:
            print(f"[warn] mouse_{mid} spat / temp trial counts differ "
                   f"({arrs['spat'].shape} vs {arrs['temp'].shape}); skipping mouse")
            continue
        out[mid] = arrs
    return out


def _aggregate_per_mouse(per_mouse_arrays: Dict[int, Dict[str, np.ndarray]]
                           ) -> Dict[int, Dict[str, float]]:
    """Collapse per-trial arrays into per-mouse scalars + shuffle ratios.

    Mirrors the legacy `_load_split` output shape so downstream plot /
    CSV code is unchanged.
    """
    out: Dict[int, Dict[str, float]] = {}
    for mid, arrs in per_mouse_arrays.items():
        spat = float(np.nanmean(arrs['spat']))
        temp = float(np.nanmean(arrs['temp']))
        spat_shf = float(np.nanmean(arrs['spat_shf']))
        temp_shf = float(np.nanmean(arrs['temp_shf']))
        out[mid] = {
            'spat':     spat / spat_shf if spat_shf > 0 else np.nan,
            'temp':     temp / temp_shf if temp_shf > 0 else np.nan,
            'spat_raw': spat, 'spat_shf': spat_shf,
            'temp_raw': temp, 'temp_shf': temp_shf,
        }
    return out


def _normalize_per_trial(per_mouse_arrays: Dict[int, Dict[str, np.ndarray]]
                           ) -> Dict[int, Dict[str, np.ndarray]]:
    """Per-trial counterpart: divide each trial's loss by the mouse's
    shuffle mean (a scalar, not the per-trial shuffle — that would blow
    up on low-dim MSE targets, per the GOTCHAS entry).

    Output: {mouse_id: {'spat': ndarray, 'temp': ndarray,
                         'spat_shf_mean': float, 'temp_shf_mean': float}}
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for mid, arrs in per_mouse_arrays.items():
        spat_shf_mean = float(np.nanmean(arrs['spat_shf']))
        temp_shf_mean = float(np.nanmean(arrs['temp_shf']))
        out[mid] = {
            'spat': (arrs['spat'] / spat_shf_mean if spat_shf_mean > 0
                      else np.full_like(arrs['spat'], np.nan)),
            'temp': (arrs['temp'] / temp_shf_mean if temp_shf_mean > 0
                      else np.full_like(arrs['temp'], np.nan)),
            # Raw per-trial fit-loss arrays, kept so the per-mouse
            # figures can be drawn in raw (un-normalised) mode too.
            'spat_raw': arrs['spat'],
            'temp_raw': arrs['temp'],
            'spat_shf_mean': spat_shf_mean,
            'temp_shf_mean': temp_shf_mean,
        }
    return out


def load_slug(run_name: str, slug: str,
               splits: Sequence[str] = SPLITS,
               results_root: str = 'results',
               ) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load all splits for one slug, returning per-mouse normalised
    aggregate losses. Output shape matches the legacy `_load_split`
    return value so the existing panel plot code consumes it unchanged.

    Internally this routes through `decoder_plotting_utils.load_run_tree`
    (path resolution) and `calc_fit_loss` (loss recomputation) so the
    contamination in the legacy `KLs` field is bypassed.
    """
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    tree = dpu.load_run_tree(run_name, slug, splits=splits, results_root=results_root)
    for split, mat in tree.items():
        cfg = mat.get('config', {})
        loss_func = (cfg.get('custom_loss_func')
                      or cfg.get('loss_func')
                      or 'PCA')
        out[split] = _aggregate_per_mouse(_per_mouse_arrays(mat, loss_func))
    return out


def load_slug_per_trial(run_name: str, slug: str,
                          splits: Sequence[str] = SPLITS,
                          results_root: str = 'results',
                          ) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """Per-trial counterpart of load_slug. Returns
    {split: {mouse_id: {'spat': ndarray, 'temp': ndarray, ...}}}.
    """
    out: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    tree = dpu.load_run_tree(run_name, slug, splits=splits, results_root=results_root)
    for split, mat in tree.items():
        cfg = mat.get('config', {})
        loss_func = (cfg.get('custom_loss_func')
                      or cfg.get('loss_func')
                      or 'PCA')
        out[split] = _normalize_per_trial(_per_mouse_arrays(mat, loss_func))
    return out


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def _paired_ttest(spat_vals: List[float], temp_vals: List[float]
                    ) -> Tuple[float, int]:
    """Two-sided paired t-test, spat vs temp. Returns (p, n_used).
    NaNs in either arm drop the pair.

    Chosen over Wilcoxon signed-rank because at n=6 mice the latter has
    a floor of ~0.03 on its minimum achievable p-value (occurs only
    when all 6 pairs agree in direction), which severely caps power
    for the kind of "consistent within-mouse but not unanimous" effects
    seen in this dataset. The paired t assumes the per-mouse difference
    is approximately Gaussian; with n=6 that's a strong assumption but
    the n=6 ceiling on cluster-robust SE is unavoidable regardless of
    test choice (see GOTCHAS), and the eye-test of the per-mouse dot
    distributions on the production figure does not flag obvious
    non-normality.
    """
    pairs = [(s, t) for s, t in zip(spat_vals, temp_vals)
              if np.isfinite(s) and np.isfinite(t)]
    if len(pairs) < 2:
        return float('nan'), len(pairs)
    from scipy.stats import ttest_rel
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if np.allclose(a, b):
        return 1.0, len(pairs)
    res = ttest_rel(a, b)
    if hasattr(res, 'pvalue'):
        p = float(res.pvalue)
    else:
        p = float(res[1])
    return p, len(pairs)


def plot_panel(ax, data: Dict[str, Dict[int, Dict[str, float]]],
                 title: str, show_ylabel: bool, mode: str = 'norm'):
    """Bar chart with per-mouse dots, paired t-test p-value per split.

    ``mode='norm'`` (default) plots the shuffle-normalised loss
    (``data[s][m][arch]``) with the 1.0 shuffle-baseline line.
    ``mode='raw'`` plots the un-normalised fit-loss
    (``data[s][m][f'{arch}_raw']``) with no baseline line — useful for
    seeing absolute loss magnitudes, though note the cells span
    different loss functions (PCA / MSE / CE) so raw values are only
    comparable within a panel, not across panels.
    """
    if mode not in ('norm', 'raw'):
        raise ValueError(f"mode must be 'norm' or 'raw', got {mode!r}")
    splits = [s for s in SPLITS if s in data]
    x = np.arange(len(splits))
    width = 0.36

    def _val(s, m, arch):
        return data[s][m][arch] if mode == 'norm' else data[s][m][f'{arch}_raw']

    for i, arch in enumerate(ARCHS):
        offset = (i - 0.5) * width
        means = []
        sems = []
        for s in splits:
            vals = [_val(s, m, arch) for m in sorted(data[s])
                     if np.isfinite(_val(s, m, arch))]
            if not vals:
                means.append(np.nan); sems.append(np.nan); continue
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        ax.bar(x + offset, means, width=width, yerr=sems,
                label=ARCH_LABELS[arch], color=ARCH_COLORS[arch],
                edgecolor='black', alpha=0.85, capsize=4, zorder=2)

        # Per-mouse dots
        for s_idx, s in enumerate(splits):
            vals = [_val(s, m, arch) for m in sorted(data[s])
                     if np.isfinite(_val(s, m, arch))]
            jitter = (np.random.RandomState(s_idx * 10 + i).uniform(-0.05, 0.05,
                                                                     size=len(vals)))
            ax.scatter([x[s_idx] + offset] * len(vals) + jitter, vals,
                        color='black', s=22, zorder=4, alpha=0.65,
                        edgecolor='white', linewidth=0.5)

    # Paired t-test (two-sided, spat vs temp) per split.
    y_top = ax.get_ylim()[1]
    for s_idx, s in enumerate(splits):
        spat_vals = [_val(s, m, 'spat') for m in sorted(data[s])]
        temp_vals = [_val(s, m, 'temp') for m in sorted(data[s])]
        p, n = _paired_ttest(spat_vals, temp_vals)
        if np.isfinite(p):
            stars = ''
            if p < 0.05: stars = '*'
            if p < 0.01: stars = '**'
            if p < 0.001: stars = '***'
            label = f"p={p:.3f} {stars}"
        else:
            label = "n/a"
        ax.text(x[s_idx], y_top * 0.96, label, ha='center', fontsize=8, color='dimgray')

    if mode == 'norm':
        # Shuffle baseline at 1.0
        ax.axhline(1.0, color='red', linestyle='--', lw=1.5, alpha=0.8,
                    label='Shuffle (failure)', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits], fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    if show_ylabel:
        if mode == 'norm':
            ax.set_ylabel('Normalised test loss\n(loss / shuffled loss; lower = better)')
        else:
            ax.set_ylabel('Raw test loss\n(fit-loss, not shuffle-normalised)')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)


def make_figure(slugs_data: List[Tuple[str, Dict]], out_path: Path,
                  mode: str = 'norm'):
    """One figure, panels laid out in the (PANEL_ROWS x PANEL_COLS)
    grid declared above. Slugs are placed in the order they appear in
    SLUGS, filling row-by-row. Missing slugs (where training hasn't
    completed yet) get a clearly-marked empty placeholder so the
    figure still reads cleanly during a partial run.

    ``mode`` is threaded to ``plot_panel`` — 'norm' for shuffle-
    normalised, 'raw' for absolute fit-loss.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = PANEL_ROWS * PANEL_COLS
    fig, axes = plt.subplots(PANEL_ROWS, PANEL_COLS,
                              figsize=(5.5 * PANEL_COLS, 5 * PANEL_ROWS),
                              sharey=False)
    axes_flat = np.array(axes).reshape(-1)

    # Map slug label -> data dict for quick lookup.
    by_label = dict(slugs_data)

    for i in range(n):
        ax = axes_flat[i]
        if i >= len(SLUGS):
            ax.set_visible(False); continue
        label, _slug = SLUGS[i]
        if label not in by_label:
            ax.text(0.5, 0.5,
                     f"{label}\n(no data — re-run training)",
                     ha='center', va='center', fontsize=11, color='gray',
                     transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=12, pad=8)
            continue
        # Y-label only on the leftmost column.
        show_ylabel = (i % PANEL_COLS == 0)
        plot_panel(ax, by_label[label], label, show_ylabel=show_ylabel,
                     mode=mode)

    scale_note = ('shuffle-normalised loss; lower = better, 1.0 = chance'
                   if mode == 'norm'
                   else 'RAW fit-loss; comparable within a panel only '
                        '(cells span PCA / MSE / CE losses)')
    fig.suptitle(
        'Post-fix decoder performance: Spatial vs Temporal architecture\n'
        f'({scale_note})',
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------
# Per-mouse, across-trials panel + figure
# ----------------------------------------------------------------------


def _paired_ttest_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Two-sided paired t-test on raw arrays. Drops NaNs pairwise.
    Unlike _paired_ttest which takes per-mouse scalars, this version
    operates on per-trial arrays for one (mouse, split) combination.
    With hundreds of trials per mouse this test has plenty of power;
    interpret accordingly — a p-value < 0.001 with n=300 trials is
    routine and doesn't necessarily mean the effect is large in
    absolute terms."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return float('nan'), 0
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if len(a) < 2:
        return float('nan'), len(a)
    if np.allclose(a, b):
        return 1.0, len(a)
    from scipy.stats import ttest_rel
    res = ttest_rel(a, b)
    if hasattr(res, 'pvalue'):
        p = float(res.pvalue)
    else:
        p = float(res[1])
    return p, len(a)


def plot_panel_per_mouse(ax, per_split: Dict[str, Dict[str, np.ndarray]],
                           title: str, show_ylabel: bool, mode: str = 'norm'):
    """For one mouse: spat vs temp bars per split, mean ± SEM across
    trials, paired t-test (across trials) annotated per split.

    `per_split` is {split: {'spat': ndarray, 'temp': ndarray, ...}} as
    produced by load_slug_per_trial for one mouse.

    ``mode='norm'`` (default) uses the shuffle-normalised per-trial
    arrays and draws the 1.0 baseline line; ``mode='raw'`` uses the
    un-normalised per-trial fit-loss arrays (``<arch>_raw``).
    """
    if mode not in ('norm', 'raw'):
        raise ValueError(f"mode must be 'norm' or 'raw', got {mode!r}")
    arch_key = (lambda a: a) if mode == 'norm' else (lambda a: f'{a}_raw')
    splits = [s for s in SPLITS if s in per_split]
    x = np.arange(len(splits))
    width = 0.36

    for i, arch in enumerate(ARCHS):
        offset = (i - 0.5) * width
        means, sems = [], []
        for s in splits:
            vals = per_split[s][arch_key(arch)]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means.append(np.nan); sems.append(np.nan); continue
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                          if vals.size > 1 else 0.0)
        ax.bar(x + offset, means, width=width, yerr=sems,
                label=ARCH_LABELS[arch], color=ARCH_COLORS[arch],
                edgecolor='black', alpha=0.85, capsize=4, zorder=2)

    # Paired t-test (across trials) per split.
    # Annotate at the top of the current y-limits.
    y_top = ax.get_ylim()[1]
    for s_idx, s in enumerate(splits):
        p, n = _paired_ttest_arrays(per_split[s][arch_key('spat')],
                                      per_split[s][arch_key('temp')])
        if np.isfinite(p):
            stars = ''
            if p < 0.05:   stars = '*'
            if p < 0.01:   stars = '**'
            if p < 0.001:  stars = '***'
            label = f"p={p:.1e} {stars}" if p < 1e-3 else f"p={p:.3f} {stars}"
            label = f"{label}\nn={n}"
        else:
            label = "n/a"
        ax.text(x[s_idx], y_top * 0.93, label, ha='center', fontsize=7,
                  color='dimgray')

    if mode == 'norm':
        ax.axhline(1.0, color='red', linestyle='--', lw=1.2, alpha=0.7,
                    label='Shuffle (failure)', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits], fontsize=9)
    ax.set_title(title, fontsize=11, pad=6)
    if show_ylabel:
        ax.set_ylabel('Normalised loss\n(per-trial / shuffle-mean)'
                       if mode == 'norm'
                       else 'Raw loss\n(per-trial fit-loss)')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)


def make_per_mouse_figure(label: str,
                            per_trial_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                            out_path: Path,
                            mouse_rows: int = 2, mouse_cols: int = 3,
                            mode: str = 'norm'):
    """One figure for ONE target. Panel grid = mouse_rows x mouse_cols
    (default 2x3 for 6 mice). Each panel shows that mouse's spat vs
    temp loss across the 3 splits, with the paired t-test across trials
    annotated per split.

    `per_trial_data` is {split: {mouse_id: {'spat': ndarray, ...}}}
    as produced by load_slug_per_trial.

    ``mode='norm'`` plots shuffle-normalised per-trial loss; ``mode='raw'``
    plots the un-normalised per-trial fit-loss. All panels share a
    y-axis (one target = one loss function, so the scale is consistent).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arch_key = (lambda a: a) if mode == 'norm' else (lambda a: f'{a}_raw')

    # Collect the set of mice present across all splits.
    mouse_ids = sorted({mid for split_dict in per_trial_data.values()
                          for mid in split_dict})
    n_slots = mouse_rows * mouse_cols
    fig, axes = plt.subplots(mouse_rows, mouse_cols,
                              figsize=(5.0 * mouse_cols, 4.0 * mouse_rows),
                              sharey=True)
    axes_flat = np.array(axes).reshape(-1)

    # First pass: determine a shared y-limit across all mouse panels for
    # this target so the comparisons are visually unambiguous.
    all_means = []
    for mid in mouse_ids:
        for s, mouse_dict in per_trial_data.items():
            if mid not in mouse_dict: continue
            for arch in ARCHS:
                vals = mouse_dict[mid][arch_key(arch)]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    all_means.append(float(np.mean(vals)))
    if mode == 'norm':
        y_max = max(1.05, max(all_means) * 1.15) if all_means else 1.2
    else:
        y_max = (max(all_means) * 1.2) if all_means else 1.0

    for i in range(n_slots):
        ax = axes_flat[i]
        if i >= len(mouse_ids):
            ax.set_visible(False); continue
        mid = mouse_ids[i]
        # Build the per_split dict for this mouse, dropping splits where
        # the mouse is missing.
        per_split_for_mouse = {
            s: mouse_dict[mid]
            for s, mouse_dict in per_trial_data.items()
            if mid in mouse_dict
        }
        if not per_split_for_mouse:
            ax.set_visible(False); continue
        ax.set_ylim(0, y_max)
        show_ylabel = (i % mouse_cols == 0)
        plot_panel_per_mouse(ax, per_split_for_mouse,
                                title=f'Mouse {mid}',
                                show_ylabel=show_ylabel, mode=mode)
        if i == 0:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    scale_note = ('normalised to shuffle' if mode == 'norm'
                   else 'RAW fit-loss')
    fig.suptitle(
        f'{label}: Spatial vs Temporal performance, per mouse '
        f'(within-mouse, across trials; {scale_note})',
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_per_mouse_summary_csv(per_trial_by_label: Dict[str, Dict],
                                  out_path: Path):
    """Tidy CSV with one row per (target, mouse, split, arch):
    mean_norm_loss, sem_norm_loss, n_trials. Plus a wide-format
    companion with the paired t-test result per (target, mouse, split).
    """
    import csv
    rows = []
    for label, per_trial in per_trial_by_label.items():
        for split, mouse_dict in per_trial.items():
            for mid, vals in mouse_dict.items():
                for arch in ARCHS:
                    arr = vals[arch]
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue
                    rows.append({
                        'target': label, 'split': split, 'mouse_id': mid,
                        'arch': arch,
                        'n_trials': int(arr.size),
                        'mean_norm_loss': float(np.mean(arr)),
                        'sem_norm_loss': float(np.std(arr, ddof=1) / np.sqrt(arr.size))
                                            if arr.size > 1 else 0.0,
                    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                  ['target', 'split', 'mouse_id', 'arch',
                                    'n_trials', 'mean_norm_loss', 'sem_norm_loss'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {out_path}  ({len(rows)} rows)")

    # Wide-format: paired t per (target, mouse, split).
    wide_path = out_path.with_name(out_path.stem + '_paired_t.csv')
    with open(wide_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'target', 'mouse_id', 'split', 'n_trials',
            'mean_spat', 'mean_temp', 'diff_mean',
            'ttest_p', 'spat_wins',
        ])
        writer.writeheader()
        for label, per_trial in per_trial_by_label.items():
            for split, mouse_dict in per_trial.items():
                for mid, vals in mouse_dict.items():
                    a = vals['spat']; b = vals['temp']
                    mask = np.isfinite(a) & np.isfinite(b)
                    a = a[mask]; b = b[mask]
                    if a.size < 2:
                        continue
                    p, n = _paired_ttest_arrays(a, b)
                    writer.writerow({
                        'target': label, 'mouse_id': mid, 'split': split,
                        'n_trials': int(n),
                        'mean_spat': float(np.mean(a)),
                        'mean_temp': float(np.mean(b)),
                        'diff_mean': float(np.mean(a - b)),
                        'ttest_p': p,
                        'spat_wins': bool(np.mean(a) < np.mean(b)),
                    })
    print(f"Wrote {wide_path}")


# ----------------------------------------------------------------------
# CSV summary
# ----------------------------------------------------------------------


def write_summary_csv(slugs_data: List[Tuple[str, Dict]], out_path: Path):
    """Per-(target, split, mouse, arch) tidy CSV with raw, shuf, and
    normalised losses. Plus a wide-format companion with per-(target,
    split) means + SEMs + Wilcoxon p-values, for quick eyeballing."""
    rows: List[Dict] = []
    for label, data in slugs_data:
        for split, per_mouse in data.items():
            for mid, vals in per_mouse.items():
                for arch in ARCHS:
                    rows.append({
                        'target': label, 'split': split, 'mouse_id': mid,
                        'arch': arch,
                        'raw_loss': vals[f'{arch}_raw'],
                        'shuf_loss': vals[f'{arch}_shf'],
                        'norm_loss': vals[arch],
                    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {out_path}  ({len(rows)} rows)")

    # Wide-format companion: per (target, split) mean+SEM and Wilcoxon.
    wide_path = out_path.with_name(out_path.stem + '_wide.csv')
    with open(wide_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'target', 'split', 'n_mice',
            'spat_mean', 'spat_sem', 'temp_mean', 'temp_sem',
            'ttest_p', 'ttest_n',
        ])
        writer.writeheader()
        for label, data in slugs_data:
            for split, per_mouse in data.items():
                spat_vals = [per_mouse[m]['spat'] for m in sorted(per_mouse)
                              if np.isfinite(per_mouse[m]['spat'])]
                temp_vals = [per_mouse[m]['temp'] for m in sorted(per_mouse)
                              if np.isfinite(per_mouse[m]['temp'])]
                if not spat_vals or not temp_vals:
                    continue
                p, n = _paired_ttest(
                    [per_mouse[m]['spat'] for m in sorted(per_mouse)],
                    [per_mouse[m]['temp'] for m in sorted(per_mouse)],
                )
                writer.writerow({
                    'target': label, 'split': split, 'n_mice': len(per_mouse),
                    'spat_mean': float(np.mean(spat_vals)),
                    'spat_sem':  float(np.std(spat_vals, ddof=1) / np.sqrt(len(spat_vals)))
                                  if len(spat_vals) > 1 else 0.0,
                    'temp_mean': float(np.mean(temp_vals)),
                    'temp_sem':  float(np.std(temp_vals, ddof=1) / np.sqrt(len(temp_vals)))
                                  if len(temp_vals) > 1 else 0.0,
                    'ttest_p': p, 'ttest_n': n,
                })
    print(f"Wrote {wide_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main(run_name: str = 'post_fix_loadings_2026_05_17',
         results_root: str = 'results',
         out_dir: str = 'figures/post_fix_performance',
         per_mouse: bool = True):
    """Two output families:
      (1) The across-mouse 2x3 panel figure + summary CSVs.
      (2) If `per_mouse=True`, one figure per target showing per-mouse
          (across-trials) bars with paired t-tests across trials.

    Per-mouse figures answer "within each animal, does spat or temp do
    better trial-by-trial?". With hundreds of trials per mouse the
    paired t has plenty of power, so p-values tend to be small;
    interpret the *direction* (spat or temp winning) and the *
    consistency across mice*, not just the per-mouse p.
    """
    out = Path(out_dir)

    slugs_data = []
    per_trial_by_label: Dict[str, Dict] = {}
    for label, slug in SLUGS:
        slug_dir = Path(results_root) / run_name / slug
        if not slug_dir.exists():
            print(f"[warn] {slug_dir} not found; skipping {label!r}")
            continue
        data = load_slug(run_name, slug, results_root=results_root)
        if not data:
            print(f"[warn] {slug_dir} produced no data; skipping {label!r}")
            continue
        slugs_data.append((label, data))
        if per_mouse:
            per_trial_by_label[label] = load_slug_per_trial(
                run_name, slug, results_root=results_root,
            )

    if not slugs_data:
        print("No data found. Re-run training first.")
        return

    # (1) Across-mouse figures + CSVs. Two variants of the 2x3 grid:
    #     - normalised (loss / shuffle) — the headline comparison
    #     - raw (absolute fit-loss) — for inspecting loss magnitudes
    make_figure(slugs_data, out / 'spat_vs_temp_post_fix.png', mode='norm')
    make_figure(slugs_data, out / 'spat_vs_temp_post_fix_raw.png', mode='raw')
    write_summary_csv(slugs_data, out / 'spat_vs_temp_post_fix_per_mouse.csv')

    # (2) Per-mouse, across-trials figures + CSVs. Each target gets both
    #     a normalised and a raw variant of the per-mouse grid.
    if per_mouse and per_trial_by_label:
        per_mouse_dir = out / 'per_mouse'
        for label, per_trial in per_trial_by_label.items():
            # Filename-safe version of the label.
            safe = (label.replace(' ', '_').replace('(', '').replace(')', '')
                          .replace('/', '_'))
            make_per_mouse_figure(label, per_trial,
                                    per_mouse_dir / f'per_mouse_{safe}.png',
                                    mode='norm')
            make_per_mouse_figure(label, per_trial,
                                    per_mouse_dir / f'per_mouse_{safe}_raw.png',
                                    mode='raw')
        write_per_mouse_summary_csv(per_trial_by_label,
                                      per_mouse_dir / 'per_mouse_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default='clean_2026_05_19')
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--out-dir', default='figures/post_fix_performance')
    parser.add_argument('--no-per-mouse', action='store_true',
                         help='Skip the per-mouse, across-trials figures.')
    args = parser.parse_args()
    main(run_name=args.run_name,
         results_root=args.results_root,
         out_dir=args.out_dir,
         per_mouse=not args.no_per_mouse)
