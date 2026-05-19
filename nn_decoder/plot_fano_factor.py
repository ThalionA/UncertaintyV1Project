# -*- coding: utf-8 -*-
"""
Fano-factor / across-trial dispersion analyses, fully self-contained.

Three computations, three viewing modes (per-mouse + grand average across mice):

  Computations
  ------------
    compute_pooled_ff_timecourse
        Churchland-style time course pooled across (neuron × condition) pairs.
        One FF per time bin per mouse; mean-matched variant included.

    compute_ff_per_condition_window
        Aggregate spike counts over the full time window per (trial, neuron),
        then for each (orientation, contrast, dispersion) cell compute the
        across-neuron mean per-neuron Fano and the slope-FF.

    compute_ff_per_condition_timecourse
        Per (orientation, contrast, dispersion) cell, the time course of the
        across-trial var/mean slope across neurons. One curve per qualifying
        condition per mouse.

  Per-mouse plots
  ---------------
    plot_pooled_ff_timecourse_per_mouse
    plot_ff_per_condition_per_mouse           (faceted heatmap)
    plot_ff_per_condition_lines_per_mouse     (line trend view)

  Grand-average plots (mean ± SEM across mice)
  ---------------------------------------------
    plot_pooled_ff_timecourse_grand
    plot_ff_per_condition_grand               (heatmap, mean across mice)
    plot_ff_per_condition_timecourse_marginal (3 panels: time course averaged
                                               by orientation / contrast /
                                               dispersion separately)
    plot_ff_per_condition_timecourse_facet    (full Ori × Disp facets, lines
                                               coloured by Contrast)

IMPORTANT — provenance caveat (see also `documents/residual_partial_correlation.md`
and the population-metrics notes):

`Gspk` from VR_Decoder_Data_Export.mat is **CASCADE-deconvolved spike rate**,
not integer Poisson counts. CASCADE smooths across time and trials, so
across-trial variance is suppressed and these slopes systematically read
sub-Poisson (FF << 1). The FF=1 reference does not apply. Use these values
RELATIVELY — across time, conditions, and animals — not against the
Poisson reference. The plot annotations make this caveat explicit.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# ==========================================
# 0. Style + small helpers
# ==========================================

PALETTE = {
    'primary': '#4361ee',
    'secondary': '#f72585',
    'accent': '#7209b7',
    'gray': '#888888',
}

# Sequential palettes that stay dark on white (no bright yellow tail).
ORI_CMAP = sns.color_palette("mako_r", as_cmap=True)
CONT_CMAP = sns.color_palette("flare", as_cmap=True)
DISP_CMAP = sns.color_palette("crest_r", as_cmap=True)


def set_plot_style():
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'sans-serif', 'axes.linewidth': 0.8,
        'axes.edgecolor': '#333', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5, 'grid.alpha': 0.7,
    })


def _cmap_color(cmap, i, n):
    """Sample a colormap at the i-th of n points, biased into the darker
    middle so we never reach the very pale ends."""
    if n <= 1:
        return cmap(0.5)
    return cmap(0.15 + 0.7 * i / (n - 1))


def _pooled_mean_sem(arr, axis=0):
    """Mean and SEM along `axis`, NaN-aware. SEM is NaN if only one row."""
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr, axis=axis)
    if arr.shape[axis] > 1:
        sem = stats.sem(arr, axis=axis, nan_policy='omit')
    else:
        sem = np.full_like(mean, np.nan, dtype=float)
    return mean, sem


def _errorbar_or_line(ax, x, mean, sem, **kw):
    if sem is not None and np.any(np.isfinite(sem)):
        ax.errorbar(x, mean, yerr=sem, capsize=3, **kw)
    else:
        clean = {k: v for k, v in kw.items() if k != 'capsize'}
        ax.plot(x, mean, **clean)


def _shaded_line(ax, x, mean, sem, color, lw=2, ls='-', label=None, alpha_fill=0.18):
    ax.plot(x, mean, ls=ls, color=color, lw=lw, label=label)
    if sem is not None and np.any(np.isfinite(sem)):
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha_fill,
                        linewidth=0)


# ==========================================
# 1. Condition labels
# ==========================================

def _build_condition_labels(trials):
    """Integer condition label per trial from (orientation, contrast, dispersion).
    Returns (labels, n_unique, unique_combos)."""
    oris = np.asarray(trials['orientation'])
    cons = np.asarray(trials['contrast'])
    disp = np.asarray(trials['dispersion'])
    combos = list(zip(oris, cons, disp))
    unique = sorted(set(combos))
    lookup = {c: i for i, c in enumerate(unique)}
    labels = np.array([lookup[c] for c in combos])
    return labels, len(unique), unique


# ==========================================
# 2. Computation
# ==========================================

def compute_pooled_ff_timecourse(raw_activity, trials,
                                 min_trials_per_cond=7,
                                 n_bootstrap=200,
                                 n_match_bins=20,
                                 rng_seed=0):
    """Churchland (2010) time-course of var-vs-mean slope, pooled across
    (neuron × condition) pairs.

    See module-level docstring for the CASCADE-rate caveat.

    Returns
    -------
    ff_raw        : (t_bins,) raw slope per time bin.
    ff_matched    : (t_bins,) bootstrap mean of mean-matched slope.
    ff_matched_ci : (2, t_bins) 5th/95th percentile CI from bootstrap reps.
    time_bins     : (t_bins,) bin indices.
    """
    n_trials, t_bins, n_neurons = raw_activity.shape
    cond_labels, n_conds, _ = _build_condition_labels(trials)

    cond_masks = {c: np.where(cond_labels == c)[0]
                  for c in range(n_conds)
                  if (cond_labels == c).sum() >= min_trials_per_cond}
    n_valid = len(cond_masks)
    print(f"  Pooled FF time-course: {n_valid}/{n_conds} conds "
          f"with >= {min_trials_per_cond} trials")

    nan_arr = np.full(t_bins, np.nan)
    nan_ci = np.stack([nan_arr, nan_arr])
    if n_valid == 0:
        return nan_arr, nan_arr, nan_ci, np.arange(t_bins)

    max_pairs = n_neurons * n_valid
    all_means = np.zeros((t_bins, max_pairs))
    all_vars = np.zeros((t_bins, max_pairs))
    all_weights = np.zeros((t_bins, max_pairs))
    n_pairs = np.zeros(t_bins, dtype=int)

    for cond_id, trial_idx in cond_masks.items():
        n_c = len(trial_idx)
        cond_data = raw_activity[trial_idx]
        c_mean = np.nanmean(cond_data, axis=0)
        c_var = np.nanvar(cond_data, axis=0, ddof=1)
        for t in range(t_bins):
            for n in range(n_neurons):
                m = c_mean[t, n]
                v = c_var[t, n]
                if m > 1e-10:
                    idx = n_pairs[t]
                    all_means[t, idx] = m
                    all_vars[t, idx] = v
                    all_weights[t, idx] = (n_c - 1) / (2.0 * m * m)
                    n_pairs[t] += 1

    ff_raw = np.full(t_bins, np.nan)
    for t in range(t_bins):
        n_t = n_pairs[t]
        if n_t < 5:
            continue
        m = all_means[t, :n_t]
        v = all_vars[t, :n_t]
        w = all_weights[t, :n_t]
        ff_raw[t] = np.sum(w * m * v) / np.sum(w * m * m)

    flat_means_list = [all_means[t, :n_pairs[t]] for t in range(t_bins) if n_pairs[t] > 0]
    if not flat_means_list:
        return ff_raw, nan_arr, nan_ci, np.arange(t_bins)
    flat_means = np.concatenate(flat_means_list)

    quantiles = np.linspace(0, 1, n_match_bins + 1)
    bin_edges = np.unique(np.quantile(flat_means, quantiles))
    n_actual_bins = max(len(bin_edges) - 1, 1)

    bin_assignments = {}
    hist_per_time = np.zeros((t_bins, n_actual_bins), dtype=int)
    for t in range(t_bins):
        n_t = n_pairs[t]
        if n_t == 0:
            continue
        m = all_means[t, :n_t]
        ba = np.digitize(m, bin_edges) - 1
        ba = np.clip(ba, 0, n_actual_bins - 1)
        bin_assignments[t] = ba
        for b in range(n_actual_bins):
            hist_per_time[t, b] = np.sum(ba == b)
    common_hist = np.min(hist_per_time, axis=0)

    ff_matched_all = np.full((n_bootstrap, t_bins), np.nan)
    for rep in range(n_bootstrap):
        rng = np.random.RandomState(rng_seed + rep)
        for t in range(t_bins):
            n_t = n_pairs[t]
            if n_t < 5 or t not in bin_assignments:
                continue
            ba = bin_assignments[t]
            m = all_means[t, :n_t]
            v = all_vars[t, :n_t]
            w = all_weights[t, :n_t]
            keep = np.zeros(n_t, dtype=bool)
            for b in range(n_actual_bins):
                in_bin = np.where(ba == b)[0]
                n_keep = min(common_hist[b], len(in_bin))
                if n_keep > 0:
                    chosen = rng.choice(in_bin, n_keep, replace=False)
                    keep[chosen] = True
            mk, vk, wk = m[keep], v[keep], w[keep]
            if mk.size >= 5:
                ff_matched_all[rep, t] = np.sum(wk * mk * vk) / np.sum(wk * mk * mk)

    ff_matched = np.nanmean(ff_matched_all, axis=0)
    ff_matched_ci = np.stack([
        np.nanpercentile(ff_matched_all, 5, axis=0),
        np.nanpercentile(ff_matched_all, 95, axis=0),
    ])
    return ff_raw, ff_matched, ff_matched_ci, np.arange(t_bins)


def compute_ff_per_condition_window(raw_activity, trials,
                                    min_trials_per_cond=7,
                                    time_slice=None,
                                    min_neurons=5):
    """Per-(orientation, contrast, dispersion) FF, aggregated over the window.

    Returns a list of dicts with keys: orientation, contrast, dispersion,
    n_trials, n_neurons, mean_per_neuron_fano, slope_ff, mean_rate.
    """
    n_trials, t_bins, n_neurons = raw_activity.shape
    if time_slice is None:
        time_slice = slice(None)
    cond_labels, n_conds, unique_combos = _build_condition_labels(trials)

    agg = np.nansum(raw_activity[:, time_slice, :], axis=1)

    rows = []
    for c, combo in enumerate(unique_combos):
        idx = np.where(cond_labels == c)[0]
        n_c = len(idx)
        if n_c < min_trials_per_cond:
            continue
        cond_agg = agg[idx]
        m = np.nanmean(cond_agg, axis=0)
        v = np.nanvar(cond_agg, axis=0, ddof=1)
        valid = (m > 1e-3) & np.isfinite(v)
        if valid.sum() < min_neurons:
            continue
        m_v = m[valid]
        v_v = v[valid]
        rows.append({
            'orientation': combo[0],
            'contrast': combo[1],
            'dispersion': combo[2],
            'n_trials': int(n_c),
            'n_neurons': int(valid.sum()),
            'mean_per_neuron_fano': float(np.nanmean(v_v / m_v)),
            'slope_ff': float(np.sum(m_v * v_v) / np.sum(m_v * m_v)),
            'mean_rate': float(np.nanmean(m_v)),
        })
    return rows


def compute_ff_per_condition_timecourse(raw_activity, trials,
                                        min_trials_per_cond=7,
                                        min_neurons=5):
    """For each qualifying (ori, contrast, disp), the time course of the
    across-trial var-vs-mean slope across neurons.

    Returns
    -------
    dict mapping combo tuple (ori, contrast, disp) → array shape (t_bins,).
    Conditions with fewer than `min_trials_per_cond` trials are skipped.
    Time bins where fewer than `min_neurons` neurons have positive mean are
    set to NaN.
    """
    n_trials, t_bins, n_neurons = raw_activity.shape
    cond_labels, n_conds, unique_combos = _build_condition_labels(trials)

    out = {}
    for c, combo in enumerate(unique_combos):
        idx = np.where(cond_labels == c)[0]
        if len(idx) < min_trials_per_cond:
            continue
        cond_data = raw_activity[idx]
        c_mean = np.nanmean(cond_data, axis=0)
        c_var = np.nanvar(cond_data, axis=0, ddof=1)
        ff_t = np.full(t_bins, np.nan)
        for t in range(t_bins):
            m = c_mean[t]
            v = c_var[t]
            valid = (m > 1e-3) & np.isfinite(v)
            if valid.sum() < min_neurons:
                continue
            mv = m[valid]
            vv = v[valid]
            ff_t[t] = np.sum(mv * vv) / np.sum(mv * mv)
        out[combo] = ff_t
    return out


# ==========================================
# 3. Per-mouse plots
# ==========================================

def plot_pooled_ff_timecourse_per_mouse(ff_results, bin_ms=50,
                                        output_path="ff_pooled_timecourse_per_mouse.svg"):
    """One subplot per mouse, shows raw + matched + bootstrap CI band."""
    set_plot_style()
    n_mice = len(ff_results)
    if n_mice == 0:
        return
    n_cols = min(3, n_mice)
    n_rows = (n_mice + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for i, (mid, ff_raw, ff_matched, ci, t_bins) in enumerate(ff_results):
        ax = axes[i]
        time_s = t_bins * bin_ms / 1000.0
        ax.fill_between(time_s, ci[0], ci[1], color=PALETTE['secondary'],
                        alpha=0.18, label='Matched 5–95%' if i == 0 else None)
        ax.plot(time_s, ff_raw, '-', color=PALETTE['primary'], lw=2,
                label='Raw' if i == 0 else None)
        ax.plot(time_s, ff_matched, '-', color=PALETTE['secondary'], lw=2,
                label='Mean-matched' if i == 0 else None)
        ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.4,
                   label='Poisson ref (n/a, CASCADE rates)' if i == 0 else None)
        ax.set_title(f"Mouse {mid}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("Var/Mean slope", fontsize=9)
        finite = np.concatenate([ff_raw[np.isfinite(ff_raw)],
                                  ff_matched[np.isfinite(ff_matched)]])
        if finite.size:
            ymin = max(0.0, finite.min() * 0.9)
            ymax = max(finite.max() * 1.15, 1.05)
        else:
            ymin, ymax = 0.0, 1.5
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc='best')
    for j in range(n_mice, len(axes)):
        axes[j].axis('off')
    fig.suptitle("Pooled Var/Mean slope time course (per mouse) — CASCADE rates",
                 fontsize=13, fontweight='bold', y=1.03)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


def plot_ff_per_condition_per_mouse(ff_per_cond_results,
                                    metric='mean_per_neuron_fano',
                                    output_path="ff_per_condition_per_mouse.svg"):
    """Faceted heatmap: rows = mice, cols = orientation, cells = contrast×dispersion."""
    set_plot_style()
    if not ff_per_cond_results:
        return
    all_rows = [r for (_, rows) in ff_per_cond_results for r in rows]
    if not all_rows:
        return
    all_oris = sorted({r['orientation'] for r in all_rows})
    all_cons = sorted({r['contrast'] for r in all_rows})
    all_disps = sorted({r['dispersion'] for r in all_rows})
    vals = np.array([r[metric] for r in all_rows], dtype=float)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    n_mice = len(ff_per_cond_results)
    n_oris = len(all_oris)
    fig, axes = plt.subplots(n_mice, n_oris,
                             figsize=(2.6 * n_oris, 2.4 * n_mice),
                             constrained_layout=True, squeeze=False)
    last_im = None
    for i, (mid, rows) in enumerate(ff_per_cond_results):
        for j, ori in enumerate(all_oris):
            ax = axes[i, j]
            mat = np.full((len(all_cons), len(all_disps)), np.nan)
            n_mat = np.zeros_like(mat, dtype=int)
            for r in rows:
                if r['orientation'] != ori:
                    continue
                ci = all_cons.index(r['contrast'])
                di = all_disps.index(r['dispersion'])
                mat[ci, di] = r[metric]
                n_mat[ci, di] = r['n_trials']
            im = ax.imshow(mat, aspect='auto', cmap='magma',
                           vmin=vmin, vmax=vmax, origin='lower')
            last_im = im
            for ci in range(len(all_cons)):
                for di in range(len(all_disps)):
                    if not np.isnan(mat[ci, di]):
                        ax.text(di, ci, f"{mat[ci, di]:.2f}\n(n={n_mat[ci, di]})",
                                ha='center', va='center', fontsize=6,
                                color='white' if mat[ci, di] < (vmin + vmax) / 2 else 'black')
            ax.set_xticks(range(len(all_disps)))
            ax.set_xticklabels([f"{d:g}" for d in all_disps], fontsize=7)
            ax.set_yticks(range(len(all_cons)))
            ax.set_yticklabels([f"{c:g}" for c in all_cons], fontsize=7)
            if i == 0:
                ax.set_title(f"Δ={ori:g}°", fontsize=9)
            if i == n_mice - 1:
                ax.set_xlabel("Dispersion", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Mouse {mid}\nContrast", fontsize=8)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.7, pad=0.02)
        cbar.set_label(f"{metric}\n(CASCADE rates — Poisson FF=1 not applicable)",
                       fontsize=8)
    fig.suptitle("Per-condition FF (per mouse, full window)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


def plot_ff_per_condition_lines_per_mouse(ff_per_cond_results,
                                          metric='mean_per_neuron_fano',
                                          output_path="ff_per_condition_lines_per_mouse.svg"):
    """Per-mouse trend view: rows = mice, cols = dispersion, lines = contrast,
    x = orientation."""
    set_plot_style()
    if not ff_per_cond_results:
        return
    rows = []
    for mid, cond_rows in ff_per_cond_results:
        for r in cond_rows:
            rec = dict(r)
            rec['mouse_id'] = mid
            rows.append(rec)
    if not rows:
        return
    df = pd.DataFrame(rows)
    all_disps = sorted(df['dispersion'].unique())
    cons = sorted(df['contrast'].unique())
    mice = sorted(df['mouse_id'].unique())
    n_mice = len(mice)
    n_disps = len(all_disps)
    fig, axes = plt.subplots(n_mice, n_disps,
                             figsize=(3.2 * n_disps, 2.6 * n_mice),
                             constrained_layout=True, squeeze=False, sharey=True)
    for i, mid in enumerate(mice):
        sub_m = df[df['mouse_id'] == mid]
        for j, dv in enumerate(all_disps):
            ax = axes[i, j]
            sub = sub_m[sub_m['dispersion'] == dv]
            for k, cv in enumerate(cons):
                cur = sub[sub['contrast'] == cv].sort_values('orientation')
                if cur.empty:
                    continue
                color = _cmap_color(CONT_CMAP, k, len(cons))
                ax.plot(cur['orientation'], cur[metric], marker='o', lw=2,
                        color=color,
                        label=f"C={cv:g}" if i == 0 and j == 0 else None)
            if i == 0:
                ax.set_title(f"Dispersion = {dv:g}", fontsize=9)
            if i == n_mice - 1:
                ax.set_xlabel("|Δ from Go| (deg)", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Mouse {mid}\n{metric}", fontsize=8)
            ax.tick_params(labelsize=7)
    if mice and cons:
        axes[0, 0].legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle("Per-condition FF trend (per mouse, full window, ≥ 7 trials)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


# ==========================================
# 4. Grand-average plots (mean ± SEM across mice)
# ==========================================

def plot_pooled_ff_timecourse_grand(ff_results, bin_ms=50,
                                    output_path="ff_pooled_timecourse_grand.svg",
                                    title_suffix=""):
    """Grand-average pooled FF time course across mice (mean ± SEM)."""
    set_plot_style()
    if not ff_results:
        return
    n_mice = len(ff_results)
    t_bins = len(ff_results[0][1])
    raw_arr = np.full((n_mice, t_bins), np.nan)
    matched_arr = np.full((n_mice, t_bins), np.nan)
    for i, (_, ff_raw, ff_matched, _, _) in enumerate(ff_results):
        raw_arr[i] = ff_raw
        matched_arr[i] = ff_matched

    raw_mean, raw_sem = _pooled_mean_sem(raw_arr)
    m_mean, m_sem = _pooled_mean_sem(matched_arr)
    time_s = np.arange(t_bins) * bin_ms / 1000.0

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    _shaded_line(ax, time_s, raw_mean, raw_sem,
                 color=PALETTE['primary'], lw=2,
                 label=f'Raw (mean ± SEM, n={n_mice} mice)')
    _shaded_line(ax, time_s, m_mean, m_sem,
                 color=PALETTE['secondary'], lw=2,
                 label='Mean-matched (mean ± SEM)')
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.4,
               label='Poisson ref (n/a, CASCADE rates)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Var/Mean slope")
    title = "Pooled Var/Mean slope — grand average across mice"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='best')
    finite = np.concatenate([raw_mean[np.isfinite(raw_mean)],
                              m_mean[np.isfinite(m_mean)]])
    if finite.size:
        ymin = max(0.0, finite.min() * 0.9)
        ymax = max(finite.max() * 1.15, 1.05)
        ax.set_ylim(ymin, ymax)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


def plot_ff_per_condition_grand(ff_per_cond_results,
                                metric='mean_per_neuron_fano',
                                output_path="ff_per_condition_grand.svg",
                                title_suffix=""):
    """One row of heatmaps (one per orientation), each cell = mean across
    mice of the per-condition metric (after each mouse contributed). Number
    annotated in each cell is "mean (n_mice)" to show how many animals
    contributed to that cell.
    """
    set_plot_style()
    if not ff_per_cond_results:
        return
    all_rows = []
    for mid, rows in ff_per_cond_results:
        for r in rows:
            rec = dict(r)
            rec['mouse_id'] = mid
            all_rows.append(rec)
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    all_oris = sorted(df['orientation'].unique())
    all_cons = sorted(df['contrast'].unique())
    all_disps = sorted(df['dispersion'].unique())

    # Aggregate across mice
    agg = (df.groupby(['orientation', 'contrast', 'dispersion'])[metric]
             .agg(['mean', 'sem', 'count']).reset_index())
    vmin = float(agg['mean'].min())
    vmax = float(agg['mean'].max())

    n_oris = len(all_oris)
    fig, axes = plt.subplots(1, n_oris, figsize=(3.2 * n_oris, 3.6),
                             constrained_layout=True, squeeze=False)
    last_im = None
    for j, ori in enumerate(all_oris):
        ax = axes[0, j]
        mat = np.full((len(all_cons), len(all_disps)), np.nan)
        n_mice_mat = np.zeros_like(mat, dtype=int)
        for _, r in agg.iterrows():
            if r['orientation'] != ori:
                continue
            ci = all_cons.index(r['contrast'])
            di = all_disps.index(r['dispersion'])
            mat[ci, di] = r['mean']
            n_mice_mat[ci, di] = int(r['count'])
        im = ax.imshow(mat, aspect='auto', cmap='magma',
                       vmin=vmin, vmax=vmax, origin='lower')
        last_im = im
        for ci in range(len(all_cons)):
            for di in range(len(all_disps)):
                if not np.isnan(mat[ci, di]):
                    ax.text(di, ci,
                            f"{mat[ci, di]:.2f}\n(N={n_mice_mat[ci, di]})",
                            ha='center', va='center', fontsize=7,
                            color='white' if mat[ci, di] < (vmin + vmax) / 2 else 'black')
        ax.set_xticks(range(len(all_disps)))
        ax.set_xticklabels([f"{d:g}" for d in all_disps], fontsize=8)
        ax.set_yticks(range(len(all_cons)))
        ax.set_yticklabels([f"{c:g}" for c in all_cons], fontsize=8)
        ax.set_title(f"Δ from Go = {ori:g}°", fontsize=10)
        ax.set_xlabel("Dispersion", fontsize=9)
        if j == 0:
            ax.set_ylabel("Contrast", fontsize=9)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label(
            f"Mean {metric} across mice\n(annotation: value (N=mice contributing))",
            fontsize=8)
    title = "Per-condition FF — grand average across mice (full window)"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.04)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


def plot_ff_per_condition_timecourse_marginal(per_mouse_tc_results, bin_ms=50,
                                              output_path="ff_per_condition_timecourse_marginal.svg",
                                              title_suffix=""):
    """Three-panel grand-average view of the per-condition FF time course.

    Each panel marginalises over the OTHER two stim features:
      - Panel 1: lines = orientation, averaged over (contrast, dispersion).
      - Panel 2: lines = contrast,    averaged over (orientation, dispersion).
      - Panel 3: lines = dispersion,  averaged over (orientation, contrast).

    Aggregation: at each (mouse, time bin), average the per-condition FF
    over the other-two-feature cells available for that mouse, then take
    the mean ± SEM across mice for each marginal feature value.

    `per_mouse_tc_results` : list of (mouse_id, dict{combo: ff_t})
    """
    set_plot_style()
    if not per_mouse_tc_results:
        return

    all_combos = set()
    t_bins = None
    for _, d in per_mouse_tc_results:
        all_combos.update(d.keys())
        for v in d.values():
            t_bins = len(v)
            break
    if t_bins is None:
        return
    all_combos = sorted(all_combos)

    n_mice = len(per_mouse_tc_results)
    time_s = np.arange(t_bins) * bin_ms / 1000.0

    feature_specs = [
        ('orientation', 0, ORI_CMAP, "|Δ from Go| (deg)"),
        ('contrast', 1, CONT_CMAP, "Contrast"),
        ('dispersion', 2, DISP_CMAP, "Dispersion"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5),
                             constrained_layout=True, sharey=True)

    for panel_idx, (feat_name, axis, cmap, feat_label) in enumerate(feature_specs):
        ax = axes[panel_idx]
        all_vals = sorted({c[axis] for c in all_combos})
        for k, val in enumerate(all_vals):
            # For each mouse, average ff_t across all combos with this feature value
            per_mouse_vec = np.full((n_mice, t_bins), np.nan)
            for i, (_, d) in enumerate(per_mouse_tc_results):
                matching = [d[c] for c in d if c[axis] == val]
                if matching:
                    per_mouse_vec[i] = np.nanmean(np.stack(matching), axis=0)
            mean, sem = _pooled_mean_sem(per_mouse_vec)
            color = _cmap_color(cmap, k, len(all_vals))
            _shaded_line(ax, time_s, mean, sem, color=color, lw=2,
                         label=f"{val:g}")
        ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.3)
        ax.set_xlabel("Time (s)")
        if panel_idx == 0:
            ax.set_ylabel("Var/Mean slope (mean ± SEM across mice)")
        ax.set_title(f"Marginal across {feat_label}", fontsize=11)
        ax.legend(title=feat_name.title(), fontsize=8, frameon=False, loc='best')

    title = "Per-condition FF time course — grand-average marginals"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.04)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


def plot_ff_per_condition_timecourse_facet(per_mouse_tc_results, bin_ms=50,
                                           output_path="ff_per_condition_timecourse_facet.svg",
                                           title_suffix=""):
    """Fully faceted grand-average time courses.
    Rows = orientation, cols = dispersion, lines = contrast.
    Mean ± SEM across mice in each (orientation, contrast, dispersion) cell.
    """
    set_plot_style()
    if not per_mouse_tc_results:
        return

    all_combos = set()
    t_bins = None
    for _, d in per_mouse_tc_results:
        all_combos.update(d.keys())
        for v in d.values():
            t_bins = len(v)
            break
    if t_bins is None:
        return

    all_oris = sorted({c[0] for c in all_combos})
    all_cons = sorted({c[1] for c in all_combos})
    all_disps = sorted({c[2] for c in all_combos})
    n_mice = len(per_mouse_tc_results)
    time_s = np.arange(t_bins) * bin_ms / 1000.0

    n_rows = len(all_oris)
    n_cols = len(all_disps)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.4 * n_rows),
                             constrained_layout=True, squeeze=False, sharey=True)

    for i, ori in enumerate(all_oris):
        for j, disp in enumerate(all_disps):
            ax = axes[i, j]
            for k, con in enumerate(all_cons):
                combo = (ori, con, disp)
                per_mouse_vec = np.full((n_mice, t_bins), np.nan)
                for m_idx, (_, d) in enumerate(per_mouse_tc_results):
                    if combo in d:
                        per_mouse_vec[m_idx] = d[combo]
                if np.all(np.isnan(per_mouse_vec)):
                    continue
                mean, sem = _pooled_mean_sem(per_mouse_vec)
                color = _cmap_color(CONT_CMAP, k, len(all_cons))
                _shaded_line(ax, time_s, mean, sem, color=color, lw=2,
                             label=f"C={con:g}" if i == 0 and j == 0 else None)
            ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.3)
            if i == 0:
                ax.set_title(f"Dispersion={disp:g}", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Δ={ori:g}°\nVar/Mean", fontsize=9)
            ax.tick_params(labelsize=7)
    if all_cons:
        axes[0, 0].legend(fontsize=7, frameon=False, loc='best')
    title = "Per-condition FF time course — orientation × dispersion facets, lines = contrast"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {output_path}")
    plt.close()


# ==========================================
# 5. Driver
# ==========================================

def run_ff_pipeline(mouse_ids, output_dir="figures/fano_factor", min_trials_per_cond=7,
                    aggregate_mouse_ids=None, exclude_from_aggregate=None,
                    grand_filename_tag=None):
    """Load all mice, run all FF computations, save all plots and CSVs.

    Parameters
    ----------
    mouse_ids : iterable of int
        Mice to load and run per-mouse analyses for.
    output_dir : str
        Directory under which all SVGs and CSVs are written.
    min_trials_per_cond : int
        Threshold passed to all FF computations.
    aggregate_mouse_ids : iterable of int or None
        If provided, ONLY these mice contribute to the four grand-average
        plots. The per-mouse plots still show every mouse in `mouse_ids`.
        Mutually exclusive with `exclude_from_aggregate`.
    exclude_from_aggregate : iterable of int or None
        Convenience for "all loaded mice except these". Equivalent to
        passing `aggregate_mouse_ids = set(mouse_ids) - set(exclude_from_aggregate)`.
    grand_filename_tag : str or None
        Optional suffix appended to the four grand-average filenames so
        different aggregation choices don't overwrite each other in the
        same `output_dir`. e.g. tag="excl_3" → "ff_per_condition_grand_excl_3.svg".
        If None and the aggregation set differs from the loaded set, an
        automatic tag is generated.

    Examples
    --------
    All animals but the third (id=2):
        run_ff_pipeline(mouse_ids=[0,1,2,3,4,5], exclude_from_aggregate=[2])

    Only mice 0 and 4 in the aggregates, but per-mouse plots for everyone:
        run_ff_pipeline(mouse_ids=[0,1,2,3,4,5], aggregate_mouse_ids=[0, 4])
    """
    from utils_v26 import load_vr_export  # noqa: F401  (deferred to allow tests)

    if aggregate_mouse_ids is not None and exclude_from_aggregate is not None:
        raise ValueError("Pass either aggregate_mouse_ids OR exclude_from_aggregate, not both.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to ./{output_dir}/")

    pooled_results = []
    per_cond_results = []
    per_cond_tc_results = []

    for mouse_id in mouse_ids:
        print(f"Processing Mouse {mouse_id}...")
        try:
            (activities_m, _, _, _, trials) = load_vr_export(mouse_id)
        except Exception as e:
            print(f"  Skipping Mouse {mouse_id}: {e}")
            continue
        raw = np.transpose(activities_m, (1, 2, 0))
        sample = raw[~np.isnan(raw)]
        if sample.size:
            print(f"  Data sniff: dtype={raw.dtype}, "
                  f"min={float(sample.min()):.3g}, "
                  f"max={float(sample.max()):.3g}, "
                  f"mean={float(sample.mean()):.3g} "
                  f"(non-integer ⇒ CASCADE rate)")

        ff_raw, ff_matched, ff_ci, ff_tbins = compute_pooled_ff_timecourse(
            raw, trials, min_trials_per_cond=min_trials_per_cond)
        pooled_results.append((mouse_id, ff_raw, ff_matched, ff_ci, ff_tbins))

        per_cond = compute_ff_per_condition_window(
            raw, trials, min_trials_per_cond=min_trials_per_cond)
        per_cond_results.append((mouse_id, per_cond))

        per_cond_tc = compute_ff_per_condition_timecourse(
            raw, trials, min_trials_per_cond=min_trials_per_cond)
        per_cond_tc_results.append((mouse_id, per_cond_tc))

    if not pooled_results:
        print("[!] No mice processed.")
        return

    # CSVs
    if per_cond_results:
        rows = []
        for mid, rs in per_cond_results:
            for r in rs:
                rec = dict(r)
                rec['mouse_id'] = mid
                rows.append(rec)
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(output_dir, "ff_per_condition.csv"), index=False)
            print(f"  Saved {output_dir}/ff_per_condition.csv ({len(rows)} rows)")

    # Per-mouse plots
    plot_pooled_ff_timecourse_per_mouse(
        pooled_results,
        output_path=os.path.join(output_dir, "ff_pooled_timecourse_per_mouse.svg"))
    plot_ff_per_condition_per_mouse(
        per_cond_results,
        output_path=os.path.join(output_dir, "ff_per_condition_per_mouse.svg"))
    plot_ff_per_condition_lines_per_mouse(
        per_cond_results,
        output_path=os.path.join(output_dir, "ff_per_condition_lines_per_mouse.svg"))

    # ----- Determine aggregation subset for grand-average plots -----
    loaded_ids = [mid for mid, *_ in pooled_results]
    if aggregate_mouse_ids is not None:
        agg_set = set(aggregate_mouse_ids)
    elif exclude_from_aggregate is not None:
        agg_set = set(loaded_ids) - set(exclude_from_aggregate)
    else:
        agg_set = set(loaded_ids)

    pooled_for_grand = [r for r in pooled_results if r[0] in agg_set]
    per_cond_for_grand = [r for r in per_cond_results if r[0] in agg_set]
    per_cond_tc_for_grand = [r for r in per_cond_tc_results if r[0] in agg_set]

    if not pooled_for_grand:
        print(f"[!] Aggregation set {sorted(agg_set)} matches no loaded mouse "
              f"(loaded: {sorted(loaded_ids)}). Skipping grand-average plots.")
        print("Done.")
        return

    used_ids = sorted(r[0] for r in pooled_for_grand)
    title_suffix = f"Aggregated across mice {used_ids}  (N = {len(used_ids)})"
    print(f"Grand-average aggregation: {used_ids} "
          f"(out of loaded {sorted(loaded_ids)})")

    # Filename tag: explicit > auto-generated > none
    if grand_filename_tag is not None:
        tag = f"_{grand_filename_tag}" if grand_filename_tag else ""
    elif set(used_ids) != set(loaded_ids):
        excluded = sorted(set(loaded_ids) - set(used_ids))
        tag = f"_excl_{'-'.join(str(m) for m in excluded)}" if excluded else ""
    else:
        tag = ""

    def _grand_path(name):
        base, ext = os.path.splitext(name)
        return os.path.join(output_dir, f"{base}{tag}{ext}")

    # Grand-average plots
    plot_pooled_ff_timecourse_grand(
        pooled_for_grand,
        output_path=_grand_path("ff_pooled_timecourse_grand.svg"),
        title_suffix=title_suffix)
    plot_ff_per_condition_grand(
        per_cond_for_grand,
        output_path=_grand_path("ff_per_condition_grand.svg"),
        title_suffix=title_suffix)
    plot_ff_per_condition_timecourse_marginal(
        per_cond_tc_for_grand,
        output_path=_grand_path("ff_per_condition_timecourse_marginal.svg"),
        title_suffix=title_suffix)
    plot_ff_per_condition_timecourse_facet(
        per_cond_tc_for_grand,
        output_path=_grand_path("ff_per_condition_timecourse_facet.svg"),
        title_suffix=title_suffix)

    print("Done.")


if __name__ == "__main__":
    run_ff_pipeline(mouse_ids=[0, 1, 2, 3, 4, 5], min_trials_per_cond=5)
