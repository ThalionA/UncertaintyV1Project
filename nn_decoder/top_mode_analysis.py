# -*- coding: utf-8 -*-
"""Top-mode analysis — what does the dominant within-trial covariance
direction in V1 population activity actually represent?

Background. ``population_metrics_vs_uncertainty.compute_population_geometry``
computes a per-trial Top-Mode-Dominance scalar TMD = lambda_max / sum(lambda)
from the eigenvalues of the t_bins x n_neurons sample covariance. The
meeting item "what is the top PC" is asking the complementary question:
not how dominant the top mode is, but what direction it points in
neural-state space — and whether that direction is consistent across
trials or stimulus-driven.

For each (mouse, trial) we:
  1. Z-score per neuron across (trials, time) on the training-fold
     statistics (mirrors the decoder's normalisation).
  2. SVD the centered per-trial activity (t_bins x n_neurons).
  3. Extract the top right singular vector (length n_neurons) — the
     leading eigenvector of the per-trial covariance.
  4. Fix the sign: the entry with largest |value| is made positive.
     Otherwise SVD's arbitrary sign convention pollutes the cross-trial
     average.
  5. Record TMD = (s_0**2) / sum(s**2) and the top mode v_top.

Aggregates per mouse:
  - Mean top mode across trials (per stimulus condition + grand mean).
  - Cross-trial cosine similarity matrix of top modes.
  - Mean cosine similarity within stim condition vs across — diagnoses
    whether the top mode is stimulus-driven or trial-specific noise.

Plots (per mouse):
  - Bar chart of grand-mean top mode (sorted by |weight|, top 30 neurons
    annotated). Visualises the dominant direction.
  - Heatmap of top modes (trials x neurons) sorted by stim condition.
  - Scree of mean eigenvalue spectrum (across trials).
  - TMD distribution histogram.
  - Cross-trial cosine similarity histogram (within vs across stim
    condition).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from figsave import save_fig


def _set_style():
    sns.set_context('talk')
    sns.set_style('ticks')


# ----------------------------------------------------------------------
# Per-trial top mode + eigenvalue spectrum
# ----------------------------------------------------------------------

def compute_top_mode_per_trial(z_tensor: np.ndarray):
    """For each trial in ``z_tensor`` (n_trials, t_bins, n_neurons),
    return:

        top_vec : (n_trials, n_neurons) top right singular vector
                  (sign-fixed: the entry with largest |v| is positive).
        tmd     : (n_trials,) lambda_max / sum(lambda) for each trial.
        eigs    : list of length n_trials of per-trial eigenvalue arrays
                  (variable length).
    """
    n_trials, t_bins, n_neurons = z_tensor.shape
    top_vec = np.full((n_trials, n_neurons), np.nan)
    tmd = np.full(n_trials, np.nan)
    eigs = [None] * n_trials
    denom = max(t_bins - 1, 1)
    for i in range(n_trials):
        x = z_tensor[i]
        if np.any(np.isnan(x)):
            continue
        x_c = x - x.mean(axis=0, keepdims=True)
        try:
            _, s, vt = np.linalg.svd(x_c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        e = (s ** 2) / denom
        e = e[e > 0]
        if e.size == 0:
            continue
        v = vt[0]
        # Sign convention: largest-|entry| positive
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        top_vec[i] = v
        tmd[i] = e[0] / e.sum()
        eigs[i] = e
    return top_vec, tmd, eigs


def _cosine_similarity_matrix(v: np.ndarray) -> np.ndarray:
    """v: (n_trials, n_neurons). Returns (n_trials, n_trials) cosine sim."""
    mask = ~np.any(np.isnan(v), axis=1)
    norms = np.linalg.norm(v, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = v / safe[:, None]
    sim = unit @ unit.T
    sim[~mask] = np.nan
    sim[:, ~mask] = np.nan
    np.fill_diagonal(sim, np.nan)
    return sim


def plot_cross_condition_similarity(mouse_id, top_vec, conditions, out_dir):
    """Plot pairwise cosine similarity between mean top modes for each feature."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    feature_names = ['Orientation', 'Contrast', 'Dispersion']
    
    valid = ~np.any(np.isnan(top_vec), axis=1)
    top_vec_v = top_vec[valid]
    conditions_v = conditions[valid]
    
    feature_mats = []
    
    for idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        unique_vals = np.unique(conditions_v[:, idx])
        mean_modes = []
        labels = []
        for val in unique_vals:
            mask = conditions_v[:, idx] == val
            if mask.sum() > 0:
                m = np.nanmean(top_vec_v[mask], axis=0)
                n = np.linalg.norm(m)
                if n > 0:
                    m = m / n
                mean_modes.append(m)
                labels.append(f"{val:g}")
        
        if len(mean_modes) < 2:
            ax.set_visible(False)
            continue
            
        mean_modes = np.array(mean_modes)
        sim_mat = mean_modes @ mean_modes.T
        feature_mats.append((feat_name, labels, sim_mat))
        sns.heatmap(sim_mat, xticklabels=labels, yticklabels=labels, 
                    cmap='viridis', vmin=-1, vmax=1, annot=False, ax=ax, cbar=(idx==2))
        ax.set_title(f"By {feat_name}")
        ax.set_xlabel(feat_name)
        if idx == 0:
            ax.set_ylabel(feat_name)
        
    fig.suptitle(f"Mouse {mouse_id}: Cosine similarity between mean top modes")
    fig.tight_layout()
    save_fig(fig, out_dir, f'mouse{mouse_id}_cross_condition_similarity')
    return feature_mats


def plot_within_condition_stability(mouse_id, top_vec, conditions, out_dir):
    """Plot within-condition stability vs contrast and dispersion."""
    valid = ~np.any(np.isnan(top_vec), axis=1)
    top_vec_v = top_vec[valid]
    cond_v = conditions[valid]
    
    unique_c = np.unique(cond_v[:, 1])
    unique_d = np.unique(cond_v[:, 2])
    
    stability_matrix = np.full((len(unique_c), len(unique_d)), np.nan)
    
    for i, c in enumerate(unique_c):
        for j, d in enumerate(unique_d):
            mask = (cond_v[:, 1] == c) & (cond_v[:, 2] == d)
            if mask.sum() >= 2:
                vecs = top_vec_v[mask]
                sim = _cosine_similarity_matrix(vecs)
                stability_matrix[i, j] = np.nanmean(sim)
                
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, d in enumerate(unique_d):
        ax.plot(unique_c, stability_matrix[:, j], marker='o', label=f"Disp {d:g}")
        
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title(f"Mouse {mouse_id}: Top mode stability vs stimulus quality")
    ax.legend(title="Dispersion", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_dir, f'mouse{mouse_id}_within_condition_stability', layout=None)
    return unique_c, unique_d, stability_matrix


def plot_average_cross_condition_similarity(all_feature_mats, out_dir):
    """Plot the averaged cross-condition similarity across mice."""
    if not all_feature_mats: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    feature_names = ['Orientation', 'Contrast', 'Dispersion']
    
    for idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        mats = []
        labels = None
        for mouse_mats in all_feature_mats:
            for m_feat_name, m_labels, m_sim_mat in mouse_mats:
                if m_feat_name == feat_name:
                    mats.append(m_sim_mat)
                    if labels is None:
                        labels = m_labels
        
        if not mats:
            ax.set_visible(False)
            continue
            
        avg_mat = np.nanmean(np.array(mats), axis=0)
        sns.heatmap(avg_mat, xticklabels=labels, yticklabels=labels, 
                    cmap='viridis', vmin=-1, vmax=1, annot=False, ax=ax, cbar=(idx==2))
        ax.set_title(f"By {feat_name}")
        ax.set_xlabel(feat_name)
        if idx == 0:
            ax.set_ylabel(feat_name)
        
    fig.suptitle(f"Averaged Across Mice: Cosine similarity between mean top modes")
    fig.tight_layout()
    save_fig(fig, out_dir, 'average_cross_condition_similarity')


def plot_average_within_condition_stability(all_stability_data, out_dir):
    """Plot the averaged within-condition stability across mice."""
    if not all_stability_data: return
    
    all_mats = []
    unique_c = None
    unique_d = None
    
    for c, d, mat in all_stability_data:
        if unique_c is None:
            unique_c = c
            unique_d = d
        all_mats.append(mat)
        
    avg_mat = np.nanmean(np.array(all_mats), axis=0)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, d in enumerate(unique_d):
        ax.plot(unique_c, avg_mat[:, j], marker='o', label=f"Disp {d:g}")
        
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title(f"Averaged Across Mice: Top mode stability vs stimulus quality")
    ax.legend(title="Dispersion", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_dir, 'average_within_condition_stability', layout=None)


# ----------------------------------------------------------------------
# Per-mouse driver
# ----------------------------------------------------------------------

def _load_z_tensor(mouse_id):
    """Z-scored (n_trials, t_bins, n_neurons) activity for one mouse.
    Mirrors the binning the decoder uses (half window, 100 ms bins) so
    the analysis is on the same data the decoder sees."""
    from utils import load_vr_export, apply_temporal_binning
    activities_m, _, _, _, trials = load_vr_export(mouse_id)
    # Decoder uses 'half' window, 100 ms bins
    act_t = np.transpose(activities_m, (1, 2, 0))  # (n_trials, t_bins, n_neurons)
    act_b = apply_temporal_binning(act_t, time_window='half', bin_size_ms=100)
    # Z-score per neuron across (trials, time)
    mu = np.nanmean(act_b, axis=(0, 1), keepdims=True)
    sd = np.nanstd(act_b, axis=(0, 1), keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    z = (act_b - mu) / sd
    conditions = np.column_stack([
        np.asarray(trials['orientation'], dtype=float),
        np.asarray(trials['contrast'], dtype=float),
        np.asarray(trials['dispersion'], dtype=float),
    ])
    return z, conditions


def analyse_mouse(mouse_id, out_dir):
    _set_style()
    os.makedirs(out_dir, exist_ok=True)

    z, conditions = _load_z_tensor(mouse_id)
    n_trials, t_bins, n_neurons = z.shape

    top_vec, tmd, eigs = compute_top_mode_per_trial(z)
    valid = ~np.any(np.isnan(top_vec), axis=1)
    print(f"  mouse {mouse_id}: n_trials={n_trials}, n_neurons={n_neurons}, "
          f"t_bins={t_bins}, valid={valid.sum()}/{n_trials}, "
          f"mean TMD={np.nanmean(tmd):.3f}")

    # Grand-mean top mode (signed average across trials)
    grand_mean = np.nanmean(top_vec[valid], axis=0)
    # Re-normalise so we plot a unit vector
    if np.linalg.norm(grand_mean) > 0:
        grand_mean = grand_mean / np.linalg.norm(grand_mean)

    # Cross-trial cosine similarity
    sim = _cosine_similarity_matrix(top_vec)

    # Within-vs-across stim condition similarity
    cond_ids = np.zeros(n_trials, dtype=int)
    for k, c in enumerate(np.unique(conditions, axis=0)):
        cond_ids[np.all(conditions == c, axis=1)] = k
    within_sims, across_sims = [], []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            v = sim[i, j]
            if np.isnan(v):
                continue
            if cond_ids[i] == cond_ids[j]:
                within_sims.append(v)
            else:
                across_sims.append(v)
    within = np.asarray(within_sims)
    across = np.asarray(across_sims)
    print(f"    cross-trial cosine sim: within-condition mean={within.mean():.3f}, "
          f"across-condition mean={across.mean():.3f}")

    # -- Plot 1: grand-mean top mode, sorted ----------------------------
    order = np.argsort(np.abs(grand_mean))[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(np.arange(n_neurons), grand_mean[order], color='steelblue',
            edgecolor='none')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('neuron index (sorted by |loading|, descending)')
    ax.set_ylabel('top-mode loading')
    ax.set_title(
        f'Mouse {mouse_id}: grand-mean top eigenvector across '
        f'{int(valid.sum())} trials\n'
        f'(within-condition cosine sim = {within.mean():.3f}, '
        f'across-condition = {across.mean():.3f})'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, out_dir, f'mouse{mouse_id}_top_mode_grand_mean')

    # -- Plot 2: TMD distribution + scree -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(tmd[valid], bins=30, color='steelblue', edgecolor='black')
    axes[0].axvline(np.nanmean(tmd), color='red', linestyle='--',
                     label=f'mean = {np.nanmean(tmd):.3f}')
    axes[0].set_xlabel('TMD = lambda_max / sum(lambda)')
    axes[0].set_ylabel('count (trials)')
    axes[0].set_title('Top-mode dominance per trial')
    axes[0].legend()
    axes[0].grid(linestyle='--', alpha=0.4)
    axes[0].set_axisbelow(True)
    # Mean eigenvalue spectrum (truncate to the shortest trial)
    valid_eigs = [e for e in eigs if e is not None]
    if valid_eigs:
        max_k = min(20, min(len(e) for e in valid_eigs))
        spectrum = np.stack([e[:max_k] / e.sum() for e in valid_eigs])
        mean_spec = spectrum.mean(axis=0)
        cum = np.cumsum(mean_spec)
        axes[1].plot(np.arange(1, max_k + 1), mean_spec, marker='o',
                      lw=2, label='per-mode fraction')
        axes[1].plot(np.arange(1, max_k + 1), cum, marker='s',
                      lw=2, label='cumulative', color='darkorange')
        axes[1].axhline(0.95, color='red', linestyle='--', alpha=0.6,
                         linewidth=1)
        axes[1].set_xlabel('mode index')
        axes[1].set_ylabel('fraction of variance')
        axes[1].set_title('Mean eigenvalue spectrum (across trials)')
        axes[1].legend()
        axes[1].grid(linestyle='--', alpha=0.4)
        axes[1].set_axisbelow(True)
    fig.suptitle(f'Mouse {mouse_id}: top-mode spectrum')
    fig.tight_layout()
    save_fig(fig, out_dir, f'mouse{mouse_id}_tmd_spectrum')

    # -- Plot 3: within-vs-across condition cosine similarity ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 40)
    if len(within) and len(across):
        ax.hist(within, bins=bins, color='steelblue', alpha=0.65,
                label=f'within-condition (n={len(within)}, mean={within.mean():.2f})',
                density=True)
        ax.hist(across, bins=bins, color='darkorange', alpha=0.65,
                label=f'across-condition (n={len(across)}, mean={across.mean():.2f})',
                density=True)
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('cosine similarity of top modes (trial pair)')
    ax.set_ylabel('density')
    ax.set_title(
        f'Mouse {mouse_id}: top-mode similarity across trial pairs\n'
        f'(if within > across, top mode is partly stimulus-driven)'
    )
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, out_dir, f'mouse{mouse_id}_top_mode_similarity')

    # -- Plot 4: cross-condition similarity ----------------------------
    feature_mats = plot_cross_condition_similarity(mouse_id, top_vec, conditions, out_dir)

    # -- Plot 5: within-condition stability ----------------------------
    stab_data = plot_within_condition_stability(mouse_id, top_vec, conditions, out_dir)

    return dict(
        mouse=mouse_id, n_trials=n_trials, n_neurons=n_neurons,
        valid=int(valid.sum()),
        mean_tmd=float(np.nanmean(tmd)),
        within_sim=float(within.mean()) if len(within) else np.nan,
        across_sim=float(across.mean()) if len(across) else np.nan,
        grand_mean=grand_mean,
        feature_mats=feature_mats,
        stab_data=stab_data,
    )


def main(mouse_ids=range(6), out_dir=None):
    if out_dir is None:
        out_dir = os.path.normpath(os.path.join(HERE, '..', 'figures', 'top_mode'))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[top_mode] writing to {out_dir}")
    summary_rows = []
    all_feature_mats = []
    all_stab_data = []
    
    for mid in mouse_ids:
        info = analyse_mouse(mid, out_dir)
        if 'feature_mats' in info and info['feature_mats']:
            all_feature_mats.append(info['feature_mats'])
        if 'stab_data' in info and info['stab_data']:
            all_stab_data.append(info['stab_data'])
        summary_rows.append({k: v for k, v in info.items() if k not in ['grand_mean', 'feature_mats', 'stab_data']})
        
    plot_average_cross_condition_similarity(all_feature_mats, out_dir)
    plot_average_within_condition_stability(all_stab_data, out_dir)

    import pandas as pd
    summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, 'top_mode_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary written to {csv_path}")
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
