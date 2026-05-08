# -*- coding: utf-8 -*-
"""
Normal PPC Analysis — Direct Comparison

Computes a classical population vector code and directly compares its
natural statistics to the ideal observer likelihood and posterior, trial
by trial.  No free parameters are fitted.

Steps per mouse:
  1. Estimate preferred orientation of each neuron (high contrast, low disp.)
  2. Time-average neural activity → (n_trials, n_neurons)
  3. Compute population vector:
       direction θ̂  (weighted circular mean of preferred orientations)
       resultant  R  (concentration of the population vector on the half-circle)
       circular var  V = 1 − R
  4. Compare PPC statistics to IO quantities:
       - PPC mean (θ̂)  vs  IO likelihood / posterior mean
       - PPC concentration (R)  vs  IO likelihood / posterior precision
       - PPC circular variance  vs  IO likelihood / posterior variance

References:
  Ma, Beck, Latham, Pouget (2006) Nat Neurosci
  Jazayeri & Movshon (2006) Nat Neurosci
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

from utils_v26 import (
    load_vr_export,
    estimate_preferred_orientations,
    calculate_np_variance,
)


# ==========================================
# 0. Plot Style
# ==========================================

PALETTE = {
    'ppc':  '#e63946',
    'io':   '#1d3557',
    'lik':  '#457b9d',
    'post': '#2a9d8f',
    'bg':   '#fafafa',
}


def set_plot_style():
    sns.set_context("talk", font_scale=0.8)
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5,
    })


# ==========================================
# 1. Population Vector Computation
# ==========================================

def compute_population_vector(rates, pref_oris_deg):
    """
    Compute the population vector for each trial (half-circle statistics).

    Parameters
    ----------
    rates : (n_trials, n_neurons) — time-averaged firing rates
    pref_oris_deg : (n_neurons,) — preferred orientation in degrees [0, 90]

    Returns
    -------
    pv_direction : (n_trials,) — pop vector direction in degrees [0, 90]
    pv_magnitude : (n_trials,) — resultant length R ∈ [0, 1]
    pv_circ_var  : (n_trials,) — circular variance V = 1 − R
    """
    # Half-circle: multiply by 2 to span [0, 180°] → proper circular stats
    theta_rad = np.deg2rad(2.0 * pref_oris_deg)

    # Use raw rates (non-negative for weighting)
    rates_pos = np.clip(rates, 0, None)
    sum_w = np.nansum(rates_pos, axis=1, keepdims=True)
    sum_w = np.clip(sum_w, 1e-10, None)

    C = np.nansum(rates_pos * np.cos(theta_rad), axis=1) / sum_w.squeeze()
    S = np.nansum(rates_pos * np.sin(theta_rad), axis=1) / sum_w.squeeze()

    R = np.sqrt(C**2 + S**2)

    direction_rad = np.arctan2(S, C) / 2.0  # undo doubling
    pv_direction = np.mod(np.rad2deg(direction_rad), 90.0)

    return pv_direction, R, 1.0 - R


def compute_io_moments(distribution, s_grid):
    """Compute mean and variance of a distribution over s_grid."""
    denom = np.nansum(distribution, axis=1, keepdims=True) + 1e-10
    mean = np.nansum(distribution * s_grid, axis=1, keepdims=True) / denom
    var = np.nansum(distribution * (s_grid - mean)**2, axis=1) / denom.squeeze()
    return mean.squeeze(), var


# ==========================================
# 2. Pipeline
# ==========================================

def run_ppc_pipeline(mouse_ids, time_window='half'):
    """
    For each mouse: compute population vector, extract IO moments, collate.
    """
    s_grid = np.arange(0, 91, 1)
    all_results = []

    for mouse_id in mouse_ids:
        print(f"\nProcessing Mouse {mouse_id}...")
        try:
            activities_m, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mouse_id)
        except Exception as e:
            print(f"  Skipping Mouse {mouse_id}: {e}")
            continue

        # (n_neurons, n_trials, t_bins) → (n_trials, t_bins, n_neurons)
        activities = np.transpose(activities_m, (1, 2, 0))
        n_trials, t_bins, n_neurons = activities.shape

        # Window selection
        if time_window == 'half':
            act_win = activities[:, t_bins // 2:, :]
        else:
            act_win = activities

        # Time-average → (n_trials, n_neurons)
        rates = np.nanmean(act_win, axis=1)

        # Preferred orientations (from best-condition trials)
        pref_oris = estimate_preferred_orientations(rates, trials)

        # Population vector
        pv_dir, pv_R, pv_V = compute_population_vector(rates, pref_oris)
        print(f"  n={n_neurons} neurons | R range: [{pv_R.min():.3f}, {pv_R.max():.3f}]")

        # IO moments — posterior
        io_post_mean, io_post_var = compute_io_moments(targets_perc, s_grid)

        # IO moments — likelihood (if available)
        if targets_lik is not None:
            io_lik_mean, io_lik_var = compute_io_moments(targets_lik, s_grid)
        else:
            io_lik_mean = np.full(n_trials, np.nan)
            io_lik_var = np.full(n_trials, np.nan)

        # Decision entropy
        dec_ent = -np.sum(np.clip(targets_dec, 1e-10, 1.0) *
                          np.log(np.clip(targets_dec, 1e-10, 1.0)), axis=1)

        df = pd.DataFrame({
            'Mouse_ID': mouse_id,
            'Trial_Idx': np.arange(n_trials),
            'Contrast': trials['contrast'],
            'Dispersion': trials['dispersion'],
            'Orientation': trials['orientation'],
            # PPC quantities
            'PV_Direction': pv_dir,
            'PV_Resultant': pv_R,
            'PV_CircVar': pv_V,
            # IO posterior
            'IO_Post_Mean': io_post_mean,
            'IO_Post_Var': io_post_var,
            # IO likelihood
            'IO_Lik_Mean': io_lik_mean,
            'IO_Lik_Var': io_lik_var,
            # Decision
            'Decision_Entropy': dec_ent,
        })
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


# ==========================================
# 3. Visualization
# ==========================================

def _annotate(ax, x, y, color='#333'):
    r, p = stats.pearsonr(x, y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax.text(0.05, 0.95, f"r = {r:+.3f} {sig}",
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            color=color, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
    return r


def plot_mean_scatter(df, target='Posterior'):
    """PPC direction vs IO mean — per mouse."""
    set_plot_style()
    io_col = 'IO_Post_Mean' if target == 'Posterior' else 'IO_Lik_Mean'
    mouse_ids = sorted(df['Mouse_ID'].unique())
    n = len(mouse_ids)
    colors = sns.color_palette('Set2', n)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for i, mid in enumerate(mouse_ids):
        ax = axes[i]
        m = df[df['Mouse_ID'] == mid].dropna(subset=[io_col])
        ax.scatter(m[io_col], m['PV_Direction'], s=8, alpha=0.3,
                   color=colors[i], edgecolors='none', rasterized=True)
        ax.plot([0, 90], [0, 90], 'k--', lw=1, alpha=0.4)
        _annotate(ax, m[io_col].values, m['PV_Direction'].values)
        ax.set_title(f"Mouse {mid}", fontsize=10, fontweight='bold')
        ax.set_xlim(0, 90); ax.set_ylim(0, 90)
        ax.set_aspect('equal')
        ax.set_xlabel(f'IO {target} Mean (°)' if i == n // 2 else '', fontsize=9)
        ax.set_ylabel('PPC Direction (°)' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Population Vector Direction vs IO {target} Mean",
                 fontsize=13, fontweight='bold', y=1.04)
    path = f"ppc_direction_vs_{target.lower()}_mean.svg"
    plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {path}")
    plt.close()


def plot_variance_scatter(df, target='Posterior'):
    """PPC circular variance vs IO variance — per mouse."""
    set_plot_style()
    io_col = 'IO_Post_Var' if target == 'Posterior' else 'IO_Lik_Var'
    mouse_ids = sorted(df['Mouse_ID'].unique())
    n = len(mouse_ids)
    colors = sns.color_palette('Set2', n)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for i, mid in enumerate(mouse_ids):
        ax = axes[i]
        m = df[df['Mouse_ID'] == mid].dropna(subset=[io_col])
        ax.scatter(m[io_col], m['PV_CircVar'], s=8, alpha=0.3,
                   color=colors[i], edgecolors='none', rasterized=True)
        sns.regplot(x=m[io_col].values, y=m['PV_CircVar'].values,
                    scatter=False, ax=ax, line_kws={'lw': 2, 'color': '#222'})
        _annotate(ax, m[io_col].values, m['PV_CircVar'].values)
        ax.set_title(f"Mouse {mid}", fontsize=10, fontweight='bold')
        ax.set_xlabel(f'IO {target} Variance' if i == n // 2 else '', fontsize=9)
        ax.set_ylabel('PPC Circular Variance' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"PPC Circular Variance vs IO {target} Variance",
                 fontsize=13, fontweight='bold', y=1.04)
    path = f"ppc_circvar_vs_{target.lower()}_var.svg"
    plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {path}")
    plt.close()


def plot_resultant_vs_uncertainty(df):
    """PPC resultant R vs decision entropy + posterior variance — 2-row grid."""
    set_plot_style()
    mouse_ids = sorted(df['Mouse_ID'].unique())
    n = len(mouse_ids)
    colors = sns.color_palette('Set2', n)

    targets = [
        ('IO_Post_Var', 'IO Posterior Variance'),
        ('Decision_Entropy', 'Decision Entropy'),
    ]

    fig, axes = plt.subplots(len(targets), n, figsize=(3.5 * n, 3.5 * len(targets)),
                             constrained_layout=True)

    for row, (col, label) in enumerate(targets):
        for i, mid in enumerate(mouse_ids):
            ax = axes[row, i]
            m = df[df['Mouse_ID'] == mid].dropna(subset=[col])
            ax.scatter(m[col], m['PV_Resultant'], s=8, alpha=0.3,
                       color=colors[i], edgecolors='none', rasterized=True)
            sns.regplot(x=m[col].values, y=m['PV_Resultant'].values,
                        scatter=False, ax=ax, line_kws={'lw': 2, 'color': '#222'})
            _annotate(ax, m[col].values, m['PV_Resultant'].values)

            if row == 0:
                ax.set_title(f"Mouse {mid}", fontsize=10, fontweight='bold')
            ax.set_xlabel(label if row == len(targets) - 1 else '', fontsize=9)
            ax.set_ylabel('Resultant R' if i == 0 else '', fontsize=9)
            ax.tick_params(labelsize=8)

    fig.suptitle("Population Vector Resultant R vs Uncertainty",
                 fontsize=13, fontweight='bold', y=1.03)
    plt.savefig("ppc_resultant_vs_uncertainty.svg", format='svg',
                bbox_inches='tight', dpi=150)
    print("  Saved ppc_resultant_vs_uncertainty.svg")
    plt.close()


def plot_summary_panel(df):
    """
    Compact 3-column summary: Direction–Mean | CircVar–Var | Resultant–Var.
    All mice pooled, hexbin.
    """
    set_plot_style()

    panels = [
        ('IO_Lik_Mean', 'PV_Direction', 'IO Lik Mean (°)', 'PPC Direction (°)'),
        ('IO_Lik_Var', 'PV_CircVar', 'IO Lik Variance', 'PPC Circ. Variance'),
        ('IO_Post_Var', 'PV_Resultant', 'IO Post Variance', 'PPC Resultant R'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for ax, (xcol, ycol, xlab, ylab) in zip(axes, panels):
        clean = df.dropna(subset=[xcol, ycol])
        x, y = clean[xcol].values, clean[ycol].values

        ax.hexbin(x, y, gridsize=35, cmap='magma', mincnt=1,
                  linewidths=0.1, edgecolors='none')
        sns.regplot(x=x, y=y, scatter=False, ax=ax,
                    line_kws={'lw': 2, 'color': '#4cc9f0'})

        # Diagonal reference for the mean panel
        if 'Mean' in xcol and 'Direction' in ycol:
            ax.plot([0, 90], [0, 90], 'w--', lw=1.5, alpha=0.6)
            ax.set_xlim(0, 90); ax.set_ylim(0, 90)
            ax.set_aspect('equal')

        _annotate(ax, x, y, color='#4cc9f0')
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle("Normal PPC — Pooled Summary",
                 fontsize=14, fontweight='bold', y=1.04)
    plt.savefig("ppc_summary_panel.svg", format='svg', bbox_inches='tight', dpi=150)
    print("  Saved ppc_summary_panel.svg")
    plt.close()


# ==========================================
# 4. Summary Print
# ==========================================

def print_summary(df):
    print(f"\n{'═'*72}")
    print(f"  Normal PPC — Per-Mouse Correlation Summary")
    print(f"{'═'*72}")
    header = (f"  {'Mouse':>5s}  {'R(dir,lik_m)':>12s}  {'R(dir,post_m)':>13s}  "
              f"{'R(V,lik_v)':>10s}  {'R(V,post_v)':>11s}  {'R(R,dec_H)':>10s}")
    print(header)
    print(f"  {'─'*68}")

    for mid in sorted(df['Mouse_ID'].unique()):
        m = df[df['Mouse_ID'] == mid].dropna()

        def safe_r(a, b):
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 3:
                return np.nan
            return stats.pearsonr(a[valid], b[valid])[0]

        r_dir_lik  = safe_r(m['PV_Direction'].values, m['IO_Lik_Mean'].values)
        r_dir_post = safe_r(m['PV_Direction'].values, m['IO_Post_Mean'].values)
        r_v_lik    = safe_r(m['PV_CircVar'].values, m['IO_Lik_Var'].values)
        r_v_post   = safe_r(m['PV_CircVar'].values, m['IO_Post_Var'].values)
        r_R_dec    = safe_r(m['PV_Resultant'].values, m['Decision_Entropy'].values)

        print(f"  {mid:>5d}  {r_dir_lik:>+12.3f}  {r_dir_post:>+13.3f}  "
              f"{r_v_lik:>+10.3f}  {r_v_post:>+11.3f}  {r_R_dec:>+10.3f}")


# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == "__main__":
    mouse_list = [0, 1, 2, 3, 4, 5]

    for window in ['half', 'full']:
        print(f"\n{'#'*60}")
        print(f"  NORMAL PPC — {window.upper()} WINDOW")
        print(f"{'#'*60}")

        df = run_ppc_pipeline(mouse_list, time_window=window)

        if df.empty:
            print("[!] No data.")
            continue

        csv_path = f"ppc_results_{window}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved {csv_path}")

        print_summary(df)

        # Generate plots for the half window
        if window == 'half':
            plot_mean_scatter(df, target='Likelihood')
            plot_mean_scatter(df, target='Posterior')
            plot_variance_scatter(df, target='Likelihood')
            plot_variance_scatter(df, target='Posterior')
            plot_resultant_vs_uncertainty(df)
            plot_summary_panel(df)

    print("\nDone.")
