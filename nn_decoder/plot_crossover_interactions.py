# -*- coding: utf-8 -*-
"""
Decoder Model Recovery: Interaction and Distribution Visualisations
Strictly filtered for 'full' and 'half' time windows.
Forces PPC to the left and SBC to the right across all axes and splits.
Losses are normalized to the shuffled baseline (Fraction of Chance).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. Aesthetics & Helper Functions
# ==========================================
def set_style():
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Consistent color mapping for the architectures
    return {'spat': '#ff7f0e', 'temp': '#1f77b4'} # PPC=Orange, SBC=Blue

def calc_kl(p, q):
    """ Forward KL Divergence: D_KL(Target || Prediction) """
    p_safe, q_safe = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)
    return np.sum(q_safe * np.log(q_safe / p_safe), axis=1)

# ==========================================
# 2. Data Aggregation
# ==========================================
def compile_recovery_dataframe(target_loss='KL'):
    """
    Scans the directory for all recovery caches, calculates mean loss metrics,
    normalizes to the shuffle baseline, and returns tidy DataFrames.
    Strictly filters for 'full' and 'half' windows.
    """
    files = glob.glob('recovery_cache_base_*.npy')
    if not files:
        raise FileNotFoundError("No recovery cache files found in the current directory.")
        
    print(f"Aggregating data from {len(files)} recovery cache files...")
    
    rows = []
    trial_rows = [] 
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True).item()
            cfg = data['base_config']
            
            # Filter for loss function
            if cfg.get('custom_loss_func', 'KL') != target_loss:
                continue
                
            t_win = cfg.get('time_window', 'full')
            
            # ENFORCE CONSTRAINT: 'full' and 'half' windows only
            if t_win not in ['full', 'half']:
                continue
                
            base_id = cfg.get('experiment_id', 'Unknown')
            b_sz = cfg.get('bin_size_ms', 50)
            lam = cfg.get('entropy_lambda', 0.0)
            
            for target_src in ['temp', 'spat']:
                if target_src not in data: continue
                
                for m_id in data[target_src].keys():
                    for dec_arch in ['temp', 'spat']:
                        dist = data[target_src][m_id]['Dist'][dec_arch]
                        
                        # Calculate raw trial-by-trial KL
                        kl_raw = calc_kl(dist['decoded'], dist['target'])
                        
                        # Load and calculate Shuffle KL for normalisation
                        try:
                            dist_shf = data[target_src][m_id]['Dist'][f"{dec_arch}_shf"]
                            kl_shf = calc_kl(dist_shf['decoded'], dist_shf['target'])
                            kl_vals = kl_raw / (kl_shf + 1e-10)
                        except KeyError:
                            print(f"  [!] Missing shuffle data for {m_id} {dec_arch}. Falling back to raw KL.")
                            kl_vals = kl_raw
                        
                        mean_kl = np.nanmean(kl_vals)
                        
                        target_label = 'True PPC' if target_src == 'spat' else 'True SBC'
                        decoder_label = 'PPC Decoder' if dec_arch == 'spat' else 'SBC Decoder'
                        
                        # Store mean data for interaction plots
                        rows.append({
                            'Base_ID': base_id,
                            'Mouse': m_id,
                            'Time_Window': t_win,
                            'Bin_Size': b_sz,
                            'Lambda': lam,
                            'Target_Source': target_label,
                            'Decoder_Arch': decoder_label,
                            'Mean_KL': mean_kl
                        })
                        
                        # Store raw trial data for specific deep-dive distributions
                        for val in kl_vals:
                            trial_rows.append({
                                'Base_ID': base_id,
                                'Time_Window': t_win,
                                'Lambda': lam,
                                'Target_Source': target_label,
                                'Decoder_Arch': decoder_label,
                                'KL_Trial': val
                            })
                            
        except Exception as e:
            print(f"  [!] Failed to parse {f}: {e}")

    df_means = pd.DataFrame(rows)
    df_trials = pd.DataFrame(trial_rows)
    
    return df_means, df_trials

# ==========================================
# 3. Visualisation Routines
# ==========================================
def plot_parameter_interactions(df, colors, target_bin=50):
    """
    Plots the double dissociation interaction lines.
    Forces PPC to the left (x-axis and legend).
    """
    if df.empty:
        print("[!] DataFrame is empty. Cannot plot interactions.")
        return

    df_sub = df[df['Bin_Size'] == target_bin]
    
    palette = {'PPC Decoder': colors['spat'], 'SBC Decoder': colors['temp']}
    
    g = sns.catplot(
        data=df_sub, 
        x='Target_Source', 
        y='Mean_KL', 
        hue='Decoder_Arch',
        col='Time_Window', 
        row='Lambda',
        kind='point', 
        palette=palette,
        order=['True PPC', 'True SBC'],         # Force PPC Left on X-Axis
        hue_order=['PPC Decoder', 'SBC Decoder'], # Force PPC First in Legend
        col_order=['full', 'half'],             # Order columns logically
        markers=['s', 'o'],
        linestyles=['--', '-'],
        capsize=0.1,
        height=4, 
        aspect=1.2,
        sharey=False 
    )
    
    g.set_axis_labels("Ground Truth Generator", "Norm. Recovery Loss (Fraction of Chance)")
    g.set_titles(col_template="Window: {col_name}", row_template="$\lambda$: {row_name}")
    
    g.fig.suptitle(f"Double Dissociation Interaction Across Parameters (Bin = {target_bin}ms)", y=1.02, fontweight='bold')
    
    out_path = "Recovery_Interaction_Sweep_Filtered.svg"
    g.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved {out_path}")

def plot_trial_distributions(df_trials, colors, target_lambda=0.1, clip_percentile=97):
    """
    Generates split-violin plots for both 'full' and 'half' windows.
    Forces PPC to the left half of the violin, SBC to the right half.
    Dynamically clips the y-axis to ignore extreme outlier tails.
    """
    # Filter for the target lambda
    df_lam = df_trials[df_trials['Lambda'] == target_lambda]
    
    if df_lam.empty:
        print(f"[!] No trial data found for Lambda={target_lambda}")
        return

    # Generate one plot per time window
    for window in ['full', 'half']:
        df_sub = df_lam[df_lam['Time_Window'] == window]
        if df_sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = {'PPC Decoder': colors['spat'], 'SBC Decoder': colors['temp']}
        
        sns.violinplot(
            data=df_sub, 
            x='Target_Source', 
            y='KL_Trial', 
            hue='Decoder_Arch',
            split=True,
            inner="quart",
            palette=palette,
            order=['True PPC', 'True SBC'],           
            hue_order=['PPC Decoder', 'SBC Decoder'], 
            ax=ax,
            cut=0,
            density_norm='width'
        )
        
        # --- NEW: Dynamic Y-Axis Clipping ---
        # Calculate the cutoff value based on the specified percentile
        y_max = np.percentile(df_sub['KL_Trial'].dropna(), clip_percentile)
        ax.set_ylim(bottom=0, top=y_max)
        # ------------------------------------
        
        ax.set_title(f"Trial-Level Loss Distribution\n(Window: {window} | $\lambda$: {target_lambda})", fontweight='bold')
        ax.set_xlabel("Ground Truth Generator")
        ax.set_ylabel(f"Trial Norm. Recovery Loss\n(Axis clipped to {clip_percentile}th percentile)")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Decoder Architecture", frameon=False, loc='upper center')
        sns.despine(ax=ax)
        
        out_path = f"Recovery_Violins_{window}_lam_{target_lambda}.svg"
        fig.savefig(out_path, format='svg', bbox_inches='tight')
        print(f"Saved {out_path} (Y-Max: {y_max:.2f})")
        plt.close(fig)
        
def plot_ecdf_distributions(df_trials, colors, target_lambda=0.1):
    """
    Plots the Cumulative Distribution Function (CDF) of the trial losses.
    A log-scaled X-axis prevents extreme outliers from compressing the visual space,
    while a reference line at x=1.0 clearly delineates better-than-chance vs worse-than-chance trials.
    """
    df_lam = df_trials[df_trials['Lambda'] == target_lambda]
    if df_lam.empty:
        print(f"[!] No trial data found for Lambda={target_lambda}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    
    windows = ['full', 'half']
    targets = ['True PPC', 'True SBC']
    palette = {'PPC Decoder': colors['spat'], 'SBC Decoder': colors['temp']}

    for r, window in enumerate(windows):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            df_sub = df_lam[(df_lam['Time_Window'] == window) & (df_lam['Target_Source'] == target)]
            
            if not df_sub.empty:
                sns.ecdfplot(
                    data=df_sub, 
                    x='KL_Trial', 
                    hue='Decoder_Arch',
                    palette=palette, 
                    hue_order=['PPC Decoder', 'SBC Decoder'],
                    ax=ax, 
                    linewidth=2.5
                )
            
            # Use log scale to handle the massive range of KL divergence
            ax.set_xscale('log')
            
            # Reference line: 1.0 represents the Shuffled Chance level
            ax.axvline(1.0, color='gray', linestyle='--', zorder=0, label='Chance Level')
            
            # Formatting and Grid Layout
            if r == 0: 
                ax.set_title(target, fontweight='bold', pad=15)
            if c == 0: 
                ax.set_ylabel(f"{window.capitalize()} Window\nCumulative Proportion")
            else: 
                ax.set_ylabel("")
                
            if r == 1: 
                ax.set_xlabel("Norm. Recovery Loss (Fraction of Chance)")
            else:
                ax.set_xlabel("")
            
            # Clean up redundant legends, keep only one in the top right
            if ax.get_legend():
                if r == 0 and c == 1:
                    sns.move_legend(ax, "lower right", frameon=True, title="Decoder")
                else:
                    ax.get_legend().remove()

    fig.suptitle(f"Empirical CDF of Trial Recovery Loss ($\lambda$: {target_lambda})", y=1.02, fontweight='bold', fontsize=18)
    sns.despine()
    plt.tight_layout()
    
    out_path = f"Recovery_ECDF_lam_{target_lambda}.svg"
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_histogram_distributions(df_trials, colors, target_lambda=0.1):
    """
    Plots overlapping step histograms of the trial losses.
    Uses a log scale to handle the unconstrained upper bounds of KL divergence.
    """
    df_lam = df_trials[df_trials['Lambda'] == target_lambda]
    if df_lam.empty: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    
    windows = ['full', 'half']
    targets = ['True PPC', 'True SBC']
    palette = {'PPC Decoder': colors['spat'], 'SBC Decoder': colors['temp']}

    for r, window in enumerate(windows):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            df_sub = df_lam[(df_lam['Time_Window'] == window) & (df_lam['Target_Source'] == target)]
            
            if not df_sub.empty:
                sns.histplot(
                    data=df_sub, 
                    x='KL_Trial', 
                    hue='Decoder_Arch',
                    palette=palette, 
                    hue_order=['PPC Decoder', 'SBC Decoder'],
                    ax=ax, 
                    element='step', 
                    fill=True, 
                    alpha=0.15,
                    common_norm=False, # Normalizes each decoder independently so areas sum to 1
                    stat='proportion',
                    log_scale=True,    # Crucial for handling outliers without dropping data
                    bins=40
                )
            
            ax.axvline(1.0, color='gray', linestyle='--', zorder=0)
            
            if r == 0: ax.set_title(target, fontweight='bold', pad=15)
            if c == 0: ax.set_ylabel(f"{window.capitalize()} Window\nProportion of Trials")
            else: ax.set_ylabel("")
            if r == 1: ax.set_xlabel("Norm. Recovery Loss (Fraction of Chance)")
            else: ax.set_xlabel("")
            
            if ax.get_legend():
                if r == 0 and c == 1:
                    sns.move_legend(ax, "upper right", frameon=True, title="Decoder")
                else:
                    ax.get_legend().remove()

    fig.suptitle(f"Distribution of Trial Recovery Loss ($\lambda$: {target_lambda})", y=1.02, fontweight='bold', fontsize=18)
    sns.despine()
    plt.tight_layout()
    
    out_path = f"Recovery_Histograms_lam_{target_lambda}.svg"
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)

# ==========================================
# 4. Execution
# ==========================================
if __name__ == "__main__":
    colors = set_style()
    
    print("Compiling datasets from cached recoveries...")
    df_means, df_trials = compile_recovery_dataframe(target_loss='KL')
    
    # Export compiled data for external statistical modeling
    if not df_means.empty:
        df_means.to_csv('recovery_means_compiled.csv', index=False)
        df_trials.to_csv('recovery_trials_compiled.csv', index=False)
        print("Exported compiled data to 'recovery_means_compiled.csv' and 'recovery_trials_compiled.csv'.")
    
    print("\nGenerating Parameter Interaction Line Plots...")
    plot_parameter_interactions(df_means, colors, target_bin=50)
    
    print("\nGenerating Trial Distribution Violin Plots...")
    plot_trial_distributions(df_trials, colors, target_lambda=0.1)
    
    print("\nGenerating ECDF Plots...")
    plot_ecdf_distributions(df_trials, colors, target_lambda=0.1)

    print("\nGenerating Histogram Plots...")
    plot_histogram_distributions(df_trials, colors, target_lambda=0.1)
    
    print("\nVisualisations complete.")