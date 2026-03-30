# -*- coding: utf-8 -*-
"""
Master Presentation Visualisation Pipeline
Generates two distinct, publication-ready figures for theoretical presentations.
"""

import os
import numpy as np
import scipy.io as sio
import scipy.stats as stats # NEW: For statistical testing
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import pandas as pd

# ==========================================
# 1. Aesthetics, Math, & Data Loading Helpers
# ==========================================
def set_plot_style():
    sns.set_context("talk", font_scale=0.8)
    sns.set_style("ticks", {'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2
    
    colors = {'SBC': '#1f77b4', 'PPC': '#ff7f0e', 'Heatmap': 'YlGnBu'}
    return colors

def load_config(config_id):
    file_path = f'population_results_config_{config_id}.mat'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    return sio.loadmat(file_path, simplify_cells=True)

def calc_kl(p, q):
    p_safe, q_safe = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

def compute_normalized_kl_raw(data, config_id):
    """ UPDATED: Now returns the raw arrays so we can run stats """
    sbc_norms, ppc_norms = [], []
    if not data or 'results' not in data: return [], []

    for mouse, res in data['results'].items():
        try:
            dist = res['Dist']
            val_temp = np.nanmean(calc_kl(dist['temp']['target'], dist['temp']['decoded']))
            val_spat = np.nanmean(calc_kl(dist['spat']['target'], dist['spat']['decoded']))
            val_temp_shf = np.nanmean(calc_kl(dist['temp_shf']['decoded'], dist['temp_shf']['target']))
            val_spat_shf = np.nanmean(calc_kl(dist['spat_shf']['decoded'], dist['spat_shf']['target']))
            
            norm_temp = val_temp / val_temp_shf if val_temp_shf > 0 else np.nan
            norm_spat = val_spat / val_spat_shf if val_spat_shf > 0 else np.nan
            
            sbc_norms.append(norm_temp)
            ppc_norms.append(norm_spat)
        except KeyError:
            continue

    return np.array(sbc_norms), np.array(ppc_norms)

def find_canonical_trials(targets, angles):
    peaks = angles[np.argmax(targets, axis=1)]
    entropies = -np.sum(targets * np.log(targets + 1e-10), axis=1)
    
    mask_0 = peaks < 15
    idx_0 = np.where(mask_0)[0][np.argmin(entropies[mask_0])] if np.any(mask_0) else 0
    
    mask_90 = peaks > 75
    idx_90 = np.where(mask_90)[0][np.argmin(entropies[mask_90])] if np.any(mask_90) else 1
    
    mask_45 = (peaks > 35) & (peaks < 55)
    idx_ambig = np.where(mask_45)[0][np.argmax(entropies[mask_45])] if np.any(mask_45) else np.argmax(entropies)
    
    return {'Confident Go (0°)': idx_0, 'Confident No-Go (90°)': idx_90, 'Ambiguous (~45°)': idx_ambig}

def load_recovery_matrix(base_id):
    cache_file = f'recovery_cache_base_{base_id}.npy'
    if not os.path.exists(cache_file):
        print(f"  [!] No recovery cache found at {cache_file}. Using placeholder NaNs.")
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])

    recov_data = np.load(cache_file, allow_pickle=True).item()

    def get_mean_kl(target_src, decoder_arch):
        vals = []
        for m_id in recov_data[target_src].keys():
            dist = recov_data[target_src][m_id]['Dist'][decoder_arch]
            vals.append(np.nanmean(calc_kl(dist['target'], dist['decoded'])))
        return np.nanmean(vals)

    return np.array([
        [get_mean_kl('temp', 'temp'), get_mean_kl('temp', 'spat')],
        [get_mean_kl('spat', 'temp'), get_mean_kl('spat', 'spat')]
    ])

def build_rm_anova_dataframe(configs, data_dict):
    """ 
    Extracts KL metrics into a long-form DataFrame for Repeated Measures ANOVA.
    Ensures the design is perfectly balanced across all subjects.
    """
    rows = []
    for bin_sz, cid in configs.items():
        if not data_dict[bin_sz] or 'results' not in data_dict[bin_sz]: continue
        
        for mouse_id, res in data_dict[bin_sz]['results'].items():
            try:
                dist = res['Dist']
                val_temp = np.nanmean(calc_kl(dist['temp']['decoded'], dist['temp']['target']))
                val_spat = np.nanmean(calc_kl(dist['spat']['decoded'], dist['spat']['target']))
                val_temp_shf = np.nanmean(calc_kl(dist['temp_shf']['decoded'], dist['temp_shf']['target']))
                val_spat_shf = np.nanmean(calc_kl(dist['spat_shf']['decoded'], dist['spat_shf']['target']))
                
                norm_temp = val_temp / val_temp_shf if val_temp_shf > 0 else np.nan
                norm_spat = val_spat / val_spat_shf if val_spat_shf > 0 else np.nan
                
                if not np.isnan(norm_temp):
                    rows.append({'Mouse': mouse_id, 'Bin_Size': bin_sz, 'Model': 'SBC', 'KL': norm_temp})
                if not np.isnan(norm_spat):
                    rows.append({'Mouse': mouse_id, 'Bin_Size': bin_sz, 'Model': 'PPC', 'KL': norm_spat})
            except KeyError:
                continue
                
    df = pd.DataFrame(rows)
    
    # RM ANOVA requires perfectly balanced data. Drop mice missing any condition.
    if not df.empty:
        expected_conditions = len(configs) * 2 # Bins * Models
        mouse_counts = df.groupby('Mouse').size()
        valid_mice = mouse_counts[mouse_counts == expected_conditions].index
        df = df[df['Mouse'].isin(valid_mice)]
        
    return df

# ==========================================
# 2. Figure 1: The Macro Framework
# ==========================================
def generate_macro_figure(configs, data_dict, colors, mouse_id='mouse_0', base_bin=50, recovery_matrix=None, suffix=""):
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], wspace=0.3, hspace=0.4)

    # --- Panel A: Sweep with 2-Way RM ANOVA & Post-Hocs ---
    axA = fig.add_subplot(gs[0, 0])
    bins = list(configs.keys())
    
    # 1. Build Data and Run ANOVA
    df_stats = build_rm_anova_dataframe(configs, data_dict)
    
    if not df_stats.empty and df_stats['Mouse'].nunique() > 1:
        print("\n" + "="*50)
        print("  STATISTICAL ANALYSIS: 2-Way RM ANOVA")
        print("="*50)
        try:
            anova = AnovaRM(data=df_stats, depvar='KL', subject='Mouse', within=['Bin_Size', 'Model']).fit()
            print(anova.summary())
        except Exception as e:
            print(f"[!] ANOVA Failed: {e}")
            
        # 2. Run FDR-Corrected Post-Hocs (Simple Main Effects of Model at each Bin Size)
        p_vals_raw = []
        valid_bins = []
        
        for b in bins:
            sbc_vals = df_stats[(df_stats['Bin_Size']==b) & (df_stats['Model']=='SBC')].sort_values('Mouse')['KL'].values
            ppc_vals = df_stats[(df_stats['Bin_Size']==b) & (df_stats['Model']=='PPC')].sort_values('Mouse')['KL'].values
            
            if len(sbc_vals) > 1 and len(sbc_vals) == len(ppc_vals):
                _, p = stats.ttest_rel(sbc_vals, ppc_vals)
                p_vals_raw.append(p)
                valid_bins.append(b)
                
        # Apply Benjamini-Hochberg FDR correction
        if len(p_vals_raw) > 0:
            _, pvals_corrected, _, _ = multipletests(p_vals_raw, alpha=0.05, method='fdr_bh')
            sig_dict = dict(zip(valid_bins, pvals_corrected))
            print("\nPost-Hoc Paired t-tests (SBC vs PPC, FDR Corrected):")
            for b, p in sig_dict.items():
                print(f"  Bin {b}ms: p = {p:.4f}")
        else:
            sig_dict = {}
    else:
        print("[!] Not enough balanced data to run RM ANOVA.")
        sig_dict = {}

    # 3. Plotting the Sweep
    sbc_means, sbc_sems, ppc_means, ppc_sems = [], [], [], []
    for b in bins:
        s_vals = df_stats[(df_stats['Bin_Size']==b) & (df_stats['Model']=='SBC')]['KL'].values
        p_vals = df_stats[(df_stats['Bin_Size']==b) & (df_stats['Model']=='PPC')]['KL'].values
        
        if len(s_vals) > 0 and len(p_vals) > 0:
            sbc_means.append(np.mean(s_vals)); sbc_sems.append(np.std(s_vals)/np.sqrt(len(s_vals)))
            ppc_means.append(np.mean(p_vals)); ppc_sems.append(np.std(p_vals)/np.sqrt(len(p_vals)))
            
            # Add significance markers based on our FDR corrected post-hocs
            if b in sig_dict:
                pval = sig_dict[b]
                sig_y = max(np.mean(s_vals), np.mean(p_vals)) + max(np.std(s_vals)/np.sqrt(len(s_vals)), np.std(p_vals)/np.sqrt(len(p_vals))) + 0.05
                if pval < 0.001: axA.text(b, sig_y, '***', ha='center', fontweight='bold', fontsize=14)
                elif pval < 0.01: axA.text(b, sig_y, '**', ha='center', fontweight='bold', fontsize=14)
                elif pval < 0.05: axA.text(b, sig_y, '*', ha='center', fontweight='bold', fontsize=14)
        else:
            sbc_means.append(np.nan); sbc_sems.append(np.nan)
            ppc_means.append(np.nan); ppc_sems.append(np.nan)

    axA.errorbar(bins, sbc_means, yerr=sbc_sems, label='Temporal (SBC)', color=colors['SBC'], marker='o', lw=2.5, markersize=8)
    axA.errorbar(bins, ppc_means, yerr=ppc_sems, label='Spatial (PPC)', color=colors['PPC'], marker='s', lw=2.5, markersize=8)
    axA.axhline(1.0, color='grey', linestyle='--', lw=1.5, zorder=0)
    axA.set_title("A. Temporal Integration Sweep", fontweight='bold', loc='left')
    axA.set_xlabel("Temporal Bin Size (ms)"); axA.set_ylabel("Normalized KL Divergence")
    axA.set_xticks(bins); axA.legend(frameon=False)

    # --- Panel B & C: Example Ambiguous Trial ---
    axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    if data_dict[base_bin]:
        try:
            dist_temp = data_dict[base_bin]['results'][mouse_id]['Dist']['temp']
            dist_spat = data_dict[base_bin]['results'][mouse_id]['Dist']['spat']
            
            n_angles = dist_temp['target'].shape[1]
            angles = np.linspace(0, 90, n_angles)
            ambig_idx = find_canonical_trials(dist_temp['target'], angles)['Ambiguous (~45°)']
            
            real_instant_samples = dist_temp['decoded_samp'][ambig_idx]
            n_bins = real_instant_samples.shape[1]
            max_time_s = (n_bins * base_bin) / 1000.0 
            
            # B: Heatmap (UPDATED WITH MAP OVERLAY)
            im = axB.imshow(real_instant_samples, extent=[0, max_time_s, 0, 90], origin='lower', aspect='auto', cmap=colors['Heatmap'])
            
            # Plot the MAP trajectory
            map_indices = np.argmax(real_instant_samples, axis=0)
            map_angles = angles[map_indices]
            time_vector = np.linspace(0 + (max_time_s/n_bins)/2, max_time_s - (max_time_s/n_bins)/2, n_bins)
            axB.plot(time_vector, map_angles, color='red', linestyle='--', lw=2, alpha=0.8, label="MAP Estimate")
            axB.legend(loc='upper right', frameon=False, fontsize=8, labelcolor='white')
            
            axB.set_title(f"B. SBC Samples ({mouse_id}, {base_bin}ms)", fontweight='bold', loc='left')
            axB.set_xlabel("Time (s)"); axB.set_ylabel("Decoded Orientation")
            axB.set_yticks([0, 45, 90]); axB.set_yticklabels(['0$^\circ$', '45$^\circ$', '90$^\circ$'])
            axB.set_xticks([0, max_time_s/2, max_time_s])
            
            # C: Aggregation
            for i in range(n_bins):
                axC.plot(angles, real_instant_samples[:, i], color=colors['SBC'], alpha=0.15, lw=1)
            axC.plot(angles, dist_temp['decoded'][ambig_idx], color=colors['SBC'], lw=3, label='SBC')
            axC.plot(angles, dist_spat['decoded'][ambig_idx], color=colors['PPC'], lw=3, label='PPC')
            axC.plot(angles, dist_temp['target'][ambig_idx], 'k--', lw=2, label='Target')
            
            axC.set_title("C. Trial-Level Aggregation", fontweight='bold', loc='left')
            axC.set_xlabel("Orientation"); axC.set_xticks([0, 45, 90])
            axC.legend(frameon=False, loc='upper right'); sns.despine(ax=axC, left=True); axC.set_yticks([])
        except KeyError as e:
            print(f"  [!] Missing data for {mouse_id} in Panel B/C: {e}")

    # --- Panel D: Recovery Matrix ---
    axD = fig.add_subplot(gs[1, 1])
    vmax = np.nanmax(recovery_matrix) if not np.isnan(recovery_matrix).all() else 1.0
    sns.heatmap(recovery_matrix, annot=True, cmap="Reds", fmt=".3f", cbar_kws={'label': 'Recovery Loss (KL)'}, ax=axD, vmin=0, vmax=vmax)
    axD.set_title("D. Cross-Architecture Recovery", fontweight='bold', loc='left')
    axD.set_xticklabels(['SBC Decoder', 'PPC Decoder'])
    axD.set_yticklabels(['Target:\nTrue SBC', 'Target:\nTrue PPC'], rotation=0, va='center')

    output_path = f"1_Macro_Framework_Summary_{suffix}.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Saved {output_path}")

# ==========================================
# 3. Figure 2: The Micro Mechanics Grid
# ==========================================
def generate_micro_figure(configs, data_dict, colors, mouse_id='mouse_0', base_bin=50, suffix=""):
    if not data_dict[base_bin]: return
    
    try:
        dist_base = data_dict[base_bin]['results'][mouse_id]['Dist']
    except KeyError:
        print(f"  [!] Missing data for {mouse_id} in Micro Figure. Skipping.")
        return
        
    targets = dist_base['temp']['target']
    n_angles = targets.shape[1]
    angles = np.linspace(0, 90, n_angles)
    trial_indices = find_canonical_trials(targets, angles)
    
    fig, axes = plt.subplots(3, 4, figsize=(22, 12), gridspec_kw={'width_ratios': [1.5, 1, 1, 1]})
    
    for row_idx, (trial_name, t_idx) in enumerate(trial_indices.items()):
        
        # Calculate KL for annotation (UPDATED)
        kl_sbc = calc_kl(np.expand_dims(dist_base['temp']['decoded'][t_idx], 0), np.expand_dims(targets[t_idx], 0))[0]
        kl_ppc = calc_kl(np.expand_dims(dist_base['spat']['decoded'][t_idx], 0), np.expand_dims(targets[t_idx], 0))[0]

        # --- Column 0: Aggregation ---
        ax_agg = axes[row_idx, 0]
        ax_agg.plot(angles, targets[t_idx], 'k--', lw=2.5, label='Target')
        ax_agg.plot(angles, dist_base['temp']['decoded'][t_idx], color=colors['SBC'], lw=3, label='SBC')
        ax_agg.plot(angles, dist_base['spat']['decoded'][t_idx], color=colors['PPC'], lw=3, label='PPC')
        
        # Add KL annotation text (NEW)
        ax_agg.text(0.95, 0.95, f"KL$_{{SBC}}$: {kl_sbc:.3f}\nKL$_{{PPC}}$: {kl_ppc:.3f}", 
                    transform=ax_agg.transAxes, ha='right', va='top', fontsize=11, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax_agg.set_ylabel(f"{trial_name}\n\nProbability")
        ax_agg.set_xticks([0, 45, 90]); ax_agg.set_xlim([0, 90])
        sns.despine(ax=ax_agg)
        if row_idx == 0: ax_agg.set_title(f"Trial Aggregation ({mouse_id})"); ax_agg.legend(frameon=False)
        if row_idx == 2: ax_agg.set_xlabel("Orientation")

        # --- Columns 1-3: Heatmaps (UPDATED WITH MAP OVERLAY) ---
        for col_idx, (bin_sz, cid) in enumerate(configs.items(), start=1):
            ax_heat = axes[row_idx, col_idx]
            if not data_dict[bin_sz]: continue
            
            try:
                raw_samples = data_dict[bin_sz]['results'][mouse_id]['Dist']['temp']['decoded_samp'][t_idx]
                n_bins = raw_samples.shape[1]
                max_time_s = (n_bins * bin_sz) / 1000.0
                
                ax_heat.imshow(raw_samples, extent=[0, max_time_s, 0, 90], origin='lower', aspect='auto', cmap=colors['Heatmap'])
                
                # Plot MAP trajectory
                map_indices = np.argmax(raw_samples, axis=0)
                map_angles = angles[map_indices]
                time_vector = np.linspace(0 + (max_time_s/n_bins)/2, max_time_s - (max_time_s/n_bins)/2, n_bins)
                ax_heat.plot(time_vector, map_angles, color='red', linestyle='--', lw=2, alpha=0.7)

                ax_heat.set_yticks([0, 45, 90]); ax_heat.set_xticks([0, max_time_s/2, max_time_s])
                
                if row_idx == 0: ax_heat.set_title(f"SBC Samples ({bin_sz}ms)")
                if row_idx == 2: ax_heat.set_xlabel("Time (s)")
                if col_idx > 1: ax_heat.set_yticklabels([])
            except KeyError:
                pass 
                
    fig.suptitle(f"SBC Sampling Dynamics Across Stimulus Ambiguity & Temporal Integration Windows", fontsize=20, y=1.02)
    plt.tight_layout()
    
    output_path = f"2_Micro_Trial_Mechanics_{suffix}.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    colors = set_plot_style()
    
    TARGET_MOUSE = 'mouse_0' 
    BASE_BIN = 50 
    
    # Define the specific hyperparameters to sweep
    TARGET_LOSS = 'KL'
    WINDOWS_TO_EVALUATE = ['full', 'half', 'last_quarter']
    LAMBDAS_TO_EVALUATE = [0.05, 0.1, 0.2, 0.5]
    
    # Load the tracker to dynamically find the correct experiment IDs
    tracker_path = 'experiment_tracker.json'
    if not os.path.exists(tracker_path):
        raise FileNotFoundError(f"Could not find {tracker_path}. Run from the correct directory.")
        
    with open(tracker_path, 'r') as f:
        tracker = json.load(f)

    # Nested loop to sweep both Time Windows and Entropy Lambdas
    for window in WINDOWS_TO_EVALUATE:
        for lam in LAMBDAS_TO_EVALUATE:
            print(f"\n{'='*60}")
            print(f" PROCESSING: Window = {window.upper()} | Lambda = {lam}")
            print(f"{'='*60}")
            
            # 1. Dynamically extract configs for the current parameters
            ACTIVE_CONFIGS = {}
            for exp in tracker:
                dp = exp.get('data_params', {})
                tp = exp.get('training_params', {})
                
                if (exp.get('status') == 'completed' and 
                    dp.get('time_window') == window and 
                    tp.get('custom_loss_func') == TARGET_LOSS and 
                    tp.get('entropy_lambda') == lam):
                    
                    ACTIVE_CONFIGS[dp['bin_size_ms']] = exp['experiment_id']
                    
            # Sort by bin size for ordered plotting in the grid (50, 100, 250)
            ACTIVE_CONFIGS = dict(sorted(ACTIVE_CONFIGS.items()))
            
            if not ACTIVE_CONFIGS:
                print(f"  [!] No matching completed configs found for '{window}' & Lambda {lam}. Skipping.")
                continue
                
            # Update suffix to prevent overwriting files across the 12 generated sets
            FILE_SUFFIX = f"{TARGET_MOUSE}_{window}_window_lambda_{lam}"
            
            print(f"Loading data configurations: {ACTIVE_CONFIGS} ...")
            data_dict = {b: load_config(cid) for b, cid in ACTIVE_CONFIGS.items()}
            
            if BASE_BIN not in ACTIVE_CONFIGS:
                print(f"  [!] Base bin {BASE_BIN}ms missing. Cannot build heatmaps. Skipping.")
                continue
                
            base_config_id = ACTIVE_CONFIGS[BASE_BIN]
            real_recovery_matrix = load_recovery_matrix(base_config_id)
            
            print(f"\nGenerating Figure 1...")
            generate_macro_figure(configs=ACTIVE_CONFIGS, data_dict=data_dict, colors=colors, 
                                  mouse_id=TARGET_MOUSE, base_bin=BASE_BIN, 
                                  recovery_matrix=real_recovery_matrix, suffix=FILE_SUFFIX)
            
            print(f"\nGenerating Figure 2...")
            generate_micro_figure(configs=ACTIVE_CONFIGS, data_dict=data_dict, colors=colors, 
                                  mouse_id=TARGET_MOUSE, base_bin=BASE_BIN, suffix=FILE_SUFFIX)

    print("\nAll target windows and lambdas processed.")
    plt.show()