# -*- coding: utf-8 -*-
"""
Stimulus Ambiguity Heatmaps
Maps trial-by-trial KL divergence across the 2D space of Contrast vs. Dispersion.
Generates a 3x2 grid showing Real Data, True SBC targets, and True PPC targets 
decoded by both the SBC and PPC architectures.
"""

import os
import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. Aesthetics & Style Settings
# ==========================================
def set_plot_style():
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2

def calc_kl_posthoc(pred, target):
    """
    Diagnostic KL Divergence function.
    Computes Forward KL: D_KL(Target || Prediction)
    """
    p_safe = pred + 1e-10
    t_safe = target + 1e-10
    
    p_safe /= np.sum(p_safe, axis=1, keepdims=True)
    t_safe /= np.sum(t_safe, axis=1, keepdims=True)
    
    kl = np.sum(t_safe * np.log(t_safe / p_safe), axis=1)
    return kl

# ==========================================
# 2. Data Extraction & Averaging
# ==========================================
def build_mouse_averaged_heatmap(results_dict, arch_key):
    """ 
    Calculates the Contrast vs. Dispersion heatmap independently for each mouse, 
    then averages the resulting matrices to weight all subjects equally.
    """
    mouse_pivots = []

    for mouse_id, res in results_dict.items():
        try:
            # 1. Get raw distributions
            targets = res['Dist'][arch_key]['target']
            preds = res['Dist'][arch_key]['decoded']
            preds_shf = res['Dist'][f"{arch_key}_shf"]['decoded']
            
            # 2. Calculate trial-by-trial KL Divergences
            kl_pred = calc_kl_posthoc(preds, targets)
            kl_shf = calc_kl_posthoc(preds_shf, targets)
            
            # 3. Normalize to Shuffle 
            kl_norm = kl_pred / (kl_shf + 1e-10)
            
            # 4. Get trial stimulus parameters
            contrasts = res['trials']['contrast']
            dispersions = res['trials']['dispersion']
            
            # 5. Build the dataframe strictly for the current mouse
            df = pd.DataFrame({
                'Contrast': np.round(contrasts, 3), 
                'Dispersion': np.round(dispersions, 3),
                'KL_Loss': kl_norm
            })
            
            # 6. Create the 2D spatial matrix for this specific mouse
            pivot = df.pivot_table(values='KL_Loss', index='Dispersion', columns='Contrast', aggfunc='mean')
            mouse_pivots.append(pivot)
            
        except KeyError as e:
            print(f"  [!] Missing data for {mouse_id}: {e}")
            continue

    if not mouse_pivots:
        return None

    # 7. Stack all mouse matrices and average across the mouse dimension
    avg_pivot = pd.concat(mouse_pivots).groupby(level=0).mean()
    
    # Sort index so low dispersion (easy) is at the bottom, high (hard) at the top
    return avg_pivot.sort_index(ascending=False) 

# ==========================================
# 3. Dynamic Configuration Finder
# ==========================================
def find_config_id(tracker_path, target_params):
    """ Parses the JSON tracker to find the experiment ID matching the target parameters. """
    if not os.path.exists(tracker_path):
        raise FileNotFoundError(f"Tracker file {tracker_path} not found.")
        
    with open(tracker_path, 'r') as f:
        tracker = json.load(f)
        
    for exp in tracker:
        dp = exp.get('data_params', {})
        tp = exp.get('training_params', {})
        
        match = True
        for k, v in target_params.items():
            # Check if the param exists and matches in either dict
            if dp.get(k) != v and tp.get(k) != v:
                match = False
                break
                
        if match and exp.get('status') == 'completed':
            return exp['experiment_id']
            
    raise ValueError(f"No completed configuration found matching parameters: {target_params}")

# ==========================================
# 4. Main Plotting Routine
# ==========================================
def generate_ambiguity_heatmaps(config_id, params_suffix):
    set_plot_style()
    
    print(f"Generating Ambiguity Heatmaps for Config {config_id} ({params_suffix})...")
    
    # 1. Load Real Data
    real_file = f'population_results_config_{config_id}.mat'
    if not os.path.exists(real_file):
        raise FileNotFoundError(f"Real data {real_file} not found.")
    real_data = sio.loadmat(real_file, simplify_cells=True)['results']
    
    # 2. Load Recovery Data
    cache_file = f'recovery_cache_base_{config_id}.npy'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Recovery data {cache_file} not found.")
    recov_data = np.load(cache_file, allow_pickle=True).item()

    # 3. Setup the 3x2 Grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey=True)
    
    row_configs = [
        ("Real Data Ground Truth", real_data),
        ("True SBC Target", recov_data['temp']),
        ("True PPC Target", recov_data['spat'])
    ]
    archs = [('temp', 'SBC Decoder'), ('spat', 'PPC Decoder')]
    
    heatmap_matrices = {}

    # Extract and average all data
    for r, (row_title, res_dict) in enumerate(row_configs):
        for c, (arch_key, arch_title) in enumerate(archs):
            heatmap_matrices[(r, c)] = build_mouse_averaged_heatmap(res_dict, arch_key)

    # Plot using a fixed scale where 1.0 = Chance
    for r, (row_title, res_dict) in enumerate(row_configs):
        for c, (arch_key, arch_title) in enumerate(archs):
            ax = axes[r, c]
            pivot = heatmap_matrices[(r, c)]
            
            if pivot is not None:
                sns.heatmap(pivot, cmap="YlGnBu", ax=ax, vmin=0, vmax=1.0, cbar=False)
            else:
                ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center')
            
            if r == 0:
                ax.set_title(f"{arch_title}", fontweight='bold', pad=15)
            if c == 0:
                ax.set_ylabel(f"{row_title}\n\nDispersion (Deg)")
            else:
                ax.set_ylabel("")
                
            if r == 2:
                ax.set_xlabel("Contrast")
            else:
                ax.set_xlabel("")
                
    # --- GLOBAL COLORBAR ---
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7]) 
    
    mappable = axes[0, 0].collections[0]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Norm. KL (Fraction of Chance)', rotation=270, labelpad=20, fontweight='bold')

    fig.suptitle(f"Stimulus Ambiguity Failure Maps\nConfig {config_id} ({params_suffix})", fontsize=18, fontweight='bold', y=0.98)
     
    output_path = f"3_Ambiguity_Heatmaps_Config_{config_id}_{params_suffix}.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Saved {output_path}")

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    TRACKER_FILE = 'experiment_tracker.json'
    
    # =========================================================
    # ⚙️ USER CONFIGURATION ZONE
    # Define the exact hyperparameters to dynamically fetch the config ID.
    # =========================================================
    TARGET_PARAMS = {
        'time_window': 'half',
        'bin_size_ms': 100,
        'custom_loss_func': 'KL',
        'entropy_lambda': 0.1
    }
    # =========================================================
    
    try:
        # Resolve the specific config ID
        target_id = find_config_id(TRACKER_FILE, TARGET_PARAMS)
        
        # Build a safe string for the output file name
        param_str = f"{TARGET_PARAMS['time_window']}_{TARGET_PARAMS['bin_size_ms']}ms_{TARGET_PARAMS['custom_loss_func']}_lam{TARGET_PARAMS['entropy_lambda']}"
        
        generate_ambiguity_heatmaps(target_id, param_str)
        plt.show()
        
    except Exception as e:
        print(f"\n[!] Execution Failed: {e}")