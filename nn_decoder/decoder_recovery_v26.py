# -*- coding: utf-8 -*-
"""
Decoder Model Recovery (Double Dissociation Crossover)
Automated across all 'real' configurations.
Generates Loss Bar Matrices and Hexbin Scatter Matrices to validate target recovery.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from run_experiment_v26 import run_animal_decoder

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

def get_all_real_base_configs():
    """ Scans the directory for all base configurations trained on 'real' targets """
    files = glob.glob('population_results_config_*.mat')
    real_ids = []
    for f in files:
        try:
            cid = int(os.path.basename(f).split('_')[-1].split('.')[0])
            data = sio.loadmat(f, simplify_cells=True)
            if data['config'].get('target_source') == 'real':
                real_ids.append(cid)
        except Exception:
            pass
    return sorted(real_ids)

def load_base_predictions(config_id):
    """ Loads the base config. Re-runs it quietly if full_decoded is missing. """
    file_path = f'population_results_config_{config_id}.mat'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Base config {config_id} not found.")
    
    data = sio.loadmat(file_path, simplify_cells=True)
    
    # Check if this base file has the new 'full_decoded' array
    sample_mouse = list(data['results'].keys())[0]
    if 'full_decoded' not in data['results'][sample_mouse]['Dist']['spat']:
        print(f"\n[!] Config {config_id} is missing 'full_decoded'. Re-running base to generate...")
        cfg = data['config']
        all_mice_results = {}
        for mid in [0, 1, 2, 3, 4, 5]:
            print(f"  Re-running Base Mouse {mid}...")
            all_mice_results[f"mouse_{mid}"] = run_animal_decoder(cfg, mid)
            
        sio.savemat(file_path, {'results': all_mice_results, 'config': cfg})
        data = sio.loadmat(file_path, simplify_cells=True)
        print("Base config updated successfully.\n")

    return data['results'], data['config']

def run_recovery_experiment(base_config_id):
    """ Runs (or loads cached) crossover experiments for a given base ID """
    cache_file = f'recovery_cache_base_{base_config_id}.npy'
    
    # Check if we already ran the crossover for this config to save time
    if os.path.exists(cache_file):
        print(f"\nLoading cached recovery results for Base {base_config_id}...")
        return np.load(cache_file, allow_pickle=True).item()

    results_real, base_cfg = load_base_predictions(base_config_id)
    
    # --- Extract temporal params for logging ---
    t_win = base_cfg.get('time_window', 'full')
    b_sz = base_cfg.get('bin_size_ms', 50)
    
    print(f"\nStarting Recovery Experiment using Base Config {base_config_id} ({t_win.upper()} window, {b_sz}ms bins) as Ground Truth...")
    
    mouse_ids = [int(m.split('_')[1]) for m in results_real.keys()]
    
    target_types = ['spat', 'temp']
    recovery_results = {'base_config': base_cfg}

    for t_type in target_types:
        print(f"=== CROSSOVER BRANCH: Ground Truth = Fitted {t_type.upper()} ===")
        config = base_cfg.copy()
        config['target_source'] = f'recovery_{t_type}' 
        config['base_recovery_id'] = base_config_id
        
        session_results = {}
        for mid in mouse_ids:
            print(f"  Training Crossover Models for Mouse {mid}...")
            res = run_animal_decoder(config, mid) 
            session_results[f"mouse_{mid}"] = res
            
        recovery_results[t_type] = session_results

    # Save to cache so we don't have to re-run the neural network if we just want to re-plot
    np.save(cache_file, recovery_results)
    return recovery_results

# --- Helper Math ---
def calc_kl(p, q):
    p_safe, q_safe = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

def calc_wasserstein(p, q):
    return np.sum(np.abs(np.cumsum(p, axis=1) - np.cumsum(q, axis=1)), axis=1)

# --- Plotting Functions ---

def plot_recovery_matrix(recovery_results, base_id):
    """ Generates a 2x2 Bar Matrix of Losses """
    set_style()
    metrics = ['KL', 'Wasserstein'] # Kept to 2 metrics for clean 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey='row')
    
    sources = ['spat', 'temp']
    archs = [
        ('spat', 'PPC Arch', 'darkorange'), 
        ('temp', 'SBC Arch', 'steelblue'),
        ('spat_shf', 'Shuffled Baseline', 'gray')
    ]
    
    cfg = recovery_results['base_config']
    
    # --- Extract temporal params ---
    t_win = cfg.get('time_window', 'full')
    b_sz = cfg.get('bin_size_ms', 50)
    
    for m_idx, metric_name in enumerate(metrics):
        for col, src in enumerate(sources):
            ax = axes[m_idx, col]
            
            bar_labels = [a[1] for a in archs]
            means, sems = [], []
            
            for arch_key, label, color in archs:
                vals = []
                for m_id in recovery_results[src].keys():
                    dist = recovery_results[src][m_id]['Dist'][arch_key]
                    if metric_name == 'KL':
                        vals.append(np.nanmean(calc_kl(dist['target'], dist['decoded'])))
                    else:
                        vals.append(np.nanmean(calc_wasserstein(dist['target'], dist['decoded'])))
                
                means.append(np.nanmean(vals))
                sems.append(stats.sem(vals, nan_policy='omit'))
            
            ax.bar(bar_labels, means, yerr=sems, color=[a[2] for a in archs], capsize=8, edgecolor='black')
            if col == 0: ax.set_ylabel(f"Test {metric_name}\n(Lower is Better)")
            ax.set_title(f"Target: Fitted {src.upper()}")
            ax.patches[2].set_hatch('//')

    fig.suptitle(f"Loss Recovery Matrix (Base {base_id} | Window: {t_win} | Bin: {b_sz}ms)\nLoss: {cfg['custom_loss_func']} | Lambda: {cfg['entropy_lambda']}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_dir = "Recovery_Plots"
    os.makedirs(out_dir, exist_ok=True)
    # Changed to fig.savefig for safer object-oriented exporting
    fig.savefig(os.path.join(out_dir, f"1_Loss_Matrix_Base_{base_id}.svg"), format='svg')
    plt.close(fig) # Prevent memory buildup

def plot_recovery_scatter(recovery_results, base_id):
    """ Generates a 2x2 Hexbin Density Matrix of True vs Recovered Probabilities """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    sources = ['spat', 'temp']
    archs = ['spat', 'temp']
    arch_titles = {'spat': 'Spatial (PPC) Arch', 'temp': 'Temporal (SBC) Arch'}
    
    cfg = recovery_results['base_config']
    
    # --- Extract temporal params ---
    t_win = cfg.get('time_window', 'full')
    b_sz = cfg.get('bin_size_ms', 50)

    for row, src in enumerate(sources):
        for col, arch in enumerate(archs):
            ax = axes[row, col]
            
            all_targets = []
            all_decoded = []
            
            # Pool all probabilities across all mice and trials
            for m_id in recovery_results[src].keys():
                dist = recovery_results[src][m_id]['Dist'][arch]
                all_targets.append(dist['target'].flatten())
                all_decoded.append(dist['decoded'].flatten())
                
            all_targets = np.concatenate(all_targets)
            all_decoded = np.concatenate(all_decoded)
            
            # 2D Density Hexbin (log scale colors for visibility)
            hb = ax.hexbin(all_targets, all_decoded, gridsize=50, cmap='inferno', mincnt=1, bins='log')
            
            # y = x Reference Line
            max_val = max(all_targets.max(), all_decoded.max())
            ax.plot([0, max_val], [0, max_val], color='cyan', linestyle='--', lw=2)
            
            # Calculate R-squared
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_decoded)
            r2 = r_value**2
            
            ax.text(0.05, 0.9, f"$R^2 = {r2:.3f}$", transform=ax.transAxes, 
                    fontsize=16, color='cyan', weight='bold', bbox=dict(facecolor='black', alpha=0.5))
            
            if row == 0: ax.set_title(arch_titles[arch])
            if col == 0: ax.set_ylabel(f"Ground Truth: {src.upper()}\nRecovered Probability")
            if row == 1: ax.set_xlabel("True (Target) Probability")
            
            ax.set_xlim([0, max_val])
            ax.set_ylim([0, max_val])
            
    # Add a colorbar to the side
    cb = fig.colorbar(hb, ax=axes.ravel().tolist(), pad=0.02, shrink=0.8)
    cb.set_label('Log10(Count of Bins)')

    fig.suptitle(f"Scatter Recovery Matrix (Base {base_id} | Window: {t_win} | Bin: {b_sz}ms)\nLoss: {cfg['custom_loss_func']} | Lambda: {cfg['entropy_lambda']}", fontsize=18)
    
    out_dir = "Recovery_Plots"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"2_Scatter_Matrix_Base_{base_id}.svg"), format='svg', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    # 1. Find all base configs trained on real targets
    real_base_ids = get_all_real_base_configs()
    print(f"Found {len(real_base_ids)} 'real' target configurations to run recovery on: {real_base_ids}")
    
    # 2. Loop through all of them!
    for base_id in real_base_ids:
        try:
            # Run or Load
            recov_data = run_recovery_experiment(base_id)
            
            # Plot
            print(f"Generating Recovery Plots for Base {base_id}...")
            plot_recovery_matrix(recov_data, base_id)
            plot_recovery_scatter(recov_data, base_id)
            print(f"Finished Base {base_id}.")
            
        except Exception as e:
            print(f"[!] Error processing Base Config {base_id}: {e}")
            
    print("\nAll recoveries complete! Check the ./Recovery_Plots/ directory.")