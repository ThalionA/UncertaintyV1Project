# -*- coding: utf-8 -*-
"""
Sample Dynamics Analyzer
Analyzes the instantaneous samples from the Sampling (SBC) architecture.
Quantifies:
1. Sharpness (Mean Instantaneous Entropy)
2. Similarity (Standard Deviation of the MAP estimate across time bins)
"""

import os
import json
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRACKER_FILE = 'experiment_tracker.json'

def set_plot_style():
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2

def calculate_sample_metrics(decoded_samp):
    """
    decoded_samp shape: (n_trials, n_angles, n_bins)
    Returns the trial-averaged Sharpness and Similarity.
    """
    # 1. Sharpness (Entropy)
    p_safe = np.clip(decoded_samp, 1e-10, 1.0)
    # Entropy along the angles axis (axis=1) -> shape: (n_trials, n_bins)
    instantaneous_entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    # Average entropy across all bins and all trials
    mean_entropy = np.nanmean(instantaneous_entropy)

    # 2. Similarity (MAP Variance over time)
    n_angles = decoded_samp.shape[1]
    angles = np.linspace(0, 90, n_angles)
    
    # Find the peak (MAP) angle for every bin -> shape: (n_trials, n_bins)
    map_indices = np.argmax(decoded_samp, axis=1)
    map_angles = angles[map_indices]
    
    # Calculate how much the MAP jumps around within each trial -> shape: (n_trials,)
    # If standard deviation is 0, the samples are perfectly similar (identical)
    trial_map_std = np.std(map_angles, axis=1)
    mean_map_std = np.nanmean(trial_map_std)
    
    return mean_entropy, mean_map_std

def extract_dynamics_data():
    """ Scans the tracker and extracts the metrics from completed .mat files """
    if not os.path.exists(TRACKER_FILE):
        raise FileNotFoundError(f"Tracker {TRACKER_FILE} not found.")
        
    with open(TRACKER_FILE, 'r') as f:
        tracker = json.load(f)
        
    data_rows = []
    
    print("Extracting Sample Dynamics from completed configurations...")
    for exp in tracker:
        if exp['status'] != 'completed':
            continue
            
        exp_id = exp['experiment_id']
        mat_file = f"population_results_config_{exp_id}.mat"
        
        if not os.path.exists(mat_file):
            continue
            
        try:
            mat = sio.loadmat(mat_file, simplify_cells=True)
            results = mat['results']
            
            # Save a row for EACH mouse, so seaborn can compute error bars across them
            for mouse_id, res in results.items():
                if 'decoded_samp' in res['Dist']['temp']:
                    samp = res['Dist']['temp']['decoded_samp']
                    ent, std = calculate_sample_metrics(samp)
                    
                    data_rows.append({
                        'Experiment_ID': exp_id,
                        'Mouse_ID': mouse_id,  # Track the mouse for grouped statistics
                        'Loss_Function': exp['training_params']['custom_loss_func'],
                        'Entropy_Lambda': exp['training_params']['entropy_lambda'],
                        'Bin_Size_ms': exp['data_params']['bin_size_ms'],
                        'Time_Window': exp['data_params']['time_window'],
                        'Sharpness_Entropy': ent,
                        'Similarity_MAP_Std': std
                    })
                
        except Exception as e:
            print(f"  [!] Error reading {mat_file}: {e}")
            
    return pd.DataFrame(data_rows)

def plot_dynamics(df):
    set_plot_style()
    
    # Filter to the "full" window just to keep the plots clean (you can change this!)
    df_plot = df[df['Time_Window'] == 'full'].copy()
    
    if df_plot.empty:
        print("[!] No 'full' window data found. Falling back to all data.")
        df_plot = df.copy()

    # Create a 2-panel figure: Top is Sharpness, Bottom is Similarity
    loss_funcs = df_plot['Loss_Function'].unique()
    n_losses = len(loss_funcs)
    
    fig, axes = plt.subplots(2, n_losses, figsize=(6 * n_losses, 10), sharey='row', sharex=True)
    
    # Handle the case where there's only 1 loss function (axes won't be 2D)
    if n_losses == 1:
        axes = np.expand_dims(axes, axis=1)

    # Standardize the palette for lambdas
    lambdas = sorted(df_plot['Entropy_Lambda'].unique())
    palette = sns.color_palette("flare", n_colors=len(lambdas))

    for col, loss in enumerate(loss_funcs):
        subset = df_plot[df_plot['Loss_Function'] == loss]
        
        # --- Row 0: Sharpness (Entropy) ---
        ax_sharp = axes[0, col]
        sns.pointplot(data=subset, x='Bin_Size_ms', y='Sharpness_Entropy', hue='Entropy_Lambda', 
                      palette=palette, ax=ax_sharp, dodge=True, markers=['o', 's', 'D', '^'],
                      errorbar='se', capsize=0.1) # Added errorbar='se' and capsize
        
        ax_sharp.set_title(f"Loss: {loss}", fontweight='bold')
        if col == 0:
            ax_sharp.set_ylabel("Sharpness (Instantaneous Entropy)\n$\\leftarrow$ Sharper    Flatter $\\rightarrow$")
        else:
            ax_sharp.set_ylabel("")
        ax_sharp.set_xlabel("")
        ax_sharp.get_legend().remove()

        # --- Row 1: Similarity (MAP Jumpiness) ---
        ax_sim = axes[1, col]
        sns.pointplot(data=subset, x='Bin_Size_ms', y='Similarity_MAP_Std', hue='Entropy_Lambda', 
                      palette=palette, ax=ax_sim, dodge=True, markers=['o', 's', 'D', '^'],
                      errorbar='se', capsize=0.1) # Added errorbar='se' and capsize
        
        if col == 0:
            ax_sim.set_ylabel("Similarity (MAP Std Dev in Degrees)\n$\\leftarrow$ More Similar    Jumps Around $\\rightarrow$")
        else:
            ax_sim.set_ylabel("")
        ax_sim.set_xlabel("Bin Size (ms)")
        
        # Keep legend only on the far right plot
        if col == n_losses - 1:
            ax_sim.legend(title='Entropy $\\lambda$', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        else:
            ax_sim.get_legend().remove()

    fig.suptitle("Sampling Dynamics: Sharpness & Similarity across Hyperparameters", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = "4_Sample_Dynamics_Analysis.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nSaved {output_path}")
    plt.show()

if __name__ == "__main__":
    df = extract_dynamics_data()
    if not df.empty:
        print(f"Successfully extracted data from {len(df)} configurations.")
        plot_dynamics(df)
    else:
        print("No completed configurations found in the tracker.")