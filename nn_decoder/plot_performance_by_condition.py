# -*- coding: utf-8 -*-
"""
Evaluates existing decoder performance across different stimulus conditions
(Contrast and Dispersion) on a trial-by-trial basis.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

# --- Trial-by-Trial Divergence Calculator ---
def calc_wasserstein_trial(p, q):
    """
    Calculates Wasserstein distance per trial.
    Assumes inputs are shape (N_trials, N_categories).
    Returns an array of shape (N_trials,).
    """
    # Ensure no zeros for stability if needed, but Wasserstein just needs cumulative sum
    cdf_p = np.cumsum(p, axis=1)
    cdf_q = np.cumsum(q, axis=1)
    return np.nansum(np.abs(cdf_p - cdf_q), axis=1)

def extract_trial_data(directory='.', metric='Wasserstein'):
    search_pattern = os.path.join(directory, 'population_results_config_*.mat')
    file_list = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    rows = []
    print(f"Found {len(file_list)} configurations. Extracting trial-by-trial data for 'real' targets...")
    
    for f in file_list:
        try:
            mat = sio.loadmat(f, simplify_cells=True)
            cfg = mat['config']
            
            # ONLY analyze real neural data for stimulus condition effects
            if cfg.get('target_source', 'unknown') != 'real':
                continue
                
            config_num = int(os.path.basename(f).split('_')[-1].split('.')[0])
            results = mat['results']
            
            for mouse_key, mouse_data in results.items():
                if 'Dist' in mouse_data and 'trials' in mouse_data:
                    dist = mouse_data['Dist']
                    trials = mouse_data['trials']
                    
                    # Flatten trial conditions to ensure 1D arrays
                    contrasts = np.ravel(trials['contrast'])
                    dispersions = np.ravel(trials['dispersion'])
                    
                    # Calculate raw trial-by-trial losses
                    spat_loss = calc_wasserstein_trial(dist['spat']['target'], dist['spat']['decoded'])
                    temp_loss = calc_wasserstein_trial(dist['temp']['target'], dist['temp']['decoded'])
                    
                    # Calculate shuffled trial-by-trial losses
                    spat_shf_loss = calc_wasserstein_trial(dist['spat_shf']['target'], dist['spat_shf']['decoded'])
                    temp_shf_loss = calc_wasserstein_trial(dist['temp_shf']['target'], dist['temp_shf']['decoded'])
                    
                    # Loop through each trial and store it
                    for i in range(len(contrasts)):
                        # Safely calculate normalized loss per trial (avoid div by zero)
                        norm_spat = spat_loss[i] / spat_shf_loss[i] if spat_shf_loss[i] > 1e-5 else np.nan
                        norm_temp = temp_loss[i] / temp_shf_loss[i] if temp_shf_loss[i] > 1e-5 else np.nan
                        
                        rows.append({
                            'Config_ID': config_num,
                            'Mouse_ID': mouse_key,
                            'Contrast': np.round(contrasts[i], 2), # Rounding to group similar conditions
                            'Dispersion': np.round(dispersions[i], 2),
                            'Norm_Spatial_Loss': norm_spat,
                            'Norm_Temporal_Loss': norm_temp
                        })
                        
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    df = pd.DataFrame(rows)
    return df

def plot_condition_performance(df, output_dir="Landscape_Exports"):
    if df.empty:
        print("No valid 'real' target data found.")
        return
        
    # Melt the dataframe so 'Architecture' (Spatial/Temporal) is a categorical column
    # This makes plotting side-by-side much easier in seaborn
    df_melted = df.melt(
        id_vars=['Config_ID', 'Mouse_ID', 'Contrast', 'Dispersion'],
        value_vars=['Norm_Spatial_Loss', 'Norm_Temporal_Loss'],
        var_name='Architecture',
        value_name='Normalized_Loss'
    )
    
    # Rename for cleaner plot legends
    df_melted['Architecture'] = df_melted['Architecture'].replace({
        'Norm_Spatial_Loss': 'Spatial (PPC)',
        'Norm_Temporal_Loss': 'Temporal (SBC)'
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    palette = {'Spatial (PPC)': 'blue', 'Temporal (SBC)': 'red'}
    
    # Plot 1: Contrast
    sns.pointplot(
        data=df_melted, x='Contrast', y='Normalized_Loss', 
        hue='Architecture', ax=axes[0], dodge=True, 
        markers=['o', 's'], capsize=.1, errwidth=2, palette=palette
    )
    axes[0].set_title("Decoder Performance vs. Stimulus Contrast")
    axes[0].set_ylabel("Normalized Loss (< 1.0 is better)")
    axes[0].set_xlabel("Contrast Level")
    axes[0].axhline(1.0, color='k', linestyle='--', alpha=0.5) # Baseline
    
    # Plot 2: Dispersion
    sns.pointplot(
        data=df_melted, x='Dispersion', y='Normalized_Loss', 
        hue='Architecture', ax=axes[1], dodge=True, 
        markers=['o', 's'], capsize=.1, errwidth=2, palette=palette
    )
    axes[1].set_title("Decoder Performance vs. Stimulus Dispersion")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Dispersion Level")
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5) # Baseline
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "4_Performance_by_Condition.svg"), format='svg')
    plt.close()

if __name__ == "__main__":
    set_style()
    df_trials = extract_trial_data(directory='.', metric='Wasserstein')
    
    if not df_trials.empty:
        print(f"Extracted {len(df_trials)} total trials across all real configs.")
        plot_condition_performance(df_trials)
        print("Success! Check 'Landscape_Exports/4_Performance_by_Condition.svg'.")
    else:
        print("Data extraction failed or no real data configs found.")