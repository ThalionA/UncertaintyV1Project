# -*- coding: utf-8 -*-
"""
Neural Population Heuristics vs. IO Uncertainty
Analyzes whether simple features of the neural population (Magnitude, Spatial Variance, Temporal Variance) 
trivially encode the uncertainty of the Ideal Observer (IO) posterior.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils_v26 import load_vr_export

def set_plot_style():
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2

def entropy_np(p):
    """ Calculates Shannon Entropy for numpy arrays (n_trials, n_bins) """
    p_safe = np.clip(p, 1e-10, 1.0)
    return -np.sum(p_safe * np.log(p_safe), axis=1)

def extract_population_heuristics(mouse_ids=[0, 1, 2, 3, 4, 5]):
    print("Extracting Neural Heuristics and IO Uncertainty...")
    
    all_data = []

    for mid in mouse_ids:
        try:
            # activities_m shape is (nTrials, nNeurons, tBins)
            activities_m, targets_perc, _, _, _ = load_vr_export(mid, 'VR_Decoder_Data_Export.mat')
            
            # 1. IO Posterior Uncertainty (Entropy)
            io_entropy = entropy_np(targets_perc)
            
            # 2. Population Magnitude (Mean firing rate across all neurons and time)
            # Shape: (nTrials,)
            pop_magnitude = np.mean(activities_m, axis=(1, 2))
            
            # 3. Spatial Variability (How varied the neurons are from each other, averaged over time)
            # Standard deviation across neurons (axis=1), then mean across time (axis=1 of the result)
            spatial_var = np.mean(np.std(activities_m, axis=1), axis=1)
            
            # 4. Temporal Variability (How jumpy neurons are over time, averaged across neurons)
            # Standard deviation across time (axis=2), then mean across neurons (axis=1 of the result)
            temporal_var = np.mean(np.std(activities_m, axis=2), axis=1)

            df_mouse = pd.DataFrame({
                'Mouse_ID': f"Mouse_{mid}",
                'IO_Uncertainty': io_entropy,
                'Pop_Magnitude': pop_magnitude,
                'Spatial_Variability': spatial_var,
                'Temporal_Variability': temporal_var
            })
            all_data.append(df_mouse)
            print(f"  --> Processed Mouse {mid} ({len(io_entropy)} trials)")
            
        except Exception as e:
            print(f"  [!] Failed to process Mouse {mid}: {e}")

    return pd.concat(all_data, ignore_index=True)

def plot_heuristics(df):
    set_plot_style()
    
    features = [
        ('Pop_Magnitude', "Population Magnitude\n(Mean Firing Rate)"),
        ('Spatial_Variability', "Spatial Variability\n(Std Dev Across Neurons)"),
        ('Temporal_Variability', "Temporal Variability\n(Std Dev Across Time)")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, (col_name, y_label) in zip(axes, features):
        # We use a hexbin/kde approach because thousands of dots get messy
        sns.kdeplot(
            data=df, x='IO_Uncertainty', y=col_name, 
            cmap="mako", fill=True, thresh=0.05, ax=ax, alpha=0.8
        )
        
        # Overlay a regression line to show the general trend
        sns.regplot(
            data=df, x='IO_Uncertainty', y=col_name, 
            scatter=False, color='darkorange', ax=ax, line_kws={'lw': 3, 'linestyle': '--'}
        )
        
        # Calculate Correlation
        r, p = stats.pearsonr(df['IO_Uncertainty'], df[col_name])
        
        # Annotate
        ax.set_title(f"Pearson r: {r:.3f} (p={p:.1e})", fontweight='bold')
        ax.set_xlabel("Ideal Observer Uncertainty (Entropy)")
        ax.set_ylabel(y_label)

    fig.suptitle("Do basic neural features trivially encode IO uncertainty?", fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    output_path = "5_Neural_Heuristics_Analysis.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nSaved {output_path}")
    plt.show()

if __name__ == "__main__":
    df = extract_population_heuristics()
    if not df.empty:
        plot_heuristics(df)
    else:
        print("[!] No data extracted.")