# -*- coding: utf-8 -*-
"""
Spatial vs Temporal Delta Scatter Plot (PPC vs SBC testing)
Plots individual mice as distinct data points.
Maps Entropy Lambda to marker size.
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

# --- Divergence Calculators ---
def calc_kl(p, q, eps=1e-10):
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    axis = 1 if p.ndim >= 2 else 0
    return np.nansum(p_safe * np.log(p_safe / q_safe), axis=axis)

def calc_js(p, q, eps=1e-10):
    m = 0.5 * (p + q)
    return 0.5 * calc_kl(p, m, eps) + 0.5 * calc_kl(q, m, eps)

def calc_wasserstein(p, q):
    axis = 1 if p.ndim >= 2 else 0
    cdf_p = np.cumsum(p, axis=axis)
    cdf_q = np.cumsum(q, axis=axis)
    return np.nansum(np.abs(cdf_p - cdf_q), axis=axis)

def get_specific_divergence(target, predicted, metric_name):
    """Calculates a specific metric, handling dimension expansions."""
    if predicted.ndim == 3 and target.ndim == 2:
        target = np.expand_dims(target, axis=2)
        target = np.repeat(target, predicted.shape[2], axis=2)
        
    if metric_name == 'KL':
        return np.nanmean(calc_kl(target, predicted))
    elif metric_name == 'JS':
        return np.nanmean(calc_js(target, predicted))
    elif metric_name == 'Wasserstein':
        return np.nanmean(calc_wasserstein(target, predicted))
    else:
        raise ValueError(f"Metric {metric_name} not supported.")

# --- Data Extraction (Modified for Per-Mouse rows) ---
def load_per_mouse_dataframe(directory='.', metric_to_plot='Wasserstein'):
    search_pattern = os.path.join(directory, 'population_results_config_*.mat')
    file_list = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    rows = []
    print(f"Found {len(file_list)} completed configurations. Extracting per-mouse data...")
    
    for f in file_list:
        try:
            mat = sio.loadmat(f, simplify_cells=True)
            cfg = mat['config']
            results = mat['results']
            
            config_num = int(os.path.basename(f).split('_')[-1].split('.')[0])
            target_source = cfg.get('target_source', 'unknown')
            loss_func = cfg.get('custom_loss_func', 'unknown')
            entropy_lambda = float(cfg.get('entropy_lambda', 0.0))
            
            # Extract data for EVERY mouse independently
            for mouse_key, mouse_data in results.items():
                if 'Dist' in mouse_data:
                    dist = mouse_data['Dist']
                    
                    try:
                        # True losses
                        spat_loss = get_specific_divergence(dist['spat']['target'], dist['spat']['decoded'], metric_to_plot)
                        temp_loss = get_specific_divergence(dist['temp']['target'], dist['temp']['decoded'], metric_to_plot)
                        
                        # Shuffled baseline losses
                        spat_shf_loss = get_specific_divergence(dist['spat_shf']['target'], dist['spat_shf']['decoded'], metric_to_plot)
                        temp_shf_loss = get_specific_divergence(dist['temp_shf']['target'], dist['temp_shf']['decoded'], metric_to_plot)
                        
                        # Calculate Normalized Loss per mouse
                        norm_spatial = spat_loss / spat_shf_loss if spat_shf_loss > 0 else np.nan
                        norm_temporal = temp_loss / temp_shf_loss if temp_shf_loss > 0 else np.nan
                        
                        rows.append({
                            'Config_ID': config_num,
                            'Mouse_ID': mouse_key,  # Save the specific mouse!
                            'Target_Source': target_source,
                            'Loss_Function': loss_func,
                            'Entropy_Lambda': entropy_lambda,
                            'Norm_Spatial_Loss': norm_spatial,
                            'Norm_Temporal_Loss': norm_temporal
                        })
                    except KeyError as e:
                        print(f"Missing key in {f} for mouse {mouse_key}: {e}")
                        
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return pd.DataFrame(rows)

# --- Plotting (Modified for Size and Alpha) ---
def plot_per_mouse_scatter(df, output_dir="Landscape_Exports"):
    fig, ax = plt.subplots(figsize=(12, 10))
    df_clean = df.dropna(subset=['Norm_Spatial_Loss', 'Norm_Temporal_Loss'])
    if df_clean.empty: return
        
    # We lower alpha because there are now 6x more points, preventing solid blobs
    sns.scatterplot(
        data=df_clean,
        x='Norm_Spatial_Loss', y='Norm_Temporal_Loss',
        hue='Target_Source', 
        style='Loss_Function',
        size='Entropy_Lambda',   # MAP LAMBDA TO SIZE
        sizes=(50, 400),         # Min and Max bubble sizes
        alpha=0.6, ax=ax,
        palette={'real': 'black', 'synthetic_ppc': 'blue', 'synthetic_sbc': 'red'} 
    )
    
    min_val = min(df_clean['Norm_Spatial_Loss'].min(), df_clean['Norm_Temporal_Loss'].min()) * 0.9
    max_val = max(df_clean['Norm_Spatial_Loss'].max(), df_clean['Norm_Temporal_Loss'].max()) * 1.1
    
    # Diagonal Identity Line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Spatial = Temporal')
    
    ax.text(max_val*0.9, min_val*1.1, "Temporal Decoder Better\n(Favors SBC)", 
            color='green', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax.text(min_val*1.1, max_val*0.9, "Spatial Decoder Better\n(Favors PPC)", 
            color='purple', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    
    ax.set_title("Decoding Architecture Performance (Per Mouse):\nSpatial vs. Temporal")
    ax.set_xlabel("Normalized Spatial Decoder Loss (Lower is Better)")
    ax.set_ylabel("Normalized Temporal Decoder Loss (Lower is Better)")
    
    # Adjust legend to handle the new size variable gracefully
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_Per_Mouse_Spatial_vs_Temporal.svg"), format='svg')
    plt.close()

if __name__ == "__main__":
    set_style()
    print("Extracting per-mouse summary DataFrame...")
    
    df = load_per_mouse_dataframe(directory='.', metric_to_plot='Wasserstein') 
    
    if not df.empty:
        out_dir = "Landscape_Exports"
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Data extracted ({len(df)} mouse data points). Generating plot...")
        plot_per_mouse_scatter(df, output_dir=out_dir)
        
        print("Success! Check the export folder for '3_Per_Mouse_Spatial_vs_Temporal.svg'.")
    else:
        print("No valid data extracted. Check file structures.")