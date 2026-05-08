# -*- coding: utf-8 -*-
"""
Normalized Divergence Scatter Plot
Compares Spatial (PPC) vs Temporal (SBC) architectures normalized to shuffle.
Provides a global view of all configurations and mice simultaneously.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

def load_completed_results(directory='.'):
    search_pattern = os.path.join(directory, 'population_results_config_*.mat')
    file_list = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    loaded_data = []
    print(f"Scanning for completed configurations...")
    for f in file_list:
        try:
            mat = sio.loadmat(f, simplify_cells=True)
            loaded_data.append({
                'file': os.path.basename(f),
                'config_num': int(os.path.basename(f).split('_')[-1].split('.')[0]),
                'config': mat['config'],
                'results': mat['results']
            })
        except Exception as e:
            pass
    print(f"Found {len(loaded_data)} valid configurations.")
    return loaded_data

# --- Math Helpers ---
def calc_kl(p, q):
    p_safe, q_safe = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

def calc_wasserstein(p, q):
    return np.sum(np.abs(np.cumsum(p, axis=1) - np.cumsum(q, axis=1)), axis=1)

def calc_js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * calc_kl(p, m) + 0.5 * calc_kl(q, m)

def calc_pca_dist(p, q, pcs=None, evar=None):
    if pcs is None or (isinstance(pcs, (list, np.ndarray)) and len(pcs) == 0):
        return np.full(p.shape[0], np.nan)
    if p.ndim == 3:
        proj_p = np.einsum('nct,kc->nkt', p, pcs)
        proj_q = np.einsum('nct,kc->nkt', q, pcs)
        evar_expand = evar[np.newaxis, :, np.newaxis]
        return np.sum(evar_expand * (proj_p - proj_q)**2, axis=1) * 100
    else:
        proj_p = np.dot(p, pcs.T)
        proj_q = np.dot(q, pcs.T)
        return np.sum(evar * (proj_p - proj_q)**2, axis=1) * 100

def build_dataframe(loaded_data):
    metrics = {'KL': calc_kl, 'JS': calc_js, 'Wasserstein': calc_wasserstein, 'PCA': calc_pca_dist}
    rows = []
    
    for d in loaded_data:
        cfg = d['config']
        train_loss = cfg.get('custom_loss_func', 'KL')
        b_size = cfg.get('bin_size_ms', 50)
        window = cfg.get('time_window', 'full')
        ent_lambda = cfg.get('entropy_lambda', 0.0)  # Extract entropy lambda
        c_num = d['config_num']
        
        for m_id, m_data in d['results'].items():
            dist = m_data['Dist']
            
            for metric_name, m_func in metrics.items():
                # Evaluate true models
                if metric_name == 'PCA':
                    pcs = dist.get('pcs', None)
                    evar = dist.get('explained_var', None)
                    val_spat = np.nanmean(m_func(dist['spat']['target'], dist['spat']['decoded'], pcs, evar))
                    val_temp = np.nanmean(m_func(dist['temp']['target'], dist['temp']['decoded'], pcs, evar))
                    
                    val_spat_shf = np.nanmean(m_func(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar))
                    val_temp_shf = np.nanmean(m_func(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar))
                else:
                    val_spat = np.nanmean(m_func(dist['spat']['target'], dist['spat']['decoded']))
                    val_temp = np.nanmean(m_func(dist['temp']['target'], dist['temp']['decoded']))
                    
                    val_spat_shf = np.nanmean(m_func(dist['spat_shf']['target'], dist['spat_shf']['decoded']))
                    val_temp_shf = np.nanmean(m_func(dist['temp_shf']['target'], dist['temp_shf']['decoded']))
                
                # Normalization
                norm_spat = val_spat / val_spat_shf if val_spat_shf > 0 else np.nan
                norm_temp = val_temp / val_temp_shf if val_temp_shf > 0 else np.nan
                
                rows.append({
                    'Config': c_num,
                    'Train_Loss': train_loss,
                    'Entropy_Lambda': ent_lambda,
                    'Bin_Size': b_size,
                    'Time_Window': window,
                    'Mouse': m_id,
                    'Eval_Metric': metric_name,
                    'Spat_Norm': norm_spat,
                    'Temp_Norm': norm_temp
                })
                
    return pd.DataFrame(rows)

def plot_scatter_matrices(df):
    set_style()
    unique_losses = df['Train_Loss'].unique()
    eval_metrics = ['KL', 'JS', 'Wasserstein', 'PCA']
    
    figures = {}
    
    # We create a separate figure for configs trained on KL, JS, etc. to avoid visual clutter
    for train_loss in unique_losses:
        subset = df[df['Train_Loss'] == train_loss]
        if subset.empty: continue
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for i, eval_metric in enumerate(eval_metrics):
            ax = axes[i]
            metric_data = subset[subset['Eval_Metric'] == eval_metric]
            
            # Drop NaNs before plotting
            metric_data = metric_data.dropna(subset=['Spat_Norm', 'Temp_Norm'])
            if metric_data.empty:
                ax.set_title(f"Eval: {eval_metric} (No Data)")
                continue
            
            # The Seaborn Magic
            sns.scatterplot(
                data=metric_data,
                x='Temp_Norm',
                y='Spat_Norm',
                hue='Bin_Size',
                style='Time_Window',
                size='Entropy_Lambda',  # Map size to entropy lambda
                sizes=(50, 300),        # Scale dot sizes for clear visibility
                palette='viridis',
                alpha=0.8,
                ax=ax,
                markers={'full': 'o', 'half': 's', 'last_quarter': 'p'}
            )
            
            # Reference lines
            max_val = max(metric_data['Temp_Norm'].max(), metric_data['Spat_Norm'].max()) * 1.1
            max_val = max(max_val, 1.1) # Ensure we can see the shuffle lines
            
            # Diagonal identity line (removed label)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Shuffle thresholds (removed labels)
            ax.axvline(1.0, color='red', linestyle=':', alpha=0.5)
            ax.axhline(1.0, color='red', linestyle=':', alpha=0.5)
            
            ax.set_xlim([0, max_val])
            ax.set_ylim([0, max_val])
            
            ax.set_title(f"Evaluated with: {eval_metric}")
            ax.set_xlabel("SBC (Temporal) Normalized Divergence")
            ax.set_ylabel("PPC (Spatial) Normalized Divergence")
            
            # Clean up redundant legends (keep the first one only)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            
        fig.suptitle(f"Scatter Matrix: Spatial vs Temporal Normalised Divergence\n(Models Trained with {train_loss} Loss)", fontsize=20, y=1.02)
        fig.tight_layout()
        
        figures[train_loss] = fig
        
    return figures

if __name__ == "__main__":
    completed_data = load_completed_results()
    
    if len(completed_data) > 0:
        # 1. Build & Export DataFrame
        df = build_dataframe(completed_data)
        df.to_csv("Normalized_Scatter_Data.csv", index=False)
        print("Exported raw metrics to Normalized_Scatter_Data.csv")
        
        # 2. Plot & Export Figures
        figs = plot_scatter_matrices(df)
        output_dir = "Decoder_Results_Exports"
        os.makedirs(output_dir, exist_ok=True)
        
        for train_loss, fig in figs.items():
            filename = os.path.join(output_dir, f"Scatter_Spat_vs_Temp_Trained_{train_loss}.svg")
            fig.savefig(filename, format='svg', bbox_inches='tight')
            print(f"Saved {filename}")
            
        # 3. Explicitly tell matplotlib to display all generated figures in the IDE
        print("Displaying plots in the Spyder IDE...")
        plt.show()