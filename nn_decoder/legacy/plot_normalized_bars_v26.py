# -*- coding: utf-8 -*-
"""
Normalized Divergence Bar Charts
Plots Spatial (PPC) vs Temporal (SBC) architectures normalized to shuffle.
Averages across animals (+SEM). Generates one figure per Evaluation Metric.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats
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

def plot_comprehensive_bar_charts(loaded_data):
    set_style()
    eval_metrics = {'KL': calc_kl, 'JS': calc_js, 'Wasserstein': calc_wasserstein, 'PCA': calc_pca_dist}
    
    # Identify unique training loss functions to create subplots
    train_losses = sorted(list(set([d['config'].get('custom_loss_func', 'KL') for d in loaded_data])))
    
    figures = {}
    
    # Generate ONE figure per Evaluation Metric
    for eval_name, m_func in eval_metrics.items():
        
        # Create subplots based on the number of training losses (e.g., 3 subplots for KL, JS, PCA)
        fig, axes = plt.subplots(len(train_losses), 1, figsize=(20, 6 * len(train_losses)), sharey=True)
        if len(train_losses) == 1: axes = [axes]
        
        for ax_idx, train_loss in enumerate(train_losses):
            ax = axes[ax_idx]
            
            # Filter configurations trained with this specific loss function
            subset = [d for d in loaded_data if d['config'].get('custom_loss_func') == train_loss]
            
            # Sort them logically for the X-axis: Window -> Bin Size -> Lambda
            subset = sorted(subset, key=lambda x: (
                x['config'].get('time_window', 'full'),
                x['config'].get('bin_size_ms', 50),
                x['config'].get('entropy_lambda', 0.1)
            ))
            
            x_labels = []
            spat_means, spat_sems = [], []
            temp_means, temp_sems = [], []
            
            for d in subset:
                cfg = d['config']
                win = cfg.get('time_window', 'full')
                bsz = cfg.get('bin_size_ms', 50)
                lam = cfg.get('entropy_lambda', 0.1)
                
                # Construct legible X-axis label
                x_labels.append(f"{win.upper()}\n{bsz}ms\nλ={lam}")
                
                m_spat_norms, m_temp_norms = [], []
                
                # Iterate over mice to get the animal-level average
                for m_id, m_data in d['results'].items():
                    dist = m_data['Dist']
                    
                    if eval_name == 'PCA':
                        pcs = dist.get('pcs', None)
                        evar = dist.get('explained_var', None)
                        v_spat = np.nanmean(m_func(dist['spat']['target'], dist['spat']['decoded'], pcs, evar))
                        v_temp = np.nanmean(m_func(dist['temp']['target'], dist['temp']['decoded'], pcs, evar))
                        v_spat_shf = np.nanmean(m_func(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar))
                        v_temp_shf = np.nanmean(m_func(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar))
                    else:
                        v_spat = np.nanmean(m_func(dist['spat']['target'], dist['spat']['decoded']))
                        v_temp = np.nanmean(m_func(dist['temp']['target'], dist['temp']['decoded']))
                        v_spat_shf = np.nanmean(m_func(dist['spat_shf']['target'], dist['spat_shf']['decoded']))
                        v_temp_shf = np.nanmean(m_func(dist['temp_shf']['target'], dist['temp_shf']['decoded']))
                    
                    # Normalize by shuffle (Lower is better, < 1.0 means it beat shuffle)
                    if v_spat_shf > 0: m_spat_norms.append(v_spat / v_spat_shf)
                    if v_temp_shf > 0: m_temp_norms.append(v_temp / v_temp_shf)
                
                spat_means.append(np.nanmean(m_spat_norms))
                spat_sems.append(stats.sem(m_spat_norms, nan_policy='omit'))
                temp_means.append(np.nanmean(m_temp_norms))
                temp_sems.append(stats.sem(m_temp_norms, nan_policy='omit'))
            
            # --- Plotting the Bars ---
            x = np.arange(len(x_labels))
            width = 0.35
            
            ax.bar(x - width/2, spat_means, width, yerr=spat_sems, label='Spatial (PPC)', color='darkorange', edgecolor='black', capsize=4)
            ax.bar(x + width/2, temp_means, width, yerr=temp_sems, label='Temporal (SBC)', color='steelblue', edgecolor='black', capsize=4)
            
            # Shuffle Baseline Reference Line
            ax.axhline(1.0, color='red', linestyle='--', lw=2, label='Shuffled Baseline (Failure)')
            
            ax.set_title(f"Models Trained With: {train_loss} Loss", fontsize=16, pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
            ax.set_ylabel(f"Normalized {eval_name}\n(Fraction of Shuffle)")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            if ax_idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

        fig.suptitle(f"Evaluated with: {eval_name} Divergence\n(Normalized to Shuffled Baseline - Lower is Better)", fontsize=20, y=1.02)
        fig.tight_layout()
        
        figures[eval_name] = fig
        
    return figures

if __name__ == "__main__":
    completed_data = load_completed_results()
    
    if len(completed_data) > 0:
        figs = plot_comprehensive_bar_charts(completed_data)
        
        output_dir = "Decoder_Results_Exports"
        os.makedirs(output_dir, exist_ok=True)
        
        for eval_name, fig in figs.items():
            filename = os.path.join(output_dir, f"Normalized_BarChart_Eval_{eval_name}.svg")
            fig.savefig(filename, format='svg', bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {filename}")