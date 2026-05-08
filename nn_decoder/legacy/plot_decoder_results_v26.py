# -*- coding: utf-8 -*-
"""
Post-Hoc Divergence Evaluation for VR Decoder Temporal Sweep.
Plots KL, JS, Wasserstein, and PCA divergence as a function of Temporal Bin Size.
Reveals if SBC representations degrade into PPC representations at larger integration windows.
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
    print(f"Found {len(file_list)} completed configurations.")
    
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
            print(f"Error loading {f}: {e}")
            
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
    """ Calculates variance-weighted Euclidean distance in PCA space """
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

def plot_temporal_sweep(loaded_data):
    """ Plots Performance vs Bin Size to visualize temporal degradation """
    set_style()
    # ALL FOUR METRICS INCLUDED
    metrics = {'KL': calc_kl, 'JS': calc_js, 'Wasserstein': calc_wasserstein, 'PCA': calc_pca_dist}
    archs = [('spat', 'Spatial (PPC)', 'darkorange', '-'), ('temp', 'Temporal (SBC)', 'steelblue', '-')]
    
    unique_losses = list(set([d['config'].get('custom_loss_func', 'KL') for d in loaded_data]))
    unique_windows = list(set([d['config'].get('time_window', 'full') for d in loaded_data]))
    
    figures = {}
    
    for loss_func in unique_losses:
        for window in unique_windows:
            
            # Filter data for this specific plot
            subset = [d for d in loaded_data if d['config'].get('custom_loss_func') == loss_func and d['config'].get('time_window') == window]
            if not subset: continue
                
            bin_sizes = sorted(list(set([d['config'].get('bin_size_ms', 50) for d in subset])))
            
            # Increased width to 22 inches to comfortably fit 4 subplot panels side-by-side
            fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6), sharex=True)
            if len(metrics) == 1: axes = [axes]
            
            for m_idx, (m_name, m_func) in enumerate(metrics.items()):
                ax = axes[m_idx]
                
                for arch_key, arch_label, color, ls in archs:
                    means = []
                    sems = []
                    
                    for b_size in bin_sizes:
                        cfg_data = [d for d in subset if d['config'].get('bin_size_ms') == b_size]
                        
                        if not cfg_data:
                            means.append(np.nan)
                            sems.append(np.nan)
                            continue
                            
                        c_data = cfg_data[0] 
                        
                        all_vals = []
                        for m_id in c_data['results'].keys():
                            dist = c_data['results'][m_id]['Dist'][arch_key]
                            
                            # --- Handle PCA signature difference ---
                            if m_name == 'PCA':
                                # Fetch the global PCS matrices for this specific animal configuration
                                pcs = c_data['results'][m_id]['Dist'].get('pcs', None)
                                evar = c_data['results'][m_id]['Dist'].get('explained_var', None)
                                val = np.nanmean(m_func(dist['target'], dist['decoded'], pcs, evar))
                            else:
                                val = np.nanmean(m_func(dist['target'], dist['decoded']))
                                
                            all_vals.append(val)
                            
                        means.append(np.nanmean(all_vals))
                        sems.append(stats.sem(all_vals, nan_policy='omit'))
                        
                    # Plot the line across bin sizes
                    ax.errorbar(bin_sizes, means, yerr=sems, label=arch_label, color=color, linestyle=ls, marker='o', markersize=8, capsize=5, lw=2)
                
                ax.set_title(f"Test {m_name} Divergence")
                ax.set_xlabel("Temporal Bin Size (ms)")
                if m_idx == 0: ax.set_ylabel("Divergence (Lower is Better)")
                ax.set_xticks(bin_sizes)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                if m_idx == 0: ax.legend()

            fig.suptitle(f"Temporal Integration Sweep (Loss: {loss_func} | Window: {window.upper()})", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            fig_key = f"{loss_func}_{window}"
            figures[fig_key] = fig

    return figures

if __name__ == "__main__":
    completed_data = load_completed_results()
    
    if len(completed_data) > 0:
        figs_dict = plot_temporal_sweep(completed_data)
        
        output_dir = "Decoder_Results_Exports"
        os.makedirs(output_dir, exist_ok=True)
        
        for key, fig in figs_dict.items():
            filename = os.path.join(output_dir, f"Temporal_Sweep_{key}.svg")
            fig.savefig(filename, format='svg', bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {filename}")