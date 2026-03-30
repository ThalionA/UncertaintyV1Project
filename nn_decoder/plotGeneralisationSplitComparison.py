# -*- coding: utf-8 -*-
"""
Generalization & Target PC Stability Visualizer
1. Evaluates OOD generalization performance (PPC vs SBC) separated by time_window.
2. Extracts the exact Target PCs used for the PCA loss during training.
3. Visualizes the ground-truth loss subspace alignment across generalization splits,
   averaging across random splits (configurations) and animals.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Aesthetics & Helper Functions
# ==========================================
def set_style():
    sns.set_context("talk", font_scale=0.8)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})
    plt.rcParams['font.family'] = 'sans-serif'
    return {'PPC': '#ff7f0e', 'SBC': '#1f77b4'}

def calc_pca_dist(p, q, pcs, evar):
    """ Calculates the weighted squared distance in PCA space. """
    if pcs is None or len(pcs) == 0: 
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

# ==========================================
# 2. Data Loading & Filtering
# ==========================================
def load_and_filter_data(directory='.'):
    search_pattern = os.path.join(directory, 'population_results_config_*.mat')
    file_list = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    rows = []
    target_pc_store = {} 
    
    print(f"Scanning {len(file_list)} files for metrics & target PCs...")
    for f in file_list:
        try:
            mat = sio.loadmat(f, simplify_cells=True)
            cfg = mat['config']
            
            # Only analyze PCA loss configurations
            if cfg.get('custom_loss_func') != 'PCA': continue
            
            split = cfg.get('split_type')
            b_size = cfg.get('bin_size_ms', 50)
            window = cfg.get('time_window', 'full')
            lam = cfg.get('entropy_lambda', 0.0)
            config_id = f"{window}_{b_size}_{lam}"
                
            for m_id, m_data in mat['results'].items():
                dist = m_data.get('Dist', {})
                pcs = dist.get('pcs', None)
                evar = dist.get('explained_var', None)
                
                if pcs is not None and evar is not None:
                    # 1. Store the exact Target PCs used during training
                    if m_id not in target_pc_store: target_pc_store[m_id] = {}
                    if config_id not in target_pc_store[m_id]: target_pc_store[m_id][config_id] = {}
                    target_pc_store[m_id][config_id][split] = pcs
                    
                    # 2. Extract and compute Distance Metrics 
                    val_spat = np.nanmean(calc_pca_dist(dist['spat']['target'], dist['spat']['decoded'], pcs, evar))
                    val_temp = np.nanmean(calc_pca_dist(dist['temp']['target'], dist['temp']['decoded'], pcs, evar))
                    val_spat_shf = np.nanmean(calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar))
                    val_temp_shf = np.nanmean(calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar))
                    
                    norm_spat = val_spat / val_spat_shf if val_spat_shf > 0 else np.nan
                    norm_temp = val_temp / val_temp_shf if val_temp_shf > 0 else np.nan
                    
                    base_dict = {'Mouse': m_id, 'Split_Type': split, 'Time_Window': window, 
                                 'Bin_Size': b_size, 'Lambda': lam, 'Config_Group': config_id}
                                 
                    rows.append({**base_dict, 'Architecture': 'PPC', 'Norm_Loss': norm_spat})
                    rows.append({**base_dict, 'Architecture': 'SBC', 'Norm_Loss': norm_temp})
                    
        except Exception as e:
            # Silently pass corrupted or incomplete files
            pass
            
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("[!] No valid PCA configurations extracted. Check file availability.")
        return df, {}

    # Strict Intersection Filtering (Keep only configs with ALL 3 splits completed)
    required_splits = {'stratified_balanced', 'generalize_contrast', 'generalize_dispersion'}
    valid_groups = []
    
    for c_id, group in df.groupby('Config_Group'):
        if set(group['Split_Type'].unique()) == required_splits:
            valid_groups.append(c_id)
                
    df_filtered = df[df['Config_Group'].isin(valid_groups)].copy()
    
    # Filter PC store to match the strictly valid groups
    filtered_pcs = {}
    for m_id, configs in target_pc_store.items():
        filtered_pcs[m_id] = {}
        for c_id, splits in configs.items():
            if c_id in valid_groups and set(splits.keys()) == required_splits:
                filtered_pcs[m_id][c_id] = splits
                
    print(f"Retained {len(valid_groups)} complete hyperparameter configurations (having all 3 splits).")
    return df_filtered, filtered_pcs

# ==========================================
# 3. Performance Visualization
# ==========================================
def plot_performance_by_window(df, colors):
    if df.empty: return
    
    df['Split_Type'] = df['Split_Type'].replace({
        'stratified_balanced': 'Stratified (Full Space)',
        'generalize_contrast': 'Train: Contrast -> Test: Disp.',
        'generalize_dispersion': 'Train: Disp. -> Test: Contrast'
    })

    # Separate figures for full and half windows to decouple temporal dynamics
    for window in ['full', 'half']:
        df_win = df[df['Time_Window'] == window]
        if df_win.empty: continue

        g = sns.catplot(
            data=df_win, x='Bin_Size', y='Norm_Loss', hue='Architecture',
            col='Split_Type', row='Lambda', kind='point',
            palette=colors, dodge=True, markers=['s', 'o'], capsize=0.1,
            height=3.0, aspect=1.3, sharey=False
        )
        
        for ax in g.axes.flatten():
            ax.axhline(1.0, color='gray', linestyle=':', zorder=0)
            
        g.set_axis_labels("Temporal Bin Size (ms)", "Norm. PCA Dist (Frac. Chance)")
        g.set_titles(col_template="{col_name}", row_template="Lambda: {row_name}")
        
        # Add custom unified legend at the bottom
        if g._legend: g._legend.remove()
        handles = [plt.Line2D([0], [0], marker='s', color=colors['PPC'], label='PPC (Spatial)', markersize=8),
                   plt.Line2D([0], [0], marker='o', color=colors['SBC'], label='SBC (Temporal)', markersize=8),
                   plt.Line2D([0], [0], linestyle=':', color='gray', label='Shuffled Baseline')]
        g.fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))

        g.fig.suptitle(f"Generalization Performance (Time Window: {window.upper()})", y=1.05, fontweight='bold')
        out_name = f"PCA_Generalization_{window.capitalize()}.svg"
        plt.savefig(out_name, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_name}")

# ==========================================
# 4. Target PC Stability Analysis
# ==========================================
def compute_target_pc_alignments(target_pcs, n_components=5):
    """ Computes subspace alignments using the actual target PCs fit during training """
    animal_matrices_cont = {}
    animal_matrices_disp = {}
    
    for m_id, configs in target_pcs.items():
        if not configs: continue
        
        sims_cont_list = []
        sims_disp_list = []
        
        for c_id, splits in configs.items():
            try:
                # Scikit-learn PCA components are shaped (n_components, n_features)
                pcs_full = splits['stratified_balanced'][:n_components]
                pcs_cont = splits['generalize_contrast'][:n_components]
                pcs_disp = splits['generalize_dispersion'][:n_components]
                
                # Absolute cosine similarity between the exact loss projection spaces
                sim_cont = np.abs(cosine_similarity(pcs_full, pcs_cont))
                sim_disp = np.abs(cosine_similarity(pcs_full, pcs_disp))
                
                sims_cont_list.append(sim_cont)
                sims_disp_list.append(sim_disp)
            except Exception as e:
                continue
                
        # Average across all valid random splits (configurations) for this animal
        if sims_cont_list:
            animal_matrices_cont[m_id] = np.mean(sims_cont_list, axis=0)
            animal_matrices_disp[m_id] = np.mean(sims_disp_list, axis=0)
            
    return animal_matrices_cont, animal_matrices_disp

def plot_target_pc_stability(target_pcs):
    animal_cont, animal_disp = compute_target_pc_alignments(target_pcs)
    if not animal_cont: 
        print("[!] No alignment data could be computed.")
        return
    
    # --- 1. GRAND AVERAGE ACROSS ANIMALS ---
    grand_avg_cont = np.mean(list(animal_cont.values()), axis=0)
    grand_avg_disp = np.mean(list(animal_disp.values()), axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(grand_avg_cont, annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0],
                xticklabels=np.arange(1, 6), yticklabels=np.arange(1, 6))
    axes[0].set_title("Target PCs: Stratified vs Contrast")
    axes[0].set_xlabel("PC (Train-Contrast)")
    axes[0].set_ylabel("PC (Stratified)")
    
    sns.heatmap(grand_avg_disp, annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[1],
                xticklabels=np.arange(1, 6), yticklabels=np.arange(1, 6))
    axes[1].set_title("Target PCs: Stratified vs Dispersion")
    axes[1].set_xlabel("PC (Train-Dispersion)")
    
    fig.suptitle("Ground Truth (Target) Loss Subspace Alignment\n(Averaged across animals & random splits)", fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig("Target_PC_Stability_Grand_Average.svg", bbox_inches='tight')
    plt.close()
    print("Saved Target_PC_Stability_Grand_Average.svg")
    
    # --- 2. INDIVIDUAL ANIMAL HEATMAPS ---
    n_animals = len(animal_cont)
    fig, axes = plt.subplots(n_animals, 2, figsize=(10, 3 * n_animals))
    
    if n_animals == 1: axes = np.array([axes])
        
    for idx, (m_id, mat_cont) in enumerate(animal_cont.items()):
        mat_disp = animal_disp[m_id]
        
        sns.heatmap(mat_cont, annot=False, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[idx, 0])
        axes[idx, 0].set_ylabel(f"{m_id}\nPC (Strat)")
        if idx == 0: axes[idx, 0].set_title("Vs Target Train-Contrast")
            
        sns.heatmap(mat_disp, annot=False, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[idx, 1])
        if idx == 0: axes[idx, 1].set_title("Vs Target Train-Dispersion")
            
    fig.suptitle("Per-Animal Target Subspace Alignment", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("Target_PC_Stability_Per_Animal.svg", bbox_inches='tight')
    plt.close()
    print("Saved Target_PC_Stability_Per_Animal.svg")

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    palette = set_style()
    df_strict, dict_target_pcs = load_and_filter_data()
    
    if not df_strict.empty:
        print("\n--- Generating Performance Plots ---")
        plot_performance_by_window(df_strict, palette)
        
        print("\n--- Generating Subspace Alignment Plots ---")
        plot_target_pc_stability(dict_target_pcs)
        
        print("\nAll visualizations complete!")