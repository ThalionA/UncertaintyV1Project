# -*- coding: utf-8 -*-
"""
PCA Visualisation of Perceptual Posteriors for VR Decoder
Generates Condition Averages, Per-Mouse PCA, and Global Universal PCA.
Exports all figures as SVGs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA
from utils_v26 import load_vr_export

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

def get_mouse_data(mouse_ids=[0, 1, 2, 3, 4, 5]):
    all_data = {}
    for mouse_id in mouse_ids:
        try:
            _, targets_perc, _, _, trials = load_vr_export(mouse_id, 'VR_Decoder_Data_Export.mat')
            all_data[mouse_id] = {
                'targets_perc': targets_perc,
                'trials': trials
            }
        except Exception as e:
            print(f"Skipping Mouse {mouse_id}: {e}")
    return all_data

def plot_condition_averages(all_data):
    print("Generating Condition Averages...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    s_grid = np.arange(0, 91, 1)
    
    all_trials = [d['trials'] for d in all_data.values()]
    oris = np.unique(np.concatenate([t['orientation'] for t in all_trials]))
    conts = np.unique(np.concatenate([t['contrast'] for t in all_trials]))
    disps = np.unique(np.concatenate([t['dispersion'] for t in all_trials]))

    # 1. By Orientation
    cmap_ori = cm.viridis
    for i, o in enumerate(oris):
        pooled = []
        for d in all_data.values():
            mask = d['trials']['orientation'] == o
            if np.sum(mask) > 0:
                pooled.append(np.nanmean(d['targets_perc'][mask], axis=0))
        if pooled:
            axes[0].plot(s_grid, np.nanmean(pooled, axis=0), color=cmap_ori(i/len(oris)), lw=2, label=f'{o}°')
    axes[0].set_title('Average by |Δ from Go|')
    axes[0].set_xlabel('|Δ from Go| (deg)')
    axes[0].set_ylabel('Probability')
    axes[0].legend(fontsize='small', ncol=2)

    # 2. By Contrast
    cmap_cont = cm.plasma
    for i, c in enumerate(conts):
        pooled = []
        for d in all_data.values():
            mask = d['trials']['contrast'] == c
            if np.sum(mask) > 0:
                pooled.append(np.nanmean(d['targets_perc'][mask], axis=0))
        if pooled:
            axes[1].plot(s_grid, np.nanmean(pooled, axis=0), color=cmap_cont(i/len(conts)), lw=2, label=f'{c}')
    axes[1].set_title('Average by Contrast')
    axes[1].set_xlabel('|Δ from Go| (deg)')
    axes[1].legend(fontsize='small')

    # 3. By Dispersion
    cmap_disp = cm.magma
    for i, dp in enumerate(disps):
        pooled = []
        for d in all_data.values():
            mask = d['trials']['dispersion'] == dp
            if np.sum(mask) > 0:
                pooled.append(np.nanmean(d['targets_perc'][mask], axis=0))
        if pooled:
            axes[2].plot(s_grid, np.nanmean(pooled, axis=0), color=cmap_disp(i/len(disps)), lw=2, label=f'{dp}°')
    axes[2].set_title('Average by Dispersion')
    axes[2].set_xlabel('|Δ from Go| (deg)')
    axes[2].legend(fontsize='small')

    plt.tight_layout()
    return fig

def run_per_mouse_pcas(all_data):
    """ PCA individually for each mouse: Variance, Loadings, and Evolution """
    print("Running Per-Mouse PCA Analysis...")
    mouse_ids = list(all_data.keys())
    pca_results = {}
    s_grid = np.arange(0, 91, 1)
    
    # 1. Cumulative Variance Plot
    fig_vars = plt.figure(figsize=(8, 6))
    all_vars = []
    
    for mid in mouse_ids:
        targets = all_data[mid]['targets_perc']
        trials = all_data[mid]['trials']
        
        stim_conditions = np.column_stack((trials['orientation'], trials['contrast'], trials['dispersion']))
        unique_cats = np.unique(stim_conditions, axis=0)
        
        avg_dists = np.zeros((len(unique_cats), targets.shape[1]))
        for i, cat in enumerate(unique_cats):
            mask = np.all(stim_conditions == cat, axis=1)
            avg_dists[i] = np.nanmean(targets[mask], axis=0)
            
        pca = PCA()
        pca.fit(avg_dists)
        
        # FIX EIGENVECTOR SIGN AMBIGUITY
        for pc_idx in range(2):
            max_idx = np.argmax(np.abs(pca.components_[pc_idx]))
            if pca.components_[pc_idx, max_idx] < 0:
                pca.components_[pc_idx] = -pca.components_[pc_idx]
        
        cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
        all_vars.append(cum_var)
        
        pca_results[mid] = {'pca': pca, 'avg_dists': avg_dists, 'unique_cats': unique_cats}
        plt.plot(range(1, len(cum_var)+1), cum_var, alpha=0.3, label=f'Mouse {mid}')

    max_len = max([len(v) for v in all_vars])
    padded_vars = np.array([np.pad(v, (0, max_len - len(v)), 'edge') for v in all_vars])
    plt.plot(range(1, max_len+1), np.mean(padded_vars, axis=0), 'k-', lw=3, label='Average')
    plt.axhline(90, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Per-Mouse PCA Variance Explained')
    plt.legend()
    plt.xlim(0, 15) 
    plt.tight_layout()

    # 2. PCA Loadings (PC1 and PC2)
    num_mice = len(mouse_ids)
    fig_loadings, axes_l = plt.subplots(num_mice, 2, figsize=(15, 3 * num_mice), sharex=True)
    if num_mice == 1: axes_l = np.expand_dims(axes_l, axis=0)
    
    # 3. PCA Evolution (PC1 and PC2)
    fig_evol, axes_e = plt.subplots(num_mice, 2, figsize=(15, 3 * num_mice), sharex=True)
    if num_mice == 1: axes_e = np.expand_dims(axes_e, axis=0)
    
    for idx, mid in enumerate(mouse_ids):
        pca = pca_results[mid]['pca']
        pcs = pca.components_
        avg_dists = pca_results[mid]['avg_dists']
        
        # --- Plot Loadings ---
        axes_l[idx, 0].bar(s_grid, pcs[0], color='steelblue')
        axes_l[idx, 0].set_ylabel('Loading')
        axes_l[idx, 1].bar(s_grid, pcs[1], color='darkorange')
        axes_l[idx, 1].text(1.05, 0.5, f'Mouse {mid}', transform=axes_l[idx, 1].transAxes, va='center', fontweight='bold')
        if idx == 0: 
            axes_l[idx, 0].set_title('PC1 Loadings')
            axes_l[idx, 1].set_title('PC2 Loadings')

        # --- Plot Evolution ---
        transformed = pca.transform(avg_dists)
        origin_pca = pca.transform(np.mean(avg_dists, axis=0).reshape(1, -1))
        std_pc1, std_pc2 = np.std(transformed[:, 0]), np.std(transformed[:, 1])
        inc_pc1 = np.linspace(-2*std_pc1, 2*std_pc1, 11)
        inc_pc2 = np.linspace(-2*std_pc2, 2*std_pc2, 11)

        cmap1 = cm.cool
        norm1 = plt.Normalize(vmin=inc_pc1.min(), vmax=inc_pc1.max())
        pt = np.copy(origin_pca)
        for inc in inc_pc1:
            pt[0, 0] = origin_pca[0, 0] + inc
            axes_e[idx, 0].plot(s_grid, pca.inverse_transform(pt)[0], color=cmap1(norm1(inc)), lw=2)
            
        cmap2 = cm.winter
        norm2 = plt.Normalize(vmin=inc_pc2.min(), vmax=inc_pc2.max())
        pt = np.copy(origin_pca)
        for inc in inc_pc2:
            pt[0, 1] = origin_pca[0, 1] + inc
            axes_e[idx, 1].plot(s_grid, pca.inverse_transform(pt)[0], color=cmap2(norm2(inc)), lw=2)
            
        axes_e[idx, 0].set_ylabel('Prob')
        axes_e[idx, 1].text(1.05, 0.5, f'Mouse {mid}', transform=axes_e[idx, 1].transAxes, va='center', fontweight='bold')
        if idx == 0: 
            axes_e[idx, 0].set_title('Evolution PC1 (± 2 Std)')
            axes_e[idx, 1].set_title('Evolution PC2 (± 2 Std)')

    axes_l[-1, 0].set_xlabel('|Δ from Go| (deg)')
    axes_l[-1, 1].set_xlabel('|Δ from Go| (deg)')
    axes_e[-1, 0].set_xlabel('|Δ from Go| (deg)')
    axes_e[-1, 1].set_xlabel('|Δ from Go| (deg)')
    
    fig_loadings.tight_layout()
    fig_evol.tight_layout()
    
    return fig_vars, fig_loadings, fig_evol

def run_global_pca(all_data):
    """ PCA globally pooling all Universal Conditions across all mice """
    print("Running Global Universal PCA Analysis...")
    all_targets = []
    all_conditions = []
    
    for d in all_data.values():
        all_targets.append(d['targets_perc'])
        stim_conds = np.column_stack((d['trials']['orientation'], d['trials']['contrast'], d['trials']['dispersion']))
        all_conditions.append(stim_conds)
        
    all_targets = np.vstack(all_targets)
    all_conditions = np.vstack(all_conditions)
    
    unique_cats = np.unique(all_conditions, axis=0)
    print(f"  -> Extracted {len(unique_cats)} unique universal stimulus conditions.")
    
    global_avg_dists = np.zeros((len(unique_cats), all_targets.shape[1]))
    for i, cat in enumerate(unique_cats):
        mask = np.all(all_conditions == cat, axis=1)
        global_avg_dists[i] = np.nanmean(all_targets[mask], axis=0)
    
    pca_global = PCA()
    pca_global.fit(global_avg_dists)
    
    # Fix global sign ambiguity
    for pc_idx in range(2):
        max_idx = np.argmax(np.abs(pca_global.components_[pc_idx]))
        if pca_global.components_[pc_idx, max_idx] < 0:
            pca_global.components_[pc_idx] = -pca_global.components_[pc_idx]
            
    s_grid = np.arange(0, 91, 1)
    
    # 1. Global Loadings
    fig_global_loadings, axes_l = plt.subplots(1, 2, figsize=(16, 5))
    axes_l[0].bar(s_grid, pca_global.components_[0], color='steelblue')
    axes_l[0].set_title('GLOBAL PC1 Loadings')
    axes_l[0].set_ylabel('Loading')
    axes_l[1].bar(s_grid, pca_global.components_[1], color='darkorange')
    axes_l[1].set_title('GLOBAL PC2 Loadings')
    for ax in axes_l: ax.set_xlabel('|Δ from Go| (deg)')
    fig_global_loadings.tight_layout()
    
    # 2. Global Evolution
    transformed = pca_global.transform(global_avg_dists)
    origin_pca = pca_global.transform(np.mean(global_avg_dists, axis=0).reshape(1, -1))
    
    std_pc1, std_pc2 = np.std(transformed[:, 0]), np.std(transformed[:, 1])
    inc_pc1 = np.linspace(-2*std_pc1, 2*std_pc1, 11)
    inc_pc2 = np.linspace(-2*std_pc2, 2*std_pc2, 11)
    
    fig_global_evol, axes_e = plt.subplots(1, 2, figsize=(16, 6))
    
    cmap1 = cm.cool
    norm1 = plt.Normalize(vmin=inc_pc1.min(), vmax=inc_pc1.max())
    pt = np.copy(origin_pca)
    for inc in inc_pc1:
        pt[0, 0] = origin_pca[0, 0] + inc
        axes_e[0].plot(s_grid, pca_global.inverse_transform(pt)[0], color=cmap1(norm1(inc)), lw=2)
    axes_e[0].set_title('GLOBAL Evolution PC1 (± 2 Std)')
    axes_e[0].set_ylabel('Probability')

    cmap2 = cm.winter
    norm2 = plt.Normalize(vmin=inc_pc2.min(), vmax=inc_pc2.max())
    pt = np.copy(origin_pca)
    for inc in inc_pc2:
        pt[0, 1] = origin_pca[0, 1] + inc
        axes_e[1].plot(s_grid, pca_global.inverse_transform(pt)[0], color=cmap2(norm2(inc)), lw=2)
    axes_e[1].set_title('GLOBAL Evolution PC2 (± 2 Std)')
    for ax in axes_e: ax.set_xlabel('|Δ from Go| (deg)')
    fig_global_evol.tight_layout()

    # =========================================================================
    # 3. Global Projection Scatter (WIDE figure for legend placement)
    # =========================================================================
    # Updated figsize (15, 7) provides ample room to prevent horizontal "squishing"
    fig_global_scatter = plt.figure(figsize=(15, 7)) 
    
    oris = unique_cats[:, 0]
    conts = unique_cats[:, 1]
    disps = unique_cats[:, 2]
    
    # Calculate Certainty Metric: (C * 5) / D
    safe_disps = np.clip(disps, 1e-5, None)
    cert_metric = (conts * 5) / safe_disps
    
    if cert_metric.max() > cert_metric.min():
        norm_cert = (cert_metric - cert_metric.min()) / (cert_metric.max() - cert_metric.min())
        sizes = 40 + 260 * norm_cert
    else:
        sizes = np.full_like(cert_metric, 100)

    scatter = plt.scatter(transformed[:, 0], transformed[:, 1], 
                          c=oris, cmap='viridis', s=sizes, 
                          alpha=0.8, edgecolor='k', zorder=5)
    
    # Standard colorbar (now has plenty of room)
    cbar = plt.colorbar(scatter)
    cbar.set_label('|Δ from Go| (deg)')
    
    plt.scatter(origin_pca[0, 0], origin_pca[0, 1], marker='*', s=400, color='white', edgecolor='k', linewidth=1.5, zorder=6, label='Grand Mean')
    
    pc1_traj = np.tile(origin_pca, (len(inc_pc1), 1))
    pc1_traj[:, 0] += inc_pc1
    plt.scatter(pc1_traj[:, 0], pc1_traj[:, 1], c=range(len(inc_pc1)), cmap='cool', edgecolor='k', s=60, alpha=0.5, zorder=3)
    
    pc2_traj = np.tile(origin_pca, (len(inc_pc2), 1))
    pc2_traj[:, 1] += inc_pc2
    plt.scatter(pc2_traj[:, 0], pc2_traj[:, 1], c=range(len(inc_pc2)), cmap='winter', edgecolor='k', s=60, alpha=0.5, zorder=3)

    plt.axhline(origin_pca[0,1], color='k', linestyle='--', alpha=0.3, zorder=1)
    plt.axvline(origin_pca[0,0], color='k', linestyle='--', alpha=0.3, zorder=1)
    
    # Size legend helpers
    plt.scatter([], [], c='gray', s=40, edgecolor='k', label='Low (Uncertain)')
    plt.scatter([], [], c='gray', s=300, edgecolor='k', label='High (Certain)')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('GLOBAL Projection (Color = Ori, Size = Cert)')
    
    # Bbox adjusted back to 1.05 now that the figure width itself is 15 inches.
    # It clears the colorbar and tight_layout won't need to shrink the main axes.
    plt.legend(title='Certainty (C×5/D)', loc='upper left', bbox_to_anchor=(1.05, 1), scatterpoints=1)
    fig_global_scatter.tight_layout()
    
    return fig_global_loadings, fig_global_evol, fig_global_scatter

if __name__ == "__main__":
    set_style()
    all_data = get_mouse_data()
    
    if len(all_data) > 0:
        # 1. Generate Figures
        fig_avgs = plot_condition_averages(all_data)
        
        # Per-Mouse figures
        fig_vars, fig_pm_loadings, fig_pm_evol = run_per_mouse_pcas(all_data)
        
        # Global figures
        fig_gl_loadings, fig_gl_evol, fig_gl_scatter = run_global_pca(all_data)
        
        # 2. Export Section
        output_dir = "PCA_SVG_Exports"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving SVG figures to folder: ./{output_dir}/")
        
        fig_avgs.savefig(os.path.join(output_dir, "1_Condition_Averages.svg"), format='svg', bbox_inches='tight')
        fig_vars.savefig(os.path.join(output_dir, "2_PerMouse_Cumulative_Variance.svg"), format='svg', bbox_inches='tight')
        fig_pm_loadings.savefig(os.path.join(output_dir, "3_PerMouse_Loadings.svg"), format='svg', bbox_inches='tight')
        fig_pm_evol.savefig(os.path.join(output_dir, "4_PerMouse_Evolution.svg"), format='svg', bbox_inches='tight')
        
        fig_gl_loadings.savefig(os.path.join(output_dir, "5_Global_Loadings.svg"), format='svg', bbox_inches='tight')
        fig_gl_evol.savefig(os.path.join(output_dir, "6_Global_Evolution.svg"), format='svg', bbox_inches='tight')
        fig_gl_scatter.savefig(os.path.join(output_dir, "7_Global_Projection_Scatter.svg"), format='svg', bbox_inches='tight')
        
        print("Export complete. Displaying figures...")
        plt.show()
        
    else:
        print("No data loaded. Please check the export file.")