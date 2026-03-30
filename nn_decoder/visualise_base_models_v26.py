# -*- coding: utf-8 -*-
"""
Visualizes the internal predictions of the base VR Decoders trained on Real IO targets.
Diagnoses Spatial vs Temporal differences and verifies Temporal Bin "sampling" behavior.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

def load_real_target_results(directory='.'):
    """ Loads only configurations trained on 'real' targets """
    search_pattern = os.path.join(directory, 'population_results_config_*.mat')
    file_list = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    valid_data = []
    for f in file_list:
        try:
            mat = sio.loadmat(f, simplify_cells=True)
            cfg = mat['config']
            if cfg.get('target_source') == 'real' and cfg.get('which_model') == 'perception':
                valid_data.append({
                    'file': os.path.basename(f),
                    'config_num': int(os.path.basename(f).split('_')[-1].split('.')[0]),
                    'config': cfg,
                    'results': mat['results']
                })
        except Exception as e:
            pass
            
    print(f"Found {len(valid_data)} configurations trained on Real IO targets.")
    return valid_data

def calculate_variance(targets, s_grid=np.arange(0, 91, 1), axis=-1):
    """ Calculates variance (width) of probability distributions along a specified axis. """
    # Reshape s_grid to broadcast correctly against the target array
    grid_shape = [1] * targets.ndim
    grid_shape[axis] = len(s_grid)
    s_grid_broad = s_grid.reshape(grid_shape)
    
    denom = np.nansum(targets, axis=axis, keepdims=True) + 1e-10
    weighted_mean = np.nansum(targets * s_grid_broad, axis=axis, keepdims=True) / denom
    squared_diffs = (s_grid_broad - weighted_mean) ** 2
    
    # Calculate variance and squeeze out the probability dimension we just collapsed
    variance = np.nansum(targets * squared_diffs, axis=axis) / np.squeeze(denom, axis=axis)
    return variance

def plot_condition_averages(data, config_num):
    """ Plots average posteriors for Spatial, Temporal, and Real IO """
    results = data['results']
    s_grid = np.arange(0, 91, 1)
    
    # Pool all trials across mice
    all_targets, all_spat, all_temp, all_oris = [], [], [], []
    for m_data in results.values():
        if len(m_data['trials']['orientation']) == 0: continue
        all_targets.append(m_data['Dist']['spat']['target']) # Target is same for spat/temp
        all_spat.append(m_data['Dist']['spat']['decoded'])
        all_temp.append(m_data['Dist']['temp']['decoded'])
        all_oris.append(m_data['trials']['orientation'])
        
    all_targets = np.vstack(all_targets)
    all_spat = np.vstack(all_spat)
    all_temp = np.vstack(all_temp)
    all_oris = np.concatenate(all_oris)
    
    unique_oris = np.unique(all_oris)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cmap = plt.cm.viridis
    
    for i, o in enumerate(unique_oris):
        mask = (all_oris == o)
        color = cmap(i / len(unique_oris))
        
        axes[0].plot(s_grid, np.nanmean(all_targets[mask], axis=0), color=color, lw=2, label=f'{o}°' if i%2==0 else "")
        axes[1].plot(s_grid, np.nanmean(all_spat[mask], axis=0), color=color, lw=2)
        axes[2].plot(s_grid, np.nanmean(all_temp[mask], axis=0), color=color, lw=2)
        
    axes[0].set_title('Real IO (Target)')
    axes[1].set_title('Spatial (PPC) Prediction')
    axes[2].set_title('Temporal (SBC) Prediction')
    
    axes[0].set_ylabel('Probability')
    axes[0].legend(fontsize='x-small', ncol=2)
    for ax in axes: ax.set_xlabel('|Δ from Go| (deg)')
    
    fig.suptitle(f"Configuration {config_num} (Loss: {data['config']['custom_loss_func']}) - Condition Averages", fontsize=14)
    fig.tight_layout()
    return fig

def plot_single_trial_dynamics(data, config_num, n_trials=3):
    """ Plots heatmaps of 50ms Temporal Bins for a few randomly selected trials """
    results = data['results']
    s_grid = np.arange(0, 91, 1)
    
    # Extract data from the first mouse with trials
    m_data = next(iter([m for m in results.values() if len(m['trials']['orientation']) > 0]))
    
    oris = m_data['trials']['orientation']
    conts = m_data['trials']['contrast']
    
    # Pick a few diverse trials (e.g., high vs low contrast)
    np.random.seed(42)
    trial_idxs = np.random.choice(len(oris), n_trials, replace=False)
    
    fig, axes = plt.subplots(n_trials, 2, figsize=(15, 4 * n_trials))
    if n_trials == 1: axes = [axes]
    
    for row, t_idx in enumerate(trial_idxs):
        true_ori = oris[t_idx]
        cont = conts[t_idx]
        
        target = m_data['Dist']['temp']['target'][t_idx]
        temp_avg = m_data['Dist']['temp']['decoded'][t_idx]
        
        # decoded_samp shape can be (N, 91, T) or (N, T, 91). We want (91, T) for heatmap.
        samp = m_data['Dist']['temp']['decoded_samp'][t_idx]
        if samp.shape[0] != 91: 
            samp = samp.T
            
        T_bins = samp.shape[1]
        
        # 1. Heatmap
        ax_hm = axes[row][0]
        im = ax_hm.imshow(samp, aspect='auto', origin='lower', extent=[0, T_bins, 0, 90], cmap='magma')
        ax_hm.axhline(true_ori, color='cyan', linestyle='--', lw=2, label='True Stim')
        ax_hm.set_title(f'Trial {t_idx} (Ori: {true_ori}°, Cont: {cont})\nTemporal Bins Heatmap')
        ax_hm.set_ylabel('Decoded Ori (deg)')
        if row == n_trials - 1: ax_hm.set_xlabel('Time Bin')
        ax_hm.legend(loc='upper right', fontsize='x-small')
        fig.colorbar(im, ax=ax_hm, label='Probability')
        
        # 2. Spaghetti Plot (Overlaid Bins vs Trial Avg)
        ax_sp = axes[row][1]
        ax_sp.plot(s_grid, samp, color='gray', alpha=0.3, lw=1)
        ax_sp.plot([], [], color='gray', label='Individual 50ms Bins') # Dummy for legend
        ax_sp.plot(s_grid, temp_avg, color='blue', lw=3, label='Temporal (Trial-Avg)')
        ax_sp.plot(s_grid, target, color='black', lw=3, linestyle='--', label='Real IO (Target)')
        ax_sp.axvline(true_ori, color='cyan', linestyle='--', lw=2)
        
        ax_sp.set_title('Instantaneous Bins vs Averaged Output')
        if row == n_trials - 1: ax_sp.set_xlabel('Orientation (deg)')
        ax_sp.legend(fontsize='x-small')

    fig.tight_layout()
    return fig

def plot_bin_sharpness(data, config_num):
    """ Compares the variance (width) of the bins vs the trial-averaged posterior """
    results = data['results']
    
    var_io, var_spat, var_temp_avg, var_temp_bins = [], [], [], []
    s_grid = np.arange(0, 91, 1)
    
    for m_data in results.values():
        if len(m_data['trials']['orientation']) == 0: continue
        
        targ = m_data['Dist']['spat']['target']
        spat = m_data['Dist']['spat']['decoded']
        temp_avg = m_data['Dist']['temp']['decoded']
        
        samp = m_data['Dist']['temp']['decoded_samp']
        axis_91 = 1 if samp.shape[1] == 91 else 2
        
        # Trial-averaged variances (2D arrays, probability is the last axis)
        var_io.extend(calculate_variance(targ, s_grid, axis=-1))
        var_spat.extend(calculate_variance(spat, s_grid, axis=-1))
        var_temp_avg.extend(calculate_variance(temp_avg, s_grid, axis=-1))
        
        # 3D array variance: calculate along the probability axis
        # This collapses the 91 bins, leaving (Trials, Timebins)
        bin_vars = calculate_variance(samp, s_grid, axis=axis_91) 
        
        # Average the variance across the timebins (which is now the last axis) to get one number per trial
        mean_bin_vars = np.nanmean(bin_vars, axis=-1)
        var_temp_bins.extend(mean_bin_vars)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = [var_io, var_spat, var_temp_avg, var_temp_bins]
    labels = ['Real IO\n(Target)', 'Spatial\n(Trial-Avg)', 'Temporal\n(Trial-Avg)', 'Temporal Bins\n(Instantaneous)']
    colors = ['gray', 'darkorange', 'steelblue', 'skyblue']
    
    sns.violinplot(data=data_to_plot, palette=colors, ax=ax, inner="quartile")
    
    ax.set_xticklabels(labels)
    ax.set_ylabel('Posterior Variance (deg²)')
    ax.set_title(f'Config {config_num}: Are instantaneous bins sharper than the average?')
    
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    set_style()
    valid_data = load_real_target_results()
    
    if len(valid_data) > 0:
        output_dir = "Base_Model_Diagnostics"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving diagnostic figures to ./{output_dir}/")
        
        # Loop through a few configurations (e.g., Wasserstein and KL)
        for data in valid_data:
            c_num = data['config_num']
            loss_func = data['config']['custom_loss_func']
            print(f"Plotting Config {c_num} ({loss_func})...")
            
            fig_avgs = plot_condition_averages(data, c_num)
            fig_dyn = plot_single_trial_dynamics(data, c_num)
            fig_sharp = plot_bin_sharpness(data, c_num)
            
            fig_avgs.savefig(os.path.join(output_dir, f"Cfg{c_num}_{loss_func}_1_Averages.svg"), format='svg')
            fig_dyn.savefig(os.path.join(output_dir, f"Cfg{c_num}_{loss_func}_2_SingleTrials.svg"), format='svg')
            fig_sharp.savefig(os.path.join(output_dir, f"Cfg{c_num}_{loss_func}_3_Sharpness.svg"), format='svg')
            
            # Close figures to save memory
            plt.close(fig_avgs)
            plt.close(fig_dyn)
            plt.close(fig_sharp)
            
        print("Done!")
    else:
        print("No configurations trained on 'real' targets found. Ensure your grid search included target_source='real'.")