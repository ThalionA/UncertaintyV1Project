# -*- coding: utf-8 -*-
"""
Multi-Animal Diagnostic & Sanity Check Plotting for VR Go/No-Go Decoder
Plots Grand Mean +/- SEM across all loaded animals, aligned to Go-Stimulus.
Evaluates empirical behavior, neural dynamics, and Bayesian synthetic models.
Exports all figures as SVGs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import scipy.stats as stats
from utils_v26 import (
    load_vr_export, 
    estimate_preferred_orientations, 
    generate_PPC_targets, 
    generate_SBC_targets,
    optimize_synthetic_params,
    calculate_np_variance,
    get_tuning_templates
)

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

def plot_true_behavior_pooled(all_data, unique_oris, unique_conts, unique_disps):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ('choice', 'True P(Go) (Actual Choice)'),
        ('velocity', 'Pre-RZ Velocity (cm/s)'),
        ('lick_rate', 'Pre-RZ Lick Rate')
    ]
    
    for row, (conditions, label_prefix) in enumerate([(unique_conts, 'Contrast'), (unique_disps, 'Dispersion')]):
        for col, (metric_name, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for val in conditions:
                animal_means = np.zeros((len(all_data), len(unique_oris)))
                for i, data in enumerate(all_data):
                    trials = data['trials']
                    mask_cond = (trials[label_prefix.lower()] == val)
                    for j, o in enumerate(unique_oris):
                        valid_trials = trials[metric_name][mask_cond & (trials['orientation'] == o)]
                        animal_means[i, j] = np.nanmean(valid_trials) if len(valid_trials) > 0 else np.nan
                
                mean_val = np.nanmean(animal_means, axis=0)
                sem_val = stats.sem(animal_means, axis=0, nan_policy='omit')
                ax.errorbar(unique_oris, mean_val, yerr=sem_val, marker='o', label=f'{label_prefix} {val}', capsize=4, lw=2)
                
            ax.axvline(45, color='k', linestyle='--', alpha=0.5)
            if row == 1: ax.set_xlabel('|Δ from Go| (deg)')
            ax.set_ylabel(ylabel.split('(')[0].strip())
            if row == 0: ax.set_title(ylabel.split('(')[0].strip())
            ax.legend(fontsize='small')
            
    fig.tight_layout()
    return fig

def plot_uncertainty_relationships_pooled(all_data, unique_oris, unique_conts, unique_disps):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, (conditions, label_prefix) in enumerate([(unique_conts, 'Contrast'), (unique_disps, 'Dispersion')]):
        for val in conditions:
            perc_means = np.zeros((len(all_data), len(unique_oris)))
            dec_means = np.zeros((len(all_data), len(unique_oris)))
            for i, data in enumerate(all_data):
                trials = data['trials']
                mask_cond = (trials[label_prefix.lower()] == val)
                for j, o in enumerate(unique_oris):
                    mask = mask_cond & (trials['orientation'] == o)
                    if np.sum(mask) > 0:
                        perc_means[i, j] = np.nanmean(data['perc_unc'][mask])
                        dec_means[i, j] = np.nanmean(data['dec_unc'][mask])
                    else:
                        perc_means[i, j], dec_means[i, j] = np.nan, np.nan
            
            p_mean = np.nanmean(perc_means, axis=0)
            p_sem = stats.sem(perc_means, axis=0, nan_policy='omit')
            axes[row, 0].errorbar(unique_oris, p_mean, yerr=p_sem, marker='o', label=f'{label_prefix} {val}', capsize=4, lw=2)
            
            d_mean = np.nanmean(dec_means, axis=0)
            d_sem = stats.sem(dec_means, axis=0, nan_policy='omit')
            axes[row, 1].errorbar(unique_oris, d_mean, yerr=d_sem, marker='o', label=f'{label_prefix} {val}', capsize=4, lw=2)
            
        axes[row, 0].set_ylabel('Perceptual Var (deg^2)')
        axes[row, 1].set_ylabel('Decision Var [p*(1-p)]')
        axes[row, 1].axvline(45, color='k', linestyle='--', alpha=0.5)
        for col in range(2):
            axes[row, col].legend(fontsize='small')
            if row == 1: axes[row, col].set_xlabel('|Δ from Go| (deg)')
            
    axes[0, 0].set_title('Perceptual Uncertainty')
    axes[0, 1].set_title('Decision Uncertainty')
    fig.tight_layout()
    return fig

def plot_neural_sanity_checks_pooled(all_data, unique_oris, unique_conts):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    target_prefs = [0, 45, 90]
    colors = ['blue', 'green', 'red']
    
    for pref, color in zip(target_prefs, colors):
        tuning_matrix = np.zeros((len(all_data), len(unique_oris)))
        for i, data in enumerate(all_data):
            neurons_mask = (data['pref_oris'] == pref)
            if np.sum(neurons_mask) == 0:
                tuning_matrix[i, :] = np.nan
                continue
            mean_act = np.nanmean(data['activities'], axis=1) 
            for j, o in enumerate(unique_oris):
                t_mask = (data['trials']['orientation'] == o)
                if np.sum(t_mask) == 0:
                    tuning_matrix[i, j] = np.nan
                    continue
                max_c = np.nanmax(data['trials']['contrast'][t_mask])
                best_mask = t_mask & (data['trials']['contrast'] == max_c)
                tuning_matrix[i, j] = np.nanmean(mean_act[best_mask][:, neurons_mask])
                
        t_mean = np.nanmean(tuning_matrix, axis=0)
        t_sem = stats.sem(tuning_matrix, axis=0, nan_policy='omit')
        axes[0].errorbar(unique_oris, t_mean, yerr=t_sem, marker='o', label=f'Pref: {pref} deg', color=color, capsize=4, lw=2)
        
    axes[0].set_xlabel('|Δ from Go| (deg)')
    axes[0].set_ylabel('Mean Activity')
    axes[0].set_title('Pop Tuning (0=Go, 90=NoGo)')
    axes[0].legend()
    
    t_bins = all_data[0]['activities'].shape[1]
    time_axis = np.linspace(1.0, 2.0, t_bins) 
    
    for c in unique_conts:
        psth_matrix = np.zeros((len(all_data), t_bins))
        for i, data in enumerate(all_data):
            mask_c = (data['trials']['contrast'] == c)
            if np.sum(mask_c) > 0:
                psth_matrix[i, :] = np.nanmean(data['activities'][mask_c, :, :], axis=(0, 2))
            else:
                psth_matrix[i, :] = np.nan
                
        psth_mean = np.nanmean(psth_matrix, axis=0)
        psth_sem = stats.sem(psth_matrix, axis=0, nan_policy='omit')
        axes[1].plot(time_axis, psth_mean, lw=2, label=f'Contrast {c}')
        axes[1].fill_between(time_axis, psth_mean - psth_sem, psth_mean + psth_sem, alpha=0.2)
        
    axes[1].set_xlabel('Time in Epoch (s)')
    axes[1].set_ylabel('Mean Population Activity')
    axes[1].set_title('Population Activity vs Contrast')
    axes[1].legend()
    
    pooled_prefs = np.concatenate([d['pref_oris'] for d in all_data])
    counts = [np.sum(pooled_prefs == o) for o in unique_oris]
    x_positions = np.arange(len(unique_oris)) 
    axes[2].bar(x_positions, counts, color='skyblue', edgecolor='black', alpha=0.8, width=1.0)
    
    if 45.0 in unique_oris:
        boundary_idx = np.where(unique_oris == 45.0)[0][0]
        axes[2].axvline(boundary_idx, color='r', linestyle='--', alpha=0.5, label='Boundary')
        axes[2].legend()
        
    axes[2].set_xlabel('Preferred |Δ from Go| (deg)')
    axes[2].set_ylabel('Total Number of Neurons')
    axes[2].set_title('Global Distribution of Preferences')
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels([str(int(o)) for o in unique_oris])
    
    fig.tight_layout()
    return fig

def plot_example_neural_traces(all_data, unique_oris):
    num_mice = len(all_data)
    fig, axes = plt.subplots(num_mice, 3, figsize=(15, 3 * num_mice), sharex=True)
    if num_mice == 1: axes = np.expand_dims(axes, axis=0)
    
    for i, data in enumerate(all_data):
        activities = data['activities'] 
        trials = data['trials']
        t_bins = activities.shape[1]
        time_axis = np.linspace(1.0, 2.0, t_bins)
        
        # Find 3 most active neurons
        mean_rates = np.nanmean(activities, axis=(0, 1))
        top_neurons = np.argsort(mean_rates)[-3:][::-1]
        
        for j, n in enumerate(top_neurons):
            pref = data['pref_oris'][n]
            # Furthest available orientation as "Orthogonal"
            non_pref = unique_oris[np.argmax(np.abs(unique_oris - pref))]
            
            mask_pref = (trials['orientation'] == pref)
            mask_non_pref = (trials['orientation'] == non_pref)
            
            if np.sum(mask_pref) > 0:
                psth_pref = np.nanmean(activities[mask_pref, :, n], axis=0)
                sem_pref = stats.sem(activities[mask_pref, :, n], axis=0, nan_policy='omit')
                axes[i, j].plot(time_axis, psth_pref, color='blue', label=f'Pref ({pref}°)')
                axes[i, j].fill_between(time_axis, psth_pref - sem_pref, psth_pref + sem_pref, color='blue', alpha=0.2)
                
            if np.sum(mask_non_pref) > 0:
                psth_non = np.nanmean(activities[mask_non_pref, :, n], axis=0)
                sem_non = stats.sem(activities[mask_non_pref, :, n], axis=0, nan_policy='omit')
                axes[i, j].plot(time_axis, psth_non, color='red', label=f'Orth ({non_pref}°)')
                axes[i, j].fill_between(time_axis, psth_non - sem_non, psth_non + sem_non, color='red', alpha=0.2)
                
            axes[i, j].set_title(f'Mouse {data["mouse_id"]} | Neuron {n}')
            if i == num_mice - 1: axes[i, j].set_xlabel('Time (s)')
            if j == 0: axes[i, j].set_ylabel('Firing Rate')
            axes[i, j].legend(fontsize='x-small')

    fig.tight_layout()
    return fig

def plot_synthetic_posteriors_systematic(all_data, unique_oris):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    s_grid = np.arange(0, 91, 1)
    
    metrics = {
        'Real IO': {'peak': np.zeros((len(all_data), len(unique_oris))), 'var': np.zeros((len(all_data), len(unique_oris))), 'color': 'k', 'ls': '-'},
        'PPC Posterior': {'peak': np.zeros((len(all_data), len(unique_oris))), 'var': np.zeros((len(all_data), len(unique_oris))), 'color': 'b', 'ls': '--'},
        'SBC Posterior': {'peak': np.zeros((len(all_data), len(unique_oris))), 'var': np.zeros((len(all_data), len(unique_oris))), 'color': 'r', 'ls': '-.'}
    }
    
    for i, data in enumerate(all_data):
        for j, o in enumerate(unique_oris):
            mask = (data['trials']['orientation'] == o)
            if np.sum(mask) == 0:
                for m in metrics: metrics[m]['peak'][i, j], metrics[m]['var'][i, j] = np.nan, np.nan
                continue
            
            metrics['Real IO']['peak'][i, j] = np.nanmean([s_grid[np.argmax(p)] for p in data['targets_perc'][mask]])
            metrics['Real IO']['var'][i, j] = np.nanmean(calculate_np_variance(data['targets_perc'][mask]))
            metrics['PPC Posterior']['peak'][i, j] = np.nanmean([s_grid[np.argmax(p)] for p in data['post_ppc'][mask]])
            metrics['PPC Posterior']['var'][i, j] = np.nanmean(calculate_np_variance(data['post_ppc'][mask]))
            metrics['SBC Posterior']['peak'][i, j] = np.nanmean([s_grid[np.argmax(p)] for p in data['post_sbc'][mask]])
            metrics['SBC Posterior']['var'][i, j] = np.nanmean(calculate_np_variance(data['post_sbc'][mask]))

    axes[0].plot(unique_oris, unique_oris, 'gray', alpha=0.5, lw=4, label='Perfect Decoding')
    for label, m in metrics.items():
        axes[0].errorbar(unique_oris, np.nanmean(m['peak'], axis=0), yerr=stats.sem(m['peak'], axis=0, nan_policy='omit'), 
                         color=m['color'], linestyle=m['ls'], marker='o', label=label, capsize=4)
        
    axes[0].set_title('Systematic Decoding: Peak vs True Stimulus')
    axes[0].set_xlabel('True |Δ from Go| (deg)')
    axes[0].set_ylabel('Decoded MAP Peak (deg)')
    axes[0].legend()

    for label, m in metrics.items():
        axes[1].errorbar(unique_oris, np.nanmean(m['var'], axis=0), yerr=stats.sem(m['var'], axis=0, nan_policy='omit'), 
                         color=m['color'], linestyle=m['ls'], marker='o', label=label, capsize=4)
        
    axes[1].set_title('Systematic Decoding: Uncertainty vs True Stimulus')
    axes[1].set_xlabel('True |Δ from Go| (deg)')
    axes[1].set_ylabel('Posterior Variance (deg^2)')
    axes[1].legend()

    fig.tight_layout()
    return fig

def plot_model_posteriors_comparison(data_dict_list, unique_oris, unique_conts, unique_disps, title_prefix="Grand Average"):
    fig, axes = plt.subplots(3, 5, figsize=(26, 15), sharex=True)
    s_grid = np.arange(0, 91, 1)
    
    models = [
        ('Real IO (Posterior)', 'targets_perc'), 
        ('PPC (Likelihood)', 'lik_ppc'), 
        ('PPC (Posterior)', 'post_ppc'), 
        ('SBC (Likelihood)', 'lik_sbc'), 
        ('SBC (Posterior)', 'post_sbc')
    ]
    splits = [
        ('Orientation', unique_oris, 'orientation', cm.viridis),
        ('Contrast', unique_conts, 'contrast', cm.plasma),
        ('Dispersion', unique_disps, 'dispersion', cm.magma)
    ]
    
    for row, (split_name, conditions, trial_key, cmap) in enumerate(splits):
        for col, (model_name, target_key) in enumerate(models):
            ax = axes[row, col]
            for i, val in enumerate(conditions):
                pooled_dists = []
                for data in data_dict_list:
                    mask = (data['trials'][trial_key] == val)
                    if np.sum(mask) > 0:
                        pooled_dists.append(np.nanmean(data[target_key][mask], axis=0))
                
                if pooled_dists:
                    mean_dist = np.nanmean(pooled_dists, axis=0)
                    ax.plot(s_grid, mean_dist, color=cmap(i/len(conditions)), lw=2, label=f'{val}')
            
            if row == 0: ax.set_title(f'{model_name}')
            if col == 0: ax.set_ylabel(f'Split by {split_name}\nProbability')
            if row == 2: ax.set_xlabel('|Δ from Go| (deg)')
            if col == 4: ax.legend(fontsize='x-small', loc='upper right')

    fig.suptitle(f'{title_prefix} Likelihoods vs Posteriors', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


if __name__ == "__main__":
    set_style()
    filepath = 'VR_Decoder_Data_Export.mat' 
    all_data = []
    
    print("Pre-loading and optimizing data across all animals...")
    for mouse_id in range(6): 
        try:
            activities_m, targets_perc, targets_dec, trials = load_vr_export(mouse_id, filepath)
        except Exception as e:
            print(f"Skipping Mouse {mouse_id} (Not found or error): {e}")
            continue
            
        activities = np.transpose(activities_m, (1, 2, 0))
        mean_act = np.nanmean(activities, axis=1)
        
        pref_oris = estimate_preferred_orientations(mean_act, trials)
        templates = get_tuning_templates(mean_act, trials)
        
        print(f"  Mouse {mouse_id} Loaded. Optimizing synthetic Bayesian bounds...")
        opt_beta, opt_kde = optimize_synthetic_params(activities, templates, targets_perc)
        
        lik_ppc, post_ppc = generate_PPC_targets(activities, templates, beta=opt_beta)
        lik_sbc, post_sbc = generate_SBC_targets(activities, templates, kde_std=opt_kde)
        
        perc_unc = calculate_np_variance(targets_perc)
        p_go = targets_dec[:, 0]
        dec_unc = p_go * (1 - p_go)
        
        all_data.append({
            'mouse_id': mouse_id, 
            'trials': trials, 
            'activities': activities,
            'pref_oris': pref_oris, 
            'perc_unc': perc_unc, 
            'dec_unc': dec_unc,
            'targets_perc': targets_perc, 
            'lik_ppc': lik_ppc, 
            'post_ppc': post_ppc, 
            'lik_sbc': lik_sbc, 
            'post_sbc': post_sbc
        })
        
    if not all_data: 
        print("No animal data loaded! Check filepath.")
        exit()
        
    global_oris = np.unique(np.concatenate([d['trials']['orientation'] for d in all_data]))
    global_conts = np.unique(np.concatenate([d['trials']['contrast'] for d in all_data]))
    global_disps = np.unique(np.concatenate([d['trials']['dispersion'] for d in all_data]))
    
    print(f"\nSuccessfully loaded {len(all_data)} animals.")
    
    # 1. Generate Base Figures
    fig_beh = plot_true_behavior_pooled(all_data, global_oris, global_conts, global_disps)
    fig_unc = plot_uncertainty_relationships_pooled(all_data, global_oris, global_conts, global_disps)
    fig_dyn = plot_neural_sanity_checks_pooled(all_data, global_oris, global_conts) 
    fig_tra = plot_example_neural_traces(all_data, global_oris)
    fig_syn = plot_synthetic_posteriors_systematic(all_data, global_oris)
    
    # 2. Generate Posterior Comparison Figures (3x5 Grids)
    figs_posteriors = {}
    figs_posteriors['Grand_Average'] = plot_model_posteriors_comparison(all_data, global_oris, global_conts, global_disps, "Grand Average")
    for d in all_data:
        m_id = d['mouse_id']
        figs_posteriors[f'Mouse_{m_id}'] = plot_model_posteriors_comparison([d], global_oris, global_conts, global_disps, f"Mouse {m_id}")
    
    # 3. Export Section
    output_dir = "Sanity_Check_Exports"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving SVG figures to folder: ./{output_dir}/")
    
    fig_beh.savefig(os.path.join(output_dir, "1_Behavior.svg"), format='svg', bbox_inches='tight')
    fig_unc.savefig(os.path.join(output_dir, "2_Uncertainty_Relationships.svg"), format='svg', bbox_inches='tight')
    fig_dyn.savefig(os.path.join(output_dir, "3_Neural_Dynamics.svg"), format='svg', bbox_inches='tight')
    fig_tra.savefig(os.path.join(output_dir, "4_Example_Neural_Traces.svg"), format='svg', bbox_inches='tight')
    fig_syn.savefig(os.path.join(output_dir, "5_Systematic_Synthetic_Validation.svg"), format='svg', bbox_inches='tight')
    
    for name, fig in figs_posteriors.items():
        fig.savefig(os.path.join(output_dir, f"6_Posteriors_{name}.svg"), format='svg', bbox_inches='tight')
        
    print(f"Export complete. All files saved to ./{output_dir}/")
    plt.show()