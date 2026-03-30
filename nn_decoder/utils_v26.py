import numpy as np
import torch
import scipy.stats as stats
import scipy.io as sio
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

def zscore_activity(activity):
    mean_act = np.mean(activity, axis=(0, 1), keepdims=True)
    std_act = np.std(activity, axis=(0, 1), keepdims=True)
    std_act[std_act == 0] = 1 
    return (activity - mean_act) / std_act

def get_stratified_train_test_indices(trial_categories, test_size=0.5, random_state=42):
    indices = np.arange(len(trial_categories))
    unique_classes, class_counts = np.unique(trial_categories, return_counts=True)
    
    valid_classes = unique_classes[class_counts > 1]
    valid_mask = np.isin(trial_categories, valid_classes)
    rare_mask = ~valid_mask
    
    valid_indices = indices[valid_mask]
    rare_indices = indices[rare_mask]
    
    if len(valid_indices) > 0:
        train_valid, test_valid = train_test_split(
            valid_indices, test_size=test_size, random_state=random_state, stratify=trial_categories[valid_mask]
        )
    else:
        train_valid, test_valid = np.array([], dtype=int), np.array([], dtype=int)
        
    if len(rare_indices) > 0:
        np.random.seed(random_state)
        np.random.shuffle(rare_indices)
        split_idx = int(np.round(len(rare_indices) * (1.0 - test_size)))
        train_rare, test_rare = rare_indices[:split_idx], rare_indices[split_idx:]
    else:
        train_rare, test_rare = np.array([], dtype=int), np.array([], dtype=int)
        
    train_idx = np.concatenate([train_valid, train_rare]).astype(int)
    test_idx = np.concatenate([test_valid, test_rare]).astype(int)
    
    np.random.seed(random_state)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx

def weighted_circular_variance_torch(angles, weights, circle_type='full'):
    coeff = 2 if circle_type == 'half' else 1
    angles = torch.tensor(angles, dtype=torch.float32)
    if torch.max(angles) > 2 * torch.pi:
        angles = torch.deg2rad(angles)
    if len(angles.shape) == 1:
        angles = angles.unsqueeze(0)
    C = torch.sum(weights * torch.cos(coeff * angles), dim=-1) / torch.sum(weights, dim=-1)
    S = torch.sum(weights * torch.sin(coeff * angles), dim=-1) / torch.sum(weights, dim=-1)
    Rw = torch.sqrt(C**2 + S**2)
    return (1 - Rw).squeeze()

class ToTensor:
    def __init__(self, device):
        self.device = device
    def __call__(self, sample):
        inputs, targets = sample
        return (torch.from_numpy(inputs).to(torch.float32).to(self.device), 
                torch.from_numpy(targets).to(torch.float32).to(self.device))

def weighted_variance_torch(angles, weights):
    angles = torch.tensor(angles, dtype=torch.float32).to(weights.device)
    if len(angles.shape) == 1:
        angles = angles.unsqueeze(0) 
    weighted_mean = torch.sum(weights * angles, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True)
    squared_diffs = (angles - weighted_mean) ** 2
    variance = torch.sum(weights * squared_diffs, dim=-1) / torch.sum(weights, dim=-1)
    return variance.squeeze()

def load_vr_export(mouse_id, filepath='VR_Decoder_Data_Export.mat'):
    mat = sio.loadmat(filepath, simplify_cells=True)
    neural_store = mat['NeuralStore'][mouse_id]
    trial_tbl = mat['TrialTbl_Struct'] 
    
    animal_tag = neural_store['tag'] if 'tag' in neural_store else f"Mouse_{mouse_id}"
    
    if 'animal' in trial_tbl:
        animals_array = np.array(trial_tbl['animal'])
        animal_mask = (animals_array == animal_tag)
    else:
        animal_mask = np.ones(len(trial_tbl['stimulus']), dtype=bool)

    trials_dict = {
        'orientation': trial_tbl['abs_from_go'][animal_mask], 
        'dispersion': trial_tbl['dispersion'][animal_mask],
        'contrast': trial_tbl['contrast'][animal_mask],
        'choice': trial_tbl['goChoice'][animal_mask],
        'velocity': trial_tbl['preRZ_velocity'][animal_mask],
        'lick_rate': trial_tbl['preRZ_lick_rate'][animal_mask],
        'true_orientation': trial_tbl['stimulus'][animal_mask] 
    }
    
    xG = neural_store['xG']
    time_mask = (xG >= 0.0) & (xG <= 2.0)  
    raw_spikes = neural_store['Gspk'] 
    sliced_spikes = raw_spikes[:, :, time_mask]
    activities_m = np.transpose(sliced_spikes, (1, 0, 2))
    
    targets_perc = trial_tbl['post_s_given_map'][animal_mask] 
    targets_dec = trial_tbl['decision_posterior'][animal_mask] 
    
    return activities_m, targets_perc, targets_dec, trials_dict

def estimate_preferred_orientations(activities, trials):
    n_trials, n_neurons = activities.shape
    orientations = trials['orientation']
    contrasts = trials['contrast']
    unique_oris = np.unique(orientations)
    
    pref_oris = np.zeros(n_neurons)
    for i in range(n_neurons):
        rates = activities[:, i]
        if np.nansum(rates) <= 0:
            pref_oris[i] = 45.0
            continue
            
        mean_resps = []
        for o in unique_oris:
            ori_mask = (orientations == o)
            if np.sum(ori_mask) == 0:
                mean_resps.append(np.nan)
                continue
            max_c_for_ori = np.nanmax(contrasts[ori_mask])
            best_trials_mask = ori_mask & (contrasts == max_c_for_ori)
            
            if np.sum(best_trials_mask) > 0:
                mean_resps.append(np.nanmean(rates[best_trials_mask]))
            else:
                mean_resps.append(np.nanmean(rates[ori_mask]))
                
        if np.all(np.isnan(mean_resps)):
            pref_oris[i] = 45.0
        else:
            pref_oris[i] = unique_oris[np.nanargmax(mean_resps)]
            
    return np.clip(pref_oris, 0, 90)

# ==========================================
# BAYESIAN GENERATIVE FUNCTIONS
# ==========================================

def get_prior(s_grid, prior_type='bimodal', prior_sigma=15.0):
    """ Generates the prior probability distribution over orientations. """
    if prior_type == 'bimodal':
        # Mixture of Gaussians centered at 0 (Go) and 90 (No-Go)
        p = stats.norm.pdf(s_grid, 0, prior_sigma) + stats.norm.pdf(s_grid, 90, prior_sigma)
    else:
        p = np.ones_like(s_grid)
    return p / np.sum(p)

def get_tuning_templates(activities, trials, s_grid=np.arange(0, 91, 1)):
    """ Extracts templates using Max Contrast AND Min Dispersion """
    orientations = trials['orientation']
    contrasts = trials['contrast']
    dispersions = trials['dispersion']
    unique_oris = np.unique(orientations)
    n_neurons = activities.shape[1]

    templates_emp = np.zeros((len(unique_oris), n_neurons))

    for i, o in enumerate(unique_oris):
        ori_mask = (orientations == o)
        if np.sum(ori_mask) == 0: continue
            
        max_c = np.nanmax(contrasts[ori_mask])
        c_mask = ori_mask & (contrasts == max_c)
        
        min_d = np.nanmin(dispersions[c_mask])
        best_mask = c_mask & (dispersions == min_d)
        
        templates_emp[i, :] = np.nanmean(activities[best_mask, :], axis=0)

    f = interp1d(unique_oris, templates_emp, axis=0, kind='linear', fill_value="extrapolate")
    templates = np.clip(f(s_grid), 1e-6, None)
    return templates

def generate_PPC_targets(activities, templates, s_grid=np.arange(0, 91, 1), beta=1.0, prior_type='bimodal'):
    """ Spatial Code: Returns both Likelihoods and Posteriors """
    mean_rates = np.nanmean(activities, axis=1) 
    r = mean_rates[:, np.newaxis, :]
    f = templates[np.newaxis, :, :]

    # Log-Likelihood
    LL = np.nansum(r * np.log(f) - f, axis=2) 
    scaled_LL = beta * LL
    scaled_LL -= np.max(scaled_LL, axis=1, keepdims=True)
    
    # Raw Likelihood Distribution
    L_unnorm = np.exp(scaled_LL)
    likelihoods = L_unnorm / np.nansum(L_unnorm, axis=1, keepdims=True)
    
    # Posterior (Likelihood * Prior)
    prior = get_prior(s_grid, prior_type)[np.newaxis, :]
    P_unnorm = L_unnorm * prior
    posteriors = P_unnorm / np.nansum(P_unnorm, axis=1, keepdims=True)
    
    return likelihoods, posteriors

def generate_SBC_targets(activities, templates, s_grid=np.arange(0, 91, 1), kde_std=2.0, prior_type='bimodal'):
    """ Temporal Code: Extracts MAPs for Likelihood and Posterior independently, then KDEs """
    n_trials, t_bins, n_neurons = activities.shape
    likelihoods = np.zeros((n_trials, len(s_grid)))
    posteriors = np.zeros((n_trials, len(s_grid)))

    f = templates[np.newaxis, :, :] 
    prior_log = np.log(get_prior(s_grid, prior_type) + 1e-10)[np.newaxis, :]

    for i in range(n_trials):
        r_t = activities[i, :, :][:, np.newaxis, :]

        # Instantaneous Log-Likelihood
        LL_t = np.nansum(r_t * np.log(f) - f, axis=2) 
        
        # 1. Likelihood Samples (MAP of LL_t)
        map_idx_L = np.argmax(LL_t, axis=1)
        samples_L = s_grid[map_idx_L]
        
        # 2. Posterior Samples (MAP of LL_t + LogPrior)
        LPost_t = LL_t + prior_log
        map_idx_P = np.argmax(LPost_t, axis=1)
        samples_P = s_grid[map_idx_P]

        # KDE for Likelihoods
        kde_L = np.zeros_like(s_grid, dtype=float)
        for s in samples_L:
            if not np.isnan(s): kde_L += stats.norm.pdf(s_grid, loc=s, scale=kde_std)
        likelihoods[i, :] = kde_L / np.nansum(kde_L) if np.nansum(kde_L) > 0 else 1.0/len(s_grid)

        # KDE for Posteriors
        kde_P = np.zeros_like(s_grid, dtype=float)
        for s in samples_P:
            if not np.isnan(s): kde_P += stats.norm.pdf(s_grid, loc=s, scale=kde_std)
        posteriors[i, :] = kde_P / np.nansum(kde_P) if np.nansum(kde_P) > 0 else 1.0/len(s_grid)
            
    return likelihoods, posteriors

def calculate_np_variance(targets, s_grid=np.arange(0, 91, 1)):
    denom = np.nansum(targets, axis=1, keepdims=True) + 1e-10
    weighted_mean = np.nansum(targets * s_grid, axis=1, keepdims=True) / denom
    squared_diffs = (s_grid - weighted_mean) ** 2
    return np.nansum(targets * squared_diffs, axis=1) / denom.squeeze()

def optimize_synthetic_params(activities, templates, io_targets, s_grid=np.arange(0, 91, 1)):
    valid_mask = ~np.isnan(np.sum(io_targets, axis=1))
    valid_io_targets = io_targets[valid_mask]
    
    if len(valid_io_targets) == 0: return 1.0, 2.0
    target_mean_var = np.mean(calculate_np_variance(valid_io_targets, s_grid))
    
    def ppc_loss(beta):
        _, ppc_post = generate_PPC_targets(activities, templates, s_grid, beta)
        ppc_mean_var = np.mean(calculate_np_variance(ppc_post[valid_mask], s_grid))
        return (ppc_mean_var - target_mean_var)**2
        
    res_ppc = minimize_scalar(ppc_loss, bounds=(0.001, 20.0), method='bounded')
    
    def sbc_loss(kde_std):
        _, sbc_post = generate_SBC_targets(activities, templates, s_grid, kde_std)
        sbc_mean_var = np.mean(calculate_np_variance(sbc_post[valid_mask], s_grid))
        return (sbc_mean_var - target_mean_var)**2
        
    res_sbc = minimize_scalar(sbc_loss, bounds=(0.1, 40.0), method='bounded')
    
    return res_ppc.x, res_sbc.x

def apply_temporal_binning(activities, time_window='full', bin_size_ms=50, base_bin_ms=50):
    """
    Slices and downsamples neural activity.
    activities: numpy array of shape (n_trials, n_bins, n_neurons)
    """
    total_bins = activities.shape[1]
    
    # 1. Apply Windowing
    if time_window == 'full':
        pass # Keep all 40 bins (0.0s to 2.0s)
    elif time_window == 'half':
        activities = activities[:, total_bins//2:, :] # Keep last 20 bins (1.0s to 2.0s)
    elif time_window == 'last_quarter':
        activities = activities[:, (3*total_bins)//4:, :] # Keep last 10 bins (1.5s to 2.0s)
    else:
        raise ValueError("time_window must be 'full', 'half', or 'last_quarter'")

    # 2. Apply Binning
    if bin_size_ms == base_bin_ms:
        return activities

    group_size = int(bin_size_ms / base_bin_ms)
    n_trials, n_bins_current, n_neurons = activities.shape
    n_groups = n_bins_current // group_size

    # Truncate any remainder bins that don't fit perfectly
    activities = activities[:, :n_groups * group_size, :]
    
    # Reshape and average across the grouped dimension
    activities = activities.reshape(n_trials, n_groups, group_size, n_neurons)
    binned_activities = np.nanmean(activities, axis=2)
    
    return binned_activities

def get_generalization_split_indices(trials, split_type='generalize_contrast', random_state=42):
    """
    Creates train/test splits for Out-of-Distribution generalization testing.
    Anchors both sets with half of the baseline (High Contrast, Low Dispersion) trials.
    """
    np.random.seed(random_state)
    indices = np.arange(len(trials['orientation']))
    
    # Dynamically find the "Baseline" (Most certain) conditions
    base_cont = np.max(trials['contrast'])
    base_disp = np.min(trials['dispersion'])
    
    # Define boolean masks for the three trial types
    is_baseline = (trials['contrast'] == base_cont) & (trials['dispersion'] == base_disp)
    is_cont_manip = (trials['contrast'] < base_cont) & (trials['dispersion'] == base_disp)
    is_disp_manip = (trials['dispersion'] > base_disp) & (trials['contrast'] == base_cont)
    
    # Split the baseline trials 50/50 so both train and test sets have a "certain" anchor
    baseline_idx = indices[is_baseline]
    np.random.shuffle(baseline_idx)
    half_base = len(baseline_idx) // 2
    base_train, base_test = baseline_idx[:half_base], baseline_idx[half_base:]
    
    # Route the manipulated trials based on the split_type
    if split_type == 'generalize_contrast':
        # Train on Contrast variations, Test on Dispersion variations
        train_manip = indices[is_cont_manip]
        test_manip = indices[is_disp_manip]
        
    elif split_type == 'generalize_dispersion':
        # Train on Dispersion variations, Test on Contrast variations
        train_manip = indices[is_disp_manip]
        test_manip = indices[is_cont_manip]
        
    else:
        raise ValueError(f"Unknown generalization split_type: {split_type}")
        
    train_indices = np.concatenate([base_train, train_manip])
    test_indices = np.concatenate([base_test, test_manip])
    
    # Shuffle the final arrays so batches are mixed
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, test_indices