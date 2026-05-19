# -*- coding: utf-8 -*-
"""
VR Go/No-Go Decoder (Full Pipeline with Robust Resume & Recovery Logic)
Grouped by Hyperparameter Configuration across all animals.
"""

#%% Imports
import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from utils import (
    zscore_activity, 
    get_stratified_train_test_indices,
    get_generalization_split_indices,
    load_vr_export,
    get_tuning_templates,
    generate_PPC_targets,
    generate_SBC_targets,
    optimize_synthetic_params,
    ToTensor,
    apply_temporal_binning # <-- NEW IMPORT
)
from neural_dataset import NeuralDataset
from nn_classifier import evaluate_model_entropy, train_and_select_best_model

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_weights(model):
    """Snapshot the trained model's first- and last-layer weights as
    numpy arrays, ready to be embedded in the returned dict and saved
    to the per-split .mat file by training.run.run_config.

    Both PPC and SBC share the SimpleFlexibleNNClassifier MLP backbone;
    the first layer is the V1 -> hidden readout (what the loadings
    comparison consumes) and the last layer is the hidden -> output
    projection (what functional alignment matches on). Bias terms are
    saved alongside for completeness even though the alignment doesn't
    need them.

    Shapes (for the production single-hidden-layer backbone):
      W_in:  (hidden_size, n_neurons)
      b_in:  (hidden_size,)
      W_out: (n_output, hidden_size)
      b_out: (n_output,)
    """
    return {
        'W_in':  model.layers[0].weight.detach().cpu().numpy(),
        'b_in':  model.layers[0].bias.detach().cpu().numpy(),
        'W_out': model.layers[-1].weight.detach().cpu().numpy(),
        'b_out': model.layers[-1].bias.detach().cpu().numpy(),
    }


#%% Define the single-animal processing function

def run_animal_decoder(config, mouse_id):
    """ Runs the complete decoding pipeline for a single animal given a configuration """
    
    hidden_sizes = config['hidden_sizes']
    learning_rate = config['learning_rate']
    # weight_decay is sourced from training.config.Config.weight_decay via
    # to_legacy_dict (default 1e-4, or the per-target Optuna-tuned preset
    # e.g. 1.388e-5 for ('Q', 100ms)). Previously this field was dropped on
    # the way through, so train_and_select_best_model's hardcoded 3e-4 was
    # silently overriding the Optuna value. Default here matches the Config
    # default so hand-built test dicts get the same behaviour.
    weight_decay = config.get('weight_decay', 1e-4)
    # pca_basis controls how the PCA loss-basis is fit (PCA-loss targets
    # only; ignored for CE/MSE). See training.config.Config.pca_basis.
    pca_basis = config.get('pca_basis', 'condition_mean')
    if pca_basis not in ('condition_mean', 'residual'):
        raise ValueError(f"Unknown pca_basis {pca_basis!r}")
    num_epochs = config['num_epochs']
    REP = config['REP']
    entropy_lambda = config['entropy_lambda']
    minibatch_size = config['minibatch_size']
    momentum = config['momentum']
    optimizer_type = config['optimizer_type']
    activation_function = config['activation_function']
    
    custom_loss_func = config['custom_loss_func']
    model_post_to_use = config['which_model']
    target_source = config.get('target_source', 'real') 
    
    time_window = config.get('time_window', 'full')
    bin_size_ms = config.get('bin_size_ms', 50)

    # 1. Import aligned data (|Δ from Go|)
    # activities_m shape is naturally (nNeurons, nTrials, tBins)
    activities_m, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mouse_id)
    
    # --- Apply Temporal Sweep Parameters ---
    # Transpose to (nTrials, tBins, nNeurons) for the utility function, then back
    act_transposed = np.transpose(activities_m, (1, 2, 0))
    act_binned = apply_temporal_binning(act_transposed, time_window=time_window, bin_size_ms=bin_size_ms)
    activities_m = np.transpose(act_binned, (2, 0, 1))
    # --------------------------------------------

    # 2. Target Mapping & Generation
    if target_source in ['synthetic_ppc', 'synthetic_sbc']:
        # Using the now-binned activity for synthetic target generation
        act_for_gen = np.transpose(activities_m, (1, 2, 0)) # (nTrials, tBins, nNeurons)
        mean_act = np.nanmean(act_for_gen, axis=1)
        
        templates = get_tuning_templates(mean_act, trials)
        opt_beta, opt_kde = optimize_synthetic_params(act_for_gen, templates, targets_perc)
        
        if target_source == 'synthetic_ppc':
            generated_targets = generate_PPC_targets(act_for_gen, templates, beta=opt_beta)
        else:
            generated_targets = generate_SBC_targets(act_for_gen, templates, kde_std=opt_kde)
            
        raw_targets = generated_targets

    elif 'recovery' in target_source:
        # CROSSOVER LOGIC: Load the 'full_decoded' predictions from the base model
        if 'base_file_path' in config:
            base_file = config['base_file_path']
        else:
            base_id = config.get('base_recovery_id')
            base_file = f'population_results_config_{base_id}.mat'
            
        if not os.path.exists(base_file):
            raise FileNotFoundError(f"Missing {base_file}. Base config must be run first.")
            
        base_data = sio.loadmat(base_file, simplify_cells=True)
        t_type = 'spat' if 'spat' in target_source else 'temp'
        
        # Load the base network's full predictions across all trials as the new ground truth
        raw_targets = base_data['results'][f'mouse_{mouse_id}']['Dist'][t_type]['full_decoded']
        
    else:
        # Real Targets — single source of truth lives in
        # ``training.targets.make_target``. Adding a new target type is a
        # one-line change there, not here.
        from training.targets import make_target
        raw_targets = make_target(
            model_post_to_use, trials,
            targets_perc=targets_perc,
            targets_dec=targets_dec,
            targets_lik=targets_lik,
        )

    # Mapping Params — distributional (91-D) vs binary (2-D) targets.
    if model_post_to_use in ['perception', 'likelihood',
                              'stim_kernel', 'stim_cat']:
        angles = np.arange(0, 91, 1)
        circle_type = 'linear'
    elif model_post_to_use in ['detection', 'decision', 'true_choice']:
        angles = np.array([0, 1])
        circle_type = 'linear'
    else:
        raise ValueError(f"Unknown which_model {model_post_to_use!r}")
        
    # activities_m_z = zscore_activity(activities_m)
        
    # Stratification based on full stimulus condition
    split_type = config.get('split_type', 'stratified_balanced')
    
    # Evaluate global categories first (required for PCA extraction later)
    stimulus_conditions_full = np.array(list(zip(trials['orientation'], trials['contrast'], trials['dispersion'])))
    unique_stimulus_categories, trial_categories_all = np.unique(stimulus_conditions_full, axis=0, return_inverse=True)
    
    if split_type == 'stratified_balanced':
        # Standard random split balanced across all stimulus combinations
        train_indices, test_indices = get_stratified_train_test_indices(trial_categories_all, test_size=0.5, random_state=42)
        
    elif split_type in ['generalize_contrast', 'generalize_dispersion']:
        # Out-of-Distribution Generalization split
        train_indices, test_indices = get_generalization_split_indices(trials, split_type=split_type, random_state=42)
        
    else:
        raise ValueError(f"Unknown split_type in config: {split_type}")
        
    N_training = len(train_indices)
    T = activities_m.shape[2] # Dynamically adapts to the new binned length
    N_cats = raw_targets.shape[1]
    
    # Format target_distr_: (N_cats, nTrials, T)
    target_distr_ = np.expand_dims(raw_targets.T, axis=2) 
    target_distr_ = np.repeat(target_distr_, T, axis=2)
    
    # z-score activity based on training set only!!
    # activities_m is (nNeurons, nTrials, tBins). Index axis 1 for trials.
    # Average across trials (axis 1) and time bins (axis 2) to get per-neuron stats.
    mean_train = np.mean(activities_m[:, train_indices, :], axis=(1, 2), keepdims=True)
    std_train = np.std(activities_m[:, train_indices, :], axis=(1, 2), keepdims=True)
    std_train[std_train == 0] = 1.0
    
    activities_m_z = (activities_m - mean_train) / std_train
        
    X_train = activities_m_z[:, train_indices, :]
    X_train_in = np.copy( X_train.reshape(X_train.shape[0],-1) ).T
    
    Y_train    = target_distr_[:,train_indices,:]
    Y_train_in = np.copy( Y_train.reshape(Y_train.shape[0],-1) ).T
    
    X_test = activities_m_z[:, test_indices, :]
    X_test_in = np.copy( X_test.reshape(X_test.shape[0],-1) ).T
    Y_test    = target_distr_[:,test_indices,:]
    Y_test_in = np.copy( Y_test.reshape(Y_test.shape[0],-1) ).T
    
    # Shuffling
    shuffle_idxs = np.arange(0, N_training, 1)
    np.random.shuffle(shuffle_idxs)
    shuffle_idxs       = np.repeat(shuffle_idxs,T,axis=0) + np.tile(np.arange(0,T,1),N_training)
    Y_train_in_shuffle = np.copy(Y_train_in[shuffle_idxs,:])
    
    # ------------------------------------------------------------------
    # PCA Baseline Extraction
    #
    # BUG FIX: Previously, unique_stimulus_categories was derived from ALL
    # trials (train + test). For OOD generalization splits, stimulus conditions
    # that exist only in the test set produced zero-filled rows in
    # averaged_distributions, which corrupted the PCA subspace.
    #
    # Fix: derive unique categories from TRAINING trials only. Every category
    # in unique_train_categories is guaranteed to have at least one training
    # trial, so averaged_distributions contains no zero rows.
    #
    # pca_basis controls *what* the PCA is fit on:
    #   - 'condition_mean': per-(o,c,d)-cell mean Q (historical default).
    #     Dominant PCs are across-condition Q axes; loss minimum is the
    #     per-cell training mean, which stim_mean_baseline.py provides
    #     closed-form. This is the basis flagged in GOTCHAS as "measures
    #     across-condition variation only".
    #   - 'residual': per-trial (Q_trial - cond_mean_train_Q_in_cell).
    #     Dominant PCs are within-cell trial-level axes; loss scores
    #     trial-by-trial information stim_mean cannot capture.
    # Cells with fewer than 2 training trials fall back to subtracting the
    # global training-set mean (so the residual isn't degenerately zero).
    # ------------------------------------------------------------------
    stim_conditions_train = stimulus_conditions_full[train_indices]
    training_posteriors = Y_train[:, :, 0].T

    unique_train_categories = np.unique(stim_conditions_train, axis=0)
    averaged_distributions = np.zeros((len(unique_train_categories), training_posteriors.shape[1]))

    for i, stimulus in enumerate(unique_train_categories):
        condition_indices = np.where(np.all(stim_conditions_train == stimulus, axis=1))[0]
        # All entries in unique_train_categories are guaranteed to have at least
        # one match in stim_conditions_train by construction, so no guard needed.
        averaged_distributions[i] = np.mean(training_posteriors[condition_indices, :], axis=0)

    if N_cats > 2:
        if pca_basis == 'condition_mean':
            pca_input = averaged_distributions
        else:  # 'residual'
            # For each well-populated cell (>= 2 train trials) subtract the
            # within-cell mean to isolate trial-level deviation. Singleton
            # cells (1 train trial) get a zero residual row — they have no
            # within-cell deviation to estimate, so they contribute nothing
            # to the PCA. The alternative (subtract a global mean) lets a
            # singleton that sits far from the global mean inject an
            # outlier row that can dominate the leading PC and re-introduce
            # the across-condition shift the residual basis is supposed to
            # filter out. Verified in test_pca_basis with a planted
            # singleton outlier.
            cell_mean_per_trial = np.zeros_like(training_posteriors)
            keep_mask = np.zeros(len(training_posteriors), dtype=bool)
            for i, stimulus in enumerate(unique_train_categories):
                condition_indices = np.where(np.all(stim_conditions_train == stimulus, axis=1))[0]
                if len(condition_indices) >= 2:
                    cell_mean_per_trial[condition_indices] = training_posteriors[condition_indices].mean(axis=0)
                    keep_mask[condition_indices] = True
                # else: leave cell_mean as zeros and keep_mask as False;
                # those trials contribute zero residual rows below.
            residuals = training_posteriors - cell_mean_per_trial
            residuals[~keep_mask] = 0.0
            pca_input = residuals

        pca = PCA()
        pca.fit(pca_input)
        pcs = torch.tensor(pca.components_, dtype=torch.float32, device=default_device)
        explained_variance = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32, device=default_device)
    else:
        pcs = None
        explained_variance = None

    # Datasets
    training_set         = NeuralDataset(X_train_in,Y_train_in,transform=ToTensor(default_device))
    training_set_shuffle = NeuralDataset(X_train_in,Y_train_in_shuffle,transform=ToTensor(default_device))
    test_set             = NeuralDataset(X_test_in,Y_test_in,transform=ToTensor(default_device))
    
    train_loader         = DataLoader(training_set, batch_size=T, shuffle=False)
    train_loader_shuffle = DataLoader(training_set_shuffle, batch_size=T, shuffle=False)
    test_loader          = DataLoader(test_set, batch_size=T, shuffle=False)    
    
    input_size  = X_train_in.shape[1]
    output_size = target_distr_.shape[0]
    
    model_params = {
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size,
        'activation_function': activation_function
    }

    training_params = {
        'loss_func': custom_loss_func,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'minibatch_size': minibatch_size,
        'momentum': momentum,
        'device': default_device,
        'entropy_lambda': entropy_lambda,
        'pcs': pcs,
        'explained_variance': explained_variance,
        'angles': angles,
        'circle_type': circle_type,
        'optimizer_type': optimizer_type
    }

    # Model Training
    print("      [1/4] Training SBC...")
    best_model_sampling, _ = train_and_select_best_model(REP, 'sampling', train_loader, model_params, training_params, verbose=False)
    
    print("      [2/4] Training SBC - SHUFFLED...")
    best_model_sampling_shf, _ = train_and_select_best_model(REP, 'sampling', train_loader_shuffle, model_params, training_params, verbose=False)
    
    print("      [3/4] Training PPC...")
    best_model_ppc, _ = train_and_select_best_model(REP, 'ppc', train_loader, model_params, training_params, verbose=False)
    
    print("      [4/4] Training PPC - SHUFFLED...")
    best_model_ppc_shf, _ = train_and_select_best_model(REP, 'ppc', train_loader_shuffle, model_params, training_params, verbose=False)

    # Initialize storage dictionaries
    Distr = {'temp': [], 'temp_shf': [], 'spat': [], 'spat_shf': []}
    for key in list(Distr.keys()):
        Distr[key] = {'decoded': np.empty([0,N_cats]), 'decoded_samp': np.empty([0,N_cats,T]), 'target': np.empty([0,N_cats]), 'full_decoded': np.empty([0,N_cats])}
    Losses = {'temp': [], 'temp_shf': [], 'spat': [], 'spat_shf': []}
    
    Distr['pcs'] = pcs.cpu().numpy() if pcs is not None else []
    Distr['explained_var'] = explained_variance.cpu().numpy() if explained_variance is not None else []
        
    # Evaluation on Held-Out Test Set
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            for m_type, model_obj, key in [('sampling', best_model_sampling, 'temp'), 
                                           ('sampling', best_model_sampling_shf, 'temp_shf'), 
                                           ('ppc', best_model_ppc, 'spat'), 
                                           ('ppc', best_model_ppc_shf, 'spat_shf')]:
                l, p_s, p_m, t_m, _ = evaluate_model_entropy(batch_inputs, batch_targets, model_obj, custom_loss_func, entropy_lambda, m_type, pcs, explained_variance, angles, circle_type, default_device)
                Losses[key] = np.append(Losses[key], l.reshape(1,-1).cpu().numpy())  
                Distr[key]["decoded"] = np.vstack((Distr[key]["decoded"], p_m))
                Distr[key]["target"] = np.vstack((Distr[key]["target"], t_m))
                if 'temp' in key: Distr[key]["decoded_samp"] = np.vstack((Distr[key]["decoded_samp"], p_s))

    # Evaluate on FULL dataset (required for recovery experiments downstream)
    X_all_in = np.copy(activities_m_z.reshape(activities_m_z.shape[0], -1)).T
    Y_all_in = np.copy(target_distr_.reshape(target_distr_.shape[0], -1)).T
    full_loader = DataLoader(NeuralDataset(X_all_in, Y_all_in, transform=ToTensor(default_device)), batch_size=T, shuffle=False)

    with torch.no_grad():
        for batch_inputs, batch_targets in full_loader:
            for m_type, model_obj, key in [('sampling', best_model_sampling, 'temp'), 
                                           ('sampling', best_model_sampling_shf, 'temp_shf'), 
                                           ('ppc', best_model_ppc, 'spat'), 
                                           ('ppc', best_model_ppc_shf, 'spat_shf')]:
                _, _, p_m, _, _ = evaluate_model_entropy(batch_inputs, batch_targets, model_obj, custom_loss_func, entropy_lambda, m_type, pcs, explained_variance, angles, circle_type, default_device)
                Distr[key]["full_decoded"] = np.vstack((Distr[key]["full_decoded"], p_m))

    trials_out = {
        'orientation': np.copy(trials['orientation'][test_indices]),
        'dispersion': np.copy(trials['dispersion'][test_indices]),
        'contrast': np.copy(trials['contrast'][test_indices])
    }
    trial_cats_out = {'orientation': np.copy(trial_categories_all[test_indices])}

    # Trained weights snapshot — consumed downstream by the loadings-
    # comparison module (decoder_loadings_comparison.py) and any future
    # interpretability work. Mirrors the four-key Distr structure so the
    # shuffle controls are saved alongside the real models.
    Weights = {
        'spat':     _extract_weights(best_model_ppc),
        'spat_shf': _extract_weights(best_model_ppc_shf),
        'temp':     _extract_weights(best_model_sampling),
        'temp_shf': _extract_weights(best_model_sampling_shf),
    }

    return {'KLs': Losses, 'trials': trials_out, 'trial_cats': trial_cats_out,
            'Dist': Distr, 'Weights': Weights}
    