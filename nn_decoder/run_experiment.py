# -*- coding: utf-8 -*-
"""
VR Go/No-Go Decoder (Full Pipeline with Robust Resume & Recovery Logic)
Grouped by Hyperparameter Configuration across all animals.
"""

#%% Imports
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from utils import (
    get_stratified_train_test_indices,
    get_generalization_split_indices,
    load_vr_export,
    get_tuning_templates,
    generate_PPC_targets,
    generate_SBC_targets,
    optimize_synthetic_params,
    ToTensor,
    apply_temporal_binning,
    select_neuron_subset,
)
from neural_dataset import NeuralDataset
from nn_classifier import evaluate_model_entropy, train_and_select_best_model

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_weights(model):
    """Snapshot the trained model's first- and last-layer weights as
    numpy arrays, ready to be embedded in the returned dict and saved
    to the per-split .mat file by training.run.run_config.

    Both spatial and temporal share the SimpleFlexibleNNClassifier MLP backbone;
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


def _extract_checkpoint(model, batches, pred_probs_list, model_params,
                         model_type, pcs, explained_var,
                         loss_func, entropy_lambda):
    """Build a self-contained checkpoint dict for round-trip sanity check.

    The sanity-check script (``nn_decoder/recovery_sanity_check.py``)
    reloads this and confirms that:
      1. ``state_dict`` + ``model_params`` reconstructs the same network
         the production .mat metrics were computed from.
      2. Forward-passing the saved ``X_test`` through that network
         reproduces the saved ``pred_probs`` bit-for-bit.
      3. Computing fit-loss with the saved ``pred_probs`` as both the
         prediction AND the target lands at the fp32 floor — the
         "model perfectly reproduces its own outputs" identity check.

    Everything is moved to CPU so the checkpoint round-trips on a
    laptop after training on the cluster's GPU.

    Parameters
    ----------
    model : torch.nn.Module
        The trained ``SimpleFlexibleNNClassifier`` whose weights to snapshot.
    batches : list of torch.Tensor
        Per-trial held-out test inputs, one ``(T, n_neurons)`` tensor
        per trial, in the order the test_loader yielded them.
    pred_probs_list : list of torch.Tensor
        Per-trial predicted probabilities from
        ``get_model_probabilities(model, batch, model_type)``. Shape is
        ``(1, n_cats)`` for ``model_type='ppc'`` (time-averaged before
        softmax) and ``(T, n_cats)`` for ``'sampling'`` (per-bin softmax).
    model_params : dict
        ``{input_size, hidden_sizes, output_size, activation_function}``
        — enough to rebuild the network from scratch.
    model_type : {'ppc', 'sampling'}
        Architecture flag; the sanity script dispatches its forward
        pass on this.
    pcs, explained_var : torch.Tensor or None
        PCA basis for the loss recomputation. ``None`` for MSE/CE cells.
    loss_func : {'PCA', 'MSE', 'CE', 'JS', 'KL', 'Wasserstein'}
        Loss the model was trained on, in
        ``training.config.Config.loss_func`` notation.
    entropy_lambda : float
        Training-time entropy regulariser weight. Kept so the sanity
        script can report the diagnostic penalty.
    """
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    X_test = torch.stack([b.detach().cpu() for b in batches], dim=0)
    pred_probs = torch.stack([p.detach().cpu() for p in pred_probs_list], dim=0)
    pcs_cpu = pcs.detach().cpu() if torch.is_tensor(pcs) else None
    evar_cpu = (explained_var.detach().cpu()
                 if torch.is_tensor(explained_var) else None)
    return {
        'state_dict': state_dict,
        'X_test': X_test,
        'pred_probs': pred_probs,
        'model_params': dict(model_params),
        'model_type': model_type,
        'pcs': pcs_cpu,
        'explained_var': evar_cpu,
        'loss_func': loss_func,
        'entropy_lambda': float(entropy_lambda),
    }


#%% Pipeline helpers (extracted from run_animal_decoder)

KNOWN_WHICH_MODELS = ('perception', 'likelihood', 'stim_kernel', 'stim_cat',
                      'detection', 'decision', 'true_choice')


def fit_pca_basis(training_posteriors, stim_conditions_train, n_cats,
                  pca_basis='all_trials', flat_evar=False, shape_lambda=0.0,
                  evar_alpha=1.0, device=default_device):
    """Fit the PCA loss-basis for one cell's training targets.

    Returns ``(pcs, explained_variance)`` as torch tensors on ``device``, or
    ``(None, None)`` when ``n_cats <= 2`` (PCA loss is undefined for the 2-D
    decision target — those cells train under MSE). Shared by
    ``run_animal_decoder`` and the recovery/optuna analyses so the eval-time
    basis matches the training basis exactly.

    Parameters
    ----------
    training_posteriors : np.ndarray, shape (n_train, n_cats)
        Per-trial training target distributions.
    stim_conditions_train : np.ndarray, shape (n_train, 3)
        (orientation, contrast, dispersion) per training trial — used to form
        condition cells for the 'condition_mean' / 'residual' bases.
    n_cats : int
    pca_basis : {'all_trials', 'condition_mean', 'residual'}
        What the PCA is fit on:
          - 'all_trials' (default): raw per-trial training targets — no
            condition averaging, no cell-mean subtraction. Dominant PCs capture
            across- and within-condition variance together.
          - 'condition_mean': per-(o,c,d)-cell mean target. Dominant PCs are
            across-condition axes; the loss minimum is the per-cell training
            mean (stim_mean_baseline.py provides it closed-form). Categories are
            derived from TRAINING trials only, so OOD splits never inject
            zero-filled rows (a fixed bug). Flagged in GOTCHAS as "measures
            across-condition variation only".
          - 'residual': per-trial (target - within-cell train mean). Dominant
            PCs are within-cell trial-level axes. Cells with < 2 train trials
            contribute a zero residual row (excluded), not a global-mean
            fallback — a singleton far from the global mean would otherwise
            dominate the leading PC and re-introduce the across-condition shift
            the residual basis filters out. Verified in test_pca_basis.
    flat_evar : bool
        Replace the explained-variance weights with a uniform 1/n_pc vector so
        the PCA loss becomes the unweighted L2 / Brier loss (diagnostic control).
    shape_lambda : float
        When > 0, floor every weight by shape_lambda/100 so the loss becomes
        Σ(evar+shape_lambda/100)·err² = PCA + (shape_lambda/100)·Brier (width-matched
        fix; the clean Brier weight is λ = shape_lambda/100). Mutually exclusive
        with flat_evar.
    evar_alpha : float
        When != 1, raise every weight to this power and renormalise to sum 1
        (evar_k -> evar_k**alpha / Σ evar**alpha). alpha<1 compresses the weight
        dynamic range — the multiplicative cousin of shape_lambda, interpolating
        from plain PCA (alpha=1) to flat-evar (alpha=0). Mutually exclusive with
        flat_evar / shape_lambda (checked by the caller's if/elif order).
    """
    if n_cats <= 2:
        return None, None

    unique_train_categories = np.unique(stim_conditions_train, axis=0)
    if pca_basis == 'all_trials':
        pca_input = training_posteriors
    elif pca_basis == 'condition_mean':
        averaged = np.zeros((len(unique_train_categories), training_posteriors.shape[1]))
        for i, stimulus in enumerate(unique_train_categories):
            idx = np.where(np.all(stim_conditions_train == stimulus, axis=1))[0]
            averaged[i] = np.mean(training_posteriors[idx, :], axis=0)
        pca_input = averaged
    else:  # 'residual'
        cell_mean_per_trial = np.zeros_like(training_posteriors)
        keep_mask = np.zeros(len(training_posteriors), dtype=bool)
        for stimulus in unique_train_categories:
            idx = np.where(np.all(stim_conditions_train == stimulus, axis=1))[0]
            if len(idx) >= 2:
                cell_mean_per_trial[idx] = training_posteriors[idx].mean(axis=0)
                keep_mask[idx] = True
        residuals = training_posteriors - cell_mean_per_trial
        residuals[~keep_mask] = 0.0
        pca_input = residuals

    pca = PCA()
    pca.fit(pca_input)
    pcs = torch.tensor(pca.components_, dtype=torch.float32, device=device)
    explained_variance = torch.tensor(
        pca.explained_variance_ratio_, dtype=torch.float32, device=device)
    if flat_evar:
        # Uniform weights (sum to 1) -> with all PCs kept this is unweighted L2.
        n_pc = explained_variance.numel()
        explained_variance = torch.full_like(explained_variance, 1.0 / n_pc)
    elif shape_lambda > 0.0:
        explained_variance = explained_variance + (shape_lambda / 100.0)
    elif evar_alpha != 1.0:
        # Compress the weight dynamic range (soft cousin of shape_lambda): raise
        # every evar to alpha and renormalise to sum 1. alpha=1 is the no-op
        # (skipped); alpha->0 recovers the flat-evar uniform vector. clamp_min
        # guards against tiny negative fp noise from the PCA.
        w = explained_variance.clamp_min(0.0) ** evar_alpha
        explained_variance = w / w.sum()
    return pcs, explained_variance


def _decode_models_over_loader(loader, model_specs, loss_func, entropy_lambda,
                               pcs, explained_variance, n_cats, t_bins,
                               capture_ckpt_keys=()):
    """Decode every ``(model_type, model, key)`` spec over every batch of
    ``loader``, returning per-key accumulated arrays. The single loop body that
    used to be copy-pasted as the held-out-test pass and the full-dataset pass.

    Returns ``(acc, ckpt)`` where ``acc[key]`` has keys ``decoded`` (n, n_cats),
    ``target`` (n, n_cats), ``decoded_samp`` ((n, n_cats, t_bins); only filled
    for sampling 'temp*' keys), ``fit_loss`` (n,) and ``entropy_penalty`` (n,);
    and ``ckpt[key]`` (only for keys in ``capture_ckpt_keys``) carries the
    per-batch ``batches`` / ``pred_probs`` lists the sanity-check checkpoint needs.
    """
    from nn_classifier import get_model_probabilities
    keys = [k for _, _, k in model_specs]
    acc = {k: {'decoded': np.empty([0, n_cats]),
               'decoded_samp': np.empty([0, n_cats, t_bins]),
               'target': np.empty([0, n_cats]),
               'fit_loss': np.empty(0),
               'entropy_penalty': np.empty(0)} for k in keys}
    ckpt = {k: {'batches': [], 'pred_probs': []} for k in capture_ckpt_keys}
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            for m_type, model_obj, key in model_specs:
                fl, p_s, p_m, t_m, _, ep = evaluate_model_entropy(
                    batch_inputs, batch_targets, model_obj, loss_func,
                    entropy_lambda, m_type, pcs, explained_variance,
                )
                acc[key]['fit_loss'] = np.append(acc[key]['fit_loss'], fl.reshape(1, -1).cpu().numpy())
                acc[key]['entropy_penalty'] = np.append(acc[key]['entropy_penalty'], ep.reshape(1, -1).cpu().numpy())
                acc[key]['decoded'] = np.vstack((acc[key]['decoded'], p_m))
                acc[key]['target'] = np.vstack((acc[key]['target'], t_m))
                if 'temp' in key:
                    acc[key]['decoded_samp'] = np.vstack((acc[key]['decoded_samp'], p_s))
                if key in ckpt:
                    # Capture inputs + pred_probs for the round-trip sanity-check
                    # checkpoint (real models only). One extra forward pass/batch.
                    pp = get_model_probabilities(model_obj, batch_inputs, m_type)
                    ckpt[key]['batches'].append(batch_inputs)
                    ckpt[key]['pred_probs'].append(pp)
    return acc, ckpt


#%% Define the single-animal processing function

def run_animal_decoder(config, mouse_id, neuron_subset=None, preloaded=None):
    """Run the complete decoding pipeline for a single animal.

    Parameters
    ----------
    config : dict
        Legacy config dict (see ``training.config.Config.to_legacy_dict``).
    mouse_id : int
        Animal index passed to :func:`utils.load_vr_export`.
    neuron_subset : array-like of int, or None
        If given, train/evaluate the decoder on only this subset of the
        recorded population. ``None`` (default) uses all neurons — the
        path every pre-existing caller takes. The network input
        dimension is the neuron count, so subsetting here makes
        ``input_size`` become ``len(neuron_subset)`` automatically.
        Used by the population-scaling analysis (``neuron_scaling.py``).
    preloaded : tuple, or None
        Optional pre-loaded ``load_vr_export`` output
        ``(activities_m, targets_perc, targets_dec, targets_lik, trials)``.
        Lets a caller that runs many fits for one animal (e.g. the
        scaling sweep) load the export once instead of re-reading the
        ``.mat`` on every call. ``None`` (default) loads it here.
    """

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
    pca_basis = config.get('pca_basis', 'all_trials')
    if pca_basis not in ('all_trials', 'condition_mean', 'residual'):
        raise ValueError(f"Unknown pca_basis {pca_basis!r}")
    # flat_evar: replace the per-PC variance weights with a uniform vector so the
    # PCA loss becomes unweighted L2 (Brier). Diagnostic control only. PCA-loss.
    flat_evar = bool(config.get('flat_evar', False))
    # shape_lambda > 0: width-matched loss = PCA + shape_lambda*Brier, applied as
    # a floor on the evar weights (evar -> evar + shape_lambda/100). PCA-loss.
    shape_lambda = float(config.get('shape_lambda', 0.0))
    # evar_alpha != 1: compress the evar weight dynamic range (evar -> evar**alpha,
    # renormalised). Soft multiplicative cousin of shape_lambda. PCA-loss.
    evar_alpha = float(config.get('evar_alpha', 1.0))
    num_epochs = config['num_epochs']
    REP = config['REP']
    entropy_lambda = config['entropy_lambda']
    minibatch_size = config['minibatch_size']
    activation_function = config['activation_function']
    # optimizer_type / momentum are recorded in the config for provenance but
    # not applied here — train_and_select_best_model fixes Adam (see the
    # RECORDED-ONLY note in training/config.py), so they are not read.
    
    custom_loss_func = config['custom_loss_func']
    model_post_to_use = config['which_model']
    target_source = config.get('target_source', 'real') 
    
    time_window = config.get('time_window', 'full')
    bin_size_ms = config.get('bin_size_ms', 50)

    # 1. Import aligned data (|Δ from Go|)
    # activities_m shape is naturally (nNeurons, nTrials, tBins)
    if preloaded is not None:
        activities_m, targets_perc, targets_dec, targets_lik, trials = preloaded
    else:
        activities_m, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mouse_id)

    # Optional neuron-population subsetting (population-scaling analysis).
    # Applied to the neurons-first axis before binning / z-scoring, so
    # every downstream shape (input_size, W_in columns) follows along.
    # No-op when neuron_subset is None — the default for all other callers.
    activities_m = select_neuron_subset(activities_m, neuron_subset)

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

    # Validate the target name. (The 91-D-vs-2-D distinction this block used to
    # draw — via an `angles`/`circle_type` pair — is now read directly off
    # N_cats below; nothing downstream consumed those two variables.)
    if model_post_to_use not in KNOWN_WHICH_MODELS:
        raise ValueError(f"Unknown which_model {model_post_to_use!r}")

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

    # Optional validation carve-out from the training set. Stratified on
    # the stim cell category so val matches train's per-condition
    # composition (avoids the artefact of putting all of e.g. the high-
    # contrast trials into val). Default val_frac=0 keeps the prior
    # behaviour (no val) bit-identical. PCA basis fitting below uses
    # the post-carve sub-train only, so val never leaks into the basis.
    val_frac = float(config.get('val_frac', 0.0))
    if val_frac > 0:
        train_cats_for_carve = trial_categories_all[train_indices]
        sub_train, sub_val = get_stratified_train_test_indices(
            train_cats_for_carve, test_size=val_frac, random_state=42)
        val_indices = train_indices[sub_val]
        train_indices = train_indices[sub_train]
        print(f"  Carved val_frac={val_frac:.2f}: "
              f"{len(train_indices)} train / {len(val_indices)} val / "
              f"{len(test_indices)} test")
    else:
        val_indices = None

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

    # Val activations + targets — z-scored with the same per-neuron
    # mean/std as train and test (computed above from the post-carve
    # train_indices, so val never leaks into the z-scoring statistics).
    if val_indices is not None:
        X_val = activities_m_z[:, val_indices, :]
        X_val_in = np.copy(X_val.reshape(X_val.shape[0], -1)).T
        Y_val = target_distr_[:, val_indices, :]
        Y_val_in = np.copy(Y_val.reshape(Y_val.shape[0], -1)).T
    else:
        X_val_in = None
        Y_val_in = None
    
    # Shuffling — the chance/shuffle control: permute WHOLE trials' targets
    # against the inputs (a true trial permutation that preserves the target
    # marginal). Y_train_in is trial-major (row = trial*T + bin), so output row
    # (trial_out*T + bin) must take source row perm[trial_out]*T + bin — i.e.
    # the trial index is MULTIPLIED by T before adding the bin offset.
    shuffle_idxs = np.arange(0, N_training, 1)
    np.random.shuffle(shuffle_idxs)
    shuffle_idxs       = np.repeat(shuffle_idxs,T,axis=0) * T + np.tile(np.arange(0,T,1),N_training)
    Y_train_in_shuffle = np.copy(Y_train_in[shuffle_idxs,:])
    
    # PCA loss-basis. Categories are derived from TRAINING trials only (the OOD
    # zero-row bug fix), and the pca_basis / flat_evar / shape_lambda semantics
    # all live on fit_pca_basis. No-op -> (None, None) for 2-D decision targets.
    stim_conditions_train = stimulus_conditions_full[train_indices]
    training_posteriors = Y_train[:, :, 0].T
    pcs, explained_variance = fit_pca_basis(
        training_posteriors, stim_conditions_train, N_cats,
        pca_basis=pca_basis, flat_evar=flat_evar, shape_lambda=shape_lambda,
        evar_alpha=evar_alpha, device=default_device)

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
        'device': default_device,
        'entropy_lambda': entropy_lambda,
        'pcs': pcs,
        'explained_variance': explained_variance,
        # (momentum / optimizer_type / angles / circle_type were threaded here
        # but never read by train_and_select_best_model — dropped 2026-06-09.)
        # Diagnostic checkpointing — default off so production runs are
        # untouched. Forwarded by train_and_select_best_model to
        # fit_model when set; see training.config.Config docstring.
        'track_training_history':
            bool(config.get('track_training_history', False)),
        'weight_snapshot_every':
            int(config.get('weight_snapshot_every', 0)),
        # Early stopping — default off (patience=0 -> fixed num_epochs).
        # When >0, fit_model stops on the held-out val fit-loss and
        # restores best weights. Reuses the val_frac carve below as its
        # signal; if val_frac=0 it carves its own seeded slice.
        'patience': int(config.get('patience', 0)),
        'min_epochs': int(config.get('min_epochs', 0)),
        'val_fraction': float(config.get('val_fraction', 0.2)),
        # Validation arrays (CPU numpy, n*T flat layout matching
        # X_train_in / Y_train_in). train_and_select_best_model lifts
        # these into the trial-stacked (n_trials, T, ...) layout
        # fit_model expects, in lock-step with how it materialises
        # X_train from the legacy DataLoader. None when val_frac=0 —
        # fit_model skips val logging in that case.
        'X_val_in': X_val_in,
        'Y_val_in': Y_val_in,
        'T_bins': T,
    }

    # Model Training. The third return value is the per-epoch history
    # dict from the winning REP restart (None when tracking is off).
    # Only collected for the REAL models (spat / temp) — shuffle decoders
    # are diagnostic-only and their training curves aren't reported.
    print("      [1/4] Training temporal...")
    best_model_sampling, _, history_temp = train_and_select_best_model(
        REP, 'sampling', train_loader, model_params, training_params, verbose=False)

    print("      [2/4] Training temporal - SHUFFLED...")
    best_model_sampling_shf, _, _ = train_and_select_best_model(
        REP, 'sampling', train_loader_shuffle, model_params, training_params, verbose=False)

    print("      [3/4] Training spatial...")
    best_model_ppc, _, history_spat = train_and_select_best_model(
        REP, 'ppc', train_loader, model_params, training_params, verbose=False)

    print("      [4/4] Training spatial - SHUFFLED...")
    best_model_ppc_shf, _, _ = train_and_select_best_model(
        REP, 'ppc', train_loader_shuffle, model_params, training_params, verbose=False)

    # ------------------------------------------------------------------
    # Evaluation. Two passes share one loop body (_decode_models_over_loader):
    #   1. held-out test set -> Distr decoded/target/decoded_samp + the clean
    #      per-trial fit_loss / entropy_penalty + the sanity-check checkpoint
    #      inputs for the REAL models (spat, temp).
    #   2. FULL dataset -> Distr full_decoded (needed by recovery experiments).
    # fit_loss is the canonical held-out test loss (no entropy regulariser),
    # replacing the contaminated `KLs` field; entropy_penalty is kept as a
    # diagnostic so `fit_loss + entropy_penalty` reconstructs the legacy total.
    # See nn_decoder/audit/AUDIT_loss_consumers.md.
    # ------------------------------------------------------------------
    model_specs = [('sampling', best_model_sampling, 'temp'),
                   ('sampling', best_model_sampling_shf, 'temp_shf'),
                   ('ppc', best_model_ppc, 'spat'),
                   ('ppc', best_model_ppc_shf, 'spat_shf')]
    KEYS = [k for _, _, k in model_specs]

    test_acc, ckpt_collect = _decode_models_over_loader(
        test_loader, model_specs, custom_loss_func, entropy_lambda,
        pcs, explained_variance, N_cats, T, capture_ckpt_keys=('spat', 'temp'))
    fit_loss = {k: test_acc[k]['fit_loss'] for k in KEYS}
    entropy_penalty = {k: test_acc[k]['entropy_penalty'] for k in KEYS}
    Distr = {k: {'decoded': test_acc[k]['decoded'],
                 'decoded_samp': test_acc[k]['decoded_samp'],
                 'target': test_acc[k]['target'],
                 'full_decoded': np.empty([0, N_cats])} for k in KEYS}

    X_all_in = np.copy(activities_m_z.reshape(activities_m_z.shape[0], -1)).T
    Y_all_in = np.copy(target_distr_.reshape(target_distr_.shape[0], -1)).T
    full_loader = DataLoader(NeuralDataset(X_all_in, Y_all_in, transform=ToTensor(default_device)), batch_size=T, shuffle=False)
    full_acc, _ = _decode_models_over_loader(
        full_loader, model_specs, custom_loss_func, entropy_lambda,
        pcs, explained_variance, N_cats, T)
    for k in KEYS:
        Distr[k]['full_decoded'] = full_acc[k]['decoded']

    Distr['pcs'] = pcs.cpu().numpy() if pcs is not None else []
    Distr['explained_var'] = explained_variance.cpu().numpy() if explained_variance is not None else []

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

    # Self-contained checkpoint bundle for the round-trip sanity check
    # (see nn_decoder/recovery_sanity_check.py). One entry per REAL
    # model — shuffle controls are not included because the sanity
    # check only needs to confirm production models reproduce their own
    # outputs. Saved as part of the run output via training/run.py.
    Checkpoints = {
        'spat': _extract_checkpoint(
            model=best_model_ppc,
            batches=ckpt_collect['spat']['batches'],
            pred_probs_list=ckpt_collect['spat']['pred_probs'],
            model_params=model_params, model_type='ppc',
            pcs=pcs, explained_var=explained_variance,
            loss_func=custom_loss_func, entropy_lambda=entropy_lambda,
        ),
        'temp': _extract_checkpoint(
            model=best_model_sampling,
            batches=ckpt_collect['temp']['batches'],
            pred_probs_list=ckpt_collect['temp']['pred_probs'],
            model_params=model_params, model_type='sampling',
            pcs=pcs, explained_var=explained_variance,
            loss_func=custom_loss_func, entropy_lambda=entropy_lambda,
        ),
    }

    # Training-history sidecar — populated only when the caller set
    # config['track_training_history'] (default off). Carried alongside
    # the final-state checkpoint so it gets saved to disk by
    # training/run.py::_pop_and_save_checkpoints. Empty dicts when
    # tracking was off; None-valued entries when the winning restart
    # didn't produce a history (shouldn't happen but defensive).
    if history_spat is not None:
        Checkpoints['spat']['history'] = history_spat
    if history_temp is not None:
        Checkpoints['temp']['history'] = history_temp

    return {'fit_loss': fit_loss,
            'entropy_penalty': entropy_penalty,
            'trials': trials_out, 'trial_cats': trial_cats_out,
            'Dist': Distr, 'Weights': Weights,
            'Checkpoints': Checkpoints}
    