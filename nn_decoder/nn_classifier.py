# -*- coding: utf-8 -*-
"""
Neural Network Classifier and Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import torch.nn.init as init

# ==========================================
# 1. Divergence and Loss Metrics
# ==========================================

def cross_entropy(X, Y):
    """ X is Prediction, Y is Target """
    return -torch.sum(Y * torch.log(X + torch.finfo(torch.float32).eps), dim=-1)

def entropy_calc(Y):
    """ Calculates entropy of a distribution Y """
    return -torch.sum(Y * torch.log(Y + torch.finfo(torch.float32).eps), dim=-1)

def KL_calc(X, Y):
    """
    Forward KL Divergence: D_KL(Target || Prediction)
    X: Prediction, Y: Target
    """
    KL = cross_entropy(X, Y) - entropy_calc(Y)
    return torch.clamp(KL, min=0.0)

def JS_calc(X, Y):
    """
    Jensen-Shannon Divergence
    X: Prediction, Y: Target
    """
    M = 0.5 * (X + Y)
    # D_KL(Target || M) -> KL_calc(M, Target)
    kl_xm = KL_calc(M, X) 
    kl_ym = KL_calc(M, Y)
    return 0.5 * kl_xm + 0.5 * kl_ym

def Wasserstein_calc_1D(X, Y):
    """
    1D Wasserstein Distance (Earth Mover's Distance)
    Calculated as the L1 distance between the Cumulative Distribution Functions (CDFs).
    """
    cdf_X = torch.cumsum(X, dim=-1)
    cdf_Y = torch.cumsum(Y, dim=-1)
    w_dist = torch.sum(torch.abs(cdf_X - cdf_Y), dim=-1)
    return w_dist

# ==========================================
# 2. Neural Network Architectures
# ==========================================

class NN_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN_classifier, self).__init__()
        # Standard configuration as per legacy architecture
        if isinstance(hidden_size, list):
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], output_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SimpleFlexibleNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        Initializes a neural network with flexible hidden layers and activations.
        """
        super(SimpleFlexibleNNClassifier, self).__init__()
        
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation.lower(), nn.ReLU()) 
        
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        for layer in self.layers:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x) 
        return x

# ==========================================
# 3. Model Forward and Evaluation Logic
# ==========================================

def get_model_probabilities(model, batch_inputs, model_type):
    if model_type == 'ppc':
        integrated_inputs = torch.mean(batch_inputs, dim=0, keepdim=True)
        logits = model(integrated_inputs)
        probs = F.softmax(logits, dim=-1)
    elif model_type == 'sampling':
        logits = model(batch_inputs) 
        probs = F.softmax(logits, dim=-1)
    return probs

def custom_loss_all_H(pred_probs, targets, entropy_lambda, model_type, pcs=None, explained_variance=None, loss_func_type='JS'):
    """Compute the training objective, decomposed into its fit and
    regularisation components.

    Returns
    -------
    total_loss : torch.Tensor (0-d)
        ``fit_loss + entropy_penalty``. The training loop backprops on
        this and ``train_and_select_best_model`` uses it for rep
        selection — the same scalar the legacy API returned as its first
        element.
    fit_loss : torch.Tensor (0-d)
        Pure divergence between pred_probs and targets, independent of
        ``entropy_lambda``. This is the metric that should be reported at
        eval time (e.g. saved in the .mat as the held-out test loss).
    entropy_penalty : torch.Tensor (0-d)
        ``entropy_lambda * mean(H(pred_probs))`` for ``model_type='sampling'``;
        always 0 for ``model_type='ppc'`` (PPC branch never has a
        sharpness penalty by design).

    Before this split, ``custom_loss_all_H`` returned
    ``(total_loss, entropy_log_val)`` where ``total_loss`` baked the
    penalty into the fit-loss. Because ``evaluate_model_entropy`` passes
    the production ``entropy_lambda`` through verbatim, the saved
    ``KLs[temp]`` arrays in every pre-2026-05-19 .mat carried this
    contamination (a few % for PCA-loss cells; smaller for MSE/CE).
    See ``nn_decoder/audit/AUDIT_loss_consumers.md``.
    """
    # 1. Route the Logic based on Architecture
    if model_type == 'sampling':
        # Calculate Instantaneous Entropy BEFORE averaging
        instantaneous_entropy = torch.mean(entropy_calc(pred_probs))

        # Average the predictions for the base divergence loss
        pred_probs_loss = torch.mean(pred_probs, dim=0, keepdim=True)

        # Apply the sharpness penalty
        entropy_penalty = entropy_lambda * instantaneous_entropy
        # `entropy_lambda` is sometimes a Python float; in that case the
        # multiplication above produces a Python float (lambda=0 -> 0.0)
        # or a 0-d tensor (lambda>0). Normalise to a 0-d tensor so the
        # return contract is uniform.
        if not torch.is_tensor(entropy_penalty):
            entropy_penalty = torch.tensor(float(entropy_penalty),
                                            device=pred_probs.device,
                                            dtype=pred_probs.dtype)

    else:  # model_type == 'ppc'
        pred_probs_loss = pred_probs

        # PPC branch never gets a sharpness penalty. Return a 0-d
        # tensor on the same device/dtype as pred_probs so the
        # signature stays uniform with the sampling branch.
        entropy_penalty = torch.zeros((), device=pred_probs.device,
                                        dtype=pred_probs.dtype)

    targets_mean = torch.mean(targets, dim=0, keepdim=True)

    # 2. Calculate Base Divergence
    if loss_func_type == 'JS':
        loss_val = JS_calc(pred_probs_loss, targets_mean)
    elif loss_func_type == 'KL':
        loss_val = KL_calc(pred_probs_loss, targets_mean)
    elif loss_func_type == 'Wasserstein':
        loss_val = Wasserstein_calc_1D(pred_probs_loss, targets_mean)
    elif loss_func_type == 'PCA':
        if pcs is None or explained_variance is None:
            raise ValueError(
                "custom_loss_all_H: loss_func_type='PCA' requires a PCA "
                "basis, but pcs/explained_variance is None. PCA loss is "
                "only defined for multi-dimensional targets (>2 categories) "
                "— for 2-D targets set the loss to 'MSE' explicitly. "
                "Previously this condition fell through to cross-entropy, "
                "silently training a different objective than requested. "
                "See nn_decoder/pca_loss.py."
            )
        # NOTE: torch twin of pca_loss.pca_distance — keep the two in sync
        # (tests/test_pca_loss.py pins their numerical agreement).
        pred_proj = torch.matmul(pred_probs_loss, pcs.T)
        target_proj = torch.matmul(targets_mean, pcs.T)
        loss_val = torch.sum(explained_variance * (pred_proj - target_proj)**2, dim=-1) * 100
    elif loss_func_type == 'MSE':
        # Mean Squared Error — suitable for low-dimensional soft targets (e.g. 2D decision posterior)
        loss_val = torch.mean((pred_probs_loss - targets_mean)**2, dim=-1)
    else:
        loss_val = cross_entropy(pred_probs_loss, targets_mean)

    fit_loss = torch.mean(loss_val)

    # 3. Total Loss: Base Divergence + (Conditional) Sharpness Penalty
    total_loss = fit_loss + entropy_penalty

    return total_loss, fit_loss, entropy_penalty


def evaluate_model_entropy(batch_inputs, batch_targets, model, loss_func_type, entropy_lambda, model_type, pcs, explained_variance, angles, circle_type, device):
    """Evaluate ``model`` on one batch, returning the clean fit-loss and
    the entropy penalty as separate values.

    Returns
    -------
    fit_loss : torch.Tensor (0-d)
        Held-out fit-loss for reporting (no entropy regulariser added).
        This is what should be stored in the .mat as the test-set loss.
    pred_samp : np.ndarray
        Per-bin predicted distributions for the sampling architecture
        (shape ``(1, n_cats, n_bins)``); zeros for ppc.
    pred_m : np.ndarray
        Time-averaged predicted distribution, shape ``(1, n_cats)``.
    targ_m : np.ndarray
        Time-averaged target distribution, shape ``(1, n_cats)``.
    cv_val : np.ndarray
        Placeholder (unused), kept for legacy unpacking.
    entropy_penalty : torch.Tensor (0-d)
        The training-time sharpness regulariser
        ``entropy_lambda * mean(H(pred_probs))``; always 0 for ppc.
        Saved alongside fit_loss as a diagnostic.
    """
    model.eval()

    with torch.no_grad():
        pred_probs = get_model_probabilities(model, batch_inputs, model_type)

        _, fit_loss, entropy_penalty = custom_loss_all_H(
            pred_probs, batch_targets, entropy_lambda, model_type,
            pcs, explained_variance, loss_func_type,
        )

        if model_type == 'sampling':
            # This captures the instantaneous samples for your heatmaps! Shape: (1, n_angles, n_bins)
            pred_samp = np.expand_dims(pred_probs.cpu().numpy().transpose(1,0), axis=0)
            pred_m = torch.mean(pred_probs, dim=0).reshape(1,-1).cpu().numpy()
        else:
            # PPC has no instantaneous samples, returning zeros of matching shape
            pred_samp = np.zeros((1, batch_targets.shape[1], batch_inputs.shape[0]))
            pred_m = pred_probs.reshape(1,-1).cpu().numpy()

        targ_m = torch.mean(batch_targets, dim=0).reshape(1,-1).cpu().numpy()
        cv_val = np.zeros(1)

    return fit_loss, pred_samp, pred_m, targ_m, cv_val, entropy_penalty


# ==========================================
# 4. Vectorised training primitives
# ==========================================
# fit_model / evaluate replace the per-trial DataLoader loop shared by
# train_and_select_best_model and optuna_per_target.py::train_eval. The
# whole per-mouse dataset is a single device tensor; a minibatch of `mb`
# trials is one batched forward/backward over a (mb, T, n_neurons) slab.
# The time axis T is reduced exactly where the legacy code reduced it --
# input-mean for PPC, output-distribution-mean for SBC -- so each trial
# still yields one per-trial loss and one per-trial gradient contribution.
#
# Two deliberate corrections vs. the legacy loop:
#   1. clip_grad_norm_ is applied once per minibatch (before step()), not
#      after every trial's partially-accumulated gradient.
#   2. each minibatch loss is the MEAN over its trials, so the trailing
#      partial minibatch is averaged over its actual count -- the legacy
#      loop divided every trial by the full `mb`, underweighting the
#      trailing chunk by k/mb.

def _batched_predict(model, xb, model_type):
    """Vectorised counterpart of get_model_probabilities for a batch of trials.

    Parameters
    ----------
    xb : torch.Tensor, shape (B, T, n_neurons)
        B trials, each T time bins of population activity.
    model_type : {'ppc', 'sampling'}

    Returns
    -------
    pred : torch.Tensor, shape (B, n_cats)
        Per-trial predicted distribution. PPC averages the input over time
        then decodes once; SBC decodes every bin then averages the output
        distributions over time -- identical to get_model_probabilities plus
        the pred_probs_loss reduction inside custom_loss_all_H.
    entropy : torch.Tensor or None, shape (B,)
        Per-trial mean-over-time entropy of the per-bin distributions, for
        the SBC sharpness penalty. None for PPC (no penalty by design).
    """
    if model_type == 'ppc':
        integrated = torch.mean(xb, dim=1)                  # (B, n_neurons)
        probs = F.softmax(model(integrated), dim=-1)        # (B, n_cats)
        return probs, None
    elif model_type == 'sampling':
        probs = F.softmax(model(xb), dim=-1)                # (B, T, n_cats)
        pred = torch.mean(probs, dim=1)                     # (B, n_cats)
        entropy = torch.mean(entropy_calc(probs), dim=1)    # (B,)
        return pred, entropy
    raise ValueError(f"unknown model_type {model_type!r}")


def _batched_fit_loss(pred, target, loss_func_type, pcs=None,
                      explained_variance=None):
    """Per-trial fit-loss for a batch -- the divergence branch of
    custom_loss_all_H evaluated per row instead of mean-reduced.

    pred, target : (B, n_cats). Returns (B,) per-trial fit loss. Branch
    selection mirrors custom_loss_all_H exactly, including the PCA ``* 100``
    scale and the fall-through to cross-entropy when loss_func_type='PCA'
    but pcs is None.
    """
    if loss_func_type == 'JS':
        return JS_calc(pred, target)
    elif loss_func_type == 'KL':
        return KL_calc(pred, target)
    elif loss_func_type == 'Wasserstein':
        return Wasserstein_calc_1D(pred, target)
    elif loss_func_type == 'PCA' and pcs is not None:
        pred_proj = torch.matmul(pred, pcs.T)
        target_proj = torch.matmul(target, pcs.T)
        return torch.sum(
            explained_variance * (pred_proj - target_proj) ** 2, dim=-1) * 100
    elif loss_func_type == 'MSE':
        return torch.mean((pred - target) ** 2, dim=-1)
    else:
        return cross_entropy(pred, target)


def _batched_total_loss(model, xb, yb, model_type, loss_func, pcs,
                        explained_variance, entropy_lambda):
    """Per-trial total loss (fit + SBC sharpness penalty) for a batch.

    xb : (B, T, n_neurons) ; yb : (B, T, n_cats). The target is reduced over
    time by the same mean custom_loss_all_H applies. Returns (B,).
    """
    pred, entropy = _batched_predict(model, xb, model_type)
    target = torch.mean(yb, dim=1)                          # (B, n_cats)
    fit = _batched_fit_loss(pred, target, loss_func, pcs, explained_variance)
    if model_type == 'sampling':
        return fit + entropy_lambda * entropy
    return fit


def fit_model(model, optimizer, X_train, Y_train, *,
              model_type, loss_func, pcs, explained_variance,
              entropy_lambda, minibatch_size, num_epochs, max_grad_norm=1.0,
              patience=0, min_epochs=0, val_fraction=0.2):
    """Train ``model`` in place, vectorised over minibatches of trials.

    Parameters
    ----------
    X_train : (n_trials, T, n_neurons) ; Y_train : (n_trials, T, n_cats)
        On the model's device. Y is per-bin; it is mean-reduced over T
        internally, matching custom_loss_all_H's ``targets_mean``.
    model_type : {'ppc', 'sampling'}
    loss_func : {'PCA', 'MSE', 'CE', 'JS', 'KL', 'Wasserstein'}
    pcs, explained_variance : PCA basis for loss_func='PCA', else None.
    entropy_lambda : SBC sharpness weight (ignored for PPC).
    minibatch_size : trials per optimizer step.
    num_epochs : passes over the training set (the upper bound on epochs
        when early stopping is enabled).
    max_grad_norm : gradient clipped to this norm once per minibatch.
    patience : int, default 0
        Early stopping. ``0`` disables it entirely -- the loop is then the
        original fixed ``num_epochs`` pass over all training trials, byte
        for byte (no validation holdout, no weight snapshots), so existing
        runs are unchanged. When ``> 0``, a seeded ``val_fraction`` slice of
        the training trials is held out; after each epoch the validation
        *fit*-loss (entropy_lambda=0, so the sharpness penalty never leaks
        into the stopping signal) is measured, and training stops once it
        has not improved for ``patience`` consecutive epochs. The weights
        with the lowest validation fit-loss are restored before returning.
    min_epochs : int, default 0
        Floor on epochs before early stopping may trigger (ignored when
        ``patience == 0``).
    val_fraction : float, default 0.2
        Fraction of training trials reserved for the early-stopping
        validation signal (ignored when ``patience == 0``).

    Returns the trained model (same object).
    """
    n_trials = X_train.shape[0]

    # ---- Fixed-schedule path (patience=0): unchanged from the original. ----
    if patience <= 0:
        model.train()
        for _ in range(num_epochs):
            for s in range(0, n_trials, minibatch_size):
                e = min(s + minibatch_size, n_trials)
                total = _batched_total_loss(
                    model, X_train[s:e], Y_train[s:e], model_type,
                    loss_func, pcs, explained_variance, entropy_lambda)
                loss = total.mean()     # mean over this minibatch's trials
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        return model

    # ---- Early-stopping path: carve a seeded validation slice. ----
    # A dedicated generator keeps the shuffle independent of the global RNG
    # (so model init / rep restarts are unaffected) and reproducible.
    g = torch.Generator(device='cpu').manual_seed(1234)
    perm = torch.randperm(n_trials, generator=g).to(X_train.device)
    n_val = max(1, int(round(n_trials * val_fraction)))
    # Guard tiny training sets: never leave the training slice empty.
    n_val = min(n_val, n_trials - 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
    X_val, Y_val = X_train[val_idx], Y_train[val_idx]
    n_tr = X_tr.shape[0]

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for s in range(0, n_tr, minibatch_size):
            e = min(s + minibatch_size, n_tr)
            total = _batched_total_loss(
                model, X_tr[s:e], Y_tr[s:e], model_type,
                loss_func, pcs, explained_variance, entropy_lambda)
            loss = total.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Validation fit-loss (entropy_lambda=0): the pure held-out metric.
        val_loss = evaluate(
            model, X_val, Y_val, model_type=model_type, loss_func=loss_func,
            pcs=pcs, explained_variance=explained_variance,
            entropy_lambda=0.0, reduction='mean')

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, X, Y, *, model_type, loss_func, pcs, explained_variance,
             entropy_lambda=0.0, reduction='mean'):
    """Vectorised held-out evaluation. Returns the reduced total loss as a float.

    reduction='mean' -- mean over trials (Optuna validation metric; used with
        entropy_lambda=0 so total == fit).
    reduction='sum'  -- sum over trials (rep-selection score in
        train_and_select_best_model; used with the real entropy_lambda).

    X : (n_trials, T, n_neurons) ; Y : (n_trials, T, n_cats), on device.
    """
    if reduction not in ('mean', 'sum'):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    model.eval()
    with torch.no_grad():
        total = _batched_total_loss(
            model, X, Y, model_type, loss_func, pcs,
            explained_variance, entropy_lambda)
        reduced = total.mean() if reduction == 'mean' else total.sum()
        return float(reduced.item())


# ==========================================
# 5. Training loop (REP-restart wrapper)
# ==========================================

def train_and_select_best_model(REP, model_type, train_loader, model_params, training_params, verbose=True):
    """REP random restarts; keep the model with the lowest training-set
    score. Per-restart training is delegated to the vectorised ``fit_model``
    -- this wrapper only does restart bookkeeping.

    ``train_loader`` is the legacy interface: a DataLoader with
    ``batch_size=T, shuffle=False`` over a NeuralDataset, so each batch is
    one trial's ``(T, n_neurons)`` activity and ``(T, n_cats)`` target. It
    is materialised once into the ``(n_trials, T, ...)`` tensors fit_model
    expects -- the DataLoader is plumbing, not part of the training maths.
    """
    input_size = model_params['input_size']
    output_size = model_params['output_size']
    hidden_sizes = model_params['hidden_sizes']

    # Safely extract activation, default to 'relu' if not provided
    activation = model_params.get('activation_function', 'relu')
    device = training_params['device']
    minibatch_size = training_params.get('minibatch_size', 32)
    loss_func = training_params['loss_func']
    pcs = training_params['pcs']
    explained_variance = training_params['explained_variance']
    entropy_lambda = training_params['entropy_lambda']
    num_epochs = training_params['num_epochs']

    # Materialise the per-trial DataLoader into single (n_trials, T, ...)
    # device tensors. batch_size=T + shuffle=False => each batch is one
    # trial in order, so stacking preserves trial order. The loader's
    # ToTensor transform has already placed each batch on `device`.
    #
    # Iterating a DataLoader draws once from the global RNG (the iterator's
    # base seed). The legacy loop built each model *before* touching the
    # loader, so we save/restore the RNG state around the iteration --
    # keeping model init independent of data loading. Without this, a
    # seeded caller would get different random restarts than the
    # pre-rewrite code.
    _rng_state = torch.get_rng_state()
    X_list, Y_list = [], []
    for batch_inputs, batch_targets in train_loader:
        X_list.append(batch_inputs)
        Y_list.append(batch_targets)
    torch.set_rng_state(_rng_state)
    X_train = torch.stack(X_list)      # (n_trials, T, n_neurons)
    Y_train = torch.stack(Y_list)      # (n_trials, T, n_cats)

    best_overall_loss = float('inf')
    best_overall_model = None

    for r in range(REP):
        # Instantiate using the flexible architecture!
        model = SimpleFlexibleNNClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation
        ).to(device)

        # weight_decay is read from training_params (sourced from
        # training.config.Config.weight_decay via run_experiment's
        # config -> training_params plumbing). Default 1e-4 matches the
        # Config default; the historical hardcoded 3e-4 silently overrode
        # the Optuna-tuned per-target value (e.g. 1.388e-5 for Q-100ms),
        # so the regression is guarded by
        # test_training_config::test_to_legacy_dict_carries_weight_decay.
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params.get('weight_decay', 1e-4),
        )

        fit_model(
            model, optimizer, X_train, Y_train,
            model_type=model_type, loss_func=loss_func,
            pcs=pcs, explained_variance=explained_variance,
            entropy_lambda=entropy_lambda,
            minibatch_size=minibatch_size, num_epochs=num_epochs,
            patience=training_params.get('patience', 0),
            min_epochs=training_params.get('min_epochs', 0),
            val_fraction=training_params.get('val_fraction', 0.2),
        )

        # Rep-selection score: total loss (fit + penalty) at the real
        # entropy_lambda, summed over the training trials -- the same
        # metric the legacy loop accumulated.
        rep_loss = evaluate(
            model, X_train, Y_train,
            model_type=model_type, loss_func=loss_func,
            pcs=pcs, explained_variance=explained_variance,
            entropy_lambda=entropy_lambda, reduction='sum',
        )

        if rep_loss < best_overall_loss:
            best_overall_loss = rep_loss
            best_overall_model = copy.deepcopy(model)

        if verbose:
            print(f"    Rep {r+1}/{REP} | Loss: {rep_loss:.4f} | Best: {best_overall_loss:.4f}")

    if verbose:
        print(f"  -> Best {model_type.upper()} Loss: {best_overall_loss:.4f}\n")

    return best_overall_model, best_overall_loss