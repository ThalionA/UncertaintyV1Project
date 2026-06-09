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

# The legacy fixed-2-layer ``NN_classifier`` lived here; it was never
# instantiated anywhere (superseded by the flexible backbone below) and was
# removed 2026-06-09. Recover it from git history if ever needed.

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
        always 0 for ``model_type='ppc'`` (spatial branch never has a
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

        # spatial branch never gets a sharpness penalty. Return a 0-d
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


def evaluate_model_entropy(batch_inputs, batch_targets, model, loss_func_type, entropy_lambda, model_type, pcs, explained_variance):
    """Evaluate ``model`` on one batch, returning the clean fit-loss and
    the entropy penalty as separate values.

    (Previously also took ``angles, circle_type, device`` — all three were
    unused: the batch is already on-device and the loss branches never touch
    the angle grid. Dropped 2026-06-09 to narrow the signature.)

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
            # spatial has no instantaneous samples, returning zeros of matching shape
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
# input-mean for spatial, output-distribution-mean for temporal -- so each trial
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
        Per-trial predicted distribution. spatial averages the input over time
        then decodes once; temporal decodes every bin then averages the output
        distributions over time -- identical to get_model_probabilities plus
        the pred_probs_loss reduction inside custom_loss_all_H.
    entropy : torch.Tensor or None, shape (B,)
        Per-trial mean-over-time entropy of the per-bin distributions, for
        the temporal sharpness penalty. None for spatial (no penalty by design).
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


def fit_loss_per_trial(pred, target, loss_func_type, pcs=None,
                       explained_variance=None):
    """Per-trial held-out fit-loss for a batch of (time-reduced) distributions
    — the **public eval entry point** for scoring saved posteriors.

    This is the divergence branch of :func:`custom_loss_all_H` evaluated per
    row instead of mean-reduced: given ``pred, target`` of shape ``(B, n_cats)``
    it returns the ``(B,)`` vector of per-trial losses under ``loss_func_type``.
    Analysis scripts import it to score held-out decoded/target arrays with
    *exactly* the maths the training objective used, so eval-time numbers match
    the loss the model was trained on. The PCA branch equals the numpy
    :func:`pca_loss.pca_distance` — both agreements are pinned by
    ``tests/test_pca_loss.py``.

    (Historically named ``_batched_fit_loss`` and underscore-private, yet
    imported by several analysis scripts; renamed to a public name. The old
    name remains as a backwards-compatibility alias below.)

    Parameters
    ----------
    pred, target : torch.Tensor, shape (B, n_cats)
        Per-trial predicted and target distributions, time already reduced.
        Symmetric for the squared-distance losses (PCA/MSE).
    loss_func_type : {'PCA', 'MSE', 'CE', 'JS', 'KL', 'Wasserstein'}
        Any other string falls through to cross-entropy (the historical default).
    pcs, explained_variance : torch.Tensor, optional
        PCA basis; required only for ``loss_func_type='PCA'``.

    Returns
    -------
    torch.Tensor, shape (B,)
        Per-trial fit-loss. Branch selection mirrors ``custom_loss_all_H``
        exactly, including the PCA ``* 100`` scale.

    Raises
    ------
    ValueError
        When ``loss_func_type='PCA'`` but ``pcs`` is None — PCA loss is
        undefined without a basis, and silently substituting cross-entropy
        would optimise/report a different metric than the label claims
        (2026-06-05). When iterating over sessions where some legitimately
        lack a basis, guard the call with ``if pcs is None``.
    """
    if loss_func_type == 'JS':
        return JS_calc(pred, target)
    elif loss_func_type == 'KL':
        return KL_calc(pred, target)
    elif loss_func_type == 'Wasserstein':
        return Wasserstein_calc_1D(pred, target)
    elif loss_func_type == 'PCA':
        if pcs is None:
            raise ValueError(
                "fit_loss_per_trial: 'PCA' requested but no PCA basis (pcs is "
                "None). PCA loss is undefined without a basis; pass pcs/"
                "explained_variance or request an explicit loss for basis-less "
                "targets.")
        pred_proj = torch.matmul(pred, pcs.T)
        target_proj = torch.matmul(target, pcs.T)
        return torch.sum(
            explained_variance * (pred_proj - target_proj) ** 2, dim=-1) * 100
    elif loss_func_type == 'MSE':
        return torch.mean((pred - target) ** 2, dim=-1)
    else:
        return cross_entropy(pred, target)


# Backwards-compatibility alias. Prefer ``fit_loss_per_trial`` — this private
# name predates the rename and is kept so any straggler import keeps working.
_batched_fit_loss = fit_loss_per_trial


def _batched_total_loss(model, xb, yb, model_type, loss_func, pcs,
                        explained_variance, entropy_lambda):
    """Per-trial total loss (fit + temporal sharpness penalty) for a batch.

    xb : (B, T, n_neurons) ; yb : (B, T, n_cats). The target is reduced over
    time by the same mean custom_loss_all_H applies. Returns (B,).
    """
    pred, entropy = _batched_predict(model, xb, model_type)
    target = torch.mean(yb, dim=1)                          # (B, n_cats)
    fit = fit_loss_per_trial(pred, target, loss_func, pcs, explained_variance)
    if model_type == 'sampling':
        return fit + entropy_lambda * entropy
    return fit


def fit_model(model, optimizer, X_train, Y_train, *,
              model_type, loss_func, pcs, explained_variance,
              entropy_lambda, minibatch_size, num_epochs, max_grad_norm=1.0,
              history=None, snapshot_every=0,
              X_val=None, Y_val=None,
              patience=0, min_epochs=0, val_fraction=0.2):
    """Train ``model`` in place, vectorised over minibatches of trials.

    Parameters
    ----------
    X_train : (n_trials, T, n_neurons) ; Y_train : (n_trials, T, n_cats)
        On the model's device. Y is per-bin; it is mean-reduced over T
        internally, matching custom_loss_all_H's ``targets_mean``.
    model_type : {'ppc', 'sampling'}
    loss_func : {'PCA', 'MSE', 'CE', 'JS', 'KL', 'Wasserstein'}
    pcs, explained_variance : PCA basis. Required for loss_func='PCA';
        for non-PCA losses still used (when provided) to compute the
        per-epoch PCA-projected yardstick stored in ``history``.
    entropy_lambda : temporal sharpness weight (ignored for spatial).
    minibatch_size : trials per optimizer step.
    num_epochs : passes over the training set.
    max_grad_norm : gradient clipped to this norm once per minibatch.
    history : dict or None
        If not None, populated in place each epoch with diagnostic
        arrays. Keys created:
          - ``train_total_loss``   per-epoch mean of fit + λ·H over training
          - ``train_fit_loss``     per-epoch mean fit-only loss (no penalty)
          - ``train_entropy_pen``  per-epoch mean λ·H (temporal; 0 for spatial)
          - ``train_pca_yardstick`` per-epoch PCA-projected eval loss
                                   (independent of loss_func); ``None``
                                   list when no PCA basis was supplied.
          - ``weight_norms``       per-epoch list of per-parameter L2 norms
          - ``snapshot_epochs``    epochs at which a state_dict was saved
          - ``state_dicts``        CPU-deep-copied state_dicts at those epochs
        Default ``None`` — the function behaves identically to the
        pre-history version (no eval overhead, no allocation).
    snapshot_every : int
        Save the full ``state_dict`` every N epochs (and at the last
        epoch). 0 disables snapshots even when ``history`` is supplied
        — useful when you want loss curves but not weight trajectories.
    X_val, Y_val : torch.Tensor or None
        Optional held-out validation tensors in the same
        ``(n_trials, T, ...)`` layout as ``X_train`` / ``Y_train``.
        When both are provided AND ``history`` is being tracked, the
        per-epoch logs additionally include ``val_total_loss``,
        ``val_fit_loss``, ``val_pca_yardstick``. The val set never
        affects gradients (no training step touches it) and never
        affects REP selection — purely diagnostic for the train-vs-val
        gap plot.
    patience : int, default 0
        Early stopping. ``0`` disables it: the loop runs the full fixed
        ``num_epochs`` (unchanged behaviour). When ``> 0``, the
        validation *fit*-loss is checked each epoch and training stops
        once it has not improved for ``patience`` consecutive epochs
        (after at least ``min_epochs``); the weights with the lowest
        validation fit-loss are restored before returning. The
        validation set is whichever ``X_val``/``Y_val`` the caller
        supplied (e.g. the stratified ``val_frac`` slice from
        ``train_and_select_best_model``); if none was supplied, a seeded
        ``val_fraction`` slice of the training trials is carved off here
        so early stopping still has a held-out signal. The stop signal
        is the fit-loss only — ``entropy_lambda`` never leaks into it.
    min_epochs : int, default 0
        Floor on epochs before early stopping may trigger (ignored when
        ``patience == 0``).
    val_fraction : float, default 0.2
        Fraction of training trials used for the internal early-stopping
        holdout, but only when ``patience > 0`` AND the caller did not
        already pass ``X_val``/``Y_val``.

    Returns the trained model (same object).
    """
    n_trials = X_train.shape[0]
    _have_val = (X_val is not None and Y_val is not None
                  and X_val.shape[0] > 0)

    # Early stopping needs a held-out signal. If the caller supplied a
    # val set we reuse it; otherwise carve a seeded slice off the
    # training trials (private generator -> independent of global RNG,
    # so model init / rep restarts are unaffected and the split is
    # reproducible). The slice is removed from the training tensors so
    # it never contributes a gradient.
    if patience > 0 and not _have_val:
        g = torch.Generator(device='cpu').manual_seed(1234)
        perm = torch.randperm(n_trials, generator=g).to(X_train.device)
        n_val = min(max(1, int(round(n_trials * val_fraction))), n_trials - 1)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_val, Y_val = X_train[val_idx], Y_train[val_idx]
        X_train, Y_train = X_train[tr_idx], Y_train[tr_idx]
        n_trials = X_train.shape[0]
        _have_val = True

    # Early-stopping bookkeeping (only active when patience > 0).
    _es_best_val = float('inf')
    _es_best_state = None
    _es_no_improve = 0
    _es_best_epoch = -1
    _es_stopped_epoch = -1   # last epoch actually run (0-based); set in the loop

    if history is not None:
        history.setdefault('train_total_loss', [])
        history.setdefault('train_fit_loss', [])
        history.setdefault('train_entropy_pen', [])
        history.setdefault('train_pca_yardstick', [])
        history.setdefault('weight_norms', [])
        history.setdefault('snapshot_epochs', [])
        history.setdefault('state_dicts', [])
        if _have_val:
            history.setdefault('val_total_loss', [])
            history.setdefault('val_fit_loss', [])
            history.setdefault('val_pca_yardstick', [])
        # PCA yardstick is well-defined only when a basis was supplied
        # (i.e. n_cats > 2). For 2-D targets we leave it as an array of
        # NaNs so the column length matches the other curves.
        _have_pca_basis = pcs is not None and explained_variance is not None

    model.train()
    for epoch in range(num_epochs):
        for s in range(0, n_trials, minibatch_size):
            e = min(s + minibatch_size, n_trials)
            total = _batched_total_loss(
                model, X_train[s:e], Y_train[s:e], model_type,
                loss_func, pcs, explained_variance, entropy_lambda)
            loss = total.mean()         # mean over this minibatch's trials
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # ---- Early stopping (independent of history tracking) -------
        # Check the held-out validation fit-loss (entropy_lambda=0 so the
        # sharpness penalty never enters the stop signal); snapshot the
        # best weights and stop after `patience` epochs without
        # improvement. Runs every epoch regardless of `history` so loss
        # curves and early stopping are orthogonal features.
        if patience > 0 and _have_val:
            val_fit_es = evaluate(
                model, X_val, Y_val, model_type=model_type,
                loss_func=loss_func, pcs=pcs,
                explained_variance=explained_variance,
                entropy_lambda=0.0, reduction='mean')
            if val_fit_es < _es_best_val:
                _es_best_val = val_fit_es
                _es_best_state = copy.deepcopy(model.state_dict())
                _es_no_improve = 0
                _es_best_epoch = epoch
            else:
                _es_no_improve += 1
            model.train()  # evaluate() set eval mode; restore for next epoch
            _es_stopped_epoch = epoch
            _stop_now = (epoch + 1 >= min_epochs
                          and _es_no_improve >= patience)
        else:
            _stop_now = False

        if history is None:
            if _stop_now:
                break
            continue

        # ---- Per-epoch diagnostics (only when tracking history) -----
        # All evals here run with the model back in eval mode and
        # torch.no_grad, so they don't affect the training trajectory.
        model.eval()
        with torch.no_grad():
            # Full-train fit loss + total loss under the trained loss_func.
            total_train = _batched_total_loss(
                model, X_train, Y_train, model_type,
                loss_func, pcs, explained_variance, entropy_lambda)
            fit_train = fit_loss_per_trial(
                _batched_predict(model, X_train, model_type)[0],
                torch.mean(Y_train, dim=1),
                loss_func, pcs, explained_variance,
            )
            history['train_total_loss'].append(float(total_train.mean().item()))
            history['train_fit_loss'].append(float(fit_train.mean().item()))
            history['train_entropy_pen'].append(
                float((total_train - fit_train).mean().item()))

            # PCA yardstick: same predictions, scored under PCA loss
            # against the same targets. When the model was trained on
            # PCA loss already, this duplicates train_fit_loss exactly.
            if _have_pca_basis:
                pred, _ = _batched_predict(model, X_train, model_type)
                target = torch.mean(Y_train, dim=1)
                pca_y = fit_loss_per_trial(
                    pred, target, 'PCA', pcs, explained_variance)
                history['train_pca_yardstick'].append(
                    float(pca_y.mean().item()))
            else:
                history['train_pca_yardstick'].append(float('nan'))

            # Per-parameter L2 norms — cheap diagnostic for weight growth.
            history['weight_norms'].append(
                [float(p.detach().norm().item())
                  for p in model.parameters() if p.requires_grad])

            # State-dict snapshot at the requested cadence + always at
            # the final epoch so the trained weights are captured.
            if snapshot_every > 0 and (
                    epoch % snapshot_every == 0 or epoch == num_epochs - 1):
                history['snapshot_epochs'].append(int(epoch))
                history['state_dicts'].append(
                    {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()})

            # Validation curves — same metrics as train, computed on
            # X_val / Y_val with no gradient. Skipped silently when
            # the caller didn't supply a val set.
            if _have_val:
                val_total = _batched_total_loss(
                    model, X_val, Y_val, model_type,
                    loss_func, pcs, explained_variance, entropy_lambda)
                val_pred, _ = _batched_predict(model, X_val, model_type)
                val_target = torch.mean(Y_val, dim=1)
                val_fit = fit_loss_per_trial(
                    val_pred, val_target,
                    loss_func, pcs, explained_variance)
                history['val_total_loss'].append(
                    float(val_total.mean().item()))
                history['val_fit_loss'].append(float(val_fit.mean().item()))
                if _have_pca_basis:
                    val_pca = fit_loss_per_trial(
                        val_pred, val_target, 'PCA',
                        pcs, explained_variance)
                    history['val_pca_yardstick'].append(
                        float(val_pca.mean().item()))
                else:
                    history['val_pca_yardstick'].append(float('nan'))
        model.train()

        # Early stopping also applies on the history-tracking path.
        if _stop_now:
            break

    # Restore the best-validation weights if early stopping was active
    # and ever recorded an improvement.
    if patience > 0 and _es_best_state is not None:
        model.load_state_dict(_es_best_state)

    # Record where early stopping landed so the training-curve plots can mark
    # the stop point and the restored-best epoch. Scalars (not lists), so they
    # never interfere with the per-epoch arrays. Only written when ES ran.
    if history is not None and patience > 0 and _es_stopped_epoch >= 0:
        history['early_stopped_epoch'] = int(_es_stopped_epoch)
        history['best_epoch'] = int(_es_best_epoch)
        history['epoch_cap'] = int(num_epochs)
        history['patience'] = int(patience)

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

    Returns
    -------
    (best_overall_model, best_overall_loss, best_history)
        ``best_history`` is the history dict from the winning restart
        when ``training_params['track_training_history']`` is true
        (populated by ``fit_model``); otherwise ``None``. Existing
        callers that unpack only the first two elements continue to
        work via ``a, b, _ = train_and_select_best_model(...)`` or
        ``(a, b, _)`` slicing — the unpacking site is the only place
        that needs touching when a caller wants the history.
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
    best_overall_history = None

    # Per-restart diagnostic capture — only collect when the caller
    # opted in via training_params (default off so production runs pay
    # zero overhead). The winning restart's history is the only one
    # retained: the lower-scoring restarts' histories are dropped.
    track_history = bool(training_params.get('track_training_history', False))
    snapshot_every = int(training_params.get('weight_snapshot_every', 0))

    # Optional validation tensors. ``run_experiment.run_animal_decoder``
    # passes them as flat (n*T, ...) numpy arrays (matching the legacy
    # X_train_in / Y_train_in layout); reshape here into the
    # (n_trials, T, ...) tensors fit_model consumes. None when the
    # caller's Config didn't set val_frac.
    X_val_in = training_params.get('X_val_in')
    Y_val_in = training_params.get('Y_val_in')
    T_bins = training_params.get('T_bins')
    if X_val_in is not None and Y_val_in is not None and T_bins:
        n_val_trials = X_val_in.shape[0] // T_bins
        n_neurons = X_val_in.shape[1]
        n_cats = Y_val_in.shape[1]
        X_val = torch.tensor(
            X_val_in.reshape(n_val_trials, T_bins, n_neurons),
            dtype=torch.float32, device=device)
        Y_val = torch.tensor(
            Y_val_in.reshape(n_val_trials, T_bins, n_cats),
            dtype=torch.float32, device=device)
    else:
        X_val = None
        Y_val = None

    for r in range(REP):
        # Instantiate using the flexible architecture!
        model = SimpleFlexibleNNClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation
        ).to(device)

        # The optimiser is ALWAYS Adam. Config.optimizer_type / Config.momentum
        # are recorded in provenance but intentionally not consumed here (every
        # preset uses Adam) — see the RECORDED-ONLY note in training/config.py.
        # weight_decay IS read from training_params (sourced from
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

        rep_history = {} if track_history else None
        fit_model(
            model, optimizer, X_train, Y_train,
            model_type=model_type, loss_func=loss_func,
            pcs=pcs, explained_variance=explained_variance,
            entropy_lambda=entropy_lambda,
            minibatch_size=minibatch_size, num_epochs=num_epochs,
            history=rep_history, snapshot_every=snapshot_every,
            X_val=X_val, Y_val=Y_val,
            patience=int(training_params.get('patience', 0)),
            min_epochs=int(training_params.get('min_epochs', 0)),
            val_fraction=float(training_params.get('val_fraction', 0.2)),
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
            best_overall_history = rep_history

        if verbose:
            print(f"    Rep {r+1}/{REP} | Loss: {rep_loss:.4f} | Best: {best_overall_loss:.4f}")

    if verbose:
        print(f"  -> Best {model_type.upper()} Loss: {best_overall_loss:.4f}\n")

    return best_overall_model, best_overall_loss, best_overall_history