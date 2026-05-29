# -*- coding: utf-8 -*-
"""Model-recovery harness for the generative simulation.

Given a synthetic population (``simulation.generative``) whose activity
encodes known latents under PPC or SBC, this module fits **every**
decoder — the constrained PPC and SBC arms plus the three unconstrained
"free" arms (mlp / gru / tcn) — against the ground-truth target and scores
how well each recovers it. Running this for both generative schemes builds
the recovery matrix: a genuine architectural difference shows up as a
**double dissociation** (PPC-generated data recovered best by the PPC
decoder, SBC-generated data by the SBC decoder), with the free arms
tracking the information ceiling.

Targets (the latents to recover):
  'Q' — perceptual posterior (91 bins, PCA-all-trials loss)
  'L' — marginalised likelihood (91 bins, PCA-all-trials loss)
  'd' — decision posterior [P(Go), P(No-Go)] (2 bins, MSE loss)

There is no entropy / sharpness penalty on any arm in this harness — the
recovery question is about *architecture*, not regularisation — and the
PCA loss basis is always fit on all training trials.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from neural_dataset import NeuralDataset
from utils import ToTensor, get_stratified_train_test_indices, calculate_np_variance
from nn_classifier import train_and_select_best_model, get_model_probabilities, _batched_fit_loss
from simulation.generative import (
    S_GRID, make_tuning_curves, make_latents, simulate_population,
)


# Target name -> (latent attribute, n_cats, loss_func)
TARGET_SPEC = {
    'Q': ('posterior', 91, 'PCA'),
    'L': ('likelihood', 91, 'PCA'),
    'd': ('decision', 2, 'MSE'),
}

# Decoder name -> (model_type, free_arch, hidden_sizes)
DECODER_SPEC = {
    'PPC':      ('ppc', None, [32]),
    'SBC':      ('sampling', None, [32]),
    'free_mlp': ('free', 'mlp', [128, 64]),
    'free_gru': ('free', 'gru', [128, 64]),
    'free_tcn': ('free', 'tcn', [128, 64]),
}


# ----------------------------------------------------------------------
# Distribution metrics
# ----------------------------------------------------------------------

def _kl(p, q, eps=1e-12):
    """KL(p || q) per row for (n, bins) distributions."""
    p = p + eps
    q = q + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


def _wasserstein1d(p, q):
    """1-D Wasserstein (CDF L1) per row for (n, bins) distributions."""
    cp = np.cumsum(p, axis=1)
    cq = np.cumsum(q, axis=1)
    return np.sum(np.abs(cp - cq), axis=1)


def dist_sd(dist, s_grid=S_GRID):
    """Per-row SD of a distribution over ``s_grid`` (deg)."""
    return np.sqrt(np.clip(calculate_np_variance(dist, np.asarray(s_grid)), 0, None))


# ----------------------------------------------------------------------
# Synthetic loaders (mirrors run_experiment.prepare_trial_tensors)
# ----------------------------------------------------------------------

@dataclass
class SimLoaders:
    train_loader: object
    train_loader_shuffle: object
    test_loader: object
    pcs: object
    explained_variance: object
    T: int
    n_neurons: int
    test_indices: np.ndarray


def build_synthetic_loaders(activity, target, level, *, n_cats, device,
                            seed=42, test_size=0.5):
    """Build train / train-shuffled / test loaders from a synthetic dataset.

    ``activity`` is ``(n_neurons, n_trials, T)``; ``target`` is
    ``(n_trials, n_cats)``; ``level`` is the per-trial uncertainty-level
    index used to stratify the split. Z-scoring uses training-fold stats
    only and the PCA loss basis (n_cats > 2) is fit on all training-trial
    targets — identical conventions to the real pipeline.
    """
    n_neurons, n_trials, T = activity.shape
    train_idx, test_idx = get_stratified_train_test_indices(
        level, test_size=test_size, random_state=seed)

    # Train-fold z-scoring (per neuron, over train trials and bins).
    mean_tr = activity[:, train_idx, :].mean(axis=(1, 2), keepdims=True)
    std_tr = activity[:, train_idx, :].std(axis=(1, 2), keepdims=True)
    std_tr[std_tr == 0] = 1.0
    act_z = (activity - mean_tr) / std_tr

    # Targets repeated across T bins -> (n_cats, n_trials, T).
    target_distr = np.repeat(target.T[:, :, None], T, axis=2)

    def _flatten(a):
        return np.copy(a.reshape(a.shape[0], -1)).T

    X_train = _flatten(act_z[:, train_idx, :])
    Y_train = _flatten(target_distr[:, train_idx, :])
    X_test = _flatten(act_z[:, test_idx, :])
    Y_test = _flatten(target_distr[:, test_idx, :])

    # Trial-shuffled training targets (same permutation across a trial's T
    # bins), matching the production shuffle control.
    n_train = len(train_idx)
    perm = np.arange(n_train)
    np.random.RandomState(seed).shuffle(perm)
    row_perm = np.repeat(perm, T) * T + np.tile(np.arange(T), n_train)
    Y_train_shuf = np.copy(Y_train[row_perm, :])

    # PCA basis on all training-trial targets (distributional targets only).
    if n_cats > 2:
        pca = PCA().fit(Y_train[::T])  # one row per train trial (bins identical)
        pcs = torch.tensor(pca.components_, dtype=torch.float32, device=device)
        evar = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32, device=device)
    else:
        pcs, evar = None, None

    mk = lambda X, Y: DataLoader(
        NeuralDataset(X, Y, transform=ToTensor(device)), batch_size=T, shuffle=False)
    return SimLoaders(
        train_loader=mk(X_train, Y_train),
        train_loader_shuffle=mk(X_train, Y_train_shuf),
        test_loader=mk(X_test, Y_test),
        pcs=pcs, explained_variance=evar, T=T, n_neurons=n_neurons,
        test_indices=test_idx)


# ----------------------------------------------------------------------
# Fitting + evaluation
# ----------------------------------------------------------------------

def _eval_decoder(model, loader, model_type, loss_func, pcs, evar):
    """Per-trial predicted distribution + fit-loss over a loader."""
    decoded, losses = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            probs = get_model_probabilities(model, xb, model_type)  # (1|T, n_cats)
            pred = probs.mean(dim=0, keepdim=True)                  # (1, n_cats)
            tgt = yb.mean(dim=0, keepdim=True)
            fit = _batched_fit_loss(pred, tgt, loss_func, pcs, evar)
            decoded.append(pred.cpu().numpy())
            losses.append(float(fit.item()))
    return np.vstack(decoded), np.asarray(losses, dtype=float)


def fit_one_decoder(name, loaders, *, loss_func, n_cats, device, REP=3,
                    num_epochs=40, learning_rate=1e-3, weight_decay=1e-4,
                    minibatch_size=16):
    """Train one decoder (real + shuffle control) and evaluate on test.

    Returns ``{'decoded', 'fit_loss', 'fit_loss_shuffle'}`` where decoded
    is ``(n_test, n_cats)`` and the losses are per-test-trial arrays.
    """
    model_type, free_arch, hidden = DECODER_SPEC[name]
    model_params = {
        'input_size': loaders.n_neurons,
        'hidden_sizes': hidden,
        'output_size': n_cats,
        'activation_function': 'tanh',
    }
    if model_type == 'free':
        model_params['T'] = loaders.T
        model_params['free_arch'] = free_arch
    training_params = {
        'loss_func': loss_func,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'minibatch_size': minibatch_size,
        'momentum': 0.9,
        'device': device,
        'entropy_lambda': 0.0,                  # no sharpness penalty anywhere
        'pcs': loaders.pcs,
        'explained_variance': loaders.explained_variance,
        'optimizer_type': 'adam',
    }
    best_real, _ = train_and_select_best_model(
        REP, model_type, loaders.train_loader, model_params, training_params, verbose=False)
    best_shuf, _ = train_and_select_best_model(
        REP, model_type, loaders.train_loader_shuffle, model_params, training_params, verbose=False)

    dec, fl = _eval_decoder(best_real, loaders.test_loader, model_type,
                            loss_func, loaders.pcs, loaders.explained_variance)
    _, fl_shuf = _eval_decoder(best_shuf, loaders.test_loader, model_type,
                               loss_func, loaders.pcs, loaders.explained_variance)
    return {'decoded': dec, 'fit_loss': fl, 'fit_loss_shuffle': fl_shuf}


def run_cell(scheme, target_name, *, n_trials=400, n_neurons=80, T=10,
             uncertainty_levels=(4.0, 8.0, 16.0), tuning_width=12.0,
             sample_gain=12.0, REP=3, num_epochs=40, seed=0,
             device=None, decoders=tuple(DECODER_SPEC)):
    """Simulate one (scheme, target) dataset and fit every decoder on it.

    Returns a dict keyed by decoder name, each with the recovery metrics
    (fit-loss normalised to shuffle, KL and 1-D Wasserstein to the
    ground-truth target, and the uncertainty-calibration correlation
    between predicted distribution SD and true likelihood SD on test
    trials). Also returns ``ground_truth`` test-set arrays for plotting.
    """
    device = device or torch.device('cpu')
    attr, n_cats, loss_func = TARGET_SPEC[target_name]

    tuning, _ = make_tuning_curves(n_neurons, tuning_width=tuning_width, seed=seed)
    latents = make_latents(n_trials, uncertainty_levels=uncertainty_levels, seed=seed)
    activity = simulate_population(scheme, latents, tuning, T,
                                   seed=seed, sample_gain=sample_gain)
    target = getattr(latents, attr)                       # (n_trials, n_cats)

    loaders = build_synthetic_loaders(activity, target, latents.level,
                                      n_cats=n_cats, device=device, seed=42)
    test_idx = loaders.test_indices
    true_test = target[test_idx]
    sigma_test = latents.sigma[test_idx]

    results = {}
    for name in decoders:
        out = fit_one_decoder(name, loaders, loss_func=loss_func, n_cats=n_cats,
                              device=device, REP=REP, num_epochs=num_epochs)
        dec = out['decoded']
        real = float(np.mean(out['fit_loss']))
        shuf = float(np.mean(out['fit_loss_shuffle']))
        metrics = {
            'fit_loss': real,
            'fit_loss_shuffle': shuf,
            'fit_loss_norm': real / shuf if shuf > 0 else np.nan,
        }
        if n_cats > 2:
            metrics['kl_to_truth'] = float(np.mean(_kl(true_test, dec)))
            metrics['wasserstein_to_truth'] = float(np.mean(_wasserstein1d(true_test, dec)))
            # Uncertainty calibration: does predicted spread track true sigma?
            pred_sd = dist_sd(dec)
            if np.std(pred_sd) > 0 and np.std(sigma_test) > 0:
                metrics['sd_corr'] = float(np.corrcoef(pred_sd, sigma_test)[0, 1])
            else:
                metrics['sd_corr'] = np.nan
        else:
            # Decision posterior: report MSE and P(Go) recovery correlation.
            metrics['mse_to_truth'] = float(np.mean((dec - true_test) ** 2))
            if np.std(dec[:, 0]) > 0 and np.std(true_test[:, 0]) > 0:
                metrics['pgo_corr'] = float(np.corrcoef(dec[:, 0], true_test[:, 0])[0, 1])
            else:
                metrics['pgo_corr'] = np.nan
        results[name] = metrics

    return {'metrics': results,
            'ground_truth': {'target_test': true_test, 'sigma_test': sigma_test,
                             'decoded': {n: None for n in decoders}}}


def run_recovery_matrix(*, schemes=('ppc', 'sbc'), targets=('Q', 'L', 'd'),
                        decoders=tuple(DECODER_SPEC), seed=0, verbose=True,
                        **cell_kwargs):
    """Full recovery matrix: every (scheme, target) cell, every decoder.

    Returns ``{target: {scheme: {decoder: metrics}}}``.
    """
    out = {}
    for target_name in targets:
        out[target_name] = {}
        for scheme in schemes:
            if verbose:
                print(f"=== target={target_name} | generative scheme={scheme} ===")
            cell = run_cell(scheme, target_name, seed=seed, decoders=decoders,
                            **cell_kwargs)
            out[target_name][scheme] = cell['metrics']
            if verbose:
                for name, m in cell['metrics'].items():
                    key = 'kl_to_truth' if target_name != 'd' else 'mse_to_truth'
                    print(f"    {name:9s} fit_norm={m['fit_loss_norm']:.3f} "
                          f"{key}={m.get(key, float('nan')):.4f}")
    return out
