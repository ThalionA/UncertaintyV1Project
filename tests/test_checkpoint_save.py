# -*- coding: utf-8 -*-
"""Unit tests for the checkpoint-save plumbing in
``run_experiment._extract_checkpoint``.

The checkpoint is a self-contained bundle saved alongside each
``(mouse, split)`` .mat output so the sanity-check script can:
  1. Reconstruct the trained model (state_dict + architecture params).
  2. Replay the held-out forward pass on the exact same X_test the
     reported metrics were computed on (no dependence on dataloader
     reproducibility / RNG state).
  3. Compute fit-loss with the saved pred_probs as target and confirm
     it lands at the fp32 floor — the round-trip identity check the
     user asked for.

Tests cover:
  * Returned dict shape and keys.
  * state_dict round-trips into a fresh ``SimpleFlexibleNNClassifier``
    built from the saved ``model_params``.
  * Forward pass on the saved X_test reproduces the saved pred_probs
    bit-for-bit (modulo fp32 noise).
  * spat/temp pred_probs have the architecture-specific shape
    (n_trials × 1 × n_cats vs n_trials × T × n_cats).
  * Optional fields (pcs, explained_var) are preserved when present,
    None when absent.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

torch = pytest.importorskip('torch')

from nn_classifier import (  # noqa: E402
    SimpleFlexibleNNClassifier,
    get_model_probabilities,
)
from run_experiment import _extract_checkpoint  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures: tiny trained-ish model + synthetic batches
# ----------------------------------------------------------------------

N_NEURONS = 8
N_CATS = 5
T_BINS = 4
N_TRIALS = 6


def _build_model(seed=0):
    """Reproducible tiny MLP — small enough that tests run in ms."""
    torch.manual_seed(seed)
    return SimpleFlexibleNNClassifier(
        input_size=N_NEURONS, hidden_sizes=[12],
        output_size=N_CATS, activation='tanh',
    )


def _synthetic_batches(seed=1):
    """Per-trial batches as the test_loader would yield: one (T, n_neurons)
    tensor per trial."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(T_BINS, N_NEURONS, generator=g) for _ in range(N_TRIALS)]


def _model_params():
    return {
        'input_size': N_NEURONS,
        'hidden_sizes': [12],
        'output_size': N_CATS,
        'activation_function': 'tanh',
    }


def _collect_pred_probs(model, batches, model_type):
    """Forward each batch through, returning the per-trial pred_probs
    list (one tensor per trial). Mirrors what run_animal_decoder does
    when it captures the checkpoint."""
    with torch.no_grad():
        return [get_model_probabilities(model, b, model_type) for b in batches]


# ----------------------------------------------------------------------
# Dict shape and keys
# ----------------------------------------------------------------------

REQUIRED_KEYS = {
    'state_dict', 'X_test', 'pred_probs', 'model_params',
    'pcs', 'explained_var', 'loss_func', 'entropy_lambda',
    'model_type',
}


def test_extract_checkpoint_returns_required_keys_spat():
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'ppc')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )
    assert REQUIRED_KEYS <= set(ckpt), (
        f"missing keys: {REQUIRED_KEYS - set(ckpt)}"
    )


def test_extract_checkpoint_returns_required_keys_temp():
    model = _build_model(seed=1)
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'sampling')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='sampling',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )
    assert REQUIRED_KEYS <= set(ckpt)


def test_loss_func_and_entropy_lambda_preserved():
    """Metadata round-trip: the sanity script depends on these to know
    which fit-loss formula to apply and to report the entropy penalty
    diagnostic."""
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'ppc')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=None, explained_var=None,
        loss_func='PCA', entropy_lambda=0.005,
    )
    assert ckpt['loss_func'] == 'PCA'
    assert ckpt['entropy_lambda'] == 0.005
    assert ckpt['model_type'] == 'ppc'


# ----------------------------------------------------------------------
# Tensor shapes
# ----------------------------------------------------------------------

def test_X_test_shape_is_trials_T_neurons():
    """X_test is stacked across trials: (n_trials, T, n_neurons)."""
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'sampling')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='sampling',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )
    assert tuple(ckpt['X_test'].shape) == (N_TRIALS, T_BINS, N_NEURONS)


def test_pred_probs_shape_spat_is_trials_1_cats():
    """PPC integrates time before softmax, so per-trial pred_probs is
    (1, n_cats); stacked across trials this is (n_trials, 1, n_cats)."""
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'ppc')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )
    assert tuple(ckpt['pred_probs'].shape) == (N_TRIALS, 1, N_CATS)


def test_pred_probs_shape_temp_is_trials_T_cats():
    """Sampling architecture produces per-bin pred_probs (T, n_cats);
    stacked across trials this is (n_trials, T, n_cats)."""
    model = _build_model(seed=2)
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'sampling')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='sampling',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )
    assert tuple(ckpt['pred_probs'].shape) == (N_TRIALS, T_BINS, N_CATS)


# ----------------------------------------------------------------------
# state_dict round-trip
# ----------------------------------------------------------------------

def test_state_dict_round_trips_into_fresh_model():
    """The saved state_dict + model_params must rebuild an identical
    network. This is the contract the sanity-check script depends on."""
    model = _build_model(seed=10)
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'ppc')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )

    # Rebuild a fresh model from model_params.
    fresh = SimpleFlexibleNNClassifier(
        input_size=ckpt['model_params']['input_size'],
        hidden_sizes=ckpt['model_params']['hidden_sizes'],
        output_size=ckpt['model_params']['output_size'],
        activation=ckpt['model_params']['activation_function'],
    )
    fresh.load_state_dict(ckpt['state_dict'])

    # Forward through both — parameters identical => outputs identical.
    test_input = torch.randn(T_BINS, N_NEURONS)
    with torch.no_grad():
        orig_out = model(test_input)
        fresh_out = fresh(test_input)
    assert torch.allclose(orig_out, fresh_out, atol=1e-7)


def test_forward_pass_on_saved_X_reproduces_saved_pred_probs_spat():
    """The headline round-trip: running the loaded model on saved X_test
    yields the saved pred_probs bit-for-bit. This IS the sanity-test
    invariant — if it fails, the saved checkpoint doesn't match the
    saved reported metrics, and every downstream "0 loss" claim falls
    apart."""
    model = _build_model(seed=20)
    batches = _synthetic_batches(seed=21)
    pps = _collect_pred_probs(model, batches, 'ppc')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )

    fresh = SimpleFlexibleNNClassifier(
        input_size=ckpt['model_params']['input_size'],
        hidden_sizes=ckpt['model_params']['hidden_sizes'],
        output_size=ckpt['model_params']['output_size'],
        activation=ckpt['model_params']['activation_function'],
    )
    fresh.load_state_dict(ckpt['state_dict'])
    fresh.eval()

    X = ckpt['X_test']  # (n_trials, T, n_neurons)
    saved = ckpt['pred_probs']  # (n_trials, 1, n_cats)
    with torch.no_grad():
        for i in range(X.shape[0]):
            new_pp = get_model_probabilities(fresh, X[i], 'ppc')
            assert torch.allclose(new_pp, saved[i], atol=1e-6), (
                f"trial {i}: round-trip drift "
                f"max|Δ|={torch.max(torch.abs(new_pp - saved[i])).item():.2e}"
            )


def test_forward_pass_on_saved_X_reproduces_saved_pred_probs_temp():
    """Same round-trip invariant for the sampling architecture."""
    model = _build_model(seed=30)
    batches = _synthetic_batches(seed=31)
    pps = _collect_pred_probs(model, batches, 'sampling')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='sampling',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )

    fresh = SimpleFlexibleNNClassifier(
        input_size=ckpt['model_params']['input_size'],
        hidden_sizes=ckpt['model_params']['hidden_sizes'],
        output_size=ckpt['model_params']['output_size'],
        activation=ckpt['model_params']['activation_function'],
    )
    fresh.load_state_dict(ckpt['state_dict'])
    fresh.eval()

    X = ckpt['X_test']
    saved = ckpt['pred_probs']
    with torch.no_grad():
        for i in range(X.shape[0]):
            new_pp = get_model_probabilities(fresh, X[i], 'sampling')
            assert torch.allclose(new_pp, saved[i], atol=1e-6)


# ----------------------------------------------------------------------
# pcs / evar handling
# ----------------------------------------------------------------------

def test_pcs_and_evar_preserved_when_present():
    """For PCA-loss cells, pcs and explained_var are needed at sanity-
    check time to evaluate the loss formula. Saved on CPU so the
    checkpoint is portable across GPU/CPU machines."""
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'ppc')
    pcs = torch.randn(3, N_CATS)
    evar = torch.tensor([0.5, 0.3, 0.2])
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=pcs, explained_var=evar,
        loss_func='PCA', entropy_lambda=0.003,
    )
    assert ckpt['pcs'] is not None
    assert ckpt['explained_var'] is not None
    assert tuple(ckpt['pcs'].shape) == (3, N_CATS)
    assert tuple(ckpt['explained_var'].shape) == (3,)
    # CPU storage so the checkpoint round-trips on machines without GPU.
    assert ckpt['pcs'].device.type == 'cpu'


def test_pcs_and_evar_are_none_for_non_pca_cells():
    """MSE / CE cells don't have a PCA basis; the corresponding fields
    are None rather than empty arrays so the sanity script can
    distinguish "PCA basis not applicable" from "PCA basis missing"."""
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'ppc')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='ppc',
        pcs=None, explained_var=None,
        loss_func='MSE', entropy_lambda=0.003,
    )
    assert ckpt['pcs'] is None
    assert ckpt['explained_var'] is None


# ----------------------------------------------------------------------
# CPU storage invariant (for portability)
# ----------------------------------------------------------------------

def test_all_tensors_on_cpu():
    """Every tensor in the checkpoint must be on CPU. Production runs
    happen on the cluster's GPU; the sanity script may run on a laptop.
    Saving GPU tensors and loading them on a CPU-only machine fails."""
    model = _build_model()
    batches = _synthetic_batches()
    pps = _collect_pred_probs(model, batches, 'sampling')
    ckpt = _extract_checkpoint(
        model=model, batches=batches, pred_probs_list=pps,
        model_params=_model_params(), model_type='sampling',
        pcs=torch.randn(2, N_CATS), explained_var=torch.tensor([0.6, 0.4]),
        loss_func='PCA', entropy_lambda=0.003,
    )
    for k, v in ckpt['state_dict'].items():
        assert v.device.type == 'cpu', f"state_dict[{k}] on {v.device}"
    for k in ('X_test', 'pred_probs', 'pcs', 'explained_var'):
        assert ckpt[k].device.type == 'cpu', f"{k} on {ckpt[k].device}"
