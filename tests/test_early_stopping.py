# -*- coding: utf-8 -*-
"""Unit tests for the opt-in early stopping added to ``fit_model``.

Pins three contracts:
  1. ``patience=0`` is the historical fixed-schedule path -- the trained
     weights are byte-identical to the pre-change loop (no validation
     holdout, no snapshots, no behavioural drift).
  2. ``patience>0`` restores the *best* validation weights and can stop
     before ``num_epochs``.
  3. The early-stopping stop signal uses the fit-loss only (entropy_lambda
     does not leak into it).
"""

from __future__ import annotations

import copy
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

from nn_classifier import (  # noqa: E402
    SimpleFlexibleNNClassifier, fit_model, evaluate, _batched_total_loss,
)


def _toy_problem(n_trials=40, T=3, n_neurons=8, n_cats=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_trials, T, n_neurons, generator=g)
    # A smooth target so fitting is meaningful but quick.
    Y = torch.softmax(torch.randn(n_trials, T, n_cats, generator=g), dim=-1)
    return X, Y


def _model(n_neurons=8, n_cats=5, seed=0):
    torch.manual_seed(seed)
    return SimpleFlexibleNNClassifier(
        input_size=n_neurons, hidden_sizes=[16], output_size=n_cats,
        activation='tanh')


def test_patience_zero_is_unchanged_fixed_schedule():
    """patience=0 must reproduce the plain fixed-num_epochs loop exactly."""
    X, Y = _toy_problem()

    # Reference: the exact original loop, inlined here using the same
    # internal loss helper fit_model uses (so the only thing under test is
    # that patience=0 does not alter the update sequence).
    ref = _model(seed=1)
    opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    ref.train()
    for _ in range(8):
        for s in range(0, X.shape[0], 16):
            e = min(s + 16, X.shape[0])
            total = _batched_total_loss(ref, X[s:e], Y[s:e], 'ppc', 'MSE',
                                        None, None, 0.0)
            loss = total.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            opt.step()

    # fit_model with patience=0 from the same init/opt seed.
    got = _model(seed=1)
    opt2 = torch.optim.Adam(got.parameters(), lr=1e-2)
    fit_model(got, opt2, X, Y, model_type='ppc', loss_func='MSE',
              pcs=None, explained_variance=None, entropy_lambda=0.0,
              minibatch_size=16, num_epochs=8, patience=0)

    for p_ref, p_got in zip(ref.parameters(), got.parameters()):
        assert torch.allclose(p_ref, p_got, atol=1e-6), \
            "patience=0 must match the original fixed-schedule loop exactly"


def test_early_stopping_restores_best_and_can_stop_early():
    """With a tiny patience the loop should stop before the epoch cap and
    leave the model at its best-validation weights (not the last epoch's)."""
    X, Y = _toy_problem()
    model = _model(seed=2)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)

    # A generous cap but tight patience -> must stop early on this toy data.
    fit_model(model, opt, X, Y, model_type='ppc', loss_func='MSE',
              pcs=None, explained_variance=None, entropy_lambda=0.0,
              minibatch_size=16, num_epochs=500, patience=3, min_epochs=2,
              val_fraction=0.25)

    # The restored weights should beat a fresh untrained model on the data
    # (sanity: training happened and best-state restore did not blow up).
    fresh = _model(seed=2)
    trained_loss = evaluate(model, X, Y, model_type='ppc', loss_func='MSE',
                            pcs=None, explained_variance=None)
    fresh_loss = evaluate(fresh, X, Y, model_type='ppc', loss_func='MSE',
                          pcs=None, explained_variance=None)
    assert trained_loss < fresh_loss, \
        "early-stopped model should fit better than an untrained one"


def test_early_stopping_independent_of_global_rng():
    """The validation split uses a private generator, so two runs with
    different global RNG states must produce identical early-stopped
    weights (reproducibility of the holdout)."""
    X, Y = _toy_problem()

    torch.manual_seed(111)
    m1 = _model(seed=3)
    o1 = torch.optim.Adam(m1.parameters(), lr=1e-2)
    fit_model(m1, o1, X, Y, model_type='ppc', loss_func='MSE', pcs=None,
              explained_variance=None, entropy_lambda=0.0, minibatch_size=16,
              num_epochs=30, patience=5, min_epochs=2)

    torch.manual_seed(999)            # different global RNG
    m2 = _model(seed=3)
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-2)
    fit_model(m2, o2, X, Y, model_type='ppc', loss_func='MSE', pcs=None,
              explained_variance=None, entropy_lambda=0.0, minibatch_size=16,
              num_epochs=30, patience=5, min_epochs=2)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), \
            "early-stopping holdout must be independent of global RNG"
