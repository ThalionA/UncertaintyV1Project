# -*- coding: utf-8 -*-
"""Equivalence + correctness tests for the vectorised training primitive
``nn_classifier.fit_model`` and its evaluation counterpart
``nn_classifier.evaluate``.

Background
----------
The legacy training loop (``train_and_select_best_model``, and the inline
copy in ``optuna_per_target.py::train_eval``) iterated a DataLoader one
*trial* at a time -- ``batch_size=T`` -- and faked minibatching by
gradient accumulation: ``(loss / mb).backward()`` every trial,
``optimizer.step()`` every ``mb`` trials. That is latency-bound: ~one
tiny GPU kernel set per trial per epoch, each followed by a CPU<->GPU
sync from ``clip_grad_norm_``.

``fit_model`` vectorises this: the whole per-mouse dataset lives on the
device as one tensor, and a minibatch of ``mb`` trials is one batched
forward/backward over a ``(mb, T, n_neurons)`` slab. The time axis ``T``
is kept as an explicit dimension and reduced exactly where the legacy
code reduced it -- input-mean for PPC, output-mean for SBC -- so each
trial still yields exactly one per-trial loss and one per-trial gradient
contribution.

Two deliberate behavioural changes vs. the literal legacy loop:

  1. ``clip_grad_norm_`` is applied once per minibatch (before
     ``step()``), not after every trial's partially-accumulated
     gradient -- the per-trial clip was order-dependent and almost
     certainly unintended.
  2. each minibatch loss is the mean over its trials, so the trailing
     partial minibatch is averaged over its actual count; the legacy
     loop divided every trial by the full ``mb``, underweighting the
     trailing chunk by ``k/mb``.

The reference loop below (``_fit_model_reference``) applies both
corrections: it is the oracle ``fit_model`` must match.

These tests are pure synthetic-data unit tests -- no real data, no GPU
required (everything runs on CPU). The real-data equivalence + speed
check is a separate verification step.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

torch = pytest.importorskip('torch')

from nn_classifier import (  # noqa: E402
    SimpleFlexibleNNClassifier,
    get_model_probabilities,
    custom_loss_all_H,
    fit_model,
    evaluate,
    train_and_select_best_model,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

def _make_data(n_trials, T, n_neurons, n_cats, seed):
    """Synthetic training data in the shape ``fit_model`` expects.

    X : (n_trials, T, n_neurons)  -- per-trial, per-bin activity
    Y : (n_trials, T, n_cats)     -- per-trial, per-bin target distributions
                                     (softmax rows; not repeated over T, so
                                     the mean-over-T reduction is exercised).
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_trials, T, n_neurons, generator=g, dtype=torch.float32)
    Y = torch.softmax(
        torch.randn(n_trials, T, n_cats, generator=g, dtype=torch.float32), dim=-1)
    return X, Y


def _make_pca(n_cats, seed):
    """A valid (pcs, explained_variance) pair for the PCA loss branch.

    pcs : (k, n_cats) orthonormal rows; explained_variance : (k,) positive,
    summing to 1. Shapes match what ``get_mouse_data`` builds from
    ``sklearn.decomposition.PCA`` (``components_`` and
    ``explained_variance_ratio_``).
    """
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n_cats, n_cats, generator=g))
    pcs = q.to(torch.float32)
    raw = torch.rand(n_cats, generator=g) + 0.1
    var = (raw / raw.sum()).to(torch.float32)
    return pcs, var


def _new_model(n_neurons, n_cats, seed, hidden=(8,)):
    """A SimpleFlexibleNNClassifier with deterministic init."""
    torch.manual_seed(seed)
    return SimpleFlexibleNNClassifier(
        input_size=n_neurons, hidden_sizes=list(hidden),
        output_size=n_cats, activation='tanh')


def _fit_model_reference(model, optimizer, X_train, Y_train, *,
                         model_type, loss_func, pcs, explained_variance,
                         entropy_lambda, minibatch_size, num_epochs,
                         max_grad_norm=1.0):
    """Naive, obviously-correct training loop -- the oracle for ``fit_model``.

    Trains one trial at a time using the *authored* ``get_model_probabilities``
    + ``custom_loss_all_H``, sums per-trial total losses within a minibatch,
    does one backward, clips once, steps. By linearity of autograd this builds
    exactly the gradient the legacy per-trial accumulation built -- with the
    two agreed corrections applied.

    Each minibatch loss is the MEAN over its trials -- ``mb_loss / (e - s)``
    -- so the trailing partial minibatch is averaged over its actual count.
    The legacy loop instead divided every trial by the full ``minibatch_size``
    regardless, which underweighted the trailing chunk by ``k/mb``.
    """
    n_trials = X_train.shape[0]
    for _ in range(num_epochs):
        model.train()
        for s in range(0, n_trials, minibatch_size):
            e = min(s + minibatch_size, n_trials)
            optimizer.zero_grad()
            mb_loss = 0.0
            for i in range(s, e):
                p = get_model_probabilities(model, X_train[i], model_type)
                total, _, _ = custom_loss_all_H(
                    p, Y_train[i], entropy_lambda, model_type,
                    pcs, explained_variance, loss_func)
                mb_loss = mb_loss + total
            (mb_loss / (e - s)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
    return model


# ----------------------------------------------------------------------
# 1. Core equivalence: fit_model == reference loop
# ----------------------------------------------------------------------

@pytest.mark.parametrize('model_type', ['ppc', 'sampling'])
@pytest.mark.parametrize('loss_func', ['PCA', 'MSE', 'CE', 'JS', 'KL'])
@pytest.mark.parametrize('n_trials,minibatch_size', [
    (24, 8),   # evenly divisible
    (23, 8),   # trailing partial minibatch (7 trials in the last chunk)
])
def test_fit_model_matches_reference(model_type, loss_func,
                                     n_trials, minibatch_size):
    """The vectorised ``fit_model`` must reproduce the trial-by-trial
    reference loop bit-for-bit (to fp32 tolerance) -- same data, same init,
    same optimizer. This is THE correctness anchor: it proves the batched
    reimplementation of the per-trial PPC/SBC math agrees with the authored
    ``get_model_probabilities`` + ``custom_loss_all_H``, and that the
    minibatch boundaries (incl. the trailing partial) line up."""
    T, n_neurons, n_cats = 5, 6, 5
    X, Y = _make_data(n_trials, T, n_neurons, n_cats, seed=0)
    pcs, var = _make_pca(n_cats, seed=1) if loss_func == 'PCA' else (None, None)
    entropy_lambda = 0.01 if model_type == 'sampling' else 0.0

    model_ref = _new_model(n_neurons, n_cats, seed=42)
    model_vec = _new_model(n_neurons, n_cats, seed=42)
    # Sanity: the two models start identical.
    for pr, pv in zip(model_ref.parameters(), model_vec.parameters()):
        assert torch.equal(pr, pv)

    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_vec = torch.optim.Adam(model_vec.parameters(), lr=1e-3, weight_decay=1e-4)

    kw = dict(model_type=model_type, loss_func=loss_func, pcs=pcs,
              explained_variance=var, entropy_lambda=entropy_lambda,
              minibatch_size=minibatch_size, num_epochs=8)
    _fit_model_reference(model_ref, opt_ref, X, Y, **kw)
    fit_model(model_vec, opt_vec, X, Y, **kw)

    for (name, pr), (_, pv) in zip(model_ref.named_parameters(),
                                   model_vec.named_parameters()):
        assert torch.allclose(pr, pv, rtol=1e-4, atol=1e-6), (
            f"{model_type}/{loss_func} n={n_trials}: param '{name}' diverged "
            f"-- max abs diff {(pr - pv).abs().max().item():.2e}")


# ----------------------------------------------------------------------
# 2. evaluate() reduction contract
# ----------------------------------------------------------------------

@pytest.mark.parametrize('model_type', ['ppc', 'sampling'])
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_evaluate_matches_manual(model_type, reduction):
    """``evaluate`` returns the reduced *total* loss (fit + penalty). The
    Optuna val path needs mean-at-lambda-0; rep-selection needs sum-at-real-
    lambda. Both reduce to: compare against a manual per-trial loop over the
    authored ``custom_loss_all_H``."""
    T, n_neurons, n_cats, n_trials = 5, 6, 5, 12
    X, Y = _make_data(n_trials, T, n_neurons, n_cats, seed=2)
    entropy_lambda = 0.01 if model_type == 'sampling' else 0.0
    model = _new_model(n_neurons, n_cats, seed=7)

    got = evaluate(model, X, Y, model_type=model_type, loss_func='MSE',
                   pcs=None, explained_variance=None,
                   entropy_lambda=entropy_lambda, reduction=reduction)

    model.eval()
    with torch.no_grad():
        per_trial = []
        for i in range(n_trials):
            p = get_model_probabilities(model, X[i], model_type)
            total, _, _ = custom_loss_all_H(
                p, Y[i], entropy_lambda, model_type, None, None, 'MSE')
            per_trial.append(total.item())
    expected = sum(per_trial) / n_trials if reduction == 'mean' else sum(per_trial)
    assert abs(got - expected) < 1e-5, f"got {got}, expected {expected}"


# ----------------------------------------------------------------------
# 3. Training actually optimises
# ----------------------------------------------------------------------

def test_fit_model_reduces_training_loss():
    """Sanity check on synthetic data: after training, the evaluated loss
    on the same data is lower than at init. Guards against a vectorisation
    that runs without error but silently fails to learn (e.g. a detached
    graph or a backward over the wrong axis)."""
    T, n_neurons, n_cats, n_trials = 5, 6, 5, 32
    X, Y = _make_data(n_trials, T, n_neurons, n_cats, seed=3)
    model = _new_model(n_neurons, n_cats, seed=11)

    before = evaluate(model, X, Y, model_type='ppc', loss_func='MSE',
                      pcs=None, explained_variance=None)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    fit_model(model, opt, X, Y, model_type='ppc', loss_func='MSE',
              pcs=None, explained_variance=None, entropy_lambda=0.0,
              minibatch_size=8, num_epochs=50)
    after = evaluate(model, X, Y, model_type='ppc', loss_func='MSE',
                     pcs=None, explained_variance=None)
    assert after < before, f"loss did not drop: before={before}, after={after}"


# ----------------------------------------------------------------------
# 4. Time bins stay unified per trial (the design property)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('model_type', ['ppc', 'sampling'])
def test_loss_invariant_to_time_bin_permutation(model_type):
    """The T bins of a trial are pooled into ONE per-trial prediction --
    input-mean for PPC, output-distribution-mean for SBC -- so permuting
    the time axis within trials must not change the loss. This is the formal
    guard for the 'all bins enter together, one unified gradient per trial'
    design property of the spatial-vs-temporal comparison: bins are pooled,
    never treated as ordered independent samples."""
    T, n_neurons, n_cats, n_trials = 6, 6, 5, 10
    X, Y = _make_data(n_trials, T, n_neurons, n_cats, seed=5)
    model = _new_model(n_neurons, n_cats, seed=9)
    entropy_lambda = 0.01 if model_type == 'sampling' else 0.0

    base = evaluate(model, X, Y, model_type=model_type, loss_func='MSE',
                    pcs=None, explained_variance=None,
                    entropy_lambda=entropy_lambda)
    perm = torch.randperm(T)
    permd = evaluate(model, X[:, perm, :], Y[:, perm, :],
                     model_type=model_type, loss_func='MSE',
                     pcs=None, explained_variance=None,
                     entropy_lambda=entropy_lambda)
    assert abs(base - permd) < 1e-5, (
        f"{model_type}: loss changed under time-bin permutation "
        f"({base} vs {permd}) -- bins are NOT being pooled per trial")


# ----------------------------------------------------------------------
# 5. PPC ignores the entropy penalty
# ----------------------------------------------------------------------

def test_ppc_ignores_entropy_lambda():
    """The PPC branch has no sharpness penalty by design. fit_model must
    produce identical weights whether entropy_lambda is 0 or large."""
    T, n_neurons, n_cats, n_trials = 5, 6, 5, 24
    X, Y = _make_data(n_trials, T, n_neurons, n_cats, seed=6)

    runs = []
    for lam in (0.0, 0.5):
        model = _new_model(n_neurons, n_cats, seed=99)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fit_model(model, opt, X, Y, model_type='ppc', loss_func='MSE',
                  pcs=None, explained_variance=None, entropy_lambda=lam,
                  minibatch_size=8, num_epochs=10)
        runs.append([p.clone() for p in model.parameters()])
    for a, b in zip(runs[0], runs[1]):
        assert torch.equal(a, b), "ppc training depended on entropy_lambda"


# ----------------------------------------------------------------------
# 6. train_and_select_best_model wrapper (run_experiment.py path)
# ----------------------------------------------------------------------

def test_train_and_select_best_model_uses_fit_model():
    """train_and_select_best_model must materialise its legacy DataLoader
    (batch_size=T, one trial per batch) into the (n_trials, T, ...) layout
    and train via fit_model. With REP=1 and a fixed seed, its weights must
    match a direct fit_model call on the equivalent hand-stacked tensors --
    this pins the loader-materialisation (stacking order) and the wrapper's
    rep-selection score against evaluate(reduction='sum')."""
    from torch.utils.data import DataLoader
    from neural_dataset import NeuralDataset
    from utils import ToTensor

    n_trials, T, n_neurons, n_cats = 16, 5, 6, 5
    X3, Y3 = _make_data(n_trials, T, n_neurons, n_cats, seed=8)  # (n_trials, T, *)
    device = torch.device('cpu')

    # Legacy flat layout: (n_trials*T, *), trial-major -- what the
    # DataLoader chops back into one-trial (T, *) batches.
    X_flat = X3.reshape(n_trials * T, n_neurons).numpy()
    Y_flat = Y3.reshape(n_trials * T, n_cats).numpy()
    loader = DataLoader(NeuralDataset(X_flat, Y_flat, transform=ToTensor(device)),
                        batch_size=T, shuffle=False)

    model_params = dict(input_size=n_neurons, hidden_sizes=[8],
                        output_size=n_cats, activation_function='tanh')
    training_params = dict(loss_func='MSE', num_epochs=6, learning_rate=1e-3,
                           weight_decay=1e-4, minibatch_size=8, device=device,
                           entropy_lambda=0.0, pcs=None, explained_variance=None)

    torch.manual_seed(123)
    best_model, best_loss, _ = train_and_select_best_model(
        1, 'ppc', loader, model_params, training_params, verbose=False)

    # Direct fit_model on the same data, identical init seed.
    torch.manual_seed(123)
    ref = SimpleFlexibleNNClassifier(n_neurons, [8], n_cats, 'tanh')
    opt = torch.optim.Adam(ref.parameters(), lr=1e-3, weight_decay=1e-4)
    fit_model(ref, opt, X3, Y3, model_type='ppc', loss_func='MSE',
              pcs=None, explained_variance=None, entropy_lambda=0.0,
              minibatch_size=8, num_epochs=6)

    for (name, pb), (_, pr) in zip(best_model.named_parameters(),
                                   ref.named_parameters()):
        assert torch.allclose(pb, pr, rtol=1e-4, atol=1e-6), (
            f"param '{name}' mismatch -- loader materialisation likely wrong")

    ref_loss = evaluate(ref, X3, Y3, model_type='ppc', loss_func='MSE',
                        pcs=None, explained_variance=None,
                        entropy_lambda=0.0, reduction='sum')
    assert best_loss == pytest.approx(ref_loss, rel=1e-4)


# ----------------------------------------------------------------------
# Restart selection (2026-07 audit): the winning restart must be chosen on
# HELD-OUT validation loss, not training loss. Selecting on training loss
# systematically favours the most overfit restart -- a confound for exactly
# the overfitting comparisons this project reports.
# ----------------------------------------------------------------------

def _selection_setup(n_trials=20, T=4, n_neurons=6, n_cats=5, seed=11, **tp):
    """(loader, model_params, training_params) with a validation slice carved."""
    from torch.utils.data import DataLoader
    from neural_dataset import NeuralDataset
    from utils import ToTensor

    X3, Y3 = _make_data(n_trials, T, n_neurons, n_cats, seed=seed)
    device = torch.device('cpu')
    loader = DataLoader(
        NeuralDataset(X3.reshape(n_trials * T, n_neurons).numpy(),
                      Y3.reshape(n_trials * T, n_cats).numpy(),
                      transform=ToTensor(device)),
        batch_size=T, shuffle=False)
    model_params = dict(input_size=n_neurons, hidden_sizes=[8],
                        output_size=n_cats, activation_function='tanh')
    training_params = dict(loss_func='MSE', num_epochs=5, learning_rate=1e-3,
                           weight_decay=1e-4, minibatch_size=8, device=device,
                           entropy_lambda=0.0, pcs=None, explained_variance=None,
                           track_training_history=True,
                           monitor_val=True, val_fraction=0.25)
    training_params.update(tp)
    return loader, model_params, training_params


def test_restart_selection_defaults_to_validation_loss():
    """Default rule: the returned score is the MINIMUM VAL loss across restarts."""
    loader, mp, tp = _selection_setup()
    torch.manual_seed(3)
    _, best_loss, hist = train_and_select_best_model(3, 'ppc', loader, mp, tp,
                                                     verbose=False)
    scores = hist['restart_scores']
    assert len(scores) == 3
    assert all(v is not None for _, v in scores), 'val slice should exist'
    assert hist['restart_selection'] == 'val'
    assert best_loss == pytest.approx(min(v for _, v in scores))


def test_restart_selection_train_reproduces_historical_rule():
    """restart_selection='train' must select on training loss (pre-2026-07-16)."""
    loader, mp, tp = _selection_setup(restart_selection='train')
    torch.manual_seed(3)
    _, best_loss, hist = train_and_select_best_model(3, 'ppc', loader, mp, tp,
                                                     verbose=False)
    scores = hist['restart_scores']
    assert hist['restart_selection'] == 'train'
    assert best_loss == pytest.approx(min(t for t, _ in scores))


def test_restart_selection_falls_back_to_train_without_validation():
    """No val slice (patience=0, monitor_val=False, no X_val) -> nothing to select
    on, so the historical training-loss rule applies and such runs are unchanged."""
    loader, mp, tp = _selection_setup(monitor_val=False)
    torch.manual_seed(3)
    _, best_loss, hist = train_and_select_best_model(2, 'ppc', loader, mp, tp,
                                                     verbose=False)
    scores = hist['restart_scores']
    assert all(v is None for _, v in scores), 'no val slice should be carved'
    assert best_loss == pytest.approx(min(t for t, _ in scores))


def test_fit_model_val_out_reports_the_slice_it_used():
    """fit_model's opt-in val_out exposes the validation slice actually used, and
    leaves the return value untouched for every existing caller."""
    n_trials, T, n_neurons, n_cats = 20, 4, 6, 5
    X3, Y3 = _make_data(n_trials, T, n_neurons, n_cats, seed=5)
    model = SimpleFlexibleNNClassifier(n_neurons, [8], n_cats, 'tanh')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = {}
    ret = fit_model(model, opt, X3, Y3, model_type='ppc', loss_func='MSE',
                    pcs=None, explained_variance=None, entropy_lambda=0.0,
                    minibatch_size=8, num_epochs=3, monitor_val=True,
                    val_fraction=0.25, val_out=out)
    assert ret is model                      # unchanged contract
    assert out['have_val'] is True
    assert out['X_val'].shape[0] == round(n_trials * 0.25)
    assert out['X_val'].shape[0] + 0 < n_trials
