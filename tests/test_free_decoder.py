# -*- coding: utf-8 -*-
"""Tests for the unconstrained ("free") spatiotemporal decoders.

Covers the three architectures (``free_decoder.py``), the factory, their
integration with the shared training primitives in ``nn_classifier`` via
the new ``model_type='free'`` branch, and the distinguishing property
that — unlike the PPC/SBC arms — a free decoder is allowed to read
temporal structure (so its output is *not* invariant to permuting the
time axis).

Pure synthetic-data unit tests; CPU only.
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

from free_decoder import (  # noqa: E402
    FreeMLP, FreeGRU, FreeTCN, build_free_model, VALID_FREE_ARCHS,
)
from nn_classifier import (  # noqa: E402
    _batched_predict,
    get_model_probabilities,
    fit_model,
    evaluate,
    train_and_select_best_model,
)


ARCHS = list(VALID_FREE_ARCHS)


def _data(n_trials, T, n_neurons, n_cats, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_trials, T, n_neurons, generator=g, dtype=torch.float32)
    Y = torch.softmax(
        torch.randn(n_trials, T, n_cats, generator=g, dtype=torch.float32), dim=-1)
    return X, Y


# ----------------------------------------------------------------------
# 1. Architecture I/O contract
# ----------------------------------------------------------------------

@pytest.mark.parametrize('arch', ARCHS)
@pytest.mark.parametrize('n_cats', [2, 91])
def test_forward_shape(arch, n_cats):
    """Every free arch maps (B, T, n_neurons) -> (B, output_size) logits."""
    B, T, n_neurons = 4, 6, 7
    model = build_free_model(arch, n_neurons=n_neurons, T=T,
                             output_size=n_cats, hidden_sizes=[16])
    x = torch.randn(B, T, n_neurons)
    out = model(x)
    assert out.shape == (B, n_cats)
    assert torch.isfinite(out).all()


def test_factory_dispatch_types():
    """The factory builds the right class per name and rejects unknowns."""
    kw = dict(n_neurons=5, T=4, output_size=3, hidden_sizes=[8])
    assert isinstance(build_free_model('mlp', **kw), FreeMLP)
    assert isinstance(build_free_model('gru', **kw), FreeGRU)
    assert isinstance(build_free_model('tcn', **kw), FreeTCN)
    with pytest.raises(ValueError):
        build_free_model('nope', **kw)


def test_tcn_kernel_clamped_to_T():
    """A kernel wider than the sequence must not crash — it's clamped to T."""
    model = build_free_model('tcn', n_neurons=4, T=3, output_size=5,
                             hidden_sizes=[8], kernel_size=99)
    out = model(torch.randn(2, 3, 4))
    assert out.shape == (2, 5)


# ----------------------------------------------------------------------
# 2. _batched_predict / get_model_probabilities 'free' branch
# ----------------------------------------------------------------------

@pytest.mark.parametrize('arch', ARCHS)
def test_batched_predict_free(arch):
    """The 'free' branch softmaxes the model output to a per-trial
    distribution and returns no entropy (no per-bin sharpness penalty)."""
    B, T, n_neurons, n_cats = 5, 6, 7, 4
    model = build_free_model(arch, n_neurons=n_neurons, T=T,
                             output_size=n_cats, hidden_sizes=[8])
    xb, _ = _data(B, T, n_neurons, n_cats)
    probs, entropy = _batched_predict(model, xb, 'free')
    assert probs.shape == (B, n_cats)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert entropy is None


def test_get_model_probabilities_free_single_trial():
    """The per-trial eval path feeds one (T, n_neurons) trial and gets a
    (1, n_cats) distribution back — same shape as the PPC branch."""
    T, n_neurons, n_cats = 6, 7, 4
    model = build_free_model('mlp', n_neurons=n_neurons, T=T,
                             output_size=n_cats, hidden_sizes=[8])
    one_trial = torch.randn(T, n_neurons)
    probs = get_model_probabilities(model, one_trial, 'free')
    assert probs.shape == (1, n_cats)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


# ----------------------------------------------------------------------
# 3. Training integration: fit_model optimises a free model
# ----------------------------------------------------------------------

@pytest.mark.parametrize('arch', ARCHS)
def test_fit_model_reduces_loss_free(arch):
    """fit_model with model_type='free' must actually learn: the evaluated
    loss after training is below the loss at init. Guards the whole
    free training path (predict -> fit-loss -> backward)."""
    n_trials, T, n_neurons, n_cats = 24, 6, 7, 5
    X, Y = _data(n_trials, T, n_neurons, n_cats, seed=3)
    torch.manual_seed(11)
    model = build_free_model(arch, n_neurons=n_neurons, T=T,
                             output_size=n_cats, hidden_sizes=[16])

    common = dict(model_type='free', loss_func='MSE', pcs=None,
                  explained_variance=None)
    before = evaluate(model, X, Y, **common)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    fit_model(model, opt, X, Y, entropy_lambda=0.0, minibatch_size=8,
              num_epochs=60, **common)
    after = evaluate(model, X, Y, **common)
    assert after < before, f"{arch}: loss did not drop ({before} -> {after})"


def _make_pca(n_cats, seed=1):
    """Orthonormal (pcs, explained_variance) pair for the PCA-loss branch,
    matching the shapes prepare_trial_tensors builds from sklearn PCA."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n_cats, n_cats, generator=g))
    raw = torch.rand(n_cats, generator=g) + 0.1
    return q.to(torch.float32), (raw / raw.sum()).to(torch.float32)


@pytest.mark.parametrize('arch', ARCHS)
@pytest.mark.parametrize('loss_func', ['PCA', 'KL', 'JS'])
def test_fit_model_reduces_loss_free_distributional(arch, loss_func):
    """The free arm must train under the three distributional losses the
    ceiling uses on Q/L: PCA (all-trials basis), KL, and JS. Loss after
    training must be below loss at init for every (arch, loss)."""
    n_trials, T, n_neurons, n_cats = 24, 6, 7, 9
    X, Y = _data(n_trials, T, n_neurons, n_cats, seed=4)
    pcs, var = _make_pca(n_cats) if loss_func == 'PCA' else (None, None)
    torch.manual_seed(13)
    model = build_free_model(arch, n_neurons=n_neurons, T=T,
                             output_size=n_cats, hidden_sizes=[16])
    common = dict(model_type='free', loss_func=loss_func, pcs=pcs,
                  explained_variance=var)
    before = evaluate(model, X, Y, **common)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    fit_model(model, opt, X, Y, entropy_lambda=0.0, minibatch_size=8,
              num_epochs=60, **common)
    after = evaluate(model, X, Y, **common)
    assert after < before, (
        f"{arch}/{loss_func}: loss did not drop ({before} -> {after})")


def test_free_ignores_entropy_lambda():
    """The free arm has no sharpness penalty (single output per trial), so
    training must be identical for any entropy_lambda — same as PPC."""
    n_trials, T, n_neurons, n_cats = 16, 5, 6, 4
    X, Y = _data(n_trials, T, n_neurons, n_cats, seed=6)
    runs = []
    for lam in (0.0, 0.7):
        torch.manual_seed(99)
        model = build_free_model('mlp', n_neurons=n_neurons, T=T,
                                 output_size=n_cats, hidden_sizes=[8])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fit_model(model, opt, X, Y, model_type='free', loss_func='MSE',
                  pcs=None, explained_variance=None, entropy_lambda=lam,
                  minibatch_size=8, num_epochs=10)
        runs.append([p.clone() for p in model.parameters()])
    for a, b in zip(runs[0], runs[1]):
        assert torch.equal(a, b), "free training depended on entropy_lambda"


# ----------------------------------------------------------------------
# 4. The distinguishing property: free decoders CAN see time order
# ----------------------------------------------------------------------

@pytest.mark.parametrize('arch', ['mlp', 'gru', 'tcn'])
def test_free_not_invariant_to_time_permutation(arch):
    """Unlike PPC (time-averaged input) and SBC (per-bin output average),
    a free decoder reads the trial jointly, so permuting the time axis can
    change its prediction. This is the whole point of the unconstrained
    arm — the formal counterpart to test_fit_model's pooling-invariance
    test for the constrained arms."""
    B, T, n_neurons, n_cats = 8, 6, 7, 5
    torch.manual_seed(5)
    model = build_free_model(arch, n_neurons=n_neurons, T=T,
                             output_size=n_cats, hidden_sizes=[16])
    x = torch.randn(B, T, n_neurons)
    perm = torch.tensor([T - 1 - i for i in range(T)])  # reverse time
    base, _ = _batched_predict(model, x, 'free')
    permd, _ = _batched_predict(model, x[:, perm, :], 'free')
    assert not torch.allclose(base, permd, atol=1e-4), (
        f"{arch}: prediction is invariant to time permutation -- it is not "
        f"actually reading temporal structure")


# ----------------------------------------------------------------------
# 5. train_and_select_best_model wrapper with model_type='free'
# ----------------------------------------------------------------------

@pytest.mark.parametrize('arch', ARCHS)
def test_train_and_select_best_model_free(arch):
    """The REP-restart wrapper must build the free model from
    model_params['free_arch'] + ['T'] and return a trained model whose
    rep-selection score matches a direct evaluate(reduction='sum')."""
    from torch.utils.data import DataLoader
    from neural_dataset import NeuralDataset
    from utils import ToTensor

    n_trials, T, n_neurons, n_cats = 16, 5, 6, 5
    X3, Y3 = _data(n_trials, T, n_neurons, n_cats, seed=8)
    device = torch.device('cpu')

    X_flat = X3.reshape(n_trials * T, n_neurons).numpy()
    Y_flat = Y3.reshape(n_trials * T, n_cats).numpy()
    loader = DataLoader(NeuralDataset(X_flat, Y_flat, transform=ToTensor(device)),
                        batch_size=T, shuffle=False)

    model_params = dict(input_size=n_neurons, T=T, hidden_sizes=[8],
                        output_size=n_cats, activation_function='tanh',
                        free_arch=arch)
    training_params = dict(loss_func='MSE', num_epochs=4, learning_rate=1e-3,
                           weight_decay=1e-4, minibatch_size=8, device=device,
                           entropy_lambda=0.0, pcs=None, explained_variance=None)

    torch.manual_seed(123)
    best_model, best_loss = train_and_select_best_model(
        1, 'free', loader, model_params, training_params, verbose=False)

    ref_loss = evaluate(best_model, X3, Y3, model_type='free', loss_func='MSE',
                        pcs=None, explained_variance=None,
                        entropy_lambda=0.0, reduction='sum')
    assert best_loss == pytest.approx(ref_loss, rel=1e-4)
