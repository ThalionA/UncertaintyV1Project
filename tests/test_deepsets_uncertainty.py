"""Contracts for the leakage-safe DeepSets uncertainty analysis."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NN = os.path.join(REPO, "nn_decoder")
if NN not in sys.path:
    sys.path.insert(0, NN)

torch = pytest.importorskip("torch")

from deepsets_uncertainty import (  # noqa: E402
    AnalysisConfig, DeepSetsDecoder, MeanDecoder, MomentsDecoder,
    build_model, fit_projection_basis, gaussian_posterior, loss_per_trial,
    make_synthetic, parameter_count, posterior_metrics, prepare_arrays,
    train_model, within_condition_shuffle,
)


def _toy(n=120, t=10, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, t, p)).astype(np.float32)
    mu = rng.choice([25, 45, 65], n)
    sigma = rng.choice([5, 10], n)
    Q = gaussian_posterior(mu, sigma)
    C = np.column_stack([mu, sigma, np.zeros(n)])
    return X, Q, C


@pytest.mark.parametrize("name", ["mean", "moments", "deepsets"])
def test_models_return_distribution_logits(name):
    cfg = AnalysisConfig(hidden_dim=16, phi_hidden=8, phi_dim=8)
    model = build_model(name, 7, cfg)
    out = model(torch.randn(5, 10, 7))
    assert out.shape == (5, 91)
    p = torch.softmax(out, -1)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-6)


@pytest.mark.parametrize("cls,args", [
    (MeanDecoder, (7, 12)),
    (MomentsDecoder, (7, 12)),
    (DeepSetsDecoder, (7, 8, 8, 12)),
])
def test_all_proposed_models_are_permutation_invariant(cls, args):
    torch.manual_seed(0)
    model = cls(*args).eval()
    x = torch.randn(9, 10, 7)
    perm = torch.tensor([7, 1, 9, 0, 4, 8, 2, 6, 3, 5])
    with torch.no_grad():
        a, b = model(x), model(x[:, perm])
    assert torch.allclose(a, b, atol=2e-6)


def test_moments_decoder_receives_population_variance_not_sample_variance():
    model = MomentsDecoder(1, 2, n_bins=2)
    captured = {}
    def capture(_module, inputs):
        captured["summary"] = inputs[0].detach().clone()
    hook = model.readout.register_forward_pre_hook(capture)
    model(torch.tensor([[[0.0], [2.0]]]))
    hook.remove()
    assert torch.allclose(captured["summary"], torch.tensor([[1.0, 1.0]]))


def test_parameter_matching_is_close_not_unbounded():
    cfg = AnalysisConfig(hidden_dim=32, phi_hidden=16, phi_dim=16)
    counts = [parameter_count(build_model(n, 40, cfg)) for n in
              ("mean", "moments", "deepsets")]
    assert max(counts) / min(counts) < 1.08


@pytest.mark.parametrize("kind", ["KL", "JS", "MSE", "PCA"])
def test_losses_are_zero_on_identity(kind):
    torch.manual_seed(1)
    target = torch.softmax(torch.randn(6, 9), -1)
    logits = torch.log(target)
    pcs, evar = fit_projection_basis(target.numpy())
    got = loss_per_trial(logits, target, kind,
                         torch.tensor(pcs), torch.tensor(evar))
    assert torch.allclose(got, torch.zeros_like(got), atol=2e-6)


def test_unknown_loss_raises_instead_of_falling_through():
    with pytest.raises(ValueError):
        loss_per_trial(torch.zeros(2, 3), torch.ones(2, 3) / 3, "typo")


def test_nested_split_is_disjoint_and_preprocessing_is_train_only():
    X, Q, C = _toy()
    cfg = AnalysisConfig(seed=4)
    a = prepare_arrays(X, Q, C, cfg)
    assert not set(a.train_idx) & set(a.val_idx)
    assert not set(a.train_idx) & set(a.test_idx)
    assert not set(a.val_idx) & set(a.test_idx)

    X2, Q2 = X.copy(), Q.copy()
    X2[a.val_idx] += 1000
    Q2[a.val_idx] = np.roll(Q2[a.val_idx], 20, axis=1)
    b = prepare_arrays(X2, Q2, C, cfg)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.allclose(a.z_mean, b.z_mean)
    assert np.allclose(np.abs(a.pcs), np.abs(b.pcs))
    assert np.allclose(a.evar, b.evar)


def test_within_condition_shuffle_never_crosses_cells_and_preserves_rows():
    _, Q, C = _toy(n=60)
    shuffled, perm, eligible = within_condition_shuffle(Q, C, seed=9)
    assert eligible == 1.0
    assert np.all(C == C[perm])
    for cell in np.unique(C, axis=0):
        mask = np.all(C == cell, axis=1)
        assert np.allclose(np.sort(Q[mask], axis=0),
                           np.sort(shuffled[mask], axis=0))


def test_synthetic_variance_code_has_exactly_matched_temporal_means():
    X, _Q, _C, audit = make_synthetic("variance", 2, n_base=200)
    u = audit["u_idx"]
    means = X.mean(axis=1)
    # The dedicated width channel is index 4; its mean is exactly zero at all u.
    group = np.stack([means[u == k, 4].mean() for k in np.unique(u)])
    assert np.max(np.abs(group - group[0])) < 2e-6
    variances = X.var(axis=1)[:, 4]
    assert np.all(np.diff([variances[u == k].mean() for k in np.unique(u)]) > 0)


def test_synthetic_order_counterfactuals_have_identical_unordered_states():
    X, _Q, _C, audit = make_synthetic("order", 3, n_base=100)
    assert np.array_equal(audit["u_idx"][:5], np.arange(5))
    reference = np.sort(X[0], axis=0)
    for i in range(1, 5):
        assert np.allclose(np.sort(X[i], axis=0), reference)
        assert np.allclose(X[i].mean(0), X[0].mean(0), atol=1e-6)
        assert np.allclose(X[i].var(0), X[0].var(0), atol=1e-6)


def test_posterior_metrics_report_width_separately_from_location():
    target = gaussian_posterior(np.array([45.0]), np.array([5.0]))
    pred = gaussian_posterior(np.array([45.0]), np.array([15.0]))
    m = posterior_metrics(pred, target)
    assert m.loc[0, "mean_abs_deg"] < 1e-3
    assert m.loc[0, "variance_abs_deg2"] > 100


def test_first_validation_epoch_always_initialises_checkpoint():
    X, Q, C = _toy(n=60, p=5)
    cfg = AnalysisConfig(seed=1, max_epochs=1, min_epochs=0, patience=1,
                         restarts=1, hidden_dim=8, phi_hidden=4, phi_dim=4)
    prepared = prepare_arrays(X, Q, C, cfg)
    result = train_model("mean", "KL", prepared, cfg,
                         device=torch.device("cpu"))
    assert result["best_epoch"] == 0
    assert np.isfinite(result["best_val_loss"])
