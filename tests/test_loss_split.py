# -*- coding: utf-8 -*-
"""Unit tests for the clean fit_loss / entropy_penalty split in
``nn_classifier.custom_loss_all_H`` and ``evaluate_model_entropy``.

Before the split, ``custom_loss_all_H`` returned ``(total_loss,
entropy_log_val)`` where ``total_loss = fit_loss + entropy_lambda *
H(pred_probs)``. The entropy term is a training-time regulariser to
encourage per-bin sharpness in the temporal (sampling) architecture; it
has no place in a held-out test metric. Because ``evaluate_model_entropy``
called ``custom_loss_all_H`` with the actual ``entropy_lambda`` (not
zeroed), the saved ``KLs`` field in every production .mat carried this
contamination for temporal models (see
``nn_decoder/audit/AUDIT_loss_consumers.md``).

The new contract:

    total_loss, fit_loss, entropy_penalty = custom_loss_all_H(
        pred_probs, targets, entropy_lambda, model_type, pcs, evar, loss_func)

  * ``total_loss == fit_loss + entropy_penalty`` exactly (training and
    rep selection still optimise the total).
  * ``entropy_penalty == 0`` for ``model_type='ppc'`` (PPC branch has
    no penalty by design).
  * ``entropy_penalty == entropy_lambda * mean(H(pred_probs))`` for
    ``model_type='sampling'``.
  * ``fit_loss`` is the bare divergence between pred and target,
    independent of ``entropy_lambda``.

``evaluate_model_entropy`` likewise returns ``fit_loss`` as the headline
loss for reporting, with ``entropy_penalty`` available as a separate
diagnostic.
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
import torch.nn.functional as F  # noqa: E402

from nn_classifier import (  # noqa: E402
    custom_loss_all_H,
    entropy_calc,
    SimpleFlexibleNNClassifier,
    evaluate_model_entropy,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _rand_softmax(shape, seed=0):
    """Random softmax distributions, shape e.g. (n_time, n_cats)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(*shape, generator=g)
    return F.softmax(logits, dim=-1)


# ----------------------------------------------------------------------
# Return signature
# ----------------------------------------------------------------------

def test_custom_loss_returns_three_values_ppc():
    """New contract: (total_loss, fit_loss, entropy_penalty)."""
    pred = _rand_softmax((1, 5), seed=1)
    tgt = _rand_softmax((1, 5), seed=2)
    out = custom_loss_all_H(pred, tgt, entropy_lambda=0.003, model_type='ppc',
                             loss_func_type='MSE')
    assert isinstance(out, tuple)
    assert len(out) == 3, f"Expected 3-tuple, got {len(out)}-tuple"


def test_custom_loss_returns_three_values_sampling():
    """Same 3-tuple contract for the sampling branch."""
    pred = _rand_softmax((10, 5), seed=3)
    tgt = _rand_softmax((10, 5), seed=4)
    out = custom_loss_all_H(pred, tgt, entropy_lambda=0.003,
                             model_type='sampling', loss_func_type='MSE')
    assert len(out) == 3


# ----------------------------------------------------------------------
# total == fit + penalty
# ----------------------------------------------------------------------

@pytest.mark.parametrize('loss_func', ['MSE', 'KL', 'JS', 'CE'])
def test_total_equals_fit_plus_penalty_sampling(loss_func):
    """For the sampling branch, total_loss must equal fit_loss +
    entropy_penalty exactly. The training loop backprops on total; eval
    reports fit. Both must add up so model selection stays consistent."""
    pred = _rand_softmax((20, 5), seed=10)
    tgt = _rand_softmax((20, 5), seed=11)
    total, fit, penalty = custom_loss_all_H(
        pred, tgt, entropy_lambda=0.003, model_type='sampling',
        loss_func_type=loss_func,
    )
    assert torch.allclose(total, fit + penalty, atol=1e-6), (
        f"{loss_func}: total={total.item()} fit={fit.item()} "
        f"penalty={penalty.item()}"
    )


@pytest.mark.parametrize('loss_func', ['MSE', 'KL', 'JS', 'CE'])
def test_total_equals_fit_plus_penalty_ppc(loss_func):
    """PPC branch: penalty is 0 by design, so total == fit."""
    pred = _rand_softmax((1, 5), seed=20)
    tgt = _rand_softmax((1, 5), seed=21)
    total, fit, penalty = custom_loss_all_H(
        pred, tgt, entropy_lambda=0.003, model_type='ppc',
        loss_func_type=loss_func,
    )
    assert torch.allclose(total, fit, atol=1e-6)
    assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-9)


# ----------------------------------------------------------------------
# Penalty semantics
# ----------------------------------------------------------------------

def test_ppc_penalty_is_zero_regardless_of_lambda():
    """PPC branch hardcodes penalty=0, so even λ=100 produces 0."""
    pred = _rand_softmax((1, 5), seed=30)
    tgt = _rand_softmax((1, 5), seed=31)
    for lam in (0.0, 0.003, 0.1, 100.0):
        _, _, penalty = custom_loss_all_H(
            pred, tgt, entropy_lambda=lam, model_type='ppc',
            loss_func_type='MSE',
        )
        assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-9), (
            f"lambda={lam} produced nonzero ppc penalty: {penalty.item()}"
        )


def test_sampling_penalty_matches_lambda_times_mean_entropy():
    """For sampling, entropy_penalty == entropy_lambda * mean(H(pred))
    where H is computed per time bin and averaged. Matches the legacy
    formula exactly."""
    pred = _rand_softmax((15, 7), seed=40)
    tgt = _rand_softmax((15, 7), seed=41)
    lam = 0.005
    _, _, penalty = custom_loss_all_H(
        pred, tgt, entropy_lambda=lam, model_type='sampling',
        loss_func_type='MSE',
    )
    expected = lam * torch.mean(entropy_calc(pred))
    assert torch.allclose(penalty, expected, atol=1e-6), (
        f"penalty={penalty.item()} expected={expected.item()}"
    )


def test_sampling_penalty_scales_linearly_with_lambda():
    """Doubling entropy_lambda doubles the penalty (with everything else
    fixed). Anchors that the penalty really is λ·H, not some other
    form."""
    pred = _rand_softmax((10, 5), seed=50)
    tgt = _rand_softmax((10, 5), seed=51)
    _, _, p1 = custom_loss_all_H(pred, tgt, entropy_lambda=0.003,
                                   model_type='sampling', loss_func_type='MSE')
    _, _, p2 = custom_loss_all_H(pred, tgt, entropy_lambda=0.006,
                                   model_type='sampling', loss_func_type='MSE')
    assert torch.allclose(p2, 2 * p1, atol=1e-6)


# ----------------------------------------------------------------------
# Self-target: fit ≈ 0 but penalty is positive (sampling)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('loss_func', ['MSE', 'KL', 'JS'])
def test_self_target_fit_is_zero_sampling(loss_func):
    """When target equals pred_probs, fit_loss is ~0 (modulo fp32) for
    divergence metrics. This is THE sanity-test invariant for the upcoming
    checkpoint round-trip work."""
    pred = _rand_softmax((10, 5), seed=60)
    _, fit, _ = custom_loss_all_H(
        pred, pred.clone(), entropy_lambda=0.003,
        model_type='sampling', loss_func_type=loss_func,
    )
    assert torch.allclose(fit, torch.zeros_like(fit), atol=1e-5), (
        f"{loss_func}: self-target fit not ~0: {fit.item()}"
    )


@pytest.mark.parametrize('loss_func', ['MSE', 'KL', 'JS'])
def test_self_target_fit_is_zero_ppc(loss_func):
    """Same invariant for PPC: pred == target => fit ~ 0."""
    pred = _rand_softmax((1, 5), seed=70)
    _, fit, _ = custom_loss_all_H(
        pred, pred.clone(), entropy_lambda=0.003,
        model_type='ppc', loss_func_type=loss_func,
    )
    assert torch.allclose(fit, torch.zeros_like(fit), atol=1e-5)


def test_self_target_penalty_is_nonzero_sampling():
    """The other half of the sanity-test story: with self-target, fit
    is 0 but the penalty is NOT — it's still λ·H(pred). The sanity
    check report needs to separate these two."""
    pred = _rand_softmax((10, 5), seed=80)
    _, fit, penalty = custom_loss_all_H(
        pred, pred.clone(), entropy_lambda=0.003,
        model_type='sampling', loss_func_type='MSE',
    )
    assert torch.allclose(fit, torch.zeros_like(fit), atol=1e-5)
    assert penalty.item() > 1e-6, (
        f"self-target penalty unexpectedly ~0: {penalty.item()} "
        "(should be λ·H(pred) > 0)"
    )


# ----------------------------------------------------------------------
# fit_loss is independent of entropy_lambda
# ----------------------------------------------------------------------

def test_fit_loss_does_not_depend_on_lambda_sampling():
    """fit_loss should be identical across lambdas — only the penalty
    changes. This invariant lets eval-time reporting be lambda-agnostic
    and is the whole point of the split."""
    pred = _rand_softmax((10, 5), seed=90)
    tgt = _rand_softmax((10, 5), seed=91)
    fits = []
    for lam in (0.0, 0.003, 0.1):
        _, fit, _ = custom_loss_all_H(
            pred, tgt, entropy_lambda=lam, model_type='sampling',
            loss_func_type='MSE',
        )
        fits.append(fit.item())
    # All three fit_losses should be the same to fp32 precision.
    assert max(fits) - min(fits) < 1e-6, f"fit varied across lambdas: {fits}"


# ----------------------------------------------------------------------
# evaluate_model_entropy plumbing
# ----------------------------------------------------------------------

def test_evaluate_model_entropy_surfaces_fit_loss_separately():
    """evaluate_model_entropy must expose fit_loss (the headline number
    to save in the .mat) and entropy_penalty (the diagnostic) as
    separate returns. Production callers populate Losses[key]['fit']
    and Losses[key]['entropy_penalty'] from these."""
    torch.manual_seed(0)
    n_neurons, n_time, n_cats = 6, 4, 5
    model = SimpleFlexibleNNClassifier(
        input_size=n_neurons, hidden_sizes=[8],
        output_size=n_cats, activation='tanh',
    )
    batch_inputs = torch.randn(n_time, n_neurons)
    batch_targets = _rand_softmax((n_time, n_cats), seed=200)
    device = torch.device('cpu')

    out = evaluate_model_entropy(
        batch_inputs, batch_targets, model, 'MSE', 0.003, 'sampling',
        None, None, np.arange(n_cats), 'linear', device,
    )
    # Contract: returns (fit_loss, pred_samp, pred_m, targ_m, cv_val,
    # entropy_penalty). The last slot is new.
    assert len(out) == 6, (
        f"evaluate_model_entropy returned {len(out)}-tuple; expected 6 "
        "(fit_loss, pred_samp, pred_m, targ_m, cv_val, entropy_penalty)"
    )
    fit_loss, _, _, _, _, entropy_penalty = out
    # fit_loss should be a torch scalar, NOT including the penalty.
    assert torch.is_tensor(fit_loss)
    # For sampling, entropy_penalty should be strictly positive on
    # non-degenerate random predictions.
    assert torch.is_tensor(entropy_penalty)
    assert entropy_penalty.item() > 0


def test_evaluate_model_entropy_ppc_penalty_is_zero():
    """PPC branch has no penalty by construction; evaluate must report
    entropy_penalty == 0 for ppc models even when entropy_lambda > 0."""
    torch.manual_seed(1)
    n_neurons, n_time, n_cats = 6, 4, 5
    model = SimpleFlexibleNNClassifier(
        input_size=n_neurons, hidden_sizes=[8],
        output_size=n_cats, activation='tanh',
    )
    batch_inputs = torch.randn(n_time, n_neurons)
    batch_targets = _rand_softmax((n_time, n_cats), seed=210)
    device = torch.device('cpu')

    _, _, _, _, _, entropy_penalty = evaluate_model_entropy(
        batch_inputs, batch_targets, model, 'MSE', 0.003, 'ppc',
        None, None, np.arange(n_cats), 'linear', device,
    )
    assert entropy_penalty.item() == 0.0


# ----------------------------------------------------------------------
# Backward-compat: total_loss matches legacy (fit + penalty)
# ----------------------------------------------------------------------

def test_total_loss_matches_legacy_combined_value():
    """Anchor: total_loss is the same scalar the old API returned as the
    first element. Training and rep-selection callers that unpack
    ``total, _, _ = ...`` continue to optimise the same objective."""
    pred = _rand_softmax((10, 5), seed=300)
    tgt = _rand_softmax((10, 5), seed=301)
    total, fit, penalty = custom_loss_all_H(
        pred, tgt, entropy_lambda=0.003, model_type='sampling',
        loss_func_type='MSE',
    )
    # Hand-compute the legacy formula directly.
    pred_loss = torch.mean(pred, dim=0, keepdim=True)
    tgt_loss = torch.mean(tgt, dim=0, keepdim=True)
    legacy_fit = torch.mean(torch.mean((pred_loss - tgt_loss) ** 2, dim=-1))
    legacy_penalty = 0.003 * torch.mean(entropy_calc(pred))
    legacy_total = legacy_fit + legacy_penalty
    assert torch.allclose(total, legacy_total, atol=1e-6)
