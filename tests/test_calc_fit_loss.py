# -*- coding: utf-8 -*-
"""Unit tests for ``decoder_plotting_utils.calc_fit_loss``.

``calc_fit_loss`` dispatches on the trained model's ``loss_func``
('PCA' | 'MSE' | 'CE') and returns the per-trial fit-loss array
recomputed from the saved ``Dist[arch]['target']`` and ``['decoded']``
arrays. It is the clean replacement for reading ``KLs[arch]`` directly,
which carried the entropy-penalty contamination for temporal/sampling
models (see ``audit/AUDIT_loss_consumers.md``).

Tests anchor:
  * Identity case: target == decoded -> all-zeros loss for each branch.
  * MSE matches a hand-computed value.
  * CE matches a hand-computed value (no entropy penalty added).
  * PCA branch delegates to ``calc_pca_dist`` and inherits its semantics.
  * Unknown loss_func raises a loud error rather than silently returning
    something misleading.
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

pytest.importorskip('scipy.io')
import decoder_plotting_utils as dpu  # noqa: E402


# ----------------------------------------------------------------------
# Identity: target == decoded -> zero fit-loss
# ----------------------------------------------------------------------

def test_calc_fit_loss_mse_identity_is_zero():
    """For MSE, target == decoded => per-trial loss is exactly 0."""
    decoded = np.array([[0.3, 0.7], [0.1, 0.9], [0.5, 0.5]])
    arch_dist = {'decoded': decoded, 'target': decoded.copy()}
    loss = dpu.calc_fit_loss(arch_dist, 'MSE')
    assert loss.shape == (3,)
    assert np.allclose(loss, 0.0)


def test_calc_fit_loss_ce_identity_is_entropy_of_target():
    """For CE, target == decoded => per-trial loss equals H(target) — the
    entropy of the target distribution itself, NOT zero. (KL(p||p) = 0
    but CE(p, p) = H(p).) This anchors that calc_fit_loss is computing
    CE the same way the training loss did, with no entropy penalty
    added on top."""
    target = np.array([[0.5, 0.5], [0.1, 0.9], [0.3, 0.7]])
    arch_dist = {'decoded': target.copy(), 'target': target.copy()}
    loss = dpu.calc_fit_loss(arch_dist, 'CE')
    expected_H = -np.sum(target * np.log(target + 1e-12), axis=-1)
    assert loss.shape == (3,)
    assert np.allclose(loss, expected_H)


def test_calc_fit_loss_pca_identity_is_zero():
    """For PCA, target == decoded => loss is 0 regardless of the basis."""
    decoded = np.array([[0.3, 0.7], [0.1, 0.9], [0.5, 0.5]])
    pcs = np.eye(2, dtype=np.float64)
    evar = np.array([0.6, 0.4], dtype=np.float64)
    arch_dist = {'decoded': decoded, 'target': decoded.copy()}
    loss = dpu.calc_fit_loss(arch_dist, 'PCA', pcs=pcs, evar=evar)
    assert loss.shape == (3,)
    assert np.allclose(loss, 0.0)


# ----------------------------------------------------------------------
# Hand-computed values
# ----------------------------------------------------------------------

def test_calc_fit_loss_mse_matches_hand_value():
    """MSE = mean((p - q)**2, axis=-1) per trial. Anchored on a small
    case so future refactors can't silently change the convention."""
    decoded = np.array([[1.0, 0.0], [0.5, 0.5]])
    target  = np.array([[0.0, 1.0], [0.5, 0.5]])
    arch_dist = {'decoded': decoded, 'target': target}
    loss = dpu.calc_fit_loss(arch_dist, 'MSE')
    # Trial 0: mean([1, 1]) = 1.0; Trial 1: mean([0, 0]) = 0.0
    assert np.allclose(loss, [1.0, 0.0])


def test_calc_fit_loss_ce_matches_hand_value():
    """CE = -sum(target * log(decoded + eps), axis=-1) per trial.
    Matches nn_classifier.cross_entropy. NO entropy penalty added —
    that was the original bug we're routing around."""
    decoded = np.array([[0.9, 0.1], [0.5, 0.5]])
    target  = np.array([[1.0, 0.0], [0.5, 0.5]])
    arch_dist = {'decoded': decoded, 'target': target}
    loss = dpu.calc_fit_loss(arch_dist, 'CE')
    # Trial 0: -1*log(0.9) ≈ 0.10536
    # Trial 1: -0.5*log(0.5) - 0.5*log(0.5) = log(2) ≈ 0.69315
    expected = np.array([-np.log(0.9 + 1e-12), np.log(2.0)])
    assert np.allclose(loss, expected, atol=1e-6)


# ----------------------------------------------------------------------
# Routing / dispatch
# ----------------------------------------------------------------------

def test_calc_fit_loss_pca_delegates_to_calc_pca_dist():
    """The PCA branch must use the same calc_pca_dist that
    decoder_plotting_utils' other plotters use, so all per-cell loss
    numbers in plot_post_fix_performance.py and
    plot_normalized_performance_with_lines stay comparable."""
    decoded = np.array([[1.0, 0.0], [0.0, 1.0]])
    target  = np.array([[0.0, 1.0], [1.0, 0.0]])
    pcs = np.eye(2, dtype=np.float64)
    evar = np.array([0.5, 0.5], dtype=np.float64)
    arch_dist = {'decoded': decoded, 'target': target}
    via_dispatcher = dpu.calc_fit_loss(arch_dist, 'PCA', pcs=pcs, evar=evar)
    via_direct = dpu.calc_pca_dist(target, decoded, pcs, evar)
    assert np.allclose(via_dispatcher, via_direct)


def test_calc_fit_loss_unknown_loss_func_raises():
    """A typo in loss_func must surface loudly, not silently return
    cross-entropy or zeros."""
    arch_dist = {'decoded': np.zeros((2, 2)), 'target': np.zeros((2, 2))}
    with pytest.raises(ValueError):
        dpu.calc_fit_loss(arch_dist, 'JS')  # JS is real but unsupported here
    with pytest.raises(ValueError):
        dpu.calc_fit_loss(arch_dist, 'definitely_not_a_loss')


# ----------------------------------------------------------------------
# Contract: the function returns per-trial 1-D arrays
# ----------------------------------------------------------------------

def test_calc_fit_loss_returns_1d_per_trial_array():
    """Per-trial means: shape == (n_trials,) for all three branches.
    Downstream code (per-mouse aggregation, paired t-tests across
    trials) depends on this."""
    decoded = np.random.RandomState(0).rand(5, 4)
    target = np.random.RandomState(1).rand(5, 4)
    arch_dist = {'decoded': decoded, 'target': target}

    mse = dpu.calc_fit_loss(arch_dist, 'MSE')
    ce = dpu.calc_fit_loss(arch_dist, 'CE')

    pcs = np.eye(4, dtype=np.float64)
    evar = np.array([0.4, 0.3, 0.2, 0.1])
    pca = dpu.calc_fit_loss(arch_dist, 'PCA', pcs=pcs, evar=evar)

    for name, arr in [('MSE', mse), ('CE', ce), ('PCA', pca)]:
        assert arr.ndim == 1 and arr.shape[0] == 5, (
            f"{name} returned {arr.shape}, expected (5,)"
        )
