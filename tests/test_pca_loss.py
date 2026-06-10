# -*- coding: utf-8 -*-
"""Unit tests for the consolidated PCA loss (``nn_decoder/pca_loss.py``).

``pca_loss.pca_distance`` is the single source of truth for the
PCA-weighted squared distance used as the decoder's fit-loss. It
replaces seven byte-similar copies that previously lived in
``decoder_plotting_utils``, ``run_fixed_recovery``,
``quick_diagonal_loss_check``, ``time_binned_ppc`` and the ``legacy/``
tree.

The method, for one trial's predicted distribution ``pred`` and target
distribution ``target`` (each length ``n_cats``):

  1. project both onto the PC basis  ``proj = X @ pcs.T``
  2. square the per-component difference, weight by that PC's
     explained-variance ratio  ``evar_k * (proj_pred_k - proj_target_k)**2``
  3. sum over components and multiply by 100.

Tests anchor:
  * hand-computed values for an identity basis and a rotation basis;
  * the 3-D ``(trials, cats, t_bins)`` path matches a per-bin loop;
  * a missing basis RAISES (no silent fallback to MSE / CE / NaN) —
    the design decision recorded with this consolidation;
  * the numpy ``pca_distance`` and the torch training-loss twin in
    ``nn_classifier.custom_loss_all_H`` agree numerically — the guard
    against the two implementations drifting apart;
  * ``custom_loss_all_H`` raises (rather than silently training on
    cross-entropy) when ``loss_func_type='PCA'`` but no basis is given.
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

import pca_loss  # noqa: E402


# ----------------------------------------------------------------------
# Hand-computed values — 2-D (trials, cats)
# ----------------------------------------------------------------------

def test_identity_basis_matches_hand_value():
    """With ``pcs = I`` the projection is a no-op, so the loss reduces to
    ``sum(evar * (pred - target)**2)`` (no overall scale)."""
    pred = np.array([[1.0, 0.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0]])
    pcs = np.eye(3)
    evar = np.array([0.5, 0.3, 0.2])
    # diff = [1, -1, 0]; sq = [1, 1, 0]; weighted sum = 0.5 + 0.3 = 0.8
    loss = pca_loss.pca_distance(pred, target, pcs, evar)
    assert loss.shape == (1,)
    assert np.allclose(loss, [0.8])


def test_rotation_basis_matches_hand_value():
    """A genuine (non-identity) orthonormal basis. ``pcs`` rotates by 45°.

    diff (original space) = [2, 1]
    proj diff            = (1/sqrt2) * [2+1, 2-1] = [3/sqrt2, 1/sqrt2]
    sq proj              = [4.5, 0.5]
    weighted sum         = 0.7*4.5 + 0.3*0.5 = 3.30
    """
    pred = np.array([[3.0, 1.0]])
    target = np.array([[1.0, 0.0]])
    r = 1.0 / np.sqrt(2.0)
    pcs = np.array([[r, r], [r, -r]])
    evar = np.array([0.7, 0.3])
    loss = pca_loss.pca_distance(pred, target, pcs, evar)
    assert np.allclose(loss, [3.30])


def test_identity_when_pred_equals_target_is_zero():
    """pred == target => exactly zero loss regardless of the basis."""
    rng = np.random.RandomState(0)
    pred = rng.rand(7, 5)
    pcs = rng.rand(5, 5)
    evar = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    loss = pca_loss.pca_distance(pred, pred.copy(), pcs, evar)
    assert loss.shape == (7,)
    assert np.allclose(loss, 0.0)


def test_symmetric_in_pred_and_target():
    """The loss is a squared difference, so swapping pred/target must
    not change the value. (Several call sites pass the two arrays in
    opposite orders; this pins that it is harmless.)"""
    rng = np.random.RandomState(1)
    a, b = rng.rand(4, 6), rng.rand(4, 6)
    pcs = rng.rand(6, 6)
    evar = rng.rand(6)
    assert np.allclose(
        pca_loss.pca_distance(a, b, pcs, evar),
        pca_loss.pca_distance(b, a, pcs, evar),
    )


def test_returns_one_value_per_trial():
    rng = np.random.RandomState(2)
    pred, target = rng.rand(9, 8), rng.rand(9, 8)
    pcs, evar = rng.rand(8, 8), rng.rand(8)
    loss = pca_loss.pca_distance(pred, target, pcs, evar)
    assert loss.ndim == 1 and loss.shape[0] == 9


# ----------------------------------------------------------------------
# 3-D path  (trials, cats, t_bins)
# ----------------------------------------------------------------------

def test_3d_path_equals_per_bin_2d_loop():
    """The 3-D einsum path must equal looping the 2-D path over t_bins."""
    rng = np.random.RandomState(3)
    n_trials, n_cats, t_bins = 5, 7, 4
    pred = rng.rand(n_trials, n_cats, t_bins)
    target = rng.rand(n_trials, n_cats, t_bins)
    pcs = rng.rand(6, n_cats)
    evar = rng.rand(6)

    loss_3d = pca_loss.pca_distance(pred, target, pcs, evar)
    assert loss_3d.shape == (n_trials, t_bins)

    for t in range(t_bins):
        loss_2d = pca_loss.pca_distance(pred[:, :, t], target[:, :, t], pcs, evar)
        assert np.allclose(loss_3d[:, t], loss_2d)


def test_3d_with_single_bin_matches_2d():
    """A 3-D array with t_bins == 1 collapses to the 2-D result."""
    rng = np.random.RandomState(4)
    pred2d, target2d = rng.rand(6, 5), rng.rand(6, 5)
    pcs, evar = rng.rand(5, 5), rng.rand(5)

    loss2d = pca_loss.pca_distance(pred2d, target2d, pcs, evar)
    loss3d = pca_loss.pca_distance(
        pred2d[:, :, None], target2d[:, :, None], pcs, evar)
    assert loss3d.shape == (6, 1)
    assert np.allclose(loss3d[:, 0], loss2d)


# ----------------------------------------------------------------------
# No basis => RAISE (no silent fallback)
# ----------------------------------------------------------------------

def test_no_basis_none_raises():
    """pcs is None => ValueError. PCA loss is undefined without a basis;
    silently returning MSE / CE / NaN would let a run optimise or report
    a different metric than its label claims."""
    pred, target = np.zeros((3, 4)), np.zeros((3, 4))
    with pytest.raises(ValueError):
        pca_loss.pca_distance(pred, target, None, None)


def test_no_basis_empty_array_raises():
    """An empty ``pcs`` array (how a basis-less cell is saved in .mat
    files) is also a missing basis and must raise."""
    pred, target = np.zeros((3, 4)), np.zeros((3, 4))
    with pytest.raises(ValueError):
        pca_loss.pca_distance(pred, target, np.array([]), np.array([]))


# ----------------------------------------------------------------------
# fit_loss dispatcher
# ----------------------------------------------------------------------

def test_fit_loss_pca_no_basis_raises():
    """The dispatcher inherits the strict contract: asking for 'PCA'
    with no basis raises rather than silently degrading."""
    arch_dist = {'decoded': np.zeros((2, 5)), 'target': np.zeros((2, 5))}
    with pytest.raises(ValueError):
        pca_loss.fit_loss(arch_dist, 'PCA', pcs=None, evar=None)


def test_fit_loss_unknown_loss_func_raises():
    arch_dist = {'decoded': np.zeros((2, 2)), 'target': np.zeros((2, 2))}
    with pytest.raises(ValueError):
        pca_loss.fit_loss(arch_dist, 'definitely_not_a_loss')


def test_fit_loss_mse_matches_hand_value():
    decoded = np.array([[1.0, 0.0], [0.5, 0.5]])
    target = np.array([[0.0, 1.0], [0.5, 0.5]])
    arch_dist = {'decoded': decoded, 'target': target}
    loss = pca_loss.fit_loss(arch_dist, 'MSE')
    assert np.allclose(loss, [1.0, 0.0])


def test_fit_loss_ce_matches_hand_value():
    decoded = np.array([[0.9, 0.1], [0.5, 0.5]])
    target = np.array([[1.0, 0.0], [0.5, 0.5]])
    arch_dist = {'decoded': decoded, 'target': target}
    loss = pca_loss.fit_loss(arch_dist, 'CE')
    expected = np.array([-np.log(0.9 + 1e-12), np.log(2.0)])
    assert np.allclose(loss, expected, atol=1e-6)


def test_fit_loss_pca_delegates_to_pca_distance():
    decoded = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    pcs = np.eye(2)
    evar = np.array([0.5, 0.5])
    arch_dist = {'decoded': decoded, 'target': target}
    via_dispatcher = pca_loss.fit_loss(arch_dist, 'PCA', pcs=pcs, evar=evar)
    via_direct = pca_loss.pca_distance(decoded, target, pcs, evar)
    assert np.allclose(via_dispatcher, via_direct)


# ----------------------------------------------------------------------
# numpy <-> torch agreement  (anti-drift guard)
# ----------------------------------------------------------------------

def test_numpy_matches_torch_training_loss():
    """``pca_distance`` (numpy, analysis) and the PCA branch of
    ``nn_classifier.custom_loss_all_H`` (torch, training) implement the
    same formula in two languages. They cannot be one function (autograd
    needs torch), so this test pins that they agree numerically."""
    torch = pytest.importorskip('torch')
    import nn_classifier  # noqa: E402

    rng = np.random.RandomState(5)
    n_cats, k = 9, 6
    pred_np = rng.rand(1, n_cats)
    target_np = rng.rand(1, n_cats)
    pcs_np = rng.rand(k, n_cats)
    evar_np = rng.rand(k)

    # numpy analysis path
    loss_np = pca_loss.pca_distance(pred_np, target_np, pcs_np, evar_np)

    # torch training path (model_type='ppc' => no time-averaging, no
    # entropy penalty, so fit_loss is purely the PCA distance).
    pred_t = torch.tensor(pred_np, dtype=torch.float64)
    target_t = torch.tensor(target_np, dtype=torch.float64)
    pcs_t = torch.tensor(pcs_np, dtype=torch.float64)
    evar_t = torch.tensor(evar_np, dtype=torch.float64)
    _, fit_loss, penalty = nn_classifier.custom_loss_all_H(
        pred_t, target_t, entropy_lambda=0.0, model_type='ppc',
        pcs=pcs_t, explained_variance=evar_t, loss_func_type='PCA',
    )
    assert float(penalty) == 0.0
    assert np.isclose(float(fit_loss), loss_np[0], rtol=1e-6, atol=1e-8)


def test_fit_loss_per_trial_pca_matches_pca_distance():
    """``nn_classifier.fit_loss_per_trial`` (formerly the underscore-private
    ``_batched_fit_loss``) is the per-trial eval entry point imported by 6
    analysis scripts to score saved posteriors. Its PCA branch must agree
    row-for-row with the numpy ``pca_distance`` — the agreement the scripts
    rely on but which only the mean-reduced ``custom_loss_all_H`` path pinned
    before. Tested on a batch (B>1) so the per-trial vectorisation is covered."""
    torch = pytest.importorskip('torch')
    import nn_classifier  # noqa: E402

    rng = np.random.RandomState(13)
    B, n_cats, k = 7, 9, 6
    pred_np = rng.rand(B, n_cats)
    target_np = rng.rand(B, n_cats)
    pcs_np = rng.rand(k, n_cats)
    evar_np = rng.rand(k)

    loss_np = pca_loss.pca_distance(pred_np, target_np, pcs_np, evar_np)  # (B,)
    loss_t = nn_classifier.fit_loss_per_trial(
        torch.tensor(pred_np, dtype=torch.float64),
        torch.tensor(target_np, dtype=torch.float64),
        'PCA',
        torch.tensor(pcs_np, dtype=torch.float64),
        torch.tensor(evar_np, dtype=torch.float64),
    )
    assert np.allclose(loss_t.numpy(), loss_np, rtol=1e-6, atol=1e-8)
    # The retained backwards-compat alias points at the same function.
    assert nn_classifier._batched_fit_loss is nn_classifier.fit_loss_per_trial


def test_fit_loss_per_trial_raises_on_pca_without_basis():
    """Mirror of the custom_loss_all_H guard for the per-trial eval path:
    PCA requested with no basis must raise, never silently fall through to CE."""
    torch = pytest.importorskip('torch')
    import nn_classifier  # noqa: E402

    pred_t = torch.rand(3, 5, dtype=torch.float64)
    target_t = torch.rand(3, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        nn_classifier.fit_loss_per_trial(pred_t, target_t, 'PCA')


def test_custom_loss_all_H_raises_on_pca_without_basis():
    """The core fix: ``loss_func_type='PCA'`` with ``pcs=None`` must
    raise, not silently fall through to cross-entropy."""
    torch = pytest.importorskip('torch')
    import nn_classifier  # noqa: E402

    pred_t = torch.rand(1, 5, dtype=torch.float64)
    target_t = torch.rand(1, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        nn_classifier.custom_loss_all_H(
            pred_t, target_t, entropy_lambda=0.0, model_type='ppc',
            pcs=None, explained_variance=None, loss_func_type='PCA',
        )


# ----------------------------------------------------------------------
# Plotting wrapper: decoder_plotting_utils.calc_pca_dist tolerates a
# missing basis by returning NaN (per-mouse loops use np.nanmean).
# ----------------------------------------------------------------------

def test_dpu_calc_pca_dist_wrapper_returns_nan_without_basis():
    """``decoder_plotting_utils.calc_pca_dist`` is the plotting-layer
    wrapper: it deliberately returns an all-NaN per-trial array when no
    basis exists, so per-mouse ``np.nanmean`` loops skip that cell
    instead of crashing. The strict, raising contract stays in
    ``pca_loss.pca_distance``."""
    pytest.importorskip('matplotlib')
    pytest.importorskip('scipy.io')
    import decoder_plotting_utils as dpu  # noqa: E402

    pred, target = np.zeros((3, 4)), np.zeros((3, 4))
    out = dpu.calc_pca_dist(pred, target, None, None)
    assert out.shape == (3,)
    assert np.all(np.isnan(out))

    # With a basis it must agree with the canonical function.
    rng = np.random.RandomState(6)
    p, q = rng.rand(3, 4), rng.rand(3, 4)
    pcs, evar = rng.rand(4, 4), rng.rand(4)
    assert np.allclose(
        dpu.calc_pca_dist(p, q, pcs, evar),
        pca_loss.pca_distance(p, q, pcs, evar),
    )
