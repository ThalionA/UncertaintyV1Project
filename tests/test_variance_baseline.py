# -*- coding: utf-8 -*-
"""Tests for the variance-baseline helpers in decoder_plotting_utils.

The variance baseline drives the new `_varnorm` figure variant in
generate_clean_run_plots.py. It must agree numerically with
optuna_per_target.marginal_baseline_loss (which the Optuna sweep
already uses to normalise validation losses) — otherwise the
shuffle / variance / raw plot suite would be reporting one quantity in
the figures and a different quantity in the hyperparameter sweep.

This file pins the contract on synthetic data with a known generative
variance:

  1. variance_baseline(target_test, train_target_mean, pcs, evar).mean()
     equals marginal_baseline_loss(Y_train_flat, Y_val_flat, 'PCA', pcs, var)
     when train_target_mean = Y_train_flat.mean(0).

  2. The baseline approaches the analytic PC-weighted target variance
     (sum_i evar_i * Var_t[<P_t, PC_i>]) * 100 as N grows — i.e. the
     formula is what we think it is.

  3. With pcs=None or train_target_mean=None the helper returns a
     NaN-filled array (the plot-layer NaN-tolerance contract — same
     pattern decoder_plotting_utils.calc_pca_dist uses).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make ``nn_decoder/`` importable without depending on the test runner
# being launched from that directory.
HERE = os.path.dirname(os.path.abspath(__file__))
NN_DECODER = os.path.abspath(os.path.join(HERE, '..', 'nn_decoder'))
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)


def _plant_targets(rng, n_cells=10, n_trials_per_cell=40, n_bins=91,
                    shift_scale=1.0, wiggle_scale=0.3):
    """Synthetic per-trial targets with structured across- and
    within-cell variance.

    Mirrors the helper in tests/test_pca_basis.py — same two-axis
    construction so the planted variance has a closed form: total
    PC-weighted variance is dominated by the across-cell shift plus
    the within-cell wiggle, both orthonormal directions on the n_bins
    grid.
    """
    shift_axis = np.zeros(n_bins)
    shift_axis[10:30] = 1.0
    shift_axis -= shift_axis.mean()
    shift_axis /= np.linalg.norm(shift_axis)

    wiggle_axis = np.zeros(n_bins)
    wiggle_axis[55:75] = 1.0
    wiggle_axis -= wiggle_axis.mean()
    wiggle_axis -= np.dot(wiggle_axis, shift_axis) * shift_axis
    wiggle_axis /= np.linalg.norm(wiggle_axis)

    n = n_cells * n_trials_per_cell
    posteriors = np.zeros((n, n_bins))
    for c in range(n_cells):
        cell_mean = c * shift_scale * shift_axis
        for k in range(n_trials_per_cell):
            i = c * n_trials_per_cell + k
            posteriors[i] = (cell_mean
                              + rng.normal(scale=wiggle_scale) * wiggle_axis)
    return posteriors, shift_axis, wiggle_axis


def _fit_pca_basis(posteriors):
    """Fit a full-rank PCA on the per-trial training targets — the
    'all_trials' basis used in production for Q / L / stim_kernel."""
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(posteriors)
    return pca.components_, pca.explained_variance_ratio_


def _marginal_baseline_loss_numpy(Y_train, Y_val, pcs, evar):
    """Pure-numpy reference for the PCA branch of
    :func:`optuna_per_target.marginal_baseline_loss`. Kept in
    lock-step with the torch implementation (the test below pins them
    together when torch is available) so the plot-layer helper can
    still be verified against the formula without depending on torch
    being installed in the test environment."""
    p = Y_train.mean(axis=0)
    P = np.tile(p[None, :], (Y_val.shape[0], 1))
    Pp = P @ pcs.T
    Yp = Y_val @ pcs.T
    return float(np.mean(np.sum(evar * (Pp - Yp) ** 2, axis=-1)) * 100)


def test_variance_baseline_matches_marginal_formula():
    """The plot-layer helper must compute the same formula as the
    Optuna search baseline: PCA-weighted mean squared distance from
    each val target to the train target mean (×100).

    Verified here against a pure-numpy reference (the formula); the
    next test pins that reference to the actual torch implementation
    when torch is available.
    """
    pytest.importorskip("sklearn")
    from decoder_plotting_utils import variance_baseline

    rng = np.random.default_rng(0)
    Y_train, _, _ = _plant_targets(rng)
    Y_val, _, _ = _plant_targets(np.random.default_rng(1))
    pcs, evar = _fit_pca_basis(Y_train)

    train_mean = Y_train.mean(axis=0)
    per_trial = variance_baseline(Y_val, train_mean, pcs, evar)
    plot_value = float(np.nanmean(per_trial))

    reference = _marginal_baseline_loss_numpy(Y_train, Y_val, pcs, evar)

    assert np.isfinite(plot_value)
    np.testing.assert_allclose(plot_value, reference, rtol=1e-10,
                                err_msg="variance_baseline does not match "
                                        "the marginal-mean formula.")


def test_marginal_baseline_numpy_matches_torch_implementation():
    """Pin the numpy reference in this file to the production torch
    ``optuna_per_target.marginal_baseline_loss``. Skipped if torch
    isn't installed in the test environment; runs on the dev machine
    and in CI where torch is available.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    # optuna_per_target imports optuna at module top; skip rather than fail when
    # that optional sweep dependency is absent (same policy as the ssm guard).
    pytest.importorskip("optuna")
    import torch
    from optuna_per_target import marginal_baseline_loss, DEVICE

    rng = np.random.default_rng(7)
    Y_train, _, _ = _plant_targets(rng)
    Y_val, _, _ = _plant_targets(np.random.default_rng(8))
    pcs, evar = _fit_pca_basis(Y_train)

    pcs_t = torch.tensor(pcs, dtype=torch.float32, device=DEVICE)
    var_t = torch.tensor(evar, dtype=torch.float32, device=DEVICE)
    torch_value = marginal_baseline_loss(Y_train, Y_val, 'PCA', pcs_t, var_t)
    numpy_value = _marginal_baseline_loss_numpy(Y_train, Y_val, pcs, evar)

    np.testing.assert_allclose(torch_value, numpy_value, rtol=1e-5,
                                err_msg="numpy reference diverged from the "
                                        "torch marginal_baseline_loss — if "
                                        "the production formula changed, "
                                        "update _marginal_baseline_loss_numpy "
                                        "in this file in lock-step.")


def test_variance_baseline_recovers_pc_weighted_variance():
    """In the large-sample limit, the variance baseline equals
    sum_i evar_i * Var_t[<P_t, PC_i>] * 100. The synthetic targets have
    a closed-form structure (across-cell shift + within-cell wiggle),
    so we can check the value approaches the planted variance."""
    pytest.importorskip("sklearn")
    from decoder_plotting_utils import variance_baseline

    rng = np.random.default_rng(2)
    Y, _, _ = _plant_targets(rng, n_cells=12, n_trials_per_cell=200)
    pcs, evar = _fit_pca_basis(Y)

    # Centroid = Y mean. Computing the baseline on the same set as the
    # basis was fit on should give: sum_i evar_i * total_variance_ratio_i.
    # PCA().explained_variance_ratio_ is normalised so it sums to 1; the
    # total raw variance (sum of components) lives in explained_variance_.
    from sklearn.decomposition import PCA
    pca_full = PCA().fit(Y)
    expected = float(np.sum(pca_full.explained_variance_ratio_
                             * pca_full.explained_variance_)) * 100

    centroid = Y.mean(axis=0)
    per_trial = variance_baseline(Y, centroid, pcs, evar)
    observed = float(np.nanmean(per_trial))

    # The variance baseline should agree with the analytic
    # PC-weighted variance up to a small N-sized residual.
    np.testing.assert_allclose(observed, expected, rtol=1e-3,
                                err_msg="variance_baseline did not recover "
                                        "the analytic PC-weighted variance "
                                        "on the planted targets.")


def test_variance_baseline_missing_basis_returns_nan():
    """Plot-layer NaN-tolerance contract: a missing basis or missing
    train mean gives a NaN-filled per-trial array (matches the pattern
    in decoder_plotting_utils.calc_pca_dist) so downstream
    ``np.nanmean`` reductions skip that cell instead of raising."""
    from decoder_plotting_utils import variance_baseline

    target = np.random.default_rng(3).random((20, 91))
    train_mean = target.mean(axis=0)
    pcs = np.eye(91)
    evar = np.full(91, 1.0 / 91)

    # No basis -> all NaN.
    out_no_pcs = variance_baseline(target, train_mean, None, evar)
    assert out_no_pcs.shape == (20,)
    assert np.isnan(out_no_pcs).all()

    # Empty basis -> all NaN.
    out_empty = variance_baseline(target, train_mean, np.array([]), evar)
    assert np.isnan(out_empty).all()

    # No train mean -> all NaN.
    out_no_mean = variance_baseline(target, None, pcs, evar)
    assert np.isnan(out_no_mean).all()

    # With everything present, output is finite.
    out_ok = variance_baseline(target, train_mean, pcs, evar)
    assert np.isfinite(out_ok).all()


def test_variance_baseline_invariant_to_centroid_constant():
    """The variance baseline depends on the test targets and the train
    centroid. Shifting both by the same constant should leave the
    answer unchanged (squared difference is translation-invariant).
    Subtle but important: when we fall back to test-target mean as the
    centroid for stim_kernel, the answer should not depend on overall
    DC offsets in the IO posterior representation."""
    from decoder_plotting_utils import variance_baseline

    rng = np.random.default_rng(4)
    Y, _, _ = _plant_targets(rng, n_cells=6, n_trials_per_cell=40)
    pcs, evar = _fit_pca_basis(Y)
    train_mean = Y.mean(axis=0)

    a = variance_baseline(Y, train_mean, pcs, evar).mean()

    offset = np.full(Y.shape[1], 0.7)
    b = variance_baseline(Y + offset, train_mean + offset, pcs, evar).mean()

    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
