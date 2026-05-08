# -*- coding: utf-8 -*-
"""Synthetic-data tests for the moment-decomposition analysis in
``nn_decoder/decomposition_analysis.py``.

Decomposes decoded vs target distributional losses into three moment-error
components (mean, variance, skewness) for the 91-bin Q(theta) / L(theta)
targets, and into a (P(Go) bias, entropy error) pair for the 2-bin decision
posterior.

Tests use known closed-form distributions on the production grids so the
expected component values can be derived analytically.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

import decomposition_analysis as da  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

GRID = np.arange(0, 91, 1, dtype=float)


def _gauss_on_grid(mu, sigma, grid=GRID):
    """Truncated Gaussian on the [0, 90] grid, normalised to sum to 1."""
    p = np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    return p / p.sum()


def _bimodal_on_grid(mu1, mu2, sigma, grid=GRID, w=0.5):
    p = w * _gauss_on_grid(mu1, sigma, grid) + (1 - w) * _gauss_on_grid(mu2, sigma, grid)
    return p / p.sum()


# ----------------------------------------------------------------------
# trial_moments
# ----------------------------------------------------------------------

def test_trial_moments_recovers_known_gaussian():
    """A narrow Gaussian on the bounded grid should recover its mu and sigma^2
    closely enough that the bounded-grid bias is < 1% relative error."""
    mu_true, sigma_true = 45.0, 5.0
    p = _gauss_on_grid(mu_true, sigma_true)[None, :]  # (1, n_bins)
    mu, var = da.trial_moments(p, GRID)
    assert mu.shape == (1,)
    assert var.shape == (1,)
    # Narrow Gaussian on [0, 90]: bounded-grid bias is tiny.
    assert mu[0] == pytest.approx(mu_true, abs=0.05)
    assert var[0] == pytest.approx(sigma_true ** 2, rel=0.02)


def test_trial_moments_uniform_on_grid():
    """Uniform on a 91-bin [0, 90] grid: mean = 45, var = (n^2 - 1)/12 of the
    discrete grid spacing. With grid spacing 1 deg and n=91 points,
    var ≈ (91^2 - 1)/12 = 690."""
    p = np.ones((1, len(GRID))) / len(GRID)
    mu, var = da.trial_moments(p, GRID)
    assert mu[0] == pytest.approx(45.0, abs=1e-9)
    assert var[0] == pytest.approx((91 ** 2 - 1) / 12.0, rel=1e-9)


def test_trial_moments_handles_multiple_trials():
    p = np.stack([
        _gauss_on_grid(20.0, 5.0),
        _gauss_on_grid(60.0, 10.0),
        _gauss_on_grid(45.0, 3.0),
    ])
    mu, var = da.trial_moments(p, GRID)
    assert mu.shape == (3,)
    assert mu[0] == pytest.approx(20.0, abs=0.1)
    assert mu[1] == pytest.approx(60.0, abs=0.1)
    assert mu[2] == pytest.approx(45.0, abs=0.05)
    assert var[0] == pytest.approx(25.0, rel=0.05)
    assert var[1] == pytest.approx(100.0, rel=0.05)
    assert var[2] == pytest.approx(9.0, rel=0.05)


# ----------------------------------------------------------------------
# trial_skewness
# ----------------------------------------------------------------------

def test_skewness_symmetric_is_zero():
    p = _gauss_on_grid(45.0, 5.0)[None, :]
    s = da.trial_skewness(p, GRID)
    assert s[0] == pytest.approx(0.0, abs=1e-6)


def test_skewness_right_skewed_is_positive():
    """Place a Gaussian near the lower edge so the bounded grid clips the
    left tail less than the right; the resulting distribution leans right."""
    p_skewed = _gauss_on_grid(20.0, 15.0)[None, :]
    s = da.trial_skewness(p_skewed, GRID)
    # Truncated near the left edge -> mass extends toward larger values -> +ve skew
    assert s[0] > 0.05


def test_skewness_left_skewed_is_negative():
    p_skewed = _gauss_on_grid(70.0, 15.0)[None, :]
    s = da.trial_skewness(p_skewed, GRID)
    assert s[0] < -0.05


# ----------------------------------------------------------------------
# decompose_distributional
# ----------------------------------------------------------------------

def test_decompose_identical_yields_zero():
    target = np.stack([_gauss_on_grid(30.0, 5.0), _gauss_on_grid(60.0, 10.0)])
    decoded = target.copy()
    res = da.decompose_distributional(decoded, target, GRID)
    assert np.allclose(res['mean_error'], 0.0, atol=1e-10)
    assert np.allclose(res['var_error'], 0.0, atol=1e-10)
    assert np.allclose(res['skew_error'], 0.0, atol=1e-10)
    assert np.allclose(res['total_l2_sq'], 0.0, atol=1e-10)


def test_decompose_pure_mean_shift():
    """Decoded is the target shifted by +5 deg. Mean error per trial ≈ 25;
    variance error and skewness error ≈ 0."""
    target = _gauss_on_grid(30.0, 5.0)[None, :]
    decoded = _gauss_on_grid(35.0, 5.0)[None, :]
    res = da.decompose_distributional(decoded, target, GRID)
    assert res['mean_error'][0] == pytest.approx(25.0, rel=0.05)
    assert abs(res['var_error'][0]) < 1.0
    assert abs(res['skew_error'][0]) < 0.05


def test_decompose_pure_scale_shift():
    """Same mean, different variance: mean error ≈ 0, variance error ≈
    (sigma_decoded^2 - sigma_target^2)^2 = (100 - 25)^2 = 5625."""
    target = _gauss_on_grid(45.0, 5.0)[None, :]
    decoded = _gauss_on_grid(45.0, 10.0)[None, :]
    res = da.decompose_distributional(decoded, target, GRID)
    assert abs(res['mean_error'][0]) < 0.5
    assert res['var_error'][0] == pytest.approx(5625.0, rel=0.05)


def test_decompose_pure_shape_change():
    """Match first two moments but change shape (unimodal -> bimodal). Mean
    error and variance error should be small; the residual L2 should be
    larger than what mean+variance can account for.

    Variance of bimodal(45-d, 45+d, sigma_b) = d^2 + sigma_b^2, so to match
    a target Gaussian with sigma_t = 12 (var=144) we pick d=10 and
    sigma_b=sqrt(44) so the bimodal has nominal var 100 + 44 = 144 too."""
    sigma_t = 12.0
    d = 10.0
    sigma_b = float(np.sqrt(sigma_t ** 2 - d ** 2))  # 44 -> sqrt ~ 6.633
    target = _gauss_on_grid(45.0, sigma_t)[None, :]
    decoded = _bimodal_on_grid(45.0 - d, 45.0 + d, sigma_b)[None, :]
    res = da.decompose_distributional(decoded, target, GRID)
    # First two moments matched by construction (modulo grid truncation,
    # which is small at this center/spread).
    assert abs(res['mean_error'][0]) < 1.0
    assert abs(res['var_error'][0]) < 25.0
    # Shape residual must be meaningfully positive: the bimodal shape
    # cannot be explained by a Gaussian moment match.
    assert res['shape_residual_l2_sq'][0] > 1e-4


def test_decompose_handles_multiple_trials_independently():
    """Pick centered Gaussians with sigma small enough that grid-truncation
    bias on the variance is negligible (< ~0.1%)."""
    target = np.stack([_gauss_on_grid(30.0, 5.0), _gauss_on_grid(50.0, 8.0)])
    decoded = np.stack([_gauss_on_grid(35.0, 5.0), _gauss_on_grid(50.0, 10.0)])
    res = da.decompose_distributional(decoded, target, GRID)
    # Trial 0: pure mean shift
    assert res['mean_error'][0] == pytest.approx(25.0, rel=0.05)
    assert abs(res['var_error'][0]) < 1.0
    # Trial 1: pure scale shift, var goes 64 -> 100 -> err = 36^2 = 1296
    assert abs(res['mean_error'][1]) < 0.5
    assert res['var_error'][1] == pytest.approx((100 - 64) ** 2, rel=0.05)


# ----------------------------------------------------------------------
# decompose_decision
# ----------------------------------------------------------------------

def _binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def test_decompose_decision_identical():
    target = np.array([[0.7, 0.3], [0.5, 0.5]])
    decoded = target.copy()
    res = da.decompose_decision(decoded, target)
    assert np.allclose(res['pgo_bias'], 0.0)
    assert np.allclose(res['entropy_error'], 0.0)


def test_decompose_decision_known_values():
    target = np.array([[0.5, 0.5]])
    decoded = np.array([[0.7, 0.3]])
    res = da.decompose_decision(decoded, target)
    # Bias = (0.7 - 0.5)^2 = 0.04
    assert res['pgo_bias'][0] == pytest.approx(0.04, abs=1e-9)
    # Entropy: H(0.7) ~ 0.881, H(0.5) = 1.0 -> diff^2 ~ 0.01415
    expected_h_err = (_binary_entropy(0.7) - _binary_entropy(0.5)) ** 2
    assert res['entropy_error'][0] == pytest.approx(expected_h_err, rel=1e-9)


def test_decompose_decision_handles_extreme_probs():
    """Avoid log(0) blowups when decoded approaches 0 or 1."""
    target = np.array([[1.0, 0.0]])
    decoded = np.array([[0.99, 0.01]])
    res = da.decompose_decision(decoded, target)
    # Should not be nan or inf
    assert np.isfinite(res['pgo_bias'][0])
    assert np.isfinite(res['entropy_error'][0])


# ----------------------------------------------------------------------
# Per-mouse aggregation
# ----------------------------------------------------------------------

def test_aggregate_components_to_per_mouse():
    """A long-form trial-level frame should aggregate to (mouse, arch, target,
    component) means matching np.mean across the appropriate slices."""
    rng = np.random.default_rng(0)
    rows = []
    for mouse in [0, 1]:
        for arch in ['spat', 'temp']:
            n_trials = 50
            for i in range(n_trials):
                rows.append(dict(
                    mouse=mouse, arch=arch, target='Q', split='strat',
                    trial=i, mean_error=rng.uniform(0, 1),
                    var_error=rng.uniform(0, 4), skew_error=rng.uniform(0, 0.5),
                    total_l2_sq=rng.uniform(0, 0.01),
                    shape_residual_l2_sq=rng.uniform(0, 0.001),
                ))
    df = pd.DataFrame(rows)
    summary = da.aggregate_per_mouse(df)
    # Should have one row per (mouse, arch, target, split)
    assert len(summary) == 2 * 2 * 1 * 1
    # Means must match a manual groupby.mean
    manual = (
        df.groupby(['mouse', 'arch', 'target', 'split'])
        ['mean_error'].mean().reset_index()
    )
    assert np.allclose(
        summary.sort_values(['mouse', 'arch'])['mean_error'].to_numpy(),
        manual.sort_values(['mouse', 'arch'])['mean_error'].to_numpy(),
    )


# ----------------------------------------------------------------------
# Load helpers — mock the .mat / .npy structure
# ----------------------------------------------------------------------

def _mock_production_results_dict(grid_size=91, n_trials=10, n_mice=2, seed=0):
    """Build the same in-memory shape that scipy.io.loadmat(simplify_cells=True)
    would return for population_results_*.mat files."""
    rng = np.random.default_rng(seed)
    results = {}
    for mid in range(n_mice):
        archs = {}
        for k in ['spat', 'temp', 'spat_shf', 'temp_shf']:
            decoded = rng.dirichlet(np.ones(grid_size), size=n_trials)
            target = rng.dirichlet(np.ones(grid_size), size=n_trials)
            archs[k] = {'decoded': decoded, 'target': target}
        archs['pcs'] = rng.normal(size=(grid_size, grid_size))
        archs['explained_var'] = rng.dirichlet(np.ones(grid_size))
        results[f'mouse_{mid}'] = {'Dist': archs}
    return {'results': results, 'config': {'custom_loss_func': 'PCA'}}


def test_extract_decompositions_distributional():
    mock = _mock_production_results_dict(grid_size=91, n_trials=8, n_mice=2)
    df = da.extract_decompositions_from_dict(
        mock, target_type='Q', split='strat',
        archs=('spat', 'temp', 'spat_shf', 'temp_shf'),
    )
    # 2 mice × 4 archs × 8 trials = 64 rows
    assert len(df) == 64
    # All expected columns present
    for col in ['mouse', 'arch', 'target', 'split', 'mean_error', 'var_error',
                'skew_error', 'total_l2_sq']:
        assert col in df.columns


def test_extract_decompositions_decision_skips_skew():
    """For 2-bin decision targets, skewness is undefined; the function should
    use decompose_decision and emit pgo_bias / entropy_error columns instead."""
    mock = _mock_production_results_dict(grid_size=2, n_trials=5, n_mice=1)
    df = da.extract_decompositions_from_dict(
        mock, target_type='d', split='strat',
        archs=('spat', 'temp', 'spat_shf', 'temp_shf'),
    )
    assert 'pgo_bias' in df.columns
    assert 'entropy_error' in df.columns
    assert 'mean_error' not in df.columns
