# -*- coding: utf-8 -*-
"""Unit tests for ``plot_spat_temp_divergence``.

The module ranks per-trial decoder performance by
``delta = pca_loss(spat) - pca_loss(temp)`` and exposes helpers to
(a) compute the per-trial delta, (b) pick top/bottom quantile masks,
(c) align posteriors so the true-orientation bin sits at the centre,
and (d) average posteriors within a quantile. These tests anchor each
helper on synthetic data with a known ground truth, so regressions in
sign convention, alignment, or quantile selection surface as test
failures before they reach a figure.
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
import plot_spat_temp_divergence as div  # noqa: E402


# ----------------------------------------------------------------------
# Helpers — build a minimal Dist-shaped dict and an orthonormal PCA basis
# ----------------------------------------------------------------------

def _identity_basis(n_cats: int):
    """PC basis = identity (n_cats components), evar uniform.

    With this basis, ``pca_distance`` reduces to ``100 * mean MSE / n_cats``
    times a constant — it preserves ranking but the absolute value is
    different from raw MSE. We only use *ranking* in the tests, so this
    is fine, and using the identity keeps the algebra trivial to follow.
    """
    pcs = np.eye(n_cats, dtype=np.float64)
    evar = np.full(n_cats, 1.0 / n_cats, dtype=np.float64)
    return pcs, evar


def _make_dist(decoded_spat, decoded_temp, target, n_cats):
    """Build a Dist-shaped nested dict the way ``compute_per_trial_delta``
    expects after the .mat is squeezed out of its object arrays."""
    pcs, evar = _identity_basis(n_cats)
    return {
        'spat': {'decoded': decoded_spat, 'target': target},
        'temp': {'decoded': decoded_temp, 'target': target},
        'spat_shf': {'decoded': decoded_spat.copy(), 'target': target},
        'temp_shf': {'decoded': decoded_temp.copy(), 'target': target},
        'pcs': pcs,
        'explained_var': evar,
    }


# ----------------------------------------------------------------------
# compute_per_trial_delta — sign convention with known ground truth
# ----------------------------------------------------------------------

def test_delta_sign_when_spat_perfect_and_temp_noisy():
    """If spat == target exactly and temp is noisy, delta = loss_spat -
    loss_temp must be negative on every trial (spat much better)."""
    rng = np.random.default_rng(0)
    n_trials, n_cats = 50, 9
    target = rng.dirichlet(np.ones(n_cats), size=n_trials)
    decoded_spat = target.copy()
    decoded_temp = target + 0.1 * rng.standard_normal(target.shape)

    out = div.compute_per_trial_delta(
        _make_dist(decoded_spat, decoded_temp, target, n_cats))

    assert out['delta'].shape == (n_trials,)
    assert np.all(out['delta'] <= 0), \
        f"expected delta<=0 everywhere (spat perfect), got max={out['delta'].max()}"
    assert np.allclose(out['loss_spat'], 0.0, atol=1e-12), \
        "loss_spat should be 0 when decoded == target"


def test_delta_sign_flipped_when_temp_perfect():
    """Symmetric: temp == target, spat noisy => delta >= 0 everywhere."""
    rng = np.random.default_rng(1)
    n_trials, n_cats = 50, 9
    target = rng.dirichlet(np.ones(n_cats), size=n_trials)
    decoded_temp = target.copy()
    decoded_spat = target + 0.1 * rng.standard_normal(target.shape)

    out = div.compute_per_trial_delta(
        _make_dist(decoded_spat, decoded_temp, target, n_cats))

    assert np.all(out['delta'] >= 0), \
        f"expected delta>=0 everywhere (temp perfect), got min={out['delta'].min()}"


def test_delta_target_mismatch_raises():
    """spat and temp must share the same target array — they do by
    construction in the production .mat, and an assertion guards
    against schema drift."""
    n_trials, n_cats = 8, 9
    rng = np.random.default_rng(2)
    target = rng.dirichlet(np.ones(n_cats), size=n_trials)
    other = rng.dirichlet(np.ones(n_cats), size=n_trials)
    pcs, evar = _identity_basis(n_cats)
    bad_dist = {
        'spat': {'decoded': target.copy(), 'target': target},
        'temp': {'decoded': target.copy(), 'target': other},
        'pcs': pcs, 'explained_var': evar,
    }
    with pytest.raises(AssertionError):
        div.compute_per_trial_delta(bad_dist)


# ----------------------------------------------------------------------
# select_quantiles — masks pick the right trials
# ----------------------------------------------------------------------

def test_select_quantiles_picks_extremes():
    """delta = arange(n). Bottom decile = smallest 10%, top decile =
    largest 10% of indices."""
    delta = np.arange(100, dtype=float)
    top, bot = div.select_quantiles(delta, q=0.10)
    assert np.array_equal(np.where(bot)[0], np.arange(10))
    assert np.array_equal(np.where(top)[0], np.arange(90, 100))


def test_select_quantiles_quartile():
    delta = np.arange(40, dtype=float)
    top, bot = div.select_quantiles(delta, q=0.25)
    assert bot.sum() == 10
    assert top.sum() == 10
    # Bottom = lowest delta, top = highest delta.
    assert delta[bot].max() < delta[top].min()


def test_select_quantiles_nan_excluded():
    """NaN deltas must not be selected into either quantile."""
    delta = np.arange(20, dtype=float)
    delta[[3, 17]] = np.nan
    top, bot = div.select_quantiles(delta, q=0.10)
    assert not bot[3] and not bot[17]
    assert not top[3] and not top[17]


# ----------------------------------------------------------------------
# align_to_truth — Kronecker delta rolls to the centre
# ----------------------------------------------------------------------

def test_align_delta_posteriors_land_at_centre():
    """Posterior is a Kronecker delta at index `c`; after aligning by
    true category `c`, the delta must sit at n_cats//2."""
    n_cats = 91
    centre = n_cats // 2
    cats = np.array([0, 10, 45, 70, 90])
    post = np.zeros((len(cats), n_cats))
    post[np.arange(len(cats)), cats] = 1.0
    aligned = div.align_to_truth(post, cats, n_cats=n_cats)
    assert aligned.shape == post.shape
    assert np.all(aligned[:, centre] == 1.0)
    # All mass should be at the centre column, nowhere else.
    others = np.delete(aligned, centre, axis=1)
    assert np.allclose(others, 0.0)


def test_align_preserves_row_sums():
    """Alignment is a per-row roll; it must not change total mass."""
    rng = np.random.default_rng(3)
    n_trials, n_cats = 30, 91
    post = rng.dirichlet(np.ones(n_cats), size=n_trials)
    cats = rng.integers(0, n_cats, size=n_trials)
    aligned = div.align_to_truth(post, cats, n_cats=n_cats)
    assert np.allclose(aligned.sum(axis=1), post.sum(axis=1))


# ----------------------------------------------------------------------
# mean_in_quantile — uniform inputs return uniform
# ----------------------------------------------------------------------

def test_mean_in_quantile_uniform_returns_uniform():
    """If every selected trial has identical posteriors, the mean must
    be the same posterior — guards against accidental normalisation or
    axis bugs."""
    n_trials, n_cats = 20, 9
    p_spat = np.tile(np.linspace(0, 1, n_cats), (n_trials, 1))
    p_temp = np.tile(np.linspace(1, 0, n_cats), (n_trials, 1))
    target = np.tile(np.ones(n_cats) / n_cats, (n_trials, 1))
    cats = np.full(n_trials, n_cats // 2)  # already centred
    idx = np.ones(n_trials, dtype=bool)
    idx[:5] = False  # pick 15 of 20

    m_spat, m_temp, m_tgt = div.mean_in_quantile(
        p_spat, p_temp, target, idx, align=False, true_cat=cats)
    assert np.allclose(m_spat, p_spat[0])
    assert np.allclose(m_temp, p_temp[0])
    assert np.allclose(m_tgt, target[0])


def test_mean_in_quantile_empty_idx_returns_nan():
    """Empty selection should return NaN curves, not raise — keeps the
    plotting loop simple when a quantile is empty for a small mouse."""
    n_trials, n_cats = 10, 9
    p = np.zeros((n_trials, n_cats))
    idx = np.zeros(n_trials, dtype=bool)
    cats = np.zeros(n_trials, dtype=int)
    m_spat, m_temp, m_tgt = div.mean_in_quantile(
        p, p, p, idx, align=False, true_cat=cats)
    assert np.all(np.isnan(m_spat))
    assert np.all(np.isnan(m_temp))
    assert np.all(np.isnan(m_tgt))


# ----------------------------------------------------------------------
# weighted_pool_across_mice — trial-count weighting (per memory note)
# ----------------------------------------------------------------------

def test_weighted_pool_matches_trial_count_weighting():
    """Per-mouse mean curves pooled across mice must weight by trial
    count, not by mouse (matches feedback_trial_count_weighting)."""
    n_cats = 5
    per_mouse_curves = [np.full(n_cats, 1.0), np.full(n_cats, 3.0)]
    weights = [10, 30]  # 25%/75%
    pooled = div.weighted_pool(per_mouse_curves, weights)
    # weighted mean = (10*1 + 30*3) / 40 = 100/40 = 2.5
    assert np.allclose(pooled, 2.5)


def test_weighted_pool_skips_nan_curves():
    """A mouse with NaN curve (empty quantile) should be excluded; the
    pooled curve uses only the remaining mice/weights."""
    n_cats = 5
    curves = [np.full(n_cats, 2.0), np.full(n_cats, np.nan), np.full(n_cats, 4.0)]
    weights = [10, 20, 10]
    pooled = div.weighted_pool(curves, weights)
    # (10*2 + 10*4) / 20 = 3.0
    assert np.allclose(pooled, 3.0)
