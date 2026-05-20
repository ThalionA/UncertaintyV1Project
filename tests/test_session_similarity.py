# -*- coding: utf-8 -*-
"""Synthetic-data tests for the per-session similarity analysis.

Covers the two genuinely new pieces of logic:

  - ``session_aggregation``: the trial-weighted hierarchical aggregation
    core — ``weighted_nanmean`` / ``weighted_mean_sem`` (Kish n_eff,
    NaN-aware), ``stability_vs_lag``, ``session_usable``. Equal-weight
    inputs must reduce exactly to the ordinary mean / SEM.
  - ``session_similarity_analysis._archetype_stability``: pairwise cosine
    between per-session archetypes — identical archetypes give 1, mutually
    orthogonal archetypes give 0, and the end-to-end split -> extract ->
    compare chain is directionally sensitive to a planted across-session
    drift.

The multi-session builder ``_make_multisession_mouse`` constructs a
``MouseData`` with a known ``session`` axis and easy Go / NoGo trials that
satisfy the fixed easy-block archetype definition (contrast = 1.0,
dispersion = 5.0, cardinal stimulus). ``drift`` rotates each session's
archetype directions so stability can be dialled down on demand.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NN_DECODER = os.path.join(REPO_ROOT, "nn_decoder")
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

import session_aggregation as agg  # noqa: E402
from dim_reduction_explore import MouseData, split_by_session  # noqa: E402
from similarity_analysis import _compute_archetypes  # noqa: E402
from session_similarity_analysis import (  # noqa: E402
    _archetype_stability,
    EASY_BLOCK_SIZE,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ======================================================================
# session_aggregation — weighted_nanmean
# ======================================================================


def test_weighted_nanmean_equal_weights_matches_plain_mean():
    rng = _rng(1)
    vals = rng.standard_normal((6, 4))
    got = agg.weighted_nanmean(vals, np.ones(6))
    assert np.allclose(got, vals.mean(axis=0), atol=1e-12)


def test_weighted_nanmean_known_scalar():
    # (1*1 + 3*3) / (1 + 3) = 10 / 4 = 2.5
    got = agg.weighted_nanmean(np.array([1.0, 3.0]), np.array([1.0, 3.0]))
    assert np.isclose(float(got), 2.5, atol=1e-12)


def test_weighted_nanmean_ignores_nan():
    # NaN entry drops out; mean of [1, 3] = 2.
    got = agg.weighted_nanmean(np.array([1.0, np.nan, 3.0]), np.ones(3))
    assert np.isclose(float(got), 2.0, atol=1e-12)


def test_weighted_nanmean_zero_total_weight_is_nan():
    got = agg.weighted_nanmean(np.array([np.nan, np.nan]), np.ones(2))
    assert np.isnan(float(got))
    got2 = agg.weighted_nanmean(np.array([1.0, 2.0]), np.zeros(2))
    assert np.isnan(float(got2))


def test_weighted_nanmean_negative_weight_raises():
    with pytest.raises(ValueError):
        agg.weighted_nanmean(np.array([1.0, 2.0]), np.array([1.0, -1.0]))


def test_weighted_nanmean_vector_shape_and_partial_nan():
    # Column 0 fully finite, column 1 has a NaN in row 1.
    vals = np.array([[1.0, 10.0], [3.0, np.nan]])
    got = agg.weighted_nanmean(vals, np.array([1.0, 3.0]))
    assert got.shape == (2,)
    assert np.isclose(got[0], (1 + 9) / 4)     # trial-weighted
    assert np.isclose(got[1], 10.0)            # only row 0 contributes


# ======================================================================
# session_aggregation — weighted_mean_sem
# ======================================================================


def test_weighted_mean_sem_equal_weights_matches_numpy():
    rng = _rng(2)
    vals = rng.standard_normal(8)
    mean, sem, n_eff = agg.weighted_mean_sem(vals, np.ones(8))
    assert np.isclose(float(mean), vals.mean(), atol=1e-12)
    expected_sem = np.std(vals, ddof=1) / np.sqrt(8)
    assert np.isclose(float(sem), expected_sem, atol=1e-12)
    assert np.isclose(float(n_eff), 8.0, atol=1e-12)


def test_weighted_mean_sem_kish_n_eff():
    # weights [1, 1, 2] -> n_eff = (4)^2 / (1 + 1 + 4) = 16 / 6.
    _, _, n_eff = agg.weighted_mean_sem(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]))
    assert np.isclose(float(n_eff), 16.0 / 6.0, atol=1e-12)


def test_weighted_mean_sem_single_sample_sem_nan():
    mean, sem, n_eff = agg.weighted_mean_sem(np.array([4.2]), np.array([7.0]))
    assert np.isclose(float(mean), 4.2, atol=1e-12)
    assert np.isnan(float(sem))            # cannot estimate spread from one
    assert np.isclose(float(n_eff), 1.0, atol=1e-12)


def test_weighted_mean_sem_vector_shape():
    rng = _rng(3)
    vals = rng.standard_normal((5, 7))
    mean, sem, n_eff = agg.weighted_mean_sem(vals, np.ones(5))
    assert mean.shape == sem.shape == n_eff.shape == (7,)
    assert np.allclose(mean, vals.mean(axis=0), atol=1e-12)


def test_session_to_mouse_then_mouse_to_group():
    # Two mice, each with two sessions. Trial-weighted all the way up.
    m1 = agg.session_to_mouse(np.array([2.0, 4.0]), np.array([10.0, 30.0]))
    m2 = agg.session_to_mouse(np.array([6.0, 6.0]), np.array([5.0, 5.0]))
    assert np.isclose(float(m1), (2 * 10 + 4 * 30) / 40.0)   # 3.5
    assert np.isclose(float(m2), 6.0)
    grp_mean, grp_sem, _ = agg.mouse_to_group(
        np.array([float(m1), float(m2)]), np.array([40.0, 10.0]))
    assert np.isclose(float(grp_mean), (3.5 * 40 + 6.0 * 10) / 50.0)
    assert np.isfinite(float(grp_sem))


# ======================================================================
# session_aggregation — stability_vs_lag
# ======================================================================


def _sym_matrix(n: int, fill_fn) -> np.ndarray:
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = fill_fn(i, j)
    return M


def test_stability_vs_lag_constant():
    M = _sym_matrix(5, lambda i, j: 0.9)
    lags, mean, sem, n_pairs = agg.stability_vs_lag(M)
    assert list(lags) == [1, 2, 3, 4]
    assert np.allclose(mean, 0.9)


def test_stability_vs_lag_linear_decay():
    # similarity falls 0.1 per unit of lag.
    M = _sym_matrix(6, lambda i, j: 1.0 - 0.1 * abs(i - j))
    lags, mean, _, _ = agg.stability_vs_lag(M)
    for lag, mu in zip(lags, mean):
        assert np.isclose(mu, 1.0 - 0.1 * lag, atol=1e-12)


def test_stability_vs_lag_pair_counts():
    M = _sym_matrix(4, lambda i, j: 0.5)
    _, _, _, n_pairs = agg.stability_vs_lag(M)
    assert list(n_pairs) == [3, 2, 1]      # lag 1, 2, 3


def test_stability_vs_lag_skips_nan():
    M = _sym_matrix(4, lambda i, j: 0.8)
    M[0, 1] = M[1, 0] = np.nan             # drop one lag-1 pair
    lags, mean, _, n_pairs = agg.stability_vs_lag(M)
    assert n_pairs[0] == 2                  # lag 1 lost one pair
    assert np.isclose(mean[0], 0.8)


def test_stability_vs_lag_requires_square():
    with pytest.raises(ValueError):
        agg.stability_vs_lag(np.zeros((3, 4)))


# ======================================================================
# session_aggregation — session_usable
# ======================================================================


def test_session_usable():
    assert agg.session_usable({"Go": 1, "NoGo": 2})
    assert not agg.session_usable({"Go": 1})
    assert not agg.session_usable({})


# ======================================================================
# _archetype_stability — hand-built archetype dicts (no z-scoring)
# ======================================================================


def test_archetype_stability_identical_sessions():
    rng = _rng(10)
    A = rng.standard_normal((30, 12))
    B = rng.standard_normal((30, 12))
    sessions = [{"Go": A.copy(), "NoGo": B.copy()} for _ in range(3)]
    stab = _archetype_stability(sessions)
    assert np.isclose(stab["Go"]["mean"], 1.0, atol=1e-9)
    assert np.isclose(stab["NoGo"]["mean"], 1.0, atol=1e-9)
    assert np.allclose(stab["Go"]["xg_trace"], 1.0, atol=1e-9)


def test_archetype_stability_orthogonal_sessions():
    # Each session's Go archetype lives along a distinct orthonormal neuron
    # direction -> pairwise cosine is exactly zero.
    n_neurons, n_xG = 12, 8
    basis = np.linalg.qr(_rng(11).standard_normal((n_neurons, n_neurons)))[0]
    ramp = np.linspace(0.2, 1.0, n_xG)
    sessions = []
    for k in range(3):
        go = basis[:, k][:, None] * ramp[None, :]
        nogo = basis[:, k + 3][:, None] * ramp[None, :]
        sessions.append({"Go": go, "NoGo": nogo})
    stab = _archetype_stability(sessions)
    assert abs(stab["Go"]["mean"]) < 1e-9
    assert abs(stab["NoGo"]["mean"]) < 1e-9


def test_archetype_stability_shapes():
    rng = _rng(12)
    sessions = [{"Go": rng.standard_normal((20, 10)),
                 "NoGo": rng.standard_normal((20, 10))} for _ in range(4)]
    stab = _archetype_stability(sessions)
    assert stab["Go"]["matrix"].shape == (4, 4)
    assert stab["Go"]["xg_trace"].shape == (10,)
    assert "xg_trace_sem" in stab["Go"]


def test_archetype_stability_single_session_is_nan():
    rng = _rng(13)
    stab = _archetype_stability(
        [{"Go": rng.standard_normal((10, 6)),
          "NoGo": rng.standard_normal((10, 6))}])
    assert np.isnan(stab["Go"]["mean"])
    assert stab["Go"]["xg_trace"] is None


# ======================================================================
# End-to-end: split -> per-session archetypes -> stability
# ======================================================================


def _make_multisession_mouse(
    n_sessions: int = 4,
    n_easy_per_side: int = 10,
    n_neurons: int = 40,
    n_xG: int = 12,
    *,
    seed: int = 0,
    noise_amp: float = 0.02,
    drift: float = 0.0,
) -> MouseData:
    """Build a MouseData with a known session axis and easy Go/NoGo trials.

    Each session's archetype directions are rotated by ``drift * session``
    radians away from a fixed base pair, so ``drift = 0`` yields sessions
    with identical generative archetypes (high stability) and larger
    ``drift`` dials stability down.
    """
    rng = np.random.default_rng(seed)
    # Two orthonormal base directions, plus orthonormal "drift" directions.
    Q = np.linalg.qr(rng.standard_normal((n_neurons, n_neurons)))[0]
    go0, nogo0 = Q[:, 0], Q[:, 1]
    go_perp, nogo_perp = Q[:, 2], Q[:, 3]
    ramp = np.linspace(0.2, 1.0, n_xG)

    acts, sess, ori, true_ori, contrast, disp = [], [], [], [], [], []
    for sid in range(n_sessions):
        theta = drift * sid
        go_s = go0 * np.cos(theta) + go_perp * np.sin(theta)
        nogo_s = nogo0 * np.cos(theta) + nogo_perp * np.sin(theta)
        for direction, side_ori, side_true in (
            (go_s, 15.0, 0.0), (nogo_s, 75.0, 90.0)):
            for _ in range(n_easy_per_side):
                a = (direction[:, None] * ramp[None, :]
                     + noise_amp * rng.standard_normal((n_neurons, n_xG)))
                acts.append(a)
                sess.append(sid)
                ori.append(side_ori)
                true_ori.append(side_true)
                contrast.append(1.0)        # easy-block contrast
                disp.append(5.0)            # easy-block dispersion

    activity = np.asarray(acts)
    n_trials = activity.shape[0]
    return MouseData(
        tag="synthMS",
        activity=activity,
        xG=np.linspace(0.0, 2.0, n_xG),
        orientation=np.asarray(ori),
        true_orientation=np.asarray(true_ori),
        contrast=np.asarray(contrast),
        dispersion=np.asarray(disp),
        choice=np.zeros(n_trials),
        velocity=np.zeros(n_trials),
        lick_rate=np.zeros(n_trials),
        post_s_marginal=np.zeros((n_trials, 3)),
        decision_posterior=np.zeros((n_trials, 2)),
        session=np.asarray(sess, dtype=int),
        go_orientation=0.0,
    )


def test_end_to_end_split_produces_one_object_per_session():
    m = _make_multisession_mouse(n_sessions=5)
    sessions = split_by_session(m)
    assert len(sessions) == 5
    assert {s.n_trials for s in sessions} == {20}   # 10 Go + 10 NoGo each


def test_end_to_end_stability_high_without_drift():
    m = _make_multisession_mouse(n_sessions=4, drift=0.0, noise_amp=0.02)
    archs = [_compute_archetypes(s, easy_block_size=EASY_BLOCK_SIZE)
             for s in split_by_session(m)]
    assert all(agg.session_usable(a) for a in archs)
    stab = _archetype_stability(archs)
    assert stab["Go"]["mean"] > 0.8
    assert stab["NoGo"]["mean"] > 0.8


def test_end_to_end_drift_lowers_stability():
    common = dict(n_sessions=4, noise_amp=0.02, seed=7)
    archs_flat = [_compute_archetypes(s, easy_block_size=EASY_BLOCK_SIZE)
                  for s in split_by_session(
                      _make_multisession_mouse(drift=0.0, **common))]
    archs_drift = [_compute_archetypes(s, easy_block_size=EASY_BLOCK_SIZE)
                   for s in split_by_session(
                       _make_multisession_mouse(drift=0.5, **common))]
    stab_flat = _archetype_stability(archs_flat)
    stab_drift = _archetype_stability(archs_drift)
    for side in ("Go", "NoGo"):
        assert stab_drift[side]["mean"] < stab_flat[side]["mean"]


def test_end_to_end_stability_vs_lag_runs_on_real_matrix():
    m = _make_multisession_mouse(n_sessions=5, drift=0.3)
    archs = [_compute_archetypes(s, easy_block_size=EASY_BLOCK_SIZE)
             for s in split_by_session(m)]
    stab = _archetype_stability(archs)
    lags, mean, sem, n_pairs = agg.stability_vs_lag(stab["Go"]["matrix"])
    assert list(lags) == [1, 2, 3, 4]
    assert list(n_pairs) == [4, 3, 2, 1]
    assert np.all(np.isfinite(mean))
