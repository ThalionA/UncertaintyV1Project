# -*- coding: utf-8 -*-
"""Unit tests for the choice-from-distributions core in
``predict_choice_from_decoders``.

The script predicts the animal's trial-by-trial Go/NoGo choice from a
decoder's fitted distribution, in two variants:

  * WITH the IO stage-2 psychometric: decoded orientation posterior →
    log-odds → ``predict_choice_lapse(g, α_r, β_r, γ_r, δ_r)``.
  * WITHOUT: decoded log-odds → plain sigmoid (parameter-free; the
    lapse-free, slope-1, offset-0 special case).

For the d (decision) target there is no orientation posterior — its
decoded output is already a 2-D [P(Go), P(NoGo)], so P(Go) is read off
directly (without-variant only).

Tests anchor: monotonicity / direction of ``predict_pgo``, the lapse
bounds of the with-variant, the d-target passthrough, and the three
metrics (AUROC, accuracy, NLL) on synthetic data with known answers.
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

pytest.importorskip('scipy')
import predict_choice_from_decoders as pcd  # noqa: E402

GRID = np.arange(0, 91, 1, dtype=float)  # orientation grid, boundary at 45 deg


def _peaked_posterior(angle, n_trials=1, sharp=4.0):
    """A softmax posterior peaked at `angle` deg on the 0..90 grid."""
    logits = -sharp * np.abs(GRID[None, :] - angle) / 10.0
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return np.repeat(p, n_trials, axis=0)


# ----------------------------------------------------------------------
# predict_pgo — direction / monotonicity
# ----------------------------------------------------------------------

def test_without_variant_go_posterior_predicts_go():
    """A posterior peaked well inside the Go region (angle < 45) should
    give P(Go) > 0.5 in the parameter-free without-variant."""
    Q = _peaked_posterior(15, n_trials=3)
    p_go = pcd.predict_pgo(Q, target_type='Q', variant='without', grid=GRID)
    assert p_go.shape == (3,)
    assert np.all(p_go > 0.5)


def test_without_variant_nogo_posterior_predicts_nogo():
    """A posterior peaked in the No-Go region (angle > 45) → P(Go) < 0.5."""
    Q = _peaked_posterior(75, n_trials=3)
    p_go = pcd.predict_pgo(Q, target_type='Q', variant='without', grid=GRID)
    assert np.all(p_go < 0.5)


def test_without_variant_is_monotonic_in_peak_angle():
    """Sweeping the posterior peak from Go-side to No-Go-side, P(Go)
    must decrease monotonically."""
    angles = [5, 20, 35, 50, 65, 80]
    p_gos = [float(pcd.predict_pgo(_peaked_posterior(a), 'Q', 'without', grid=GRID)[0])
              for a in angles]
    diffs = np.diff(p_gos)
    assert np.all(diffs <= 1e-9), f"P(Go) not monotone decreasing: {p_gos}"


# ----------------------------------------------------------------------
# predict_pgo — with-variant (IO stage-2 lapse psychometric)
# ----------------------------------------------------------------------

def test_with_variant_respects_lapse_bounds():
    """The IO stage-2 psychometric is γ + (1-γ-δ)·sigmoid(...). With
    γ=0.1, δ=0.08 the output must stay within [0.1, 0.92] even for an
    extreme (near-deterministic) posterior."""
    io_params = (1.0, 0.0, 0.10, 0.08)  # (alpha, beta, gamma, delta)
    Q_go = _peaked_posterior(2, sharp=20.0)
    Q_nogo = _peaked_posterior(88, sharp=20.0)
    p_hi = pcd.predict_pgo(Q_go, 'Q', 'with', grid=GRID, io_params=io_params)
    p_lo = pcd.predict_pgo(Q_nogo, 'Q', 'with', grid=GRID, io_params=io_params)
    assert p_hi[0] <= 0.92 + 1e-9 and p_hi[0] >= 0.10 - 1e-9
    assert p_lo[0] <= 0.92 + 1e-9 and p_lo[0] >= 0.10 - 1e-9
    # Direction still preserved.
    assert p_hi[0] > p_lo[0]


def test_with_variant_zero_lapse_unit_slope_equals_without():
    """The without-variant is exactly the with-variant at
    (α=1, β=0, γ=0, δ=0) — a plain sigmoid of the log-odds."""
    Q = _peaked_posterior(30, n_trials=4)
    p_without = pcd.predict_pgo(Q, 'Q', 'without', grid=GRID)
    p_with_trivial = pcd.predict_pgo(Q, 'Q', 'with', grid=GRID,
                                       io_params=(1.0, 0.0, 0.0, 0.0))
    assert np.allclose(p_without, p_with_trivial, atol=1e-9)


# ----------------------------------------------------------------------
# predict_pgo — d (decision) target passthrough
# ----------------------------------------------------------------------

def test_d_target_returns_raw_pgo():
    """The d decoder outputs a 2-D [P(Go), P(NoGo)]; predict_pgo must
    return column 0 unchanged (no psychometric stage)."""
    decoded_d = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    p_go = pcd.predict_pgo(decoded_d, target_type='d', variant='without')
    assert np.allclose(p_go, [0.8, 0.3, 0.5])


# ----------------------------------------------------------------------
# choice_metrics
# ----------------------------------------------------------------------

def test_metrics_perfect_prediction():
    """Perfectly calibrated, perfectly ordered prediction: AUROC 1,
    accuracy 1, NLL ~ 0."""
    actual = np.array([0, 0, 1, 1, 1, 0], dtype=float)
    p_pred = np.where(actual == 1, 0.999, 0.001)
    m = pcd.choice_metrics(p_pred, actual)
    assert m['auroc'] == pytest.approx(1.0, abs=1e-9)
    assert m['accuracy'] == pytest.approx(1.0, abs=1e-9)
    assert m['nll'] < 0.02


def test_metrics_anti_perfect_prediction():
    """Predictions ranked exactly backwards → AUROC 0, accuracy 0."""
    actual = np.array([0, 0, 1, 1], dtype=float)
    p_pred = np.where(actual == 1, 0.01, 0.99)
    m = pcd.choice_metrics(p_pred, actual)
    assert m['auroc'] == pytest.approx(0.0, abs=1e-9)
    assert m['accuracy'] == pytest.approx(0.0, abs=1e-9)


def test_metrics_accuracy_uses_half_threshold():
    """Accuracy thresholds P(Go) at 0.5: hand-checked case."""
    actual = np.array([1, 1, 0, 0], dtype=float)
    # p>=0.5 -> predict Go. Trial 0,1 correct; trial 2 wrong (0.6->Go);
    # trial 3 correct (0.2->NoGo).  => 3/4 = 0.75
    p_pred = np.array([0.9, 0.55, 0.6, 0.2])
    m = pcd.choice_metrics(p_pred, actual)
    assert m['accuracy'] == pytest.approx(0.75, abs=1e-9)


def test_metrics_chance_prediction_auroc_near_half():
    """A random predictor scores AUROC ≈ 0.5 on a balanced set."""
    rng = np.random.RandomState(0)
    actual = np.array([0, 1] * 200, dtype=float)
    p_pred = rng.rand(400)
    m = pcd.choice_metrics(p_pred, actual)
    assert 0.40 < m['auroc'] < 0.60


def test_metrics_nll_penalises_confident_wrong():
    """NLL of a confidently-wrong prediction is much larger than NLL of
    a confidently-right one."""
    actual = np.array([1.0, 1.0])
    nll_right = pcd.choice_metrics(np.array([0.99, 0.99]), actual)['nll']
    nll_wrong = pcd.choice_metrics(np.array([0.01, 0.01]), actual)['nll']
    assert nll_wrong > 10 * nll_right
