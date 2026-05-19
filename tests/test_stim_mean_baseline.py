# -*- coding: utf-8 -*-
"""Synthetic-data tests for the stimulus-only baseline in
``nn_decoder/stim_mean_baseline.py``.

The stim-mean baseline is the closed-form 'predictor' that takes the
trial's stimulus condition (orientation, contrast, dispersion) and outputs
the per-condition mean training target. Tests verify:

  - compute_condition_means matches np.mean per group on synthetic data.
  - predict_stim_mean returns the right mean per condition and falls back
    correctly for unseen conditions on OOD splits.
  - predict_stim_mean_with_orientation_fallback drops to the orientation-
    only mean when the full triplet isn't seen and to the global mean
    when even the orientation isn't seen.
  - Predictions have zero within-condition variance (every trial of the
    same condition gets the same row).
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


def _baseline():
    """Lazy import: stim_mean_baseline pulls in utils (torch). Skip the
    test module if that import chain fails in the runner's environment."""
    try:
        import stim_mean_baseline as smb
    except Exception as exc:
        pytest.skip(f"stim_mean_baseline import failed: {exc}")
    return smb


# ----------------------------------------------------------------------
# compute_condition_means
# ----------------------------------------------------------------------

def test_compute_condition_means_matches_groupby():
    smb = _baseline()
    rng = np.random.default_rng(0)
    # Two conditions, 5 trials each, 91-D targets
    cond_a = (15.0, 0.5, 30.0)
    cond_b = (75.0, 1.0, 5.0)
    conditions = np.array([cond_a] * 5 + [cond_b] * 5)
    targets = rng.normal(size=(10, 91))

    means = smb.compute_condition_means(conditions, targets)
    assert tuple(map(float, cond_a)) in means
    assert tuple(map(float, cond_b)) in means

    expected_a = targets[:5].mean(axis=0)
    expected_b = targets[5:].mean(axis=0)
    assert np.allclose(means[tuple(map(float, cond_a))], expected_a)
    assert np.allclose(means[tuple(map(float, cond_b))], expected_b)


def test_compute_condition_means_handles_singleton_condition():
    smb = _baseline()
    conditions = np.array([(0.0, 1.0, 5.0)])
    targets = np.arange(91, dtype=float).reshape(1, -1)
    means = smb.compute_condition_means(conditions, targets)
    assert np.allclose(means[(0.0, 1.0, 5.0)], targets[0])


# ----------------------------------------------------------------------
# predict_stim_mean
# ----------------------------------------------------------------------

def test_predict_stim_mean_returns_known_means():
    smb = _baseline()
    means = {(0.0, 1.0, 5.0): np.full(91, 0.1),
             (90.0, 1.0, 5.0): np.full(91, 0.9)}
    fallback = np.full(91, 0.5)
    test_conds = np.array([(0.0, 1.0, 5.0), (90.0, 1.0, 5.0), (0.0, 1.0, 5.0)])
    pred = smb.predict_stim_mean(test_conds, means, fallback)
    assert np.allclose(pred[0], 0.1)
    assert np.allclose(pred[1], 0.9)
    assert np.allclose(pred[2], 0.1)


def test_predict_stim_mean_falls_back_for_unseen_condition():
    smb = _baseline()
    means = {(0.0, 1.0, 5.0): np.full(91, 0.1)}
    fallback = np.full(91, 0.5)
    test_conds = np.array([(45.0, 0.25, 90.0)])
    pred = smb.predict_stim_mean(test_conds, means, fallback)
    assert np.allclose(pred[0], 0.5)


# ----------------------------------------------------------------------
# Orientation fallback (OOD case)
# ----------------------------------------------------------------------

def test_orientation_fallback_takes_priority_over_global():
    smb = _baseline()
    cond_means = {(0.0, 1.0, 5.0): np.full(91, 0.1)}
    orientation_means = {0.0: np.full(91, 0.2),
                         90.0: np.full(91, 0.8)}
    global_mean = np.full(91, 0.5)
    # Condition not seen, but orientation 0 is in orientation_means.
    test_conds = np.array([(0.0, 0.5, 30.0)])
    pred = smb.predict_stim_mean_with_orientation_fallback(
        test_conds, cond_means, orientation_means, global_mean
    )
    assert np.allclose(pred[0], 0.2)


def test_orientation_fallback_drops_to_global_when_orientation_unseen():
    smb = _baseline()
    cond_means = {(0.0, 1.0, 5.0): np.full(91, 0.1)}
    orientation_means = {0.0: np.full(91, 0.2)}
    global_mean = np.full(91, 0.5)
    # Orientation 60 isn't in either dict -> global mean.
    test_conds = np.array([(60.0, 0.5, 30.0)])
    pred = smb.predict_stim_mean_with_orientation_fallback(
        test_conds, cond_means, orientation_means, global_mean
    )
    assert np.allclose(pred[0], 0.5)


def test_orientation_fallback_full_match_uses_full_condition_mean():
    smb = _baseline()
    cond_means = {(0.0, 1.0, 5.0): np.full(91, 0.1)}
    orientation_means = {0.0: np.full(91, 0.2)}
    global_mean = np.full(91, 0.5)
    test_conds = np.array([(0.0, 1.0, 5.0)])
    pred = smb.predict_stim_mean_with_orientation_fallback(
        test_conds, cond_means, orientation_means, global_mean
    )
    # Full condition seen -> use full-condition mean, not orientation mean.
    assert np.allclose(pred[0], 0.1)


# ----------------------------------------------------------------------
# Within-condition variance
# ----------------------------------------------------------------------

def test_predictions_have_zero_within_condition_variance():
    """Every trial sharing a condition must get an identical row."""
    smb = _baseline()
    cond = (45.0, 0.5, 30.0)
    cond_means = {cond: np.arange(91, dtype=float)}
    fallback = np.zeros(91)
    test_conds = np.array([cond] * 8)
    pred = smb.predict_stim_mean(test_conds, cond_means, fallback)
    # All 8 rows identical
    assert np.allclose(pred, pred[0:1])


# ----------------------------------------------------------------------
# 2-bin decision target works the same way
# ----------------------------------------------------------------------

def test_compute_condition_means_recovers_per_condition_pgo():
    """For target_type='choice', the condition mean of one-hot rows
    should be exactly the per-condition empirical Go rate."""
    smb = _baseline()
    cond_a = (15.0, 0.5, 30.0)
    cond_b = (75.0, 1.0, 5.0)
    # cond A: 3 Go / 4 trials -> P(Go)=0.75
    # cond B: 1 Go / 4 trials -> P(Go)=0.25
    conditions = np.array([cond_a]*4 + [cond_b]*4)
    choices_one_hot = np.array([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
    ])
    means = smb.compute_condition_means(conditions, choices_one_hot)
    assert np.allclose(means[tuple(map(float, cond_a))], [0.75, 0.25])
    assert np.allclose(means[tuple(map(float, cond_b))], [0.25, 0.75])


def test_select_target_choice_branch_constructs_one_hot():
    """The 'choice' branch of _select_target should turn binary goChoice
    into a 2-bin [P(Go), P(NoGo)] target."""
    smb = _baseline()
    trials = {'choice': np.array([1, 0, 1, 1, 0])}
    target = smb._select_target('choice', None, None, None, trials=trials)
    assert target.shape == (5, 2)
    expected = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]], dtype=float)
    assert np.allclose(target, expected)


def test_select_target_choice_raises_without_trials():
    smb = _baseline()
    with pytest.raises(ValueError, match="trials dict"):
        smb._select_target('choice', None, None, None)


def test_predict_stim_mean_handles_2bin_decision():
    smb = _baseline()
    cond_means = {(0.0, 1.0, 5.0): np.array([0.9, 0.1]),
                  (90.0, 1.0, 5.0): np.array([0.1, 0.9])}
    fallback = np.array([0.5, 0.5])
    test_conds = np.array([(0.0, 1.0, 5.0), (90.0, 1.0, 5.0), (45.0, 1.0, 5.0)])
    pred = smb.predict_stim_mean(test_conds, cond_means, fallback)
    assert pred.shape == (3, 2)
    assert np.allclose(pred[0], [0.9, 0.1])
    assert np.allclose(pred[1], [0.1, 0.9])
    assert np.allclose(pred[2], [0.5, 0.5])
