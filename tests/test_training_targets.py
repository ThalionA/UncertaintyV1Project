# -*- coding: utf-8 -*-
"""Unit tests for ``nn_decoder/training/targets.py``.

Verifies the per-target supervised-label construction is correct for
each which_model, given synthetic trial inputs.
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

from training.targets import make_target, n_bins_for, GRID_DEG, STIM_KERNEL_SIGMA_DEG  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _trials(orientations, choices=None):
    if choices is None:
        choices = np.zeros(len(orientations))
    return {
        'orientation': np.asarray(orientations, dtype=float),
        'choice': np.asarray(choices, dtype=float),
        'contrast': np.ones(len(orientations)),
        'dispersion': np.full(len(orientations), 5),
    }


# ----------------------------------------------------------------------
# Pre-existing IO targets pass through
# ----------------------------------------------------------------------

def test_perception_passes_through_targets_perc():
    Q = np.random.default_rng(0).dirichlet(np.ones(91), size=10)
    out = make_target('perception', _trials(np.zeros(10)), targets_perc=Q)
    assert np.allclose(out, Q)


def test_likelihood_raises_when_targets_lik_missing():
    with pytest.raises(ValueError, match='requires targets_lik'):
        make_target('likelihood', _trials(np.zeros(5)), targets_lik=None)


def test_decision_passes_through_targets_dec():
    d = np.random.default_rng(1).dirichlet(np.ones(2), size=5)
    out = make_target('decision', _trials(np.zeros(5)), targets_dec=d)
    assert np.allclose(out, d)


def test_unknown_which_model_raises():
    with pytest.raises(ValueError, match='Unknown which_model'):
        make_target('not_a_target', _trials(np.zeros(3)))


# ----------------------------------------------------------------------
# true_choice — animal goChoice one-hot
# ----------------------------------------------------------------------

def test_true_choice_one_hots_binary_choice():
    choices = np.array([1, 0, 1, 1, 0])
    out = make_target('true_choice', _trials(np.zeros(5), choices=choices))
    assert out.shape == (5, 2)
    expected = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]], dtype=float)
    assert np.allclose(out, expected)


def test_true_choice_each_row_sums_to_one():
    choices = np.array([0, 1, 0, 1])
    out = make_target('true_choice', _trials(np.zeros(4), choices=choices))
    assert np.allclose(out.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# stim_kernel — Gaussian smoothed delta
# ----------------------------------------------------------------------

def test_stim_kernel_rows_sum_to_one():
    oris = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
    out = make_target('stim_kernel', _trials(oris))
    assert out.shape == (5, 91)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_stim_kernel_peaks_at_true_orientation():
    oris = np.array([15.0, 45.0, 75.0])
    out = make_target('stim_kernel', _trials(oris))
    peak_bins = np.argmax(out, axis=1)
    assert np.allclose(peak_bins, oris.astype(int))


def test_stim_kernel_width_matches_sigma():
    """A Gaussian on the bounded grid should have empirical SD close to
    STIM_KERNEL_SIGMA_DEG when far enough from the boundary."""
    oris = np.array([45.0])
    out = make_target('stim_kernel', _trials(oris))
    p = out[0]
    mu = (p * GRID_DEG).sum()
    var = (p * (GRID_DEG - mu) ** 2).sum()
    sigma_emp = float(np.sqrt(var))
    assert abs(sigma_emp - STIM_KERNEL_SIGMA_DEG) < 0.5


# ----------------------------------------------------------------------
# stim_cat — one-hot at true bin
# ----------------------------------------------------------------------

def test_stim_cat_is_one_hot_at_true_bin():
    oris = np.array([0.0, 30.0, 45.0, 90.0])
    out = make_target('stim_cat', _trials(oris))
    assert out.shape == (4, 91)
    for i, ori in enumerate(oris):
        bin_idx = int(round(ori))
        assert out[i, bin_idx] == 1.0
        assert out[i].sum() == 1.0
        assert (out[i] > 0).sum() == 1


def test_stim_cat_clips_out_of_grid_orientations():
    """If an orientation falls slightly outside [0, 90] (rounding error)
    the one-hot still lands on a valid bin (clipped)."""
    oris = np.array([-0.4, 90.4, 45.5])
    out = make_target('stim_cat', _trials(oris))
    bin_idx = np.argmax(out, axis=1)
    # -0.4 rounds to 0 then clipped to 0; 90.4 rounds to 90 (clipped from
    # 90); 45.5 rounds half-to-even -> 46 (numpy convention)
    assert bin_idx[0] == 0
    assert bin_idx[1] == 90
    assert bin_idx[2] in (45, 46)


# ----------------------------------------------------------------------
# n_bins_for
# ----------------------------------------------------------------------

@pytest.mark.parametrize('which,expected', [
    ('perception', 91),
    ('likelihood', 91),
    ('stim_kernel', 91),
    ('stim_cat', 91),
    ('decision', 2),
    ('true_choice', 2),
    ('detection', 2),
])
def test_n_bins_for(which, expected):
    assert n_bins_for(which) == expected


def test_n_bins_for_unknown_raises():
    with pytest.raises(ValueError):
        n_bins_for('foo')
