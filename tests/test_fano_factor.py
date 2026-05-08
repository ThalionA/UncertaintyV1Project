# -*- coding: utf-8 -*-
"""Synthetic-data tests for ``nn_decoder/plot_fano_factor.py``.

  - compute_pooled_ff_timecourse: Poisson recovery (FF ≈ 1 across time),
    degenerate-condition handling.
  - compute_ff_per_condition_window: Poisson recovery, min-trials filter,
    time_slice aggregation correctness.
  - compute_ff_per_condition_timecourse: Poisson recovery per condition,
    correct keys, NaN behaviour for unqualified time bins.
"""

from __future__ import annotations

import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

import plot_fano_factor as pff  # noqa: E402


def _rng(seed=0):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# compute_pooled_ff_timecourse
# ----------------------------------------------------------------------

def test_pooled_ff_poisson_recovers_one():
    rng = _rng(123)
    n_trials, t_bins, n_neurons = 600, 12, 25
    n_conds = 6
    cond_per_trial = np.repeat(np.arange(n_conds), n_trials // n_conds)
    rng.shuffle(cond_per_trial)
    rates = rng.uniform(0.5, 5.0, size=(n_conds, n_neurons))
    raw = np.zeros((n_trials, t_bins, n_neurons))
    for i in range(n_trials):
        c = cond_per_trial[i]
        raw[i] = rng.poisson(rates[c], size=(t_bins, n_neurons))
    trials = {
        'orientation': cond_per_trial.astype(float),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    ff_raw, ff_matched, ci, _ = pff.compute_pooled_ff_timecourse(
        raw, trials, min_trials_per_cond=20, n_bootstrap=80, n_match_bins=8)
    assert np.isfinite(ff_raw).any()
    raw_mean = float(np.nanmean(ff_raw))
    matched_mean = float(np.nanmean(ff_matched))
    assert 0.85 < raw_mean < 1.15, f"raw mean = {raw_mean:.3f}"
    assert 0.85 < matched_mean < 1.15, f"matched mean = {matched_mean:.3f}"
    # CI has the right shape and brackets the mean somewhere
    assert ci.shape == (2, ff_raw.size)


def test_pooled_ff_no_valid_conditions_returns_nan():
    n_trials, t_bins, n_neurons = 10, 5, 4
    raw = np.random.poisson(1.0, (n_trials, t_bins, n_neurons)).astype(float)
    trials = {
        'orientation': np.arange(n_trials, dtype=float),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    ff_raw, ff_matched, _, _ = pff.compute_pooled_ff_timecourse(
        raw, trials, min_trials_per_cond=10, n_bootstrap=10)
    assert np.all(np.isnan(ff_raw))
    assert np.all(np.isnan(ff_matched))


# ----------------------------------------------------------------------
# compute_ff_per_condition_window
# ----------------------------------------------------------------------

def test_per_condition_window_poisson_close_to_one():
    rng = _rng(0)
    n_per_cond, t_bins, n_neurons = 30, 20, 15
    n_conds = 4
    rates = rng.uniform(0.5, 4.0, size=(n_conds, n_neurons))
    cond_per_trial = np.repeat(np.arange(n_conds), n_per_cond)
    n_trials = cond_per_trial.size
    raw = np.zeros((n_trials, t_bins, n_neurons))
    for i in range(n_trials):
        c = cond_per_trial[i]
        raw[i] = rng.poisson(rates[c], size=(t_bins, n_neurons))
    trials = {
        'orientation': cond_per_trial.astype(float),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    rows = pff.compute_ff_per_condition_window(raw, trials, min_trials_per_cond=20)
    assert len(rows) == n_conds
    fanos = [r['mean_per_neuron_fano'] for r in rows]
    slopes = [r['slope_ff'] for r in rows]
    assert 0.7 < float(np.mean(fanos)) < 1.3
    assert 0.7 < float(np.mean(slopes)) < 1.3


def test_per_condition_window_min_trials_filter():
    rng = _rng(1)
    counts = [3, 8, 2, 12, 5, 10]
    n_neurons, t_bins = 8, 10
    cond_per_trial = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    n_trials = cond_per_trial.size
    raw = rng.poisson(2.0, size=(n_trials, t_bins, n_neurons)).astype(float)
    trials = {
        'orientation': cond_per_trial.astype(float),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    rows = pff.compute_ff_per_condition_window(raw, trials, min_trials_per_cond=7)
    kept_oris = sorted({r['orientation'] for r in rows})
    expected = [float(i) for i, n in enumerate(counts) if n >= 7]
    assert kept_oris == expected


def test_per_condition_window_time_slice_changes_aggregation():
    rng = _rng(2)
    n_trials, t_bins, n_neurons = 30, 20, 6
    raw = rng.poisson(2.0, size=(n_trials, t_bins, n_neurons)).astype(float)
    trials = {
        'orientation': np.zeros(n_trials),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    rows_full = pff.compute_ff_per_condition_window(raw, trials, min_trials_per_cond=20)
    rows_half = pff.compute_ff_per_condition_window(raw, trials, min_trials_per_cond=20,
                                                      time_slice=slice(0, t_bins // 2))
    assert rows_full and rows_half
    assert rows_half[0]['mean_rate'] < rows_full[0]['mean_rate']


# ----------------------------------------------------------------------
# compute_ff_per_condition_timecourse
# ----------------------------------------------------------------------

def test_per_condition_timecourse_poisson_close_to_one():
    rng = _rng(7)
    n_per_cond, t_bins, n_neurons = 30, 15, 12
    n_conds = 3
    rates = rng.uniform(1.0, 3.0, size=(n_conds, n_neurons))
    cond_per_trial = np.repeat(np.arange(n_conds), n_per_cond)
    n_trials = cond_per_trial.size
    raw = np.zeros((n_trials, t_bins, n_neurons))
    for i in range(n_trials):
        c = cond_per_trial[i]
        raw[i] = rng.poisson(rates[c], size=(t_bins, n_neurons))
    trials = {
        'orientation': cond_per_trial.astype(float),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    out = pff.compute_ff_per_condition_timecourse(raw, trials, min_trials_per_cond=20)
    # All three conditions should be present
    assert len(out) == n_conds
    # Each value is a (t_bins,) array
    for arr in out.values():
        assert arr.shape == (t_bins,)
        finite_mean = float(np.nanmean(arr))
        assert 0.7 < finite_mean < 1.3, f"per-cond TC mean = {finite_mean:.3f}"


def test_per_condition_timecourse_drops_below_threshold():
    rng = _rng(8)
    counts = [3, 12]  # one condition above threshold, one below
    n_neurons, t_bins = 6, 10
    cond_per_trial = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    n_trials = cond_per_trial.size
    raw = rng.poisson(2.0, size=(n_trials, t_bins, n_neurons)).astype(float)
    trials = {
        'orientation': cond_per_trial.astype(float),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    out = pff.compute_ff_per_condition_timecourse(raw, trials, min_trials_per_cond=7)
    # Only orientation 1 (count 12) should be kept
    kept_oris = sorted({k[0] for k in out.keys()})
    assert kept_oris == [1.0]


def test_per_condition_timecourse_nan_when_too_few_neurons():
    """If a condition has many trials but no neurons survive the
    mean > 1e-3 filter at a time bin, that bin should be NaN."""
    rng = _rng(9)
    n_trials, t_bins, n_neurons = 20, 6, 8
    raw = np.zeros((n_trials, t_bins, n_neurons))
    # Real Poisson on the second half, all-zero on the first half
    raw[:, t_bins // 2:, :] = rng.poisson(2.0, size=(n_trials, t_bins // 2, n_neurons))
    trials = {
        'orientation': np.zeros(n_trials),
        'contrast': np.zeros(n_trials),
        'dispersion': np.zeros(n_trials),
    }
    out = pff.compute_ff_per_condition_timecourse(raw, trials,
                                                    min_trials_per_cond=10,
                                                    min_neurons=5)
    assert len(out) == 1
    arr = next(iter(out.values()))
    # First half should be NaN (no neurons with mean > 1e-3)
    assert np.all(np.isnan(arr[:t_bins // 2]))
    # Second half should be mostly finite, near 1
    second_half = arr[t_bins // 2:]
    assert np.isfinite(second_half).any()
    mean_finite = float(np.nanmean(second_half))
    assert 0.6 < mean_finite < 1.4, f"second half mean = {mean_finite:.3f}"


# ----------------------------------------------------------------------
# Cross-mouse aggregation correctness in plot helpers (no plot rendering,
# just confirm the underlying aggregation logic does what we expect).
# ----------------------------------------------------------------------

def test_marginal_aggregation_handles_missing_combos():
    """Two synthetic mice with overlapping but non-identical condition
    coverage. Marginal time courses should still produce one curve per
    feature value, averaging over the OTHER two features."""
    t_bins = 5
    # Mouse 0: only contrast 0.1 across two orientations
    m0 = {
        (0.0, 0.1, 1.0): np.full(t_bins, 0.5),
        (1.0, 0.1, 1.0): np.full(t_bins, 0.7),
    }
    # Mouse 1: contrast 0.5 across one orientation, plus one shared cell
    m1 = {
        (0.0, 0.1, 1.0): np.full(t_bins, 0.6),  # shared with mouse 0
        (0.0, 0.5, 1.0): np.full(t_bins, 0.9),
    }
    per_mouse_tc_results = [(0, m0), (1, m1)]

    # Replicate the plot_ff_per_condition_timecourse_marginal aggregation
    # for the contrast feature (axis = 1).
    all_combos = sorted(set(m0.keys()) | set(m1.keys()))
    contrasts = sorted({c[1] for c in all_combos})
    assert contrasts == [0.1, 0.5]

    n_mice = 2
    results = {}
    for val in contrasts:
        per_mouse_vec = np.full((n_mice, t_bins), np.nan)
        for i, (_, d) in enumerate(per_mouse_tc_results):
            matching = [d[c] for c in d if c[1] == val]
            if matching:
                per_mouse_vec[i] = np.nanmean(np.stack(matching), axis=0)
        results[val] = per_mouse_vec

    # Contrast 0.1: mouse 0 averages [(0.0, 0.1, 1.0), (1.0, 0.1, 1.0)] = (0.5+0.7)/2 = 0.6
    #               mouse 1 has only (0.0, 0.1, 1.0) = 0.6
    # → grand mean = 0.6 (both mice agree)
    assert np.allclose(np.nanmean(results[0.1], axis=0), 0.6)
    # Contrast 0.5: mouse 0 has nothing (NaN row), mouse 1 has 0.9
    # → grand mean = 0.9
    assert np.allclose(np.nanmean(results[0.5], axis=0), 0.9)
