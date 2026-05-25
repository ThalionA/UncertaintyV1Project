# -*- coding: utf-8 -*-
"""Synthetic-data tests for ``nn_decoder/neuron_scaling.py``.

The population-scaling analysis trains the decoder on subsets of the
recorded neural population. These tests pin the correctness of the
answer-independent core on synthetic data with known ground truth:

  - build_neuron_grid produces the agreed step grid, capped per mouse.
  - select_neuron_subset (utils) subsets the neuron axis and validates.
  - The three neuron rankings recover a planted ordering:
      * orientation tuning   -> the strongly tuned neuron ranks first;
      * mean activity        -> the highest-rate neuron ranks first;
      * weight magnitude     -> the heaviest-weighted neuron ranks first.
  - subsets_from_ranking takes the top / bottom N of a ranking.
  - iter_random_subsets is reproducible and well-formed.
  - mean_fit_loss averages the per-trial held-out loss.
  - run_scaling_for_mouse orchestrates the sweep — verified with a stub
    decoder so no torch training is needed.
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

import neuron_scaling as ns
from utils import select_neuron_subset


# ----------------------------------------------------------------------
# build_neuron_grid
# ----------------------------------------------------------------------

def test_build_neuron_grid_appends_full_population():
    # 95 is not a multiple of 10 -> the full count is appended.
    assert ns.build_neuron_grid(95, step=10) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def test_build_neuron_grid_no_duplicate_when_full_is_multiple():
    # 40 is already the last step -> not appended twice.
    assert ns.build_neuron_grid(40, step=10) == [10, 20, 30, 40]
    assert ns.build_neuron_grid(100, step=10) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def test_build_neuron_grid_small_population():
    # Fewer neurons than one step -> just the full population.
    assert ns.build_neuron_grid(8, step=10) == [8]
    assert ns.build_neuron_grid(1, step=10) == [1]


def test_build_neuron_grid_rejects_empty_population():
    with pytest.raises(ValueError):
        ns.build_neuron_grid(0, step=10)


# ----------------------------------------------------------------------
# select_neuron_subset  (lives in utils, exercised by the decoder hook)
# ----------------------------------------------------------------------

def test_select_neuron_subset_keeps_requested_rows():
    act = np.arange(5 * 4 * 3).reshape(5, 4, 3).astype(float)
    out = select_neuron_subset(act, np.array([4, 2, 0]))
    assert out.shape == (3, 4, 3)
    np.testing.assert_array_equal(out, act[[4, 2, 0]])


def test_select_neuron_subset_none_is_noop():
    act = np.zeros((6, 3, 2))
    assert select_neuron_subset(act, None) is act


def test_select_neuron_subset_validation():
    act = np.zeros((5, 3, 2))
    with pytest.raises(ValueError):                       # out of range
        select_neuron_subset(act, np.array([0, 5]))
    with pytest.raises(ValueError):                       # duplicates
        select_neuron_subset(act, np.array([1, 1]))
    with pytest.raises(ValueError):                       # empty
        select_neuron_subset(act, np.array([], dtype=int))
    with pytest.raises(ValueError):                       # non-integer
        select_neuron_subset(act, np.array([0.0, 1.0]))


# ----------------------------------------------------------------------
# Neuron rankings — planted ground truth
# ----------------------------------------------------------------------

def _tuning_activity():
    """3 neurons, 8 trials, 3 time bins. Per-orientation mean response:
        neuron 0: strongly tuned   [10, 7, 4, 1]
        neuron 1: flat             [ 5, 5, 5, 5]
        neuron 2: weakly tuned     [ 5, 5.5, 6, 6.5]
    Expected ranking by tuning strength: [0, 2, 1].
    """
    oris = np.array([0, 30, 60, 90, 0, 30, 60, 90])
    per_ori = {
        0: {0: 10.0, 30: 7.0, 60: 4.0, 90: 1.0},
        1: {0: 5.0, 30: 5.0, 60: 5.0, 90: 5.0},
        2: {0: 5.0, 30: 5.5, 60: 6.0, 90: 6.5},
    }
    act = np.zeros((3, 8, 3))
    for n in range(3):
        for t, o in enumerate(oris):
            act[n, t, :] = per_ori[n][o]
    trials = {
        'orientation': oris,
        'contrast': np.ones(8),
        'dispersion': np.zeros(8),
        'choice': np.array([0, 1, 0, 1, 0, 1, 0, 1]),
    }
    return act, trials


def test_rank_by_orientation_tuning_recovers_planted_order():
    act, trials = _tuning_activity()
    ranking, scores = ns.rank_by_orientation_tuning(act, trials)
    assert list(ranking) == [0, 2, 1]
    assert scores[0] > scores[2] > scores[1]
    assert scores[1] == pytest.approx(0.0)               # flat neuron, no tuning


def test_rank_by_mean_activity_recovers_planted_order():
    act = np.zeros((3, 6, 2))
    act[0] = 2.0
    act[1] = 9.0
    act[2] = 5.0
    ranking, scores = ns.rank_by_mean_activity(act)
    assert list(ranking) == [1, 2, 0]
    assert scores[1] == pytest.approx(9.0)


def test_rank_by_weight_magnitude_recovers_planted_order():
    # W_in is (hidden, n_neurons); per-neuron score is the column norm.
    w_spat = np.zeros((3, 4))
    w_temp = np.zeros((3, 4))
    for w in (w_spat, w_temp):
        w[:, 0] = 0.1
        w[:, 1] = 10.0
        w[:, 2] = 2.0
        w[:, 3] = 5.0
    ranking, scores = ns.rank_by_weight_magnitude(w_spat, w_temp)
    assert list(ranking) == [1, 3, 2, 0]


def test_rank_by_weight_magnitude_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ns.rank_by_weight_magnitude(np.zeros((3, 4)), np.zeros((3, 5)))


# ----------------------------------------------------------------------
# Subset construction
# ----------------------------------------------------------------------

def test_subsets_from_ranking_top_and_bottom():
    ranking = np.array([3, 1, 4, 0, 2])
    top = ns.subsets_from_ranking(ranking, [2, 4], direction='top')
    np.testing.assert_array_equal(top[2], [3, 1])
    np.testing.assert_array_equal(top[4], [3, 1, 4, 0])

    bottom = ns.subsets_from_ranking(ranking, [2, 4], direction='bottom')
    np.testing.assert_array_equal(bottom[2], [0, 2])
    np.testing.assert_array_equal(bottom[4], [1, 4, 0, 2])


def test_subsets_from_ranking_rejects_oversize():
    with pytest.raises(ValueError):
        ns.subsets_from_ranking(np.arange(5), [6], direction='top')


def test_iter_random_subsets_is_reproducible_and_well_formed():
    sizes = [5, 10]
    a = list(ns.iter_random_subsets(20, sizes, n_draws=3, seed=0))
    b = list(ns.iter_random_subsets(20, sizes, n_draws=3, seed=0))
    assert len(a) == len(sizes) * 3
    for (sa, da, ia), (sb, db, ib) in zip(a, b):
        assert sa == sb and da == db
        np.testing.assert_array_equal(ia, ib)            # same seed -> identical
    for size, draw, idx in a:
        assert idx.size == size
        assert np.unique(idx).size == size               # no duplicate neurons
        assert idx.min() >= 0 and idx.max() < 20

    c = list(ns.iter_random_subsets(20, sizes, n_draws=3, seed=1))
    assert any(not np.array_equal(x[2], y[2]) for x, y in zip(a, c))


# ----------------------------------------------------------------------
# Performance extraction
# ----------------------------------------------------------------------

def test_mean_fit_loss_averages_per_trial_loss():
    result = {'fit_loss': {'spat': np.array([1.0, 2.0, 3.0])}}
    assert ns.mean_fit_loss(result, 'spat') == pytest.approx(2.0)


def test_mean_fit_loss_empty_is_nan():
    result = {'fit_loss': {'spat': np.array([])}}
    assert np.isnan(ns.mean_fit_loss(result, 'spat'))


# ----------------------------------------------------------------------
# Runner orchestration — stub decoder, no torch training
# ----------------------------------------------------------------------

def _synthetic_mouse(n_neurons=12, n_trials=24, t_bins=4):
    rng = np.random.default_rng(42)
    act = rng.random((n_neurons, n_trials, t_bins))
    trials = {
        'orientation': np.tile([0, 30, 60, 90], n_trials // 4).astype(float),
        'contrast': np.tile([0.5, 1.0], n_trials // 2),
        'dispersion': np.tile([0.0, 10.0], n_trials // 2),
        'choice': np.tile([0, 1], n_trials // 2),
    }
    return act, trials


def test_run_scaling_for_mouse_orchestration():
    act, trials = _synthetic_mouse(n_neurons=12)
    calls = []

    def stub_decoder(config, mouse_id, neuron_subset=None, preloaded=None):
        n = 12 if neuron_subset is None else len(neuron_subset)
        calls.append(n)
        rng = np.random.default_rng(len(calls))
        return {
            'fit_loss': {k: rng.random(5)
                         for k in ('spat', 'temp', 'spat_shf', 'temp_shf')},
            'Weights': {'spat': {'W_in': rng.random((4, n))},
                        'temp': {'W_in': rng.random((4, n))}},
        }

    df = ns.run_scaling_for_mouse(
        0, {'which_model': 'perception', 'split_type': 'stratified_balanced'},
        act, trials, target='Q', split='stratified_balanced',
        decoder_fn=stub_decoder, step=10, n_draws=2, seed=0, progress=False,
    )

    # grid = [10, 12]; sweep = [10]. calls = 1 full + 2 random + 3*2 targeted.
    assert len(calls) == 9
    # 9 decoder calls x 4 model keys per call.
    assert len(df) == 9 * 4
    assert set(df['selection']) == {
        'full', 'random', 'orientation_tuning', 'mean_activity', 'weight_magnitude'}
    assert set(df['model']) == {'spat', 'temp', 'spat_shf', 'temp_shf'}

    # The full-population point is recorded once, at n_neurons = 12.
    full = df[df['selection'] == 'full']
    assert set(full['n_neurons']) == {12}
    assert set(full['direction']) == {'na'}

    # Targeted curves carry both directions; random carries none.
    targeted = df[df['selection'].isin(
        ['orientation_tuning', 'mean_activity', 'weight_magnitude'])]
    assert set(targeted['direction']) == {'top', 'bottom'}
    assert set(targeted['n_neurons']) == {10}
    assert set(df[df['selection'] == 'random']['n_neurons']) == {10}
    assert set(df[df['selection'] == 'random']['draw']) == {0, 1}

    assert df['fit_loss'].notna().all()
    # Stub returns 5-trial fit_loss vectors -> n_test_trials recorded as 5.
    assert set(df['n_test_trials']) == {5}


# ----------------------------------------------------------------------
# n_test_trials in the row schema
# ----------------------------------------------------------------------

def test_result_rows_records_n_test_trials():
    result = {'fit_loss': {k: np.arange(7.0) for k in ns.MODEL_KEYS}}
    rows = ns.result_rows(result, {'mouse': 0, 'selection': 'full'})
    assert len(rows) == 4
    for row in rows:
        assert row['n_test_trials'] == 7
        assert row['fit_loss'] == pytest.approx(np.arange(7.0).mean())


# ----------------------------------------------------------------------
# Cross-mouse aggregation
# ----------------------------------------------------------------------

def _agg_row(mouse, selection, direction, n_neurons, model, fit_loss,
             n_test_trials=None, draw=0):
    row = dict(mouse=mouse, target='Q', split='stratified_balanced',
               selection=selection, direction=direction, n_neurons=n_neurons,
               draw=draw, model=model, fit_loss=fit_loss)
    if n_test_trials is not None:
        row['n_test_trials'] = n_test_trials
    return row


def test_aggregate_across_mice_weights_by_trial_count():
    # mouse 0: 100 test trials, loss 0.2; mouse 3: 300 trials, loss 0.6.
    # trial-count-weighted mean = (100*0.2 + 300*0.6) / 400 = 0.5
    df = pd.DataFrame([
        _agg_row(0, 'random', 'na', 30, 'spat', 0.2, n_test_trials=100),
        _agg_row(3, 'random', 'na', 30, 'spat', 0.6, n_test_trials=300),
    ])
    agg = ns.aggregate_across_mice(df)
    assert len(agg) == 1
    assert agg['fit_loss'].iloc[0] == pytest.approx(0.5)
    assert agg['weight_basis'].iloc[0] == 'trial_count'
    assert agg['n_mice'].iloc[0] == 2


def test_aggregate_across_mice_collapses_draws_within_mouse():
    # One mouse, two random draws -> averaged before the cross-mouse step,
    # so the draw count cannot skew the trial-count weighting.
    df = pd.DataFrame([
        _agg_row(0, 'random', 'na', 30, 'spat', 0.2, n_test_trials=100, draw=0),
        _agg_row(0, 'random', 'na', 30, 'spat', 0.4, n_test_trials=100, draw=1),
    ])
    agg = ns.aggregate_across_mice(df)
    assert len(agg) == 1
    assert agg['fit_loss'].iloc[0] == pytest.approx(0.3)


def test_aggregate_across_mice_equal_weight_fallback():
    # No n_test_trials column -> equal weight, with a warning.
    df = pd.DataFrame([
        _agg_row(0, 'random', 'na', 30, 'spat', 0.2),
        _agg_row(3, 'random', 'na', 30, 'spat', 0.6),
    ])
    with pytest.warns(UserWarning, match='n_test_trials'):
        agg = ns.aggregate_across_mice(df)
    assert agg['fit_loss'].iloc[0] == pytest.approx(0.4)
    assert agg['weight_basis'].iloc[0] == 'equal'


def test_aggregate_across_mice_records_partial_coverage():
    # N=30 reached by both mice; N=120 only by the larger-population mouse.
    df = pd.DataFrame([
        _agg_row(0, 'random', 'na', 30, 'spat', 0.5, n_test_trials=100),
        _agg_row(3, 'random', 'na', 30, 'spat', 0.5, n_test_trials=100),
        _agg_row(3, 'random', 'na', 120, 'spat', 0.3, n_test_trials=100),
    ])
    agg = ns.aggregate_across_mice(df)
    assert int(agg[agg['n_neurons'] == 30]['n_mice'].iloc[0]) == 2
    assert int(agg[agg['n_neurons'] == 120]['n_mice'].iloc[0]) == 1


# ----------------------------------------------------------------------
# Normalisation to the full-population loss
# ----------------------------------------------------------------------

def test_normalise_to_full_anchors_each_mouse_at_one():
    # mouse 0: full loss 0.40, a smaller-N loss 0.60 -> ratio 1.5
    # mouse 3: full loss 0.50, a smaller-N loss 1.00 -> ratio 2.0
    df = pd.DataFrame([
        _agg_row(0, 'full', 'na', 60, 'spat', 0.40, n_test_trials=100, draw=-1),
        _agg_row(0, 'random', 'na', 30, 'spat', 0.60, n_test_trials=100),
        _agg_row(3, 'full', 'na', 150, 'spat', 0.50, n_test_trials=200, draw=-1),
        _agg_row(3, 'random', 'na', 30, 'spat', 1.00, n_test_trials=200),
    ])
    nd = ns.normalise_to_full(df)
    # full-population rows normalise to exactly 1.0
    assert list(nd[nd['selection'] == 'full']['fit_loss_rel_full']) == [1.0, 1.0]
    m0 = nd[(nd['mouse'] == 0) & (nd['selection'] == 'random')]
    m3 = nd[(nd['mouse'] == 3) & (nd['selection'] == 'random')]
    assert m0['fit_loss_rel_full'].iloc[0] == pytest.approx(1.5)
    assert m3['fit_loss_rel_full'].iloc[0] == pytest.approx(2.0)


def test_aggregate_across_mice_pools_a_chosen_value_column():
    df = pd.DataFrame([
        _agg_row(0, 'random', 'na', 30, 'spat', 0.0, n_test_trials=100),
        _agg_row(3, 'random', 'na', 30, 'spat', 0.0, n_test_trials=100),
    ])
    df['fit_loss_rel_full'] = [1.5, 2.5]
    agg = ns.aggregate_across_mice(df, value_col='fit_loss_rel_full')
    assert 'fit_loss_rel_full' in agg.columns
    assert agg['fit_loss_rel_full'].iloc[0] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# build_scaling_config — Optuna preset with an epoch floor
# ----------------------------------------------------------------------

def test_build_scaling_config_applies_epoch_floor():
    """min_epochs raises num_epochs only when the preset sits below it.
    For the production run (min_epochs=100): Q's preset (30 epochs) is
    floored to 100, while d's (200) already exceeds the floor and is
    left untouched."""
    q = ns.build_scaling_config('Q', min_epochs=100)
    d = ns.build_scaling_config('d', min_epochs=100)
    assert q.num_epochs == 100      # preset 30 -> floored up
    assert d.num_epochs == 200      # preset 200 -> already above the floor


def test_build_scaling_config_none_keeps_preset_epochs():
    """min_epochs=None leaves num_epochs at the Optuna preset value for
    every target."""
    from training.config import default_config_for_target
    for target in ('Q', 'L', 'd', 'choice', 'stim_kernel', 'stim_cat'):
        preset = default_config_for_target(target, bin_size_ms=100)
        cfg = ns.build_scaling_config(target, min_epochs=None)
        assert cfg.num_epochs == preset.num_epochs


def test_build_scaling_config_floor_touches_only_num_epochs():
    """The epoch floor must not perturb any other hyperparameter — every
    field except num_epochs stays at the Optuna preset."""
    from dataclasses import asdict
    from training.config import default_config_for_target
    preset = default_config_for_target('Q', bin_size_ms=100,
                                        split_type='stratified_balanced',
                                        run_name='neuron_scaling_Q')
    floored = ns.build_scaling_config('Q', min_epochs=9999)
    assert floored.num_epochs == 9999
    pa, pf = asdict(preset), asdict(floored)
    pa.pop('num_epochs'); pf.pop('num_epochs')
    assert pa == pf                # every other field identical
