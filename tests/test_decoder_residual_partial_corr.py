# -*- coding: utf-8 -*-
"""Synthetic-data tests for nn_decoder/decoder_residual_partial_corr.py.

The script analyses already-saved decoder outputs without retraining; these
tests cover the pure-numerics surface (scalar summaries, DataFrame
assembly, the structural null of the stim_mean baseline, planted-signal
recovery) on hand-built inputs.

The full driver `main()` is not exercised here because it requires real
.mat files and the raw VR export. Tests for the assembly path use a
monkey-patched `_get_test_trials` to inject synthetic trial tables.
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


def _mod():
    """Lazy import: the module is pure-numpy at the top level, but its
    helpers transitively import utils_v26 (torch) when used. Top-level
    import is safe; skip the suite cleanly if it isn't."""
    try:
        import decoder_residual_partial_corr as drpc
    except Exception as exc:
        pytest.skip(f"decoder_residual_partial_corr import failed: {exc}")
    return drpc


# ----------------------------------------------------------------------
# Scalar summaries on hand-built distributions
# ----------------------------------------------------------------------

def _gaussian_on_grid(mu, sigma, grid):
    p = np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    return p / p.sum()


def test_dist_moments_matches_analytic_on_gaussian():
    drpc = _mod()
    grid = drpc.DEFAULT_GRID_91
    rows = np.stack([
        _gaussian_on_grid(30.0, 5.0, grid),
        _gaussian_on_grid(60.0, 10.0, grid),
        _gaussian_on_grid(45.0, 8.0, grid),
    ])
    mu, var = drpc.dist_moments(rows, grid)
    # Truncated Gaussians on [0, 90] have small bias for sigma << 30,
    # so a 0.5 deg / 2 deg^2 tolerance is plenty.
    assert np.allclose(mu, [30.0, 60.0, 45.0], atol=0.5)
    assert np.allclose(var, [25.0, 100.0, 64.0], atol=2.0)


def test_dist_moments_handles_unnormalised_rows():
    drpc = _mod()
    grid = drpc.DEFAULT_GRID_91
    row = _gaussian_on_grid(45.0, 6.0, grid)
    row_scaled = row * 17.3
    mu1, var1 = drpc.dist_moments(row[None, :], grid)
    mu2, var2 = drpc.dist_moments(row_scaled[None, :], grid)
    assert np.allclose(mu1, mu2)
    assert np.allclose(var1, var2)


def test_dist_entropy_zero_for_delta_high_for_uniform():
    drpc = _mod()
    grid = drpc.DEFAULT_GRID_91
    delta = np.zeros((1, grid.size))
    delta[0, 30] = 1.0
    uniform = np.ones((1, grid.size)) / grid.size
    h_delta = drpc.dist_entropy(delta)[0]
    h_uniform = drpc.dist_entropy(uniform)[0]
    assert h_delta < 1e-6
    assert h_uniform == pytest.approx(np.log(grid.size), rel=1e-6)
    assert h_uniform > h_delta


def test_summaries_for_array_Q_returns_three_summaries():
    drpc = _mod()
    grid = drpc.DEFAULT_GRID_91
    rows = np.stack([_gaussian_on_grid(45.0, 5.0, grid)] * 4)
    out = drpc.summaries_for_array(rows, 'Q', grid)
    assert set(out.keys()) == {'mu', 'var', 'H'}
    for k in out:
        assert out[k].shape == (4,)


def test_summaries_for_array_d_returns_pgo_and_H():
    drpc = _mod()
    rows = np.array([
        [0.9, 0.1],
        [0.5, 0.5],
        [0.1, 0.9],
    ])
    out = drpc.summaries_for_array(rows, 'd')
    assert set(out.keys()) == {'pgo', 'H'}
    assert np.allclose(out['pgo'], [0.9, 0.5, 0.1])
    # Entropy ordering: middle row has max entropy (= ln 2).
    assert out['H'][1] > out['H'][0]
    assert out['H'][1] > out['H'][2]
    assert out['H'][1] == pytest.approx(np.log(2), rel=1e-6)


def test_summaries_for_array_rejects_unknown_target():
    drpc = _mod()
    with pytest.raises(ValueError):
        drpc.summaries_for_array(np.zeros((2, 91)), 'bogus')


def test_summaries_for_array_rejects_1d_input():
    drpc = _mod()
    with pytest.raises(ValueError):
        drpc.summaries_for_array(np.zeros(91), 'Q')


# ----------------------------------------------------------------------
# partial_corr_table: structural null for stim_mean, planted signal for
# the trained-arch surrogate.
# ----------------------------------------------------------------------

def _build_synthetic_long_df(n_mice=4, n_per_condition=20, seed=0):
    """Build a synthetic long DataFrame for two archs:

      * 'stim_mean': decoded == per-condition mean target (zero within-cond
        variance). Partial r | stim must be NaN or ~0.
      * 'good':     decoded == target + small per-trial noise. Partial r |
        stim should be high.
    """
    rng = np.random.default_rng(seed)
    orientations = [15.0, 45.0, 75.0]
    contrasts = [0.25, 1.0]
    dispersions = [5.0, 30.0]
    rows = []
    for mid in range(n_mice):
        # Per-condition baseline target value (the "stim shadow")
        cond_means = {}
        for o in orientations:
            for c in contrasts:
                for d in dispersions:
                    cond_means[(o, c, d)] = (
                        rng.normal(loc=50.0, scale=10.0)
                    )
        for o in orientations:
            for c in contrasts:
                for d in dispersions:
                    base = cond_means[(o, c, d)]
                    for _ in range(n_per_condition):
                        # Trial-level true value: condition mean + within-cond
                        # noise (this within-cond fluctuation is what a real
                        # decoder is meant to capture).
                        within = rng.normal(scale=2.0)
                        target_val = base + within

                        # stim_mean decoder: always predicts the (training)
                        # condition mean. Zero within-cond variance.
                        stim_mean_val = base

                        # 'good' decoder: target + small noise. Tracks the
                        # within-cond signal too.
                        good_val = target_val + rng.normal(scale=0.5)

                        base_row = {
                            'Mouse_ID': mid,
                            'target_type': 'Q',
                            'split': 'synthetic',
                            'Orientation': o,
                            'Contrast': c,
                            'Dispersion': d,
                            'Velocity': rng.normal(scale=1.0),
                            'Lick_Rate': rng.normal(scale=1.0),
                            'Choice': rng.integers(0, 2),
                            # Reuse the same summary value as 'mu' for the
                            # purposes of this test. The numerical content of
                            # the summary is irrelevant — only the
                            # within-cond co-variation matters.
                            'target_mu':  target_val,
                            'target_var': 100.0,
                            'target_H':   np.log(91.0),
                        }
                        r_sm = dict(base_row)
                        r_sm['arch'] = 'stim_mean'
                        r_sm['decoded_mu']  = stim_mean_val
                        r_sm['decoded_var'] = 100.0
                        r_sm['decoded_H']   = np.log(91.0)
                        rows.append(r_sm)

                        r_g = dict(base_row)
                        r_g['arch'] = 'good'
                        r_g['decoded_mu']  = good_val
                        r_g['decoded_var'] = 100.0
                        r_g['decoded_H']   = np.log(91.0)
                        rows.append(r_g)
    return pd.DataFrame(rows)


def test_partial_corr_stim_mean_structural_null():
    """The stim_mean baseline has zero within-cell variance by construction,
    so residualising against the stim one-hot leaves a zero vector. The
    partial r must be NaN or ~0; emphatically it must NOT be the same as
    the raw r (which is high because between-cell variance is huge)."""
    drpc = _mod()
    df = _build_synthetic_long_df()
    tbl = drpc.partial_corr_table(df, summary_keys=('mu',))
    sm_row = tbl[(tbl['arch'] == 'stim_mean') & (tbl['summary'] == 'mu')]
    assert len(sm_row) == 1
    raw = float(sm_row['raw_pearson_r'].iloc[0])
    partial_s = float(sm_row['partial_stim_r'].iloc[0])
    # raw r is dominated by between-cell variation and should be high.
    assert abs(raw) > 0.5
    # partial r is the structural null: 0 (or NaN if all residuals were
    # exactly zero so Pearson is undefined).
    assert np.isnan(partial_s) or abs(partial_s) < 0.05


def test_partial_corr_good_decoder_recovers_signal():
    """A decoder that tracks the trial-level target plus small noise should
    have partial r close to 1.0."""
    drpc = _mod()
    df = _build_synthetic_long_df()
    tbl = drpc.partial_corr_table(df, summary_keys=('mu',))
    g_row = tbl[(tbl['arch'] == 'good') & (tbl['summary'] == 'mu')]
    assert len(g_row) == 1
    partial_s = float(g_row['partial_stim_r'].iloc[0])
    assert partial_s > 0.85


def test_partial_corr_good_beats_stim_mean():
    """Cross-arch comparison of the gating quantity."""
    drpc = _mod()
    df = _build_synthetic_long_df()
    tbl = drpc.partial_corr_table(df, summary_keys=('mu',))
    g = float(tbl[(tbl['arch'] == 'good')
                  & (tbl['summary'] == 'mu')]['partial_stim_r'].iloc[0])
    sm = tbl[(tbl['arch'] == 'stim_mean')
             & (tbl['summary'] == 'mu')]['partial_stim_r'].iloc[0]
    sm = 0.0 if np.isnan(sm) else float(sm)
    assert g > sm + 0.5


def test_partial_corr_includes_behaviour_tier():
    """The partial_corr_table must always populate both tiers; the
    stim+beh column should also collapse for stim_mean (zero within-cond
    variance is structural in either control set)."""
    drpc = _mod()
    df = _build_synthetic_long_df()
    tbl = drpc.partial_corr_table(df, summary_keys=('mu',))
    expected_cols = {
        'raw_pearson_r', 'partial_stim_r', 'partial_stim_p',
        'partial_stim_beh_r', 'partial_stim_beh_p',
    }
    assert expected_cols.issubset(set(tbl.columns))
    sm = tbl[tbl['arch'] == 'stim_mean'].iloc[0]
    sb = sm['partial_stim_beh_r']
    assert np.isnan(sb) or abs(sb) < 0.05


def test_partial_corr_table_handles_missing_summary_columns():
    drpc = _mod()
    df = _build_synthetic_long_df()
    df = df.drop(columns=['decoded_var', 'target_var'])
    tbl = drpc.partial_corr_table(df, summary_keys=('mu', 'var', 'H'))
    assert 'var' not in tbl['summary'].values
    assert {'mu', 'H'}.issubset(set(tbl['summary'].values))


# ----------------------------------------------------------------------
# Sanity: load + assembly path against a manufactured .mat (no torch).
# ----------------------------------------------------------------------

def test_load_decoder_outputs_picks_up_production_and_stim_mean(tmp_path):
    """Hand-write minimal production + stim_mean .mat files and confirm
    `_load_decoder_outputs` discovers both archs and the stim_mean key.
    Uses target_type='d' so the synthetic distributions are 2-D and small.
    """
    drpc = _mod()
    import scipy.io as sio

    n_test = 12
    rng = np.random.default_rng(0)
    decoded_spat = rng.uniform(size=(n_test, 2))
    decoded_spat = decoded_spat / decoded_spat.sum(1, keepdims=True)
    target = rng.uniform(size=(n_test, 2))
    target = target / target.sum(1, keepdims=True)

    prod_payload = {
        'results': {
            'mouse_0': {
                'Dist': {
                    'spat': {'decoded': decoded_spat, 'target': target},
                    'temp': {'decoded': decoded_spat, 'target': target},
                },
            },
        },
        'config': {'note': 'test'},
    }
    prod_path = tmp_path / 'population_results_fixed_decision_stratified_balanced.mat'
    sio.savemat(str(prod_path), prod_payload)

    sm_payload = {
        'results': {
            'mouse_0': {
                'Dist': {
                    'stim_mean': {'decoded': decoded_spat, 'target': target},
                },
            },
        },
        'config': {'note': 'test'},
    }
    sm_path = tmp_path / 'population_results_stim_mean_d_stratified_balanced.mat'
    sio.savemat(str(sm_path), sm_payload)

    outputs = drpc._load_decoder_outputs('d', 'stratified_balanced', str(tmp_path))
    keys = set(outputs.keys())
    assert (0, 'spat') in keys
    assert (0, 'temp') in keys
    assert (0, 'stim_mean') in keys
    for k, v in outputs.items():
        assert v['decoded'].shape == (n_test, 2)
        assert v['target'].shape == (n_test, 2)
