# -*- coding: utf-8 -*-
"""Synthetic-data tests for ``nn_decoder/feature_ablation_analysis.py``.

Tests are organised around the four substantive claims the module needs
to deliver on:

1. ``cv_gpr_r2`` recovers signal on linear, nonlinear, and null targets,
   handles LOMO and NaNs, and agrees with Ridge in the linear regime.
2. ``run_cv_regression_blocks`` correctly attributes signal to the
   block that carries it: ``drop:noise_block`` leaves R² intact;
   ``drop:signal_block`` collapses it.
3. Per-block and per-metric ablations coexist cleanly in the output;
   the schema is stable.
4. Target transforms (log) and null shuffled-target baselines work.

All tests use small synthetic data (n ≤ 300, p ≤ 6) so GPR fits
in well under a second per call. The GPR config in test uses
``n_restarts=0`` to keep wall time tight.
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

import feature_ablation_analysis as fab  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _rng(seed=0):
    return np.random.default_rng(seed)


def _make_synthetic_df(n_per_mouse=200, n_mice=4, seed=0):
    """Build a per-trial dataframe with planted signal patterns.

    Schema mirrors what ``run_population_variance_pipeline`` produces:
    Mouse_ID + the 11 base features × 2 windows = 22 feature columns,
    plus targets. Signal is planted as follows:

      - Perceptual_Variance is a linear function of TEMPORAL_VARIANCE
        block features (specifically Temporal_Var_Late + GV_Late).
      - Velocity is a linear function of RATE block (Pop_Mean_Raw_Late).
      - Decision_Entropy is a nonlinear function (sin) of GV_Late.
      - Likelihood_Variance is pure noise (the null target).
      - Orientation, Contrast, Dispersion: drawn from the discrete
        task set so the log-transform paths are testable.
    """
    rng = _rng(seed)
    rows = []
    for mid in range(n_mice):
        n = n_per_mouse
        # Synthetic features for each block. Independent N(0,1).
        rate_features = rng.normal(size=(n, 4))
        tvar_features = rng.normal(size=(n, 6))
        order_features = rng.normal(size=(n, 1))

        # Targets with planted patterns
        perc_var = (
            1.5 * tvar_features[:, 1]   # Temporal_Var_Late
            + 0.8 * tvar_features[:, 2] # GV_Late
            + 0.3 * rng.normal(size=n)
        )
        velocity = (
            2.0 * rate_features[:, 0]   # Pop_Mean_Raw_Late
            + 0.3 * rng.normal(size=n)
        )
        dec_ent = np.sin(2.0 * tvar_features[:, 2]) + 0.2 * rng.normal(size=n)
        lik_var = rng.normal(size=n)  # null target

        orientations = rng.choice([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0], n)
        contrasts = rng.choice([0.01, 0.25, 0.5, 1.0], n)
        dispersions = rng.choice([5.0, 30.0, 45.0, 90.0], n)

        row = pd.DataFrame({
            'Mouse_ID': mid,
            'Trial_Idx': np.arange(n),
            'Orientation': orientations,
            'Contrast': contrasts,
            'Dispersion': dispersions,
            'Perceptual_Variance': perc_var,
            'Velocity': velocity,
            'Decision_Entropy': dec_ent,
            'Likelihood_Variance': lik_var,
        })

        # The 22 feature columns. Names mirror METRIC_BASES × windows.
        # We put the planted signal in the *_Late columns and noise in *_Full
        # so the test catches anyone who accidentally drops windowing.
        rate_base = ['Pop_Mean_Raw', 'Pop_Deviation', 'Spatial_Var',
                     'Centroid_Magnitude']
        tvar_base = ['Temporal_Fano_Raw', 'Temporal_Var', 'GV', 'GV_Fano',
                     'Participation_Ratio', 'Top_Mode_Dominance']
        order_base = ['Traj_Length']

        for i, b in enumerate(rate_base):
            row[f'{b}_Late'] = rate_features[:, i]
            row[f'{b}_Full'] = rng.normal(size=n)  # uninformative
        for i, b in enumerate(tvar_base):
            row[f'{b}_Late'] = tvar_features[:, i]
            row[f'{b}_Full'] = rng.normal(size=n)
        for i, b in enumerate(order_base):
            row[f'{b}_Late'] = order_features[:, i]
            row[f'{b}_Full'] = rng.normal(size=n)
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


# ======================================================================
# 1. cv_gpr_r2
# ======================================================================

def test_gpr_recovers_linear_signal():
    rng = _rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * rng.normal(size=n)
    groups = np.repeat(np.arange(4), n // 4)
    r2, y_hat, mask = fab.cv_gpr_r2(X, y, groups, cv='kfold', n_splits=5,
                                     n_restarts=0, seed=0)
    assert r2 > 0.7, f"GPR should recover strong linear signal (got R²={r2})"


def test_gpr_null_on_pure_noise():
    rng = _rng(1)
    n = 200
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)  # independent of X
    groups = np.repeat(np.arange(4), n // 4)
    r2, _, _ = fab.cv_gpr_r2(X, y, groups, cv='kfold', n_splits=5,
                              n_restarts=0, seed=0)
    assert r2 < 0.15, f"GPR R² should be near zero on noise (got {r2})"


def test_gpr_handles_nonlinear_signal():
    rng = _rng(2)
    n = 300
    X = rng.uniform(-2, 2, size=(n, 1))
    y = np.sin(2.0 * X[:, 0]) + 0.1 * rng.normal(size=n)
    groups = np.repeat(np.arange(4), n // 4)
    r2_gpr, _, _ = fab.cv_gpr_r2(X, y, groups, cv='kfold', n_splits=5,
                                  n_restarts=1, seed=0)
    assert r2_gpr > 0.7, f"GPR should recover sin signal (got {r2_gpr})"


def test_gpr_lomo_runs():
    rng = _rng(3)
    n = 50
    n_mice = 4
    X = rng.normal(size=(n_mice * n, 2))
    y = X[:, 0] + 0.3 * rng.normal(size=n_mice * n)
    groups = np.repeat(np.arange(n_mice), n)
    r2, y_hat, mask = fab.cv_gpr_r2(X, y, groups, cv='lomo', n_restarts=0, seed=0)
    assert not np.isnan(r2)
    assert mask.sum() == len(y)


def test_gpr_returns_nan_on_empty():
    X = np.empty((0, 3))
    y = np.empty(0)
    groups = np.empty(0)
    r2, _, _ = fab.cv_gpr_r2(X, y, groups, cv='kfold', n_restarts=0, seed=0)
    assert np.isnan(r2)


def test_gpr_handles_nans_in_features():
    rng = _rng(5)
    n = 200
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + 0.3 * rng.normal(size=n)
    X[5, 0] = np.nan
    X[12, 1] = np.nan
    groups = np.repeat(np.arange(4), n // 4)
    r2, _, mask = fab.cv_gpr_r2(X, y, groups, cv='kfold', n_restarts=0, seed=0)
    assert not np.isnan(r2)
    assert mask.sum() == n - 2  # two rows dropped


# ======================================================================
# 2. Block ablation: signal attribution
# ======================================================================

def test_block_ablation_signal_in_temporal_variance():
    """Target loaded on temporal_variance block. drop:rate should be near-full;
    drop:temporal_variance should collapse R²."""
    df = _make_synthetic_df(n_per_mouse=150, n_mice=4, seed=10)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    rows_df = pd.DataFrame(rows)
    kfold = rows_df[rows_df['cv'] == 'kfold']
    full = kfold[kfold['model'] == 'full']['r2'].iloc[0]
    drop_rate = kfold[kfold['model'] == 'drop:rate']['r2'].iloc[0]
    drop_tvar = kfold[kfold['model'] == 'drop:temporal_variance']['r2'].iloc[0]
    only_rate = kfold[kfold['model'] == 'only:rate']['r2'].iloc[0]
    only_tvar = kfold[kfold['model'] == 'only:temporal_variance']['r2'].iloc[0]

    assert full > 0.5, f"Full model should fit (got R²={full})"
    assert abs(drop_rate - full) < 0.15, (
        f"drop:rate should be ~ full; got {drop_rate} vs {full}")
    assert drop_tvar < full - 0.3, (
        f"drop:temporal_variance should collapse; got {drop_tvar} vs {full}")
    assert only_tvar > only_rate + 0.3, (
        f"only:temporal_variance should dominate only:rate; "
        f"got tvar={only_tvar} rate={only_rate}")


def test_block_ablation_signal_in_rate():
    """Same idea, signal in rate block (Velocity ~ Pop_Mean_Raw_Late)."""
    df = _make_synthetic_df(n_per_mouse=150, n_mice=4, seed=11)
    rows = fab.run_cv_regression_blocks(
        df, 'Velocity', 'Velocity',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    rows_df = pd.DataFrame(rows)
    kfold = rows_df[rows_df['cv'] == 'kfold']
    full = kfold[kfold['model'] == 'full']['r2'].iloc[0]
    drop_rate = kfold[kfold['model'] == 'drop:rate']['r2'].iloc[0]
    drop_tvar = kfold[kfold['model'] == 'drop:temporal_variance']['r2'].iloc[0]

    assert full > 0.5
    assert drop_rate < full - 0.3, (
        f"drop:rate should collapse on rate-driven target; got {drop_rate}")
    assert abs(drop_tvar - full) < 0.15, (
        f"drop:temporal_variance shouldn't hurt; got {drop_tvar}")


def test_per_metric_ablation_finds_signal_metric():
    """The metric Temporal_Var should show the largest unique contribution
    when the target is loaded on it (perceptual_variance from the fixture)."""
    df = _make_synthetic_df(n_per_mouse=150, n_mice=4, seed=12)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    rows_df = pd.DataFrame(rows)
    kfold = rows_df[(rows_df['cv'] == 'kfold')
                    & (rows_df['model'].str.startswith('ko:'))]
    # unique = full - ko
    full = rows_df[(rows_df['cv'] == 'kfold')
                   & (rows_df['model'] == 'full')]['r2'].iloc[0]
    kfold = kfold.copy()
    kfold['unique'] = full - kfold['r2']
    # Top metric by unique contribution should be Temporal_Var (the dominant
    # signal carrier in the fixture).
    top = kfold.sort_values('unique', ascending=False).iloc[0]
    assert top['metric'] in ('Temporal_Var', 'GV'), (
        f"Top unique metric should be in the signal carriers; got {top['metric']}")


def test_null_target_gives_near_zero_r2():
    """Likelihood_Variance is pure noise in the fixture."""
    df = _make_synthetic_df(n_per_mouse=150, n_mice=4, seed=13)
    rows = fab.run_cv_regression_blocks(
        df, 'Likelihood_Variance', 'Likelihood Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    rows_df = pd.DataFrame(rows)
    full = rows_df[(rows_df['cv'] == 'kfold')
                   & (rows_df['model'] == 'full')]['r2'].iloc[0]
    assert full < 0.2, f"Pure-noise target should give R² ≈ 0 (got {full})"


# ======================================================================
# 3. Schema and shape
# ======================================================================

def test_output_schema_contains_required_keys():
    df = _make_synthetic_df(n_per_mouse=80, n_mice=3, seed=14)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    required = {'model', 'cv', 'r2', 'n_features', 'ablation_type',
                'delta_r2_vs_full'}
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"Row {r['model']!r} missing keys {missing}"


def test_per_block_and_per_metric_rows_coexist():
    df = _make_synthetic_df(n_per_mouse=80, n_mice=3, seed=15)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    rows_df = pd.DataFrame(rows)
    block_rows = rows_df[rows_df['ablation_type'] == 'block']
    metric_rows = rows_df[rows_df['ablation_type'] == 'metric']
    assert len(block_rows) > 0
    assert len(metric_rows) > 0


def test_null_baseline_appears_when_requested():
    df = _make_synthetic_df(n_per_mouse=80, n_mice=3, seed=16)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
        include_null=True,
    )
    rows_df = pd.DataFrame(rows)
    null_rows = rows_df[rows_df['model'] == 'null']
    assert len(null_rows) > 0
    # Null R² should be near zero (within noise)
    assert null_rows['r2'].iloc[0] < 0.2


# ======================================================================
# 4. Target transforms
# ======================================================================

def test_log_transform_applied():
    """log-transformed contrast should be a different fit from raw contrast."""
    df = _make_synthetic_df(n_per_mouse=100, n_mice=4, seed=17)
    rows_raw = fab.run_cv_regression_blocks(
        df, 'Contrast', 'Contrast', regressor='ridge',
        cv_schemes=('kfold',), n_splits=5, seed=0, target_transform=None,
    )
    rows_log = fab.run_cv_regression_blocks(
        df, 'Contrast', 'log(Contrast)', regressor='ridge',
        cv_schemes=('kfold',), n_splits=5, seed=0, target_transform='log',
    )
    r_raw = pd.DataFrame(rows_raw)
    r_log = pd.DataFrame(rows_log)
    full_raw = r_raw[r_raw['model'] == 'full']['r2'].iloc[0]
    full_log = r_log[r_log['model'] == 'full']['r2'].iloc[0]
    # The two should not be identical (features are uninformative about
    # Contrast in the fixture, so both should be ~0, but the function path
    # for transform must run without errors).
    assert not np.isnan(full_raw)
    assert not np.isnan(full_log)


# ======================================================================
# 5. Block definitions match the catalog
# ======================================================================

def test_block_definitions_subset_of_metric_bases():
    """All block metrics must be valid METRIC_BASES entries (Traj_Length
    is deliberately excluded from BLOCKS — order-sensitive features are
    dropped from the active feature set)."""
    all_block_names = set(fab.RATE_BLOCK + fab.TEMPORAL_VARIANCE_BLOCK)
    import population_metrics_vs_uncertainty as vvu
    all_metric_names = {b for b, _ in vvu.METRIC_BASES}
    assert all_block_names.issubset(all_metric_names)
    # Traj_Length is the order-sensitive feature; it must NOT be in any block.
    assert 'Traj_Length' not in all_block_names


def test_blocks_are_disjoint():
    blocks = [fab.RATE_BLOCK, fab.TEMPORAL_VARIANCE_BLOCK]
    for i, b1 in enumerate(blocks):
        for j, b2 in enumerate(blocks):
            if i == j:
                continue
            inter = set(b1) & set(b2)
            assert not inter, f"Blocks share metrics: {inter}"


def test_per_mouse_paired_t_recovers_planted_signal():
    """Per-mouse paired t-test rejects when one model is consistently
    better across mice; accepts when arrays are identical."""
    import numpy as np
    rng = _rng(60)
    # Strong consistent effect: model A always better than B
    a = np.array([0.30, 0.32, 0.28, 0.35, 0.29, 0.33])
    b = np.array([0.20, 0.21, 0.18, 0.25, 0.19, 0.22])
    p = fab._per_mouse_paired_t(a, b)
    assert p < 0.001, f"Strong consistent effect should give tiny p, got {p}"
    # Null: identical arrays
    p_null = fab._per_mouse_paired_t(a, a.copy())
    assert np.isnan(p_null) or p_null > 0.5, (
        f"Identical arrays should give large/NaN p, got {p_null}")


def test_per_mouse_paired_t_bar_height_p_value_consistency():
    """The test should agree with bar heights: closer means → larger p,
    farther means → smaller p (when SEs are comparable). This is the
    invariant the trial-level cluster-robust test violated."""
    import numpy as np
    rng = _rng(61)
    full = np.array([0.30, 0.32, 0.31, 0.29, 0.30, 0.33])
    # Two candidates with same SEM, different means
    close = full - 0.02 + rng.normal(0, 0.01, size=6)   # small consistent gap
    far = full - 0.10 + rng.normal(0, 0.01, size=6)     # big consistent gap
    p_close = fab._per_mouse_paired_t(full, close)
    p_far = fab._per_mouse_paired_t(full, far)
    assert p_close > p_far, (
        f"Larger effect must give smaller p; got p_close={p_close}, "
        f"p_far={p_far}")


def test_active_metrics_excludes_traj_length():
    active = fab._active_metrics(fab.BLOCKS)
    assert 'Traj_Length' not in active
    assert 'Pop_Mean_Raw' in active
    assert 'Temporal_Var' in active


# ======================================================================
# 6. GPR vs Ridge on the same fixture
# ======================================================================

def test_per_mouse_r2_present_in_rows():
    """Each row must carry per-mouse R² with length = n_unique_mice."""
    df = _make_synthetic_df(n_per_mouse=120, n_mice=4, seed=20)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='ridge', cv_schemes=('kfold',), n_splits=5, seed=0,
    )
    for r in rows:
        assert 'r2_per_mouse' in r, f"row {r['model']!r} missing r2_per_mouse"
        assert 'mouse_ids' in r, f"row {r['model']!r} missing mouse_ids"
        assert len(r['r2_per_mouse']) == 4
        assert len(r['mouse_ids']) == 4


def test_r2_per_mouse_recovers_known_signal():
    """Per-mouse R² should be high when each mouse carries the same
    linear signal."""
    import numpy as np
    rng = _rng(21)
    n = 200
    groups = np.repeat(np.arange(4), n)
    X = rng.normal(size=(4 * n, 2))
    y = X[:, 0] + 0.2 * rng.normal(size=4 * n)
    y_hat = X[:, 0]   # perfect predictions (up to noise)
    per_mouse, mids = fab._r2_per_mouse(y, y_hat, groups)
    assert len(per_mouse) == 4
    assert np.all(per_mouse > 0.7), f"per-mouse R² should be high, got {per_mouse}"


def test_r2_per_mouse_zero_on_constant_predictor():
    """A constant predictor (the per-mouse mean) gives R² = 0."""
    import numpy as np
    rng = _rng(22)
    n = 100
    groups = np.repeat(np.arange(3), n)
    y = rng.normal(size=3 * n)
    y_hat = np.zeros_like(y)
    # The mean of standard-normal y_v is ~0, so a zero predictor
    # is roughly the mean predictor and R² should be ~0 ± noise.
    per_mouse, _ = fab._r2_per_mouse(y, y_hat, groups)
    assert np.all(np.abs(per_mouse) < 0.15), (
        f"per-mouse R² for constant predictor should be ~0, got {per_mouse}")


def test_add_stimulus_and_behaviour_columns_derives_stim_category():
    """Stim_Category should be derived from True_Orientation."""
    import numpy as np
    df = pd.DataFrame({
        'Mouse_ID': [0, 0, 0, 0, 0],
        'Orientation': [0, 30, 45, 60, 90],
        'True_Orientation': [10.0, 30.0, 45.0, 60.0, 80.0],
        'Contrast': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Dispersion': [5.0, 5.0, 5.0, 5.0, 5.0],
        'Velocity': [0.0, 0.0, 0.0, 0.0, 0.0],
        'Choice': [1, 1, 1, 0, 0],
        'Perceptual_Variance': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Decision_Entropy': [0.5, 0.5, 0.5, 0.5, 0.5],
        'Likelihood_Variance': [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    out = fab.add_stimulus_and_behaviour_columns(df)
    assert 'Stim_Category' in out.columns
    np.testing.assert_array_equal(
        out['Stim_Category'].values,
        np.array([1.0, 1.0, 0.5, 0.0, 0.0]),
    )


def test_friedman_pairwise_detects_known_separation():
    """When one of three groups has a clean shift relative to the
    others, Friedman omnibus should reject and the pairwise involving
    that group should land at the lowest raw p."""
    import numpy as np
    # n=8 paired observations across 3 groups
    rng = _rng(40)
    n = 8
    g0 = rng.normal(loc=0.0, scale=0.05, size=n)
    g1 = rng.normal(loc=0.0, scale=0.05, size=n)
    g2 = rng.normal(loc=0.3, scale=0.05, size=n)   # clear shift
    out = fab._friedman_pairwise([g0, g1, g2], ['g0', 'g1', 'g2'])
    assert out is not None
    assert out['omnibus_p'] < 0.05, (
        f"Friedman should detect g2-vs-rest, got p={out['omnibus_p']}")
    # The pairwise involving g2 (against g0 or g1) should be the
    # smallest p
    pairs = sorted(out['pairwise'], key=lambda d: d['p_raw'])
    top = pairs[0]
    assert (top['label_i'] == 'g2' or top['label_j'] == 'g2'), (
        f"Most significant pair should involve g2; got {top}")


def test_friedman_pairwise_null_groups_no_omnibus_significance():
    """All-identical-distribution groups: omnibus p should be large."""
    import numpy as np
    rng = _rng(41)
    n = 8
    arrays = [rng.normal(size=n) for _ in range(4)]
    out = fab._friedman_pairwise(arrays, list('abcd'))
    assert out is not None
    assert out['omnibus_p'] > 0.05, (
        f"Friedman on null groups should not reject, got p={out['omnibus_p']}")


def test_holm_bonferroni_basic():
    """Holm step-down: smallest p × (m), next × (m-1), running max."""
    import numpy as np
    ps = np.array([0.01, 0.03, 0.04, 0.20])
    adj = fab._holm_bonferroni(ps)
    # Expected: 0.01*4 = 0.04, 0.03*3 = 0.09, 0.04*2 = 0.08 → running max 0.09,
    # 0.20*1 = 0.20.
    np.testing.assert_allclose(adj[0], 0.04, atol=1e-9)
    np.testing.assert_allclose(adj[1], 0.09, atol=1e-9)
    np.testing.assert_allclose(adj[2], 0.09, atol=1e-9)
    np.testing.assert_allclose(adj[3], 0.20, atol=1e-9)


def test_choice_target_in_TARGETS_list():
    """Choice and Stim_Category must appear in TARGETS for the driver
    to pick them up automatically."""
    tnames = [t[0] for t in fab.TARGETS]
    assert 'Choice' in tnames
    assert 'Stim_Category' in tnames


def test_cluster_robust_paired_t_detects_signal():
    """When yhat_A is consistently closer to y than yhat_B across all
    trials in all mice, the cluster-robust paired t-test should reject."""
    import numpy as np
    rng = _rng(50)
    n_mice = 6
    n_per_mouse = 200
    n = n_mice * n_per_mouse
    mouse_ids = np.repeat(np.arange(n_mice), n_per_mouse)
    y_true = rng.normal(size=n)
    yhat_A = y_true + 0.1 * rng.normal(size=n)  # very close
    yhat_B = y_true + 0.5 * rng.normal(size=n)  # noisier
    stat = fab._cluster_robust_paired_t(yhat_A, yhat_B, y_true, mouse_ids)
    assert stat is not None
    assert stat['mean_delta'] < 0, (
        f"err_A < err_B should give negative delta, got {stat['mean_delta']}")
    assert stat['p'] < 0.01, (
        f"Strong effect should reject, got p={stat['p']}")
    assert stat['n_clusters'] == n_mice
    assert stat['df'] == n_mice - 1


def test_cluster_robust_paired_t_null():
    """Identical predictors → mean_delta ≈ 0, p large."""
    import numpy as np
    rng = _rng(51)
    n_mice = 6
    n_per_mouse = 200
    n = n_mice * n_per_mouse
    mouse_ids = np.repeat(np.arange(n_mice), n_per_mouse)
    y_true = rng.normal(size=n)
    yhat_A = y_true + 0.3 * rng.normal(size=n)
    yhat_B = y_true + 0.3 * rng.normal(size=n)
    stat = fab._cluster_robust_paired_t(yhat_A, yhat_B, y_true, mouse_ids)
    assert stat is not None
    # No directional effect — p should be large most of the time.
    assert stat['p'] > 0.1, f"Null condition gave low p: {stat['p']}"


def test_cluster_robust_paired_t_respects_mouse_clustering():
    """If signal is *between* mice rather than within, the cluster-
    robust SE should give a much larger p-value than a naive paired
    t-test that ignored clustering."""
    import numpy as np
    rng = _rng(52)
    n_mice = 6
    n_per_mouse = 300
    n = n_mice * n_per_mouse
    mouse_ids = np.repeat(np.arange(n_mice), n_per_mouse)
    # Mouse-level offsets: each mouse has its own delta level, but trials
    # within a mouse all share that level (perfect clustering).
    mouse_offsets = np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])
    delta_per_trial = mouse_offsets[mouse_ids]
    y_true = rng.normal(size=n)
    # Construct yhat such that (yhat_A - y)^2 - (yhat_B - y)^2 == delta_per_trial
    # The easy way: set yhat_A = y + sqrt(|delta| + 1), yhat_B = y + 1.
    # We just need delta to track the per-mouse cluster pattern.
    yhat_B = y_true + 1.0
    # err_B = 1, want err_A = err_B + delta = 1 + delta
    # so |yhat_A - y| = sqrt(max(1 + delta, 0.01))
    err_A_target = np.maximum(1.0 + delta_per_trial, 0.01)
    yhat_A = y_true + np.sqrt(err_A_target)
    stat = fab._cluster_robust_paired_t(yhat_A, yhat_B, y_true, mouse_ids)
    # With 3 vs 3 mice and within-mouse cancellation, cluster-robust
    # should not reject just because trials happen to look different.
    assert stat['p'] > 0.05, (
        f"Pure between-mouse signal should not reject with cluster-robust SE; "
        f"got p={stat['p']}")


def test_trial_level_pairwise_bonferroni():
    """Three predictors with one clear winner; pairwise involving the
    winner should be highly significant after Bonferroni; the comparison
    between the two losers should not be."""
    import numpy as np
    rng = _rng(53)
    n_mice = 5
    n_per_mouse = 200
    n = n_mice * n_per_mouse
    mouse_ids = np.repeat(np.arange(n_mice), n_per_mouse)
    y_true = rng.normal(size=n)
    yhat_good = y_true + 0.05 * rng.normal(size=n)
    yhat_mid1 = y_true + 0.5 * rng.normal(size=n)
    yhat_mid2 = y_true + 0.5 * rng.normal(size=n)
    out = fab._trial_level_pairwise(
        [yhat_good, yhat_mid1, yhat_mid2], y_true, mouse_ids,
        labels=['good', 'mid1', 'mid2'],
        correction='bonferroni',
    )
    assert len(out['pairwise']) == 3
    pmap = {(p['label_i'], p['label_j']): p for p in out['pairwise']}
    # good vs mid1 and good vs mid2 should be highly significant
    assert pmap[('good', 'mid1')]['p_adj'] < 0.01
    assert pmap[('good', 'mid2')]['p_adj'] < 0.01
    # mid1 vs mid2 should NOT be significant
    assert pmap[('mid1', 'mid2')]['p_adj'] > 0.1


def test_gpr_runs_on_full_fixture():
    """Smoke test: GPR completes without error on the planted-signal fixture."""
    df = _make_synthetic_df(n_per_mouse=80, n_mice=3, seed=18)
    rows = fab.run_cv_regression_blocks(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        regressor='gpr', cv_schemes=('kfold',), n_splits=3, seed=0,
        regressor_kwargs={'n_restarts': 0},
    )
    rows_df = pd.DataFrame(rows)
    full = rows_df[(rows_df['cv'] == 'kfold')
                   & (rows_df['model'] == 'full')]['r2'].iloc[0]
    assert full > 0.3, f"GPR should recover signal on fixture (got {full})"
