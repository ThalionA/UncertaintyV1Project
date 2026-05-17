# -*- coding: utf-8 -*-
"""Synthetic-data tests for the analysis functions in
``nn_decoder/population_metrics_vs_uncertainty.py``.

The pure analysis functions are tested with known ground truth:
  - benjamini_hochberg: classical worked example.
  - partial_correlation: pure-confound → r ≈ 0; orthogonal → r ≈ raw.
  - compute_population_geometry: identity-covariance and rank-1 limits.
  - compute_churchland_ff: Poisson with rate λ → FF ≈ 1.
  - cv_ridge_r2: orthogonal feature-block ablations sum (approximately) to
    the full-model R² when blocks contribute independent signal.
  - compute_raw_metrics, compute_zscored_metrics: shape and sentinel values.
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

import population_metrics_vs_uncertainty as vvu  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _rng(seed=0):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# benjamini_hochberg
# ----------------------------------------------------------------------

def test_bh_classical_example():
    # Standard textbook example
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205])
    q = vvu.benjamini_hochberg(p)
    # Monotonic non-decreasing in p
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    # All bounded in [0, 1]
    assert np.all((q >= 0) & (q <= 1))
    # Smallest p has the smallest q
    assert q[0] == q.min()


def test_bh_with_nans():
    p = np.array([0.01, np.nan, 0.5])
    q = vvu.benjamini_hochberg(p)
    assert np.isnan(q[1])
    assert np.all(q[[0, 2]] >= 0)


def test_bh_all_significant_threshold():
    # Uniformly small p-values: every q should be < 0.05 if all p < 0.05/n
    p = np.full(10, 0.001)
    q = vvu.benjamini_hochberg(p)
    assert np.all(q < 0.05)


# ----------------------------------------------------------------------
# pearson_spearman
# ----------------------------------------------------------------------

def test_pearson_spearman_perfect_linear():
    rng = _rng(42)
    x = rng.normal(size=200)
    y = 2.0 * x + 0.5
    rp, pp, rs, ps = vvu.pearson_spearman(x, y)
    assert rp == pytest.approx(1.0, abs=1e-10)
    assert rs == pytest.approx(1.0, abs=1e-10)


def test_pearson_spearman_handles_nans():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    rp, pp, rs, ps = vvu.pearson_spearman(x, y)
    assert rp == pytest.approx(1.0, abs=1e-10)


# ----------------------------------------------------------------------
# partial_correlation
# ----------------------------------------------------------------------

def _make_partial_df(n_per_mouse=400, n_mice=3, scenario='confound', seed=0):
    """Build a dataframe matching the schema partial_correlation expects.

    scenario:
      - 'confound' : x and y are both driven by Contrast/Dispersion only;
                      after partialling there is no residual correlation.
      - 'pure'     : x and y share a stimulus-orthogonal component;
                      partial r should match the raw r.
    """
    rng = _rng(seed)
    rows = []
    for mid in range(n_mice):
        ori = rng.choice([0, 30, 60, 90], size=n_per_mouse)
        con = rng.choice([0.1, 0.5, 1.0], size=n_per_mouse)
        disp = rng.choice([0.1, 0.5, 1.0], size=n_per_mouse)

        # Stimulus signal
        stim_signal = (con * 2.0) + (disp * -1.5) + (ori * 0.01)
        eps_x = rng.normal(scale=0.1, size=n_per_mouse)
        eps_y = rng.normal(scale=0.1, size=n_per_mouse)
        if scenario == 'confound':
            # Both driven only by stimulus + independent noise
            x = stim_signal + eps_x
            y = stim_signal + eps_y
        elif scenario == 'pure':
            # x has independent component that drives y (stimulus is independent)
            shared = rng.normal(size=n_per_mouse)
            x = shared + eps_x
            y = shared + eps_y
        else:
            raise ValueError(scenario)

        rows.append(pd.DataFrame({
            'Mouse_ID': mid,
            'Orientation': ori,
            'Contrast': con,
            'Dispersion': disp,
            'Velocity': rng.normal(size=n_per_mouse),
            'Lick_Rate': rng.normal(size=n_per_mouse),
            'x': x,
            'y': y,
        }))
    return pd.concat(rows, ignore_index=True)


def test_partial_correlation_pure_confound_collapses():
    df = _make_partial_df(scenario='confound', seed=7)
    # Raw correlation should be high (both driven by stimulus)
    rp, _, _, _ = vvu.pearson_spearman(df['x'].values, df['y'].values)
    assert abs(rp) > 0.7
    # After partialling out (Ori, Contrast, Disp), residual r should be near zero
    res = vvu.partial_correlation(df, 'x', 'y',
                                   controls=('Orientation', 'Contrast', 'Dispersion'),
                                   per_mouse=True)
    assert abs(res['r']) < 0.1, f"expected r ≈ 0, got {res['r']:.3f}"


def test_partial_correlation_orthogonal_signal_survives():
    df = _make_partial_df(scenario='pure', seed=11)
    rp, _, _, _ = vvu.pearson_spearman(df['x'].values, df['y'].values)
    assert rp > 0.5  # strong shared signal
    res = vvu.partial_correlation(df, 'x', 'y',
                                   controls=('Orientation', 'Contrast', 'Dispersion'),
                                   per_mouse=True)
    # Partial r should be similar to raw r (within 0.1)
    assert abs(res['r'] - rp) < 0.1, f"raw={rp:.3f}, partial={res['r']:.3f}"


# ----------------------------------------------------------------------
# Stimulus-design parameter: 'joint' (default) vs 'additive' (legacy).
# Switched 2026-05-16 after the empirical comparison in
# nn_decoder/compare_partial_corr_designs.py.
# ----------------------------------------------------------------------

def test_partial_correlation_default_is_joint():
    """Default `design=` should be 'joint'. Caller-facing regression test
    so the default doesn't silently flip back to additive."""
    df = _make_partial_df(scenario='confound', seed=3)
    res_default = vvu.partial_correlation(
        df, 'x', 'y',
        controls=('Orientation', 'Contrast', 'Dispersion'),
        per_mouse=True,
    )
    res_joint = vvu.partial_correlation(
        df, 'x', 'y',
        controls=('Orientation', 'Contrast', 'Dispersion'),
        per_mouse=True, design='joint',
    )
    assert res_default['r'] == pytest.approx(res_joint['r'], abs=1e-12)


def test_partial_correlation_additive_still_callable():
    """Legacy main-effects-only behaviour must remain reachable for
    backward compatibility with pre-2026-05-16 figures."""
    df = _make_partial_df(scenario='pure', seed=4)
    res_add = vvu.partial_correlation(
        df, 'x', 'y',
        controls=('Orientation', 'Contrast', 'Dispersion'),
        per_mouse=True, design='additive',
    )
    # 'pure' scenario has stim-independent signal, so additive partial r
    # should still be high.
    assert res_add['r'] > 0.5


def test_partial_correlation_joint_collapses_per_cell_predictor():
    """A predictor that is constant within each (mouse, stim cell) — i.e.
    equals the per-cell training mean of y — should give NaN partial r
    under joint-cell residualisation (structural null). Under additive
    residualisation the same predictor leaves non-zero residuals because
    cell-interaction structure isn't absorbed."""
    rng = _rng(13)
    n_per_cell = 30
    n_mice = 3
    rows = []
    for mid in range(n_mice):
        for ori in (0, 30, 60):
            for con in (0.1, 0.5):
                for disp in (0.1, 0.5):
                    # interaction-shaped per-cell mean
                    base = (con * disp * 10.0) + (ori * 0.05)
                    for _ in range(n_per_cell):
                        within = rng.normal(scale=1.0)
                        rows.append({
                            'Mouse_ID': mid,
                            'Orientation': ori,
                            'Contrast': con,
                            'Dispersion': disp,
                            'y': base + within,
                            'x_per_cell': base,  # per-cell predictor
                        })
    df = pd.DataFrame(rows)

    res_joint = vvu.partial_correlation(
        df, 'x_per_cell', 'y',
        controls=('Orientation', 'Contrast', 'Dispersion'),
        per_mouse=True, design='joint',
    )
    # All mice dropped (residual std = 0 for x); partial r is NaN.
    assert np.isnan(res_joint['r']) or abs(res_joint['r']) < 0.05

    res_add = vvu.partial_correlation(
        df, 'x_per_cell', 'y',
        controls=('Orientation', 'Contrast', 'Dispersion'),
        per_mouse=True, design='additive',
    )
    # Additive can't absorb the contrast*dispersion interaction, so the
    # x_per_cell predictor still has surviving residual variance and the
    # partial r is meaningfully non-zero.
    assert abs(res_add['r']) > 0.4


def test_stimulus_design_matrix_joint_has_more_columns_than_additive():
    """Sanity: joint design produces (# unique cells) + intercept columns;
    additive produces (Σ levels − Σ drop_first) + intercept. Joint is
    therefore wider on any DataFrame with non-degenerate interactions."""
    df = _make_partial_df(scenario='pure', seed=5)
    Z_add = vvu._stimulus_design_matrix(
        df, design='additive',
    )
    Z_joint = vvu._stimulus_design_matrix(
        df, design='joint',
    )
    assert Z_joint.shape[1] > Z_add.shape[1]


def test_stimulus_design_matrix_rejects_unknown_design():
    df = _make_partial_df(scenario='pure', seed=6)
    with pytest.raises(ValueError):
        vvu._stimulus_design_matrix(df, design='bogus')


# ----------------------------------------------------------------------
# compute_population_geometry
# ----------------------------------------------------------------------

def test_participation_ratio_isotropic_limit():
    """Isotropic noise across t_bins, n_neurons → PR ~ min(t-1, n) (data-rank-limited)."""
    rng = _rng(0)
    n_trials, t_bins, n_neurons = 30, 50, 8
    z = rng.normal(size=(n_trials, t_bins, n_neurons))
    geom = vvu.compute_population_geometry(z)
    pr_mean = np.nanmean(geom['Participation_Ratio'])
    # With t_bins much larger than n_neurons, all n_neurons modes get sampled
    # and PR approaches n_neurons. We allow a margin for finite-sample noise.
    assert pr_mean > 0.55 * n_neurons, f"PR={pr_mean:.2f}, expected near {n_neurons}"


def test_participation_ratio_rank1_limit():
    """A single shared mode across neurons within each trial → PR ≈ 1."""
    rng = _rng(0)
    n_trials, t_bins, n_neurons = 20, 30, 12
    z = np.zeros((n_trials, t_bins, n_neurons))
    for i in range(n_trials):
        # All neurons follow the same temporal signal (with tiny per-neuron noise)
        signal = rng.normal(size=t_bins)
        z[i] = signal[:, None] + 1e-3 * rng.normal(size=(t_bins, n_neurons))
    geom = vvu.compute_population_geometry(z)
    pr_mean = np.nanmean(geom['Participation_Ratio'])
    tmd_mean = np.nanmean(geom['Top_Mode_Dominance'])
    assert pr_mean < 1.5, f"expected PR ≈ 1, got {pr_mean:.3f}"
    assert tmd_mean > 0.95, f"expected TMD ≈ 1, got {tmd_mean:.3f}"


# ----------------------------------------------------------------------
# cv_ridge_r2
# ----------------------------------------------------------------------

def test_cv_ridge_pure_noise_r2_near_zero():
    rng = _rng(0)
    n = 500
    X = rng.normal(size=(n, 10))
    y = rng.normal(size=n)
    groups = np.repeat(np.arange(5), n // 5)
    r2_kf, _, _ = vvu.cv_ridge_r2(X, y, groups, cv='kfold', n_splits=5)
    # Should hover near 0 (or mildly negative due to CV noise)
    assert r2_kf < 0.05, f"R²={r2_kf:.3f}, expected near 0"


def test_cv_ridge_recovers_linear_signal():
    rng = _rng(0)
    n = 500
    X = rng.normal(size=(n, 5))
    beta = np.array([1.0, -0.5, 0.0, 0.0, 0.3])
    y = X @ beta + 0.5 * rng.normal(size=n)
    groups = np.repeat(np.arange(5), n // 5)
    r2_kf, _, _ = vvu.cv_ridge_r2(X, y, groups, cv='kfold',
                                   alpha=0.1, n_splits=5)
    assert r2_kf > 0.6, f"R²={r2_kf:.3f}, expected > 0.6"


def test_cv_ridge_lomo_runs():
    rng = _rng(0)
    n_per = 100
    n_groups = 4
    X = rng.normal(size=(n_per * n_groups, 4))
    y = X @ np.array([1.0, -0.5, 0.0, 0.5]) + rng.normal(size=X.shape[0])
    groups = np.repeat(np.arange(n_groups), n_per)
    r2, y_hat, mask = vvu.cv_ridge_r2(X, y, groups, cv='lomo')
    assert np.isfinite(r2)


def test_cv_ridge_drop_one_block_reduces_r2_when_block_carries_signal():
    rng = _rng(1)
    n = 600
    X_signal = rng.normal(size=(n, 3))           # block A — carries the signal
    X_noise = rng.normal(size=(n, 3))            # block B — noise
    X_full = np.column_stack([X_signal, X_noise])
    y = X_signal @ np.array([1.0, -0.5, 0.7]) + 0.3 * rng.normal(size=n)
    groups = np.repeat(np.arange(5), n // 5)

    r2_full, _, _ = vvu.cv_ridge_r2(X_full, y, groups, cv='kfold',
                                     alpha=0.1, n_splits=5)
    r2_drop_signal, _, _ = vvu.cv_ridge_r2(X_noise, y, groups, cv='kfold',
                                            alpha=0.1, n_splits=5)
    r2_drop_noise, _, _ = vvu.cv_ridge_r2(X_signal, y, groups, cv='kfold',
                                           alpha=0.1, n_splits=5)
    # Dropping the signal block must hurt; dropping noise should leave it ~unchanged
    assert r2_full - r2_drop_signal > 0.3, \
        f"full={r2_full:.3f}, drop_signal={r2_drop_signal:.3f}"
    assert abs(r2_full - r2_drop_noise) < 0.1, \
        f"full={r2_full:.3f}, drop_noise={r2_drop_noise:.3f}"


# ----------------------------------------------------------------------
# compute_raw_metrics, compute_zscored_metrics — shape & sentinel checks
# ----------------------------------------------------------------------

def test_raw_metrics_shapes_and_poisson_recovery():
    rng = _rng(0)
    n_trials, t_bins, n_neurons = 100, 30, 20
    raw = rng.poisson(2.5, size=(n_trials, t_bins, n_neurons)).astype(float)
    out = vvu.compute_raw_metrics(raw)
    assert out['Pop_Mean_Raw'].shape == (n_trials,)
    assert out['Temporal_Fano_Raw'].shape == (n_trials,)
    assert abs(np.nanmean(out['Pop_Mean_Raw']) - 2.5) < 0.1
    # Per-neuron temporal Fano: var/mean of Poisson within a trial → ≈ 1
    assert 0.85 < np.nanmean(out['Temporal_Fano_Raw']) < 1.15


def test_zscored_metrics_pop_deviation_is_zero():
    rng = _rng(0)
    raw = rng.normal(size=(80, 25, 12))
    z = vvu._zscore_per_neuron(raw)
    out = vvu.compute_zscored_metrics(z)
    # Mean of Pop_Deviation across trials must be ~0 (z-scoring removes per-neuron mean)
    assert abs(np.mean(out['Pop_Deviation'])) < 1e-10


# ----------------------------------------------------------------------
# compute_trajectory_metrics — return signature, centroid, GV_Fano
# ----------------------------------------------------------------------

def test_trajectory_metrics_returns_five_values_with_correct_shapes():
    rng = _rng(3)
    z = rng.normal(size=(40, 30, 16))
    out = vvu.compute_trajectory_metrics(z, fixed_k=8)
    # Now returns 5 things
    assert len(out) == 5
    traj_len, gv, var_expl, centroid_mag, gv_fano = out
    n_trials = z.shape[0]
    assert traj_len.shape == (n_trials,)
    assert gv.shape == (n_trials,)
    assert centroid_mag.shape == (n_trials,)
    assert gv_fano.shape == (n_trials,)
    assert 0.0 <= var_expl <= 1.0
    # Centroid magnitude is a non-negative L2 norm
    assert np.all(centroid_mag >= 0)
    # GV_Fano is consistent with the published formula
    eps = 1e-8
    expected = gv - np.log(centroid_mag + eps)
    assert np.allclose(gv_fano, expected)


def test_centroid_magnitude_zero_for_centred_trajectories():
    """If every trial's trajectory has zero mean (already centred per trial),
    centroid magnitude should be ~0."""
    rng = _rng(4)
    n_trials, t_bins, n_neurons = 20, 30, 12
    raw = rng.normal(size=(n_trials, t_bins, n_neurons))
    # Subtract per-trial mean over time so each centroid is identically 0
    raw -= raw.mean(axis=1, keepdims=True)
    _, _, _, centroid_mag, _ = vvu.compute_trajectory_metrics(raw, fixed_k=6)
    assert np.all(centroid_mag < 1e-8)


def test_gv_fano_subtracts_log_centroid_in_high_displacement_trial():
    """Adding a constant displacement to every neuron in one trial should
    leave GV unchanged but increase centroid magnitude → GV_Fano shrinks."""
    rng = _rng(5)
    n_trials, t_bins, n_neurons = 30, 25, 10
    z = rng.normal(size=(n_trials, t_bins, n_neurons))
    # Trial 0 left as-is, trial 1 displaced by +5 in every dim
    z_disp = z.copy()
    z_disp[1] += 5.0
    _, gv0, _, c0, gvf0 = vvu.compute_trajectory_metrics(z, fixed_k=6)
    _, gv1, _, c1, gvf1 = vvu.compute_trajectory_metrics(z_disp, fixed_k=6)
    # Trial 1's displaced version: centroid magnitude grew, GV roughly
    # the same, so GV_Fano should be smaller for the displaced trial
    assert c1[1] > c0[1] + 1.0
    # Compare GV_Fano change for trial 1 vs the displacement-induced
    # change in -log(centroid)
    expected_drop = np.log(c1[1] + 1e-8) - np.log(c0[1] + 1e-8)
    actual_drop = gvf0[1] - gvf1[1]
    # GV may shift slightly due to PCA refit, but the dominant effect is
    # the centroid term
    assert abs(actual_drop - expected_drop) < 0.5 * abs(expected_drop) + 0.5


def test_gv_regularisation_default_is_1e_minus_4():
    """Document the bumped default explicitly."""
    import inspect
    sig = inspect.signature(vvu.compute_trajectory_metrics)
    assert sig.parameters['reg'].default == 1e-4


def test_lomo_handles_between_mouse_offsets():
    """LOMO must NOT collapse to large-negative R² when the only between-mouse
    difference is a per-mouse feature offset (which carries no signal).

    Build 4 mice where every mouse has the same within-mouse linear
    relationship between feature and target, but each mouse's feature
    column has a large additive offset. Within-mouse feature z-scoring
    should remove the offset so LOMO recovers a positive R²."""
    rng = _rng(11)
    n_per_mouse = 200
    n_mice = 4
    rows = []
    for mid in range(n_mice):
        x = rng.normal(size=n_per_mouse)
        y = 1.5 * x + 0.3 * rng.normal(size=n_per_mouse)
        # Add a mouse-specific feature offset that has nothing to do with y
        x_offset = x + 10.0 * mid
        # Add a mouse-specific TARGET mean shift too — within-mouse target
        # z-scoring should remove this. (Same handling on both sides.)
        y_offset = y + 5.0 * mid
        rows.append(pd.DataFrame({
            'Mouse_ID': mid,
            'Pop_Mean_Raw_Full': x_offset,
            'Pop_Mean_Raw_Late': x_offset,  # paired window column
            'Perceptual_Variance': y_offset,
        }))
    df = pd.concat(rows, ignore_index=True)

    cv_rows = vvu.run_cv_regression(
        df, 'Perceptual_Variance', 'Perceptual Variance',
        per_mouse_neuron_rates_late=None, n_splits=3)

    full_kfold = next(r['r2'] for r in cv_rows
                      if r['model'] == 'full' and r['cv'] == 'kfold')
    full_lomo = next(r['r2'] for r in cv_rows
                     if r['model'] == 'full' and r['cv'] == 'lomo')
    # Both should recover the within-mouse linear relationship
    assert full_kfold > 0.7, f"kfold R² = {full_kfold:.3f}"
    assert full_lomo > 0.5, f"LOMO R² = {full_lomo:.3f} (should be positive — between-mouse offsets shouldn't punish it)"


def test_trajectory_length_now_in_pca_subspace():
    """After the migration into PCA-k space, trajectory length must be
    bounded above by the trajectory length computed in the full
    n_neurons space (PCA projection is non-expansive in L2)."""
    rng = _rng(6)
    n_trials, t_bins, n_neurons = 25, 30, 16
    z = rng.normal(size=(n_trials, t_bins, n_neurons))
    k = 8
    out = vvu.compute_trajectory_metrics(z, fixed_k=k)
    traj_len_pca = out[0]

    # Reference: trajectory length in full neuron space
    diffs_full = np.diff(z, axis=1)
    traj_len_full = np.sum(np.linalg.norm(diffs_full, axis=2), axis=1)

    # PCA projection cannot increase L2 step norms → per-trial inequality
    assert np.all(traj_len_pca <= traj_len_full + 1e-9)
    # And the reduction is non-trivial (we kept k=8 of n_neurons=16)
    assert np.mean(traj_len_pca) < 0.95 * np.mean(traj_len_full)


# ----------------------------------------------------------------------
# Smoke test: run the full DataFrame-level analysis path on synthetic data
# ----------------------------------------------------------------------

def test_full_dataframe_analysis_smoke():
    """End-to-end smoke: build a synthetic results_df and run the
    correlation / partial-correlation / CV regression analysis path."""
    rng = _rng(2024)
    n_per_mouse = 300
    n_mice = 3
    rows = []
    per_mouse_neuron_rates = {}
    for mid in range(n_mice):
        ori = rng.choice([0, 30, 60, 90], size=n_per_mouse)
        con = rng.choice([0.1, 0.5, 1.0], size=n_per_mouse)
        disp = rng.choice([0.1, 0.5, 1.0], size=n_per_mouse)
        # Synthetic uncertainty target driven partly by stim, partly by latent
        latent = rng.normal(size=n_per_mouse)
        unc = 0.5 * con + 0.3 * disp + 0.4 * latent + 0.1 * rng.normal(size=n_per_mouse)

        # Synthetic neural metrics: each correlated with a different driver
        d = {
            'Mouse_ID': mid,
            'Trial_Idx': np.arange(n_per_mouse),
            'Orientation': ori, 'Contrast': con, 'Dispersion': disp,
            'Velocity': rng.normal(size=n_per_mouse),
            'Lick_Rate': rng.normal(size=n_per_mouse),
            'Choice': rng.integers(0, 2, size=n_per_mouse),
            'Perceptual_Variance': unc,
            'Decision_Entropy': unc + rng.normal(scale=0.2, size=n_per_mouse),
            'Likelihood_Variance': unc + rng.normal(scale=0.2, size=n_per_mouse),
        }
        # Add metric columns for both windows and all bases
        for win in ['Full', 'Late']:
            d[f'Pop_Mean_Raw_{win}']      = 2.0 + 0.5 * con + rng.normal(scale=0.1, size=n_per_mouse)
            d[f'Temporal_Fano_Raw_{win}'] = 1.0 + 0.2 * latent + rng.normal(scale=0.1, size=n_per_mouse)
            d[f'Pop_Deviation_{win}']     = rng.normal(scale=0.1, size=n_per_mouse)
            d[f'Spatial_Var_{win}']       = 0.5 + 0.3 * disp + rng.normal(scale=0.05, size=n_per_mouse)
            d[f'Temporal_Var_{win}']      = 1.0 + 0.2 * con + rng.normal(scale=0.05, size=n_per_mouse)
            d[f'Traj_Length_{win}']         = 100 + 10 * latent + rng.normal(scale=2, size=n_per_mouse)
            d[f'GV_{win}']                  = -1 + 0.1 * latent + rng.normal(scale=0.05, size=n_per_mouse)
            d[f'Centroid_Magnitude_{win}']  = 1.0 + 0.05 * latent + rng.normal(scale=0.02, size=n_per_mouse)
            d[f'GV_Fano_{win}']             = -1.0 + 0.1 * latent + rng.normal(scale=0.05, size=n_per_mouse)
            d[f'Participation_Ratio_{win}'] = 5 + 1.0 * latent + rng.normal(scale=0.2, size=n_per_mouse)
            d[f'Top_Mode_Dominance_{win}']  = 0.3 + 0.05 * latent + rng.normal(scale=0.02, size=n_per_mouse)
        rows.append(pd.DataFrame(d))
        per_mouse_neuron_rates[mid] = rng.normal(loc=2.0, size=(n_per_mouse, 30))

    df = pd.concat(rows, ignore_index=True)

    # 1. Correlation summary table runs and returns expected schema
    corr_df = vvu.print_correlation_summary(df, 'Perceptual_Variance', 'Perceptual Variance')
    assert {'metric', 'window', 'pearson_r', 'pearson_q', 'spearman_r', 'spearman_q'}.issubset(corr_df.columns)
    # 2. Partial correlation runs
    partial_df = vvu.print_partial_correlation_summary(df, 'Perceptual_Variance', 'Perceptual Variance')
    assert 'partial_r_stim' in partial_df.columns
    # 3. CV regression runs
    cv_rows = vvu.run_cv_regression(df, 'Perceptual_Variance', 'Perceptual Variance',
                                     per_mouse_neuron_rates_late=per_mouse_neuron_rates,
                                     n_splits=3)
    assert any(r['model'] == 'full' for r in cv_rows)
    # Per-metric only and ko ablations present
    assert any(r['model'].startswith('only:') for r in cv_rows)
    assert any(r['model'].startswith('ko:') for r in cv_rows)
    # delta_r2_vs_full present for non-full rows
    full_kfold_r2 = next(r['r2'] for r in cv_rows if r['model'] == 'full' and r['cv'] == 'kfold')
    other = [r for r in cv_rows if r['model'] != 'full' and r['cv'] == 'kfold']
    for r in other:
        assert 'delta_r2_vs_full' in r
        assert r['delta_r2_vs_full'] == pytest.approx(r['r2'] - full_kfold_r2, abs=1e-9)
    # KO rows additionally have unique_r2_vs_full = full - r2
    ko_kfold = [r for r in cv_rows if r['model'].startswith('ko:') and r['cv'] == 'kfold']
    assert ko_kfold, "expected at least one ko: row in kfold scheme"
    for r in ko_kfold:
        assert 'unique_r2_vs_full' in r
        assert r['unique_r2_vs_full'] == pytest.approx(full_kfold_r2 - r['r2'], abs=1e-9)
        # And metric / metric_label fields exist
        assert 'metric' in r and isinstance(r['metric'], str)
        assert 'metric_label' in r
