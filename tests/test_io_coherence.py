# -*- coding: utf-8 -*-
"""Synthetic-data tests for the IO–decoder coherence analysis in
``nn_decoder/io_coherence.py``.

Tests verify:
  - Log posterior odds g(Q) on the boundary grid (uniform, peaked).
  - Stage 2 lapse psychometric saturates at (gamma_r, 1 - delta_r).
  - Naive choice prediction is the boundary integral of Q.
  - Velocity prediction matches the IO's Stage 1 emission for known Q shapes.
  - Choice metrics (NLL, AUROC) on degenerate cases.
  - Velocity metrics (R^2, Pearson r) on perfect-prediction and constant
    cases (the constant-prediction case must not return NaN).
  - The trial-mask logic that drops the last trial of each session.
  - Trial-alignment sanity check raises on length and value mismatches.
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

import io_coherence as ioc  # noqa: E402


GRID = np.arange(0, 91, 1, dtype=float)
DEFAULT_UTILITY = dict(R_hit=1.0, R_miss=0.0, R_cr=0.1, R_fa=-0.2)


def _gauss(mu, sigma, grid=GRID):
    p = np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    return p / p.sum()


# ----------------------------------------------------------------------
# Log posterior odds g(Q)
# ----------------------------------------------------------------------

def test_g_uniform_q_is_zero():
    """g(uniform Q on 0..90) = 0 because P(Go) = P(NoGo) = 0.5 by symmetry
    around the 45 deg boundary on a 91-point grid that splits 45/45 with
    boundary mass shared 50/50."""
    Q = np.ones((1, len(GRID))) / len(GRID)
    g = ioc.compute_log_posterior_odds(Q, GRID)
    assert g.shape == (1,)
    assert abs(g[0]) < 1e-9


def test_g_peaked_go_is_large_positive():
    Q = np.zeros((1, len(GRID)))
    Q[0, 0] = 1.0  # all mass on theta=0 (Go)
    g = ioc.compute_log_posterior_odds(Q, GRID)
    assert g[0] > 10.0


def test_g_peaked_nogo_is_large_negative():
    Q = np.zeros((1, len(GRID)))
    Q[0, 90] = 1.0  # all mass on theta=90 (NoGo)
    g = ioc.compute_log_posterior_odds(Q, GRID)
    assert g[0] < -10.0


def test_g_clips_infinities():
    """A Q with literal zero on one side must not return inf; the clipping
    inside compute_log_posterior_odds keeps it finite."""
    Q = np.zeros((2, len(GRID)))
    Q[0, 0] = 1.0
    Q[1, 90] = 1.0
    g = ioc.compute_log_posterior_odds(Q, GRID)
    assert np.all(np.isfinite(g))


# ----------------------------------------------------------------------
# Stage 2 lapse psychometric
# ----------------------------------------------------------------------

def test_lapse_psychometric_zero_g_no_lapse_is_half():
    p = ioc.predict_choice_lapse(np.array([0.0]), 1.0, 0.0, 0.0, 0.0)
    assert abs(p[0] - 0.5) < 1e-12


def test_lapse_psychometric_saturates_to_one_minus_delta():
    p_high = ioc.predict_choice_lapse(np.array([100.0]), 1.0, 0.0, 0.05, 0.03)
    p_low = ioc.predict_choice_lapse(np.array([-100.0]), 1.0, 0.0, 0.05, 0.03)
    assert p_high[0] == pytest.approx(1.0 - 0.03, abs=1e-6)
    assert p_low[0] == pytest.approx(0.05, abs=1e-6)


def test_lapse_psychometric_offset_by_beta():
    """At g=0, beta_r=2 with alpha_r=1 should give P(Go) = sigmoid(2)."""
    p = ioc.predict_choice_lapse(np.array([0.0]), 1.0, 2.0, 0.0, 0.0)
    expected = 1.0 / (1.0 + np.exp(-2.0))
    assert abs(p[0] - expected) < 1e-12


# ----------------------------------------------------------------------
# Naive choice prediction
# ----------------------------------------------------------------------

def test_naive_pgo_uniform_is_half():
    Q = np.ones((1, len(GRID))) / len(GRID)
    p = ioc.predict_choice_naive(Q, GRID)
    assert abs(p[0] - 0.5) < 1e-12


def test_naive_pgo_peaked_go_is_one():
    Q = np.zeros((1, len(GRID)))
    Q[0, 0] = 1.0
    p = ioc.predict_choice_naive(Q, GRID)
    assert p[0] == 1.0


def test_naive_pgo_peaked_nogo_is_zero():
    Q = np.zeros((1, len(GRID)))
    Q[0, 90] = 1.0
    p = ioc.predict_choice_naive(Q, GRID)
    assert p[0] == 0.0


def test_naive_pgo_at_boundary_is_half():
    """All mass at theta=45 -> P(Go) = 0.5 (boundary mass split)."""
    Q = np.zeros((1, len(GRID)))
    Q[0, 45] = 1.0
    p = ioc.predict_choice_naive(Q, GRID)
    assert p[0] == pytest.approx(0.5, abs=1e-12)


# ----------------------------------------------------------------------
# Velocity prediction
# ----------------------------------------------------------------------

def test_predict_velocity_peaked_go_uses_full_dv():
    """Q peaked at theta=0: U(Go|0)=R_hit=1, U(NoGo|0)=R_miss=0 -> DV=1.
    vel = beta_vel * DV + alpha_vel."""
    Q = np.zeros((1, len(GRID)))
    Q[0, 0] = 1.0
    vel = ioc.predict_velocity(Q, GRID, beta_vel=-2.0, alpha_vel=0.5,
                               utility=DEFAULT_UTILITY)
    expected = -2.0 * 1.0 + 0.5
    assert abs(vel[0] - expected) < 1e-12


def test_predict_velocity_peaked_nogo_uses_negative_dv():
    """Q peaked at theta=90: U(Go|90)=R_fa=-0.2, U(NoGo|90)=R_cr=0.1 ->
    DV = -0.3."""
    Q = np.zeros((1, len(GRID)))
    Q[0, 90] = 1.0
    vel = ioc.predict_velocity(Q, GRID, beta_vel=-2.0, alpha_vel=0.5,
                               utility=DEFAULT_UTILITY)
    expected = -2.0 * (-0.3) + 0.5
    assert vel[0] == pytest.approx(expected, abs=1e-12)


def test_predict_velocity_uniform_q_matches_boundary_logic():
    """For the 91-point grid with a 45 deg boundary, 45 bins below + the
    boundary bin (split 50/50) + 45 bins above. Compute analytically."""
    Q = np.ones((1, len(GRID))) / len(GRID)
    n = len(GRID)
    # EU(Go) under uniform Q: 45*R_hit + 0.5*(R_hit + R_fa) + 45*R_fa, divided by n
    eu_go = (45 * 1.0 + 0.5 * (1.0 - 0.2) + 45 * -0.2) / n
    eu_nogo = (45 * 0.0 + 0.5 * (0.0 + 0.1) + 45 * 0.1) / n
    expected_dv = eu_go - eu_nogo
    expected_vel = -2.0 * expected_dv + 0.0
    vel = ioc.predict_velocity(Q, GRID, beta_vel=-2.0, alpha_vel=0.0,
                               utility=DEFAULT_UTILITY)
    assert vel[0] == pytest.approx(expected_vel, abs=1e-12)


# ----------------------------------------------------------------------
# Choice metrics
# ----------------------------------------------------------------------

def test_choice_metrics_perfect_prediction_low_nll_high_auroc():
    p_pred = np.array([0.99, 0.01, 0.99, 0.01, 0.99, 0.01])
    actual = np.array([1, 0, 1, 0, 1, 0])
    nll, auroc = ioc.compute_choice_metrics(p_pred, actual)
    assert nll < 0.05  # log(0.99) ~ -0.01
    assert auroc == 1.0


def test_choice_metrics_random_prediction_chance_nll():
    p_pred = np.full(200, 0.5)
    rng = np.random.default_rng(0)
    actual = rng.integers(0, 2, size=200)
    nll, auroc = ioc.compute_choice_metrics(p_pred, actual)
    # Mean NLL of constant 0.5 prediction = log(2) regardless of actual.
    assert abs(nll - np.log(2)) < 1e-6


def test_choice_metrics_handles_extreme_probabilities():
    # P near 0 or 1 must not cause -inf log
    p_pred = np.array([1.0, 0.0, 1.0, 0.0])
    actual = np.array([1, 0, 1, 0])
    nll, auroc = ioc.compute_choice_metrics(p_pred, actual)
    assert np.isfinite(nll)


# ----------------------------------------------------------------------
# Velocity metrics
# ----------------------------------------------------------------------

def test_velocity_metrics_perfect():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = actual.copy()
    r2, r = ioc.compute_velocity_metrics(pred, actual)
    assert r2 == pytest.approx(1.0, abs=1e-12)
    assert r == pytest.approx(1.0, abs=1e-12)


def test_velocity_metrics_zero_signal_constant_prediction():
    """A constant prediction has zero variance, so Pearson r is undefined.
    The function must report this as NaN rather than crashing."""
    rng = np.random.default_rng(0)
    actual = rng.normal(size=100)
    pred = np.full(100, actual.mean())
    r2, r = ioc.compute_velocity_metrics(pred, actual)
    # R² of a constant predictor equal to the mean is exactly 0.
    assert abs(r2) < 1e-9
    # Pearson r is undefined for zero-variance pred -> NaN
    assert np.isnan(r)


def test_velocity_metrics_anticorrelated():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = -actual + 10.0  # perfectly negatively correlated, biased
    r2, r = ioc.compute_velocity_metrics(pred, actual)
    # Pearson r = -1
    assert r == pytest.approx(-1.0, abs=1e-12)
    # R² of a biased anticorrelated predictor can be very negative
    assert r2 < 0.0


# ----------------------------------------------------------------------
# Trial mask: drop last trial of each session
# ----------------------------------------------------------------------

def test_trials_to_keep_mask_drops_last_per_session():
    sessions = np.array(['s1', 's1', 's1', 's2', 's2', 's3'])
    mask = ioc.get_trials_to_keep_mask(sessions)
    expected = np.array([True, True, False, True, False, False])
    assert np.array_equal(mask, expected)


def test_trials_to_keep_mask_handles_single_session():
    sessions = np.array(['s1', 's1', 's1', 's1'])
    mask = ioc.get_trials_to_keep_mask(sessions)
    assert np.array_equal(mask, np.array([True, True, True, False]))


def test_trials_to_keep_mask_handles_single_trial():
    sessions = np.array(['s1'])
    mask = ioc.get_trials_to_keep_mask(sessions)
    assert mask.shape == (1,)
    assert mask[0] == False  # noqa: E712


# ----------------------------------------------------------------------
# Alignment sanity check
# ----------------------------------------------------------------------

def test_alignment_passes_for_matching_arrays():
    a = np.array([0, 1, 1, 0, 1])
    b = np.array([0, 1, 1, 0, 1])
    # Should not raise
    ioc.assert_choices_aligned(a, b)


def test_alignment_raises_for_value_mismatch():
    a = np.array([0, 1, 1, 0, 1])
    b = np.array([0, 1, 0, 0, 1])
    with pytest.raises(AssertionError):
        ioc.assert_choices_aligned(a, b)


def test_alignment_raises_for_length_mismatch():
    with pytest.raises(AssertionError):
        ioc.assert_choices_aligned(np.array([0, 1]), np.array([0, 1, 0]))


# ----------------------------------------------------------------------
# Integration: end-to-end on a mocked single animal
# ----------------------------------------------------------------------

def _mock_animal(n_trials=20, n_sessions=2, seed=0):
    """Build a struct that mirrors the simplify_cells layout of one entry of
    IOResults.animals."""
    rng = np.random.default_rng(seed)
    # Distribute trials evenly across sessions
    per_session = n_trials // n_sessions
    session_names = np.repeat([f's{i+1}' for i in range(n_sessions)], per_session)
    if len(session_names) < n_trials:
        session_names = np.concatenate([
            session_names,
            np.full(n_trials - len(session_names), session_names[-1]),
        ])

    Q = rng.dirichlet(np.ones(91), size=n_trials)
    L = rng.dirichlet(np.ones(91), size=n_trials)
    choices = rng.integers(0, 2, size=n_trials).astype(float)
    conf_vel = rng.normal(size=n_trials)
    cv_p_choice = rng.uniform(0.05, 0.95, size=n_trials)
    cv_vel = rng.normal(size=n_trials)

    return {
        'tag': 'MOCK',
        'data': {
            'choices': choices,
            'conf_vel': conf_vel,
            'orientation': rng.uniform(0, 90, size=n_trials),
            'contrast': rng.choice([0.25, 0.5, 1.0], size=n_trials),
            'dispersion': rng.choice([5, 30, 90], size=n_trials),
            'trial_keys': {
                'session_name': session_names,
                'trial_in_session': np.arange(n_trials),
            },
        },
        'fit': {
            'choice': {'alpha_r': 1.5, 'beta_r': 0.0,
                       'gamma_r': 0.05, 'delta_r': 0.05},
            'full_params': {'vel_slope': -2.0, 'vel_intercept': 0.0,
                            'vel_std': 0.5},
            'utility': dict(DEFAULT_UTILITY),
        },
        'pred': {
            'cv_p_choice_stage2': cv_p_choice,
            'cv_vel': cv_vel,
        },
        'inferred': {
            'post_s_marginal': Q,
            'L_s_marginal': L,
        },
    }


def test_compute_io_coherence_end_to_end_smoke():
    """Build a mock animal, mock decoder full_decoded distributions for two
    archs, and verify the driver returns a long-form DataFrame with
    expected columns and one row per (arch, target_type, trial)."""
    pd = pytest.importorskip('pandas')
    animal = _mock_animal(n_trials=20, n_sessions=2)

    mask = ioc.get_trials_to_keep_mask(animal['data']['trial_keys']['session_name'])
    n_kept = int(mask.sum())

    rng = np.random.default_rng(7)
    decoder_full_decoded = {
        'Q': {
            'spat': rng.dirichlet(np.ones(91), size=n_kept),
            'temp': rng.dirichlet(np.ones(91), size=n_kept),
        },
        'L': {
            'spat': rng.dirichlet(np.ones(91), size=n_kept),
            'temp': rng.dirichlet(np.ones(91), size=n_kept),
        },
        'd': {
            'spat': rng.dirichlet(np.ones(2),  size=n_kept),
            'temp': rng.dirichlet(np.ones(2),  size=n_kept),
        },
    }

    df = ioc.compute_io_coherence_for_animal(
        animal, mouse_id=0, split='strat',
        decoder_full_decoded_by_target=decoder_full_decoded,
        grid=GRID,
    )
    assert isinstance(df, pd.DataFrame)
    # 3 targets * 2 archs * n_kept trials
    assert len(df) == 3 * 2 * n_kept
    for col in ['mouse', 'arch', 'target', 'split', 'trial',
                'pred_pgo_lapse', 'pred_pgo_naive',
                'actual_choice', 'io_pred_pgo']:
        assert col in df.columns


# ----------------------------------------------------------------------
# v7.3 (HDF5) loader
# ----------------------------------------------------------------------

def _build_synthetic_v73_iofile(path, n_animals=2, n_trials=12, n_sessions=2):
    """Build a small HDF5 file that mimics MATLAB v7.3 IOResults.mat.

    Sets the MATLAB_class attributes that ``_h5_to_native`` keys on so the
    walker takes the right branches for cells, chars, and numerics.
    """
    h5py = pytest.importorskip('h5py')

    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        ior = f.create_group('IOResults')
        ior.attrs['MATLAB_class'] = np.bytes_('struct')

        # Build per-animal groups under '#refs#' (mimicking how MATLAB
        # stores cell-array contents) and put refs in IOResults/animals.
        refs_group = f.create_group('#refs#')
        animal_refs = []
        for ai in range(n_animals):
            grp = refs_group.create_group(f'animal_{ai}')
            grp.attrs['MATLAB_class'] = np.bytes_('struct')

            # Tag (string)
            tag = f'MOUSE_{ai}'
            tag_chars = np.array([ord(c) for c in tag], dtype=np.uint16).reshape(1, -1)
            ds = grp.create_dataset('tag', data=tag_chars)
            ds.attrs['MATLAB_class'] = np.bytes_('char')

            # data group
            data_grp = grp.create_group('data')
            data_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            choices = rng.integers(0, 2, size=n_trials).astype(np.float64)
            # MATLAB col-major would store as (n,1); HDF5 sees it as (1,n).
            ds = data_grp.create_dataset('choices', data=choices.reshape(1, -1))
            ds.attrs['MATLAB_class'] = np.bytes_('double')
            conf_vel = rng.normal(size=n_trials).astype(np.float64)
            ds = data_grp.create_dataset('conf_vel', data=conf_vel.reshape(1, -1))
            ds.attrs['MATLAB_class'] = np.bytes_('double')

            # trial_keys group — session_name is a cell array of strings
            tk = data_grp.create_group('trial_keys')
            tk.attrs['MATLAB_class'] = np.bytes_('struct')
            session_refs = []
            per_session = n_trials // n_sessions
            for ti in range(n_trials):
                sess_name = f's{ti // per_session + 1}'
                sn_chars = np.array([ord(c) for c in sess_name], dtype=np.uint16).reshape(1, -1)
                sds = refs_group.create_dataset(f'sn_a{ai}_t{ti}', data=sn_chars)
                sds.attrs['MATLAB_class'] = np.bytes_('char')
                session_refs.append(sds.ref)
            sn_arr = np.array(session_refs, dtype=h5py.ref_dtype).reshape(1, -1)
            sn_ds = tk.create_dataset('session_name', data=sn_arr)
            sn_ds.attrs['MATLAB_class'] = np.bytes_('cell')
            tis = np.arange(n_trials, dtype=np.float64).reshape(1, -1)
            ds = tk.create_dataset('trial_in_session', data=tis)
            ds.attrs['MATLAB_class'] = np.bytes_('double')

            # fit group
            fit_grp = grp.create_group('fit')
            fit_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            choice_grp = fit_grp.create_group('choice')
            choice_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            for k, v in [('alpha_r', 1.5), ('beta_r', 0.0),
                          ('gamma_r', 0.05), ('delta_r', 0.05)]:
                ds = choice_grp.create_dataset(k, data=np.array([[v]], dtype=np.float64))
                ds.attrs['MATLAB_class'] = np.bytes_('double')
            fp_grp = fit_grp.create_group('full_params')
            fp_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            for k, v in [('vel_slope', -2.0), ('vel_intercept', 0.0),
                          ('vel_std', 0.5)]:
                ds = fp_grp.create_dataset(k, data=np.array([[v]], dtype=np.float64))
                ds.attrs['MATLAB_class'] = np.bytes_('double')
            util_grp = fit_grp.create_group('utility')
            util_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            for k, v in [('R_hit', 1.0), ('R_miss', 0.0),
                          ('R_cr', 0.1), ('R_fa', -0.2)]:
                ds = util_grp.create_dataset(k, data=np.array([[v]], dtype=np.float64))
                ds.attrs['MATLAB_class'] = np.bytes_('double')

            # pred group
            pred_grp = grp.create_group('pred')
            pred_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            for k in ('cv_p_choice_stage2', 'cv_vel'):
                vals = rng.uniform(0.05, 0.95, size=n_trials).astype(np.float64)
                ds = pred_grp.create_dataset(k, data=vals.reshape(1, -1))
                ds.attrs['MATLAB_class'] = np.bytes_('double')

            # inferred group: post_s_marginal is (n_trials, 91) in MATLAB,
            # written as (91, n_trials) in HDF5 (col-major -> row-major flip)
            inf_grp = grp.create_group('inferred')
            inf_grp.attrs['MATLAB_class'] = np.bytes_('struct')
            psm = rng.dirichlet(np.ones(91), size=n_trials).astype(np.float64)
            ds = inf_grp.create_dataset('post_s_marginal', data=psm.T)  # (91, n_trials)
            ds.attrs['MATLAB_class'] = np.bytes_('double')
            lsm = rng.dirichlet(np.ones(91), size=n_trials).astype(np.float64)
            ds = inf_grp.create_dataset('L_s_marginal', data=lsm.T)
            ds.attrs['MATLAB_class'] = np.bytes_('double')

            animal_refs.append(grp.ref)

        # IOResults/animals: cell array of object refs to the animal groups
        animals_arr = np.array(animal_refs, dtype=h5py.ref_dtype).reshape(1, -1)
        ds = ior.create_dataset('animals', data=animals_arr)
        ds.attrs['MATLAB_class'] = np.bytes_('cell')


def test_v73_loader_returns_animal_list(tmp_path):
    """End-to-end: build a synthetic v7.3 IOResults file with two animals,
    call load_io_results, verify the per-animal dict layout matches what
    the rest of io_coherence expects."""
    pytest.importorskip('h5py')
    path = str(tmp_path / 'IOResults_synth.mat')
    _build_synthetic_v73_iofile(path, n_animals=2, n_trials=12, n_sessions=2)

    # Force the h5py path even though scipy could in principle read this
    # file (it can't — it's HDF5; sio raises NotImplementedError anyway).
    animals = ioc.load_io_results(path)
    assert isinstance(animals, list)
    assert len(animals) == 2
    a0 = animals[0]
    # Tag string round-trip
    assert a0['tag'] == 'MOUSE_0'
    # data sub-struct
    assert 'choices' in a0['data']
    assert a0['data']['choices'].shape == (12,)
    # cell-array of session names round-trips to a list of strings
    sess = a0['data']['trial_keys']['session_name']
    assert isinstance(sess, list)
    assert len(sess) == 12
    assert all(isinstance(s, str) for s in sess)
    # Stage 2 lapse params under fit.choice
    assert a0['fit']['choice']['alpha_r'] == pytest.approx(1.5)
    assert a0['fit']['choice']['gamma_r'] == pytest.approx(0.05)
    # Stage 1 vel emission under fit.full_params
    assert a0['fit']['full_params']['vel_slope'] == pytest.approx(-2.0)
    # Utility under fit.utility
    assert a0['fit']['utility']['R_hit'] == pytest.approx(1.0)
    # post_s_marginal must be (n_trials, 91), not (91, n_trials)
    assert a0['inferred']['post_s_marginal'].shape == (12, 91)


def test_v73_loader_then_compute_coherence(tmp_path):
    """Build a synthetic v7.3 file, build matching mock decoder full_decoded,
    and run the full coherence driver end to end. Catches alignment and
    type-coercion bugs in the loader path."""
    pytest.importorskip('h5py')
    pd = pytest.importorskip('pandas')
    path = str(tmp_path / 'IOResults_synth.mat')
    _build_synthetic_v73_iofile(path, n_animals=2, n_trials=12, n_sessions=2)
    animals = ioc.load_io_results(path)

    mask = ioc.get_trials_to_keep_mask(animals[0]['data']['trial_keys']['session_name'])
    n_kept = int(mask.sum())

    rng = np.random.default_rng(11)
    decoder = {
        'Q': {'spat': rng.dirichlet(np.ones(91), size=n_kept),
              'temp': rng.dirichlet(np.ones(91), size=n_kept)},
    }
    df = ioc.compute_io_coherence_for_animal(
        animals[0], mouse_id=0, split='strat',
        decoder_full_decoded_by_target=decoder, grid=GRID,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1 * 2 * n_kept
    assert df['actual_choice'].isin([0.0, 1.0]).all()


def test_compute_io_coherence_alignment_failure_surfaces():
    """If the decoder array length doesn't match the masked IO arrays, the
    driver must raise rather than silently truncating."""
    animal = _mock_animal(n_trials=20, n_sessions=2)
    mask = ioc.get_trials_to_keep_mask(animal['data']['trial_keys']['session_name'])
    n_kept = int(mask.sum())

    decoder_full_decoded = {
        'Q': {'spat': np.zeros((n_kept - 1, 91)),  # WRONG LENGTH
              'temp': np.zeros((n_kept, 91))},
    }
    with pytest.raises(AssertionError):
        ioc.compute_io_coherence_for_animal(
            animal, mouse_id=0, split='strat',
            decoder_full_decoded_by_target=decoder_full_decoded,
            grid=GRID,
        )
