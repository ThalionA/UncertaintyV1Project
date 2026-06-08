# -*- coding: utf-8 -*-
"""Tests for ``ideal_observer/io_hmm/fit.py`` (EM driver) and ``simulate.py``.

The headline tests are parameter recovery: simulate sequences from known
``pi`` / ``A`` / psychometric params and check EM recovers them. Because the
states used here are not exchangeable (distinct fixed priors / psych
patterns), recovery is compared directly by state name with no
label-switching.

Identifiability note
--------------------
There are two *distinct* identifiability constraints, quantified empirically by
``scripts/make_recovery_figures.py`` (see ``figures/README.md``):

1. Psychometric recovery *given the state label* (single-state, K=1). Bias
   ``alpha`` recovers tightly across +/-2 (sd <~ 0.08) and the lapse ``gamma``
   across 0-0.3 (sd ~ 0.01). The *slope* ``beta``, however, is only recovered
   up to ~1.5: the IO log-odds ``g`` already span ~+/-5, so the sigmoid is
   saturated for larger beta (beta=2 vs 3 give near-identical choices) and the
   gamma-weighted MLE undergoes logistic *separation* (diverges to the bound).
   This saturation wall -- not a bug -- is the concrete motivation for the
   planned v0.5 velocity emissions.

2. Latent-path / per-state attribution in the *HMM*. Choice-only emissions
   carry ~1 bit/trial, so Viterbi decoding stays at chance until the states'
   P(go) curves differ enough (perm-corrected agreement crosses chance around a
   bias gap of ~3). This separation requirement -- not any need for extreme
   parameters per se -- is why the recovery tests below use behaviourally
   well-separated states.

Because the states used here are not exchangeable (distinct fixed priors /
psych patterns) and are well separated, recovery is compared directly by state
name with no label-switching. The tests exercise EM on the rigorously
identifiable pieces (transition matrix, constant-rate bias);
``test_v0_four_state_fit_runs`` is a smoke test that the full v0 four-state
spec fits end-to-end (monotone, valid params) without asserting slope recovery.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IO_HMM = os.path.join(REPO_ROOT, 'ideal_observer', 'io_hmm')
if IO_HMM not in sys.path:
    sys.path.insert(0, IO_HMM)

import io_core  # noqa: E402
import states as states_mod  # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod  # noqa: E402
import simulate as sim  # noqa: E402

PS = states_mod.PsychSpec


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _stage1() -> emissions_mod.Stage1Params:
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _condition_grid() -> np.ndarray:
    """Boundary-spanning stimuli: intermediate P(go), informative for psych."""
    s_vals = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65], float)
    c_vals = np.array([0.5, 1.0])
    return np.array([[s, c, 0.0] for s in s_vals for c in c_vals], float)


def _sticky_A(K: int, p_self: float) -> np.ndarray:
    A = np.full((K, K), (1.0 - p_self) / (K - 1))
    np.fill_diagonal(A, p_self)
    return A


def _simulate(state_list, true_psych, true_A, T, n_sessions, seed):
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    conds = _condition_grid()
    K = len(state_list)
    true_pi = np.ones(K) / K
    rng = np.random.default_rng(seed)
    trials_list, z_true_list = [], []
    for _ in range(n_sessions):
        tr, z = sim.simulate_sequence(state_list, stage1, grids, true_pi,
                                      true_A, true_psych, conds, T, rng)
        trials_list.append(tr)
        z_true_list.append(z)
    return grids, stage1, trials_list, z_true_list


# ---------------------------------------------------------------------------
# Closed-form transition M-step
# ---------------------------------------------------------------------------


def test_m_step_transitions_matches_hand_computation():
    K = 2
    gamma = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    xi = np.array([
        [[0.75, 0.15], [0.05, 0.05]],   # t=0 -> t=1
        [[0.25, 0.55], [0.05, 0.15]],   # t=1 -> t=2
    ])
    pi, A = fit_mod.m_step_transitions([gamma], [xi], K)
    assert np.allclose(pi, gamma[0])             # pi is the first-step posterior
    num = xi.sum(axis=0)
    den = gamma[:-1].sum(axis=0)
    expected = num / den[:, None]
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(A, expected)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_m_step_transitions_zero_row_falls_back_to_uniform():
    K = 3
    gamma = np.array([[0.0, 0.5, 0.5], [0.0, 0.4, 0.6]])
    xi = np.zeros((1, K, K))
    xi[0, 1, 2] = 0.5
    xi[0, 2, 1] = 0.5
    _, A = fit_mod.m_step_transitions([gamma], [xi], K)
    assert np.allclose(A[0], 1.0 / K)            # state 0 had no mass -> uniform
    assert np.allclose(A.sum(axis=1), 1.0)


def test_m_step_transitions_accumulates_across_sequences():
    """Two sequences must give the same A as one concatenated sufficient stat."""
    K = 2
    g1 = np.array([[0.7, 0.3], [0.6, 0.4]])
    x1 = np.array([[[0.5, 0.2], [0.1, 0.2]]])
    g2 = np.array([[0.4, 0.6], [0.5, 0.5]])
    x2 = np.array([[[0.3, 0.1], [0.2, 0.4]]])
    _, A = fit_mod.m_step_transitions([g1, g2], [x1, x2], K)
    num = x1.sum(axis=0) + x2.sum(axis=0)
    den = g1[:-1].sum(axis=0) + g2[:-1].sum(axis=0)
    expected = num / den[:, None]
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(A, expected)


# ---------------------------------------------------------------------------
# Per-state psychometric M-step (isolated, exact under expected counts)
# ---------------------------------------------------------------------------


def test_psych_mstep_recovers_known_params_from_weighted_counts():
    """Expected gamma-weighted counts are the population MLE, so the per-state
    M-step recovers the generating (alpha, beta) for a free-alpha-and-beta
    state (Thirsty)."""
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    thirsty = [s for s in states_mod.default_v0_states(grids)
               if s.name == 'Thirsty'][0]
    conds = _condition_grid()
    true = {'alpha': 0.8, 'beta': 2.0}

    g_m, _dv_m, p_m = emissions_mod.precompute_state_terms(grids, stage1, thirsty, conds)
    full = thirsty.psych.resolve(true)
    p_go = emissions_mod.p_go_from_terms(g_m, p_m, full)
    N = 20000.0
    w_go, w_nogo = N * p_go, N * (1.0 - p_go)

    fitted = fit_mod._fit_psych_state(
        thirsty, {'alpha': 0.0, 'beta': 1.0}, g_m, p_m, w_go, w_nogo
    )
    assert abs(fitted['alpha'] - true['alpha']) < 0.1
    assert abs(fitted['beta'] - true['beta']) < 0.2


def test_psych_mstep_noop_for_state_without_free_params():
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    conds = _condition_grid()
    fixed = states_mod.IOState('Fixed', io_core.prior_bimodal(grids),
                               PS(alpha=0.0, beta=3.0, gamma=0.0, delta=0.0))
    g_m, _dv_m, p_m = emissions_mod.precompute_state_terms(grids, stage1, fixed, conds)
    out = fit_mod._fit_psych_state(fixed, {}, g_m, p_m,
                                   np.ones(len(conds)), np.ones(len(conds)))
    assert out == {}


# ---------------------------------------------------------------------------
# EM log-likelihood monotonicity
# ---------------------------------------------------------------------------


def test_em_loglik_monotonic_increasing():
    grids = io_core.IOGrids.default()
    state_list = states_mod.default_v0_states(grids)
    K = len(state_list)
    true_psych = {'Perfect': {'beta': 2.0}, 'Thirsty': {'alpha': 0.8, 'beta': 2.0},
                  'Disengaged': {'alpha': -0.4}, 'Naive': {'beta': 1.5}}
    grids, stage1, trials_list, _ = _simulate(
        state_list, true_psych, _sticky_A(K, 0.9), T=400, n_sessions=2, seed=3
    )
    init = fit_mod.random_init(state_list, np.random.default_rng(0))
    _, history = fit_mod.fit(trials_list, state_list, stage1, grids,
                             init=init, n_restarts=1, max_iters=40, seed=0)
    diffs = np.diff(history)
    assert np.all(diffs >= -1e-6), f"EM log-lik decreased: min diff {diffs.min()}"


# ---------------------------------------------------------------------------
# Headline 1: transition + latent-path recovery (distinct fixed-psych states)
# ---------------------------------------------------------------------------


def test_parameter_recovery_transitions_and_path():
    """Three behaviourally distinct states with fully-fixed psychometrics. EM
    fits only pi / A; the transition matrix and Viterbi path recover tightly."""
    grids = io_core.IOGrids.default()
    bimodal = io_core.prior_bimodal(grids)
    state_list = [
        states_mod.IOState('Sensory', bimodal, PS(alpha=0.0, beta=3.0, gamma=0.0, delta=0.0)),
        states_mod.IOState('GoBias', bimodal, PS(alpha=2.2, beta=0.0, gamma=0.0, delta=0.0)),
        states_mod.IOState('NoGoBias', bimodal, PS(alpha=-2.2, beta=0.0, gamma=0.0, delta=0.0)),
    ]
    K = len(state_list)
    true_A = _sticky_A(K, 0.92)
    true_psych = {'Sensory': {}, 'GoBias': {}, 'NoGoBias': {}}
    grids, stage1, trials_list, z_true_list = _simulate(
        state_list, true_psych, true_A, T=700, n_sessions=3, seed=0
    )
    params, history = fit_mod.fit(trials_list, state_list, stage1, grids,
                                  n_restarts=3, max_iters=60, seed=1)

    assert np.all(np.diff(history) >= -1e-6)
    assert np.abs(params.A - true_A).mean() < 0.04
    paths = fit_mod.viterbi_paths(params, trials_list, state_list, stage1, grids)
    agree = np.mean([(p == z).mean() for p, z in zip(paths, z_true_list)])
    assert agree > 0.80, f"Viterbi agreement only {agree:.3f}"


# ---------------------------------------------------------------------------
# Headline 2: joint recovery exercising the free-psych M-step inside EM
# ---------------------------------------------------------------------------


def test_parameter_recovery_with_free_psych():
    """Sensory (fixed psych) + Disengaged (free alpha = constant Bernoulli
    rate). EM jointly fits pi / A and the identifiable free bias param."""
    grids = io_core.IOGrids.default()
    bimodal = io_core.prior_bimodal(grids)
    state_list = [
        states_mod.IOState('Sensory', bimodal, PS(alpha=0.0, beta=3.0, gamma=0.0, delta=0.0)),
        states_mod.IOState('Disengaged', bimodal, PS(beta=0.0, gamma=0.0, delta=0.0)),
    ]
    K = len(state_list)
    true_A = _sticky_A(K, 0.92)
    true_alpha = -0.5
    true_psych = {'Sensory': {}, 'Disengaged': {'alpha': true_alpha}}
    grids, stage1, trials_list, z_true_list = _simulate(
        state_list, true_psych, true_A, T=700, n_sessions=3, seed=0
    )
    params, history = fit_mod.fit(trials_list, state_list, stage1, grids,
                                  n_restarts=3, max_iters=60, seed=1)

    assert np.all(np.diff(history) >= -1e-6)
    assert np.abs(params.A - true_A).mean() < 0.08
    assert abs(params.psych_per_state['Disengaged']['alpha'] - true_alpha) < 0.2
    paths = fit_mod.viterbi_paths(params, trials_list, state_list, stage1, grids)
    agree = np.mean([(p == z).mean() for p, z in zip(paths, z_true_list)])
    assert agree > 0.70, f"Viterbi agreement only {agree:.3f}"


# ---------------------------------------------------------------------------
# Smoke: the full v0 four-state spec fits end-to-end
# ---------------------------------------------------------------------------


def test_v0_four_state_fit_runs():
    """The real four-state spec fits without error: monotone LL, valid
    stochastic A, finite params. (Slope recovery is not asserted -- see the
    identifiability note in the module docstring.)"""
    grids = io_core.IOGrids.default()
    state_list = states_mod.default_v0_states(grids)
    K = len(state_list)
    true_psych = {'Perfect': {'beta': 2.0}, 'Thirsty': {'alpha': 0.8, 'beta': 2.0},
                  'Disengaged': {'alpha': -0.4}, 'Naive': {'beta': 1.5}}
    grids, stage1, trials_list, _ = _simulate(
        state_list, true_psych, _sticky_A(K, 0.9), T=300, n_sessions=2, seed=1
    )
    params, history = fit_mod.fit(trials_list, state_list, stage1, grids,
                                  n_restarts=2, max_iters=25, seed=2)
    assert np.all(np.diff(history) >= -1e-6)
    assert params.A.shape == (K, K)
    assert np.allclose(params.A.sum(axis=1), 1.0)
    assert np.all(params.A >= 0)
    assert np.isclose(params.pi.sum(), 1.0)
    for name, vals in params.psych_per_state.items():
        for v in vals.values():
            assert np.isfinite(v)


# ---------------------------------------------------------------------------
# simulate.py sanity
# ---------------------------------------------------------------------------


def test_simulate_shapes_and_state_range():
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    state_list = states_mod.default_v0_states(grids)
    K = len(state_list)
    conds = _condition_grid()
    psych = {'Perfect': {'beta': 2.0}, 'Thirsty': {'alpha': 0.5, 'beta': 2.0},
             'Disengaged': {'alpha': 0.0}, 'Naive': {'beta': 1.5}}
    rng = np.random.default_rng(0)
    trials, z = sim.simulate_sequence(state_list, stage1, grids, np.ones(K) / K,
                                      _sticky_A(K, 0.9), psych, conds, T=50, rng=rng)
    assert trials.n_trials == 50
    assert z.shape == (50,)
    assert set(np.unique(z)).issubset(set(range(K)))
    assert set(np.unique(trials.choice)).issubset({0.0, 1.0})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
