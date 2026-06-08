# -*- coding: utf-8 -*-
"""Tests for the confidence-velocity emission channel (paper Eqs. 9-10).

Velocity is a graded readout of the decision variable: ``v ~ N(beta_vel*DV(m) +
alpha_vel, sigma_vel)``, marginalised over the latent measurement ``m`` jointly
with choice (so velocity updates choice via Bayes on ``m``). ``beta_vel`` is the
per-state *confidence gain*; ``beta_vel = 0`` recovers a stimulus-independent
baseline marker (the joint emission then factorises into choice x Gaussian).

These cover: the VelocitySpec; the joint emission (choice-only reduction,
beta_vel=0 additivity, and the velocity->choice Bayes coupling); EM
monotonicity; and parameter recovery of beta_vel/alpha_vel when states are
separable.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IO_HMM = os.path.join(REPO_ROOT, 'ideal_observer', 'io_hmm')
if IO_HMM not in sys.path:
    sys.path.insert(0, IO_HMM)

import io_core            # noqa: E402
import states as states_mod   # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod     # noqa: E402
import simulate as sim    # noqa: E402

PS = states_mod.PsychSpec
VS = states_mod.VelocitySpec


def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _conds():
    s_vals = np.array([20, 35, 45, 55, 70], float)
    return np.array([[s, c, 0.0] for s in s_vals for c in (0.5, 1.0)], float)


def _perm_agree(paths, z_list, K):
    return max(np.mean([(np.array(perm)[p] == z).mean()
                        for p, z in zip(paths, z_list)])
               for perm in itertools.permutations(range(K)))


# ---------------------------------------------------------------------------
# Trials velocity field
# ---------------------------------------------------------------------------


def test_trials_velocity_optional_and_validated():
    base = dict(s_deg=np.zeros(4), c=np.ones(4), d=np.zeros(4),
                choice=np.array([0., 1., 0., 1.]))
    assert not emissions_mod.Trials(**base).has_velocity
    assert emissions_mod.Trials(**base, velocity=np.arange(4.0)).has_velocity
    with pytest.raises(ValueError):
        emissions_mod.Trials(**base, velocity=np.arange(3.0))


# ---------------------------------------------------------------------------
# VelocitySpec
# ---------------------------------------------------------------------------


def test_velocity_spec_free_and_resolve():
    spec = VS()  # all free
    assert spec.free_params == ('beta_vel', 'alpha_vel', 'sigma_vel')
    full = spec.resolve({'beta_vel': 1.0, 'alpha_vel': 0.0, 'sigma_vel': 0.5})
    assert full == {'beta_vel': 1.0, 'alpha_vel': 0.0, 'sigma_vel': 0.5}
    decoupled = VS(beta_vel=0.0)
    assert decoupled.free_params == ('alpha_vel', 'sigma_vel')
    assert decoupled.resolve({'alpha_vel': 1.0, 'sigma_vel': 2.0})['beta_vel'] == 0.0
    with pytest.raises(KeyError):
        spec.resolve({'beta_vel': 1.0})  # missing alpha_vel / sigma_vel


# ---------------------------------------------------------------------------
# Joint emission
# ---------------------------------------------------------------------------


def _terms(state, n=30, seed=0):
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    conds = _conds()
    g_m, dv_m, p_m = emissions_mod.precompute_state_terms(grids, stage1, state, conds)
    rng = np.random.default_rng(seed)
    gidx = rng.integers(0, len(conds), n)
    choice = rng.integers(0, 2, n).astype(float)
    velocity = rng.normal(size=n)
    return g_m, dv_m, p_m, gidx, choice, velocity


def test_joint_emission_choice_only_matches_marginal_bernoulli():
    state = states_mod.IOState('A', io_core.prior_bimodal(io_core.IOGrids.default()),
                               PS(alpha=0., beta=2., gamma=0., delta=0.))
    g_m, dv_m, p_m, gidx, choice, _ = _terms(state)
    full = state.psych.resolve({})
    le = emissions_mod.joint_log_emission_state(g_m, dv_m, p_m, gidx, choice, full)
    # manual: P(go|cond) = sum_m psi(g(m)) p(m|cond); Bernoulli on choice
    psi = states_mod.psychometric(g_m[gidx], **full)
    p_go = np.clip((psi * p_m[gidx]).sum(1), emissions_mod.EPS, 1 - emissions_mod.EPS)
    manual = choice * np.log(p_go) + (1 - choice) * np.log(1 - p_go)
    assert np.allclose(le, manual)


def test_joint_emission_decoupled_is_additive():
    """beta_vel=0: velocity is N(alpha_vel, sigma_vel) independent of choice, so
    the joint log-emission is the choice term plus a Gaussian term."""
    state = states_mod.IOState('A', io_core.prior_bimodal(io_core.IOGrids.default()),
                               PS(alpha=0., beta=2., gamma=0., delta=0.),
                               vel=VS(beta_vel=0.0))
    g_m, dv_m, p_m, gidx, choice, v = _terms(state)
    full = state.psych.resolve({})
    vel_full = state.vel.resolve({'alpha_vel': 0.5, 'sigma_vel': 1.2})
    le_joint = emissions_mod.joint_log_emission_state(
        g_m, dv_m, p_m, gidx, choice, full, velocity=v, vel_full=vel_full)
    le_choice = emissions_mod.joint_log_emission_state(g_m, dv_m, p_m, gidx, choice, full)
    sig = 1.2
    log_gauss = -0.5 * np.log(2 * np.pi * sig * sig) - 0.5 * ((v - 0.5) / sig) ** 2
    assert np.allclose(le_joint, le_choice + log_gauss, atol=1e-8)


def test_velocity_updates_choice_via_bayes():
    """With beta_vel>0, a faster velocity (evidence for high DV / Go) raises the
    implied P(go|v). P(go|v) = sigmoid(le[choice=1] - le[choice=0])."""
    grids = io_core.IOGrids.default()
    state = states_mod.IOState('A', io_core.prior_bimodal(grids),
                               PS(alpha=0., beta=4., gamma=0., delta=0.), vel=VS())
    g_m, dv_m, p_m = emissions_mod.precompute_state_terms(
        grids, _stage1(), state, np.array([[45.0, 1.0, 0.0]]))  # ambiguous stimulus
    vel_full = {'beta_vel': 3.0, 'alpha_vel': 0.0, 'sigma_vel': 0.5}
    full = state.psych.resolve({})
    vs = np.linspace(-1.5, 1.5, 13)
    gidx = np.zeros(len(vs), dtype=int)
    le1 = emissions_mod.joint_log_emission_state(
        g_m, dv_m, p_m, gidx, np.ones(len(vs)), full, velocity=vs, vel_full=vel_full)
    le0 = emissions_mod.joint_log_emission_state(
        g_m, dv_m, p_m, gidx, np.zeros(len(vs)), full, velocity=vs, vel_full=vel_full)
    p_go_given_v = 1.0 / (1.0 + np.exp(-(le1 - le0)))
    assert np.all(np.diff(p_go_given_v) > 0), "P(go|v) should rise with velocity"


# ---------------------------------------------------------------------------
# EM with the confidence channel: monotonicity + recovery
# ---------------------------------------------------------------------------


def _engaged_disengaged(beta_vel_eng=2.5, dis_alpha=1.5, T=500, n_sess=3, seed=0):
    """Engaged (velocity tracks confidence) vs Disengaged (decoupled, raised
    baseline, shallow psychometric) -- separable, so params are recoverable."""
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    bim = io_core.prior_bimodal(grids)
    sl = [states_mod.IOState('Eng', bim, PS(alpha=0., gamma=0., delta=0.), vel=VS()),
          states_mod.IOState('Dis', bim, PS(alpha=0., gamma=0., delta=0.), vel=VS(beta_vel=0.))]
    tp = {'Eng': {'beta': 3.0}, 'Dis': {'beta': 0.3}}
    vel = {'Eng': {'beta_vel': beta_vel_eng, 'alpha_vel': 0.0, 'sigma_vel': 0.5},
           'Dis': {'alpha_vel': dis_alpha, 'sigma_vel': 0.5}}
    A = np.array([[0.93, 0.07], [0.07, 0.93]]); pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(seed)
    conds = _conds()
    tl, zl = [], []
    for _ in range(n_sess):
        tr, z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds, T,
                                      rng, vel_per_state=vel)
        tl.append(tr); zl.append(z)
    return grids, stage1, sl, tl, zl


def test_em_confidence_velocity_is_monotonic():
    grids, stage1, sl, tl, zl = _engaged_disengaged(T=400, n_sess=2)
    _, history = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                             n_restarts=1, max_iters=30, seed=0)
    assert np.all(np.diff(history) >= -1e-6), f"decreased: {np.diff(history).min()}"


def test_confidence_coupling_beta_vel_recovered():
    """The headline: beta_vel (confidence gain) is read out per state, and the
    engaged/disengaged states separate."""
    grids, stage1, sl, tl, zl = _engaged_disengaged(beta_vel_eng=2.5, seed=0)
    params, history = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                                  n_restarts=3, max_iters=60, seed=1)
    assert np.all(np.diff(history) >= -1e-6)
    agree = _perm_agree(fit_mod.viterbi_paths(params, tl, sl, stage1, grids), zl, 2)
    assert agree > 0.85

    # the engaged state's confidence gain recovers; the decoupled state has none.
    betas = {n: v.get('beta_vel', 0.0) for n, v in params.vel_per_state.items()}
    eng = max(betas, key=lambda n: betas[n])      # the state that fit a coupling
    assert betas[eng] > 1.5                        # ~2.5 true, generous tolerance
    # the other state carries no free beta_vel (decoupled spec)
    other = [n for n in betas if n != eng][0]
    assert 'beta_vel' not in params.vel_per_state[other]


def test_decoupled_baseline_marker_recovered():
    """beta_vel=0 for both states reduces to the engagement-marker model: the
    baselines (alpha_vel) must recover and the states separate cleanly."""
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    bim = io_core.prior_bimodal(grids)
    sl = [states_mod.IOState('Lo', bim, PS(alpha=-0.5, gamma=0., delta=0.), vel=VS(beta_vel=0.)),
          states_mod.IOState('Hi', bim, PS(alpha=0.5, gamma=0., delta=0.), vel=VS(beta_vel=0.))]
    tp = {'Lo': {'beta': 1.0}, 'Hi': {'beta': 1.0}}
    vel = {'Lo': {'alpha_vel': 0.0, 'sigma_vel': 1.0},
           'Hi': {'alpha_vel': 3.0, 'sigma_vel': 1.0}}
    A = np.array([[0.92, 0.08], [0.08, 0.92]]); pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(0)
    conds = _conds()
    tl, zl = [], []
    for _ in range(3):
        tr, z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds, 600,
                                      rng, vel_per_state=vel)
        tl.append(tr); zl.append(z)
    params, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                            n_restarts=3, max_iters=60, seed=1)
    agree = _perm_agree(fit_mod.viterbi_paths(params, tl, sl, stage1, grids), zl, 2)
    assert agree > 0.9
    alphas = sorted(v['alpha_vel'] for v in params.vel_per_state.values())
    assert np.allclose(alphas, [0.0, 3.0], atol=0.4)


def test_velocity_params_only_when_requested():
    grids, stage1, sl, tl, zl = _engaged_disengaged(T=300, n_sess=2, seed=2)
    pc, _ = fit_mod.fit(tl, sl, stage1, grids, n_restarts=1, max_iters=15, seed=0)
    assert pc.vel_per_state == {}                       # choice-only: empty
    pv, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                        n_restarts=1, max_iters=15, seed=0)
    assert set(pv.vel_per_state) == {'Eng', 'Dis'}
    # Eng fits all three; Dis fits only the two free ones (beta_vel fixed 0)
    assert set(pv.vel_per_state['Eng']) == {'beta_vel', 'alpha_vel', 'sigma_vel'}
    assert set(pv.vel_per_state['Dis']) == {'alpha_vel', 'sigma_vel'}


def test_fit_use_velocity_without_velocity_field_raises():
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    sl = states_mod.default_v0_states(grids, with_velocity=True)
    conds = _conds()
    rng = np.random.default_rng(0)
    tr, _ = sim.simulate_sequence(  # choice-only trials (no velocity)
        sl, stage1, grids, np.full(4, 0.25), 0.9 * np.eye(4) + 0.1 / 4,
        {'Perfect': {'beta': 1.}, 'Thirsty': {'alpha': 0., 'beta': 1.},
         'Disengaged': {'alpha': 0.}, 'Naive': {'beta': 1.}}, conds, 50, rng)
    with pytest.raises(ValueError):
        fit_mod.fit(tr, sl, stage1, grids, use_velocity=True, n_restarts=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
