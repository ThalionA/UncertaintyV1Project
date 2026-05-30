# -*- coding: utf-8 -*-
"""Tests for the v0.5 velocity emission channel (emissions / simulate / fit).

Headline: two behaviourally *similar* states (small choice bias gap) that the
choice-only HMM cannot separate become cleanly recoverable once a per-state
Gaussian velocity (engagement) marker is added. This is the concrete payoff of
multi-channel emissions characterised in scripts/velocity_emission_demo.py and
figures/README.md: extra trial-by-trial signal raises the ~1 bit/trial that
gates HMM state separability.
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


def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _conds():
    s_vals = np.array([30, 35, 40, 45, 50, 55, 60], float)
    return np.array([[s, c, 0.0] for s in s_vals for c in (0.5, 1.0)], float)


def _perm_agree(paths, z_list, K):
    return max(np.mean([(np.array(perm)[p] == z).mean()
                        for p, z in zip(paths, z_list)])
               for perm in itertools.permutations(range(K)))


# ---------------------------------------------------------------------------
# Trials container
# ---------------------------------------------------------------------------


def test_trials_velocity_optional_and_validated():
    base = dict(s_deg=np.zeros(4), c=np.ones(4), d=np.zeros(4),
                choice=np.array([0., 1., 0., 1.]))
    t0 = emissions_mod.Trials(**base)
    assert not t0.has_velocity
    t1 = emissions_mod.Trials(**base, velocity=np.arange(4.0))
    assert t1.has_velocity
    with pytest.raises(ValueError):
        emissions_mod.Trials(**base, velocity=np.arange(3.0))


# ---------------------------------------------------------------------------
# Velocity emission matrix
# ---------------------------------------------------------------------------


def test_log_velocity_matrix_matches_gaussian():
    grids = io_core.IOGrids.default()
    state_list = [states_mod.IOState('A', io_core.prior_bimodal(grids),
                                     PS(beta=1., gamma=0., delta=0.)),
                  states_mod.IOState('B', io_core.prior_bimodal(grids),
                                     PS(beta=1., gamma=0., delta=0.))]
    v = np.array([-1.0, 0.0, 2.5])
    vp = {'A': {'mu': 0.0, 'sigma': 1.0}, 'B': {'mu': 2.0, 'sigma': 0.5}}
    out = emissions_mod.log_velocity_matrix(v, state_list, vp)
    for k, name in enumerate(['A', 'B']):
        mu, sig = vp[name]['mu'], vp[name]['sigma']
        expected = (-0.5 * np.log(2 * np.pi * sig ** 2)
                    - 0.5 * ((v - mu) / sig) ** 2)
        assert np.allclose(out[:, k], expected)


def test_log_emission_matrix_adds_velocity_channel():
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    state_list = [states_mod.IOState('A', io_core.prior_bimodal(grids),
                                     PS(alpha=0., beta=2., gamma=0., delta=0.)),
                  states_mod.IOState('B', io_core.prior_flat(grids),
                                     PS(alpha=0., beta=1., gamma=0., delta=0.))]
    rng = np.random.default_rng(0)
    conds = _conds()
    idx = rng.integers(0, len(conds), 20)
    tr = emissions_mod.Trials(s_deg=conds[idx, 0], c=conds[idx, 1],
                              d=conds[idx, 2],
                              choice=rng.integers(0, 2, 20).astype(float),
                              velocity=rng.normal(size=20))
    vp = {'A': {'mu': 0.0, 'sigma': 1.0}, 'B': {'mu': 1.0, 'sigma': 2.0}}
    le_choice = emissions_mod.log_emission_matrix(
        grids, stage1, state_list, {}, tr)
    le_joint = emissions_mod.log_emission_matrix(
        grids, stage1, state_list, {}, tr, vel_per_state=vp)
    le_vel = emissions_mod.log_velocity_matrix(tr.velocity, state_list, vp)
    assert np.allclose(le_joint, le_choice + le_vel)


def test_log_emission_matrix_velocity_requires_velocity_field():
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    state_list = [states_mod.IOState('A', io_core.prior_bimodal(grids),
                                     PS(alpha=0., beta=1., gamma=0., delta=0.))]
    tr = emissions_mod.Trials(s_deg=np.array([45.]), c=np.array([1.]),
                              d=np.array([0.]), choice=np.array([1.]))
    with pytest.raises(ValueError):
        emissions_mod.log_emission_matrix(
            grids, stage1, state_list, {}, tr,
            vel_per_state={'A': {'mu': 0., 'sigma': 1.}})


# ---------------------------------------------------------------------------
# Velocity M-step (closed form)
# ---------------------------------------------------------------------------


def test_m_step_velocity_matches_weighted_moments():
    grids = io_core.IOGrids.default()
    state_list = [states_mod.IOState('A', io_core.prior_bimodal(grids), PS()),
                  states_mod.IOState('B', io_core.prior_bimodal(grids), PS())]
    gamma = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
    v = np.array([1.0, 5.0, 3.0, 2.0])
    out = fit_mod.m_step_velocity([gamma], [v], state_list)
    for k, name in enumerate(['A', 'B']):
        w = gamma[:, k]
        mu = (w * v).sum() / w.sum()
        sig = np.sqrt((w * (v - mu) ** 2).sum() / w.sum())
        assert out[name]['mu'] == pytest.approx(mu)
        assert out[name]['sigma'] == pytest.approx(sig)


def test_m_step_velocity_accumulates_across_sequences():
    grids = io_core.IOGrids.default()
    state_list = [states_mod.IOState('A', io_core.prior_bimodal(grids), PS())]
    g1 = np.array([[0.6], [0.4]])
    g2 = np.array([[0.5], [0.5]])
    v1 = np.array([2.0, 4.0])
    v2 = np.array([6.0, 0.0])
    out = fit_mod.m_step_velocity([g1, g2], [v1, v2], state_list)
    w = np.concatenate([g1[:, 0], g2[:, 0]])
    v = np.concatenate([v1, v2])
    mu = (w * v).sum() / w.sum()
    assert out['A']['mu'] == pytest.approx(mu)


def test_m_step_velocity_zero_mass_falls_back_to_global():
    grids = io_core.IOGrids.default()
    state_list = [states_mod.IOState('A', io_core.prior_bimodal(grids), PS()),
                  states_mod.IOState('B', io_core.prior_bimodal(grids), PS())]
    gamma = np.array([[1.0, 0.0], [1.0, 0.0]])   # state B gets no mass
    v = np.array([3.0, 5.0])
    out = fit_mod.m_step_velocity([gamma], [v], state_list)
    assert out['B']['mu'] == pytest.approx(v.mean())     # global fallback
    assert out['B']['sigma'] >= 1e-2


# ---------------------------------------------------------------------------
# EM with velocity: monotonicity + the headline recovery
# ---------------------------------------------------------------------------


def _two_state_setup(bias_gap, dprime, T, n_sessions, seed, sigma=1.0):
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    bimodal = io_core.prior_bimodal(grids)
    state_list = [states_mod.IOState('Lo', bimodal, PS(beta=1., gamma=0., delta=0.)),
                  states_mod.IOState('Hi', bimodal, PS(beta=1., gamma=0., delta=0.))]
    true_psych = {'Lo': {'alpha': -bias_gap / 2}, 'Hi': {'alpha': bias_gap / 2}}
    vel = {'Lo': {'mu': 0.0, 'sigma': sigma},
           'Hi': {'mu': dprime * sigma, 'sigma': sigma}}
    A = np.array([[0.92, 0.08], [0.08, 0.92]])
    pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(seed)
    conds = _conds()
    tl, zl = [], []
    for _ in range(n_sessions):
        tr, z = sim.simulate_sequence(state_list, stage1, grids, pi, A,
                                      true_psych, conds, T, rng,
                                      vel_per_state=vel)
        tl.append(tr); zl.append(z)
    return grids, stage1, state_list, tl, zl, A, vel


def test_em_with_velocity_is_monotonic():
    grids, stage1, sl, tl, zl, _, _ = _two_state_setup(
        bias_gap=1.0, dprime=3.0, T=400, n_sessions=2, seed=1)
    _, history = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                             n_restarts=1, max_iters=40, seed=0)
    diffs = np.diff(history)
    assert np.all(diffs >= -1e-6), f"velocity EM decreased: {diffs.min()}"


def test_velocity_rescues_separability_of_similar_states():
    """Two states with a small choice bias gap: choice-only Viterbi is near
    chance, but adding the velocity engagement marker recovers the latent path
    and the per-state velocity params."""
    grids, stage1, sl, tl, zl, A_true, vel_true = _two_state_setup(
        bias_gap=1.0, dprime=3.0, T=600, n_sessions=3, seed=0)

    pc, hc = fit_mod.fit(tl, sl, stage1, grids, n_restarts=3, max_iters=60, seed=1)
    agree_choice = _perm_agree(
        fit_mod.viterbi_paths(pc, tl, sl, stage1, grids), zl, 2)

    pv, hv = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                         n_restarts=3, max_iters=60, seed=1)
    agree_vel = _perm_agree(
        fit_mod.viterbi_paths(pv, tl, sl, stage1, grids), zl, 2)

    assert np.all(np.diff(hv) >= -1e-6)
    # choices alone cannot separate these states; velocity clearly can
    assert agree_choice < 0.70
    assert agree_vel > 0.90
    assert agree_vel > agree_choice + 0.2

    # fitted velocity markers recover (up to the label permutation EM picks)
    mus = sorted(v['mu'] for v in pv.vel_per_state.values())
    true_mus = sorted(v['mu'] for v in vel_true.values())
    assert np.allclose(mus, true_mus, atol=0.3)


def test_velocity_params_populated_only_when_requested():
    grids, stage1, sl, tl, zl, _, _ = _two_state_setup(
        bias_gap=2.0, dprime=2.0, T=300, n_sessions=2, seed=2)
    pc, _ = fit_mod.fit(tl, sl, stage1, grids, n_restarts=1, max_iters=20, seed=0)
    assert pc.vel_per_state == {}                       # choice-only: empty
    pv, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                        n_restarts=1, max_iters=20, seed=0)
    assert set(pv.vel_per_state) == {'Lo', 'Hi'}
    for v in pv.vel_per_state.values():
        assert np.isfinite(v['mu']) and v['sigma'] > 0


def test_fit_use_velocity_without_velocity_field_raises():
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    bimodal = io_core.prior_bimodal(grids)
    sl = [states_mod.IOState('Lo', bimodal, PS(beta=1., gamma=0., delta=0.)),
          states_mod.IOState('Hi', bimodal, PS(beta=1., gamma=0., delta=0.))]
    conds = _conds()
    rng = np.random.default_rng(0)
    tr, _ = sim.simulate_sequence(sl, stage1, grids, np.array([.5, .5]),
                                  np.array([[.9, .1], [.1, .9]]),
                                  {'Lo': {'alpha': -1.}, 'Hi': {'alpha': 1.}},
                                  conds, 50, rng)   # no velocity
    with pytest.raises(ValueError):
        fit_mod.fit(tr, sl, stage1, grids, use_velocity=True, n_restarts=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
