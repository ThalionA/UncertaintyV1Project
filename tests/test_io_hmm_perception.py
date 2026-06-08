# -*- coding: utf-8 -*-
"""Tests for Phase 2: free per-state perception (sensory precision ``lambda_``).

``lambda_`` multiplies the Stage-1 kappa for *both* the generative ``p(m|s)``
and the inference posterior, so it reshapes ``g(m)`` / ``DV(m)`` / ``p(m|s)``
together (sharper or blurrier perception) -- unlike the psychometric, which only
rescales an otherwise-fixed ``g(m)``. The headline is the perception-vs-action
dissociation: two states with the *same* choice behaviour but different
``lambda_`` are separated, and ``lambda_`` is recovered, because the velocity
channel reads ``DV(m)`` (perception) independently of the choice psychometric.
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
PC = states_mod.PerceptionSpec


def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _perm_agree(paths, z_list, K):
    return max(np.mean([(np.array(perm)[p] == z).mean()
                        for p, z in zip(paths, z_list)])
               for perm in itertools.permutations(range(K)))


# ---------------------------------------------------------------------------
# PerceptionSpec
# ---------------------------------------------------------------------------


def test_perception_spec_default_is_frozen_lambda_one():
    spec = PC()  # default
    assert spec.is_frozen and spec.free_params == ()
    assert spec.resolve({})['lambda_'] == 1.0


def test_perception_spec_free_and_resolve():
    spec = PC(lambda_=None)
    assert not spec.is_frozen and spec.free_params == ('lambda_',)
    assert spec.resolve({'lambda_': 2.3})['lambda_'] == 2.3
    with pytest.raises(KeyError):
        spec.resolve({})


def test_iostate_defaults_to_frozen_perception():
    grids = io_core.IOGrids.default()
    st = states_mod.IOState('A', io_core.prior_bimodal(grids), PS())
    assert st.perc is None and st.perception.is_frozen
    assert st.perception.resolve({})['lambda_'] == 1.0


# ---------------------------------------------------------------------------
# lambda_ reshapes the IO terms (no fit)
# ---------------------------------------------------------------------------


def test_lambda_sharpens_g_and_dv():
    """Higher lambda_ (sharper sensory evidence) makes g(m) and DV(m) more
    extreme; lower lambda_ flattens them."""
    grids = io_core.IOGrids.default()
    st = states_mod.IOState('A', io_core.prior_bimodal(grids), PS())
    conds = np.array([[s, 1.0, 0.0] for s in [20, 35, 55, 70]], float)
    g_lo, dv_lo, _ = emissions_mod.precompute_state_terms(
        grids, _stage1(), st, conds, kappa_scale=0.5)
    g_hi, dv_hi, _ = emissions_mod.precompute_state_terms(
        grids, _stage1(), st, conds, kappa_scale=2.0)
    assert np.abs(g_hi).max() > np.abs(g_lo).max()
    assert (dv_hi.max() - dv_hi.min()) > (dv_lo.max() - dv_lo.min())


def test_vectorized_precompute_matches_io_core_loop():
    """The vectorised precompute is numerically identical to the per-condition
    io_core primitives it replaces (across sensory gains)."""
    grids = io_core.IOGrids.default()
    st = states_mod.IOState('A', io_core.prior_bimodal(grids), PS())
    conds = np.array([[s, c, d] for s in [15, 45, 75]
                      for c in (0.5, 1.0) for d in (0.0, 0.5)], float)
    for scale in (1.0, 2.0, 0.5, 5.0):
        g, dv, pm = emissions_mod.precompute_state_terms(
            grids, _stage1(), st, conds, kappa_scale=scale)
        for j, (s_j, c_j, d_j) in enumerate(conds):
            kap = scale * float(io_core.kappa_for_trial(
                np.array([s_j]), np.array([c_j]), np.array([d_j]),
                kappa_amp=8., c_power=1., d_power=1., kappa_min=1.)[0])
            post = io_core.posterior_s_given_m(grids, kappa=kap, prior=st.prior)
            assert np.allclose(g[j], io_core.log_posterior_odds(grids, post))
            assert np.allclose(dv[j], io_core.decision_variable(grids, post))
            assert np.allclose(pm[j], io_core.p_m_given_s(grids, kappa=kap, s_deg=float(s_j)))


# ---------------------------------------------------------------------------
# Fit plumbing: perc_per_state populated only for free perception
# ---------------------------------------------------------------------------


def _two_states(free_perc):
    grids = io_core.IOGrids.default()
    bim = io_core.prior_bimodal(grids)
    perc = PC(lambda_=None) if free_perc else None
    return grids, [
        states_mod.IOState('Sharp', bim, PS(alpha=0., gamma=0., delta=0.),
                           vel=VS(), perc=perc),
        states_mod.IOState('Blur', bim, PS(alpha=0., gamma=0., delta=0.),
                           vel=VS(), perc=perc),
    ]


def _sim_dissociation(grids, sl, T=350, n_sess=3, seed=0):
    """Same action (beta, beta_vel); different perception (lambda) and a
    velocity baseline offset that makes the two states separable."""
    stage1 = _stage1()
    tp = {'Sharp': {'beta': 2.0}, 'Blur': {'beta': 2.0}}
    velp = {'Sharp': {'beta_vel': 2.0, 'alpha_vel': 0.0, 'sigma_vel': 0.5},
            'Blur': {'beta_vel': 2.0, 'alpha_vel': 1.5, 'sigma_vel': 0.5}}
    percp = {'Sharp': {'lambda_': 2.0}, 'Blur': {'lambda_': 0.5}}
    A = np.array([[0.93, 0.07], [0.07, 0.93]]); pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(seed)
    conds = np.array([[s, 1.0, 0.0] for s in [25, 40, 55, 70]], float)
    tl, zl = [], []
    for _ in range(n_sess):
        tr, z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds, T,
                                      rng, vel_per_state=velp, perc_per_state=percp)
        tl.append(tr); zl.append(z)
    return stage1, tl, zl


def test_perc_per_state_empty_when_perception_frozen():
    grids, sl = _two_states(free_perc=False)
    stage1, tl, _ = _sim_dissociation(grids, sl, T=250, n_sess=2)
    params, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                            n_restarts=1, max_iters=12, seed=0)
    assert params.perc_per_state == {} or all(not v for v in params.perc_per_state.values())


# ---------------------------------------------------------------------------
# Headline: free lambda_ recovers and perception dissociates from action
# ---------------------------------------------------------------------------


def test_free_lambda_recovers_and_dissociates_from_action():
    grids, sl = _two_states(free_perc=True)
    stage1, tl, zl = _sim_dissociation(grids, sl, T=350, n_sess=3, seed=0)
    params, history = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                                  n_restarts=2, max_iters=30, seed=1)
    assert np.all(np.diff(history) >= -1e-6)                  # EM monotone
    agree = _perm_agree(fit_mod.viterbi_paths(params, tl, sl, stage1, grids), zl, 2)
    assert agree > 0.85                                       # states separated

    lams = {n: v['lambda_'] for n, v in params.perc_per_state.items()}
    sharp = max(lams, key=lambda n: lams[n])
    blur = min(lams, key=lambda n: lams[n])
    # perception recovered: one state clearly sharper than the other, in range
    assert lams[sharp] > 1.3 and lams[blur] < 0.8
    assert lams[sharp] / lams[blur] > 2.0
    # action (velocity gain) recovered for both despite the perceptual difference
    for n in (sharp, blur):
        assert abs(params.vel_per_state[n]['beta_vel'] - 2.0) < 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
