# -*- coding: utf-8 -*-
"""Tests for ``ideal_observer/io_hmm/emissions.py``.

Pin shape and sanity behaviour of the per-trial state-conditioned log
emission matrix, plus the load-bearing identifying property of each v0
state shape (Disengaged constant; Perfect monotonic in stimulus; etc.).
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
import emissions  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_psych_v0() -> dict:
    """A baseline psych dict that satisfies all v0 free-param requirements."""
    return {
        'Perfect': {'beta': 2.0},
        'Thirsty': {'alpha': 0.5, 'beta': 2.0},
        'Disengaged': {'alpha': 0.3},
        'Naive': {'beta': 2.0},
    }


def _default_stage1() -> emissions.Stage1Params:
    return emissions.Stage1Params(kappa_amp=8.0, c_power=1.5, d_power=2.0)


def _make_trials(s_deg, c, d, choice) -> emissions.Trials:
    return emissions.Trials(
        s_deg=np.asarray(s_deg, dtype=float),
        c=np.asarray(c, dtype=float),
        d=np.asarray(d, dtype=float),
        choice=np.asarray(choice, dtype=float),
    )


# ---------------------------------------------------------------------------
# Trials container
# ---------------------------------------------------------------------------


def test_trials_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="length"):
        emissions.Trials(
            s_deg=np.array([10.0, 20.0]),
            c=np.array([1.0]),  # wrong length
            d=np.array([0.0, 0.0]),
            choice=np.array([1.0, 0.0]),
        )


# ---------------------------------------------------------------------------
# unique_conditions grouping
# ---------------------------------------------------------------------------


def test_unique_conditions_dedupes_and_indexes():
    trials = _make_trials(
        s_deg=[10, 30, 10, 30, 60, 10],
        c=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        choice=[1, 1, 0, 1, 0, 1],
    )
    G_unique, G_idx = emissions.unique_conditions(trials)
    assert G_unique.shape == (3, 3)  # three distinct (s, c, d) rows
    # Every trial's (s, c, d) must equal the indexed unique row.
    cond_t = np.column_stack([trials.s_deg, trials.c, trials.d])
    assert np.allclose(cond_t, G_unique[G_idx])


# ---------------------------------------------------------------------------
# Emission matrix shape and finiteness
# ---------------------------------------------------------------------------


def test_emission_matrix_shape_and_finiteness():
    grids = io_core.IOGrids.default()
    stage1 = _default_stage1()
    trials = _make_trials(
        s_deg=[10, 30, 60, 80, 30, 60],
        c=[0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
        d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        choice=[1, 1, 0, 0, 1, 0],
    )
    state_list = states_mod.default_v0_states(grids)
    log_emiss = emissions.log_emission_matrix(grids, stage1, state_list,
                                               _default_psych_v0(), trials)
    assert log_emiss.shape == (6, 4)
    assert np.all(np.isfinite(log_emiss))
    # Bernoulli log-likelihood must be <= 0 always.
    assert np.all(log_emiss <= 0)


# ---------------------------------------------------------------------------
# Disengaged: stimulus-independent constant P(go)
# ---------------------------------------------------------------------------


def test_disengaged_emission_constant_across_stim():
    """Disengaged has beta=0 fixed, so P(go) = sigmoid(alpha) regardless
    of (s, c, d). The choice-conditioned log-emission therefore only
    depends on the choice value, not the trial features."""
    grids = io_core.IOGrids.default()
    stage1 = _default_stage1()
    trials = _make_trials(
        s_deg=[5.0, 40.0, 75.0],     # very different stim
        c=[0.3, 0.6, 1.0],           # different contrasts too
        d=[0.0, 0.2, 0.4],           # different dispersions too
        choice=[1, 1, 1],            # all go => log_emiss[:, k] = log P(go)
    )
    state_list = states_mod.default_v0_states(grids)
    log_emiss = emissions.log_emission_matrix(grids, stage1, state_list,
                                               _default_psych_v0(), trials)
    disengaged_idx = [i for i, s in enumerate(state_list)
                      if s.name == 'Disengaged'][0]
    col = log_emiss[:, disengaged_idx]
    assert np.allclose(col, col[0], atol=1e-12), \
        f"Disengaged emission varies across stim: {col}"
    # Value matches log(sigmoid(alpha)).
    alpha = _default_psych_v0()['Disengaged']['alpha']
    expected = np.log(1.0 / (1.0 + np.exp(-alpha)))
    assert np.isclose(col[0], expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Perfect / Thirsty / Naive: stimulus-driven
# ---------------------------------------------------------------------------


def test_perfect_high_kappa_predicts_go_for_go_stimulus():
    """With strong sensory + bimodal prior + high beta, Perfect should
    predict P(go) close to 1 for stim near 0 deg."""
    grids = io_core.IOGrids.default()
    stage1 = emissions.Stage1Params(kappa_amp=30.0, c_power=1.0, d_power=0.0)
    trials = _make_trials(s_deg=[5.0], c=[1.0], d=[0.0], choice=[1.0])
    state_list = states_mod.default_v0_states(grids)
    psych = _default_psych_v0()
    psych['Perfect'] = {'beta': 5.0}
    log_emiss = emissions.log_emission_matrix(grids, stage1, state_list,
                                               psych, trials)
    perfect_idx = [i for i, s in enumerate(state_list) if s.name == 'Perfect'][0]
    p_go = float(np.exp(log_emiss[0, perfect_idx]))
    assert p_go > 0.95, f"Perfect P(go|s=5) = {p_go} unexpectedly low"


def test_perfect_high_kappa_predicts_nogo_for_nogo_stimulus():
    grids = io_core.IOGrids.default()
    stage1 = emissions.Stage1Params(kappa_amp=30.0, c_power=1.0, d_power=0.0)
    trials = _make_trials(s_deg=[85.0], c=[1.0], d=[0.0], choice=[0.0])
    state_list = states_mod.default_v0_states(grids)
    psych = _default_psych_v0()
    psych['Perfect'] = {'beta': 5.0}
    log_emiss = emissions.log_emission_matrix(grids, stage1, state_list,
                                               psych, trials)
    perfect_idx = [i for i, s in enumerate(state_list) if s.name == 'Perfect'][0]
    p_nogo = float(np.exp(log_emiss[0, perfect_idx]))  # choice=0 => log(1-p_go)
    assert p_nogo > 0.95, f"Perfect P(nogo|s=85) = {p_nogo} unexpectedly low"


def test_thirsty_with_positive_alpha_pushes_pgo_above_perfect():
    """At a borderline orientation, Thirsty (positive alpha) gives higher
    P(go) than Perfect (alpha=0), holding everything else equal."""
    grids = io_core.IOGrids.default()
    stage1 = emissions.Stage1Params(kappa_amp=4.0, c_power=1.0, d_power=0.0)
    trials = _make_trials(s_deg=[45.0], c=[1.0], d=[0.0], choice=[1.0])
    state_list = states_mod.default_v0_states(grids)
    psych = {
        'Perfect': {'beta': 2.0},
        'Thirsty': {'alpha': 1.5, 'beta': 2.0},
        'Disengaged': {'alpha': 0.0},
        'Naive': {'beta': 2.0},
    }
    log_emiss = emissions.log_emission_matrix(grids, stage1, state_list,
                                               psych, trials)
    perfect_idx = [i for i, s in enumerate(state_list) if s.name == 'Perfect'][0]
    thirsty_idx = [i for i, s in enumerate(state_list) if s.name == 'Thirsty'][0]
    # choice=1 => log_emiss = log P(go); higher P(go) => higher log_emiss.
    assert log_emiss[0, thirsty_idx] > log_emiss[0, perfect_idx]


def test_perfect_and_naive_emissions_differ_on_intermediate_stim():
    """Perfect uses bimodal prior; Naive uses flat. At an intermediate
    orientation (e.g. 30 deg), the priors push g(m) differently and the
    emission probabilities differ. This is the load-bearing differentiation
    between the two states."""
    grids = io_core.IOGrids.default()
    stage1 = emissions.Stage1Params(kappa_amp=4.0, c_power=1.0, d_power=0.0)
    trials = _make_trials(
        s_deg=[20.0, 40.0, 60.0],
        c=[1.0, 1.0, 1.0],
        d=[0.0, 0.0, 0.0],
        choice=[1, 1, 0],
    )
    state_list = states_mod.default_v0_states(grids)
    psych = _default_psych_v0()
    log_emiss = emissions.log_emission_matrix(grids, stage1, state_list,
                                               psych, trials)
    perfect_idx = [i for i, s in enumerate(state_list) if s.name == 'Perfect'][0]
    naive_idx = [i for i, s in enumerate(state_list) if s.name == 'Naive'][0]
    diffs = log_emiss[:, perfect_idx] - log_emiss[:, naive_idx]
    # At least one trial must show a non-trivial emission gap between
    # Perfect and Naive (they share beta but differ in the IO prior).
    assert np.max(np.abs(diffs)) > 1e-3, \
        f"Perfect and Naive give identical emissions: max |diff| = {np.max(np.abs(diffs))}"


# ---------------------------------------------------------------------------
# Choice flip flips emission sign correctly
# ---------------------------------------------------------------------------


def test_emission_choice_flip_complements_log_p():
    """log P(go) and log P(nogo) on the same trial should obey
    exp(log P(go)) + exp(log P(nogo)) = 1 for a binary Bernoulli."""
    grids = io_core.IOGrids.default()
    stage1 = _default_stage1()
    trials_go = _make_trials(s_deg=[30.0], c=[0.5], d=[0.0], choice=[1.0])
    trials_nogo = _make_trials(s_deg=[30.0], c=[0.5], d=[0.0], choice=[0.0])
    state_list = states_mod.default_v0_states(grids)
    psych = _default_psych_v0()
    L_go = emissions.log_emission_matrix(grids, stage1, state_list, psych, trials_go)
    L_nogo = emissions.log_emission_matrix(grids, stage1, state_list, psych, trials_nogo)
    p_sum = np.exp(L_go[0]) + np.exp(L_nogo[0])
    assert np.allclose(p_sum, 1.0, atol=1e-10), \
        f"Bernoulli probs do not sum to 1: {p_sum}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
