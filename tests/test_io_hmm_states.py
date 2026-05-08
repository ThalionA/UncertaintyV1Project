# -*- coding: utf-8 -*-
"""Tests for ``ideal_observer/io_hmm/states.py``.

Pin the v0 four-state spec (free param patterns) and the 4-param
psychometric's behaviour in the limits we rely on (boundary, asymptotes,
lapse).
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
import states  # noqa: E402


# ---------------------------------------------------------------------------
# PsychSpec
# ---------------------------------------------------------------------------


def test_psych_spec_perfect_has_one_free_param_beta():
    spec = states.PsychSpec(alpha=0.0, gamma=0.0, delta=0.0)
    assert spec.n_free == 1
    assert spec.free_params == ('beta',)


def test_psych_spec_thirsty_has_two_free_alpha_beta():
    spec = states.PsychSpec(gamma=0.0, delta=0.0)
    assert spec.free_params == ('alpha', 'beta')


def test_psych_spec_disengaged_has_one_free_alpha():
    spec = states.PsychSpec(beta=0.0, gamma=0.0, delta=0.0)
    assert spec.free_params == ('alpha',)


def test_psych_spec_resolve_combines_fixed_and_free():
    spec = states.PsychSpec(alpha=0.0, gamma=0.0, delta=0.0)
    out = spec.resolve({'beta': 2.5})
    assert out == {'alpha': 0.0, 'beta': 2.5, 'gamma': 0.0, 'delta': 0.0}


def test_psych_spec_resolve_raises_on_missing_free():
    spec = states.PsychSpec(gamma=0.0, delta=0.0)  # alpha and beta free
    with pytest.raises(KeyError, match="alpha"):
        spec.resolve({'beta': 1.0})  # missing alpha


def test_psych_spec_resolve_ignores_extra_keys():
    spec = states.PsychSpec(alpha=0.0, gamma=0.0, delta=0.0)
    out = spec.resolve({'beta': 1.0, 'unrelated': 99.0})
    assert out['beta'] == 1.0


# ---------------------------------------------------------------------------
# 4-param psychometric
# ---------------------------------------------------------------------------


def test_psychometric_at_g_zero_no_lapse_is_half():
    g = np.array([0.0])
    val = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
    assert np.isclose(val[0], 0.5)


def test_psychometric_at_g_zero_with_lapse_is_lapse_plus_half_span():
    g = np.array([0.0])
    val = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.1, delta=0.05)
    expected = 0.1 + (1.0 - 0.1 - 0.05) * 0.5
    assert np.isclose(val[0], expected)


def test_psychometric_high_g_no_lapse_approaches_one():
    g = np.array([20.0])
    val = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
    assert val[0] > 0.999


def test_psychometric_low_g_no_lapse_approaches_zero():
    g = np.array([-20.0])
    val = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
    assert val[0] < 0.001


def test_psychometric_high_g_with_upper_lapse_caps_at_one_minus_delta():
    g = np.array([20.0])
    val = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.0, delta=0.1)
    assert np.isclose(val[0], 0.9, atol=1e-3)


def test_psychometric_low_g_with_lower_lapse_floors_at_gamma():
    g = np.array([-20.0])
    val = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.15, delta=0.0)
    assert np.isclose(val[0], 0.15, atol=1e-3)


def test_psychometric_beta_zero_yields_constant_independent_of_g():
    g = np.linspace(-5, 5, 11)
    val = states.psychometric(g, alpha=0.7, beta=0.0, gamma=0.0, delta=0.0)
    assert np.allclose(val, val[0])
    expected = 1.0 / (1.0 + np.exp(-0.7))
    assert np.isclose(val[0], expected)


def test_psychometric_alpha_shifts_curve():
    """Adding alpha shifts the inflection point of the psychometric in g."""
    g = np.array([0.0])
    base = states.psychometric(g, alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
    pos = states.psychometric(g, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0)
    assert pos[0] > base[0]  # positive bias raises P(go) at g=0


# ---------------------------------------------------------------------------
# v0 four-state factory
# ---------------------------------------------------------------------------


def test_default_v0_states_correct_count_and_names():
    grids = io_core.IOGrids.default()
    sts = states.default_v0_states(grids)
    assert len(sts) == 4
    assert [s.name for s in sts] == ['Perfect', 'Thirsty', 'Disengaged', 'Naive']


def test_default_v0_states_have_expected_free_params():
    grids = io_core.IOGrids.default()
    sts = states.default_v0_states(grids)
    expected = {
        'Perfect': ('beta',),
        'Thirsty': ('alpha', 'beta'),
        'Disengaged': ('alpha',),
        'Naive': ('beta',),
    }
    for s in sts:
        assert s.psych.free_params == expected[s.name], \
            f"{s.name}: got {s.psych.free_params}, expected {expected[s.name]}"


def test_default_v0_states_priors_normalised_and_distinct():
    grids = io_core.IOGrids.default()
    sts = states.default_v0_states(grids)
    for s in sts:
        assert np.isclose(s.prior.sum(), 1.0, atol=1e-8), s.name
    # Naive's prior is flat; the other three share the bimodal prior.
    perfect = next(s for s in sts if s.name == 'Perfect')
    thirsty = next(s for s in sts if s.name == 'Thirsty')
    disengaged = next(s for s in sts if s.name == 'Disengaged')
    naive = next(s for s in sts if s.name == 'Naive')
    assert np.allclose(perfect.prior, thirsty.prior)
    assert np.allclose(perfect.prior, disengaged.prior)
    assert not np.allclose(perfect.prior, naive.prior)
    # Naive prior is flat (uniform).
    assert np.allclose(naive.prior, naive.prior[0])


def test_iostate_rejects_unnormalised_prior():
    grids = io_core.IOGrids.default()
    bad = np.ones_like(grids.s_grid_deg, dtype=float)  # sums to n_s, not 1
    with pytest.raises(ValueError, match='not normalised'):
        states.IOState(name='Bad', prior=bad,
                       psych=states.PsychSpec(alpha=0.0, gamma=0.0, delta=0.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
