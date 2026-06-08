# -*- coding: utf-8 -*-
"""Tests for ``ideal_observer/io_hmm/io_core.py``.

Focus: numerical correctness of the pure IO computation (von Mises with
doubled-angle convention, kappa schedule, posterior, g(m)).
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


# ---------------------------------------------------------------------------
# von Mises pdf
# ---------------------------------------------------------------------------


def test_vonmises_kappa_zero_is_uniform():
    """With kappa=0 the von Mises is uniform on the (doubled) circle."""
    grids = io_core.IOGrids.default()
    p = io_core.vonmises_pdf_doubled(grids.m_grid_rad, np.deg2rad(45.0), 0.0)
    assert np.allclose(p, p[0]), "kappa=0 must yield a constant density"
    # Uniform density on doubled circle is 1 / (2*pi).
    assert np.isclose(p[0], 1.0 / (2 * np.pi))


def test_vonmises_period_180deg():
    """Doubled-angle: orientations differing by 180 deg give the same density."""
    grids = io_core.IOGrids.default()
    mu = np.deg2rad(30.0)
    p_a = io_core.vonmises_pdf_doubled(np.deg2rad(50.0), mu, kappa=2.0)
    p_b = io_core.vonmises_pdf_doubled(np.deg2rad(50.0 + 180.0), mu, kappa=2.0)
    assert np.isclose(p_a, p_b)


def test_vonmises_integrates_to_half_on_orientation_grid():
    """Raw doubled-angle vM density integrates to 0.5 over [0, 180 deg].

    The pdf is normalised on the full doubled circle [0, 360 deg] which
    integrates to 1; the orientation grid [0, 180 deg] = [0, pi rad] is
    exactly half of that, so the raw integral is 0.5. Production code uses
    ``p_m_given_s`` which renormalises the discrete sum to 1
    (covered by a separate test). This test pins the underlying math.
    """
    grids = io_core.IOGrids.default()
    dx_rad = np.deg2rad(grids.m_grid_deg[1] - grids.m_grid_deg[0])
    for s_deg in [10.0, 45.0, 80.0]:
        p = io_core.vonmises_pdf_doubled(grids.m_grid_rad, np.deg2rad(s_deg), kappa=4.0)
        integral = np.trapezoid(p, dx=dx_rad)
        assert 0.48 < integral < 0.52, f"integral {integral} for s={s_deg}"


# ---------------------------------------------------------------------------
# Sensory precision schedule
# ---------------------------------------------------------------------------


def test_kappa_for_trial_independent_of_orientation():
    """v2's IO is isotropic: kappa shouldn't depend on s."""
    k1 = io_core.kappa_for_trial(np.array([10.0]), c=np.array([0.5]),
                                 d=np.array([0.2]), kappa_amp=4.0,
                                 c_power=1.5, d_power=2.0)
    k2 = io_core.kappa_for_trial(np.array([80.0]), c=np.array([0.5]),
                                 d=np.array([0.2]), kappa_amp=4.0,
                                 c_power=1.5, d_power=2.0)
    assert np.isclose(k1, k2)


def test_kappa_for_trial_increases_with_contrast():
    base = io_core.kappa_for_trial(np.array([45.0]), c=np.array([0.2]),
                                   d=np.array([0.0]), kappa_amp=4.0,
                                   c_power=1.5, d_power=2.0)
    higher = io_core.kappa_for_trial(np.array([45.0]), c=np.array([0.8]),
                                     d=np.array([0.0]), kappa_amp=4.0,
                                     c_power=1.5, d_power=2.0)
    assert higher > base


def test_kappa_for_trial_decreases_with_dispersion():
    base = io_core.kappa_for_trial(np.array([45.0]), c=np.array([0.5]),
                                   d=np.array([0.0]), kappa_amp=4.0,
                                   c_power=1.5, d_power=2.0)
    disp = io_core.kappa_for_trial(np.array([45.0]), c=np.array([0.5]),
                                   d=np.array([0.5]), kappa_amp=4.0,
                                   c_power=1.5, d_power=2.0)
    assert disp < base


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def test_prior_flat_normalises():
    grids = io_core.IOGrids.default()
    p = io_core.prior_flat(grids)
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, p[0])


def test_prior_bimodal_peaks_at_zero_and_ninety():
    grids = io_core.IOGrids.default()
    p = io_core.prior_bimodal(grids, prior_strength=3.0)
    assert np.isclose(p.sum(), 1.0)
    # Peaks should be at 0 and 90 deg, troughs around 45 deg.
    idx_0 = int(np.where(grids.s_grid_deg == 0.0)[0][0])
    idx_45 = int(np.where(grids.s_grid_deg == 45.0)[0][0])
    idx_90 = int(np.where(grids.s_grid_deg == 90.0)[0][0])
    assert p[idx_0] > p[idx_45]
    assert p[idx_90] > p[idx_45]
    # Symmetry: p(0) and p(90) approximately equal.
    assert np.isclose(p[idx_0], p[idx_90], rtol=1e-6)


# ---------------------------------------------------------------------------
# Posterior P(s | m) and g(m)
# ---------------------------------------------------------------------------


def test_posterior_with_flat_prior_is_normalised_likelihood():
    grids = io_core.IOGrids.default()
    prior = io_core.prior_flat(grids)
    post = io_core.posterior_s_given_m(grids, kappa=5.0, prior=prior)
    # Each row of post sums to 1.
    assert np.allclose(post.sum(axis=1), 1.0)
    # With flat prior, posterior(s|m) = likelihood(m|s) renormalised over s
    # at fixed m.
    lik = io_core.likelihood_m_given_s(grids, kappa=5.0)
    expected = lik / lik.sum(axis=1, keepdims=True)
    assert np.allclose(post, expected, atol=1e-8)


def test_posterior_with_strong_prior_concentrates_on_prior_peaks():
    grids = io_core.IOGrids.default()
    # Bimodal with very high strength -> strong pull toward 0 and 90 deg.
    prior = io_core.prior_bimodal(grids, prior_strength=50.0)
    post = io_core.posterior_s_given_m(grids, kappa=0.5, prior=prior)
    # Pick m corresponding to the 45 deg boundary.
    m_idx = int(np.where(grids.m_grid_deg == 45.0)[0][0])
    row = post[m_idx, :]
    # Mass on s in [0..15] U [75..90] should exceed mass in [40..50].
    near_peaks = np.where((grids.s_grid_deg <= 15) | (grids.s_grid_deg >= 75))[0]
    near_boundary = np.where((grids.s_grid_deg >= 40) & (grids.s_grid_deg <= 50))[0]
    assert row[near_peaks].sum() > 5 * row[near_boundary].sum()


def test_g_of_m_sign_with_flat_prior():
    """With a flat prior and a high-precision sensor, g(m) > 0 when m is
    near a Go orientation (s < 45 deg) and g(m) < 0 near a NoGo orientation."""
    grids = io_core.IOGrids.default()
    prior = io_core.prior_flat(grids)
    post = io_core.posterior_s_given_m(grids, kappa=10.0, prior=prior)
    g = io_core.log_posterior_odds(grids, post)
    idx_20 = int(np.where(grids.m_grid_deg == 20.0)[0][0])  # Go side
    idx_70 = int(np.where(grids.m_grid_deg == 70.0)[0][0])  # NoGo side
    idx_45 = int(np.where(grids.m_grid_deg == 45.0)[0][0])  # Boundary
    assert g[idx_20] > 0
    assert g[idx_70] < 0
    # Boundary should be near zero (symmetry).
    assert abs(g[idx_45]) < 1e-3


def test_g_of_m_antisymmetric_around_boundary_with_flat_prior():
    """Doubled-angle symmetry: g(45 - x) ~ -g(45 + x) under a flat prior."""
    grids = io_core.IOGrids.default()
    prior = io_core.prior_flat(grids)
    post = io_core.posterior_s_given_m(grids, kappa=8.0, prior=prior)
    g = io_core.log_posterior_odds(grids, post)
    for delta in [5.0, 15.0, 25.0]:
        idx_lo = int(np.where(grids.m_grid_deg == 45.0 - delta)[0][0])
        idx_hi = int(np.where(grids.m_grid_deg == 45.0 + delta)[0][0])
        assert np.isclose(g[idx_lo], -g[idx_hi], atol=1e-6), \
            f"delta={delta}, g_lo={g[idx_lo]}, g_hi={g[idx_hi]}"


def test_p_m_given_s_normalises_and_peaks_at_s():
    grids = io_core.IOGrids.default()
    for s_deg in [15.0, 45.0, 75.0]:
        p = io_core.p_m_given_s(grids, kappa=5.0, s_deg=s_deg)
        assert np.isclose(p.sum(), 1.0)
        peak_m = grids.m_grid_deg[int(np.argmax(p))]
        assert abs(peak_m - s_deg) <= 1.0, f"peak {peak_m} far from s={s_deg}"


# ---------------------------------------------------------------------------
# Utility-weighted decision variable DV(m)
# ---------------------------------------------------------------------------


def test_utility_difference_step_values():
    grids = io_core.IOGrids.default()
    diff = io_core.utility_difference(grids)  # default Utility
    # Go bins (s<45): R_hit - R_miss = 1 - 0 = 1
    assert np.allclose(diff[grids.is_go], 1.0)
    # NoGo bins (s>45): R_fa - R_cr = -0.2 - 0.1 = -0.3
    assert np.allclose(diff[grids.is_nogo], -0.3)
    # Boundary (s=45): 0.5(R_hit+R_fa) - 0.5(R_miss+R_cr) = 0.4 - 0.05 = 0.35
    assert np.allclose(diff[grids.is_boundary], 0.35)


def test_decision_variable_delta_posteriors():
    """A posterior concentrated on one orientation gives that bin's utility diff."""
    grids = io_core.IOGrids.default()
    n_s = grids.s_grid_deg.shape[0]
    go_idx = int(np.where(grids.s_grid_deg == 10.0)[0][0])
    nogo_idx = int(np.where(grids.s_grid_deg == 80.0)[0][0])
    bnd_idx = int(np.where(grids.s_grid_deg == 45.0)[0][0])
    post = np.zeros((3, n_s))
    post[0, go_idx] = 1.0
    post[1, nogo_idx] = 1.0
    post[2, bnd_idx] = 1.0
    dv = io_core.decision_variable(grids, post)
    assert np.allclose(dv, [1.0, -0.3, 0.35])


def test_decision_variable_monotone_in_go_mass():
    """DV increases monotonically as posterior mass shifts Go -> from NoGo."""
    grids = io_core.IOGrids.default()
    n_s = grids.s_grid_deg.shape[0]
    go_idx = int(np.where(grids.s_grid_deg == 10.0)[0][0])
    nogo_idx = int(np.where(grids.s_grid_deg == 80.0)[0][0])
    ws = np.linspace(0, 1, 11)
    post = np.zeros((len(ws), n_s))
    post[:, go_idx] = ws
    post[:, nogo_idx] = 1 - ws
    dv = io_core.decision_variable(grids, post)
    assert np.all(np.diff(dv) > 0)
    # endpoints are the pure-Go / pure-NoGo utility diffs
    assert np.isclose(dv[0], -0.3) and np.isclose(dv[-1], 1.0)


def test_decision_variable_sign_tracks_g_of_m():
    """DV(m) and g(m) share sign structure (both rise with P(Go|m))."""
    grids = io_core.IOGrids.default()
    prior = io_core.prior_bimodal(grids)
    post = io_core.posterior_s_given_m(grids, kappa=6.0, prior=prior)
    g = io_core.log_posterior_odds(grids, post)
    dv = io_core.decision_variable(grids, post)
    # Both monotone in P(Go|m); compare ranks across the measurement grid.
    order_g = np.argsort(g)
    assert np.all(np.diff(dv[order_g]) >= -1e-9), "DV not monotone in g(m)"


def test_custom_utility_changes_dv():
    grids = io_core.IOGrids.default()
    n_s = grids.s_grid_deg.shape[0]
    nogo_idx = int(np.where(grids.s_grid_deg == 80.0)[0][0])
    post = np.zeros((1, n_s)); post[0, nogo_idx] = 1.0
    # Harsher false-alarm penalty pushes the NoGo decision variable more negative.
    dv_default = io_core.decision_variable(grids, post)[0]
    dv_harsh = io_core.decision_variable(
        grids, post, io_core.Utility(R_fa=-1.0))[0]
    assert dv_harsh < dv_default


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
