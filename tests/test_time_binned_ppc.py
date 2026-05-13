# -*- coding: utf-8 -*-
"""Synthetic-data tests for ``nn_decoder/time_binned_ppc.py``.

Three checks:

1. Stationary equivalence — TimeInt-stationary = softmax(T · LL_TimeAvg)
   to numerical precision, verifying the closed-form claim in the
   module docstring.

2. Decoding sanity — under known Poisson tuning and reasonable SNR,
   the per-trial PPC mean recovers the presented orientation within a
   few degrees on average, for all three variants.

3. Time-varying templates matter — when bin-specific tuning genuinely
   differs from the trial-averaged tuning (a deliberately time-varying
   generative model), TimeInt-timevarying is meaningfully sharper /
   better-aligned with the true orientation than TimeInt-stationary.
"""

from __future__ import annotations

import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

import time_binned_ppc as tbp


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _rng(seed=0):
    return np.random.default_rng(seed)


def _bell_template(n_neurons, s_grid, gain=4.0, baseline=0.2, tuning_sigma=12.0,
                   rng=None):
    """A bank of bell-shaped (half-circle) tuning curves with preferred
    orientations uniformly tiling [0, 90]. Returns templates of shape
    (S, n_neurons) and the preferred orientations."""
    if rng is None:
        rng = np.random.default_rng(0)
    prefs = np.linspace(0, 90, n_neurons, endpoint=False) + rng.uniform(
        -1, 1, size=n_neurons)
    prefs = np.clip(prefs, 0, 90)
    # half-circle distance: min(|Δ|, 90 - |Δ|)
    d = np.abs(s_grid[:, None] - prefs[None, :])
    d = np.minimum(d, 90 - d)
    f = baseline + gain * np.exp(-0.5 * (d / tuning_sigma) ** 2)
    return f, prefs


def _make_trials(orientations, contrast=1.0, dispersion=0.0):
    n = len(orientations)
    return {
        'orientation': np.asarray(orientations, dtype=float),
        'contrast':    np.full(n, contrast, dtype=float),
        'dispersion':  np.full(n, dispersion, dtype=float),
    }


# --------------------------------------------------------------------------
# 1. Stationary equivalence
# --------------------------------------------------------------------------

def test_stationary_equivalence_matches_temperature_rescaling():
    rng = _rng(7)
    s_grid = tbp.S_GRID
    n_neurons, t_bins, n_trials = 25, 8, 60

    template, _ = _bell_template(n_neurons, s_grid, rng=rng)
    # Sample true orientations from the grid, generate Poisson activity.
    true_idx = rng.integers(0, len(s_grid), size=n_trials)
    rate = template[true_idx, :]                              # (n_trials, n_neurons)
    activity = rng.poisson(np.broadcast_to(rate[:, None, :],
                                          (n_trials, t_bins, n_neurons))).astype(float)

    L_avg, _ = tbp.ppc_time_avg(activity, template)
    L_int, _ = tbp.ppc_time_int_stationary(activity, template)

    # LL_int = T · LL_avg up to a per-trial constant (the - T·f_n vs - f_n
    # offset is shared across s; the r·log f term is exactly T-fold larger
    # under stationary templates because Σ_t r_t = T · r̄ even though
    # `activity` carries Poisson noise).
    # After softmax, that becomes L_int(s) ∝ L_avg(s)^T.
    L_avg_safe = np.clip(L_avg, 1e-12, 1.0)
    L_avg_pow = L_avg_safe ** t_bins
    L_avg_pow = L_avg_pow / L_avg_pow.sum(axis=-1, keepdims=True)
    assert np.allclose(L_int, L_avg_pow, atol=1e-9), \
        "TimeInt-stationary should equal softmax(T · LL_avg)."


# --------------------------------------------------------------------------
# 2. Decoding sanity — all three variants
# --------------------------------------------------------------------------

def test_all_variants_recover_orientation_under_clean_tuning():
    rng = _rng(11)
    s_grid = tbp.S_GRID
    n_neurons, t_bins, n_trials = 40, 10, 120

    template, _ = _bell_template(n_neurons, s_grid, gain=6.0, baseline=0.3,
                                 rng=rng)
    # Trials: orientations sampled from a coarse grid spanning [0, 90].
    coarse = np.array([5, 20, 35, 50, 65, 80], dtype=float)
    orientations = rng.choice(coarse, size=n_trials)
    true_idx = np.array([np.argmin(np.abs(s_grid - o)) for o in orientations])
    rate = template[true_idx, :]                               # (n_trials, n_neurons)
    activity = rng.poisson(np.broadcast_to(rate[:, None, :],
                                           (n_trials, t_bins, n_neurons))).astype(float)
    trials = _make_trials(orientations)

    # Fit templates the same way the pipeline does.
    rates_2d = activity.mean(axis=1)
    template_stat = tbp.fit_template_from_rates(rates_2d, trials, s_grid)
    templates_tv  = tbp.fit_templates_per_bin(activity, trials, s_grid)

    _, P_avg = tbp.ppc_time_avg(activity, template_stat)
    _, P_int_s = tbp.ppc_time_int_stationary(activity, template_stat)
    _, P_int_v = tbp.ppc_time_int_timevarying(activity, templates_tv)

    for name, P in [('TimeAvg', P_avg),
                    ('TimeInt_stat', P_int_s),
                    ('TimeInt_timevary', P_int_v)]:
        mu, _ = tbp.dist_mean_var(P, s_grid)
        err = np.abs(mu - orientations)
        err = np.minimum(err, 90 - err)
        assert np.mean(err) < 5.0, \
            f"{name}: mean half-circle err {np.mean(err):.2f}° too large"


# --------------------------------------------------------------------------
# 3. Time-varying templates beat stationary when generative model varies
# --------------------------------------------------------------------------

def test_time_varying_helps_when_tuning_is_time_varying():
    """Construct a generative model where each bin uses a *different*
    tuning curve (different preferred-orientation tiling and different
    gain). The trial-averaged template smears these together; per-bin
    templates can recover them. Decoding accuracy of TimeInt-timevarying
    should beat TimeInt-stationary."""
    rng = _rng(23)
    s_grid = tbp.S_GRID
    n_neurons, t_bins, n_trials = 40, 10, 200

    # Build T distinct templates with rotated preferred-orientation tilings.
    templates_true = np.empty((t_bins, len(s_grid), n_neurons))
    for t in range(t_bins):
        shift = (t * 7) % 90  # rotate prefs each bin
        prefs = np.mod(np.linspace(0, 90, n_neurons, endpoint=False) + shift, 90)
        d = np.abs(s_grid[:, None] - prefs[None, :])
        d = np.minimum(d, 90 - d)
        gain = 3.0 + 0.5 * t  # mild ramp
        templates_true[t] = 0.3 + gain * np.exp(-0.5 * (d / 10.0) ** 2)

    coarse = np.array([5, 20, 35, 50, 65, 80], dtype=float)
    orientations = rng.choice(coarse, size=n_trials)
    true_idx = np.array([np.argmin(np.abs(s_grid - o)) for o in orientations])

    activity = np.empty((n_trials, t_bins, n_neurons))
    for t in range(t_bins):
        rate_t = templates_true[t, true_idx, :]                # (n_trials, n_neurons)
        activity[:, t, :] = rng.poisson(rate_t).astype(float)
    trials = _make_trials(orientations)

    rates_2d = activity.mean(axis=1)
    template_stat = tbp.fit_template_from_rates(rates_2d, trials, s_grid)
    templates_tv  = tbp.fit_templates_per_bin(activity, trials, s_grid)

    _, P_int_s = tbp.ppc_time_int_stationary(activity, template_stat)
    _, P_int_v = tbp.ppc_time_int_timevarying(activity, templates_tv)

    mu_s, _ = tbp.dist_mean_var(P_int_s, s_grid)
    mu_v, _ = tbp.dist_mean_var(P_int_v, s_grid)
    err_s = np.minimum(np.abs(mu_s - orientations),
                       90 - np.abs(mu_s - orientations))
    err_v = np.minimum(np.abs(mu_v - orientations),
                       90 - np.abs(mu_v - orientations))
    assert np.mean(err_v) < np.mean(err_s), (
        f"Time-varying templates should outperform stationary under "
        f"time-varying tuning. stat err={np.mean(err_s):.2f}°, "
        f"timevary err={np.mean(err_v):.2f}°")


# --------------------------------------------------------------------------
# Sanity for distribution summaries
# --------------------------------------------------------------------------

def test_kl_zero_when_equal_and_moments_recover_delta():
    s_grid = tbp.S_GRID
    delta = np.zeros((3, len(s_grid)))
    delta[0, 10] = 1.0
    delta[1, 45] = 1.0
    delta[2, 80] = 1.0
    mu, var = tbp.dist_mean_var(delta, s_grid)
    assert np.allclose(mu, [10, 45, 80])
    assert np.allclose(var, [0, 0, 0])
    assert np.allclose(tbp.dist_kl(delta, delta), 0.0, atol=1e-9)
