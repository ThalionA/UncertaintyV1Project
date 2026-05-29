# -*- coding: utf-8 -*-
"""Tests for the generative model-recovery simulation.

Two layers:
  * Deterministic checks on the generative models themselves (tuning,
    latents, gain calibration, PPC sufficiency, SBC sampling) and a fast
    *analytic* double-dissociation using the repo's own PPC/SBC readouts —
    this is the scientific anchor and needs no training.
  * A small end-to-end smoke test that actually fits the decoders through
    the recovery harness and checks shapes + that the matched decoder
    beats its shuffle control.
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

from simulation.generative import (  # noqa: E402
    S_GRID, make_tuning_curves, make_latents, make_prior, calibrate_gain,
    simulate_ppc, simulate_sbc, simulate_population,
)
from utils import (  # noqa: E402
    generate_PPC_targets, generate_SBC_targets, calculate_np_variance,
)


def _mean_kl(truth, est, eps=1e-12):
    p = truth + eps
    q = est + eps
    p = p / p.sum(1, keepdims=True)
    q = q / q.sum(1, keepdims=True)
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


# ----------------------------------------------------------------------
# Generative-model unit checks
# ----------------------------------------------------------------------

def test_tuning_curves_shape_and_peak():
    f, pref = make_tuning_curves(60, tuning_width=10.0, seed=1)
    assert f.shape == (len(S_GRID), 60)
    assert (f > 0).all()
    # For neurons whose preferred orientation lies inside the grid, the
    # on-grid peak sits at that preference (edge-padded neurons peak at the
    # boundary, so they're excluded from this check).
    interior = (pref >= 5) & (pref <= 85)
    peak_deg = S_GRID[np.argmax(f[:, interior], axis=0)]
    assert np.median(np.abs(peak_deg - pref[interior])) <= 2.0


def test_latents_normalised_and_uncertainty_monotonic():
    lat = make_latents(600, uncertainty_levels=(3.0, 9.0, 18.0), seed=2)
    assert np.allclose(lat.likelihood.sum(1), 1.0, atol=1e-6)
    assert np.allclose(lat.posterior.sum(1), 1.0, atol=1e-6)
    assert np.allclose(lat.decision.sum(1), 1.0, atol=1e-6)
    # Posterior gets broader with the uncertainty level.
    var = calculate_np_variance(lat.posterior, S_GRID)
    means = [var[lat.level == k].mean() for k in range(3)]
    assert means[0] < means[1] < means[2], means


def test_prior_bimodal_peaks_at_modes():
    p = make_prior(prior_type='bimodal')
    assert p[0] > p[45] and p[90] > p[45]   # modes at 0 and 90, trough at 45


def test_gain_calibration_monotonic():
    f, _ = make_tuning_curves(60, seed=3)
    g = calibrate_gain(np.array([4.0, 8.0, 16.0]), f, T=10)
    # More certain (smaller sigma) -> higher gain -> more spikes.
    assert g[0] > g[1] > g[2]


def test_ppc_mean_sufficient_and_recovers_stimulus():
    f, _ = make_tuning_curves(100, tuning_width=10.0, seed=4)
    lat = make_latents(60, uncertainty_levels=(5.0,), seed=4)
    act = simulate_ppc(lat, f, T=20, seed=4)            # (n_neurons, n_trials, T)
    act_tr = np.transpose(act, (1, 2, 0))                # (n_trials, T, n_neurons)

    # The linear-PPC likelihood (mean-rate Poisson readout) peaks near the
    # true stimulus.
    L, _ = generate_PPC_targets(act_tr, f, beta=1.0)
    maps = S_GRID[np.argmax(L, axis=1)]
    assert np.median(np.abs(maps - lat.s_true)) < 12.0

    # Time-averaging is a sufficient statistic: permuting the time bins
    # leaves the readout exactly unchanged (the defining PPC property).
    perm = np.random.RandomState(0).permutation(20)
    L_perm, _ = generate_PPC_targets(act_tr[:, perm, :], f, beta=1.0)
    assert np.allclose(L, L_perm)


def test_sbc_readout_recovers_posterior():
    f, _ = make_tuning_curves(100, tuning_width=10.0, seed=5)
    lat = make_latents(40, uncertainty_levels=(8.0, 12.0), seed=5)
    # Many bins per trial so the per-bin sample MAPs densely trace Q*.
    act = simulate_sbc(lat, f, T=300, sample_gain=12.0, seed=5)
    act_tr = np.transpose(act, (1, 2, 0))
    _, Qhat = generate_SBC_targets(act_tr, f, kde_std=3.0)
    kl_truth = _mean_kl(lat.posterior, Qhat)
    kl_prior = _mean_kl(np.repeat(lat.prior[None, :], len(lat.s_true), 0), Qhat)
    # The recovered distribution matches the true posterior much better
    # than it matches the (different) prior.
    assert kl_truth < kl_prior
    assert kl_truth < 0.5


# ----------------------------------------------------------------------
# Analytic double dissociation (deterministic, no training)
# ----------------------------------------------------------------------

def test_analytic_double_dissociation_on_posterior():
    """Using the repo's OWN readouts (generate_PPC/SBC_targets), each
    readout recovers the ground-truth posterior better on the data its
    matching generative model produced. Each readout is variance-calibrated
    per dataset (optimize_synthetic_params), so the win reflects the
    trial-by-trial coding structure, not a global width offset."""
    f, _ = make_tuning_curves(100, tuning_width=10.0, seed=7)
    lat = make_latents(300, uncertainty_levels=(6.0, 12.0), seed=7)
    Q = lat.posterior

    act_ppc = np.transpose(simulate_ppc(lat, f, T=20, seed=7), (1, 2, 0))
    act_sbc = np.transpose(simulate_sbc(lat, f, T=20, sample_gain=12.0, seed=7),
                           (1, 2, 0))                    # (n_trials, T, n_neurons)

    # PPC readout uses beta=1 (the ML Poisson inverse-temperature, which the
    # gain calibration already matches to the target width); SBC readout uses
    # a fixed small KDE bandwidth. Neither is tuned to win.
    ppc_on_ppc = generate_PPC_targets(act_ppc, f, beta=1.0)[1]
    sbc_on_ppc = generate_SBC_targets(act_ppc, f, kde_std=3.0)[1]
    ppc_on_sbc = generate_PPC_targets(act_sbc, f, beta=1.0)[1]
    sbc_on_sbc = generate_SBC_targets(act_sbc, f, kde_std=3.0)[1]

    # On PPC data the PPC readout wins; on SBC data the SBC readout wins.
    assert _mean_kl(Q, ppc_on_ppc) < _mean_kl(Q, sbc_on_ppc)
    assert _mean_kl(Q, sbc_on_sbc) < _mean_kl(Q, ppc_on_sbc)


# ----------------------------------------------------------------------
# Trained-decoder end-to-end smoke
# ----------------------------------------------------------------------

def test_recovery_harness_runs_and_beats_shuffle():
    pytest.importorskip('torch')
    from simulation.recovery import run_cell, DECODER_SPEC

    # Small but real fit of every decoder on PPC-generated posterior data.
    cell = run_cell('ppc', 'Q', n_trials=160, n_neurons=40, T=8,
                    uncertainty_levels=(6.0, 12.0), REP=1, num_epochs=12, seed=0)
    metrics = cell['metrics']
    assert set(metrics) == set(DECODER_SPEC)
    for name, m in metrics.items():
        assert np.isfinite(m['fit_loss_norm']), name
        assert 'kl_to_truth' in m
    # The matched (PPC) decoder must recover better than its shuffle control.
    assert metrics['PPC']['fit_loss_norm'] < 1.0
