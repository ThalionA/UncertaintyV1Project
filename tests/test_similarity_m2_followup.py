"""Synthetic-ground-truth tests for the M2 follow-up battery.

Covers the distinctive pieces of ``similarity_m2_followup`` (its own CV ΔLL,
the within-trial-dynamics summary, and the permutation null) — confirming each
has sensitivity and specificity so the real-data verdict (Cb22 = artifact,
Cb17 = genuine-but-non-SBC) rests on trustworthy machinery.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nn_decoder"))

m2 = pytest.importorskip("similarity_m2_followup")
pytest.importorskip("sklearn")


def _folds(y, k=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(
        np.zeros((len(y), 1)), y))


# --- _delta: held-out ΔLL is honest (noise ≤ 0, signal > 0) ----------------


def test_delta_noise_feature_not_positive():
    rng = np.random.default_rng(0)
    n = 800
    c = rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * c))).astype(int)
    feat = {"c": c, "noise": rng.standard_normal(n), "y": y}
    folds = _folds(y)
    d = m2._delta(feat, ["c"], "noise", y, folds)
    assert d < 0.005          # a pure-noise feature does not help held-out


def test_delta_signal_feature_positive():
    rng = np.random.default_rng(1)
    n = 800
    c = rng.standard_normal(n)
    extra = rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.0 * c + 1.3 * extra)))).astype(int)
    feat = {"c": c, "extra": extra, "y": y}
    folds = _folds(y)
    assert m2._delta(feat, ["c"], "extra", y, folds) > 0.02


# --- within_trial_dynamics: flat vs dynamic SI(t) --------------------------


def _ar1(n, T, rho, rng):
    x = np.zeros((n, T))
    x[:, 0] = rng.standard_normal(n)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return x


def test_within_trial_dynamics_smooth_vs_white():
    rng = np.random.default_rng(2)
    T = 25
    # smooth within-trial trajectory (AR(1), high ρ) → high autocorr, ESS→1
    smooth = _ar1(300, T, 0.9, rng)
    # white: independent bins → low autocorr, ESS large (more 'samples')
    white = _ar1(300, T, 0.0, rng)
    ds = m2.within_trial_dynamics(smooth)
    dw = m2.within_trial_dynamics(white)
    assert ds["lag1_autocorr"] > 0.7
    assert dw["lag1_autocorr"] < 0.3
    assert dw["ess_per_trial"] > ds["ess_per_trial"]   # high-ρ → fewer effective samples


# --- permutation null structure: a real signal exceeds the shuffle band ----


def test_permutation_null_detects_real_signal():
    rng = np.random.default_rng(3)
    n = 700
    levels = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    c = rng.choice(levels, n)
    io = 1.3 * c + 0.2 * rng.standard_normal(n)
    var_si = np.abs(rng.standard_normal(n))
    # choice depends on the within-condition variance → permutation must break it
    z = 1.0 * c + 1.4 * (var_si - var_si.mean())
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    bundle = {
        "feat": {"c": c, "io_logodds": io, "var_si": var_si, "y": y,
                 "correct": np.full(n, np.nan), "H_dec": np.full(n, np.nan)},
        "folds": _folds(y),
    }
    out = m2.permutation_and_bootstrap(bundle, n_perm=120, n_boot=10)
    io_res = out["C2_IO"]
    assert io_res["real_dLL"] > io_res["null_mean"]
    assert io_res["z"] > 3.0
    assert io_res["p_right"] < 0.05
