"""Synthetic-ground-truth tests for ``similarity_readout_tests``.

Each test plants a known generative structure and checks that the RD-1 / RD-2
estimators recover it — covering both *sensitivity* (they detect an effect that
is really there) and *specificity* (they return ≈0 when it is not).  This is the
trust anchor for the real-data conclusions: a test that always fires, or never
fires, would make the empirical result meaningless.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nn_decoder"))

rt = pytest.importorskip("similarity_readout_tests")
pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Linear-algebra core
# ---------------------------------------------------------------------------


def test_class_direction_recovers_planted_dmu():
    rng = np.random.default_rng(0)
    p = 8
    dmu_true = np.zeros(p); dmu_true[0] = 2.0
    y = rng.integers(0, 2, size=2000)
    X = y[:, None] * dmu_true[None, :] + rng.standard_normal((2000, p)) * 0.3
    dmu = rt.class_direction(X, y)
    assert np.argmax(np.abs(dmu)) == 0
    assert abs(dmu[0] - 2.0) < 0.1


def test_whitened_direction_endpoints():
    # λ=1 → template Δμ; λ=0 → Σ⁻¹Δμ (rotated by anisotropic Σ)
    p = 6
    dmu = np.ones(p)
    Sigma = np.diag(np.linspace(1.0, 10.0, p))   # anisotropic
    w_template = rt.whitened_direction(dmu, Sigma, 1.0)
    w_lda = rt.whitened_direction(dmu, Sigma, 0.0)
    cos_t = abs(np.dot(w_template, dmu / np.linalg.norm(dmu)))
    assert cos_t > 0.999                          # template == Δμ direction
    # Σ⁻¹Δμ down-weights the high-variance dims → direction tilts to dim 0
    assert np.argmax(np.abs(w_lda)) == 0
    assert abs(np.dot(w_lda, dmu / np.linalg.norm(dmu))) < 0.95   # genuinely rotated


# ---------------------------------------------------------------------------
# RD-1 — dissociation: whitening buys stimulus decoding, not choice
# ---------------------------------------------------------------------------


def _make_rd1_data(seed=0, n=900, p=12, rho=0.85):
    # Signal lives in dim 0; noise in dim 0 is strongly *correlated* with dim 1,
    # so a whitened decoder uses dim 1 as a noise reference to clean dim 0 and
    # separates the classes better than the raw template.  (Per-feature
    # z-scoring inside the estimator removes marginal-variance anisotropy, so the
    # discriminating structure must live in the *correlations*, and Δμ must not
    # be a correlation-eigenvector — here it is dim 0, not [1,1]/[1,-1].)
    rng = np.random.default_rng(seed)
    y_stim = rng.integers(0, 2, size=n)
    dmu = np.zeros(p); dmu[0] = 1.0
    C = np.eye(p); C[0, 1] = C[1, 0] = rho
    L = np.linalg.cholesky(C)
    noise = rng.standard_normal((n, p)) @ L.T
    X = y_stim[:, None] * dmu[None, :] * 1.3 + noise
    # choice follows the raw template (dim 0), not the whitened readout
    proj_t = X[:, 0]
    proj_t = (proj_t - proj_t.mean()) / proj_t.std()
    p_go = 1.0 / (1.0 + np.exp(-1.8 * proj_t))
    y_choice = (rng.random(n) < p_go).astype(int)
    return X, y_stim, y_choice


def test_rd1_whitening_helps_stimulus():
    X, ys, yc = _make_rd1_data()
    r = rt.rd1_template_vs_whitened(X, ys, yc, n_folds=5, seed=1)
    # covariance buys stimulus-decoding accuracy: auc_stim higher at λ=0 (LDA)
    assert r["d_auc_stim_whiten"] > 0.03


def test_rd1_whitening_does_not_help_choice():
    X, ys, yc = _make_rd1_data()
    r = rt.rd1_template_vs_whitened(X, ys, yc, n_folds=5, seed=1)
    # the dissociation: choice does NOT ride the whitening gain that stimulus does
    assert r["d_auc_choice_whiten"] < r["d_auc_stim_whiten"]


# ---------------------------------------------------------------------------
# RD-2 — nested choice models: sensitivity + specificity per step
# ---------------------------------------------------------------------------


def _base_rd2(seed=0, n=900, p=10):
    rng = np.random.default_rng(seed)
    levels = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0])
    c = rng.choice(levels, size=n)
    y_stim = (c > 0).astype(int)
    # si_mean carries the stimulus + a within-condition fluctuation
    within = rng.standard_normal(n)
    si_mean = 1.2 * c + 0.8 * within + 0.2 * rng.standard_normal(n)
    si_var = np.abs(rng.standard_normal(n))      # pure noise by default
    dmu = np.zeros(p); dmu[0] = 1.0
    X = y_stim[:, None] * dmu[None, :] + rng.standard_normal((n, p)) * 0.5
    return rng, c, y_stim, si_mean, si_var, within, X, dmu


def test_rd2_meanSI_adds_within_condition():
    rng, c, y_stim, si_mean, si_var, within, X, dmu = _base_rd2(seed=1)
    # choice driven by stimulus AND the within-condition si fluctuation
    z = 1.5 * c + 1.2 * within
    y_choice = (rng.random(len(c)) < 1 / (1 + np.exp(-z))).astype(int)
    r = rt.rd2_nested_choice_models(X, y_stim, y_choice, c, si_mean, si_var, seed=2)
    assert r["delta"]["M1-M0 (meanSI adds | Pred16)"] > 0.005


def test_rd2_varSI_specificity_and_sensitivity():
    # specificity: si_var is noise → M2−M1 ≈ 0 (not positive)
    rng, c, y_stim, si_mean, si_var, within, X, dmu = _base_rd2(seed=3)
    z = 1.5 * c + 1.2 * within
    y_choice = (rng.random(len(c)) < 1 / (1 + np.exp(-z))).astype(int)
    r0 = rt.rd2_nested_choice_models(X, y_stim, y_choice, c, si_mean, si_var, seed=4)
    assert r0["delta"]["M2-M1 (varSI adds | SBC wedge)"] < 0.004

    # sensitivity: now choice DOES depend on si_var → M2−M1 > 0
    rng, c, y_stim, si_mean, si_var, within, X, dmu = _base_rd2(seed=3)
    si_var = np.abs(rng.standard_normal(len(c)))
    z = 1.5 * c + 1.2 * within + 1.4 * (si_var - si_var.mean())
    y_choice = (rng.random(len(c)) < 1 / (1 + np.exp(-z))).astype(int)
    r1 = rt.rd2_nested_choice_models(X, y_stim, y_choice, c, si_mean, si_var, seed=4)
    assert r1["delta"]["M2-M1 (varSI adds | SBC wedge)"] > 0.01


def _make_rd2_aniso(seed, n=1100, p=10, rho=0.85):
    """Correlated-noise data where the whitened direction Σ⁻¹Δμ ≠ template Δμ.

    ``si_mean`` is the template (dim-0) readout — the framework SI surrogate.
    Returns the true template and whitened projections so a test can drive
    choice from either.
    """
    rng = np.random.default_rng(seed)
    levels = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0])
    c = rng.choice(levels, n)
    y_stim = (c > 0).astype(int)
    dmu = np.zeros(p); dmu[0] = 1.0
    C = np.eye(p); C[0, 1] = C[1, 0] = rho
    L = np.linalg.cholesky(C)
    noise = rng.standard_normal((n, p)) @ L.T
    X = y_stim[:, None] * dmu[None, :] * 1.3 + noise + c[:, None] * dmu[None, :] * 0.6
    tmpl = X[:, 0]
    w = np.linalg.inv(C) @ dmu; w /= np.linalg.norm(w)
    whit = X @ w
    si_mean = (tmpl - tmpl.mean()) / tmpl.std()
    si_var = np.abs(rng.standard_normal(n))
    return rng, c, y_stim, X, si_mean, si_var, tmpl, whit


def test_rd2_whitened_specificity():
    # choice follows the template only → the whitened direction adds ≈0
    rng, c, ys, X, si_mean, si_var, tmpl, whit = _make_rd2_aniso(5)
    z = 1.2 * c + 1.3 * (tmpl - tmpl.mean()) / tmpl.std()
    yc = (rng.random(len(c)) < 1 / (1 + np.exp(-z))).astype(int)
    r = rt.rd2_nested_choice_models(X, ys, yc, c, si_mean, si_var, seed=6)
    assert r["delta"]["M3-M1 (whitened adds | premise)"] < 0.01


def test_rd2_whitened_sensitivity():
    # choice follows the whitened direction → M3−M1 > 0 (test has power)
    rng, c, ys, X, si_mean, si_var, tmpl, whit = _make_rd2_aniso(5)
    z = 1.0 * c + 1.6 * (whit - whit.mean()) / whit.std()
    yc = (rng.random(len(c)) < 1 / (1 + np.exp(-z))).astype(int)
    r = rt.rd2_nested_choice_models(X, ys, yc, c, si_mean, si_var, seed=6)
    assert r["delta"]["M3-M1 (whitened adds | premise)"] > 0.01
