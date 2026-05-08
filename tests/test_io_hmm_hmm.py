# -*- coding: utf-8 -*-
"""Tests for ``ideal_observer/io_hmm/hmm.py``.

Pinned to analytic ground truth wherever possible. The brute-force
enumeration helper is the load-bearing reference for forward log-likelihood
and posterior gamma; xi is checked via its marginal-consistency identities.
"""

from __future__ import annotations

import os
import sys
from itertools import product

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IO_HMM = os.path.join(REPO_ROOT, 'ideal_observer', 'io_hmm')
if IO_HMM not in sys.path:
    sys.path.insert(0, IO_HMM)

import hmm  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _brute_force_loglik(pi: np.ndarray, A: np.ndarray,
                        emiss: np.ndarray) -> float:
    """Sum_path P(y, z) by enumeration. T must be small (K**T paths)."""
    T, K = emiss.shape
    total = 0.0
    for path in product(range(K), repeat=T):
        p = pi[path[0]] * emiss[0, path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * emiss[t, path[t]]
        total += p
    return float(np.log(total))


def _brute_force_gamma(pi: np.ndarray, A: np.ndarray,
                       emiss: np.ndarray) -> np.ndarray:
    """Posterior P(z_t = k | y) by enumeration. Returns (T, K)."""
    T, K = emiss.shape
    joint = np.zeros((T, K))
    Z = 0.0
    for path in product(range(K), repeat=T):
        p = pi[path[0]] * emiss[0, path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * emiss[t, path[t]]
        for t, k in enumerate(path):
            joint[t, k] += p
        Z += p
    return joint / Z


# ---------------------------------------------------------------------------
# logsumexp helper
# ---------------------------------------------------------------------------


def test_logsumexp_matches_naive():
    rng = _rng(0)
    x = rng.normal(size=(7, 5))
    expected = np.log(np.sum(np.exp(x), axis=1))
    got = hmm._logsumexp(x, axis=1)
    assert np.allclose(got, expected, atol=1e-12)


def test_logsumexp_handles_neg_inf_row():
    """All -inf along the reduction axis must give -inf, not NaN."""
    x = np.array([[-np.inf, -np.inf], [0.0, 0.0]])
    got = hmm._logsumexp(x, axis=1)
    assert got[0] == -np.inf
    assert np.isfinite(got[1])


# ---------------------------------------------------------------------------
# Forward / log-likelihood
# ---------------------------------------------------------------------------


def test_forward_loglik_matches_brute_force_2state():
    rng = _rng(0)
    K, T = 2, 4
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    emiss = rng.uniform(0.05, 1.0, size=(T, K))
    log_alpha, log_lik = hmm.log_forward(np.log(pi), np.log(A), np.log(emiss))
    brute = _brute_force_loglik(pi, A, emiss)
    assert np.isclose(log_lik, brute, atol=1e-10)
    assert log_alpha.shape == (T, K)


def test_forward_loglik_matches_brute_force_3state():
    rng = _rng(1)
    K, T = 3, 5
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    emiss = rng.uniform(0.05, 1.0, size=(T, K))
    _, log_lik = hmm.log_forward(np.log(pi), np.log(A), np.log(emiss))
    brute = _brute_force_loglik(pi, A, emiss)
    assert np.isclose(log_lik, brute, atol=1e-10)


def test_forward_T1_returns_pi_times_emiss():
    K = 3
    pi = np.array([0.2, 0.5, 0.3])
    A = np.eye(K)  # transition irrelevant when T=1; log produces -inf entries
    emiss = np.array([[0.1, 0.9, 0.5]])  # (T=1, K=3)
    with np.errstate(divide='ignore'):
        log_A = np.log(A)
    _, log_lik = hmm.log_forward(np.log(pi), log_A, np.log(emiss))
    assert np.isclose(log_lik, np.log((pi * emiss[0]).sum()))


# ---------------------------------------------------------------------------
# Backward and posteriors
# ---------------------------------------------------------------------------


def test_posterior_gamma_matches_brute_force():
    rng = _rng(2)
    K, T = 3, 5
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    emiss = rng.uniform(0.05, 1.0, size=(T, K))
    log_alpha, _ = hmm.log_forward(np.log(pi), np.log(A), np.log(emiss))
    log_beta = hmm.log_backward(np.log(A), np.log(emiss))
    gamma = hmm.posterior_states(log_alpha, log_beta)
    brute = _brute_force_gamma(pi, A, emiss)
    assert gamma.shape == (T, K)
    assert np.allclose(gamma, brute, atol=1e-10)


def test_posterior_gamma_rows_sum_to_one():
    rng = _rng(3)
    K, T = 4, 6
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    emiss = rng.uniform(0.05, 1.0, size=(T, K))
    log_alpha, _ = hmm.log_forward(np.log(pi), np.log(A), np.log(emiss))
    log_beta = hmm.log_backward(np.log(A), np.log(emiss))
    gamma = hmm.posterior_states(log_alpha, log_beta)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)


def test_xi_marginal_consistency():
    """sum_j xi[t,i,j] = gamma[t,i]; sum_i xi[t,i,j] = gamma[t+1,j]."""
    rng = _rng(4)
    K, T = 3, 6
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    emiss = rng.uniform(0.05, 1.0, size=(T, K))
    log_alpha, _ = hmm.log_forward(np.log(pi), np.log(A), np.log(emiss))
    log_beta = hmm.log_backward(np.log(A), np.log(emiss))
    gamma = hmm.posterior_states(log_alpha, log_beta)
    xi = hmm.posterior_transitions(log_alpha, log_beta, np.log(A), np.log(emiss))
    assert xi.shape == (T - 1, K, K)
    # Each (i, j) slice sums to 1 per t.
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-10)
    # Marginal over j is gamma[t, i] for t = 0..T-2.
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)
    # Marginal over i is gamma[t+1, j] for t = 0..T-2.
    assert np.allclose(xi.sum(axis=1), gamma[1:], atol=1e-10)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------


def test_viterbi_recovers_deterministic_path():
    """When emissions strongly prefer a known sequence and transitions are
    near-uniform, Viterbi recovers the emission-pinned path."""
    K, T = 3, 8
    rng = _rng(5)
    true_path = rng.integers(K, size=T)
    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 1.0 / K)  # uniform transitions
    emiss = np.full((T, K), 0.01)
    for t in range(T):
        emiss[t, true_path[t]] = 1.0
    path, _ = hmm.viterbi(np.log(pi), np.log(A), np.log(emiss))
    assert np.array_equal(path, true_path)


def test_viterbi_matches_brute_force_2state():
    """Brute-force argmax over all paths matches Viterbi."""
    rng = _rng(6)
    K, T = 2, 5
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    emiss = rng.uniform(0.05, 1.0, size=(T, K))
    # Brute-force best path.
    best_p = -np.inf
    best_path = None
    for path in product(range(K), repeat=T):
        p = np.log(pi[path[0]]) + np.log(emiss[0, path[0]])
        for t in range(1, T):
            p += np.log(A[path[t - 1], path[t]]) + np.log(emiss[t, path[t]])
        if p > best_p:
            best_p = p
            best_path = path
    path, log_p = hmm.viterbi(np.log(pi), np.log(A), np.log(emiss))
    assert np.array_equal(path, np.array(best_path))
    assert np.isclose(log_p, best_p, atol=1e-10)


def test_viterbi_sticky_chain_stays_in_dominant_state():
    """High self-transition probability + ambiguous emissions: path collapses
    to a single state most of the time."""
    K, T = 2, 50
    pi = np.array([0.5, 0.5])
    A = np.array([[0.99, 0.01], [0.01, 0.99]])  # very sticky
    rng = _rng(7)
    emiss = rng.uniform(0.4, 0.6, size=(T, K))  # nearly uninformative
    path, _ = hmm.viterbi(np.log(pi), np.log(A), np.log(emiss))
    # Either all 0s or all 1s (one switch is improbable).
    n_changes = int(np.sum(np.diff(path) != 0))
    assert n_changes <= 1, f"too many state changes ({n_changes}) for sticky chain"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
