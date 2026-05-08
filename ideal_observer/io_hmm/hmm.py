# -*- coding: utf-8 -*-
"""Categorical HMM core: log-domain forward, backward, posterior, Viterbi.

These routines operate purely on log-emission matrices ``log_emiss`` of shape
``(T, K)`` plus an initial-distribution vector and a stationary transition
matrix. Emission semantics (``log P(y_t | z_t = k)``) are owned by
``emissions.py``; the HMM core does not know what ``y_t`` is.

Conventions
-----------
- ``log_pi``: shape (K,), ``log P(z_1 = k)``.
- ``log_A``:  shape (K, K), ``log P(z_{t+1} = j | z_t = i)``.
- ``log_emiss``: shape (T, K), ``log P(y_t | z_t = k)``.

All quantities are computed in log space with the numerically stable
``_logsumexp``. For T or K large, the routines remain memory-friendly:
forward/backward are O(T*K^2), Viterbi is O(T*K^2), xi materialises an
``(T-1, K, K)`` array which is the only O(T*K^2) memory cost.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -np.inf


# ---------------------------------------------------------------------------
# Numerically stable log-sum-exp (numpy-only; mirrors scipy.special.logsumexp
# for the cases we use)
# ---------------------------------------------------------------------------


def _logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable ``log(sum(exp(x), axis=axis))``.

    Treats ``-inf`` rows correctly (returns ``-inf`` rather than NaN). Does
    not return a keepdims-shaped array; squeezes the reduced axis.
    """
    x = np.asarray(x, dtype=float)
    if axis is None:
        m = np.max(x)
        if not np.isfinite(m):
            # All -inf -> log(sum(exp)) = -inf.
            return NEG_INF
        return float(m + np.log(np.sum(np.exp(x - m))))
    m = np.max(x, axis=axis, keepdims=True)
    finite = np.isfinite(m)
    # Where the max is -inf, the entire slice is -inf; substitute 0 to avoid
    # NaN in the subtraction, then mask the result back to -inf.
    safe_m = np.where(finite, m, 0.0)
    s = np.sum(np.exp(x - safe_m), axis=axis, keepdims=True)
    with np.errstate(divide='ignore'):  # log(0) -> -inf is the intended path
        out = safe_m + np.log(s)
    out = np.where(finite, out, NEG_INF)
    return np.squeeze(out, axis=axis)


# ---------------------------------------------------------------------------
# Forward and backward
# ---------------------------------------------------------------------------


def log_forward(log_pi: np.ndarray, log_A: np.ndarray,
                log_emiss: np.ndarray) -> tuple[np.ndarray, float]:
    """Forward recursion in log space.

    Returns
    -------
    log_alpha : (T, K) array
        ``log_alpha[t, k] = log P(y_{1:t}, z_t = k)``.
    log_lik : float
        ``log P(y_{1:T})`` (the marginal likelihood of the observation
        sequence under the model).
    """
    T, K = log_emiss.shape
    log_alpha = np.full((T, K), NEG_INF, dtype=float)
    log_alpha[0] = log_pi + log_emiss[0]
    for t in range(1, T):
        # log_alpha[t, j] = logsumexp_i (log_alpha[t-1, i] + log_A[i, j])
        #                  + log_emiss[t, j]
        log_alpha[t] = _logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) \
                       + log_emiss[t]
    log_lik = float(_logsumexp(log_alpha[-1]))
    return log_alpha, log_lik


def log_backward(log_A: np.ndarray, log_emiss: np.ndarray) -> np.ndarray:
    """Backward recursion in log space.

    Returns
    -------
    log_beta : (T, K) array
        ``log_beta[t, k] = log P(y_{t+1:T} | z_t = k)``. By convention
        ``log_beta[T-1] = 0`` (i.e. P(empty future | anything) = 1).
    """
    T, _ = log_emiss.shape
    log_beta = np.zeros_like(log_emiss)  # log 1 = 0
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j (log_A[i, j] + log_emiss[t+1, j]
        #                                          + log_beta[t+1, j])
        log_beta[t] = _logsumexp(log_A + log_emiss[t + 1] + log_beta[t + 1],
                                  axis=1)
    return log_beta


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------


def posterior_states(log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
    """``gamma[t, k] = P(z_t = k | y_{1:T})``. Returns (T, K) in linear space."""
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    return np.exp(log_gamma)


def posterior_transitions(log_alpha: np.ndarray, log_beta: np.ndarray,
                          log_A: np.ndarray,
                          log_emiss: np.ndarray) -> np.ndarray:
    """``xi[t, i, j] = P(z_t = i, z_{t+1} = j | y_{1:T})`` for t = 0..T-2.

    Returns shape (T-1, K, K) in linear space. Each (T-1, :, :) slice sums
    to 1.
    """
    # Outer combination of forward, backward, transition, and next-step emission.
    # Shapes: alpha[:-1] (T-1, K, 1); A (1, K, K); emiss[1:] (T-1, 1, K);
    # beta[1:] (T-1, 1, K).
    T_minus_1 = log_emiss.shape[0] - 1
    log_xi = (log_alpha[:-1, :, None]
              + log_A[None, :, :]
              + log_emiss[1:, None, :]
              + log_beta[1:, None, :])
    # Normalise over (i, j) per t.
    flat = log_xi.reshape(T_minus_1, -1)
    log_xi = log_xi - _logsumexp(flat, axis=1)[:, None, None]
    return np.exp(log_xi)


# ---------------------------------------------------------------------------
# Viterbi (most-likely-path)
# ---------------------------------------------------------------------------


def viterbi(log_pi: np.ndarray, log_A: np.ndarray,
            log_emiss: np.ndarray) -> tuple[np.ndarray, float]:
    """Most likely state sequence given observations.

    Returns
    -------
    path : (T,) int array of state indices.
    log_p : float, ``log P(z_{1:T} = path, y_{1:T})``.
    """
    T, K = log_emiss.shape
    delta = np.full((T, K), NEG_INF, dtype=float)
    psi = np.zeros((T, K), dtype=np.int64)
    delta[0] = log_pi + log_emiss[0]
    for t in range(1, T):
        scores = delta[t - 1, :, None] + log_A  # (K_prev, K_next)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_emiss[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    log_p = float(delta[-1, path[-1]])
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])
    return path, log_p


__all__ = [
    "log_forward",
    "log_backward",
    "posterior_states",
    "posterior_transitions",
    "viterbi",
]
