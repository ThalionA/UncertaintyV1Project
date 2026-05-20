"""Trial-weighted hierarchical aggregation for the per-session similarity analysis.

Pure-numpy helpers — no plotting, no project imports — so they are cheap to
import and straightforward to unit-test on synthetic data with known ground
truth.

The per-session analysis produces a value per session (or per session ×
grouping-level). Results are combined in two stages:

    trial   -> session   (done inside each per-session metric)
    session -> mouse     ``session_to_mouse``  — weight = trials per session
    mouse   -> group     ``mouse_to_group``    — weight = trials per mouse

Both stages use **trial-count weights** (this project chose trial-count
weighting over equal weighting, so sessions / mice with more data count
more). The across-mice error bar is a *weighted* SEM based on Kish's
effective sample size

    n_eff = (Sum w)^2 / Sum w^2

so that with equal weights it reduces *exactly* to the ordinary mean and
``std(ddof=1) / sqrt(n)``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "weighted_nanmean",
    "weighted_mean_sem",
    "session_to_mouse",
    "mouse_to_group",
    "stability_vs_lag",
    "session_usable",
]


def _broadcast_weights(weights: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a 1-D weight vector to broadcast against an ndim-D value array."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return w.reshape((-1,) + (1,) * (ndim - 1))


def weighted_nanmean(values, weights) -> np.ndarray:
    """Weighted mean along axis 0, ignoring non-finite entries.

    Parameters
    ----------
    values : array-like, shape (n, *dims)
        Stack of ``n`` items to combine. Non-finite entries are dropped
        element-wise — they contribute zero weight at that position, so a
        partially-NaN stack still yields a valid mean where any item is
        finite.
    weights : array-like, shape (n,)
        Per-item weights (e.g. trial counts). Must be non-negative.

    Returns
    -------
    np.ndarray, shape (*dims)
        Weighted mean. Output positions with zero total weight are NaN.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        raise ValueError("values must have at least one axis")
    w = _broadcast_weights(weights, values.ndim)
    if w.shape[0] != values.shape[0]:
        raise ValueError(
            f"weights length {w.shape[0]} != values axis-0 {values.shape[0]}")
    finite = np.isfinite(values)
    w_eff = np.where(finite, w, 0.0)
    v0 = np.where(finite, values, 0.0)
    W = w_eff.sum(axis=0)
    num = (w_eff * v0).sum(axis=0)
    out = np.full(np.shape(W), np.nan)
    ok = W > 0
    out[ok] = num[ok] / W[ok]
    return out


def weighted_mean_sem(values, weights):
    """Weighted mean and weighted SEM across axis 0, NaN-aware.

    The SEM uses Kish's effective sample size ``n_eff = (Sum w)^2 / Sum w^2``.
    The weighted unbiased variance is

        var = Sum w (v - mu)^2 / (Sum w - Sum w^2 / Sum w)

    and ``sem = sqrt(var / n_eff)``. With equal weights this reduces exactly
    to the ordinary mean and ``std(ddof=1) / sqrt(n)``.

    Parameters
    ----------
    values : array-like, shape (n, *dims)
    weights : array-like, shape (n,)
        Non-negative per-item weights.

    Returns
    -------
    (mean, sem, n_eff) : tuple of np.ndarray, each shape (*dims)
        ``sem`` and the variance are NaN where fewer than one effective
        sample contributed (n_eff <= 1); ``mean`` is still valid there.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        raise ValueError("values must have at least one axis")
    w = _broadcast_weights(weights, values.ndim)
    if w.shape[0] != values.shape[0]:
        raise ValueError(
            f"weights length {w.shape[0]} != values axis-0 {values.shape[0]}")
    finite = np.isfinite(values)
    w_eff = np.where(finite, w, 0.0)
    v0 = np.where(finite, values, 0.0)
    W = w_eff.sum(axis=0)
    W2 = (w_eff ** 2).sum(axis=0)

    mean = np.full(np.shape(W), np.nan)
    sem = np.full(np.shape(W), np.nan)
    n_eff = np.full(np.shape(W), np.nan)

    ok = W > 0
    mean[ok] = (w_eff * v0).sum(axis=0)[ok] / W[ok]
    n_eff[ok] = W[ok] ** 2 / W2[ok]

    # Weighted unbiased variance: only meaningful when n_eff > 1.
    diff2 = (values - mean[None]) ** 2
    resid2 = np.where(w_eff > 0, w_eff * diff2, 0.0)
    ssr = resid2.sum(axis=0)
    denom = np.full(np.shape(W), np.nan)
    denom[ok] = W[ok] - W2[ok] / W[ok]          # == W * (n_eff - 1) / n_eff
    good = ok & (denom > 0)
    var = np.full(np.shape(W), np.nan)
    var[good] = ssr[good] / denom[good]
    sem[good] = np.sqrt(var[good] / n_eff[good])
    return mean, sem, n_eff


def session_to_mouse(session_values, session_trial_counts) -> np.ndarray:
    """Combine per-session values into one per-mouse value (trial-weighted).

    Thin, self-documenting wrapper over :func:`weighted_nanmean` — sessions
    are weighted by their trial counts.
    """
    return weighted_nanmean(session_values, session_trial_counts)


def mouse_to_group(mouse_values, mouse_trial_totals):
    """Combine per-mouse values into a group value (trial-weighted).

    Thin wrapper over :func:`weighted_mean_sem` — mice are weighted by their
    total trial counts. Returns ``(mean, sem, n_eff)``.
    """
    return weighted_mean_sem(mouse_values, mouse_trial_totals)


def stability_vs_lag(pairwise_matrix):
    """Aggregate a pairwise-similarity matrix by session lag.

    Parameters
    ----------
    pairwise_matrix : array-like, shape (n, n)
        Symmetric matrix of pairwise similarities; entry ``[i, j]`` compares
        the session at rank ``i`` with the session at rank ``j``. The
        diagonal and any missing pairs should be NaN.

    Returns
    -------
    (lags, mean, sem, n_pairs) : tuple of np.ndarray
        ``lags`` runs ``1 .. n-1``. ``mean`` / ``sem`` are the mean and
        standard error of the similarity over all session pairs that many
        ranks apart; ``n_pairs`` counts the finite pairs at each lag. Lags
        with no finite pair give NaN mean / sem and a zero count. With
        ``n < 2`` all four arrays are empty.
    """
    M = np.asarray(pairwise_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {M.shape}")
    n = M.shape[0]
    lags = np.arange(1, n)
    mean = np.full(max(n - 1, 0), np.nan)
    sem = np.full(max(n - 1, 0), np.nan)
    n_pairs = np.zeros(max(n - 1, 0), dtype=int)
    for k, d in enumerate(lags):
        vals = np.array([M[i, i + d] for i in range(n - d)], dtype=float)
        vals = vals[np.isfinite(vals)]
        n_pairs[k] = vals.size
        if vals.size > 0:
            mean[k] = float(np.mean(vals))
        if vals.size > 1:
            sem[k] = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    return lags, mean, sem, n_pairs


def session_usable(archetypes: dict) -> bool:
    """True if an archetype dict carries both a Go and a NoGo prototype.

    A session that fails this test could not raise the minimum number of
    easy trials on one or both sides, so it cannot contribute archetypes,
    SI, or any downstream per-session metric.
    """
    return "Go" in archetypes and "NoGo" in archetypes
