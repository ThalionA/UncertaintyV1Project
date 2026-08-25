# -*- coding: utf-8 -*-
"""Per-trial loss/baseline metrics for the decoder figures.

These are the pure scoring helpers that used to live inside
``decoder_plotting_utils`` — loss maths has no business in a plotting file. They
depend only on :mod:`pca_loss` and numpy (no IO, no matplotlib), so they import
cheaply and are unit-tested directly (``tests/test_calc_fit_loss.py``,
``tests/test_variance_baseline.py``, ``tests/test_pca_loss.py``).

``decoder_plotting_utils`` re-exports every name here under its historical
spelling, so existing ``import decoder_plotting_utils as dpu`` /
``from decoder_plotting_utils import calc_fit_loss`` call sites are unchanged.

Two PCA-distance entry points with deliberately different missing-basis policy:
  * :func:`calc_pca_dist` — the NaN-tolerant *plotting* wrapper: a missing basis
    yields an all-NaN per-trial array so ``np.nanmean`` loops skip that cell.
  * :data:`calc_fit_loss` (= :func:`pca_loss.fit_loss`) — the strict dispatcher
    whose 'PCA' branch RAISES on a missing basis (a 'PCA' request always has a
    basis in correct configs, so a raise surfaces a genuine misconfiguration).
Neither ever substitutes a *different* metric (MSE/CE) for an absent basis.
"""

from __future__ import annotations

import numpy as np

from pca_loss import pca_distance, fit_loss


def calc_pca_dist(p, q, pcs, evar):
    """Plotting-layer wrapper around ``pca_loss.pca_distance``.

    A missing PCA basis yields an all-NaN per-trial array, so the per-mouse
    plotting loops (which reduce with ``np.nanmean``) skip that cell instead of
    raising. This is the *only* place that tolerates a missing basis by design;
    the strict, raising contract lives in ``pca_loss.pca_distance`` and is what
    the training loss and the test suite use. An absent basis becomes NaN, never
    another metric.
    """
    if pcs is None or (isinstance(pcs, (list, np.ndarray)) and len(pcs) == 0):
        return np.full(np.asarray(p).shape[0], np.nan)
    return pca_distance(p, q, pcs, evar)


# The PCA/MSE/CE fit-loss dispatcher lives in ``pca_loss``; re-exported here
# under its historical name for external importers (recovery_sanity_check,
# plot_post_fix_performance, run_recovery_Q_spyder) and the test suite. The
# 'PCA' branch RAISES on a missing basis (unlike the NaN-tolerant
# ``calc_pca_dist`` above).
calc_fit_loss = fit_loss


# ----------------------------------------------------------------------
# Normalisation modes — shuffle (historical default), variance
# (marginal-mean baseline; see optuna_per_target.marginal_baseline_loss),
# or raw (no division). The plotters share these so the three modes stay
# numerically consistent across plots.
# ----------------------------------------------------------------------

_NORM_SUFFIX = {'shuffle': '', 'variance': '_varnorm', 'raw': '_raw'}
_NORM_BASELINE_LABEL = {'shuffle': 'Shuffle baseline',
                        'variance': 'Marginal-mean baseline',
                        'raw': None}


def _normalize_mode(normalize) -> str:
    """Coerce the legacy bool flag and the new string flag to one of
    ``'shuffle'`` / ``'variance'`` / ``'raw'``.

    ``True`` → ``'shuffle'`` (the historical default), ``False`` → ``'raw'``.
    String values pass through verbatim. Anything else raises so a typo can't
    silently fall through to a wrong baseline.
    """
    if normalize is True:
        return 'shuffle'
    if normalize is False:
        return 'raw'
    s = str(normalize).lower()
    if s in _NORM_SUFFIX:
        return s
    raise ValueError(
        f"normalize must be True/False or one of "
        f"{tuple(_NORM_SUFFIX)}, got {normalize!r}"
    )


def variance_baseline(target_test, train_target_mean, pcs, evar):
    """Per-trial PCA-weighted squared distance from each test target to the
    train-target mean.

    Matches ``optuna_per_target.marginal_baseline_loss`` (same scale as
    ``pca_distance``) when averaged over trials::

        mean_t Σ_i evar_i (⟨P_t^test, PC_i⟩ − ⟨P̄^train, PC_i⟩)²

    This is the variance baseline drawn in the figure denominator — "predict the
    marginal-mean target" — and equals the test-set target variance projected on
    the PC basis when the train and test marginals are matched.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials,)``. NaN-filled when ``pcs`` or ``train_target_mean``
        is missing.
    """
    target_test = np.asarray(target_test)
    if pcs is None or np.asarray(pcs).size == 0:
        return np.full(target_test.shape[0], np.nan)
    if train_target_mean is None:
        return np.full(target_test.shape[0], np.nan)
    pcs = np.asarray(pcs)
    evar = np.asarray(evar)
    train_target_mean = np.asarray(train_target_mean)
    proj_test = target_test @ pcs.T               # (n_trials, n_components)
    proj_mean = train_target_mean @ pcs.T         # (n_components,)
    return np.sum(evar * (proj_test - proj_mean) ** 2, axis=-1)


# ----------------------------------------------------------------------
# Row-wise distribution metrics (2026-08-25 audit, item D1).
#
# THE canonical numpy implementations. Before this section existed, KL had
# drifted into 7 per-script copies and entropy into 11.
#
# ⚠ THE TWO EPS CONVENTIONS ARE DIFFERENT METRICS, NOT ROUNDING VARIANTS.
# Measured 2026-08-25 on peaked predictions against broad targets — the
# project's own over-sharpening regime — the mean per-trial KL is 17.83
# under 'clip' and 9.36 under 'additive': a factor of ~1.9.
#
#   'clip'      p -> max(p, 1e-12) inside the log. What all seven diagnostic
#               copies used, so it is what every KL number currently in the
#               figures, CSVs and vault notes was computed with. An almost-
#               zero prediction where the target has mass costs up to
#               log(1e-12) ≈ -27.6 nats: essentially unbounded, no saturation.
#   'additive'  log(p + 1.19e-7) — the float32 eps ADDED, which is what the
#               training side does (``nn_classifier.cross_entropy``). This
#               SATURATES: the same confidently-wrong bin costs at most
#               ≈ 15.9 nats, capping loss and gradient alike (the CE
#               saturation recorded in documents/AUDIT_2026-07.md, C4).
#
# Default is 'clip' — the historical scoring convention — so consolidating
# the copies did not silently move any published number. Pass
# ``eps_mode='additive'`` when you need the number the TRAINING loss saw
# (that path is pinned against the torch implementations in
# ``tests/test_decoder_metrics_rows.py``). Whichever you pick, say which one
# in the figure/caption: for sharp predictions the two disagree ~2x.
#
# All take (n, bins) arrays of row-normalised distributions, return (n,).
# ----------------------------------------------------------------------

EPS_ADDITIVE = float(np.finfo(np.float32).eps)   # training-side (saturating)
EPS_CLIP = 1e-12                                 # historical scoring-side


def _prep(x, eps_mode, eps):
    """Apply the chosen eps convention, returning (x_for_log, x_weight).

    The two conventions also differ in what they use as the OUTER weighting
    factor, and both are reproduced faithfully here:
      'clip'      weights by the CLIPPED value (what the replaced diagnostic
                  copies did: ``a = np.clip(a, 1e-12, None); (a * (log a - log b))``).
      'additive'  weights by the RAW value (what the training side does:
                  ``-sum(Y * log(X + eps))`` with Y unmodified).
    Ignoring this makes the clip path disagree with the historical numbers at
    ~1e-9 on rows containing exact zeros — small, but not zero, and the point
    of the consolidation is that it moves nothing.
    """
    x = np.asarray(x, float)
    if eps_mode == 'clip':
        xc = np.clip(x, EPS_CLIP if eps is None else eps, None)
        return xc, xc
    if eps_mode == 'additive':
        return x + (EPS_ADDITIVE if eps is None else eps), x
    raise ValueError(
        f"eps_mode must be 'clip' or 'additive', got {eps_mode!r}")


def cross_entropy_rows(pred, target, eps_mode='clip', eps=None):
    """Per-row cross-entropy −Σ target·log(pred). See the eps-convention note
    above; ``eps_mode='additive'`` is the twin of ``nn_classifier.cross_entropy``."""
    p_log, _ = _prep(pred, eps_mode, eps)
    _, t_weight = _prep(target, eps_mode, eps)
    return -np.sum(t_weight * np.log(p_log), axis=-1)


def entropy_rows(p, eps_mode='clip', eps=None):
    """Per-row Shannon entropy −Σ p·log(p). ``eps_mode='additive'`` is the
    twin of ``nn_classifier.entropy_calc``."""
    p_log, p_w = _prep(p, eps_mode, eps)
    return -np.sum(p_w * np.log(p_log), axis=-1)


def kl_rows(pred, target, eps_mode='clip', eps=None):
    """Per-row forward KL D(target ‖ pred), clamped at 0. Argument order is
    prediction first, target second — matching ``nn_classifier.KL_calc``
    (note the seven local copies this replaces used the OPPOSITE order,
    ``_kl(target, decoded)``; call sites were swapped when ported).

    With ``eps_mode='additive'`` this is the numpy twin of
    ``nn_classifier.KL_calc`` / ``fit_loss_per_trial(..., 'KL')``. With the
    default ``'clip'`` it reproduces the historical diagnostic scoring."""
    ce = cross_entropy_rows(pred, target, eps_mode, eps)
    h = entropy_rows(target, eps_mode, eps)
    return np.clip(ce - h, 0.0, None)


def js_rows(pred, target, eps_mode='clip', eps=None):
    """Per-row Jensen–Shannon divergence, built from the clamped KLs exactly
    as ``nn_classifier.JS_calc`` does."""
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    m = 0.5 * (pred + target)
    return (0.5 * kl_rows(m, pred, eps_mode, eps)
            + 0.5 * kl_rows(m, target, eps_mode, eps))


def peakiness_rows(p):
    """Per-row peakiness = max probability across bins (the primitive under
    ``diagnostics/story_figures.peaky``, which is ``peakiness_rows(...).mean()``
    per mouse)."""
    return np.asarray(p, float).max(axis=-1)


def mean_sem(x, axis=None):
    """(mean, sem) with ddof=1 — the repo's canonical error-bar convention
    (July 2026 audit item M3: ddof=0 copies made n=6 error bars ~9.5% small).
    NaN-tolerant: uses nanmean/nanstd and counts only finite entries."""
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    mean = np.nanmean(x, axis=axis)
    sem = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)
    return mean, sem


def get_mouse_pca_losses(res_dict, arch_key, target_key='target'):
    """Concatenate per-trial PCA losses across every mouse in a loaded run.

    Uses the NaN-tolerant :func:`calc_pca_dist`, so mice/cells without a PCA
    basis contribute NaNs (skipped by downstream ``np.nanmean``) rather than
    raising.
    """
    all_losses = []
    for _m_id, m_data in res_dict['results'].items():
        dist = m_data['Dist']
        pcs = dist.get('pcs', None)
        evar = dist.get('explained_var', None)
        loss = calc_pca_dist(dist[arch_key][target_key], dist[arch_key]['decoded'], pcs, evar)
        all_losses.append(loss)
    return np.concatenate(all_losses) if all_losses else np.array([])
