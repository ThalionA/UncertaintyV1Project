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
