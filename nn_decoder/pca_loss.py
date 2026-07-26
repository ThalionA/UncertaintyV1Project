# -*- coding: utf-8 -*-
"""Canonical PCA-weighted fit-loss for the neural decoder.

This module is the **single source of truth** for the PCA-based loss.
It replaces seven byte-similar copies of the same formula that had
drifted across the codebase (``decoder_plotting_utils``,
``run_fixed_recovery``, ``quick_diagonal_loss_check``,
``time_binned_ppc`` and five files under ``legacy/``).

The method
----------
For one trial's predicted distribution ``pred`` and target distribution
``target`` (each a vector of length ``n_cats`` — e.g. 91 angle bins):

  1. Project both onto the PC basis:  ``proj = X @ pcs.T``.
  2. Take the squared difference per component, weighted by that PC's
     explained-variance ratio:  ``evar_k * (proj_pred_k - proj_target_k)**2``.
  3. Sum over components.

      loss = sum_k  evar_k * (proj_pred_k - proj_target_k)**2

  (An overall ×100 scale was dropped 2026-06-10 — it was arbitrary and, with the
  Adam optimiser, model-neutral; skill/ratio metrics are scale-invariant anyway.)

Because ``PCA()`` keeps every component, the projection is a pure
rotation; the ``evar`` weighting is what distinguishes this from plain
MSE — it is a Mahalanobis-like distance that emphasises the dominant
axes of the target manifold.

The PC basis itself (``pcs``, ``evar``) is fit elsewhere — on
condition-averaged target distributions (``run_experiment.py`` /
``optuna_per_target.py`` / ``time_binned_ppc.fit_pca_basis``). This
module only consumes a basis; it never fits one.

No basis => raise
-----------------
``pcs is None`` (or an empty array) means PCA loss is undefined. That
happens only for targets with <=2 categories (the 2-D decision target),
which are trained with an explicit MSE loss instead, or for a session
with too few stimulus conditions to fit a basis. In every correct
configuration a 'PCA' loss request has a basis. ``pca_distance``
therefore **raises** on a missing basis rather than silently returning
MSE, cross-entropy or NaN — a silent substitution would let a run
optimise or report a different metric than its label claims.

Callers that legitimately iterate over sessions where some lack a basis
(e.g. plotting loops) must guard the call explicitly — see
``decoder_plotting_utils.calc_pca_dist``, the plotting-layer wrapper
that returns an all-NaN per-trial array so ``np.nanmean`` skips that
cell.

Torch twin
----------
The training loop needs an autograd-friendly version, so the identical
formula also exists in torch inside
``nn_classifier.custom_loss_all_H`` (the ``loss_func_type == 'PCA'``
branch) and ``optuna_per_target.marginal_baseline_loss``. A numpy/torch
agreement test (``tests/test_pca_loss.py``) pins that the two stay
numerically identical — if you change the formula here, change it there
too.
"""

from __future__ import annotations

import numpy as np


def _has_basis(arr) -> bool:
    """True when ``arr`` is a usable PCA-basis component — i.e. not
    ``None`` and not an empty array/list. A basis-less cell is stored in
    the .mat outputs as an empty array, so both forms must be caught."""
    if arr is None:
        return False
    return np.asarray(arr).size > 0


def pca_distance(pred, target, pcs, evar):
    """Per-trial PCA-weighted squared distance between ``pred`` and
    ``target``.

    Parameters
    ----------
    pred, target : np.ndarray
        Either 2-D ``(n_trials, n_cats)`` or 3-D
        ``(n_trials, n_cats, t_bins)``. The function is symmetric in the
        two arguments (it is a squared difference), so decoded/target
        order does not affect the result.
    pcs : np.ndarray
        Principal components, shape ``(n_components, n_cats)``.
    evar : np.ndarray
        Explained-variance ratio per component, shape ``(n_components,)``.

    Returns
    -------
    np.ndarray
        Per-trial loss: shape ``(n_trials,)`` for 2-D input, or
        ``(n_trials, t_bins)`` for 3-D input.

    Raises
    ------
    ValueError
        If ``pcs`` or ``evar`` is ``None`` or empty. PCA loss is
        undefined without a basis; see the module docstring.
    """
    if not (_has_basis(pcs) and _has_basis(evar)):
        raise ValueError(
            "pca_distance: no PCA basis supplied (pcs/evar is None or "
            "empty). PCA loss is undefined without a basis. It is only "
            "used for multi-dimensional targets (>2 categories) — for 2-D "
            "targets request an MSE loss explicitly. If you are iterating "
            "over sessions/mice and some legitimately lack a basis, guard "
            "the call with `if pcs is None` and record NaN for that cell."
        )

    pred = np.asarray(pred)
    target = np.asarray(target)
    pcs = np.asarray(pcs)
    evar = np.asarray(evar)

    if pred.ndim == 3:
        # (n_trials, n_cats, t_bins) -> project the n_cats axis.
        proj_pred = np.einsum('nct,kc->nkt', pred, pcs)
        proj_target = np.einsum('nct,kc->nkt', target, pcs)
        evar_expand = evar[np.newaxis, :, np.newaxis]
        return np.sum(evar_expand * (proj_pred - proj_target) ** 2, axis=1)

    # 2-D: (n_trials, n_cats).
    proj_pred = np.dot(pred, pcs.T)
    proj_target = np.dot(target, pcs.T)
    return np.sum(evar * (proj_pred - proj_target) ** 2, axis=1)


def fit_loss(arch_dist, loss_func, pcs=None, evar=None):
    """Per-trial fit-loss recomputed from saved decoded/target arrays.

    This is the clean replacement for reading ``mouse_data['KLs'][arch]``
    directly: the legacy ``KLs`` field is contaminated for temporal
    (sampling) models — it carries the training-time entropy regulariser
    on top of the fit-loss, which has no place in a held-out test metric
    (see ``nn_decoder/audit/AUDIT_loss_consumers.md``). ``fit_loss``
    recomputes the pure fit-loss from ``arch_dist['decoded']`` and
    ``arch_dist['target']``, which were always saved cleanly.

    Parameters
    ----------
    arch_dist : dict
        Per-architecture sub-dict from a loaded .mat, e.g. ``Dist['spat']``.
        Must contain 'decoded' and 'target'.
    loss_func : {'PCA', 'MSE', 'CE'}
        Which fit-loss to compute — the one the model was trained on for
        this cell (read from ``mat['config']['custom_loss_func']``).
    pcs, evar : np.ndarray, optional
        Principal components and explained-variance ratios. Required for
        the 'PCA' branch; ignored otherwise. They live at ``Dist['pcs']``
        / ``Dist['explained_var']`` — the *parent* of ``arch_dist``.

    Returns
    -------
    np.ndarray
        Per-trial fit-loss, shape ``(n_trials,)``.

    Raises
    ------
    ValueError
        If ``loss_func`` is unknown, or 'PCA' is requested with no basis.
    """
    pred = np.asarray(arch_dist['decoded'])
    target = np.asarray(arch_dist['target'])

    if loss_func == 'PCA':
        return pca_distance(pred, target, pcs, evar)
    elif loss_func == 'MSE':
        return np.mean((pred - target) ** 2, axis=-1)
    elif loss_func == 'CE':
        # Matches nn_classifier.cross_entropy: -sum(target * log(decoded + eps)).
        # The eps MUST be float32 machine epsilon (1.19e-7), which is what the
        # training-side torch implementation uses (nn_classifier.py:20). This file
        # previously used 1e-12 under this same "matches" comment, so the scorer
        # disagreed with the loss it claims to reproduce by up to ~1.3 nats on
        # confidently-wrong bins (-log 1.19e-7 = 15.94 vs -log 1e-12 = 27.63).
        # NB this also means training-side CE SATURATES at 15.94 nats — loss and
        # gradient are capped for confidently-wrong predictions. That is a genuine
        # training-side defect (the fix is log_softmax on logits), but changing it
        # alters every future run, so it is flagged in documents/AUDIT_2026-07.md
        # rather than silently changed here. 2026-07 audit.
        return -np.sum(target * np.log(pred + np.finfo(np.float32).eps), axis=-1)
    else:
        raise ValueError(
            f"fit_loss: unknown loss_func {loss_func!r}; "
            f"supported: 'PCA', 'MSE', 'CE'."
        )
