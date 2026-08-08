# -*- coding: utf-8 -*-
"""Single source of truth for per-target supervised-label construction.

``run_experiment.run_animal_decoder`` calls into ``make_target`` for
the real-targets branch; the recovery branch (which loads predictions
from a base run as new targets) is handled separately in the runner.

All targets are pure-numpy. No torch dependency, so this module is
testable in isolation without a GPU/MPS environment.

Convention: column 0 of any 2-bin target is P(Go); first axis of any
91-bin target is the orientation grid 0..90 degrees in 1-degree steps.
Under ``target_source='io_hmm_pkl'`` the 'perception' target is instead
72 CIRCULAR 2.5-deg bins spanning [0, 180) (``io_hmm_data.GRID_DEG_IO``);
the perception/decision branches are pure pass-throughs, so no code here
assumes the 91-bin grid for them (GRID_DEG serves stim_kernel/stim_cat only).
"""

from __future__ import annotations

import numpy as np


GRID_DEG = np.arange(0, 91, 1, dtype=float)
STIM_KERNEL_SIGMA_DEG = 5.0  # Gaussian width for stim_kernel target


def make_target(which_model: str, trials: dict,
                 targets_perc=None, targets_dec=None, targets_lik=None
                 ) -> np.ndarray:
    """Construct the per-trial supervised target for a given decoding job.

    Parameters
    ----------
    which_model : str
        Legacy name as used by ``run_experiment.run_animal_decoder``:
        ``perception`` (Q), ``likelihood`` (L), ``decision`` (d),
        ``true_choice`` (animal goChoice), ``stim_kernel`` (Gaussian
        smoothed), ``stim_cat`` (one-hot).
    trials : dict
        Per-trial metadata with at least 'orientation' and 'choice' keys
        (as returned by ``utils.load_vr_export``).
    targets_perc, targets_dec, targets_lik : np.ndarray or None
        Pre-loaded IO target arrays for Q, d, L respectively. Required
        only for the corresponding which_model values.

    Returns
    -------
    np.ndarray
        ``(n_trials, n_bins)`` float array, normalised so each row sums
        to 1 (probabilities) for distributional targets, and one-hot
        for categorical targets.
    """
    if which_model == 'perception':
        if targets_perc is None:
            raise ValueError("'perception' requires targets_perc.")
        return np.asarray(targets_perc, dtype=float)

    if which_model == 'likelihood':
        if targets_lik is None:
            raise ValueError(
                "'likelihood' requires targets_lik (L_s_marginal). "
                "Re-run the IO export to populate this field."
            )
        return np.asarray(targets_lik, dtype=float)

    if which_model == 'decision':
        if targets_dec is None:
            raise ValueError("'decision' requires targets_dec.")
        return np.asarray(targets_dec, dtype=float)

    if which_model == 'true_choice':
        # Animal's actual binary goChoice, one-hot encoded as
        # [P(Go), P(NoGo)] = [c, 1-c]. Bypasses the IO entirely.
        c = np.asarray(trials['choice']).astype(float).reshape(-1)
        return np.column_stack([c, 1.0 - c])

    if which_model == 'stim_kernel':
        # Gaussian-smoothed delta at the true orientation. Output shape
        # mirrors Q (91-D) so neural-state-space comparisons with the Q
        # decoder are apples-to-apples (same loss, same arch, same
        # target shape). Sigma fixed at STIM_KERNEL_SIGMA_DEG.
        true_ori = np.asarray(trials['orientation'], dtype=float).reshape(-1)
        target = np.exp(
            -0.5 * ((GRID_DEG[None, :] - true_ori[:, None]) / STIM_KERNEL_SIGMA_DEG) ** 2
        )
        # Normalise each row to sum to 1.
        s = target.sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        return target / s

    if which_model == 'stim_cat':
        # One-hot at the true orientation bin (categorical, CE loss).
        true_ori = np.asarray(trials['orientation']).astype(float).reshape(-1)
        # Round to nearest integer degree, clip to grid.
        bin_idx = np.clip(np.rint(true_ori).astype(int), 0, 90)
        target = np.zeros((len(bin_idx), len(GRID_DEG)))
        target[np.arange(len(bin_idx)), bin_idx] = 1.0
        return target

    if which_model == 'detection':
        # Historical alias: targets the IO's decision_posterior. Kept
        # for backward compatibility with already-saved population_results
        # files. Use 'true_choice' for the actually-meaningful equivalent.
        if targets_dec is None:
            raise ValueError("'detection' requires targets_dec.")
        return np.asarray(targets_dec, dtype=float)

    raise ValueError(f"Unknown which_model {which_model!r}")


def n_bins_for(which_model: str) -> int:
    """How many output bins a given target produces."""
    if which_model in ('perception', 'likelihood', 'stim_kernel', 'stim_cat'):
        return 91
    if which_model in ('decision', 'true_choice', 'detection'):
        return 2
    raise ValueError(f"Unknown which_model {which_model!r}")
