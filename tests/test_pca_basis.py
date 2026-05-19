# -*- coding: utf-8 -*-
"""Tests for the pca_basis flag in run_experiment_v26.run_animal_decoder.

Re-implementing the small PCA-fit block in isolation to test numerical
correctness without spinning up the full torch training loop. The
implementation under test (in run_experiment_v26.py) is duplicated here
to avoid pulling torch / scipy / the full module import chain.

The end-to-end contract (Config -> to_legacy_dict -> run_animal_decoder
reads pca_basis -> branches the PCA fit) is exercised by the schema
checks in test_training_config.py; this file tests *what* the basis
captures on synthetic Q's with known structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.decomposition import PCA


def _fit_pca_basis(training_posteriors: np.ndarray,
                   stim_conditions_train: np.ndarray,
                   pca_basis: str) -> np.ndarray:
    """Re-implementation of the run_experiment_v26 PCA-basis fit.

    Mirrors the production code in nn_decoder/run_experiment_v26.py
    (the block guarded by `if N_cats > 2`). Returns the PCA components
    (shape n_components x n_bins) for the requested basis.
    """
    unique_train_categories = np.unique(stim_conditions_train, axis=0)
    averaged_distributions = np.zeros(
        (len(unique_train_categories), training_posteriors.shape[1])
    )
    for i, stimulus in enumerate(unique_train_categories):
        idx = np.where(np.all(stim_conditions_train == stimulus, axis=1))[0]
        averaged_distributions[i] = training_posteriors[idx].mean(axis=0)

    if pca_basis == 'condition_mean':
        pca_input = averaged_distributions
    elif pca_basis == 'residual':
        cell_mean_per_trial = np.zeros_like(training_posteriors)
        keep_mask = np.zeros(len(training_posteriors), dtype=bool)
        for i, stimulus in enumerate(unique_train_categories):
            idx = np.where(np.all(stim_conditions_train == stimulus, axis=1))[0]
            if len(idx) >= 2:
                cell_mean_per_trial[idx] = training_posteriors[idx].mean(axis=0)
                keep_mask[idx] = True
            # else: leave residual = 0 (excluded from PCA basis).
        residuals = training_posteriors - cell_mean_per_trial
        residuals[~keep_mask] = 0.0
        pca_input = residuals
    else:
        raise ValueError(pca_basis)

    pca = PCA()
    pca.fit(pca_input)
    return pca.components_


def _plant_targets(rng, n_cells=12, n_trials_per_cell=30, n_bins=91,
                    shift_axis=None, wiggle_axis=None,
                    shift_scale=1.0, wiggle_scale=0.3):
    """Construct synthetic training targets with KNOWN structure.

    Each of n_cells stimulus cells gets a mean = (cell_index * shift_axis).
    Each trial within the cell gets an additional (gaussian noise) *
    wiggle_axis perturbation. So:

      Between-cell variance: lives entirely along shift_axis.
      Within-cell variance:  lives entirely along wiggle_axis.

    A well-designed condition_mean basis should put its leading PC on
    shift_axis; a residual basis should put its leading PC on
    wiggle_axis. The two axes are constructed orthonormal — naive
    indicator-and-mean-centre construction does NOT yield orthogonality
    in general (the centring constant mixes them), so we explicitly
    Gram-Schmidt the wiggle axis against the shift axis.
    """
    if shift_axis is None:
        shift_axis = np.zeros(n_bins)
        shift_axis[10:30] = 1.0
        shift_axis -= shift_axis.mean()
        shift_axis /= np.linalg.norm(shift_axis)
    if wiggle_axis is None:
        wiggle_axis = np.zeros(n_bins)
        wiggle_axis[55:75] = 1.0
        wiggle_axis -= wiggle_axis.mean()
        # Gram-Schmidt against shift_axis: subtract the shift component
        # before normalising. Without this step the two "indicator-bump"
        # axes have inner product ~-0.28 because their centring constants
        # mix.
        wiggle_axis -= np.dot(wiggle_axis, shift_axis) * shift_axis
        wiggle_axis /= np.linalg.norm(wiggle_axis)

    n_train = n_cells * n_trials_per_cell
    posteriors = np.zeros((n_train, n_bins))
    conditions = np.zeros((n_train, 3))
    for c in range(n_cells):
        cell_mean = c * shift_scale * shift_axis
        for k in range(n_trials_per_cell):
            i = c * n_trials_per_cell + k
            posteriors[i] = cell_mean + rng.normal(scale=wiggle_scale) * wiggle_axis
            # Use (o, c, d) = (c, 0, 0) so every cell has a unique triplet.
            conditions[i] = [float(c), 0.0, 0.0]
    return posteriors, conditions, shift_axis, wiggle_axis


def _cosine(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))


def test_condition_mean_basis_leading_pc_aligns_with_shift_axis():
    """condition_mean basis should put its first PC along the
    across-cell shift direction, ignoring the within-cell wiggle."""
    rng = np.random.default_rng(0)
    posteriors, conditions, shift_axis, wiggle_axis = _plant_targets(rng)
    components = _fit_pca_basis(posteriors, conditions, 'condition_mean')
    cos_shift = _cosine(components[0], shift_axis)
    cos_wiggle = _cosine(components[0], wiggle_axis)
    assert cos_shift > 0.95, (
        f"condition_mean leading PC should track shift axis; got |cos|={cos_shift:.3f}"
    )
    assert cos_wiggle < 0.1, (
        f"condition_mean leading PC should NOT track wiggle; got |cos|={cos_wiggle:.3f}"
    )


def test_residual_basis_leading_pc_aligns_with_wiggle_axis():
    """residual basis should put its first PC along the within-cell
    wiggle direction, ignoring the across-cell shift."""
    rng = np.random.default_rng(0)
    posteriors, conditions, shift_axis, wiggle_axis = _plant_targets(rng)
    components = _fit_pca_basis(posteriors, conditions, 'residual')
    cos_shift = _cosine(components[0], shift_axis)
    cos_wiggle = _cosine(components[0], wiggle_axis)
    assert cos_wiggle > 0.95, (
        f"residual leading PC should track wiggle axis; got |cos|={cos_wiggle:.3f}"
    )
    assert cos_shift < 0.1, (
        f"residual leading PC should NOT track shift; got |cos|={cos_shift:.3f}"
    )


def test_residual_basis_singleton_cells_are_excluded_from_pca():
    """Cells with only 1 training trial cannot be used to estimate a
    within-cell deviation (the residual would be exactly 0). The
    implementation zeros out the singleton's residual row so it
    contributes nothing to the PCA. An earlier global-mean-fallback
    design let singletons that sat far from the global mean inject an
    outlier residual that dominated the leading PC and re-introduced
    the across-condition shift the residual basis was supposed to
    filter out — see the regression
    test_residual_basis_with_singleton_doesnt_introduce_shift_signal_for_well_populated_cells."""
    rng = np.random.default_rng(1)
    # 10 trials in cell 0, 1 trial in cell 1
    n_bins = 91
    shift_axis = np.zeros(n_bins); shift_axis[10:30] = 1.0
    shift_axis -= shift_axis.mean(); shift_axis /= np.linalg.norm(shift_axis)
    wiggle_axis = np.zeros(n_bins); wiggle_axis[55:75] = 1.0
    wiggle_axis -= wiggle_axis.mean()
    wiggle_axis -= np.dot(wiggle_axis, shift_axis) * shift_axis  # orthogonalise
    wiggle_axis /= np.linalg.norm(wiggle_axis)

    posteriors = np.zeros((11, n_bins))
    conditions = np.zeros((11, 3))
    for k in range(10):
        posteriors[k] = 0.0 + rng.normal(scale=0.3) * wiggle_axis
        conditions[k] = [0.0, 0.0, 0.0]
    # singleton cell with a distinct mean far from the bulk
    posteriors[10] = 2.0 * shift_axis
    conditions[10] = [1.0, 0.0, 0.0]

    components = _fit_pca_basis(posteriors, conditions, 'residual')
    # The singleton's residual row is exactly 0, so the PCA basis is
    # dominated by the well-populated cell's within-cell variation.
    # Verify the basis is finite and the leading PC tracks the wiggle
    # axis (i.e. the singleton didn't contaminate the basis).
    assert np.isfinite(components).all()
    cos_wiggle = _cosine(components[0], wiggle_axis)
    cos_shift = _cosine(components[0], shift_axis)
    assert cos_wiggle > 0.9, (
        f"Leading PC should track wiggle (singleton contributes nothing); "
        f"got |cos vs wiggle|={cos_wiggle:.3f}"
    )
    assert cos_shift < 0.1, (
        f"Leading PC should NOT track shift; got |cos vs shift|={cos_shift:.3f}"
    )


def test_residual_basis_with_singleton_doesnt_introduce_shift_signal_for_well_populated_cells():
    """An earlier design subtracted the global training mean for singleton
    cells, which let a singleton sitting far from the global mean inject
    a large outlier residual that dominated the leading PC and re-
    introduced the across-condition shift. The current design zeros out
    singleton residuals; this test plants a singleton at 10*shift_axis
    (far from the rest of the data) and confirms the leading PC still
    tracks the wiggle axis."""
    rng = np.random.default_rng(2)
    n_bins = 91
    shift_axis = np.zeros(n_bins); shift_axis[10:30] = 1.0
    shift_axis -= shift_axis.mean(); shift_axis /= np.linalg.norm(shift_axis)
    wiggle_axis = np.zeros(n_bins); wiggle_axis[55:75] = 1.0
    wiggle_axis -= wiggle_axis.mean()
    wiggle_axis -= np.dot(wiggle_axis, shift_axis) * shift_axis  # orthogonalise
    wiggle_axis /= np.linalg.norm(wiggle_axis)

    rows = []
    conds = []
    for c in range(5):
        cell_mean = c * shift_axis
        for k in range(30):
            rows.append(cell_mean + rng.normal(scale=0.3) * wiggle_axis)
            conds.append([float(c), 0.0, 0.0])
    # plus one singleton cell
    rows.append(10.0 * shift_axis)
    conds.append([99.0, 0.0, 0.0])

    posteriors = np.array(rows)
    conditions = np.array(conds)
    components = _fit_pca_basis(posteriors, conditions, 'residual')
    # The leading PC should still be the wiggle axis (the well-populated
    # cells dominate). The singleton's residual is one outlier row that
    # PCA will treat as an outlier rather than the dominant pattern.
    cos_wiggle = _cosine(components[0], wiggle_axis)
    assert cos_wiggle > 0.7, (
        f"singleton contamination shouldn't dominate; leading PC |cos vs wiggle|={cos_wiggle:.3f}"
    )


def test_unknown_pca_basis_raises():
    rng = np.random.default_rng(3)
    posteriors, conditions, _, _ = _plant_targets(rng, n_cells=3)
    with pytest.raises(ValueError):
        _fit_pca_basis(posteriors, conditions, 'garbage')
