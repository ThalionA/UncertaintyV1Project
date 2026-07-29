# -*- coding: utf-8 -*-
"""Tests for the n_neural_pcs flag — the INPUT-side (neural) PCA.

Asked for by Máté 2026-07-29 ("do PCA on neural resp. and decode from PCs;
also decode from different n of PCs").

Following the convention of test_pca_basis.py: the small projection block in
run_experiment.run_animal_decoder is re-implemented here in isolation so the
numerics can be tested without spinning up torch / the full training loop.
The Config schema half of the contract (field default, validation,
to_legacy_dict threading) is tested directly against the real Config.

The property that actually matters scientifically is the LEAKAGE one:
the basis must be fit on training trials only, exactly like the per-neuron
z-scoring it sits next to.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from sklearn.decomposition import PCA

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

from dataclasses import replace  # noqa: E402

from training.config import Config, default_config_for_target  # noqa: E402


def _project(activities_z: np.ndarray, train_indices: np.ndarray,
             n_neural_pcs):
    """Re-implementation of the run_experiment neural-PCA block.

    Mirrors nn_decoder/run_experiment.py. ``activities_z`` is
    (nNeurons, nTrials, tBins). Returns (projected, retained_evr) with
    projected shape (k, nTrials, tBins), or the input unchanged when
    ``n_neural_pcs`` is None.
    """
    if n_neural_pcs is None:
        return activities_z, None
    n_neur = activities_z.shape[0]
    train_flat = activities_z[:, train_indices, :].reshape(n_neur, -1).T
    k = int(min(int(n_neural_pcs), train_flat.shape[0], n_neur))
    pca = PCA(n_components=k)
    pca.fit(train_flat)
    all_flat = activities_z.reshape(n_neur, -1).T
    proj = pca.transform(all_flat)
    out = proj.T.reshape(k, activities_z.shape[1], activities_z.shape[2])
    return out, float(pca.explained_variance_ratio_.sum())


@pytest.fixture
def toy():
    """20 neurons x 40 trials x 5 bins, with only 3 real latent dimensions."""
    rng = np.random.default_rng(0)
    n_neur, n_trials, n_bins, rank = 20, 40, 5, 3
    latents = rng.normal(size=(rank, n_trials * n_bins))
    mixing = rng.normal(size=(n_neur, rank))
    flat = mixing @ latents + 0.01 * rng.normal(size=(n_neur, n_trials * n_bins))
    acts = flat.reshape(n_neur, n_trials, n_bins)
    train_indices = np.arange(0, n_trials, 2)      # every other trial
    return acts, train_indices


# --------------------------------------------------------------- schema ----

def test_default_is_none_so_production_runs_are_unchanged():
    assert Config.__dataclass_fields__['n_neural_pcs'].default is None
    cfg = default_config_for_target('Q')
    assert cfg.n_neural_pcs is None
    assert cfg.to_legacy_dict()['n_neural_pcs'] is None


def test_value_is_threaded_into_the_legacy_dict():
    cfg = replace(default_config_for_target('Q'), n_neural_pcs=8)
    assert cfg.to_legacy_dict()['n_neural_pcs'] == 8


@pytest.mark.parametrize('bad', [0, -1, -32])
def test_non_positive_is_rejected(bad):
    with pytest.raises(ValueError, match='n_neural_pcs'):
        replace(default_config_for_target('Q'), n_neural_pcs=bad)


def test_is_independent_of_pca_basis():
    """Input-side and target-side PCA are orthogonal knobs; both can be set."""
    cfg = replace(default_config_for_target('Q'),
                  n_neural_pcs=4, pca_basis='residual')
    legacy = cfg.to_legacy_dict()
    assert legacy['n_neural_pcs'] == 4
    assert legacy['pca_basis'] == 'residual'


# ------------------------------------------------------------ numerics ----

def test_none_is_a_bit_identical_no_op(toy):
    acts, train_indices = toy
    out, evr = _project(acts, train_indices, None)
    assert evr is None
    assert out is acts


def test_projection_sets_the_input_dimension(toy):
    acts, train_indices = toy
    for k in (1, 3, 8, 20):
        out, _ = _project(acts, train_indices, k)
        assert out.shape == (k, acts.shape[1], acts.shape[2])


def test_k_is_clamped_to_available_components(toy):
    """Asking for more PCs than neurons yields n_neurons, not an error."""
    acts, train_indices = toy
    out, evr = _project(acts, train_indices, 500)
    assert out.shape[0] == acts.shape[0] == 20
    assert evr == pytest.approx(1.0, abs=1e-6)


def test_rank_3_data_is_captured_by_3_pcs(toy):
    """Sanity: the toy data has 3 latent dimensions, so 3 PCs retain ~all of it."""
    acts, train_indices = toy
    _, evr = _project(acts, train_indices, 3)
    assert evr > 0.999


def test_trial_and_bin_structure_survives_the_reshape_round_trip(toy):
    """The (k, nTrials, tBins) reshape must not scramble trials against bins.

    With k = n_neurons the PCA is a pure rotation, so projecting and then
    inverting must return the original array element-for-element. That
    round-trip is only true if the flatten/unflatten ordering is consistent.
    """
    acts, train_indices = toy
    n_neur = acts.shape[0]
    train_flat = acts[:, train_indices, :].reshape(n_neur, -1).T
    pca = PCA(n_components=n_neur)
    pca.fit(train_flat)

    out, _ = _project(acts, train_indices, n_neur)
    back_flat = pca.inverse_transform(out.reshape(n_neur, -1).T)
    back = back_flat.T.reshape(acts.shape)
    np.testing.assert_allclose(back, acts, atol=1e-8)


# ------------------------------------------------------------- leakage ----

def test_basis_is_fit_on_training_trials_only(toy):
    """THE leakage pin: the projection must not depend on held-out trials.

    Corrupting the non-training trials with huge values must leave the
    training trials' PC scores untouched. If the basis were fit on all
    trials, those scores would move.
    """
    acts, train_indices = toy
    test_mask = np.ones(acts.shape[1], dtype=bool)
    test_mask[train_indices] = False

    clean, _ = _project(acts, train_indices, 3)

    corrupted = acts.copy()
    corrupted[:, test_mask, :] += 1000.0
    dirty, _ = _project(corrupted, train_indices, 3)

    np.testing.assert_allclose(dirty[:, train_indices, :],
                               clean[:, train_indices, :], atol=1e-6)


def test_held_out_trials_are_projected_not_dropped(toy):
    """Test trials must still come out the other side, transformed by the
    train-fit basis — the point is to project them, not to exclude them."""
    acts, train_indices = toy
    out, _ = _project(acts, train_indices, 3)
    assert out.shape[1] == acts.shape[1]
    test_mask = np.ones(acts.shape[1], dtype=bool)
    test_mask[train_indices] = False
    assert np.isfinite(out[:, test_mask, :]).all()
    assert not np.allclose(out[:, test_mask, :], 0.0)
