# -*- coding: utf-8 -*-
"""Contract tests for the synthetic-target generators (``utils.py``).

Both generators return a ``(likelihoods, posteriors)`` TUPLE; the training
targets are element [1]. ``run_experiment``'s ``synthetic_ppc``/
``synthetic_sbc`` branch once assigned the whole tuple to ``raw_targets``
and crashed downstream (2026-08-25 audit, item B3) — these tests pin the
return contract so a future change to the generators' return shape fails
here instead of inside a run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

from utils import generate_PPC_targets, generate_SBC_targets  # noqa: E402

S_GRID = np.arange(0, 91, 1)
N_TRIALS, T_BINS, N_NEURONS = 5, 4, 3


@pytest.fixture()
def activities_and_templates():
    rng = np.random.default_rng(0)
    activities = rng.poisson(2.0, size=(N_TRIALS, T_BINS, N_NEURONS)).astype(float)
    # Smooth positive tuning templates over the 91-bin grid.
    centres = np.linspace(10, 80, N_NEURONS)
    templates = np.stack(
        [1.0 + 4.0 * np.exp(-0.5 * ((S_GRID - c) / 15.0) ** 2) for c in centres],
        axis=1)  # (91, n_neurons)
    return activities, templates


@pytest.mark.parametrize('gen', [generate_PPC_targets, generate_SBC_targets])
def test_generators_return_likelihood_posterior_tuple(gen, activities_and_templates):
    activities, templates = activities_and_templates
    out = gen(activities, templates)
    assert isinstance(out, tuple) and len(out) == 2, (
        f"{gen.__name__} must return (likelihoods, posteriors); "
        f"run_experiment's synthetic branch takes element [1] as the targets.")
    likelihoods, posteriors = out
    for arr in (likelihoods, posteriors):
        assert arr.shape == (N_TRIALS, len(S_GRID))
        np.testing.assert_allclose(np.nansum(arr, axis=1), 1.0, atol=1e-8)
        assert np.all(arr >= 0)
