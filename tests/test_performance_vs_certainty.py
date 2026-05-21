# -*- coding: utf-8 -*-
"""Unit tests for ``decoder_plotting_utils._posterior_entropy`` and
``plot_performance_vs_certainty``.

The certainty plot bins per-trial decoder fit-loss by how peaked the IO
target posterior is. The numeric core is ``_posterior_entropy`` —
Shannon entropy per row — which the tests anchor against known cases
(uniform → log N, delta → 0). The plot function itself gets a smoke
test: it must run end-to-end on a synthetic results dict and write the
expected SVG.
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

pytest.importorskip('scipy')
pytest.importorskip('matplotlib')
import decoder_plotting_utils as dpu  # noqa: E402


# ----------------------------------------------------------------------
# _posterior_entropy
# ----------------------------------------------------------------------

def test_entropy_of_uniform_is_log_n():
    """A uniform distribution over N categories has entropy log N."""
    for n in (2, 5, 91):
        p = np.full((3, n), 1.0 / n)
        H = dpu._posterior_entropy(p)
        assert H.shape == (3,)
        assert np.allclose(H, np.log(n), atol=1e-6)


def test_entropy_of_delta_is_zero():
    """A one-hot (delta) distribution has zero entropy."""
    p = np.zeros((4, 7))
    p[np.arange(4), [0, 3, 6, 1]] = 1.0
    H = dpu._posterior_entropy(p)
    assert np.allclose(H, 0.0, atol=1e-6)


def test_entropy_renormalises_unnormalised_rows():
    """Rows that don't sum to 1 are renormalised before the entropy is
    taken — a uniform row scaled by 10 still gives log N."""
    n = 5
    p = np.full((2, n), 10.0 / n)  # sums to 10, not 1
    H = dpu._posterior_entropy(p)
    assert np.allclose(H, np.log(n), atol=1e-6)


def test_entropy_matches_hand_value():
    """Anchored hand-computed case so the convention can't drift."""
    p = np.array([[0.5, 0.5], [0.9, 0.1]])
    H = dpu._posterior_entropy(p)
    expected = np.array([
        np.log(2.0),
        -(0.9 * np.log(0.9) + 0.1 * np.log(0.1)),
    ])
    assert np.allclose(H, expected, atol=1e-6)


# ----------------------------------------------------------------------
# plot_performance_vs_certainty smoke test
# ----------------------------------------------------------------------

def _synthetic_perception_results(n_trials=120, n_cats=20, seed=0):
    """A minimal {split: {'results': {mouse: {'Dist': ...}}}} structure
    with a real PCA basis, enough for plot_performance_vs_certainty."""
    rng = np.random.RandomState(seed)

    def _softmax_rows(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # Target posteriors spanning a range of sharpness (so certainty bins
    # are populated).
    sharp = rng.uniform(0.2, 6.0, size=(n_trials, 1))
    base = rng.randn(n_trials, n_cats)
    target = _softmax_rows(base * sharp)
    # Decoded = target plus noise, per arch.
    dist = {}
    for arch in ('spat', 'temp', 'spat_shf', 'temp_shf'):
        noise = rng.randn(n_trials, n_cats) * (0.5 if 'shf' in arch else 0.15)
        decoded = _softmax_rows(np.log(target + 1e-9) + noise)
        dist[arch] = {'target': target.copy(), 'decoded': decoded}
    # PCA basis fit on the targets.
    from sklearn.decomposition import PCA
    pca = PCA().fit(target)
    dist['pcs'] = pca.components_
    dist['explained_var'] = pca.explained_variance_ratio_

    return {'stratified_balanced': {'results': {'mouse_0': {'Dist': dist}}}}


def test_plot_performance_vs_certainty_writes_svg(tmp_path):
    """End-to-end: the function runs on a synthetic results dict and
    writes 4B_Performance_vs_Certainty_<split>.svg."""
    results = _synthetic_perception_results()
    dpu.plot_performance_vs_certainty(
        results, ['stratified_balanced'], n_bins=5, out_dir=str(tmp_path),
    )
    svg = tmp_path / '4B_Performance_vs_Certainty_stratified_balanced.svg'
    assert svg.exists(), "certainty plot SVG was not written"
    assert svg.stat().st_size > 0


def test_plot_performance_vs_certainty_raw_variant_writes_suffixed_svg(tmp_path):
    """normalize=False writes a *_raw.svg and does not clobber the
    normalised file — both can coexist in one output dir."""
    results = _synthetic_perception_results()
    dpu.plot_performance_vs_certainty(
        results, ['stratified_balanced'], n_bins=5, out_dir=str(tmp_path),
        normalize=True,
    )
    dpu.plot_performance_vs_certainty(
        results, ['stratified_balanced'], n_bins=5, out_dir=str(tmp_path),
        normalize=False,
    )
    norm_svg = tmp_path / '4B_Performance_vs_Certainty_stratified_balanced.svg'
    raw_svg = tmp_path / '4B_Performance_vs_Certainty_stratified_balanced_raw.svg'
    assert norm_svg.exists(), "normalised certainty plot missing"
    assert raw_svg.exists(), "raw certainty plot missing"


def test_plot_performance_vs_certainty_skips_non_pca_cell(tmp_path):
    """A cell with no PCA basis (pcs empty) is skipped silently — the
    certainty-vs-loss metric is only defined for the 91-D PCA targets."""
    results = _synthetic_perception_results()
    # Blank out the PCA basis to simulate an MSE/CE cell.
    results['stratified_balanced']['results']['mouse_0']['Dist']['pcs'] = np.array([])
    results['stratified_balanced']['results']['mouse_0']['Dist']['explained_var'] = np.array([])
    dpu.plot_performance_vs_certainty(
        results, ['stratified_balanced'], n_bins=5, out_dir=str(tmp_path),
    )
    # No mice contributed -> no SVG written, and no exception raised.
    svg = tmp_path / '4B_Performance_vs_Certainty_stratified_balanced.svg'
    assert not svg.exists()
