# -*- coding: utf-8 -*-
"""Unit tests for ``decoder_plotting_utils.load_run_tree``.

``load_run_tree`` is the companion to ``load_results_dict``. Where
``load_results_dict`` reads the legacy flat-named files
(``population_results_*_<split>.mat``) written by ``run_fixed_hyperparams.py``,
``load_run_tree`` reads the nested tree (``results/<run_name>/<slug>/<split>.mat``)
written by ``training/run.py``. Both return the same shape
(``{split: scipy.io.loadmat output}``) so downstream plotters in
``decoder_plotting_utils`` work unchanged on either input.

Tests cover:
  * Path resolution against ``paths.RESULTS`` and explicit ``results_root``.
  * Default split list.
  * Missing slug / missing split behaviour (warn, return empty / skip).
  * Returned dict shape matches what plotters consume from
    ``load_results_dict`` (top-level ``results`` and ``config`` keys,
    per-mouse sub-dicts).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

scipy_io = pytest.importorskip('scipy.io')
import decoder_plotting_utils as dpu  # noqa: E402
import paths  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

PRODUCTION_RUN = 'post_fix_loadings_2026_05_17'
PRODUCTION_SLUG = 'd_MSE_half_100ms'
PRODUCTION_SPLITS = ('stratified_balanced',
                      'generalize_contrast',
                      'generalize_dispersion')


def _has_production_run() -> bool:
    """True iff the canonical post_fix_loadings run is on disk."""
    return (paths.RESULTS / PRODUCTION_RUN / PRODUCTION_SLUG /
            'stratified_balanced.mat').exists()


def _write_fake_split(path: Path, mouse_ids=(0, 1)):
    """Write a minimal .mat file with the shape downstream plotters expect.

    Just enough structure that load_run_tree can return a non-empty dict
    and downstream code that does `data['results']['mouse_0']` doesn't
    crash. The values themselves are placeholders — these tests are
    about loading and path resolution, not numerical correctness.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for mid in mouse_ids:
        results[f'mouse_{mid}'] = {
            'KLs': {
                'spat': np.array([0.1, 0.2, 0.3], dtype=np.float64),
                'temp': np.array([0.15, 0.25, 0.35], dtype=np.float64),
                'spat_shf': np.array([0.5, 0.5, 0.5], dtype=np.float64),
                'temp_shf': np.array([0.55, 0.55, 0.55], dtype=np.float64),
            },
            'Dist': {
                'spat': {
                    'decoded': np.zeros((3, 2), dtype=np.float64),
                    'target':  np.zeros((3, 2), dtype=np.float64),
                    'decoded_samp': np.zeros((3, 2, 4), dtype=np.float64),
                    'full_decoded': np.zeros((6, 2), dtype=np.float64),
                },
                'temp': {
                    'decoded': np.zeros((3, 2), dtype=np.float64),
                    'target':  np.zeros((3, 2), dtype=np.float64),
                    'decoded_samp': np.zeros((3, 2, 4), dtype=np.float64),
                    'full_decoded': np.zeros((6, 2), dtype=np.float64),
                },
                'pcs': np.array([]),
                'explained_var': np.array([]),
            },
        }
    scipy_io.savemat(str(path),
                      {'results': results, 'config': {'split_type': path.stem}})


# ----------------------------------------------------------------------
# Production-data integration tests
# ----------------------------------------------------------------------

@pytest.mark.skipif(not _has_production_run(),
                    reason="post_fix_loadings_2026_05_17 not on disk")
def test_load_run_tree_finds_all_three_splits():
    """Calling with defaults loads the three production splits."""
    out = dpu.load_run_tree(PRODUCTION_RUN, PRODUCTION_SLUG)
    assert set(out) == set(PRODUCTION_SPLITS)


@pytest.mark.skipif(not _has_production_run(),
                    reason="post_fix_loadings_2026_05_17 not on disk")
def test_load_run_tree_each_split_has_results_and_config_keys():
    """Output shape matches load_results_dict: each split's value is the
    full .mat dict, including 'results' and 'config' top-level keys.
    Plotters in decoder_plotting_utils consume this exact shape."""
    out = dpu.load_run_tree(PRODUCTION_RUN, PRODUCTION_SLUG)
    for split, mat in out.items():
        assert 'results' in mat, f"{split} missing 'results' key"
        assert 'config' in mat, f"{split} missing 'config' key"


@pytest.mark.skipif(not _has_production_run(),
                    reason="post_fix_loadings_2026_05_17 not on disk")
def test_load_run_tree_mouse_keys_present():
    """Every loaded split should have at least one mouse entry — the
    contract that downstream per-mouse plotters depend on."""
    out = dpu.load_run_tree(PRODUCTION_RUN, PRODUCTION_SLUG)
    for split, mat in out.items():
        mouse_keys = [k for k in mat['results'] if k.startswith('mouse_')]
        assert mouse_keys, f"{split} has no mouse_ keys in results"


# ----------------------------------------------------------------------
# Synthetic-fixture tests (run in any environment)
# ----------------------------------------------------------------------

def test_load_run_tree_custom_results_root(tmp_path):
    """results_root kwarg overrides paths.RESULTS, letting tests use a
    temporary tree."""
    run_dir = tmp_path / 'fake_run' / 'fake_slug'
    for split in ('stratified_balanced', 'generalize_contrast'):
        _write_fake_split(run_dir / f'{split}.mat')

    out = dpu.load_run_tree('fake_run', 'fake_slug',
                              splits=('stratified_balanced', 'generalize_contrast'),
                              results_root=str(tmp_path))
    assert set(out) == {'stratified_balanced', 'generalize_contrast'}
    assert 'results' in out['stratified_balanced']
    assert 'mouse_0' in out['stratified_balanced']['results']


def test_load_run_tree_missing_slug_returns_empty(tmp_path, capsys):
    """Pointing at a slug directory that doesn't exist returns {} and
    prints a warning per split, mirroring load_results_dict behaviour."""
    out = dpu.load_run_tree('nonexistent_run', 'nonexistent_slug',
                              splits=('stratified_balanced',),
                              results_root=str(tmp_path))
    assert out == {}
    captured = capsys.readouterr()
    assert 'not found' in captured.out.lower()


def test_load_run_tree_missing_split_warns_and_skips(tmp_path, capsys):
    """If one split's .mat is missing, that split is skipped with a
    warning but the others still load."""
    run_dir = tmp_path / 'fake_run' / 'fake_slug'
    _write_fake_split(run_dir / 'stratified_balanced.mat')
    # generalize_contrast.mat intentionally not written.

    out = dpu.load_run_tree('fake_run', 'fake_slug',
                              splits=('stratified_balanced', 'generalize_contrast'),
                              results_root=str(tmp_path))
    assert set(out) == {'stratified_balanced'}
    captured = capsys.readouterr()
    assert 'generalize_contrast' in captured.out


def test_load_run_tree_default_splits_match_production(tmp_path):
    """When splits=None, load_run_tree falls back to the canonical
    production triple — the same default as load_results_dict."""
    run_dir = tmp_path / 'fake_run' / 'fake_slug'
    for split in PRODUCTION_SPLITS:
        _write_fake_split(run_dir / f'{split}.mat')

    out = dpu.load_run_tree('fake_run', 'fake_slug',
                              results_root=str(tmp_path))
    assert set(out) == set(PRODUCTION_SPLITS)


def test_load_run_tree_returns_same_shape_as_load_results_dict(tmp_path):
    """Direct shape comparison: load_run_tree's output should be a
    drop-in for load_results_dict's output — downstream plotters
    iterate over splits and consume mat['results']['mouse_<n>'] the
    same way regardless of which loader fed them."""
    run_dir = tmp_path / 'fake_run' / 'fake_slug'
    _write_fake_split(run_dir / 'stratified_balanced.mat')

    out = dpu.load_run_tree('fake_run', 'fake_slug',
                              splits=('stratified_balanced',),
                              results_root=str(tmp_path))
    mat = out['stratified_balanced']
    # Same top-level keys as load_results_dict produces (sio.loadmat output).
    assert 'results' in mat and 'config' in mat
    # Same per-mouse access pattern.
    m = mat['results']['mouse_0']
    assert 'KLs' in m
    assert 'Dist' in m
    assert {'spat', 'temp'} <= set(m['Dist'])
