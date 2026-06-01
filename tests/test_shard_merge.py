# -*- coding: utf-8 -*-
"""Tests for the per-mouse shard + merge_shards flow.

The 2026-05-27 loss sweep parallelises across mice. Each per-mouse
worker writes ``<slug>/shards/<split>__mouse_<mid>.mat``; a final
``merge_shards`` step combines them into the canonical
``<slug>/<split>.mat`` that ``load_run_tree`` and the plot suite
expect.

These tests pin the contract without triggering an actual training
fit — they fabricate per-mouse result dicts, write shards, run the
merge, and assert the merged file has the right shape and content.
The shard round-trip is the only new I/O contract introduced by the
per-mouse refactor; everything downstream (load_run_tree, plotters)
keeps reading the canonical .mat.
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
import scipy.io as sio   # noqa: E402
from training.run import (   # noqa: E402
    merge_shards, _shard_path, _checkpoint_path,
)


def _fake_animal_result(n_trials=4, n_cats=3):
    """Smallest non-trivial dict shape ``run_animal_decoder`` returns."""
    rng = np.random.default_rng(0)
    return {
        'Dist': {
            'spat': {
                'decoded': rng.random((n_trials, n_cats)),
                'target':  rng.random((n_trials, n_cats)),
            },
        },
        'trials': {
            'orientation': rng.integers(0, 90, n_trials),
        },
    }


def _write_fake_shard(slug_dir: Path, split: str, mouse_id: int,
                       cfg: dict | None = None):
    """Write one fake per-mouse shard with the shape run_config produces."""
    shard_dir = slug_dir / 'shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = _shard_path(slug_dir, split, mouse_id)
    sio.savemat(
        str(shard_path),
        {
            'results': {f'mouse_{mouse_id}': _fake_animal_result()},
            'config': cfg or {'custom_loss_func': 'PCA', 'which_model': 'perception'},
        },
    )
    return shard_path


# ----------------------------------------------------------------------
# merge_shards
# ----------------------------------------------------------------------

def test_merge_shards_combines_all_mice(tmp_path):
    """Multiple per-mouse shards merge into one .mat whose ``results``
    dict has one entry per mouse, keyed ``mouse_<mid>``."""
    slug_dir = tmp_path / 'Q_PCA_half_100ms_all'
    for mid in (0, 1, 2):
        _write_fake_shard(slug_dir, 'stratified_balanced', mid)

    final = merge_shards(slug_dir, 'stratified_balanced')
    assert final is not None
    assert final.name == 'stratified_balanced.mat'

    merged = sio.loadmat(str(final), simplify_cells=True)
    assert set(merged['results']) == {'mouse_0', 'mouse_1', 'mouse_2'}
    # Each animal's nested dict has the shape downstream plotters read.
    for mid in (0, 1, 2):
        a = merged['results'][f'mouse_{mid}']
        assert 'Dist' in a and 'spat' in a['Dist']
        assert 'target' in a['Dist']['spat']
        assert 'decoded' in a['Dist']['spat']


def test_merge_shards_returns_none_when_no_shards(tmp_path):
    """A slug with no shards directory (or an empty one) must not crash
    — it just returns None so the parallel wrapper can move on."""
    slug_dir = tmp_path / 'Q_PCA_half_100ms_all'
    slug_dir.mkdir()

    # No shards/ subdir at all.
    assert merge_shards(slug_dir, 'stratified_balanced') is None

    # Empty shards/ subdir is also a no-op.
    (slug_dir / 'shards').mkdir()
    assert merge_shards(slug_dir, 'stratified_balanced') is None


def test_merge_shards_idempotent(tmp_path):
    """Running merge_shards twice produces the same canonical .mat; the
    second call simply overwrites the first with identical content."""
    slug_dir = tmp_path / 'Q_PCA_half_100ms_all'
    for mid in (0, 1):
        _write_fake_shard(slug_dir, 'stratified_balanced', mid)

    first = merge_shards(slug_dir, 'stratified_balanced')
    first_bytes = first.read_bytes()

    second = merge_shards(slug_dir, 'stratified_balanced')
    # Byte-identical re-write — sio.savemat is deterministic given the
    # same input dict order. (Shards are sorted in merge_shards.)
    assert first == second
    assert first.read_bytes() == first_bytes


def test_merge_shards_split_isolation(tmp_path):
    """Shards for different splits don't bleed into each other —
    merge_shards picks up only the shards whose filename matches the
    requested split prefix."""
    slug_dir = tmp_path / 'Q_PCA_half_100ms_all'
    _write_fake_shard(slug_dir, 'stratified_balanced', 0)
    _write_fake_shard(slug_dir, 'stratified_balanced', 1)
    _write_fake_shard(slug_dir, 'generalize_contrast', 0)

    merged_a = sio.loadmat(
        str(merge_shards(slug_dir, 'stratified_balanced')),
        simplify_cells=True)
    assert set(merged_a['results']) == {'mouse_0', 'mouse_1'}

    merged_b = sio.loadmat(
        str(merge_shards(slug_dir, 'generalize_contrast')),
        simplify_cells=True)
    assert set(merged_b['results']) == {'mouse_0'}


def test_merge_shards_delete_cleans_up(tmp_path):
    """``delete_shards=True`` removes the per-mouse files (and the
    shards/ dir when empty) after a successful merge."""
    slug_dir = tmp_path / 'Q_PCA_half_100ms_all'
    for mid in (0, 1):
        _write_fake_shard(slug_dir, 'stratified_balanced', mid)

    merge_shards(slug_dir, 'stratified_balanced', delete_shards=True)
    assert not (slug_dir / 'shards' / 'stratified_balanced__mouse_0.mat').exists()
    assert not (slug_dir / 'shards' / 'stratified_balanced__mouse_1.mat').exists()
    # Final .mat is still there.
    assert (slug_dir / 'stratified_balanced.mat').exists()


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------

def test_shard_and_checkpoint_paths(tmp_path):
    """The per-mouse path helpers must produce the exact filenames the
    bash wrapper and run_config consume — pinned so a rename in one
    place doesn't silently desync the other."""
    out = tmp_path / 'Q_PCA_half_100ms_all'
    sp = _shard_path(out, 'stratified_balanced', 3)
    cp = _checkpoint_path(out, 'stratified_balanced', 3)
    assert sp.name == 'stratified_balanced__mouse_3.mat'
    assert sp.parent.name == 'shards'
    assert cp.name == 'mouse_3_stratified_balanced.pt'
    assert cp.parent.name == 'checkpoints'
