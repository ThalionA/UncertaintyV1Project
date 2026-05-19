# -*- coding: utf-8 -*-
"""Unit tests for ``training.run._pop_and_save_checkpoints``.

This is the plumbing that pulls the Checkpoints dict out of each
animal's results, writes one ``.pt`` file per animal/split under
``<out_dir>/checkpoints/``, and strips Checkpoints out of the in-memory
dict so the subsequent ``scipy.io.savemat`` call doesn't try to
serialise torch tensors (which scipy can't handle).

Tests cover:
  * .pt files appear at the expected paths.
  * Loaded .pt content matches what was saved.
  * Checkpoints key is removed from the in-memory results dict.
  * Animals missing a Checkpoints key are tolerated (no crash, no
    spurious file).
  * The checkpoints/ subdirectory is created on demand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

torch = pytest.importorskip('torch')

from training.run import _pop_and_save_checkpoints  # noqa: E402


def _fake_animal_result(mid_seed: int, with_ckpt: bool = True) -> dict:
    """Minimal results dict shaped like run_animal_decoder's return.
    Carries fit_loss/Dist/Weights placeholders so the dict-keys check
    upstream stays happy."""
    res = {
        'fit_loss': {'spat': [], 'temp': []},
        'Dist': {},
        'Weights': {},
    }
    if with_ckpt:
        res['Checkpoints'] = {
            'spat': {
                'state_dict': {'w': torch.tensor([float(mid_seed)])},
                'X_test': torch.randn(2, 3, 4),
                'pred_probs': torch.randn(2, 1, 5),
                'model_params': {'input_size': 4, 'hidden_sizes': [6],
                                  'output_size': 5,
                                  'activation_function': 'tanh'},
                'model_type': 'ppc',
                'pcs': None, 'explained_var': None,
                'loss_func': 'MSE', 'entropy_lambda': 0.003,
            },
            'temp': {
                'state_dict': {'w': torch.tensor([float(mid_seed) + 0.5])},
                'X_test': torch.randn(2, 3, 4),
                'pred_probs': torch.randn(2, 3, 5),
                'model_params': {'input_size': 4, 'hidden_sizes': [6],
                                  'output_size': 5,
                                  'activation_function': 'tanh'},
                'model_type': 'sampling',
                'pcs': None, 'explained_var': None,
                'loss_func': 'MSE', 'entropy_lambda': 0.003,
            },
        }
    return res


def test_pop_and_save_creates_pt_files(tmp_path):
    """One .pt file per animal at the canonical path."""
    out_dir = tmp_path / 'fake_run' / 'fake_slug'
    out_dir.mkdir(parents=True)
    all_mice = {
        'mouse_0': _fake_animal_result(0),
        'mouse_1': _fake_animal_result(1),
    }
    _pop_and_save_checkpoints(all_mice, out_dir, 'stratified_balanced')

    for mid in (0, 1):
        p = out_dir / 'checkpoints' / f'mouse_{mid}_stratified_balanced.pt'
        assert p.exists(), f"missing {p}"


def test_pop_and_save_round_trips_content(tmp_path):
    """Loaded .pt content matches what was saved — the checkpoint is
    the only durable record of the trained model, so corruption here
    breaks the whole sanity-check pipeline."""
    out_dir = tmp_path / 'fake_run' / 'fake_slug'
    out_dir.mkdir(parents=True)
    original = _fake_animal_result(42)
    saved_ckpt = {
        arch: {k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in entry.items()}
        for arch, entry in original['Checkpoints'].items()
    }
    all_mice = {'mouse_0': original}
    _pop_and_save_checkpoints(all_mice, out_dir, 'stratified_balanced')

    pt_path = out_dir / 'checkpoints' / 'mouse_0_stratified_balanced.pt'
    loaded = torch.load(str(pt_path), weights_only=False)
    for arch in ('spat', 'temp'):
        assert arch in loaded
        # Tensor content survives.
        assert torch.allclose(loaded[arch]['state_dict']['w'],
                                saved_ckpt[arch]['state_dict']['w'])
        # Metadata survives.
        assert loaded[arch]['model_type'] == saved_ckpt[arch]['model_type']
        assert loaded[arch]['loss_func'] == saved_ckpt[arch]['loss_func']
        assert loaded[arch]['entropy_lambda'] == saved_ckpt[arch]['entropy_lambda']


def test_pop_removes_checkpoints_from_in_memory_dict(tmp_path):
    """After the call, Checkpoints must be GONE from each animal's
    dict. If left behind, scipy.io.savemat downstream would choke on
    the torch tensors."""
    out_dir = tmp_path / 'fake_run' / 'fake_slug'
    out_dir.mkdir(parents=True)
    all_mice = {'mouse_0': _fake_animal_result(0)}
    _pop_and_save_checkpoints(all_mice, out_dir, 'stratified_balanced')
    assert 'Checkpoints' not in all_mice['mouse_0']
    # Other keys preserved.
    assert 'fit_loss' in all_mice['mouse_0']
    assert 'Dist' in all_mice['mouse_0']


def test_pop_tolerates_animals_without_checkpoints(tmp_path):
    """Older run_animal_decoder runs (or partial / failed mice) may not
    have a Checkpoints key. Skip them silently rather than crashing
    the whole run."""
    out_dir = tmp_path / 'fake_run' / 'fake_slug'
    out_dir.mkdir(parents=True)
    all_mice = {
        'mouse_0': _fake_animal_result(0, with_ckpt=True),
        'mouse_1': _fake_animal_result(1, with_ckpt=False),
    }
    _pop_and_save_checkpoints(all_mice, out_dir, 'stratified_balanced')

    assert (out_dir / 'checkpoints' / 'mouse_0_stratified_balanced.pt').exists()
    assert not (out_dir / 'checkpoints' / 'mouse_1_stratified_balanced.pt').exists()
    # mouse_1 dict is untouched (no Checkpoints key existed to pop).
    assert 'Checkpoints' not in all_mice['mouse_1']


def test_pop_creates_checkpoints_dir_on_demand(tmp_path):
    """checkpoints/ subdirectory is created lazily by the saver — we
    don't require run_config to pre-create it."""
    out_dir = tmp_path / 'fake_run' / 'fake_slug'
    out_dir.mkdir(parents=True)
    assert not (out_dir / 'checkpoints').exists()
    all_mice = {'mouse_0': _fake_animal_result(0)}
    _pop_and_save_checkpoints(all_mice, out_dir, 'stratified_balanced')
    assert (out_dir / 'checkpoints').is_dir()


def test_pop_with_no_animals_having_checkpoints_skips_dir_creation(tmp_path):
    """If no animal carries a Checkpoints field, the saver should be a
    no-op — no empty checkpoints/ directory left behind."""
    out_dir = tmp_path / 'fake_run' / 'fake_slug'
    out_dir.mkdir(parents=True)
    all_mice = {'mouse_0': _fake_animal_result(0, with_ckpt=False)}
    _pop_and_save_checkpoints(all_mice, out_dir, 'stratified_balanced')
    assert not (out_dir / 'checkpoints').exists()
