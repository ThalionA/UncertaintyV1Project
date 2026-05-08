# -*- coding: utf-8 -*-
"""Unit tests for ``nn_decoder/training/config.py``.

Verifies:
  - default_config_for_target returns a valid Config for every target.
  - to_legacy_dict produces a dict with all the keys the legacy
    run_animal_decoder consumes.
  - Slug + output_dir match the agreed nested-tree convention.
  - YAML round-trip preserves every field.
  - Validation rejects unknown values.
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

from training.config import (  # noqa: E402
    Config, default_config_for_target, TARGET_TO_WHICH_MODEL,
)


# ----------------------------------------------------------------------
# default_config_for_target
# ----------------------------------------------------------------------

@pytest.mark.parametrize('target', list(TARGET_TO_WHICH_MODEL))
def test_default_config_returns_valid_for_every_target(target):
    cfg = default_config_for_target(target)
    assert cfg.target_type == target
    assert cfg.loss_func in ('PCA', 'MSE', 'CE')
    assert cfg.bin_size_ms in (50, 100, 250)
    assert cfg.entropy_lambda == 3e-3  # post-bug-fix invariant


def test_default_config_choice_uses_smaller_net_higher_lr():
    """Documented per-target deviation: choice uses [16] hidden, lr=5e-3,
    epochs=50 vs the [32]/1e-3/30 baseline for distributional targets."""
    cfg = default_config_for_target('choice')
    assert cfg.hidden_sizes == [16]
    assert cfg.learning_rate == 5e-3
    assert cfg.num_epochs == 50


def test_default_config_overrides_apply():
    cfg = default_config_for_target('Q', bin_size_ms=50, run_name='probe')
    assert cfg.target_type == 'Q'
    assert cfg.bin_size_ms == 50
    assert cfg.run_name == 'probe'


def test_default_config_unknown_target_raises():
    with pytest.raises(ValueError):
        default_config_for_target('unknown_target')


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def test_invalid_target_type_raises():
    with pytest.raises(ValueError, match='Unknown target_type'):
        Config(target_type='foo', loss_func='PCA')


def test_invalid_loss_raises():
    with pytest.raises(ValueError, match='Unknown loss_func'):
        Config(target_type='Q', loss_func='not_a_loss')


def test_invalid_bin_size_raises():
    with pytest.raises(ValueError, match='Unknown bin_size_ms'):
        Config(target_type='Q', loss_func='PCA', bin_size_ms=42)


def test_invalid_time_window_raises():
    with pytest.raises(ValueError, match='Unknown time_window'):
        Config(target_type='Q', loss_func='PCA', time_window='garbage')


# ----------------------------------------------------------------------
# Legacy dict translation
# ----------------------------------------------------------------------

def test_to_legacy_dict_translates_short_target_name_to_which_model():
    cfg = default_config_for_target('choice', run_name='x')
    legacy = cfg.to_legacy_dict()
    assert legacy['which_model'] == 'true_choice'   # short -> legacy
    assert 'target_type' not in legacy              # don't leak short name


def test_to_legacy_dict_has_all_keys_run_animal_decoder_reads():
    """run_animal_decoder reads these keys (see run_experiment_v26)."""
    needed = {
        'target_source', 'time_window', 'bin_size_ms', 'split_type',
        'which_model', 'hidden_sizes', 'activation_function',
        'weight_initialization', 'custom_loss_func', 'entropy_lambda',
        'learning_rate', 'optimizer_type', 'momentum', 'num_epochs',
        'minibatch_size', 'REP',
    }
    cfg = default_config_for_target('Q')
    legacy = cfg.to_legacy_dict()
    missing = needed - set(legacy)
    assert not missing, f"to_legacy_dict missing keys: {missing}"


def test_to_legacy_dict_translates_every_target():
    for target, expected_which in TARGET_TO_WHICH_MODEL.items():
        cfg = default_config_for_target(target)
        assert cfg.to_legacy_dict()['which_model'] == expected_which


# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------

def test_slug_encodes_all_axes():
    cfg = default_config_for_target('Q', bin_size_ms=50)
    assert cfg.slug() == 'Q_PCA_half_50ms'

    cfg = default_config_for_target('choice')
    assert cfg.slug() == 'choice_CE_half_100ms'


def test_output_dir_is_nested_under_run_name():
    cfg = default_config_for_target('Q', run_name='production_2026_05_06',
                                     bin_size_ms=50)
    p = cfg.output_dir('results')
    assert p == Path('results/production_2026_05_06/Q_PCA_half_50ms')


# ----------------------------------------------------------------------
# YAML round-trip
# ----------------------------------------------------------------------

def test_yaml_round_trip_preserves_all_fields(tmp_path):
    pytest.importorskip('yaml')
    cfg_in = default_config_for_target(
        'stim_kernel', bin_size_ms=50, learning_rate=2e-3, run_name='testrun',
        notes='probing the orientation decoder',
    )
    out = tmp_path / 'config.yaml'
    cfg_in.save_yaml(out)
    assert out.exists()
    cfg_out = Config.from_yaml(out)
    # Every field equal
    from dataclasses import asdict
    assert asdict(cfg_in) == asdict(cfg_out)
