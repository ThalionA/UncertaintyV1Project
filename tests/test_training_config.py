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
    # entropy_lambda must be in a sensible range. The historical bug was
    # entropy_lambda=3e3 (3000) which broke SBC + one-hot targets. Anything
    # at 3e-3 magnitude or nearby is fine; the strict bound here just
    # excludes the broken value and other absurdities.
    assert 1e-5 <= cfg.entropy_lambda <= 1.0


def test_default_config_choice_matches_optuna_preset():
    """choice's default (100 ms) preset is the 2026-05-23 Optuna sweep
    winner: [32] hidden, CE loss, lr~3.2e-4, 30 epochs."""
    cfg = default_config_for_target('choice')
    assert cfg.hidden_sizes == [32]
    assert cfg.loss_func == 'CE'
    assert cfg.num_epochs == 30
    assert cfg.learning_rate == 0.0003177


def test_default_config_overrides_apply():
    cfg = default_config_for_target('Q', bin_size_ms=50, run_name='probe')
    assert cfg.target_type == 'Q'
    assert cfg.bin_size_ms == 50
    assert cfg.run_name == 'probe'


def test_default_config_q_uses_per_bin_size_optuna_winner():
    """Q has separate Optuna-tuned presets for 50 ms and 100 ms,
    retrievable by their exact bin. The 100 ms preset is the 2026-05-23
    all-targets sweep winner; 50 ms is the older 2026-05-06 sweep, so the
    two are distinct (and not directly comparable -- different training
    loops)."""
    c50 = default_config_for_target('Q', bin_size_ms=50)
    c100 = default_config_for_target('Q', bin_size_ms=100)
    assert c50.bin_size_ms == 50
    assert c100.bin_size_ms == 100
    # Distinct presets from distinct sweeps.
    assert (c50.learning_rate, c50.num_epochs) != (c100.learning_rate, c100.num_epochs)


def test_default_config_falls_back_to_bin_agnostic_preset():
    """L has an explicit (L, 100) Optuna preset but no 50/250 ms entries;
    those bins fall back to the bin-agnostic (L, None) preset."""
    c50 = default_config_for_target('L', bin_size_ms=50)
    c250 = default_config_for_target('L', bin_size_ms=250)
    # Both fall back to (L, None): same hyperparams, different bin tag.
    assert c50.learning_rate == c250.learning_rate
    assert c50.num_epochs == c250.num_epochs
    assert c50.hidden_sizes == c250.hidden_sizes
    assert c50.bin_size_ms == 50
    assert c250.bin_size_ms == 250
    # The explicit (L, 100) preset wins over the fallback at 100 ms.
    c100 = default_config_for_target('L', bin_size_ms=100)
    assert c100.learning_rate != c50.learning_rate


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


# Config keys run_animal_decoder reads that to_legacy_dict deliberately does
# NOT provide. Both are guarded reads inside the legacy 'recovery' target
# branch, which is driven by hand-built dicts (run_fixed_recovery.py), not
# Config. Anything else that shows up in the diff is a plumbing regression.
_LEGACY_ONLY_KEYS = {'base_file_path', 'base_recovery_id'}


def test_to_legacy_dict_covers_every_key_run_animal_decoder_reads():
    """Derive the needed-key set from run_experiment's SOURCE instead of a
    hand-frozen list — the frozen list rotted (21 keys vs the ~37 actually
    read by 2026-08) and would have missed exactly the regression class it
    exists for (2026-08-25 audit, item B9)."""
    import re
    src = (Path(NN_DECODER) / 'run_experiment.py').read_text()
    reads = set(re.findall(r"config\.get\(\s*['\"](\w+)['\"]", src))
    reads |= set(re.findall(r"config\[\s*['\"](\w+)['\"]\s*\]", src))
    legacy = set(default_config_for_target('Q').to_legacy_dict())
    missing = reads - legacy - _LEGACY_ONLY_KEYS
    assert not missing, (
        f"run_animal_decoder reads keys to_legacy_dict does not provide: "
        f"{sorted(missing)} — add them to to_legacy_dict (no-op default) or, "
        f"if truly legacy-dict-only, to _LEGACY_ONLY_KEYS with a reason.")


# Config fields that never enter the legacy dict, with the reason. A field
# added to Config must land in to_legacy_dict, in _FIELD_RENAMES, or here —
# otherwise it is a knob that silently does nothing (the random_state bug,
# 2026-08-25 audit item B2, and the restart_selection incident before it).
_FIELD_RENAMES = {'target_type': 'which_model', 'loss_func': 'custom_loss_func'}
_NOT_PLUMBED_FIELDS = {
    'run_name',  # output-path identity only; consumed by slug()/output_dir()
    'notes',     # free-text provenance; recorded in config.yaml only
}


def test_every_config_field_is_plumbed_or_declared():
    import dataclasses
    cfg = default_config_for_target('Q')
    legacy = set(cfg.to_legacy_dict())
    unplumbed = set()
    for f in dataclasses.fields(cfg):
        if f.name in legacy or f.name in _NOT_PLUMBED_FIELDS:
            continue
        if _FIELD_RENAMES.get(f.name) in legacy:
            continue
        unplumbed.add(f.name)
    assert not unplumbed, (
        f"Config fields that never reach run_animal_decoder: "
        f"{sorted(unplumbed)} — plumb through to_legacy_dict or declare in "
        f"_NOT_PLUMBED_FIELDS/_FIELD_RENAMES with a reason.")


def test_cell_slug_is_the_single_producer_consumer_implementation():
    """`Config.slug` (producer — names the directories run_config writes) and
    the plotters' `_slug` (consumer — finds them again) must be the same
    function. Four plotters used to re-implement it, all hardcoding the '_all'
    PCA token, so a condition_mean/residual run was invisible to them
    (2026-08-25 audit, D2)."""
    from training.config import cell_slug, PCA_BASIS_SLUG

    for basis in PCA_BASIS_SLUG:
        cfg = default_config_for_target('Q', loss_func='PCA', pca_basis=basis)
        assert cfg.slug() == cell_slug(
            'Q', 'PCA', cfg.time_window, cfg.bin_size_ms, basis)
    # The three bases must land in DIFFERENT directories.
    slugs = {cell_slug('Q', 'PCA', 'half', 100, b) for b in PCA_BASIS_SLUG}
    assert len(slugs) == len(PCA_BASIS_SLUG)

    # Non-PCA losses carry no basis token, whatever pca_basis says.
    for loss in ('KL', 'JS', 'CE', 'MSE', 'Wasserstein'):
        assert cell_slug('Q', loss, 'half', 100) == f'Q_{loss}_half_100ms'
        assert cell_slug('Q', loss, 'half', 100, 'residual') == f'Q_{loss}_half_100ms'

    # On-disk stability: these are the names the existing results tree uses.
    assert cell_slug('Q', 'PCA', 'half', 100) == 'Q_PCA_half_100ms_all'
    assert cell_slug('Q', 'KL', 'half', 100) == 'Q_KL_half_100ms'

    with pytest.raises(ValueError, match='pca_basis'):
        cell_slug('Q', 'PCA', 'half', 100, 'nonsense')


def test_plotter_slug_helpers_delegate_to_cell_slug():
    """The four ported consumers must stay in lock-step with the producer."""
    pytest.importorskip('matplotlib').use('Agg')
    from training.config import cell_slug
    mods = ['plot_weight_evolution_cell', 'posterior_pca_views',
            'plot_overfit_vs_width', 'within_mouse_loss_plots']
    import importlib
    for name in mods:
        mod = importlib.import_module(name)
        for loss in ('PCA', 'KL'):
            assert mod._slug('Q', loss, 'half', 100) == \
                cell_slug('Q', loss, 'half', 100), name
        assert mod._slug('Q', 'PCA', 'half', 100, 'residual') == \
            cell_slug('Q', 'PCA', 'half', 100, 'residual'), name


def test_random_state_is_plumbed_and_defaults_to_42():
    """random_state must reach the legacy dict (2026-08-25 audit, B2): before
    the fix run_experiment hardcoded 42, so Config(random_state=7) silently
    reproduced the default split."""
    assert default_config_for_target('Q').to_legacy_dict()['random_state'] == 42
    cfg = default_config_for_target('Q', random_state=7)
    assert cfg.to_legacy_dict()['random_state'] == 7


def test_io_hmm_rejects_wasserstein_and_smooth_lambda():
    """The IO-HMM targets live on a CIRCULAR 72-bin support; the 1-D
    Wasserstein cumsum and the (wrap-less) Dirichlet smoothness penalty both
    assume a linear support (2026-08-25 audit, B5 — previously only a
    docstring warning)."""
    with pytest.raises(ValueError, match='[Ww]asserstein'):
        default_config_for_target(
            'Q', target_source='io_hmm_pkl', loss_func='Wasserstein')
    with pytest.raises(ValueError, match='smooth_lambda'):
        default_config_for_target(
            'Q', target_source='io_hmm_pkl', smooth_lambda=0.1)
    # The valid pairing still constructs.
    default_config_for_target('Q', target_source='io_hmm_pkl', loss_func='KL')


def test_track_training_history_default_off_and_opt_in():
    """Diagnostic checkpointing is opt-in. The defaults must be off so
    production runs pay no overhead and the saved YAML self-documents
    when an exploratory run had history tracking enabled."""
    cfg_default = default_config_for_target('Q')
    legacy_default = cfg_default.to_legacy_dict()
    assert legacy_default['track_training_history'] is False
    assert legacy_default['weight_snapshot_every'] == 0

    cfg_on = default_config_for_target(
        'Q', track_training_history=True, weight_snapshot_every=10)
    legacy_on = cfg_on.to_legacy_dict()
    assert legacy_on['track_training_history'] is True
    assert legacy_on['weight_snapshot_every'] == 10


def test_val_frac_default_off_and_opt_in():
    """val_frac defaults to 0 (no carve-out — production unchanged).
    When set, run_animal_decoder will further split train into
    train+val; the legacy dict carries the value through. Per the
    2026-05-27 meeting plan this is the train-vs-val diagnostic switch."""
    cfg_default = default_config_for_target('Q')
    assert cfg_default.val_frac == 0.0
    assert cfg_default.to_legacy_dict()['val_frac'] == 0.0

    cfg_on = default_config_for_target('Q', val_frac=0.15)
    assert cfg_on.val_frac == 0.15
    assert cfg_on.to_legacy_dict()['val_frac'] == 0.15


def test_to_legacy_dict_translates_every_target():
    for target, expected_which in TARGET_TO_WHICH_MODEL.items():
        cfg = default_config_for_target(target)
        assert cfg.to_legacy_dict()['which_model'] == expected_which


def test_to_legacy_dict_carries_weight_decay():
    """weight_decay was previously dropped from to_legacy_dict, then ignored
    by run_animal_decoder (which never read it from config), then silently
    overridden to 3e-4 in train_and_select_best_model's hardcoded Adam call.
    The Optuna-tuned Q-100ms value (weight_decay=1.388e-5) was therefore
    never honoured in production training. This test guards against
    re-introducing that silent override on any of the three layers."""
    cfg = default_config_for_target('Q', bin_size_ms=100)
    legacy = cfg.to_legacy_dict()
    assert 'weight_decay' in legacy
    assert legacy['weight_decay'] == cfg.weight_decay
    # The Q-100ms preset uses Optuna-tuned weight_decay ~1.4e-5, which is
    # ~20x smaller than the historical hardcoded 3e-4. A regression that
    # replaces the preset with the hardcoded default would push this above
    # 1e-3, so the bound flags it.
    assert legacy['weight_decay'] < 1e-3, (
        f"Q-100ms weight_decay should be Optuna-tuned (~1.4e-5), "
        f"got {legacy['weight_decay']}; suggests the preset drifted or "
        f"the historical 3e-4 default has leaked back in."
    )


# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------

def test_slug_encodes_all_axes():
    # PCA-loss targets carry the pca_basis suffix so all_trials /
    # condmean / residual runs don't collide on disk. Default is
    # 'all_trials'.
    cfg = default_config_for_target('Q', bin_size_ms=50)
    assert cfg.slug() == 'Q_PCA_half_50ms_all'

    cfg_cm = default_config_for_target('Q', bin_size_ms=50,
                                       pca_basis='condition_mean')
    assert cfg_cm.slug() == 'Q_PCA_half_50ms_condmean'

    cfg_res = default_config_for_target('Q', bin_size_ms=50,
                                        pca_basis='residual')
    assert cfg_res.slug() == 'Q_PCA_half_50ms_residual'

    # Non-PCA losses are not suffixed (slug stable across the change).
    cfg_choice = default_config_for_target('choice')
    assert cfg_choice.slug() == 'choice_CE_half_100ms'


def test_output_dir_is_nested_under_run_name():
    cfg = default_config_for_target('Q', run_name='production_2026_05_06',
                                     bin_size_ms=50)
    p = cfg.output_dir('results')
    assert p == Path('results/production_2026_05_06/Q_PCA_half_50ms_all')


def test_invalid_pca_basis_raises():
    with pytest.raises(ValueError, match='Unknown pca_basis'):
        Config(target_type='Q', loss_func='PCA', pca_basis='garbage')


def test_to_legacy_dict_carries_pca_basis():
    """pca_basis selects what the loss-basis PCA is fit on: the raw
    per-trial training targets ('all_trials', the current default),
    per-condition averaged targets ('condition_mean'), or per-trial
    (target - cond_mean) deviations ('residual'). Loss is scored in the
    chosen basis; the model still predicts the raw target. Ignored for
    CE/MSE losses, but always present on the legacy dict for schema
    consistency / provenance."""
    cfg_cm = default_config_for_target('Q', bin_size_ms=100,
                                        pca_basis='condition_mean')
    cfg_res = default_config_for_target('Q', bin_size_ms=100,
                                         pca_basis='residual')
    assert cfg_cm.to_legacy_dict()['pca_basis'] == 'condition_mean'
    assert cfg_res.to_legacy_dict()['pca_basis'] == 'residual'


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
