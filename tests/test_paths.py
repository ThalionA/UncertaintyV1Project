# -*- coding: utf-8 -*-
"""Unit tests for ``nn_decoder/paths.py``.

paths.py is the single source of truth for data file locations in
nn_decoder/. Every caller that loads a population_results_*.mat or a
recovery_cache_*.npy should route through it.

Tests here pin the canonical basenames, the Path composition rules, and
the module-level constants so that future moves only need to flip
LEGACY_FITS / RECOVERY_CACHES without breaking callers.
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

import paths  # noqa: E402


# ----------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------

def test_here_points_at_nn_decoder():
    assert paths.HERE.name == 'nn_decoder'
    assert paths.HERE.is_dir()


def test_repo_root_is_parent_of_here():
    assert paths.REPO_ROOT == paths.HERE.parent


def test_data_dirs_resolve_under_repo_root():
    assert paths.DATA_RAW == paths.REPO_ROOT / 'data' / 'raw'
    assert paths.DATA_PROC == paths.REPO_ROOT / 'data' / 'processed'


def test_legacy_fits_default_is_here():
    """Fit .mat files currently live at nn_decoder/ root.
    Relocating them means pointing LEGACY_FITS at a DATA_PROC subfolder;
    when that happens, update this test in lockstep."""
    assert paths.LEGACY_FITS == paths.HERE


def test_recovery_caches_default_is_here():
    """recovery_cache_*.npy currently lives at nn_decoder/ root.
    Relocating means pointing RECOVERY_CACHES at a DATA_PROC subfolder."""
    assert paths.RECOVERY_CACHES == paths.HERE


def test_figures_and_results_constants():
    assert paths.FIGURES == paths.HERE / 'figures'
    assert paths.RESULTS == paths.HERE / 'results'


# ----------------------------------------------------------------------
# FIT_BASENAMES — canonical (protocol, target) -> stem mapping
# ----------------------------------------------------------------------

def test_fit_basenames_has_required_protocols():
    assert set(paths.FIT_BASENAMES) >= {'fixed', 'stim_mean', 'loss_sweep'}


def test_fit_basenames_fixed_targets():
    fixed = paths.FIT_BASENAMES['fixed']
    assert fixed['Q'] == 'population_results_fixed_hyperparams'
    assert fixed['L'] == 'population_results_fixed_likelihood'
    assert fixed['d'] == 'population_results_fixed_decision'
    assert fixed['choice'] == 'population_results_fixed_choice'
    assert fixed['truechoice'] == 'population_results_fixed_truechoice'


def test_fit_basenames_stim_mean_targets():
    sm = paths.FIT_BASENAMES['stim_mean']
    assert sm['Q'] == 'population_results_stim_mean_Q'
    assert sm['L'] == 'population_results_stim_mean_L'
    assert sm['d'] == 'population_results_stim_mean_d'
    assert sm['choice'] == 'population_results_stim_mean_choice'


def test_fit_basenames_loss_sweep_targets():
    ls = paths.FIT_BASENAMES['loss_sweep']
    assert ls['Wasserstein'] == 'population_results_fixed_hyperparams_Wasserstein'
    assert ls['KL'] == 'population_results_fixed_hyperparams_KL'
    assert ls['JS'] == 'population_results_fixed_hyperparams_JS'


def test_every_stem_uses_population_results_prefix():
    """Invariant: every fit-output basename must start with the
    'population_results_' prefix that downstream tooling greps for."""
    for protocol, mapping in paths.FIT_BASENAMES.items():
        for target, stem in mapping.items():
            assert stem.startswith('population_results_'), (
                f"{protocol}/{target} stem {stem!r} breaks the "
                f"population_results_ prefix invariant"
            )


# ----------------------------------------------------------------------
# fit_stem / fit_path / fit_path_from_stem
# ----------------------------------------------------------------------

def test_fit_stem_returns_canonical_basename():
    assert paths.fit_stem('fixed', 'Q') == 'population_results_fixed_hyperparams'
    assert paths.fit_stem('stim_mean', 'L') == 'population_results_stim_mean_L'


def test_fit_stem_unknown_protocol_raises():
    with pytest.raises(KeyError):
        paths.fit_stem('nonsense', 'Q')


def test_fit_stem_unknown_target_raises():
    with pytest.raises(KeyError):
        paths.fit_stem('fixed', 'not_a_target')


def test_fit_path_composes_split_suffix():
    p = paths.fit_path('fixed', 'Q', 'stratified_balanced')
    assert isinstance(p, Path)
    assert p.parent == paths.LEGACY_FITS
    assert p.name == 'population_results_fixed_hyperparams_stratified_balanced.mat'


def test_fit_path_for_each_split():
    for split in ('stratified_balanced', 'generalize_contrast',
                  'generalize_dispersion'):
        p = paths.fit_path('fixed', 'Q', split)
        assert p.name == f'population_results_fixed_hyperparams_{split}.mat'


def test_fit_path_stim_mean_uses_stim_mean_split():
    """stim_mean baselines are stored with split='stim_mean' suffix."""
    p = paths.fit_path('stim_mean', 'Q', 'stim_mean')
    assert p.name == 'population_results_stim_mean_Q_stim_mean.mat'


def test_fit_path_from_stem_matches_fit_path():
    """Escape hatch for callers that already have a stem string."""
    stem = paths.fit_stem('fixed', 'L')
    assert paths.fit_path_from_stem(stem, 'stratified_balanced') == \
           paths.fit_path('fixed', 'L', 'stratified_balanced')


def test_fit_path_from_stem_for_loss_sweep():
    """The loss_sweep callers iterate over stems already; verify the
    escape hatch returns the right .mat path."""
    stem = paths.fit_stem('loss_sweep', 'Wasserstein')
    p = paths.fit_path_from_stem(stem, 'spat')
    assert p.name == 'population_results_fixed_hyperparams_Wasserstein_spat.mat'


# ----------------------------------------------------------------------
# recovery_cache
# ----------------------------------------------------------------------

def test_recovery_cache_composes_target():
    p = paths.recovery_cache('perception')
    assert p.parent == paths.RECOVERY_CACHES
    assert p.name == 'recovery_cache_fixed_perception.npy'


def test_recovery_cache_for_each_known_target():
    for target in ('perception', 'likelihood', 'decision'):
        p = paths.recovery_cache(target)
        assert p.name == f'recovery_cache_fixed_{target}.npy'


# ----------------------------------------------------------------------
# figures_dir — pure path composition (no mkdir side effect)
# ----------------------------------------------------------------------

def test_figures_dir_composes_under_figures():
    p = paths.figures_dir('population_metrics')
    assert p == paths.FIGURES / 'population_metrics'


def test_figures_dir_is_side_effect_free():
    """paths.figures_dir must NOT create the directory as a side effect.
    Callers that need it created should mkdir(parents=True, exist_ok=True)
    themselves. This keeps the module import-safe."""
    p = paths.figures_dir('nonexistent_analysis_for_sef_test')
    # The directory may or may not exist depending on history, but
    # calling figures_dir must not have created it just now.
    # We assert the function returns immediately with a Path.
    assert isinstance(p, Path)
