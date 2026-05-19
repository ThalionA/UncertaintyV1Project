# -*- coding: utf-8 -*-
"""Single source of truth for data file locations in nn_decoder/.

Every caller that loads a ``population_results_*.mat`` or a
``recovery_cache_*.npy`` should route through this module. Hardcoded
basenames anywhere else are a bug — they spread the knowledge of where
the data lives across the codebase, which is exactly what made the
audit's "move 600 MB of fit outputs" (items 3.5, 3.6) so expensive.

Phase A: ``LEGACY_FITS`` and ``RECOVERY_CACHES`` point at ``nn_decoder/``
itself, matching today's on-disk layout. Phase C of the action plan
moves the files to ``data/processed/`` and flips just those two
constants; no caller needs to change.

Usage::

    from paths import fit_path, recovery_cache, FIT_BASENAMES

    # Resolve a single fit by (protocol, target, split):
    mat = sio.loadmat(fit_path('fixed', 'Q', 'stratified_balanced'),
                      simplify_cells=True)

    # Iterate over a protocol's targets:
    for target, stem in FIT_BASENAMES['fixed'].items():
        ...

    # Recovery cache for a target:
    np.load(recovery_cache('perception'), allow_pickle=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


# ----------------------------------------------------------------------
# Directory anchors
# ----------------------------------------------------------------------

HERE      = Path(__file__).resolve().parent            # nn_decoder/
REPO_ROOT = HERE.parent

DATA_RAW  = REPO_ROOT / 'data' / 'raw'
DATA_PROC = REPO_ROOT / 'data' / 'processed'

# Where pre-refactor fit outputs and recovery caches live *today*. Both
# default to nn_decoder/ to match the current on-disk reality. Phase C
# will flip these to DATA_PROC subfolders; nothing else has to change.
LEGACY_FITS     = HERE
RECOVERY_CACHES = HERE

FIGURES = HERE / 'figures'
RESULTS = HERE / 'results'


# ----------------------------------------------------------------------
# Canonical basename mapping  (the only place these strings live)
# ----------------------------------------------------------------------
#
# Structure:  protocol -> target -> basename stem (no split, no extension)
#
# The "stem" is the filename prefix; full files are formed as
# ``f"{stem}_{split}.mat"``.

FIT_BASENAMES: dict[str, dict[str, str]] = {
    'fixed': {
        'Q':          'population_results_fixed_hyperparams',
        'L':          'population_results_fixed_likelihood',
        'd':          'population_results_fixed_decision',
        'choice':     'population_results_fixed_choice',
        'truechoice': 'population_results_fixed_truechoice',
    },
    'stim_mean': {
        'Q':      'population_results_stim_mean_Q',
        'L':      'population_results_stim_mean_L',
        'd':      'population_results_stim_mean_d',
        'choice': 'population_results_stim_mean_choice',
    },
    # Q-target fits with non-default loss functions
    'loss_sweep': {
        'Wasserstein': 'population_results_fixed_hyperparams_Wasserstein',
        'KL':          'population_results_fixed_hyperparams_KL',
        'JS':          'population_results_fixed_hyperparams_JS',
    },
}

Split = Literal[
    'stratified_balanced',
    'generalize_contrast',
    'generalize_dispersion',
    'stim_mean',
    'spat',
    'temp',
]


# ----------------------------------------------------------------------
# Resolvers
# ----------------------------------------------------------------------

def fit_stem(protocol: str, target: str) -> str:
    """Return the bare basename stem for a (protocol, target).

    Raises ``KeyError`` on unknown protocol or target so typos surface
    loudly instead of producing a "file not found" further downstream.
    """
    return FIT_BASENAMES[protocol][target]


def fit_path(protocol: str, target: str, split: Split) -> Path:
    """Resolve a fit ``.mat`` file by (protocol, target, split)."""
    return LEGACY_FITS / f"{fit_stem(protocol, target)}_{split}.mat"


def fit_path_from_stem(stem: str, split: Split) -> Path:
    """Escape hatch for callers that already have a stem string —
    e.g. ``io_coherence`` iterates over ``(target_type, stem)`` tuples
    that combine multiple protocols. Prefer ``fit_path`` when you have
    the (protocol, target) pair."""
    return LEGACY_FITS / f"{stem}_{split}.mat"


def recovery_cache(target: str) -> Path:
    """Resolve a recovery-cache ``.npy`` file, e.g.
    ``recovery_cache('perception')`` →
    ``recovery_cache_fixed_perception.npy``."""
    return RECOVERY_CACHES / f"recovery_cache_fixed_{target}.npy"


def figures_dir(analysis: str) -> Path:
    """Pure path composition for ``figures/<analysis>/``.

    Does NOT create the directory — callers that need it created
    should ``mkdir(parents=True, exist_ok=True)`` themselves. Keeping
    this side-effect-free means the module is safe to import in test
    setup paths."""
    return FIGURES / analysis


__all__ = [
    'HERE', 'REPO_ROOT', 'DATA_RAW', 'DATA_PROC',
    'LEGACY_FITS', 'RECOVERY_CACHES', 'FIGURES', 'RESULTS',
    'FIT_BASENAMES', 'Split',
    'fit_stem', 'fit_path', 'fit_path_from_stem',
    'recovery_cache', 'figures_dir',
]
