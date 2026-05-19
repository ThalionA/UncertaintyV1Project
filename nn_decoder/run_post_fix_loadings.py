# -*- coding: utf-8 -*-
"""Training driver for the post-bug-fix decoder runs.

Spyder-friendly: open this file and run, or call ``main()`` from a
cell. The driver is declarative — every configuration we want to
train is registered in ``CONFIGS`` below — and self-resumes: any
configuration whose output directory already contains all three
split .mat files is skipped (unless ``force=True``).

Default lineup, all at ``bin_size_ms=100`` under one ``RUN_NAME``:

  results/<RUN_NAME>/
    Q_PCA_half_100ms_condmean/    # Q, current cond-mean basis, post bug fix
    Q_PCA_half_100ms_residual/    # Q, new residual basis, post bug fix
    L_PCA_half_100ms_condmean/    # L, cond-mean basis
    L_PCA_half_100ms_residual/    # L, residual basis
    d_MSE_half_100ms/             # d, MSE (pca_basis n/a — no PCA loss)
    stim_cat_CE_half_100ms/       # stim_cat, hidden_sizes=[32] matched to Q

Each split .mat file contains, per mouse, both the trained-decoder
predictions (``Dist``) and the trained MLP weights (``Weights``).

Total fits when all configs are run from scratch:
  6 configs * 6 mice * 3 splits = 108 fits, roughly 2-3 hours on MPS.
The skip-if-complete logic means re-running this script after a
partial completion only trains what's missing.

Post-fix relative to pre-2026-05-17:
  - weight_decay is now honoured end-to-end (Optuna-tuned per-target
    presets flow through instead of the hardcoded 3e-4 from
    train_and_select_best_model).
  - pca_basis flag selects the loss-basis fit (condition_mean default,
    or residual which isolates within-cell trial deviations).
  - The trained MLP weights are saved alongside the predictions, into
    a new ``Weights`` key per mouse.

See NOTES.md "Session 2026-05-17" for context.
"""

from __future__ import annotations

# Defensive: re-bind the real builtin ``all`` (and friends used in the
# skip-decision below). Scripts invoked via Spyder's ``%runcell -i`` use
# the kernel namespace as globals, so a previous ``from numpy import *``
# (or any other shadowing import) in the same kernel will silently make
# ``all(<generator>)`` evaluate the generator object as a 0-d object
# array — always truthy — and the skip-if-complete check returns True
# for every directory regardless of contents. See GOTCHAS.md
# "%runcell -i" entry. Same defence as nn_decoder/recovery_convergence_probe.py.
from builtins import all  # noqa: A004

from pathlib import Path
from typing import Iterable, Optional

from training import default_config_for_target, run_config


RUN_NAME = 'post_fix_loadings_2026_05_17'

# ----------------------------------------------------------------------
# Declarative lineup. Each entry is (key, target_type, overrides).
#   key       — short human-readable name used by `which=`
#   target_type — short Config name (Q, L, d, choice, stim_kernel, stim_cat)
#   overrides — kwargs threaded into default_config_for_target
# ----------------------------------------------------------------------

CONFIGS = [
    ('Q_condmean',  'Q',        dict(pca_basis='condition_mean')),
    ('Q_residual',  'Q',        dict(pca_basis='residual')),
    ('L_condmean',  'L',        dict(pca_basis='condition_mean')),
    ('L_residual',  'L',        dict(pca_basis='residual')),
    ('d',           'd',        dict()),                          # MSE — pca_basis n/a
    ('stim_cat',    'stim_cat', dict(hidden_sizes=[32])),         # CE, matched to Q for loadings
]

SPLITS = ('stratified_balanced', 'generalize_contrast', 'generalize_dispersion')


def _is_complete(out_dir: Path) -> bool:
    """A slug is considered already-done if all 3 split .mat files
    exist. Used to skip Q + stim_cat on a re-run after they've completed.

    Implemented as a manual loop rather than ``all(<generator>)`` so the
    check is robust even when the builtin ``all`` has been shadowed by
    a kernel-level import (e.g. ``from numpy import *``); see
    ``GOTCHAS.md`` "%runcell -i" entry. The earlier ``all(...)`` form
    of this function returned True for every directory regardless of
    contents — fixed 2026-05-17.
    """
    for s in SPLITS:
        if not (out_dir / f'{s}.mat').exists():
            return False
    return True


def _file_status(out_dir: Path) -> str:
    """Return a short string showing which of the 3 split .mat files
    are present. Used by the runner's diagnostic log so it's clear from
    the console output whether a SKIP was justified."""
    present = [(out_dir / f'{s}.mat').exists() for s in SPLITS]
    marks = ''.join('+' if x else '-' for x in present)
    return f"[{marks}]"  # e.g. '[+++]' (complete), '[---]' (none), '[+-+]' (partial)


def main(run_name: str = RUN_NAME,
         which: Optional[Iterable[str]] = None,
         force: bool = False,
         results_root: str = 'results'):
    """Drive the post-fix training lineup.

    Parameters
    ----------
    run_name : str
        Output run directory name under ``results/``.
    which : iterable of CONFIGS keys, optional
        If given, only the named configurations are considered. Default
        is all six.
    force : bool
        If True, re-train regardless of existing outputs. Default False
        (skip any slug whose 3 split .mat files are already on disk).
    results_root : str
        Root for the output tree.
    """
    selected = list(which) if which else [k for k, _, _ in CONFIGS]
    unknown = set(selected) - {k for k, _, _ in CONFIGS}
    if unknown:
        raise ValueError(
            f"Unknown config keys: {sorted(unknown)}. "
            f"Valid keys: {[k for k, _, _ in CONFIGS]}"
        )
    spec_by_key = {k: (target, overrides) for k, target, overrides in CONFIGS}

    print(f"Post-bug-fix training run: run_name={run_name!r}")
    print(f"  module version: 6 configs (Q condmean/residual, "
          f"L condmean/residual, d, stim_cat)")
    print(f"  selected      : {selected}  (n={len(selected)})")
    print(f"  force         : {force}")
    print(f"  cwd           : {Path.cwd()}")
    print(f"  results_root  : {Path(results_root).resolve()}")
    print()

    for i, key in enumerate(selected, 1):
        target, overrides = spec_by_key[key]
        cfg = default_config_for_target(
            target, bin_size_ms=100, run_name=run_name, **overrides
        )
        out_dir = Path(results_root) / run_name / cfg.slug()
        status = _file_status(out_dir)  # '[+++]' / '[---]' / '[+-+]' etc.
        print('=' * 60)
        if not force and _is_complete(out_dir):
            print(f"[{i}/{len(selected)}] SKIP  {key}  ({cfg.slug()})  "
                  f"— all 3 splits on disk {status}")
            print(f"        path: {out_dir}/")
            print('=' * 60)
            continue
        print(f"[{i}/{len(selected)}] RUN   {key}  ({cfg.slug()})  "
              f"on-disk status: {status}")
        print(f"        path: {out_dir}/")
        print('=' * 60)
        run_config(cfg, results_root=results_root)

    print()
    print('=' * 60)
    print(f"Done. Outputs under {results_root}/{run_name}/")
    print('=' * 60)


if __name__ == '__main__':
    main()
