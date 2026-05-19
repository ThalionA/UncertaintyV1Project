# -*- coding: utf-8 -*-
"""Single config-driven driver replacing the four ``run_fixed_hyperparams_*.py``
ad-hoc loops. Writes a structured nested tree:

    <results_root>/<run_name>/<slug>/
        config.yaml
        stratified_balanced.mat
        generalize_contrast.mat
        generalize_dispersion.mat

where ``<slug>`` encodes (target, loss, time_window, bin_size_ms).

Existing analysis scripts read flat-named files (e.g.
``population_results_fixed_hyperparams_<split>.mat``); they will be
updated to glob the new tree in a follow-up pass once the fresh run
lands.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, Mapping

import scipy.io as sio

# run_experiment is at the project's nn_decoder/ level. This module
# lives one directory deeper (nn_decoder/training/), so add the parent
# to sys.path defensively.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from .config import Config


def _pop_and_save_checkpoints(all_mice_results: Mapping[str, dict],
                                out_dir: Path, split: str) -> None:
    """Strip 'Checkpoints' out of each animal's results dict and save
    each to ``<out_dir>/checkpoints/mouse_<mid>_<split>.pt``.

    Mutates ``all_mice_results`` so the subsequent ``sio.savemat`` call
    doesn't try to serialise torch tensors (which scipy can't handle).
    Skips animals whose dict has no ``Checkpoints`` key — keeps the
    saver tolerant of older ``run_animal_decoder`` versions that didn't
    populate this field.
    """
    ckpts_to_save = {}
    for key, animal_res in all_mice_results.items():
        ckpt = animal_res.pop('Checkpoints', None)
        if ckpt is None:
            continue
        # key is "mouse_<mid>"; strip the prefix for the filename.
        mid = key.split('_', 1)[1] if '_' in key else key
        ckpts_to_save[mid] = ckpt
    if not ckpts_to_save:
        return
    # Lazy torch import — keeps `run_config` importable in environments
    # that have scipy but not torch (e.g. for tests of the config layer).
    import torch
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for mid, ckpt in ckpts_to_save.items():
        ckpt_path = ckpt_dir / f'mouse_{mid}_{split}.pt'
        torch.save(ckpt, str(ckpt_path))
    print(f"Saved {len(ckpts_to_save)} checkpoint(s) under "
           f"{ckpt_dir.relative_to(out_dir.parent.parent) if out_dir.is_absolute() else ckpt_dir}")


def run_config(
    config: Config,
    mouse_ids: Iterable[int] = range(6),
    splits: Iterable[str] = ('stratified_balanced',
                              'generalize_contrast',
                              'generalize_dispersion'),
    results_root: str = 'results',
    on_error: str = 'continue',
) -> Path:
    """Run a Config across (mice x splits), saving per-split .mat files
    plus a config.yaml provenance dump.

    Parameters
    ----------
    config : Config
        The training configuration. Determines the output directory via
        ``config.output_dir(results_root)``.
    mouse_ids : iterable of int
        Mice to process. Default 0..5.
    splits : iterable of str
        Train/test splits to run. Default all three production splits.
    results_root : str
        Root directory for the nested output tree. Default ``'results'``.
    on_error : str
        ``'continue'`` (default) skips a failing mouse and proceeds;
        ``'raise'`` aborts the entire run on any single-mouse failure.

    Returns
    -------
    Path
        The output directory (``results_root/run_name/slug``).
    """
    if on_error not in ('continue', 'raise'):
        raise ValueError(f"on_error must be 'continue' or 'raise', got {on_error!r}")

    # Lazy import: run_experiment pulls torch at module top, so we
    # only import it when we actually need to run a fit. Lets tests that
    # exercise Config / make_target import this module without torch.
    from run_experiment import run_animal_decoder

    out_dir = config.output_dir(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provenance dump first — even if subsequent runs fail, we know the
    # config that was attempted.
    config.save_yaml(out_dir / 'config.yaml')

    legacy_base = config.to_legacy_dict()

    for split in splits:
        cfg = dict(legacy_base)
        cfg['split_type'] = split

        print()
        print('=' * 60)
        print(f"RUNNING: {config.target_type} ({config.loss_func}) "
              f"| {config.time_window} {config.bin_size_ms}ms "
              f"| split = {split}")
        print('=' * 60)

        all_mice_results = {}
        run_failed = False
        for mid in mouse_ids:
            print(f"  --> Processing Mouse {mid}...")
            try:
                animal_results = run_animal_decoder(cfg, mid)
                all_mice_results[f"mouse_{mid}"] = animal_results
            except Exception as exc:
                print(f"  [!] Failed for Mouse {mid}: {exc}")
                traceback.print_exc()
                if on_error == 'raise':
                    raise
                run_failed = True
                # continue to next mouse with on_error='continue'

        if all_mice_results:
            # Extract Checkpoints (torch tensors) before the .mat write —
            # scipy.io.savemat can't serialise torch tensors. The
            # Checkpoints bundle ends up at out_dir/checkpoints/.
            _pop_and_save_checkpoints(all_mice_results, out_dir, split)

            save_path = out_dir / f'{split}.mat'
            sio.savemat(str(save_path),
                        {'results': all_mice_results, 'config': cfg})
            tag = ' (partial)' if run_failed else ''
            print(f"Saved {save_path.name}{tag} ({len(all_mice_results)} animals)")
        else:
            print(f"No animals succeeded for split={split}; not writing.")

    return out_dir
