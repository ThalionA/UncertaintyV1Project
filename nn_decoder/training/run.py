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
from typing import Iterable

import scipy.io as sio

# run_experiment_v26 is at the project's nn_decoder/ level. This module
# lives one directory deeper (nn_decoder/training/), so add the parent
# to sys.path defensively.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from .config import Config


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

    # Lazy import: run_experiment_v26 pulls torch at module top, so we
    # only import it when we actually need to run a fit. Lets tests that
    # exercise Config / make_target import this module without torch.
    from run_experiment_v26 import run_animal_decoder

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
            save_path = out_dir / f'{split}.mat'
            sio.savemat(str(save_path),
                        {'results': all_mice_results, 'config': cfg})
            tag = ' (partial)' if run_failed else ''
            print(f"Saved {save_path.name}{tag} ({len(all_mice_results)} animals)")
        else:
            print(f"No animals succeeded for split={split}; not writing.")

    return out_dir
