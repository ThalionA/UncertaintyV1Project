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


def _matlab_safe_config(cfg: Mapping) -> dict:
    """Make a legacy-config dict serialisable by ``scipy.io.savemat``.

    ``savemat`` raises ``TypeError: Could not convert None`` on any None-valued
    entry, and the legacy dict carries several genuinely optional fields whose
    no-op value is None (``seed``, ``n_neural_pcs``). None is stored as an empty
    array — the same convention ``run_experiment`` already uses for an absent
    PCA basis (``Distr['pcs'] = ... if pcs is not None else []``).

    Values are otherwise passed through untouched, so this only changes runs
    that previously *crashed* on save; nothing that already worked is affected.
    Round-tripping through ``loadmat`` returns an empty array, not None, which
    is why consumers should test these fields for emptiness rather than
    identity. (Found 2026-07-29: adding the optional ``n_neural_pcs`` field made
    this fire on every run that left it at its default.)
    """
    return {k: ([] if v is None else v) for k, v in cfg.items()}


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


def _shard_path(out_dir: Path, split: str, mouse_id: int) -> Path:
    """Path to one mouse's shard for one split."""
    return out_dir / 'shards' / f'{split}__mouse_{mouse_id}.mat'


def _checkpoint_path(out_dir: Path, split: str, mouse_id: int) -> Path:
    """Path to one mouse's checkpoint for one split."""
    return out_dir / 'checkpoints' / f'mouse_{mouse_id}_{split}.pt'


def merge_shards(out_dir: Path, split: str,
                  delete_shards: bool = False) -> Path | None:
    """Combine per-mouse shards under ``<out_dir>/shards/`` into the
    canonical ``<out_dir>/<split>.mat``.

    Mirrors what the in-process ``run_config(merge=True)`` path writes
    — a single ``.mat`` with ``{'results': {mouse_<mid>: ...},
    'config': cfg}`` — so downstream loaders (``load_run_tree``, the
    plot suite, ``recovery_sanity_check``) read the merged output
    identically to a single-process run.

    Parameters
    ----------
    out_dir : Path
        The slug directory (e.g.
        ``results/loss_sweep_2026_05_27/Q_PCA_half_100ms_all``).
    split : str
        Split name; combined with shards globbed as
        ``shards/<split>__mouse_*.mat``.
    delete_shards : bool
        If True, remove the per-mouse shard files after a successful
        merge. Default False — shards are tiny and useful for
        debugging (e.g. spotting a divergent mouse). Set True to clean
        up after a sweep.

    Returns
    -------
    Path or None
        Path to the merged ``.mat``, or ``None`` if no shards were
        found (e.g. the loss was never run).
    """
    out_dir = Path(out_dir)
    shards_dir = out_dir / 'shards'
    if not shards_dir.is_dir():
        return None
    shards = sorted(shards_dir.glob(f'{split}__mouse_*.mat'))
    if not shards:
        return None

    merged_results: dict = {}
    merged_config = None
    for sp in shards:
        m = sio.loadmat(str(sp), simplify_cells=True)
        merged_results.update(m['results'])
        # All shards from the same run carry an identical config block;
        # the last one wins (they should all agree).
        merged_config = m.get('config', merged_config)

    final_path = out_dir / f'{split}.mat'
    sio.savemat(str(final_path),
                {'results': merged_results, 'config': merged_config})
    print(f"Merged {len(shards)} shard(s) -> {final_path.name} "
           f"({len(merged_results)} animals)")

    if delete_shards:
        for sp in shards:
            sp.unlink()
        # Clean up the (now-empty) shards directory.
        try:
            shards_dir.rmdir()
        except OSError:
            pass  # not empty (e.g. an in-flight shard); leave it

    return final_path


def run_config(
    config: Config,
    mouse_ids: Iterable[int] = range(6),
    splits: Iterable[str] = ('stratified_balanced',
                              'generalize_contrast',
                              'generalize_dispersion'),
    results_root: str | None = None,
    on_error: str = 'continue',
    merge: bool = True,
    skip_existing: bool = True,
) -> Path:
    """Run a Config across (mice x splits), saving per-mouse shards
    under ``<out_dir>/shards/`` and optionally merging them into the
    canonical ``<out_dir>/<split>.mat`` at the end.

    Each mouse is independent: its result is written immediately to a
    per-mouse shard on success. This makes the run idempotent (a
    crashed or killed run resumes by picking up the missing mice) and
    enables external parallel orchestration (one process per mouse —
    see ``run_loss_sweep_per_mouse.sh``). When ``merge=True`` (default,
    single-process invocation), the shards are combined into the
    canonical ``<split>.mat`` before this function returns; downstream
    consumers (``load_run_tree``, the plot suite,
    ``recovery_sanity_check``) read the merged file unchanged.

    Parameters
    ----------
    config : Config
        The training configuration. Determines the output directory via
        ``config.output_dir(results_root)``.
    mouse_ids : iterable of int
        Mice to process. Default 0..5.
    splits : iterable of str
        Train/test splits to run. Default all three production splits.
    results_root : str or None
        Root directory for the nested output tree. Default ``None`` →
        ``paths.RESULTS`` (``nn_decoder/results/``, anchored to the package
        location so the output tree is independent of the invocation cwd).
    on_error : str
        ``'continue'`` (default) skips a failing mouse and proceeds;
        ``'raise'`` aborts the entire run on any single-mouse failure.
    merge : bool
        Whether to merge per-mouse shards into the canonical
        ``<split>.mat`` after all mice for a given split have run.
        Set ``False`` when invoking ``run_config`` per-mouse from a
        parallel wrapper; the wrapper should call
        :func:`merge_shards` once all mice have completed.
    skip_existing : bool
        Skip any (mouse, split) pair whose shard already exists on
        disk. Idempotent restart — re-running the same config picks
        up where a previous run left off. Set ``False`` to force
        re-fitting every mouse.

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

    if results_root is None:
        from paths import RESULTS as results_root
    out_dir = config.output_dir(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'shards').mkdir(exist_ok=True)

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

        ran_any = False
        run_failed = False
        for mid in mouse_ids:
            shard_path = _shard_path(out_dir, split, mid)
            ckpt_path = _checkpoint_path(out_dir, split, mid)

            if skip_existing and shard_path.exists():
                # Checkpoint may or may not exist (older runs didn't
                # always populate it); the shard alone is the
                # authoritative "this mouse is done" marker.
                print(f"  [skip] mouse {mid}: shard already exists "
                      f"({shard_path.name})")
                continue

            print(f"  --> Processing Mouse {mid}...")
            try:
                animal_results = run_animal_decoder(cfg, mid)
            except Exception as exc:
                print(f"  [!] Failed for Mouse {mid}: {exc}")
                traceback.print_exc()
                if on_error == 'raise':
                    raise
                run_failed = True
                continue

            # Save the checkpoint (torch tensors → .pt) BEFORE the shard
            # so a kill between the two writes leaves the shard absent
            # and the next run repeats the mouse cleanly. The checkpoint
            # being orphaned is harmless — it gets overwritten on rerun.
            single = {f'mouse_{mid}': animal_results}
            _pop_and_save_checkpoints(single, out_dir, split)

            sio.savemat(str(shard_path),
                        {'results': single, 'config': _matlab_safe_config(cfg)})
            ran_any = True
            print(f"  saved shard {shard_path.name}")

        if not ran_any:
            print(f"No new mice fit for split={split} "
                   f"(all shards present or all failed).")

        if merge and not run_failed:
            merge_shards(out_dir, split)
        elif merge and run_failed:
            print(f"  [!] skipping merge for split={split}: "
                   f"some mice failed (re-run to fill in, then merge).")

    return out_dir
