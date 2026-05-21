# -*- coding: utf-8 -*-
"""Neural-population scaling analysis for the V1 decoder.

Answers the meeting action item (2026-05-20): *"Scale number of decoded
neurons — check how performance scales."*

Two families of subset-construction strategies are supported:

  - **Random subsampling.** For each subset size N, draw ``n_draws``
    random N-neuron subsets and record decoder performance for each.
    Averaging across draws gives a mean curve with a Monte-Carlo spread.
  - **Targeted selection.** Rank neurons by a criterion, then build each
    size-N subset deterministically from the ranking — ``top`` takes the
    N best-ranked neurons, ``bottom`` the N worst. Three rankings ship:
    orientation-tuning strength, decoder input-weight magnitude, and
    mean activity. The set is pluggable — add a ``rank_by_*`` function
    and wire it into :func:`run_scaling_for_mouse`.

Every ``run_animal_decoder`` call already trains target-shuffled control
models alongside the real ones, so the shuffle baseline is recorded at
every N for free (model keys ``spat_shf`` / ``temp_shf``).

Design notes / caveats
----------------------
* Hyperparameters are held **fixed** at the per-target preset across all
  N. Re-tuning per N would be the rigorous choice but is infeasible
  (Optuna per N); the fixed-hyperparameter curve therefore conflates
  population size with a capacity / regularisation mismatch at small N.
  Treat absolute small-N values with that caveat in mind.
* Neuron rankings are computed on **training trials only** (the same
  stratified split ``run_animal_decoder`` uses) so subset selection does
  not leak held-out information.
* Rankings are computed on **raw** (pre-z-score) activity — z-scoring is
  a per-neuron affine transform and would distort the modulation index
  and the firing-rate ranking.

Output is a tidy long-format table — one row per
``(mouse, selection, direction, n_neurons, draw, model)`` — written per
mouse (resumable) and as an aggregate CSV. Plotting the curves lives in a
separate script, per the repo's processing / visualisation separation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import RESULTS

# The four model keys every run_animal_decoder call returns: the real
# spatial (PPC) and temporal (SBC) decoders, plus their target-shuffled
# controls.
MODEL_KEYS = ('spat', 'temp', 'spat_shf', 'temp_shf')

DEFAULT_STRATEGIES = (
    'random', 'orientation_tuning', 'mean_activity', 'weight_magnitude')

ROW_COLUMNS = ['mouse', 'target', 'split', 'selection', 'direction',
               'n_neurons', 'draw', 'model', 'fit_loss']

__all__ = [
    'MODEL_KEYS', 'DEFAULT_STRATEGIES', 'ROW_COLUMNS',
    'build_neuron_grid', 'training_indices_for',
    'rank_by_orientation_tuning', 'rank_by_mean_activity',
    'rank_by_weight_magnitude',
    'subsets_from_ranking', 'iter_random_subsets',
    'mean_fit_loss', 'result_rows',
    'run_scaling_for_mouse', 'run_scaling',
]


# ----------------------------------------------------------------------
# Neuron-count grid
# ----------------------------------------------------------------------

def build_neuron_grid(n_neurons, step=10):
    """Subset sizes to sweep for a mouse with ``n_neurons`` recorded.

    ``[step, 2*step, ...]`` up to ``n_neurons``, with the full population
    always present as the final entry (appended only if it is not
    already a step multiple, so it never appears twice). Mice have
    different population sizes, so the grid is capped per mouse.
    """
    if n_neurons < 1:
        raise ValueError(f"n_neurons must be >= 1; got {n_neurons}")
    if step < 1:
        raise ValueError(f"step must be >= 1; got {step}")
    grid = list(range(step, n_neurons + 1, step))
    if not grid or grid[-1] != n_neurons:
        grid.append(int(n_neurons))
    return grid


def training_indices_for(trials, split='stratified_balanced', random_state=42):
    """Reproduce the training-trial indices ``run_animal_decoder`` uses.

    Neuron rankings are computed on these indices only, so subset
    selection never sees held-out trials. Mirrors the split logic in
    ``run_experiment.run_animal_decoder`` exactly (``random_state=42``).
    """
    from utils import (get_stratified_train_test_indices,
                        get_generalization_split_indices)
    if split == 'stratified_balanced':
        cond = np.column_stack([
            np.asarray(trials['orientation']).reshape(-1),
            np.asarray(trials['contrast']).reshape(-1),
            np.asarray(trials['dispersion']).reshape(-1),
        ])
        _, cats = np.unique(cond, axis=0, return_inverse=True)
        train_idx, _ = get_stratified_train_test_indices(
            cats, test_size=0.5, random_state=random_state)
        return train_idx
    if split in ('generalize_contrast', 'generalize_dispersion'):
        train_idx, _ = get_generalization_split_indices(
            trials, split_type=split, random_state=random_state)
        return train_idx
    raise ValueError(f"Unknown split {split!r}")


# ----------------------------------------------------------------------
# Neuron rankings  (each returns (ranking, scores), best-ranked first)
# ----------------------------------------------------------------------

def rank_by_orientation_tuning(activities_m, trials, train_indices=None):
    """Rank neurons by orientation-tuning strength.

    Tuning strength is the modulation index of the per-orientation
    tuning curve, ``(r_max - r_min) / (r_max + r_min)`` — bounded in
    ``[0, 1]`` for non-negative firing rates. The tuning curve is the
    time-and-trial-averaged response per unique orientation, pooled
    across contrast / dispersion (a neuron seen mostly at low contrast
    will read as less tuned — an acceptable noise source for a ranking;
    a max-contrast variant is a one-line change here).

    Parameters
    ----------
    activities_m : np.ndarray
        ``(n_neurons, n_trials, n_tbins)`` raw activity.
    trials : dict
        Must contain ``'orientation'``.
    train_indices : array-like of int, or None
        Restrict the tuning estimate to these trials (avoids leakage).

    Returns
    -------
    (ranking, scores) : (np.ndarray, np.ndarray)
        ``ranking`` indexes neurons most- to least-tuned;
        ``scores[i]`` is neuron ``i``'s modulation index.
    """
    act = activities_m
    ori = np.asarray(trials['orientation'], dtype=float).reshape(-1)
    if train_indices is not None:
        act = act[:, train_indices, :]
        ori = ori[train_indices]
    mean_resp = np.nanmean(act, axis=2)            # (n_neurons, n_trials_used)
    keep = ~np.isnan(ori)
    mean_resp = mean_resp[:, keep]
    ori = ori[keep]

    unique_oris = np.unique(ori)
    n_neurons = mean_resp.shape[0]
    tuning_curve = np.full((n_neurons, unique_oris.size), np.nan)
    for j, o in enumerate(unique_oris):
        m = (ori == o)
        if np.any(m):
            tuning_curve[:, j] = np.nanmean(mean_resp[:, m], axis=1)

    r_max = np.nanmax(tuning_curve, axis=1)
    r_min = np.nanmin(tuning_curve, axis=1)
    denom = r_max + r_min
    scores = np.where(np.abs(denom) > 1e-12, (r_max - r_min) / denom, 0.0)
    scores = np.nan_to_num(scores, nan=0.0)
    ranking = np.argsort(-scores, kind='stable')
    return ranking, scores


def rank_by_mean_activity(activities_m, train_indices=None):
    """Rank neurons by mean firing rate (highest first).

    Target-agnostic and cheap — a useful sanity contrast against the
    tuning-based ranking: if the scaling curve looks the same whether
    neurons are added by tuning or by raw activity, the decoder is not
    exploiting tuning structure specifically.
    """
    act = activities_m
    if train_indices is not None:
        act = act[:, train_indices, :]
    scores = np.nanmean(act, axis=(1, 2))
    order_key = np.nan_to_num(scores, nan=-np.inf)
    ranking = np.argsort(-order_key, kind='stable')
    return ranking, scores


def rank_by_weight_magnitude(*w_in):
    """Rank neurons by trained-decoder input-weight magnitude.

    Each ``W_in`` is a first-layer weight matrix ``(hidden, n_neurons)``
    from a full-population fit (``run_animal_decoder`` returns these in
    ``Weights['spat'|'temp']['W_in']``). A neuron's score is its column
    L2 norm, averaged across the supplied matrices (typically the
    spatial and temporal decoders). This is the decoder's own notion of
    "important" — mildly circular, since it needs a full fit first, but
    a direct importance measure.
    """
    if len(w_in) == 0:
        raise ValueError("rank_by_weight_magnitude needs >= 1 W_in array")
    norms = []
    n_neurons = None
    for W in w_in:
        W = np.asarray(W, dtype=float)
        if W.ndim != 2:
            raise ValueError(
                f"W_in must be 2-D (hidden, n_neurons); got ndim={W.ndim}")
        if n_neurons is None:
            n_neurons = W.shape[1]
        elif W.shape[1] != n_neurons:
            raise ValueError(
                f"W_in arrays disagree on n_neurons: {W.shape[1]} vs {n_neurons}")
        norms.append(np.linalg.norm(W, axis=0))
    scores = np.mean(np.stack(norms, axis=0), axis=0)
    ranking = np.argsort(-scores, kind='stable')
    return ranking, scores


# ----------------------------------------------------------------------
# Subset construction
# ----------------------------------------------------------------------

def subsets_from_ranking(ranking, sizes, direction='top'):
    """Build size-N neuron subsets from a ranking.

    ``direction='top'`` returns the N best-ranked neurons,
    ``'bottom'`` the N worst. Returns ``{size: indices}``.
    """
    ranking = np.asarray(ranking)
    if direction not in ('top', 'bottom'):
        raise ValueError(f"direction must be 'top' or 'bottom'; got {direction!r}")
    out = {}
    for n in sizes:
        n = int(n)
        if n > ranking.size:
            raise ValueError(
                f"requested {n} neurons but ranking has only {ranking.size}")
        out[n] = ranking[:n] if direction == 'top' else ranking[-n:]
    return out


def iter_random_subsets(n_neurons, sizes, n_draws, seed):
    """Yield ``(size, draw, indices)`` for random N-neuron subsets.

    Each subset is sampled without replacement. Deterministic given
    ``seed`` (uses ``np.random.default_rng``) so a run is reproducible
    and resumable.
    """
    rng = np.random.default_rng(seed)
    for size in sizes:
        for draw in range(n_draws):
            idx = np.sort(rng.choice(n_neurons, size=int(size), replace=False))
            yield int(size), draw, idx


# ----------------------------------------------------------------------
# Performance extraction
# ----------------------------------------------------------------------

def mean_fit_loss(result, model_key):
    """Mean held-out per-trial ``fit_loss`` for one model key.

    ``fit_loss`` is the canonical uncontaminated held-out test loss
    (see ``nn_decoder/audit/AUDIT_loss_consumers.md``); lower is better.
    """
    arr = np.asarray(result['fit_loss'][model_key], dtype=float).reshape(-1)
    if arr.size == 0:
        return float('nan')
    return float(np.nanmean(arr))


def result_rows(result, base):
    """Expand one ``run_animal_decoder`` result into four tidy rows —
    one per model key — each carrying the shared ``base`` metadata."""
    return [dict(base, model=k, fit_loss=mean_fit_loss(result, k))
            for k in MODEL_KEYS]


# ----------------------------------------------------------------------
# Per-mouse runner
# ----------------------------------------------------------------------

def run_scaling_for_mouse(mouse_id, config_legacy, activities_m, trials, *,
                          target, split, decoder_fn=None, preloaded=None,
                          step=10, n_draws=10, seed=0,
                          strategies=DEFAULT_STRATEGIES, progress=True):
    """Run the full scaling sweep for one animal.

    Trains a full-population decoder once (the shared end-point of every
    curve, and the source of the weight-magnitude ranking), then sweeps
    random subsets and targeted top/bottom subsets across the neuron
    grid. ``decoder_fn`` defaults to ``run_experiment.run_animal_decoder``
    — inject a stub to test the orchestration without torch training.

    Returns a tidy long-format ``DataFrame`` (columns: :data:`ROW_COLUMNS`).
    """
    if decoder_fn is None:
        from run_experiment import run_animal_decoder
        decoder_fn = run_animal_decoder

    n_neurons = int(activities_m.shape[0])
    grid = build_neuron_grid(n_neurons, step=step)
    sweep = [s for s in grid if s < n_neurons]   # full population done once below
    train_idx = training_indices_for(trials, split=split)

    def _base(selection, direction, n, draw):
        return dict(mouse=mouse_id, target=target, split=split,
                    selection=selection, direction=direction,
                    n_neurons=int(n), draw=int(draw))

    rows = []

    # --- Full population: shared end-point of every curve ---
    if progress:
        print(f"  [mouse {mouse_id}] full population (N={n_neurons})")
    full_res = decoder_fn(config_legacy, mouse_id,
                          neuron_subset=None, preloaded=preloaded)
    rows += result_rows(full_res, _base('full', 'na', n_neurons, -1))

    # --- Rankings for the targeted curves ---
    rankings = {}
    if 'orientation_tuning' in strategies:
        rankings['orientation_tuning'] = rank_by_orientation_tuning(
            activities_m, trials, train_indices=train_idx)[0]
    if 'mean_activity' in strategies:
        rankings['mean_activity'] = rank_by_mean_activity(
            activities_m, train_indices=train_idx)[0]
    if 'weight_magnitude' in strategies:
        rankings['weight_magnitude'] = rank_by_weight_magnitude(
            full_res['Weights']['spat']['W_in'],
            full_res['Weights']['temp']['W_in'])[0]

    # --- Random subsampling (seed offset per mouse for independent draws) ---
    if 'random' in strategies:
        for size, draw, idx in iter_random_subsets(
                n_neurons, sweep, n_draws, seed + mouse_id):
            if progress:
                print(f"  [mouse {mouse_id}] random            "
                      f"N={size:>4d} draw {draw}")
            res = decoder_fn(config_legacy, mouse_id,
                             neuron_subset=idx, preloaded=preloaded)
            rows += result_rows(res, _base('random', 'na', size, draw))

    # --- Targeted selection: top-first and bottom-first per ranking ---
    for name, ranking in rankings.items():
        for direction in ('top', 'bottom'):
            for size, idx in subsets_from_ranking(
                    ranking, sweep, direction=direction).items():
                if progress:
                    print(f"  [mouse {mouse_id}] {name:<18s} "
                          f"{direction:<6s} N={size:>4d}")
                res = decoder_fn(config_legacy, mouse_id,
                                 neuron_subset=idx, preloaded=preloaded)
                rows += result_rows(res, _base(name, direction, size, -1))

    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------

def _write_provenance(path, payload):
    """Dump a run-config JSON next to the outputs for provenance."""
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def run_scaling(mouse_ids=range(6), target='Q', split='stratified_balanced',
                bin_size_ms=100, step=10, n_draws=10, seed=0,
                strategies=DEFAULT_STRATEGIES, out_dir=None, force=False):
    """Run the population-scaling sweep across mice.

    Writes one CSV per mouse (resumable — an existing per-mouse file is
    reused unless ``force=True``) plus an aggregate CSV and a provenance
    JSON, under ``results/neuron_scaling/`` by default.

    Returns the aggregate long-format ``DataFrame``.
    """
    from utils import load_vr_export
    from run_experiment import run_animal_decoder
    from training.config import default_config_for_target

    mouse_ids = list(mouse_ids)
    cfg = default_config_for_target(
        target, bin_size_ms=bin_size_ms, split_type=split,
        run_name=f'neuron_scaling_{target}')
    config_legacy = cfg.to_legacy_dict()

    out_dir = Path(out_dir) if out_dir is not None else (RESULTS / 'neuron_scaling')
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_provenance(out_dir / f'run_config_{target}_{split}.json', dict(
        target=target, split=split, bin_size_ms=bin_size_ms, step=step,
        n_draws=n_draws, seed=seed, strategies=list(strategies),
        mouse_ids=mouse_ids, config_legacy=config_legacy))

    per_mouse = []
    for mid in mouse_ids:
        mouse_path = out_dir / f'mouse_{mid}_{target}_{split}.csv'
        if mouse_path.exists() and not force:
            print(f"[mouse {mid}] cached -> {mouse_path.name}")
            per_mouse.append(pd.read_csv(mouse_path))
            continue
        print(f"[mouse {mid}] loading export ...")
        preloaded = load_vr_export(mid)
        activities_m, trials = preloaded[0], preloaded[4]
        df = run_scaling_for_mouse(
            mid, config_legacy, activities_m, trials,
            target=target, split=split, decoder_fn=run_animal_decoder,
            preloaded=preloaded, step=step, n_draws=n_draws, seed=seed,
            strategies=strategies, progress=True)
        df.to_csv(mouse_path, index=False)
        print(f"[mouse {mid}] wrote {len(df)} rows -> {mouse_path.name}")
        per_mouse.append(df)

    agg = pd.concat(per_mouse, ignore_index=True)
    agg_path = out_dir / f'neuron_scaling_{target}_{split}.csv'
    agg.to_csv(agg_path, index=False)
    print(f"\nAggregate: {len(agg)} rows -> {agg_path}")
    return agg
