# -*- coding: utf-8 -*-
"""Stimulus-only baseline: a closed-form 'decoder' that takes a trial's
stimulus condition (orientation, contrast, dispersion) and outputs the
per-condition mean of the training-set targets.

Anchors what V1-based decoders are actually adding beyond
'extrapolate from stimulus condition'. Any decoder that only matches
this baseline is doing nothing more than predicting the per-condition
average; one that beats it is using V1's trial-by-trial fluctuations.

Output layout mirrors run_animal_decoder so existing analysis modules
can pick the file up as a 5th architecture by reading a new arch key
``'stim_mean'`` under ``Dist``.
"""

from __future__ import annotations

import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

# utils_v26 imports torch at module top-level (it's also where the neural
# decoder helpers live). The prediction primitives in this module do not
# need torch, so we defer the utils_v26 import to the only function that
# actually uses it (compute_stim_mean_for_animal). Lets unit tests for the
# math run without torch installed.


# ----------------------------------------------------------------------
# Closed-form predictor primitives
# ----------------------------------------------------------------------

def compute_condition_means(conditions, targets):
    """Group rows of ``targets`` by their (full) stimulus condition tuple
    and return the per-condition mean.

    Parameters
    ----------
    conditions : ndarray, shape (n_trials, n_features)
        Stimulus condition vectors, e.g. (orientation, contrast, dispersion).
    targets : ndarray, shape (n_trials, n_categories)
        Target distributions or scalars.

    Returns
    -------
    dict mapping tuple(condition) -> ndarray of shape (n_categories,)
    """
    conditions = np.asarray(conditions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    out = {}
    if len(conditions) == 0:
        return out
    for cond in np.unique(conditions, axis=0):
        mask = np.all(conditions == cond, axis=1)
        out[tuple(map(float, cond))] = targets[mask].mean(axis=0)
    return out


def compute_orientation_means(conditions, targets):
    """Per-orientation mean of ``targets``, ignoring contrast and dispersion.

    Used as a fallback in OOD splits where many test (o,c,d) triplets are
    not present in training but the orientation alone is.
    """
    conditions = np.asarray(conditions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    out = {}
    if len(conditions) == 0:
        return out
    orientations = conditions[:, 0]
    for ori in np.unique(orientations):
        mask = orientations == ori
        out[float(ori)] = targets[mask].mean(axis=0)
    return out


def predict_stim_mean(test_conditions, condition_means, fallback):
    """For each row of ``test_conditions`` return the matching condition
    mean, or ``fallback`` for unseen conditions."""
    test_conditions = np.asarray(test_conditions, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    n_cats = fallback.shape[0]
    out = np.zeros((len(test_conditions), n_cats))
    for i, cond in enumerate(test_conditions):
        key = tuple(map(float, cond))
        out[i] = condition_means.get(key, fallback)
    return out


def predict_stim_mean_with_orientation_fallback(
    test_conditions, condition_means, orientation_means, global_mean,
):
    """Prediction with three-tier fallback for OOD splits:

      1. Full (orientation, contrast, dispersion) condition seen in training.
      2. Otherwise, orientation alone seen in training.
      3. Otherwise, global training-set mean.
    """
    test_conditions = np.asarray(test_conditions, dtype=float)
    global_mean = np.asarray(global_mean, dtype=float)
    n_cats = global_mean.shape[0]
    out = np.zeros((len(test_conditions), n_cats))
    for i, cond in enumerate(test_conditions):
        key = tuple(map(float, cond))
        if key in condition_means:
            out[i] = condition_means[key]
            continue
        ori = float(cond[0])
        if ori in orientation_means:
            out[i] = orientation_means[ori]
            continue
        out[i] = global_mean
    return out


# ----------------------------------------------------------------------
# Per-animal driver
# ----------------------------------------------------------------------

def _select_target(target_type, targets_perc, targets_dec, targets_lik,
                    trials=None):
    if target_type == 'Q':
        return targets_perc
    if target_type == 'L':
        if targets_lik is None:
            raise ValueError(
                "L_s_marginal not present in VR_Decoder_Data_Export.mat; "
                "re-run the IO export to add the marginalised likelihood."
            )
        return targets_lik
    if target_type == 'd':
        return targets_dec
    if target_type == 'choice':
        if trials is None or 'choice' not in trials:
            raise ValueError(
                "target_type='choice' requires the trials dict with a "
                "'choice' field (animal's binary goChoice)."
            )
        c = np.asarray(trials['choice']).astype(float).reshape(-1)
        return np.column_stack([c, 1.0 - c])
    raise ValueError(f"Unknown target_type {target_type!r}")


def compute_stim_mean_for_animal(
    mouse_id, target_type, split_type, random_state=42,
):
    """Compute the stim-mean baseline for one (animal, target, split) combo.

    Returns a dict matching the production architecture sub-dict layout:
    ``{'decoded': ..., 'target': ..., 'full_decoded': ..., 'pcs': ...,
       'explained_var': ...}``. The PCA basis is fit on per-condition-
    averaged training targets, exactly as run_animal_decoder does, so that
    downstream PCA-weighted analyses see a comparable basis.
    """
    # Deferred import — utils_v26 pulls torch at module top, which the
    # prediction primitives above do not need.
    from utils_v26 import (
        load_vr_export,
        get_stratified_train_test_indices,
        get_generalization_split_indices,
    )

    activities_m, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mouse_id)
    targets = _select_target(target_type, targets_perc, targets_dec, targets_lik,
                              trials=trials)

    conditions = np.column_stack([
        np.asarray(trials['orientation'], dtype=float),
        np.asarray(trials['contrast'], dtype=float),
        np.asarray(trials['dispersion'], dtype=float),
    ])

    # Replicate the production split (same random_state).
    if split_type == 'stratified_balanced':
        _, trial_categories_all = np.unique(conditions, axis=0, return_inverse=True)
        train_idx, test_idx = get_stratified_train_test_indices(
            trial_categories_all, test_size=0.5, random_state=random_state,
        )
    elif split_type in ('generalize_contrast', 'generalize_dispersion'):
        train_idx, test_idx = get_generalization_split_indices(
            trials, split_type=split_type, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown split_type {split_type!r}")

    train_conditions = conditions[train_idx]
    train_targets = targets[train_idx]
    test_conditions = conditions[test_idx]
    test_targets = targets[test_idx]

    cond_means = compute_condition_means(train_conditions, train_targets)
    orientation_means = compute_orientation_means(train_conditions, train_targets)
    global_mean = train_targets.mean(axis=0)

    decoded_test = predict_stim_mean_with_orientation_fallback(
        test_conditions, cond_means, orientation_means, global_mean,
    )
    full_decoded = predict_stim_mean_with_orientation_fallback(
        conditions, cond_means, orientation_means, global_mean,
    )

    # PCA basis from per-condition averaged training targets, mirroring
    # run_animal_decoder. Skipped (empty arrays) for the 2-D decision
    # target where PCA is undefined.
    n_cats = targets.shape[1]
    if n_cats > 2:
        unique_train_cats = np.unique(train_conditions, axis=0)
        avg_dist = np.zeros((len(unique_train_cats), n_cats))
        for i, c in enumerate(unique_train_cats):
            mask = np.all(train_conditions == c, axis=1)
            avg_dist[i] = train_targets[mask].mean(axis=0)
        pca = PCA().fit(avg_dist)
        pcs = pca.components_
        evar = pca.explained_variance_ratio_
    else:
        pcs = np.array([])
        evar = np.array([])

    return dict(
        decoded=decoded_test,
        target=test_targets,
        full_decoded=full_decoded,
        pcs=pcs,
        explained_var=evar,
    )


# ----------------------------------------------------------------------
# Output naming + driver
# ----------------------------------------------------------------------

# File-name pattern parallels the production prefixes.
TARGET_PREFIXES = {
    'Q':      'population_results_stim_mean_Q',
    'L':      'population_results_stim_mean_L',
    'd':      'population_results_stim_mean_d',
    'choice': 'population_results_stim_mean_choice',
}


def _compute_stim_mean_from_loaded(
    targets, trials, target_type, split_type, random_state=42,
):
    """Same as compute_stim_mean_for_animal but takes pre-loaded
    (targets, trials) for the animal — avoids re-reading the 500 MB+
    VR_Decoder_Data_Export.mat once per (target, split) combination.
    """
    from utils_v26 import (
        get_stratified_train_test_indices,
        get_generalization_split_indices,
    )

    conditions = np.column_stack([
        np.asarray(trials['orientation'], dtype=float),
        np.asarray(trials['contrast'], dtype=float),
        np.asarray(trials['dispersion'], dtype=float),
    ])

    if split_type == 'stratified_balanced':
        _, trial_categories_all = np.unique(conditions, axis=0, return_inverse=True)
        train_idx, test_idx = get_stratified_train_test_indices(
            trial_categories_all, test_size=0.5, random_state=random_state,
        )
    elif split_type in ('generalize_contrast', 'generalize_dispersion'):
        train_idx, test_idx = get_generalization_split_indices(
            trials, split_type=split_type, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown split_type {split_type!r}")

    train_conditions = conditions[train_idx]
    train_targets = targets[train_idx]
    test_conditions = conditions[test_idx]
    test_targets = targets[test_idx]

    cond_means = compute_condition_means(train_conditions, train_targets)
    orientation_means = compute_orientation_means(train_conditions, train_targets)
    global_mean = train_targets.mean(axis=0)

    decoded_test = predict_stim_mean_with_orientation_fallback(
        test_conditions, cond_means, orientation_means, global_mean,
    )
    full_decoded = predict_stim_mean_with_orientation_fallback(
        conditions, cond_means, orientation_means, global_mean,
    )

    n_cats = targets.shape[1]
    if n_cats > 2:
        unique_train_cats = np.unique(train_conditions, axis=0)
        avg_dist = np.zeros((len(unique_train_cats), n_cats))
        for i, c in enumerate(unique_train_cats):
            mask = np.all(train_conditions == c, axis=1)
            avg_dist[i] = train_targets[mask].mean(axis=0)
        pca = PCA().fit(avg_dist)
        pcs = pca.components_
        evar = pca.explained_variance_ratio_
    else:
        pcs = np.array([])
        evar = np.array([])

    return dict(
        decoded=decoded_test,
        target=test_targets,
        full_decoded=full_decoded,
        pcs=pcs,
        explained_var=evar,
    )


def run_all_stim_mean_baselines(
    splits=('stratified_balanced', 'generalize_contrast', 'generalize_dispersion'),
    target_types=('Q', 'L', 'd', 'choice'),
    mouse_ids=range(6),
    output_dir=None,
):
    """Driver: iterate over (mouse, target, split) and write one
    ``population_results_stim_mean_{target}_{split}.mat`` per (target, split)
    combo. Each file mirrors the production layout.

    Loads VR_Decoder_Data_Export.mat ONCE per animal (the file is ~500 MB)
    and reuses the loaded targets/trials for every (target, split) combo
    of that animal — otherwise we'd re-read the same 500 MB file 9 times
    per animal.
    """
    from utils_v26 import load_vr_export

    here = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = here

    # Per-animal cache: load the export once.
    print('[stim_mean] caching per-animal exports (one read of '
          'VR_Decoder_Data_Export.mat per mouse)...')
    animal_cache = {}
    for mid in mouse_ids:
        try:
            _, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mid)
            # One-hot encode the animal's actual goChoice as a 2-bin
            # [P(Go), P(NoGo)] target. Used by target_type='choice'.
            c = np.asarray(trials['choice']).astype(float).reshape(-1)
            choice_target = np.column_stack([c, 1.0 - c])
            animal_cache[mid] = dict(
                Q=targets_perc, L=targets_lik, d=targets_dec,
                choice=choice_target, trials=trials,
            )
            print(f'  mouse {mid}: '
                  f'Q={None if targets_perc is None else targets_perc.shape}  '
                  f'L={None if targets_lik is None else targets_lik.shape}  '
                  f'd={None if targets_dec is None else targets_dec.shape}')
        except (FileNotFoundError, KeyError) as exc:
            print(f'  mouse {mid}: load failed: {exc}')

    written = []
    for split in splits:
        for target_type in target_types:
            results = {}
            skipped = []
            for mid in mouse_ids:
                if mid not in animal_cache:
                    skipped.append((mid, 'export load failed'))
                    continue
                tgt = animal_cache[mid][target_type]
                if tgt is None:
                    skipped.append((mid, f'{target_type} target absent'))
                    continue
                try:
                    sub = _compute_stim_mean_from_loaded(
                        tgt, animal_cache[mid]['trials'],
                        target_type, split,
                    )
                except (ValueError, KeyError) as exc:
                    skipped.append((mid, str(exc)))
                    continue
                results[f'mouse_{mid}'] = {
                    'Dist': {
                        'stim_mean': {
                            'decoded': sub['decoded'],
                            'target': sub['target'],
                            'full_decoded': sub['full_decoded'],
                        },
                        'pcs': sub['pcs'],
                        'explained_var': sub['explained_var'],
                    },
                }

            if not results:
                print(f'[stim_mean] {target_type}/{split}: no animals produced; skipping write.')
                continue

            cfg = {
                'baseline': 'stim_mean',
                'target_type': target_type,
                'split_type': split,
                'random_state': 42,
                'fallback_strategy': 'orientation_then_global',
            }
            out_path = os.path.join(output_dir, f'{TARGET_PREFIXES[target_type]}_{split}.mat')
            sio.savemat(out_path, {'results': results, 'config': cfg})
            written.append(out_path)
            msg = f'[stim_mean] wrote {out_path}  (n_animals={len(results)})'
            if skipped:
                msg += f'  skipped: {skipped}'
            print(msg)
    return written


if __name__ == '__main__':
    run_all_stim_mean_baselines()
