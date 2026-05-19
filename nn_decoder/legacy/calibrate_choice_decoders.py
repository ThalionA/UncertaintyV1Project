"""Temperature-scaling calibration for choice-shaped decoder outputs.

The 'true_choice' decoder produces high AUROC (better Go/NoGo
discrimination than IO baseline on stratified_balanced: 0.910 vs 0.876)
but high NLL (0.716 vs 0.390 for stim_mean). That's a textbook
over-confidence pattern — the decoder ranks trials correctly but its
absolute probabilities are too sharp.

Temperature scaling (Guo et al. 2017) fits one scalar T per
(mouse, arch) on calibration data:
    p_cal = sigmoid(logit(p_raw) / T)

Calibration set: training trials extracted from `full_decoded[train_idx]`,
where train_idx is the same deterministic split (random_state=42) used by
the production decoder. Evaluation set: test trials.

This is a proper holdout — the temperature is fit on data the decoder
saw during training, evaluated on data it didn't. AUROC is invariant
under monotonic transforms (so should be unchanged before/after).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_coherence as ioc  # noqa: E402


def _binary_nll(p_pred, actual, eps=1e-12):
    p = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))


def fit_temperature(p_pred_calib, actual_calib):
    """Fit T > 0 minimising binary NLL on calibration data."""
    eps = 1e-12
    p_safe = np.clip(p_pred_calib, eps, 1 - eps)
    logits = np.log(p_safe / (1 - p_safe))

    def neg_log_lik(log_T):
        T = float(np.exp(log_T))
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        p_cal = np.clip(p_cal, eps, 1 - eps)
        return -float(np.mean(actual_calib * np.log(p_cal)
                              + (1 - actual_calib) * np.log(1 - p_cal)))

    # bracket log T around 0 (T=1, no scaling)
    res = minimize_scalar(neg_log_lik, bracket=(-5, 0, 5))
    return float(np.exp(res.x))


def apply_temperature(p_pred, T, eps=1e-12):
    p_safe = np.clip(p_pred, eps, 1 - eps)
    logits = np.log(p_safe / (1 - p_safe))
    return 1.0 / (1.0 + np.exp(-logits / T))


def _train_test_indices(mouse_id, split_type, random_state=42):
    """Replicate the production split exactly."""
    from utils import (
        load_vr_export,
        get_stratified_train_test_indices,
        get_generalization_split_indices,
    )
    _, _, _, _, trials = load_vr_export(mouse_id)
    conditions = np.column_stack([
        np.asarray(trials['orientation'], dtype=float),
        np.asarray(trials['contrast'], dtype=float),
        np.asarray(trials['dispersion'], dtype=float),
    ])
    if split_type == 'stratified_balanced':
        _, cats = np.unique(conditions, axis=0, return_inverse=True)
        train_idx, test_idx = get_stratified_train_test_indices(
            cats, test_size=0.5, random_state=random_state)
    else:
        train_idx, test_idx = get_generalization_split_indices(
            trials, split_type=split_type, random_state=random_state)
    return train_idx, test_idx, trials


def calibrate_decoder(full_decoded_pgo, actual_choice_full,
                       train_idx, test_idx):
    """Temperature-scale a per-trial P(Go) prediction.

    Returns dict with uncal/cal NLL, AUROC, fitted T."""
    p_train = full_decoded_pgo[train_idx]
    a_train = actual_choice_full[train_idx]
    p_test = full_decoded_pgo[test_idx]
    a_test = actual_choice_full[test_idx]

    # Fit T on train predictions vs train choices.
    T = fit_temperature(p_train, a_train)

    # Evaluate on test set.
    p_test_cal = apply_temperature(p_test, T)
    nll_uncal = _binary_nll(p_test, a_test)
    nll_cal = _binary_nll(p_test_cal, a_test)
    _, auroc = ioc.compute_choice_metrics(p_test, a_test)
    return dict(T=T, nll_uncal=nll_uncal, nll_cal=nll_cal, auroc=auroc,
                n_train=len(train_idx), n_test=len(test_idx))


def evaluate_split(split, animals, directory='.'):
    """Calibrate every choice-shaped decoder on the given split."""
    rows = []

    files = {
        'direct_choice':  os.path.join(directory, f'population_results_fixed_choice_{split}.mat'),
        'true_choice':    os.path.join(directory, f'population_results_fixed_truechoice_{split}.mat'),
        'stim_mean_choice': os.path.join(directory, f'population_results_stim_mean_choice_{split}.mat'),
    }
    mats = {k: sio.loadmat(p, simplify_cells=True) if os.path.exists(p) else None
            for k, p in files.items()}

    for mid in range(len(animals)):
        a = animals[mid]
        mask = ioc.get_trials_to_keep_mask(a['data']['trial_keys']['session_name'])
        actual_full = np.asarray(a['data']['choices'])[mask].astype(float)
        train_idx, test_idx, _ = _train_test_indices(mid, split)

        # IO baseline (no temperature scaling needed; already calibrated
        # by the IO Stage 2 lapse fit). Reported here for reference.
        cv = np.asarray(a['pred']['cv_p_choice_stage2'])[mask].astype(float)
        nll_io = _binary_nll(cv[test_idx], actual_full[test_idx])
        _, auc_io = ioc.compute_choice_metrics(cv[test_idx], actual_full[test_idx])
        rows.append(dict(mouse=mid, split=split, method='IO_baseline',
                         arch='-', T=np.nan,
                         nll_uncal=nll_io, nll_cal=nll_io, auroc=auc_io))

        # stim_mean_choice: only one 'arch' key
        if mats['stim_mean_choice'] is not None:
            d = mats['stim_mean_choice']['results'][f'mouse_{mid}']['Dist']['stim_mean']
            full_pgo = np.asarray(d['full_decoded'])[:, 0]
            res = calibrate_decoder(full_pgo, actual_full, train_idx, test_idx)
            rows.append(dict(mouse=mid, split=split,
                             method='stim_mean_choice', arch='-',
                             **res))

        # direct_choice / true_choice: spat and temp arches
        for method_key in ('direct_choice', 'true_choice'):
            mat = mats[method_key]
            if mat is None: continue
            for arch in ('spat', 'temp'):
                try:
                    d = mat['results'][f'mouse_{mid}']['Dist'][arch]
                except KeyError:
                    continue
                full_pgo = np.asarray(d['full_decoded'])[:, 0]
                res = calibrate_decoder(full_pgo, actual_full, train_idx, test_idx)
                rows.append(dict(mouse=mid, split=split, method=method_key,
                                 arch=arch, **res))
    return rows


def main(io_path='data/IOResults.mat', directory='nn_decoder',
         splits=('stratified_balanced',),
         out_csv='choice_calibration.csv'):
    animals = ioc.load_io_results(io_path)
    all_rows = []
    for split in splits:
        all_rows.extend(evaluate_split(split, animals, directory=directory))
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(directory, out_csv), index=False)

    print('\nPer-mouse calibration (split = {}):'.format(splits[0]))
    print(df[df['split'] == splits[0]].round(3).to_string(index=False))

    print('\nMean across mice — uncalibrated vs calibrated NLL, AUROC:')
    agg = (df.groupby(['split', 'method', 'arch'])
             [['nll_uncal', 'nll_cal', 'auroc', 'T']]
             .mean()
             .round(3))
    print(agg.to_string())
    return df


if __name__ == '__main__':
    main()
