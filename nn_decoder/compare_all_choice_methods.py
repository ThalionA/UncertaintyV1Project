"""End-to-end comparison of every method that produces a Go/No-Go
prediction for the animal's choice.

Pulls together the four analyses we have on disk:

  - PCA-Euclidean trained Q decoder (spatial + temporal):
      population_results_fixed_hyperparams_<split>.mat
  - Wasserstein-trained Q decoder (spatial + temporal), the loss-sweep probe:
      population_results_fixed_hyperparams_Wasserstein_<split>.mat
  - True choice decoder, trained CE against animal goChoice (when
    available):
      population_results_fixed_truechoice_<split>.mat
    Note: the 'direct_choice' decoder
    (population_results_fixed_choice_<split>.mat) is intentionally NOT
    loaded — it targets the IO's decision_posterior rather than the
    animal's goChoice, so its best-case is matching the IO baseline at
    predicting choice. We keep its files on disk for record but exclude
    them from the comparison.
  - Stim-mean baselines (Q and choice variants):
      population_results_stim_mean_Q_<split>.mat
      population_results_stim_mean_choice_<split>.mat
  - IO Stage 2 baseline (cv_p_choice_stage2 from IOResults.mat).

Each (mouse, method, split) gets:
  - choice NLL via Stage 2 lapse psychometric for the Q-shaped methods
    (Q decoded posterior is run through g(Q) -> sigmoid)
  - choice NLL directly for the choice-shaped methods (direct_choice,
    truechoice, stim_mean_choice — outputs are already P(Go))
  - choice AUROC under the same prediction
  - Mean / variance error from decomposition_analysis for Q-shaped
    methods only (decomposition is over a 91-bin distribution).

Missing files are skipped with a warning so the script remains useful
while a training run is still in progress.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_coherence as ioc  # noqa: E402
import decomposition_analysis as da  # noqa: E402
import paths  # noqa: E402

GRID = np.arange(0, 91, 1, dtype=float)

# Default file locations resolved relative to this script, so the entry
# points work the same whether the script is invoked from the project
# root, from the nn_decoder/ directory, or via %runcell in IPython.
DEFAULT_IO_PATH = os.path.normpath(os.path.join(HERE, '..', 'data', 'IOResults.mat'))
DEFAULT_DIRECTORY = HERE
DEFAULT_OUT_DIR = os.path.join(HERE, 'figures', 'choice_method_comparison')

# Loss-sweep variants of the Q decoder. Each entry maps a label suffix
# used in the comparison output to the file-name tag in the
# population_results_fixed_hyperparams_<tag>_<split>.mat naming.
LOSS_VARIANTS = (
    # (label_suffix, file_tag)
    ('W',  'Wasserstein'),
    ('KL', 'KL'),
    ('JS', 'JS'),
)


def _safe_load_mat(path):
    if not os.path.exists(path):
        return None
    return sio.loadmat(path, simplify_cells=True)


def _per_mouse_metric(decoded_pgo, actual_choice):
    return ioc.compute_choice_metrics(decoded_pgo, actual_choice)


def _moment_decomp(decoded_q, target_q, grid=GRID):
    """Mean and variance errors for a Q-shaped (n, 91) prediction."""
    res = da.decompose_distributional(decoded_q, target_q, grid)
    return float(np.nanmean(res['mean_error'])), float(np.nanmean(res['var_error']))


def evaluate_split(split, animals, directory='.'):
    """Return a list of per-(mouse, method) dicts for one split."""
    rows = []

    # Compose under the caller's `directory` override; pull canonical
    # stems from paths.FIT_BASENAMES so the names live in exactly one place.
    def _path(stem):
        return os.path.join(directory, f'{stem}_{split}.mat')

    pca_q = _safe_load_mat(_path(paths.fit_stem('fixed', 'Q')))
    # All loss-sweep variants (Wasserstein, KL, JS, ...) detected on disk.
    loss_variant_mats = {}
    for label, tag in LOSS_VARIANTS:
        path = _path(paths.fit_stem('loss_sweep', tag))
        m = _safe_load_mat(path)
        if m is not None:
            loss_variant_mats[label] = (m, tag)
    sm_q        = _safe_load_mat(_path(paths.fit_stem('stim_mean', 'Q')))
    sm_choice   = _safe_load_mat(_path(paths.fit_stem('stim_mean', 'choice')))
    true_choice = _safe_load_mat(_path(paths.fit_stem('fixed', 'truechoice')))

    for mid in range(len(animals)):
        a = animals[mid]
        mask = ioc.get_trials_to_keep_mask(a['data']['trial_keys']['session_name'])
        actual_choice = np.asarray(a['data']['choices'])[mask].astype(float)

        c = a['fit']['choice']
        alpha_r, beta_r = float(c['alpha_r']), float(c['beta_r'])
        gamma_r, delta_r = float(c['gamma_r']), float(c['delta_r'])

        # Helper: turn a Q-shaped (n,91) array into a lapse-corrected P(Go)
        def _lapse_p_go_from_q(Q):
            g = ioc.compute_log_posterior_odds(Q, GRID)
            return ioc.predict_choice_lapse(g, alpha_r, beta_r, gamma_r, delta_r)

        # 1) IO Stage 2 baseline (always)
        cv = np.asarray(a['pred']['cv_p_choice_stage2'])[mask].astype(float)
        nll, auc = _per_mouse_metric(cv, actual_choice)
        rows.append(dict(mouse=mid, split=split, method='IO_baseline',
                         loss='-', shape='choice',
                         nll=nll, auroc=auc, mean_err=np.nan, var_err=np.nan))

        # Helpers to extract a per-mouse Dist sub-dict robustly
        def _dist(mat, arch):
            if mat is None:
                return None
            try:
                return mat['results'][f'mouse_{mid}']['Dist'][arch]
            except KeyError:
                return None

        # 2) PCA-Euclidean Q decoder, lapse-corrected
        for arch, label in (('spat', 'PPC_PCA'), ('temp', 'SBC_PCA')):
            d = _dist(pca_q, arch)
            if d is None: continue
            Q_full = np.asarray(d['full_decoded'])
            target_full = np.asarray(d['target'])  # test-set targets only
            decoded_test = np.asarray(d['decoded'])
            p_go = _lapse_p_go_from_q(Q_full)
            nll, auc = _per_mouse_metric(p_go, actual_choice)
            mean_err, var_err = _moment_decomp(decoded_test, target_full)
            rows.append(dict(mouse=mid, split=split, method=label,
                             loss='PCA-Euclid', shape='Q',
                             nll=nll, auroc=auc,
                             mean_err=mean_err, var_err=var_err))

        # 3) All loss-sweep variants of the Q decoder (Wasserstein, KL,
        #    JS, ...). Picked up from disk based on the file-name tag.
        for variant_label, (mat, file_tag) in loss_variant_mats.items():
            for arch, arch_label in (('spat', 'PPC'), ('temp', 'SBC')):
                d = _dist(mat, arch)
                if d is None: continue
                method_label = f'{arch_label}_{variant_label}'
                Q_full = np.asarray(d['full_decoded'])
                decoded_test = np.asarray(d['decoded'])
                target_test = np.asarray(d['target'])
                p_go = _lapse_p_go_from_q(Q_full)
                nll, auc = _per_mouse_metric(p_go, actual_choice)
                mean_err, var_err = _moment_decomp(decoded_test, target_test)
                rows.append(dict(mouse=mid, split=split, method=method_label,
                                 loss=file_tag, shape='Q',
                                 nll=nll, auroc=auc,
                                 mean_err=mean_err, var_err=var_err))

        # 4) Stim-mean of Q -> lapse correction
        d_sm = _dist(sm_q, 'stim_mean')
        if d_sm is not None:
            Q_full = np.asarray(d_sm['full_decoded'])
            decoded_test = np.asarray(d_sm['decoded'])
            target_test = np.asarray(d_sm['target'])
            p_go = _lapse_p_go_from_q(Q_full)
            nll, auc = _per_mouse_metric(p_go, actual_choice)
            mean_err, var_err = _moment_decomp(decoded_test, target_test)
            rows.append(dict(mouse=mid, split=split, method='stim_mean_Q',
                             loss='-', shape='Q',
                             nll=nll, auroc=auc,
                             mean_err=mean_err, var_err=var_err))

        # 5) Stim-mean of choices (direct calibrated P(Go))
        d_smc = _dist(sm_choice, 'stim_mean')
        if d_smc is not None:
            full = np.asarray(d_smc['full_decoded'])
            nll, auc = _per_mouse_metric(full[:, 0], actual_choice)
            rows.append(dict(mouse=mid, split=split, method='stim_mean_choice',
                             loss='-', shape='choice',
                             nll=nll, auroc=auc, mean_err=np.nan, var_err=np.nan))

        # 6) True choice decoder (CE on animal goChoice). Output is
        # already P(Go); no further lapse correction.
        for arch, label in (('spat', 'true_choice_PPC'),
                             ('temp', 'true_choice_SBC')):
            d = _dist(true_choice, arch)
            if d is None: continue
            full = np.asarray(d['full_decoded'])
            nll, auc = _per_mouse_metric(full[:, 0], actual_choice)
            rows.append(dict(mouse=mid, split=split, method=label,
                             loss='CE', shape='choice',
                             nll=nll, auroc=auc, mean_err=np.nan, var_err=np.nan))

    return rows


def main(io_path=None, directory=None, out_dir=None,
         splits=('stratified_balanced', 'generalize_contrast', 'generalize_dispersion'),
         out_csv='choice_method_comparison.csv'):
    if io_path is None:
        io_path = DEFAULT_IO_PATH
    if directory is None:
        directory = DEFAULT_DIRECTORY
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    animals = ioc.load_io_results(io_path)
    all_rows = []
    for split in splits:
        all_rows.extend(evaluate_split(split, animals, directory=directory))
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, out_csv), index=False)

    # Mean across mice per (split, method)
    agg = (df.groupby(['split', 'method'])
             [['nll', 'auroc', 'mean_err', 'var_err']]
             .mean()
             .round(3))
    print('\nMean across 6 mice — choice prediction NLL (lower = better) and AUROC:')
    print(agg.to_string())
    return df


if __name__ == '__main__':
    main()
