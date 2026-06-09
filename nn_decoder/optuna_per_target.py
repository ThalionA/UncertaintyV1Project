# -*- coding: utf-8 -*-
"""Parametric per-target Optuna sweep.

Replaces the previous two scripts:
  - ``optuna_joint_v27.py``         (Q-only joint spatial+temporal search)
  - ``optuna_phase2_sbc_lambda.py`` (Phase 2 entropy_lambda sweep)

Both retired to ``legacy/``.

Usage
-----
    python optuna_per_target.py --target Q
    python optuna_per_target.py --target choice --n-trials 80
    python optuna_per_target.py --target stim_kernel --bin-sizes-ms 50
    python optuna_per_target.py --target Q --pca-basis condition_mean

One sweep per target. Each writes its study to
``optuna_studies/<target>_<bin>ms_<window>_<pca_basis>.db`` and prints
the winning hyperparameters in a form ready to paste into
``training/config.py::_PRESETS[<target>]``.

The objective is the same as ``optuna_joint_v27``: train both spatial and
temporal under the same trial config, normalise each architecture's
validation loss against a marginal-mean baseline (so the score is
comparable across loss functions), and minimise the mean of the two
normalised scores. This produces hyperparameters that are *jointly
fair* across architectures rather than favouring whichever one happens
to be easier to fit.

Each (mouse, architecture) is trained ``REPS`` times from independent
random initialisations; the restart with the lowest training loss is
kept and its validation loss is the score -- the same best-of-N
selection ``nn_classifier.train_and_select_best_model`` uses in
production. REPS is fixed (not searched), so every config is evaluated
under the production restart regime.

Per-target loss is fixed (not searchable):
    Q, L, stim_kernel  -> PCA-weighted Euclidean
    d                  -> MSE
    choice, stim_cat   -> CE

Search space is currently shared across targets:
    hidden_sizes:    [8, 16, 32, [16,16]]
    learning_rate:   1e-4 .. 1e-2 log
    weight_decay:    1e-5 .. 1e-3 log
    minibatch_size:  [8, 16, 32]
    num_epochs:      [30, 50, 75, 100, 200]
    entropy_lambda:  1e-4 .. 1e-1 log  (matters only for temporal)
Fixed (not searched): REPS = 5 random initialisations per fit;
pca_basis (the PCA loss-basis, default 'all_trials') is a per-run flag.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# --- Path bootstrap so this script works whether invoked from project
# root or from nn_decoder/ ---
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils import load_vr_export, apply_temporal_binning
from nn_classifier import (
    SimpleFlexibleNNClassifier,
    fit_model,
    evaluate,
    JS_calc,
    KL_calc,
    Wasserstein_calc_1D,
)
from training.targets import make_target
from training.config import TARGET_TO_WHICH_MODEL


# =====================================================================
# Device
# =====================================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =====================================================================
# Per-target loss (fixed; not searchable)
# =====================================================================
TARGET_TO_LOSS = {
    'Q':           'PCA',
    'L':           'PCA',
    'd':           'MSE',
    'choice':      'CE',
    'stim_kernel': 'PCA',
    'stim_cat':    'CE',
}


# =====================================================================
# Interactive defaults (Spyder %runcell uses these; CLI flags override)
# Edit these in-place when running from an IDE — that way the script
# is identical whether invoked via `python optuna_per_target.py
# --target=Q` or via `%runcell -i 0`.
# =====================================================================
TARGET       = 'choice'                       # 'Q' | 'L' | 'd' | 'choice' | 'stim_kernel' | 'stim_cat'
N_TRIALS     = 100
MOUSE_IDS    = (0, 1, 2, 3, 4, 5)         # full cohort for the search
TIME_WINDOW  = 'half'                     # 'full' | 'half' | 'last_quarter'
PCA_BASIS    = 'all_trials'               # 'all_trials' | 'condition_mean' | 'residual' (PCA-loss targets only)
BIN_SIZES_MS = [100]                 # iterable: one Optuna study per bin size
STUDY_DIR    = 'optuna_studies'
REPS         = 5                          # random inits per fit; best-by-train-loss kept (fixed, not searched)


# =====================================================================
# Cached per-mouse data
# =====================================================================
_DATA_CACHE = {}


def get_mouse_data(mouse_id, target_type, time_window, bin_size_ms, pca_basis):
    """Load + bin + z-score + 80/20 split + per-target PCA basis.
    Cached by (mouse, target, window, bin, pca_basis)."""
    key = (mouse_id, target_type, time_window, bin_size_ms, pca_basis)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    print(f"  [data] mouse {mouse_id}: loading export "
          f"(target={target_type}, {time_window}, {bin_size_ms}ms)...",
          flush=True)
    activities_m, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mouse_id)

    # Construct the target via the unified make_target helper.
    target = make_target(
        TARGET_TO_WHICH_MODEL[target_type], trials,
        targets_perc=targets_perc, targets_dec=targets_dec, targets_lik=targets_lik,
    )

    # Temporal binning expects (nTrials, tBins, nNeurons)
    act_t = np.transpose(activities_m, (1, 2, 0))
    act_b = apply_temporal_binning(act_t, time_window=time_window, bin_size_ms=bin_size_ms)
    activities_m = np.transpose(act_b, (2, 0, 1))

    # Drop singletons (categories with only one trial would break stratify)
    stim = np.array(list(zip(trials['orientation'], trials['contrast'], trials['dispersion'])))
    _, cats, counts = np.unique(stim, axis=0, return_inverse=True, return_counts=True)
    valid_cat_mask = counts[cats] > 1
    if not np.all(valid_cat_mask):
        activities_m = activities_m[:, valid_cat_mask, :]
        target = target[valid_cat_mask]
        stim = stim[valid_cat_mask]
        _, cats = np.unique(stim, axis=0, return_inverse=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(cats)), test_size=0.2, random_state=42, stratify=cats,
    )

    # Per-neuron z-score using TRAINING trials only
    mu = np.mean(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
    sd = np.std(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
    sd[sd == 0] = 1.0
    activities_z = (activities_m - mu) / sd

    # PCA loss-basis (only used when the loss is PCA; computed regardless
    # for caching). `pca_basis` selects what the PCA is fit on -- kept in
    # sync with run_experiment.run_animal_decoder.
    pcs, var = None, None
    if target.shape[1] > 2:
        train_stim = stim[train_idx]
        train_t = target[train_idx]
        uniq_train = np.unique(train_stim, axis=0)
        if pca_basis == 'all_trials':
            # Raw per-trial training targets: no averaging, no cell-mean
            # subtraction. Captures across- and within-condition variance.
            pca_input = train_t
        elif pca_basis == 'condition_mean':
            # Per-(o,c,d)-cell averaged training targets.
            pca_input = np.zeros((len(uniq_train), train_t.shape[1]))
            for i, s in enumerate(uniq_train):
                m = np.all(train_stim == s, axis=1)
                pca_input[i] = np.mean(train_t[m], axis=0)
        else:  # 'residual'
            # Per-trial (target - within-cell mean); singleton cells get
            # a zero residual row (excluded). Mirrors run_experiment.py.
            cell_mean = np.zeros_like(train_t)
            keep = np.zeros(len(train_t), dtype=bool)
            for s in uniq_train:
                m = np.all(train_stim == s, axis=1)
                if m.sum() >= 2:
                    cell_mean[m] = train_t[m].mean(axis=0)
                    keep[m] = True
            pca_input = train_t - cell_mean
            pca_input[~keep] = 0.0
        pca = PCA().fit(pca_input)
        pcs = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)
        var = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32, device=DEVICE)

    out = dict(
        activities_z=activities_z, targets=target,
        train_idx=train_idx, val_idx=val_idx,
        pcs=pcs, explained_variance=var,
    )
    _DATA_CACHE[key] = out
    return out


# =====================================================================
# Marginal-mean baseline (replaces shuffled control during search)
# =====================================================================
def marginal_baseline_loss(Y_train_flat, Y_val_flat, loss_func, pcs, var):
    p = np.mean(Y_train_flat, axis=0)
    P = np.tile(p[None, :], (Y_val_flat.shape[0], 1))
    P_t = torch.tensor(P, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y_val_flat, dtype=torch.float32, device=DEVICE)

    if loss_func == "PCA":
        if pcs is None:
            raise ValueError(
                "marginal_baseline_loss: loss_func='PCA' requires a PCA "
                "basis, but pcs is None. PCA loss is only defined for "
                "multi-dimensional targets (>2 categories). See "
                "nn_decoder/pca_loss.py."
            )
        # torch twin of pca_loss.pca_distance — keep in sync.
        Pp = P_t @ pcs.T
        Yp = Y_t @ pcs.T
        return float(torch.mean(torch.sum(var * (Pp - Yp) ** 2, dim=-1) * 100).item())
    if loss_func == "JS":
        return float(torch.mean(JS_calc(P_t, Y_t)).item())
    if loss_func == "KL":
        return float(torch.mean(KL_calc(P_t, Y_t)).item())
    if loss_func == "Wasserstein":
        return float(torch.mean(Wasserstein_calc_1D(P_t, Y_t)).item())
    if loss_func == "MSE":
        return float(torch.mean((P_t - Y_t) ** 2).item())
    if loss_func == "CE":
        eps = 1e-12
        Pc = torch.clamp(P_t, eps, 1 - eps)
        return float(-torch.mean(torch.sum(Y_t * torch.log(Pc), dim=-1)).item())
    raise ValueError(f"Unsupported loss_func for baseline: {loss_func}")


# =====================================================================
# Single-mouse joint train (spatial + temporal)
# =====================================================================
def train_one_mouse(mouse_id, target_type, config, loss_func,
                     time_window, bin_size_ms, pca_basis):
    data = get_mouse_data(mouse_id, target_type, time_window, bin_size_ms,
                          pca_basis)
    A = data["activities_z"]            # (n_neurons, n_trials, T)
    Y = data["targets"]                 # (n_trials, n_cats)
    pcs = data["pcs"]
    var = data["explained_variance"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    n_neurons, n_trials, T = A.shape
    n_cats = Y.shape[1]

    # Per-trial tensors in the (n_trials, T, ...) layout fit_model expects.
    # X: activity transposed to trials-first. Y: the per-trial target
    # broadcast across the T bins -- fit_model mean-reduces it over T, so
    # this reproduces the legacy Y_d repeat.
    X_all = np.transpose(A, (1, 2, 0))                  # (n_trials, T, n_neurons)
    Y_all = np.repeat(Y[:, None, :], T, axis=1)         # (n_trials, T, n_cats)

    def _to_dev(arr):
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

    X_tr = _to_dev(X_all[train_idx])
    Y_tr = _to_dev(Y_all[train_idx])
    X_va = _to_dev(X_all[val_idx])
    Y_va = _to_dev(Y_all[val_idx])

    # Flattened (n*T, n_cats) targets only for the marginal baseline, which
    # still consumes the legacy flat layout.
    Y_tr_flat = Y_all[train_idx].reshape(-1, n_cats)
    Y_va_flat = Y_all[val_idx].reshape(-1, n_cats)

    base_arch = dict(
        input_size=n_neurons,
        hidden_sizes=config["hidden_sizes"],
        output_size=n_cats,
        activation='tanh',
    )

    def train_eval(model_type):
        # REPS independent random inits; keep the restart with the lowest
        # training loss (best-of-N, the same selection
        # nn_classifier.train_and_select_best_model uses in production)
        # and report its held-out loss. Seed once -- the REPS models then
        # draw distinct inits, deterministically per trial.
        torch.manual_seed(0)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(0)
        best_train = float("inf")
        best_val = None
        for _ in range(REPS):
            model = SimpleFlexibleNNClassifier(**base_arch).to(DEVICE)
            opt = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"], weight_decay=config["weight_decay"],
            )
            fit_model(
                model, opt, X_tr, Y_tr,
                model_type=model_type, loss_func=loss_func,
                pcs=pcs, explained_variance=var,
                entropy_lambda=config["entropy_lambda"],
                minibatch_size=config["minibatch_size"],
                num_epochs=config["num_epochs"],
            )
            # Restart selection: total loss (fit + penalty) at the real
            # entropy_lambda, summed over train trials -- matches
            # train_and_select_best_model's rep-selection score.
            train_loss = evaluate(
                model, X_tr, Y_tr,
                model_type=model_type, loss_func=loss_func,
                pcs=pcs, explained_variance=var,
                entropy_lambda=config["entropy_lambda"], reduction='sum',
            )
            if train_loss < best_train:
                best_train = train_loss
                # Held-out metric for the winning restart: fit loss at
                # entropy_lambda=0, meaned over the val trials.
                best_val = evaluate(
                    model, X_va, Y_va,
                    model_type=model_type, loss_func=loss_func,
                    pcs=pcs, explained_variance=var,
                    entropy_lambda=0.0, reduction='mean',
                )
        return best_val

    val_ppc = train_eval("ppc")
    val_sbc = train_eval("sampling")

    base = marginal_baseline_loss(Y_tr_flat, Y_va_flat, loss_func, pcs, var) + 1e-9
    return val_ppc / base, val_sbc / base


# =====================================================================
# Optuna objective
# =====================================================================
def make_objective(target_type, mouse_ids, time_window, bin_size_ms, pca_basis):
    loss_func = TARGET_TO_LOSS[target_type]

    def objective(trial):
        hidden_choice = trial.suggest_categorical(
            'hidden_choice', ['8', '16', '32', '16,16'])
        config = {
            'hidden_sizes':   [int(x) for x in hidden_choice.split(',')],
            'learning_rate':  trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay':   trial.suggest_float('weight_decay',  1e-5, 1e-3, log=True),
            'minibatch_size': trial.suggest_categorical('minibatch_size', [8, 16, 32]),
            'num_epochs':     trial.suggest_categorical('num_epochs',     [30, 50, 75, 100, 200]),
            'entropy_lambda': trial.suggest_float('entropy_lambda', 1e-4, 1e-1, log=True),
        }

        # Trial header — picks up the parameter sample so you can see what
        # this trial is exploring before it starts grinding through mice.
        print(f"  [trial {trial.number}] starting: "
              f"hidden={hidden_choice} lr={config['learning_rate']:.2e} "
              f"wd={config['weight_decay']:.2e} mb={config['minibatch_size']} "
              f"epochs={config['num_epochs']} ent_lambda={config['entropy_lambda']:.2e}",
              flush=True)

        ppc_scores, sbc_scores = [], []
        for i, mid in enumerate(mouse_ids):
            print(f"    [trial {trial.number}] mouse {mid} "
                  f"({i+1}/{len(mouse_ids)})...", flush=True)
            try:
                ppc, sbc = train_one_mouse(
                    mid, target_type, config, loss_func, time_window,
                    bin_size_ms, pca_basis,
                )
                print(f"    [trial {trial.number}] mouse {mid} done: "
                      f"spatial={ppc:.4f}  temporal={sbc:.4f}", flush=True)
            except Exception:
                logging.error(f"Trial {trial.number} mouse {mid}:\n{traceback.format_exc()}")
                raise optuna.TrialPruned()

            ppc_scores.append(ppc)
            sbc_scores.append(sbc)
            interim = 0.5 * (np.mean(ppc_scores) + np.mean(sbc_scores))

            if np.isnan(interim):
                raise optuna.TrialPruned()

            trial.report(interim, step=i + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        ppc_mean = float(np.mean(ppc_scores))
        sbc_mean = float(np.mean(sbc_scores))
        trial.set_user_attr('ppc_mean', ppc_mean)
        trial.set_user_attr('sbc_mean', sbc_mean)
        return 0.5 * (ppc_mean + sbc_mean)

    return objective


# =====================================================================
# CLI
# =====================================================================
def run(target, n_trials=60, mouse_ids=(0, 1, 2, 3, 4, 5),
         time_window='half', bin_size_ms=100, pca_basis='all_trials',
         study_dir='optuna_studies'):
    os.makedirs(study_dir, exist_ok=True)
    # pca_basis is part of the study name so studies under different
    # loss-bases never resume into one another (load_if_exists=True).
    study_name = f'{target}_{bin_size_ms}ms_{time_window}_{pca_basis}'
    storage = f'sqlite:///{study_dir}/{study_name}.db'

    print(f"Optuna sweep | target={target} | loss={TARGET_TO_LOSS[target]} "
          f"| {time_window} {bin_size_ms}ms | pca_basis={pca_basis} "
          f"| mice={mouse_ids} | trials={n_trials}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction='minimize',
        sampler=TPESampler(seed=42, multivariate=True, group=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    def _log_cb(study, trial):
        """Per-trial progress — fires after each trial finishes (or is
        pruned). Without this, study.optimize is silent."""
        state = trial.state.name
        msg = f"  [trial {trial.number:>3}/{n_trials}] {state}"
        if trial.state == optuna.trial.TrialState.COMPLETE:
            ppc = trial.user_attrs.get('ppc_mean')
            sbc = trial.user_attrs.get('sbc_mean')
            msg += f"  value={trial.value:.4f}"
            if ppc is not None and sbc is not None:
                msg += f"  (spatial={ppc:.4f}, temporal={sbc:.4f})"
            best = study.best_value
            best_n = study.best_trial.number
            msg += f"  best={best:.4f} (trial {best_n})"
        print(msg, flush=True)

    obj = make_objective(target, list(mouse_ids), time_window, bin_size_ms,
                         pca_basis)
    study.optimize(obj, n_trials=n_trials, n_jobs=1, callbacks=[_log_cb])

    print('\n' + '=' * 60)
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best hyperparameters for target = {target}:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    ppc = study.best_trial.user_attrs.get('ppc_mean')
    sbc = study.best_trial.user_attrs.get('sbc_mean')
    if ppc is not None and sbc is not None:
        print(f"  per-arch: spatial={ppc:.4f}  temporal={sbc:.4f}")

    # Print as a paste-ready Python snippet for _PRESETS
    print('\nPaste into training/config.py::_PRESETS[%r]:' % target)
    p = study.best_params
    hidden = [int(x) for x in p['hidden_choice'].split(',')]
    print('    %r: dict(' % target)
    print(f"        loss_func={TARGET_TO_LOSS[target]!r},")
    print(f"        hidden_sizes={hidden},")
    print(f"        learning_rate={p['learning_rate']:.4g},")
    print(f"        weight_decay={p['weight_decay']:.4g},")
    print(f"        minibatch_size={p['minibatch_size']},")
    print(f"        num_epochs={p['num_epochs']},")
    print(f"        entropy_lambda={p['entropy_lambda']:.4g},")
    print('    ),')

    study.trials_dataframe().to_csv(f'{study_dir}/{study_name}_results.csv')


def _running_interactively() -> bool:
    """True when invoked from IPython/Spyder %runcell or similar — no CLI
    args are passed in those cases, so we should fall back to the
    module-level constants instead of failing argparse's required check."""
    # Spyder / IPython sets sys.argv[0] to the script name and may pass
    # nothing else. Treat 'no extra args' as interactive.
    return len(sys.argv) <= 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    # --target defaults to module-level TARGET so %runcell / direct
    # interactive invocation works. Override with --target=<...> from CLI.
    p.add_argument('--target', default=TARGET, choices=list(TARGET_TO_LOSS),
                   help='Target type to optimise (default: module-level TARGET)')
    p.add_argument('--n-trials', type=int, default=N_TRIALS)
    p.add_argument('--mouse-ids', nargs='+', type=int, default=list(MOUSE_IDS),
                   help='Subset of mice for the search (validation on full cohort '
                        'happens during the production sweep)')
    p.add_argument('--time-window', default=TIME_WINDOW,
                   choices=['full', 'half', 'last_quarter'])
    p.add_argument('--bin-sizes-ms', nargs='+', type=int,
                   default=list(BIN_SIZES_MS), choices=[50, 100, 250],
                   help='One Optuna study per bin size (separate .db each).')
    p.add_argument('--pca-basis', default=PCA_BASIS,
                   choices=['all_trials', 'condition_mean', 'residual'],
                   help='What the PCA loss-basis is fit on (PCA-loss '
                        'targets only; default: module-level PCA_BASIS).')
    p.add_argument('--study-dir', default=STUDY_DIR)
    # parse_known_args lets IPython / Spyder pass through any spurious
    # cell-runner flags without crashing argparse.
    args, _unknown = p.parse_known_args()

    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler('optuna_per_target.log')])
    # Default WARNING was too quiet — interactive runs should see
    # optuna's own per-trial progress lines too.
    optuna.logging.set_verbosity(optuna.logging.INFO)

    if _running_interactively():
        print(f"[optuna_per_target] interactive mode — using module-level "
              f"defaults: TARGET={TARGET!r}, N_TRIALS={N_TRIALS}, "
              f"BIN_SIZES_MS={tuple(args.bin_sizes_ms)}, "
              f"PCA_BASIS={args.pca_basis!r}. Edit the constants "
              f"at the top of this script to change them.")

    # One sweep per bin size. Each writes its own SQLite study.
    for bs in args.bin_sizes_ms:
        print('\n' + '#' * 60)
        print(f"#  STARTING SWEEP: target={args.target}  bin_size_ms={bs}")
        print('#' * 60)
        run(target=args.target, n_trials=args.n_trials,
            mouse_ids=tuple(args.mouse_ids),
            time_window=args.time_window, bin_size_ms=bs,
            pca_basis=args.pca_basis, study_dir=args.study_dir)


if __name__ == '__main__':
    main()
