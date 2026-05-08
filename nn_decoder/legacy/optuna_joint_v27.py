#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint PPC + SBC hyperparameter search (v27)
============================================

Replaces the per-architecture studies in optuna_universal_cv_v26.py with a
single fair joint search.

Key changes vs v26:

  * One trial fits BOTH PPC and SBC under the same config and returns a
    joint objective (sum / max / Pareto), so the headline architecture
    comparison is over a fairly-tuned shared hyperparameter set.
  * Uses the production PCA-weighted Euclidean loss (v26 silently dropped
    to cross-entropy because it passed pcs=None into custom_loss_all_H).
  * Tightened search space anchored on the production fixed hyperparameters.
    Activation, optimizer, time window, and bin size are LOCKED IN —
    sweep them separately if you ever want to revisit them.
  * Production config is enqueued as the first trial so TPE anchors near
    the known-good region instead of cold-starting.
  * Per-mouse data is loaded, temporally binned, z-scored, and PCA-fitted
    ONCE and cached across trials.
  * Replaces shuffled-baseline training with a hyperparameter-invariant
    marginal baseline (predict the training-set mean target). Same flavor
    of cross-mouse normalisation, ~2x faster per trial. Train the shuffled
    control only for the final winner via run_fixed_hyperparams.py.
  * MedianPruner replaces Hyperband to allow early TPE exploration without
    being instantly culled by the seed trial's baseline.
  * Optionally evaluates on a SEARCH_MICE subset; the winner should be
    re-validated on TARGET_MICE via run_fixed_hyperparams.py.

Phase 2 (SBC-only entropy_lambda sweep) lives in
optuna_phase2_sbc_lambda.py and consumes the winner of this script.
"""

import os
import sys
import logging
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.pruners import MedianPruner

# --- Bulletproof path forcing for interactive kernels (Spyder/Jupyter)
try:
    SCRIPT_DIR = str(Path(__file__).parent.absolute())
except NameError:
    SCRIPT_DIR = str(Path().absolute())
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils_v26 import load_vr_export, apply_temporal_binning, ToTensor
from neural_dataset import NeuralDataset
from neural_network_classifier_v26 import (
    SimpleFlexibleNNClassifier,
    get_model_probabilities,
    custom_loss_all_H,
    JS_calc,
    KL_calc,
    Wasserstein_calc_1D,
)

# =====================================================================
# Device
# =====================================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS (GPU) acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA (GPU) acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("optuna_joint_v27.log")],
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================================
# Run config (edit these)
# =====================================================================
TARGET_MICE = [0, 1, 2, 3, 4, 5]    # full cohort (used to validate the winner)
SEARCH_MICE = [0, 2, 4]             # subset evaluated during the search
LOSS_FUNC = "PCA"                   # production loss for distributional targets
OBJECTIVE_MODE = "sum"              # 'sum' | 'max' | 'pareto'
N_TRIALS = 60
N_PARALLEL_TRIALS = 1               # keep 1 on MPS/single GPU; bump to 2-4 on CPU

# Hyperparameters that are no longer searched (locked at production values).
# If you ever want to revisit them, run a one-off coarse sweep on the side.
FIXED = dict(
    activation_function="tanh",
    optimizer_type="adam",
    time_window="half",
    bin_size_ms=100,
    entropy_lambda=3e-3,            # Phase 1 default; Phase 2 sweeps this for SBC
)

# Production config — enqueued as the very first trial so TPE has a strong prior.
PRODUCTION_SEED = dict(
    hidden_choice="16",
    learning_rate=5e-3,
    weight_decay=1e-4,
    minibatch_size=16,
    num_epochs=50,
)


# =====================================================================
# Cached per-mouse data (binning + z-scoring + PCA done once)
# =====================================================================
_DATA_CACHE = {}


def get_mouse_data(mouse_id, time_window, bin_size_ms):
    """Load + temporally bin + drop singletons + z-score + train/val split + per-fold PCA, once."""
    key = (mouse_id, time_window, bin_size_ms)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    activities_m, targets_perc, _, _, trials = load_vr_export(mouse_id)

    # Temporal binning expects (nTrials, tBins, nNeurons)
    act_t = np.transpose(activities_m, (1, 2, 0))
    act_b = apply_temporal_binning(act_t, time_window=time_window, bin_size_ms=bin_size_ms)
    activities_m = np.transpose(act_b, (2, 0, 1))   # back to (nNeurons, nTrials, tBins)

    # Extract unique stimulus classes
    stim = np.array(list(zip(trials["orientation"], trials["contrast"], trials["dispersion"])))
    _, cats, counts = np.unique(stim, axis=0, return_inverse=True, return_counts=True)
    
    # DROP SINGLETONS BEFORE SPLIT
    valid_cat_mask = counts[cats] > 1
    if not np.all(valid_cat_mask):
        activities_m = activities_m[:, valid_cat_mask, :]
        targets_perc = targets_perc[valid_cat_mask]
        stim = stim[valid_cat_mask]
        # Recompute cats for the filtered data
        _, cats = np.unique(stim, axis=0, return_inverse=True)

    # Stratified 80/20 split keyed on the full stim triplet, deterministic.
    train_idx, val_idx = train_test_split(
        np.arange(len(cats)),
        test_size=0.2,
        random_state=42,
        stratify=cats,
    )

    # Per-neuron z-score using TRAINING trials only.
    mu = np.mean(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
    sd = np.std(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
    sd[sd == 0] = 1.0
    activities_z = (activities_m - mu) / sd

    # PCA basis from per-condition-averaged training-fold targets (no leakage).
    train_stim = stim[train_idx]
    train_post = targets_perc[train_idx]
    uniq_train = np.unique(train_stim, axis=0)
    cond_avg = np.zeros((len(uniq_train), train_post.shape[1]))
    for i, s in enumerate(uniq_train):
        m = np.all(train_stim == s, axis=1)
        cond_avg[i] = np.mean(train_post[m], axis=0)
    pca = PCA().fit(cond_avg)
    pcs = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)
    var = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32, device=DEVICE)

    out = dict(
        activities_z=activities_z,
        targets=targets_perc,
        train_idx=train_idx,
        val_idx=val_idx,
        pcs=pcs,
        explained_variance=var,
    )
    _DATA_CACHE[key] = out
    return out


# =====================================================================
# Marginal baseline — replaces shuffled training during the search
# =====================================================================
def marginal_baseline_loss(Y_train_flat, Y_val_flat, loss_func, pcs, var):
    p = np.mean(Y_train_flat, axis=0)                                  # (n_bins,)
    P = np.tile(p[None, :], (Y_val_flat.shape[0], 1))
    P_t = torch.tensor(P, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y_val_flat, dtype=torch.float32, device=DEVICE)

    if loss_func == "PCA":
        Pp = P_t @ pcs.T
        Yp = Y_t @ pcs.T
        return float(torch.mean(torch.sum(var * (Pp - Yp) ** 2, dim=-1) * 100).item())
    if loss_func == "JS":
        return float(torch.mean(JS_calc(P_t, Y_t)).item())
    if loss_func == "KL":
        return float(torch.mean(KL_calc(P_t, Y_t)).item())
    if loss_func == "Wasserstein":
        return float(torch.mean(Wasserstein_calc_1D(P_t, Y_t)).item())
    raise ValueError(f"Unsupported loss_func for baseline: {loss_func}")


# =====================================================================
# Single-mouse joint train: PPC + SBC under the same config
# =====================================================================
def train_one_mouse(mouse_id, config, loss_func, fixed):
    data = get_mouse_data(mouse_id, fixed["time_window"], fixed["bin_size_ms"])
    A = data["activities_z"]
    Y = data["targets"]
    pcs = data["pcs"]
    var = data["explained_variance"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    n_neurons, n_trials, T = A.shape

    Y_d = np.expand_dims(Y.T, axis=2)
    Y_d = np.repeat(Y_d, T, axis=2)

    X_tr = np.copy(A[:, train_idx, :].reshape(n_neurons, -1)).T
    Y_tr = np.copy(Y_d[:, train_idx, :].reshape(Y_d.shape[0], -1)).T
    X_va = np.copy(A[:, val_idx, :].reshape(n_neurons, -1)).T
    Y_va = np.copy(Y_d[:, val_idx, :].reshape(Y_d.shape[0], -1)).T

    train_loader = DataLoader(
        NeuralDataset(X_tr, Y_tr, transform=ToTensor(DEVICE)), batch_size=T, shuffle=False
    )
    val_loader = DataLoader(
        NeuralDataset(X_va, Y_va, transform=ToTensor(DEVICE)), batch_size=T, shuffle=False
    )

    base_arch = dict(
        input_size=X_tr.shape[1],
        hidden_sizes=config["hidden_sizes"],
        output_size=Y_tr.shape[1],
        activation=fixed["activation_function"],
    )

    def train_eval(model_type):
        torch.manual_seed(0)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(0)
        model = SimpleFlexibleNNClassifier(**base_arch).to(DEVICE)
        opt = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        mb = config["minibatch_size"]
        for _ in range(config["num_epochs"]):
            model.train()
            opt.zero_grad()
            count = 0
            for x, y in train_loader:
                p = get_model_probabilities(model, x, model_type)
                loss, _ = custom_loss_all_H(
                    p, y, fixed["entropy_lambda"], model_type, pcs, var, loss_func
                )
                (loss / mb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                count += 1
                if count % mb == 0:
                    opt.step()
                    opt.zero_grad()
            if count % mb != 0:
                opt.step()
                opt.zero_grad()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                p = get_model_probabilities(model, x, model_type)
                l, _ = custom_loss_all_H(p, y, 0.0, model_type, pcs, var, loss_func)
                val_loss += l.item()
        return val_loss / len(val_loader)

    val_ppc = train_eval("ppc")
    val_sbc = train_eval("sampling")

    base = marginal_baseline_loss(Y_tr, Y_va, loss_func, pcs, var) + 1e-9
    return val_ppc / base, val_sbc / base


# =====================================================================
# Optuna objective
# =====================================================================
def make_objective(mouse_ids, loss_func, fixed, mode):
    def objective(trial):
        hidden_choice = trial.suggest_categorical(
            "hidden_choice", ["8", "16", "32", "16,16"]
        )
        config = {
            "hidden_sizes":   [int(x) for x in hidden_choice.split(",")],
            "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
            "weight_decay":   trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True),
            "minibatch_size": trial.suggest_categorical("minibatch_size", [8, 16, 32]),
            "num_epochs":     trial.suggest_categorical("num_epochs",     [30, 50, 75]),
        }

        ppc_scores, sbc_scores = [], []
        for i, mid in enumerate(mouse_ids):
            try:
                ppc, sbc = train_one_mouse(mid, config, loss_func, fixed)
            except Exception as e:
                # Log the actual traceback so it doesn't fail silently
                logging.error(f"Trial {trial.number} mouse {mid} failed:\n{traceback.format_exc()}")
                raise optuna.TrialPruned()
            
            ppc_scores.append(ppc)
            sbc_scores.append(sbc)

            if mode == "sum":
                interim = 0.5 * (np.mean(ppc_scores) + np.mean(sbc_scores))
            elif mode == "max":
                interim = max(np.mean(ppc_scores), np.mean(sbc_scores))
            else:  # pareto
                interim = 0.5 * (np.mean(ppc_scores) + np.mean(sbc_scores))
            
            if np.isnan(interim):
                logging.warning(f"Trial {trial.number} produced NaN loss. Auto-pruning.")
                raise optuna.TrialPruned()

            # Fix step index so pruner respects resource limits correctly
            trial.report(interim, step=i+1)
            
            if trial.should_prune():
                raise optuna.TrialPruned()

        ppc_mean = float(np.mean(ppc_scores))
        sbc_mean = float(np.mean(sbc_scores))
        trial.set_user_attr("ppc_mean", ppc_mean)
        trial.set_user_attr("sbc_mean", sbc_mean)

        if mode == "sum":
            return 0.5 * (ppc_mean + sbc_mean)
        if mode == "max":
            return max(ppc_mean, sbc_mean)
        if mode == "pareto":
            return ppc_mean, sbc_mean
        raise ValueError(f"Unknown OBJECTIVE_MODE: {mode}")

    return objective


# =====================================================================
# Logging callback
# =====================================================================
def log_cb(study, trial):
    print(f"\n{'='*60}\nTrial {trial.number} | {trial.state.name}\n{'='*60}")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        ppc_m = trial.user_attrs.get("ppc_mean")
        sbc_m = trial.user_attrs.get("sbc_mean")
        if ppc_m is not None and sbc_m is not None:
            print(f"  PPC = {ppc_m:.4f}   SBC = {sbc_m:.4f}")
        if not study._is_multi_objective():
            print(f"  Best so far: {study.best_value:.4f} (trial {study.best_trial.number})")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print("  Pruned.")


# =====================================================================
# Main
# =====================================================================
def run():
    study_name = f"phase1_joint_{LOSS_FUNC}_{OBJECTIVE_MODE}_v2"
    storage = f"sqlite:///{study_name}.db"

    if OBJECTIVE_MODE == "pareto":
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=["minimize", "minimize"],
            sampler=NSGAIISampler(seed=42),
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(seed=42, multivariate=True, group=True),
            # Replaced Hyperband with MedianPruner to allow robust baseline formation
            pruner=MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=1,
            ),
        )

    # Enqueue the production seed trial so TPE anchors near the known-good region.
    seeded = any(
        t.params.get("hidden_choice") == PRODUCTION_SEED["hidden_choice"]
        and abs(t.params.get("learning_rate", 0.0) - PRODUCTION_SEED["learning_rate"]) < 1e-12
        for t in study.trials
    )
    if not seeded:
        study.enqueue_trial(PRODUCTION_SEED)
        print("Enqueued production seed trial as trial 0.")

    obj = make_objective(SEARCH_MICE, LOSS_FUNC, FIXED, OBJECTIVE_MODE)
    study.optimize(
        obj,
        n_trials=N_TRIALS,
        n_jobs=N_PARALLEL_TRIALS,
        callbacks=[log_cb],
    )

    print("\n" + "=" * 60)
    if OBJECTIVE_MODE == "pareto":
        front = study.best_trials
        print(f"Pareto front: {len(front)} non-dominated trials")
        for t in front:
            print(f"  trial {t.number}  PPC={t.values[0]:.4f}  SBC={t.values[1]:.4f}  params={t.params}")
    else:
        print(f"Best joint score: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        ppc_m = study.best_trial.user_attrs.get("ppc_mean")
        sbc_m = study.best_trial.user_attrs.get("sbc_mean")
        if ppc_m is not None and sbc_m is not None:
            print(f"  -> per-arch:  PPC = {ppc_m:.4f}   SBC = {sbc_m:.4f}")

    study.trials_dataframe().to_csv(f"{study_name}_results.csv")
    print(f"\nResults written to {study_name}_results.csv")
    print(f"Storage: {storage}")
    print("Next step: copy the winning hyperparameters into PHASE1_WINNER")
    print("           in optuna_phase2_sbc_lambda.py, then run that script.")


if __name__ == "__main__":
    run()