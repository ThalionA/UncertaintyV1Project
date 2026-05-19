#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: SBC-only entropy_lambda sweep
======================================

Picks up where optuna_joint_v27.py left off. The Phase 1 study found a fair
shared hyperparameter set for PPC and SBC, holding entropy_lambda fixed at the
production default (3e-3). entropy_lambda only affects the SBC architecture
(PPC's branch in custom_loss_all_H sets penalty = 0.0 regardless of lambda),
so we can sweep it cheaply in a 1D study without touching PPC.

Procedure:
  * Load the Phase 1 winner into PHASE1_WINNER (paste from Phase 1 logs).
  * Sweep entropy_lambda in [1e-4, 1e-1] log-uniform, ~15 trials.
  * Train SBC only, evaluate on the same 80/20 split as Phase 1.
  * Normalise to the same hyperparameter-invariant marginal baseline so the
    cross-mouse aggregation matches Phase 1.
  * Seed at lambda = 3e-3 (production) so we can confirm or beat that value.

Output:
  * sqlite study + CSV of trial results.
  * Best lambda value to plug into run_fixed_hyperparams.py for SBC training.

Note: PPC training is intentionally NOT repeated here. Use the Phase 1 winner's
PPC results directly when assembling the production refit.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    SCRIPT_DIR = str(Path(__file__).parent.absolute())
except NameError:
    SCRIPT_DIR = str(Path().absolute())
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Reuse Phase 1 plumbing
from optuna_joint_v27 import (
    DEVICE,
    FIXED,
    SEARCH_MICE,
    TARGET_MICE,
    LOSS_FUNC,
    get_mouse_data,
    marginal_baseline_loss,
)
from utils import ToTensor
from neural_dataset import NeuralDataset
from nn_classifier import (
    SimpleFlexibleNNClassifier,
    get_model_probabilities,
    custom_loss_all_H,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================================
# Phase 1 winner — PASTE FROM PHASE 1 LOG OUTPUT BEFORE RUNNING
# =====================================================================
# These are placeholders that match the production seed. After Phase 1
# finishes, replace these with the printed best params.
PHASE1_WINNER = dict(
    hidden_sizes=[32],
    learning_rate=1e-3,
    weight_decay=3.7e-4,
    minibatch_size=16,
    num_epochs=30,
)

# =====================================================================
# Sweep config
# =====================================================================
N_TRIALS = 15
MICE = TARGET_MICE                  # full 6-mouse cohort
LAMBDA_RANGE = (1e-4, 1e-1)         # log-uniform
SEED_LAMBDA = 3e-3                  # production default — enqueued as trial 0


# =====================================================================
# Single-mouse SBC train + eval at a given lambda
# =====================================================================
def train_sbc_with_lambda(mouse_id, lam):
    data = get_mouse_data(mouse_id, FIXED["time_window"], FIXED["bin_size_ms"])
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

    torch.manual_seed(0)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = SimpleFlexibleNNClassifier(
        input_size=X_tr.shape[1],
        hidden_sizes=PHASE1_WINNER["hidden_sizes"],
        output_size=Y_tr.shape[1],
        activation=FIXED["activation_function"],
    ).to(DEVICE)
    opt = optim.Adam(
        model.parameters(),
        lr=PHASE1_WINNER["learning_rate"],
        weight_decay=PHASE1_WINNER["weight_decay"],
    )
    mb = PHASE1_WINNER["minibatch_size"]

    for _ in range(PHASE1_WINNER["num_epochs"]):
        model.train()
        opt.zero_grad()
        count = 0
        for x, y in train_loader:
            p = get_model_probabilities(model, x, "sampling")
            loss, _ = custom_loss_all_H(p, y, lam, "sampling", pcs, var, LOSS_FUNC)
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
            p = get_model_probabilities(model, x, "sampling")
            l, _ = custom_loss_all_H(p, y, 0.0, "sampling", pcs, var, LOSS_FUNC)
            val_loss += l.item()

    base = marginal_baseline_loss(Y_tr, Y_va, LOSS_FUNC, pcs, var) + 1e-9
    return (val_loss / len(val_loader)) / base


# =====================================================================
# Optuna objective
# =====================================================================
def objective(trial):
    lam = trial.suggest_float("entropy_lambda", LAMBDA_RANGE[0], LAMBDA_RANGE[1], log=True)

    scores = []
    for i, mid in enumerate(MICE):
        try:
            s = train_sbc_with_lambda(mid, lam)
        except Exception as e:
            print(f"[!] Trial {trial.number} mouse {mid} failed: {e}")
            raise optuna.TrialPruned()
        scores.append(s)

        # Median pruner has something to chew on after each mouse.
        trial.report(float(np.mean(scores)), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    val = float(np.mean(scores))
    trial.set_user_attr("sbc_per_mouse", [float(s) for s in scores])
    return val


def log_cb(study, trial):
    print(f"\n{'='*60}\nTrial {trial.number} | {trial.state.name}\n{'='*60}")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        lam = trial.params.get("entropy_lambda")
        per_mouse = trial.user_attrs.get("sbc_per_mouse", [])
        print(f"  entropy_lambda = {lam:.5g}")
        print(f"  joint SBC score = {trial.value:.4f}")
        print(f"  per-mouse: {['%.3f' % v for v in per_mouse]}")
        print(f"  Best so far: {study.best_value:.4f} at lambda = {study.best_params['entropy_lambda']:.5g}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print("  Pruned by MedianPruner.")


def run():
    study_name = f"phase2_sbc_entropy_lambda_{LOSS_FUNC}"
    storage = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    # Seed at the production lambda so we always have it for direct comparison.
    seeded = any(
        abs(t.params.get("entropy_lambda", -1) - SEED_LAMBDA) < 1e-12
        for t in study.trials
    )
    if not seeded:
        study.enqueue_trial({"entropy_lambda": SEED_LAMBDA})
        print(f"Enqueued production seed lambda = {SEED_LAMBDA} as trial 0.")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_cb])

    print("\n" + "=" * 60)
    print(f"Phase 2 complete. Best entropy_lambda for SBC:")
    print(f"  lambda = {study.best_params['entropy_lambda']:.5g}")
    print(f"  joint SBC score = {study.best_value:.4f}")
    print("\nNext step: plug this lambda into run_fixed_hyperparams.py for the")
    print("SBC training pass. PPC is invariant to lambda and uses the Phase 1 winner directly.")

    study.trials_dataframe().to_csv(f"{study_name}_results.csv")
    print(f"\nResults written to {study_name}_results.csv")


if __name__ == "__main__":
    run()
