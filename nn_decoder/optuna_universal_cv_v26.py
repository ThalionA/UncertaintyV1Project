#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Mouse Optuna Hyperparameter Optimization with K-Fold CV (v26)
Optimizes a single universal hyperparameter set across all subjects.
Minimizes the relative performance: (Real CV Loss / Shuffled CV Loss).
Fully parallelized across mice using joblib for Mac M3 Max (MPS).
Includes Temporal Window and Bin Size optimization.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from joblib import Parallel, delayed

# =====================================================================
# --- BULLETPROOF PATH FORCING FOR INTERACTIVE KERNELS (SPYDER/JUPYTER)
# =====================================================================
try:
    script_dir = str(Path(__file__).parent.absolute())
except NameError:
    script_dir = str(Path().absolute())

os.chdir(script_dir)

if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)

import utils_v26
from utils_v26 import load_vr_export, zscore_activity, apply_temporal_binning, ToTensor
from neural_dataset import NeuralDataset
from neural_network_classifier_v26 import SimpleFlexibleNNClassifier, get_model_probabilities, custom_loss_all_H 

print(f"\n{'='*60}")
print(f"✅ SUCCESSFULLY LOADED utils_v26 from:\n{utils_v26.__file__}")
print(f"{'='*60}\n")

# --- DEVICE SETUP (APPLE SILICON SUPPORT) ---
if torch.backends.mps.is_available():
    default_device = torch.device("mps")
    print("🚀 Using Apple Silicon MPS (GPU) Acceleration!")
elif torch.cuda.is_available():
    default_device = torch.device("cuda")
    print("🚀 Using CUDA (GPU) Acceleration!")
else:
    default_device = torch.device("cpu")
    print("⚠️ Using CPU.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("optuna_multimouse_PCA.log")])

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_single_mouse_cv(mouse_id, config, model_type, loss_func):
    """Handles the 3-Fold CV for a single mouse. Designed to run in parallel."""
    activities_m, targets_perc, targets_dec, _, trials_dict = load_vr_export(mouse_id)
    
    # --- APPLY TEMPORAL BINNING & WINDOWING ---
    time_window = config.get('time_window', 'full')
    bin_size_ms = config.get('bin_size_ms', 50)
    
    act_transposed = np.transpose(activities_m, (1, 2, 0))
    act_binned = apply_temporal_binning(act_transposed, time_window=time_window, bin_size_ms=bin_size_ms)
    activities_m = np.transpose(act_binned, (2, 0, 1)) 
    
    n_neurons, n_trials, T = activities_m.shape
    
    # Target Formatting
    target_distr = targets_perc.T  
    target_distr_ = np.expand_dims(target_distr, axis=2) 
    target_distr_ = np.repeat(target_distr_, T, axis=2) 

    # K-Fold must split across the TRIAL dimension, not the flattened bin dimension
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mouse_fold_real_losses = []
    mouse_fold_shuffled_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_trials))):
        
        # --- Z-SCORE (TRAINING SET ONLY to prevent data leakage) ---
        mean_train = np.mean(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
        std_train = np.std(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
        std_train[std_train == 0] = 1.0
        activities_m_z = (activities_m - mean_train) / std_train

        # --- REAL DATA PREP ---
        X_train = activities_m_z[:, train_idx, :]
        X_train_in = np.copy(X_train.reshape(X_train.shape[0], -1)).T
        
        Y_train = target_distr_[:, train_idx, :]
        Y_train_in = np.copy(Y_train.reshape(Y_train.shape[0], -1)).T
        
        X_val = activities_m_z[:, val_idx, :]
        X_val_in = np.copy(X_val.reshape(X_val.shape[0], -1)).T
        
        Y_val = target_distr_[:, val_idx, :]
        Y_val_in = np.copy(Y_val.reshape(Y_val.shape[0], -1)).T

        # --- SHUFFLED DATA PREP (Shuffle entirely by trial structure) ---
        shuffled_train_idx = np.random.permutation(len(train_idx))
        Y_train_shuffled = Y_train[:, shuffled_train_idx, :]
        Y_train_in_shuff = np.copy(Y_train_shuffled.reshape(Y_train_shuffled.shape[0], -1)).T

        # --- DATASETS ---
        train_set_real = NeuralDataset(X_train_in, Y_train_in, transform=ToTensor(default_device))
        train_set_shuff = NeuralDataset(X_train_in, Y_train_in_shuff, transform=ToTensor(default_device))
        val_set = NeuralDataset(X_val_in, Y_val_in, transform=ToTensor(default_device))

        # CRITICAL FIX: batch_size=T and shuffle=False keeps time bins strictly grouped by trial!
        train_loader_real = DataLoader(train_set_real, batch_size=T, shuffle=False)
        train_loader_shuff = DataLoader(train_set_shuff, batch_size=T, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=T, shuffle=False)
        
        minibatch_size = config['minibatch_size']

        # ==================================
        # 1. TRAIN REAL MODEL
        # ==================================
        model_real = SimpleFlexibleNNClassifier(
            input_size=X_train_in.shape[1], 
            hidden_sizes=config['hidden_sizes'], 
            output_size=Y_train_in.shape[1], 
            activation=config['activation_function']
        ).to(default_device)
        
        optimizer_class = optim.Adam if config['optimizer_type'] == 'adam' else optim.SGD
        optimizer_real = optimizer_class(model_real.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        for epoch in range(config['num_epochs']):
            model_real.train()
            optimizer_real.zero_grad()
            count = 0
            for batch_inputs, batch_targets in train_loader_real:
                pred_probs = get_model_probabilities(model_real, batch_inputs, model_type)
                loss, _ = custom_loss_all_H(pred_probs, batch_targets, config['entropy_lambda'], model_type, None, None, loss_func)
                
                loss = loss / minibatch_size # Gradient accumulation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_real.parameters(), max_norm=1.0)
                count += 1
                
                if count % minibatch_size == 0:
                    optimizer_real.step()
                    optimizer_real.zero_grad()
            
            if count % minibatch_size != 0:
                optimizer_real.step()
                optimizer_real.zero_grad()

        # Eval (Real)
        model_real.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                pred_probs = get_model_probabilities(model_real, batch_inputs, model_type)
                loss, _ = custom_loss_all_H(pred_probs, batch_targets, 0.0, model_type, None, None, loss_func)
                val_loss_accum += loss.item()
        mouse_fold_real_losses.append(val_loss_accum / len(val_loader))

        # ==================================
        # 2. TRAIN SHUFFLED MODEL
        # ==================================
        model_shuff = SimpleFlexibleNNClassifier(
            input_size=X_train_in.shape[1], 
            hidden_sizes=config['hidden_sizes'], 
            output_size=Y_train_in.shape[1], 
            activation=config['activation_function']
        ).to(default_device)
        
        optimizer_shuff = optimizer_class(model_shuff.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        for epoch in range(config['num_epochs']):
            model_shuff.train()
            optimizer_shuff.zero_grad()
            count = 0
            for batch_inputs, batch_targets in train_loader_shuff:
                pred_probs = get_model_probabilities(model_shuff, batch_inputs, model_type)
                loss, _ = custom_loss_all_H(pred_probs, batch_targets, config['entropy_lambda'], model_type, None, None, loss_func)
                
                loss = loss / minibatch_size
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_shuff.parameters(), max_norm=1.0)
                count += 1
                
                if count % minibatch_size == 0:
                    optimizer_shuff.step()
                    optimizer_shuff.zero_grad()
            
            if count % minibatch_size != 0:
                optimizer_shuff.step()
                optimizer_shuff.zero_grad()

        # Eval (Shuffled) - Note: Evaluated against REAL val_loader targets to match original logic
        model_shuff.eval()
        val_loss_accum_shuff = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                pred_probs = get_model_probabilities(model_shuff, batch_inputs, model_type)
                loss, _ = custom_loss_all_H(pred_probs, batch_targets, 0.0, model_type, None, None, loss_func)
                val_loss_accum_shuff += loss.item()
        mouse_fold_shuffled_losses.append(val_loss_accum_shuff / len(val_loader))

    mean_real_loss = np.mean(mouse_fold_real_losses)
    mean_shuff_loss = np.mean(mouse_fold_shuffled_losses)
    
    return mean_real_loss / (mean_shuff_loss + 1e-8)


def objective(trial, mouse_ids, model_type, loss_func):
    """Evaluates a config across all mice IN PARALLEL."""
    
    hidden_size_choice = trial.suggest_categorical('hidden_sizes', ["16", "32", "64", "16,16", "32,32"])
    hidden_sizes_list = [int(x) for x in hidden_size_choice.split(',')]
    
    config = {
        'hidden_sizes': hidden_sizes_list, 
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'num_epochs': trial.suggest_int('num_epochs', 10, 60),
        'entropy_lambda': trial.suggest_float('entropy_lambda', 1e-4, 0.5, log=True),
        'minibatch_size': trial.suggest_categorical('minibatch_size', [16, 32, 64]),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'sgd']),
        'activation_function': trial.suggest_categorical('activation_function', ['relu', 'tanh']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        
        # --- Temporal Processing Params ---
        'time_window': trial.suggest_categorical('time_window', ['full', 'half']),
        'bin_size_ms': trial.suggest_categorical('bin_size_ms', [50, 100, 200])
    }

    relative_performances = Parallel(n_jobs=len(mouse_ids), backend='threading')(
        delayed(evaluate_single_mouse_cv)(mouse_id, config, model_type, loss_func) for mouse_id in mouse_ids
    )

    return np.mean(relative_performances)


def verbose_logging_callback(study, trial):
    """Prints a highly readable status update after every trial."""
    print(f"\n{'='*60}")
    print(f"🟢 Trial {trial.number} Finished")
    print(f"{'='*60}")
    
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"📉 Relative Loss (Real / Shuffled): {trial.value:.4f}")
        
        if trial.value < 1.0:
            improvement = (1.0 - trial.value) * 100
            print(f"🧠 SUCCESS: The network is {improvement:.1f}% better than chance.")
        else:
            print("🛑 FAILED: The network is performing worse than the shuffled baseline.")
            
        print(f"🏆 Best Ratio So Far: {study.best_value:.4f} (Found in Trial {study.best_trial.number})")
        print(f"⚙️  Params Used:")
        for k, v in trial.params.items():
            print(f"    - {k}: {v}")
    
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"✂️  Trial Pruned! (Did not meet threshold).")
    print(f"{'='*60}\n")


def run_universal_optuna_study(mouse_ids, model_type, loss_func='JS', n_trials=50):
    study_name = f"universal_mice_{model_type}_{loss_func}_relative"
    db_path = f"sqlite:///{study_name}.db"
    
    print(f"\nStarting Parallel CV optimization for {study_name}")
    print(f"Tracking live in terminal, and saving to SQLite database for Optuna Dashboard.")
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize', 
        storage=db_path,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1) 
    )
    
    study.optimize(lambda trial: objective(trial, mouse_ids, model_type, loss_func), 
                   n_trials=n_trials,
                   callbacks=[verbose_logging_callback])

    print(f"\n🏁 Complete! Best universal parameters for {study_name}:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
    study.trials_dataframe().to_csv(f"{study_name}_results.csv")
    
if __name__ == "__main__":
    target_mice = [0, 1, 2, 3, 4, 5] 
    
    # 1. Optimize for Sampling-Based Code (SBC) targets relative to shuffled
    run_universal_optuna_study(mouse_ids=target_mice, model_type='sampling', loss_func='JS', n_trials=50)
    
    # 2. Optimize for Probabilistic Population Code (PPC) targets relative to shuffled
    run_universal_optuna_study(mouse_ids=target_mice, model_type='ppc', loss_func='JS', n_trials=50)