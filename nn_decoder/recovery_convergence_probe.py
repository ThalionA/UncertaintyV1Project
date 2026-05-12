# -*- coding: utf-8 -*-
"""Recovery-convergence probe — answers the supervisor's question:
"In the crossover experiment, the diagonal (PPC fit to PPC-generated
targets, SBC fit to SBC-generated targets) doesn't go to 0. Is that
because training didn't converge, or is there something more
fundamental?"

By construction, loss=0 IS achievable: the same trained model that
generated the targets achieves zero loss on them. The recovery training
fails to reach that floor because:

  1. Fresh random initialisation lands in a different optimum.
  2. The 30-50 epoch / REP=5 budget may not be enough to converge.
  3. (SBC only) per-bin sharpness penalty allows multiple sharp per-bin
     sequences that all time-average to the same target, so multiple
     equally good optima exist.

This script tests (1) and (2) directly across **all 6 mice and both
target architectures** (PPC-generated targets AND SBC-generated targets,
the two halves of the crossover experiment).

For each (mouse, target_arch) we train both PPC and SBC decoders against
the saved full_decoded predictions from a previous production run,
sweeping over (num_epochs, REP). To save compute we exploit the fact
that REP just selects best-of-N from independent inits: we train
max(REP_GRID) models once per (mouse, target_arch, decoder_arch,
num_epochs) and slice for each REP value.

Output (per (mouse, target_arch)):
  figures/recovery_probe/sweep_mouse<i>_target_<arch>.csv
  figures/recovery_probe/sweep_mouse<i>_target_<arch>.svg

Plus an aggregate across all (mouse, target_arch) pairs:
  figures/recovery_probe/sweep_aggregate.csv
  figures/recovery_probe/sweep_aggregate.svg

Progress is checkpointed to the aggregate CSV after each
(mouse, target_arch) completes, so re-running picks up where it left
off — already-computed (mouse, target_arch, num_epochs, REP,
decoder_arch) rows are skipped.

Reading the result:
  - If both diagonals drop monotonically with more training and
    approach 0, the issue is purely under-training. Bump epochs / REP
    in the recovery script.
  - If the diagonals plateau well above 0 even with 200 epochs and
    REP=20, we have a deeper issue (likely the SBC entropy-penalty
    multi-optimum problem; PPC should still go to 0).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
# Defensive: re-bind the real builtin `all` in this script's globals so the
# skip-decision below isn't broken if `all` has been shadowed in the
# IPython/Spyder kernel that runs us via `%runcell -i` (see GOTCHAS).
from builtins import all

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils_v26 import (
    load_vr_export, apply_temporal_binning, ToTensor,
    get_stratified_train_test_indices, get_generalization_split_indices,
)
from neural_dataset import NeuralDataset
from neural_network_classifier_v26 import (
    SimpleFlexibleNNClassifier, get_model_probabilities, custom_loss_all_H,
)
from sklearn.decomposition import PCA


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =====================================================================
# Defaults (overridable from CLI or %runcell-time edits)
# =====================================================================
MOUSE_IDS    = (0, 1, 2, 3, 4, 5)
TARGET_ARCHS = ('spat', 'temp')                # both crossover sides
BASE_FILE    = 'population_results_fixed_hyperparams_stratified_balanced.mat'
SPLIT_TYPE   = 'stratified_balanced'

EPOCH_GRID   = (50, 100, 200)
REP_GRID     = (5, 10, 20)

LOSS_FUNC      = 'PCA'
ENTROPY_LAMBDA = 3e-3
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
HIDDEN_SIZES   = (32,)
MINIBATCH_SIZE = 16


# =====================================================================
# Data prep
# =====================================================================
def prepare_data(mouse_id, base_file, target_arch, split_type='stratified_balanced'):
    """Reproduce the train/test split + activity z-scoring, and load the
    saved full_decoded from base_file as the new target. Returns
    everything needed for a training run (X_tr, Y_tr, X_te, Y_te, T,
    pcs, var)."""
    activities_m, _, _, _, trials = load_vr_export(mouse_id)
    act_t = np.transpose(activities_m, (1, 2, 0))
    act_b = apply_temporal_binning(act_t, time_window='half', bin_size_ms=100)
    activities_m = np.transpose(act_b, (2, 0, 1))

    stim = np.array(list(zip(trials['orientation'], trials['contrast'], trials['dispersion'])))
    _, cats = np.unique(stim, axis=0, return_inverse=True)

    if split_type == 'stratified_balanced':
        train_idx, test_idx = get_stratified_train_test_indices(
            cats, test_size=0.5, random_state=42)
    else:
        train_idx, test_idx = get_generalization_split_indices(
            trials, split_type=split_type, random_state=42)

    # Per-neuron z-score on training trials only.
    mu = np.mean(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
    sd = np.std(activities_m[:, train_idx, :], axis=(1, 2), keepdims=True)
    sd[sd == 0] = 1.0
    activities_z = (activities_m - mu) / sd

    # Targets: the saved full_decoded from the base PPC or SBC run.
    mat = sio.loadmat(base_file, simplify_cells=True)
    Q_target = np.asarray(mat['results'][f'mouse_{mouse_id}']['Dist'][target_arch]['full_decoded'])

    n_neurons, n_trials, T = activities_z.shape
    Y_d = np.expand_dims(Q_target.T, axis=2)
    Y_d = np.repeat(Y_d, T, axis=2)

    X_tr = np.copy(activities_z[:, train_idx, :].reshape(n_neurons, -1)).T
    Y_tr = np.copy(Y_d[:, train_idx, :].reshape(Y_d.shape[0], -1)).T
    X_te = np.copy(activities_z[:, test_idx, :].reshape(n_neurons, -1)).T
    Y_te = np.copy(Y_d[:, test_idx, :].reshape(Y_d.shape[0], -1)).T

    # PCA basis from per-condition-averaged training targets.
    train_stim = stim[train_idx]
    uniq = np.unique(train_stim, axis=0)
    cond_avg = np.zeros((len(uniq), Q_target.shape[1]))
    for i, s in enumerate(uniq):
        m = np.all(train_stim == s, axis=1)
        cond_avg[i] = Q_target[train_idx][m].mean(axis=0)
    pca = PCA().fit(cond_avg)
    pcs = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)
    var = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32, device=DEVICE)

    return X_tr, Y_tr, X_te, Y_te, T, pcs, var


# =====================================================================
# Single train+eval
# =====================================================================
def train_eval(model_type, X_tr, Y_tr, X_te, Y_te, T, pcs, var,
                hidden_sizes, num_epochs, lr, weight_decay, mb,
                entropy_lambda, loss_func, seed):
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif DEVICE.type == 'mps' and hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)

    train_loader = DataLoader(
        NeuralDataset(X_tr, Y_tr, transform=ToTensor(DEVICE)),
        batch_size=T, shuffle=False)
    val_loader = DataLoader(
        NeuralDataset(X_te, Y_te, transform=ToTensor(DEVICE)),
        batch_size=T, shuffle=False)

    model = SimpleFlexibleNNClassifier(
        input_size=X_tr.shape[1], hidden_sizes=list(hidden_sizes),
        output_size=Y_tr.shape[1], activation='tanh',
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(num_epochs):
        model.train()
        opt.zero_grad()
        cnt = 0
        for x, y in train_loader:
            p = get_model_probabilities(model, x, model_type)
            loss, _ = custom_loss_all_H(p, y, entropy_lambda, model_type,
                                         pcs, var, loss_func)
            (loss / mb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            cnt += 1
            if cnt % mb == 0:
                opt.step()
                opt.zero_grad()
        if cnt % mb != 0:
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


# =====================================================================
# Per-(mouse, target_arch) sweep — trains max(REP_GRID) seeds once and
# slices for each rep value
# =====================================================================
def sweep_one(mouse_id, base_file, target_arch, epoch_grid, rep_grid,
               existing_keys=None):
    """Returns rows. ``existing_keys`` is a set of
    (num_epochs, REP, decoder_arch) tuples already in the aggregate CSV
    for this (mouse, target_arch); rows in that set are skipped."""
    if existing_keys is None:
        existing_keys = set()

    # Defensive guard: an empty rep_grid or epoch_grid means there's no
    # work to do, and the vacuous-truth bug in ``all(... for k in [])``
    # would have silently "skipped" everything. Surface this loudly.
    if not list(rep_grid) or not list(epoch_grid):
        raise ValueError(
            f"sweep_one called with empty grid — refusing to silently "
            f"do nothing. epoch_grid={list(epoch_grid)!r}, "
            f"rep_grid={list(rep_grid)!r}. Most likely cause: sys.argv "
            f"in your IPython kernel has a stale '--rep-grid' / "
            f"'--epoch-grid' override from a previous cell run. Reset "
            f"with `sys.argv = ['recovery_convergence_probe.py']` and "
            f"re-run."
        )

    print(f"\n[recovery_probe] mouse {mouse_id}  target_arch={target_arch}", flush=True)
    X_tr, Y_tr, X_te, Y_te, T, pcs, var = prepare_data(
        mouse_id, base_file, target_arch, SPLIT_TYPE)
    print(f"  X_tr={X_tr.shape}  Y_tr={Y_tr.shape}  T={T}  device={DEVICE}", flush=True)

    max_reps = max(rep_grid)
    rows = []
    for n_epochs in epoch_grid:
        for model_type, arch_label in (('ppc', 'PPC'), ('sampling', 'SBC')):
            # Skip if every rep value for this (epochs, arch) is already done
            need_keys = [(n_epochs, r, arch_label) for r in rep_grid]
            # `all(...)` is vacuously True on an empty iterable; the
            # `if not list(rep_grid)` guard above prevents this branch
            # from being reached with an empty need_keys. We import
            # `all` from `builtins` at the top of the file to defend
            # against the name being shadowed in IPython kernels.
            if all(k in existing_keys for k in need_keys):
                print(f"  epochs={n_epochs:>4}  arch={arch_label:>3}  "
                      f"all REP values already on disk — skipping", flush=True)
                continue

            t0 = time.time()
            losses = []
            for seed in range(max_reps):
                l = train_eval(
                    model_type, X_tr, Y_tr, X_te, Y_te, T, pcs, var,
                    HIDDEN_SIZES, n_epochs, LEARNING_RATE, WEIGHT_DECAY,
                    MINIBATCH_SIZE, ENTROPY_LAMBDA, LOSS_FUNC, seed=seed,
                )
                losses.append(l)
            elapsed = time.time() - t0
            losses = np.array(losses)

            for n_rep in rep_grid:
                if (n_epochs, n_rep, arch_label) in existing_keys:
                    continue
                sub = losses[:n_rep]
                best = float(np.min(sub))
                mean = float(np.mean(sub))
                print(f"  epochs={n_epochs:>4}  rep={n_rep:>2}  arch={arch_label:>3}  "
                      f"best={best:.5f}  mean={mean:.5f}")
                rows.append(dict(
                    mouse=mouse_id, target_arch=target_arch,
                    decoder_arch=arch_label,
                    num_epochs=n_epochs, REP=n_rep,
                    best_val_loss=best, mean_val_loss=mean,
                    elapsed_s=elapsed,
                ))
    return rows


# =====================================================================
# Plotting
# =====================================================================
def plot_one(df_one, out_path):
    """Plot loss curves for a single (mouse, target_arch)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('talk'); sns.set_style('ticks')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    target_arch = df_one['target_arch'].iloc[0]
    for ax, arch in zip(axes, ('PPC', 'SBC')):
        s = df_one[df_one['decoder_arch'] == arch]
        for rep, c in zip(sorted(s['REP'].unique()),
                           ('steelblue', 'darkorange', 'firebrick')):
            ss = s[s['REP'] == rep].sort_values('num_epochs')
            ax.plot(ss['num_epochs'], ss['best_val_loss'],
                     marker='o', lw=2, color=c, label=f'REP={rep}')
        ax.set_xlabel('num_epochs')
        ax.set_ylabel('best val loss')
        ax.set_title(f'{arch} decoder fit to {target_arch.upper()} target')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1,
                    label='by-construction floor (0)')
        ax.legend(fontsize=9)
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Mouse {df_one['mouse'].iloc[0]}  target={target_arch.upper()}: "
        f"does diagonal recovery loss reach 0 with more training?"
    )
    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    import matplotlib.pyplot as _plt; _plt.close(fig)


def plot_aggregate(df, out_path):
    """Aggregate plot: mean ± SEM across mice of best_val_loss, per
    (target_arch, decoder_arch, num_epochs, REP). 2 rows (target_arch)
    x 2 cols (decoder_arch)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    sns.set_context('talk'); sns.set_style('ticks')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    target_archs = sorted(df['target_arch'].unique())
    decoder_archs = ('PPC', 'SBC')

    for r, ta in enumerate(target_archs):
        for c, da in enumerate(decoder_archs):
            ax = axes[r, c]
            s = df[(df['target_arch'] == ta) & (df['decoder_arch'] == da)]
            for rep, col in zip(sorted(s['REP'].unique()),
                                 ('steelblue', 'darkorange', 'firebrick')):
                ss = s[s['REP'] == rep]
                grouped = ss.groupby('num_epochs')['best_val_loss']
                means = grouped.mean()
                sems = grouped.apply(lambda x: stats.sem(x) if len(x) > 1 else 0)
                xs = means.index.values
                ax.errorbar(xs, means.values, yerr=sems.values,
                             marker='o', lw=2, capsize=4, color=col,
                             label=f'REP={rep}')
            ax.axhline(0, color='grey', linestyle='--', linewidth=1,
                        label='floor (0)')
            ax.set_xlabel('num_epochs')
            ax.set_ylabel('mean best val loss (across 6 mice)')
            diag = '(DIAGONAL)' if (ta == 'spat' and da == 'PPC') or \
                                    (ta == 'temp' and da == 'SBC') else ''
            ax.set_title(f'target={ta.upper()}  decoder={da}  {diag}')
            ax.legend(fontsize=9)
            ax.grid(linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
    fig.suptitle(
        'Recovery convergence — mean across 6 mice. '
        'DIAGONAL cells answer the supervisor\'s question.'
    )
    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    import matplotlib.pyplot as _plt; _plt.close(fig)


# =====================================================================
# Driver
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mouse-ids', nargs='+', type=int, default=list(MOUSE_IDS))
    parser.add_argument('--target-archs', nargs='+', default=list(TARGET_ARCHS),
                        choices=['spat', 'temp'])
    parser.add_argument('--base-file', default=BASE_FILE)
    parser.add_argument('--out-dir', default='figures/recovery_probe')
    parser.add_argument('--epoch-grid', nargs='+', type=int,
                        default=list(EPOCH_GRID))
    parser.add_argument('--rep-grid', nargs='+', type=int,
                        default=list(REP_GRID))
    args, _ = parser.parse_known_args()

    # Echo resolved args so the user can spot e.g. a stale --rep-grid
    # override before training spends an hour silently skipping cells.
    print(f"[recovery_probe] sys.argv = {sys.argv}", flush=True)
    print(f"[recovery_probe] resolved args:", flush=True)
    print(f"  mouse_ids    = {args.mouse_ids}", flush=True)
    print(f"  target_archs = {args.target_archs}", flush=True)
    print(f"  epoch_grid   = {args.epoch_grid}", flush=True)
    print(f"  rep_grid     = {args.rep_grid}", flush=True)
    print(f"  base_file    = {args.base_file}", flush=True)
    print(f"  out_dir      = {args.out_dir}", flush=True)
    if not args.epoch_grid or not args.rep_grid:
        raise ValueError(
            "epoch_grid and rep_grid must both be non-empty. Reset "
            "sys.argv = ['recovery_convergence_probe.py'] in IPython "
            "and re-run if you're seeing an empty list here."
        )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    aggregate_csv = os.path.join(out_dir, 'sweep_aggregate.csv')

    # Resume from existing CSV if present
    if os.path.exists(aggregate_csv):
        existing = pd.read_csv(aggregate_csv)
        print(f"[recovery_probe] resuming from {aggregate_csv}  "
              f"({len(existing)} rows already on disk)")
    else:
        existing = pd.DataFrame()

    all_rows = existing.to_dict('records') if not existing.empty else []

    for mouse_id in args.mouse_ids:
        for target_arch in args.target_archs:
            existing_keys = set()
            if not existing.empty:
                done = existing[(existing['mouse'] == mouse_id)
                                 & (existing['target_arch'] == target_arch)]
                existing_keys = set(zip(done['num_epochs'].astype(int),
                                          done['REP'].astype(int),
                                          done['decoder_arch']))

            new_rows = sweep_one(
                mouse_id, args.base_file, target_arch,
                tuple(args.epoch_grid), tuple(args.rep_grid),
                existing_keys=existing_keys,
            )
            if not new_rows:
                continue
            all_rows.extend(new_rows)

            # Checkpoint after each (mouse, target_arch) completes
            df = pd.DataFrame(all_rows)
            df.to_csv(aggregate_csv, index=False)
            print(f"[recovery_probe] checkpointed -> {aggregate_csv}  "
                  f"(total rows: {len(df)})")

            # Per-(mouse, target_arch) plot + CSV slice
            df_one = df[(df['mouse'] == mouse_id) & (df['target_arch'] == target_arch)]
            slice_csv = os.path.join(
                out_dir, f'sweep_mouse{mouse_id}_target_{target_arch}.csv')
            slice_svg = os.path.join(
                out_dir, f'sweep_mouse{mouse_id}_target_{target_arch}.svg')
            df_one.to_csv(slice_csv, index=False)
            plot_one(df_one, slice_svg)

    # Final aggregate plot
    df = pd.DataFrame(all_rows)
    if len(df):
        agg_svg = os.path.join(out_dir, 'sweep_aggregate.svg')
        plot_aggregate(df, agg_svg)
        print(f"\n[recovery_probe] aggregate plot -> {agg_svg}")
        # Headline diagonal summary
        print("\nDiagonal-cell summary (mean across mice, REP=max):")
        max_rep = max(args.rep_grid)
        diag = df[
            ((df['target_arch'] == 'spat') & (df['decoder_arch'] == 'PPC'))
            | ((df['target_arch'] == 'temp') & (df['decoder_arch'] == 'SBC'))
        ]
        diag = diag[diag['REP'] == max_rep]
        if not diag.empty:
            summary = (diag.groupby(['target_arch', 'num_epochs'])['best_val_loss']
                       .agg(['mean', 'std', 'count']).round(5))
            print(summary)


if __name__ == '__main__':
    main()
