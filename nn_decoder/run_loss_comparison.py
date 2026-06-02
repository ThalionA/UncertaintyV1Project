# -*- coding: utf-8 -*-
"""Matched-parameter loss comparison: PCA vs KL vs JS, full export.

Motivation
----------
The KL/JS entropy sweep (`run_kl_js_entropy_sweep.py`) explored several
entropy_lambda values but did NOT save training/validation curves or weight
snapshots, and there is no PCA run on disk with comparable saved outputs. To
compare the three losses fairly we need a single run where:

  * every loss uses the SAME hyperparameters (lr, weight_decay, architecture,
    epochs, entropy_lambda, val split) — only the loss function differs,
  * everything is exported: per-epoch train/val curves, weight snapshots,
    decoded posteriors (incl. per-bin), final weights, held-out fit-loss,
  * early stopping is on, with the stop epoch recorded.

That makes PCA / KL / JS directly comparable on identical footing — the thing
the sweep could not give.

Design
------
ONE parameter set per (target, bin_size). We take the per-target preset's
architecture / optimiser values (so the network and training budget are the
ones the target was tuned for) but FIX the loss-specific knob — entropy_lambda
— to a single shared value across all three losses, and force history +
snapshot export on. The only thing that varies within a (target, bin) cell is
`loss_func`.

Output layout (entropy_lambda is not in the slug, so it is fixed here):

    results/<run_name>/<target>_<loss>_<window>_<bin>ms[_all]/
        <split>.mat
        config.yaml
        checkpoints/mouse_<id>_<split>.pt   (carries the 'history' sidecar)

Run (remote GPU, see wiki/Cluster_Commands.md)
----------------------------------------------
    ssh gpu1 ; tmux new -s loss_cmp
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY -u run_loss_comparison.py 2>&1 | tee loss_cmp.log

Smaller probe first:
    $PY -u run_loss_comparison.py --targets Q --bin-sizes-ms 100 \
        --windows half --splits stratified_balanced
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config


RUN_NAME_DEFAULT = 'loss_comparison_v1'
TARGETS = ('Q', 'L')
LOSSES = ('PCA', 'KL', 'JS')
BIN_SIZES_MS = (50, 100)
WINDOWS = ('full', 'half')
SPLITS = ('stratified_balanced',)

# Shared, fixed across all losses so the comparison is matched. entropy_lambda
# is the one loss-coupled knob; pinning it equal removes it as a confound. The
# 3e-3 default is the production value (see Config.entropy_lambda docstring).
ENTROPY_LAMBDA = 3e-3
NUM_EPOCHS_CAP = 200
PATIENCE = 15
MIN_EPOCHS = 20
VAL_FRACTION = 0.2
SNAPSHOT_EVERY = 10      # weight state_dict every N epochs (+ last)


def main(run_name=RUN_NAME_DEFAULT, targets=TARGETS, losses=LOSSES,
         bin_sizes_ms=BIN_SIZES_MS, windows=WINDOWS, splits=SPLITS,
         entropy_lambda=ENTROPY_LAMBDA, num_epochs=NUM_EPOCHS_CAP,
         patience=PATIENCE, min_epochs=MIN_EPOCHS, val_fraction=VAL_FRACTION,
         snapshot_every=SNAPSHOT_EVERY):
    splits = tuple(splits)
    n_cfg = len(targets) * len(losses) * len(bin_sizes_ms) * len(windows)
    print(f"Matched loss comparison: run_name={run_name!r}")
    print(f"  targets        : {targets}")
    print(f"  losses         : {losses}  (matched params; only loss differs)")
    print(f"  bin sizes      : {bin_sizes_ms} ms")
    print(f"  time windows   : {windows}")
    print(f"  entropy_lambda : {entropy_lambda} (FIXED across losses)")
    print(f"  splits         : {splits}")
    print(f"  schedule       : up to {num_epochs} epochs, early stop "
          f"patience={patience}, min_epochs={min_epochs}, "
          f"val_fraction={val_fraction}")
    print(f"  export         : history ON, weight snapshots every "
          f"{snapshot_every} epochs")
    print(f"  total configs  : {n_cfg}")
    print(f"  total fits     : {n_cfg * 6 * len(splits)} "
          f"(6 mice * {len(splits)} split(s))")

    done = 0
    for target in targets:
        for bs in bin_sizes_ms:
            for win in windows:
                for loss in losses:
                    done += 1
                    print(f"\n[{done}/{n_cfg}] target={target} loss={loss} "
                          f"bin={bs}ms window={win}")
                    # default_config_for_target gives the per-target preset
                    # (architecture / lr / wd / epochs / minibatch). We then
                    # override loss_func, the window, and pin every shared
                    # knob so all three losses run on identical footing.
                    cfg = default_config_for_target(
                        target,
                        run_name=run_name,
                        bin_size_ms=bs,
                        loss_func=loss,
                        time_window=win,
                        entropy_lambda=entropy_lambda,
                        num_epochs=num_epochs,
                        patience=patience,
                        min_epochs=min_epochs,
                        val_fraction=val_fraction,
                        track_training_history=True,
                        weight_snapshot_every=snapshot_every,
                    )
                    run_config(cfg, splits=splits)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-name', default=RUN_NAME_DEFAULT)
    p.add_argument('--targets', nargs='+', default=list(TARGETS))
    p.add_argument('--losses', nargs='+', default=list(LOSSES))
    p.add_argument('--bin-sizes-ms', nargs='+', type=int, default=list(BIN_SIZES_MS))
    p.add_argument('--windows', nargs='+', default=list(WINDOWS))
    p.add_argument('--splits', nargs='+', default=list(SPLITS))
    p.add_argument('--entropy-lambda', type=float, default=ENTROPY_LAMBDA)
    p.add_argument('--num-epochs', type=int, default=NUM_EPOCHS_CAP)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--min-epochs', type=int, default=MIN_EPOCHS)
    p.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    p.add_argument('--snapshot-every', type=int, default=SNAPSHOT_EVERY)
    a = p.parse_args()
    main(run_name=a.run_name, targets=tuple(a.targets), losses=tuple(a.losses),
         bin_sizes_ms=tuple(a.bin_sizes_ms), windows=tuple(a.windows),
         splits=tuple(a.splits), entropy_lambda=a.entropy_lambda,
         num_epochs=a.num_epochs, patience=a.patience, min_epochs=a.min_epochs,
         val_fraction=a.val_fraction, snapshot_every=a.snapshot_every)
