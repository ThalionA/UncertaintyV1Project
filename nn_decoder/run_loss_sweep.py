# -*- coding: utf-8 -*-
"""Loss-function sweep on the Q target with training-history capture.

Exploratory run per the 2026-05-27 meeting. The intent is to diagnose
why the spatial (PPC) decoder produces peaky output distributions
under the production PCA loss. Each loss function is trained on the
same split, same seed, same architecture — only the loss differs —
and the per-epoch training curves + per-layer weight norms + periodic
state_dict snapshots are saved so the post-hoc diagnostics can
inspect each loss's training trajectory.

Spec (from the meeting):
    target          : Q (perceptual posterior, 91-D)
    losses          : PCA, MSE, CE, JS, KL, Wasserstein
    split           : stratified_balanced ONLY (in-distribution probe)
    num_epochs      : 100 (>= 100 as requested)
    mice            : all 6 (default of run_config)
    history capture : ON, with weight_snapshot_every=10

Yardstick. All losses live on different absolute scales (KL ~ a few
nats, Wasserstein in bin units, PCA × the cosmetic 100 factor, MSE on
the [0,1] probability scale, etc.). To make the cross-loss
training-trajectory plot meaningful, ``fit_model`` always also logs
the PCA-projected eval loss under the same per-mouse 'all_trials'
basis, in the history dict's ``train_pca_yardstick`` key. That's the
column to use as the y-axis when comparing across losses.

Output:
    results/<run_name>/Q_<LOSS>_half_100ms_all/
        config.yaml
        stratified_balanced.mat
        checkpoints/mouse_<mid>_stratified_balanced.pt   <-- has ['history']

Usage:
    python run_loss_sweep.py
    python run_loss_sweep.py --run-name my_sweep_name
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config


# Default run name encodes intent + date so older sweeps don't get
# stomped on. Override via --run-name on the CLI.
RUN_NAME_DEFAULT = 'loss_sweep_peakiness_2026_05_27'

# MSE dropped after the 2026-05-27 h32 + h10 runs both showed it
# collapses to the marginal-mean baseline — degenerate, no peakiness-
# investigation signal. Re-add 'MSE' here if a future experiment wants
# it back; the rest of the pipeline still supports it.
LOSSES = ('PCA', 'CE', 'JS', 'KL', 'Wasserstein')
SPLITS = ('stratified_balanced',)        # in-distribution probe only
NUM_EPOCHS = 100
SNAPSHOT_EVERY = 10                       # state_dict every 10 epochs + final

# When set on the CLI, override the per-target Optuna preset
# hidden_sizes (default [32]). Useful for the capacity-ablation
# diagnostic from the 2026-05-27 meeting: dropping to [10] reduces the
# PPC overfit by ~3× in parameter count and lets the train-vs-val gap
# be read cleanly.
HIDDEN_SIZES_DEFAULT: list[int] | None = None
VAL_FRAC_DEFAULT = 0.0                    # carve from train; 0 = off


def main(run_name: str = RUN_NAME_DEFAULT,
         losses=LOSSES,
         splits=SPLITS,
         num_epochs: int = NUM_EPOCHS,
         snapshot_every: int = SNAPSHOT_EVERY,
         bin_size_ms: int = 100,
         mouse_id: int | None = None,
         hidden_sizes: list[int] | None = HIDDEN_SIZES_DEFAULT,
         val_frac: float = VAL_FRAC_DEFAULT):
    """Run the loss sweep.

    ``mouse_id`` selects parallel-per-mouse mode: when set, only that
    mouse runs and its result is written to a shard (no merge). When
    ``None`` (default), all 6 mice run in this process and the
    canonical ``<split>.mat`` is merged in-place. Idempotent on
    reruns — a mouse whose shard already exists is skipped.

    ``hidden_sizes`` overrides each loss's per-target Optuna preset
    (default [32]). Pass e.g. [10] for the capacity-ablation
    diagnostic.

    ``val_frac`` carves that fraction of training trials into a
    stratified validation set. Per-epoch val curves are logged into
    the history dict alongside train curves; the val set is never
    used for gradients or REP selection.
    """
    print(f"Q-target loss sweep with training history capture")
    print(f"  run_name       : {run_name}")
    print(f"  losses         : {losses}")
    print(f"  splits         : {splits}")
    print(f"  num_epochs     : {num_epochs}")
    print(f"  snapshot every : {snapshot_every} epochs")
    if hidden_sizes is not None:
        print(f"  hidden_sizes   : {hidden_sizes} (overriding Optuna preset)")
    if val_frac > 0:
        print(f"  val_frac       : {val_frac:.2f} (carved from train)")
    if mouse_id is not None:
        print(f"  mouse_id       : {mouse_id} (shards-only, no merge)")
    print()

    # Parallel-per-mouse mode runs one mouse, skips merging. The
    # surrounding wrapper script (run_loss_sweep_per_mouse.sh) calls
    # merge_shards explicitly once all per-mouse processes have
    # finished for a given loss.
    if mouse_id is not None:
        mouse_ids = (mouse_id,)
        merge = False
    else:
        mouse_ids = range(6)
        merge = True

    for loss in losses:
        overrides = dict(
            run_name=run_name,
            bin_size_ms=bin_size_ms,
            loss_func=loss,
            num_epochs=num_epochs,
            track_training_history=True,
            weight_snapshot_every=snapshot_every,
            val_frac=val_frac,
        )
        if hidden_sizes is not None:
            overrides['hidden_sizes'] = list(hidden_sizes)
        cfg = default_config_for_target('Q', **overrides)
        run_config(cfg, splits=splits, mouse_ids=mouse_ids, merge=merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default=RUN_NAME_DEFAULT)
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--snapshot-every', type=int, default=SNAPSHOT_EVERY)
    parser.add_argument('--losses', nargs='+', default=list(LOSSES),
                         help='Subset of losses to sweep')
    parser.add_argument('--mouse-id', type=int, default=None,
                         help='If set, run only this mouse (parallel-per-mouse '
                              'mode — writes a shard, skips merge). Omit to '
                              'run all 6 mice in this process and merge in-place.')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=None,
                         help='Override hidden layer sizes (default = per-target '
                              'Optuna preset). Pass space-separated ints, e.g. '
                              '"--hidden-sizes 10" for a single layer of 10.')
    parser.add_argument('--val-frac', type=float, default=0.0,
                         help='Fraction of train carved off as a stratified val '
                              'set (default 0 = no val). Per-epoch val curves '
                              'logged in the history dict; never affects '
                              'gradients or REP selection.')
    args = parser.parse_args()
    main(run_name=args.run_name,
          losses=tuple(args.losses),
          num_epochs=args.num_epochs,
          snapshot_every=args.snapshot_every,
          mouse_id=args.mouse_id,
          hidden_sizes=args.hidden_sizes,
          val_frac=args.val_frac)
