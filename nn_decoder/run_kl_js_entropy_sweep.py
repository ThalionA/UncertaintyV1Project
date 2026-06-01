# -*- coding: utf-8 -*-
"""KL / JS loss sweep with an entropy-lambda grid and early stopping.

Motivated by the PCA-vs-KL/JS posterior-smoothness investigation
(``figures/loss_smoothness_demo/LOSS_SMOOTHNESS_REPORT.md``): re-decode the
multi-bin targets under the divergence losses that *do* encourage calibrated
posteriors, on the full population, sweeping the knobs that matter for posterior
sharpness.

Grid
----
    targets         : Q, L
    losses          : KL, JS
    bin sizes       : 50 ms, 100 ms
    time windows    : full, half
    entropy_lambda  : 1e-3, 3e-3, 1e-2

That is 2 x 2 x 2 x 2 x 3 = 48 configs. Each config runs the production splits
across all 6 mice (full population -- no neuron subsampling), so the divergence
losses are evaluated on exactly the same data the PCA production runs used.

Schedule
--------
Up to 200 epochs with early stopping (patience-based on a held-out validation
fit-loss; best weights restored). See ``nn_classifier.fit_model`` and the
``patience`` / ``min_epochs`` / ``val_fraction`` Config fields.

Output layout
-------------
The directory slug (``training.config.Config.slug``) encodes target, loss,
window and bin size but NOT entropy_lambda, so each lambda is given its own
``run_name`` subtree to avoid collisions:

    results/<run_name>/lam1e-03/Q_KL_full_50ms/stratified_balanced.mat
    results/<run_name>/lam3e-03/Q_JS_half_100ms/...
    ...

Run (remote GPU, via wiki/Cluster_Commands.md)
----------------------------------------------
    ssh gpu1
    tmux new -s kljs_sweep
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_kl_js_entropy_sweep.py 2>&1 | tee kljs_sweep.log
    # detach: Ctrl-B then D

A smaller probe first is recommended, e.g.
    $PY run_kl_js_entropy_sweep.py --targets Q --losses KL \
        --bin-sizes-ms 100 --windows half --entropy-lambdas 3e-3
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config


RUN_NAME_DEFAULT = 'kl_js_entropy_sweep_v1'
TARGETS = ('Q', 'L')
LOSSES = ('KL', 'JS')
BIN_SIZES_MS = (50, 100)
WINDOWS = ('full', 'half')
ENTROPY_LAMBDAS = (1e-3, 3e-3, 1e-2)

# Splits to run per config. Default is the full production trio; pass a subset
# via --splits (e.g. just 'stratified_balanced') to run a faster in-distribution
# probe without the two OOD-generalization splits.
SPLITS = ('stratified_balanced', 'generalize_contrast', 'generalize_dispersion')

# Early-stopping schedule.
NUM_EPOCHS_CAP = 200    # upper bound; early stopping usually finishes sooner
PATIENCE = 15           # epochs of no val-improvement before stopping
MIN_EPOCHS = 20         # never stop before this many epochs
VAL_FRACTION = 0.2      # held-out slice of training trials for the stop signal


def _lam_tag(lam: float) -> str:
    """Filesystem-safe run_name suffix for an entropy_lambda value."""
    return f"lam{lam:.0e}"        # e.g. 1e-03 -> 'lam1e-03'


def main(run_name: str = RUN_NAME_DEFAULT,
         targets=TARGETS,
         losses=LOSSES,
         bin_sizes_ms=BIN_SIZES_MS,
         windows=WINDOWS,
         entropy_lambdas=ENTROPY_LAMBDAS,
         splits=SPLITS,
         num_epochs=NUM_EPOCHS_CAP,
         patience=PATIENCE,
         min_epochs=MIN_EPOCHS,
         val_fraction=VAL_FRACTION):
    n_cfg = (len(targets) * len(losses) * len(bin_sizes_ms)
             * len(windows) * len(entropy_lambdas))
    splits = tuple(splits)
    print(f"KL/JS entropy sweep: run_name={run_name!r}")
    print(f"  targets        : {targets}")
    print(f"  losses         : {losses}")
    print(f"  bin sizes      : {bin_sizes_ms} ms")
    print(f"  time windows   : {windows}")
    print(f"  entropy_lambda : {entropy_lambdas}")
    print(f"  splits         : {splits}")
    print(f"  schedule       : up to {num_epochs} epochs, early stop "
          f"patience={patience}, min_epochs={min_epochs}, "
          f"val_fraction={val_fraction}")
    print(f"  total configs  : {n_cfg}")
    print(f"  total fits     : {n_cfg * 6 * len(splits)} "
          f"(6 mice * {len(splits)} split(s))")

    done = 0
    for lam in entropy_lambdas:
        lam_run = f"{run_name}/{_lam_tag(lam)}"
        for target in targets:
            for loss in losses:
                for bs in bin_sizes_ms:
                    for win in windows:
                        done += 1
                        print(f"\n[{done}/{n_cfg}] target={target} loss={loss} "
                              f"bin={bs}ms window={win} entropy_lambda={lam:g}")
                        cfg = default_config_for_target(
                            target,
                            run_name=lam_run,
                            bin_size_ms=bs,
                            loss_func=loss,
                            time_window=win,
                            entropy_lambda=lam,
                            num_epochs=num_epochs,
                            patience=patience,
                            min_epochs=min_epochs,
                            val_fraction=val_fraction,
                        )
                        run_config(cfg, splits=splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default=RUN_NAME_DEFAULT,
                        help='Output directory name under results/')
    parser.add_argument('--targets', nargs='+', default=list(TARGETS),
                        help='Subset of target types (Q L)')
    parser.add_argument('--losses', nargs='+', default=list(LOSSES),
                        help='Subset of losses (KL JS)')
    parser.add_argument('--bin-sizes-ms', nargs='+', type=int,
                        default=list(BIN_SIZES_MS), help='Bin sizes in ms')
    parser.add_argument('--windows', nargs='+', default=list(WINDOWS),
                        help='Time windows (full half)')
    parser.add_argument('--entropy-lambdas', nargs='+', type=float,
                        default=list(ENTROPY_LAMBDAS),
                        help='Entropy-lambda values to sweep')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS),
                        help='Train/test splits to run (default: all three '
                             'production splits). Pass a subset for a faster '
                             'probe, e.g. --splits stratified_balanced')
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS_CAP,
                        help='Epoch cap (early stopping usually finishes sooner)')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early-stop patience in epochs (0 disables)')
    parser.add_argument('--min-epochs', type=int, default=MIN_EPOCHS,
                        help='Minimum epochs before early stopping may trigger')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION,
                        help='Validation slice fraction for the stop signal')
    args = parser.parse_args()
    main(run_name=args.run_name, targets=tuple(args.targets),
         losses=tuple(args.losses), bin_sizes_ms=tuple(args.bin_sizes_ms),
         windows=tuple(args.windows),
         entropy_lambdas=tuple(args.entropy_lambdas),
         splits=tuple(args.splits),
         num_epochs=args.num_epochs, patience=args.patience,
         min_epochs=args.min_epochs, val_fraction=args.val_fraction)
