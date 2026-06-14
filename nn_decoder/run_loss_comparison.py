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
# All non-degenerate losses, matched + early-stopped, so plot_loss_sweep.py and
# cross_loss_eval.py can compare every loss on equal (early-stopped) footing —
# the gap the non-early-stopped run_loss_sweep.py data left open. MSE is omitted
# on purpose: it collapses to the marginal-mean baseline (degenerate). Pass
# --losses to subset, or add MSE back explicitly if a control run wants it.
LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
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
         snapshot_every=SNAPSHOT_EVERY, hidden_sizes=None, flat_evar=False,
         shape_lambda=0.0, evar_alpha=1.0, entropy_lambdas=None):
    splits = tuple(splits)
    # hidden_sizes=None -> use each target's preset architecture (default,
    # unchanged behaviour). A list -> run the whole grid once per hidden width,
    # each isolated under run_name + '_h<H>' so cells don't collide. This is the
    # overfitting-vs-width ablation (2026-06-03 meeting item 1).
    hs_list = [None] if not hidden_sizes else list(hidden_sizes)
    # entropy_lambdas=None -> single fixed entropy_lambda, no suffix (unchanged).
    # A list -> sweep the entropy/sharpness penalty lambda_H, running the whole
    # grid once per value, each isolated under run_name + '_entlam<lambda>'. This
    # is the loss x lambda_H temporal sweep (2026-06-10 meeting). The penalty acts
    # on the temporal (sampling) model only, so the spatial fit is lambda_H-invariant.
    sweeping_el = entropy_lambdas is not None
    el_list = list(entropy_lambdas) if sweeping_el else [entropy_lambda]
    n_cfg = len(targets) * len(losses) * len(bin_sizes_ms) * len(windows)
    print(f"Matched loss comparison: run_name={run_name!r}")
    print(f"  targets        : {targets}")
    print(f"  losses         : {losses}  (matched params; only loss differs)")
    print(f"  bin sizes      : {bin_sizes_ms} ms")
    print(f"  time windows   : {windows}")
    if sweeping_el:
        print(f"  entropy_lambda : SWEEP {el_list}  (lambda_H; temporal only)")
    else:
        print(f"  entropy_lambda : {entropy_lambda} (FIXED across losses)")
    print(f"  splits         : {splits}")
    print(f"  hidden sizes   : {hs_list}  ('None' = per-target preset)")
    print(f"  schedule       : up to {num_epochs} epochs, early stop "
          f"patience={patience}, min_epochs={min_epochs}, "
          f"val_fraction={val_fraction}")
    print(f"  export         : history ON, weight snapshots every "
          f"{snapshot_every} epochs")
    print(f"  total configs  : {n_cfg} x {len(hs_list)} hidden-size(s) "
          f"x {len(el_list)} entropy-lambda(s)")
    print(f"  total fits     : "
          f"{n_cfg * len(hs_list) * len(el_list) * 6 * len(splits)} "
          f"(6 mice * {len(splits)} split(s))")

    for hs in hs_list:
        base_rn = run_name if hs is None else f"{run_name}_h{hs}"
        if flat_evar:
            base_rn = f"{base_rn}_flatevar"
        elif shape_lambda > 0:
            base_rn = f"{base_rn}_shape{shape_lambda:g}".replace('.', 'p')
        elif evar_alpha != 1.0:
            base_rn = f"{base_rn}_alpha{evar_alpha:g}".replace('.', 'p')
        if hs is not None:
            print(f"\n=== hidden_sizes=[{hs}]  ->  base run_name={base_rn!r} ===")
        for el in el_list:
            rn = f"{base_rn}_entlam{el:g}".replace('.', 'p') if sweeping_el else base_rn
            if sweeping_el:
                print(f"\n=== entropy_lambda={el:g}  ->  run_name={rn!r} ===")
            done = 0
            for target in targets:
                for bs in bin_sizes_ms:
                    for win in windows:
                        for loss in losses:
                            done += 1
                            print(f"\n[{done}/{n_cfg}] target={target} loss={loss} "
                                  f"bin={bs}ms window={win}"
                                  + (f" H={hs}" if hs is not None else "")
                                  + (f" lambda_H={el:g}" if sweeping_el else ""))
                            # default_config_for_target gives the per-target preset
                            # (architecture / lr / wd / epochs / minibatch). We then
                            # override loss_func, the window, and pin every shared
                            # knob so all losses run on identical footing. When
                            # ablating width we also override hidden_sizes; when
                            # sweeping lambda_H we override entropy_lambda per value.
                            extra = {} if hs is None else {'hidden_sizes': [hs]}
                            if flat_evar:
                                extra['flat_evar'] = True
                            elif shape_lambda > 0:
                                extra['shape_lambda'] = shape_lambda
                            elif evar_alpha != 1.0:
                                extra['evar_alpha'] = evar_alpha
                            cfg = default_config_for_target(
                                target,
                                run_name=rn,
                                bin_size_ms=bs,
                                loss_func=loss,
                                time_window=win,
                                entropy_lambda=el,
                                num_epochs=num_epochs,
                                patience=patience,
                                min_epochs=min_epochs,
                                val_fraction=val_fraction,
                                track_training_history=True,
                                weight_snapshot_every=snapshot_every,
                                **extra,
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
    p.add_argument('--hidden-sizes', nargs='+', type=int, default=None,
                   help='overfitting-vs-width ablation: run the grid once per '
                        'hidden width, each isolated under run_name+"_h<H>". '
                        'Omit to use each target preset architecture.')
    p.add_argument('--flat-evar', action='store_true',
                   help='PCA-loss control: flatten the per-PC variance weights '
                        '(uniform) so the PCA loss is unweighted L2 / Brier. '
                        'Run isolated under run_name+"_flatevar".')
    p.add_argument('--shape-lambda', type=float, default=0.0,
                   help='PCA-loss width-matched term: loss = PCA + lambda*Brier '
                        '(evar floored by lambda/100). Run isolated under '
                        'run_name+"_shape<lambda>".')
    p.add_argument('--evar-alpha', type=float, default=1.0,
                   help='PCA-loss soft weighting: raise the per-PC weights to '
                        'this power and renormalise (evar -> evar**alpha). '
                        'alpha<1 compresses the dynamic range (multiplicative '
                        'cousin of --shape-lambda; alpha=0 -> flat-evar, alpha=1 '
                        '-> no-op). Run isolated under run_name+"_alpha<alpha>".')
    p.add_argument('--entropy-lambdas', nargs='+', type=float, default=None,
                   help='lambda_H sweep (2026-06-10 meeting): run the whole grid '
                        'once per entropy-penalty value, each isolated under '
                        'run_name+"_entlam<lambda>". The sharpness penalty acts on '
                        'the temporal (sampling) model only. Omit to use the single '
                        'fixed --entropy-lambda (default, unchanged behaviour).')
    a = p.parse_args()
    main(run_name=a.run_name, targets=tuple(a.targets), losses=tuple(a.losses),
         bin_sizes_ms=tuple(a.bin_sizes_ms), windows=tuple(a.windows),
         splits=tuple(a.splits), entropy_lambda=a.entropy_lambda,
         num_epochs=a.num_epochs, patience=a.patience, min_epochs=a.min_epochs,
         val_fraction=a.val_fraction, snapshot_every=a.snapshot_every,
         hidden_sizes=a.hidden_sizes, flat_evar=a.flat_evar,
         shape_lambda=a.shape_lambda, evar_alpha=a.evar_alpha,
         entropy_lambdas=a.entropy_lambdas)
