# -*- coding: utf-8 -*-
"""Can we get PEAKY TIME BINS with a CALIBRATED trial average? (fast probe, < 1 h)

THE TARGET STATE. A sampling code predicts that each individual time bin is a sharp commitment --
a sample -- while the average over bins recovers the broad ideal-observer posterior. At the current
projection-loss winner (shape_lambda = 30, i.e. lambda_Brier = 0.3) we have NEITHER: measured
per-bin peakiness 0.043 vs trial-average 0.037 vs IO target 0.037. The bins are as broad as their
own average, so the temporal decoder is emitting essentially the same posterior every bin rather
than sampling.

THE TWO KNOBS, AND WHY THEY SHOULD COMBINE. They act on different objects:
  * shape_lambda floors the per-PC weights of the projection loss, which is computed on the
    TIME-AVERAGED posterior -> it pins the AVERAGE to the IO target.
  * entropy_lambda adds +lambda_H * mean_t H(p_t), evaluated on the INDIVIDUAL per-bin posteriors
    before averaging, and applies to the temporal arch only (nn_classifier.py:431) -> minimising it
    SHARPENS THE BINS.
Raising lambda_H alone is known to wreck the temporal decoder (it drove trial-average peakiness
0.34 -> 0.79 and normalised loss to ~18). The untested question is whether it behaves differently
once shape_lambda is holding the average in place. hpsweep_v2 is one-at-a-time, so it swept
lambda_H at shape_lambda = 0 and shape_lambda at lambda_H = 3e-3 -- never the pair.

SPEED. Budgeted for under an hour: early stopping (patience 10, min_epochs 20) is both the
regulariser asked for and the main time lever, and REP is cut 5 -> 3. Ordered so the most
informative cells run first; killing it early still leaves a usable lambda_H curve.

JUDGE ON three things together, not one:
  1. PER-BIN peakiness  -- should RISE above the IO target (that is the point)
  2. TRIAL-AVERAGE peakiness -- should STAY at the IO target 0.05943
  3. normalised loss under BOTH the projection metric and KL -- must stay below chance
A cell that sharpens the bins by wrecking the average is not a success.

Run on gpu1:
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_shapefix_v1.py --dry-run
    $PY -u run_shapefix_v1.py 2>&1 | tee shapefix.log
Idempotent: per-mouse shards resume.
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config

RUN_ROOT = 'shapefix_v1'
SPLITS = ('stratified_balanced',)
SHAPE = 30.0                 # lambda_Brier = 0.3 — pins the trial-AVERAGE to the IO target
BASE = dict(target='Q', window='half', bin_ms=100, width=8, act='tanh',
            wd=1e-4, epochs=200, rep=3, patience=10, min_epochs=20,
            val_fraction=0.2, seed=0)

# (cell, overrides, what it asks) — ordered by decision value.
CELLS = [
    ('lamH0p003_drop0',  dict(entropy_lambda=3e-3, dropout=0.0),
     'reference: current lambda_H, now with early stopping'),
    ('lamH0p03_drop0',   dict(entropy_lambda=3e-2, dropout=0.0),
     '10x lambda_H — do the BINS sharpen while the average holds?'),
    ('lamH0p1_drop0',    dict(entropy_lambda=1e-1, dropout=0.0),
     'strongest lambda_H — how far can bins sharpen before the average breaks?'),
    ('lamH0_drop0',      dict(entropy_lambda=0.0, dropout=0.0),
     'lower bound: no bin-sharpening pressure at all'),
    ('lamH0p01_drop0',   dict(entropy_lambda=1e-2, dropout=0.0),
     'fills the curve between 3e-3 and 3e-2'),
    ('lamH0p03_drop0p5', dict(entropy_lambda=3e-2, dropout=0.5),
     'add a second regulariser at the promising lambda_H'),
]


def build(cell, extra):
    kw = dict(run_name=f'{RUN_ROOT}/{cell}', bin_size_ms=BASE['bin_ms'],
              loss_func='PCA', time_window=BASE['window'],
              hidden_sizes=[BASE['width']], activation_function=BASE['act'],
              shape_lambda=SHAPE, weight_decay=BASE['wd'],
              num_epochs=BASE['epochs'], REP=BASE['rep'],
              patience=BASE['patience'], min_epochs=BASE['min_epochs'],
              val_fraction=BASE['val_fraction'], monitor_val=True,
              restart_selection='val', seed=BASE['seed'],
              track_training_history=True, weight_snapshot_every=0)
    kw.update(extra)
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None)
    a = p.parse_args()

    cells = [c for c in CELLS if not a.only or c[0] in set(a.only)]
    print(f'Peaky-bins probe: {RUN_ROOT}   (shape_lambda = {SHAPE:g}, i.e. lambda_Brier = 0.3)')
    print(f"  fixed : Q half 100ms H={BASE['width']} tanh wd {BASE['wd']} "
          f"patience {BASE['patience']} REP {BASE['rep']} cap {BASE['epochs']} ep, 6 mice, "
          f"restart_selection=val, seed {BASE['seed']}")
    print(f'  cells : {len(cells)}   fits: {len(cells) * 6 * 4 * BASE["rep"]} net trainings '
          f'(early-stopped, REP {BASE["rep"]} — budgeted < 1 h)\n')
    for i, (cell, _e, why) in enumerate(cells, 1):
        print(f'    {i}. {cell:20s} {why}')
    print('\n  Judge on ALL THREE: per-bin peakiness should RISE, trial-average peakiness should '
          'STAY at 0.05943, and normalised loss (projection AND KL) must stay below chance.')
    if a.dry_run:
        return
    mice = range(6) if a.mouse_ids is None else a.mouse_ids
    for i, (cell, extra, why) in enumerate(cells, 1):
        print(f'\n[{i}/{len(cells)}] {cell}  {extra}\n    -> {why}')
        run_config(build(cell, extra), splits=SPLITS, mouse_ids=mice)


if __name__ == '__main__':
    main()
