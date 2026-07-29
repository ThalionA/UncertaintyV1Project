# -*- coding: utf-8 -*-
"""Can the projection loss do BETTER at shape_lambda = 0.3 with regularisation?

WHY THIS RUN EXISTS. hpsweep_v2 is a one-at-a-time design: every axis was swept with all other
knobs held at the baseline. So the winning projection cell, shape_lambda = 30 (lambda_Brier = 0.3),
was trained at dropout 0, patience 0, H = 8, tanh, weight_decay 1e-4, lambda_H 3e-3 — it has NEVER
been combined with any regulariser. Its overfitting is 3.5 (spatial) / 2.2 (temporal), while the
best KL cell reached 1.8 / 1.9 using dropout 0.75 + patience 10 and improved its normalised loss at
the same time. So the obvious question is whether the same regularisation improves the projection
loss once the width term has already fixed its calibration.

DESIGN. A small factorial around the winner, holding shape_lambda = 30:
    dropout  in {0, 0.25, 0.5, 0.75}
    patience in {0, 10}
= 8 cells, of which (dropout 0, patience 0) already exists in hpsweep_v2 and is re-run here only so
that every cell in this run shares one restart rule and one seed. Plus two probes:
    shape_lambda = 100 at baseline  -- does yet more Brier help, or has it saturated?
    shape_lambda = 30, H = 16       -- does the width term change the capacity optimum?

Everything else matches hpsweep_v2: Q / half / 100 ms, tanh, weight_decay 1e-4, lambda_H 3e-3,
REP 5, 200 epochs, monitor_val, val_fraction 0.2, 6 mice, stratified_balanced,
restart_selection='val' (the 2026-07-16 fix) and seed 0 throughout.

JUDGE ON: decoded peakiness against the IO target 0.05943, AND normalised loss under BOTH the
projection metric and KL. A cell that lowers the projection-metric loss while letting peakiness
drift is not an improvement -- that metric is blind to width.

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
BASE = dict(target='Q', window='half', bin_ms=100, width=8, act='tanh',
            wd=1e-4, lam=3e-3, epochs=200, rep=5, val_fraction=0.2, seed=0)
SHAPE = 30.0                       # lambda_Brier = 0.3, the projection-loss winner

# (cell, overrides, what it asks)
CELLS = []
for pat in (0, 10):
    for drop in (0.0, 0.25, 0.5, 0.75):
        CELLS.append((f'shape30_drop{drop:g}_pat{pat}'.replace('.', 'p'),
                      dict(shape_lambda=SHAPE, dropout=drop, patience=pat,
                           min_epochs=20 if pat > 0 else 0),
                      f'shape 0.3 + dropout {drop:g} + patience {pat}'))
CELLS += [
    ('shape100_baseline', dict(shape_lambda=100.0),
     'has the Brier term saturated, or does more help?'),
    ('shape30_h16', dict(shape_lambda=SHAPE, hidden_sizes=[16]),
     'does the width term move the capacity optimum?'),
]


def build(cell, extra):
    kw = dict(run_name=f'{RUN_ROOT}/{cell}', bin_size_ms=BASE['bin_ms'],
              loss_func='PCA', time_window=BASE['window'],
              hidden_sizes=[BASE['width']], activation_function=BASE['act'],
              dropout=0.0, weight_decay=BASE['wd'], entropy_lambda=BASE['lam'],
              num_epochs=BASE['epochs'], REP=BASE['rep'], patience=0, min_epochs=0,
              val_fraction=BASE['val_fraction'], monitor_val=True,
              restart_selection='val', seed=BASE['seed'],
              track_training_history=True, weight_snapshot_every=10)
    kw.update(extra)
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None)
    a = p.parse_args()

    cells = [c for c in CELLS if not a.only or c[0] in set(a.only)]
    print(f'Projection-loss regularisation probe: {RUN_ROOT}')
    print(f"  fixed : Q half 100ms tanh wd {BASE['wd']} lambda_H {BASE['lam']} REP {BASE['rep']} "
          f"{BASE['epochs']} ep, 6 mice, restart_selection=val, seed {BASE['seed']}")
    print(f'  cells : {len(cells)}   fits: {len(cells) * 6 * 4 * BASE["rep"]} net trainings\n')
    for i, (cell, _e, why) in enumerate(cells, 1):
        print(f'    {i:2d}. {cell:26s} {why}')
    print('\n  Judge on peakiness vs the IO target 0.05943 AND normalised loss under BOTH the '
          'projection metric and KL.')
    if a.dry_run:
        return
    mice = range(6) if a.mouse_ids is None else a.mouse_ids
    for i, (cell, extra, why) in enumerate(cells, 1):
        print(f'\n[{i}/{len(cells)}] {cell}  {extra}\n    -> {why}')
        run_config(build(cell, extra), splits=SPLITS, mouse_ids=mice)


if __name__ == '__main__':
    main()
