# -*- coding: utf-8 -*-
"""Production-fix decision run: shape_lambda vs smooth_lambda, matched — plus the
residual-basis arm (does V1 carry uncertainty BEYOND stimulus condition?).

Two standing items, one launch. Both arms are internally matched (each ships its own
baseline cell), so nothing here depends on comparing against `hpsweep_v2`, which was
trained before the 2026-07-16 restart-selection fix.

ARM A — "which over-sharpening fix do we adopt?" (last residual of the 2026-06-10
follow-ups; blocks "decide the production loss"). Both known fixes land peakiness on the
IO target, but they have never been compared in the same regime: `smoothsweep_*` was
H=32 + early-stopping on the loss_comparison_v1 baseline, while hpsweep_v2's shape cells
were H=8 + patience-0. Here they run side by side at the v2 baseline:
    shape_lambda in {3, 10, 30}   (PCA + (shape/100)*Brier — floors the evar weights)
    smooth_lambda in {0.1, 0.3, 1.0}   (Dirichlet energy of the decoded posterior)
    + a shared PCA baseline and a KL cell as the calibrated reference.
Decide on: decoded peakiness vs the IO target, AND normalised loss under KL (a fix that
lands peakiness but does not beat chance is a `weight_decay`-style lobotomy, not a cure).

ARM B — "is there any trial-specific uncertainty signal at all?" (audit E1, the highest-
value open question). The DeepSets analysis found no model beats its within-condition
null while a condition-mean oracle is ~20x better than any decoder, which raises the
possibility that the decoding programme is largely measuring CONDITION decoding. The
`residual` PCA basis (fit on target - condition_mean) scores within-condition, trial-level
structure only. If residual-basis decoders collapse to chance, that is the answer.
    pca_basis='residual' x {PCA, KL}, plus the matching all_trials cells as the contrast.

Everything else = hpsweep_v2 baseline: Q, half, 100 ms, H=8, tanh, dropout 0, wd 1e-4,
lambda_H 3e-3, patience 0 + monitor_val, val_fraction 0.2, REP 5, 200 epochs, 6 mice,
stratified_balanced. `restart_selection='val'` (the fix) and `seed=0` throughout, so the
whole run is reproducible and internally consistent.

Run on gpu1 (see CLUSTER_RUNBOOK / the launch block in the session log):
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_prodfix_v1.py --dry-run     # prints the plan
    $PY -u run_prodfix_v1.py 2>&1 | tee prodfix.log
Idempotent: per-mouse shards resume, so a re-run continues where it stopped.
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config

RUN_ROOT = 'prodfix_v1'
SPLITS = ('stratified_balanced',)
BASE = dict(target='Q', window='half', bin_ms=100, width=8, act='tanh',
            dropout=0.0, wd=1e-4, lam=3e-3, epochs=200, rep=5,
            patience=0, val_fraction=0.2, seed=0)

# (cell_name, loss, extra Config overrides, what this cell COMPLETES)
#
# ORDERED BY DECISION VALUE, not by arm. The run is long and may be stopped early,
# so every PREFIX should answer a complete question — a cell is only interpretable
# once its comparison partner exists. Projection-based (the loss under suspicion)
# leads; the dose-response ladder points that merely refine an already-answered
# question come last. Wasserstein is not in the default set at all (it is a second
# over-sharpening loss, not a candidate fix); --with-wasserstein appends it last.
CELLS = [
    # 1-3: the production-fix decision — the single most valuable prefix.
    ('A_baseline_pca',  'PCA', {},
     'the reference point every Arm-A cell is measured against'),
    ('A_shape10',       'PCA', dict(shape_lambda=10.0),
     'does the incumbent fix (lambda_Brier=0.1) reproduce at the H=8 baseline?'),
    ('A_smooth0p3',     'PCA', dict(smooth_lambda=0.3),
     'HEAD-TO-HEAD DECIDABLE: shape vs smooth at their known operating points'),
    # 4: bias vs capacity — interpretable against the IO target + cell 1.
    ('C_nohidden_pca',  'PCA', dict(hidden_sizes=[]),
     'does projection-based still over-sharpen with ZERO hidden units? (bias vs capacity)'),
    # 5-6: the two calibrated generalists, paired. JS is here as a live PRODUCTION-LOSS
    # CANDIDATE, not just a replication: at the hpsweep_v2 baseline it matches KL's
    # calibration exactly (peakiness 0.059/0.051 spat/temp, both on the IO target) while
    # overfitting 2.7x LESS (val/train 10.5 vs 27.9 spatial, 2.7 vs 3.8 temporal) and
    # scoring BETTER on held-out normalised loss (0.82 vs 0.92 spatial — KL-spatial sits
    # only marginally below chance). If that holds here, JS dominates KL and the standing
    # "decide the production loss" thread has an answer.
    ('A_reference_kl',  'KL',  {},
     'what a calibrated decoder scores in this exact regime'),
    ('A_reference_js',  'JS',  {},
     'KL-vs-JS DECIDABLE: does JS keep the calibration with less overfitting?'),
    # 7-8: Arm C completed across BOTH calibrated losses.
    ('C_nohidden_kl',   'KL',  dict(hidden_sizes=[]),
     'does KL stay calibrated without a hidden layer?'),
    ('C_nohidden_js',   'JS',  dict(hidden_sizes=[]),
     'ARM C COMPLETE: bias-vs-capacity replicated on a second calibrated loss'),
    # 9-10: the framing question.
    ('B_residual_pca',  'PCA', dict(pca_basis='residual'),
     'is there trial-level signal beyond the condition mean? (projection)'),
    ('B_residual_kl',   'KL',  dict(pca_basis='residual'),
     'ARM B COMPLETE: same question under the calibrated loss'),
    # 9-12: dose-response ladders. These refine an answer rather than provide one.
    ('A_shape3',        'PCA', dict(shape_lambda=3.0),   'shape ladder (below the sweet spot)'),
    ('A_shape30',       'PCA', dict(shape_lambda=30.0),  'shape ladder (above)'),
    ('A_smooth0p1',     'PCA', dict(smooth_lambda=0.1),  'smooth ladder (below)'),
    ('A_smooth1',       'PCA', dict(smooth_lambda=1.0),  'smooth ladder (above)'),
]

# Optional, appended last (--with-wasserstein): Wasserstein is the OTHER loss that
# over-sharpens, so a linear Wasserstein decoder is a second, independent test of
# the bias-vs-capacity account. It is not a candidate production fix, which is why
# it is not in the default set.
WASSERSTEIN_CELLS = [
    ('D_reference_wass', 'Wasserstein', {},
     'Wasserstein baseline in this regime'),
    ('D_nohidden_wass',  'Wasserstein', dict(hidden_sizes=[]),
     'does the OTHER over-sharpening loss also over-sharpen with zero hidden units?'),
]


def build(cell, loss, extra):
    kw = dict(
        run_name=f'{RUN_ROOT}/{cell}', bin_size_ms=BASE['bin_ms'],
        loss_func=loss, time_window=BASE['window'],
        hidden_sizes=[BASE['width']], activation_function=BASE['act'],
        dropout=BASE['dropout'], weight_decay=BASE['wd'],
        entropy_lambda=BASE['lam'], num_epochs=BASE['epochs'], REP=BASE['rep'],
        patience=BASE['patience'], min_epochs=0, val_fraction=BASE['val_fraction'],
        monitor_val=True, restart_selection='val', seed=BASE['seed'],
        track_training_history=True, weight_snapshot_every=10,
    )
    kw.update(extra)          # per-cell overrides win (e.g. hidden_sizes=[])
    # Weight snapshots are indexed by 'layers.1.weight' downstream, which does not
    # exist without a hidden layer — skip them for the linear cells.
    if not kw['hidden_sizes']:
        kw['weight_snapshot_every'] = 0
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None, help='cell names to run')
    p.add_argument('--with-wasserstein', action='store_true',
                   help='append the Wasserstein cells LAST (a second over-sharpening '
                        'loss for the bias-vs-capacity test; not a candidate fix)')
    a = p.parse_args()

    cells = list(CELLS) + (WASSERSTEIN_CELLS if a.with_wasserstein else [])
    if a.only:
        cells = [c for c in cells if c[0] in set(a.only)]
    print(f"Production-fix decision run: {RUN_ROOT}")
    print(f"  fixed : Q half 100ms H={BASE['width']} tanh dropout 0 wd {BASE['wd']} "
          f"lambda_H {BASE['lam']} patience 0 + monitor_val REP {BASE['rep']} "
          f"{BASE['epochs']} ep, 6 mice")
    print(f"  rule  : restart_selection='val' (2026-07-16 fix), seed={BASE['seed']}")
    print(f"  cells : {len(cells)}   fits: {len(cells) * 6 * 4 * BASE['rep']} net "
          f"trainings (6 mice x 4 archs x REP)")
    print("  ORDER = decision value: every prefix answers a complete question, so "
          "stopping early still yields usable answers.\n")
    for i, (cell, loss, _extra, why) in enumerate(cells, 1):
        print(f"    {i:2d}. {cell:18s} {loss:12s} {why}")
    print("\n  Judge Arm A on peakiness AND normalised loss — a fix that lands peakiness "
          "without beating chance is a lobotomy (cf. weight_decay).")
    if not a.with_wasserstein:
        print("  Wasserstein is NOT included (it is a second over-sharpening loss, not a "
              "candidate fix); pass --with-wasserstein to append it last.")
    if a.dry_run:
        return
    mice = range(6) if a.mouse_ids is None else a.mouse_ids
    for i, (cell, loss, extra, why) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell}  loss={loss}  {extra}\n    -> {why}")
        run_config(build(cell, loss, extra), splits=SPLITS, mouse_ids=mice)


if __name__ == '__main__':
    main()
