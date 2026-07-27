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

# (cell_name, loss, extra Config overrides)
CELLS = [
    # --- Arm A: the two candidate production fixes, matched ---------------
    ('A_baseline_pca',      'PCA', {}),
    ('A_shape3',            'PCA', dict(shape_lambda=3.0)),
    ('A_shape10',           'PCA', dict(shape_lambda=10.0)),
    ('A_shape30',           'PCA', dict(shape_lambda=30.0)),
    ('A_smooth0p1',         'PCA', dict(smooth_lambda=0.1)),
    ('A_smooth0p3',         'PCA', dict(smooth_lambda=0.3)),
    ('A_smooth1',           'PCA', dict(smooth_lambda=1.0)),
    ('A_reference_kl',      'KL',  {}),
    # --- Arm B: residual basis = trial-level signal only -------------------
    ('B_residual_pca',      'PCA', dict(pca_basis='residual')),
    ('B_residual_kl',       'KL',  dict(pca_basis='residual')),
    ('B_alltrials_kl',      'KL',  {}),          # == A_reference_kl, deduped below
    # --- Arm C: no hidden layer (2026-06-18 meeting item #6) ---------------
    # hidden_sizes=[] -> a single linear map input->output (multinomial logistic
    # regression after the softmax). The peakiness mechanism attributes the
    # sharpening drift to the softmax Jacobian and the shared weights, NOT to
    # capacity, so the prediction is that projection-based STILL over-sharpens
    # here while KL stays calibrated. If instead it lands on the IO target, the
    # over-sharpening is capacity-driven after all and the bias account is wrong.
    # Zero hidden units is also the extreme left end of the params-per-trial axis
    # (~0.3 vs 3.6 at H=8), so it doubles as the capacity story's end point.
    ('C_nohidden_pca',      'PCA', dict(hidden_sizes=[])),
    ('C_nohidden_kl',       'KL',  dict(hidden_sizes=[])),
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
    a = p.parse_args()

    cells = [c for c in CELLS if c[0] != 'B_alltrials_kl']    # dedup: == A_reference_kl
    if a.only:
        cells = [c for c in cells if c[0] in set(a.only)]
    print(f"Production-fix decision run: {RUN_ROOT}")
    print(f"  fixed : Q half 100ms H={BASE['width']} tanh dropout 0 wd {BASE['wd']} "
          f"lambda_H {BASE['lam']} patience 0 + monitor_val REP {BASE['rep']} "
          f"{BASE['epochs']} ep, 6 mice")
    print(f"  rule  : restart_selection='val' (2026-07-16 fix), seed={BASE['seed']}")
    print(f"  cells : {len(cells)}  -> {[c[0] for c in cells]}")
    print(f"  fits  : {len(cells) * 6 * 4 * BASE['rep']} net trainings "
          f"(6 mice x 4 archs x REP)")
    print("  NB Arm A decides the production fix (peakiness AND normalised loss — a fix "
          "that lands peakiness without beating chance is a lobotomy, cf. weight_decay).")
    print("     Arm B tests whether any trial-level signal survives the condition mean.")
    print("     Arm C (no hidden layer) separates the bias account of over-sharpening "
          "from the capacity account.")
    if a.dry_run:
        return
    mice = range(6) if a.mouse_ids is None else a.mouse_ids
    for i, (cell, loss, extra) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell}  loss={loss}  {extra}")
        run_config(build(cell, loss, extra), splits=SPLITS, mouse_ids=mice)


if __name__ == '__main__':
    main()
