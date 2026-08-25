# -*- coding: utf-8 -*-
"""Flat-evar decision run: take the width fix to its limit, with and without a
hidden layer, and ask how much anything else matters — plus the neural-side PCA.

The 2026-07-29 meeting narrowed the programme to the projection-based loss and
its tradeoffs. Three asks, one launch:

  2. "shape lambda = infinity + linear decoding (0 hidden), and check how much
      the other params matter"
  3. "PCA on neural resp. and decode from PCs; also decode from different n of PCs"

`shape_lambda = infinity` is already a knob. The projection loss weights each PC
by its explained variance; `shape_lambda` floors those weights
(evar_k -> evar_k + lambda/100), so lambda -> infinity is the uniform-weight
limit — which is exactly `flat_evar=True` (weights replaced by 1/n_pcs). With all
PCs kept that makes the projection loss an unweighted L2 / Brier distance.

WHY THIS IS NOT A REPEAT. `flat_evar` was last run on 2026-06-03
(`brier_ctrl_flatevar`): H=32, BEFORE the 2026-07-16 restart-selection fix, and
scored on decoded *entropy* (3.24 peaky -> 4.03, landing at the calibrated line
3.95). It has never been measured at the H=8 baseline, under the fixed restart
rule, on the current pair of readouts (decoded peakiness AND chance-normalised
loss), and never with zero hidden units.

ARM R — the reference frame. Plain evar-weighted PCA at H=8 and linear, plus KL
and JS. Every flat-evar cell is read against these, so the run is self-contained
and does not depend on comparing across `prodfix_v1` / `hpsweep_v2`.

ARM A — flat_evar at the H=8 baseline, then a one-at-a-time sweep of every other
knob (width, dropout, weight_decay, lambda_H, early stopping). This is the
"how much do the other params matter" half. Prediction on file: once the
weighting is flat they should matter much LESS than they did under evar
weighting, because the over-sharpening was never a capacity/regularisation
phenomenon (see `prodfix_v1` arm C).

ARM B — the same sweep with `hidden_sizes=[]` (multinomial logistic regression).
`B_flat_linear` is THE decisive cell of the run: `prodfix_v1` arm C showed the
projection loss over-sharpens 5.6x/10.5x with zero hidden units, i.e. the bias
survives deleting all capacity. If the evar weighting is the cause, then removing
the weighting must fix it even with no hidden layer. If `B_flat_linear` still
over-sharpens, the loss-geometry account has a problem.

ARM C — neural-side PCA (`n_neural_pcs`, new 2026-07-29): project the INPUT
activity onto its leading k PCs, fit on training trials only, and decode from
those. Ladder k in {2,4,8,16,32,64} at H=8 and linear. Note this is the input
side and is unrelated to `pca_basis`, which is the target-posterior basis.

JUDGE EVERY CELL TWICE — decoded peakiness against the IO target AND normalised
loss against the leave-one-out predict-mean. A manipulation that lands peakiness
without beating chance is a lobotomy, not a cure (weight_decay=0.01 is the
standing cautionary case: peakiness 0.011 = 1/91 = uniform, skill stuck at 1.59).

Everything else = the prodfix_v1 / hpsweep_v2 baseline: Q, half, 100 ms, H=8,
tanh, dropout 0, wd 1e-4, lambda_H 3e-3, patience 0 + monitor_val,
val_fraction 0.2, REP 5, 200 epochs, 6 mice, stratified_balanced,
`restart_selection='val'` and `seed=0` throughout.

Run on gpu1 (see CLUSTER_RUNBOOK / the launch block in the session log):
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_flatevar_v1.py --dry-run          # prints the plan, runs nothing
    $PY run_flatevar_v1.py --smoke            # 1 mouse, 2 epochs, the 4 key cells
    $PY -u run_flatevar_v1.py 2>&1 | tee flatevar.log
Idempotent: per-mouse shards resume, so a re-run continues where it stopped.
Trim with --arms / --only if the full grid is too long.
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config

RUN_ROOT = 'flatevar_v1'
SPLITS = ('stratified_balanced',)
BASE = dict(target='Q', window='half', bin_ms=100, width=8, act='tanh',
            dropout=0.0, wd=1e-4, lam=3e-3, epochs=200, rep=5,
            patience=0, val_fraction=0.2, seed=0)

FLAT = dict(flat_evar=True)
LINEAR = dict(hidden_sizes=[])

# (cell_name, loss, extra Config overrides, what this cell COMPLETES)
#
# ORDERED BY DECISION VALUE, not by arm. The run is long and may be stopped
# early, so every PREFIX should answer a complete question — a cell is only
# interpretable once its comparison partner exists.

# Cells 1-6: the whole thesis of the run. Stop here and you still have an answer.
CELLS_CORE = [
    ('R_evar_base',    'PCA', {},
     'reference: evar-weighted projection loss at H=8 (the broken baseline)'),
    ('A_flat_base',    'PCA', dict(FLAT),
     'does flat weighting (shape_lambda -> inf) fix peakiness AND beat chance?'),
    ('R_evar_linear',  'PCA', dict(LINEAR),
     'reference: the prodfix arm-C result re-run here (over-sharpens ~5.6x/10.5x)'),
    ('B_flat_linear',  'PCA', {**FLAT, **LINEAR},
     'THE DECISIVE CELL: flat weighting with ZERO hidden units — if the weighting '
     'is the cause, this must land on target'),
    ('R_reference_kl', 'KL',  {},
     'calibrated reference: what "on target and beating chance" looks like'),
    ('R_reference_js', 'JS',  {},
     'the other calibrated generalist (JS >= KL spatially in prodfix_v1)'),
]

# Cells 7-12: the new question — decode from neural PCs, and how many are needed.
CELLS_NEURAL_PCA = [
    ('C_npc2',  'PCA', {**FLAT, 'n_neural_pcs': 2},  'neural-PC ladder: k=2'),
    ('C_npc4',  'PCA', {**FLAT, 'n_neural_pcs': 4},  'neural-PC ladder: k=4'),
    ('C_npc8',  'PCA', {**FLAT, 'n_neural_pcs': 8},  'neural-PC ladder: k=8'),
    ('C_npc16', 'PCA', {**FLAT, 'n_neural_pcs': 16}, 'neural-PC ladder: k=16'),
    ('C_npc32', 'PCA', {**FLAT, 'n_neural_pcs': 32}, 'neural-PC ladder: k=32'),
    ('C_npc64', 'PCA', {**FLAT, 'n_neural_pcs': 64},
     'neural-PC ladder: k=64 (near full rank for most mice — the plateau check)'),
    # The linear leg: with no hidden layer the decoder IS a linear map from PC
    # scores, so this is the most interpretable version of the ask.
    ('C_lin_npc4',  'PCA', {**FLAT, **LINEAR, 'n_neural_pcs': 4},
     'linear decoder from 4 neural PCs'),
    ('C_lin_npc16', 'PCA', {**FLAT, **LINEAR, 'n_neural_pcs': 16},
     'linear decoder from 16 neural PCs'),
    ('C_lin_npc64', 'PCA', {**FLAT, **LINEAR, 'n_neural_pcs': 64},
     'linear decoder from 64 neural PCs'),
    # One evar-weighted control so the ladder is not confounded with flat_evar:
    # does input-side PCA change the over-sharpening on its own?
    ('C_evar_npc16', 'PCA', {'n_neural_pcs': 16},
     'CONTROL: neural PCA with the evar weighting still on — does input-side '
     'PCA alone change anything?'),
]

# Cells 17-25: "how much do the other params matter" at H=8, one axis at a time.
CELLS_SWEEP_H8 = [
    ('A_flat_h2',        'PCA', {**FLAT, 'hidden_sizes': [2]},   'width ladder: H=2'),
    ('A_flat_h4',        'PCA', {**FLAT, 'hidden_sizes': [4]},   'width ladder: H=4'),
    ('A_flat_h16',       'PCA', {**FLAT, 'hidden_sizes': [16]},  'width ladder: H=16'),
    ('A_flat_h32',       'PCA', {**FLAT, 'hidden_sizes': [32]},  'width ladder: H=32'),
    ('A_flat_drop0p25',  'PCA', {**FLAT, 'dropout': 0.25},       'dropout 0.25'),
    ('A_flat_drop0p5',   'PCA', {**FLAT, 'dropout': 0.5},        'dropout 0.5'),
    ('A_flat_wd0',       'PCA', {**FLAT, 'weight_decay': 0.0},   'weight decay 0'),
    ('A_flat_wd1em3',    'PCA', {**FLAT, 'weight_decay': 1e-3},  'weight decay 1e-3'),
    ('A_flat_wd1em2',    'PCA', {**FLAT, 'weight_decay': 1e-2},
     'weight decay 1e-2 — the known lobotomy dose, now under flat weighting'),
    ('A_flat_lam0',      'PCA', {**FLAT, 'entropy_lambda': 0.0}, 'lambda_H 0'),
    ('A_flat_lam1em2',   'PCA', {**FLAT, 'entropy_lambda': 1e-2}, 'lambda_H 1e-2'),
    ('A_flat_earlystop', 'PCA', {**FLAT, 'patience': 20, 'min_epochs': 20},
     'early stopping — the only knob that bit under evar weighting'),
]

# Cells 26-33: the same sweep with no hidden layer (no width axis to sweep).
CELLS_SWEEP_LINEAR = [
    ('B_flat_lin_drop0p25', 'PCA', {**FLAT, **LINEAR, 'dropout': 0.25},
     'linear + dropout 0.25'),
    ('B_flat_lin_drop0p5',  'PCA', {**FLAT, **LINEAR, 'dropout': 0.5},
     'linear + dropout 0.5'),
    ('B_flat_lin_wd0',      'PCA', {**FLAT, **LINEAR, 'weight_decay': 0.0},
     'linear + weight decay 0'),
    ('B_flat_lin_wd1em3',   'PCA', {**FLAT, **LINEAR, 'weight_decay': 1e-3},
     'linear + weight decay 1e-3'),
    ('B_flat_lin_wd1em2',   'PCA', {**FLAT, **LINEAR, 'weight_decay': 1e-2},
     'linear + weight decay 1e-2 (the lobotomy dose)'),
    ('B_flat_lin_lam0',     'PCA', {**FLAT, **LINEAR, 'entropy_lambda': 0.0},
     'linear + lambda_H 0'),
    ('B_flat_lin_lam1em2',  'PCA', {**FLAT, **LINEAR, 'entropy_lambda': 1e-2},
     'linear + lambda_H 1e-2'),
    ('B_flat_lin_earlystop', 'PCA',
     {**FLAT, **LINEAR, 'patience': 20, 'min_epochs': 20},
     'linear + early stopping'),
]

ARMS = {
    'core':   CELLS_CORE,
    'neural': CELLS_NEURAL_PCA,
    'sweep':  CELLS_SWEEP_H8,
    'linear': CELLS_SWEEP_LINEAR,
}
DEFAULT_ARMS = ('core', 'neural', 'sweep', 'linear')


def build(cell, loss, extra, root=RUN_ROOT):
    kw = dict(
        run_name=f'{root}/{cell}', bin_size_ms=BASE['bin_ms'],
        loss_func=loss, time_window=BASE['window'],
        hidden_sizes=[BASE['width']], activation_function=BASE['act'],
        dropout=BASE['dropout'], weight_decay=BASE['wd'],
        entropy_lambda=BASE['lam'], num_epochs=BASE['epochs'], REP=BASE['rep'],
        patience=BASE['patience'], min_epochs=0, val_fraction=BASE['val_fraction'],
        monitor_val=True, restart_selection='val', seed=BASE['seed'],
        track_training_history=True, weight_snapshot_every=10,
    )
    kw.update(extra)          # per-cell overrides win (e.g. hidden_sizes=[])
    # Weight snapshots are indexed by 'layers.1.weight' downstream, which does
    # not exist without a hidden layer — skip them for the linear cells.
    # (Same guard as run_prodfix_v1.)
    if not kw['hidden_sizes']:
        kw['weight_snapshot_every'] = 0
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true',
                   help='print the plan and exit without training')
    p.add_argument('--smoke', action='store_true',
                   help='1 mouse, 2 epochs, REP 1, core cells only — a wiring check')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None, help='cell names to run')
    p.add_argument('--arms', nargs='+', default=list(DEFAULT_ARMS),
                   choices=list(ARMS), help='which arms to run (default: all)')
    a = p.parse_args()

    arms = [x for x in DEFAULT_ARMS if x in set(a.arms)]   # keep canonical order
    cells = [c for arm in arms for c in ARMS[arm]]
    # --only wins over --smoke's core-only restriction, so any single cell can be
    # smoke-tested (e.g. --smoke --only C_npc4 to exercise the neural-PCA path).
    if a.only:
        cells = [c for c in cells if c[0] in set(a.only)]
    elif a.smoke:
        cells = list(CELLS_CORE)

    epochs, rep = (2, 1) if a.smoke else (BASE['epochs'], BASE['rep'])
    mice = [0] if a.smoke else (range(6) if a.mouse_ids is None else a.mouse_ids)
    n_mice = len(list(mice)) if not isinstance(mice, range) else len(mice)
    # Smoke runs write to their OWN tree. Per-mouse shards are resumable, so a
    # 2-epoch smoke shard sitting in the real tree would be silently SKIPPED by
    # the production run — mouse 0 would keep its garbage fit. Separate roots
    # make that impossible rather than merely documented.
    root = f'{RUN_ROOT}_smoke' if a.smoke else RUN_ROOT

    print(f"Flat-evar decision run: {root}"
          + ("   [SMOKE — 1 mouse, 2 epochs, REP 1]" if a.smoke else ""))
    print(f"  fixed : Q half 100ms H={BASE['width']} tanh dropout 0 wd {BASE['wd']} "
          f"lambda_H {BASE['lam']} patience 0 + monitor_val REP {rep} "
          f"{epochs} ep, {n_mice} mice")
    print(f"  rule  : restart_selection='val' (2026-07-16 fix), seed={BASE['seed']}")
    print(f"  arms  : {', '.join(arms)}")
    print(f"  cells : {len(cells)}   fits: {len(cells) * n_mice * 4 * rep} net "
          f"trainings ({n_mice} mice x 4 archs x REP)")
    print("  ORDER = decision value: every prefix answers a complete question, so "
          "stopping early still yields usable answers.\n")
    for i, (cell, loss, _extra, why) in enumerate(cells, 1):
        print(f"    {i:2d}. {cell:22s} {loss:5s} {why}")
    print("\n  JUDGE EVERY CELL TWICE — decoded peakiness vs the IO target AND "
          "normalised loss vs leave-one-out predict-mean. Landing peakiness "
          "without beating chance is a lobotomy, not a cure.")
    print("  The single decisive cell is B_flat_linear: flat weighting + zero "
          "hidden units. If it still over-sharpens, the loss-geometry account "
          "is in trouble.")
    if a.dry_run:
        return

    for i, (cell, loss, extra, why) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell}  loss={loss}  {extra}\n    -> {why}")
        cfg_extra = dict(extra)
        if a.smoke:
            cfg_extra.update(num_epochs=2, REP=1)
        run_config(build(cell, loss, cfg_extra, root=root),
                   splits=SPLITS, mouse_ids=mice)


if __name__ == '__main__':
    main()
