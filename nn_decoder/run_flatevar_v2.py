# -*- coding: utf-8 -*-
"""Flat-evar, take 2 — scale-matched. Re-runs `flatevar_v1` with the weight decay
that annihilated it removed, and adds the cells v1 could not contain.

WHY V1 IS VOID. Flat weighting puts 1/91 ~ 0.011 on every PC where the evar
weighting puts ~0.5 on the leading location PCs, so the fit-loss gradient is ~45x
weaker. Adam's decay term `wd*theta` is unchanged, so at v1's baseline
`weight_decay=1e-4` the decay simply won: input weights went to EXACTLY zero
(||W_in|| = 0.0000 vs 3.29 for the evar baseline) and the decoder emitted the
uniform posterior (peakiness 1/91, normalised loss ~1.55). 30 of 36 v1 cells are
affected, including the entire neural-PC ladder. The tell was peakiness identical
to 4 d.p. across widths 2-32 and dropout 0-0.5.

This is the mirror of the documented `shape_lambda` trap: shape_lambda inflates
the loss (sum(evar) = 1 + n*lambda) and DILUTES wd up to 28x; flat_evar deflates
the gradient and AMPLIFIES it. Both are the same lesson — a change to the loss
weighting silently re-tunes every knob measured in gradient units.

WHAT SURVIVED V1 (the `*_wd0` cells), and what it says:
  * flat + wd=0, H=8, SPATIAL: peakiness 1.02x target, normalised loss 0.948.
    Flat weighting genuinely fixes spatial over-sharpening (evar: 6.14x / 2.388)
    and crosses below chance.
  * flat + wd=0, TEMPORAL: 6.41x (H=8) / 7.44x (linear), and the normalised loss
    gets WORSE, 8.18 -> 18.23. Flat weighting does NOT fix the temporal decoder.
  * input-side PCA alone does not fix over-sharpening (`C_evar_npc16`: 4.5x target).

THE ONE CELL V1 COULD NOT CONTAIN, AND THE POINT OF THIS RUN. `entropy_lambda`
(lambda_H) is temporal-only and is ALSO unscaled against a 45x-weaker fit
gradient, so under flat weighting it is relatively ~45x stronger — and it
sharpens (v1's `A_flat_lam1em2` temporal reached 14.6x target). So the temporal
failure above has two candidate causes that v1 cannot separate, because a
one-at-a-time sweep only varies one axis from a baseline that was itself broken.
`F_flat_lam0` (flat + wd=0 + lambda_H=0) separates them:
    temporal lands on target  -> the temporal failure was the entropy penalty's
                                 relative scale, and flat weighting DOES fix both
                                 architectures once every knob is scale-matched.
    temporal still ~6-7x      -> flat weighting genuinely fails for the
                                 Jensen-averaged temporal decoder, and the
                                 loss-geometry account is incomplete.
Either answer is worth the run; the second is the more interesting.

DESIGN RULE HERE: `weight_decay=0` for EVERY cell including the evar and KL/JS
references, so the flat-vs-evar contrast is not confounded with the decay that
broke v1. That does mean these references are not directly comparable to
`prodfix_v1` (wd=1e-4) — each arm below therefore ships its own matched control.

Everything else = the v1 / prodfix_v1 baseline: Q, half, 100 ms, H=8, tanh,
dropout 0, patience 0 + monitor_val, val_fraction 0.2, REP 5, 200 epochs, 6 mice,
stratified_balanced, restart_selection='val', seed=0.

Run on gpu1:
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_flatevar_v2.py --dry-run
    $PY run_flatevar_v2.py --smoke
    $PY -u run_flatevar_v2.py --arms core 2>&1 | tee flatevar_v2_core.log
Idempotent: per-mouse shards resume. Smoke writes to `<run>_smoke`.
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config

RUN_ROOT = 'flatevar_v2'
SPLITS = ('stratified_balanced',)
# wd=0 THROUGHOUT — the whole point of this run.
BASE = dict(target='Q', window='half', bin_ms=100, width=8, act='tanh',
            dropout=0.0, wd=0.0, lam=3e-3, epochs=200, rep=5,
            patience=0, val_fraction=0.2, seed=0)

FLAT = dict(flat_evar=True)
LINEAR = dict(hidden_sizes=[])
NOLAM = dict(entropy_lambda=0.0)

# Cells 1-6: the clean flat-vs-evar contrast, all at wd=0.
CELLS_CORE = [
    ('E_evar_h8',   'PCA', {},
     'matched evar reference at wd=0 (v1 R_evar_base was wd=1e-4)'),
    ('F_flat_h8',   'PCA', dict(FLAT),
     'flat weighting, H=8 — v1 said spatial 1.02x / 0.948, temporal 6.41x / 18.2'),
    ('E_evar_lin',  'PCA', dict(LINEAR),
     'matched evar reference, zero hidden units'),
    ('F_flat_lin',  'PCA', {**FLAT, **LINEAR},
     'flat weighting, zero hidden units'),
    ('E_kl_h8',     'KL',  {},
     'calibrated reference at wd=0'),
    ('E_js_h8',     'JS',  {},
     'the other calibrated generalist at wd=0'),
]

# Cells 7-10: THE POINT OF THE RUN — separate the temporal failure from the
# entropy penalty's relative scale. Ordered so the decisive cell is first.
CELLS_TEMPORAL = [
    ('F_flat_h8_lam0',  'PCA', {**FLAT, **NOLAM},
     'DECISIVE: flat + wd=0 + lambda_H=0. Does temporal land on target once EVERY '
     'gradient-scaled knob is off? The cell v1 could not contain.'),
    ('F_flat_lin_lam0', 'PCA', {**FLAT, **LINEAR, **NOLAM},
     'the same with zero hidden units'),
    ('E_evar_h8_lam0',  'PCA', dict(NOLAM),
     'CONTROL: does lambda_H=0 change the EVAR temporal decoder? If it does not, '
     'the penalty only bites when the fit gradient is weak — which is the claim.'),
    ('F_flat_h8_wd1em6', 'PCA', {**FLAT, 'weight_decay': 1e-6},
     'where is the annihilation threshold? 1e-4 killed it; is 1e-6 survivable?'),
]

# Cells 11-17: the neural-PC ladder, re-run at wd=0 (v1's was entirely dead).
CELLS_NEURAL = [
    ('G_flat_npc2',  'PCA', {**FLAT, 'n_neural_pcs': 2},  'neural-PC ladder k=2'),
    ('G_flat_npc4',  'PCA', {**FLAT, 'n_neural_pcs': 4},  'neural-PC ladder k=4'),
    ('G_flat_npc8',  'PCA', {**FLAT, 'n_neural_pcs': 8},  'neural-PC ladder k=8'),
    ('G_flat_npc16', 'PCA', {**FLAT, 'n_neural_pcs': 16}, 'neural-PC ladder k=16'),
    ('G_flat_npc32', 'PCA', {**FLAT, 'n_neural_pcs': 32}, 'neural-PC ladder k=32'),
    ('G_flat_npc64', 'PCA', {**FLAT, 'n_neural_pcs': 64}, 'neural-PC ladder k=64'),
    ('G_evar_npc16', 'PCA', {'n_neural_pcs': 16},
     'CONTROL at wd=0 (v1 got 4.5x target — input PCA alone is not the fix)'),
]

# Cells 18-25: the OAT sweep, now on a baseline that is actually alive.
CELLS_SWEEP = [
    ('F_flat_h2',       'PCA', {**FLAT, 'hidden_sizes': [2]},  'width ladder H=2'),
    ('F_flat_h4',       'PCA', {**FLAT, 'hidden_sizes': [4]},  'width ladder H=4'),
    ('F_flat_h16',      'PCA', {**FLAT, 'hidden_sizes': [16]}, 'width ladder H=16'),
    ('F_flat_h32',      'PCA', {**FLAT, 'hidden_sizes': [32]}, 'width ladder H=32'),
    ('F_flat_drop0p25', 'PCA', {**FLAT, 'dropout': 0.25},      'dropout 0.25'),
    ('F_flat_drop0p5',  'PCA', {**FLAT, 'dropout': 0.5},       'dropout 0.5'),
    ('F_flat_lam1em2',  'PCA', {**FLAT, 'entropy_lambda': 1e-2},
     'lambda_H 1e-2 — v1 sent temporal to 14.6x; is that scale or penalty?'),
    ('F_flat_earlystop', 'PCA', {**FLAT, 'patience': 20, 'min_epochs': 20},
     'early stopping'),
]

ARMS = {'core': CELLS_CORE, 'temporal': CELLS_TEMPORAL,
        'neural': CELLS_NEURAL, 'sweep': CELLS_SWEEP}
DEFAULT_ARMS = ('core', 'temporal', 'neural', 'sweep')


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
    kw.update(extra)
    if not kw['hidden_sizes']:
        kw['weight_snapshot_every'] = 0        # no 'layers.1.weight' to snapshot
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='1 mouse, 2 epochs, REP 1, core cells only')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None)
    p.add_argument('--arms', nargs='+', default=list(DEFAULT_ARMS), choices=list(ARMS))
    a = p.parse_args()

    arms = [x for x in DEFAULT_ARMS if x in set(a.arms)]
    cells = [c for arm in arms for c in ARMS[arm]]
    if a.only:
        cells = [c for c in cells if c[0] in set(a.only)]
    elif a.smoke:
        cells = list(CELLS_CORE)

    epochs, rep = (2, 1) if a.smoke else (BASE['epochs'], BASE['rep'])
    mice = [0] if a.smoke else (range(6) if a.mouse_ids is None else a.mouse_ids)
    n_mice = len(mice) if not isinstance(mice, range) else len(mice)
    root = f'{RUN_ROOT}_smoke' if a.smoke else RUN_ROOT

    print(f"Flat-evar v2 (scale-matched, wd=0): {root}"
          + ("   [SMOKE]" if a.smoke else ""))
    print(f"  fixed : Q half 100ms H={BASE['width']} tanh dropout 0 "
          f"**weight_decay=0** lambda_H {BASE['lam']} patience 0 + monitor_val "
          f"REP {rep} {epochs} ep, {n_mice} mice")
    print(f"  why   : v1's wd=1e-4 annihilated every flat_evar cell "
          f"(||W_in||=0, output uniform) — 30/36 cells void")
    print(f"  arms  : {', '.join(arms)}")
    print(f"  cells : {len(cells)}   fits: {len(cells) * n_mice * 4 * rep}\n")
    for i, (cell, loss, _e, why) in enumerate(cells, 1):
        print(f"    {i:2d}. {cell:20s} {loss:5s} {why}")
    print("\n  FIRST CHECK ON ANY RESULT: ||W_in|| must be non-zero and peakiness "
          "must NOT sit on 1/91 = 0.011. diagnostics/flatevar_report.py flags both.")
    print("  The decisive cell is F_flat_h8_lam0 — it separates 'flat weighting "
          "fails for temporal' from 'the entropy penalty was 45x too strong'.")
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
