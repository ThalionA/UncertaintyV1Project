# -*- coding: utf-8 -*-
"""Control: does restart-selection-on-training-loss confound the overfitting results?

The 2026-07 audit's sharpest finding. `train_and_select_best_model` used to pick the
restart with the lowest TRAINING loss, which systematically favours the most overfit
restart — and plausibly does so more strongly for losses with richer objectives, i.e.
exactly the "KL overfits most / projection-based least" ordering the 2026-07-08 analysis
explained mechanistically. If that ordering is partly an artefact of the selection rule,
the mechanistic story needs revising.

Design — a matched pair. With `Config.seed` fixed, the two runs generate *identical*
restarts; only which one is kept differs:

    restart_ctrl_val   : restart_selection='val'    (the fixed rule)
    restart_ctrl_train : restart_selection='train'  (the historical rule)

Everything else matches the hpsweep_v2 baseline (Q/half/100 ms, H=8, patience 0 +
monitor_val, val_fraction 0.2, REP=5, 200 epochs, 6 mice). Two losses are enough to test
the ordering claim: KL (said to overfit most) and PCA/projection-based (least).

Because every run now records `history['restart_scores']` = per-restart (train, val), the
DISAGREEMENT RATE between the two rules is also measurable directly from either run.

Usage:
    python diagnostics/restart_selection_control.py --run       # trains (slow)
    python diagnostics/restart_selection_control.py --report    # reads results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import default_config_for_target, run_config  # noqa: E402

LOSSES = ('KL', 'PCA')
RULES = ('val', 'train')
SEED = 0
SPLIT = 'stratified_balanced'


def slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def run(mouse_ids):
    for rule in RULES:
        for loss in LOSSES:
            cfg = default_config_for_target(
                'Q', run_name=f'restart_ctrl_{rule}', bin_size_ms=100,
                loss_func=loss, time_window='half',
                hidden_sizes=[8], entropy_lambda=3e-3, dropout=0.0,
                activation_function='tanh', weight_decay=1e-4,
                num_epochs=200, REP=5, patience=0, min_epochs=0,
                val_fraction=0.2, monitor_val=True,
                restart_selection=rule, seed=SEED,
                track_training_history=True, weight_snapshot_every=0,
            )
            print(f"\n=== restart_selection={rule}  loss={loss} ===")
            run_config(cfg, splits=(SPLIT,), mouse_ids=mouse_ids)


def _load(results_root, rule, loss, arch):
    """per-mouse (train_final, val_final, val/train) + restart score table.

    Also verifies the rule the run ACTUALLY used, read back from
    history['restart_selection']. A Config field can reach `to_legacy_dict` and
    still be inert if it is not threaded into run_experiment's `training_params`
    — that happened on the first attempt here and made the control compare 'val'
    against 'val'. Never trust the directory name alone."""
    ck = Path(results_root) / f'restart_ctrl_{rule}' / slug(loss) / 'checkpoints'
    rows, disagree, total, used = [], 0, 0, set()
    for pt in sorted(ck.glob(f'mouse_*_{SPLIT}.pt')):
        h = (torch.load(str(pt), map_location='cpu', weights_only=False)
             .get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if not (t and v):
            continue
        used.add(h.get('restart_selection', '?'))
        rows.append((t[-1], v[-1], v[-1] / t[-1] if t[-1] > 0 else np.nan))
        sc = h.get('restart_scores') or []
        if len(sc) > 1 and all(s[1] is not None for s in sc):
            total += 1
            if int(np.argmin([s[0] for s in sc])) != int(np.argmin([s[1] for s in sc])):
                disagree += 1
    if used and used != {rule}:
        raise SystemExit(
            f"PLUMBING ERROR: results/restart_ctrl_{rule}/{slug(loss)} was trained with "
            f"restart_selection={used}, not {rule!r}. The Config field is not reaching "
            f"train_and_select_best_model — check run_experiment's training_params dict. "
            f"Delete results/restart_ctrl_* and re-run.")
    return rows, disagree, total


def report(results_root):
    print("Restart-selection control — hpsweep_v2 baseline (Q/half/100ms, H=8, 6 mice)\n")
    print(f"{'loss':6s}{'arch':9s}{'rule':7s}{'train':>10s}{'val':>10s}{'val/train':>11s}  n")
    summary = {}
    for loss in LOSSES:
        for arch in ('spat', 'temp'):
            for rule in RULES:
                rows, dis, tot = _load(results_root, rule, loss, arch)
                if not rows:
                    print(f"{loss:6s}{arch:9s}{rule:7s}   (no results — run with --run)")
                    continue
                a = np.asarray(rows, float)
                summary[(loss, arch, rule)] = a[:, 2].mean()
                print(f"{loss:6s}{arch:9s}{rule:7s}{a[:,0].mean():10.4f}{a[:,1].mean():10.4f}"
                      f"{a[:,2].mean():11.2f}  {len(rows)}")
                if rule == RULES[0] and tot:
                    print(f"       -> the two rules picked a DIFFERENT restart in "
                          f"{dis}/{tot} mice")
    # The claim under test: does the val/train ordering across losses survive?
    print("\nOrdering check (val/train ratio, spatial):")
    for rule in RULES:
        got = {l: summary.get((l, 'spat', rule)) for l in LOSSES}
        if all(v is not None for v in got.values()):
            order = ' > '.join(k for k, _ in sorted(got.items(), key=lambda kv: -kv[1]))
            print(f"  {rule:6s}: " + '  '.join(f'{k} {v:.1f}' for k, v in got.items())
                  + f"   ordering: {order}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    a = ap.parse_args()
    if a.run:
        run(range(6) if a.mouse_ids is None else a.mouse_ids)
    if a.report or not a.run:
        report(a.results_root)
