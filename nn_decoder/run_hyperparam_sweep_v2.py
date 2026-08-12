# -*- coding: utf-8 -*-
"""Re-based hyperparameter sweep (v2) — sensible capacity + the loss-side fix.

`run_hyperparam_sweep.py` (v1, `hpsweep_wide`) used H=32, which we then showed is ~18×
overparameterised (~6k params vs ~350 training trials) — the dominant cause of the
across-the-board overfitting, and unnecessary: for PCA/KL/JS, H=4–8 generalises as well
or better with far less overfitting. So v2 re-bases at **H=8** (~5× params/trial), re-centres
the width axis on the now-relevant small regime, and adds two axes v1 lacked:

  * **weight_decay** — a capacity/overfitting lever (live in optim.Adam; nn_classifier.py:805).
  * **shape_lambda** — the PCA loss-side over-sharpening cure (PCA + (λ=shape/100)·Brier;
    PCA-loss only), so this sweep demonstrates the *fix*, not just the failure modes.

Structure is otherwise v1's "Wide": each non-loss axis swept one-at-a-time around the
baseline, repeated under every loss (shape_lambda is PCA-only), plus width×dropout and
patience×dropout 2-D grids at the baseline loss (KL). Full export (history + val curves +
weight snapshots) for spat/temp/per-bin AND shuffle. Q/half/100ms, 6 mice, REP 5,
patience-0 + monitor_val baseline (full 200-epoch trajectories).

Output: results/hpsweep_v2/<cell>/<slug>/… ; results/hpsweep_v2/MANIFEST.csv maps cells.
Idempotent (per-mouse shards resume). Run on gpu1 (agent can't ssh) — see the v1 docstring
for the rsync/tmux launch block, substituting run_hyperparam_sweep_v2.py.

Usage:  python run_hyperparam_sweep_v2.py --dry-run | --smoke | (bare = full run)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training import default_config_for_target, run_config


# ---------------------------------------------------------------- per-axis grids
LOSSES          = ('PCA', 'KL', 'JS', 'Wasserstein')
ENTROPY_LAMBDAS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DROPOUTS        = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
ACTIVATIONS     = ('tanh', 'relu', 'gelu')
HIDDEN_WIDTHS   = (2, 4, 8, 16, 32)                    # re-centred small (was 4..64)
PATIENCES       = (0, 10, 20, 40)
VAL_FRACTIONS   = (0.1, 0.2, 0.3)
WEIGHT_DECAYS   = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)        # NEW; baseline 1e-4
SHAPE_LAMBDAS   = (0, 1, 3, 10, 30)                    # NEW, PCA-only; λ_Brier = shape/100

# Imported, not mirrored — a hand-copied set drifts out of sync with the model's
# own registry (it missed 'identity' when that was added for reduced-rank cells).
from nn_classifier import VALID_ACTIVATIONS  # noqa: E402

# ------------------------------------------------------------------- baseline
BASE = dict(lam=3e-3, drop=0.0, act='tanh', width=8, pat=0, vf=0.2, wd=1e-4, shape=0)
BASE_LOSS_2D = 'KL'
VF_SWEEP_PATIENCE = 20

# ---------------------------------------------------------------- fixed scope
TARGET, WINDOW, BIN_MS = 'Q', 'half', 100
SPLITS = ('stratified_balanced',)
NUM_EPOCHS = 200
REP = 5
SNAPSHOT_EVERY = 10
RUN_ROOT = 'hpsweep_v2'

_KEYS = ('loss', 'lam', 'drop', 'act', 'width', 'pat', 'vf', 'wd', 'shape')


def _g(x) -> str:
    return f"{x:g}".replace('.', 'p').replace('-', 'm')


def cell_name(p: dict) -> str:
    return (f"lam{_g(p['lam'])}_drop{_g(p['drop'])}_act{p['act']}_h{p['width']}"
            f"_pat{p['pat']}_vf{_g(p['vf'])}_wd{_g(p['wd'])}_shp{_g(p['shape'])}")


def run_name_for(p: dict) -> str:
    return f"{RUN_ROOT}/{cell_name(p)}"


def enumerate_points() -> list[dict]:
    pts: list[dict] = []
    # (1) OAT under EVERY loss — vary one non-loss axis, others at baseline.
    oat_axes = {'lam': ENTROPY_LAMBDAS, 'drop': DROPOUTS, 'act': ACTIVATIONS,
                'width': HIDDEN_WIDTHS, 'pat': PATIENCES, 'wd': WEIGHT_DECAYS}
    for loss in LOSSES:
        for axis, vals in oat_axes.items():
            for v in vals:
                pts.append({**BASE, axis: v, 'loss': loss})
        for vf in VAL_FRACTIONS:                          # val-frac at an ES-active patience
            pts.append({**BASE, 'vf': vf, 'pat': VF_SWEEP_PATIENCE, 'loss': loss})

    # (2) shape_lambda — the loss-side cure — PCA only.
    for s in SHAPE_LAMBDAS:
        pts.append({**BASE, 'shape': s, 'loss': 'PCA'})

    # (3) 2-D interaction grids at the baseline loss.
    for w in HIDDEN_WIDTHS:
        for d in DROPOUTS:
            pts.append({**BASE, 'width': w, 'drop': d, 'loss': BASE_LOSS_2D})
    for pat in PATIENCES:
        for d in DROPOUTS:
            pts.append({**BASE, 'pat': pat, 'drop': d, 'loss': BASE_LOSS_2D})

    seen, uniq = set(), []
    for p in pts:
        key = tuple(p[k] for k in _KEYS)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def build_config(p: dict):
    if p['act'] not in VALID_ACTIVATIONS:
        raise ValueError(f"activation {p['act']!r} not in {sorted(VALID_ACTIVATIONS)}")
    return default_config_for_target(
        TARGET, run_name=run_name_for(p), bin_size_ms=BIN_MS,
        loss_func=p['loss'], time_window=WINDOW,
        entropy_lambda=p['lam'], dropout=p['drop'],
        activation_function=p['act'], hidden_sizes=[p['width']],
        weight_decay=p['wd'], shape_lambda=float(p['shape']),
        num_epochs=NUM_EPOCHS, REP=REP,
        patience=p['pat'], min_epochs=(20 if p['pat'] > 0 else 0),
        val_fraction=p['vf'], monitor_val=True,
        track_training_history=True, weight_snapshot_every=SNAPSHOT_EVERY,
    )


def write_manifest(points, results_root='results') -> Path:
    out = Path(results_root) / RUN_ROOT / 'MANIFEST.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'run_name', 'cell', 'loss', 'entropy_lambda', 'dropout',
                    'activation', 'hidden_width', 'patience', 'val_fraction',
                    'weight_decay', 'shape_lambda'])
        for i, p in enumerate(points):
            w.writerow([i, run_name_for(p), cell_name(p), p['loss'], p['lam'], p['drop'],
                        p['act'], p['width'], p['pat'], p['vf'], p['wd'], p['shape']])
    return out


def _print_plan(points) -> None:
    n = len(points)
    print(f"Re-based sweep v2: parent run_name={RUN_ROOT!r}")
    print(f"  fixed: target={TARGET} window={WINDOW} bin={BIN_MS}ms mice=6 REP={REP} "
          f"epochs<= {NUM_EPOCHS} snapshot_every={SNAPSHOT_EVERY}  BASE H={BASE['width']}")
    print(f"  axes : loss{list(LOSSES)} lam{list(ENTROPY_LAMBDAS)} drop{list(DROPOUTS)} "
          f"act{list(ACTIVATIONS)} width{list(HIDDEN_WIDTHS)} pat{list(PATIENCES)} "
          f"vf{list(VAL_FRACTIONS)} wd{list(WEIGHT_DECAYS)} shape{list(SHAPE_LAMBDAS)}(PCA-only)")
    print(f"  unique config cells : {n}")
    print(f"  fits  : {n} cells x 6 mice x 4 archs x REP {REP} = {n * 6 * 4 * REP} net trainings")
    print(f"  rough disk : ~{n * 0.15:.0f}-{n * 0.25:.0f} GB   (smaller nets → smaller checkpoints)")


def _verify_smoke(cfg, split) -> None:
    import torch
    ckpt = cfg.output_dir() / 'checkpoints' / f'mouse_0_{split}.pt'
    print(f"\n--- smoke verify: {ckpt} ---")
    if not ckpt.exists():
        print("  [FAIL] no checkpoint"); return
    b = torch.load(str(ckpt), weights_only=False)
    print(f"  arch keys: {sorted(b)}  | shape_lambda in config: {cfg.shape_lambda}  wd: {cfg.weight_decay}")
    ok = all((b.get(a) or {}).get('history', {}).get('train_fit_loss') for a in ('spat', 'temp', 'spat_shf', 'temp_shf'))
    print("  => SMOKE OK" if ok else "  => SMOKE FAILED (missing history)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--start-at', type=int, default=0)
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    a = p.parse_args()

    points = enumerate_points()
    _print_plan(points)
    print(f"  manifest: {write_manifest(points)}")
    if a.dry_run:
        return

    if a.smoke:
        sp = {**BASE, 'shape': 10, 'loss': 'PCA'}          # exercise the shape_lambda path
        cfg = build_config(sp)
        cfg.num_epochs, cfg.REP, cfg.weight_snapshot_every = 3, 1, 1
        cfg.run_name = f"{RUN_ROOT}/_smoke/{cell_name(sp)}"
        print(f"\n[smoke] {cfg.run_name}  (3 epochs, REP 1, mouse 0, shape_lambda=10)")
        run_config(cfg, splits=SPLITS, mouse_ids=[0])
        _verify_smoke(cfg, SPLITS[0])
        return

    mouse_ids = range(6) if a.mouse_ids is None else a.mouse_ids
    end = len(points) if a.limit is None else min(len(points), a.start_at + a.limit)
    for i in range(a.start_at, end):
        pt = points[i]
        print(f"\n[{i + 1}/{len(points)}] {cell_name(pt)}  loss={pt['loss']}")
        run_config(build_config(pt), splits=SPLITS, mouse_ids=mouse_ids)


if __name__ == '__main__':
    main()
