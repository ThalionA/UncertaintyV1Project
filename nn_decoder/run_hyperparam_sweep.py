# -*- coding: utf-8 -*-
"""Wide hyperparameter sweep over six axes, FULL export.

Sweeps, with full diagnostic export (per-epoch train + val curves, weight
snapshots, decoded posteriors incl. per-bin) for **spatial, temporal AND the
shuffle controls**:

    loss_func, entropy_lambda (lambda_H), dropout, activation_function,
    hidden width, early stopping (patience + val-fraction).

Structure ("Wide", 2026-06-18)
------------------------------
A full 6-axis factorial is infeasible (~10k grids -> TB / weeks), so we cross
the axes one-at-a-time (OAT) around a baseline, *repeated under every loss*,
plus two 2-D interaction grids:

  * For EACH loss in LOSSES: sweep each non-loss axis on its own, holding the
    others at baseline. This exposes whether a knob's effect depends on the loss.
  * Two interaction grids at the baseline loss (KL): width x dropout and
    patience x dropout (the classic capacity-vs-regularisation and
    regularise-vs-early-stop interactions).

Baseline = patience 0 + monitor_val ON, i.e. the full `num_epochs` trajectory
with a logged validation curve but NO early-stopping truncation -- the cleanest
view of each knob's train-vs-val gap. The early-stopping axis turns patience on
separately. The val-FRACTION sub-sweep is run at a representative active
patience (20), where the holdout size actually changes the stop decision.

Everything else is held fixed (matched comparison): target Q, window 'half',
100 ms, stratified_balanced split, 6 mice, REP, lr/wd/minibatch from the
Q preset. Each unique (loss, lam, drop, act, width, pat, vf) config gets its own
nested run_name so cells never collide:

    results/hpsweep_wide/<cell>/<target>_<loss>_half_100ms[_all]/
        stratified_balanced.mat                 (decoded/target/per-bin, 4 archs)
        config.yaml
        checkpoints/mouse_<id>_stratified_balanced.pt
            -> {'spat','temp'}     : round-trip bundle + 'history'
               {'spat_shf','temp_shf'} : 'history' only (curves + snapshots)

A manifest CSV (results/hpsweep_wide/MANIFEST.csv) maps every cell directory to
its hyperparameters for analysis. The run is idempotent: run_config writes a
per-mouse shard on success and skips mice whose shard already exists, so a
re-run resumes.

Run (remote GPU -- the agent cannot ssh gpu1; Theo launches)
------------------------------------------------------------
    # from this Mac: push the code (NOT results/) up
    rsync -av --exclude results/ --exclude figures/ \
        ~/Desktop/Experiments/UncertaintyV1/ gpu1:~/UncertaintyV1/
    ssh gpu1 ; tmux new -s hpsweep
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY -u run_hyperparam_sweep.py 2>&1 | tee hpsweep.log
    # detach: Ctrl-b d ; later pull results down:
    rsync -av gpu1:~/UncertaintyV1/nn_decoder/results/hpsweep_wide/ \
        ~/Desktop/Experiments/UncertaintyV1/nn_decoder/results/hpsweep_wide/

Sanity first (cheap):
    $PY run_hyperparam_sweep.py --dry-run      # print the plan + write MANIFEST
    $PY run_hyperparam_sweep.py --smoke        # 1 cell, 1 mouse, 3 epochs + verify
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training import default_config_for_target, run_config


# ---------------------------------------------------------------- per-axis grids
LOSSES          = ('PCA', 'KL', 'JS', 'Wasserstein')   # CE dropped (2026-06-18);
#                  KL is the calibrated generalist (~CE per cross-loss work).
ENTROPY_LAMBDAS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # lambda_H (temporal only)
DROPOUTS        = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)     # 0.4 dropped; 0.75/0.9 added
ACTIVATIONS     = ('tanh', 'relu', 'gelu')             # gelu newly wired
HIDDEN_WIDTHS   = (4, 8, 16, 32, 64)                   # 128 dropped
PATIENCES       = (0, 10, 20, 40)                      # 0 = no early stop
VAL_FRACTIONS   = (0.1, 0.2, 0.3)                      # ES holdout %; swept at pat=20

# Imported from nn_classifier rather than mirrored: a hand-copied set drifts (it
# missed 'identity' when that was added for reduced-rank cells), and the whole
# point of validating here is to stop a name the model would reject — or, before
# 2026-08-12, silently turn into ReLU — from reaching training.
from nn_classifier import VALID_ACTIVATIONS  # noqa: E402

# ------------------------------------------------------------------- baselines
BASE = dict(lam=3e-3, drop=0.0, act='tanh', width=32, pat=0, vf=0.2)
BASE_LOSS_2D = 'KL'        # baseline loss for the 2-D interaction grids
VF_SWEEP_PATIENCE = 20     # val-fraction only bites with early stopping active

# ---------------------------------------------------------------- fixed scope
TARGET, WINDOW, BIN_MS = 'Q', 'half', 100
SPLITS = ('stratified_balanced',)
NUM_EPOCHS = 200
REP = 5
SNAPSHOT_EVERY = 10
RUN_ROOT = 'hpsweep_wide'

_KEYS = ('loss', 'lam', 'drop', 'act', 'width', 'pat', 'vf')


def _g(x) -> str:
    """Compact, path-safe number format: 0.003 -> '0p003'."""
    return f"{x:g}".replace('.', 'p').replace('-', 'm')


def cell_name(p: dict) -> str:
    return (f"lam{_g(p['lam'])}_drop{_g(p['drop'])}_act{p['act']}"
            f"_h{p['width']}_pat{p['pat']}_vf{_g(p['vf'])}")


def run_name_for(p: dict) -> str:
    """Nested under one parent so all cells live in results/hpsweep_wide/."""
    return f"{RUN_ROOT}/{cell_name(p)}"


def enumerate_points() -> list[dict]:
    """All unique config points for the Wide structure (deduped)."""
    pts: list[dict] = []

    # (1) OAT under EVERY loss -- vary one non-loss axis, others at baseline.
    oat_axes = {'lam': ENTROPY_LAMBDAS, 'drop': DROPOUTS, 'act': ACTIVATIONS,
                'width': HIDDEN_WIDTHS, 'pat': PATIENCES}
    for loss in LOSSES:
        for axis, vals in oat_axes.items():
            for v in vals:
                pts.append({**BASE, axis: v, 'loss': loss})
        # val-fraction sub-sweep at an ES-active patience (else it's a no-op).
        for vf in VAL_FRACTIONS:
            pts.append({**BASE, 'vf': vf, 'pat': VF_SWEEP_PATIENCE, 'loss': loss})

    # (2) 2-D interaction grids at the baseline loss.
    for w in HIDDEN_WIDTHS:
        for d in DROPOUTS:
            pts.append({**BASE, 'width': w, 'drop': d, 'loss': BASE_LOSS_2D})
    for pat in PATIENCES:
        for d in DROPOUTS:
            pts.append({**BASE, 'pat': pat, 'drop': d, 'loss': BASE_LOSS_2D})

    # dedup on the full hyperparameter tuple, preserving order.
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
        raise ValueError(f"activation {p['act']!r} not in {sorted(VALID_ACTIVATIONS)} "
                         "(would silently train ReLU)")
    return default_config_for_target(
        TARGET, run_name=run_name_for(p), bin_size_ms=BIN_MS,
        loss_func=p['loss'], time_window=WINDOW,
        entropy_lambda=p['lam'], dropout=p['drop'],
        activation_function=p['act'], hidden_sizes=[p['width']],
        num_epochs=NUM_EPOCHS, REP=REP,
        patience=p['pat'], min_epochs=(20 if p['pat'] > 0 else 0),
        val_fraction=p['vf'], monitor_val=True,
        track_training_history=True, weight_snapshot_every=SNAPSHOT_EVERY,
    )


def write_manifest(points: list[dict], results_root='results') -> Path:
    out = Path(results_root) / RUN_ROOT / 'MANIFEST.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'run_name', 'cell', 'target', 'window', 'bin_ms',
                    'loss', 'entropy_lambda', 'dropout', 'activation',
                    'hidden_width', 'patience', 'val_fraction'])
        for i, p in enumerate(points):
            w.writerow([i, run_name_for(p), cell_name(p), TARGET, WINDOW, BIN_MS,
                        p['loss'], p['lam'], p['drop'], p['act'],
                        p['width'], p['pat'], p['vf']])
    return out


def _print_plan(points: list[dict]) -> None:
    n = len(points)
    print(f"Wide hyperparameter sweep: parent run_name={RUN_ROOT!r}")
    print(f"  fixed: target={TARGET} window={WINDOW} bin={BIN_MS}ms "
          f"split={SPLITS[0]} mice=6 REP={REP} epochs<= {NUM_EPOCHS} "
          f"snapshot_every={SNAPSHOT_EVERY}")
    print(f"  axes : loss{list(LOSSES)} lam{list(ENTROPY_LAMBDAS)} "
          f"drop{list(DROPOUTS)} act{list(ACTIVATIONS)} "
          f"width{list(HIDDEN_WIDTHS)} pat{list(PATIENCES)} vf{list(VAL_FRACTIONS)}")
    print(f"  unique config cells : {n}")
    print(f"  fits  : {n} cells x 6 mice x 4 archs (spat/temp + 2 shuffle) "
          f"x REP {REP}  = {n * 6 * 4 * REP} net trainings")
    print(f"  rough disk : ~{n * 0.18:.0f}-{n * 0.30:.0f} GB   "
          f"(~0.18-0.30 GB/cell: 1 .mat + 6 checkpoints w/ history+snapshots)")
    print(f"  export : history + val curves + weight snapshots for "
          f"spat, temp, spat_shf, temp_shf; per-bin in decoded_samp")


def _verify_smoke(cfg, split: str) -> None:
    """Load the smoke checkpoint and confirm train+val curves were saved for
    all four archs (the whole point of the run)."""
    import torch
    ckpt = cfg.output_dir() / 'checkpoints' / f'mouse_0_{split}.pt'
    print(f"\n--- smoke verify: {ckpt} ---")
    if not ckpt.exists():
        print("  [FAIL] checkpoint not written"); return
    bundle = torch.load(str(ckpt), weights_only=False)
    print(f"  arch keys: {sorted(bundle)}")
    ok = True
    for arch in ('spat', 'temp', 'spat_shf', 'temp_shf'):
        h = (bundle.get(arch) or {}).get('history')
        if not h:
            print(f"  [FAIL] {arch}: no history"); ok = False; continue
        nt = len(h.get('train_fit_loss', []))
        nv = len(h.get('val_fit_loss', []))
        ns = len(h.get('state_dicts', []))
        print(f"  {arch:9s}: train_fit={nt}  val_fit={nv}  snapshots={ns}")
        if nt == 0 or nv == 0 or ns == 0:
            ok = False
    print("  => SMOKE OK (train+val curves + snapshots present for all 4 archs)"
          if ok else "  => SMOKE FAILED")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true',
                   help='print the plan + write MANIFEST.csv, train nothing')
    p.add_argument('--smoke', action='store_true',
                   help='train ONE cell (KL baseline) on 1 mouse / 3 epochs and verify')
    p.add_argument('--limit', type=int, default=None,
                   help='run only the first N cells (debug / partial)')
    p.add_argument('--start-at', type=int, default=0,
                   help='skip the first K cells (manual resume)')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None,
                   help='subset of mice (default all 6)')
    a = p.parse_args()

    points = enumerate_points()
    _print_plan(points)
    manifest = write_manifest(points)
    print(f"  manifest written: {manifest}")

    if a.dry_run:
        return

    if a.smoke:
        sp = {**BASE, 'loss': BASE_LOSS_2D}
        cfg = build_config(sp)
        cfg.num_epochs, cfg.REP, cfg.weight_snapshot_every = 3, 1, 1
        cfg.run_name = f"{RUN_ROOT}/_smoke/{cell_name(sp)}"
        print(f"\n[smoke] {cfg.run_name}  (3 epochs, REP 1, mouse 0)")
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
