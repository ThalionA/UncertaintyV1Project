# -*- coding: utf-8 -*-
"""Ideal-observer HMM targets (collaborator pkl): loss x weighting x architecture
x entropy penalty. The 2026-08-08 run spec.

Every cell trains against the collaborator's ideal-observer HMM posteriors
(`target_source='io_hmm_pkl'`, 72 circular orientation bins of 2.5 deg) instead
of the export's 91-bin targets. The loader (`nn_decoder/io_hmm_data.py`) aligns
pkl trials to the .mat export by the stimulus-condition barcode; the alignment
report travels with the results. Everything else follows the projflat_v1 house
conventions.

FIXED throughout: Q target, 100 ms bins, second half of the trial (1.0-2.0 s,
10 bins), tanh, val_fraction 0.2, patience 20 (+ min_epochs 20, monitor_val),
REP 5, max 200 epochs, restart_selection='val', seed 0, 6 mice,
stratified_balanced.

GRID (full factorial, 24 cells per mouse):
  loss x weighting — pca     (projection loss, eigenvalue weighting: the default)
                     pcaflat (projection loss, flat 1/K per PC = MSE over bins)
                     kl, js  (distribution losses; JS >= KL on prodfix_v1)
  architecture     — h8 ([8]) and lin ([], the bigger model by parameter count)
  entropy_lambda   — 0, 1e-3, 3e-3

Note entropy_lambda is the SBC entropy penalty and affects the TEMPORAL decoder
only, so the spatial answer effectively repeats across the lambda axis. Both
decoders train in every cell regardless, so this costs nothing extra — accept
the duplication, as every existing sweep does.

ANALYSIS (all three axes, per the standing rule):
  peakiness   — decoded max-prob / IO target
  overfitting — val / train fit-loss ratio
  performance — held-out loss / leave-one-out predict-mean, scored under BOTH
                KL and the projection metric on a common evar weighting

DATA DEPENDENCY: `data/fitted_data_and_posteriors.pkl` must be the COMPLETE
file. The local Slack download is truncated at 33 MiB; a truncated file raises
a RuntimeError from the loader, which this runner catches in smoke mode and
turns into a re-download instruction rather than a traceback.

Run on gpu1:
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_io_hmm_v1.py --dry-run
    $PY run_io_hmm_v1.py --smoke
    $PY -u run_io_hmm_v1.py 2>&1 | tee io_hmm.log
Idempotent: per-mouse shards resume. Smoke writes to `<run>_smoke`.
"""

from __future__ import annotations

import argparse
import sys

from training import default_config_for_target, run_config

RUN_ROOT = 'io_hmm_v2'   # v1 (mouse 0 only) was trained on the NON-HMM marginal by mistake — see PREDICTIONS 2026-08-18
SPLITS = ('stratified_balanced',)
IO_HMM_PKL = 'data/fitted_data_and_posteriors_hmm.pkl'   # the HMM export (full, 6 mice, 2026-08-18)

BASE = dict(target='Q', window='half', bin_ms=100, act='tanh',
            epochs=200, rep=5, patience=20, min_epochs=20, val_fraction=0.2,
            seed=0)

# (slug token, loss_func, extra overrides). 'pca' is the eigenvalue-weighted
# default (flat_evar=False is the Config default); 'pcaflat' flips it on.
LOSSES = [
    ('pca',     'PCA', {}),
    ('pcaflat', 'PCA', {'flat_evar': True}),
    ('kl',      'KL',  {}),
    ('js',      'JS',  {}),
]
ARCHS = [('h8', [8]), ('lin', [])]
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3]   # extended below 1e-3 on 2026-08-09


def _lam_tok(v):
    """Compact lambda token: 0 -> '0', 1e-3 -> '1e-3', 3e-3 -> '3e-3'."""
    if v == 0:
        return '0'
    return f'{v:.0e}'.replace('e-0', 'e-')


def cell_name(losstok, arch, lam):
    return f'{losstok}_{arch}_lh{_lam_tok(lam)}'


def build_cells():
    """(name, loss, overrides, why) — the full 4 x 2 x 3 factorial."""
    cells = []
    for losstok, loss, loss_ov in LOSSES:
        for aname, hs in ARCHS:
            for lam in LAMBDAS:
                ov = dict(loss_ov)
                ov.update(hidden_sizes=hs, entropy_lambda=lam)
                cells.append((cell_name(losstok, aname, lam), loss, ov,
                              f'{losstok} loss, {aname}, lambda_H={lam:g}'))
    return cells


def build(name, loss, extra, root=RUN_ROOT, target_source='io_hmm_pkl'):
    kw = dict(
        run_name=f'{root}/{name}', bin_size_ms=BASE['bin_ms'],
        loss_func=loss, time_window=BASE['window'],
        activation_function=BASE['act'], num_epochs=BASE['epochs'],
        REP=BASE['rep'], patience=BASE['patience'], min_epochs=BASE['min_epochs'],
        val_fraction=BASE['val_fraction'], monitor_val=True,
        restart_selection='val', seed=BASE['seed'],
        track_training_history=True, weight_snapshot_every=10,
        # The point of the run: swap the export targets for the collaborator's
        # ideal-observer HMM posteriors (72-bin circular support).
        target_source='io_hmm_pkl', io_hmm_pkl_path=IO_HMM_PKL,
    )
    kw.update(target_source=target_source)
    kw.update(extra)
    if not kw.get('hidden_sizes'):
        kw['weight_snapshot_every'] = 0      # no 'layers.1.weight' to snapshot
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='mouse 0 only, 2 epochs, REP 1, lambda_H=0 cells only '
                        '(one per loss x architecture)')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None)
    p.add_argument('--seed', type=int, default=None,
                   help='override BASE seed AND write to <run>_seed<N> so the '
                        'seed-0 tree is never overwritten (reproducibility check)')
    p.add_argument('--target-source', choices=('io_hmm_pkl', 'export'),
                   default='io_hmm_pkl',
                   help="'export' trains the SAME grid on the old 91-bin "
                        "post_s_marginal targets, writing to <run>_exportref — "
                        'the matched reference arm for old-vs-new comparison')
    p.add_argument('--allow-partial', action='store_true',
                   help='opt in to running on a truncated pkl copy (memo '
                        'recovery; only fully-parsed mice available — the '
                        'early mouse-0 run before the full re-download)')
    a = p.parse_args()

    cells = build_cells()
    if a.only:
        cells = [c for c in cells if c[0] in set(a.only)]
    elif a.smoke:
        cells = [c for c in cells if c[0].endswith('_lh0')]

    epochs, rep = (2, 1) if a.smoke else (BASE['epochs'], BASE['rep'])
    mice = [0] if a.smoke else (range(6) if a.mouse_ids is None else a.mouse_ids)
    n_mice = len(mice)
    root = f'{RUN_ROOT}_smoke' if a.smoke else RUN_ROOT
    if a.target_source == 'export':
        # Matched reference arm: identical cells/seeds/splits, old 91-bin targets.
        root = f'{root}_exportref'
    if a.seed is not None:
        BASE['seed'] = int(a.seed)
        root = f'{root}_seed{a.seed}'

    banner = ("Ideal-observer HMM targets (io_hmm_pkl, 72 circular bins)"
              if a.target_source == 'io_hmm_pkl' else
              "OLD export targets (post_s_marginal, 91 linear bins) "
              "— matched reference arm")
    print(f"{banner}: {root}" + ("   [SMOKE]" if a.smoke else ""))
    print(f"  fixed  : Q, 100ms, second half (1.0-2.0s), tanh, val 0.2, "
          f"patience {BASE['patience']} (min {BASE['min_epochs']}), REP {rep}, "
          f"max {epochs} ep, {n_mice} mice")
    print(f"  target : target_source={a.target_source!r}"
          + (f"  pkl={IO_HMM_PKL}" if a.target_source == 'io_hmm_pkl' else ''))
    print(f"  axes   : loss {[l for l, _, _ in LOSSES]}  arch "
          f"{[a_ for a_, _ in ARCHS]}  lambda_H {LAMBDAS}")
    print(f"  cells  : {len(cells)} per mouse x {n_mice} mice = "
          f"{len(cells) * n_mice} mouse-cells"
          f"   fits: {len(cells) * n_mice * 4 * rep}\n")
    for i, (name, loss, _ov, why) in enumerate(cells, 1):
        print(f"    {i:3d}. {name:22s} {loss:4s} {why}")
    print("\n  Judge on ALL THREE axes — peakiness, overfitting, performance under "
          "both metrics.\n  lambda_H touches the temporal decoder only; spatial "
          "results repeat across it.")
    if a.dry_run:
        return

    for i, (name, loss, ov, why) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {name}  loss={loss}  {ov}\n    -> {why}")
        cfg_extra = dict(ov)
        if a.smoke:
            cfg_extra.update(num_epochs=2, REP=1)
        if a.allow_partial:
            cfg_extra.update(io_hmm_allow_partial=True)
        try:
            # on_error='raise' in smoke so a truncated-pkl RuntimeError surfaces
            # here instead of being swallowed per-mouse with a traceback.
            run_config(build(name, loss, cfg_extra, root=root,
                             target_source=a.target_source),
                       splits=SPLITS, mouse_ids=mice,
                       on_error='raise' if a.smoke else 'continue')
        except RuntimeError as exc:
            if 'truncated' in str(exc).lower():
                print(f"\n[!] {exc}")
                print("    The local copy of the ideal-observer pkl is incomplete "
                      "(Slack download cut off at 33 MiB).")
                print(f"    Re-download the full file from Slack and replace "
                      f"{IO_HMM_PKL}, then re-run.")
                sys.exit(1)
            raise


if __name__ == '__main__':
    main()
