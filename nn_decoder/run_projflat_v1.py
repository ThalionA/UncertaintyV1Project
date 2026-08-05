# -*- coding: utf-8 -*-
"""Projection loss with FLAT per-PC weighting: architecture x input dimensionality
x regularisation. The 2026-08-03 run spec.

Flat weighting keeps all 91 PCs at 1/91 each. PCA is an orthogonal rotation, so
that is exactly the MEAN SQUARED ERROR across the 91 orientation bins — the
projection loss stripped of its eigenvalue weighting.

FIXED throughout: Q target, 100 ms bins, second half of the trial (1.0-2.0 s,
10 bins), tanh, val_fraction 0.2, **patience 20** (+ min_epochs 20, monitor_val),
REP 5, max 200 epochs, restart_selection='val', seed 0, 6 mice,
stratified_balanced.

Note patience 20 makes EARLY STOPPING the primary regulariser here — every
previous flatevar cell ran at patience 0. That is deliberate: at patience 0 the
flat-weighted spatial decoder overfits with val/train = 23.5 at wd=0, so
something has to hold it back, and early stopping is the candidate that does not
risk annihilating the network.

WEIGHT DECAY IS ON A DELIBERATELY SMALL LADDER: {0, 1e-6, 1e-5}. Under flat
weighting every PC carries 1/91 ~ 0.011 where the evar weighting puts ~0.9 on the
leading PC, so the fit gradient is ~45x weaker while Adam's `wd*theta` term is
unchanged. At wd=1e-4 the decay wins outright and the weights go to EXACTLY zero
(measured: ||W_in|| = 0.0000, output uniform, 30/36 cells of `flatevar_v1`), and
patience 20 does not rescue it (`A_flat_earlystop`: ||W_in|| 1.57 vs 9.66 at
wd=0, still uniform). 1e-4 is therefore excluded rather than swept.

ARMS
  base  — the 8 architecture x input combinations at the un-regularised setting
          (lambda_H 0, dropout 0, wd 0). Establishes which configurations are
          alive before any regularisation is layered on.
  grid  — the full lambda_H x dropout x wd grid at RAW input, both architectures.
          27 combinations x 2 = 54 cells.
  ref   — one KL-trained cell per architecture, raw input. Training is
          projection-only otherwise; these exist so the figures have a
          "what good looks like" anchor on the performance axis.
  evar  — the 8 base configurations again with the ORIGINAL eigenvalue weighting
          (flat_evar=False). Without these there is no contrast: the run could
          describe how flat/MSE behaves but not show it beats the weighting it
          replaces. Every existing evar baseline (prodfix_v1, hpsweep_v2,
          flatevar_v1) was trained at patience 0, so comparing this run against
          them would confound the weighting with early stopping — the same shape
          of error as reading flat@wd=0 against evar@wd=1e-4, which produced a
          spurious temporal result on 2026-08-03. Drop with `--arms base ref grid`
          if the contrast is not wanted.

Note lambda_H is the SBC entropy penalty and affects the TEMPORAL decoder only,
so the spatial answer effectively comes from the dropout x wd sub-grid (9 points)
however many lambda_H levels are run. Both architectures train in every cell, so
this costs nothing extra — it just means spatial results repeat across lambda_H.

ANALYSIS (all three axes, per the standing rule):
  peakiness   — decoded max-prob / IO target
  overfitting — val / train fit-loss ratio
  performance — held-out loss / leave-one-out predict-mean, scored under BOTH
                KL and the projection metric on a common evar weighting
A decoder must be judged on all three: annihilated cells score 1.00 on the
overfitting axis (a corpse fits train and val equally badly) and below target on
peakiness, and only the performance axis exposes them.

Run on gpu1:
    cd ~/UncertaintyV1/nn_decoder
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PY=~/cluster-env/.venv/bin/python
    $PY run_projflat_v1.py --dry-run
    $PY run_projflat_v1.py --smoke
    $PY -u run_projflat_v1.py --arms base ref 2>&1 | tee projflat_base.log
    $PY -u run_projflat_v1.py 2>&1 | tee projflat.log
Idempotent: per-mouse shards resume. Smoke writes to `<run>_smoke`.
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config

RUN_ROOT = 'projflat_v1'
SPLITS = ('stratified_balanced',)

BASE = dict(target='Q', window='half', bin_ms=100, act='tanh',
            epochs=200, rep=5, patience=20, min_epochs=20, val_fraction=0.2,
            seed=0)

ARCHS = [('h8', [8]), ('lin', [])]
INPUTS = [('raw', None), ('pc3', 3), ('pc5', 5), ('pc10', 10)]
LAMBDAS = [0.0, 3e-3, 1e-2]
DROPOUTS = [0.0, 0.25, 0.5]
# 1e-4 excluded on purpose — it annihilates the network under flat weighting.
WDECAYS = [0.0, 1e-6, 1e-5]

# The un-regularised corner, used for the base and reference cells.
L0, D0, W0 = 0.0, 0.0, 0.0


def _tok(v):
    """Compact, filesystem-safe token: 0 -> '0', 0.003 -> '0p003', 1e-06 -> '1em6'."""
    if v == 0:
        return '0'
    if v < 1e-3:
        return f'{v:.0e}'.replace('e-0', 'em').replace('e-', 'em')
    return str(v).replace('.', 'p')


def cell_name(arch, inp, lam, drop, wd):
    return f'{arch}_{inp}_l{_tok(lam)}_d{_tok(drop)}_w{_tok(wd)}'


def build_cells():
    """(name, loss, overrides, why) — deduplicated, ordered by decision value."""
    seen, cells = set(), []

    def add(name, loss, ov, why, bucket):
        if name in seen:
            return
        seen.add(name)
        cells.append((name, loss, ov, why, bucket))

    # 1. base — the 8 architecture x input combinations, un-regularised.
    for aname, hs in ARCHS:
        for iname, k in INPUTS:
            ov = {'hidden_sizes': hs, 'entropy_lambda': L0, 'dropout': D0,
                  'weight_decay': W0}
            if k is not None:
                ov['n_neural_pcs'] = k
            add(cell_name(aname, iname, L0, D0, W0), 'PCA', ov,
                f'base: {aname}, {iname} input, no regularisation', 'base')

    # 2. ref — one KL cell per architecture, for the performance anchor.
    for aname, hs in ARCHS:
        add(f'{aname}_raw_KLref', 'KL',
            {'hidden_sizes': hs, 'entropy_lambda': L0, 'dropout': D0,
             'weight_decay': W0},
            f'KL reference ({aname}) — the performance anchor', 'ref')

    # 3. evar — matched comparison: same 8 configurations, eigenvalue weighting.
    for aname, hs in ARCHS:
        for iname, k in INPUTS:
            ov = {'hidden_sizes': hs, 'entropy_lambda': L0, 'dropout': D0,
                  'weight_decay': W0, 'flat_evar': False}
            if k is not None:
                ov['n_neural_pcs'] = k
            add(f'{aname}_{iname}_EVAR', 'PCA', ov,
                f'evar-weighted control: {aname}, {iname} input', 'evar')

    # 3b. evarlam — evar controls at lambda_H > 0, so the flat-vs-evar contrast can
    #     be made at the same lambda_H (the 'evar' arm above is lambda_H=0 only).
    #     NOT in DEFAULT_ARMS: request explicitly with --arms evarlam.
    for aname, hs in ARCHS:
        for lam in [l for l in LAMBDAS if l != L0]:
            add(f'{aname}_raw_EVAR_l{_tok(lam)}', 'PCA',
                {'hidden_sizes': hs, 'entropy_lambda': lam, 'dropout': D0,
                 'weight_decay': W0, 'flat_evar': False},
                f'evar control at lambda_H={lam} ({aname})', 'evarlam')

    # 4. grid — full lambda_H x dropout x wd at raw input, both architectures.
    #    wd varies fastest (the axis most likely to kill a cell), then dropout.
    for aname, hs in ARCHS:
        for lam in LAMBDAS:
            for drop in DROPOUTS:
                for wd in WDECAYS:
                    add(cell_name(aname, 'raw', lam, drop, wd), 'PCA',
                        {'hidden_sizes': hs, 'entropy_lambda': lam,
                         'dropout': drop, 'weight_decay': wd},
                        f'grid: {aname} lam={lam} drop={drop} wd={wd}', 'grid')
    return cells


def build(name, loss, extra, root=RUN_ROOT):
    kw = dict(
        run_name=f'{root}/{name}', bin_size_ms=BASE['bin_ms'],
        loss_func=loss, time_window=BASE['window'],
        activation_function=BASE['act'], num_epochs=BASE['epochs'],
        REP=BASE['rep'], patience=BASE['patience'], min_epochs=BASE['min_epochs'],
        val_fraction=BASE['val_fraction'], monitor_val=True,
        restart_selection='val', seed=BASE['seed'],
        track_training_history=True, weight_snapshot_every=10,
    )
    # flat weighting applies to the projection loss only (it is a no-op for KL,
    # but keep it off there so the provenance says what was intended).
    if loss == 'PCA':
        kw['flat_evar'] = True
    kw.update(extra)
    if not kw.get('hidden_sizes'):
        kw['weight_snapshot_every'] = 0      # no 'layers.1.weight' to snapshot
    return default_config_for_target(BASE['target'], **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='1 mouse, 2 epochs, REP 1, base+ref cells only')
    p.add_argument('--mouse-ids', nargs='+', type=int, default=None)
    p.add_argument('--only', nargs='+', default=None)
    p.add_argument('--seed', type=int, default=None,
                   help='override BASE seed AND write to <run>_seed<N> so the '
                        'seed-0 tree is never overwritten (reproducibility check)')
    p.add_argument('--arms', nargs='+', default=['base', 'ref', 'evar', 'grid'],
                   choices=['base', 'ref', 'evar', 'evarlam', 'grid'])
    a = p.parse_args()

    cells = [c for c in build_cells() if c[4] in set(a.arms)]
    if a.only:
        cells = [c for c in cells if c[0] in set(a.only)]
    elif a.smoke:
        cells = [c for c in cells if c[4] in ('base', 'ref', 'evar')]

    epochs, rep = (2, 1) if a.smoke else (BASE['epochs'], BASE['rep'])
    mice = [0] if a.smoke else (range(6) if a.mouse_ids is None else a.mouse_ids)
    n_mice = len(mice) if not isinstance(mice, range) else len(mice)
    root = f'{RUN_ROOT}_smoke' if a.smoke else RUN_ROOT
    if a.seed is not None:
        BASE['seed'] = int(a.seed)
        root = f'{root}_seed{a.seed}'

    print(f"Projection + FLAT weighting (= MSE over 91 bins): {root}"
          + ("   [SMOKE]" if a.smoke else ""))
    print(f"  fixed  : Q, 100ms, second half (1.0-2.0s), tanh, val 0.2, "
          f"patience {BASE['patience']} (min {BASE['min_epochs']}), REP {rep}, "
          f"max {epochs} ep, {n_mice} mice")
    print(f"  axes   : arch {[a_ for a_, _ in ARCHS]}  input "
          f"{[i for i, _ in INPUTS]}  lambda_H {LAMBDAS}  dropout {DROPOUTS}  "
          f"wd {WDECAYS}")
    print(f"  wd     : 1e-4 EXCLUDED — it drives ||W_in|| to 0 under flat weighting")
    by_arm = {}
    for c in cells:
        by_arm[c[4]] = by_arm.get(c[4], 0) + 1
    print(f"  cells  : {len(cells)}  ({', '.join(f'{k} {v}' for k, v in by_arm.items())})"
          f"   fits: {len(cells) * n_mice * 4 * rep}\n")
    for i, (name, loss, _ov, why, arm) in enumerate(cells, 1):
        print(f"    {i:3d}. [{arm:5s}] {name:34s} {loss:4s} {why}")
    print("\n  Judge on ALL THREE axes — peakiness, overfitting, performance under "
          "both metrics.\n  An annihilated cell scores 1.00 on overfitting and below "
          "target on peakiness;\n  only performance exposes it. Check ||W_in|| != 0 first.")
    if a.dry_run:
        return

    for i, (name, loss, ov, why, arm) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {name}  loss={loss}  {ov}\n    -> {why}")
        cfg_extra = dict(ov)
        if a.smoke:
            cfg_extra.update(num_epochs=2, REP=1)
        run_config(build(name, loss, cfg_extra, root=root),
                   splits=SPLITS, mouse_ids=mice)


if __name__ == '__main__':
    main()
