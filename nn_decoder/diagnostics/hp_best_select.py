# -*- coding: utf-8 -*-
"""Rank every scanned cell on the three explicit selection criteria and print the table.

Reads the CSV written by `hp_best_scan.py`. The selection rule, stated in full so the
choice is auditable:

  GATE 1  completeness — all 6 mice present, both architectures fitted.
  GATE 2  performance  — KL normalised loss (vs the LOO predict-mean null) < 1 on
                         BOTH architectures. This is criterion (c) as a hard gate:
                         a decoder that does not beat chance is not a candidate.
  SCORE   three penalties, each computed per architecture and then taken at the WORST
          architecture (so no cell wins by being excellent spatially and broken
          temporally):
            C = max(pk / IO, IO / pk)   calibration   1 = exactly on the IO target,
                                                      symmetric in over/under-sharpening
            O = max(val/train, 1)       overfitting   1 = no overfitting
            P = nl_KL                   performance   < 1 beats chance
          score = (C * O * P)^(1/3)  — equal weight in log space, lower is better.
  GATE 3  stability (tie-break) — the top of the ranking is a near-tie, so a cell only
          counts as "chosen" if it sits on a PLATEAU rather than a cliff: every
          neighbouring cell one grid step away along each swept axis must itself pass
          GATE 2, and the cell must not be the endpoint of the axis that carries it.
          Motivation: cells adjacent to a regularisation cliff score well only because
          shrinkage suppresses the overfitting term while the decoder is on its way to
          the uniform attractor (`story_figures.fig_cure`: cures that destroy the decoder).

Usage:  python diagnostics/hp_best_select.py --csv <csv> [--top 12]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

IO = 0.0605

# swept grids, in order, per axis key in the scan CSV (activation is categorical -> no
# neighbour relation defined, so it is exempt from the stability test)
GRIDS = {
    'entropy_lambda': [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    'dropout':        [0, 0.1, 0.25, 0.5, 0.75, 0.9],
    'hidden_width':   [2, 4, 8, 16, 32],
    'patience':       [0, 10, 20, 40],
    'val_fraction':   [0.1, 0.2, 0.3],
    'weight_decay':   [0, 1e-4, 1e-3, 1e-2, 1e-1],
    'shape_lambda':   [0, 1, 3, 10, 30],
}
BASE = {'entropy_lambda': 3e-3, 'dropout': 0.0, 'hidden_width': 8, 'patience': 0,
        'val_fraction': 0.2, 'weight_decay': 1e-4, 'shape_lambda': 0}
HPKEYS = list(GRIDS) + ['activation']


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ('peaky', 'tgt_peaky', 'nl_KL', 'nl_PCA', 'overfit'):
                r[k] = float(r[k]) if r[k] not in ('', 'nan', None) else np.nan
            r['n_mice'] = int(r['n_mice'])
            rows.append(r)
    return rows


def _hp(r):
    return tuple([r['activation']] + [float(r[k]) for k in GRIDS])


def build(path, io):
    rows = load(path)
    byc = defaultdict(dict)
    for r in rows:
        byc[(r['run'], r['cell'], r['loss'])][r['arch']] = r
    recs = {}
    for key, d in byc.items():
        if set(d) != {'spat', 'temp'} or any(d[a]['n_mice'] != 6 for a in d):
            continue
        C = max(max(d[a]['peaky'] / io, io / d[a]['peaky']) for a in d)
        O = max(max(d[a]['overfit'], 1.0) for a in d)
        P = max(d[a]['nl_KL'] for a in d)
        recs[key] = dict(run=key[0], cell=key[1], loss=key[2], C=C, O=O, P=P,
                         score=(C * O * P) ** (1 / 3),
                         gate=all(d[a]['nl_KL'] < 1.0 for a in d),
                         pk_s=d['spat']['peaky'], pk_t=d['temp']['peaky'],
                         kl_s=d['spat']['nl_KL'], kl_t=d['temp']['nl_KL'],
                         of_s=d['spat']['overfit'], of_t=d['temp']['overfit'],
                         row=d['spat'])
    return recs


def stability(rec, recs):
    """(is_stable, notes). Looks up neighbours one grid step away on every swept axis."""
    if rec['run'] != 'hpsweep_v2':
        return None, ['not on the sweep grid — no neighbours defined']
    r = rec['row']
    idx = {(v['run'], v['cell'], v['loss']): v for v in recs.values()}
    lookup = {(_hp(v['row']), v['loss']): v for v in recs.values() if v['run'] == 'hpsweep_v2'}
    me = _hp(r)
    ok, notes = True, []
    for ai, key in enumerate(GRIDS):
        grid = GRIDS[key]
        val = float(r[key])
        try:
            j = min(range(len(grid)), key=lambda k: abs(grid[k] - val))
        except ValueError:
            continue
        off_base = abs(grid[j] - BASE[key]) > 1e-12
        if off_base and j in (0, len(grid) - 1):
            notes.append(f'{key}={grid[j]:g} is a GRID ENDPOINT (no outer neighbour tested)')
        for k in (j - 1, j + 1):
            if not 0 <= k < len(grid):
                continue
            nb = list(me)
            nb[1 + ai] = float(grid[k])
            v = lookup.get((tuple(nb), rec['loss']))
            if v is None:
                continue
            if not v['gate']:
                ok = False
                notes.append(f'neighbour {key}={grid[k]:g} FAILS chance gate '
                             f'(nl_KL {v["kl_s"]:.2f}/{v["kl_t"]:.2f}, peakiness '
                             f'{v["pk_s"]:.4f}/{v["pk_t"]:.4f})')
    return ok, notes


def main(path, top, io):
    recs = build(path, io)
    tp = [v['row']['tgt_peaky'] for v in recs.values()]
    print(f'IO target peakiness measured on the saved targets: {np.nanmean(tp):.5f} '
          f'(sd {np.nanstd(tp):.5f} across all {len(recs)} cells — identical targets '
          f'everywhere).  Constant used for the calibration ratio: {io}')

    passed = sorted([v for v in recs.values() if v['gate']], key=lambda v: v['score'])
    print(f'\n{len(recs)} complete cell x loss combinations; {len(passed)} pass GATE 2 '
          f'(nl_KL < 1 on BOTH architectures)\n')

    hdr = (f"{'#':>3} {'score':>6} {'C':>5} {'O':>5} {'P':>5}  {'loss':>5} {'run':>11}  "
           f"{'cell':<50} {'pk_s':>6} {'pk_t':>6} {'KL_s':>5} {'KL_t':>5} {'of_s':>6} {'of_t':>6}  stable")
    print(hdr)
    print('-' * len(hdr))
    for i, r in enumerate(passed[:top], 1):
        st, notes = stability(r, recs)
        tag = {True: 'yes', False: 'NO', None: 'n/a'}[st]
        if st and any('ENDPOINT' in n for n in notes):
            tag = 'edge'
        print(f"{i:>3} {r['score']:6.3f} {r['C']:5.2f} {r['O']:5.2f} {r['P']:5.3f}  "
              f"{r['loss']:>5} {r['run']:>11}  {r['cell']:<50} "
              f"{r['pk_s']:6.4f} {r['pk_t']:6.4f} {r['kl_s']:5.3f} {r['kl_t']:5.3f} "
              f"{r['of_s']:6.2f} {r['of_t']:6.2f}  {tag}")

    print('\nStability notes for the top cells:')
    for i, r in enumerate(passed[:top], 1):
        st, notes = stability(r, recs)
        if notes:
            print(f"  #{i} {r['loss']} {r['cell']}")
            for n in notes:
                print(f'        {n}')

    chosen = None
    for r in passed:
        st, notes = stability(r, recs)
        if st and not any('ENDPOINT' in n for n in notes):
            chosen = r
            break
    print(f"\nGATE-3 winner (best score that is also on a plateau and interior on every axis):")
    for k in ('run', 'cell', 'loss', 'slug') + tuple(HPKEYS):
        print(f"   {k:>15} = {chosen['row'].get(k)}")
    print(f"   {'raw rank':>15} = #{passed.index(chosen)+1} of {len(passed)}   "
          f"score {chosen['score']:.3f}  (C {chosen['C']:.2f}, O {chosen['O']:.2f}, "
          f"P {chosen['P']:.3f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', required=True)
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--io', type=float, default=IO)
    a = ap.parse_args()
    main(a.csv, a.top, a.io)
