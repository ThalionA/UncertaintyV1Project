# -*- coding: utf-8 -*-
"""Spatial vs temporal split by IO-HMM behavioural state — io_hmm_v3 kl_h8_lh0, KL metric.

The spat_temp_best_cell comparison, conditioned on the collaborator's IO-HMM latent state.
Because K and state indices are arbitrary per mouse (K = 3,2,2,3,3,3), the across-animals
view groups trials into each mouse's ENGAGED state (from the state suite: M0·s0, M1·s1,
M2·s1, M3·s2, M4·s1, M5·s0) versus ALL OTHER states pooled.

Per-trial losses and both normalisations come from spat_temp_best_cell.load() (imported,
not copied); the divisors stay the per-mouse all-trials scalars, so state subsets remain
on the same scale as the all-trials figure and their union must reproduce it exactly
(asserted). Test trials are mapped back to export order by bit-equal row matching of
Dist.<arch>.decoded into Dist.<arch>.full_decoded (uniqueness asserted), then labelled
with hard_state = argmax gamma from the HMM pkl (export-aligned via load_io_hmm_targets).

Shuffle normalisation is deliberately absent (architecturally biased for spat-vs-temp,
log 2026-07-29): norms are raw and ÷ predict-mean only. States with fewer than
--min-n (15) test trials are suppressed (annotated grey, no mean plotted).

Outputs (PNG+SVG) under figures/io_hmm_wide/spat_temp/:
  spat_temp_by_state_across_<run>_<cell>_<metric>   engaged vs other, across animals
  spat_temp_by_state_within_<run>_<cell>_<metric>   per-mouse Δ(temp − spat) per state
Usage:  python diagnostics/spat_temp_by_state.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spat_temp_best_cell as stbc  # noqa: E402

RUN, CELL, LOSS = 'io_hmm_v3', 'kl_h8_lh0', 'KL'
METRIC = 'KL'
PKL = 'data/fitted_data_and_posteriors_hmm.pkl'
ENGAGED = {0: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 0}   # per-mouse engaged state (state suite)
NORMS = [('raw', 'raw KL loss'), ('pm', 'normalised loss (÷ predict-mean)')]
MIN_N = 15
REF_PM = {'spat': 0.768, 'temp': 0.682}          # all-trials pm figures from the earlier run
ENG_C, OTH_C = '#2171b5', '0.55'


# ------------------------------------------------------------------ data
def match_rows(dec, full):
    """Index of each decoded (test) row in full_decoded (export order); rows are
    bit-equal. Asserts every row matches exactly one full row, no row reused."""
    fmap = {}
    for i, row in enumerate(full):
        fmap.setdefault(row.tobytes(), []).append(i)
    idx = np.empty(len(dec), int)
    for j, row in enumerate(dec):
        hits = fmap.get(row.tobytes(), [])
        assert len(hits) == 1, f'decoded row {j}: {len(hits)} matches in full_decoded'
        idx[j] = hits[0]
    assert len(set(idx)) == len(idx), 'test rows map onto duplicate export trials'
    return idx


def hard_states(mouse):
    """Export-order hard state labels (argmax gamma) and K for one mouse."""
    from utils import load_vr_export
    from io_hmm_data import load_io_hmm_targets
    *_, trials = load_vr_export(mouse)
    io = load_io_hmm_targets(mouse, PKL, trials)
    hs = np.asarray(io['hard_state']).ravel().astype(int)
    return hs, int(io['n_states'])


def gather(results_root):
    """per[m]: state labels + paired per-trial losses (export-order sorted), both norms."""
    data, mice, extras = stbc.load(results_root, METRIC, run=RUN, cell=CELL, loss=LOSS,
                                   return_extras=True)
    per = {}
    for m in mice:
        hs, K = hard_states(m)
        assert ENGAGED[m] < K, f'm{m}: engaged state {ENGAGED[m]} out of range (K={K})'
        idx = {}
        for arch in ('spat', 'temp'):
            ex = extras[(m, arch)]
            assert ex['full_decoded'] is not None, f'm{m} {arch}: run saved no full_decoded'
            assert len(hs) == len(ex['full_decoded']), \
                f'm{m} {arch}: hard_state ({len(hs)}) vs full_decoded ({len(ex["full_decoded"])})'
            idx[arch] = match_rows(ex['decoded'], ex['full_decoded'])[ex['ok']]
        assert set(idx['spat']) == set(idx['temp']), f'm{m}: spat/temp test sets differ'
        osp, otp = np.argsort(idx['spat']), np.argsort(idx['temp'])
        assert np.array_equal(idx['spat'][osp], idx['temp'][otp])
        per[m] = {'K': K, 'eng': ENGAGED[m], 'state': hs[idx['spat'][osp]],
                  'spat': {nk: data[(m, 'spat', nk)][osp] for nk, _ in NORMS},
                  'temp': {nk: data[(m, 'temp', nk)][otp] for nk, _ in NORMS}}
    return data, mice, per


def sanity_union(data, mice, per):
    """Union over states must reproduce the all-trials MULTISET exactly —
    compared elementwise after sorting (a sum/mean check would pass a
    permutation with compensating errors)."""
    worst = 0.0
    for m in mice:
        for arch in ('spat', 'temp'):
            allv = data[(m, arch, 'pm')]
            parts = [per[m][arch]['pm'][per[m]['state'] == s] for s in range(per[m]['K'])]
            u = np.concatenate(parts)
            assert u.size == allv.size, f'm{m} {arch}: state union {u.size} != all {allv.size}'
            worst = max(worst, float(np.max(np.abs(np.sort(u) - np.sort(allv)))))
    assert worst < 1e-6, f'state union deviates from all-trials by {worst:.3e}'
    g = {arch: np.mean([data[(m, arch, 'pm')].mean() for m in mice]) for arch in ('spat', 'temp')}
    flag = all(round(g[a], 3) == REF_PM[a] for a in g)
    print(f'sanity: union over states == all-trials per mouse+arch (max dev {worst:.2e})')
    print(f'sanity: grand pm-normalised means  spat {g["spat"]:.4f}  temp {g["temp"]:.4f}  '
          f'(earlier run {REF_PM["spat"]:.3f}/{REF_PM["temp"]:.3f}) '
          f'{"OK" if flag else "** MISMATCH — check cell/run **"}')


# ------------------------------------------------------------------ figures
def fig_across(mice, per, out_root, min_n):
    groups = [('engaged state', True), ('other states (pooled)', False)]
    fig, ax = plt.subplots(2, len(NORMS), figsize=ps.figsize(len(NORMS), 2), squeeze=False)
    for r, (glab, want_eng) in enumerate(groups):
        for c, (nk, nlab) in enumerate(NORMS):
            a = ax[r][c]
            shown, ns, sp, tp, supp = [], [], [], [], []
            for m in mice:
                msk = (per[m]['state'] == per[m]['eng']) == want_eng
                n = int(msk.sum())
                if n < min_n:
                    supp.append((m, n))
                    continue
                shown.append(m); ns.append(n)
                sp.append(per[m]['spat'][nk][msk].mean())
                tp.append(per[m]['temp'][nk][msk].mean())
            stbc.across_panel(a, np.array(sp), np.array(tp), shown,
                              chance_line=(nk != 'raw'))
            txt = 'test trials/mouse: ' + '  '.join(f'M{m}:{n}' for m, n in zip(shown, ns))
            if supp:
                txt += '\nsuppressed (n<%d): ' % min_n + ', '.join(
                    f'M{m} (n={n})' for m, n in supp)
            a.text(0.03, 0.985, txt, transform=a.transAxes, fontsize=5.2,
                   va='top', ha='left', color='0.35')
            if r == 0:
                a.set_title(nlab, fontsize=9)
            if c == 0:
                a.set_ylabel(glab, fontsize=9)
    ax[0][0].legend(handles=[
        Line2D([0], [0], color='0.62', lw=1.2, marker='o', ms=4, label='mouse'),
        Line2D([0], [0], color='#cb181d', lw=1.4, marker='o', ms=4,
               label=f'mouse {stbc.EXCLUDE}'),
        Line2D([0], [0], color='k', lw=2.4, marker='o', ms=6, label='mean ± SEM')],
        fontsize=6, frameon=True, loc='upper right')
    ax[-1][0].set_xlabel('architecture', fontsize=8)
    ax[-1][1].set_xlabel('architecture', fontsize=8)
    fig.suptitle(f'{RUN} {CELL} — {METRIC} metric by IO-HMM state '
                 '(engaged: ' + ' '.join(f'M{m}·s{ENGAGED[m]}' for m in mice) + ')',
                 fontsize=9)
    ps.label_panels(ax.ravel())
    ps.save_fig(fig, Path(out_root), f'spat_temp_by_state_across_{RUN}_{CELL}_{METRIC}')


def fig_within(mice, per, out_root, min_n):
    fig, ax = plt.subplots(2, 3, figsize=ps.figsize(3, 2), squeeze=False)
    for i, m in enumerate(mice):
        b = ax.ravel()[i]
        K, eng, st = per[m]['K'], per[m]['eng'], per[m]['state']
        d_all = per[m]['temp']['pm'] - per[m]['spat']['pm']
        tlabs, tcols = [], []
        for s in range(K):
            msk = st == s
            n = int(msk.sum())
            iseng = s == eng
            tlabs.append(('s%d (eng)' % s if iseng else 's%d' % s) + f'\nn={n}')
            if n < min_n:
                tcols.append('0.62')
                trans = mtransforms.blended_transform_factory(b.transData, b.transAxes)
                b.text(s, 0.5, f'n<{min_n} — suppressed', transform=trans, rotation=90,
                       fontsize=5.5, color='0.55', ha='center', va='center')
                continue
            tcols.append(ENG_C if iseng else '0.2')
            d = d_all[msk]
            sem = d.std(ddof=1) / np.sqrt(d.size)
            _, p = sstats.ttest_rel(per[m]['temp']['pm'][msk], per[m]['spat']['pm'][msk])
            b.bar(s, d.mean(), 0.66, yerr=sem, capsize=3,
                  color=ENG_C if iseng else OTH_C, edgecolor='k', lw=0.4)
            star = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            if star:   # beyond the error-bar tip, not behind it
                neg = d.mean() < 0
                b.annotate(star, xy=(s, d.mean() + (-1 if neg else 1) * sem),
                           xytext=(0, -3 if neg else 3), textcoords='offset points',
                           ha='center', va='top' if neg else 'bottom', fontsize=8)
        b.axhline(0, color='k', lw=1.0)
        b.set_xticks(range(K)); b.set_xticklabels(tlabs, fontsize=7)
        for tick, cc in zip(b.get_xticklabels(), tcols):
            tick.set_color(cc)
        b.set_xlim(-0.6, K - 0.4)
        b.set_title(f'mouse {m}', fontsize=9,
                    color='#cb181d' if m == stbc.EXCLUDE else 'k')
        if i % 3 == 0:
            b.set_ylabel('Δ (temporal − spatial)\nnormalised loss (÷ predict-mean)',
                         fontsize=8)
        if i >= 3:
            b.set_xlabel('IO-HMM state (own indices)', fontsize=8)
    ax.ravel()[0].legend(handles=[
        Patch(fc=ENG_C, ec='k', lw=0.4, label='engaged state'),
        Patch(fc=OTH_C, ec='k', lw=0.4, label='other state')],
        fontsize=6, frameon=True)
    fig.suptitle(f'{RUN} {CELL} — within-animal Δ({METRIC}) by IO-HMM state; '
                 f'* p<0.05, ** p<0.01 paired t (raw-unit Δ gives identical t/p)',
                 fontsize=9)
    ps.label_panels(ax.ravel())
    ps.save_fig(fig, Path(out_root), f'spat_temp_by_state_within_{RUN}_{CELL}_{METRIC}')


# ------------------------------------------------------------------ printout
def printout(mice, per, min_n):
    print('\n== per mouse, per state (pm-normalised KL; paired t over test trials) ==')
    print(f'{"mouse":>5} {"state":>6} {"eng":>4} {"n_test":>6} {"spat_pm":>8} '
          f'{"temp_pm":>8} {"Δ":>8} {"p":>8}')
    for m in mice:
        for s in range(per[m]['K']):
            msk = per[m]['state'] == s
            n = int(msk.sum())
            tag = 'yes' if s == per[m]['eng'] else ''
            if n < min_n:
                print(f'{m:>5} {s:>6} {tag:>4} {n:>6}   — suppressed (n < {min_n})')
                continue
            x = per[m]['spat']['pm'][msk]; y = per[m]['temp']['pm'][msk]
            _, p = sstats.ttest_rel(y, x)
            print(f'{m:>5} {s:>6} {tag:>4} {n:>6} {x.mean():>8.3f} {y.mean():>8.3f} '
                  f'{(y - x).mean():>+8.3f} {p:>8.4f}')

    print('\n== engaged vs other (per-mouse Δ(temp − spat), pm-normalised) ==')
    dE, dO, used = [], [], []
    for m in mice:
        me = per[m]['state'] == per[m]['eng']
        if me.sum() < min_n or (~me).sum() < min_n:
            print(f'  M{m}: skipped (engaged n={int(me.sum())}, other n={int((~me).sum())})')
            continue
        d = per[m]['temp']['pm'] - per[m]['spat']['pm']
        dE.append(d[me].mean()); dO.append(d[~me].mean()); used.append(m)
    dE, dO = np.array(dE), np.array(dO)
    t, p = sstats.ttest_rel(dE, dO)
    print('  ' + '  '.join(f'M{m}: eng {a:+.3f} oth {b:+.3f}'
                           for m, a, b in zip(used, dE, dO)))
    print(f'  Δ_engaged vs Δ_other: paired t({len(used) - 1})={t:+.2f}, p={p:.3f}  '
          f'(n={len(used)} mice)')

    try:
        for glab, want_eng in [('engaged', True), ('other', False)]:
            rows = []
            for m in mice:
                msk = (per[m]['state'] == per[m]['eng']) == want_eng
                if msk.sum() < min_n:
                    continue
                for arch in ('spat', 'temp'):
                    rows.append(pd.DataFrame({'loss': per[m][arch]['pm'][msk],
                                              'arch': arch, 'mouse': m}))
            c, se, p, warned = stbc.mixedlm_arch(pd.concat(rows, ignore_index=True))
            print(f'  mixedlm ({glab:>7}, pm): β={c:+.4f} (SE {se:.4f}), p={p:.4f}'
                  + (' [boundary — β/SE/p optimiser-robust to 4 dp]' if warned else ''))
    except Exception as e:  # statsmodels genuinely absent — never silence a fit failure
        print(f'  mixedlm skipped: {e}')


def main(results_root, out_root, min_n):
    ps.apply()
    print(f'cell: {RUN}/{CELL} (trained with {LOSS}), scored with {METRIC}; states from {PKL}')
    data, mice, per = gather(results_root)
    sanity_union(data, mice, per)
    fig_across(mice, per, out_root, min_n)
    fig_within(mice, per, out_root, min_n)
    printout(mice, per, min_n)
    print(f'\nDone -> {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/io_hmm_wide/spat_temp')
    ap.add_argument('--min-n', type=int, default=MIN_N,
                    help='suppress states with fewer test trials than this (default 15)')
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.min_n)
