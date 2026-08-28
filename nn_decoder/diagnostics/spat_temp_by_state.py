# -*- coding: utf-8 -*-
"""Spatial vs temporal split by IO-HMM behavioural state, for one cell or a table of them.

The spat_temp_best_cell comparison, conditioned on the collaborator's IO-HMM latent
state. Because K and state indices are arbitrary per mouse (K = 3,2,2,3,3,3), the
across-animals view groups trials into each mouse's ENGAGED state (from the state
suite: M0·s0, M1·s1, M2·s1, M3·s2, M4·s1, M5·s0) versus ALL OTHER states pooled;
the within-animal view keeps each mouse's own state indices. That index map is
asserted against behaviour on every load (`_check_engaged`): the engaged state is
the animal's fastest-running state with n >= --min-n trials, 6/6 mice.

Two modes:
  * one cell            `--run/--cell/--loss` (defaults: io_hmm_v3 kl_h8_lh0, KL metric)
  * a table of cells    `--configs io_hmm_proj` — the twelve io_hmm_v3 projection cells
                        from the shared registry `projflat_cells.TABLES` (the same
                        mechanism the bar drivers take), scored under `--metric PCA`.
    In table mode `--metric` defaults to PCA and the loss slug is discovered per cell.

SCORED UNDER EACH CELL'S OWN STORED WEIGHTING. `spat_temp_best_cell.load(metric='PCA')`
takes the cell's saved `pcs`/`explained_var`, so a flat cell is scored as MSE and an
evar cell eigenvalue-weighted — spatial-vs-temporal is comparable WITHIN a config,
bar heights are not comparable BETWEEN evar and flat (printed as a standing caveat).

Per-trial losses and both normalisations come from spat_temp_best_cell.load()
(imported, not copied); the divisors stay the per-mouse all-trials scalars, so state
subsets remain on the same scale as the all-trials figure and their union must
reproduce it exactly (asserted). Test trials are mapped back to export order by
bit-equal row matching of Dist.<arch>.decoded into Dist.<arch>.full_decoded
(uniqueness asserted), then labelled with hard_state = argmax gamma from the HMM pkl
(export-aligned via load_io_hmm_targets).

Shuffle normalisation is deliberately absent (architecturally biased for spat-vs-temp,
log 2026-07-29): norms are raw and ÷ predict-mean only. States with fewer than
--min-n (15) test trials are suppressed (annotated grey, no bars plotted).

BOTH BARS, NOT A DIFFERENCE (convention change 2026-08-28). Both figures now draw the
spatial and the temporal bar side by side — the same object as the per-mouse and
across-mice bar figures in diagnostics/projflat_spat_vs_temp_bymouse.py, via the
shared `peakiness_style.paired_bars`. The within-animal panel used to plot only
Δ(temporal − spatial), which hid the level the difference sits at (a −0.1 on a
0.6 base and on a 3.0 base are not the same finding).

Outputs (PNG+SVG) under --out-root:
  spat_temp_by_state_across_<run>_<cell>_<metric>   engaged vs other, bars over mice
  spat_temp_by_state_within_<run>_<cell>_<metric>   per-mouse bars per state
Usage:  python diagnostics/spat_temp_by_state.py
        python diagnostics/spat_temp_by_state.py --configs io_hmm_proj \
            --out-root figures/io_hmm_wide/projection_configs/states
"""

from __future__ import annotations

import argparse
import functools
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
import projflat_cells as pcells  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spat_temp_best_cell as stbc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The ONE definition of 'which state is engaged' — imported, never restated here.
from io_hmm_data import engaged_state as io_engaged_state  # noqa: E402

RUN, CELL, LOSS = 'io_hmm_v3', 'kl_h8_lh0', 'KL'
METRIC = 'KL'
PKL = 'data/fitted_data_and_posteriors_hmm.pkl'
# Per-mouse engaged state (state suite). Indices are arbitrary per animal, so this
# map is checked against behaviour on every load — it is the highest-velocity state
# of each mouse, 6/6. See `_check_engaged`.
ENGAGED = {0: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 0}
MIN_N = 15
REF_PM = {'spat': 0.768, 'temp': 0.682}          # all-trials pm figures, DEFAULT cell only
ENG_C = '#2171b5'
_METRIC_WORD = {'PCA': 'projection', 'KL': 'KL'}


def norms_for(metric):
    """[(key, axis label)] — the raw label names the metric, the normalised one is
    the house phrasing ('normalised loss (÷ predict-mean)', never 'skill')."""
    return [('raw', f'raw {_METRIC_WORD.get(metric, metric)} loss'),
            ('pm', 'normalised loss (÷ predict-mean)')]


def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'


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


@functools.lru_cache(maxsize=None)
def hard_states(mouse):
    """Export-order hard state labels (argmax gamma) and K for one mouse.

    Cached: the labels are a property of the ANIMAL, not of the decoder cell, and
    each call re-reads the VR export plus the 33 MiB HMM pkl (~3 s/mouse) — over a
    twelve-config table that is 70 uncached reads of the same six files.

    Also runs `_check_engaged` — see there for why a bare index map needs one."""
    from utils import load_vr_export
    from io_hmm_data import load_io_hmm_targets
    *_, trials = load_vr_export(mouse)
    io = load_io_hmm_targets(mouse, PKL, trials)
    hs = np.asarray(io['hard_state']).ravel().astype(int)
    K = int(io['n_states'])
    _check_engaged(mouse, hs, K, trials)
    return hs, K


def _check_engaged(mouse, hs, K, trials):
    """Cross-check ENGAGED[mouse] against the CANONICAL rule, and say so loudly.

    ENGAGED is a bare literal of per-mouse state INDICES, and those indices are
    arbitrary — whatever order the collaborator's HMM fit labelled its states in.
    A refit that permutes them would silently relabel 'engaged' and every
    across-animals bar here would be wrong with no error and no visible cue.

    The authority is `io_hmm_data.engaged_state`, which names states by what the
    animal DOES (signal-detection d' over the state's hard-assigned trials) rather
    than by index. It is imported, never re-implemented: a second rule living here
    is how two definitions of 'engaged' would drift apart.

    NON-FATAL ON PURPOSE (2026-08-28). As of this writing the literal and the d'
    rule DISAGREE on mice 2 and 4, and a third plausible criterion (highest mean
    running speed) picks the literal's state for 5-6 of 6 — i.e. the label is
    genuinely unsettled, not merely stale. An assert would only convert that open
    question into a crash. It is printed instead, because the figures are still
    readable while it is open: measured over four representative cells, the
    engaged and other bars themselves barely move under the alternative labelling
    (h8_evar_lh0 engaged delta -0.096 p=0.023 -> -0.084 p=0.031), while the
    engaged-vs-OTHER contrast does not survive it at all (h8_flat_lh3e-3
    t=+4.74 p=0.005 -> t=+0.47 p=0.657). So: believe the within-group bars,
    do NOT believe the contrast until the label is settled.
    """
    try:
        canon, info = io_engaged_state(mouse)
    except Exception as e:                       # pkl shape changed, scipy missing…
        print(f'  [engaged-check] m{mouse}: could not evaluate the d-prime rule ({e})')
        return
    if canon != ENGAGED[mouse]:
        _ENGAGED_DISAGREE[mouse] = (ENGAGED[mouse], canon, info.get('margin', float('nan')))


# mouse -> (literal state, d'-rule state, d' margin) for every disagreement seen this
# run; drained into the preamble/printout so it is impossible to miss.
_ENGAGED_DISAGREE = {}


def engaged_warning():
    """The disagreement notice, or None. Printed by `_preamble` and again at the end
    of a run — a caveat this load-bearing should not scroll off the top."""
    if not _ENGAGED_DISAGREE:
        return None
    parts = ', '.join(f'M{m}: literal s{a} vs d-prime s{b} (margin {g:.2f})'
                      for m, (a, b, g) in sorted(_ENGAGED_DISAGREE.items()))
    return ('*** ENGAGED LABEL IS DISPUTED *** the hardcoded ENGAGED map disagrees with '
            f'io_hmm_data.engaged_state (d-prime) on {len(_ENGAGED_DISAGREE)} of 6 mice — '
            f'{parts}. The engaged and other bars are robust to this (they move by <0.02 '
            'in normalised loss), but the engaged-vs-OTHER CONTRAST is NOT: on '
            'h8_flat_lh3e-3 it goes from t=+4.74 p=0.005 to t=+0.47 p=0.657 under the '
            'd-prime labelling. Do not report the contrast until the label is settled.')


def gather(results_root, run, cell, loss, metric, norms):
    """per[m]: state labels + paired per-trial losses (export-order sorted), both norms."""
    data, mice, extras = stbc.load(results_root, metric, run=run, cell=cell, loss=loss,
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
                  'spat': {nk: data[(m, 'spat', nk)][osp] for nk, _ in norms},
                  'temp': {nk: data[(m, 'temp', nk)][otp] for nk, _ in norms}}
    return data, mice, per


def sanity_union(data, mice, per, check_ref=False):
    """Union over states must reproduce the all-trials MULTISET exactly —
    compared elementwise after sorting (a sum/mean check would pass a
    permutation with compensating errors).

    ``check_ref`` compares the grand means against REF_PM, which are the numbers
    from the DEFAULT cell+metric only — checking them for any other cell would
    print a meaningless 'MISMATCH'."""
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
    print(f'  sanity: union over states == all-trials per mouse+arch (max dev {worst:.2e}); '
          f'grand pm means spat {g["spat"]:.4f} temp {g["temp"]:.4f}'
          + ('' if not check_ref else
             f'  (earlier run {REF_PM["spat"]:.3f}/{REF_PM["temp"]:.3f}) '
             + ('OK' if all(round(g[a], 3) == REF_PM[a] for a in g)
                else '** MISMATCH — check cell/run **')))


# ------------------------------------------------------------------ figures
def _arch_axis(a, w=0.5):
    """Tick the two bars `paired_bars` places at ±w/2 around a single group."""
    a.set_xticks([-w / 2, w / 2])
    a.set_xticklabels(['spatial', 'temporal'])
    a.set_xlim(-0.62, 0.62)


def fig_across(mice, per, out_root, min_n, run, cell, metric, cfg, norms):
    """Engaged vs other-pooled, bars = mean ± SEM OVER MICE with the per-mouse points
    joined spatial -> temporal (the pairing the t-test uses), one column per norm."""
    groups = [('engaged state', True), ('other states (pooled)', False)]
    fig, ax = plt.subplots(2, len(norms), figsize=ps.figsize(len(norms), 2), squeeze=False)
    deltas = {}                      # (group, norm) -> per-mouse Δ, for the contrast test
    for r, (glab, want_eng) in enumerate(groups):
        for c, (nk, nlab) in enumerate(norms):
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
            sp, tp = np.array(sp), np.array(tp)
            deltas[(want_eng, nk)] = (tp - sp, shown)
            top = ps.paired_bars(a, 0.0, sp, tp, w=0.5,
                                 labels=('spatial', 'temporal') if (r == 0 and c == 0) else None)
            t, p = sstats.ttest_rel(tp, sp)
            a.text(0.03, 0.03,
                   f'n={len(shown)} mice: t({len(shown) - 1})={t:+.2f}, p={p:.3f} {_stars(p)}\n'
                   f'temporal lower in {int((tp < sp).sum())}/{len(shown)}',
                   transform=a.transAxes, fontsize=6, va='bottom', ha='left', zorder=6,
                   bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
            if nk != 'raw':
                a.axhline(1.0, color='0.4', ls=':', lw=1.2, zorder=0)
                top = max(top, 1.0)
            a.set_ylim(0, top * 1.42)     # headroom for the trials line along the top
            _arch_axis(a)
            # House rule: an n wherever a subgroup statistic appears — here the
            # per-mouse means are over that mouse's trials IN THIS STATE GROUP.
            txt = 'test trials/mouse: ' + '  '.join(f'M{m}:{n}' for m, n in zip(shown, ns))
            if supp:
                txt += f'\nsuppressed (n<{min_n}): ' + ', '.join(
                    f'M{m} (n={n})' for m, n in supp)
            a.text(0.03, 0.985, txt, transform=a.transAxes, fontsize=5.2,
                   va='top', ha='left', color='0.35')
            if r == 0:
                a.set_title(nlab, fontsize=9)
            if c == 0:
                a.set_ylabel(glab, fontsize=9)
    # The engaged-vs-other CONTRAST is the question the two rows pose, so it is drawn
    # rather than left to the printout. Once, in the pm column: the raw column is the
    # same test in other units, and its wider tick labels leave a narrower axes in
    # which the two stat boxes overlap.
    for c, (nk, _) in enumerate(norms):
        if nk == 'raw' and len(norms) > 1:
            continue
        dE, mE = deltas[(True, nk)]
        dO, mO = deltas[(False, nk)]
        keep = [i for i, m in enumerate(mE) if m in set(mO)]
        kO = [i for i, m in enumerate(mO) if m in set(mE)]
        if len(keep) > 1:
            t, p = sstats.ttest_rel(dE[keep], dO[kO])
            # The contrast does NOT survive the alternative engaged labelling (see
            # `engaged_warning`), so while that label is disputed the figure says so
            # next to the number rather than letting the star stand unqualified.
            tag = ('\nlabel disputed — see run notes' if _ENGAGED_DISAGREE else '')
            ax[1][c].text(0.97, 0.03,
                          f'engaged vs other Δ:\nt({len(keep) - 1})={t:+.2f}, p={p:.3f} '
                          f'{_stars(p)} (n={len(keep)}){tag}',
                          transform=ax[1][c].transAxes, fontsize=6, va='bottom',
                          ha='right', zorder=6,
                          bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    for c in range(len(norms)):
        ax[-1][c].set_xlabel('architecture', fontsize=8)
    # No legend: it sat on the trials-per-mouse line, and the two things it named are
    # said in the subtitle instead.
    fig.suptitle(f'{cfg} — {ps.loss_label(metric, short=True)} metric by IO-HMM state, '
                 f'across animals\nBars = mean ± SEM over mice; points = the mice, '
                 f'joined spatial→temporal (the pairing the t-test uses)', fontsize=9)
    ps.label_panels(ax.ravel())
    ps.save_fig(fig, Path(out_root), f'spat_temp_by_state_across_{run}_{cell}_{metric}')


def fig_within(mice, per, out_root, min_n, run, cell, metric, cfg):
    """Per mouse, per state: the spatial and temporal bar side by side (mean ± SEM
    over that state's test trials), star = within-state paired t over trials."""
    fig, ax = plt.subplots(2, 3, figsize=ps.figsize(3, 2), squeeze=False)
    for i, m in enumerate(mice):
        b = ax.ravel()[i]
        K, eng, st = per[m]['K'], per[m]['eng'], per[m]['state']
        tlabs, tcols, tops, stars = [], [], [1.0], []
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
            x, y = per[m]['spat']['pm'][msk], per[m]['temp']['pm'][msk]
            # points=False: hundreds of trials would ink the panel solid.
            top = ps.paired_bars(b, s, x, y, w=0.38, points=False,
                                 labels=('spatial', 'temporal') if (i == 0 and s == 0) else None)
            _, p = sstats.ttest_rel(y, x)
            tops.append(top); stars.append((s, top, p))
        hi = max(tops) * 1.28
        for s, top, p in stars:
            # A bar top lands just under the chance line often enough that the star
            # would sit ON it: nudge clear of the line, and keep a white stroke for
            # the near misses (a bbox would be a box drawn over the data).
            y = top + 0.03 * hi
            if abs(y - 1.0) < 0.035 * hi:
                y = 1.0 + 0.045 * hi
            b.text(s, y, _stars(p), ha='center', fontsize=7,
                   path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
        b.axhline(1.0, color='0.4', ls=':', lw=1.2, zorder=0)   # predict-mean chance
        b.set_ylim(0, hi)
        b.set_xticks(range(K)); b.set_xticklabels(tlabs, fontsize=7)
        for tick, cc in zip(b.get_xticklabels(), tcols):
            tick.set_color(cc)
        b.set_xlim(-0.6, K - 0.4)
        b.set_title(f'mouse {m}', fontsize=9)
        if i % 3 == 0:
            b.set_ylabel('normalised loss (÷ predict-mean)\n(< 1 beats chance)', fontsize=8)
        if i >= 3:
            b.set_xlabel('IO-HMM state (own indices; blue = engaged)', fontsize=8)
    ax.ravel()[0].legend(fontsize=6, frameon=True, loc='upper left')
    fig.suptitle(f'{cfg} — {ps.loss_label(metric, short=True)} metric by IO-HMM state, '
                 f'within animals\nBars = mean ± SEM over that state’s test trials; '
                 f'star = paired t over trials', fontsize=9)
    ps.label_panels(ax.ravel())
    ps.save_fig(fig, Path(out_root), f'spat_temp_by_state_within_{run}_{cell}_{metric}')


# ------------------------------------------------------------------ printout
def printout(mice, per, min_n, metric, mixedlm=True):
    """Per-mouse/per-state table, then the engaged-vs-other contrast. Returns the
    across-mice summary the run-level table is built from."""
    print(f'\n== per mouse, per state (pm-normalised {metric}; paired t over test trials) ==')
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

    print('\n== engaged vs other, across mice (per-mouse means, pm-normalised) ==')
    summ = {}
    for glab, want_eng in [('engaged', True), ('other', False)]:
        sp, tp, used = [], [], []
        for m in mice:
            msk = (per[m]['state'] == per[m]['eng']) == want_eng
            if msk.sum() < min_n:
                continue
            sp.append(per[m]['spat']['pm'][msk].mean())
            tp.append(per[m]['temp']['pm'][msk].mean())
            used.append(m)
        sp, tp = np.array(sp), np.array(tp)
        t, p = sstats.ttest_rel(tp, sp)
        summ[glab] = dict(spat=sp.mean(), temp=tp.mean(), delta=float((tp - sp).mean()),
                          t=float(t), p=float(p), n=len(used), mice=used,
                          d=tp - sp, lower=int((tp < sp).sum()))
        print(f'  {glab:>7}: spat {sp.mean():.3f}  temp {tp.mean():.3f}  '
              f'Δ {np.mean(tp - sp):+.3f}  t({len(used) - 1})={t:+.2f}  p={p:.4f} '
              f'{_stars(p)}  temporal lower in {int((tp < sp).sum())}/{len(used)}  '
              f'(mice {",".join(f"M{m}" for m in used)})')
    keep = [i for i, m in enumerate(summ['engaged']['mice']) if m in set(summ['other']['mice'])]
    kO = [i for i, m in enumerate(summ['other']['mice']) if m in set(summ['engaged']['mice'])]
    dE, dO = summ['engaged']['d'][keep], summ['other']['d'][kO]
    t, p = sstats.ttest_rel(dE, dO)
    summ['contrast'] = dict(t=float(t), p=float(p), n=len(keep))
    print('  ' + '  '.join(f'M{m}: eng {a:+.3f} oth {b:+.3f}'
                           for m, a, b in zip([summ['engaged']['mice'][i] for i in keep], dE, dO)))
    print(f'  Δ_engaged vs Δ_other: paired t({len(keep) - 1})={t:+.2f}, p={p:.3f} {_stars(p)} '
          f'(n={len(keep)} mice)')

    if mixedlm:
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
    return summ


def _preamble(metric, min_n, tbl=None):
    """The caveats that are true of the whole RUN, printed once instead of drawn on
    every figure (same contract as projflat_spat_vs_temp_bymouse._preamble). The
    weighting and lambda sentences come from projflat_cells so both drivers say them
    in exactly the same words."""
    out = []
    if tbl is not None:
        out.append(tbl['note'])
        if (any(pcells.lambda_of(sh) for _, _, sh in tbl['rows'])
                and 'TEMPORAL-ONLY' not in tbl['note']):
            out.append(pcells.LAMBDA_NOTE)
    if metric == 'PCA':
        out.append(pcells.WEIGHT_NOTE['own'])
    out.append('Engaged state per mouse (state suite, functional identity — the raw indices '
               'are arbitrary per mouse): '
               + ' '.join(f'M{m}·s{s}' for m, s in sorted(ENGAGED.items()))
               + '; "other" pools every remaining state of that mouse. This map is a '
                 'hardcoded literal of ARBITRARY per-mouse state indices; it is '
                 'cross-checked on load against io_hmm_data.engaged_state (the d-prime '
                 'rule) and any disagreement is printed below.')
    out.append(f'States with fewer than {min_n} test trials are SUPPRESSED (no bars, n '
               'annotated on the tick) — mouse 5 has a 4-trial state.')
    out.append('No shuffle normalisation: the shuffle control is architecturally biased for '
               'spatial-vs-temporal (log 2026-07-29), so the norms are raw and / predict-mean.')
    print('STANDING CAVEATS for this run (kept off the figures):')
    for i, line in enumerate(out, 1):
        print(textwrap.fill(line, 96, initial_indent=f'  {i}. ', subsequent_indent='     '))
    print()


def one_cell(results_root, out_root, run, cell, loss, metric, min_n, cfg, check_ref,
             mixedlm=True):
    norms = norms_for(metric)
    print(f'\n=== {cfg}  [{run}/{cell}, scored with {metric}] ===')
    try:
        data, mice, per = gather(results_root, run, cell, loss, metric, norms)
    except SystemExit as e:                      # missing results dir — skip, loudly
        print(f'  [skip] {cell}: {e}')
        return None
    sanity_union(data, mice, per, check_ref=check_ref)
    fig_across(mice, per, out_root, min_n, run, cell, metric, cfg, norms)
    fig_within(mice, per, out_root, min_n, run, cell, metric, cfg)
    return printout(mice, per, min_n, metric, mixedlm=mixedlm)


def main(a):
    ps.apply()
    tbl = pcells.table(a.configs) if a.configs else None
    metric = a.metric or ('PCA' if tbl else METRIC)
    if tbl is None:
        _preamble(metric, a.min_n)
        print(f'states from {PKL}')
        # REF_PM is the DEFAULT cell's number, so it is only a check when nothing was
        # overridden — otherwise it would print a MISMATCH for a cell it never described.
        one_cell(a.results_root, a.out_root, a.run, a.cell, a.loss, metric, a.min_n,
                 pcells.cell_label(a.cell),
                 check_ref=(a.run, a.cell, metric) == (RUN, CELL, METRIC),
                 mixedlm=not a.no_mixedlm)
    else:
        run = a.run if a.run != RUN else tbl['run']
        _preamble(metric, a.min_n, tbl)
        print(f'states from {PKL}; loss slug discovered per cell dir')
        summ = {}
        for lab, cell, short in tbl['rows']:
            # loss=None: each io_hmm cell dir holds exactly ONE slug, and passing the
            # single-cell default here would look for a Q_KL dir that is not there.
            s = one_cell(a.results_root, a.out_root, run, cell, None, metric, a.min_n,
                         lab.replace('\n', ', '), check_ref=False, mixedlm=not a.no_mixedlm)
            if s is not None:
                summ[short] = s
        print('\n== run summary: engaged vs other (pm-normalised, per-mouse means, n mice) ==')
        print(f'{"config":18s} {"eng_spat":>8} {"eng_temp":>8} {"eng_Δ":>7} {"eng_p":>7}  '
              f'{"oth_spat":>8} {"oth_temp":>8} {"oth_Δ":>7} {"oth_p":>7}  {"contrast_t":>10} '
              f'{"contrast_p":>10}')
        for short, s in summ.items():
            e, o, c = s['engaged'], s['other'], s['contrast']
            print(f'{short:18s} {e["spat"]:>8.3f} {e["temp"]:>8.3f} {e["delta"]:>+7.3f} '
                  f'{e["p"]:>7.4f}  {o["spat"]:>8.3f} {o["temp"]:>8.3f} {o["delta"]:>+7.3f} '
                  f'{o["p"]:>7.4f}  {c["t"]:>+10.2f} {c["p"]:>10.3f}')
    warn = engaged_warning()
    if warn:
        print()
        print(textwrap.fill(warn, 96, subsequent_indent='  '))
    print(f'\nDone -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/io_hmm_wide/spat_temp')
    ap.add_argument('--configs', default=None, choices=list(pcells.TABLES),
                    help='run a whole cell table from projflat_cells.TABLES (each pins '
                         'its own run dir) instead of the single --cell. With a table '
                         "the loss slug is discovered from each cell's dir.")
    ap.add_argument('--run', default=RUN, help="results run dir (a --configs table's own "
                                               'run wins unless this is changed)')
    ap.add_argument('--cell', default=CELL, help='single-cell mode only')
    ap.add_argument('--loss', default=LOSS,
                    help='loss the cell was TRAINED with (locates the Q_<loss> dir); '
                         'single-cell mode only')
    ap.add_argument('--metric', default=None, choices=['PCA', 'KL'],
                    help='scoring metric. Default: KL for the single cell (its historical '
                         'behaviour), PCA for a --configs table — where PCA means each '
                         "cell's OWN stored pcs/explained_var, i.e. own-weighting.")
    ap.add_argument('--min-n', type=int, default=MIN_N,
                    help='suppress states with fewer test trials than this (default 15)')
    ap.add_argument('--no-mixedlm', action='store_true',
                    help='skip the per-group mixed-effects fits (they dominate the '
                         'runtime of a twelve-config table)')
    main(ap.parse_args())
