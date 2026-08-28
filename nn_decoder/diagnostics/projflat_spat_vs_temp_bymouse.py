# -*- coding: utf-8 -*-
"""Spatial vs temporal, judged on all the standing axes, for a table of configs
(2026-08-04, extended 2026-08-12, run/table-parametrised 2026-08-28).

`--configs` picks a table from `projflat_cells.TABLES`, and each table pins the run
dir and common-basis anchor that travel with its cell names:
  headline     (default) the nine `projflat_v1` cells — lin / rr8 / h8 crossed with
               variance-weighting (EVAR) / flat-weighting (MSE) / the KL reference,
               matched hyperparameters (raw input, lambda_H 0, dropout 0, wd 0).
               Within each weighting block `lin -> rr8` adds the RANK bottleneck and
               `rr8 -> h8` adds the tanh.
  io_hmm_proj  the twelve `io_hmm_v3` projection-loss cells — {h8, rr8} x {evar
               (`pca`), flat (`pcaflat`)} x lambda_H {0, 1e-4, 3e-3}, IO-HMM targets on
               the 72-bin circular support. lambda_H is TEMPORAL-ONLY, so the cells of
               each lambda group share one spatial fit (bit-equal); the figures bracket
               the groups and say so, and the spatial difference is never a result.

Two deliverables, for each `--measure`:
  (A) per-mouse, one figure per config: two bars/mouse (spatial, temporal), error
      bar over that mouse's trials, star = within-mouse paired test over TRIALS,
      trial count on the tick label.
  (B) across-mice, ALL configs in ONE figure: two bars/config, mean over the
      6 mice, SEM over mice, star = paired t OVER THE 6 MICE (the population test —
      the legitimate, non-pseudoreplicated one), plus the per-mouse points JOINED
      spatial -> temporal so the pairing the test uses is visible.

`--measure proj | kl | peakiness | overfit` (several may be given):
  proj       normalised projection loss (per-trial distance / that mouse's
             leave-one-out predict-mean; < 1 beats chance). PER-TRIAL.
  kl         the same ratio under KL(target || decoded) — basis-free, and the metric
             the projection loss is BLIND to (over-sharpening). Kept for the
             projflat_v1 work, where the disagreement between the two is the
             finding. NOT part of the `io_hmm_proj` deliverable (2026-08-28): that
             one is PROJECTION-ONLY, and no documented command here asks for KL.
             `peakiness` is what carries the over-sharpening story instead.
  peakiness  decoded peak / IO target peak (1 = on target, > 1 over-sharpened).
             PER-TRIAL, and weighting-INDEPENDENT — no PCA basis is involved, so
             its bar heights are comparable across every config in a table.
  overfit    val/train fit-loss at the RESTORED best epoch (1 = none). ONE VALUE
             PER MOUSE, read off the training history — so the per-mouse figure has
             no within-animal error bar and no within-animal test; the n=6 points in
             (B) are the whole distribution.

WEIGHTING (`proj` only): each cell is scored under ITS OWN stored projection
weighting — eigenvalue-weighted for the evar AND the KL-reference cells, uniform
= MSE for the flat cells. The metric therefore DIFFERS between flat and evar
configs, so compare spatial vs temporal WITHIN a config, not bar heights ACROSS
configs. `--weighting common` rescores everything under one evar basis, which makes
configs comparable but answers a different question. KL-trained cells are scored
under the projection loss deliberately (judge under both metrics) but that is not
the metric they were trained on, and their titles carry a `[trained on KL]` tag.

`--trial-stat median` is the robust cross-check for the per-trial measures: under
flat/MSE the per-trial projection loss is heavy-tailed, and the mean-vs-median
disagreement is itself a finding (see GOTCHAS).

TITLES ARE TWO LINES (2026-08-28). What a title carries is: what the panel shows,
the essential normalisation, and what the star is. The STANDING caveats — lambda_H
is temporal-only so a lambda group shares one spatial fit; the mean-vs-median heavy
tail; which projection basis; whether bar heights compare across configs — are true
of the whole RUN, not of any one panel, so `_preamble` prints them once to stdout.
They used to be title text and ran to 4-6 wrapped lines on every figure, squeezing
the bars into a strip. The across-mice figure still draws the "same spatial fit"
brackets under the x-axis, which makes the lambda caveat visible where it matters.

Outputs (PNG+SVG) under `--out-root` (default figures/projflat/); stems are prefixed
`projflat_*` for projflat_v1 and by the run name otherwise.
Usage:  python diagnostics/projflat_spat_vs_temp_bymouse.py --weighting common
        python diagnostics/projflat_spat_vs_temp_bymouse.py --measure peakiness overfit
        python diagnostics/projflat_spat_vs_temp_bymouse.py --trial-stat median
        python diagnostics/projflat_spat_vs_temp_bymouse.py --configs io_hmm_proj \
            --measure proj --weighting common --out-root figures/io_hmm_wide/projection_configs
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import projflat_report as pr  # noqa: E402
from projflat_report import _res, _mice, have, _common_basis, _slug  # noqa: E402
from overfitting_vs_hparams import _overfit_ratios  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial  # noqa: E402

import projflat_cells as pcells  # noqa: E402

# The nine headline cells, from the one shared table (see projflat_cells.py for why
# this is not another local literal). Ordered lin -> rr8 -> h8 within each weighting
# block, so the architecture progression reads left to right: lin->rr8 adds the RANK
# bottleneck, rr8->h8 adds the tanh at the same width and parameter count. The
# KL-trained anchors are scored under the PROJECTION loss deliberately (the standing
# "judge under both metrics" rule) but that is NOT their training metric, and the
# figures say so.
CONFIGS = pcells.HEADLINE            # default; --configs picks another registry table
DEFAULT_ANCHOR = pcells.TABLES['headline']['anchor']


def _metric_label(short, weighting='own'):
    """What the projection loss on the y-axis ACTUALLY is.

    Under `--weighting own` that is the cell's stored weighting (uniform = MSE for
    the flat cells, the eigenvalue spectrum for the evar and KL-reference ones).
    Under `--weighting common` every cell is rescored under the one evar anchor
    basis, so it is evar-weighted whatever the cell stores — the y-label used to
    read "MSE (flat projection)" on a flat cell even in a figure whose title said
    "common evar weighting" (fixed 2026-08-28)."""
    if weighting == 'common':
        return 'evar-weighted projection'
    return ('MSE (flat projection)' if pcells.weighting_of(short) == 'flat'
            else 'evar-weighted projection')


def _is_klref(short):
    return pcells.is_klref(short)


def _t(x):
    return torch.tensor(np.asarray(x, float))


def _basis(pcs, evar):
    """Basis arguments for `fit_loss_per_trial`. KL needs none, and passing None
    through is what tells it to take the divergence branch rather than the
    projection one — never substitute a basis-free loss silently."""
    return (None if pcs is None else _t(pcs), None if evar is None else _t(evar))


def _pt(D, arch, metric, pcs=None, evar=None):
    return fit_loss_per_trial(_t(D[arch]['decoded']), _t(D[arch]['target']),
                              metric, *_basis(pcs, evar)).numpy()


def _chance(D, metric, pcs=None, evar=None):
    tgt = np.asarray(D['spat']['target'], float)     # spat & temp share the target
    ok = np.isfinite(tgt).all(1)
    n = int(ok.sum())
    tot = tgt[ok].sum(0)
    pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
    if n > 1:
        pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)
    return float(np.nanmean(fit_loss_per_trial(_t(pm), _t(tgt), metric,
                                               *_basis(pcs, evar)).numpy()))


# The three standing axes, all judged spatial-vs-temporal in the same bar format.
# `per_trial` says whether the measure HAS a within-animal distribution: projection
# loss and peakiness are per trial, but overfitting is one val/train ratio per mouse
# read off the training history, so it supports no within-mouse error bar and no
# within-mouse test (the across-mice figure is unaffected — there n=6 either way).
# `metric` is the `fit_loss_per_trial` branch: the projection loss needs a PCA
# basis, KL needs none. The projection metric is BLIND to over-sharpening, which is
# exactly why KL is run alongside it and never instead of it.
_METRIC = {'proj': 'PCA', 'kl': 'KL'}

MEASURES = {
    'proj': dict(
        name='projection loss', ref=1.0, per_trial=True,
        ylab='normalised projection loss\n(/ predict-mean; < 1 beats chance)'),
    'kl': dict(
        name='KL', ref=1.0, per_trial=True,
        ylab='normalised KL loss\n(/ predict-mean; < 1 beats chance)'),
    'peakiness': dict(
        name='peakiness', ref=1.0, per_trial=True,
        ylab='peakiness (decoded peak / target peak)\n(1 = on target, > 1 over-sharpened)'),
    'overfit': dict(
        name='overfitting', ref=1.0, per_trial=False,
        ylab='overfitting (val / train fit-loss\nat best epoch; 1 = none)'),
}


def _peakiness_pertrial(D, arch):
    """Per-trial decoded peak / target peak. Weighting-independent (no PCA basis
    involved). NB this is the MEAN OF PER-TRIAL RATIOS, whereas
    `projflat_report.measures` reports the RATIO OF MEANS — both estimate the same
    construct but they are not equal (e.g. lin_raw_EVAR mouse_0 spatial: 9.66 vs
    8.62), so do not expect this figure to match that scorecard digit for digit.
    Denominators are safe: the smallest IO target peak across cells is ~0.015, so
    unlike per-bin ratios this one has no near-zero-denominator tail (GOTCHAS)."""
    dec = np.asarray(D[arch]['decoded'], float).max(1)
    tgt = np.asarray(D[arch]['target'], float).max(1)
    return dec / tgt


def per_mouse(results_root, cell, weighting, measure='proj', anchor=DEFAULT_ANCHOR):
    """{mouse -> (spatial_values, temporal_values)}.

    For the per-trial measures each value array is one entry per trial; for
    `overfit` it is a single-element array (that mouse's val/train ratio), which
    keeps every downstream summariser working unchanged."""
    if measure == 'overfit':
        ck = _slug(results_root, cell) / 'checkpoints'
        sp_d, te_d = _overfit_ratios(ck, 'spat'), _overfit_ratios(ck, 'temp')
        return {m: (np.array([sp_d[m]]), np.array([te_d[m]]))       # paired by mouse
                for m in sorted(set(sp_d) & set(te_d))}
    r = _res(results_root, cell)
    metric = _METRIC.get(measure)
    # KL is basis-free, so it never loads (or needs) the common-basis anchor.
    common = (_common_basis(results_root, 'spat', anchor)
              if weighting == 'common' and measure != 'kl' else None)
    out = {}
    for m in _mice(r):
        D = r[m]['Dist']
        if measure == 'peakiness':
            sp, te = _peakiness_pertrial(D, 'spat'), _peakiness_pertrial(D, 'temp')
            ok = np.isfinite(sp) & np.isfinite(te)
            out[m] = (sp[ok], te[ok])
            continue
        if measure == 'kl':
            pcs = evar = None
        elif weighting == 'own':
            pcs, evar = D.get('pcs'), D.get('explained_var')
        else:
            if m not in common:
                continue
            pcs, evar = common[m]
        sp, te = _pt(D, 'spat', metric, pcs, evar), _pt(D, 'temp', metric, pcs, evar)
        ok = np.isfinite(sp) & np.isfinite(te)
        ch = _chance(D, metric, pcs, evar)
        out[m] = (sp[ok] / ch, te[ok] / ch)
    return out


def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'


def _wrap(text, fig_w_in, fontsize, frac=0.94):
    """Wrap a long title to the FIGURE's own width.

    These titles carry the caveats, so they are long; left on one line they are
    wider than the axes, and `bbox_inches='tight'` then grows the canvas to fit
    the title — which is how a 6-bar panel ended up as a thin strip of axes inside
    a 1600 px-wide PNG with empty margins either side. Measured on these very
    titles (2026-08-28): the default sans face runs ~0.62 * fontsize points per
    character, so that is the character budget per figure inch."""
    ncol = max(40, int(frac * fig_w_in * 72 / (0.62 * fontsize)))
    return '\n'.join(textwrap.fill(par, ncol) for par in text.split('\n'))


def _ref_label(measure):
    """What the dotted reference line MEANS for this measure. Both loss ratios are
    normalised by the leave-one-out predict-mean, so their reference is chance —
    an earlier fall-through labelled the KL figures' chance line 'no overfitting'."""
    return {'proj': 'chance', 'kl': 'chance',
            'peakiness': 'on target'}.get(measure, 'no overfitting')


def _headroom(ax, tops, ref, mult=1.26):
    """Leave room above the tallest bar for the stars and the legend, which
    otherwise land on top of the leftmost bars."""
    hi = max(list(tops) + [ref])
    if np.isfinite(hi) and hi > 0:
        ax.set_ylim(top=hi * mult)
    return hi


def _star_y(pair_top, hi, ref, frac=0.055):
    """Where a significance star goes: clear of the taller error bar, and nudged
    clear of the dotted reference line — a star struck through by the chance line
    is unreadable. The window is asymmetric because the glyph grows UPWARD from its
    baseline `y`, so a reference line a little ABOVE the anchor still crosses it."""
    y = pair_top + frac * hi
    d = ref - y
    return ref + 1.15 * frac * hi if -0.3 * frac * hi <= d <= 1.4 * frac * hi else y


def _prefix():
    """Figure-stem prefix for the run currently loaded. The historical
    `projflat_*` stems are kept for projflat_v1 so nothing existing is renamed;
    any other run tags its own stems (same convention as the trial explorer)."""
    return 'projflat' if pr.RUN == 'projflat_v1' else pr.RUN


# lambda_H weights mean H(per-bin predicted posterior), a term that only EXISTS for
# the per-bin temporal decoder — so the cells of a lambda group share one spatial
# fit. Re-measured 2026-08-28 on all four io_hmm_v3 groups x 6 mice: the spatial
# decoded arrays are bit-equal (max |difference| = 0) while the temporal ones differ
# by up to 1.0. This is a property of the RUN, not of a panel, so it is printed once
# by `_preamble` rather than repeated on every title; the across-mice figure draws
# the "same spatial fit" brackets under its x-axis, which says it visually where a
# reader could actually be misled.
_LAMBDA_NOTE = ('lambda_H is TEMPORAL-ONLY: within a lambda group the spatial bars are '
                'the SAME fit (decoded arrays bit-equal) -- only the temporal bar can '
                'move, so a spatial difference across lambda is structurally zero and '
                'is never a result.')

_WEIGHT_NOTE = {
    'own': ('Own stored weighting: flat cells are scored as MSE, evar AND KL-trained cells '
            'eigenvalue-weighted -- so compare spatial-vs-temporal WITHIN a config, never '
            'bar heights ACROSS configs (--weighting common does that).'),
    'common': ('Common evar basis: every cell is rescored under the one anchor basis, so bar '
               'heights ARE comparable across configs -- a different question from each '
               "cell's own training metric."),
}

_MEAN_NOTE = ('Bars are MEANS: the per-trial projection loss is heavy-tailed under flat/MSE '
              '(the artefact that retired the "worse than chance" claim on 2026-08-04) -- '
              'cross-check any bar with --trial-stat median.')

# Named per measure, not as one fixed sentence: a run that asks for peakiness and
# overfit should not be told something about KL it never computed.
_FREE_NAME = {'peakiness': 'Peakiness', 'kl': 'KL', 'overfit': 'Overfitting'}


def _free_note(measures):
    got = [_FREE_NAME[m] for m in ('peakiness', 'kl', 'overfit') if m in measures]
    subj = got[0] if len(got) == 1 else ' and '.join([', '.join(got[:-1]), got[-1]])
    verb = 'needs' if len(got) == 1 else 'need'
    return (f'{subj} {verb} no PCA basis, so they are weighting-independent and their '
            f'bar heights compare across all configs.'
            if len(got) > 1 else
            f'{subj} {verb} no PCA basis, so it is weighting-independent and its bar '
            f'heights compare across all configs.')


def _preamble(tbl, weighting, stat, measures):
    """Print the run's standing caveats ONCE, instead of drawing them on every figure.

    Each of these is true of the whole run rather than of any one panel, which is why
    they are not title text any more: as titles they wrapped to 4-6 lines and dominated
    the frame. Printed here they are still in the record (they land in the run log next
    to the numbers), and the figures keep only the clause that stops a misreading of the
    panel in front of you."""
    out = [tbl['note']]
    # Only if the table's own note does not already say it — io_hmm_proj's note does,
    # and printing both put the same sentence on screen twice.
    if (any(pcells.lambda_of(sh) for _, _, sh in tbl['rows'])
            and 'TEMPORAL-ONLY' not in tbl['note']):
        out.append(_LAMBDA_NOTE)
    if 'proj' in measures:
        out.append(_WEIGHT_NOTE[weighting])
        if stat == 'mean':
            out.append(_MEAN_NOTE)
    if {'peakiness', 'kl', 'overfit'} & set(measures):
        out.append(_free_note(measures))
    print('STANDING CAVEATS for this run (kept off the figures; see the titles for what '
          'each panel shows):')
    for i, line in enumerate(out, 1):
        print(textwrap.fill(line, 96, initial_indent=f'  {i}. ',
                            subsequent_indent='     '))
    print()


# --- trial-level summary -----------------------------------------------------
# The per-trial normalised projection loss is HEAVY-TAILED under flat/MSE, and the
# mean is not robust to it: the 2026-08-04 scrutiny pass RETIRED the claim
# "linear + flat/MSE + raw SPATIAL is worse than chance" precisely because it is a
# mean artefact of a 1% tail (median 0.43 = 2.3x BETTER than chance; the worst 1%
# of trials carry 28% of all squared error — see PROJECT_LOG 2026-08-04 and
# diagnostics/projflat_tail_diagnosis.py). A bar chart of MEANS reproduces that
# artefact, so `--trial-stat median` exists to check any such bar before believing
# it. Mean pairs with a paired t; median pairs with Wilcoxon (robust, and with
# hundreds of TRIALS it has none of the n=6 p-floor problem flagged in GOTCHAS).
def _summ(a, stat):
    return float(np.median(a)) if stat == 'median' else float(np.mean(a))


def _err(a, stat, n_boot=400, seed=0):
    """SEM for the mean; percentile-bootstrap SE for the median."""
    a = np.asarray(a, float)
    if a.size < 2:
        return 0.0
    if stat == 'mean':
        return float(a.std(ddof=1) / np.sqrt(a.size))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    return float(np.median(a[idx], axis=1).std(ddof=1))


def _paired_p(sp, te, stat):
    """Within-mouse paired test over TRIALS, matched to the summary statistic."""
    from scipy.stats import wilcoxon
    if stat == 'median':
        try:
            return float(wilcoxon(sp, te).pvalue)
        except Exception:                                   # noqa: BLE001
            return float('nan')
    return float(ttest_rel(sp, te).pvalue)


# ---------------------------------------------------------- (A) per-mouse figure
def fig_per_mouse(results_root, out_root, title, cell, short, weighting, stat='mean',
                  measure='proj', anchor=DEFAULT_ANCHOR):
    if not have(results_root, cell):
        print(f"  [skip] {short}: {cell} not downloaded")
        return
    M = MEASURES[measure]
    ps.apply()
    pm = per_mouse(results_root, cell, weighting, measure, anchor)
    mice = sorted(pm)
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(mice))
    w = 0.38
    rows, tops, stars = [], [], []
    for xi, m in zip(x, mice):
        sp, te = pm[m]
        pair_top = 0.0
        for v, off, colr, lab in [(sp, -w / 2, ps.SPATIAL, 'spatial'),
                                  (te, +w / 2, ps.TEMPORAL, 'temporal')]:
            # NO error bar for a measure with no within-animal distribution.
            # `_err` returns 0.0 for a single value, but `yerr=0` with `capsize`
            # still DRAWS a cap, and a cap sitting on the bar top reads as a
            # (very tight) error bar — i.e. as within-mouse precision that does
            # not exist, on the very figure whose title says "no error bar".
            e = _err(v, stat) if M['per_trial'] else None
            ax.bar(xi + off, _summ(v, stat), w, yerr=e,
                   color=colr, edgecolor='k', linewidth=0.5,
                   capsize=3 if e is not None else 0,
                   label=lab if xi == 0 else None)
            pair_top = max(pair_top, _summ(v, stat) + (e or 0.0))
        tops.append(pair_top)
        if M['per_trial']:                       # no within-animal test for overfit
            p = _paired_p(sp, te, stat)
            stars.append((xi, pair_top, p))
            rows.append((m, _summ(sp, stat), _summ(te, stat), sp.size, p))
    ax.axhline(M['ref'], color='k', ls=':', lw=1.4, label=_ref_label(measure))
    # Stars sit above the TALLER bar's error bar, and the axis gets headroom, so a
    # star is never hidden behind a cap or behind the legend.
    hi = _headroom(ax, tops, M['ref'])
    for xi, pair_top, p in stars:
        ax.text(xi, _star_y(pair_top, hi, M['ref'], 0.045), _stars(p),
                ha='center', fontsize=8)
    # The within-mouse star is a test over TRIALS, so the trial count is labelled
    # per bar-pair (house rule: an n wherever a subgroup statistic appears). The
    # per-mouse measures have no within-animal distribution, so no n is shown there.
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('mouse_', 'M')
                        + (f'\nn={pm[m][0].size}' if M['per_trial'] else '')
                        for m in mice])
    if measure == 'proj':
        ax.set_ylabel(f'{stat} normalised {_metric_label(short, weighting)} loss\n'
                      f'(/ predict-mean; < 1 beats chance)', fontsize=8)
    else:
        ax.set_ylabel((f'{stat} ' if M['per_trial'] else '') + M['ylab'], fontsize=8)
    ax.set_xlabel('mouse')
    # Legend OUTSIDE the axes: at upper-left it sat on top of the first mouse's
    # significance star.
    ax.legend(fontsize=7, frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    # TWO LINES: (1) which cell and which measure, (2) the normalisation the bars are
    # in and what the star is. Everything else that used to live here — the lambda_H
    # replicate caveat, the mean-vs-median heavy tail, what the weighting does to
    # cross-config comparability — is a property of the RUN and is printed once by
    # `_preamble`. The two clauses kept are the ones that stop a misreading of THIS
    # panel: which basis a projection loss is in, and why the overfit panel has no
    # error bars and no stars.
    cfg = title.replace(chr(10), ', ')
    test = 'Wilcoxon' if stat == 'median' else 'paired t'
    star = f'Star = within-mouse {test} (n = trials).'
    if measure == 'proj':
        wtag = 'own stored weighting' if weighting == 'own' else 'common evar basis'
        head = f'{cfg} — projection loss ({wtag})'
        sub_ = f'Bars = per-trial loss / predict-mean.  {star}'
    elif measure == 'kl':
        head = f'{cfg} — KL loss (basis-free)'
        sub_ = f'Bars = per-trial KL / predict-mean.  {star}'
    elif measure == 'peakiness':
        head = f'{cfg} — peakiness'
        sub_ = f'Bars = decoded peak / target peak.  {star}'
    else:
        head = f'{cfg} — overfitting'
        sub_ = ('Bars = val / train fit-loss at the restored best epoch: ONE value per '
                'mouse, so no error bar and no within-mouse test.')
    if _is_klref(short):
        head += '  [trained on KL]'
    ax.set_title(_wrap(f'{head}\n{sub_}', fig.get_size_inches()[0], 7.4), fontsize=7.4)
    fig.tight_layout()
    stem = (f'{_prefix()}_spmouse_{short}'
            + ('' if measure == 'proj' else f'_{measure}')
            + (('_own' if weighting == 'own' else '') if measure == 'proj' else '')
            + ('_median' if stat == 'median' and M['per_trial'] else ''))
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}")
    for m, a, b, n, pv in rows:
        print(f"      {m}: spatial={a:.4f}  temporal={b:.4f}  "
              f"delta={b - a:+.4f}  n_trials={n}  p={pv:.3g} {_stars(pv)}")


# ------------------------------------------------------ (B) across-mice figure
def fig_across_mice(results_root, out_root, weighting, stat='mean', measure='proj',
                    configs=None, anchor=DEFAULT_ANCHOR):
    # No `note` argument any more: the table note was title text, and is now part of
    # the run's stdout preamble (`_preamble`) instead of being redrawn on the figure.
    configs = configs or CONFIGS
    M = MEASURES[measure]
    ps.apply()
    # Width scales with the config count (9 groups do not fit the 2-col default).
    base_w, base_h = ps.figsize(2, 1)
    fig, ax = plt.subplots(figsize=(max(base_w, 1.45 * len(configs) + 1.5), base_h + 0.5))
    x = np.arange(len(configs))
    w = 0.38
    n_mice = 0
    tops, lows, bars, stars = [], [], [], []
    for xi, (title, cell, short) in zip(x, configs):
        if not have(results_root, cell):
            print(f"  [skip] {short}: {cell} not downloaded")
            continue
        pm = per_mouse(results_root, cell, weighting, measure, anchor)
        mice = sorted(pm)
        n_mice = max(n_mice, len(mice))
        # One value per mouse (mean or median over that mouse's trials), then the
        # test is paired OVER MICE — n=6, the non-pseudoreplicated unit.
        sp = np.array([_summ(pm[m][0], stat) for m in mice])
        te = np.array([_summ(pm[m][1], stat) for m in mice])
        _, p = ttest_rel(sp, te)                              # PAIRED OVER MICE (n=6)
        n_temp = int((te < sp).sum())
        for v, off, colr, lab in [(sp, -w / 2, ps.SPATIAL, 'spatial'),
                                  (te, +w / 2, ps.TEMPORAL, 'temporal')]:
            ax.bar(xi + off, v.mean(), w, yerr=v.std(ddof=1) / np.sqrt(v.size),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=lab if xi == 0 else None)
        # Per-mouse points overlaid AND JOINED spatial -> temporal, because the
        # test is PAIRED: the line is the pairing, so a reader sees which animals
        # move together and whether any one reverses, not just two clouds.
        for si, ti in zip(sp, te):
            ax.plot([xi - w / 2, xi + w / 2], [si, ti], '-', lw=0.6, color='0.45',
                    alpha=0.75, zorder=3)
        for a_, off in [(sp, -w / 2), (te, +w / 2)]:
            ax.plot(np.full_like(a_, xi + off), a_, 'o', ms=2.5, color='0.25',
                    alpha=0.6, zorder=4)
        pair_top = max(v.mean() + v.std(ddof=1) / np.sqrt(v.size) for v in (sp, te))
        pair_top = max(pair_top, sp.max(), te.max())     # the per-mouse points too
        tops.append(pair_top)
        lows.append(min(sp.min(), te.min()))
        bars += [sp.mean(), te.mean()]
        stars.append((xi, pair_top, p, n_temp, len(mice)))
        print(f"  {short:16s} spatial={sp.mean():.4f}+-{sp.std(ddof=1) / np.sqrt(sp.size):.4f}"
              f"  temporal={te.mean():.4f}+-{te.std(ddof=1) / np.sqrt(te.size):.4f}"
              f"  delta={te.mean() - sp.mean():+.4f}  n_mice={len(mice)}"
              f"  p={p:.3g} {_stars(p)}  temporal lower in {n_temp}/{len(mice)}")
    ax.axhline(M['ref'], color='k', ls=':', lw=1.4, label=_ref_label(measure))
    # A LOG y-axis when the bar heights span more than a decade — otherwise one
    # blown-up config (the evar cells under KL reach ~9x chance, with a per-mouse
    # point at 21x) flattens every other config onto the axis floor. Judged on the
    # BAR heights, not the per-mouse points, so a single outlying animal cannot
    # flip the scale; below the threshold nothing changes (the projflat_v1
    # defaults stay linear).
    # 2026-08-28: threshold 10 -> 4. Under OWN weighting the flat cells at high
    # lambda_H reach ~3x the best bar, which on a linear axis squashes the eight
    # configs that matter into the bottom fifth of the frame.
    logy = bool(bars) and min(bars) > 0 and max(bars) / min(bars) > 4
    if logy:
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.set_yscale('log')
        lo, hi = 0.7 * min(lows + [M['ref']]), 1.9 * max(tops)
        ax.set_ylim(bottom=lo, top=hi)
        # A default log axis labels only the decades, which for a 0.5-20 range is
        # two ticks; put readable ratio ticks on instead.
        tk = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi]
        ax.yaxis.set_major_locator(FixedLocator(tk))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        for xi, pair_top, p, n_temp, nm in stars:
            ax.text(xi, pair_top * 1.12, f'{_stars(p)}\n{n_temp}/{nm}',
                    ha='center', fontsize=7)
    else:
        hi = _headroom(ax, tops, M['ref'], mult=1.32)
        for xi, pair_top, p, n_temp, nm in stars:
            ax.text(xi, _star_y(pair_top, hi, M['ref']), f'{_stars(p)}\n{n_temp}/{nm}',
                    ha='center', fontsize=7)
    # Faint rules where the WEIGHTING block changes — read off the table, not
    # assumed to be every third column (only true of the 3x3 HEADLINE grid).
    for b in pcells.block_bounds(configs):
        ax.axvline(b - 0.5, color='0.75', lw=0.9, ls='-', zorder=0)
    # Bracket the lambda pairs under the axis: within a bracket the SPATIAL bars are
    # one and the same fit (lambda_H is temporal-only), so the spatial difference is
    # structurally zero and must not be read as a result. Empty for tables with no
    # lambda axis, so this is a no-op for the projflat_v1 default.
    for i, j in pcells.replicate_groups(configs):
        tr = ax.get_xaxis_transform()
        ax.plot([i - 0.42, j + 0.42], [-0.30, -0.30], color='0.45', lw=1.0,
                transform=tr, clip_on=False, zorder=5)
        ax.text((i + j) / 2, -0.335, 'same spatial fit', transform=tr, ha='center',
                va='top', fontsize=6.5, color='0.35')
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in configs], fontsize=7)
    if measure == 'proj':
        ax.set_ylabel(f'normalised projection loss\n(per-mouse {stat}; / predict-mean; '
                      f'< 1 beats chance)', fontsize=8)
    else:
        ax.set_ylabel((f'per-mouse {stat} ' if M['per_trial'] else '') + M['ylab'],
                      fontsize=8)
    # Legend OUTSIDE the axes: at upper-left it sat on top of the first mouse's
    # significance star.
    ax.legend(fontsize=7, frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    # TWO LINES, same contract as the per-mouse figure: (1) what is plotted and under
    # which basis, (2) what the bars are and what the star is. The table note and the
    # standing caveats go to stdout via `_preamble`; the lambda replicate caveat is
    # already drawn, as brackets, under the x-axis. The log-y clause stays because it
    # changes how the bars themselves must be read.
    head = f"Spatial vs temporal across mice — {M['name']}"
    if measure == 'proj':
        head += (' (own stored weighting)' if weighting == 'own' else ' (common evar basis)')
    if logy:
        head += '  [LOG y: read the bar TOPS against the reference line, not bar lengths]'
    unit = {'proj': f'per-mouse {stat} loss / predict-mean',
            'kl': f'per-mouse {stat} KL / predict-mean',
            'peakiness': f'per-mouse {stat} decoded peak / target peak',
            'overfit': 'val / train fit-loss at the restored best epoch, one per mouse',
            }[measure]
    sub_ = (f'Bars = mean +- SEM over n={n_mice} mice of the {unit}.  '
            f'Star = paired t over mice; k/{n_mice} = mice where temporal is lower.')
    ax.set_title(_wrap(f'{head}\n{sub_}', fig.get_size_inches()[0], 7.1), fontsize=7.1)
    fig.tight_layout()
    # The weighting suffix is only meaningful for the projection loss; peakiness and
    # overfitting never touch a PCA basis, so tagging them '_common'/'_own' would
    # imply a distinction that does not exist.
    stem = (f'{_prefix()}_spat_vs_temp_acrossmice'
            + ('' if measure == 'proj' else f'_{measure}')
            + (('_own' if weighting == 'own' else '_common') if measure == 'proj' else '')
            + ('_median' if stat == 'median' and M['per_trial'] else ''))
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--weighting', choices=['own', 'common'], default='own')
    ap.add_argument('--trial-stat', choices=['mean', 'median'], default='mean',
                    dest='trial_stat',
                    help="how each mouse's per-trial losses are summarised. 'mean' "
                         "(default, historical) is heavy-tail sensitive under flat/MSE; "
                         "'median' is the robust cross-check that retired the "
                         "'worse than chance' claim on 2026-08-04.")
    ap.add_argument('--measure', nargs='+', default=['proj'], choices=list(MEASURES),
                    help="which axis to plot: proj (projection loss, default), "
                         "peakiness (decoded peak / target peak), overfit (val/train "
                         "fit-loss at the restored best epoch), kl (the basis-free "
                         "second metric — the projection loss is blind to "
                         "over-sharpening; NOT part of the io_hmm_proj deliverable, "
                         "which is projection-only). Pass several, e.g. "
                         "--measure proj peakiness overfit.")
    ap.add_argument('--configs', default='headline', choices=list(pcells.TABLES),
                    help='which cell table from projflat_cells.TABLES to plot. Each '
                         'table pins its own run dir and common-basis anchor; --run / '
                         '--basis-anchor override those.')
    ap.add_argument('--run', default=None,
                    help="results run dir, overriding the table's (projflat_report's "
                         'loader keys off this). Cell dirs must hold one slug with '
                         'stratified_balanced.mat, e.g. io_hmm_v3.')
    ap.add_argument('--basis-anchor', default=None, dest='basis_anchor',
                    help="cell whose stored (pcs, explained_var) define the COMMON "
                         "projection basis under --weighting common, overriding the "
                         "table's. Must be an evar cell from the SAME run — the support "
                         'differs between runs (91-bin export vs 72-bin IO-HMM).')
    a = ap.parse_args()
    tbl = pcells.table(a.configs)
    pr.RUN = a.run or tbl['run']          # projflat_report's loader keys off this global
    anchor = a.basis_anchor or tbl['anchor']
    configs = tbl['rows']
    _preamble(tbl, a.weighting, a.trial_stat, a.measure)
    for measure in a.measure:
        print(f"Spatial vs temporal, {MEASURES[measure]['name']}, "
              f"weighting={a.weighting}, trial-stat={a.trial_stat}\n")
        for title, cell, short in configs:
            fig_per_mouse(a.results_root, a.out_root, title, cell, short, a.weighting,
                          a.trial_stat, measure, anchor)
        fig_across_mice(a.results_root, a.out_root, a.weighting, a.trial_stat, measure,
                        configs, anchor)


if __name__ == '__main__':
    main()
