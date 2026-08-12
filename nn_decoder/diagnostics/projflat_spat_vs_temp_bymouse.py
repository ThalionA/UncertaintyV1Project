# -*- coding: utf-8 -*-
"""Spatial vs temporal, judged on all three standing axes, across the nine headline
`projflat_v1` configs (2026-08-04, extended 2026-08-12).

Configs come from `projflat_cells.HEADLINE` — lin / rr8 / h8 crossed with
variance-weighting (EVAR) / flat-weighting (MSE) / the KL reference, matched
hyperparameters throughout (raw input, lambda_H 0, dropout 0, wd 0). Within each
weighting block `lin -> rr8` adds the RANK bottleneck and `rr8 -> h8` adds the tanh.

Two deliverables, for each `--measure`:
  (A) per-mouse, one figure per config: two bars/mouse (spatial, temporal).
  (B) across-mice, ALL nine configs in ONE figure: two bars/config, mean over the
      6 mice, SEM over mice, star = paired t OVER THE 6 MICE (the population test —
      the legitimate, non-pseudoreplicated one), plus the per-mouse points.

`--measure proj | peakiness | overfit` (several may be given):
  proj       normalised projection loss (per-trial distance / that mouse's
             leave-one-out predict-mean; < 1 beats chance). PER-TRIAL.
  peakiness  decoded peak / IO target peak (1 = on target, > 1 over-sharpened).
             PER-TRIAL, and weighting-INDEPENDENT — no PCA basis is involved, so
             its bar heights are comparable across all nine configs.
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
the metric they were trained on, and the titles say so.

`--trial-stat median` is the robust cross-check for the per-trial measures: under
flat/MSE the per-trial projection loss is heavy-tailed, and the mean-vs-median
disagreement is itself a finding (see GOTCHAS).

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_spat_vs_temp_bymouse.py --weighting common
        python diagnostics/projflat_spat_vs_temp_bymouse.py --measure peakiness overfit
        python diagnostics/projflat_spat_vs_temp_bymouse.py --trial-stat median
"""

from __future__ import annotations

import argparse
import sys
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
CONFIGS = pcells.HEADLINE


def _metric_label(short):
    """What the stored projection weighting actually is for this cell."""
    return ('MSE (flat projection)' if pcells.weighting_of(short) == 'flat'
            else 'evar-weighted projection')


def _is_klref(short):
    return pcells.is_klref(short)


def _t(x):
    return torch.tensor(np.asarray(x, float))


def _pt(D, arch, pcs, evar):
    return fit_loss_per_trial(_t(D[arch]['decoded']), _t(D[arch]['target']),
                              'PCA', _t(pcs), _t(evar)).numpy()


def _chance(D, pcs, evar):
    tgt = np.asarray(D['spat']['target'], float)     # spat & temp share the target
    ok = np.isfinite(tgt).all(1)
    n = int(ok.sum())
    tot = tgt[ok].sum(0)
    pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
    if n > 1:
        pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)
    return float(np.nanmean(fit_loss_per_trial(_t(pm), _t(tgt), 'PCA',
                                               _t(pcs), _t(evar)).numpy()))


# The three standing axes, all judged spatial-vs-temporal in the same bar format.
# `per_trial` says whether the measure HAS a within-animal distribution: projection
# loss and peakiness are per trial, but overfitting is one val/train ratio per mouse
# read off the training history, so it supports no within-mouse error bar and no
# within-mouse test (the across-mice figure is unaffected — there n=6 either way).
MEASURES = {
    'proj': dict(
        name='projection loss', ref=1.0, per_trial=True,
        ylab='normalised projection loss\n(/ predict-mean; < 1 beats chance)'),
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


def per_mouse(results_root, cell, weighting, measure='proj'):
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
    common = _common_basis(results_root, 'spat') if weighting == 'common' else None
    out = {}
    for m in _mice(r):
        D = r[m]['Dist']
        if measure == 'peakiness':
            sp, te = _peakiness_pertrial(D, 'spat'), _peakiness_pertrial(D, 'temp')
            ok = np.isfinite(sp) & np.isfinite(te)
            out[m] = (sp[ok], te[ok])
            continue
        if weighting == 'own':
            pcs, evar = D.get('pcs'), D.get('explained_var')
        else:
            if m not in common:
                continue
            pcs, evar = common[m]
        sp, te = _pt(D, 'spat', pcs, evar), _pt(D, 'temp', pcs, evar)
        ok = np.isfinite(sp) & np.isfinite(te)
        ch = _chance(D, pcs, evar)
        out[m] = (sp[ok] / ch, te[ok] / ch)
    return out


def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'


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
                  measure='proj'):
    if not have(results_root, cell):
        print(f"  [skip] {short}: {cell} not downloaded")
        return
    M = MEASURES[measure]
    ps.apply()
    pm = per_mouse(results_root, cell, weighting, measure)
    mice = sorted(pm)
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(mice))
    w = 0.38
    for xi, m in zip(x, mice):
        sp, te = pm[m]
        for v, off, colr, lab in [(sp, -w / 2, ps.SPATIAL, 'spatial'),
                                  (te, +w / 2, ps.TEMPORAL, 'temporal')]:
            ax.bar(xi + off, _summ(v, stat), w, yerr=_err(v, stat),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=lab if xi == 0 else None)
        if M['per_trial']:                       # no within-animal test for overfit
            p = _paired_p(sp, te, stat)
            ax.text(xi, max(_summ(sp, stat), _summ(te, stat)) * 1.04 + 0.02,
                    _stars(p), ha='center', fontsize=8)
    ax.axhline(M['ref'], color='k', ls=':', lw=1.4,
               label='chance' if measure == 'proj' else 'on target' if measure == 'peakiness'
               else 'no overfitting')
    ax.set_xticks(x); ax.set_xticklabels([m.replace('mouse_', 'M') for m in mice])
    if measure == 'proj':
        ax.set_ylabel(f'{stat} normalised {_metric_label(short)} loss\n'
                      f'(/ predict-mean; < 1 beats chance)', fontsize=8)
    else:
        ax.set_ylabel((f'{stat} ' if M['per_trial'] else '') + M['ylab'], fontsize=8)
    ax.set_xlabel('mouse')
    ax.legend(fontsize=7, frameon=True)
    if measure == 'proj':
        wtag = ('own stored weighting' if weighting == 'own' else 'common evar weighting')
        ctx = f" ({wtag})" + (' — cell trained on KL, scored here on projection'
                              if _is_klref(short) else '')
        test = 'Wilcoxon' if stat == 'median' else 'paired t'
        stat_note = (f'  Star = within-mouse {test} (n = trials).'
                     + ('' if stat == 'median' else '  Bars are MEANS: heavy-tailed under '
                        'flat/MSE — cross-check with --trial-stat median.'))
    elif measure == 'peakiness':
        ctx = ' (weighting-independent: no PCA basis involved)'
        stat_note = (f"  Star = within-mouse "
                     f"{'Wilcoxon' if stat == 'median' else 'paired t'} (n = trials).")
    else:
        ctx = ' (val/train fit-loss at the RESTORED best epoch)'
        stat_note = ('  One value per mouse — no within-animal distribution, so no '
                     'error bar and no within-mouse test here; see the across-mice figure.')
    ax.set_title(f"{title.replace(chr(10), ', ')} — {M['name']}{ctx}.{stat_note}", fontsize=7.4)
    fig.tight_layout()
    stem = (f'projflat_spmouse_{short}'
            + ('' if measure == 'proj' else f'_{measure}')
            + (('_own' if weighting == 'own' else '') if measure == 'proj' else '')
            + ('_median' if stat == 'median' and M['per_trial'] else ''))
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}")


# ------------------------------------------------------ (B) across-mice figure
def fig_across_mice(results_root, out_root, weighting, stat='mean', measure='proj'):
    M = MEASURES[measure]
    ps.apply()
    # Width scales with the config count (9 groups do not fit the 2-col default).
    base_w, base_h = ps.figsize(2, 1)
    fig, ax = plt.subplots(figsize=(max(base_w, 1.45 * len(CONFIGS) + 1.5), base_h + 0.5))
    x = np.arange(len(CONFIGS))
    w = 0.38
    for xi, (title, cell, short) in zip(x, CONFIGS):
        if not have(results_root, cell):
            continue
        pm = per_mouse(results_root, cell, weighting, measure)
        mice = sorted(pm)
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
        # per-mouse points overlaid (paired), so the n=6 is visible
        for a_, off in [(sp, -w / 2), (te, +w / 2)]:
            ax.plot(np.full_like(a_, xi + off), a_, 'o', ms=2.5, color='0.25',
                    alpha=0.6, zorder=4)
        top = max(sp.mean(), te.mean())
        ax.text(xi, top * 1.04 + 0.03, f'{_stars(p)}\n{n_temp}/{len(mice)}',
                ha='center', fontsize=7)
    ax.axhline(M['ref'], color='k', ls=':', lw=1.4,
               label='chance' if measure == 'proj' else 'on target' if measure == 'peakiness'
               else 'no overfitting')
    # Faint rules between the weighting blocks (evar | flat | KL-trained).
    for b in range(1, len(CONFIGS) // 3):
        ax.axvline(3 * b - 0.5, color='0.75', lw=0.9, ls='-', zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=7)
    if measure == 'proj':
        ax.set_ylabel(f'normalised projection loss\n(per-mouse {stat}; / predict-mean; '
                      f'< 1 beats chance)', fontsize=8)
    else:
        ax.set_ylabel((f'per-mouse {stat} ' if M['per_trial'] else '') + M['ylab'],
                      fontsize=8)
    ax.legend(fontsize=7, frameon=True)
    if measure == 'proj':
        if weighting == 'own':
            wtag = ', own stored weighting'
            caveat = ('Own weighting: flat cells scored as MSE, evar AND KL-trained cells '
                      'eigenvalue-weighted — so compare spatial-vs-temporal WITHIN a config, not bar '
                      'heights ACROSS configs (use --weighting common for that).')
        else:
            wtag = ', common evar weighting'
            caveat = ('Common weighting: every cell rescored under one evar basis, so bar heights ARE '
                      'comparable across configs.')
        tail = ('' if stat == 'median' else
                '  Per-mouse MEANS are heavy-tailed under flat/MSE (the retired "worse than chance" '
                'artefact) — cross-check with --trial-stat median.')
    elif measure == 'peakiness':
        wtag = ''
        caveat = ('Peakiness needs no PCA basis, so it is weighting-independent and bar heights ARE '
                  'comparable across all nine configs.')
        tail = ''
    else:
        wtag = ''
        caveat = ('Overfitting is val/train fit-loss at the RESTORED best epoch, one value per '
                  'mouse — the n=6 points shown ARE the whole distribution.')
        tail = ''
    ax.set_title(f"Spatial vs temporal ACROSS MICE (paired t over n=6), {M['name']}{wtag}"
                 f"{f', per-mouse {stat}' if M['per_trial'] else ''}. "
                 f'Star = paired t; n/6 = mice where temporal is lower.\n'
                 f'{caveat}  Within each block: lin -> rr8 adds the RANK bottleneck, rr8 -> h8 adds '
                 f'the tanh. KL-trained cells are scored on a metric they were NOT trained on.{tail}',
                 fontsize=7.1)
    fig.tight_layout()
    # The weighting suffix is only meaningful for the projection loss; peakiness and
    # overfitting never touch a PCA basis, so tagging them '_common'/'_own' would
    # imply a distinction that does not exist.
    stem = ('projflat_spat_vs_temp_acrossmice'
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
                         "fit-loss at the restored best epoch). Pass several, or 'all' "
                         "via --measure proj peakiness overfit.")
    a = ap.parse_args()
    for measure in a.measure:
        print(f"Spatial vs temporal, {MEASURES[measure]['name']}, "
              f"weighting={a.weighting}, trial-stat={a.trial_stat}\n")
        for title, cell, short in CONFIGS:
            fig_per_mouse(a.results_root, a.out_root, title, cell, short, a.weighting,
                          a.trial_stat, measure)
        fig_across_mice(a.results_root, a.out_root, a.weighting, a.trial_stat, measure)


if __name__ == '__main__':
    main()
