# -*- coding: utf-8 -*-
"""Spatial vs temporal performance under the PROJECTION LOSS, scored under each
decoder's OWN training weighting (2026-08-04, Theo's request).

Matched hyperparameters throughout (raw input, lambda_H 0, dropout 0, wd 0):
  linear (0 hidden)  x  variance-weighting   ->  lin_raw_EVAR
  linear (0 hidden)  x  flat-weighting        ->  lin_raw_l0_d0_w0
  8 hidden units     x  variance-weighting   ->  h8_raw_EVAR
  8 hidden units     x  flat-weighting        ->  h8_raw_l0_d0_w0

Two deliverables:
  (A) per-mouse, one figure per config: two bars/mouse (spatial, temporal), height
      = mean normalised projection loss (per-trial projection distance / the
      mouse's leave-one-out predict-mean; < 1 beats chance), SEM over trials, star
      = WITHIN-mouse paired t (paired by trial; spat & temp decode the same
      trials, verified identical row-for-row).
  (B) across-mice, ALL configs in ONE figure: two bars/config (spatial, temporal),
      height = mean over the 6 mice, SEM over mice, star = paired t OVER THE 6 MICE
      (the population test — the legitimate, non-pseudoreplicated one).

WEIGHTING: each cell is scored under ITS OWN stored projection weighting —
eigenvalue-weighted for the variance-weighting (evar) cells, uniform = MSE for the
flat cells. Consequence: the metric DIFFERS between flat and evar configs, so
compare spatial vs temporal WITHIN a config, not bar heights ACROSS configs.
(`--weighting common` rescoress everything under one evar weighting instead, which
makes configs comparable but is a different question.)

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_spat_vs_temp_bymouse.py            # own weighting
        python diagnostics/projflat_spat_vs_temp_bymouse.py --weighting common
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
from projflat_report import _res, _mice, have, _common_basis  # noqa: E402
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


def per_mouse(results_root, cell, weighting):
    """{mouse -> (spatial_norm_pertrial, temporal_norm_pertrial)}."""
    r = _res(results_root, cell)
    common = _common_basis(results_root, 'spat') if weighting == 'common' else None
    out = {}
    for m in _mice(r):
        D = r[m]['Dist']
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
def fig_per_mouse(results_root, out_root, title, cell, short, weighting, stat='mean'):
    if not have(results_root, cell):
        print(f"  [skip] {short}: {cell} not downloaded")
        return
    ps.apply()
    pm = per_mouse(results_root, cell, weighting)
    mice = sorted(pm)
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(mice))
    w = 0.38
    for xi, m in zip(x, mice):
        sp, te = pm[m]
        p = _paired_p(sp, te, stat)
        for v, off, colr, lab in [(sp, -w / 2, ps.SPATIAL, 'spatial'),
                                  (te, +w / 2, ps.TEMPORAL, 'temporal')]:
            ax.bar(xi + off, _summ(v, stat), w, yerr=_err(v, stat),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=lab if xi == 0 else None)
        ax.text(xi, max(_summ(sp, stat), _summ(te, stat)) * 1.04 + 0.02, _stars(p),
                ha='center', fontsize=8)
    ax.axhline(1.0, color='k', ls=':', lw=1.4, label='chance')
    ax.set_xticks(x); ax.set_xticklabels([m.replace('mouse_', 'M') for m in mice])
    metric = _metric_label(short)
    ax.set_ylabel(f'{stat} normalised {metric} loss\n(/ predict-mean; < 1 beats chance)', fontsize=8)
    ax.set_xlabel('mouse')
    ax.legend(fontsize=7, frameon=True)
    wtag = ('own stored weighting' if weighting == 'own' else 'common evar weighting')
    note = ' — cell trained on KL, scored here on projection' if _is_klref(short) else ''
    test = 'Wilcoxon' if stat == 'median' else 'paired t'
    tail = ('' if stat == 'median' else
            '  Bars are MEANS: heavy-tailed under flat/MSE — cross-check with --trial-stat median.')
    ax.set_title(f'{title.replace(chr(10), ", ")} — projection loss ({wtag}){note}.  '
                 f'Star = within-mouse {test} (n = trials).{tail}', fontsize=7.4)
    fig.tight_layout()
    stem = (f'projflat_spmouse_{short}' + ('_own' if weighting == 'own' else '')
            + ('_median' if stat == 'median' else ''))
    ps.save_fig(fig, Path(out_root), stem)
    print(f"  {stem}")


# ------------------------------------------------------ (B) across-mice figure
def fig_across_mice(results_root, out_root, weighting, stat='mean'):
    ps.apply()
    # Width scales with the config count (9 groups do not fit the 2-col default).
    base_w, base_h = ps.figsize(2, 1)
    fig, ax = plt.subplots(figsize=(max(base_w, 1.45 * len(CONFIGS) + 1.5), base_h + 0.5))
    x = np.arange(len(CONFIGS))
    w = 0.38
    for xi, (title, cell, short) in zip(x, CONFIGS):
        if not have(results_root, cell):
            continue
        pm = per_mouse(results_root, cell, weighting)
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
    ax.axhline(1.0, color='k', ls=':', lw=1.4, label='chance')
    # Faint rules between the weighting blocks (evar | flat | KL-trained).
    for b in range(1, len(CONFIGS) // 3):
        ax.axvline(3 * b - 0.5, color='0.75', lw=0.9, ls='-', zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=7)
    ax.set_ylabel(f'normalised projection loss\n(per-mouse {stat}; / predict-mean; '
                  f'< 1 beats chance)', fontsize=8)
    ax.legend(fontsize=7, frameon=True)
    if weighting == 'own':
        wtag = 'own stored weighting'
        caveat = ('Own weighting: flat cells scored as MSE, evar AND KL-trained cells '
                  'eigenvalue-weighted — so compare spatial-vs-temporal WITHIN a config, not bar '
                  'heights ACROSS configs (use --weighting common for that).')
    else:
        wtag = 'common evar weighting'
        caveat = ('Common weighting: every cell rescored under one evar basis, so bar heights ARE '
                  'comparable across configs.')
    tail = ('' if stat == 'median' else
            '  Per-mouse MEANS are heavy-tailed under flat/MSE (the retired "worse than chance" '
            'artefact) — cross-check with --trial-stat median.')
    ax.set_title(f'Spatial vs temporal ACROSS MICE (paired t over n=6), projection loss ({wtag}), '
                 f'per-mouse {stat}. Star = paired t; n/6 = mice where temporal wins.\n'
                 f'{caveat}  Within each block: lin -> rr8 adds the RANK bottleneck, rr8 -> h8 adds '
                 f'the tanh. KL-trained cells are scored on a metric they were NOT trained on.{tail}',
                 fontsize=7.1)
    fig.tight_layout()
    stem = ('projflat_spat_vs_temp_acrossmice' + ('_own' if weighting == 'own' else '_common')
            + ('_median' if stat == 'median' else ''))
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
    a = ap.parse_args()
    print(f"Spatial vs temporal, projection loss, weighting={a.weighting}, "
          f"trial-stat={a.trial_stat}\n")
    for title, cell, short in CONFIGS:
        fig_per_mouse(a.results_root, a.out_root, title, cell, short, a.weighting,
                      a.trial_stat)
    fig_across_mice(a.results_root, a.out_root, a.weighting, a.trial_stat)


if __name__ == '__main__':
    main()
