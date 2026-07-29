# -*- coding: utf-8 -*-
"""Spatial vs temporal at the selected hyperparameters — scored under BOTH metrics.

Selected cell: PROJECTION-BASED loss with shape_lambda = 30 (lambda_Brier = 0.3), H=8,
dropout 0, patience 0 (hpsweep_v2). Chosen by restricting the search to projection-loss cells --
the question is what makes the projection loss work, not which loss wins overall. It is the ONLY
projection configuration that beats chance on both metrics and both architectures while landing
peakiness on the IO target (0.057/0.058 vs 0.05943).

Performance is reported under the PROJECTION-BASED metric (the minimum, always) and under KL.
Same decoders, same trials; only the scoring metric differs.

Three tests, all on NORMALISED loss (loss / leave-one-out predict-mean null, so <1 beats chance):

  (i)   ACROSS animals — paired t over the 6 mice, and again excluding mouse_2 (n=5).
  (ii)  WITHIN each animal — a standard paired t-test between the per-trial normalised loss of the
        temporal and spatial decoders, n = that animal's held-out trials. Both decoders are scored
        on the same trials, so they pair one-to-one. Per-trial losses are divided by that mouse's
        mean null loss; because the same divisor applies to both architectures this leaves t and p
        exactly unchanged and only puts the mean difference into interpretable "fraction of chance"
        units.
  (iii) HIERARCHICAL — per-trial normalised loss ~ architecture + (1 | mouse), statsmodels MixedLM.
        The random intercept absorbs each animal's own offset, so the architecture coefficient is
        estimated from within-animal contrasts rather than from pooling trials across animals.

Outputs (PNG+SVG) under figures/hparam_summary/: spat_temp_best_cell
Usage:  python diagnostics/spat_temp_best_cell.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sstats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from nn_classifier import fit_loss_per_trial  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from story_figures import _results  # noqa: E402

RUN = 'hpsweep_v2'
CELL = 'lam0p003_drop0_acttanh_h8_pat0_vf0p2_wd0p0001_shp30'
LOSS = 'PCA'                      # the loss the chosen decoders were TRAINED with
METRICS = [('PCA', 'Projection-based'), ('KL', 'KL')]   # projection first, always
EXCLUDE = 2
ARCH_C = {'spat': '#d95f02', 'temp': '#7570b3'}


def per_trial_normalised(res, mk, arch, metric):
    """Per-trial normalised loss for one mouse/arch/metric.

    Divisor = that mouse's MEAN leave-one-out predict-mean loss. Same divisor for both
    architectures, so a paired test is unaffected; it only sets the units.
    """
    D = res[mk]['Dist']
    dec = np.asarray(D[arch]['decoded'], float)
    tgt = np.asarray(D[arch]['target'], float)
    ok = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
    pcs_t = evar_t = None
    if metric == 'PCA':
        pcs_t = torch.tensor(np.asarray(D['pcs'], float))
        evar_t = torch.tensor(np.asarray(D['explained_var'], float))
    v = np.asarray(fit_loss_per_trial(torch.tensor(dec), torch.tensor(tgt),
                                      metric, pcs_t, evar_t).detach().cpu(), float)
    # leave-one-out predict-mean null, per trial, then its mean as the divisor
    n_ok = int(ok.sum())
    tot = tgt[ok].sum(axis=0)
    pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
    if n_ok > 1:
        pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
    nul = np.asarray(fit_loss_per_trial(torch.tensor(pm), torch.tensor(tgt),
                                        metric, pcs_t, evar_t).detach().cpu(), float)
    div = np.nanmean(nul[ok])
    return (v / div)[ok]


def main(results_root, out_root):
    ps.apply()
    res = _results(results_root, RUN, CELL, loss=LOSS)
    mice = sorted(int(k.split('_')[-1]) for k in res if str(k).startswith('mouse_'))

    data = {}          # (metric, mouse, arch) -> per-trial normalised loss
    for metric, _ in METRICS:
        for m in mice:
            for arch in ('spat', 'temp'):
                data[(metric, m, arch)] = per_trial_normalised(res, f'mouse_{m}', arch, metric)

    fig, ax = plt.subplots(2, 2, figsize=ps.figsize(2, 2))
    for r, (metric, mlab) in enumerate(METRICS):
        # ---- (i) across animals: per-mouse means, paired ----
        sp = np.array([data[(metric, m, 'spat')].mean() for m in mice])
        tp = np.array([data[(metric, m, 'temp')].mean() for m in mice])
        for i, m in enumerate(mice):
            ax[r][0].plot([0, 1], [sp[i], tp[i]], '-o', ms=4, lw=1.2,
                          color='#cb181d' if m == EXCLUDE else '0.62',
                          zorder=4 if m == EXCLUDE else 2)
        ax[r][0].errorbar([0, 1], [sp.mean(), tp.mean()],
                          yerr=[sp.std(ddof=1) / np.sqrt(sp.size), tp.std(ddof=1) / np.sqrt(tp.size)],
                          color='k', lw=2.4, marker='o', ms=7, capsize=4, zorder=5)
        ax[r][0].axhline(1.0, color='0.4', ls=':', lw=1.2)
        ax[r][0].set_xticks([0, 1]); ax[r][0].set_xticklabels(['spatial', 'temporal'])
        ax[r][0].set_xlim(-0.35, 1.35)
        ax[r][0].set_ylabel(f'{mlab}\nnormalised loss', fontsize=8)
        if r == 0:
            ax[r][0].set_title('across animals', fontsize=9)

        # ---- (ii) within animals: paired t on per-trial normalised loss ----
        diffs, sems, ps_ = [], [], []
        for m in mice:
            a, b = data[(metric, m, 'spat')], data[(metric, m, 'temp')]
            n = min(a.size, b.size)
            d = b[:n] - a[:n]                       # temporal - spatial
            t, p = sstats.ttest_rel(b[:n], a[:n])
            diffs.append(d.mean()); sems.append(d.std(ddof=1) / np.sqrt(d.size)); ps_.append(p)
        bars = ax[r][1].bar(range(len(mice)), diffs, 0.66, yerr=sems, capsize=3,
                            color=['#cb181d' if m == EXCLUDE else '0.45' for m in mice],
                            edgecolor='k', lw=0.4)
        for b_, p, d in zip(bars, ps_, diffs):
            s = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            if s:
                ax[r][1].text(b_.get_x() + b_.get_width() / 2, d, s, ha='center',
                              va='bottom' if d >= 0 else 'top', fontsize=8)
        ax[r][1].axhline(0, color='k', lw=1.0)
        ax[r][1].set_xticks(range(len(mice)))
        ax[r][1].set_xticklabels([f'M{m}' for m in mice], fontsize=7)
        ax[r][1].set_ylabel('Δ normalised loss\n(temporal − spatial)', fontsize=8)
        if r == 0:
            ax[r][1].set_title('within each animal', fontsize=9)
        if r == 1:
            ax[r][0].set_xlabel('architecture', fontsize=8)
            ax[r][1].set_xlabel('mouse', fontsize=8)

    ax[0][0].legend(handles=[
        Line2D([0], [0], color='0.62', lw=1.2, marker='o', ms=4, label='mouse'),
        Line2D([0], [0], color='#cb181d', lw=1.4, marker='o', ms=4, label=f'mouse {EXCLUDE}'),
        Line2D([0], [0], color='k', lw=2.4, marker='o', ms=6, label='mean ± SEM'),
        Line2D([0], [0], color='0.4', ls=':', lw=1.2, label='chance')],
        fontsize=6, frameon=True)
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'spat_temp_best_cell')

    # ------------------------------------------------------------------ stats
    print(f'cell: {RUN}/{CELL}  (trained with {LOSS})\n')
    for metric, mlab in METRICS:
        sp = np.array([data[(metric, m, 'spat')].mean() for m in mice])
        tp = np.array([data[(metric, m, 'temp')].mean() for m in mice])
        keep = [i for i, m in enumerate(mice) if m != EXCLUDE]
        t6, p6 = sstats.ttest_rel(tp, sp)
        t5, p5 = sstats.ttest_rel(tp[keep], sp[keep])
        print(f'== {mlab} metric ==')
        print(f'  ACROSS animals  spat {sp.mean():.3f}±{sp.std(ddof=1)/np.sqrt(6):.3f}   '
              f'temp {tp.mean():.3f}±{tp.std(ddof=1)/np.sqrt(6):.3f}   '
              f'Δ {np.mean(tp-sp):+.3f}  t(5)={t6:+.2f} p={p6:.4f}')
        print(f'  excl mouse_{EXCLUDE} (n=5)                                        '
              f'Δ {np.mean(tp[keep]-sp[keep]):+.3f}  t(4)={t5:+.2f} p={p5:.4f}')
        print(f'  WITHIN animals (paired t on per-trial normalised loss)')
        print(f'    {"mouse":6s}{"n":>6s}{"spat":>9s}{"temp":>9s}{"Δ":>9s}{"dz":>7s}{"t":>8s}{"p":>11s}')
        for m in mice:
            a, b = data[(metric, m, 'spat')], data[(metric, m, 'temp')]
            n = min(a.size, b.size); d = b[:n] - a[:n]
            t, p = sstats.ttest_rel(b[:n], a[:n])
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            print(f'    {m:<6d}{n:6d}{a[:n].mean():9.3f}{b[:n].mean():9.3f}'
                  f'{d.mean():+9.3f}{dz:+7.2f}{t:+8.2f}{p:11.2e}')
        # ---- (iii) hierarchical ----
        try:
            import statsmodels.formula.api as smf
            rows = []
            for m in mice:
                for arch in ('spat', 'temp'):
                    v = data[(metric, m, arch)]
                    rows.append(pd.DataFrame({'loss': v, 'arch': arch, 'mouse': m}))
            df = pd.concat(rows, ignore_index=True)
            fit = smf.mixedlm('loss ~ arch', df, groups=df['mouse']).fit(reml=True)
            coef = fit.params.get('arch[T.temp]', np.nan)
            se = fit.bse.get('arch[T.temp]', np.nan)
            pv = fit.pvalues.get('arch[T.temp]', np.nan)
            print(f'  HIERARCHICAL  loss ~ arch + (1|mouse)   arch[temp] = {coef:+.4f} '
                  f'(SE {se:.4f}), z={coef/se:+.2f}, p={pv:.4g};  '
                  f'group var = {float(fit.cov_re.iloc[0,0]):.4f}, n = {len(df)}')
        except Exception as exc:
            print(f'  HIERARCHICAL  failed: {type(exc).__name__}: {exc}')
        print()
    print(f'Done -> {Path(out_root).resolve()}/spat_temp_best_cell.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hparam_summary')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
