# -*- coding: utf-8 -*-
"""Spatial vs temporal at the selected projection-loss cell — every metric, every normalisation.

Cell: projection-based loss with shape_lambda = 30 (lambda_Brier = 0.3), H=8, dropout 0,
patience 0 (hpsweep_v2). Chosen by restricting the search to projection-loss cells: it is the only
projection configuration that beats chance on both metrics and both architectures while landing
peakiness on the IO target.

ONE FIGURE PER SCORING METRIC (projection-based first, then KL), each 3 rows x 2 cols:
  rows = normalisation:  raw (unnormalised)  /  divided by the SHUFFLE control  /  divided by the
         leave-one-out PREDICT-MEAN null
  cols = level:          across animals (paired over the 6 mice)  /  within each animal (paired
         over that animal's held-out trials)

All divisors are per-mouse scalars applied identically to both architectures. Consequence worth
knowing when reading the figure: the WITHIN-animal t and p are therefore IDENTICAL down a column —
only the units change. The ACROSS-animal test does change with normalisation, because each mouse
gets its own divisor and that reweights the animals relative to one another.

Shuffle null convention follows cross_loss_eval: the shuffle-trained decoder's predictions scored
against its own (shuffled) target.

Stats are drawn ON the figure: paired t across animals (n=6 and n=5 excluding mouse_2), per-mouse
stars within animals, and a mixed-effects estimate (per-trial loss ~ arch + (1|mouse)).

Outputs (PNG+SVG) under figures/hparam_summary/: spat_temp_best_cell_<metric>
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
LOSS = 'PCA'
METRICS = [('PCA', 'Projection-based'), ('KL', 'KL')]
NORMS = [('raw', 'raw loss'), ('shf', '÷ shuffle'), ('pm', 'normalised loss (÷ predict-mean)')]
EXCLUDE = 2


def _per_trial(dec, tgt, metric, pcs_t, evar_t):
    return np.asarray(fit_loss_per_trial(torch.tensor(dec), torch.tensor(tgt),
                                         metric, pcs_t, evar_t).detach().cpu(), float)


_UNSET = object()


def load(results_root, metric, run=None, cell=None, loss=_UNSET, return_extras=False):
    """{(mouse, arch, norm): per-trial loss array}.

    ``run``/``cell``/``loss`` default to the module-level constants (what the CLI
    flags set), so existing callers are unchanged; pass them explicitly to reuse
    this loader from another script (e.g. spat_temp_by_lambda). ``loss=None``
    means "the cell dir holds exactly one loss slug — discover it".

    ``return_extras=True`` additionally returns ``extras[(m, arch)] = {'decoded',
    'full_decoded' (None if the run didn't save it), 'ok'}`` so a caller can map
    the (ok-masked) per-trial losses back to export-order trial indices by
    matching decoded rows into full_decoded (see spat_temp_by_state.py).
    """
    res = _results(results_root, run or RUN, cell or CELL,
                   loss=(LOSS if loss is _UNSET else loss))
    mice = sorted(int(k.split('_')[-1]) for k in res if str(k).startswith('mouse_'))
    out, extras = {}, {}
    for m in mice:
        D = res[f'mouse_{m}']['Dist']
        pcs_t = evar_t = None
        if metric == 'PCA':
            pcs_t = torch.tensor(np.asarray(D['pcs'], float))
            evar_t = torch.tensor(np.asarray(D['explained_var'], float))
        for arch in ('spat', 'temp'):
            dec = np.asarray(D[arch]['decoded'], float)
            tgt = np.asarray(D[arch]['target'], float)
            ok = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
            v = _per_trial(dec, tgt, metric, pcs_t, evar_t)[ok]
            # divisor 1: shuffle control (its own predictions vs its own target)
            shf = arch + '_shf'
            sdec = np.asarray(D[shf]['decoded'], float)
            stgt = np.asarray(D[shf].get('target', tgt), float)
            sok = np.isfinite(sdec).all(1) & np.isfinite(stgt).all(1)
            d_shf = np.nanmean(_per_trial(sdec, stgt, metric, pcs_t, evar_t)[sok])
            # divisor 2: leave-one-out predict-mean
            n_ok = int(ok.sum()); tot = tgt[ok].sum(axis=0)
            pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
            if n_ok > 1:
                pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
            d_pm = np.nanmean(_per_trial(pm, tgt, metric, pcs_t, evar_t)[ok])
            out[(m, arch, 'raw')] = v
            out[(m, arch, 'shf')] = v / d_shf
            out[(m, arch, 'pm')] = v / d_pm
            if return_extras:
                fd = D[arch].get('full_decoded')
                extras[(m, arch)] = {'decoded': dec, 'ok': ok,
                                     'full_decoded': None if fd is None else np.asarray(fd, float)}
    return (out, mice, extras) if return_extras else (out, mice)


def mixedlm_arch(df, method='powell'):
    """β/SE/p for temporal-vs-spatial from ``loss ~ arch + (1|mouse)``, REML.

    ``df`` needs columns ``loss``/``arch``/``mouse``. Returns ``(beta, se, p,
    warned)`` where ``warned`` is True when statsmodels raised a
    ConvergenceWarning. Default optimiser is powell, NOT statsmodels' bfgs:
    bfgs fails its gradient check on these per-trial losses and raises
    convergence warnings that are pure optimiser artefacts, while powell
    converges clean wherever the fit is healthy — so a warning that survives
    powell is a genuine group-variance boundary and should be flagged wherever
    the numbers are shown. β/SE/p themselves are optimiser-robust to 4 dp
    either way (verified on io_hmm_v3 kl_h8_lh0, raw+pm, and the by-state
    engaged/other subsets). Shared with spat_temp_by_state /
    spat_temp_by_lambda — keep the model here, don't copy."""
    import warnings
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        fit = smf.mixedlm('loss ~ arch', df, groups=df['mouse']).fit(
            reml=True, **({} if method is None else {'method': method}))
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return (fit.params['arch[T.temp]'], fit.bse['arch[T.temp]'],
            fit.pvalues['arch[T.temp]'], warned)


def across_panel(a, sp, tp, mice, exclude=EXCLUDE, chance_line=False):
    """Across-animals paired spat-vs-temp panel: per-mouse lines (``exclude`` in
    red), mean ± SEM, and the paired t (all mice / excluding ``exclude``) drawn on
    the axes. ``sp``/``tp`` are per-mouse mean losses in the order of ``mice``.
    Shared with spat_temp_by_state.py — keep the drawing here, don't copy it."""
    keep = [i for i, m in enumerate(mice) if m != exclude]
    for i, m in enumerate(mice):
        a.plot([0, 1], [sp[i], tp[i]], '-o', ms=4, lw=1.2,
               color='#cb181d' if m == exclude else '0.62', zorder=4 if m == exclude else 2)
    a.errorbar([0, 1], [sp.mean(), tp.mean()],
               yerr=[sp.std(ddof=1) / np.sqrt(sp.size), tp.std(ddof=1) / np.sqrt(tp.size)],
               color='k', lw=2.4, marker='o', ms=7, capsize=4, zorder=5)
    tA, pA = sstats.ttest_rel(tp, sp)
    tK, pK = sstats.ttest_rel(tp[keep], sp[keep])
    a.text(0.03, 0.03,
           f'n={len(mice)}: t({len(mice) - 1})={tA:+.2f}, p={pA:.3f}\n'
           f'n={len(keep)}: t({len(keep) - 1})={tK:+.2f}, p={pK:.3f}',
           transform=a.transAxes, fontsize=6, va='bottom', ha='left',
           bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    if chance_line:
        a.axhline(1.0, color='0.4', ls=':', lw=1.2)
    a.set_xticks([0, 1]); a.set_xticklabels(['spatial', 'temporal'])
    a.set_xlim(-0.35, 1.35)


def figure(data, mice, metric, mlab, out_root):
    ps.apply()
    fig, ax = plt.subplots(len(NORMS), 2, figsize=ps.figsize(2, len(NORMS)))
    for r, (nk, nlab) in enumerate(NORMS):
        sp = np.array([data[(m, 'spat', nk)].mean() for m in mice])
        tp = np.array([data[(m, 'temp', nk)].mean() for m in mice])

        # ---- across animals -------------------------------------------------
        a = ax[r][0]
        across_panel(a, sp, tp, mice, chance_line=(nk != 'raw'))
        a.set_ylabel(f'{nlab}', fontsize=8)
        if r == 0:
            a.set_title('across animals', fontsize=9)

        # ---- within animals -------------------------------------------------
        b = ax[r][1]
        diffs, sems, pv = [], [], []
        for m in mice:
            x, y = data[(m, 'spat', nk)], data[(m, 'temp', nk)]
            n = min(x.size, y.size); d = y[:n] - x[:n]
            t, p = sstats.ttest_rel(y[:n], x[:n])
            diffs.append(d.mean()); sems.append(d.std(ddof=1) / np.sqrt(d.size)); pv.append(p)
        bars = b.bar(range(len(mice)), diffs, 0.66, yerr=sems, capsize=3,
                     color=['#cb181d' if m == EXCLUDE else '0.45' for m in mice],
                     edgecolor='k', lw=0.4)
        for bb, p, d in zip(bars, pv, diffs):
            s = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            if s:
                b.text(bb.get_x() + bb.get_width() / 2, d, s, ha='center',
                       va='bottom' if d >= 0 else 'top', fontsize=8)
        b.axhline(0, color='k', lw=1.0)
        b.set_xticks(range(len(mice))); b.set_xticklabels([f'M{m}' for m in mice], fontsize=7)
        b.set_ylabel('Δ (temporal − spatial)', fontsize=8)
        if r == 0:
            b.set_title('within each animal', fontsize=9)
        # mixed-effects estimate on this normalisation
        try:
            rows = [pd.DataFrame({'loss': data[(m, arch, nk)], 'arch': arch, 'mouse': m})
                    for m in mice for arch in ('spat', 'temp')]
            c, se, p, warned = mixedlm_arch(pd.concat(rows, ignore_index=True))
            b.text(0.03, 0.03, f'mixedlm β={c:+.3f}\n(SE {se:.3f}), p={p:.3f}'
                   + (' [boundary]' if warned else ''),
                   transform=b.transAxes, fontsize=6, va='bottom', ha='left',
                   bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
        except Exception:
            pass
    ax[-1][0].set_xlabel('architecture', fontsize=8)
    ax[-1][1].set_xlabel('mouse', fontsize=8)
    ax[0][0].legend(handles=[
        Line2D([0], [0], color='0.62', lw=1.2, marker='o', ms=4, label='mouse'),
        Line2D([0], [0], color='#cb181d', lw=1.4, marker='o', ms=4, label=f'mouse {EXCLUDE}'),
        Line2D([0], [0], color='k', lw=2.4, marker='o', ms=6, label='mean ± SEM')],
        fontsize=6, frameon=True)
    ps.label_panels(ax.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), f'spat_temp_{RUN}_{CELL}_{metric}' if RUN != 'hpsweep_v2' else f'spat_temp_best_cell_{metric}')


def main(results_root, out_root):
    print(f'cell: {RUN}/{CELL}  (trained with {LOSS})')
    print('NB within-animal t/p are identical down a normalisation column — the divisor is one '
          'per-mouse constant applied to both arches, so it sets units only.\n')
    for metric, mlab in METRICS:
        data, mice = load(results_root, metric)
        figure(data, mice, metric, mlab, out_root)
        print(f'== {mlab} metric ==')
        for nk, nlab in NORMS:
            sp = np.array([data[(m, 'spat', nk)].mean() for m in mice])
            tp = np.array([data[(m, 'temp', nk)].mean() for m in mice])
            keep = [i for i, m in enumerate(mice) if m != EXCLUDE]
            p6 = sstats.ttest_rel(tp, sp).pvalue
            p5 = sstats.ttest_rel(tp[keep], sp[keep]).pvalue
            print(f'  {nlab:16s} spat {sp.mean():9.4f}  temp {tp.mean():9.4f}  '
                  f'Δ {np.mean(tp - sp):+9.4f}   p(n=6)={p6:.4f}  p(n=5)={p5:.4f}')
        print()
    print(f'Done -> {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hparam_summary')
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--cell', default=CELL)
    ap.add_argument('--loss', default=LOSS,
                    help='loss the cell was TRAINED with (locates the Q_<loss> dir)')
    ap.add_argument('--norms', nargs='+', default=[k for k, _ in NORMS],
                    choices=[k for k, _ in NORMS],
                    help="normalisation rows; pass 'raw pm' to drop the shuffle row "
                         '(architecturally biased for spat-vs-temp, log 2026-07-29)')
    a = ap.parse_args()
    RUN, CELL, LOSS = a.run, a.cell, a.loss
    NORMS = [(k, l) for k, l in NORMS if k in set(a.norms)]
    main(a.results_root, a.out_root)
