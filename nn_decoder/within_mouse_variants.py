# -*- coding: utf-8 -*-
"""Within-mouse PPC-vs-SBC stats for the three loss variants — reusing the
established clean-run machinery (2026-06-03, Theo's request).

No reinvention: the figures are the standard clean-run within-mouse plots
  - dpu.plot_per_mouse_performance_with_stats  -> 1b_PerMouse_Performance (per
    mouse, trial-by-trial paired t, significance stars), shuffle + raw;
  - dpu.plot_normalized_performance_with_lines -> 1_Normalized_Performance_Bars_
    Paired_Stats (across-mouse paired t).
Each variant is loaded with load_run_tree and scored on the COMMON true-evar PCA
basis (swapped in from the evar run) so the variants are comparable. We also
print the within-mouse and across-mouse paired-t p-values (reusing dpu.calc_pca_dist
+ scipy.ttest_rel — the exact stats the figures annotate) since they are only
shown as stars on the SVGs.

Outputs: figures/loss_variants/within_mouse/<variant>/{1b_*,1_*}.svg
         figures/loss_variants/within_mouse_paired_stats.csv

Usage:  python within_mouse_variants.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import decoder_plotting_utils as dpu
from decoder_plotting_utils import (plot_per_mouse_performance_with_stats,
                                    plot_normalized_performance_with_lines,
                                    calc_pca_dist, _variance_denominator_per_mouse,
                                    _which_model_from_res)
from nn_classifier import fit_loss_per_trial


def _kl_per_trial(target, decoded):
    """Per-trial forward-KL (same fit_loss_per_trial used everywhere)."""
    dec = np.asarray(decoded, float); tgt = np.asarray(target, float)
    out = np.full(dec.shape[0], np.nan)
    g = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
    if g.any():
        out[g] = fit_loss_per_trial(torch.tensor(dec[g]), torch.tensor(tgt[g]), 'KL').numpy()
    return out

SLUG = 'Q_PCA_half_100ms_all'
VARIANTS = [('PCA_evar', 'wm3'), ('flat_evar', 'wm3_flatevar'),
            ('PCA_shape_lam10', 'wm3_shape10')]


def _safe_div(num, den):
    return num / den if den and np.isfinite(den) and den != 0 else np.full_like(num, np.nan)


def main(split, results_root, out_root):
    dpu.set_style()
    # common true-evar basis from the evar run
    evar_tree = dpu.load_run_tree('wm3', SLUG, splits=[split], results_root=results_root)
    if split not in evar_tree:
        raise SystemExit('evar run wm3 not found.')
    true = {mk: (m['Dist'].get('pcs'), m['Dist'].get('explained_var'))
            for mk, m in evar_tree[split]['results'].items()}

    def sig(p):
        return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'

    def per_trial(metric, d, arch):
        pcs, ev = d.get('pcs'), d.get('explained_var')
        if metric == 'PCA':
            return calc_pca_dist(d[arch]['target'], d[arch]['decoded'], pcs, ev)
        return _kl_per_trial(d[arch]['target'], d[arch]['decoded'])

    def shuf_mean(metric, d, arch):
        if metric == 'PCA':
            return np.nanmean(calc_pca_dist(d[arch + '_shf']['target'], d[arch + '_shf']['decoded'],
                                            d.get('pcs'), d.get('explained_var')))
        return np.nanmean(_kl_per_trial(d[arch + '_shf']['target'], d[arch + '_shf']['decoded']))

    # generate the standard clean-run within-mouse figures (PCA distance) once,
    # with the common true-evar basis swapped in
    trees = {}
    for vname, run in VARIANTS:
        tree = dpu.load_run_tree(run, SLUG, splits=[split], results_root=results_root)
        if split not in tree:
            print(f'  [skip] {vname}: {run} not found'); continue
        for mk, m in tree[split]['results'].items():
            if mk in true:
                m['Dist']['pcs'], m['Dist']['explained_var'] = true[mk]
        trees[vname] = tree
        vdir = Path(out_root) / 'within_mouse' / vname
        vdir.mkdir(parents=True, exist_ok=True)
        # all three established normalisations, same as the clean-run plots
        for norm in ('shuffle', 'variance', 'raw'):
            plot_per_mouse_performance_with_stats(tree, [split], out_dir=str(vdir), normalize=norm)
            plot_normalized_performance_with_lines(tree, [split], out_dir=str(vdir), normalize=norm)

    def denom(metric, norm, d, mk, m, which_model):
        """Per-arch normalisation denominator (returns (spat_denom, temp_denom))."""
        if norm == 'raw':
            return 1.0, 1.0
        if norm == 'variance':   # PCA-metric only, architecture-independent
            v = _variance_denominator_per_mouse(mk, m, which_model, split)
            return v, v
        return shuf_mean(metric, d, 'spat'), shuf_mean(metric, d, 'temp')  # shuffle

    # PCA gets shuffle + variance + raw (variance baseline is PCA-specific);
    # KL gets shuffle + raw. Matches what the established plotters support.
    COMBOS = [('PCA', 'shuffle'), ('PCA', 'variance'), ('PCA', 'raw'),
              ('KL', 'shuffle'), ('KL', 'raw')]
    rows = ['metric,norm,variant,scope,mouse,n,ppc,sbc,diff,paired_t_p']
    for metric, norm in COMBOS:
        base = '/shuffle' if norm == 'shuffle' else ('/variance baseline' if norm == 'variance' else ' (raw)')
        print(f'\n=== WITHIN-MOUSE PPC vs SBC — {metric} loss{base}, '
              'trial-level paired t per mouse (common metric)'
              f'{"; <1 beats chance" if norm!="raw" else ""} ===')
        for vname in trees:
            which_model = _which_model_from_res(trees[vname][split])
            print(f'  {vname}:')
            per_ppc, per_sbc = [], []
            for mk, m in trees[vname][split]['results'].items():
                d = m['Dist']
                ds, dt = denom(metric, norm, d, mk, m, which_model)
                ns = _safe_div(per_trial(metric, d, 'spat'), ds)
                nt = _safe_div(per_trial(metric, d, 'temp'), dt)
                g = ~np.isnan(ns) & ~np.isnan(nt)
                p = stats.ttest_rel(ns[g], nt[g]).pvalue if g.sum() > 1 else np.nan
                mp, mt = np.nanmean(ns), np.nanmean(nt)
                per_ppc.append(mp); per_sbc.append(mt)
                print(f'    {mk}: PPC={mp:.2f}  SBC={mt:.2f}  diff={mp-mt:+.2f}  '
                      f'n={int(g.sum())}  p={p:.1e} {sig(p)}')
                rows.append(f'{metric},{norm},{vname},within,{mk},{int(g.sum())},{mp:.4f},{mt:.4f},{mp-mt:.4f},{p:.4e}')
            pa = stats.ttest_rel(per_ppc, per_sbc).pvalue
            nsig = sum(1 for a, b in zip(per_ppc, per_sbc) if b < a)
            print(f'    across-mice (n={len(per_ppc)} paired t): PPC={np.mean(per_ppc):.2f}  '
                  f'SBC={np.mean(per_sbc):.2f}  diff={np.mean(per_ppc)-np.mean(per_sbc):+.2f}  '
                  f'p={pa:.3f} {sig(pa)}  [{nsig}/{len(per_ppc)} mice SBC<PPC]')
            rows.append(f'{metric},{norm},{vname},across,ALL,{len(per_ppc)},{np.mean(per_ppc):.4f},'
                        f'{np.mean(per_sbc):.4f},{np.mean(per_ppc)-np.mean(per_sbc):.4f},{pa:.4e}')
    (Path(out_root) / 'within_mouse_paired_stats.csv').write_text('\n'.join(rows) + '\n')
    print(f'\n  -> figures: {Path(out_root)/"within_mouse"}/<variant>/  '
          '(1b_*=within-mouse, 1_*=across; shuffle/variance/raw; PCA distance)')
    print(f'  -> within_mouse_paired_stats.csv  (PCA+KL × shuffle/variance/raw, within + across)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_variants')
    a = ap.parse_args()
    main(a.split, a.results_root, a.out_root)
