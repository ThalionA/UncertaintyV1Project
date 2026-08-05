# -*- coding: utf-8 -*-
"""Peakiness, overfitting, and SHUFFLE-normalised performance for the four matched
projflat_v1 configurations (2026-08-04, Theo's request).

Configs (all raw input, lambda_H 0, dropout 0, wd 0 — the shared corner):
  linear x variance-weighting · linear x flat · 8-hidden x variance · 8-hidden x flat

  fig A  peakiness    — decoded max-prob per mouse, spatial vs temporal, with the
                        IO target AND the predict-mean line drawn. Over-sharpening
                        above target is what MSE punishes; the predict-mean line
                        shows how broad the chance decoder is.
  fig B  overfitting  — val / train fit-loss per config x arch (n=6, over mice).
  fig C  performance  — normalised to EACH MODEL'S OWN SHUFFLE (its `*_shf` twin,
                        trained on trial-permuted targets) instead of predict-mean.

WHY THE SHUFFLE NULL DIFFERS FROM PREDICT-MEAN. Predict-mean is the strictest null
(the optimal constant posterior). The shuffle null is LOOSER: the shuffled model
still fits scrambled labels and emits misplaced peaks, so its loss is higher and
ratios come out smaller. Both are shown across this project because they disagree,
and the ordering predict-mean < kill-weights < shuffle is a known result here.
Scored under each decoder's OWN training weighting (flat = MSE, evar =
eigenvalue-weighted), matching the previous figures.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_config_axes.py
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from projflat_report import _res, _mice, have, _slug  # noqa: E402
from overfitting_vs_hparams import _overfit_ratio  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial  # noqa: E402

def configs_for(lam):
    """Cells at a given lambda_H. NOTE: the evar (variance-weighting) controls were
    only run at lambda_H=0, so lam>0 yields the FLAT configs only."""
    if lam == '0':
        return [('linear\nvariance-wt', 'lin_raw_EVAR'),
                ('linear\nflat (MSE)',  'lin_raw_l0_d0_w0'),
                ('8 hidden\nvariance-wt', 'h8_raw_EVAR'),
                ('8 hidden\nflat (MSE)',  'h8_raw_l0_d0_w0')]
    return [(f'linear\nflat (MSE)', f'lin_raw_l{lam}_d0_w0'),
            (f'8 hidden\nflat (MSE)', f'h8_raw_l{lam}_d0_w0')]


CONFIGS = configs_for('0')
ARCHS = [('spat', 'spatial', ps.SPATIAL), ('temp', 'temporal', ps.TEMPORAL)]


def _t(x):
    return torch.tensor(np.asarray(x, float))


def _loss(dec, tgt, pcs, evar):
    return fit_loss_per_trial(_t(dec), _t(tgt), 'PCA', _t(pcs), _t(evar)).numpy()


def collect(results_root, cell):
    """Per-mouse arrays for all three axes, scored under the cell's OWN weighting."""
    r = _res(results_root, cell)
    out = {a: dict(peak=[], tgt=[], pm_peak=[], shf=[], pm=[]) for a, _, _ in ARCHS}
    for m in _mice(r):
        D = r[m]['Dist']
        pcs, evar = D.get('pcs'), D.get('explained_var')
        for arch, _, _ in ARCHS:
            dec = np.asarray(D[arch]['decoded'], float)
            tgt = np.asarray(D[arch]['target'], float)
            ok = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
            dec, tgt = dec[ok], tgt[ok]
            n = len(tgt); tot = tgt.sum(0)
            pm = (tot[None, :] - tgt) / (n - 1)          # LOO predict-mean
            num = np.nanmean(_loss(dec, tgt, pcs, evar))
            out[arch]['peak'].append(dec.max(1).mean())
            out[arch]['tgt'].append(tgt.max(1).mean())
            out[arch]['pm_peak'].append(pm.max(1).mean())
            out[arch]['pm'].append(num / np.nanmean(_loss(pm, tgt, pcs, evar)))
            # shuffle null: this model's own *_shf twin
            key = f'{arch}_shf'
            if key in D:
                sd = np.asarray(D[key]['decoded'], float)
                st = np.asarray(D[key].get('target', tgt), float)
                okk = np.isfinite(sd).all(1) & np.isfinite(st).all(1)
                den = np.nanmean(_loss(sd[okk], st[okk], pcs, evar))
                out[arch]['shf'].append(num / den if den > 0 else np.nan)
            else:
                out[arch]['shf'].append(np.nan)
    return {a: {k: np.array(v, float) for k, v in d.items()} for a, d in out.items()}


def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'


def _grouped(ax, data, key, ylab, logy=False):
    """Two bars (spat/temp) per config, per-mouse points, paired t over mice."""
    x = np.arange(len(CONFIGS))
    w = 0.38
    for xi, (lab, cell) in enumerate(CONFIGS):
        d = data[cell]
        vals = {}
        for arch, alab, colr in ARCHS:
            v = d[arch][key]
            vals[arch] = v
            off = -w / 2 if arch == 'spat' else w / 2
            ax.bar(xi + off, np.nanmean(v), w,
                   yerr=np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()),
                   color=colr, edgecolor='k', linewidth=0.5, capsize=3,
                   label=alab if xi == 0 else None)
            ax.plot(np.full(v.size, xi + off), v, 'o', ms=2.5, color='0.25',
                    alpha=0.6, zorder=4)
        ok = np.isfinite(vals['spat']) & np.isfinite(vals['temp'])
        if ok.sum() > 1:
            _, p = ttest_rel(vals['spat'][ok], vals['temp'][ok])
            n_t = int((vals['temp'][ok] < vals['spat'][ok]).sum())
            top = max(np.nanmean(vals['spat']), np.nanmean(vals['temp']))
            ax.text(xi, top * 1.06, f'{_stars(p)}\n{n_t}/{int(ok.sum())}',
                    ha='center', fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=8)
    if logy:
        ax.set_yscale('log')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--lam', default='0', choices=['0', '0p003', '0p01'],
                    help="lambda_H token; evar controls exist only at 0")
    a = ap.parse_args()
    global CONFIGS
    CONFIGS = configs_for(a.lam)
    suffix = '' if a.lam == '0' else f'_lam{a.lam}'
    lamlab = {'0': '0', '0p003': '3e-3', '0p01': '1e-2'}[a.lam]

    missing = [c for _, c in CONFIGS if not have(a.results_root, c)]
    if missing:
        raise SystemExit(f'missing cells: {missing}')
    data = {cell: collect(a.results_root, cell) for _, cell in CONFIGS}

    ps.apply()

    # ---------------- fig A: peakiness ----------------
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    _grouped(ax, data, 'peak', 'decoded peakiness (max-prob)')
    tgt = np.nanmean([data[c]['spat']['tgt'].mean() for _, c in CONFIGS])
    pmk = np.nanmean([data[c]['spat']['pm_peak'].mean() for _, c in CONFIGS])
    ax.axhline(tgt, color='k', ls=':', lw=1.4)
    ax.axhline(pmk, color='0.45', ls='--', lw=1.2)
    ax.text(len(CONFIGS) - 0.45, tgt, ' IO target', va='bottom', ha='right', fontsize=6.5)
    ax.text(len(CONFIGS) - 0.45, pmk, ' predict-mean', va='top', ha='right', fontsize=6.5,
            color='0.45')
    ax.legend(fontsize=7, frameon=True)
    ax.set_title(f'Peakiness (λ_H = {lamlab}) — above the IO target = over-sharpened (what MSE '
                 f'punishes). n=6, points = mice.', fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), f'projflat_cfg_peakiness{suffix}')

    # ---------------- fig B: overfitting ----------------
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    x = np.arange(len(CONFIGS))
    w = 0.38
    for xi, (lab, cell) in enumerate(CONFIGS):
        ck = _slug(a.results_root, cell) / 'checkpoints'
        for arch, alab, colr in ARCHS:
            m_, s_ = _overfit_ratio(ck, arch)
            off = -w / 2 if arch == 'spat' else w / 2
            ax.bar(xi + off, m_, w, yerr=s_, color=colr, edgecolor='k',
                   linewidth=0.5, capsize=3, label=alab if xi == 0 else None)
    ax.axhline(1.0, color='k', ls=':', lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=7.5)
    ax.set_ylabel('val / train fit-loss\n(1 = no overfitting)', fontsize=8)
    ax.set_yscale('log')
    ax.legend(fontsize=7, frameon=True)
    ax.set_title(f'Overfitting (λ_H = {lamlab}) — val ÷ train fit-loss AT THE RESTORED best epoch '
                 f'(not the final epoch). n=6.', fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), f'projflat_cfg_overfitting{suffix}')

    # ---------------- fig C: shuffle-normalised performance ----------------
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    _grouped(axes[0], data, 'shf', 'loss ÷ own shuffle\n(< 1 beats shuffle)')
    axes[0].axhline(1.0, color='k', ls=':', lw=1.4)
    axes[0].set_title('normalised to each model’s OWN SHUFFLE', fontsize=8.5)
    axes[0].legend(fontsize=7, frameon=True)
    _grouped(axes[1], data, 'pm', 'loss ÷ predict-mean\n(< 1 beats chance)')
    axes[1].axhline(1.0, color='k', ls=':', lw=1.4)
    axes[1].set_title('normalised to predict-mean (for comparison)', fontsize=8.5)
    ps.label_panels(axes)
    fig.suptitle('Performance under each decoder’s OWN training weighting (flat = MSE, variance = '
                 'eigenvalue-weighted).\nShuffle is the LOOSER null — it still fits scrambled labels — '
                 f'so ratios are smaller than against predict-mean. Star/n = paired t over 6 mice.  '
                 f'(λ_H = {lamlab})', y=1.03, fontsize=7.8)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), f'projflat_cfg_performance_shuffle{suffix}')

    # console summary
    print(f"{'config':22s}{'arch':6s}{'peak':>8s}{'/tgt':>7s}{'overfit':>9s}"
          f"{'/shuffle':>10s}{'/pm':>8s}")
    for lab, cell in CONFIGS:
        ck = _slug(a.results_root, cell) / 'checkpoints'
        for arch, alab, _ in ARCHS:
            d = data[cell][arch]
            of, _s = _overfit_ratio(ck, arch)
            print(f"{lab.replace(chr(10),' '):22s}{alab:6s}{d['peak'].mean():8.4f}"
                  f"{d['peak'].mean()/d['tgt'].mean():7.2f}{of:9.2f}"
                  f"{np.nanmean(d['shf']):10.3f}{np.nanmean(d['pm']):8.3f}")


if __name__ == '__main__':
    main()
