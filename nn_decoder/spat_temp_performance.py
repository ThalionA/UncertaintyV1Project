# -*- coding: utf-8 -*-
"""Spatial vs temporal held-out PERFORMANCE, with paired stats, across cell groups.

Theo's ask (2026-06-09): compare spatial (PPC) vs temporal (SBC) performance for
EVERYTHING — all losses, all variants — with statistics. This is the unified
engine for the bar-form axes; the hidden-width axis (skill vs H) lives in
`plot_overfit_vs_width.py`.

For each cell we score the held-out decoded posteriors under a common metric,
normalise to that cell's own shuffle control (skill = loss / shuffle; <1 beats
chance), per mouse, then compare spatial vs temporal with a paired t (n = mice):

  * KL  — the discriminating metric (sees over-confidence);
  * PCA — the blind metric (note §8: scores every variant ≈ equally).

Both metric scorers are `cross_loss_eval._eval_one`, i.e. the exact training-loss
maths. The PCA metric uses one common basis per group (the reference cell's
per-mouse pcs/evar — identical across cells since the targets are shared).

Groups:
  losses    loss_comparison_v1: PCA / CE / KL / JS / Wasserstein (vary slug, one run)
  variants  wm3*: evar / flat-evar / shape λ (vary run, one slug); λ in clean
            Brier-weight units (config --shape-lambda / 100)

Outputs (PNG+SVG) + a stats CSV under figures/spat_temp/:
  spat_temp_<group>_<metric>.png   raw + skill grouped bars, per-mouse dots, paired-t
  spat_temp_<group>_stats.csv      every (cell, arch, metric): skill mean/sem, paired p

Usage:  python spat_temp_performance.py            # both groups
        python spat_temp_performance.py --group losses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import peakiness_style as ps
from cross_loss_eval import _eval_one
from nn_classifier import fit_loss_per_trial


def _kl_per_trial(decoded, target):
    """Per-trial forward-KL (the same fit_loss_per_trial used everywhere)."""
    dec = np.asarray(decoded, float)
    tgt = np.asarray(target, float)
    out = np.full(dec.shape[0], np.nan)
    g = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
    if g.any():
        out[g] = fit_loss_per_trial(torch.tensor(dec[g]), torch.tensor(tgt[g]), 'KL').numpy()
    return out


def _stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'

ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
ARCH_COLOR = {'spat': '#d95f02', 'temp': '#1f78b4'}
METRICS = ['KL', 'PCA']
LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']


def _loss_slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def discover_cells(group, results_root):
    """Return ([(label, run, slug)], (basis_run, basis_slug)) for a group."""
    if group == 'losses':
        run = 'loss_comparison_v1'
        cells = [(loss, run, _loss_slug(loss)) for loss in LOSSES]
        return cells, (run, _loss_slug('PCA'))
    if group == 'variants':
        cells = [('PCA (evar)', 'wm3', 'Q_PCA_half_100ms_all'),
                 ('flat-evar', 'wm3_flatevar', 'Q_PCA_half_100ms_all')]
        shapes = sorted(Path(results_root).glob('wm3_shape*'),
                        key=lambda p: float(p.name.replace('wm3_shape', '').replace('p', '.')))
        for p in shapes:                                   # config knob is 100·λ
            lam = float(p.name.replace('wm3_shape', '').replace('p', '.')) / 100
            cells.append((f'PCA+shape λ={lam:g}', p.name, 'Q_PCA_half_100ms_all'))
        return cells, ('wm3', 'Q_PCA_half_100ms_all')
    raise SystemExit(f'unknown group {group!r} (losses|variants)')


def _load(results_root, run, slug, split):
    f = Path(results_root) / run / slug / f'{split}.mat'
    if not f.is_file():
        return None
    return sio.loadmat(str(f), simplify_cells=True).get('results')


def common_basis(res):
    """Per-mouse (pcs, evar) for the PCA metric, from the group's reference cell."""
    return {mk: (np.asarray(m['Dist']['pcs'], float),
                 np.asarray(m['Dist']['explained_var'], float))
            for mk, m in res.items()}


def collect(results_root, group, split):
    """data[cell][arch][metric] = {real, shuf, skill} (per-mouse lists)."""
    cells, (brun, bslug) = discover_cells(group, results_root)
    bres = _load(results_root, brun, bslug, split)
    if not isinstance(bres, dict):
        raise SystemExit(f'basis/reference cell missing: {brun}/{bslug} ({split}.mat). '
                         'rsync the run down first.')
    basis = common_basis(bres)
    mice = sorted(bres)
    data = {}
    for label, run, slug in cells:
        res = _load(results_root, run, slug, split)
        if not isinstance(res, dict):
            print(f'  [skip] {label}: {run}/{slug} not on disk')
            continue
        data[label] = {a: {m: {'real': [], 'shuf': [], 'skill': []} for m in METRICS}
                       for a, _ in ARCHS}
        for mk in mice:
            if mk not in res:
                continue
            pcs, evar = basis[mk]
            D = res[mk]['Dist']
            for a, _ in ARCHS:
                dec = np.asarray(D[a]['decoded'], float)
                tgt = np.asarray(D[a]['target'], float)
                shf = np.asarray(D[a + '_shf']['decoded'], float)
                for met in METRICS:
                    real = _eval_one(dec, tgt, met, pcs, evar)
                    shuf = _eval_one(shf, tgt, met, pcs, evar)
                    data[label][a][met]['real'].append(real)
                    data[label][a][met]['shuf'].append(shuf)
                    data[label][a][met]['skill'].append(real / shuf if shuf else np.nan)
    if not data:
        raise SystemExit(f'no cells found for group {group!r} under {results_root}/.')
    return data, mice


def _paired_p(s, t):
    s, t = np.asarray(s, float), np.asarray(t, float)
    good = np.isfinite(s) & np.isfinite(t)
    return float(stats.ttest_rel(s[good], t[good]).pvalue) if good.sum() >= 3 else np.nan


def fig_bars(data, met, group, out_dir):
    """Two panels (raw, skill): spat vs temp grouped by cell, per-mouse dots,
    paired-t spat-vs-temp p annotated above each cell."""
    names = list(data)
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    for ax, key, ylab in ((axes[0], 'real', f'{met} loss (raw)'),
                          (axes[1], 'skill', f'{met} skill = loss / shuffle')):
        for i, name in enumerate(names):
            for j, (a, alab) in enumerate(ARCHS):
                v = np.asarray(data[name][a][met][key], float)
                xpos = i + (j - 0.5) * 0.36
                ax.bar(xpos, np.nanmean(v), 0.34, color=ARCH_COLOR[a],
                       alpha=0.45 if j == 0 else 0.85, hatch='' if j == 0 else '//',
                       yerr=np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()),
                       capsize=3, label=(alab if i == 0 else None))
                ax.scatter(np.full(len(v), xpos), v, color='k', s=10, zorder=3, alpha=0.55)
        if key == 'skill':
            ax.axhline(1.0, ls=':', color='0.5', lw=1, label='chance')
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.22 * (y1 - y0))
        pty = y1 + 0.05 * (y1 - y0)
        for i, name in enumerate(names):
            p = _paired_p(data[name]['spat'][met][key], data[name]['temp'][met][key])
            if np.isfinite(p):
                star = '*' if p < 0.05 else ''
                ax.text(i, pty, f'p={p:.3f}{star}', ha='center', fontsize=7.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel(ylab)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    ps.label_panels(axes)
    fig.suptitle(f'Spatial vs temporal performance — {group} ({met} metric, n=6 mice, paired-t)',
                 y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_dir), f'spat_temp_{group}_{met}')


def collect_per_trial(results_root, group, split):
    """ptdata[cell][mouse][arch] = per-trial KL-skill (per-trial KL / that mouse's
    shuffle-mean KL). For the per-mouse (n=trials) view."""
    cells, (brun, bslug) = discover_cells(group, results_root)
    bres = _load(results_root, brun, bslug, split)
    if not isinstance(bres, dict):
        raise SystemExit(f'reference cell missing: {brun}/{bslug}.')
    mice = sorted(bres)
    ptdata = {}
    for label, run, slug in cells:
        res = _load(results_root, run, slug, split)
        if not isinstance(res, dict):
            continue
        ptdata[label] = {}
        for mk in mice:
            if mk not in res:
                continue
            D = res[mk]['Dist']
            cell = {}
            for a, _ in ARCHS:
                kl = _kl_per_trial(D[a]['decoded'], D[a]['target'])
                shuf = np.nanmean(_kl_per_trial(D[a + '_shf']['decoded'], D[a + '_shf']['target']))
                cell[a] = kl / shuf if shuf else kl * np.nan
            ptdata[label][mk] = cell
    return ptdata, mice


def fig_per_mouse(ptdata, mice, group, out_dir):
    """Per-mouse spatial-vs-temporal KL-skill (bars = that mouse's mean over its
    trials; star = TRIAL-level paired-t for that mouse), one panel per cell, with
    a hatched aggregate bar (mean ± sem over the 6 mice; star = across-mouse
    paired-t, n=6) — so within-animal (n=trials) and across-animal (n=6) sit side
    by side for every loss / λ-variant."""
    cells = list(ptdata)
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=ps.figsize(n, 1, panel_w=2.9),
                             squeeze=False, sharey=True)
    nm = len(mice)
    xpos = list(range(nm)) + [nm + 0.7]
    xlabs = [mk.replace('mouse_', 'm') for mk in mice] + ['agg\n(n=6)']
    for c, cell in enumerate(cells):
        ax = axes[0][c]
        per_s, per_t = [], []
        for j, mk in enumerate(mice):
            d = ptdata[cell].get(mk)
            if d is None:
                continue
            s, t = np.asarray(d['spat'], float), np.asarray(d['temp'], float)
            ms, mt = np.nanmean(s), np.nanmean(t)
            per_s.append(ms); per_t.append(mt)
            ax.bar(j - 0.19, ms, 0.36, color=ARCH_COLOR['spat'], alpha=0.85)
            ax.bar(j + 0.19, mt, 0.36, color=ARCH_COLOR['temp'], alpha=0.85)
            gd = np.isfinite(s) & np.isfinite(t)
            if gd.sum() >= 3:
                st = _stars(stats.ttest_rel(s[gd], t[gd]).pvalue)
                if st != 'ns':
                    ax.text(j, max(ms, mt), st, ha='center', va='bottom', fontsize=7)
        xa = nm + 0.7                                  # aggregate (n=6 mice)
        for off, arch, vals in ((-0.19, 'spat', per_s), (0.19, 'temp', per_t)):
            ax.bar(xa + off, np.nanmean(vals), 0.36, color=ARCH_COLOR[arch], alpha=0.45,
                   hatch='//', yerr=np.nanstd(vals, ddof=1) / np.sqrt(len(vals)), capsize=2)
        pa = _paired_p(per_s, per_t)
        if np.isfinite(pa):
            ax.text(xa, max(np.nanmean(per_s), np.nanmean(per_t)), _stars(pa),
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.axhline(1.0, ls=':', color='0.5', lw=1)
        ax.set_xticks(xpos); ax.set_xticklabels(xlabs, fontsize=7)
        ax.set_title(cell, fontsize=9.5)
        if c == 0:
            ax.set_ylabel('KL skill = loss / shuffle\n(<1 beats chance)')
    axes[0][0].legend(handles=[Patch(color=ARCH_COLOR['spat'], label='spatial'),
                               Patch(color=ARCH_COLOR['temp'], label='temporal')],
                      fontsize=8, loc='best')
    ps.label_panels(axes)
    fig.suptitle(f'Per-mouse (n=trials, trial-level paired-t stars) + aggregate '
                 f'(n=6, bold star) — {group}, KL skill', y=1.02)
    ps.save_fig(fig, Path(out_dir), f'per_mouse_{group}')


def write_csv(data, group, out_dir):
    rows = ['cell,arch,metric,skill_mean,skill_sem,raw_mean,spat_vs_temp_paired_p']
    for name in data:
        for met in METRICS:
            p = _paired_p(data[name]['spat'][met]['skill'], data[name]['temp'][met]['skill'])
            for a, _ in ARCHS:
                sk = np.asarray(data[name][a][met]['skill'], float)
                rw = np.asarray(data[name][a][met]['real'], float)
                rows.append(f'{name},{a},{met},{np.nanmean(sk):.4f},'
                            f'{np.nanstd(sk, ddof=1)/np.sqrt(np.isfinite(sk).sum()):.4f},'
                            f'{np.nanmean(rw):.4f},{p:.4f}')
    out = Path(out_dir) / f'spat_temp_{group}_stats.csv'
    out.write_text('\n'.join(rows) + '\n')
    print(f'  -> {out.name}')


def main(results_root, group, split, out_root):
    ps.apply()
    groups = ['losses', 'variants'] if group == 'all' else [group]
    for g in groups:
        data, mice = collect(results_root, g, split)
        print(f'[{g}] cells={list(data)}  mice={len(mice)}')
        for met in METRICS:
            print(f'  -- {met} (spat / temp skill, Δ=spat−temp, paired-t):')
            for name in data:
                s = np.nanmean(data[name]['spat'][met]['skill'])
                t = np.nanmean(data[name]['temp'][met]['skill'])
                p = _paired_p(data[name]['spat'][met]['skill'], data[name]['temp'][met]['skill'])
                print(f'     {name:18s} {s:.3f} / {t:.3f}  Δ={s-t:+.3f}  p={p:.3f}'
                      f'{" *" if np.isfinite(p) and p < 0.05 else ""}')
            fig_bars(data, met, g, out_root)
        write_csv(data, g, out_root)
        ptdata, _ = collect_per_trial(results_root, g, split)
        fig_per_mouse(ptdata, mice, g, out_root)
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--group', default='all', choices=['losses', 'variants', 'all'])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/spat_temp')
    a = ap.parse_args()
    main(a.results_root, a.group, a.split, a.out_root)
