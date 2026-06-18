# -*- coding: utf-8 -*-
"""Spatial vs temporal head-to-head per loss, with the Mouse-2 leave-one-out and the
per-animal neuron count (2026-06-17, meeting follow-up).

Metric: KL-skill = test KL / shuffle KL, **per arch ÷ its own shuffle** (the chance
floor differs for spat vs temp — see GOTCHAS `spat_shf` asymmetry — so this is the
right scale-free spat-vs-temp comparison). <1 beats chance; lower = better.

Three questions:
  1. Head-to-head spat vs temp for every loss, all mice vs **Mouse-2-excluded** — does
     dropping the flagged animal change the spat/temp story?
  2. Neuron count per animal (from each checkpoint's `model_params['input_size']`).
  3. Does n_neurons relate to skill / the spat-temp gap (n=6, so correlations are
     weak-powered — reported with that caveat)?

Reads `loss_comparison_v1` (matched 5-loss grid, both archs, 6 mice).
NB "Mouse 2" assumed = `mouse_2` (0-indexed .mat key) — per-mouse values printed so
the choice is transparent; switch with --exclude.

Outputs (PNG+SVG) under figures/peakiness_scatter/:  spat_temp_per_animal.png

Usage:  python diagnostics/spat_temp_per_animal.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.stats as sstats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS, 'Wasserstein': ps.WASSERSTEIN}
ARCH_COL = {'spat': '#525252', 'temp': '#d95f02'}


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _skill_one(D, arch):
    shf = arch + '_shf'
    if arch not in D or shf not in D:
        return np.nan
    tgt = np.asarray(D[arch]['target'], float)
    r = _eval_one(np.asarray(D[arch]['decoded'], float), tgt, 'KL', D.get('pcs'), D.get('explained_var'))
    s = _eval_one(np.asarray(D[shf]['decoded'], float), np.asarray(D[shf]['target'], float), 'KL',
                  D.get('pcs'), D.get('explained_var'))
    return r / s if (np.isfinite(r) and np.isfinite(s) and s > 0) else np.nan


def collect(results_root, run, split):
    """skill[loss][arch] = {mouse: KL-skill}; n_neurons{mouse}; mice (sorted)."""
    skill = {l: {'spat': {}, 'temp': {}} for l in LOSSES}
    nneur = {}
    mice = []
    for loss in LOSSES:
        f = Path(results_root) / run / _slug(loss) / f'{split}.mat'
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get('results')
        for mk in sorted(res):
            if mk not in mice:
                mice.append(mk)
            D = res[mk]['Dist']
            skill[loss]['spat'][mk] = _skill_one(D, 'spat')
            skill[loss]['temp'][mk] = _skill_one(D, 'temp')
            if mk not in nneur:                       # n_neurons from the checkpoint (loss-invariant)
                cpt = Path(results_root) / run / _slug(loss) / 'checkpoints' / f'{mk}_{split}.pt'
                if cpt.is_file():
                    c = torch.load(str(cpt), map_location='cpu', weights_only=False)
                    nneur[mk] = int(c['spat']['model_params']['input_size'])
    return skill, nneur, sorted(mice)


def _agg(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def main(results_root, run, split, exclude, out_root):
    ps.apply()
    skill, nneur, mice = collect(results_root, run, split)
    if not mice:
        raise SystemExit('no cells found.')
    keep = [m for m in mice if m != exclude]
    x = np.arange(len(LOSSES)); w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2))

    # (a),(b) spat vs temp KL-skill per loss — all mice, then exclude
    for ax, msel, title in ((axes[0, 0], mice, f'all mice (n={len(mice)})'),
                            (axes[0, 1], keep, f'{exclude} excluded (n={len(keep)})')):
        for k, arch in enumerate(('spat', 'temp')):
            m = [_agg([skill[l][arch][mk] for mk in msel])[0] for l in LOSSES]
            e = [_agg([skill[l][arch][mk] for mk in msel])[1] for l in LOSSES]
            ax.bar(x + (k - 0.5) * w, m, w, yerr=e, capsize=2, color=ARCH_COL[arch],
                   label={'spat': 'spatial', 'temp': 'temporal'}[arch])
        ax.axhline(1.0, color='k', ls='--', lw=1.1)
        ax.set_xticks(x); ax.set_xticklabels(ps.loss_labels(LOSSES), rotation=20, ha='right')
        ax.set_ylabel('normalised KL loss (test / shuffle)'); ax.set_title(f'spat vs temp — {title}', fontsize=9)
        if ax is axes[0, 0]:
            ax.legend(fontsize=7.5, loc='upper left')

    # (c) n_neurons per mouse
    ax = axes[1, 0]
    nn = [nneur.get(mk, np.nan) for mk in mice]
    cols = ['#cb181d' if mk == exclude else '0.5' for mk in mice]
    ax.bar(np.arange(len(mice)), nn, color=cols)
    ax.set_xticks(np.arange(len(mice))); ax.set_xticklabels([m.replace('mouse_', 'm') for m in mice])
    ax.set_ylabel('# neurons'); ax.set_title(f'neurons per animal ({exclude} red)', fontsize=9)

    # (d) n_neurons vs KL-skill (per mouse, averaged over losses), spat & temp + Pearson r
    ax = axes[1, 1]
    nn_arr = np.array([nneur.get(mk, np.nan) for mk in mice], float)
    for arch in ('spat', 'temp'):
        y = np.array([np.nanmean([skill[l][arch][mk] for l in LOSSES]) for mk in mice])
        ax.scatter(nn_arr, y, color=ARCH_COL[arch], s=30,
                   label={'spat': 'spatial', 'temp': 'temporal'}[arch])
        good = np.isfinite(nn_arr) & np.isfinite(y)
        if good.sum() >= 3:
            r, p = sstats.pearsonr(nn_arr[good], y[good])
            ax.plot([], [], ' ', label=f'  r={r:+.2f} (p={p:.2f})')
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xlabel('# neurons'); ax.set_ylabel('normalised KL loss (mean over losses)')
    ax.set_title('does neuron count predict performance?', fontsize=9); ax.legend(fontsize=6.5)

    ps.label_panels(axes.ravel())
    fig.suptitle('Spatial vs temporal per animal — loss head-to-head, Mouse-2 leave-out, neuron count '
                 f'({run})', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'spat_temp_per_animal')

    # ---- numeric ----
    print(f'n_neurons per mouse: ' + ', '.join(f'{m.replace("mouse_","m")}={nneur.get(m,"?")}' for m in mice))
    print(f'\nspat vs temp normalised KL loss (lower=better), all mice vs {exclude}-excluded; paired t over mice:')
    print(f"  {'loss':12s} {'spat(all)':>9s} {'temp(all)':>9s} {'Δ(all)':>7s} {'p':>6s} | "
          f"{'spat(ex)':>9s} {'temp(ex)':>9s} {'Δ(ex)':>7s} {'p':>6s}")
    for l in LOSSES:
        row = []
        for msel in (mice, keep):
            sp = np.array([skill[l]['spat'][m] for m in msel]); tp = np.array([skill[l]['temp'][m] for m in msel])
            g = np.isfinite(sp) & np.isfinite(tp)
            p = sstats.ttest_rel(sp[g], tp[g]).pvalue if g.sum() >= 3 else np.nan
            row += [np.nanmean(sp), np.nanmean(tp), np.nanmean(sp) - np.nanmean(tp), p]
        print(f"  {l:12s} {row[0]:9.2f} {row[1]:9.2f} {row[2]:+7.2f} {row[3]:6.2f} | "
              f"{row[4]:9.2f} {row[5]:9.2f} {row[6]:+7.2f} {row[7]:6.2f}")
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--exclude', default='mouse_2', help='animal to leave out (.mat key)')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.exclude, a.out_root)
