# -*- coding: utf-8 -*-
"""Three views of the ideal-observer perceptual posteriors and their PCA basis
(2026-06-10, Theo): (1) all perceptual (IO-target) posteriors for ONE animal,
(2) the explained-variance spectrum across PCs for ALL animals, (3) the first
5–7 PC loadings for each animal.

All read from the saved IO targets + per-mouse PCA basis in the .mat (the target
and basis are loss- and arch-invariant, so any decoded cell works).

Outputs (PNG+SVG) under figures/perceptual_pca/:
  perceptual_posteriors_<mouse>.png
  evar_spectrum_all_mice.png
  pc_loadings_per_mouse.png

Usage:  python diagnostics/perceptual_posteriors_and_pca.py [--mouse mouse_0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

C = 91


def load(results_root, run, slug, split):
    f = Path(results_root) / run / slug / f'{split}.mat'
    res = sio.loadmat(str(f), simplify_cells=True)['results']
    mice = sorted(res)
    data = {}
    for mk in mice:
        D = res[mk]['Dist']
        data[mk] = {'target': np.asarray(D['spat']['target'], float),
                    'pcs': np.asarray(D['pcs'], float),
                    'evar': np.asarray(D['explained_var'], float)}
    return data, mice


def _sign_align(pcs):
    """PCA component signs are arbitrary; flip each so its largest-magnitude
    entry is positive, for cross-animal comparability."""
    out = pcs.copy()
    for k in range(len(out)):
        if out[k][int(np.argmax(np.abs(out[k])))] < 0:
            out[k] = -out[k]
    return out


def fig_perceptual_posteriors(data, mouse, out_dir):
    T = data[mouse]['target']
    order = np.argsort(T.argmax(1))
    x = np.arange(C)
    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1))

    ax = axes[0]
    im = ax.imshow(T[order], aspect='auto', cmap='magma', origin='lower')
    ax.set_xlabel('orientation bin'); ax.set_ylabel('trial (sorted by peak bin)')
    ax.set_title(f'all {len(T)} IO posteriors')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='probability')

    ax = axes[1]
    picks = order[np.linspace(0, len(order) - 1, 12).astype(int)]
    cols = plt.get_cmap('viridis')(np.linspace(0, 1, len(picks)))
    for tr, c in zip(picks, cols):
        ax.plot(x, T[tr], color=c, lw=1.3, alpha=0.85)
    ax.set_xlabel('orientation bin'); ax.set_ylabel('probability')
    ax.set_title('12 examples (spanning peak range)')

    ax = axes[2]
    step = max(1, len(T) // 400)                       # thin for speed if many
    for tr in range(0, len(T), step):
        ax.plot(x, T[tr], color='0.6', lw=0.4, alpha=0.10)
    ax.plot(x, T.mean(0), color=ps.PCA_EVAR, lw=2.6, label='mean over trials')
    ax.set_xlabel('orientation bin'); ax.set_ylabel('probability')
    ax.set_title('all (faint) + mean'); ax.legend(loc='upper right', fontsize=8)

    ps.label_panels(axes)
    fig.suptitle(f'Perceptual (ideal-observer) posteriors — {mouse} '
                 f'(orientation, 0–90°)', y=1.02)
    ps.save_fig(fig, Path(out_dir), f'perceptual_posteriors_{mouse}')


def fig_evar_spectrum(data, mice, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    cols = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(mice), 3)))
    K = 25
    for mk, c in zip(mice, cols):
        ev = data[mk]['evar']
        kk = np.arange(1, min(K, len(ev)) + 1)
        axes[0].semilogy(kk, ev[:len(kk)] + 1e-12, color=c, marker='o', ms=3, lw=1.4,
                         label=mk.replace('mouse_', 'm'))
        cum = np.cumsum(ev) / ev.sum()
        axes[1].plot(np.arange(1, len(cum) + 1), cum, color=c, lw=1.6)
    axes[0].set_xlabel('PC index $k$'); axes[0].set_ylabel('explained-variance ratio  evar$_k$')
    axes[0].set_title('Spectrum (collapses fast)')
    axes[0].legend(fontsize=7.5, ncol=2, loc='upper right')
    axes[1].axhline(0.90, ls='--', color='0.5', lw=1, label='90%')
    axes[1].set_xlabel('PC index $k$'); axes[1].set_ylabel('cumulative explained variance')
    axes[1].set_title('Cumulative'); axes[1].set_xlim(1, 30); axes[1].legend(fontsize=8)
    ps.label_panels(axes)
    fig.suptitle('PCA explained-variance spectrum of the IO targets — all animals', y=1.02)
    ps.save_fig(fig, Path(out_dir), 'evar_spectrum_all_mice')


def fig_pc_loadings(data, mice, out_dir, npc=7):
    x = np.arange(C)
    fig, axes = plt.subplots(len(mice), npc, squeeze=False, sharex=True,
                             figsize=ps.figsize(npc, len(mice), panel_w=1.7, panel_h=1.05))
    for r, mk in enumerate(mice):
        pcs = _sign_align(data[mk]['pcs'])
        ev = data[mk]['evar']
        for k in range(npc):
            ax = axes[r][k]
            ax.axhline(0, color='0.8', lw=0.5)
            ax.plot(x, pcs[k], color='#1f3b6d', lw=1.1)
            ax.set_yticks([]); ax.set_xticks([])
            if r == 0:
                ax.set_title(f'PC{k}', fontsize=8.5)
            if k == 0:
                ax.set_ylabel(mk.replace('mouse_', 'm'), fontsize=8.5, rotation=0,
                              ha='right', va='center', labelpad=8)
            # tiny per-PC evar tag
            ax.text(0.97, 0.92, f'{ev[k] * 100:.0f}%', transform=ax.transAxes,
                    ha='right', va='top', fontsize=6, color='0.4')
    fig.suptitle(f'First {npc} PC loadings (orientation, sign-aligned) — each animal; '
                 '% = explained variance', y=1.005)
    ps.save_fig(fig, Path(out_dir), 'pc_loadings_per_mouse')


def main(results_root, run, slug, split, mouse, out_root):
    ps.apply()
    data, mice = load(results_root, run, slug, split)
    if mouse not in data:
        mouse = mice[0]
    print(f'mice={mice} | {mouse}: {data[mouse]["target"].shape[0]} target posteriors, '
          f'{data[mouse]["pcs"].shape[0]} PCs')
    fig_perceptual_posteriors(data, mouse, out_root)
    fig_evar_spectrum(data, mice, out_root)
    fig_pc_loadings(data, mice, out_root)
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--slug', default='Q_PCA_half_100ms_all')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--mouse', default='mouse_0')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/perceptual_pca')
    a = ap.parse_args()
    main(a.results_root, a.run, a.slug, a.split, a.mouse, a.out_root)
