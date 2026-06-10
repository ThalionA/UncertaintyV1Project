# -*- coding: utf-8 -*-
"""Three views of the ideal-observer perceptual posteriors and their PCA basis
(2026-06-10, Theo): (1) ALL perceptual (IO) posteriors for one animal,
(2) the explained-variance spectrum across PCs for all animals, (3) the first
5–7 PC loadings for each animal.

Reads the FULL per-trial IO perceptual posteriors directly from the VR export
(`utils.load_vr_export(mouse_id)` -> targets_perc, all trials, pre-split — not the
held-out half saved in the decoder .mat), and fits PCA on all of them per animal.

Outputs (PNG+SVG) under figures/perceptual_pca/:
  perceptual_posteriors_m<mouse>.png
  evar_spectrum_all_mice.png
  pc_loadings_per_mouse.png

Usage:  python diagnostics/perceptual_posteriors_and_pca.py [--mouse 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from utils import load_vr_export  # noqa: E402

C = 91


def load(mice):
    """All-trials IO perceptual posteriors + per-animal PCA (fit on all trials)."""
    data = {}
    for m in mice:
        _, tperc, _, _, _ = load_vr_export(m)
        T = np.asarray(tperc, float)
        if T.shape[1] != C and T.shape[0] == C:
            T = T.T
        pca = PCA().fit(T)
        data[m] = {'target': T, 'pcs': pca.components_,
                   'evar': pca.explained_variance_ratio_}
    return data


def _sign_align(pcs):
    out = pcs.copy()
    for k in range(len(out)):
        if out[k][int(np.argmax(np.abs(out[k])))] < 0:
            out[k] = -out[k]
    return out


def fig_perceptual_posteriors(T, m, out_dir):
    order = np.argsort(T.argmax(1))
    x = np.arange(C)
    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1, panel_w=4.2, panel_h=3.3))

    ax = axes[0]
    im = ax.imshow(T[order], aspect='auto', cmap='magma', origin='lower')
    ax.set_xlabel('orientation bin'); ax.set_ylabel('trial (sorted by peak bin)')
    ax.set_title('IO posteriors, sorted by peak')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='probability')

    ax = axes[1]
    picks = order[np.linspace(0, len(order) - 1, 12).astype(int)]
    cols = plt.get_cmap('viridis')(np.linspace(0, 1, len(picks)))
    for tr, c in zip(picks, cols):
        ax.plot(x, T[tr], color=c, lw=1.3, alpha=0.85)
    ax.set_xlabel('orientation bin'); ax.set_ylabel('probability')
    ax.set_title('Example posteriors')

    ax = axes[2]
    step = max(1, len(T) // 400)
    for tr in range(0, len(T), step):
        ax.plot(x, T[tr], color='0.6', lw=0.4, alpha=0.10)
    ax.plot(x, T.mean(0), color=ps.PCA_EVAR, lw=2.6, label='mean over trials')
    ax.set_xlabel('orientation bin'); ax.set_ylabel('probability')
    ax.set_title('All posteriors and their mean'); ax.legend(loc='upper right', fontsize=8)

    ps.label_panels(axes)
    fig.suptitle(f'Perceptual (ideal-observer) posteriors — m{m}, orientation 0–90° '
                 f'({len(T)} trials)', y=1.02)
    ps.save_fig(fig, Path(out_dir), f'perceptual_posteriors_m{m}')


def fig_evar_spectrum(data, mice, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1, panel_w=3.9, panel_h=3.1))
    cols = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(mice), 3)))
    K = 25
    for m, c in zip(mice, cols):
        ev = data[m]['evar']
        kk = np.arange(1, min(K, len(ev)) + 1)
        axes[0].semilogy(kk, ev[:len(kk)] + 1e-12, color=c, marker='o', ms=3, lw=1.4,
                         label=f'm{m}')
        cum = np.cumsum(ev) / ev.sum()
        axes[1].plot(np.arange(1, len(cum) + 1), cum, color=c, lw=1.6)
    axes[0].set_xlabel('PC index $k$'); axes[0].set_ylabel('explained-variance ratio  evar$_k$')
    axes[0].set_title('Explained-variance ratio per PC')
    axes[0].legend(fontsize=7.5, ncol=2, loc='upper right')
    axes[1].axhline(0.90, ls='--', color='0.5', lw=1)
    axes[1].set_xlabel('PC index $k$'); axes[1].set_ylabel('cumulative explained variance')
    axes[1].set_title('Cumulative explained variance'); axes[1].set_xlim(1, 30)
    ps.label_panels(axes)
    fig.suptitle('PCA explained-variance spectrum of the IO perceptual posteriors — all animals', y=1.02)
    ps.save_fig(fig, Path(out_dir), 'evar_spectrum_all_mice')


def fig_pc_loadings(data, mice, out_dir, npc=7):
    x = np.arange(C)
    fig, axes = plt.subplots(len(mice), npc, squeeze=False, sharex=True,
                             figsize=ps.figsize(npc, len(mice), panel_w=1.9, panel_h=1.2))
    for r, m in enumerate(mice):
        pcs = _sign_align(data[m]['pcs'])
        ev = data[m]['evar']
        for k in range(npc):
            ax = axes[r][k]
            ax.axhline(0, color='0.8', lw=0.5)
            ax.plot(x, pcs[k], color='#1f3b6d', lw=1.1)
            ax.set_yticks([]); ax.set_xticks([])
            if r == 0:
                ax.set_title(f'PC{k}', fontsize=9)
            if k == 0:
                ax.set_ylabel(f'm{m}', fontsize=9, rotation=0, ha='right', va='center', labelpad=8)
            ax.text(0.96, 0.93, f'{ev[k] * 100:.0f}%', transform=ax.transAxes,
                    ha='right', va='top', fontsize=6.5, color='0.45')
    fig.suptitle(f'First {npc} PC loadings per animal (orientation; sign-aligned, '
                 '% = variance explained)', y=1.01)
    ps.save_fig(fig, Path(out_dir), 'pc_loadings_per_mouse')


def main(mouse, n_mice, out_root):
    ps.apply()
    mice = list(range(n_mice))
    data = load(mice)
    for m in mice:
        print(f'  m{m}: {data[m]["target"].shape[0]} perceptual posteriors')
    fig_perceptual_posteriors(data[mouse]['target'], mouse, out_root)
    fig_evar_spectrum(data, mice, out_root)
    fig_pc_loadings(data, mice, out_root)
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mouse', type=int, default=0, help='animal for the posteriors panel')
    ap.add_argument('--n-mice', type=int, default=6)
    ap.add_argument('--out-root', default='figures/perceptual_pca')
    a = ap.parse_args()
    main(a.mouse, a.n_mice, a.out_root)
