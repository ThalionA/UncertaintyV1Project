# -*- coding: utf-8 -*-
"""PCA views of decoded posteriors — the 2026-06-03 (Máté) meeting's asks.

Two requests from the handwritten list, on the decoded posteriors already saved
in a flat loss-comparison run (e.g. ``loss_comparison_v1``):

  * "Show PC1/PC2 proj of all posteriors."
        -> scatter of every trial's decoded posterior in the leading-PC plane,
           with the IO targets overlaid, coloured by a stimulus feature.
  * "Do a mean posterior + PC1·a."
        -> reconstruct  mean_posterior + a·σ₁·PC1  for a ∈ {−2,−1,0,1,2} and
           render each as an actual posterior curve, so the leading axis of
           variation is made legible (does PC1 move the peak, or widen it?).

Basis choice (the open decision in PLAN_2026-06-03): a PCA basis can be fit on
the **decoded** posteriors (how the *decoder* spreads trials) or on the **IO
target** posteriors (the ideal axis; decoded is then projected onto it to show
deviation). Both are cheap and the contrast is the point, so ``--basis both``
(default) does each. The IO targets are identical across loss runs (same trials,
same seed), so an all-losses overlay uses one shared target basis — making the
loss runs directly comparable in the same plane.

Outputs (PNG+SVG) under figures/posterior_pca/<run>/:
  scatter_<arch>_<loss>_<basis>.png        PC1/PC2 of decoded (+ target overlay)
  recon_<arch>_<loss>_<basis>.png          mean + a·σ·PC1 (and PC2) curves
  scatter_alllosses_<arch>_targetbasis.png decoded of every loss in one plane

Usage
-----
    python posterior_pca_views.py                       # Q half 100ms, all mice, both archs/bases
    python posterior_pca_views.py --loss KL --arch spat --mouse 0
    python posterior_pca_views.py --color-by dispersion --basis target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import decoder_plotting_utils as dpu  # noqa: F401  (set_style)

LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
LOSS_COLOR = {'PCA': '#e6550d', 'CE': '#008837', 'KL': '#7b3294',
              'JS': '#3690c0', 'Wasserstein': '#a6611a'}
AMPS = (-2, -1, 0, 1, 2)


def _slug(target, loss, window, bin_ms):
    return f'{target}_{loss}_{window}_{bin_ms}ms' + ('_all' if loss == 'PCA' else '')


def load_cell(results_root, run_name, target, loss, window, bin_ms, split, mice):
    """Concatenate decoded/target posteriors and trial features over ``mice``
    (a list of ints, or None for all present) for both archs. Returns
    ``{'spat': {...}, 'temp': {...}}`` each with decoded, target, feats — or
    None if the cell is absent."""
    mat = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / f'{split}.mat'
    if not mat.is_file():
        return None
    d = sio.loadmat(str(mat), simplify_cells=True)['results']
    keys = [f'mouse_{m}' for m in mice] if mice is not None else sorted(d.keys())
    out = {}
    for arch in ('spat', 'temp'):
        dec, tgt, feats = [], [], {'orientation': [], 'dispersion': [], 'contrast': [], 'peak': []}
        for mk in keys:
            if mk not in d:
                continue
            Dist = d[mk]['Dist'][arch]
            de = np.asarray(Dist['decoded'], float)
            tg = np.asarray(Dist['target'], float)
            if de.ndim != 2 or de.shape[0] == 0:
                continue
            dec.append(de); tgt.append(tg)
            tr = d[mk]['trials']
            n = de.shape[0]
            for f in ('orientation', 'dispersion', 'contrast'):
                v = np.asarray(tr[f]).ravel()
                feats[f].append(v[:n] if v.size >= n else np.full(n, np.nan))
            feats['peak'].append(np.argmax(tg, axis=1))
        if not dec:
            continue
        out[arch] = {
            'decoded': np.concatenate(dec), 'target': np.concatenate(tgt),
            'feats': {k: np.concatenate(v) for k, v in feats.items() if v},
        }
    return out or None


def fit_basis(posteriors):
    """PCA on a set of (n, n_cats) posteriors. Returns (pca, mean, scores)."""
    pca = PCA(n_components=min(10, posteriors.shape[1]))
    scores = pca.fit_transform(posteriors)
    return pca, posteriors.mean(axis=0), scores


# ----------------------------------------------------------------------

def fig_scatter(cell, arch, loss, basis_name, basis_post, out_dir, info, color_by):
    """PC1/PC2 scatter of all decoded posteriors in a basis fit on
    ``basis_post`` (decoded or target). IO targets overlaid as faint markers."""
    pca, _, _ = fit_basis(basis_post)
    dec = pca.transform(cell['decoded'])
    tgt = pca.transform(cell['target'])
    c = cell['feats'].get(color_by)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sc = ax.scatter(dec[:, 0], dec[:, 1], c=c, s=14, cmap='viridis',
                    alpha=0.8, edgecolor='none', label='decoded')
    ax.scatter(tgt[:, 0], tgt[:, 1], facecolor='none', edgecolor='0.35',
               s=22, lw=0.5, alpha=0.5, label='IO target')
    if c is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(color_by)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1  ({ev[0]*100:.0f}% var)')
    ax.set_ylabel(f'PC2  ({ev[1]*100:.0f}% var)')
    ax.set_title(f'Decoded posteriors in PC1/PC2 — {arch.upper()} {loss}\n'
                 f'basis fit on {basis_name};  {info};  open circles = IO target')
    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    _save(fig, out_dir, f'scatter_{arch}_{loss}_{basis_name}')


def fig_recon(cell, arch, loss, basis_name, basis_post, out_dir, info):
    """mean + a·σ_k·PC_k reconstruction strip for PC1 (top) and PC2 (bottom).
    Shows what the leading axes of posterior variation actually do."""
    pca, mean, scores = fit_basis(basis_post)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
    cmap = plt.get_cmap('coolwarm')
    x = np.arange(mean.size)
    for row, pc in enumerate((0, 1)):
        ax = axes[row]
        sigma = scores[:, pc].std()
        for a in AMPS:
            recon = mean + a * sigma * pca.components_[pc]
            col = cmap((a - AMPS[0]) / (AMPS[-1] - AMPS[0]))
            lw = 2.6 if a == 0 else 1.6
            ax.plot(x, recon, color=col, lw=lw,
                    label=f'a={a:+d}' if a else 'mean (a=0)')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ev = pca.explained_variance_ratio_[pc]
        ax.set_ylabel(f'prob.  (mean + a·σ·PC{pc+1})')
        ax.set_title(f'PC{pc+1} reconstruction  ({ev*100:.0f}% var)', fontsize=10)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, ncol=5, loc='upper right')
    axes[1].set_xlabel('orientation bin')
    fig.suptitle(f'Mean posterior ± a·σ·PC — {arch.upper()} {loss}  '
                 f'(basis: {basis_name}; {info})', y=1.0, fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, f'recon_{arch}_{loss}_{basis_name}')


def fig_scatter_all_losses(cells, arch, out_dir, info, color_by):
    """All losses' decoded posteriors in ONE shared basis (fit on the IO
    targets, which are identical across loss runs) — directly comparable."""
    losses = [l for l in LOSSES if l in cells and arch in cells[l]]
    if len(losses) < 2:
        return
    basis_post = cells[losses[0]][arch]['target']  # shared target basis
    pca, _, _ = fit_basis(basis_post)
    ncol = len(losses)
    fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.5), squeeze=False,
                             sharex=True, sharey=True)
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        dec = pca.transform(cells[loss][arch]['decoded'])
        col = cells[loss][arch]['feats'].get(color_by)
        ax.scatter(dec[:, 0], dec[:, 1], c=col, s=8, cmap='viridis', alpha=0.7,
                   edgecolor='none')
        ax.set_title(loss, color=LOSS_COLOR[loss], fontsize=11)
        ax.set_xlabel('PC1')
        if c == 0:
            ax.set_ylabel('PC2')
    fig.suptitle(f'Decoded posteriors per loss in shared IO-target PC basis — '
                 f'{arch.upper()}  ({info}, colour={color_by})', y=1.04, fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, f'scatter_alllosses_{arch}_targetbasis')


# Canonical PNG+SVG sink with the ≤1600px PNG cap (was an uncapped fixed-dpi=140
# local helper). Aliased to ``_save`` so existing call sites are unchanged.
from figsave import save_fig as _save


def main(run_name, target, window, bin_ms, split, loss_sel, arch_sel, mouse_sel,
         basis_sel, color_by, results_root, out_root):
    dpu.set_style()
    mice = None if mouse_sel == 'all' else [int(mouse_sel)]
    info = f'{target} {window} {bin_ms}ms {split} ' + \
           ('all mice' if mice is None else f'mouse {mice[0]}')
    losses = LOSSES if loss_sel == 'all' else (loss_sel,)
    archs = ('spat', 'temp') if arch_sel == 'both' else (arch_sel,)
    bases = ('decoded', 'target') if basis_sel == 'both' else (basis_sel,)
    out_dir = Path(out_root) / run_name

    cells = {}
    for loss in losses:
        cell = load_cell(results_root, run_name, target, loss, window, bin_ms, split, mice)
        if cell is None:
            print(f'  [skip] {loss}: cell not found')
            continue
        cells[loss] = cell
        for arch in archs:
            if arch not in cell:
                continue
            n = cell[arch]['decoded'].shape[0]
            print(f'  {loss:12s} {arch}: {n} trials')
            for basis_name in bases:
                basis_post = cell[arch]['decoded' if basis_name == 'decoded' else 'target']
                fig_scatter(cell[arch], arch, loss, basis_name, basis_post, out_dir, info, color_by)
                fig_recon(cell[arch], arch, loss, basis_name, basis_post, out_dir, info)
    if not cells:
        raise SystemExit('No cells found for this configuration.')
    for arch in archs:
        fig_scatter_all_losses(cells, arch, out_dir, info, color_by)
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--loss', default='all', help="one of PCA/CE/KL/JS/Wasserstein, or 'all'")
    ap.add_argument('--arch', default='both', choices=('spat', 'temp', 'both'))
    ap.add_argument('--mouse', default='all', help="mouse index, or 'all' (pooled)")
    ap.add_argument('--basis', default='both', choices=('decoded', 'target', 'both'))
    ap.add_argument('--color-by', default='peak',
                    choices=('peak', 'orientation', 'dispersion', 'contrast'))
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/posterior_pca')
    a = ap.parse_args()
    main(a.run_name, a.target, a.window, a.bin_ms, a.split, a.loss, a.arch,
         a.mouse, a.basis, a.color_by, a.results_root, a.out_root)
