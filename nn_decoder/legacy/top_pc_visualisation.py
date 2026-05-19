# -*- coding: utf-8 -*-
"""Visualise the top principal components of the per-condition averaged
Q(theta) distributions — the basis underlying the PCA-weighted Euclidean
loss used in the production decoder.

Two questions the meeting raised:
  - What does the top PC look like?
  - Does it encode something interpretable (peak position, peak width,
    Go-vs-NoGo asymmetry, ...) — i.e. what is the loss actually rewarding?

For each mouse:
  - Load Q(theta) per trial from VR_Decoder_Data_Export (post_s_marginal).
  - Compute per-condition (orientation, contrast, dispersion) means.
  - Run PCA on these averaged distributions (no leakage worry —
    everything is just for visualisation).
  - Plot the top 4 PCs as functions of theta, plus the cumulative
    explained variance.

Also produces a *pooled* PCA across all mice for a single canonical
basis.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


GRID = np.arange(0, 91, 1, dtype=float)


def _set_style():
    sns.set_context('talk')
    sns.set_style('ticks')


def _load_trials_and_Q(mouse_id):
    """Use the existing utils loader — keeps a single path to the data."""
    from utils import load_vr_export
    _, targets_perc, _, _, trials = load_vr_export(mouse_id)
    conditions = np.column_stack([
        np.asarray(trials['orientation'], dtype=float),
        np.asarray(trials['contrast'], dtype=float),
        np.asarray(trials['dispersion'], dtype=float),
    ])
    return conditions, np.asarray(targets_perc)


def per_condition_means(conditions, Q):
    """Per (orientation, contrast, dispersion) condition mean of Q."""
    uniq = np.unique(conditions, axis=0)
    out = np.zeros((len(uniq), Q.shape[1]))
    for i, c in enumerate(uniq):
        mask = np.all(conditions == c, axis=1)
        out[i] = Q[mask].mean(axis=0)
    return uniq, out


def plot_pcs_for_mouse(mouse_id, out_dir, n_pcs=4):
    """Per-mouse: top n_pcs of the per-condition averaged Q distributions."""
    _set_style()
    conditions, Q = _load_trials_and_Q(mouse_id)
    uniq, cond_avg = per_condition_means(conditions, Q)

    pca = PCA().fit(cond_avg)
    pcs = pca.components_
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[2, 1])
    ax = axes[0]
    for k in range(min(n_pcs, pcs.shape[0])):
        ax.plot(GRID, pcs[k], lw=2,
                label=f'PC{k+1} ({100*var[k]:.1f}%)')
    ax.axvline(45, color='grey', linestyle='--', linewidth=1, alpha=0.6,
                label='boundary (45 deg)')
    ax.axhline(0, color='grey', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('orientation theta [deg]')
    ax.set_ylabel('PC loading')
    ax.set_title(
        f'Mouse {mouse_id}: top {n_pcs} PCs of per-condition averaged Q(theta)\n'
        f'(n_conditions = {len(uniq)})'
    )
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    cum = np.cumsum(var)
    ax.plot(np.arange(1, len(var) + 1), cum, marker='o', lw=2)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.6,
                label='95% explained')
    ax.set_xlabel('PC index')
    ax.set_ylabel('cumulative explained variance')
    ax.set_title('PC scree (cumulative)')
    ax.set_xlim(0.5, min(20, len(var)) + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'top_pcs_mouse{mouse_id}.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path, pcs, var, uniq, cond_avg


def plot_pcs_pooled(mouse_ids, out_dir, n_pcs=4):
    """Pooled PCA across all mice's per-condition averaged Q distributions.
    Gives a canonical basis that doesn't depend on per-mouse stim coverage."""
    _set_style()
    all_avgs = []
    for mid in mouse_ids:
        conditions, Q = _load_trials_and_Q(mid)
        _, cond_avg = per_condition_means(conditions, Q)
        all_avgs.append(cond_avg)
    pooled = np.vstack(all_avgs)
    pca = PCA().fit(pooled)
    pcs = pca.components_
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[2, 1])
    ax = axes[0]
    for k in range(min(n_pcs, pcs.shape[0])):
        ax.plot(GRID, pcs[k], lw=2,
                label=f'PC{k+1} ({100*var[k]:.1f}%)')
    ax.axvline(45, color='grey', linestyle='--', linewidth=1, alpha=0.6,
                label='boundary (45 deg)')
    ax.axhline(0, color='grey', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('orientation theta [deg]')
    ax.set_ylabel('PC loading')
    ax.set_title(
        f'Pooled across {len(mouse_ids)} mice: top {n_pcs} PCs '
        f'of per-condition averaged Q(theta)\n'
        f'(n_condition_means = {pooled.shape[0]})'
    )
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    cum = np.cumsum(var)
    ax.plot(np.arange(1, len(var) + 1), cum, marker='o', lw=2)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.6,
                label='95%')
    ax.set_xlabel('PC index')
    ax.set_ylabel('cumulative explained variance')
    ax.set_xlim(0.5, min(20, len(var)) + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(out_dir, 'top_pcs_pooled.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path, pcs, var


def main(mouse_ids=range(6), out_dir=None):
    if out_dir is None:
        out_dir = os.path.normpath(os.path.join(HERE, '..', 'figures', 'top_pcs'))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[top_pcs] writing to {out_dir}")

    for mid in mouse_ids:
        path, pcs, var, uniq, _ = plot_pcs_for_mouse(mid, out_dir)
        print(f"  mouse {mid}: {len(uniq)} conditions, "
              f"top 3 PCs explain "
              f"{100*var[0]:.1f}/{100*var[1]:.1f}/{100*var[2]:.1f}%  -> {path}")

    path_pooled, pcs_pooled, var_pooled = plot_pcs_pooled(list(mouse_ids), out_dir)
    print(f"\n  pooled: top 3 PCs explain "
          f"{100*var_pooled[0]:.1f}/{100*var_pooled[1]:.1f}/{100*var_pooled[2]:.1f}%  "
          f"-> {path_pooled}")

    # Save the pooled basis as .npz for downstream consumers.
    np.savez(os.path.join(out_dir, 'pooled_pca.npz'),
              components=pcs_pooled, explained_variance_ratio=var_pooled,
              grid_deg=GRID)
    print(f"  pooled PCA basis saved to {os.path.join(out_dir, 'pooled_pca.npz')}")


if __name__ == '__main__':
    main()
