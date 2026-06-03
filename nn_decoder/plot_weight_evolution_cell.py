# -*- coding: utf-8 -*-
"""Weight-norm evolution for ONE cell of a flat loss-comparison run.

The production weight-evolution plot (`plot_loss_sweep.plot_weight_norms`) facets
per-layer norms by loss with cryptic 'param k' labels, and `plot_kl_js_training`
only walks the entropy-lambda sweep tree. This script targets a single
(target, window, bin, split) cell of a *flat* run (e.g. `loss_comparison_v1`)
and answers the meeting's actual question directly:

    Does the loss that doesn't punish over-confidence (PCA) drive the OUTPUT-layer
    weights larger than the calibrated losses (KL/JS/CE)? And does the SBC
    entropy penalty act as implicit weight regularisation (temp W_out < spat)?

Each saved checkpoint stores, per epoch, the L2 norm of every parameter tensor
(`history['weight_norms']`, shape (epochs, 4) for a 1-hidden-layer MLP):
    param 0 = W_in  (H x n_neurons)   param 1 = b_in (H)
    param 2 = W_out (n_cats x H)       param 3 = b_out (n_cats)
The output-layer weight (param 2) sets how large the softmax logits — hence how
peaky the posterior — can get, so it is the curve of interest.

Three figures (PNG+SVG) under figures/loss_sweep_plots/<run>/weight_evolution/:
  A_Wout_by_loss_<arch>     output-weight norm vs epoch, one line per loss
  B_allparams_<arch>        per-loss small multiples, all 4 params relabelled
  C_Wout_spat_vs_temp       per-loss, spat vs temp W_out (entropy-reg-as-reg test)

Lines end where each mouse's training stopped (early stopping), so the x-extent
itself shows how long each loss trained.

Usage
-----
    python plot_weight_evolution_cell.py                       # Q half 100ms balanced
    python plot_weight_evolution_cell.py --target L --bin 50 --window full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import decoder_plotting_utils as dpu  # noqa: F401  (for set_style)

LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
LOSS_COLOR = {'PCA': '#e6550d', 'CE': '#008837', 'KL': '#7b3294',
              'JS': '#3690c0', 'Wasserstein': '#a6611a'}
PARAM_LABEL = ['W_in (hidden×neurons)', 'b_in', 'W_out (cats×hidden)', 'b_out']
WOUT = 2  # index of the output-layer weight tensor


def _slug(target, loss, window, bin_ms):
    return f'{target}_{loss}_{window}_{bin_ms}ms' + ('_all' if loss == 'PCA' else '')


def load_cell(run_root, run_name, target, loss, window, bin_ms, split, arch):
    """Per-mouse weight_norms arrays for one (loss, arch). Returns list of
    (epochs, 4) arrays — one per mouse that has a tracked history."""
    ck_dir = Path(run_root) / run_name / _slug(target, loss, window, bin_ms) / 'checkpoints'
    out = []
    if not ck_dir.is_dir():
        return out
    for pt in sorted(ck_dir.glob(f'mouse_*_{split}.pt')):
        ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        if not (isinstance(ck, dict) and arch in ck and isinstance(ck[arch], dict)):
            continue
        hist = ck[arch].get('history')
        if not hist or not hist.get('weight_norms'):
            continue
        out.append(np.asarray(hist['weight_norms'], dtype=float))
    return out


def ragged_mean(arrays, col):
    """Mean over mice of column `col`, at each epoch averaging only the mice
    still training then. Returns (epochs_axis, mean) to the longest run."""
    if not arrays:
        return np.array([]), np.array([])
    maxlen = max(a.shape[0] for a in arrays)
    mean = np.full(maxlen, np.nan)
    for e in range(maxlen):
        vals = [a[e, col] for a in arrays if a.shape[0] > e]
        if vals:
            mean[e] = np.mean(vals)
    return np.arange(1, maxlen + 1), mean


# ----------------------------------------------------------------------

def fig_A_wout_by_loss(data, arch, out_dir, info):
    """Output-weight norm vs epoch, one bold across-mouse line per loss, with
    faint per-mouse lines. The headline: does PCA inflate W_out vs KL/JS/CE?"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for loss in LOSSES:
        arrs = data.get(loss, [])
        if not arrs:
            continue
        for a in arrs:
            ax.plot(np.arange(1, a.shape[0] + 1), a[:, WOUT],
                    color=LOSS_COLOR[loss], lw=0.7, alpha=0.25)
        xs, m = ragged_mean(arrs, WOUT)
        ax.plot(xs, m, color=LOSS_COLOR[loss], lw=2.6,
                label=f'{loss}  (n={len(arrs)})')
        # mark the across-mouse mean stop epoch
        stops = [a.shape[0] for a in arrs]
        ax.scatter([np.mean(stops)], [np.nanmean([a[-1, WOUT] for a in arrs])],
                   color=LOSS_COLOR[loss], s=40, zorder=5, edgecolor='k', lw=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('output-layer weight L2 norm  ‖W_out‖')
    ax.set_title(f'Output-weight growth by loss — {arch.upper()}  ({info})\n'
                 'faint = per mouse; bold = across-mouse mean; dot = mean stop '
                 '(early stopping)')
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out_dir, f'A_Wout_by_loss_{arch}')


def fig_B_allparams(data, arch, out_dir, info):
    """Per-loss small multiples: all four parameter norms vs epoch (relabelled),
    across-mouse mean. The full picture behind figure A."""
    losses = [l for l in LOSSES if data.get(l)]
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False,
                             sharex=True, sharey=True)
    pcols = plt.get_cmap('viridis')(np.linspace(0, 0.9, 4))
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        arrs = data[loss]
        for p in range(4):
            xs, m = ragged_mean(arrs, p)
            ax.plot(xs, m, color=pcols[p], lw=2.0,
                    label=PARAM_LABEL[p] if c == 0 else None)
        ax.set_title(f'{loss}', fontsize=11)
        ax.set_xlabel('epoch')
        if c == 0:
            ax.set_ylabel('parameter L2 norm')
            ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.suptitle(f'Per-parameter weight norms — {arch.upper()}  ({info}, '
                 'across-mouse mean)', y=1.04, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f'B_allparams_{arch}')


def fig_C_spat_vs_temp(spat, temp, out_dir, info):
    """Per-loss spat vs temp output-weight norm — does the SBC (temp) entropy
    penalty keep W_out smaller than the unregularised PPC (spat)?"""
    losses = [l for l in LOSSES if spat.get(l) or temp.get(l)]
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False,
                             sharex=True, sharey=True)
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        for arch, arrs, col, lbl in [('spat', spat.get(loss, []), '#d95f02', 'PPC (spat)'),
                                     ('temp', temp.get(loss, []), '#1f78b4', 'SBC (temp)')]:
            if not arrs:
                continue
            xs, m = ragged_mean(arrs, WOUT)
            ax.plot(xs, m, color=col, lw=2.2, label=lbl if c == 0 else None)
        ax.set_title(f'{loss}', fontsize=11)
        ax.set_xlabel('epoch')
        if c == 0:
            ax.set_ylabel('‖W_out‖')
            ax.legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle(f'Output-weight norm: PPC vs SBC  ({info}, across-mouse mean)\n'
                 'does the SBC entropy penalty act as implicit weight reg?',
                 y=1.05, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'C_Wout_spat_vs_temp')


def _save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(f'  -> {stem}.png/.svg')


def main(run_name, target, window, bin_ms, split, results_root, out_root):
    dpu.set_style()
    info = f'{target} {window} {bin_ms}ms {split}'
    print(f'Weight evolution: {run_name} | {info}')
    spat = {l: load_cell(results_root, run_name, target, l, window, bin_ms, split, 'spat')
            for l in LOSSES}
    temp = {l: load_cell(results_root, run_name, target, l, window, bin_ms, split, 'temp')
            for l in LOSSES}
    for l in LOSSES:
        print(f'  {l:12s} spat={len(spat[l])} mice, temp={len(temp[l])} mice')
    if not any(spat.values()) and not any(temp.values()):
        raise SystemExit('No tracked histories found for this cell.')
    out_dir = Path(out_root) / run_name / 'weight_evolution'
    # Both architectures: PPC (spat) and SBC (temp).
    for arch, data in (('spat', spat), ('temp', temp)):
        if any(data.values()):
            fig_A_wout_by_loss(data, arch, out_dir, info)
            fig_B_allparams(data, arch, out_dir, info)
        else:
            print(f'  [skip] {arch}: no tracked histories')
    fig_C_spat_vs_temp(spat, temp, out_dir, info)
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(a.run_name, a.target, a.window, a.bin_ms, a.split,
         a.results_root, a.out_root)
