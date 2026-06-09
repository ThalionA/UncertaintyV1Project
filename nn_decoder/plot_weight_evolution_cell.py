# -*- coding: utf-8 -*-
"""Weight-norm evolution for ONE cell of a flat loss-comparison run.

The production weight-evolution plot (`plot_loss_sweep.plot_weight_norms`) facets
per-layer norms by loss with cryptic 'param k' labels, and `plot_kl_js_training`
only walks the entropy-lambda sweep tree. This script targets a single
(target, window, bin, split) cell of a *flat* run (e.g. `loss_comparison_v1`)
and answers the meeting's actual question directly:

    Does the loss that doesn't punish over-confidence (PCA) drive the OUTPUT-layer
    weights larger than the calibrated losses (KL/JS/CE)? And does the temporal
    entropy penalty act as implicit weight regularisation (temp W_out < spat)?

Each saved checkpoint stores, per epoch, the L2 norm of every parameter tensor
(`history['weight_norms']`, shape (epochs, 4) for a 1-hidden-layer MLP):
    param 0 = W_in  (H x n_neurons)   param 1 = b_in (H)
    param 2 = W_out (n_cats x H)       param 3 = b_out (n_cats)
The output-layer weight (param 2) sets how large the softmax logits — hence how
peaky the posterior — can get, so it is the curve of interest.

Figures (PNG+SVG) under figures/loss_sweep_plots/<run>/weight_evolution/:
  A_Wout_by_loss_<arch>     output-weight norm vs epoch, one line per loss
  B_allparams_<arch>        per-loss small multiples, all 4 params relabelled
  C_Wout_spat_vs_temp       per-loss, spat vs temp W_out (entropy-reg-as-reg test)
  D_normalised_norms_<arch> raw vs fan-in-normalised ‖W_in‖, ‖W_out‖ by loss
  E_mean_std_<arch>         weight mean ± std vs snapshot-epoch (not just L2 norm)
  F_init_vs_final_<arch>    weight histograms at init (epoch 0) vs final, per loss

A/B/C read the per-epoch scalar `weight_norms`; D/E/F read the full weight
tensors in `history['state_dicts']` (snapshotted every `--snapshot-every` epochs,
epoch 0 = initialisation), which lets us answer the 2026-06-03 meeting's asks:
  - normalise the L2 norm by the number of input neurons (D);
  - report weight mean + std instead of a single L2 norm (E);
  - why are the weight norms different from the beginning, and is the init odd? (E/F).

Lines end where each mouse's training stopped (early stopping), so the x-extent
itself shows how long each loss trained. A ★ marks the as-deployed (best-val) epoch.

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
WIN = 0   # index of the input-layer weight tensor in weight_norms
WOUT = 2  # index of the output-layer weight tensor
# state_dict keys for the 1-hidden-layer MLP (see SimpleFlexibleNNClassifier).
KEY_WIN = 'layers.0.weight'   # (H, n_neurons)  -> fan-in = n_neurons
KEY_BIN = 'layers.0.bias'     # (H,)            -> init constant 0
KEY_WOUT = 'layers.1.weight'  # (n_cats, H)     -> fan-in = H
KEY_BOUT = 'layers.1.bias'    # (n_cats,)       -> init constant 0
# The four parameter groups, in forward order, with the snapshot dict key used
# in load_cell_snapshots and a human label.
PARAM_GROUPS = [('Win', 'W_in (hidden×neurons)'), ('bin', 'b_in (hidden)'),
                ('Wout', 'W_out (cats×hidden)'), ('bout', 'b_out (cats)')]


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
# Snapshot-tensor loading (for the fan-in-normalised / mean-std / init figures)
# ----------------------------------------------------------------------

def _as_np(x):
    """state_dict values may be torch tensors or already-numpy. Normalise."""
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def _best_epoch(hist):
    """As-deployed (best-val) epoch, 1-indexed to match the norm-curve x-axis.
    Prefer an explicit scalar if fit_model recorded one; else argmin val loss."""
    for k in ('best_epoch', 'best_val_epoch'):
        v = hist.get(k)
        if v is not None and np.isscalar(v):
            return int(v) + 1  # stored 0-indexed
    for k in ('val_total_loss', 'val_fit_loss'):
        v = hist.get(k)
        if v is not None and len(np.asarray(v)):
            return int(np.nanargmin(np.asarray(v, dtype=float))) + 1
    return None


def load_cell_snapshots(run_root, run_name, target, loss, window, bin_ms, split, arch):
    """Per-mouse weight-tensor snapshots for one (loss, arch). Returns a list of
    dicts: {epochs:(S,), Win:(S,H,Nin), Wout:(S,cats,H), best:int|None}."""
    ck_dir = Path(run_root) / run_name / _slug(target, loss, window, bin_ms) / 'checkpoints'
    out = []
    if not ck_dir.is_dir():
        return out
    for pt in sorted(ck_dir.glob(f'mouse_*_{split}.pt')):
        ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        if not (isinstance(ck, dict) and arch in ck and isinstance(ck[arch], dict)):
            continue
        hist = ck[arch].get('history')
        if not hist:
            continue
        sds = hist.get('state_dicts')
        eps = hist.get('snapshot_epochs')
        if sds is None or eps is None or not len(sds):
            continue
        Win = np.stack([_as_np(sd[KEY_WIN]) for sd in sds])
        Wout = np.stack([_as_np(sd[KEY_WOUT]) for sd in sds])
        bin_ = np.stack([_as_np(sd[KEY_BIN]) for sd in sds])
        bout = np.stack([_as_np(sd[KEY_BOUT]) for sd in sds])
        out.append({'epochs': np.asarray(eps, dtype=int), 'Win': Win,
                    'Wout': Wout, 'bin': bin_, 'bout': bout,
                    'best': _best_epoch(hist)})
    return out


def snap_ragged_mean(snaps, key, reduce_fn):
    """Apply reduce_fn (e.g. fan-in-normalised norm, or std) to each mouse's
    snapshot tensor at each snapshot epoch, then mean over mice at matched
    epochs. Returns (epochs, mean_over_mice). Assumes a shared epoch grid (the
    sweep snapshots on a fixed schedule); aligns on the shortest grid."""
    if not snaps:
        return np.array([]), np.array([])
    grids = [s['epochs'] for s in snaps]
    common = grids[int(np.argmin([len(g) for g in grids]))]
    vals = np.full((len(snaps), len(common)), np.nan)
    for i, s in enumerate(snaps):
        idx = {int(e): j for j, e in enumerate(s['epochs'])}
        for j, e in enumerate(common):
            if int(e) in idx:
                vals[i, j] = reduce_fn(s[key][idx[int(e)]])
    return common, np.nanmean(vals, axis=0)


# fan-in-normalised L2 norm: ‖W‖ / sqrt(fan_in). For W_in fan_in = n_neurons
# (the meeting's "normalise L2 norm by N input neurons"); for W_out fan_in = H.
def _norm_per_fanin(W):
    return float(np.linalg.norm(W) / np.sqrt(W.shape[1]))

def _norm_raw(W):
    return float(np.linalg.norm(W))

def _wmean(W):
    return float(np.mean(W))

def _wstd(W):
    return float(np.std(W))


# ----------------------------------------------------------------------

def fig_A_wout_by_loss(data, arch, out_dir, info, best_by_loss=None):
    """Output-weight norm vs epoch, one bold across-mouse line per loss, with
    faint per-mouse lines. The headline: does PCA inflate W_out vs KL/JS/CE?
    A ★ marks the across-mouse mean best-val (as-deployed) epoch."""
    best_by_loss = best_by_loss or {}
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
        # ★ as-deployed (best-val) epoch — weights early stopping restores
        bests = [b for b in best_by_loss.get(loss, []) if b]
        if bests:
            be = int(round(np.mean(bests)))
            yi = [a[min(be, a.shape[0]) - 1, WOUT] for a in arrs]
            ax.scatter([be], [np.nanmean(yi)], color=LOSS_COLOR[loss], s=140,
                       marker='*', zorder=6, edgecolor='k', lw=0.6)
    ax.set_xlabel('epoch')
    ax.set_ylabel('output-layer weight L2 norm  ‖W_out‖')
    ax.set_title(f'Output-weight growth by loss — {arch.upper()}  ({info})\n'
                 'faint = per mouse; bold = across-mouse mean; dot = mean stop; '
                 'star = mean best-val (as-deployed) epoch')
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
    """Per-loss spat vs temp output-weight norm — does the temporal entropy
    penalty keep W_out smaller than the unregularised spatial?"""
    losses = [l for l in LOSSES if spat.get(l) or temp.get(l)]
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False,
                             sharex=True, sharey=True)
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        for arch, arrs, col, lbl in [('spat', spat.get(loss, []), '#d95f02', 'spatial'),
                                     ('temp', temp.get(loss, []), '#1f78b4', 'temporal')]:
            if not arrs:
                continue
            xs, m = ragged_mean(arrs, WOUT)
            ax.plot(xs, m, color=col, lw=2.2, label=lbl if c == 0 else None)
        ax.set_title(f'{loss}', fontsize=11)
        ax.set_xlabel('epoch')
        if c == 0:
            ax.set_ylabel('‖W_out‖')
            ax.legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle(f'Output-weight norm: spatial vs temporal  ({info}, across-mouse mean)\n'
                 'does the temporal entropy penalty act as implicit weight reg?',
                 y=1.05, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'C_Wout_spat_vs_temp')


def fig_D_normalised_norms(snaps_by_loss, arch, out_dir, info):
    """Raw vs fan-in-normalised weight norms, by loss. Top row = raw ‖W‖ (what
    A/B show); bottom row = ‖W‖/√(fan-in) — ‖W_in‖ normalised by N input neurons
    (the meeting's ask) and ‖W_out‖ by H. Normalising removes the trivial
    'bigger layer ⇒ bigger norm' factor so cross-layer / cross-arch comparison
    is honest. Columns: W_in (left), W_out (right)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    specs = [(0, 0, 'Win', _norm_raw,        'raw  ‖W_in‖'),
             (0, 1, 'Wout', _norm_raw,       'raw  ‖W_out‖'),
             (1, 0, 'Win', _norm_per_fanin,  '‖W_in‖ / √N_in  (per input neuron)'),
             (1, 1, 'Wout', _norm_per_fanin, '‖W_out‖ / √H  (per hidden unit)')]
    for r, c, key, fn, ylab in specs:
        ax = axes[r][c]
        for loss in LOSSES:
            snaps = snaps_by_loss.get(loss, [])
            if not snaps:
                continue
            xs, m = snap_ragged_mean(snaps, key, fn)
            if len(xs):
                ax.plot(xs, m, color=LOSS_COLOR[loss], lw=2.2, marker='o',
                        ms=3.5, label=loss if (r == 0 and c == 0) else None)
        ax.set_ylabel(ylab)
        if r == 1:
            ax.set_xlabel('snapshot epoch')
    axes[0][0].legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.suptitle(f'Weight norms: raw vs fan-in-normalised — {arch.upper()}  '
                 f'({info}, across-mouse mean)', y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f'D_normalised_norms_{arch}')


def fig_E_mean_std(snaps_by_loss, arch, out_dir, info):
    """Weight mean ± std vs snapshot-epoch, by loss — the distributional view the
    meeting asked for ('mean + std of weights instead of L2 norms'). A single L2
    norm conflates a distribution that shifted (mean≠0) with one that merely
    spread (std grew). Line = mean over all entries; band = ±std. Dashed grey =
    init (epoch-0) std, so 'why are norms different from the beginning?' is read
    straight off the drift of the band away from it. Columns: W_in, W_out."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, key, ttl in ((axes[0], 'Win', 'W_in (hidden×neurons)'),
                         (axes[1], 'Wout', 'W_out (cats×hidden)')):
        init_std_ref = []
        for loss in LOSSES:
            snaps = snaps_by_loss.get(loss, [])
            if not snaps:
                continue
            xs, mean = snap_ragged_mean(snaps, key, _wmean)
            _, std = snap_ragged_mean(snaps, key, _wstd)
            if not len(xs):
                continue
            col = LOSS_COLOR[loss]
            ax.plot(xs, mean, color=col, lw=2.0, marker='o', ms=3, label=loss)
            ax.fill_between(xs, mean - std, mean + std, color=col, alpha=0.12)
            init_std_ref.append(std[0])
        if init_std_ref:
            s0 = float(np.nanmean(init_std_ref))
            ax.axhline(s0, ls='--', lw=1, color='0.4')
            ax.axhline(-s0, ls='--', lw=1, color='0.4')
            ax.text(ax.get_xlim()[1], s0, '  ±init std', va='center',
                    fontsize=8, color='0.4')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel('snapshot epoch')
        ax.set_ylabel('weight value  (mean ± std over entries)')
    axes[0].legend(frameon=False, fontsize=8.5, loc='best')
    fig.suptitle(f'Weight distribution over training — {arch.upper()}  '
                 f'({info}, across-mouse mean)', y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f'E_mean_std_{arch}')


def fig_F_init_vs_final(snaps_by_loss, arch, out_dir, info):
    """Histograms of parameter values at init (epoch 0) vs final snapshot, for
    ALL FOUR parameter groups (rows: W_in, b_in, W_out, b_out) × losses (cols).
    Interrogates the 'are the parameters initialised in a weird way?' question:
    a clean symmetric init that simply broadens is normal. NB the biases init to
    exactly 0 (constant_ init), so their grey 'init' bar is a single spike at 0 —
    the final spread shows how far each bias group moved off zero. Per-row x is
    shared (so init/final are comparable across losses); y is per-row (weights
    and biases live on very different scales). Pooled over mice."""
    losses = [l for l in LOSSES if snaps_by_loss.get(l)]
    if not losses:
        return
    ncol = len(losses)
    nrow = len(PARAM_GROUPS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.6 * nrow),
                             squeeze=False, sharex='row')
    for r, (key, label) in enumerate(PARAM_GROUPS):
        # shared bin edges for this parameter-group row, across all losses
        allvals = np.concatenate([np.concatenate([s[key].ravel() for s in snaps_by_loss[l]])
                                  for l in losses])
        lo, hi = float(allvals.min()), float(allvals.max())
        if lo == hi:                       # degenerate (e.g. all-zero init only)
            lo, hi = lo - 1e-3, hi + 1e-3
        bins = np.linspace(lo, hi, 60)
        for c, loss in enumerate(losses):
            ax = axes[r][c]
            snaps = snaps_by_loss[loss]
            init = np.concatenate([s[key][0].ravel() for s in snaps])
            final = np.concatenate([s[key][-1].ravel() for s in snaps])
            ax.hist(init, bins=bins, color='0.6', alpha=0.7, density=True,
                    label=f'init  μ={init.mean():+.3f} σ={init.std():.3f}')
            ax.hist(final, bins=bins, histtype='step', lw=2,
                    color=LOSS_COLOR[loss], density=True,
                    label=f'final μ={final.mean():+.3f} σ={final.std():.3f}')
            if r == 0:
                ax.set_title(loss, fontsize=11)
            if c == 0:
                ax.set_ylabel(f'{label}\ndensity', fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel('parameter value')
            ax.legend(frameon=False, fontsize=6.2, loc='upper right')
    fig.suptitle(f'Parameter values at init vs final, all 4 groups — {arch.upper()}'
                 f'  ({info}, pooled over mice; biases init to 0)',
                 y=1.005, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f'F_init_vs_final_{arch}')


# Canonical PNG+SVG sink with the ≤1600px PNG cap (was an uncapped fixed-dpi=140
# local helper). Aliased to ``_save`` so existing call sites are unchanged.
from figsave import save_fig as _save


def main(run_name, target, window, bin_ms, split, results_root, out_root):
    dpu.set_style()
    info = f'{target} {window} {bin_ms}ms {split}'
    print(f'Weight evolution: {run_name} | {info}')
    spat = {l: load_cell(results_root, run_name, target, l, window, bin_ms, split, 'spat')
            for l in LOSSES}
    temp = {l: load_cell(results_root, run_name, target, l, window, bin_ms, split, 'temp')
            for l in LOSSES}
    # Full weight-tensor snapshots (epoch 0 + every --snapshot-every) for D/E/F.
    spat_snap = {l: load_cell_snapshots(results_root, run_name, target, l, window, bin_ms, split, 'spat')
                 for l in LOSSES}
    temp_snap = {l: load_cell_snapshots(results_root, run_name, target, l, window, bin_ms, split, 'temp')
                 for l in LOSSES}
    for l in LOSSES:
        print(f'  {l:12s} spat={len(spat[l])} mice ({len(spat_snap[l])} w/ snapshots), '
              f'temp={len(temp[l])} mice ({len(temp_snap[l])} w/ snapshots)')
    if not any(spat.values()) and not any(temp.values()):
        raise SystemExit('No tracked histories found for this cell.')
    out_dir = Path(out_root) / run_name / 'weight_evolution'
    # Both architectures: spatial and temporal.
    for arch, data, snap in (('spat', spat, spat_snap), ('temp', temp, temp_snap)):
        if any(data.values()):
            best_by_loss = {l: [s['best'] for s in snap.get(l, [])] for l in LOSSES}
            fig_A_wout_by_loss(data, arch, out_dir, info, best_by_loss=best_by_loss)
            fig_B_allparams(data, arch, out_dir, info)
        else:
            print(f'  [skip] {arch}: no tracked histories')
        if any(snap.values()):
            fig_D_normalised_norms(snap, arch, out_dir, info)
            fig_E_mean_std(snap, arch, out_dir, info)
            fig_F_init_vs_final(snap, arch, out_dir, info)
        else:
            print(f'  [skip] {arch}: no weight snapshots (re-run with --snapshot-every)')
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
