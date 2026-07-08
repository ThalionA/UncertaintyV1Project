# -*- coding: utf-8 -*-
"""Trained-as-target round-trip, crossed with the refit loss.

Question (Theo, 2026-07-08): does taking *fitted* posteriors as the training
target — instead of the noisy IO posteriors — and refitting them under a
different loss fix the PCA over-sharpening / overfitting?

The round-trip's whole point is that a network's OWN output is, by construction,
EXACTLY realizable by a net of the same capacity: there is zero irreducible
approximation error and no target noise to overfit. So refitting on fitted
targets isolates the two competing stories for PCA's peakiness:

  * OVERFITTING-TO-UNACHIEVABLE-TARGET — the net dumps the irreducible residual
    (real IO target it can't reproduce) into the loss-blind shape subspace as
    spikes. If true, an achievable target removes the residual and PCA should
    stop over-sharpening.
  * LOSS-GEOMETRY DRIFT — the evar-weighted L2 leaves the shape subspace
    unconstrained and there is no restoring force, so peakiness ratchets up
    regardless of whether the target is achievable. If true, PCA over-sharpens
    even a broad, achievable KL target.

Design: a 3 (target source) x 2 (refit loss) matrix, Q / half / 100 ms, 6 mice.

  target source            base fit (its full_decoded -> new target)
  ----------------------   -----------------------------------------
  realIO   real IO posterior (no base)              target_source='real'
  pcaFit   PCA decoder's own (peaky) output         loss_comparison_v1/Q_PCA_..._all
  klFit    KL  decoder's (calibrated) output         loss_comparison_v1/Q_KL_...

  refit loss  in  {PCA, KL}.

Every cell is trained with the FULL trajectory exposed: no early stopping
(patience=0), a monitored val carve-out (val_frac=0.2, monitor_val) so we get a
val curve, weight snapshots every 10 epochs, and entropy_lambda=0 so nothing
external shapes the posterior width — the cleanest read of what the loss itself
does. We then read, per cell (spatial arch = PPC, the canonical peaky one):

  1. full per-epoch TRAIN vs VAL fit-loss curves  (the overfitting gap);
  2. decoded peakiness (mean max-prob) vs epoch, by re-decoding X_test at every
     weight snapshot, against the cell's own TARGET peakiness (dashed line):
     does the refit sail PAST its achievable target?

Per-(cell, mouse) curves are cached under results/roundtrip_refit/ so the run
is resumable; figures land in figures/roundtrip_refit/ (PNG+SVG).

Usage
-----
    python diagnostics/roundtrip_loss_refit.py            # run (cached) + plot
    python diagnostics/roundtrip_loss_refit.py --plot-only
    python diagnostics/roundtrip_loss_refit.py --mice 0 1 # subset while iterating
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps                                   # noqa: E402
from run_experiment import run_animal_decoder                  # noqa: E402
from training.config import Config                             # noqa: E402
# Reuse the snapshot -> re-decode -> peakiness machinery.
from diagnostics.decode_entropy_trajectory import (            # noqa: E402
    _build_model, _decode, _reduce,
)

RESULTS = Path(__file__).resolve().parent.parent / 'results'
FIG_ROOT = Path(__file__).resolve().parent.parent / 'figures' / 'roundtrip_refit'
CACHE = RESULTS / 'roundtrip_refit'
BASE_RUN = RESULTS / 'loss_comparison_v1'
SPLIT = 'stratified_balanced'
METRIC = 'maxprob'          # sharper read than entropy for the jagged spatial decoder
SNAP_EVERY = 10
N_EPOCHS = 200

# (cell name, target_source, base .mat for full_decoded | None, refit loss)
CELLS = [
    ('realIO',  'real',          None,                              'PCA'),
    ('realIO',  'real',          None,                              'KL'),
    ('pcaFit',  'recovery_spat', BASE_RUN / 'Q_PCA_half_100ms_all', 'PCA'),
    ('pcaFit',  'recovery_spat', BASE_RUN / 'Q_PCA_half_100ms_all', 'KL'),
    ('klFit',   'recovery_spat', BASE_RUN / 'Q_KL_half_100ms',      'PCA'),
    ('klFit',   'recovery_spat', BASE_RUN / 'Q_KL_half_100ms',      'KL'),
]
TARGET_ORDER = ['realIO', 'pcaFit', 'klFit']
LOSS_ORDER = ['PCA', 'KL']
TARGET_TITLE = {
    'realIO': 'target = real IO posterior',
    'pcaFit': 'target = PCA decoder output (peaky)',
    'klFit':  'target = KL decoder output (calibrated)',
}


def _cell_key(name, loss):
    return f'{name}_{loss}'


def build_config(refit_loss, target_source, base_file):
    """Legacy-dict config for one refit cell. All cells share window/bin/arch/lr
    with the loss_comparison_v1 bases; only the loss + target source vary. Full
    trajectory (no early stop), monitored val, snapshots, entropy_lambda=0."""
    cfg = Config(
        target_type='Q',
        loss_func=refit_loss,
        time_window='half',
        bin_size_ms=100,
        hidden_sizes=[32],
        activation_function='tanh',
        learning_rate=1.252e-3,        # matches the loss_comparison_v1 base cells
        num_epochs=N_EPOCHS,
        entropy_lambda=0.0,            # purest loss-geometry read
        pca_basis='all_trials',
        split_type=SPLIT,
        track_training_history=True,
        weight_snapshot_every=SNAP_EVERY,
        monitor_val=True,
        val_frac=0.2,
        patience=0,                    # full fixed schedule, no early-stop truncation
        val_fraction=0.2,
    ).to_legacy_dict()
    # Runtime-only keys read directly by run_experiment (not Config fields).
    cfg['target_source'] = target_source
    if base_file is not None:
        # base_file is the run-slug DIRECTORY; the recovery branch loads the .mat.
        cfg['base_file_path'] = str(base_file / f'{SPLIT}.mat')
    return cfg


def _mouse_ids(base_file, requested):
    if requested is not None:
        return list(requested)
    if base_file is not None:
        d = sio.loadmat(str(base_file / f'{SPLIT}.mat'), simplify_cells=True)['results']
        return sorted(int(k.split('_')[1]) for k in d)
    # realIO: mirror the base cohort (mice 0..5).
    d = sio.loadmat(str(BASE_RUN / 'Q_PCA_half_100ms_all' / f'{SPLIT}.mat'),
                    simplify_cells=True)['results']
    return sorted(int(k.split('_')[1]) for k in d)


def _peakiness_trajectory(hist, X_test, arch, activation):
    """maxprob at each weight snapshot by re-decoding X_test."""
    eps = np.asarray(hist['snapshot_epochs'], dtype=int)
    vals = []
    for sd in hist['state_dicts']:
        model = _build_model(sd, activation)
        vals.append(_reduce(_decode(model, X_test, arch), METRIC))
    return eps, np.asarray(vals)


def run_cell_mouse(name, target_source, base_file, refit_loss, mouse_id, activation='tanh'):
    """Train one (cell, mouse); return the curves dict, or load it from cache."""
    ck = _cell_key(name, refit_loss)
    cache = CACHE / ck / f'mouse_{mouse_id}.npz'
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))

    cfg = build_config(refit_loss, target_source, base_file)
    print(f'  [{ck}] mouse {mouse_id}: training ({N_EPOCHS} ep, loss={refit_loss}, '
          f'target={target_source}) ...', flush=True)
    res = run_animal_decoder(cfg, mouse_id)

    out = {}
    for arch in ('spat', 'temp'):
        sub = res['Checkpoints'][arch]
        hist = sub['history']
        X_test = sub['X_test']
        eps, peak = _peakiness_trajectory(hist, X_test, arch, activation)
        # target peakiness for this cell/arch (mean over trials, same measure)
        tgt = np.asarray(res['Dist'][arch]['target'], float)
        out[f'{arch}_snap_epochs'] = eps
        out[f'{arch}_peak'] = peak
        out[f'{arch}_target_peak'] = np.float64(_reduce(tgt, METRIC))
        out[f'{arch}_train_fit'] = np.asarray(hist['train_fit_loss'], float)
        out[f'{arch}_val_fit'] = np.asarray(hist.get('val_fit_loss', []), float)
        out[f'{arch}_train_yard'] = np.asarray(hist.get('train_pca_yardstick', []), float)
        out[f'{arch}_val_yard'] = np.asarray(hist.get('val_pca_yardstick', []), float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def run_all(mice):
    for name, tsrc, base, loss in CELLS:
        ids = _mouse_ids(base, mice)
        for m in ids:
            run_cell_mouse(name, tsrc, base, loss, m)
    print('All cells trained / cached.')


# ---------------------------------------------------------------- analysis

def _load_cell(name, loss, mice):
    ids = mice if mice is not None else _mouse_ids(
        dict(CELLS_BY_KEY)[_cell_key(name, loss)][2], None)
    curves = []
    for m in ids:
        f = CACHE / _cell_key(name, loss) / f'mouse_{m}.npz'
        if f.exists():
            curves.append(dict(np.load(f, allow_pickle=True)))
    return curves


CELLS_BY_KEY = {_cell_key(n, l): (n, t, b, l) for (n, t, b, l) in CELLS}


def _mean_over_mice(curves, key):
    """nanmean over mice on the shortest common length (curves are same-length
    within a cell — fixed epochs, fixed snapshot grid)."""
    arrs = [c[key] for c in curves if key in c and np.size(c[key])]
    if not arrs:
        return None, None
    n = min(len(a) for a in arrs)
    M = np.vstack([a[:n] for a in arrs])
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(M.shape[0])


def plot_peakiness_grid(mice, arch='spat'):
    """3x2 grid: rows=target source, cols=refit loss. Peakiness vs epoch, with
    each cell's own target peakiness as a dashed line. THE over-sharpening read."""
    ps.apply()
    fig, axes = plt.subplots(len(TARGET_ORDER), len(LOSS_ORDER),
                             figsize=ps.figsize(2.0, 2.4), sharex=True, sharey=True)
    for r, tgt in enumerate(TARGET_ORDER):
        for c, loss in enumerate(LOSS_ORDER):
            ax = axes[r, c]
            curves = _load_cell(tgt, loss, mice)
            if not curves:
                ax.set_visible(False); continue
            eps = curves[0][f'{arch}_snap_epochs']
            peak, sem = _mean_over_mice(curves, f'{arch}_peak')
            tpk = float(np.nanmean([c[f'{arch}_target_peak'] for c in curves]))
            ax.plot(eps[:len(peak)], peak, color=ps.color(loss), lw=2.6, marker='o', ms=3)
            if sem is not None:
                ax.fill_between(eps[:len(peak)], peak - sem, peak + sem,
                                color=ps.color(loss), alpha=0.2, lw=0)
            ax.axhline(tpk, ls='--', lw=1.6, color='0.35',
                       label=f'target ({tpk:.3f})')
            ax.text(0.03, 0.94, f'{tgt} · refit {loss}\nfinal {peak[-1]:.3f} vs tgt {tpk:.3f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
            if r == 0:
                ax.set_title(f'refit loss = {loss}')
            if c == 0:
                ax.set_ylabel(f'{TARGET_TITLE[tgt]}\ndecoded max-prob')
            if r == len(TARGET_ORDER) - 1:
                ax.set_xlabel('epoch (weight snapshot)')
    fig.suptitle(f'Round-trip: does refitting fitted targets stop over-sharpening?  '
                 f'({arch} arch, Q half 100ms, {"6" if mice is None else len(mice)} mice)',
                 fontsize=12)
    ps.save_fig(fig, FIG_ROOT, f'peakiness_grid_{arch}')


def plot_trainval_grid(mice, arch='spat'):
    """3x2 grid of full train vs val fit-loss curves (the overfitting gap).
    Within a cell train/val share a scale; across cells the loss differs, so
    each panel is autoscaled and read on its own."""
    ps.apply()
    fig, axes = plt.subplots(len(TARGET_ORDER), len(LOSS_ORDER),
                             figsize=ps.figsize(2.0, 2.4), sharex=True)
    for r, tgt in enumerate(TARGET_ORDER):
        for c, loss in enumerate(LOSS_ORDER):
            ax = axes[r, c]
            curves = _load_cell(tgt, loss, mice)
            if not curves:
                ax.set_visible(False); continue
            tr, _ = _mean_over_mice(curves, f'{arch}_train_fit')
            va, _ = _mean_over_mice(curves, f'{arch}_val_fit')
            x = np.arange(1, len(tr) + 1)
            ax.plot(x, tr, color='steelblue', lw=2.2, label='train fit')
            if va is not None and np.size(va):
                ax.plot(np.arange(1, len(va) + 1), va, color='firebrick', lw=2.2, label='val fit')
            ax.text(0.03, 0.06, f'{tgt} · refit {loss}', transform=ax.transAxes,
                    va='bottom', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
            if r == 0:
                ax.set_title(f'refit loss = {loss}')
            if c == 0:
                ax.set_ylabel(f'{TARGET_TITLE[tgt]}\nfit loss')
            if r == len(TARGET_ORDER) - 1:
                ax.set_xlabel('epoch')
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Round-trip: train vs val fit-loss ({arch} arch, Q half 100ms)', fontsize=12)
    ps.save_fig(fig, FIG_ROOT, f'trainval_grid_{arch}')


def print_scorecard(mice, arch='spat'):
    print(f'\n=== SCORECARD ({arch} arch) — final decoded max-prob vs target ===')
    print(f'{"cell":22s} {"final":>8s} {"target":>8s} {"ratio":>7s}')
    for tgt in TARGET_ORDER:
        for loss in LOSS_ORDER:
            curves = _load_cell(tgt, loss, mice)
            if not curves:
                continue
            peak, _ = _mean_over_mice(curves, f'{arch}_peak')
            tpk = float(np.nanmean([c[f'{arch}_target_peak'] for c in curves]))
            print(f'{_cell_key(tgt, loss):22s} {peak[-1]:8.3f} {tpk:8.3f} '
                  f'{peak[-1] / tpk:7.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--mice', type=int, nargs='*', default=None,
                    help='subset of mouse ids (default: all 6)')
    a = ap.parse_args()
    if not a.plot_only:
        run_all(a.mice)
    for arch in ('spat', 'temp'):
        plot_peakiness_grid(a.mice, arch)
        plot_trainval_grid(a.mice, arch)
        print_scorecard(a.mice, arch)
    print(f'\nFigures -> {FIG_ROOT.resolve()}')


if __name__ == '__main__':
    main()
