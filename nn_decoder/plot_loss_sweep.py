# -*- coding: utf-8 -*-
"""Plot the cross-loss diagnostic suite for a Q-target loss sweep.

Companion to ``run_loss_sweep.py``. Reads the nested results tree
written by ``training/run.py`` (one ``stratified_balanced.mat`` per
``Q_<LOSS>_half_100ms[_all]/`` slug) plus the per-mouse ``.pt``
checkpoints that carry the training history, and produces the
diagnostic plot set agreed at the 2026-05-27 meeting:

  1. Example posteriors per loss (visual gut check; matches the
     screenshot Theo sent showing spatial spiking on a smooth target).
  2. Peakiness histograms: max(P) and fractional entropy
     H(P)/log(n_cats) per trial, faceted by (loss, arch).
  3. Per-bin temporal peakiness — the per-time-bin distributions
     (``dist['temp']['decoded_samp']``) before Jensen-via-time-average
     smoothing. Isolates loss-driven peakiness from architectural
     smoothing.
  4. Cross-loss held-out PCA-projected loss — per (loss, arch)
     bars + per-mouse dots, both raw and variance-normalised. The
     PCA basis is the same one the training loop saw, so values
     are on a single yardstick.
  5. Per-epoch ``train_pca_yardstick`` training curves (one panel
     per loss, both archs overlaid).  Needs ``.pt`` history.
  6. Per-layer weight L2 norms over training (one panel per loss,
     per-arch).  Needs ``.pt`` history.
  7. Train-vs-test gap: final training PCA-yardstick loss vs the
     held-out test PCA-projected loss, per (loss, mouse, arch).
     Needs both ``.pt`` history and ``.mat``.
  8. Train-vs-val curves overlaid (one panel per loss × arch). Only
     drawn when ``val_frac > 0`` was set in the run's Config — the
     val curves live in the history dict alongside the train curves.
     The early-stopping epoch reads off each panel as the val
     plateau (red dotted line); the train-val gap reads off as the
     vertical offset.
  9. Posterior evolution across saved snapshot epochs — for one
     mouse, one architecture (default spatial), and a few example
     trials, the predicted distribution at every snapshot. Reads
     off when the peakiness develops over training. Requires torch
     to reconstruct saved ``state_dict`` snapshots and forward-pass
     the saved test inputs.

Plots 5–8 use a torch-less zipfile/pickle extractor when torch is
unavailable, so they still run in environments without torch (the
state_dict tensors become unusable stubs, but the numeric history
fields round-trip cleanly). Plot 9 needs torch proper.

Usage::

    python plot_loss_sweep.py --run-name new_loss_sweep_may27

Outputs land at ``figures/loss_sweep_plots/<run_name>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
from pca_loss import pca_distance  # noqa: E402
from decoder_plotting_utils import calc_pca_dist  # noqa: E402


# ----------------------------------------------------------------------
# Run-level config
# ----------------------------------------------------------------------

# Fixed loss ordering across every figure — distributional losses
# clustered together so colour-by-loss reads off the legend in the same
# order on every panel. MSE was dropped from the sweep on 2026-05-27
# after the h32 + h10 runs both showed it collapsing to the marginal
# mean (degenerate; not a useful comparison for the peakiness
# investigation). Re-add 'MSE' here and to _SLUG_SUFFIX if a future
# experiment wants it back.
LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')

# Slug suffix: PCA gets ``_all`` (the pca_basis short-name 'all_trials');
# every other loss writes the bare slug.
_SLUG_SUFFIX = {
    'PCA':         '_all',
    'CE':          '',
    'KL':          '',
    'JS':          '',
    'Wasserstein': '',
}

# The cell these plots address. Defaults reproduce the historical Q / half /
# 100 ms / stratified behaviour; generate_all() overwrites them from its
# target/window/bin/split args so the same suite runs on any cell of a
# loss-comparison grid (L, 50 ms, full window, the OOD splits). _slug() and the
# loaders read these module-level values.
TARGET = 'Q'
WINDOW = 'half'
BIN_MS = 100
SPLIT = 'stratified_balanced'


def cell_tag() -> str:
    """Filesystem tag for the current cell, used as a per-cell output subdir so
    different cells of one run don't overwrite each other's figures."""
    return f'{TARGET}_{WINDOW}_{BIN_MS}ms_{SPLIT}'


# Loss palette — keeps each loss the same colour across every figure.
# Chosen for distinguishability + colour-blind friendliness (tab10).
LOSS_COLOR = dict(zip(LOSSES, plt.get_cmap('tab10').colors))
ARCH_COLOR = {'spat': 'darkorange', 'temp': 'steelblue'}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _slug(loss: str) -> str:
    return f'{TARGET}_{loss}_{WINDOW}_{BIN_MS}ms{_SLUG_SUFFIX[loss]}'


def load_loss_sweep(run_name: str, results_root: Path | None = None) -> dict:
    """Load every loss's ``stratified_balanced.mat`` for ``run_name``.

    Returns
    -------
    dict
        ``{loss: mat_dict_or_None}``. ``None`` for losses whose .mat
        is absent (e.g. the run crashed) so downstream plot functions
        can skip-and-warn cleanly instead of raising.
    """
    if results_root is None:
        results_root = paths.RESULTS
    out = {}
    for loss in LOSSES:
        path = Path(results_root) / run_name / _slug(loss) / f'{SPLIT}.mat'
        if path.exists():
            try:
                out[loss] = sio.loadmat(str(path), simplify_cells=True)
            except Exception as exc:
                print(f"  [!] couldn't read {path}: {exc}")
                out[loss] = None
        else:
            print(f"  [!] missing: {path}")
            out[loss] = None
    return out


def _load_history_no_torch(pt_path: Path) -> dict | None:
    """Extract the per-arch history dict from a .pt without torch
    installed. The .pt is a zip of pickled data; we use a custom
    Unpickler that returns stubs for any torch.* class and substitutes
    ``None`` for tensor storage refs. Plots 5/6/7/8 only need numeric
    history fields (lists of floats and lists of lists), which survive
    this round-trip cleanly; ``state_dicts`` entries become unusable
    stubs but we don't plot them.

    Returns the bundle dict (``{arch: {...}}``) or ``None`` on failure.
    """
    import zipfile, pickle

    class _Stub:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return None
        def __getattr__(self, n): return _Stub()
        def __repr__(self): return '<stub>'

    class _NoTorchUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('torch'):
                return _Stub
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return _Stub()

    try:
        with zipfile.ZipFile(str(pt_path)) as zf:
            members = [n for n in zf.namelist() if n.endswith('data.pkl')]
            if not members:
                return None
            with zf.open(members[0]) as f:
                return _NoTorchUnpickler(f).load()
    except Exception:
        return None


def load_loss_histories(run_name: str, results_root: Path | None = None) -> dict:
    """Load per-mouse training histories from the ``.pt`` checkpoints.

    Returns
    -------
    dict
        ``{loss: {mouse_id_str: {arch: history_dict}}}``.  Uses torch
        when available; falls back to a torch-less zipfile+pickle
        extraction (state_dict tensors become unusable stubs but the
        numeric history fields — lists of floats — round-trip cleanly,
        which is all plots 5/6/7/8 need).
    """
    try:
        import torch
        have_torch = True
    except ImportError:
        torch = None
        have_torch = False
        print("  (torch not installed — using torch-less .pt extractor)")

    if results_root is None:
        results_root = paths.RESULTS
    out = {}
    for loss in LOSSES:
        slug_dir = Path(results_root) / run_name / _slug(loss)
        ckpt_dir = slug_dir / 'checkpoints'
        if not ckpt_dir.is_dir():
            continue
        out[loss] = {}
        for pt in sorted(ckpt_dir.glob(f'mouse_*_{SPLIT}.pt')):
            mid = pt.stem.split('_')[1]   # mouse_<mid>_<split>
            if have_torch:
                try:
                    bundle = torch.load(str(pt), map_location='cpu',
                                         weights_only=False)
                except Exception as exc:
                    print(f"  [!] couldn't load {pt.name}: {exc}")
                    continue
            else:
                bundle = _load_history_no_torch(pt)
                if bundle is None:
                    print(f"  [!] torch-less extract failed: {pt.name}")
                    continue
            histories = {}
            for arch in ('spat', 'temp'):
                if arch in bundle and isinstance(bundle[arch], dict) \
                        and 'history' in bundle[arch]:
                    histories[arch] = bundle[arch]['history']
            if histories:
                out[loss][mid] = histories
    return out


# ----------------------------------------------------------------------
# Math helpers (per-trial peakiness / per-trial PCA-projected loss)
# ----------------------------------------------------------------------

def _normalised_entropy(P):
    """H(p)/log(n_cats), per row. 0 = delta, 1 = uniform."""
    P = np.asarray(P, dtype=float)
    n_cats = P.shape[-1]
    row = P.sum(axis=-1, keepdims=True)
    row = np.where(row > 0, row, 1.0)
    P = P / row
    H = -np.sum(P * np.log(P + 1e-12), axis=-1)
    return H / np.log(n_cats)


def _max_prob(P):
    """Max probability per row — the 'how peaky' scalar."""
    return np.asarray(P).max(axis=-1)


def _per_mouse_pca_test_loss(mat, arch):
    """Per-mouse mean test PCA-projected loss for one arch.
    ``mat`` is one loss's loadmat output."""
    out = {}
    for mid, m_data in mat['results'].items():
        dist = m_data['Dist']
        pcs, evar = dist.get('pcs'), dist.get('explained_var')
        if pcs is None or np.asarray(pcs).size == 0:
            out[mid] = np.nan
            continue
        per_trial = pca_distance(
            dist[arch]['decoded'], dist[arch]['target'], pcs, evar)
        out[mid] = float(np.nanmean(per_trial))
    return out


def _per_mouse_variance_baseline(mat, arch):
    """Variance baseline per mouse — test-target mean as centroid (no
    train-target reload since sandbox can't reach the VR export here;
    matches the fallback the plot helper uses for synthesised targets).

    Returns ``{mouse_id_str: float}``.
    """
    out = {}
    for mid, m_data in mat['results'].items():
        dist = m_data['Dist']
        pcs, evar = dist.get('pcs'), dist.get('explained_var')
        if pcs is None or np.asarray(pcs).size == 0:
            out[mid] = np.nan
            continue
        test_target = np.asarray(dist[arch]['target'])
        centroid = np.nanmean(test_target, axis=0)
        proj_t = test_target @ np.asarray(pcs).T
        proj_c = centroid @ np.asarray(pcs).T
        var = np.asarray(evar)
        per_trial = np.sum(var * (proj_t - proj_c) ** 2, axis=-1) * 100
        out[mid] = float(np.nanmean(per_trial))
    return out


# ----------------------------------------------------------------------
# Plot 1 — Example posteriors
# ----------------------------------------------------------------------

def plot_example_posteriors(sweep, out_dir: Path,
                              mouse_id: str = 'mouse_0',
                              trial_indices=(0, 50, 150, 300, 400)):
    """For each chosen trial, overlay spatial + temporal + target distributions
    across all 6 losses on a single page.

    Trials grid: rows = losses, cols = chosen trial. Each panel shows
    the same trial under that loss's trained decoders — direct visual
    of how peaky the spatial output is under each training objective.
    """
    # No y sharing — peakiness varies orders of magnitude across losses
    # (MSE collapses to near-zero, PCA spikes to ~1), so a shared y axis
    # would crush whichever loss isn't at the extreme. Per-panel scale
    # lets each row read off the shape clearly.
    fig, axes = plt.subplots(
        len(LOSSES), len(trial_indices),
        figsize=(3.4 * len(trial_indices), 2.0 * len(LOSSES)),
        sharex=True, squeeze=False)
    s_grid = np.arange(91)

    for r, loss in enumerate(LOSSES):
        mat = sweep.get(loss)
        for c, t in enumerate(trial_indices):
            ax = axes[r, c]
            if mat is None or mouse_id not in mat['results']:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f'{loss}: no data',
                         transform=ax.transAxes, ha='center', va='center',
                         fontsize=9, color='grey')
                continue
            dist = mat['results'][mouse_id]['Dist']
            # OOD splits (generalize_*) have fewer test trials than stratified,
            # so a fixed example index can overflow — skip it cleanly.
            if t >= np.asarray(dist['spat']['target']).shape[0]:
                ax.set_axis_off()
                continue
            tgt = dist['spat']['target'][t]
            spat = dist['spat']['decoded'][t]
            temp = dist['temp']['decoded'][t]

            ax.plot(s_grid, tgt, 'k--', lw=1.2, label='target', alpha=0.8)
            ax.plot(s_grid, spat, color=ARCH_COLOR['spat'], lw=1.4,
                     label='spatial')
            ax.plot(s_grid, temp, color=ARCH_COLOR['temp'], lw=1.4,
                     label='temporal')
            ymax = max(tgt.max(), spat.max(), temp.max())
            ax.set_ylim(0, ymax * 1.08)
            if r == 0:
                ax.set_title(f'trial {t}', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{loss}\nprob.', fontsize=10)
            if r == len(LOSSES) - 1:
                ax.set_xlabel('orientation bin')

    # Single legend at the top — pull handles from the first panel that
    # produced output.
    handles, labels = None, None
    for r in range(len(LOSSES)):
        for c in range(len(trial_indices)):
            h, l = axes[r, c].get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc='upper center',
                    ncol=3, bbox_to_anchor=(0.5, 1.01), frameon=False)

    fig.suptitle(f'Example posteriors across losses ({mouse_id}, '
                  f'stratified split)', y=1.04, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '1_example_posteriors.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 2 — Peakiness histograms
# ----------------------------------------------------------------------

def plot_peakiness_histograms(sweep, out_dir: Path):
    """Pool trials across mice. Two panels per loss:
    (a) histogram of max(P_b), (b) histogram of H(P_b)/log(n_cats).
    Spatial and temporal overlaid; vertical line at target mean as
    reference (so 'how peaky compared to the target' is visible)."""
    # x shared per column (max prob and entropy each span [0, 1]); y
    # is per-row so PCA's broad spread doesn't get crushed by MSE's
    # collapse-spike, and vice versa.
    fig, axes = plt.subplots(
        len(LOSSES), 2, figsize=(10, 1.8 * len(LOSSES)),
        sharex='col', squeeze=False)

    for r, loss in enumerate(LOSSES):
        mat = sweep.get(loss)
        if mat is None:
            for c in range(2):
                axes[r, c].set_axis_off()
                axes[r, c].text(0.5, 0.5, f'{loss}: no data',
                                  transform=axes[r, c].transAxes,
                                  ha='center', va='center',
                                  fontsize=9, color='grey')
            continue

        max_spat, max_temp, max_target = [], [], []
        ent_spat, ent_temp, ent_target = [], [], []
        for m_data in mat['results'].values():
            dist = m_data['Dist']
            max_spat.append(_max_prob(dist['spat']['decoded']))
            max_temp.append(_max_prob(dist['temp']['decoded']))
            max_target.append(_max_prob(dist['spat']['target']))
            ent_spat.append(_normalised_entropy(dist['spat']['decoded']))
            ent_temp.append(_normalised_entropy(dist['temp']['decoded']))
            ent_target.append(_normalised_entropy(dist['spat']['target']))
        max_spat = np.concatenate(max_spat)
        max_temp = np.concatenate(max_temp)
        max_target = np.concatenate(max_target)
        ent_spat = np.concatenate(ent_spat)
        ent_temp = np.concatenate(ent_temp)
        ent_target = np.concatenate(ent_target)

        # Panel a — max prob
        ax = axes[r, 0]
        bins = np.linspace(0, 1, 41)
        ax.hist(max_spat, bins=bins, color=ARCH_COLOR['spat'],
                 alpha=0.55, label='spatial', density=True)
        ax.hist(max_temp, bins=bins, color=ARCH_COLOR['temp'],
                 alpha=0.55, label='temporal', density=True)
        ax.axvline(np.median(max_target), color='black', linestyle='--',
                    lw=1.0, alpha=0.7, label='target median')
        ax.set_ylabel(f'{loss}\ndensity', fontsize=10)
        if r == 0:
            ax.set_title('max(P) per trial')
            ax.legend(loc='upper right', fontsize=8, frameon=False)
        if r == len(LOSSES) - 1:
            ax.set_xlabel('max probability')

        # Panel b — fractional entropy
        ax = axes[r, 1]
        bins = np.linspace(0, 1, 41)
        ax.hist(ent_spat, bins=bins, color=ARCH_COLOR['spat'],
                 alpha=0.55, density=True)
        ax.hist(ent_temp, bins=bins, color=ARCH_COLOR['temp'],
                 alpha=0.55, density=True)
        ax.axvline(np.median(ent_target), color='black', linestyle='--',
                    lw=1.0, alpha=0.7)
        if r == 0:
            ax.set_title('H(P) / log N  per trial')
        if r == len(LOSSES) - 1:
            ax.set_xlabel('normalised entropy')

    fig.suptitle('Decoder peakiness across losses '
                  '(trials pooled across mice)', y=1.01, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '2_peakiness_histograms.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 3 — Per-bin temporal peakiness vs spatial
# ----------------------------------------------------------------------

def plot_per_bin_sbc_peakiness(sweep, out_dir: Path):
    """For each loss, compare:
      * spatial time-averaged output (one P per trial)
      * temporal time-averaged output (one P per trial — Jensen-smoothed)
      * temporal per-bin output (T per trial, pre-Jensen)
    If the per-bin temporal distributions are as peaky as spatial, that
    confirms the temporal final-output smoothness is purely Jensen-via-time-
    average, not loss-driven."""
    fig, axes = plt.subplots(
        len(LOSSES), 1,
        figsize=(8, 1.6 * len(LOSSES)),
        sharex=True, squeeze=False)

    bins = np.linspace(0, 1, 41)
    for r, loss in enumerate(LOSSES):
        ax = axes[r, 0]
        mat = sweep.get(loss)
        if mat is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'{loss}: no data',
                     transform=ax.transAxes, ha='center', va='center',
                     fontsize=9, color='grey')
            continue

        max_ppc, max_sbc_avg, max_sbc_bin = [], [], []
        for m_data in mat['results'].values():
            dist = m_data['Dist']
            max_ppc.append(_max_prob(dist['spat']['decoded']))
            max_sbc_avg.append(_max_prob(dist['temp']['decoded']))
            samp = np.asarray(dist['temp']['decoded_samp'])
            if samp.ndim == 3:
                # (n_trials, n_cats, T) — take max along the n_cats axis,
                # then flatten across (trials, time bins).
                max_sbc_bin.append(samp.max(axis=1).reshape(-1))
        max_ppc = np.concatenate(max_ppc)
        max_sbc_avg = np.concatenate(max_sbc_avg)
        if max_sbc_bin:
            max_sbc_bin = np.concatenate(max_sbc_bin)
        else:
            max_sbc_bin = np.array([])

        ax.hist(max_ppc, bins=bins, color=ARCH_COLOR['spat'],
                 alpha=0.6, label='spatial (1 per trial)', density=True)
        ax.hist(max_sbc_avg, bins=bins, color=ARCH_COLOR['temp'],
                 alpha=0.6, label='temporal time-avg', density=True)
        if max_sbc_bin.size:
            ax.hist(max_sbc_bin, bins=bins, color='black',
                     alpha=0.5, histtype='step', lw=1.6,
                     label='temporal per-bin', density=True)
        ax.set_ylabel(f'{loss}\ndensity', fontsize=10)
        if r == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=False)
            ax.set_title('max(P) — per-bin temporal isolates Jensen smoothing')
        if r == len(LOSSES) - 1:
            ax.set_xlabel('max probability')

    fig.tight_layout()
    out_path = out_dir / '3_per_bin_sbc_peakiness.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 4 — Cross-loss held-out PCA loss
# ----------------------------------------------------------------------

def plot_cross_loss_test_pca(sweep, out_dir: Path):
    """Bars per (loss, arch) on the held-out PCA-projected loss. Two
    panels: raw and variance-normalised (loss / per-mouse marginal-
    mean variance baseline). Per-mouse dots + paired lines connecting
    each mouse's spat→temp.

    Each loss uses its own decoder's PCA basis as the yardstick (same
    basis used during training for the PCA loss; computed but unused
    for the other losses). So the bar heights are directly comparable
    across losses.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    x = np.arange(len(LOSSES))
    width = 0.38

    # Collect per-mouse values for each (loss, arch).
    raw = {arch: {loss: [] for loss in LOSSES} for arch in ('spat', 'temp')}
    varnorm = {arch: {loss: [] for loss in LOSSES} for arch in ('spat', 'temp')}

    for loss in LOSSES:
        mat = sweep.get(loss)
        if mat is None:
            continue
        for arch in ('spat', 'temp'):
            losses_per_mouse = _per_mouse_pca_test_loss(mat, arch)
            bases_per_mouse = _per_mouse_variance_baseline(mat, arch)
            # Aligned: same mouse keys.
            for mid in losses_per_mouse:
                v = losses_per_mouse[mid]
                b = bases_per_mouse.get(mid, np.nan)
                raw[arch][loss].append(v)
                if np.isfinite(b) and b > 0:
                    varnorm[arch][loss].append(v / b)
                else:
                    varnorm[arch][loss].append(np.nan)

    def _draw(ax, data, ylabel, hline_at=None):
        for i, arch in enumerate(('spat', 'temp')):
            offset = (i - 0.5) * width
            means, sems = [], []
            for loss in LOSSES:
                vals = [v for v in data[arch][loss] if np.isfinite(v)]
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                             if len(vals) > 1 else 0.0)
            ax.bar(x + offset, means, width=width, yerr=sems, capsize=4,
                    color=ARCH_COLOR[arch], edgecolor='black',
                    alpha=0.85, label=f'{"spatial" if arch=="spat" else "temporal"}',
                    zorder=2)
        # Per-mouse dots + spat→temp connecting lines.
        for li, loss in enumerate(LOSSES):
            sv = data['spat'][loss]; tv = data['temp'][loss]
            xs = x[li] - 0.5 * width; xt = x[li] + 0.5 * width
            for s, t in zip(sv, tv):
                if np.isfinite(s) and np.isfinite(t):
                    ax.plot([xs, xt], [s, t], '-', color='grey',
                             alpha=0.4, lw=0.9, zorder=3)
                if np.isfinite(s):
                    ax.plot(xs, s, 'o', color='black', markersize=3.5,
                             alpha=0.7, zorder=4)
                if np.isfinite(t):
                    ax.plot(xt, t, 'o', color='black', markersize=3.5,
                             alpha=0.7, zorder=4)
        if hline_at is not None:
            ax.axhline(hline_at, color='red', linestyle='--', alpha=0.7,
                        lw=1.2, label='marginal-mean baseline')
        ax.set_xticks(x)
        ax.set_xticklabels(LOSSES, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
        ax.legend(loc='best', fontsize=9, frameon=False)

    _draw(axes[0], raw, 'Held-out PCA loss (×100)')
    _draw(axes[1], varnorm, 'Variance-normalised held-out PCA loss', hline_at=1.0)

    axes[0].set_title('Raw held-out PCA-projected loss\n(common yardstick across losses)')
    axes[1].set_title('Variance-normalised\n(1.0 = predict the marginal mean)')
    fig.suptitle('Cross-loss held-out performance on the PCA yardstick '
                  '(stratified_balanced)', y=1.03, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '4_cross_loss_test_pca.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 5 — Training curves on PCA yardstick (needs .pt)
# ----------------------------------------------------------------------

def plot_training_curves(histories, out_dir: Path):
    """Per-epoch ``train_pca_yardstick`` curves for each (loss, arch).
    Skipped if no histories were loaded (sandbox without torch)."""
    if not histories:
        print("  [skip] plot_training_curves — no histories loaded")
        return

    # One panel per loss; both archs overlaid; per-mouse curves drawn
    # in faint colour with the across-mouse mean overplotted.
    n_losses = len(LOSSES)
    n_cols = 3
    n_rows = int(np.ceil(n_losses / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.0 * n_cols, 3.2 * n_rows),
                              sharex=True, squeeze=False)

    for li, loss in enumerate(LOSSES):
        ax = axes[li // n_cols][li % n_cols]
        per_loss = histories.get(loss, {})
        # Even when a loss has no history we keep the axis visible so
        # the panel's x-tick / x-label still draws — otherwise shared-x
        # rows above lose their tick labels under matplotlib's
        # sharex=True. Mark "no data" with a centred annotation
        # instead of hiding the axis entirely.
        if not per_loss:
            ax.text(0.5, 0.5, f'{loss}: no history',
                     transform=ax.transAxes, ha='center', va='center',
                     fontsize=10, color='grey')
        else:
            for arch in ('spat', 'temp'):
                curves = []
                for mid, archs in per_loss.items():
                    if arch in archs and 'train_pca_yardstick' in archs[arch]:
                        y = np.asarray(archs[arch]['train_pca_yardstick'],
                                        dtype=float)
                        if y.size:
                            curves.append(y)
                            ax.plot(np.arange(len(y)) + 1, y,
                                     color=ARCH_COLOR[arch], alpha=0.25, lw=0.9)
                if curves:
                    # Stack on the shortest common length so a partial run
                    # doesn't kill the mean.
                    L = min(c.shape[0] for c in curves)
                    stack = np.stack([c[:L] for c in curves], axis=0)
                    ax.plot(np.arange(L) + 1, stack.mean(axis=0),
                             color=ARCH_COLOR[arch], lw=2.2,
                             label=f'{arch.upper()}')
            ax.set_yscale('log')
            ax.legend(loc='best', fontsize=8, frameon=False)
        ax.set_title(loss)
        ax.set_xlabel('epoch')
        if li % n_cols == 0:
            ax.set_ylabel('train PCA yardstick')

    # Hide truly off-grid axes (when the loss count doesn't fill a
    # rectangular grid). These never carry data or shared x-ticks, so
    # set_axis_off is safe here.
    for k in range(n_losses, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_axis_off()

    fig.suptitle('Training curves on PCA yardstick — common scale across losses '
                  '(faint = per-mouse, bold = across-mouse mean)',
                  y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '5_training_curves_pca_yardstick.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 6 — Weight norm evolution (needs .pt)
# ----------------------------------------------------------------------

def plot_weight_norms(histories, out_dir: Path):
    """Per-layer L2 weight norm trajectories across training, faceted
    by loss. Stacks layers in the legend. The diagnostic question:
    do entropy-bearing losses keep output-layer weights smaller?"""
    if not histories:
        print("  [skip] plot_weight_norms — no histories loaded")
        return

    n_losses = len(LOSSES)
    n_cols = 3
    n_rows = int(np.ceil(n_losses / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.0 * n_cols, 3.2 * n_rows),
                              sharex=True, squeeze=False)

    layer_palette = plt.get_cmap('viridis')

    for li, loss in enumerate(LOSSES):
        ax = axes[li // n_cols][li % n_cols]
        per_loss = histories.get(loss, {})
        has_data = False
        if per_loss:
            # Use 'spat' (spatial) — output-weight peakiness is a spatial question.
            per_layer_traj = []  # list across mice, each (n_epochs, n_layers)
            for mid, archs in per_loss.items():
                if 'spat' not in archs:
                    continue
                wn = archs['spat'].get('weight_norms', [])
                if not wn:
                    continue
                arr = np.asarray(wn, dtype=float)
                per_layer_traj.append(arr)
            if per_layer_traj:
                has_data = True
                L = min(a.shape[0] for a in per_layer_traj)
                K = per_layer_traj[0].shape[1]
                stack = np.stack([a[:L, :] for a in per_layer_traj], axis=0)
                mean = stack.mean(axis=0)
                for k in range(K):
                    ax.plot(np.arange(L) + 1, mean[:, k],
                             color=layer_palette(k / max(K - 1, 1)),
                             lw=1.6, label=f'param {k}')
                ax.legend(loc='best', fontsize=7, frameon=False, ncol=2)
        if not has_data:
            ax.text(0.5, 0.5, f'{loss}: no weight_norms',
                     transform=ax.transAxes, ha='center', va='center',
                     fontsize=10, color='grey')
        ax.set_title(f'{loss} — spatial weight norms')
        ax.set_xlabel('epoch')
        if li % n_cols == 0:
            ax.set_ylabel('||·||₂')

    for k in range(n_losses, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_axis_off()

    fig.suptitle('Per-layer L2 weight norms over training (spatial, across-mouse mean)',
                  y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '6_weight_norm_evolution.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 7 — Train vs test gap (needs .pt + .mat)
# ----------------------------------------------------------------------

def plot_train_vs_test_gap(sweep, histories, out_dir: Path):
    """Scatter of final-epoch train PCA yardstick (from history) vs
    held-out test PCA loss (from .mat), one dot per (loss, mouse, arch).
    Off-diagonal = generalisation gap; losses that overfit more sit
    further below the y=x line."""
    if not histories:
        print("  [skip] plot_train_vs_test_gap — no histories loaded")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    arch_axes = {'spat': axes[0], 'temp': axes[1]}

    for loss in LOSSES:
        mat = sweep.get(loss)
        if mat is None:
            continue
        per_loss = histories.get(loss, {})
        if not per_loss:
            continue
        for arch in ('spat', 'temp'):
            ax = arch_axes[arch]
            test_losses = _per_mouse_pca_test_loss(mat, arch)
            for mid, archs in per_loss.items():
                mkey = f'mouse_{mid}'
                if mkey not in test_losses:
                    continue
                yard = archs.get(arch, {}).get('train_pca_yardstick', [])
                if not yard:
                    continue
                final_train = float(yard[-1])
                test_val = test_losses[mkey]
                ax.scatter(final_train, test_val,
                            color=LOSS_COLOR[loss], s=40, edgecolor='black',
                            linewidth=0.5, alpha=0.85, zorder=3,
                            label=loss)

    for arch, ax in arch_axes.items():
        # y=x reference. Use the combined data range for a clean line.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim_lo = min(xlim[0], ylim[0])
        lim_hi = max(xlim[1], ylim[1])
        if np.isfinite(lim_lo) and np.isfinite(lim_hi):
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                     'k--', lw=1, alpha=0.5, label='train = test')
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel('final train PCA yardstick')
        ax.set_ylabel('held-out test PCA loss')
        ax.set_title(f'{"spatial" if arch == "spat" else "temporal"}')
        # Dedup legend.
        h, l = ax.get_legend_handles_labels()
        by = dict(zip(l, h))
        ax.legend(by.values(), by.keys(), loc='best', fontsize=8, frameon=False)
        ax.grid(linestyle=':', alpha=0.4)

    fig.suptitle('Final train PCA yardstick vs held-out test loss '
                  '(generalisation gap by loss)', y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '7_train_vs_test_gap.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


def _plateau_epoch(curve, tol_frac: float = 0.05) -> int:
    """First epoch the val curve gets within ``tol_frac`` of its
    full-window minimum, measured against the curve's range.

    The naive ``argmin`` is biased toward late epochs when the val
    curve plateaus: small noise-driven re-improvements late in
    training keep extending the "argmin" far past where the curve
    actually flattened. This detector instead asks "when did val
    first reach essentially its eventual floor?" — i.e. the elbow.

    Specifically: return the first epoch ``e`` such that
    ``curve[e] <= min(curve) + tol_frac * (curve[0] - min(curve))``.
    With tol_frac=0.02, the elbow is the first time val has come
    within 2% of its full-range descent. The exact tol_frac is a
    judgement call — tighten for noisier curves, loosen if the
    elbow drifts late.

    Epochs are 1-indexed in the returned value (matching the x-axis
    label scheme used in the plot).
    """
    import numpy as _np
    arr = _np.asarray(curve, dtype=float)
    arr = arr[_np.isfinite(arr)]
    if arr.size == 0:
        return 1
    floor = float(arr.min())
    head = float(arr[0])
    if head <= floor:                 # monotonic non-decreasing edge case
        return 1
    threshold = floor + tol_frac * (head - floor)
    hits = _np.where(arr <= threshold)[0]
    return int(hits[0] + 1) if hits.size else int(_np.argmin(arr) + 1)


# ----------------------------------------------------------------------
# Plot 8 — Train vs val curves overlaid (needs .pt + val history)
# ----------------------------------------------------------------------

def plot_train_val_curves(histories, out_dir: Path):
    """For each loss, per-arch panel with train (solid) and val
    (dashed) PCA-yardstick curves overlaid. Reads the early-stopping
    epoch off the val trough; reads the train-val gap off the
    vertical offset between solid and dashed.

    Skipped when no histories contain val curves (i.e. the run was
    launched with ``val_frac=0`` — no val data was carved).
    """
    if not histories:
        print("  [skip] plot_train_val_curves — no histories loaded")
        return

    # Probe: do any of the histories actually have val curves?
    has_val = any(
        'val_pca_yardstick' in archs.get('spat', {})
        or 'val_pca_yardstick' in archs.get('temp', {})
        for per_loss in histories.values() for archs in per_loss.values()
    )
    if not has_val:
        print("  [skip] plot_train_val_curves — no val curves in history "
              "(was the run launched with val_frac > 0?)")
        return

    n_losses = len(LOSSES)
    fig, axes = plt.subplots(
        n_losses, 2, figsize=(11, 2.3 * n_losses),
        sharex=True, squeeze=False)

    for li, loss in enumerate(LOSSES):
        per_loss = histories.get(loss, {})
        for ci, arch in enumerate(('spat', 'temp')):
            ax = axes[li, ci]
            if not per_loss:
                # Keep axis visible so shared-x ticks above still draw.
                ax.text(0.5, 0.5, f'{loss} / {arch}: no history',
                         transform=ax.transAxes, ha='center', va='center',
                         fontsize=10, color='grey')
                ax.set_yscale('log')
                ax.set_title(f'{loss} — '
                              f'{"spatial" if arch == "spat" else "temporal"}')
                if li == n_losses - 1:
                    ax.set_xlabel('epoch')
                if ci == 0:
                    ax.set_ylabel('PCA yardstick')
                continue
            train_curves, val_curves = [], []
            for mid, archs in per_loss.items():
                h = archs.get(arch, {})
                if 'train_pca_yardstick' in h and h['train_pca_yardstick']:
                    train_curves.append(np.asarray(
                        h['train_pca_yardstick'], dtype=float))
                if 'val_pca_yardstick' in h and h['val_pca_yardstick']:
                    val_curves.append(np.asarray(
                        h['val_pca_yardstick'], dtype=float))

            color = ARCH_COLOR[arch]
            for c in train_curves:
                ax.plot(np.arange(len(c)) + 1, c, color=color,
                         alpha=0.20, lw=0.8)
            for c in val_curves:
                ax.plot(np.arange(len(c)) + 1, c, color=color,
                         alpha=0.20, lw=0.8, linestyle='--')
            if train_curves:
                L = min(c.shape[0] for c in train_curves)
                stack = np.stack([c[:L] for c in train_curves], axis=0)
                ax.plot(np.arange(L) + 1, stack.mean(axis=0),
                         color=color, lw=2.2, label='train')
            if val_curves:
                L = min(c.shape[0] for c in val_curves)
                stack = np.stack([c[:L] for c in val_curves], axis=0)
                mean = stack.mean(axis=0)
                ax.plot(np.arange(L) + 1, mean, color=color, lw=2.2,
                         linestyle='--', label='val')
                # Plateau elbow (5% tolerance) — robust early-stop
                # epoch estimate. Looser threshold than the technical
                # argmin: the first epoch where val has covered 95% of
                # its full descent range. Catches the visual "the
                # curve has bent" point rather than the noise-floor
                # tail. Tighten to 0.02 / 0.01 if the elbow visibly
                # follows the marker; loosen to 0.10 to bias even
                # earlier.
                plateau = _plateau_epoch(mean, tol_frac=0.05)
                ax.axvline(plateau, color='red', linestyle=':', lw=1.0,
                            alpha=0.6,
                            label=f'val plateau (5%) @ {plateau}')

            ax.set_yscale('log')
            ax.set_title(f'{loss} — '
                          f'{"spatial" if arch == "spat" else "temporal"}')
            if li == n_losses - 1:
                ax.set_xlabel('epoch')
            if ci == 0:
                ax.set_ylabel('PCA yardstick')
            ax.legend(loc='best', fontsize=8, frameon=False)

    fig.suptitle('Train vs val PCA-yardstick curves '
                  '(solid=train, dashed=val; red dotted = val plateau, '
                  'first epoch within 5% of the eventual floor)',
                  y=1.01, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / '8_train_val_curves.svg'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Plot 9 — Posterior evolution across snapshots (needs torch + .pt)
# ----------------------------------------------------------------------

def plot_posterior_evolution(run_name: str, results_root: Path | None,
                              out_dir: Path,
                              mouse_id: int = 0,
                              trial_indices=(0, 50, 150, 300),
                              arch: str = 'spat'):
    """For one mouse and one architecture (default spatial, where the
    peakiness story lives), show how the predicted posterior evolves
    across training snapshots. Rows = saved snapshot epochs, cols =
    example trials. Each panel overlays the decoded posterior at
    that epoch against the (constant) target.

    The diagnostic question: at the val-plateau epoch (epoch ~16-20
    for spatial per Plot 8), is the posterior already peaky, or does the
    peakiness develop over the next 80 epochs of overfit?

    Requires torch to reconstruct the saved ``state_dict`` snapshots
    and forward-pass the saved test inputs. Skipped silently in
    environments without torch (e.g. the sandbox).
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("  [skip] plot_posterior_evolution — torch not installed")
        return

    if results_root is None:
        results_root = paths.RESULTS

    # Import the model class lazily — keeps this module importable
    # without torch on the path.
    from nn_classifier import SimpleFlexibleNNClassifier

    # One figure per loss; rows = snapshot epochs, cols = trial indices.
    arch_pretty = 'spatial' if arch == 'spat' else 'temporal'
    s_grid = np.arange(91)

    for loss in LOSSES:
        pt_path = (Path(results_root) / run_name / _slug(loss)
                    / 'checkpoints' / f'mouse_{mouse_id}_{SPLIT}.pt')
        if not pt_path.exists():
            print(f"  [skip] {loss}: no checkpoint at {pt_path}")
            continue

        bundle = torch.load(str(pt_path), map_location='cpu',
                             weights_only=False)
        arch_bundle = bundle.get(arch)
        if not arch_bundle:
            print(f"  [skip] {loss}: no '{arch}' in checkpoint bundle")
            continue
        history = arch_bundle.get('history', {})
        snap_epochs = history.get('snapshot_epochs', [])
        state_dicts = history.get('state_dicts', [])
        if not snap_epochs or not state_dicts:
            print(f"  [skip] {loss}: no snapshots in history")
            continue

        model_params = arch_bundle['model_params']
        # X_test is saved as (n_trials, T, n_neurons) from
        # _extract_checkpoint; the target arrives via the matching
        # split's .mat (so the trial indices line up).
        X_test = arch_bundle['X_test']            # torch.Tensor

        # Match the target distributions for the same mouse from the
        # merged .mat — they're the test-set targets that align with
        # X_test row-for-row.
        mat_path = Path(results_root) / run_name / _slug(loss) / f'{SPLIT}.mat'
        if not mat_path.exists():
            print(f"  [skip] {loss}: no .mat for target retrieval")
            continue
        mat = sio.loadmat(str(mat_path), simplify_cells=True)
        mkey = f'mouse_{mouse_id}'
        if mkey not in mat['results']:
            print(f"  [skip] {loss}/{mkey}: missing in .mat")
            continue
        targets = mat['results'][mkey]['Dist'][arch]['target']   # (n_trials, n_cats)

        n_epochs = len(snap_epochs)
        n_trials = len(trial_indices)
        fig, axes = plt.subplots(n_epochs, n_trials,
                                  figsize=(2.6 * n_trials, 1.5 * n_epochs),
                                  sharex=True, squeeze=False)

        for r, (ep, sd) in enumerate(zip(snap_epochs, state_dicts)):
            model = SimpleFlexibleNNClassifier(
                input_size=model_params['input_size'],
                hidden_sizes=model_params['hidden_sizes'],
                output_size=model_params['output_size'],
                activation=model_params.get('activation_function', 'tanh'),
            )
            model.load_state_dict(sd)
            model.eval()
            per_bin = None
            with torch.no_grad():
                if arch == 'spat':
                    integrated = torch.mean(X_test, dim=1)        # (B, n_neurons)
                    probs = F.softmax(model(integrated), dim=-1)  # (B, n_cats)
                else:
                    raw = F.softmax(model(X_test), dim=-1)        # (B, T, n_cats)
                    probs = torch.mean(raw, dim=1)                # (B, n_cats)
                    per_bin = raw                                 # keep T axis

            for c, t in enumerate(trial_indices):
                ax = axes[r, c]
                if t >= probs.shape[0]:
                    ax.set_axis_off()
                    continue
                p = probs[t].cpu().numpy()
                tgt = targets[t]
                ax.plot(s_grid, tgt, 'k--', lw=1.0, alpha=0.7,
                         label='target' if (r == 0 and c == 0) else None)
                # For temporal, expose the temporal structure: each time bin's
                # per-bin posterior as a faint line, with the time-average
                # (the actual decoded posterior) bold on top. spatial has no
                # per-bin axis — the integrated readout IS the posterior.
                ymax = max(p.max(), tgt.max())
                if per_bin is not None:
                    pb = per_bin[t].cpu().numpy()                 # (T, n_cats)
                    for bi in range(pb.shape[0]):
                        ax.plot(s_grid, pb[bi], color=ARCH_COLOR[arch],
                                 lw=0.5, alpha=0.25,
                                 label=('per-bin' if (r == 0 and c == 0
                                                      and bi == 0) else None))
                    ymax = max(ymax, float(pb.max()))
                ax.plot(s_grid, p, color=ARCH_COLOR[arch], lw=1.6,
                         label=(f'{arch_pretty} (time-avg)'
                                if (r == 0 and c == 0) else None))
                ax.set_ylim(0, ymax * 1.08)
                if r == 0:
                    ax.set_title(f'trial {t}', fontsize=9)
                if c == 0:
                    ax.set_ylabel(f'epoch {ep}', fontsize=9)
                if r == n_epochs - 1:
                    ax.set_xlabel('orientation bin')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=2,
                        bbox_to_anchor=(0.5, 1.01), frameon=False)

        fig.suptitle(f'Posterior evolution — {loss} / {arch_pretty} '
                      f'(mouse_{mouse_id}, {SPLIT})',
                      y=1.03, fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f'9_posterior_evolution_{loss}_{arch_pretty}.svg'
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def generate_all(run_name: str,
                  results_root: Path | None = None,
                  out_root: str = 'figures/loss_sweep_plots',
                  target: str = 'Q', window: str = 'half', bin_ms: int = 100,
                  split: str = 'stratified_balanced'):
    global TARGET, WINDOW, BIN_MS, SPLIT
    TARGET, WINDOW, BIN_MS, SPLIT = target, window, bin_ms, split
    if results_root is None:
        results_root = paths.RESULTS
    # Per-cell subdir so L / 50 ms / full / OOD don't overwrite the Q cell.
    out_dir = Path(out_root) / run_name / cell_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loss sweep: {run_name}  | cell: {cell_tag()}")
    print(f"  results -> {results_root}")
    print(f"  figures -> {out_dir}")
    print()

    print("Loading .mat files...")
    sweep = load_loss_sweep(run_name, results_root=results_root)
    print(f"  losses present: {[l for l, m in sweep.items() if m is not None]}")
    print()

    print("Loading .pt histories...")
    histories = load_loss_histories(run_name, results_root=results_root)
    if histories:
        print(f"  histories present: "
               f"{[(l, len(m)) for l, m in histories.items() if m]}")
    print()

    # Plot 1 — example posteriors per loss
    plot_example_posteriors(sweep, out_dir)
    # Plot 2 — peakiness histograms
    plot_peakiness_histograms(sweep, out_dir)
    # Plot 3 — per-bin temporal peakiness
    plot_per_bin_sbc_peakiness(sweep, out_dir)
    # Plot 4 — cross-loss held-out PCA loss
    plot_cross_loss_test_pca(sweep, out_dir)
    # Plot 5 — training curves on PCA yardstick (needs .pt)
    plot_training_curves(histories, out_dir)
    # Plot 6 — weight norm evolution (needs .pt)
    plot_weight_norms(histories, out_dir)
    # Plot 7 — train vs test gap (needs .pt + .mat)
    plot_train_vs_test_gap(sweep, histories, out_dir)
    # Plot 8 — train vs val curves overlaid (needs val_frac > 0 run)
    plot_train_val_curves(histories, out_dir)
    # Plot 9 — posterior evolution across saved snapshots (needs torch).
    # Both archs: spatial where the peakiness story lives, and temporal
    # where the per-bin temporal posteriors are overlaid (faint) under the
    # time-averaged decoded posterior.
    plot_posterior_evolution(run_name, results_root, out_dir, arch='spat')
    plot_posterior_evolution(run_name, results_root, out_dir, arch='temp')

    print()
    print(f"Done. {out_dir.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default='new_loss_sweep_may27')
    parser.add_argument('--results-root', default=None,
                         help='Default: paths.RESULTS')
    parser.add_argument('--out-root', default='figures/loss_sweep_plots')
    parser.add_argument('--target', default='Q')
    parser.add_argument('--window', default='half')
    parser.add_argument('--bin', type=int, default=100, dest='bin_ms')
    parser.add_argument('--split', default='stratified_balanced')
    args = parser.parse_args()
    generate_all(run_name=args.run_name,
                  results_root=args.results_root,
                  out_root=args.out_root,
                  target=args.target, window=args.window, bin_ms=args.bin_ms,
                  split=args.split)
