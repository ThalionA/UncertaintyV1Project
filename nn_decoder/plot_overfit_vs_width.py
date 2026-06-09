# -*- coding: utf-8 -*-
"""Overfitting & peakiness vs hidden-layer width — 2026-06-03 meeting item 1,
and the real-data confirmation of the toy capacity sweep in the PCA-peakiness note.

Two questions answered from one Tier-B hidden-width ablation
(run_loss_comparison.py --run-name hidden_ablation --hidden-sizes 4 8 16 32 64 …),
which isolates each width H under its own run dir ``<base>_h<H>`` (the slug does
NOT encode H — only the run name does):

  1. **Overfitting (Máté's literal ask):** how does the train–val gap change as the
     hidden layer shrinks? Read from the per-epoch ``train_total_loss`` /
     ``val_total_loss`` curves in ``ck[arch]['history']`` at the recorded best-val
     epoch (val exists because run_loss_comparison runs at VAL_FRACTION=0.2):
         gap = val_total_loss[best_epoch] − train_total_loss[best_epoch]

  2. **Peakiness (what the PCA-peakiness note needs):** does the decoded
     over-sharpening grow with capacity, as the toy model predicts
     ([[PCA-Peakiness-Mechanism]] fig 8a / fig 15)? Read the decoded posteriors
     from ``<split>.mat`` (``results[mouse]['Dist'][arch]['decoded']``) and take
     the across-trial mean max-probability — the note's peakiness definition.

The overfitting story (gap grows with H) and the peakiness story (max-prob grows
with H) should co-vary for PCA and stay flat for the calibrated losses — that is
the real-data version of "peakiness is overfitting into the loss-blind subspace".

Figures (PNG+SVG) under figures/loss_sweep_plots/<base>/overfit_vs_width/:
  capacity_summary_spat        the note figure: spatial peakiness (vs IO target)
                               AND train–val gap vs H, one line per loss
  peakiness_vs_width_by_loss   per-loss facets; mean max-prob vs H, spat vs temp
  gap_vs_width_by_loss         per-loss facets; train–val gap vs H, spat vs temp
  trainval_vs_width_by_loss    per-loss × arch; train (dashed) & val (solid) vs H

Read each per-loss panel WITHIN a loss for the gap/train-val figures: total-loss
scale differs across losses (evar-weighted PCA vs CE vs KL vs Wasserstein), so a
panel's height is comparable to itself across widths, not across losses. Peakiness
(max-prob) is on one common scale, so the peakiness figures ARE cross-loss
comparable and carry the IO-target line.

Usage
-----
    python plot_overfit_vs_width.py                       # base=hidden_ablation, Q half 100ms
    python plot_overfit_vs_width.py --target L --bin 50 --window full
    python plot_overfit_vs_width.py --run-name my_ablation --widths 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import decoder_plotting_utils as dpu  # noqa: F401  (for set_style)

# Canonical PNG+SVG sink with the ≤1600px PNG cap (CLAUDE.md contract).
from figsave import save_fig as _save

NCAT = 91
LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
# Per-loss colours match plot_weight_evolution_cell.py for a consistent suite.
LOSS_COLOR = {'PCA': '#e6550d', 'CE': '#008837', 'KL': '#7b3294',
              'JS': '#3690c0', 'Wasserstein': '#a6611a'}
# spatial = unregularised PPC (the loss acting alone — cleanest capacity read);
# temporal = SBC (per-bin entropy penalty). Colours match fig C of the weight script.
ARCHS = ('spat', 'temp')
ARCH_COLOR = {'spat': '#d95f02', 'temp': '#1f78b4'}
ARCH_LABEL = {'spat': 'spatial', 'temp': 'temporal'}


def _slug(target, loss, window, bin_ms):
    """Cell slug within a width run. PCA carries the all_trials-basis '_all'
    suffix (matches run_loss_comparison's PCA default + plot_weight_evolution_cell)."""
    return f'{target}_{loss}_{window}_{bin_ms}ms' + ('_all' if loss == 'PCA' else '')


def discover_widths(results_root, base_run, want=None):
    """Find ``<base_run>_h<H>`` run dirs on disk; return sorted [(H, run_name)].
    If ``want`` is given, keep only those widths (and warn about any missing)."""
    root = Path(results_root)
    found = {}
    for d in sorted(root.glob(f'{base_run}_h*')):
        if not d.is_dir():
            continue
        m = re.search(r'_h(\d+)$', d.name)   # single-layer widths only (e.g. _h16)
        if m:
            found[int(m.group(1))] = d.name
    if want:
        missing = [h for h in want if h not in found]
        if missing:
            print(f'  [warn] requested widths with no run dir (skipped): {missing}')
        return [(h, found[h]) for h in want if h in found]
    return [(h, found[h]) for h in sorted(found)]


def load_gap(results_root, run_name, target, loss, window, bin_ms, split, arch):
    """Per-mouse train/val/gap at best-val epoch for one (width, loss, arch),
    read from the .pt checkpoint histories. [] if absent or no val curve."""
    ck_dir = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / 'checkpoints'
    rows = []
    if not ck_dir.is_dir():
        return rows
    for pt in sorted(ck_dir.glob(f'mouse_*_{split}.pt')):
        ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        if not (isinstance(ck, dict) and arch in ck and isinstance(ck[arch], dict)):
            continue
        hist = ck[arch].get('history')
        if not hist:
            continue
        tr, va = hist.get('train_total_loss'), hist.get('val_total_loss')
        if not tr or not va:                 # no val curve (val_fraction==0) -> skip
            continue
        tr = np.asarray(tr, dtype=float)
        va = np.asarray(va, dtype=float)
        be = hist.get('best_epoch')           # as-deployed epoch, stored 0-indexed
        if be is None or not np.isscalar(be):
            be = int(np.nanargmin(va))
        be = int(be)
        be = min(be, len(tr) - 1, len(va) - 1)  # clamp into the logged range
        rows.append({'mouse': pt.stem, 'train': float(tr[be]), 'val': float(va[be]),
                     'gap': float(va[be] - tr[be]), 'best': be})
    return rows


def load_peak(results_root, run_name, target, loss, window, bin_ms, split, arch):
    """Per-mouse mean decoded max-probability (the note's peakiness) for one
    (width, loss, arch), read from <split>.mat. Returns (per_mouse_list,
    io_target_maxprob_or_None). Mirrors diagnostics/peakiness_distributions.py."""
    f = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / f'{split}.mat'
    if not f.is_file():
        return [], None
    res = sio.loadmat(str(f), simplify_cells=True).get('results')
    if not isinstance(res, dict):
        return [], None
    per_mouse, tgt_mp = [], []
    for mk in sorted(res):
        D = res[mk].get('Dist') if isinstance(res[mk], dict) else None
        if not (isinstance(D, dict) and arch in D):
            continue
        dec = np.asarray(D[arch]['decoded'], dtype=float)   # (n_trials, 91)
        per_mouse.append(float(np.mean(dec.max(-1))))
        if 'target' in D[arch]:                              # IO target (arch-invariant)
            tgt_mp.append(float(np.mean(np.asarray(D[arch]['target'], dtype=float).max(-1))))
    return per_mouse, (float(np.mean(tgt_mp)) if tgt_mp else None)


def _msem(values):
    """(mean, sem) over a list; sem with ddof=1, 0 when n<2."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None, None
    mean = float(np.nanmean(v))
    sem = float(np.nanstd(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return mean, sem


def collect(results_root, base_run, widths_runs, target, window, bin_ms, split):
    """stats[arch][loss][H] = {gap_*, train_*, val_*, peak_*, n*}; plus the
    pooled IO-target max-prob. Reads .pt histories (gap) and .mat posteriors (peak)."""
    stats = {a: {l: {} for l in LOSSES} for a in ARCHS}
    io_target = None
    for H, run_name in widths_runs:
        for loss in LOSSES:
            for arch in ARCHS:
                cell = {}
                rows = load_gap(results_root, run_name, target, loss, window, bin_ms, split, arch)
                if rows:
                    gm, gs = _msem([r['gap'] for r in rows])
                    tm, _ = _msem([r['train'] for r in rows])
                    vm, _ = _msem([r['val'] for r in rows])
                    cell.update(gap_mean=gm, gap_sem=gs, train_mean=tm, val_mean=vm,
                                n_gap=len(rows), gap_rows=rows)
                peaks, tgt = load_peak(results_root, run_name, target, loss, window, bin_ms, split, arch)
                if peaks:
                    pm, psm = _msem(peaks)
                    cell.update(peak_mean=pm, peak_sem=psm, n_peak=len(peaks), peaks=peaks)
                if tgt is not None and io_target is None:
                    io_target = tgt
                if cell:
                    stats[arch][loss][H] = cell
        nper = {l: (stats['spat'][l].get(H, {}).get('n_gap', 0),
                    stats['spat'][l].get(H, {}).get('n_peak', 0)) for l in LOSSES}
        print(f'  H={H:<3d} ({run_name}): ' +
              '  '.join(f'{l}={g}g/{p}p' for l, (g, p) in nper.items()))
    return stats, io_target


def _widths_with(stats, loss, key):
    """Widths (sorted) where stats has `key` for this loss, in either arch."""
    return sorted({H for a in ARCHS for H, c in stats[a][loss].items() if key in c})


def fig_capacity_summary(stats, widths, io_target, out_dir, info):
    """THE note figure (spatial = loss acting alone, no entropy confound): decoded
    peakiness (a) and train–val gap (b) vs hidden width, one line per loss. If
    peakiness is overfitting into the loss-blind subspace, both rise with H for
    PCA (and Wasserstein) and stay flat for the calibrated losses — the real-data
    version of the toy capacity sweep (PCA-Peakiness-Mechanism fig 8a / fig 15)."""
    arch = 'spat'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) peakiness vs width
    ax = axes[0]
    for loss in LOSSES:
        cells = stats[arch][loss]
        Hs = [H for H in widths if H in cells and 'peak_mean' in cells[H]]
        if not Hs:
            continue
        ax.errorbar(Hs, [cells[H]['peak_mean'] for H in Hs],
                    yerr=[cells[H]['peak_sem'] for H in Hs], color=LOSS_COLOR[loss],
                    lw=2.4, marker='o', ms=6, capsize=3, label=loss)
    if io_target is not None:
        ax.axhline(io_target, color='k', ls=':', lw=1.4, label=f'IO target ({io_target:.3f})')
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel('hidden width H')
    ax.set_ylabel('decoded peakiness  (mean max-prob)')
    ax.set_title('Peakiness vs capacity')
    ax.legend(frameon=False, fontsize=8.5, loc='best')
    # (b) train-val gap vs width
    ax = axes[1]
    for loss in LOSSES:
        cells = stats[arch][loss]
        Hs = [H for H in widths if H in cells and 'gap_mean' in cells[H]]
        if not Hs:
            continue
        ax.errorbar(Hs, [cells[H]['gap_mean'] for H in Hs],
                    yerr=[cells[H]['gap_sem'] for H in Hs], color=LOSS_COLOR[loss],
                    lw=2.4, marker='o', ms=6, capsize=3, label=loss)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel('hidden width H')
    ax.set_ylabel('train–val gap at best-val epoch')
    ax.set_title('Overfitting vs capacity')
    fig.suptitle(f'Capacity ablation — spatial decoder (loss alone)  ({info})', y=1.02,
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'capacity_summary_spat')


def fig_peakiness_vs_width_by_loss(stats, widths, io_target, out_dir, info):
    """Decoded peakiness (mean max-prob) vs hidden width, one panel per loss,
    spatial vs temporal, with the IO target. Cross-loss comparable (one scale)."""
    losses = [l for l in LOSSES if _widths_with(stats, l, 'peak_mean')]
    if not losses:
        return
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.8), squeeze=False, sharex=True, sharey=True)
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        for arch in ARCHS:
            cells = stats[arch][loss]
            Hs = [H for H in widths if H in cells and 'peak_mean' in cells[H]]
            if not Hs:
                continue
            ax.errorbar(Hs, [cells[H]['peak_mean'] for H in Hs],
                        yerr=[cells[H]['peak_sem'] for H in Hs], color=ARCH_COLOR[arch],
                        lw=2.2, marker='o', ms=5, capsize=3,
                        label=ARCH_LABEL[arch] if c == 0 else None)
            for H in Hs:                       # faint per-mouse
                ax.plot([H] * len(cells[H]['peaks']), cells[H]['peaks'], '.',
                        color=ARCH_COLOR[arch], alpha=0.22, ms=4)
        if io_target is not None:
            ax.axhline(io_target, color='k', ls=':', lw=1.3,
                       label='IO target' if c == 0 else None)
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_title(loss, fontsize=11)
        ax.set_xlabel('hidden width H')
        if c == 0:
            ax.set_ylabel('decoded peakiness\n(mean max-prob)')
            ax.legend(frameon=False, fontsize=9, loc='best')
    fig.suptitle(f'Decoded peakiness vs hidden width  ({info})  —  '
                 'rises with H ⇒ over-sharpening grows with capacity',
                 y=1.04, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'peakiness_vs_width_by_loss')


def fig_gap_vs_width_by_loss(stats, widths, out_dir, info):
    """Generalisation gap (val − train, at best-val epoch) vs hidden width, one
    panel per loss, spatial vs temporal. Gap growing with H = overfitting. Read
    each panel against itself (total-loss scale differs across losses)."""
    losses = [l for l in LOSSES if _widths_with(stats, l, 'gap_mean')]
    if not losses:
        return
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.8), squeeze=False, sharex=True)
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        for arch in ARCHS:
            cells = stats[arch][loss]
            Hs = [H for H in widths if H in cells and 'gap_mean' in cells[H]]
            if not Hs:
                continue
            ax.errorbar(Hs, [cells[H]['gap_mean'] for H in Hs],
                        yerr=[cells[H]['gap_sem'] for H in Hs], color=ARCH_COLOR[arch],
                        lw=2.2, marker='o', ms=5, capsize=3,
                        label=ARCH_LABEL[arch] if c == 0 else None)
            for H in Hs:                       # faint per-mouse
                ax.plot([H] * cells[H]['n_gap'], [r['gap'] for r in cells[H]['gap_rows']],
                        '.', color=ARCH_COLOR[arch], alpha=0.22, ms=4)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_title(loss, fontsize=11)
        ax.set_xlabel('hidden width H')
        if c == 0:
            ax.set_ylabel('gap at best-val epoch\n(val − train total loss)')
            ax.legend(frameon=False, fontsize=9, loc='best')
    fig.suptitle(f'Overfitting vs hidden width  ({info})  —  gap grows with H ⇒ '
                 'overfitting; read each loss panel against itself',
                 y=1.04, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'gap_vs_width_by_loss')


def fig_trainval_vs_width_by_loss(stats, widths, out_dir, info):
    """Absolute train (dashed) and val (solid) total loss at best-val epoch vs
    hidden width. Rows = arch, cols = loss. Small H with both curves high =
    underfitting; the curves splaying apart as H grows = the overfitting the gap
    figure summarises."""
    losses = [l for l in LOSSES if _widths_with(stats, l, 'gap_mean')]
    if not losses:
        return
    n = len(losses)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.6), squeeze=False, sharex=True)
    for r, arch in enumerate(ARCHS):
        for c, loss in enumerate(losses):
            ax = axes[r][c]
            cells = stats[arch][loss]
            Hs = [H for H in widths if H in cells and 'gap_mean' in cells[H]]
            if Hs:
                col = ARCH_COLOR[arch]
                ax.plot(Hs, [cells[H]['val_mean'] for H in Hs], color=col, lw=2.2,
                        marker='o', ms=5, label='val' if c == 0 else None)
                ax.plot(Hs, [cells[H]['train_mean'] for H in Hs], color=col, lw=1.6,
                        ls='--', marker='s', ms=4, alpha=0.85,
                        label='train' if c == 0 else None)
            ax.set_xscale('log', base=2)
            ax.set_xticks(widths)
            ax.set_xticklabels([str(w) for w in widths])
            if r == 0:
                ax.set_title(loss, fontsize=11)
            if r == 1:
                ax.set_xlabel('hidden width H')
            if c == 0:
                ax.set_ylabel(f'{ARCH_LABEL[arch]}\ntotal loss @ best-val')
                ax.legend(frameon=False, fontsize=8.5, loc='best')
    fig.suptitle(f'Train (dashed) vs val (solid) total loss at best-val epoch  ({info})',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'trainval_vs_width_by_loss')


def _print_summary(stats, widths, io_target):
    """Console tables: mean peakiness and mean gap per (arch, loss, H)."""
    for metric, key, fmt in (('Mean decoded peakiness (max-prob; IO target '
                               f'{io_target:.3f})' if io_target else
                               'Mean decoded peakiness (max-prob)', 'peak_mean', '{:.3f}'),
                             ('Mean generalisation gap (val − train @ best-val)', 'gap_mean', '{:+.4f}')):
        print(f'\n{metric}:')
        for arch in ARCHS:
            rows = [l for l in LOSSES if any(key in stats[arch][l].get(H, {}) for H in widths)]
            if not rows:
                continue
            print(f'  [{ARCH_LABEL[arch]}]')
            print('    ' + 'loss'.ljust(12) + ''.join(f'H={H}'.rjust(11) for H in widths))
            for loss in rows:
                cells = stats[arch][loss]
                line = '    ' + loss.ljust(12)
                for H in widths:
                    line += (fmt.format(cells[H][key]) if (H in cells and key in cells[H]) else '—').rjust(11)
                print(line)


def main(run_name, target, window, bin_ms, split, widths, results_root, out_root):
    dpu.set_style()
    info = f'{target} {window} {bin_ms}ms {split}'
    print(f'Overfit/peakiness-vs-width: base run "{run_name}" | {info}')
    widths_runs = discover_widths(results_root, run_name, want=widths)
    if not widths_runs:
        raise SystemExit(
            f'No "{run_name}_h<H>" run dirs under {results_root}/. '
            'Run the hidden-width ablation and rsync the results down first '
            '(see PROJECT_LOG / PLAN Tier B).')
    found_widths = [H for H, _ in widths_runs]
    print(f'  widths on disk: {found_widths}')
    stats, io_target = collect(results_root, run_name, widths_runs, target, window, bin_ms, split)
    has_peak = any('peak_mean' in stats[a][l].get(H, {}) for a in ARCHS for l in LOSSES for H in found_widths)
    has_gap = any('gap_mean' in stats[a][l].get(H, {}) for a in ARCHS for l in LOSSES for H in found_widths)
    if not (has_peak or has_gap):
        raise SystemExit('Found width dirs but no decoded posteriors (.mat) or '
                         'tracked train/val histories (.pt) in them.')
    out_dir = Path(out_root) / run_name / 'overfit_vs_width'
    if has_peak or has_gap:
        fig_capacity_summary(stats, found_widths, io_target, out_dir, info)
    if has_peak:
        fig_peakiness_vs_width_by_loss(stats, found_widths, io_target, out_dir, info)
    if has_gap:
        fig_gap_vs_width_by_loss(stats, found_widths, out_dir, info)
        fig_trainval_vs_width_by_loss(stats, found_widths, out_dir, info)
    _print_summary(stats, found_widths, io_target)
    print(f'\nDone. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='hidden_ablation',
                    help='base run name; width dirs are <run-name>_h<H>')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--widths', type=int, nargs='+', default=None,
                    help='restrict/order widths (default: every <run>_h<H> on disk)')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(a.run_name, a.target, a.window, a.bin_ms, a.split, a.widths,
         a.results_root, a.out_root)
