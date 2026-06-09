# -*- coding: utf-8 -*-
"""Overfitting vs hidden-layer width — the 2026-06-03 meeting's item 1.

Máté's ask: *show how the train–test gap changes as the hidden layer shrinks*
(the explicit "overfitting comparison for fewer hidden units"). This is the
companion plotter for the Tier-B hidden-width ablation produced by

    run_loss_comparison.py --run-name hidden_ablation --hidden-sizes 4 8 16 32 64 ...

which isolates each width H under its own run dir ``<base>_h<H>`` (the slug
itself does NOT encode H — only the run name does). Within each width run the
layout is the usual flat-comparison cell:

    results/<base>_h<H>/<target>_<loss>_<window>_<bin>ms[_all]/
        checkpoints/mouse_<id>_<split>.pt   ->  ck[arch]['history']

Each ``history`` carries per-epoch ``train_total_loss`` and ``val_total_loss``
(val exists because run_loss_comparison runs at VAL_FRACTION=0.2) plus the
recorded early-stopping ``best_epoch`` (0-indexed). The generalisation gap we
plot is taken **at the as-deployed (best-val) epoch**:

    gap = val_total_loss[best_epoch] − train_total_loss[best_epoch]

Figures (PNG+SVG) under figures/loss_sweep_plots/<base>/overfit_vs_width/:
  gap_vs_width_by_loss        per-loss facets; gap vs H, spatial vs temporal
  trainval_vs_width_by_loss   per-loss facets × arch rows; train (dashed) & val
                              (solid) total loss vs H — the underfit→overfit shape

Read each panel WITHIN a loss: the total-loss scale differs across losses
(evar-weighted PCA vs CE vs KL vs Wasserstein), so a panel's height is only
comparable to itself across widths, not across losses. The shape that answers
the meeting — does the gap open up as H grows (overfitting) or stay flat /
collapse at small H (underfitting)? — is read per panel.

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
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import decoder_plotting_utils as dpu  # noqa: F401  (for set_style)

# Canonical PNG+SVG sink with the ≤1600px PNG cap (CLAUDE.md contract).
from figsave import save_fig as _save

LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
# spatial = unregularised PPC; temporal = SBC (per-bin entropy penalty). Colours
# match fig C of plot_weight_evolution_cell.py for a consistent visual language.
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
    """Per-mouse train/val/gap at best-val epoch for one (width, loss, arch).
    Returns a list of dicts; empty if the cell is absent or has no val curve."""
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
                     'gap': float(va[be] - tr[be]), 'best': be,
                     'stopped': hist.get('early_stopped_epoch'),
                     'cap': hist.get('epoch_cap')})
    return rows


def _agg(rows):
    """mean ± sem over mice of gap/train/val (sem with ddof=1; 0 when n<2)."""
    if not rows:
        return None
    out = {'n': len(rows), 'rows': rows}
    for k in ('gap', 'train', 'val'):
        v = np.array([r[k] for r in rows], dtype=float)
        out[f'{k}_mean'] = float(np.nanmean(v))
        out[f'{k}_sem'] = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return out


def collect(results_root, base_run, widths_runs, target, window, bin_ms, split):
    """stats[arch][loss][H] = _agg(...). Also prints a per-cell mouse count."""
    stats = {a: {l: {} for l in LOSSES} for a in ARCHS}
    for H, run_name in widths_runs:
        for loss in LOSSES:
            for arch in ARCHS:
                rows = load_gap(results_root, run_name, target, loss,
                                window, bin_ms, split, arch)
                if rows:
                    stats[arch][loss][H] = _agg(rows)
        nper = {l: (stats['spat'][l].get(H, {}).get('n', 0),
                    stats['temp'][l].get(H, {}).get('n', 0)) for l in LOSSES}
        print(f'  H={H:<3d} ({run_name}): ' +
              '  '.join(f'{l}={s}/{t}' for l, (s, t) in nper.items()))
    return stats


def fig_gap_vs_width_by_loss(stats, widths, out_dir, info):
    """Headline: generalisation gap (val − train, at best-val epoch) vs hidden
    width, one panel per loss, spatial vs temporal. A gap that grows with H =
    overfitting; a gap that is large already at small H with high train loss =
    the small net can't even fit. Faint = per mouse; bold = mean ± sem."""
    losses = [l for l in LOSSES if any(stats[a][l] for a in ARCHS)]
    if not losses:
        return
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.8), squeeze=False, sharex=True)
    for c, loss in enumerate(losses):
        ax = axes[0][c]
        for arch in ARCHS:
            cells = stats[arch][loss]
            Hs = [H for H in widths if H in cells]
            if not Hs:
                continue
            means = [cells[H]['gap_mean'] for H in Hs]
            sems = [cells[H]['gap_sem'] for H in Hs]
            ax.errorbar(Hs, means, yerr=sems, color=ARCH_COLOR[arch], lw=2.2,
                        marker='o', ms=5, capsize=3,
                        label=ARCH_LABEL[arch] if c == 0 else None)
            for H in Hs:                      # faint per-mouse points
                ax.plot([H] * cells[H]['n'], [r['gap'] for r in cells[H]['rows']],
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
    """Context for the gap: absolute train (dashed) and val (solid) total loss at
    the best-val epoch vs hidden width. Rows = arch, cols = loss. Small H with
    both curves high = underfitting; the curves splaying apart as H grows = the
    overfitting the gap figure summarises."""
    losses = [l for l in LOSSES if any(stats[a][l] for a in ARCHS)]
    if not losses:
        return
    n = len(losses)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.6), squeeze=False, sharex=True)
    for r, arch in enumerate(ARCHS):
        for c, loss in enumerate(losses):
            ax = axes[r][c]
            cells = stats[arch][loss]
            Hs = [H for H in widths if H in cells]
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


def _print_summary(stats, widths):
    """Console table: mean gap per (arch, loss, H)."""
    print('\nMean generalisation gap (val − train @ best-val epoch):')
    for arch in ARCHS:
        print(f'  [{ARCH_LABEL[arch]}]')
        header = '    ' + 'loss'.ljust(12) + ''.join(f'H={H}'.rjust(11) for H in widths)
        print(header)
        for loss in LOSSES:
            cells = stats[arch][loss]
            if not cells:
                continue
            row = '    ' + loss.ljust(12)
            for H in widths:
                row += (f'{cells[H]["gap_mean"]:+.4f}' if H in cells else '—').rjust(11)
            print(row)


def main(run_name, target, window, bin_ms, split, widths, results_root, out_root):
    dpu.set_style()
    info = f'{target} {window} {bin_ms}ms {split}'
    print(f'Overfit-vs-width: base run "{run_name}" | {info}')
    widths_runs = discover_widths(results_root, run_name, want=widths)
    if not widths_runs:
        raise SystemExit(
            f'No "{run_name}_h<H>" run dirs under {results_root}/. '
            'Run the hidden-width ablation and rsync the results down first '
            '(see PROJECT_LOG / PLAN Tier B).')
    found_widths = [H for H, _ in widths_runs]
    print(f'  widths on disk: {found_widths}')
    stats = collect(results_root, run_name, widths_runs, target, window, bin_ms, split)
    if not any(stats[a][l] for a in ARCHS for l in LOSSES):
        raise SystemExit('Found width dirs but no tracked train/val histories in them.')
    out_dir = Path(out_root) / run_name / 'overfit_vs_width'
    fig_gap_vs_width_by_loss(stats, found_widths, out_dir, info)
    fig_trainval_vs_width_by_loss(stats, found_widths, out_dir, info)
    _print_summary(stats, found_widths)
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
