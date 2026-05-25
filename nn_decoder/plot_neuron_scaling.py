# -*- coding: utf-8 -*-
"""Plot neural-population scaling curves from ``neuron_scaling.py`` output.

Reads the per-mouse CSVs written by ``run_neuron_scaling.py`` (so it
picks up whatever has finished — it does not wait for the aggregate)
and produces, per mouse:

  - ``mouse_<id>_<target>_<split>_detail.png`` — a decoder x ranking
    grid. Each panel compares the random-subsampling curve against that
    ranking's top-first and bottom-first curves, with the target-shuffle
    control as the chance reference.
  - ``mouse_<id>_<target>_<split>_headline.png`` — the plain "how does
    performance scale" view: the random curve for both decoders.

Pure visualisation — no decoder training, no heavy processing (those
live in ``neuron_scaling.py``), per the repo's processing / plotting
split. Figures land in ``nn_decoder/figures/neuron_scaling/``; the raw
data they are built from persists in ``results/neuron_scaling/``.

Usage
-----
    python plot_neuron_scaling.py                       # all mice found
    python plot_neuron_scaling.py --mice 0 3            # specific mice
    python plot_neuron_scaling.py --target Q --split stratified_balanced
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — write PNGs, never open a window
import matplotlib.pyplot as plt

from paths import RESULTS, figures_dir
from neuron_scaling import aggregate_across_mice, normalise_to_full

TARGET_LABELS = {
    'Q': 'Q (perceptual posterior)', 'L': 'L (likelihood)',
    'd': 'd (decision posterior)', 'choice': 'choice',
    'stim_kernel': 'stim kernel', 'stim_cat': 'stim category',
}
# (model key, real/shuffle, human label)
DECODERS = [('spat', 'Spatial decoder (PPC)'),
            ('temp', 'Temporal decoder (SBC)')]
RANKINGS = [('orientation_tuning', 'Orientation tuning'),
            ('weight_magnitude', 'Weight magnitude'),
            ('mean_activity', 'Mean activity')]
RANK_COLOR = {'orientation_tuning': '#1f77b4',
              'weight_magnitude': '#d62728',
              'mean_activity': '#2ca02c'}

Y_LABEL = 'Held-out fit loss  (lower = better)'
X_LABEL = 'Number of decoded neurons'


# ----------------------------------------------------------------------
# Curve extraction
# ----------------------------------------------------------------------

def _full_point(df, model, value_col='fit_loss'):
    """(n_neurons, value) for the full-population fit, or None."""
    sub = df[(df['selection'] == 'full') & (df['model'] == model)]
    if sub.empty:
        return None
    return int(sub['n_neurons'].iloc[0]), float(sub[value_col].iloc[0])


def random_curve(df, model, value_col='fit_loss'):
    """Per-N mean / std / draw-points for the random-subsampling sweep,
    with the full-population point appended as the final N.

    ``value_col`` selects the y-quantity — ``'fit_loss'`` for the raw
    curve or ``'fit_loss_rel_full'`` for the full-normalised curve."""
    sub = df[(df['selection'] == 'random') & (df['model'] == model)]
    g = (sub.groupby('n_neurons')[value_col]
            .agg(['mean', 'std', 'count']).reset_index()
            .sort_values('n_neurons'))
    xs = list(g['n_neurons']); mean = list(g['mean'])
    std = [0.0 if np.isnan(s) else s for s in g['std']]
    fp = _full_point(df, model, value_col)
    if fp is not None and fp[0] not in xs:
        xs.append(fp[0]); mean.append(fp[1]); std.append(0.0)
    order = np.argsort(xs)
    return (np.array(xs)[order], np.array(mean)[order], np.array(std)[order],
            sub[['n_neurons', value_col]])


def targeted_curve(df, selection, direction, model):
    """(n_neurons, fit_loss) for a targeted curve, full point appended."""
    sub = df[(df['selection'] == selection)
             & (df['direction'] == direction)
             & (df['model'] == model)].sort_values('n_neurons')
    xs = list(sub['n_neurons']); ys = list(sub['fit_loss'])
    fp = _full_point(df, model)
    if fp is not None and fp[0] not in xs:
        xs.append(fp[0]); ys.append(fp[1])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_detail(df, mouse, target, split, out_path):
    """decoder (rows) x ranking (cols) grid: random vs top/bottom."""
    fig, axes = plt.subplots(len(DECODERS), len(RANKINGS),
                             figsize=(15, 8.5), sharex=True, sharey='row')
    axes = np.atleast_2d(axes)

    for r, (model, dec_label) in enumerate(DECODERS):
        rx, rmean, rstd, rpts = random_curve(df, model)
        sx, smean, _, _ = random_curve(df, model + '_shf')
        for c, (rank_key, rank_label) in enumerate(RANKINGS):
            ax = axes[r, c]
            colour = RANK_COLOR[rank_key]

            # Random subsampling: mean +/- spread over draws, draws as dots.
            if rx.size:
                ax.fill_between(rx, rmean - rstd, rmean + rstd,
                                color='0.6', alpha=0.25, zorder=1)
                ax.plot(rx, rmean, '-o', color='0.35', lw=1.8, ms=5,
                        label='Random (mean)', zorder=3)
                ax.scatter(rpts['n_neurons'], rpts['fit_loss'], s=14,
                           color='0.55', alpha=0.7, zorder=2,
                           label='Random (draws)')
            # Target-shuffle control — chance reference.
            if sx.size:
                ax.plot(sx, smean, ':', color='0.5', lw=1.5,
                        label='Shuffle control', zorder=2)
            # Targeted: top-first and bottom-first by this ranking.
            tx, ty = targeted_curve(df, rank_key, 'top', model)
            bx, by = targeted_curve(df, rank_key, 'bottom', model)
            if tx.size:
                ax.plot(tx, ty, '-s', color=colour, lw=2.0, ms=6,
                        label='Targeted: top-ranked first', zorder=4)
            if bx.size:
                ax.plot(bx, by, '--^', color=colour, lw=2.0, ms=6,
                        label='Targeted: bottom-ranked first', zorder=4)

            if r == 0:
                ax.set_title(rank_label, fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{dec_label}\n{Y_LABEL}', fontsize=9)
            if r == len(DECODERS) - 1:
                ax.set_xlabel(X_LABEL, fontsize=9)
            ax.grid(alpha=0.25)

    # One shared legend (entries are identical across panels).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Population scaling — Mouse {mouse}  |  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_headline(df, mouse, target, split, out_path):
    """The plain scaling curve: random subsampling, both decoders."""
    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11, 4.6),
                             sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        rx, rmean, rstd, rpts = random_curve(df, model)
        sx, smean, _, _ = random_curve(df, model + '_shf')
        if rx.size:
            ax.fill_between(rx, rmean - rstd, rmean + rstd,
                            color='#1f77b4', alpha=0.18)
            ax.plot(rx, rmean, '-o', color='#1f77b4', lw=2.2, ms=6,
                    label='Random subsampling (mean)')
            ax.scatter(rpts['n_neurons'], rpts['fit_loss'], s=18,
                       color='#1f77b4', alpha=0.55, label='Individual draws')
        if sx.size:
            ax.plot(sx, smean, ':', color='0.5', lw=1.8,
                    label='Shuffle control')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(Y_LABEL, fontsize=10)
    axes[-1].legend(fontsize=9, frameon=False)
    fig.suptitle(
        f'How decoding scales with population size — Mouse {mouse}  |  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Cross-mouse aggregate figures
# ----------------------------------------------------------------------

def _agg_curve(agg, selection, direction, model, full_only_n_mice=None,
               value_col='fit_loss'):
    """(n_neurons, value) from an aggregate frame, sorted by N.

    ``full_only_n_mice`` keeps only rows where every mouse contributed —
    i.e. genuine cross-mouse means rather than partial-coverage points.
    """
    a = agg[(agg['selection'] == selection)
            & (agg['direction'] == direction)
            & (agg['model'] == model)]
    if full_only_n_mice is not None:
        a = a[a['n_mice'] == full_only_n_mice]
    a = a.sort_values('n_neurons')
    return a['n_neurons'].to_numpy(), a[value_col].to_numpy()


def plot_aggregate_headline(df, agg, target, split, out_path):
    """Per-mouse random curves overlaid with the cross-mouse weighted mean."""
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11.5, 4.8), sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        for i, mid in enumerate(mice):
            rx, rmean, _, _ = random_curve(df[df['mouse'] == mid], model)
            if rx.size:
                ax.plot(rx, rmean, '-o', color=cmap(i % 10), lw=1.3, ms=4,
                        alpha=0.7, label=f'Mouse {mid}')
        gx, gy = _agg_curve(agg, 'random', 'na', model, full_only_n_mice=n_mice)
        if gx.size:
            ax.plot(gx, gy, '-o', color='k', lw=2.8, ms=8, zorder=5,
                    label=f'Cross-mouse mean ({basis}-weighted)')
        sx, sy = _agg_curve(agg, 'random', 'na', model + '_shf',
                            full_only_n_mice=n_mice)
        if sx.size:
            ax.plot(sx, sy, ':', color='0.5', lw=1.8, label='Shuffle (mean)')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(Y_LABEL, fontsize=10)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(
        f'Population scaling across mice (n={n_mice})  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_aggregate_detail(df, agg, target, split, out_path):
    """decoder x ranking grid of cross-mouse weighted means: random vs
    targeted top/bottom. Only genuine all-mice means are drawn."""
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'

    fig, axes = plt.subplots(len(DECODERS), len(RANKINGS),
                             figsize=(15, 8.5), sharex=True, sharey='row')
    axes = np.atleast_2d(axes)
    for r, (model, dec_label) in enumerate(DECODERS):
        for c, (rank_key, rank_label) in enumerate(RANKINGS):
            ax = axes[r, c]
            colour = RANK_COLOR[rank_key]

            rx, ry = _agg_curve(agg, 'random', 'na', model, n_mice)
            sx, sy = _agg_curve(agg, 'random', 'na', model + '_shf', n_mice)
            tx, ty = _agg_curve(agg, rank_key, 'top', model, n_mice)
            bx, by = _agg_curve(agg, rank_key, 'bottom', model, n_mice)
            if rx.size:
                ax.plot(rx, ry, '-o', color='0.35', lw=1.8, ms=5,
                        label='Random')
            if sx.size:
                ax.plot(sx, sy, ':', color='0.5', lw=1.5,
                        label='Shuffle control')
            if tx.size:
                ax.plot(tx, ty, '-s', color=colour, lw=2.0, ms=6,
                        label='Targeted: top-ranked first')
            if bx.size:
                ax.plot(bx, by, '--^', color=colour, lw=2.0, ms=6,
                        label='Targeted: bottom-ranked first')

            if r == 0:
                ax.set_title(rank_label, fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{dec_label}\n{Y_LABEL}', fontsize=9)
            if r == len(DECODERS) - 1:
                ax.set_xlabel(X_LABEL, fontsize=9)
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Cross-mouse mean ({basis}-weighted, n={n_mice} mice)  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_aggregate_normalised(df, target, split, out_path):
    """Per-mouse random curves normalised to each mouse's own full-
    population loss (= 1.0), with the cross-mouse weighted mean.

    Dividing out each mouse's full-population loss removes the
    between-mouse absolute-performance offset, so the curves show the
    *shape* of scaling — how much is given up by decoding from fewer
    neurons — on a common scale.
    """
    ndf = normalise_to_full(df)
    agg = aggregate_across_mice(ndf, value_col='fit_loss_rel_full')
    mice = sorted(df['mouse'].unique())
    n_mice = len(mice)
    basis = agg['weight_basis'].iloc[0] if not agg.empty else 'n/a'
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, len(DECODERS), figsize=(11.5, 4.8), sharey=True)
    for ax, (model, dec_label) in zip(axes, DECODERS):
        for i, mid in enumerate(mice):
            rx, rmean, _, _ = random_curve(
                ndf[ndf['mouse'] == mid], model, value_col='fit_loss_rel_full')
            if rx.size:
                ax.plot(rx, rmean, '-o', color=cmap(i % 10), lw=1.3, ms=4,
                        alpha=0.7, label=f'Mouse {mid}')
        gx, gy = _agg_curve(agg, 'random', 'na', model,
                            full_only_n_mice=n_mice,
                            value_col='fit_loss_rel_full')
        if gx.size:
            ax.plot(gx, gy, '-o', color='k', lw=2.8, ms=8, zorder=5,
                    label=f'Cross-mouse mean ({basis}-weighted)')
        ax.axhline(1.0, color='0.55', ls='--', lw=1.3, zorder=1,
                   label='Full population (= 1.0)')
        ax.set_title(dec_label, fontsize=11, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Fit loss relative to full population\n'
                       '(1.0 = full;  higher = worse)', fontsize=10)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(
        f'Scaling normalised to full-population loss (n={n_mice} mice)  —  '
        f'{TARGET_LABELS.get(target, target)}  |  {split}',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def discover_mice(results_dir, target, split, wanted=None):
    """Mouse ids that have a per-mouse CSV on disk."""
    pattern = os.path.join(results_dir, f'mouse_*_{target}_{split}.csv')
    found = {}
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        mid = int(base.split('_')[1])
        if wanted is None or mid in wanted:
            found[mid] = path
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default='Q')
    parser.add_argument('--split', default='stratified_balanced')
    parser.add_argument('--mice', type=int, nargs='+', default=None,
                        help='Mouse ids to plot (default: all found).')
    parser.add_argument('--results-dir', default=None,
                        help='Default: nn_decoder/results/neuron_scaling/.')
    parser.add_argument('--out-dir', default=None,
                        help='Where to write the PNGs. '
                             'Default: nn_decoder/figures/neuron_scaling/.')
    args = parser.parse_args(argv)

    results_dir = (args.results_dir if args.results_dir
                   else str(RESULTS / 'neuron_scaling'))
    mice = discover_mice(results_dir, args.target, args.split,
                         wanted=set(args.mice) if args.mice else None)
    if not mice:
        raise SystemExit(
            f"No per-mouse CSVs found in {results_dir} for "
            f"target={args.target} split={args.split}.")

    out_dir = args.out_dir if args.out_dir else str(figures_dir('neuron_scaling'))
    os.makedirs(out_dir, exist_ok=True)

    written = []
    frames = []
    for mid, path in mice.items():
        df = pd.read_csv(path)
        frames.append(df)
        stem = f'mouse_{mid}_{args.target}_{args.split}'
        written.append(plot_detail(
            df, mid, args.target, args.split,
            os.path.join(out_dir, f'{stem}_detail.png')))
        written.append(plot_headline(
            df, mid, args.target, args.split,
            os.path.join(out_dir, f'{stem}_headline.png')))
        print(f"[mouse {mid}] plotted from {os.path.basename(path)}")

    # --- Cross-mouse aggregate (needs >= 2 mice) ---
    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        agg = aggregate_across_mice(combined)
        agg_csv = os.path.join(results_dir,
                               f'aggregate_{args.target}_{args.split}.csv')
        try:
            agg.to_csv(agg_csv, index=False)
            print(f"[aggregate] {len(agg)} rows "
                  f"({agg['weight_basis'].iloc[0]}-weighted) -> {agg_csv}")
        except OSError as exc:
            print(f"[aggregate] could not write CSV ({exc}); figures still produced")
        stem = f'aggregate_{args.target}_{args.split}'
        written.append(plot_aggregate_headline(
            combined, agg, args.target, args.split,
            os.path.join(out_dir, f'{stem}_headline.png')))
        written.append(plot_aggregate_detail(
            combined, agg, args.target, args.split,
            os.path.join(out_dir, f'{stem}_detail.png')))
        written.append(plot_aggregate_normalised(
            combined, args.target, args.split,
            os.path.join(out_dir, f'{stem}_normalised.png')))
    else:
        print("[aggregate] only one mouse found — skipping cross-mouse figures")

    print(f"\n{len(written)} figure(s) written to {out_dir}")
    for p in written:
        print(f"  {p}")
    return written


if __name__ == '__main__':
    main()
