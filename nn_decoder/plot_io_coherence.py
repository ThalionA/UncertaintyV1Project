# -*- coding: utf-8 -*-
"""Plotting for IO–decoder coherence (closing the loop V1 -> Q-hat -> behaviour).

Reads per-trial coherence predictions from ``io_coherence`` and renders:

  - Per-metric bar charts (choice NLL, choice AUROC, vel R^2) comparing the
    IO baseline, the spatial and temporal decoders, their shuffled controls, and a
    stimulus-condition mean baseline. One panel per (target, split).
  - Per-mouse scatter: IO metric on x, decoder metric on y. A dot in the
    upper-left quadrant is a decoder that predicts behaviour better than
    the IO does — the strong claim. Diagonal reference line included.
  - Coherence vs decoding-loss scatter: ties this analysis to the
    decomposition output of plot_decomposition. For every (mouse, arch),
    x = mean total_l2_sq from the decomposition, y = choice NLL from this
    analysis. Tests whether decoders that decode Q better also predict
    behaviour better.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import io_coherence as ioc

# Decomposition module is optional; the cross-plot only runs when both
# analyses have outputs available.
try:
    import decomposition_analysis as da
    HAVE_DECOMPOSITION = True
except ImportError:
    HAVE_DECOMPOSITION = False


# ----------------------------------------------------------------------
# Style / palette
# ----------------------------------------------------------------------

ARCH_COLORS = {
    'spat':      'darkorange',
    'temp':      'steelblue',
    'spat_shf':  'darkorange',
    'temp_shf':  'steelblue',
    'stim_mean': 'mediumseagreen',
    'io':        'black',
    'naive':     'grey',
}
ARCH_LABELS = {
    'spat':      'spatial',
    'temp':      'temporal',
    'spat_shf':  'spatial (shf)',
    'temp_shf':  'temporal (shf)',
    'stim_mean': 'stim-mean',
    'io':        'IO',
    'naive':     'stim-mean (legacy)',
}
TARGET_LABELS = {'Q': 'Perceptual posterior Q(theta)',
                 'L': 'Likelihood L(theta)',
                 'd': 'Decision posterior'}
PRODUCTION_SPLITS = ('stratified_balanced', 'generalize_contrast', 'generalize_dispersion')


def _set_style():
    sns.set_context('talk')
    sns.set_style('ticks')


# ----------------------------------------------------------------------
# Stimulus-condition naive baseline (also serves piece (m))
# ----------------------------------------------------------------------

def _stim_mean_pgo_baseline(trial_df, group_cols=('mouse', 'split')):
    """Per (mouse, split) compute the per-trial-condition mean of actual
    choice and use it as the predicted P(Go) for that condition's trials.

    This is the trivial 'condition mean tells you everything' baseline.
    Returns a dict {(mouse, split, target): {'pred_pgo': array, 'actual': array}}.
    """
    # We don't have stimulus condition columns in the coherence frame; this
    # function is a placeholder so the bar plot has a sensible 'stim-mean'
    # bar even before (m) lands. For the headline figure we use the mean of
    # actual_choice across all trials as a per-mouse baseline. A proper
    # condition-keyed version belongs in (m).
    out = {}
    for keys, sub in trial_df.groupby(list(group_cols) + ['target']):
        mouse, split, target = keys
        pred = np.full(len(sub), sub['actual_choice'].mean())
        actual = sub['actual_choice'].to_numpy()
        out[(mouse, split, target)] = (pred, actual)
    return out


# ----------------------------------------------------------------------
# Per-metric bar chart
# ----------------------------------------------------------------------

def plot_metric_bars(per_mouse_df, target_type, split, out_dir):
    """One figure per (target_type, split). Three panels: choice NLL,
    choice AUROC (lapse pathway), velocity R^2. Bars: IO, spatial, temporal,
    spatial-shf, temporal-shf, stim-mean naive baseline. Stim-mean is computed
    here from the per-trial frame; the others come from per_mouse_df.
    """
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Pull per-mouse scalar columns out of per_mouse_df, restricted to this
    # (target, split). For each architecture we average across mice.
    sub = per_mouse_df[
        (per_mouse_df['target'] == target_type) & (per_mouse_df['split'] == split)
    ]

    arch_order = ['io', 'spat', 'temp', 'stim_mean', 'spat_shf', 'temp_shf']

    def _by_arch(metric_col):
        """Build the per-architecture getter dict for a given decoder
        metric column name (e.g. 'nll_lapse'). The IO baseline column
        differs ('nll_io', 'auroc_io', 'io_vel_r2') and is wired by
        the caller."""
        return {
            arch: (lambda s, a=arch, c=metric_col:
                   s[s['arch'] == a][c].to_numpy())
            for arch in ('spat', 'temp', 'stim_mean', 'spat_shf', 'temp_shf')
        }

    panel_specs = [
        (0, 'Choice NLL  (lower = better)',
            dict(io=lambda s: s['nll_io'].to_numpy(), **_by_arch('nll_lapse'))),
        (1, 'Choice AUROC  (higher = better)',
            dict(io=lambda s: s['auroc_io'].to_numpy(), **_by_arch('auroc_lapse'))),
        (2, r'Velocity $R^2$  (higher = better)',
            dict(io=lambda s: s['io_vel_r2'].to_numpy(), **_by_arch('vel_r2'))),
    ]

    for ax_idx, ylabel, getters in panel_specs:
        ax = axes[ax_idx]
        means, sems, vals_per_arch = [], [], []
        # IO baseline is identical across rows of sub for the same mouse;
        # take the unique-per-mouse values to avoid double counting.
        io_unique = sub.drop_duplicates(subset='mouse')
        for arch in arch_order:
            if arch == 'io':
                vals = getters['io'](io_unique)
            else:
                vals = getters[arch](sub)
            vals = np.asarray(vals, dtype=float)
            vals_per_arch.append(vals)
            means.append(np.nanmean(vals) if len(vals) else np.nan)
            sems.append(stats.sem(vals, nan_policy='omit') if len(vals) > 1 else np.nan)

        x = np.arange(len(arch_order))
        for i, arch in enumerate(arch_order):
            color = ARCH_COLORS[arch]
            hatch = '//' if arch.endswith('_shf') else None
            ax.bar(x[i], means[i], yerr=sems[i],
                   color=color, edgecolor='black', capsize=6,
                   hatch=hatch, alpha=0.95)
        # Per-mouse strip
        rng = np.random.default_rng(42)
        for i, vals in enumerate(vals_per_arch):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(x[i] + jitter, vals, s=22, color='black', alpha=0.55,
                       zorder=3, linewidths=0)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in arch_order], rotation=20)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"IO–decoder coherence  |  Target = {TARGET_LABELS.get(target_type, target_type)}  |  Split = {split}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'metric_bars_{target_type}_{split}.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# IO-vs-decoder per-mouse scatter
# ----------------------------------------------------------------------

def plot_io_vs_decoder_scatter(per_mouse_df, target_type, split, metric_pair, out_dir):
    """Per-mouse scatter: IO metric on x, decoder metric on y. A diagonal
    line shows parity. ``metric_pair`` is one of:
        ('nll_io', 'nll_lapse')      — choice NLL
        ('auroc_io', 'auroc_lapse')  — choice AUROC
        ('io_vel_r2', 'vel_r2')      — velocity R^2
    """
    _set_style()
    io_col, dec_col = metric_pair
    sub = per_mouse_df[
        (per_mouse_df['target'] == target_type) & (per_mouse_df['split'] == split)
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    for arch in ('spat', 'temp'):
        s = sub[sub['arch'] == arch]
        if len(s) == 0:
            continue
        x = s[io_col].to_numpy()
        y = s[dec_col].to_numpy()
        ax.scatter(x, y, s=120, color=ARCH_COLORS[arch],
                   edgecolor='black', label=ARCH_LABELS[arch], zorder=3)

    # Diagonal y = x
    pooled = []
    for c in (io_col, dec_col):
        pooled.append(sub[c].to_numpy())
    pooled = np.concatenate(pooled)
    valid = pooled[np.isfinite(pooled)]
    if len(valid):
        lo, hi = float(np.min(valid)), float(np.max(valid))
        margin = (hi - lo) * 0.05 if hi > lo else 1.0
        lo -= margin; hi += margin
        ax.plot([lo, hi], [lo, hi], color='grey', linestyle='--', linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(f'{io_col} (IO baseline)')
    ax.set_ylabel(f'{dec_col} (decoder)')
    ax.set_title(
        f"IO vs decoder per mouse  |  {TARGET_LABELS.get(target_type, target_type)}  |  {split}\n"
        f"y < x = decoder beats IO (for NLL); y > x = decoder beats IO (for AUROC / R$^2$)"
    )
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    metric_tag = io_col.replace('_io', '').replace('io_', '')
    path = os.path.join(out_dir, f'io_vs_decoder_{metric_tag}_{target_type}_{split}.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Coherence vs decoding scatter (cross-plot with (l))
# ----------------------------------------------------------------------

def plot_coherence_vs_decoding(coherence_per_mouse, decomposition_per_mouse,
                               target_type, split, out_dir):
    """One panel per (target, split). x = mean total_l2_sq from the
    decomposition output, y = choice NLL from the coherence output. One
    dot per (mouse, arch).

    A negative relationship across the cloud says 'better V1 decoding
    -> better behaviour prediction', which is what the loop-closure
    claim wants to see.
    """
    _set_style()
    coh = coherence_per_mouse[
        (coherence_per_mouse['target'] == target_type)
        & (coherence_per_mouse['split'] == split)
    ]
    dec = decomposition_per_mouse[
        (decomposition_per_mouse['target'] == target_type)
        & (decomposition_per_mouse['split'] == split)
    ]
    if len(coh) == 0 or len(dec) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 7))
    pooled_x, pooled_y = [], []
    for arch in ('spat', 'temp'):
        cs = coh[coh['arch'] == arch].set_index('mouse')['nll_lapse']
        if 'total_l2_sq' in dec.columns:
            ds = dec[dec['arch'] == arch].set_index('mouse')['total_l2_sq']
        else:
            # 2-D decision target uses 'total_mse' instead.
            ds = dec[dec['arch'] == arch].set_index('mouse').get('total_mse')
            if ds is None:
                continue
        common = sorted(set(cs.index) & set(ds.index))
        x = np.array([ds[m] for m in common])
        y = np.array([cs[m] for m in common])
        ax.scatter(x, y, s=110, color=ARCH_COLORS[arch], edgecolor='black',
                   label=ARCH_LABELS[arch], zorder=3)
        for m, xi, yi in zip(common, x, y):
            ax.annotate(f'm{m}', (xi, yi), xytext=(6, 6),
                        textcoords='offset points', fontsize=10)
        pooled_x.append(x); pooled_y.append(y)

    if pooled_x:
        all_x = np.concatenate(pooled_x); all_y = np.concatenate(pooled_y)
        valid = np.isfinite(all_x) & np.isfinite(all_y)
        if valid.sum() >= 3:
            r, p = stats.spearmanr(all_x[valid], all_y[valid])
            ax.text(0.05, 0.95,
                    f'Spearman rho = {r:.2f}, p = {p:.2g}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_xlabel('Decoding loss (mean total_l2_sq)  [from decomposition]')
    ax.set_ylabel('Choice NLL  [from IO coherence]')
    ax.set_title(
        f"Coherence vs decoding accuracy  |  {TARGET_LABELS.get(target_type, target_type)}  |  {split}\n"
        f"negative slope = better decoding -> better behaviour prediction"
    )
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'coherence_vs_decoding_{target_type}_{split}.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def make_all_plots(
    io_path,
    splits=PRODUCTION_SPLITS,
    directory='.',
    output_dir='IO_Coherence_Plots',
):
    """End-to-end driver. Loads IOResults, all production decoder files,
    computes the coherence frame, aggregates per mouse, and writes:

      - metric bar charts per (target, split)
      - IO-vs-decoder scatters per metric per (target, split)
      - coherence-vs-decoding scatters per (target, split) when the
        decomposition outputs are available
      - per-trial and per-mouse CSVs

    Returns a list of files written.
    """
    written = []

    trial_df = ioc.compute_io_coherence(io_path, splits, directory=directory)
    if len(trial_df) == 0:
        print(f"[plot_io_coherence] no trial-level coherence rows; "
              f"is IOResults at {io_path} present and any "
              f"population_results_fixed_*_<split>.mat in {directory}?")
        return written

    per_mouse = ioc.aggregate_metrics_per_mouse(trial_df)

    os.makedirs(output_dir, exist_ok=True)
    trial_csv = os.path.join(output_dir, 'coherence_per_trial.csv')
    pm_csv = os.path.join(output_dir, 'coherence_per_mouse.csv')
    trial_df.to_csv(trial_csv, index=False)
    per_mouse.to_csv(pm_csv, index=False)
    written.extend([trial_csv, pm_csv])

    # Per-(target, split) bar charts and scatters
    for split in splits:
        for target_type in ('Q', 'L', 'd'):
            sub = per_mouse[
                (per_mouse['target'] == target_type) & (per_mouse['split'] == split)
            ]
            if len(sub) == 0:
                continue
            written.append(plot_metric_bars(per_mouse, target_type, split, output_dir))
            written.append(plot_io_vs_decoder_scatter(
                per_mouse, target_type, split, ('nll_io', 'nll_lapse'), output_dir))
            written.append(plot_io_vs_decoder_scatter(
                per_mouse, target_type, split, ('auroc_io', 'auroc_lapse'), output_dir))
            if target_type in ('Q', 'L'):
                written.append(plot_io_vs_decoder_scatter(
                    per_mouse, target_type, split, ('io_vel_r2', 'vel_r2'), output_dir))

    # Cross-plot with the decomposition output if it's available
    if HAVE_DECOMPOSITION:
        for split in splits:
            try:
                dec_trial = da.load_all_production(split, directory=directory)
            except Exception as exc:
                print(f"[plot_io_coherence] decomposition load failed for "
                      f"split={split}: {exc}")
                continue
            if len(dec_trial) == 0:
                continue
            dec_per_mouse = da.aggregate_per_mouse(dec_trial)
            for target_type in ('Q', 'L', 'd'):
                path = plot_coherence_vs_decoding(
                    per_mouse, dec_per_mouse, target_type, split, output_dir,
                )
                if path:
                    written.append(path)

    print(f"[plot_io_coherence] wrote {len(written)} files to {output_dir}/")
    return written


if __name__ == '__main__':
    # Default IOResults location: ../data/IOResults.mat relative to this script
    here = os.path.dirname(os.path.abspath(__file__))
    default_io = os.path.normpath(os.path.join(here, '..', 'data', 'IOResults.mat'))
    if not os.path.exists(default_io):
        # Fallback to current working directory (caller may symlink)
        default_io = 'IOResults.mat'
    make_all_plots(default_io)
