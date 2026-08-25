# -*- coding: utf-8 -*-
"""Plotting for the moment-error decomposition of decoded vs target loss.

Reads the decompositions produced by ``decomposition_analysis`` and renders
publication-style SVGs into ``Decomposition_Plots/``:

  - Bar charts of per-component error (mean, variance, skewness, shape
    residual) grouped by architecture, one panel per component, one figure
    per (target, split).
  - Per-mouse quadrant scatter of the architectural contrast on the first
    two moments — the headline panel that says whether the dissociation
    lives in mean or in variance.
  - Recovery 2x2 matrices: target_arch (rows) x decoder_arch (cols), with
    each cell showing the same component bars on the synthetic crossover
    targets.

All plots are raw (un-normalised by shuffle) per the analysis plan; the
shuffled controls are drawn as hatched bars next to the production bars so
the reader can see how far above the chance baseline each component sits.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as stats

import decomposition_analysis as da
from figsave import save_fig


# ----------------------------------------------------------------------
# Style / palette
# ----------------------------------------------------------------------

ARCH_COLORS = {
    'spat':      'darkorange',
    'temp':      'steelblue',
    'spat_shf':  'darkorange',
    'temp_shf':  'steelblue',
    'stim_mean': 'mediumseagreen',
}
ARCH_LABELS = {
    'spat':      'spatial',
    'temp':      'temporal',
    'spat_shf':  'spatial (shf)',
    'temp_shf':  'temporal (shf)',
    'stim_mean': 'stim-mean',
}

# Components and their axis labels per target type.
DIST_COMPONENTS = [
    ('mean_error',           r'Mean error  $(\hat\mu - \mu)^2$ [deg$^2$]'),
    ('var_error',            r'Variance error  $(\hat\sigma^2 - \sigma^2)^2$ [deg$^4$]'),
    ('skew_error',           r'Skewness error  $(\hat{\gamma}_1 - \gamma_1)^2$ [unitless]'),
    ('shape_residual_l2_sq', r'Shape residual  L$^2$ minus moment-explained [prob$^2$]'),
]
DEC_COMPONENTS = [
    ('pgo_bias',      r'P(Go) bias  $(\hat{p}_{Go} - p_{Go})^2$ [unitless]'),
    ('entropy_error', r'Decision entropy error  $(\hat{H} - H)^2$ [bit$^2$]'),
]

ARCHS_PROD = ('spat', 'temp', 'stim_mean', 'spat_shf', 'temp_shf')
ARCHS_RECOVERY = ('spat', 'temp')

PROD_TARGET_LABELS = {'Q': 'Perceptual posterior Q(theta)',
                      'L': 'Likelihood L(theta)',
                      'd': 'Decision posterior'}


def _set_style():
    sns.set_context('talk')
    sns.set_style('ticks')


def _per_mouse_means(trial_df, component, archs):
    """For a (component, arch) pair, return a list of per-mouse mean values
    aligned to the unique mice present in ``trial_df``."""
    out = {}
    for arch in archs:
        sub = trial_df[trial_df['arch'] == arch]
        per_mouse = sub.groupby('mouse')[component].mean().to_dict()
        out[arch] = per_mouse
    return out


# ----------------------------------------------------------------------
# Production bar charts
# ----------------------------------------------------------------------

def plot_production_bars(trial_df, target_type, split, out_dir):
    """One figure per (target_type, split). Panels = components."""
    _set_style()
    if target_type == 'd':
        components = DEC_COMPONENTS
    else:
        components = DIST_COMPONENTS

    n_panels = len(components)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    archs = ARCHS_PROD
    arch_x = np.arange(len(archs))

    for ax, (comp, ylabel) in zip(axes, components):
        per = _per_mouse_means(trial_df, comp, archs)
        means, sems = [], []
        all_mouse_vals = []
        for arch in archs:
            vals = np.array(list(per[arch].values()), dtype=float)
            means.append(np.nanmean(vals) if len(vals) else np.nan)
            sems.append(stats.sem(vals, nan_policy='omit') if len(vals) > 1 else np.nan)
            all_mouse_vals.append(vals)

        for i, arch in enumerate(archs):
            color = ARCH_COLORS[arch]
            hatch = '//' if arch.endswith('_shf') else None
            edge = 'black'
            ax.bar(
                arch_x[i], means[i], yerr=sems[i],
                color=color, edgecolor=edge, capsize=6,
                hatch=hatch, alpha=0.95, label=ARCH_LABELS[arch],
            )
        # Per-mouse strip
        for i, vals in enumerate(all_mouse_vals):
            jitter = (np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals)))
            ax.scatter(
                arch_x[i] + jitter, vals,
                s=22, color='black', alpha=0.55, zorder=3, linewidths=0,
            )

        ax.set_xticks(arch_x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in archs], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(comp.replace('_', ' '))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Decomposition  |  Target = {PROD_TARGET_LABELS[target_type]}  |  Split = {split}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'bars_production_{target_type}_{split}.svg')
    save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0])
    return path


# ----------------------------------------------------------------------
# Per-mouse quadrant scatter (mean error vs variance error contrast)
# ----------------------------------------------------------------------

def plot_quadrant_scatter(trial_df, target_type, split, out_dir):
    """For each mouse compute (mean_error_PPC - mean_error_SBC,
    var_error_PPC - var_error_SBC). A dot in the upper-right quadrant says
    temporal wins on both moments; upper-left says temporal wins on variance only,
    spatial on mean; etc.

    For 2-D decision targets the axes become (Delta P(Go) bias,
    Delta entropy error) — same logic, different units.
    """
    _set_style()
    if target_type == 'd':
        x_comp, y_comp = 'pgo_bias', 'entropy_error'
        x_label = r'$\Delta$ P(Go) bias  (spatial - temporal)'
        y_label = r'$\Delta$ entropy error  (spatial - temporal)'
    else:
        x_comp, y_comp = 'mean_error', 'var_error'
        x_label = r'$\Delta$ mean error  (spatial - temporal)  [deg$^2$]'
        y_label = r'$\Delta$ variance error  (spatial - temporal)  [deg$^4$]'

    per_mean_ppc = trial_df[trial_df['arch'] == 'spat'].groupby('mouse')[x_comp].mean()
    per_mean_sbc = trial_df[trial_df['arch'] == 'temp'].groupby('mouse')[x_comp].mean()
    per_var_ppc  = trial_df[trial_df['arch'] == 'spat'].groupby('mouse')[y_comp].mean()
    per_var_sbc  = trial_df[trial_df['arch'] == 'temp'].groupby('mouse')[y_comp].mean()

    common = sorted(set(per_mean_ppc.index) & set(per_mean_sbc.index))
    dx = np.array([per_mean_ppc[m] - per_mean_sbc[m] for m in common])
    dy = np.array([per_var_ppc[m]  - per_var_sbc[m]  for m in common])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.axvline(0, color='grey', linewidth=1, linestyle='--')
    ax.scatter(dx, dy, s=120, color='black', zorder=3)
    for m, x, y in zip(common, dx, dy):
        ax.annotate(f'm{m}', (x, y), xytext=(6, 6), textcoords='offset points', fontsize=12)

    # Quadrant labels — spatial > temporal means spatial has higher error -> temporal wins
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmax * 0.98, ymax * 0.98, 'Temporal wins both', ha='right', va='top',
            fontsize=11, color='steelblue', alpha=0.7)
    ax.text(xmin * 0.98, ymin * 0.98, 'Spatial wins both', ha='left', va='bottom',
            fontsize=11, color='darkorange', alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Architectural contrast per mouse  |  {PROD_TARGET_LABELS[target_type]}  |  {split}\n"
        f"positive = spatial has higher error than temporal on that component"
    )
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'quadrant_{target_type}_{split}.svg')
    save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0])
    return path


# ----------------------------------------------------------------------
# Recovery 2x2 panels
# ----------------------------------------------------------------------

def plot_recovery_bars(rec_df, target_type, split, out_dir):
    """2x2 grid: rows = target_arch (synthetic ground-truth source), cols =
    decoder_arch. Each cell shows component bars summed across mice.

    A diagonal-favourable dissociation produces low bars on the diagonal
    (where decoder matches the synthetic source) and high bars off-diagonal.
    """
    _set_style()
    if target_type == 'd':
        components = DEC_COMPONENTS
    else:
        components = DIST_COMPONENTS
    n_components = len(components)

    target_archs = ['spat', 'temp']
    decoder_archs = ['spat', 'temp']

    fig, axes = plt.subplots(
        len(target_archs), len(decoder_archs),
        figsize=(5 * len(decoder_archs), 4.5 * len(target_archs)),
        sharey='row',
    )

    comp_x = np.arange(n_components)

    for r, tarch in enumerate(target_archs):
        for c, darch in enumerate(decoder_archs):
            ax = axes[r, c]
            sub = rec_df[(rec_df['target_arch'] == tarch) & (rec_df['arch'] == darch)]

            per_mouse_means_by_comp = []
            sems_by_comp = []
            for comp, _label in components:
                per_mouse = sub.groupby('mouse')[comp].mean().to_numpy()
                per_mouse_means_by_comp.append(np.nanmean(per_mouse) if len(per_mouse) else np.nan)
                sems_by_comp.append(
                    stats.sem(per_mouse, nan_policy='omit') if len(per_mouse) > 1 else np.nan
                )

            color = ARCH_COLORS[darch]
            ax.bar(
                comp_x, per_mouse_means_by_comp,
                yerr=sems_by_comp,
                color=color, edgecolor='black', capsize=4,
                alpha=0.9 if r == c else 0.55,
            )

            ax.set_xticks(comp_x)
            ax.set_xticklabels([c[0].replace('_', '\n') for c in components],
                               rotation=0, fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)
            highlight = ' (DIAGONAL)' if r == c else ''
            ax.set_title(
                f"target = {ARCH_LABELS[tarch]}  decoder = {ARCH_LABELS[darch]}{highlight}",
                fontsize=11,
            )
            if c == 0:
                ax.set_ylabel('error (component-specific units)')

    fig.suptitle(
        f"Recovery decomposition  |  Target = {PROD_TARGET_LABELS[target_type]}  |  Split = {split}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'recovery_bars_{target_type}_{split}.svg')
    save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0])
    return path


def plot_recovery_quadrant(rec_df, target_type, split, out_dir):
    """Per-mouse architectural contrast on the diagonal-vs-off-diagonal axis.

    For each target_arch (synthetic source), compute per mouse:
      delta_mean = mean_error[arch != target] - mean_error[arch == target]
      delta_var  = var_error[arch != target]  - var_error[arch == target]
    A genuine architectural double dissociation puts every dot in the
    upper-right quadrant (off-diagonal worse than diagonal on both moments).
    """
    _set_style()
    if target_type == 'd':
        x_comp, y_comp = 'pgo_bias', 'entropy_error'
        x_label = r'$\Delta$ P(Go) bias  (off-diag - diag)'
        y_label = r'$\Delta$ entropy error  (off-diag - diag)'
    else:
        x_comp, y_comp = 'mean_error', 'var_error'
        x_label = r'$\Delta$ mean error  (off-diag - diag)  [deg$^2$]'
        y_label = r'$\Delta$ variance error  (off-diag - diag)  [deg$^4$]'

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.axvline(0, color='grey', linewidth=1, linestyle='--')

    point_colors = []
    point_xs = []
    point_ys = []
    point_labels = []

    for tarch in ('spat', 'temp'):
        diag_arch = tarch
        offdiag_arch = 'temp' if tarch == 'spat' else 'spat'
        sub = rec_df[rec_df['target_arch'] == tarch]
        diag_mouse = sub[sub['arch'] == diag_arch].groupby('mouse')[[x_comp, y_comp]].mean()
        off_mouse  = sub[sub['arch'] == offdiag_arch].groupby('mouse')[[x_comp, y_comp]].mean()
        common = sorted(set(diag_mouse.index) & set(off_mouse.index))
        for m in common:
            dx = off_mouse.loc[m, x_comp] - diag_mouse.loc[m, x_comp]
            dy = off_mouse.loc[m, y_comp] - diag_mouse.loc[m, y_comp]
            point_xs.append(dx)
            point_ys.append(dy)
            point_colors.append(ARCH_COLORS[tarch])
            point_labels.append(f'm{m}/{ARCH_LABELS[tarch]}-tgt')

    ax.scatter(point_xs, point_ys, s=110, c=point_colors, edgecolor='black', zorder=3)
    for x, y, lbl in zip(point_xs, point_ys, point_labels):
        ax.annotate(lbl, (x, y), xytext=(6, 6), textcoords='offset points',
                    fontsize=9)

    legend = [
        mpatches.Patch(color=ARCH_COLORS['spat'], label='target = spatial'),
        mpatches.Patch(color=ARCH_COLORS['temp'], label='target = temporal'),
    ]
    ax.legend(handles=legend, loc='best')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Recovery double-dissociation contrast per mouse\n"
        f"{PROD_TARGET_LABELS[target_type]}  |  {split}  (positive = off-diagonal worse)"
    )
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'recovery_quadrant_{target_type}_{split}.svg')
    save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0])
    return path


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

PRODUCTION_SPLITS = ('stratified_balanced', 'generalize_contrast', 'generalize_dispersion')
RECOVERY_SPLITS = ('stratified_balanced',)  # recovery only run on stratified_balanced per NOTES


def make_all_plots(
    splits=PRODUCTION_SPLITS,
    recovery_splits=RECOVERY_SPLITS,
    directory='.',
    output_dir='Decomposition_Plots',
):
    """Driver: load production and recovery decompositions for the given
    splits, write the bars + quadrant figures to ``output_dir``, and also
    write a per-mouse summary CSV that the LME / paired test can consume."""
    written = []

    for split in splits:
        prod_df = da.load_all_production(split, directory=directory)
        if len(prod_df) == 0:
            print(f"[plot_decomposition] no production data for split={split}; skipping.")
            continue

        for target_type in ('Q', 'L', 'd'):
            sub = prod_df[prod_df['target'] == target_type]
            if len(sub) == 0:
                continue
            written.append(plot_production_bars(sub, target_type, split, output_dir))
            # Quadrant uses only the production architectures.
            sub_no_shf = sub[~sub['arch'].str.endswith('_shf')]
            if len(sub_no_shf):
                written.append(plot_quadrant_scatter(sub_no_shf, target_type, split, output_dir))

        # Per-mouse summary CSV across all targets for this split
        per_mouse = da.aggregate_per_mouse(prod_df)
        csv_path = os.path.join(output_dir, f'production_per_mouse_{split}.csv')
        os.makedirs(output_dir, exist_ok=True)
        per_mouse.to_csv(csv_path, index=False)
        written.append(csv_path)

    for split in recovery_splits:
        rec_df = da.load_all_recovery(split, directory=directory)
        if len(rec_df) == 0:
            print(f"[plot_decomposition] no recovery data for split={split}; skipping.")
            continue

        for target_type in ('Q', 'L', 'd'):
            sub = rec_df[rec_df['target'] == target_type]
            if len(sub) == 0:
                continue
            written.append(plot_recovery_bars(sub, target_type, split, output_dir))
            written.append(plot_recovery_quadrant(sub, target_type, split, output_dir))

        per_mouse_rec = da.aggregate_per_mouse(
            rec_df,
            group_keys=('mouse', 'arch', 'target_arch', 'target', 'split'),
        )
        csv_path = os.path.join(output_dir, f'recovery_per_mouse_{split}.csv')
        per_mouse_rec.to_csv(csv_path, index=False)
        written.append(csv_path)

    print(f"[plot_decomposition] wrote {len(written)} files to {output_dir}/")
    return written


if __name__ == '__main__':
    make_all_plots()
