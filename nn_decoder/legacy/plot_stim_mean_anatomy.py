# -*- coding: utf-8 -*-
"""Visual anatomy of why the stim-mean baseline beats the trained V1
decoders on choice prediction.

For a chosen mouse and split, picks a handful of representative test trials
spanning the stimulus space and overlays the four predicted Q-distributions
that drive the choice NLL comparison:

    - The IO's true Q (post_s_marginal from IOResults; the per-trial
      target the decoders try to match)
    - PPC decoder Q-hat (from population_results_fixed_hyperparams)
    - SBC decoder Q-hat (same file)
    - Stim-mean Q-hat (from population_results_stim_mean)

A second figure shows the within-condition variance of Q for a few
specific (orientation, contrast, dispersion) cells: per-trial Q's drawn as
faint lines and the per-condition mean drawn bold. This is the visual
intuition for the headline result — within-condition variance of Q is
small, so the per-condition mean captures most of the structure.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_coherence as ioc  # noqa: E402

GRID = np.arange(0, 91, 1, dtype=float)
ARCH_COLORS = dict(
    io='black', spat='darkorange', temp='steelblue', stim_mean='mediumseagreen',
)
ARCH_LABELS = dict(
    io='IO Q (target)', spat='PPC', temp='SBC', stim_mean='stim-mean',
)


def _load_trial_table(mouse_id, io_path):
    """Use the io_coherence loader so the trial-mask logic stays in one
    place; returns the per-animal struct (already mask-aligned with the
    decoder side)."""
    animals = ioc.load_io_results(io_path)
    a = animals[mouse_id]
    mask = ioc.get_trials_to_keep_mask(a['data']['trial_keys']['session_name'])
    return a, mask


def _load_decoder_full(prefix, split, mouse_id, arch):
    """Pull `Dist[arch]['full_decoded']` from a population_results file."""
    path = os.path.join(HERE, f'{prefix}_{split}.mat')
    mat = sio.loadmat(path, simplify_cells=True)
    return np.asarray(mat['results'][f'mouse_{mouse_id}']['Dist'][arch]['full_decoded'])


def _pick_representative_trials(orientations, contrasts, dispersions, n=6, seed=0):
    """Pick ~n trials spanning the stimulus space: low/high contrast,
    low/high dispersion, near-boundary (45 deg) vs far-from-boundary."""
    rng = np.random.default_rng(seed)
    n_trials = len(orientations)
    chosen = []

    # Try to find at least one trial per (extreme) cell of the stim grid.
    high_c = contrasts >= np.quantile(contrasts, 0.75)
    low_c = contrasts <= np.quantile(contrasts, 0.25)
    high_d = dispersions >= np.quantile(dispersions, 0.75)
    low_d = dispersions <= np.quantile(dispersions, 0.25)
    near = np.abs(orientations - 45) <= 10
    far = np.abs(orientations - 45) >= 35

    cells = [
        ('hi-C, lo-D, far', high_c & low_d & far),
        ('hi-C, lo-D, near', high_c & low_d & near),
        ('lo-C, lo-D, far', low_c & low_d & far),
        ('lo-C, hi-D, far', low_c & high_d & far),
        ('hi-C, hi-D, far', high_c & high_d & far),
        ('lo-C, hi-D, near', low_c & high_d & near),
    ]
    for label, mask in cells:
        idx = np.where(mask)[0]
        if len(idx):
            chosen.append((label, int(rng.choice(idx))))
        if len(chosen) >= n:
            break
    return chosen


def plot_anatomy_single_trials(mouse_id=0, split='stratified_balanced',
                                io_path='../data/IOResults.mat',
                                out_dir='Stim_Mean_Anatomy_Plots'):
    """For a chosen mouse, overlay the four predicted Qs on a panel of
    representative trials. Saves one SVG per mouse."""
    a, mask = _load_trial_table(mouse_id, io_path)

    # Per-trial arrays, already masked to align with the decoder full set
    Q_io = np.asarray(a['inferred']['post_s_marginal'])[mask]
    orientations = np.asarray(a['data']['orientation'])[mask]
    contrasts = np.asarray(a['data']['contrast'])[mask]
    dispersions = np.asarray(a['data']['dispersion'])[mask]
    choices = np.asarray(a['data']['choices'])[mask].astype(float)

    Q_spat = _load_decoder_full('population_results_fixed_hyperparams', split, mouse_id, 'spat')
    Q_temp = _load_decoder_full('population_results_fixed_hyperparams', split, mouse_id, 'temp')
    Q_stim = _load_decoder_full('population_results_stim_mean_Q', split, mouse_id, 'stim_mean')

    picks = _pick_representative_trials(orientations, contrasts, dispersions, n=6)
    n = len(picks)
    if n == 0:
        print(f'[anatomy] no representative trials for mouse {mouse_id}')
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, (label, ti) in zip(axes, picks):
        ax.plot(GRID, Q_io[ti], color=ARCH_COLORS['io'], linewidth=2.2,
                label=ARCH_LABELS['io'])
        ax.plot(GRID, Q_stim[ti], color=ARCH_COLORS['stim_mean'], linewidth=1.8,
                linestyle='-', label=ARCH_LABELS['stim_mean'])
        ax.plot(GRID, Q_spat[ti], color=ARCH_COLORS['spat'], linewidth=1.4,
                alpha=0.9, label=ARCH_LABELS['spat'])
        ax.plot(GRID, Q_temp[ti], color=ARCH_COLORS['temp'], linewidth=1.4,
                alpha=0.9, label=ARCH_LABELS['temp'])
        ax.axvline(45, color='grey', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(orientations[ti], color='red', linestyle=':', linewidth=1,
                   alpha=0.7, label='true orientation')
        c = 'Go' if choices[ti] > 0.5 else 'No-Go'
        ax.set_title(
            f'trial {ti}  |  {label}\n'
            f'true theta={orientations[ti]:.0f} deg  c={contrasts[ti]:.2f}  '
            f'd={dispersions[ti]:.0f}  choice={c}',
            fontsize=10,
        )
        ax.set_xlabel('orientation theta [deg]')
        ax.set_ylabel('Q(theta)')
        ax.set_xlim(0, 90)
        ax.grid(linestyle='--', alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f'Decoded Q(theta) anatomy — mouse {mouse_id}, split = {split}',
        fontsize=14, y=1.06,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'anatomy_mouse{mouse_id}_{split}.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path


def plot_within_condition_variance(mouse_id=0, split='stratified_balanced',
                                    io_path='../data/IOResults.mat',
                                    out_dir='Stim_Mean_Anatomy_Plots'):
    """For three representative (o, c, d) cells, plot all per-trial IO Qs
    drawn faint, plus the condition-mean drawn bold. Visualises why
    stim-mean is so good — within-condition Q variance is small."""
    a, mask = _load_trial_table(mouse_id, io_path)
    Q_io = np.asarray(a['inferred']['post_s_marginal'])[mask]
    orientations = np.asarray(a['data']['orientation'])[mask]
    contrasts = np.asarray(a['data']['contrast'])[mask]
    dispersions = np.asarray(a['data']['dispersion'])[mask]
    conditions = np.column_stack([orientations, contrasts, dispersions])

    # Pick three conditions with the most trials so the within-condition
    # cloud is well-populated.
    uniq, counts = np.unique(conditions, axis=0, return_counts=True)
    top_idx = np.argsort(counts)[::-1]
    chosen_conds = []
    for i in top_idx[:6]:
        if counts[i] < 6:
            continue
        chosen_conds.append((uniq[i], int(counts[i])))
        if len(chosen_conds) >= 3:
            break

    if not chosen_conds:
        print(f'[anatomy] no cell with >=6 trials for mouse {mouse_id}')
        return None

    fig, axes = plt.subplots(1, len(chosen_conds), figsize=(6 * len(chosen_conds), 5),
                              sharey=True)
    if len(chosen_conds) == 1:
        axes = [axes]
    for ax, (cond, n_in_cond) in zip(axes, chosen_conds):
        mask_cond = np.all(conditions == cond, axis=1)
        Q_cond = Q_io[mask_cond]
        # Faint per-trial Qs
        for q in Q_cond:
            ax.plot(GRID, q, color='grey', alpha=0.25, linewidth=0.9)
        # Bold condition mean
        ax.plot(GRID, Q_cond.mean(axis=0), color=ARCH_COLORS['stim_mean'],
                linewidth=2.5, label=f'condition mean (n={n_in_cond})')
        ax.axvline(45, color='red', linestyle='--', linewidth=1, alpha=0.6,
                   label='boundary 45 deg')
        ax.axvline(cond[0], color='blue', linestyle=':', linewidth=1, alpha=0.7,
                   label=f'true theta={cond[0]:.0f}')
        ax.set_title(
            f'theta={cond[0]:.0f} deg  c={cond[1]:.2f}  d={cond[2]:.0f}\n'
            f'{n_in_cond} trials',
            fontsize=11,
        )
        ax.set_xlabel('orientation theta [deg]')
        ax.set_ylabel('Q(theta)')
        ax.set_xlim(0, 90)
        ax.grid(linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f'Within-condition Q(theta) variance — mouse {mouse_id}, split = {split}\n'
        f'grey = per-trial IO Q;  green = per-condition mean (= stim_mean prediction)',
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'within_condition_var_mouse{mouse_id}_{split}.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path


def plot_choice_nll_summary(io_path='../data/IOResults.mat',
                             splits=('stratified_balanced',
                                     'generalize_contrast',
                                     'generalize_dispersion'),
                             directory='.',
                             out_dir='Stim_Mean_Anatomy_Plots'):
    """One compact panel per split: per-mouse strip + per-arch mean.
    Highlights stim_mean vs IO vs trained decoders on choice NLL for Q."""
    df = ioc.compute_io_coherence(io_path, splits, directory=directory)
    per_mouse = ioc.aggregate_metrics_per_mouse(df)
    per_mouse = per_mouse[per_mouse['target'] == 'Q']

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 6), sharey=True)
    if len(splits) == 1:
        axes = [axes]
    arch_order = ['io', 'stim_mean', 'spat', 'temp', 'spat_shf', 'temp_shf']
    for ax, split in zip(axes, splits):
        sub = per_mouse[per_mouse['split'] == split]
        for i, arch in enumerate(arch_order):
            if arch == 'io':
                vals = sub.drop_duplicates(subset='mouse')['nll_io'].to_numpy()
            else:
                vals = sub[sub['arch'] == arch]['nll_lapse'].to_numpy()
            color = ARCH_COLORS.get(arch, 'grey')
            hatch = '//' if arch.endswith('_shf') else None
            mean = float(np.nanmean(vals))
            ax.bar(i, mean, color=color, edgecolor='black', alpha=0.9,
                   hatch=hatch)
            jitter = (np.random.default_rng(i).uniform(-0.12, 0.12, size=len(vals)))
            ax.scatter(i + jitter, vals, color='black', s=24, alpha=0.6,
                       zorder=3, linewidths=0)
        ax.set_xticks(range(len(arch_order)))
        ax.set_xticklabels(['IO', 'stim-mean', 'PPC', 'SBC', 'PPC shf', 'SBC shf'],
                           rotation=15)
        ax.set_title(split, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel('Choice NLL  (lower = better)')
    fig.suptitle(
        'Choice NLL via IO Stage 2 lapse psychometric — all 6 mice, target = Q',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'choice_nll_summary_Q.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return path


if __name__ == '__main__':
    out = []
    out.append(plot_anatomy_single_trials())
    out.append(plot_within_condition_variance())
    out.append(plot_choice_nll_summary())
    print(f'Wrote {len([p for p in out if p])} files:')
    for p in out:
        if p:
            print(f'  {p}')
