# -*- coding: utf-8 -*-
"""Side-by-side comparison: additive vs joint-cell stimulus design for the
neural-metric partial-correlation analysis in
``population_metrics_vs_uncertainty.py``.

The existing `partial_correlation` helper uses **additive** main-effects-only
one-hot for the stimulus controls. That choice is documented in
`documents/residual_partial_correlation.md` and is fine for the neural-metric
question. But for gating tests against per-cell baselines it isn't enough —
hence the joint-cell variant in `decoder_residual_partial_corr.py`. Before
deciding whether to switch the default, this script computes both for every
neural metric × uncertainty target on the existing features.csv and writes a
comparison table + delta plot so we can see how much the numbers actually
move.

Spyder usage::

    from compare_partial_corr_designs import main
    main()   # reads nn_decoder/feature_ablation_results/features.csv

Outputs land in `nn_decoder/partial_corr_design_comparison/`:
  * `comparison_additive_vs_joint.csv`
  * `comparison_delta_plot.svg`
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from population_metrics_vs_uncertainty import (
    partial_correlation,          # additive (existing)
    pearson_spearman,
)
from decoder_residual_partial_corr import (
    _partial_correlation_joint,   # joint-cell (new)
)

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEATURES_CSV = os.path.join(
    DEFAULT_DIR, 'feature_ablation_results', 'features.csv',
)

# Targets we'll compare against. These are the headline uncertainty
# columns in features.csv (see feature_ablation_analysis.py).
UNCERTAINTY_TARGETS = [
    ('Perceptual_Variance', 'Perceptual Variance (Q)'),
    ('Likelihood_Variance', 'Likelihood Variance (L)'),
    ('Decision_Entropy',    'Decision Entropy'),
]

# Per the existing population_metrics module, the Late window is the
# default for partial-r reporting.
METRIC_BASES = [
    ('Pop_Mean_Raw',         'Pop Mean (raw)'),
    ('Temporal_Fano_Raw',    'Temporal Fano (raw)'),
    ('Pop_Deviation',        'Pop Deviation'),
    ('Spatial_Var',          'Spatial Var'),
    ('Temporal_Var',         'Temporal Var'),
    ('Traj_Length',          'Traj Length'),
    ('GV',                   'GV'),
    ('Centroid_Magnitude',   'Centroid Magnitude'),
    ('GV_Fano',              'GV / Centroid'),
    ('Participation_Ratio',  'Participation Ratio'),
    ('Top_Mode_Dominance',   'Top Mode Dominance'),
]


def _row(metric_base, metric_label, target_col, target_label, df):
    """Compute raw + both partial-r tiers under both designs for one cell."""
    metric_col = f"{metric_base}_Late"
    if metric_col not in df.columns or target_col not in df.columns:
        return None
    clean = df.dropna(subset=[metric_col, target_col])
    if len(clean) < 50:
        return None

    rp, pp, _, _ = pearson_spearman(
        clean[metric_col].values, clean[target_col].values,
    )

    # Additive (existing) tier
    add_stim = partial_correlation(
        df, metric_col, target_col,
        controls=('Orientation', 'Contrast', 'Dispersion'),
        per_mouse=True,
    )
    add_stim_beh = partial_correlation(
        df, metric_col, target_col,
        controls=('Orientation', 'Contrast', 'Dispersion'),
        extra=('Velocity', 'Lick_Rate'),
        per_mouse=True,
    )
    # Joint-cell (new) tier
    joint_stim = _partial_correlation_joint(
        df, metric_col, target_col, extra=(),
    )
    joint_stim_beh = _partial_correlation_joint(
        df, metric_col, target_col, extra=('Velocity', 'Lick_Rate'),
    )

    return {
        'metric_base':           metric_base,
        'metric_label':          metric_label,
        'target_col':            target_col,
        'target_label':          target_label,
        'raw_pearson_r':         rp,
        'raw_pearson_p':         pp,
        'additive_stim_r':       add_stim['r'],
        'additive_stim_p':       add_stim['p'],
        'additive_stim_beh_r':   add_stim_beh['r'],
        'additive_stim_beh_p':   add_stim_beh['p'],
        'joint_stim_r':          joint_stim['r'],
        'joint_stim_p':          joint_stim['p'],
        'joint_stim_beh_r':      joint_stim_beh['r'],
        'joint_stim_beh_p':      joint_stim_beh['p'],
        'delta_stim':            (joint_stim['r'] - add_stim['r']
                                  if np.isfinite(joint_stim['r'])
                                  and np.isfinite(add_stim['r'])
                                  else np.nan),
        'delta_stim_beh':        (joint_stim_beh['r'] - add_stim_beh['r']
                                  if np.isfinite(joint_stim_beh['r'])
                                  and np.isfinite(add_stim_beh['r'])
                                  else np.nan),
    }


def compare(df):
    rows = []
    for tcol, tlabel in UNCERTAINTY_TARGETS:
        for mbase, mlabel in METRIC_BASES:
            r = _row(mbase, mlabel, tcol, tlabel, df)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def plot_delta(comp, target_label, output_dir):
    """Per-target dot plot: x = metric, two markers per metric showing the
    additive vs joint partial r | stim, with a connecting line."""
    sub = comp[comp['target_label'] == target_label].copy()
    if sub.empty:
        return
    sub = sub.sort_values('additive_stim_r')
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(sub) + 4), 4.5),
                           constrained_layout=True)
    x = np.arange(len(sub))
    ax.plot([x, x],
            [sub['additive_stim_r'].values, sub['joint_stim_r'].values],
            color='#bbb', lw=1, zorder=1)
    ax.scatter(x, sub['additive_stim_r'].values, color='#1f77b4', s=55,
               label='additive (main effects only)', zorder=2)
    ax.scatter(x, sub['joint_stim_r'].values, color='#d62728', s=55,
               marker='D', label='joint cell', zorder=2)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub['metric_label'].values, rotation=35,
                       ha='right', fontsize=9)
    ax.set_ylabel('partial r | stim')
    ax.set_title(f'Partial r design comparison — {target_label}',
                 fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='best')
    safe = target_label.split()[0].lower()
    out = os.path.join(output_dir, f'comparison_delta_{safe}.svg')
    plt.savefig(out, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  wrote {out}')


def main(features_csv=DEFAULT_FEATURES_CSV, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_DIR,
                                  'partial_corr_design_comparison')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    df = pd.read_csv(features_csv)
    print(f'[compare] loaded {features_csv}: {df.shape[0]} rows, '
          f'{df["Mouse_ID"].nunique()} mice')

    comp = compare(df)
    out_csv = os.path.join(output_dir, 'comparison_additive_vs_joint.csv')
    comp.to_csv(out_csv, index=False)
    print(f'[compare] wrote {out_csv}')

    for _, tlabel in UNCERTAINTY_TARGETS:
        plot_delta(comp, tlabel, output_dir)

    # Headline summary stats: how big are the deltas?
    print('\n=== Delta summary (joint - additive), partial r | stim ===')
    for tcol, tlabel in UNCERTAINTY_TARGETS:
        sub = comp[comp['target_label'] == tlabel]
        if sub.empty:
            continue
        d = sub['delta_stim'].dropna()
        sign_flips = int(((sub['additive_stim_r'] > 0)
                          != (sub['joint_stim_r'] > 0)).sum())
        print(f"  {tlabel}: median |delta|={d.abs().median():.4f}  "
              f"max |delta|={d.abs().max():.4f}  "
              f"n_sign_flips={sign_flips}")

    return comp


if __name__ == '__main__':
    main()
