# -*- coding: utf-8 -*-
"""Residualised partial-correlation analysis on existing decoder outputs.

Task #2 from documents/session_2026_05_16_handoff.md. The gating test for
whether the V1 decoder carries within-condition trial-by-trial signal beyond
stimulus condition.

The PCA-weighted Euclidean loss that the production decoder is trained on
measures *across-condition* variation: its closed-form minimum is the
per-condition mean target, which is exactly what stim_mean_baseline.py
returns. So beating-or-tying stim_mean on that loss does not establish that
V1 carries trial-level information. The question this script answers,
without retraining anything, is:

    After residualising both decoded and target scalar summaries against
    one-hot stimulus condition (and optionally behaviour), is there any
    Pearson correlation left between them?

For a decoder that genuinely tracks within-condition trial-by-trial
fluctuations of the IO posterior, the partial r is non-zero. For the
structural null (stim_mean baseline), the partial r is NaN by construction:
the decoded scalar is constant within each stim cell, so its residual is
zero. PPC (`spat`) and SBC (`temp`) sit on the spectrum between those.

Scalar summaries
----------------
Per trial, for both `decoded` and `target` 91-bin distributions (Q, L) or
2-D simplex (d):

  * mu  : mean of the distribution on the [0, 90] degree grid. Reflects the
          peak / centre of mass. Not defined for the 2-D decision target.
  * var : 2nd central moment, in deg^2. The "perceptual variance" /
          "likelihood variance" target. Not defined for the 2-D decision
          target.
  * H   : entropy (nats).
  * pgo : P(Go) for the 2-D decision target.

Outputs
-------
Per (target_type, split):
  * a CSV row in `partial_corr_summary.csv` with raw / partial-stim /
    partial-stim+beh Pearson r and p-values per (arch, summary).
  * an SVG bar chart `partial_corr_{target}_{split}.svg`.
The full trial-level scalars and stim/behaviour columns are also dumped to
`trial_level_scalars.csv` for downstream slicing.

Spyder-friendly: `main()` runs the full sweep with defaults. The
`build_long_df(target, split)` and `partial_corr_table(df, summary_keys)`
helpers are usable interactively.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse the upstream Fisher-z pooled correlation infrastructure. As of
# 2026-05-16 `partial_correlation` defaults to `design='joint'` (joint-cell
# one-hot), which is the principled control for this gating test —
# stim_mean's per-cell predictions collapse to a NaN partial r under joint
# residualisation, the correct structural null. We thread `design=` through
# explicitly here as documentation; passing nothing would give the same
# behaviour as long as the upstream default does not flip back.
from population_metrics_vs_uncertainty import (
    partial_correlation as _upstream_partial_correlation,
    pearson_spearman,
)

# Default 91-bin orientation grid in degrees.
DEFAULT_GRID_91 = np.arange(0, 91, 1, dtype=float)
DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

# Production fixed-hyperparam decoder file prefixes per target.
PRODUCTION_PREFIXES = {
    'Q': 'population_results_fixed_hyperparams',
    'L': 'population_results_fixed_likelihood',
    'd': 'population_results_fixed_decision',
}

# Stimulus-only baseline file prefixes per target.
STIM_MEAN_PREFIXES = {
    'Q': 'population_results_stim_mean_Q',
    'L': 'population_results_stim_mean_L',
    'd': 'population_results_stim_mean_d',
}

ARCH_LABEL = {'spat': 'PPC', 'temp': 'SBC', 'stim_mean': 'StimMean'}

SUMMARY_LABELS = {
    'mu':  'Mean (deg)',
    'var': 'Variance (deg^2)',
    'H':   'Entropy',
    'pgo': 'P(Go)',
}

SUMMARIES_FOR_TARGET = {
    'Q': ('mu', 'var', 'H'),
    'L': ('mu', 'var', 'H'),
    'd': ('pgo', 'H'),
}


# ----------------------------------------------------------------------
# Scalar summaries
# ----------------------------------------------------------------------

def _normalise_rows(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    s = np.clip(p.sum(axis=1, keepdims=True), eps, None)
    return p / s


def dist_moments(p, grid=DEFAULT_GRID_91):
    """Mean (deg) and variance (deg^2) of each row of `p` on `grid`.

    `p` is treated as a probability distribution over `grid`; rows are
    normalised defensively before taking moments. Returns two 1-D arrays
    of length `n_rows`.
    """
    p = _normalise_rows(p)
    grid = np.asarray(grid, dtype=float)
    mu = (p * grid).sum(axis=1)
    var = (p * (grid - mu[:, None]) ** 2).sum(axis=1)
    return mu, var


def dist_entropy(p, eps=1e-12):
    """Row-wise Shannon entropy in nats. `p` is row-normalised first."""
    p = _normalise_rows(p, eps)
    return -np.sum(p * np.log(p + eps), axis=1)


def summaries_for_array(arr, target_type, grid=DEFAULT_GRID_91):
    """Compute scalar summaries from a (n_trials, n_categories) array.

    For target_type in {'Q','L'} returns dict {'mu', 'var', 'H'}.
    For target_type == 'd' returns {'pgo', 'H'} (mu/var undefined on the
    2-bin decision simplex).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array; got shape {arr.shape}")
    if target_type in ('Q', 'L'):
        mu, var = dist_moments(arr, grid)
        H = dist_entropy(arr)
        return {'mu': mu, 'var': var, 'H': H}
    if target_type == 'd':
        normed = _normalise_rows(arr)
        pgo = normed[:, 0]  # column 0 = P(Go), per project convention
        H = dist_entropy(normed)
        return {'pgo': pgo, 'H': H}
    raise ValueError(f"Unknown target_type {target_type!r}")


# ----------------------------------------------------------------------
# Trial-table re-extraction (so we can attach Velocity, Lick_Rate, Choice
# to the existing decoded/target arrays without re-running training).
# ----------------------------------------------------------------------

_trials_cache = {}
_export_failure_warned = {'_': False}


def _compute_test_idx(trials, split, random_state=42):
    """Replicate the production split's test_idx. Same code path as
    `stim_mean_baseline._compute_stim_mean_from_loaded`. Pulls in
    utils_v26 (torch) lazily — callers should only invoke this when the
    raw export is genuinely needed (i.e. for behaviour covariates)."""
    from utils_v26 import (
        get_stratified_train_test_indices,
        get_generalization_split_indices,
    )
    conditions = np.column_stack([
        np.asarray(trials['orientation'], dtype=float),
        np.asarray(trials['contrast'], dtype=float),
        np.asarray(trials['dispersion'], dtype=float),
    ])
    if split == 'stratified_balanced':
        _, trial_cats = np.unique(conditions, axis=0, return_inverse=True)
        _, test_idx = get_stratified_train_test_indices(
            trial_cats, test_size=0.5, random_state=random_state,
        )
    elif split in ('generalize_contrast', 'generalize_dispersion'):
        _, test_idx = get_generalization_split_indices(
            trials, split_type=split, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown split {split!r}")
    return test_idx


def _try_load_behaviour(mouse_id, split):
    """Best-effort load of (velocity, lick_rate, choice) for the test trials
    of this (mouse_id, split). Returns dict or None. Failure is silent on
    repeat calls so a missing-torch / missing-export environment doesn't
    spam the log.
    """
    try:
        from utils_v26 import load_vr_export
        _, _, _, _, trials = load_vr_export(mouse_id)
        test_idx = _compute_test_idx(trials, split)
        return {
            'velocity':  np.asarray(trials['velocity'], dtype=float)[test_idx],
            'lick_rate': np.asarray(trials['lick_rate'], dtype=float)[test_idx],
            'choice':    np.asarray(trials['choice'], dtype=float)[test_idx],
        }
    except Exception as exc:
        if not _export_failure_warned['_']:
            print(f"[partial_corr] raw export unavailable "
                  f"({type(exc).__name__}: {exc}). Stim+behaviour tier will "
                  f"be skipped; stim-only and raw tiers still computed.")
            _export_failure_warned['_'] = True
        return None


def _get_test_trials(mouse_id, split, mat_trials=None):
    """Return a dict of per-test-trial fields aligned with the decoded /
    target arrays. Cached per (mouse_id, split).

    Stim columns (Orientation/Contrast/Dispersion) come from `mat_trials`
    (the test-set-filtered trials dict embedded in the .mat) when provided,
    avoiding a torch-dependent reload of the raw export. Behaviour columns
    (Velocity/Lick_Rate/Choice) require the raw export and are filled
    with NaN if the export can't be loaded; in that case the
    stim+behaviour partial correlation tier degrades gracefully.
    """
    key = (mouse_id, split)
    if key in _trials_cache:
        return _trials_cache[key]
    if mat_trials is not None:
        ori = np.asarray(mat_trials['orientation'])
        con = np.asarray(mat_trials['contrast'])
        dsp = np.asarray(mat_trials['dispersion'])
    else:
        from utils_v26 import load_vr_export
        _, _, _, _, trials_full = load_vr_export(mouse_id)
        test_idx = _compute_test_idx(trials_full, split)
        ori = np.asarray(trials_full['orientation'])[test_idx]
        con = np.asarray(trials_full['contrast'])[test_idx]
        dsp = np.asarray(trials_full['dispersion'])[test_idx]

    beh = _try_load_behaviour(mouse_id, split)
    n = len(ori)
    out = {
        'orientation': ori,
        'contrast':    con,
        'dispersion':  dsp,
        'velocity':    beh['velocity']  if beh is not None else np.full(n, np.nan),
        'lick_rate':   beh['lick_rate'] if beh is not None else np.full(n, np.nan),
        'choice':      beh['choice']    if beh is not None else np.full(n, np.nan),
    }
    _trials_cache[key] = out
    return out


# ----------------------------------------------------------------------
# .mat loading
# ----------------------------------------------------------------------

def _load_decoder_outputs(target_type, split, directory):
    """Load both production decoder outputs (spat, temp) and the stim_mean
    baseline for a (target_type, split). Shuffled controls are not loaded.

    Returns dict mapping (mouse_id, arch) -> {'decoded', 'target',
    'mat_trials'} arrays. `mat_trials` is the per-mouse test-set-filtered
    trials dict embedded in the .mat (orientation/contrast/dispersion);
    we pass it through so the stim-only partial-correlation tier works
    without re-reading the 500 MB raw export.
    """
    outputs = {}

    prod_path = os.path.join(directory, f"{PRODUCTION_PREFIXES[target_type]}_{split}.mat")
    if os.path.exists(prod_path):
        mat = sio.loadmat(prod_path, simplify_cells=True)
        results = mat.get('results', {})
        for mkey, mdata in results.items():
            try:
                mid = int(str(mkey).split('_')[-1])
            except (ValueError, AttributeError):
                continue
            if 'Dist' not in mdata:
                continue
            dist = mdata['Dist']
            mat_trials = mdata.get('trials')  # may be None
            for arch in ('spat', 'temp'):
                if arch not in dist:
                    continue
                outputs[(mid, arch)] = {
                    'decoded':    np.asarray(dist[arch]['decoded']),
                    'target':     np.asarray(dist[arch]['target']),
                    'mat_trials': mat_trials,
                }
    else:
        print(f"[partial_corr] missing production file: {prod_path}")

    sm_path = os.path.join(directory, f"{STIM_MEAN_PREFIXES[target_type]}_{split}.mat")
    if os.path.exists(sm_path):
        mat = sio.loadmat(sm_path, simplify_cells=True)
        results = mat.get('results', {})
        for mkey, mdata in results.items():
            try:
                mid = int(str(mkey).split('_')[-1])
            except (ValueError, AttributeError):
                continue
            if 'Dist' not in mdata:
                continue
            dist = mdata['Dist']
            mat_trials = mdata.get('trials')
            if 'stim_mean' in dist:
                outputs[(mid, 'stim_mean')] = {
                    'decoded':    np.asarray(dist['stim_mean']['decoded']),
                    'target':     np.asarray(dist['stim_mean']['target']),
                    'mat_trials': mat_trials,
                }
    else:
        print(f"[partial_corr] missing stim_mean file: {sm_path}")

    return outputs


# ----------------------------------------------------------------------
# Long DataFrame builder
# ----------------------------------------------------------------------

def build_long_df(target_type, split, directory=DEFAULT_DIR):
    """Build a long DataFrame with one row per (mouse, arch, test_trial),
    carrying decoded/target scalar summaries + stimulus + behaviour.
    """
    decoder_outputs = _load_decoder_outputs(target_type, split, directory)
    if not decoder_outputs:
        return pd.DataFrame()
    rows = []
    for (mid, arch), payload in decoder_outputs.items():
        decoded = payload['decoded']
        target = payload['target']
        if decoded.size == 0 or target.size == 0:
            continue
        trials = _get_test_trials(mid, split, mat_trials=payload.get('mat_trials'))
        n_dec = decoded.shape[0]
        n_trl = len(trials['orientation'])
        if n_dec != n_trl:
            print(f"[partial_corr] WARN mouse {mid} arch {arch} "
                  f"target {target_type} split {split}: "
                  f"decoded rows={n_dec} vs test_trials={n_trl}; "
                  f"truncating to min.")
        n = min(n_dec, n_trl)
        s_dec = summaries_for_array(decoded[:n], target_type)
        s_tgt = summaries_for_array(target[:n], target_type)
        for i in range(n):
            row = {
                'Mouse_ID':    mid,
                'arch':        arch,
                'target_type': target_type,
                'split':       split,
                'Orientation': float(trials['orientation'][i]),
                'Contrast':    float(trials['contrast'][i]),
                'Dispersion':  float(trials['dispersion'][i]),
                'Velocity':    float(trials['velocity'][i]),
                'Lick_Rate':   float(trials['lick_rate'][i]),
                'Choice':      float(trials['choice'][i]),
            }
            for k in s_dec:
                row[f'decoded_{k}'] = float(s_dec[k][i])
                row[f'target_{k}']  = float(s_tgt[k][i])
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Joint-cell partial correlation (delegated to upstream)
# ----------------------------------------------------------------------

def _partial_correlation_joint(df, x_col, y_col, extra=()):
    """Per-mouse partial r(x, y | joint stim cell [+ extra]), Fisher-z
    pooled. Thin wrapper around `partial_correlation(design='joint')` in
    `population_metrics_vs_uncertainty.py`, pinned explicitly so this
    module's gating semantics stay intact even if the upstream default
    is ever changed again. The upstream helper already drops mice whose
    residual series have zero standard deviation, which is what gives
    stim_mean its NaN structural null."""
    return _upstream_partial_correlation(
        df, x_col, y_col,
        controls=('Orientation', 'Contrast', 'Dispersion'),
        extra=tuple(extra),
        per_mouse=True,
        design='joint',
    )


# ----------------------------------------------------------------------
# Partial correlation table
# ----------------------------------------------------------------------

def partial_corr_table(df, summary_keys):
    """For each (arch, summary) in `df`, compute three statistics:
      * raw Pearson r between decoded_<k> and target_<k>;
      * partial r | stimulus (per-mouse residualisation, Fisher-z pooled);
      * partial r | stimulus + behaviour (extra controls Velocity, Lick_Rate).
    """
    rows = []
    for arch in sorted(df['arch'].unique()):
        sub = df[df['arch'] == arch].copy()
        for k in summary_keys:
            dec = f'decoded_{k}'
            tgt = f'target_{k}'
            if dec not in sub.columns or tgt not in sub.columns:
                continue
            clean = sub.dropna(subset=[dec, tgt])
            if len(clean) < 5:
                continue
            x_vals = clean[dec].values
            y_vals = clean[tgt].values
            if np.std(x_vals) < 1e-12 or np.std(y_vals) < 1e-12:
                # Either input is constant -> raw Pearson undefined.
                rp, pp = np.nan, np.nan
            else:
                rp, pp, _rs, _ps = pearson_spearman(x_vals, y_vals)
            pc_s = _partial_correlation_joint(sub, dec, tgt, extra=())
            pc_sb = _partial_correlation_joint(
                sub, dec, tgt, extra=('Velocity', 'Lick_Rate'),
            )
            rows.append({
                'arch':                arch,
                'arch_label':          ARCH_LABEL.get(arch, arch),
                'summary':             k,
                'summary_label':       SUMMARY_LABELS.get(k, k),
                'raw_pearson_r':       rp,
                'raw_pearson_p':       pp,
                'partial_stim_r':      pc_s['r'],
                'partial_stim_p':      pc_s['p'],
                'partial_stim_n':      pc_s['n'],
                'partial_stim_beh_r':  pc_sb['r'],
                'partial_stim_beh_p':  pc_sb['p'],
                'partial_stim_beh_n':  pc_sb['n'],
                'n_mice':              int(df.loc[df['arch'] == arch,
                                                  'Mouse_ID'].nunique()),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

PALETTE_KINDS = {
    'Raw Pearson':        '#888888',
    'Partial | stim':     '#1f77b4',
    'Partial | stim+beh': '#d62728',
}

# Consistent per-mouse colour mapping across all forest plots. 6 mice in
# this dataset; if more get added the colormap will cycle.
_MOUSE_PALETTE_NAME = 'tab10'


def _mouse_color(mid, n_mice_hint=6):
    cmap = plt.get_cmap(_MOUSE_PALETTE_NAME)
    return cmap(int(mid) % cmap.N)


# ----------------------------------------------------------------------
# Per-mouse partial r table (forest plot input)
# ----------------------------------------------------------------------

def _fisher_z_ci(r, n, n_covariates, alpha=0.05):
    """Two-sided (1-alpha) CI for a partial Pearson r via Fisher-z
    approximation. SE in z-space is 1 / sqrt(n - n_covariates - 3) when
    `n_covariates` linear controls have been partialled out. Returns
    (lo, hi) in r-space."""
    n = int(n)
    df_eff = max(n - int(n_covariates) - 3, 1)
    se_z = 1.0 / np.sqrt(df_eff)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    from scipy import stats as _sst
    crit = _sst.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - crit * se_z, z + crit * se_z
    return float(np.tanh(z_lo)), float(np.tanh(z_hi))


def per_mouse_partial_r_table(df, summary_keys,
                              arches=('spat', 'temp', 'stim_mean')):
    """One row per (arch, summary, mouse) with per-mouse partial r |
    joint stim cell, the trial count, and a Fisher-z 95% CI.

    Stim_mean rows are kept (with NaN r) so the forest plot can mark the
    structural null explicitly rather than silently dropping it.
    """
    rows = []
    for arch in arches:
        sub_arch = df[df['arch'] == arch]
        if sub_arch.empty:
            continue
        # How many one-hot columns does the joint design produce for THIS
        # arch's subframe? Count unique (Ori, Con, Disp) combos per mouse,
        # take max as a conservative estimate of n_covariates absorbed.
        # In practice every mouse sees the same cell set on stratified.
        cells_per_mouse = (sub_arch.groupby('Mouse_ID')
                           [['Orientation', 'Contrast', 'Dispersion']]
                           .apply(lambda x: x.drop_duplicates().shape[0]))
        for summ in summary_keys:
            dec = f'decoded_{summ}'
            tgt = f'target_{summ}'
            if dec not in sub_arch.columns or tgt not in sub_arch.columns:
                continue
            pc = _upstream_partial_correlation(
                sub_arch, dec, tgt,
                controls=('Orientation', 'Contrast', 'Dispersion'),
                extra=(), per_mouse=True, design='joint',
            )
            per_mouse = dict(pc.get('per_mouse', []))
            for mid, mouse_sub in sub_arch.groupby('Mouse_ID'):
                n_trials = len(mouse_sub.dropna(subset=[dec, tgt]))
                # Joint design absorbs (#unique cells - 1) dummies +
                # intercept. Use the per-mouse cell count.
                n_cov = max(int(cells_per_mouse.get(mid, 1)) - 1, 1)
                r_i = per_mouse.get(mid, np.nan)
                if np.isfinite(r_i):
                    lo, hi = _fisher_z_ci(r_i, n_trials, n_cov)
                else:
                    lo, hi = np.nan, np.nan
                rows.append({
                    'arch':          arch,
                    'arch_label':    ARCH_LABEL.get(arch, arch),
                    'summary':       summ,
                    'summary_label': SUMMARY_LABELS.get(summ, summ),
                    'Mouse_ID':      int(mid),
                    'r':             float(r_i) if np.isfinite(r_i) else np.nan,
                    'r_ci_lo':       lo,
                    'r_ci_hi':       hi,
                    'n_trials':      int(n_trials),
                    'n_covariates':  int(n_cov),
                })
    return pd.DataFrame(rows)


def plot_per_mouse_forest(per_mouse_df, pool_df, target_type, split,
                          summary_keys, output_dir):
    """Forest plot. One panel per scalar summary. Within a panel: rows are
    arches (PPC top, SBC middle, StimMean bottom), each row has six
    per-mouse dots (colour = Mouse_ID, x = partial r, horizontal whisker =
    Fisher-z 95% CI) plus a black diamond at the Fisher-z pooled estimate.
    """
    if per_mouse_df.empty:
        return
    summaries = [k for k in summary_keys
                 if k in per_mouse_df['summary'].values]
    if not summaries:
        return
    arch_order = [a for a in ('spat', 'temp', 'stim_mean')
                  if a in per_mouse_df['arch'].values]
    if not arch_order:
        return

    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.4),
                             constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, summ in zip(axes, summaries):
        ax.axvline(0, color='black', lw=0.6, zorder=1)
        for row_i, arch in enumerate(arch_order):
            sub = per_mouse_df[(per_mouse_df['arch'] == arch)
                               & (per_mouse_df['summary'] == summ)]
            # Offset per-mouse dots vertically inside the arch row so
            # overlapping CIs don't pile up
            for j, (_, r) in enumerate(sub.iterrows()):
                y = row_i + (j - (len(sub) - 1) / 2) * 0.08
                color = _mouse_color(r['Mouse_ID'])
                if np.isfinite(r['r']):
                    ax.plot([r['r_ci_lo'], r['r_ci_hi']], [y, y],
                            color=color, lw=1.2, alpha=0.75, zorder=2)
                    ax.plot(r['r'], y, 'o', color=color, ms=6,
                            mec='black', mew=0.5, zorder=3,
                            label=f"Mouse {int(r['Mouse_ID'])}"
                                  if (summ == summaries[0]
                                      and row_i == 0)
                                  else None)
                else:
                    ax.plot(0, y, 'x', color=color, ms=6, alpha=0.5,
                            zorder=2)
            # Pooled marker
            pool_row = pool_df[(pool_df['arch'] == arch)
                               & (pool_df['summary'] == summ)]
            if not pool_row.empty:
                pooled_r = float(pool_row['partial_stim_r'].iloc[0])
                if np.isfinite(pooled_r):
                    ax.plot(pooled_r, row_i, 'D', color='black', ms=9,
                            mec='white', mew=0.8, zorder=4)

        ax.set_yticks(range(len(arch_order)))
        ax.set_yticklabels([ARCH_LABEL.get(a, a) for a in arch_order],
                           fontsize=10, fontweight='bold')
        ax.set_xlabel('partial r | joint stim cell')
        ax.set_title(SUMMARY_LABELS.get(summ, summ),
                     fontsize=11, fontweight='bold')
        ax.set_ylim(-0.6, len(arch_order) - 0.4)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.2, lw=0.5)

    # Build a single mouse-id legend for the figure
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    if handles:
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(handles), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        f"Per-mouse partial r | joint stim cell — {target_type} — {split}\n"
        "(dots = per-mouse r ± Fisher-z 95% CI; black diamond = pooled)",
        fontsize=11, fontweight='bold')
    out = os.path.join(output_dir,
                       f"forest_{target_type}_{split}.svg")
    plt.savefig(out, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[partial_corr] wrote {out}")


def plot_mouse_consistency(all_per_mouse_df, output_dir):
    """One panel per Mouse_ID showing that mouse's partial r across every
    (arch, target, summary, split) cell. Reveals whether one mouse is
    consistently off (e.g. Mouse 2 has been flagged before)."""
    if all_per_mouse_df.empty:
        return
    mice = sorted(all_per_mouse_df['Mouse_ID'].unique())
    n = len(mice)
    if n == 0:
        return
    # Restrict to trained archs; stim_mean is null by construction.
    trained = all_per_mouse_df[all_per_mouse_df['arch'].isin(['spat', 'temp'])]
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4),
                             constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, mid in zip(axes, mice):
        sub = trained[trained['Mouse_ID'] == mid].dropna(subset=['r'])
        if sub.empty:
            ax.set_visible(False)
            continue
        # Group bars: x = (target_type, summary), bar height = r, two
        # bars per group (PPC, SBC).
        sub = sub.copy()
        sub['key'] = (sub['target_type'].astype(str) + '|'
                      + sub['summary'].astype(str))
        sub['split_short'] = sub['split'].map({
            'stratified_balanced':    'strat',
            'generalize_contrast':    'gen-c',
            'generalize_dispersion':  'gen-d',
        }).fillna(sub['split'])
        # Aggregate across splits for visual simplicity — show mean ± std
        # of r across the three splits per (arch, target, summary).
        agg = (sub.groupby(['arch_label', 'target_type',
                            'summary_label'])
                  ['r']
                  .agg(['mean', 'std'])
                  .reset_index())
        agg['x_label'] = (agg['target_type'] + ' '
                          + agg['summary_label'].str.split().str[0])
        order = (agg.sort_values('mean', ascending=False)
                 .drop_duplicates('x_label')['x_label'].tolist())
        sns.barplot(data=agg, x='x_label', y='mean', hue='arch_label',
                    order=order, ax=ax,
                    palette={'PPC': '#1f77b4', 'SBC': '#d62728'})
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f"Mouse {mid}", fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('partial r (mean across splits)'
                      if mid == mice[0] else '')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        if mid != mice[-1]:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:
            ax.legend(frameon=False, fontsize=7, loc='best')

    fig.suptitle("Per-mouse partial r across all (target, summary) — "
                 "averaged across 3 splits",
                 fontsize=12, fontweight='bold')
    out = os.path.join(output_dir, 'per_mouse_consistency.svg')
    plt.savefig(out, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[partial_corr] wrote {out}")


def plot_partial_bars(table, target_type, split, summary_keys, output_dir):
    """Grouped bar chart: x-axis = arch, hue = (raw | partial-stim |
    partial-stim+beh), one subplot per scalar summary."""
    if table.empty:
        return
    arch_order = [a for a in ('spat', 'temp', 'stim_mean')
                  if a in table['arch'].values]
    summaries = [k for k in summary_keys if k in table['summary'].values]
    if not arch_order or not summaries:
        return

    # melt for seaborn
    melt_rows = []
    for _, r in table.iterrows():
        for kind, col in [('Raw Pearson',       'raw_pearson_r'),
                          ('Partial | stim',     'partial_stim_r'),
                          ('Partial | stim+beh', 'partial_stim_beh_r')]:
            melt_rows.append({
                'arch':         r['arch'],
                'arch_label':   r['arch_label'],
                'summary':      r['summary'],
                'kind':         kind,
                'r':            r[col],
            })
    long_df = pd.DataFrame(melt_rows)

    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2),
                             constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, summ in zip(axes, summaries):
        sub = long_df[long_df['summary'] == summ]
        sns.barplot(
            data=sub, x='arch', y='r', hue='kind',
            order=arch_order,
            hue_order=['Raw Pearson', 'Partial | stim', 'Partial | stim+beh'],
            palette=PALETTE_KINDS, ax=ax,
        )
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xlabel('')
        ax.set_xticklabels([ARCH_LABEL.get(a, a) for a in arch_order])
        ax.set_title(SUMMARY_LABELS.get(summ, summ),
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Pearson r' if summ == summaries[0] else '')
        # Show legend only on the rightmost panel to save space.
        if summ != summaries[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title='', frameon=False, fontsize=8, loc='best')
    fig.suptitle(
        f"Decoded vs target scalars | partial r | {target_type} | {split}\n"
        f"(per-mouse residualisation, Fisher-z pooled across n="
        f"{int(table['n_mice'].max())} mice)",
        fontsize=11, fontweight='bold')
    out = os.path.join(output_dir, f"partial_corr_{target_type}_{split}.svg")
    plt.savefig(out, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[partial_corr] wrote {out}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(target_types=('Q', 'L', 'd'),
         splits=('stratified_balanced',
                 'generalize_contrast',
                 'generalize_dispersion'),
         directory=DEFAULT_DIR,
         output_dir=None):
    """Sweep over (target_type, split), build long DataFrames, compute
    partial-correlation tables and plot bar charts. Writes a summary CSV,
    a trial-level CSV, and one SVG per (target, split) into `output_dir`.

    Returns the concatenated summary DataFrame.
    """
    if output_dir is None:
        output_dir = os.path.join(directory, 'decoder_residual_partial_corr_results')
    os.makedirs(output_dir, exist_ok=True)

    all_tables = []
    all_long = []
    all_per_mouse = []
    for target_type in target_types:
        summary_keys = SUMMARIES_FOR_TARGET[target_type]
        for split in splits:
            print(f"\n[partial_corr] === target={target_type}  split={split} ===")
            df = build_long_df(target_type, split, directory)
            if df.empty:
                print('  no data assembled; skipping.')
                continue
            print(f"  assembled {len(df)} rows "
                  f"({df['Mouse_ID'].nunique()} mice, "
                  f"{df['arch'].nunique()} archs)")
            tbl = partial_corr_table(df, summary_keys)
            if tbl.empty:
                print('  no partial correlations computable; skipping plot.')
                continue
            tbl['target_type'] = target_type
            tbl['split']       = split
            all_tables.append(tbl)

            df = df.copy()
            df['target_type'] = target_type
            df['split']       = split
            all_long.append(df)

            plot_partial_bars(tbl, target_type, split, summary_keys, output_dir)

            # Per-mouse breakdown + forest plot
            per_mouse_tbl = per_mouse_partial_r_table(df, summary_keys)
            if not per_mouse_tbl.empty:
                per_mouse_tbl['target_type'] = target_type
                per_mouse_tbl['split']       = split
                all_per_mouse.append(per_mouse_tbl)
                plot_per_mouse_forest(
                    per_mouse_tbl, tbl, target_type, split,
                    summary_keys, output_dir,
                )

            display_cols = ['arch_label', 'summary_label',
                            'raw_pearson_r',
                            'partial_stim_r', 'partial_stim_p',
                            'partial_stim_beh_r', 'partial_stim_beh_p',
                            'partial_stim_n']
            print(tbl[display_cols].round(4).to_string(index=False))

    if all_tables:
        summary_df = pd.concat(all_tables, ignore_index=True)
        out_csv = os.path.join(output_dir, 'partial_corr_summary.csv')
        summary_df.to_csv(out_csv, index=False)
        print(f"\n[partial_corr] summary CSV -> {out_csv}")
    else:
        summary_df = pd.DataFrame()
        print('\n[partial_corr] no summary tables produced.')

    if all_long:
        long_df = pd.concat(all_long, ignore_index=True)
        out_long = os.path.join(output_dir, 'trial_level_scalars.csv')
        long_df.to_csv(out_long, index=False)
        print(f"[partial_corr] trial-level CSV -> {out_long}")

    if all_per_mouse:
        per_mouse_df = pd.concat(all_per_mouse, ignore_index=True)
        out_pm = os.path.join(output_dir, 'per_mouse_partial_corr.csv')
        per_mouse_df.to_csv(out_pm, index=False)
        print(f"[partial_corr] per-mouse CSV -> {out_pm}")

        plot_mouse_consistency(per_mouse_df, output_dir)

        # Surface Mouse 2 anomaly explicitly: compute, per (arch, summary,
        # target_type), the gap between mouse 2's mean partial r (across
        # splits) and the mean across the other mice.
        trained = per_mouse_df[per_mouse_df['arch'].isin(['spat', 'temp'])]
        if not trained.empty:
            print('\n=== Mouse outlier check (mean partial r across splits) ===')
            for (arch_lbl, summ_lbl, tt), g in trained.dropna(subset=['r']).groupby(
                    ['arch_label', 'summary_label', 'target_type']):
                mouse_means = g.groupby('Mouse_ID')['r'].mean()
                if len(mouse_means) < 2:
                    continue
                for mid in mouse_means.index:
                    others = mouse_means.drop(mid)
                    gap = mouse_means[mid] - others.mean()
                    # Only print suspicious outliers (>2 SD from others'
                    # mean OR |gap| > 0.10).
                    if (abs(gap) > max(0.10, 2 * (others.std() or 1e-9))
                            and others.std() > 0):
                        print(f"  Mouse {int(mid)}  {tt}  {arch_lbl}  "
                              f"{summ_lbl}: r={mouse_means[mid]:+.3f}  "
                              f"others mean={others.mean():+.3f}  "
                              f"gap={gap:+.3f}")

    return summary_df


if __name__ == '__main__':
    main()
