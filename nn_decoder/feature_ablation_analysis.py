# -*- coding: utf-8 -*-
"""Feature-ablation CV regression with per-block AND per-metric ablations,
Ridge and Gaussian Process regression backends, applied across the seven
target families: IO-derived uncertainty (Q, L, D), behaviour (velocity),
stimulus components (orientation, contrast, dispersion).

Musall/Churchland-style "what features predict the target, which
contribute uniquely?" for the population metrics in
``population_metrics_vs_uncertainty.py``.

Block definitions
-----------------

Per ``documents/feature_catalog.md``:

  RATE_BLOCK
      Pop_Mean_Raw, Pop_Deviation, Spatial_Var, Centroid_Magnitude.
      Features that depend only on the time-averaged activity.

  TEMPORAL_VARIANCE_BLOCK
      Temporal_Fano_Raw, Temporal_Var, GV, GV_Fano,
      Participation_Ratio, Top_Mode_Dominance.
      Order-invariant features that require within-trial temporal
      variation. The clean substrate for Mate's question: does
      temporal variability carry uncertainty info beyond the mean?

  ORDER_SENSITIVE_BLOCK
      Traj_Length only.
      The single feature that *depends on the order* of time bins —
      shuffling bins changes it. Kept separate so the headline answer
      to Mate's question is not contaminated by sequence information.

Output rows
-----------

Each row of the returned list is a dict with at least::

    model           : 'full' | 'null' | 'only:<block>' | 'drop:<block>'
                       | 'only:<metric>' | 'ko:<metric>'
    cv              : 'kfold' or 'lomo'
    r2              : cross-validated R²
    n_features      : number of input columns
    ablation_type   : 'full' | 'block' | 'metric' | 'null'
    delta_r2_vs_full: r2 - full_r2 in the same cv scheme

Block rows also have ``block``; metric rows also have ``metric`` +
``metric_label``. ``drop:<block>`` and ``ko:<metric>`` rows also have
``unique_r2_vs_full = full_r2 - r2`` — the metric/block's unique
contribution that nothing else can substitute for.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel,
)
from sklearn.model_selection import KFold, LeaveOneGroupOut

# Re-use the existing within-mouse helpers and Ridge implementation.
from population_metrics_vs_uncertainty import (
    _within_mouse_zscore,
    _within_mouse_zscore_features,
    _r2_score,
    cv_ridge_r2,
    METRIC_BASES,
    set_plot_style,
)

# Okabe-Ito colour palette: 8 distinguishable, colour-blind-friendly hues.
# https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    'black':         '#000000',
    'orange':        '#E69F00',
    'sky_blue':      '#56B4E9',
    'bluish_green':  '#009E73',
    'yellow':        '#F0E442',
    'blue':          '#0072B2',
    'vermilion':     '#D55E00',
    'reddish_purple':'#CC79A7',
}
COLOR_RATE     = OKABE_ITO['sky_blue']
COLOR_TVAR     = OKABE_ITO['vermilion']
COLOR_ORDER    = OKABE_ITO['reddish_purple']
COLOR_ONLY     = OKABE_ITO['sky_blue']
COLOR_DROP     = OKABE_ITO['vermilion']
COLOR_FULL_REF = '#222222'
COLOR_NULL_REF = '#888888'
COLOR_MOUSE_DOT = '#222222'


# ======================================================================
# Block definitions
# ======================================================================

RATE_BLOCK = [
    'Pop_Mean_Raw',
    'Pop_Deviation',
    'Spatial_Var',
    'Centroid_Magnitude',
]

TEMPORAL_VARIANCE_BLOCK = [
    'Temporal_Fano_Raw',
    'Temporal_Var',
    'GV',
    'GV_Fano',
    'Participation_Ratio',
    'Top_Mode_Dominance',
]

# Note: Traj_Length (order-sensitive) is intentionally NOT in either
# block. It is excluded from the full feature set entirely so that the
# "drop:" comparisons cleanly contrast rate vs temporal-variance without
# Traj_Length residual contamination.

BLOCKS = {
    'rate': RATE_BLOCK,
    'temporal_variance': TEMPORAL_VARIANCE_BLOCK,
}

BLOCK_LABELS = {
    'rate': 'Rate',
    'temporal_variance': 'Temporal variance',
}

BLOCK_COLOURS = {
    'rate': COLOR_RATE,
    'temporal_variance': COLOR_TVAR,
}


def _active_metrics(blocks):
    """Flatten all block metrics into a single ordered list (no
    duplicates). The "full" feature set for the analysis is exactly
    this set × the chosen windows."""
    seen = set()
    out = []
    for metrics in blocks.values():
        for m in metrics:
            if m not in seen:
                out.append(m)
                seen.add(m)
    return out


# ======================================================================
# Target definitions
# ======================================================================

# (column_name, display_label, transform). transform in {None, 'log'}.
TARGETS = [
    ('Perceptual_Variance', 'Perceptual Variance (Q)', None),
    ('Likelihood_Variance', 'Likelihood Variance (L)', None),
    ('Decision_Entropy',    'Decision Entropy (D, [0,1])', None),
    ('Velocity',            'Pre-RZ Velocity', None),
    ('Orientation',         'Stimulus Orientation (|Δ from Go|)', None),
    ('Contrast',            'Stimulus Contrast (log)', 'log'),
    ('Dispersion',          'Stimulus Dispersion (log)', 'log'),
    ('Choice',              "Mouse's Choice (Go=1, NoGo=0)", None),
    ('Stim_Category',       'Stimulus Category (Go=1, NoGo=0)', None),
]


# ======================================================================
# Gaussian Process regressor
# ======================================================================

def cv_gpr_r2(X, y, groups, cv='kfold', n_splits=5, seed=0,
              n_restarts=2, length_scale=1.0, noise_init=1.0):
    """Cross-validated R² using GP regression with RBF + WhiteKernel.

    Mirrors ``population_metrics_vs_uncertainty.cv_ridge_r2`` so it can
    drop in as an alternative backend. Features should be z-scored by
    the caller (use ``_within_mouse_zscore_features`` first).

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    groups : ndarray (n,)   Mouse IDs (used only for LOMO).
    cv : {'kfold', 'lomo'}
    n_splits : int          Folds for kfold; ignored for LOMO.
    seed : int
    n_restarts : int        ``n_restarts_optimizer``. 0 for tests; 2-3
                            in production.
    length_scale, noise_init : float
                            Initial kernel hyperparameters; the GP
                            re-fits them per training fold.

    Returns
    -------
    (r2, y_hat_full, mask)   Same signature as cv_ridge_r2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    if X.size == 0 or y.size == 0:
        return np.nan, np.full(y.shape, np.nan), np.zeros(y.shape, dtype=bool)

    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    Xv = X[mask]
    yv = y[mask]
    gv = groups[mask] if groups.size else groups
    if Xv.shape[0] < 10 or Xv.shape[1] == 0:
        return np.nan, np.full(y.shape, np.nan), mask

    if cv == 'kfold':
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(Xv, yv)
    elif cv == 'lomo':
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(Xv, yv, gv)
    else:
        raise ValueError(f"cv must be 'kfold' or 'lomo', got {cv!r}")

    kernel = (ConstantKernel(1.0)
              * RBF(length_scale=length_scale)
              + WhiteKernel(noise_level=noise_init))

    y_hat_v = np.full(yv.shape, np.nan)
    for tr, te in split_iter:
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
            alpha=1e-8,
            normalize_y=True,
        )
        try:
            model.fit(Xv[tr], yv[tr])
            y_hat_v[te] = model.predict(Xv[te])
        except (np.linalg.LinAlgError, ValueError):
            y_hat_v[te] = np.nan

    y_hat_full = np.full(y.shape, np.nan)
    y_hat_full[mask] = y_hat_v
    return _r2_score(y, y_hat_full), y_hat_full, mask


# ======================================================================
# Per-mouse R² evaluation
# ======================================================================

def _r2_per_mouse(y, y_hat, groups):
    """Per-mouse R² from out-of-fold predictions.

    For each mouse, compute R² using that mouse's trials and that
    mouse's own mean as the constant baseline. This gives the
    per-mouse predictive R² of the pooled-trained model, which is
    the per-mouse "statistical unit" reviewers expect.

    Returns
    -------
    (per_mouse_r2, mouse_ids)
        Arrays of length n_unique_mice (sorted by Mouse_ID).
    """
    unique_mice = np.unique(groups)
    per_mouse = np.full(len(unique_mice), np.nan)
    for i, mid in enumerate(unique_mice):
        mask = (groups == mid)
        y_m = np.asarray(y[mask], dtype=float)
        yh_m = np.asarray(y_hat[mask], dtype=float)
        valid = ~(np.isnan(y_m) | np.isnan(yh_m))
        if valid.sum() < 3:
            continue
        y_v = y_m[valid]
        yh_v = yh_m[valid]
        ss_res = float(np.sum((y_v - yh_v) ** 2))
        ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
        if ss_tot < 1e-12:
            continue
        per_mouse[i] = 1.0 - ss_res / ss_tot
    return per_mouse, unique_mice


# ======================================================================
# Block-aware CV regression
# ======================================================================

def _columns_for_metric(metric_name, df, windows):
    return [f"{metric_name}_{w}" for w in windows
            if f"{metric_name}_{w}" in df.columns]


def _columns_for_block(block_metrics, df, windows):
    cols = []
    for m in block_metrics:
        cols.extend(_columns_for_metric(m, df, windows))
    return cols


def _run_regressor(X, y, groups, regressor, cv, n_splits, seed,
                   regressor_kwargs):
    """Run the regressor and return (pooled R², per-mouse R², mouse_ids,
    y_hat). y_hat is the out-of-fold per-trial prediction (NaN for any
    trial dropped during the fit), aligned with the input y."""
    if regressor == 'ridge':
        alpha = regressor_kwargs.get('alpha', 1.0)
        r2, y_hat, _ = cv_ridge_r2(X, y, groups, cv=cv, alpha=alpha,
                                    n_splits=n_splits, seed=seed)
    elif regressor == 'gpr':
        kw = {k: v for k, v in regressor_kwargs.items()
              if k in {'n_restarts', 'length_scale', 'noise_init'}}
        r2, y_hat, _ = cv_gpr_r2(X, y, groups, cv=cv,
                                  n_splits=n_splits, seed=seed, **kw)
    else:
        raise ValueError(f"Unknown regressor {regressor!r}")
    per_mouse_r2, mouse_ids = _r2_per_mouse(np.asarray(y), y_hat,
                                              np.asarray(groups))
    return r2, per_mouse_r2, mouse_ids, y_hat


# ======================================================================
# Trial-level cluster-robust statistics
# ======================================================================

def _cluster_robust_paired_t(yhat_A, yhat_B, y_true, mouse_ids):
    """Trial-level paired t-test on squared prediction errors with the
    standard error clustered by mouse.

    Null hypothesis: E[err_A - err_B] = 0, where
    err_X = (yhat_X - y_true)^2. The mean of the within-trial paired
    difference is tested with a cluster-robust SE (Liang-Zeger) using
    mouse as the clustering variable. Strictly more powerful than the
    per-mouse Wilcoxon when the within-mouse effect is consistent.

    Returns ``None`` on infeasible inputs; otherwise a dict::

        {'mean_delta': float,    # E[err_A - err_B] (negative = A better)
         'se':         float,    # cluster-robust SE
         't':          float,    # mean_delta / se
         'df':         int,      # G - 1, where G is # clusters
         'p':          float,    # two-sided
         'n_trials':   int,
         'n_clusters': int}
    """
    y_true = np.asarray(y_true, dtype=float)
    yA = np.asarray(yhat_A, dtype=float)
    yB = np.asarray(yhat_B, dtype=float)
    mouse_ids = np.asarray(mouse_ids)
    if not (len(yA) == len(yB) == len(y_true) == len(mouse_ids)):
        return None

    err_A = (yA - y_true) ** 2
    err_B = (yB - y_true) ** 2
    delta = err_A - err_B

    valid = ~(np.isnan(delta) | pd.isna(mouse_ids))
    delta = delta[valid]
    mids = mouse_ids[valid]
    if delta.size < 10:
        return None

    unique_mice = np.unique(mids)
    G = len(unique_mice)
    if G < 2:
        return None

    n = delta.size
    mean_delta = float(np.mean(delta))
    centered = delta - mean_delta
    cluster_sums = np.array(
        [float(np.sum(centered[mids == m])) for m in unique_mice]
    )
    # Variance of the mean estimator with cluster-robust adjustment:
    # V[mean] = (1/n^2) * sum_clusters (sum_within_cluster_centered)^2
    # Liang-Zeger small-sample correction:  G/(G-1) * (n-1)/(n-K), K=1
    var_mean = float(np.sum(cluster_sums ** 2)) / (n ** 2)
    correction = (G / (G - 1)) * ((n - 1) / (n - 1))
    se = float(np.sqrt(max(var_mean * correction, 0.0)))
    if se < 1e-15:
        return None
    t = mean_delta / se
    df = G - 1
    p_two = 2.0 * (1.0 - stats.t.cdf(abs(t), df=df))
    return {'mean_delta': mean_delta, 'se': se,
            't': float(t), 'df': int(df), 'p': float(p_two),
            'n_trials': int(n), 'n_clusters': int(G)}


def _trial_level_pairwise(yhat_arrays, y_true, mouse_ids, labels,
                           correction='bonferroni'):
    """Cluster-robust paired t-test for every pair of model variants.

    yhat_arrays : list of ndarray, one per model variant.
    y_true      : true target, shared across variants.
    mouse_ids   : mouse cluster IDs, shared across variants.
    labels      : list of group labels (same length as yhat_arrays).
    correction  : 'bonferroni' | 'holm' | 'none'

    Returns a dict mirroring :func:`_friedman_pairwise`::

        {'pairwise': list of {'i', 'j', 'label_i', 'label_j',
                              'p_raw', 'p_adj',
                              'mean_delta', 't', 'df',
                              'n_trials', 'n_clusters'},
         'n_pairs':  number of pairs returned}
    """
    n = len(yhat_arrays)
    pairs = []
    raw = []
    for i in range(n):
        for j in range(i + 1, n):
            stat = _cluster_robust_paired_t(
                yhat_arrays[i], yhat_arrays[j], y_true, mouse_ids)
            if stat is None:
                row = {'i': i, 'j': j,
                       'label_i': labels[i], 'label_j': labels[j],
                       'p_raw': np.nan, 'mean_delta': np.nan,
                       't': np.nan, 'df': np.nan,
                       'n_trials': 0, 'n_clusters': 0}
                p = np.nan
            else:
                row = {'i': i, 'j': j,
                       'label_i': labels[i], 'label_j': labels[j],
                       'p_raw': stat['p'],
                       'mean_delta': stat['mean_delta'],
                       't': stat['t'], 'df': stat['df'],
                       'n_trials': stat['n_trials'],
                       'n_clusters': stat['n_clusters']}
                p = stat['p']
            pairs.append(row)
            raw.append(p if not np.isnan(p) else 1.0)

    raw = np.array(raw)
    if correction == 'bonferroni':
        m = np.sum(~np.isnan(raw))
        adj = np.minimum(raw * max(m, 1), 1.0)
    elif correction == 'holm':
        adj = _holm_bonferroni(raw)
    elif correction == 'none':
        adj = raw.copy()
    else:
        raise ValueError(f"Unknown correction {correction!r}")
    for k, pair in enumerate(pairs):
        pair['p_adj'] = float(adj[k]) if not np.isnan(adj[k]) else np.nan

    return {'pairwise': pairs, 'n_pairs': len(pairs)}


def _apply_target_transform(df, ucol, transform):
    """Optionally log-transform the target before z-scoring. Returns the
    z-scored target as an ndarray aligned with df rows."""
    if transform is None:
        return _within_mouse_zscore(df, ucol)
    if transform == 'log':
        raw = np.asarray(df[ucol].values, dtype=float)
        with np.errstate(invalid='ignore'):
            raw = np.where(raw > 0, raw, np.nan)
        log_col = '__log_' + ucol
        df = df.copy()
        df[log_col] = np.log(raw)
        return _within_mouse_zscore(df, log_col)
    raise ValueError(f"Unknown target_transform {transform!r}")


def run_cv_regression_blocks(
    df, ucol, ulabel,
    regressor='ridge',
    cv_schemes=('kfold', 'lomo'),
    n_splits=5,
    seed=0,
    blocks=None,
    windows=('Full', 'Late'),
    target_transform=None,
    include_null=False,
    include_per_metric=True,
    regressor_kwargs=None,
    return_meta=False,
):
    """Run CV regression with per-block and per-metric ablations.

    See module docstring for the full output schema. ``regressor`` is
    'ridge' or 'gpr'. ``regressor_kwargs`` is a dict of regressor-specific
    kwargs forwarded to ``cv_ridge_r2`` / ``cv_gpr_r2``.

    If ``return_meta=True`` returns ``(rows, meta)`` where meta is a dict
    with the trial-aligned target ``y_true`` and per-trial mouse IDs
    ``mouse_ids_per_trial``. These are needed by the trial-level
    cluster-robust paired tests.
    """
    if blocks is None:
        blocks = BLOCKS
    if regressor_kwargs is None:
        regressor_kwargs = {}

    target = _apply_target_transform(df, ucol, target_transform)
    groups = df['Mouse_ID'].values

    # Active metrics = union across all blocks (Traj_Length excluded).
    # The "full" model uses only these metrics, so drop:/ko: comparisons
    # don't leak signal through metrics that aren't in any block.
    active_metrics = _active_metrics(blocks)
    label_lookup = dict(METRIC_BASES)

    full_cols = [f"{base}_{w}"
                 for base in active_metrics
                 for w in windows
                 if f"{base}_{w}" in df.columns]

    rows = []

    def _append(model_name, X_, target_, ablation_type, **extra):
        for cv in cv_schemes:
            r2, per_mouse, m_ids, y_hat = _run_regressor(
                X_, target_, groups, regressor, cv,
                n_splits, seed, regressor_kwargs)
            row = {'model': model_name, 'cv': cv, 'r2': r2,
                   'r2_per_mouse': per_mouse.tolist(),
                   'mouse_ids': m_ids.tolist(),
                   'n_features': X_.shape[1],
                   'ablation_type': ablation_type,
                   'y_hat': y_hat.tolist()}
            row.update(extra)
            rows.append(row)

    # --- Full model ---
    if full_cols:
        X = _within_mouse_zscore_features(df, full_cols, groups)
        _append('full', X, target, 'full')

    # --- Null baseline: shuffled target ---
    if include_null and full_cols:
        rng = np.random.default_rng(seed)
        target_shuf = target.copy()
        for mid in np.unique(groups):
            mask = groups == mid
            idx = np.where(mask)[0]
            perm = rng.permutation(len(idx))
            target_shuf[idx] = target_shuf[idx[perm]]
        X = _within_mouse_zscore_features(df, full_cols, groups)
        _append('null', X, target_shuf, 'null')

    # --- Per-block ablations ---
    for block_name, block_metrics in blocks.items():
        block_cols = _columns_for_block(block_metrics, df, windows)
        if not block_cols:
            continue

        X_only = _within_mouse_zscore_features(df, block_cols, groups)
        _append(f'only:{block_name}', X_only, target, 'block',
                block=block_name)

        drop_cols = [c for c in full_cols if c not in block_cols]
        if drop_cols:
            X_drop = _within_mouse_zscore_features(df, drop_cols, groups)
            _append(f'drop:{block_name}', X_drop, target, 'block',
                    block=block_name)

    # --- Per-metric ablations (over active metrics only) ---
    if include_per_metric:
        for base in active_metrics:
            label = label_lookup.get(base, base)
            metric_cols = _columns_for_metric(base, df, windows)
            if not metric_cols:
                continue

            X_only = _within_mouse_zscore_features(df, metric_cols, groups)
            _append(f'only:{base}', X_only, target, 'metric',
                    metric=base, metric_label=label)

            ko_cols = [c for c in full_cols if c not in metric_cols]
            if ko_cols:
                X_ko = _within_mouse_zscore_features(df, ko_cols, groups)
                _append(f'ko:{base}', X_ko, target, 'metric',
                        metric=base, metric_label=label)

    # --- Within-CV deltas ---
    cv_names = sorted({r['cv'] for r in rows})
    for cv_name in cv_names:
        full_in_cv = [r for r in rows
                      if r['model'] == 'full' and r['cv'] == cv_name]
        if not full_in_cv:
            continue
        full_r2 = full_in_cv[0]['r2']
        for r in rows:
            if r['cv'] != cv_name:
                continue
            r['delta_r2_vs_full'] = r['r2'] - full_r2
            mname = r['model']
            if isinstance(mname, str) and (mname.startswith('ko:')
                                            or mname.startswith('drop:')):
                r['unique_r2_vs_full'] = full_r2 - r['r2']

    if return_meta:
        meta = {'y_true': np.asarray(target).tolist(),
                'mouse_ids_per_trial': np.asarray(groups).tolist(),
                'target_column': ucol,
                'target_label': ulabel}
        return rows, meta
    return rows


# ======================================================================
# Driver: loop over targets x regressors
# ======================================================================

def add_stimulus_and_behaviour_columns(df):
    """Ensure df has the columns expected by all target families and
    derive Stim_Category from True_Orientation if it's not already there.

    Stim_Category: 1 if True_Orientation < 45 (Go), 0 if > 45 (NoGo),
    0.5 if exactly 45 (boundary, randomised 50/50 by experiment design).
    """
    required = ['Mouse_ID', 'Orientation', 'Contrast', 'Dispersion',
                'Velocity', 'Choice', 'Perceptual_Variance',
                'Decision_Entropy', 'Likelihood_Variance']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataframe is missing columns expected for the target "
            f"ablation: {missing}. Re-run "
            "population_metrics_vs_uncertainty.run_population_variance_pipeline "
            "to regenerate."
        )
    if 'Stim_Category' not in df.columns:
        if 'True_Orientation' not in df.columns:
            raise ValueError(
                "Cannot derive Stim_Category: neither Stim_Category nor "
                "True_Orientation is in the dataframe. Re-run the feature "
                "pipeline after the True_Orientation patch."
            )
        ori = np.asarray(df['True_Orientation'].values, dtype=float)
        cat = np.where(ori < 45.0, 1.0,
                       np.where(ori > 45.0, 0.0, 0.5))
        df = df.copy()
        df['Stim_Category'] = cat
    return df


def run_all_targets_all_regressors(
    df,
    targets=TARGETS,
    regressors=('ridge', 'gpr'),
    cv_schemes=('kfold', 'lomo'),
    output_dir='feature_ablation_results',
    n_splits=5,
    seed=0,
    include_null=True,
    include_per_metric=True,
    regressor_kwargs=None,
    windows=('Full', 'Late'),
):
    """Loop over (target, regressor, window) combos.

    Each window is run as a separate analysis — the Full (0–2 s) and
    Late (1–2 s) feature sets are NOT pooled, so any difference in
    per-block contributions between the two windows is visible.

    Writes one CSV + NPZ companion per (target, regressor, window) and
    returns a dict keyed by ``(target, regressor, window)`` mapping to
    ``{'rows': [...], 'meta': {...}}``.
    """
    if regressor_kwargs is None:
        regressor_kwargs = {'ridge': {'alpha': 1.0},
                            'gpr': {'n_restarts': 2}}

    os.makedirs(output_dir, exist_ok=True)
    df = add_stimulus_and_behaviour_columns(df)

    all_results = {}
    for window in windows:
        for ucol, ulabel, transform in targets:
            if ucol not in df.columns or df[ucol].dropna().empty:
                print(f"  [skip] target {ucol} not in df")
                continue
            for regressor in regressors:
                print(f"  fitting {regressor} on {ucol} ({ulabel}) "
                      f"window={window}...")
                rkw = dict(regressor_kwargs.get(regressor, {}))
                rows, meta = run_cv_regression_blocks(
                    df, ucol, ulabel,
                    regressor=regressor,
                    cv_schemes=cv_schemes,
                    n_splits=n_splits,
                    seed=seed,
                    target_transform=transform,
                    include_null=include_null,
                    include_per_metric=include_per_metric,
                    regressor_kwargs=rkw,
                    windows=(window,),
                    return_meta=True,
                )
                meta['window'] = window
                all_results[(ucol, regressor, window)] = {
                    'rows': rows, 'meta': meta,
                }

                csv_rows = [{k: v for k, v in r.items() if k != 'y_hat'}
                             for r in rows]
                tag = f"{ucol}_{regressor}_{window}"
                out_csv = os.path.join(output_dir, f"ablation_{tag}.csv")
                pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
                out_npz = os.path.join(output_dir, f"ablation_{tag}_yhat.npz")
                yhat_dict = {f"{r['model']}__{r['cv']}":
                              np.asarray(r['y_hat'])
                              for r in rows if 'y_hat' in r}
                yhat_dict['__y_true'] = np.asarray(meta['y_true'])
                yhat_dict['__mouse_ids_per_trial'] = np.asarray(
                    meta['mouse_ids_per_trial'])
                np.savez_compressed(out_npz, **yhat_dict)
                print(f"    wrote {out_csv} + companion {out_npz}")
    return all_results


# ======================================================================
# Plotting helpers
# ======================================================================

def _arr(values):
    """Robust array of per-mouse R² from a row dict's r2_per_mouse field.
    Strings are tolerated (csv-loaded rows)."""
    import ast
    if isinstance(values, str):
        values = ast.literal_eval(values)
    return np.asarray(values, dtype=float)


def _wilcoxon_pair(a, b, alternative='two-sided'):
    """Paired Wilcoxon signed-rank between two per-mouse arrays. Returns
    (statistic, p_value). NaN-safe; falls back to (nan, nan) if too
    few non-NaN paired values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 4:
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(a[mask], b[mask], alternative=alternative,
                              zero_method='wilcox')
    except ValueError:
        return np.nan, np.nan
    return float(res.statistic), float(res.pvalue)


def _holm_bonferroni(pvalues):
    """Holm-Bonferroni step-down adjustment. NaN p-values pass through
    as NaN."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    valid = ~np.isnan(p)
    if not np.any(valid):
        return out
    idx_valid = np.where(valid)[0]
    p_v = p[idx_valid]
    order = np.argsort(p_v)
    n_v = p_v.size
    adj = np.empty(n_v)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, p_v[k] * (n_v - rank))
        adj[k] = min(running, 1.0)
    out[idx_valid] = adj
    return out


def _friedman_pairwise(arrays, labels):
    """Friedman omnibus + pairwise Wilcoxon signed-rank with Holm
    correction across an arbitrary number of paired groups.

    arrays : list of per-mouse arrays (must all be the same length =
             n_mice, with NaN allowed for missing per-mouse R²).

    Returns a dict::

        {'omnibus_stat':  Friedman chi²,
         'omnibus_p':     omnibus p,
         'n_valid':       number of mice with all groups present,
         'pairwise':      list of {'i', 'j', 'label_i', 'label_j',
                                   'p_raw', 'p_holm'} dicts}

    Returns ``None`` if the test is infeasible (< 2 groups, < 3 valid
    mice, or all-identical values).
    """
    if len(arrays) < 2:
        return None
    mat = np.column_stack([np.asarray(a, dtype=float) for a in arrays])
    valid_rows = ~np.any(np.isnan(mat), axis=1)
    if valid_rows.sum() < 3:
        return None
    mat = mat[valid_rows]
    n_groups = mat.shape[1]

    try:
        fres = stats.friedmanchisquare(*[mat[:, k] for k in range(n_groups)])
        omnibus_stat = float(fres.statistic)
        omnibus_p = float(fres.pvalue)
    except (ValueError, Exception):
        omnibus_stat = np.nan
        omnibus_p = np.nan

    pairs = []
    raw = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            try:
                wres = stats.wilcoxon(mat[:, i], mat[:, j],
                                       zero_method='wilcox')
                p = float(wres.pvalue)
            except ValueError:
                p = np.nan
            pairs.append({'i': i, 'j': j,
                           'label_i': labels[i] if labels else str(i),
                           'label_j': labels[j] if labels else str(j),
                           'p_raw': p})
            raw.append(p if not np.isnan(p) else 1.0)

    adj = _holm_bonferroni(np.array(raw))
    for k, pair in enumerate(pairs):
        pa = float(adj[k]) if not np.isnan(adj[k]) else np.nan
        pair['p_holm'] = pa
        pair['p_adj'] = pa  # unified key shared with trial-level results

    return {'omnibus_stat': omnibus_stat,
            'omnibus_p': omnibus_p,
            'n_valid': int(valid_rows.sum()),
            'pairwise': pairs}


def _omnibus_blurb(stats_dict, group_name='groups'):
    """Format a short omnibus result string for a panel title."""
    if stats_dict is None:
        return ''
    s = stats_dict['omnibus_stat']
    p = stats_dict['omnibus_p']
    n = stats_dict.get('n_valid', np.nan)
    if np.isnan(s) or np.isnan(p):
        return ''
    return f"Friedman across {group_name}: χ²={s:.2f}, p={p:.3g} (n={n})"


def _sig_marker(p):
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def _draw_cross_brackets(ax, pairs, x_centers, y_top, y_range,
                          colour, max_brackets=None, tier_offset=0.0,
                          show_only_significant=False):
    """Draw bracket annotations for cross-group pairwise comparisons,
    tiered vertically to avoid overlap.

    pairs : list of {'i', 'j', 'p_raw', 'p_holm'}
    x_centers : array of bar centers (length = n_groups). The bracket
                spans from x_centers[i] to x_centers[j].
    """
    if not pairs:
        return
    # Order pairs by raw p ascending (most significant first)
    pairs_sorted = sorted(pairs, key=lambda d: d.get('p_raw',
                                                       float('inf')))
    if show_only_significant:
        pairs_sorted = [d for d in pairs_sorted
                        if not np.isnan(d.get('p_holm', np.nan))
                        and d['p_holm'] < 0.05]
    if max_brackets is not None:
        pairs_sorted = pairs_sorted[:max_brackets]

    tier = 0
    for pair in pairs_sorted:
        i, j = pair['i'], pair['j']
        p_adj = pair.get('p_adj', pair.get('p_holm', np.nan))
        p_raw = pair.get('p_raw', np.nan)
        if np.isnan(p_raw):
            continue
        if i >= len(x_centers) or j >= len(x_centers):
            continue
        y = y_top + (tier_offset + 0.05 + 0.075 * tier) * y_range
        tier += 1
        xL = x_centers[i]
        xR = x_centers[j]
        # Bracket
        alpha = 1.0 if (not np.isnan(p_adj) and p_adj < 0.05) else 0.45
        ax.plot([xL, xL, xR, xR],
                [y, y + 0.012 * y_range,
                 y + 0.012 * y_range, y],
                color=colour, lw=0.8, alpha=alpha, zorder=5)
        # Label: corrected p, with significance star if appropriate.
        marker = _sig_marker(p_adj) if not np.isnan(p_adj) else ''
        if marker in ('', 'n.s.'):
            text = f"p={p_adj:.2g}" if not np.isnan(p_adj) else f"p={p_raw:.2g}"
        else:
            text = f"{marker} (p={p_adj:.3g})"
        ax.text((xL + xR) / 2, y + 0.018 * y_range,
                text, ha='center', va='bottom',
                fontsize=7, color=colour if alpha == 1.0 else '#555')


def _clustered_bar_panel(
    ax, group_names, only_arrays, paired_arrays,
    only_side_label='Only', paired_side_label='Drop',
    group_colours=None, hatch_paired=False,
    full_per_mouse=None, null_per_mouse=None,
    brackets=None, vs_full_pvalues=None,
    cluster_gap=1.2, bar_width=0.8,
    rotate_labels=0,
):
    """Two-cluster bar panel.

    Bars are clustered by SIDE: only-side bars on the left, paired-side
    (drop / ko) bars on the right, with a gap between. ``group_names``
    labels the within-cluster ordering, applied identically to both
    sides. Per-mouse dots overlay each bar with stable jitter per
    mouse. Connecting lines per mouse are drawn ONLY between bars that
    appear in ``brackets`` (the pairs being tested), within their
    cluster. No lines are drawn across only/paired.

    ``brackets`` and ``vs_full_pvalues`` describe what to annotate:
        brackets: [{'side': 'only'|'paired', 'i': k1, 'j': k2, 'p': pval}, ...]
        vs_full_pvalues: [{'side': 'only'|'paired', 'i': k, 'p': pval}, ...]
    """
    n = len(group_names)
    x_only = np.arange(n, dtype=float)
    x_paired = np.arange(n, dtype=float) + n + cluster_gap
    width = bar_width
    rng = np.random.default_rng(7)
    if group_colours is None:
        group_colours = ['#888888'] * n

    only_means = np.array([np.nanmean(a) if len(a) else np.nan
                            for a in only_arrays])
    only_sems = np.array([stats.sem(a, nan_policy='omit') if len(a) > 1
                          else 0.0 for a in only_arrays])
    paired_means = np.array([np.nanmean(a) if len(a) else np.nan
                              for a in paired_arrays])
    paired_sems = np.array([stats.sem(a, nan_policy='omit') if len(a) > 1
                            else 0.0 for a in paired_arrays])

    for k in range(n):
        ax.bar(x_only[k], only_means[k], width, yerr=only_sems[k],
               color=group_colours[k], alpha=0.78,
               edgecolor='black', linewidth=0.6, capsize=3, zorder=2)
        ax.bar(x_paired[k], paired_means[k], width, yerr=paired_sems[k],
               color=group_colours[k], alpha=0.78,
               edgecolor='black', linewidth=0.6, capsize=3, zorder=2,
               hatch='///' if hatch_paired else None)

    n_mice = max((len(a) for a in only_arrays + paired_arrays), default=0)
    per_mouse_jitter = [float(rng.uniform(-0.12, 0.12))
                         for _ in range(n_mice)]

    # Per-mouse dots: stable jitter per mouse used in both clusters.
    for m_idx in range(n_mice):
        jitter = per_mouse_jitter[m_idx]
        for k in range(n):
            ov = (only_arrays[k][m_idx]
                  if m_idx < len(only_arrays[k]) else np.nan)
            pv = (paired_arrays[k][m_idx]
                  if m_idx < len(paired_arrays[k]) else np.nan)
            if not np.isnan(ov):
                ax.scatter(x_only[k] + jitter, ov,
                           color=COLOR_MOUSE_DOT, s=14, alpha=0.7,
                           linewidths=0, zorder=4)
            if not np.isnan(pv):
                ax.scatter(x_paired[k] + jitter, pv,
                           color=COLOR_MOUSE_DOT, s=14, alpha=0.7,
                           linewidths=0, zorder=4)

    # Within-cluster connecting lines per mouse — ONLY for the pairs in
    # ``brackets``. No lines across only/paired clusters, and no lines
    # between pairs of bars that aren't being tested.
    if brackets:
        for br in brackets:
            side = br['side']
            i, j = br['i'], br['j']
            if i >= n or j >= n:
                continue
            centers = x_only if side == 'only' else x_paired
            arrs = only_arrays if side == 'only' else paired_arrays
            for m_idx in range(n_mice):
                jitter = per_mouse_jitter[m_idx]
                va = arrs[i][m_idx] if m_idx < len(arrs[i]) else np.nan
                vb = arrs[j][m_idx] if m_idx < len(arrs[j]) else np.nan
                if not np.isnan(va) and not np.isnan(vb):
                    ax.plot([centers[i] + jitter, centers[j] + jitter],
                            [va, vb], color=COLOR_MOUSE_DOT,
                            alpha=0.30, lw=0.6, zorder=3)

    # Reference lines.
    if full_per_mouse is not None and len(full_per_mouse):
        fm = float(np.nanmean(full_per_mouse))
        fs = float(stats.sem(full_per_mouse, nan_policy='omit'))
        ax.axhline(fm, color=COLOR_FULL_REF, ls='--', lw=1.3, zorder=1,
                    label=f'Full mean ({fm:+.3f})')
        ax.axhspan(fm - fs, fm + fs,
                    color=COLOR_FULL_REF, alpha=0.07, zorder=0)
    if null_per_mouse is not None and len(null_per_mouse):
        nm = float(np.nanmean(null_per_mouse))
        ax.axhline(nm, color=COLOR_NULL_REF, ls=':', lw=1.2, zorder=1,
                    label=f'Null mean ({nm:+.3f})')
    ax.axhline(0, color='#333', lw=0.6, zorder=1)

    ax.set_xticks(np.concatenate([x_only, x_paired]))
    ax.set_xticklabels(list(group_names) + list(group_names),
                       rotation=rotate_labels,
                       ha='right' if rotate_labels else 'center')

    # Y range for bracket / annotation placement.
    all_vals = np.concatenate([a for a in only_arrays + paired_arrays
                                if len(a) > 0])
    if all_vals.size == 0:
        return ax
    y_top = float(np.nanmax(all_vals))
    y_min = min(0.0, float(np.nanmin(all_vals)))
    y_range = max(y_top - y_min, 0.05)

    # Cluster headers above each cluster.
    only_center = float(np.mean(x_only))
    paired_center = float(np.mean(x_paired))
    header_y = y_top + 0.30 * y_range
    ax.text(only_center, header_y, only_side_label,
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=COLOR_ONLY)
    ax.text(paired_center, header_y, paired_side_label,
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=COLOR_DROP)

    # Within-cluster brackets.
    if brackets:
        only_tier = 0
        paired_tier = 0
        for br in brackets:
            side = br['side']
            i, j = br['i'], br['j']
            p = br.get('p', np.nan)
            if i >= n or j >= n:
                continue
            centers = x_only if side == 'only' else x_paired
            if side == 'only':
                tier = only_tier; only_tier += 1
            else:
                tier = paired_tier; paired_tier += 1
            y = y_top + (0.08 + 0.08 * tier) * y_range
            xL, xR = centers[i], centers[j]
            alpha = 1.0 if (not np.isnan(p) and p < 0.05) else 0.55
            ax.plot([xL, xL, xR, xR],
                    [y, y + 0.014 * y_range,
                     y + 0.014 * y_range, y],
                    color='black', lw=1.0, alpha=alpha, zorder=5)
            marker = _sig_marker(p) if not np.isnan(p) else ''
            text = (f"{marker} p={p:.3g}"
                    if marker not in ('', 'n.s.') else f"p={p:.2g}")
            ax.text((xL + xR) / 2, y + 0.020 * y_range, text,
                    ha='center', va='bottom', fontsize=8,
                    color='black' if alpha == 1.0 else '#555')

    # "vs full" annotations on individual bars.
    if vs_full_pvalues:
        for ann in vs_full_pvalues:
            i = ann['i']
            if i >= n:
                continue
            p = ann.get('p', np.nan)
            if ann['side'] == 'only':
                xc = x_only[i]; value_arr = only_arrays[i]
            else:
                xc = x_paired[i]; value_arr = paired_arrays[i]
            if not len(value_arr):
                continue
            top_of_bar = (float(np.nanmean(value_arr))
                          + (float(stats.sem(value_arr, nan_policy='omit'))
                             if len(value_arr) > 1 else 0.0))
            y = top_of_bar + 0.02 * y_range
            marker = _sig_marker(p) if not np.isnan(p) else ''
            text = (f"vs full\n{marker} p={p:.3g}"
                    if marker not in ('', 'n.s.')
                    else f"vs full\np={p:.2g}")
            ax.text(xc, y, text, ha='center', va='bottom',
                    fontsize=7, color='#333',
                    bbox=dict(facecolor='white', alpha=0.7,
                               edgecolor='none', pad=1))

    return ax


def _paired_bar_panel(
    ax, group_names, only_arrays, paired_arrays,
    only_label='only', paired_label='drop',
    only_colour=COLOR_ONLY, paired_colour=COLOR_DROP,
    full_per_mouse=None, null_per_mouse=None,
    brackets=None,
    vs_full_pvalues=None,
    rotate_labels=0,
    draw_connecting_lines=True,
):
    """Paired-bar panel with per-mouse points, within-group connecting
    lines, and explicit user-supplied significance annotations.

    Parameters
    ----------
    only_arrays, paired_arrays : list[ndarray]
        Per-mouse R² arrays, one per group.
    brackets : list of dict, optional
        Each entry::

            {'side': 'only' | 'paired',
             'i': group_idx_left, 'j': group_idx_right,
             'p': p-value, 'label': str}

        Draws a horizontal bracket over the ``side`` bars connecting
        groups ``i`` and ``j`` with the p-value annotated.
    vs_full_pvalues : list of dict, optional
        Each entry::

            {'side': 'only' | 'paired', 'i': group_idx, 'p': p}

        Draws a small "vs full: p=…" text annotation above that bar.
    """
    n_groups = len(group_names)
    x = np.arange(n_groups, dtype=float)
    width = 0.36
    rng = np.random.default_rng(7)

    only_means = np.array([np.nanmean(a) if len(a) else np.nan
                            for a in only_arrays])
    only_sems = np.array([stats.sem(a, nan_policy='omit') if len(a) > 1
                          else 0.0 for a in only_arrays])
    paired_means = np.array([np.nanmean(a) if len(a) else np.nan
                              for a in paired_arrays])
    paired_sems = np.array([stats.sem(a, nan_policy='omit') if len(a) > 1
                            else 0.0 for a in paired_arrays])

    ax.bar(x - width/2, only_means, width, yerr=only_sems,
           color=only_colour, alpha=0.75, edgecolor='black', linewidth=0.6,
           capsize=3, label=only_label, zorder=2)
    ax.bar(x + width/2, paired_means, width, yerr=paired_sems,
           color=paired_colour, alpha=0.75, edgecolor='black', linewidth=0.6,
           capsize=3, label=paired_label, zorder=2)

    # Per-mouse dots + within-group connecting lines.
    # Each mouse gets a stable horizontal offset drawn once, so dots for
    # the same mouse line up horizontally across all groups (within the
    # only/paired side).
    n_mice = max((len(a) for a in only_arrays + paired_arrays), default=0)
    per_mouse_jitter = [float(rng.uniform(-0.06, 0.06))
                         for _ in range(n_mice)]
    for m_idx in range(n_mice):
        jitter = per_mouse_jitter[m_idx]
        for i in range(n_groups):
            ov = (only_arrays[i][m_idx]
                  if m_idx < len(only_arrays[i]) else np.nan)
            pv = (paired_arrays[i][m_idx]
                  if m_idx < len(paired_arrays[i]) else np.nan)
            x_only = x[i] - width/2 + jitter
            x_paired = x[i] + width/2 + jitter
            if draw_connecting_lines and not np.isnan(ov) and not np.isnan(pv):
                ax.plot([x_only, x_paired], [ov, pv],
                        color=COLOR_MOUSE_DOT, alpha=0.30, lw=0.6, zorder=3)
            if not np.isnan(ov):
                ax.scatter(x_only, ov,
                           color=COLOR_MOUSE_DOT, s=14, alpha=0.7,
                           linewidths=0, zorder=4)
            if not np.isnan(pv):
                ax.scatter(x_paired, pv,
                           color=COLOR_MOUSE_DOT, s=14, alpha=0.7,
                           linewidths=0, zorder=4)

    # Reference bands.
    full_mean = None
    if full_per_mouse is not None and len(full_per_mouse):
        full_mean = float(np.nanmean(full_per_mouse))
        full_sem = float(stats.sem(full_per_mouse, nan_policy='omit'))
        ax.axhline(full_mean, color=COLOR_FULL_REF, ls='--', lw=1.3,
                    zorder=1, label=f'Full mean ({full_mean:+.3f})')
        ax.axhspan(full_mean - full_sem, full_mean + full_sem,
                    color=COLOR_FULL_REF, alpha=0.07, zorder=0)
    if null_per_mouse is not None and len(null_per_mouse):
        null_mean = float(np.nanmean(null_per_mouse))
        ax.axhline(null_mean, color=COLOR_NULL_REF, ls=':', lw=1.2,
                    zorder=1, label=f'Null mean ({null_mean:+.3f})')

    ax.axhline(0, color='#333', lw=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=rotate_labels,
                       ha='right' if rotate_labels else 'center')

    # Vertical extent for bracket / annotation placement
    all_vals = np.concatenate([a for a in only_arrays + paired_arrays
                                if len(a) > 0])
    if all_vals.size == 0:
        return ax
    y_top = float(np.nanmax(all_vals))
    y_min = min(0.0, float(np.nanmin(all_vals)))
    y_range = max(y_top - y_min, 0.05)

    # User-supplied significance brackets (cross-group).
    if brackets:
        only_x = x - width/2
        paired_x = x + width/2
        # Sort by side to stack: only brackets at tier 0, paired at tier 1+
        only_tier = 0
        paired_tier = 0
        for b in brackets:
            side = b['side']
            i, j = b['i'], b['j']
            p = b.get('p', np.nan)
            if i >= n_groups or j >= n_groups:
                continue
            if side == 'only':
                centers = only_x
                colour = only_colour
                tier = only_tier
                only_tier += 1
            else:
                centers = paired_x
                colour = paired_colour
                tier = paired_tier
                paired_tier += 1
            y = y_top + (0.07 + 0.08 * tier) * y_range
            xL, xR = centers[i], centers[j]
            alpha = 1.0 if (not np.isnan(p) and p < 0.05) else 0.55
            ax.plot([xL, xL, xR, xR],
                    [y, y + 0.014 * y_range,
                     y + 0.014 * y_range, y],
                    color=colour, lw=1.0, alpha=alpha, zorder=5)
            marker = _sig_marker(p) if not np.isnan(p) else ''
            text = (f"{marker} p={p:.3g}"
                    if marker not in ('', 'n.s.') else f"p={p:.2g}")
            ax.text((xL + xR) / 2, y + 0.020 * y_range,
                    text, ha='center', va='bottom', fontsize=8,
                    color=colour if alpha == 1.0 else '#555')

    # "vs full" small text labels on individual bars.
    if vs_full_pvalues:
        only_x = x - width/2
        paired_x = x + width/2
        for ann in vs_full_pvalues:
            i = ann['i']
            if i >= n_groups:
                continue
            p = ann.get('p', np.nan)
            if ann['side'] == 'only':
                xc = only_x[i]
                value_arr = only_arrays[i]
            else:
                xc = paired_x[i]
                value_arr = paired_arrays[i]
            if not len(value_arr):
                continue
            top_of_bar = (float(np.nanmean(value_arr))
                          + (float(stats.sem(value_arr, nan_policy='omit'))
                             if len(value_arr) > 1 else 0.0))
            y = top_of_bar + 0.02 * y_range
            marker = _sig_marker(p) if not np.isnan(p) else ''
            text = (f"vs full\n{marker} p={p:.3g}"
                    if marker not in ('', 'n.s.')
                    else f"vs full\np={p:.2g}")
            ax.text(xc, y, text, ha='center', va='bottom',
                    fontsize=7, color='#333',
                    bbox=dict(facecolor='white', alpha=0.7,
                               edgecolor='none', pad=1))

    return ax


def _per_mouse_paired_t(arr_a, arr_b):
    """Standard paired t-test on per-mouse R² arrays (n=6 → df=5).

    This is the test that matches what the bar plots show: bars are
    means across mice of per-mouse R², error bars are per-mouse SEM,
    and this p-value tests whether the within-mouse R² difference is
    consistently signed across the 6 mice.  Returns NaN if too few
    valid pairs.
    """
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    try:
        res = stats.ttest_rel(a[mask], b[mask])
        return float(res.pvalue)
    except Exception:
        return np.nan


def _paired_t_yhat(rows_df, model_a, model_b, cv_scheme):
    """Per-mouse paired t-test between two named model rows.

    Returns the p-value from a standard paired t-test on the per-mouse
    R² arrays from each row (n=6 mice → df=5). Returns NaN if either
    row is missing.
    """
    a_rows = rows_df[(rows_df['model'] == model_a)
                      & (rows_df['cv'] == cv_scheme)]
    b_rows = rows_df[(rows_df['model'] == model_b)
                      & (rows_df['cv'] == cv_scheme)]
    if a_rows.empty or b_rows.empty:
        return np.nan
    pmA = _arr(a_rows.iloc[0]['r2_per_mouse'])
    pmB = _arr(b_rows.iloc[0]['r2_per_mouse'])
    return _per_mouse_paired_t(pmA, pmB)


def plot_musall_style(
    rows, target_label, regressor_label,
    cv_scheme='kfold', output_path=None, top_k_metrics=8,
    meta=None,
):
    """Two-panel Musall/Churchland-style figure with per-mouse paired bars.

    - Block panel: rate vs temporal_variance bars (only and drop side
      each); planned paired t-tests between only:rate ↔ only:tvar and
      drop:rate ↔ drop:tvar as cross-block brackets; "vs full" paired
      t-test annotated above each drop bar.
    - Metric panel: top-k metrics by unique contribution; only:metric vs
      ko:metric bars; "vs full" paired t-test annotated above each ko bar.

    All p-values are trial-level cluster-robust paired t-tests (SE
    clustered by mouse, df = G − 1 = 5). These are planned comparisons,
    so no multiple-comparisons correction is applied. Bar = mean ± SEM
    across mice; dots = individual mice; thin grey lines pair within-block
    only→drop per mouse. ``meta`` is required for the p-values to be
    computed.
    """
    set_plot_style()
    df_all = pd.DataFrame(rows)
    df = df_all[df_all['cv'] == cv_scheme].copy()
    if df.empty:
        return None

    full = df[df['model'] == 'full']
    if full.empty:
        return None
    full_per_mouse = _arr(full.iloc[0]['r2_per_mouse'])

    null_per_mouse = None
    if (df['model'] == 'null').any():
        null_per_mouse = _arr(df[df['model'] == 'null'].iloc[0]['r2_per_mouse'])

    block_df = df[df['ablation_type'] == 'block'].copy()
    metric_df = df[df['ablation_type'] == 'metric'].copy()

    fig, (ax_block, ax_metric) = plt.subplots(
        2, 1,
        figsize=(max(11, 1.0 * min(top_k_metrics, 8) + 5), 11),
        constrained_layout=True,
        gridspec_kw={'height_ratios': [1.0, 1.4]},
    )

    # ------------------------------------------------------------------
    # Block panel
    # ------------------------------------------------------------------
    if not block_df.empty:
        block_df['kind'] = block_df['model'].str.split(':').str[0]
        # Stable order: rate first, then temporal_variance (matches BLOCKS).
        order = [b for b in BLOCKS.keys() if b in block_df['block'].values]
        only_arrays, drop_arrays = [], []
        for b in order:
            ob = block_df[(block_df['kind'] == 'only')
                          & (block_df['block'] == b)]
            db = block_df[(block_df['kind'] == 'drop')
                          & (block_df['block'] == b)]
            only_arrays.append(_arr(ob.iloc[0]['r2_per_mouse'])
                                if not ob.empty else np.array([]))
            drop_arrays.append(_arr(db.iloc[0]['r2_per_mouse'])
                                if not db.empty else np.array([]))
        block_labels_pretty = [BLOCK_LABELS.get(b, b) for b in order]

        # Planned comparisons (no multiple-comparison correction):
        # cross-block (only vs only, drop vs drop) + full vs drop per block.
        brackets = []
        vs_full = []
        if len(order) >= 2:
            p_only = _paired_t_yhat(
                df_all, f"only:{order[0]}", f"only:{order[1]}", cv_scheme)
            p_drop = _paired_t_yhat(
                df_all, f"drop:{order[0]}", f"drop:{order[1]}", cv_scheme)
            brackets.append({'side': 'only', 'i': 0, 'j': 1, 'p': p_only})
            brackets.append({'side': 'paired', 'i': 0, 'j': 1, 'p': p_drop})
        for i, b in enumerate(order):
            p_vs = _paired_t_yhat(
                df_all, 'full', f"drop:{b}", cv_scheme)
            vs_full.append({'side': 'paired', 'i': i, 'p': p_vs})

        # Map block names to their colours (rate=blue, tvar=red).
        block_colours = [BLOCK_COLOURS.get(b, '#888') for b in order]
        _clustered_bar_panel(
            ax_block,
            block_labels_pretty,
            only_arrays, drop_arrays,
            only_side_label='ONLY this block',
            paired_side_label='DROP this block',
            group_colours=block_colours,
            hatch_paired=True,
            full_per_mouse=full_per_mouse,
            null_per_mouse=null_per_mouse,
            brackets=brackets,
            vs_full_pvalues=vs_full,
        )
        ax_block.set_title(
            f"Block-level ablation  |  {regressor_label}  |  CV: {cv_scheme}",
            fontsize=11, fontweight='bold', loc='left')
        ax_block.set_ylabel('CV R² (per-mouse)')
        ax_block.set_xlabel('')
        ax_block.legend(frameon=False, fontsize=8, loc='best')

    # ------------------------------------------------------------------
    # Metric panel — top-k by unique contribution
    # ------------------------------------------------------------------
    if not metric_df.empty:
        metric_df['kind'] = metric_df['model'].str.split(':').str[0]
        ko_rows = metric_df[metric_df['kind'] == 'ko']
        if not ko_rows.empty:
            full_mean = float(np.nanmean(full_per_mouse))
            order_m = (ko_rows.assign(unique=lambda d: full_mean - d['r2'])
                              .sort_values('unique', ascending=False)
                              ['metric'].tolist())
        else:
            order_m = sorted(metric_df['metric'].unique())
        order_m = order_m[:top_k_metrics]

        label_map = (metric_df.dropna(subset=['metric_label'])
                              .groupby('metric')['metric_label']
                              .first().to_dict())

        only_arrays = []
        ko_arrays = []
        names = []
        for m in order_m:
            om = metric_df[(metric_df['kind'] == 'only')
                            & (metric_df['metric'] == m)]
            km = metric_df[(metric_df['kind'] == 'ko')
                            & (metric_df['metric'] == m)]
            only_arrays.append(_arr(om.iloc[0]['r2_per_mouse'])
                                if not om.empty else np.array([]))
            ko_arrays.append(_arr(km.iloc[0]['r2_per_mouse'])
                              if not km.empty else np.array([]))
            names.append(label_map.get(m, m))

        # Planned comparison per metric: full vs ko:metric.
        vs_full_m = []
        for i, m in enumerate(order_m):
            p_vs = _paired_t_yhat(
                df_all, 'full', f"ko:{m}", cv_scheme)
            vs_full_m.append({'side': 'paired', 'i': i, 'p': p_vs})

        # Single uniform colour per metric — distinction comes from
        # cluster (only vs ko) and the metric label, not per-metric hue.
        metric_colours = [OKABE_ITO['blue']] * len(names)
        _clustered_bar_panel(
            ax_metric, names, only_arrays, ko_arrays,
            only_side_label='ONLY this metric',
            paired_side_label='KO this metric',
            group_colours=metric_colours,
            hatch_paired=True,
            full_per_mouse=full_per_mouse,
            null_per_mouse=null_per_mouse,
            brackets=None,
            vs_full_pvalues=vs_full_m,
            rotate_labels=25,
        )
        ax_metric.set_title(
            f"Per-metric ablation (top {len(order_m)} by unique R²)"
            f"  |  {regressor_label}  |  CV: {cv_scheme}",
            fontsize=11, fontweight='bold', loc='left')
        ax_metric.set_ylabel('CV R² (per-mouse)')
        ax_metric.set_xlabel('')
        ax_metric.legend(frameon=False, fontsize=8, loc='best')

    win = meta.get('window') if meta else None
    suptitle = (
        f"{target_label}  —  population features → target  ({regressor_label})"
        + (f"  |  window: {win}" if win else '')
        + "\nBars: mean ± SEM across n=6 mice.  Dots: individual mice;  "
          "lines connect only pairs of bars being tested (within-cluster).  "
          "Brackets and 'vs full' labels: paired t-test on per-mouse R² "
          "(scipy.stats.ttest_rel, df = 5). "
          "* p<.05, ** p<.01, *** p<.001."
    )
    fig.suptitle(suptitle, fontsize=10, fontweight='bold')

    if output_path:
        fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        return output_path
    return fig


def _rows_from_results_entry(entry):
    """``all_results[key]`` may be either a flat list of rows (legacy)
    or a dict ``{'rows': [...], 'meta': {...}}``. Normalise to rows."""
    if isinstance(entry, dict) and 'rows' in entry:
        return entry['rows']
    return entry


def _meta_from_results_entry(entry):
    if isinstance(entry, dict) and 'meta' in entry:
        return entry['meta']
    return None


def plot_unique_r2_heatmap(
    all_results,
    regressor,
    window=None,
    cv_scheme='kfold',
    output_path=None,
):
    """Heatmap of per-mouse-averaged unique R² across (target × block).

    Cell (i, j) = mean across mice of (full R² − drop:block_j R²) for
    target_i. ``window`` filters the all_results dict to a single window
    (Full or Late). With windows pooled together (window=None), pre-2026
    behaviour is preserved for backward compat.
    """
    set_plot_style()

    pretty_targets = {ucol: lab for ucol, lab, _ in TARGETS}

    cells = []
    for key, entry in all_results.items():
        if len(key) == 3:
            ucol, reg, win = key
            if window is not None and win != window:
                continue
        else:
            ucol, reg = key
        if reg != regressor:
            continue
        rows = _rows_from_results_entry(entry)
        rdf = pd.DataFrame(rows)
        rdf = rdf[rdf['cv'] == cv_scheme]
        full = rdf[rdf['model'] == 'full']
        if full.empty:
            continue
        full_per_mouse = _arr(full.iloc[0]['r2_per_mouse'])
        full_mean = float(np.nanmean(full_per_mouse))
        drop = rdf[(rdf['ablation_type'] == 'block')
                   & (rdf['model'].str.startswith('drop:'))]
        if drop.empty:
            continue
        row = {'target': ucol, 'full_mean_r2': full_mean,
               'target_label': pretty_targets.get(ucol, ucol)}
        for _, r in drop.iterrows():
            drop_per_mouse = _arr(r['r2_per_mouse'])
            unique_per_mouse = full_per_mouse - drop_per_mouse
            row[r['block']] = float(np.nanmean(unique_per_mouse))
        cells.append(row)

    if not cells:
        return None

    heat_df = pd.DataFrame(cells)
    block_cols = [b for b in BLOCKS.keys() if b in heat_df.columns]
    heat = heat_df[block_cols].copy()

    target_labels = [
        f"{r['target_label']}\n(full R² = {r['full_mean_r2']:+.3f})"
        for _, r in heat_df.iterrows()
    ]
    block_labels = [BLOCK_LABELS.get(b, b) for b in block_cols]
    heat.index = target_labels
    heat.columns = block_labels

    # Spacious figure: 2.6 in per block column + ~5 in per target row.
    fig_w = max(8, 2.6 * heat.shape[1] + 4.5)
    fig_h = max(5, 0.95 * heat.shape[0] + 2.2)
    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    vmax = float(np.nanmax(np.abs(heat.values))) if heat.size else 0.05
    vmax = max(vmax, 0.005)
    sns.heatmap(
        heat, annot=True, fmt='+.3f', cmap='RdBu_r',
        center=0.0, vmin=-vmax, vmax=vmax, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'label': 'Unique R² (mean across mice)',
                  'shrink': 0.7, 'pad': 0.02},
        annot_kws={'fontsize': 11, 'fontweight': 'bold'},
    )
    win_str = f"  |  window: {window}" if window else ""
    ax.set_title(
        f"Unique block contributions  |  {regressor.upper()}{win_str}  |  CV: {cv_scheme}\n"
        "Cell = mean across mice of (full R² − drop:block R²)\n"
        "Higher = the block carries information nothing else can substitute for.",
        fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=10, rotation=0)
    ax.tick_params(axis='y', labelsize=10, rotation=0)

    if output_path:
        fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        return output_path
    return fig


# ======================================================================
# Entry point
# ======================================================================

def _parse_cli():
    """Minimal CLI for the production run.

    Examples
    --------
      # Default: Ridge + GPR, kfold only (skips LOMO — much faster)
      python feature_ablation_analysis.py

      # Add LOMO too:
      python feature_ablation_analysis.py --cv kfold lomo

      # Ridge only:
      python feature_ablation_analysis.py --regressors ridge

      # Reuse already-extracted features (skip the pipeline step):
      python feature_ablation_analysis.py --features-csv \\
          feature_ablation_results/features.csv
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Feature-ablation CV regression on V1 population metrics.")
    p.add_argument('--regressors', nargs='+', default=['ridge', 'gpr'],
                    choices=['ridge', 'gpr'],
                    help="Which regressors to run (default: both)")
    p.add_argument('--cv', nargs='+', default=['kfold'],
                    choices=['kfold', 'lomo'],
                    help="Which CV schemes to run (default: kfold only — "
                         "skips slow LOMO)")
    p.add_argument('--output-dir', default='figures/feature_ablation_results',
                    help="Where to write CSVs and plots")
    p.add_argument('--features-csv',
                    help="Reuse a previously-extracted features.csv "
                         "(skips the per-mouse feature pipeline)")
    p.add_argument('--mice', nargs='+', type=int,
                    default=[0, 1, 2, 3, 4, 5],
                    help="Mouse IDs (default: 0..5)")
    p.add_argument('--gpr-restarts', type=int, default=2,
                    help="GPR n_restarts_optimizer (default: 2)")
    return p.parse_args()


def main(regressors=('ridge', 'gpr'), cv_schemes=('kfold',),
         windows=('Full', 'Late'),
         output_dir='feature_ablation_results',
         features_csv=None, mice=tuple(range(6)),
         gpr_restarts=2, save_plots=True):
    """Run the full ablation pipeline end-to-end.

    Each window is run as a separate analysis (Full = 0–2 s, Late =
    1–2 s) so the two are not pooled and can be compared. Output files
    and plots include the window in the filename.

    Designed to be called from Spyder / IPython directly — no CLI args
    required.  ::

        from feature_ablation_analysis import main
        results = main()                                    # ridge+gpr, kfold, both windows
        results = main(regressors=('ridge',))               # ridge only
        results = main(cv_schemes=('kfold', 'lomo'))        # add LOMO
        results = main(windows=('Late',))                   # late window only
        results = main(features_csv='feature_ablation_results/features.csv')
                                                            # reuse features
    """
    os.makedirs(output_dir, exist_ok=True)

    if features_csv:
        print(f'[1/3] Loading existing features from {features_csv}...')
        df = pd.read_csv(features_csv)
    else:
        from population_metrics_vs_uncertainty import (
            run_population_variance_pipeline,
        )
        print(f'[1/3] Extracting per-trial features for mice {mice}...')
        df, _ = run_population_variance_pipeline(list(mice))
        if df.empty:
            raise RuntimeError('No data extracted from the feature pipeline.')
        df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)

    df = add_stimulus_and_behaviour_columns(df)

    print(f'[2/3] Running ablations: regressors={tuple(regressors)} '
          f'cv={tuple(cv_schemes)} windows={tuple(windows)}...')
    all_results = run_all_targets_all_regressors(
        df,
        regressors=tuple(regressors),
        cv_schemes=tuple(cv_schemes),
        output_dir=output_dir,
        windows=tuple(windows),
        regressor_kwargs={
            'ridge': {'alpha': 1.0},
            'gpr': {'n_restarts': gpr_restarts},
        },
    )

    if save_plots:
        print('[3/3] Plotting...')
        for (ucol, regressor, window), entry in all_results.items():
            rows = _rows_from_results_entry(entry)
            meta = _meta_from_results_entry(entry)
            ulabel = next((lab for c, lab, _ in TARGETS if c == ucol), ucol)
            for cv_scheme in cv_schemes:
                out_path = os.path.join(
                    output_dir,
                    f'musall_{ucol}_{regressor}_{window}_{cv_scheme}.svg',
                )
                plot_musall_style(rows, ulabel,
                                   regressor_label=regressor.upper(),
                                   cv_scheme=cv_scheme,
                                   output_path=out_path,
                                   meta=meta)

        for regressor in regressors:
            for window in windows:
                for cv_scheme in cv_schemes:
                    out_path = os.path.join(
                        output_dir,
                        f'unique_r2_heatmap_{regressor}_{window}_{cv_scheme}.svg',
                    )
                    plot_unique_r2_heatmap(
                        all_results, regressor=regressor,
                        window=window,
                        cv_scheme=cv_scheme, output_path=out_path)
        print(f'Done. Results in {output_dir}/')
    return all_results


if __name__ == '__main__':
    args = _parse_cli()
    main(
        regressors=tuple(args.regressors),
        cv_schemes=tuple(args.cv),
        output_dir=args.output_dir,
        features_csv=args.features_csv,
        mice=tuple(args.mice),
        gpr_restarts=args.gpr_restarts,
    )
