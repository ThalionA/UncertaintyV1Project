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

ORDER_SENSITIVE_BLOCK = [
    'Traj_Length',
]

BLOCKS = {
    'rate': RATE_BLOCK,
    'temporal_variance': TEMPORAL_VARIANCE_BLOCK,
    'order_sensitive': ORDER_SENSITIVE_BLOCK,
}

BLOCK_LABELS = {
    'rate': 'Rate',
    'temporal_variance': 'Temporal variance',
    'order_sensitive': 'Order-sensitive (Traj_Length)',
}

BLOCK_COLOURS = {
    'rate': COLOR_RATE,
    'temporal_variance': COLOR_TVAR,
    'order_sensitive': COLOR_ORDER,
}


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

    # Full feature column set: METRIC_BASES order × windows.
    full_cols = [f"{base}_{w}"
                 for base, _ in METRIC_BASES
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

    # --- Per-metric ablations ---
    if include_per_metric:
        for base, label in METRIC_BASES:
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
):
    """Loop over (target, regressor) combos. Writes one CSV per combo
    and returns a dict of {(target, regressor): rows}."""
    if regressor_kwargs is None:
        regressor_kwargs = {'ridge': {'alpha': 1.0},
                            'gpr': {'n_restarts': 2}}

    os.makedirs(output_dir, exist_ok=True)
    df = add_stimulus_and_behaviour_columns(df)

    all_results = {}
    for ucol, ulabel, transform in targets:
        if ucol not in df.columns or df[ucol].dropna().empty:
            print(f"  [skip] target {ucol} not in df")
            continue
        for regressor in regressors:
            print(f"  fitting {regressor} on {ucol} ({ulabel})...")
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
                return_meta=True,
            )
            all_results[(ucol, regressor)] = {'rows': rows, 'meta': meta}

            # CSV: drop y_hat (large) so re-loadable CSVs stay light.
            # y_hat is preserved in the in-memory dict for trial-level
            # stats. To round-trip via disk, save the NPZ companion file.
            csv_rows = [{k: v for k, v in r.items() if k != 'y_hat'}
                         for r in rows]
            out_csv = os.path.join(
                output_dir, f"ablation_{ucol}_{regressor}.csv")
            pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
            out_npz = os.path.join(
                output_dir, f"ablation_{ucol}_{regressor}_yhat.npz")
            yhat_dict = {f"{r['model']}__{r['cv']}": np.asarray(r['y_hat'])
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


def _paired_bar_panel(
    ax, group_names, only_arrays, paired_arrays,
    only_label='only', paired_label='drop',
    only_colour=COLOR_ONLY, paired_colour=COLOR_DROP,
    full_per_mouse=None, null_per_mouse=None,
    only_stats=None, paired_stats=None,
    max_brackets=None,
    rotate_labels=0,
):
    """Draw paired (only, drop|ko) bars per group, with per-mouse points
    and connecting lines, plus reference bands for full / null.

    Cross-group comparisons are drawn via Friedman + pairwise Wilcoxon
    (Holm-corrected) results in ``only_stats`` and ``paired_stats``,
    produced by ``_friedman_pairwise``.
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

    # Per-mouse dots
    n_mice = max((len(a) for a in only_arrays + paired_arrays), default=0)
    for m_idx in range(n_mice):
        for i in range(n_groups):
            ov = only_arrays[i][m_idx] if m_idx < len(only_arrays[i]) else np.nan
            pv = paired_arrays[i][m_idx] if m_idx < len(paired_arrays[i]) else np.nan
            jitter = float(rng.uniform(-0.04, 0.04))
            if not np.isnan(ov):
                ax.scatter(x[i] - width/2 + jitter, ov,
                           color=COLOR_MOUSE_DOT, s=10, alpha=0.6,
                           linewidths=0, zorder=4)
            if not np.isnan(pv):
                ax.scatter(x[i] + width/2 + jitter, pv,
                           color=COLOR_MOUSE_DOT, s=10, alpha=0.6,
                           linewidths=0, zorder=4)

    # Reference bands
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

    # Cross-group significance brackets — only across only-bars and across
    # paired-bars separately. Vertical positioning: only brackets stack
    # closer to the bars; paired brackets stack above them.
    all_vals = np.concatenate([a for a in only_arrays + paired_arrays
                                if len(a) > 0])
    if all_vals.size:
        y_top = float(np.nanmax(all_vals))
        y_min = min(0.0, float(np.nanmin(all_vals)))
        y_range = max(y_top - y_min, 0.05)
        only_x = x - width/2
        paired_x = x + width/2

        # Determine tier offsets so paired brackets sit above only brackets
        n_only_brackets = (
            min(max_brackets or len(only_stats.get('pairwise', [])),
                len(only_stats.get('pairwise', [])))
            if only_stats else 0
        )

        if only_stats:
            _draw_cross_brackets(
                ax, only_stats['pairwise'], only_x, y_top, y_range,
                colour=only_colour, max_brackets=max_brackets,
                tier_offset=0.0,
            )
        if paired_stats:
            _draw_cross_brackets(
                ax, paired_stats['pairwise'], paired_x, y_top, y_range,
                colour=paired_colour, max_brackets=max_brackets,
                tier_offset=0.085 * max(n_only_brackets, 0) + 0.02,
            )

    return ax


def plot_musall_style(
    rows, target_label, regressor_label,
    cv_scheme='kfold', output_path=None, top_k_metrics=8,
    meta=None,
):
    """Musall/Churchland-style figure with per-mouse paired bars.

    Top panel: block-level only/drop bars (Mate's contrast).
    Bottom panel: top-k metrics by unique contribution, only/ko bars.

    Bar = mean R² across mice ± SEM; dots = individual mouse R²;
    reference lines for the full model and the null shuffled-target
    baseline. Panel header shows Friedman omnibus (per-mouse) across
    groups. Brackets show **trial-level cluster-robust paired t-tests**
    (Bonferroni-adjusted across the pairs displayed) when ``meta`` is
    provided — this is the more powerful per-pair test that uses every
    trial as an observation with the SE clustered by mouse. Without
    ``meta``, brackets fall back to per-mouse Wilcoxon + Holm.
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
        # Order blocks by unique contribution
        drop_rows = block_df[block_df['kind'] == 'drop']
        if not drop_rows.empty:
            order = (drop_rows.assign(unique=lambda d:
                                       np.nanmean(full_per_mouse)
                                       - d['r2'])
                              .sort_values('unique', ascending=False)
                              ['block'].tolist())
        else:
            order = sorted(block_df['block'].unique())

        only_arrays = []
        drop_arrays = []
        for b in order:
            ob = block_df[(block_df['kind'] == 'only')
                          & (block_df['block'] == b)]
            db = block_df[(block_df['kind'] == 'drop')
                          & (block_df['block'] == b)]
            only_arrays.append(_arr(ob.iloc[0]['r2_per_mouse'])
                                if not ob.empty else np.array([]))
            drop_arrays.append(_arr(db.iloc[0]['r2_per_mouse'])
                                if not db.empty else np.array([]))

        # Per-mouse omnibus (Friedman). Pairwise: trial-level cluster-
        # robust + Bonferroni if y_hat + meta available; else fall back
        # to per-mouse Wilcoxon + Holm.
        block_labels_pretty = [BLOCK_LABELS.get(b, b) for b in order]
        only_stats_block = _friedman_pairwise(only_arrays, block_labels_pretty)
        drop_stats_block = _friedman_pairwise(drop_arrays, block_labels_pretty)

        if meta is not None and 'y_true' in meta:
            y_true_arr = np.asarray(meta['y_true'], dtype=float)
            mids_trial = np.asarray(meta['mouse_ids_per_trial'])
            # Collect y_hat per block from the matching rows
            def _collect_yhat(kind):
                arrs = []
                for b in order:
                    sub = block_df[(block_df['kind'] == kind)
                                   & (block_df['block'] == b)
                                   & (block_df['cv'] == cv_scheme)]
                    if sub.empty or 'y_hat' not in sub.columns:
                        return None
                    arrs.append(_arr(sub.iloc[0]['y_hat']))
                return arrs
            only_yhat = _collect_yhat('only')
            drop_yhat = _collect_yhat('drop')
            if only_yhat is not None:
                trial_only = _trial_level_pairwise(
                    only_yhat, y_true_arr, mids_trial,
                    block_labels_pretty, correction='bonferroni')
                # Override the per-mouse pairwise with the trial-level one
                only_stats_block['pairwise'] = trial_only['pairwise']
                only_stats_block['trial_level'] = True
            if drop_yhat is not None:
                trial_drop = _trial_level_pairwise(
                    drop_yhat, y_true_arr, mids_trial,
                    block_labels_pretty, correction='bonferroni')
                drop_stats_block['pairwise'] = trial_drop['pairwise']
                drop_stats_block['trial_level'] = True

        _paired_bar_panel(
            ax_block,
            block_labels_pretty,
            only_arrays, drop_arrays,
            only_label='only:block', paired_label='drop:block',
            only_colour=COLOR_ONLY, paired_colour=COLOR_DROP,
            full_per_mouse=full_per_mouse,
            null_per_mouse=null_per_mouse,
            only_stats=only_stats_block,
            paired_stats=drop_stats_block,
        )
        # Title — embed the two omnibus results
        only_blurb = _omnibus_blurb(only_stats_block, group_name='only:block')
        drop_blurb = _omnibus_blurb(drop_stats_block, group_name='drop:block')
        omnibus_line = (only_blurb + '   |   ' + drop_blurb
                        if only_blurb and drop_blurb
                        else only_blurb or drop_blurb)
        ax_block.set_title(
            f"Block-level ablation  |  {regressor_label}  |  CV: {cv_scheme}"
            + (f"\n{omnibus_line}" if omnibus_line else ''),
            fontsize=10, fontweight='bold', loc='left')
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

        # Cross-metric omnibus + pairwise. With k=top_k metrics, C(k,2)
        # comparisons; cap displayed brackets to the most-significant 4
        # to keep the plot readable. Same trial-level upgrade as the
        # block panel when meta is provided.
        only_stats_m = _friedman_pairwise(only_arrays, names)
        ko_stats_m = _friedman_pairwise(ko_arrays, names)

        if meta is not None and 'y_true' in meta:
            y_true_arr = np.asarray(meta['y_true'], dtype=float)
            mids_trial = np.asarray(meta['mouse_ids_per_trial'])
            def _collect_metric_yhat(kind):
                arrs = []
                for m in order_m:
                    sub = metric_df[(metric_df['kind'] == kind)
                                    & (metric_df['metric'] == m)
                                    & (metric_df['cv'] == cv_scheme)]
                    if sub.empty or 'y_hat' not in sub.columns:
                        return None
                    arrs.append(_arr(sub.iloc[0]['y_hat']))
                return arrs
            only_yhat_m = _collect_metric_yhat('only')
            ko_yhat_m = _collect_metric_yhat('ko')
            if only_yhat_m is not None:
                trial_only_m = _trial_level_pairwise(
                    only_yhat_m, y_true_arr, mids_trial, names,
                    correction='bonferroni')
                only_stats_m['pairwise'] = trial_only_m['pairwise']
                only_stats_m['trial_level'] = True
            if ko_yhat_m is not None:
                trial_ko_m = _trial_level_pairwise(
                    ko_yhat_m, y_true_arr, mids_trial, names,
                    correction='bonferroni')
                ko_stats_m['pairwise'] = trial_ko_m['pairwise']
                ko_stats_m['trial_level'] = True
        _paired_bar_panel(
            ax_metric, names, only_arrays, ko_arrays,
            only_label='only:metric', paired_label='ko:metric',
            only_colour=COLOR_ONLY, paired_colour=COLOR_DROP,
            full_per_mouse=full_per_mouse,
            null_per_mouse=null_per_mouse,
            only_stats=only_stats_m,
            paired_stats=ko_stats_m,
            max_brackets=4,
            rotate_labels=25,
        )
        only_blurb = _omnibus_blurb(only_stats_m, group_name='only:metric')
        ko_blurb = _omnibus_blurb(ko_stats_m, group_name='ko:metric')
        omnibus_line = (only_blurb + '   |   ' + ko_blurb
                        if only_blurb and ko_blurb
                        else only_blurb or ko_blurb)
        ax_metric.set_title(
            f"Per-metric ablation (top {len(order_m)} by unique R²)"
            f"  |  {regressor_label}  |  CV: {cv_scheme}"
            + (f"\n{omnibus_line}" if omnibus_line else ''),
            fontsize=10, fontweight='bold', loc='left')
        ax_metric.set_ylabel('CV R² (per-mouse)')
        ax_metric.set_xlabel('')
        ax_metric.legend(frameon=False, fontsize=8, loc='best')

    bracket_test = ("cluster-robust paired t-test (Bonferroni)"
                     if meta is not None else
                     "Wilcoxon signed-rank (Holm)")
    fig.suptitle(
        f"{target_label}  —  population features → target  ({regressor_label})\n"
        "Bars: mean ± SEM across n=6 mice.  Dots: individual mice.  "
        f"Brackets across only/drop bars: trial-level {bracket_test};  "
        "* p<.05, ** p<.01, *** p<.001.  "
        "Panel headers: per-mouse Friedman omnibus across groups.",
        fontsize=10, fontweight='bold')

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
    cv_scheme='kfold',
    output_path=None,
):
    """Heatmap of per-mouse-averaged unique R² across (target × block).

    Cell (i, j) = full_mean_r2(target_i) - drop:block_j_mean_r2(target_i),
    averaged across mice. Labels shown clearly with adequate figure width.
    """
    set_plot_style()

    pretty_targets = {ucol: lab for ucol, lab, _ in TARGETS}

    cells = []
    for (ucol, reg), entry in all_results.items():
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
    ax.set_title(
        f"Unique block contributions  |  {regressor.upper()}  |  CV: {cv_scheme}\n"
        "Cell = mean across mice of (full R² - drop:block R²)\n"
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
    p.add_argument('--output-dir', default='feature_ablation_results',
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
         output_dir='feature_ablation_results',
         features_csv=None, mice=tuple(range(6)),
         gpr_restarts=2, save_plots=True):
    """Run the full ablation pipeline end-to-end.

    Designed to be called from Spyder / IPython directly — no CLI args
    required.  ::

        from feature_ablation_analysis import main
        results = main()                                  # ridge+gpr, kfold
        results = main(regressors=('ridge',))             # ridge only
        results = main(cv_schemes=('kfold', 'lomo'))      # add LOMO
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
          f'cv={tuple(cv_schemes)}...')
    all_results = run_all_targets_all_regressors(
        df,
        regressors=tuple(regressors),
        cv_schemes=tuple(cv_schemes),
        output_dir=output_dir,
        regressor_kwargs={
            'ridge': {'alpha': 1.0},
            'gpr': {'n_restarts': gpr_restarts},
        },
    )

    if save_plots:
        print('[3/3] Plotting...')
        for (ucol, regressor), entry in all_results.items():
            rows = _rows_from_results_entry(entry)
            meta = _meta_from_results_entry(entry)
            ulabel = next((lab for c, lab, _ in TARGETS if c == ucol), ucol)
            for cv_scheme in cv_schemes:
                out_path = os.path.join(
                    output_dir,
                    f'musall_{ucol}_{regressor}_{cv_scheme}.svg',
                )
                plot_musall_style(rows, ulabel,
                                   regressor_label=regressor.upper(),
                                   cv_scheme=cv_scheme,
                                   output_path=out_path,
                                   meta=meta)

        for regressor in regressors:
            for cv_scheme in cv_schemes:
                out_path = os.path.join(
                    output_dir,
                    f'unique_r2_heatmap_{regressor}_{cv_scheme}.svg',
                )
                plot_unique_r2_heatmap(
                    all_results, regressor=regressor,
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
