# -*- coding: utf-8 -*-
"""
Population Metrics vs IO Uncertainty.

Per-trial neural-population metrics correlated against per-trial IO
uncertainty targets, with stimulus-controlled (partial / residualised)
follow-ups and a cross-validated regression that asks how much basic
features explain on their own.

Per-trial metrics:

  Raw spike counts (rate semantics required)
    - Pop_Mean_Raw          : mean spike count over (time, neurons), per trial
    - Temporal_Fano_Raw     : per-neuron var_t / mean_t over the trial's t_bins,
                              averaged across neurons. Computed on raw counts so
                              the "Fano" label is honest.

  Per-neuron z-scored (population-state geometry)
    - Pop_Deviation         : per-trial mean of z-scored activity. ~0 in
                              expectation across trials by construction; per
                              trial captures whether this trial sat above or
                              below the per-neuron grand mean.
    - Spatial_Var           : variance across neurons of the trial-mean activity
    - Temporal_Var          : mean across neurons of within-trial temporal var
    - Trajectory_Length     : sum of step norms in fixed-k PCA subspace
    - GV                    : log-det of trial covariance in fixed-k PCA space
                              (regularised by reg=1e-4)
    - Centroid_Magnitude    : L2 norm of trial centroid in fixed-k PCA space
                              ("where the trajectory sits")
    - GV_Fano               : GV − log(Centroid_Magnitude + ε) — Fano-spirited
                              "spread per unit displacement"
    - Participation_Ratio   : (Σλ)² / Σλ²  of the trial neuron-mode covariance
    - Top_Mode_Dominance    : λ_max / Σλ   of the trial neuron-mode covariance

Across-trial Fano-factor / dispersion analyses live in the dedicated
`nn_decoder/plot_fano_factor.py` script (this file used to host them but
they were factored out for clarity).

Headline analyses:
    - Pearson + Spearman correlations with BH-FDR adjustment.
    - Partial correlations after regressing out (orientation, contrast,
      dispersion) and optionally (velocity, lick_rate). Per-mouse, then
      Fisher-z averaged.
    - Stimulus-residualised hexbin plots (visual companion to the partial
      correlations).
    - Cross-validated Ridge regression: features → uncertainty target. Full
      model + drop-one-feature-group ablations + single-group minima +
      single-best-neuron baseline. Reports both 5-fold and leave-one-mouse-out.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut

# `utils` pulls in torch via its sibling modules. Defer the import so this
# file can be imported (e.g. for unit tests of the pure analysis functions)
# without requiring torch.


# ==========================================
# 0. Plot Style
# ==========================================

PALETTE = {
    'primary': '#4361ee',
    'secondary': '#f72585',
    'accent': '#7209b7',
    'gray': '#888888',
}

def set_plot_style():
    sns.set_context("talk", font_scale=0.8)
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'sans-serif', 'axes.linewidth': 0.8,
        'axes.edgecolor': '#333', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5, 'grid.alpha': 0.7,
    })


# ==========================================
# 1. Per-Trial Metrics
# ==========================================

def entropy_np(p):
    p_safe = np.clip(p, 1e-10, 1.0)
    return -np.sum(p_safe * np.log(p_safe), axis=1)


def compute_raw_metrics(raw_tensor, mean_thresh=1e-3):
    """Per-trial metrics that need raw spike counts.

    Parameters
    ----------
    raw_tensor : (n_trials, t_bins, n_neurons) raw spike counts.

    Returns
    -------
    dict with:
      Pop_Mean_Raw       : mean spike count per trial.
      Temporal_Fano_Raw  : per-neuron var_t / mean_t over the trial's t_bins,
                           averaged over neurons whose trial-mean exceeds
                           ``mean_thresh``.
    """
    pop_mean = np.nanmean(raw_tensor, axis=(1, 2))

    per_neuron_mean = np.nanmean(raw_tensor, axis=1)   # (n_trials, n_neurons)
    per_neuron_var = np.nanvar(raw_tensor, axis=1)
    valid = per_neuron_mean > mean_thresh
    fano_per = np.where(valid,
                        per_neuron_var / np.maximum(per_neuron_mean, mean_thresh),
                        np.nan)
    temporal_fano = np.nanmean(fano_per, axis=1)

    return {
        'Pop_Mean_Raw': pop_mean,
        'Temporal_Fano_Raw': temporal_fano,
    }


def compute_zscored_metrics(z_tensor):
    """Per-trial population-state metrics on z-scored activity."""
    pop_dev = np.nanmean(z_tensor, axis=(1, 2))
    time_avg = np.nanmean(z_tensor, axis=1)
    spatial_var = np.nanvar(time_avg, axis=1)
    temporal_var = np.nanmean(np.nanvar(z_tensor, axis=1), axis=1)
    return {
        'Pop_Deviation': pop_dev,
        'Spatial_Var': spatial_var,
        'Temporal_Var': temporal_var,
    }


def compute_trajectory_metrics(z_tensor, fixed_k=5, reg=1e-4):
    """Trajectory length, Generalized Variance, centroid magnitude, and a
    "Fano-factorised" GV per trial.

    All four geometric measures are computed in the same fixed-k PCA
    subspace fit jointly on (trials × time) of the supplied window, so
    they are dimensionally comparable.

      - Trajectory_Length: sum of step norms in the fixed-k PCA subspace.
        (Used to be the full neuron space — moved into PCA-k for
        dimensional comparability with the other geometric metrics.)
      - GV: log-det of the trial covariance in PCA-k. Regularised by
        ``reg * I_k`` (default 1e-4 — bumped from 1e-6 to give more
        stability against near-singular per-trial covariances).
      - Centroid_Magnitude: L2 norm of the trial centroid in PCA-k.
        This is the "where the trajectory sits" companion to GV's
        "how spread out the trajectory is".
      - GV_Fano: GV − log(centroid_magnitude + eps). On a linear scale
        this is det(cov) / centroid_magnitude — a "spread per unit
        displacement" that adjusts for trials whose trajectory simply
        sits further from the per-neuron grand mean. (This is the
        Fano-spirited normalisation: variance / location, on log scale.)
    """
    n_trials, t_bins, n_neurons = z_tensor.shape

    k = min(fixed_k, n_neurons, t_bins)
    stacked = z_tensor.reshape(n_trials * t_bins, n_neurons)
    pca_k = PCA(n_components=k)
    projected = pca_k.fit_transform(stacked).reshape(n_trials, t_bins, k)
    var_expl = float(np.sum(pca_k.explained_variance_ratio_))

    # Trajectory length in PCA-k subspace (was full neuron space)
    diffs = np.diff(projected, axis=1)
    traj_len = np.sum(np.linalg.norm(diffs, axis=2), axis=1)

    gv = np.zeros(n_trials)
    centroid_mag = np.zeros(n_trials)
    for i in range(n_trials):
        cov_k = np.cov(projected[i], rowvar=False)
        _, gv[i] = np.linalg.slogdet(cov_k + reg * np.eye(k))
        centroid_mag[i] = float(np.linalg.norm(np.mean(projected[i], axis=0)))

    eps = 1e-8
    gv_fano = gv - np.log(centroid_mag + eps)
    return traj_len, gv, var_expl, centroid_mag, gv_fano


def compute_population_geometry(z_tensor):
    """Per-trial participation ratio and top-mode dominance.

    For each trial, eigenvalues of the (n_neurons x n_neurons) sample covariance
    over time bins (computed via SVD on the centered data for stability). Then:

        PR  = (Σλ)² / Σλ²        (effective dimensionality, ≤ min(t_bins, n_neurons))
        TMD = λ_max / Σλ         (dominance of the top mode, ∈ (0, 1])
    """
    n_trials, t_bins, n_neurons = z_tensor.shape
    pr = np.full(n_trials, np.nan)
    tmd = np.full(n_trials, np.nan)
    denom_t = max(t_bins - 1, 1)
    for i in range(n_trials):
        x = z_tensor[i]
        if np.any(np.isnan(x)):
            continue
        x_c = x - x.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(x_c, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        eigs = (s ** 2) / denom_t
        eigs = eigs[eigs > 0]
        if eigs.size == 0:
            continue
        pr[i] = (eigs.sum() ** 2) / np.sum(eigs ** 2)
        tmd[i] = eigs[0] / eigs.sum()
    return {'Participation_Ratio': pr, 'Top_Mode_Dominance': tmd}




# ==========================================
# 3. Pipeline
# ==========================================

GV_FIXED_K = 5  # committed 2026-05-19: t_bins=10, so k=5 keeps per-trial
                #   covariance well under the t_bins-1=9 rank ceiling.
                #   At k=10 the per-trial cov in PCA-k was rank-deficient and
                #   GV/GV_Fano were partly driven by the 1e-4 regulariser
                #   along the unspanned direction. See discussion 2026-05-19.
LIK_S_GRID = np.arange(0, 91, 1)


def _zscore_per_neuron(raw):
    """Per-neuron z-scoring across (trials, time)."""
    mean_act = np.nanmean(raw, axis=(0, 1), keepdims=True)
    std_act = np.nanstd(raw, axis=(0, 1), keepdims=True)
    std_act = np.where(std_act == 0, 1.0, std_act)
    return (raw - mean_act) / std_act


def run_population_variance_pipeline(mouse_ids):
    from utils import load_vr_export, calculate_np_variance  # noqa: F401

    all_results = []
    per_mouse_neuron_rates_late = {}

    for mouse_id in mouse_ids:
        print(f"Processing Mouse {mouse_id}...")
        try:
            (activities_m, targets_perc, targets_dec,
             targets_lik, trials) = load_vr_export(mouse_id)
        except Exception as e:
            print(f"  Skipping Mouse {mouse_id}: {e}")
            continue

        # load_vr_export returns (n_neurons, n_trials, t_bins);
        # downstream metrics expect (n_trials, t_bins, n_neurons)
        raw_full = np.transpose(activities_m, (1, 2, 0))
        n_trials, t_bins, n_neurons = raw_full.shape
        print(f"  Shape: {n_trials} trials x {t_bins} bins x {n_neurons} neurons")

        z_full = _zscore_per_neuron(raw_full)
        half = t_bins // 2
        raw_late = raw_full[:, half:, :]
        z_late = z_full[:, half:, :]
        per_mouse_neuron_rates_late[mouse_id] = np.nanmean(raw_late, axis=1)

        s_grid = LIK_S_GRID
        perceptual_variance = calculate_np_variance(targets_perc, s_grid)
        decision_entropy = entropy_np(targets_dec) / np.log(2.0)  # normalised to [0,1]
        if targets_lik is not None:
            likelihood_variance = calculate_np_variance(targets_lik, s_grid)
        else:
            likelihood_variance = np.full(n_trials, np.nan)
            print("  [warn] L_s_marginal not in export — Likelihood_Variance is NaN")

        rows = {
            'Mouse_ID': mouse_id,
            'Trial_Idx': np.arange(n_trials),
            'Contrast': trials['contrast'],
            'Dispersion': trials['dispersion'],
            'Orientation': trials['orientation'],
            'True_Orientation': trials.get('true_orientation',
                                          trials['orientation']),
            'Velocity': trials['velocity'],
            'Lick_Rate': trials['lick_rate'],
            'Choice': trials['choice'],
            'Perceptual_Variance': perceptual_variance,
            'Decision_Entropy': decision_entropy,
            'Likelihood_Variance': likelihood_variance,
        }
        for win_name, raw_w, z_w in [('Full', raw_full, z_full),
                                      ('Late', raw_late, z_late)]:
            raw_metrics = compute_raw_metrics(raw_w)
            z_metrics = compute_zscored_metrics(z_w)
            tl_w, gv_w, ve_w, centroid_w, gv_fano_w = compute_trajectory_metrics(
                z_w, fixed_k=GV_FIXED_K)
            geom = compute_population_geometry(z_w)
            print(f"  GV PCA k={GV_FIXED_K} ({win_name}): {ve_w:.1%} variance explained")
            for k, v in raw_metrics.items():
                rows[f"{k}_{win_name}"] = v
            for k, v in z_metrics.items():
                rows[f"{k}_{win_name}"] = v
            rows[f"Traj_Length_{win_name}"] = tl_w
            rows[f"GV_{win_name}"] = gv_w
            rows[f"Centroid_Magnitude_{win_name}"] = centroid_w
            rows[f"GV_Fano_{win_name}"] = gv_fano_w
            for k, v in geom.items():
                rows[f"{k}_{win_name}"] = v

        all_results.append(pd.DataFrame(rows))

    if not all_results:
        return pd.DataFrame(), {}
    return (pd.concat(all_results, ignore_index=True),
            per_mouse_neuron_rates_late)


# ==========================================
# 4. Statistics utilities
# ==========================================

UNCERTAINTY_TARGETS = [
    ('Perceptual_Variance', 'Perceptual Variance'),
    ('Likelihood_Variance', 'Likelihood Variance'),
    ('Decision_Entropy', 'Decision Entropy (normalised)'),
]

METRIC_BASES = [
    ('Pop_Mean_Raw', 'Pop Mean (Raw)'),
    ('Temporal_Fano_Raw', 'Temporal Fano (Raw)'),
    ('Pop_Deviation', 'Pop Deviation (z)'),
    ('Spatial_Var', 'Spatial Variance (z)'),
    ('Temporal_Var', 'Temporal Variance (z)'),
    ('Traj_Length', f'Trajectory Length k={GV_FIXED_K} (z)'),
    ('GV', f'Generalised Variance k={GV_FIXED_K} (z)'),
    ('Centroid_Magnitude', f'Centroid Magnitude k={GV_FIXED_K} (z)'),
    ('GV_Fano', f'GV Fano-norm k={GV_FIXED_K} (z)'),
    ('Participation_Ratio', 'Participation Ratio (z)'),
    ('Top_Mode_Dominance', 'Top-Mode Dominance (z)'),
]


def benjamini_hochberg(pvalues):
    """Benjamini–Hochberg FDR-adjusted q-values, same shape as input.

    NaN inputs propagate as NaN q-values.
    """
    p = np.asarray(pvalues, dtype=float)
    flat = p.flatten()
    out = np.full_like(flat, np.nan)
    valid = ~np.isnan(flat)
    if valid.sum() == 0:
        return out.reshape(p.shape)
    pv = flat[valid]
    n = pv.size
    order = np.argsort(pv)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_raw = pv * n / ranks
    sorted_q = q_raw[order]
    sorted_q_min = np.minimum.accumulate(sorted_q[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(sorted_q_min, 1.0)
    out[valid] = q
    return out.reshape(p.shape)


def pearson_spearman(x, y):
    """Returns (pearson_r, pearson_p, spearman_r, spearman_p) on the joint
    non-NaN subset. Returns NaNs if fewer than 3 valid points."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    rp, pp = stats.pearsonr(x[mask], y[mask])
    rs, ps = stats.spearmanr(x[mask], y[mask])
    return rp, pp, rs, ps


def _stimulus_design_matrix(df, columns=('Orientation', 'Contrast', 'Dispersion'),
                            extra=(), design='joint'):
    """One-hot encode stimulus categories + optional continuous covariates,
    plus an intercept column.

    Parameters
    ----------
    df : DataFrame
    columns : tuple of str
        Stimulus columns to encode.
    extra : tuple of str
        Continuous covariates (e.g. Velocity, Lick_Rate). Appended raw.
    design : {'joint', 'additive'}
        How to encode the stimulus controls.

        * ``'joint'`` (default since 2026-05-16): one-hot encode the *full
          tuple* of stimulus column values as a single categorical, so OLS
          residuals are within-(stim-cell) deviations. The principled
          control whenever any predictor in the comparison lives in the
          joint stim-cell structure (stim_mean baselines, decoder outputs
          that pass through per-cell means, or any metric whose
          correlation with the target plausibly flows through Ori×Con×Disp
          interactions).
        * ``'additive'``: legacy main-effects-only. One-hot each stimulus
          column independently; the implicit model is ``y ~ sum_i col_i``
          (no interactions). Residuals retain interaction structure. Kept
          for backward compatibility with figures produced before
          2026-05-16 and for cases where main-effects-only is
          *deliberately* the control.

    See ``documents/residual_partial_correlation.md`` for the rationale
    and ``nn_decoder/compare_partial_corr_designs.py`` for the empirical
    comparison that motivated switching the default.
    """
    if design not in ('joint', 'additive'):
        raise ValueError(
            f"design must be 'joint' or 'additive', got {design!r}"
        )

    parts = []
    if design == 'joint':
        present = [c for c in columns if c in df.columns]
        if present:
            joint = df[present[0]].astype(str).copy()
            for c in present[1:]:
                joint = joint + '|' + df[c].astype(str)
            joint = joint.astype('category')
            oh = pd.get_dummies(joint, prefix='cell',
                                drop_first=True).astype(float)
            if not oh.empty:
                parts.append(oh.values)
    else:  # additive
        for c in columns:
            if c not in df.columns:
                continue
            col = df[c].astype('category')
            oh = pd.get_dummies(col, prefix=c, drop_first=True).astype(float)
            if not oh.empty:
                parts.append(oh.values)

    for c in extra:
        if c in df.columns:
            parts.append(df[[c]].astype(float).values)
    if not parts:
        return np.ones((len(df), 1))
    X = np.column_stack(parts)
    return np.column_stack([np.ones(X.shape[0]), X])


def _residualize(y, Z):
    """OLS residuals of y on Z (Z already includes intercept)."""
    y = np.asarray(y, dtype=float)
    res = np.full_like(y, np.nan, dtype=float)
    mask = ~np.isnan(y) & ~np.any(np.isnan(Z), axis=1)
    if mask.sum() < Z.shape[1] + 1:
        return res
    beta, *_ = np.linalg.lstsq(Z[mask], y[mask], rcond=None)
    res[mask] = y[mask] - Z[mask] @ beta
    return res


def partial_correlation(df, x_col, y_col,
                        controls=('Orientation', 'Contrast', 'Dispersion'),
                        extra=(), per_mouse=True, design='joint'):
    """Partial Pearson correlation r(x, y | controls).

    With per_mouse=True we residualise within each mouse and pool with a
    Fisher-z weighted average (weights = n_i - 3). p-value is from the pooled
    z-statistic against N(0, 1/Σw).

    The ``design`` parameter selects the stimulus-control encoding; see
    `_stimulus_design_matrix` for the full discussion. Default changed from
    ``'additive'`` to ``'joint'`` on 2026-05-16 after the empirical
    comparison in `nn_decoder/compare_partial_corr_designs.py` showed
    main-effects residualisation was leaving substantial stim-cell
    interaction structure in the residuals (most notably for
    Temporal_Fano_Raw, where the partial r dropped from ~0.08 to ~0.01).
    Pass ``design='additive'`` to reproduce pre-2026-05-16 figures.

    Pearson is undefined on a residual series with zero variance; such
    mice are dropped from the pool. (Predictors that are constant within
    each stim cell — e.g. ``stim_mean_baseline`` decoded scalars under
    ``design='joint'`` — therefore correctly yield NaN partial r.)

    Returns a dict: {'r': pooled_r, 'p': pooled_p, 'n': total_n,
                     'per_mouse': [(mouse_id, r_i), ...]}.
    """
    if per_mouse and 'Mouse_ID' in df.columns:
        rs, ns, mids = [], [], []
        for mid, sub in df.groupby('Mouse_ID'):
            Z = _stimulus_design_matrix(sub, controls, extra, design=design)
            x_res = _residualize(sub[x_col].values, Z)
            y_res = _residualize(sub[y_col].values, Z)
            mask = ~(np.isnan(x_res) | np.isnan(y_res))
            if mask.sum() < 4:
                continue
            if (np.std(x_res[mask]) < 1e-12
                    or np.std(y_res[mask]) < 1e-12):
                continue
            r, _ = stats.pearsonr(x_res[mask], y_res[mask])
            rs.append(r)
            ns.append(int(mask.sum()))
            mids.append(mid)
        if not rs:
            return {'r': np.nan, 'p': np.nan, 'n': 0, 'per_mouse': []}
        zs = np.arctanh(np.clip(np.array(rs), -0.999, 0.999))
        ws = np.maximum(np.array(ns) - 3, 1)
        z_mean = np.sum(zs * ws) / np.sum(ws)
        r_pooled = np.tanh(z_mean)
        z_stat = z_mean * np.sqrt(np.sum(ws))
        p_pooled = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return {'r': float(r_pooled), 'p': float(p_pooled), 'n': int(np.sum(ns)),
                'per_mouse': list(zip(mids, rs))}
    else:
        Z = _stimulus_design_matrix(df, controls, extra, design=design)
        x_res = _residualize(df[x_col].values, Z)
        y_res = _residualize(df[y_col].values, Z)
        mask = ~(np.isnan(x_res) | np.isnan(y_res))
        if mask.sum() < 4:
            return {'r': np.nan, 'p': np.nan, 'n': 0, 'per_mouse': []}
        r, p = stats.pearsonr(x_res[mask], y_res[mask])
        return {'r': float(r), 'p': float(p), 'n': int(mask.sum()), 'per_mouse': []}


def _r2_score(y, y_hat):
    mask = ~(np.isnan(y) | np.isnan(y_hat))
    if mask.sum() < 2:
        return np.nan
    ss_res = np.sum((y[mask] - y_hat[mask]) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot


def cv_ridge_r2(X, y, groups, cv='kfold', alpha=1.0, n_splits=5, seed=0):
    """Cross-validated Ridge R². cv='kfold' or 'lomo' (leave-one-mouse-out).

    Features are z-scored within each training fold (StandardScaler.fit on
    train only). Returns (r2, y_hat_full, mask) — y_hat is OOF predictions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    Xv = X[mask]
    yv = y[mask]
    gv = groups[mask]
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

    y_hat_v = np.full(yv.shape, np.nan)
    for tr, te in split_iter:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Xv[tr])
        X_te = scaler.transform(Xv[te])
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_tr, yv[tr])
        y_hat_v[te] = model.predict(X_te)

    y_hat_full = np.full(y.shape, np.nan)
    y_hat_full[mask] = y_hat_v
    return _r2_score(y, y_hat_full), y_hat_full, mask


def cv_best_neuron_baseline_r2(per_trial_neuron_rates, y, n_splits=5, seed=0):
    """Per-fold: pick the best-correlated neuron on the train fold, fit a
    univariate OLS, predict the test fold. Honest CV R² for the
    "best single neuron's mean rate" lower bound. Per-session only — no
    cross-mouse pooling because neuron identity is session-specific."""
    X = np.asarray(per_trial_neuron_rates, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    if X.shape[0] < 10 or X.shape[1] == 0:
        return np.nan
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_hat = np.full(y.shape, np.nan)
    for tr, te in splitter.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        x_centered = Xtr - Xtr.mean(0, keepdims=True)
        y_centered = ytr - ytr.mean()
        denom = np.sqrt((x_centered ** 2).sum(0) * (y_centered ** 2).sum() + 1e-12)
        rs = (x_centered.T @ y_centered) / denom
        best = int(np.nanargmax(np.abs(rs)))
        x_tr = Xtr[:, best]
        if np.std(x_tr) < 1e-12:
            slope, intercept = 0.0, float(np.mean(ytr))
        else:
            slope, intercept = np.polyfit(x_tr, ytr, 1)
        y_hat[te] = slope * X[te, best] + intercept
    return _r2_score(y, y_hat)


# ==========================================
# 5. Visualisation
# ==========================================

def _annotate_corr(ax, x, y, color='#333'):
    rp, pp, rs, ps = pearson_spearman(np.asarray(x), np.asarray(y))
    sig = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else 'n.s.'
    ax.text(0.05, 0.95,
            f"Pearson r = {rp:+.3f} {sig}\nSpearman ρ = {rs:+.3f}",
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            color=color, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
    return rp, pp, rs, ps


def plot_pooled(df, uncertainty_col, uncertainty_label, suffix=''):
    set_plot_style()
    n_m = len(METRIC_BASES)
    fig, axes = plt.subplots(2, n_m, figsize=(3.6 * n_m, 7), constrained_layout=True)
    for row, (window, wlabel) in enumerate([('Full', '0–2 s'), ('Late', '1–2 s')]):
        for col, (base, label) in enumerate(METRIC_BASES):
            ax = axes[row, col]
            cn = f"{base}_{window}"
            if cn not in df.columns:
                ax.set_visible(False)
                continue
            clean = df.dropna(subset=[uncertainty_col, cn])
            if len(clean) < 5:
                ax.set_visible(False)
                continue
            x, y = clean[uncertainty_col].values, clean[cn].values
            ax.hexbin(x, y, gridsize=35, cmap='magma', mincnt=1,
                      linewidths=0.1, edgecolors='none')
            sns.regplot(x=x, y=y, scatter=False, ax=ax,
                        line_kws={'lw': 2, 'color': PALETTE['primary']})
            _annotate_corr(ax, x, y, color=PALETTE['primary'])
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight='bold', pad=6)
            ax.set_xlabel(uncertainty_label if row == 1 else '', fontsize=9)
            ax.set_ylabel(wlabel if col == 0 else '', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
    fig.suptitle(f"Neural Heuristics vs {uncertainty_label}",
                 fontsize=14, fontweight='bold', y=1.01)
    path = f"heuristics_pooled_{suffix}.svg"
    plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {path}")
    plt.close()


def plot_per_mouse(df, uncertainty_col, uncertainty_label, suffix=''):
    set_plot_style()
    mouse_ids = sorted(df['Mouse_ID'].unique())
    n_mice = len(mouse_ids)
    if n_mice == 0:
        return
    colors = sns.color_palette('Set2', n_mice)
    for base, label in METRIC_BASES:
        cn = f"{base}_Late"
        if cn not in df.columns:
            continue
        if df[cn].dropna().empty:
            continue
        fig, axes = plt.subplots(1, n_mice, figsize=(3.5 * n_mice, 3.6),
                                 constrained_layout=True, sharey=True)
        if n_mice == 1:
            axes = [axes]
        y_lo, y_hi = df[cn].quantile(0.005), df[cn].quantile(0.995)
        for i, m_id in enumerate(mouse_ids):
            ax = axes[i]
            m_df = df[df['Mouse_ID'] == m_id].dropna(subset=[uncertainty_col, cn])
            if m_df.empty:
                continue
            x, y = m_df[uncertainty_col].values, m_df[cn].values
            ax.scatter(x, y, s=8, alpha=0.25, color=colors[i],
                       edgecolors='none', rasterized=True)
            sns.regplot(x=x, y=y, scatter=False, ax=ax,
                        line_kws={'lw': 2, 'color': '#222'})
            _annotate_corr(ax, x, y, color='#222')
            ax.set_title(f"Mouse {m_id}", fontsize=10, fontweight='bold')
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(uncertainty_label if i == n_mice // 2 else '', fontsize=9)
            if i == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"{label}  (1–2 s)", fontsize=12, fontweight='bold', y=1.04)
        path = f"heuristics_per_mouse_{base}_{suffix}.svg"
        plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        print(f"  Saved {path}")
        plt.close()


def plot_partial_vs_raw(df, uncertainty_col, uncertainty_label, suffix='',
                        design='joint'):
    """Bar chart: raw Pearson r vs partial r | stimulus, vs partial r | stim+behaviour.

    ``design`` is forwarded to `partial_correlation` (default joint-cell
    since 2026-05-16). Pass ``design='additive'`` for the legacy
    main-effects-only control.
    """
    set_plot_style()
    rows = []
    for base, label in METRIC_BASES:
        cn = f"{base}_Late"
        if cn not in df.columns:
            continue
        clean = df.dropna(subset=[uncertainty_col, cn])
        if len(clean) < 5:
            continue
        rp, _, _, _ = pearson_spearman(clean[cn].values, clean[uncertainty_col].values)
        pc_stim = partial_correlation(df, cn, uncertainty_col,
                                       controls=('Orientation', 'Contrast', 'Dispersion'),
                                       per_mouse=True, design=design)
        pc_stim_beh = partial_correlation(df, cn, uncertainty_col,
                                           controls=('Orientation', 'Contrast', 'Dispersion'),
                                           extra=('Velocity', 'Lick_Rate'),
                                           per_mouse=True, design=design)
        rows.append({'metric': label,
                     'Raw Pearson': rp,
                     'Partial: stimulus': pc_stim['r'],
                     'Partial: stim + behaviour': pc_stim_beh['r']})
    if not rows:
        return None
    plot_df = pd.DataFrame(rows)
    plot_df_long = plot_df.melt(id_vars='metric',
                                value_vars=['Raw Pearson', 'Partial: stimulus',
                                            'Partial: stim + behaviour'],
                                var_name='kind', value_name='r')
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(plot_df)), 5),
                           constrained_layout=True)
    sns.barplot(data=plot_df_long, x='metric', y='r', hue='kind', ax=ax,
                palette=[PALETTE['gray'], PALETTE['primary'], PALETTE['secondary']])
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel(f"Correlation with {uncertainty_label}")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')
    ax.legend(title='', frameon=False)
    fig.suptitle(f"Raw vs Partial Correlation: Neural Metrics vs {uncertainty_label}",
                 fontsize=12, fontweight='bold')
    path = f"partial_vs_raw_{suffix}.svg"
    plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {path}")
    plt.close()
    return plot_df


def plot_residuals(df, uncertainty_col, uncertainty_label, suffix='',
                   design='joint'):
    """Hexbin of stimulus-residualised metric vs stimulus-residualised target.

    The residualisation uses the same ``design`` default as
    `partial_correlation` (joint-cell since 2026-05-16).
    """
    set_plot_style()
    metrics_with_data = [(b, lab) for (b, lab) in METRIC_BASES
                         if f"{b}_Late" in df.columns]
    n_m = len(metrics_with_data)
    if n_m == 0:
        return
    n_cols = min(3, n_m)
    n_rows = (n_m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    for ax, (base, label) in zip(axes, metrics_with_data):
        cn = f"{base}_Late"
        x_chunks, y_chunks = [], []
        for mid, sub in df.groupby('Mouse_ID'):
            Z = _stimulus_design_matrix(
                sub, ('Orientation', 'Contrast', 'Dispersion'),
                design=design,
            )
            x_chunks.append(_residualize(sub[cn].values, Z))
            y_chunks.append(_residualize(sub[uncertainty_col].values, Z))
        x_res = np.concatenate(x_chunks)
        y_res = np.concatenate(y_chunks)
        mask = ~(np.isnan(x_res) | np.isnan(y_res))
        if mask.sum() < 5:
            ax.set_visible(False)
            continue
        x = x_res[mask]
        y = y_res[mask]
        ax.hexbin(x, y, gridsize=35, cmap='mako', mincnt=1,
                  linewidths=0.1, edgecolors='none')
        sns.regplot(x=x, y=y, scatter=False, ax=ax,
                    line_kws={'lw': 2, 'color': PALETTE['secondary']})
        _annotate_corr(ax, x, y, color=PALETTE['secondary'])
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel(f"{label} residual", fontsize=9)
        ax.set_ylabel(f"{uncertainty_label} residual", fontsize=9)
        ax.tick_params(labelsize=8)
    for j in range(len(metrics_with_data), len(axes)):
        axes[j].axis('off')
    fig.suptitle(f"Stimulus-Residualised: Neural Metrics vs {uncertainty_label}",
                 fontsize=13, fontweight='bold', y=1.01)
    path = f"residuals_{suffix}.svg"
    plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {path}")
    plt.close()


def plot_cv_regression_bar(cv_results, target_label, suffix=''):
    """Per-metric contribution view.

    Two stacked panels (kfold above, lomo below). In each panel, bars are
    grouped by metric; within a group, two side-by-side bars show
    ``only:<metric>`` and ``ko:<metric>``. A dashed horizontal line marks
    the full-model R² in that CV scheme so the gap between any KO bar
    and the line is visually the metric's UNIQUE contribution. A dotted
    gray line marks the best-single-neuron baseline (kfold panel only —
    the baseline is computed per mouse on kfold).

    Reading guide:
      - "only" tall   → metric is informative on its own
      - "ko" close to the dashed full line → metric is REDUNDANT
        (the others can substitute)
      - "ko" well below the dashed full line → metric carries UNIQUE
        information that nothing else captures
    """
    set_plot_style()
    df = pd.DataFrame(cv_results)
    if df.empty:
        return

    # Split rows
    full_rows = df[df['model'] == 'full']
    baseline_rows = df[df['model'] == 'best_single_neuron']
    metric_mask = df['model'].astype(str).str.contains(':', regex=False)
    metric_df = df[metric_mask].copy()
    if metric_df.empty:
        return
    metric_df['kind'] = metric_df['model'].str.split(':').str[0]
    metric_df['metric_base'] = metric_df['model'].str.split(':').str[1]

    # Pretty labels (fall back to base if no metric_label column)
    if 'metric_label' in metric_df.columns:
        label_map = (metric_df.dropna(subset=['metric_label'])
                              .groupby('metric_base')['metric_label']
                              .first().to_dict())
    else:
        label_map = {}
    metric_df['metric_pretty'] = metric_df['metric_base'].map(
        lambda b: label_map.get(b, b))

    # Order metrics by 'only' kfold R², descending — most informative first
    only_kfold = metric_df[(metric_df['kind'] == 'only')
                           & (metric_df['cv'] == 'kfold')]
    if not only_kfold.empty:
        order = only_kfold.sort_values('r2', ascending=False)['metric_pretty'].tolist()
    else:
        order = sorted(metric_df['metric_pretty'].unique())
    metric_df['metric_pretty'] = pd.Categorical(
        metric_df['metric_pretty'], categories=order, ordered=True)
    metric_df = metric_df.sort_values(['cv', 'metric_pretty', 'kind'])

    cv_schemes = ['kfold', 'lomo']
    n_metrics = len(order)
    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.95 * n_metrics + 3), 8),
                             constrained_layout=True, sharex=True)

    for ax, cv in zip(axes, cv_schemes):
        sub = metric_df[metric_df['cv'] == cv]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=sub, x='metric_pretty', y='r2', hue='kind',
                    hue_order=['only', 'ko'],
                    palette=[PALETTE['primary'], PALETTE['secondary']],
                    ax=ax)
        # Horizontal reference: full model R² in this cv scheme
        full_r = full_rows[full_rows['cv'] == cv]
        if not full_r.empty:
            ax.axhline(full_r['r2'].iloc[0], color='black', ls='--', lw=1.5,
                       label=f"Full ({full_r['r2'].iloc[0]:+.3f})")
        # Best-single-neuron baseline (kfold panel only)
        if cv == 'kfold' and not baseline_rows.empty:
            b_r2 = float(baseline_rows['r2'].iloc[0])
            ax.axhline(b_r2, color=PALETTE['gray'], ls=':', lw=1.2,
                       label=f"Best single neuron ({b_r2:+.3f})")
        ax.axhline(0, color='#333', lw=0.6)
        ax.set_title(f"CV scheme: {cv}", fontsize=11, fontweight='bold',
                     loc='left')
        ax.set_ylabel("CV R²")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=35, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        ax.legend(title='', frameon=False, fontsize=8, loc='best')

    fig.suptitle(
        f"Metric contribution: features → {target_label}\n"
        "only = metric on its own  |  ko = full model minus this metric  "
        "(gap to dashed = unique contribution)",
        fontsize=11, fontweight='bold')
    path = f"cv_regression_{suffix}.svg"
    plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    print(f"  Saved {path}")
    plt.close()


# ==========================================
# 6. Headline analysis: CV regression with ablations
# ==========================================

FEATURE_GROUPS = {
    'rate':     ['Pop_Mean_Raw'],
    'fano':     ['Temporal_Fano_Raw'],
    'state':    ['Pop_Deviation', 'Spatial_Var', 'Temporal_Var'],
    'geometry': ['Traj_Length', 'GV', 'Centroid_Magnitude', 'GV_Fano',
                 'Participation_Ratio', 'Top_Mode_Dominance'],
}


def _expand_feature_names(group_keys, windows=('Full', 'Late')):
    return [f"{f}_{w}"
            for k in group_keys
            for f in FEATURE_GROUPS[k]
            for w in windows]


def _within_mouse_zscore(df, col):
    """z-score a target within each mouse so the across-mouse pooled R² is
    well-defined and not dominated by per-mouse mean shifts."""
    z = np.full(len(df), np.nan)
    for mid, sub in df.groupby('Mouse_ID'):
        idx = sub.index.values
        v = sub[col].values
        m = np.nanmean(v)
        s = np.nanstd(v)
        if s < 1e-12:
            z[idx] = 0.0
        else:
            z[idx] = (v - m) / s
    return z


def _within_mouse_zscore_features(df, cols, groups):
    """Within-mouse z-score each feature column.

    Returns a (n_rows, n_cols) numpy array where every (mouse, column)
    pair has mean 0 and std 1 (or 0 if std was zero). This makes feature
    distributions commensurate across mice before train/test splitting,
    which is essential for leave-one-mouse-out CV — without it the
    held-out mouse's features may sit at a different mean from the
    training pool and the Ridge predictions inherit that offset,
    pushing R² negative for trivial reasons.
    """
    X = df[list(cols)].values.astype(float).copy()
    groups = np.asarray(groups)
    for mid in np.unique(groups):
        mask = groups == mid
        if mask.sum() < 2:
            continue
        col_means = np.nanmean(X[mask], axis=0)
        col_stds = np.nanstd(X[mask], axis=0)
        col_stds = np.where(col_stds < 1e-12, 1.0, col_stds)
        X[mask] = (X[mask] - col_means) / col_stds
    return X


def run_cv_regression(df, ucol, ulabel,
                      per_mouse_neuron_rates_late=None,
                      alpha=1.0, n_splits=5, seed=0,
                      windows=('Full', 'Late')):
    """Per-metric CV regression: full model + per-metric ONLY + per-metric KO.

    For each metric in METRIC_BASES we evaluate two restricted models:

      - ``only:<metric>`` — Ridge fitted on just that metric's columns
        (one column per window). Tells us how much the metric explains
        on its own.
      - ``ko:<metric>`` — Ridge fitted on the full feature set MINUS that
        metric. Tells us how much the full model loses without it; the
        gap ``full_r2 − ko_r2`` is the metric's UNIQUE contribution
        (variance that no other metric can substitute for).

    Plus the standard reference points: full-model R², and the per-mouse
    best-single-neuron baseline (a lower bound).

    Both the target and the features are z-scored within each mouse
    before fitting (see ``_within_mouse_zscore`` and
    ``_within_mouse_zscore_features``). The within-mouse feature scaling
    is critical for leave-one-mouse-out CV: without it, the held-out
    mouse's features can sit at a different absolute scale from the
    training pool and the Ridge predictions inherit that offset, which
    blows up R² (often to negative values) for trivial reasons. With
    within-mouse z-scoring, both schemes test the within-mouse
    trial-by-trial structure on a common scale.

    Each row of the returned list is a dict with at least:
        model, cv, r2, n_features
    Rows for ``only:`` and ``ko:`` also have:
        metric         — base name (e.g. 'Pop_Mean_Raw')
        metric_label   — pretty label from METRIC_BASES
    Within-CV-scheme deltas (added at the end):
        delta_r2_vs_full     = r2 − full_r2  (always)
        unique_r2_vs_full    = full_r2 − r2  (KO rows only — metric's
                                              unique contribution)
    """
    target = _within_mouse_zscore(df, ucol)
    groups = df['Mouse_ID'].values

    label_lookup = dict(METRIC_BASES)

    # All columns the full model uses, in METRIC_BASES order × windows
    full_cols = [f"{base}_{w}"
                 for base, _ in METRIC_BASES
                 for w in windows
                 if f"{base}_{w}" in df.columns]

    rows = []

    # ----- Full model -----
    if full_cols:
        X = _within_mouse_zscore_features(df, full_cols, groups)
        for cv in ['kfold', 'lomo']:
            r2, _, _ = cv_ridge_r2(X, target, groups, cv=cv, alpha=alpha,
                                   n_splits=n_splits, seed=seed)
            rows.append({'model': 'full', 'cv': cv, 'r2': r2,
                         'n_features': X.shape[1]})

    # ----- Per-metric ONLY and KO -----
    for base, label in METRIC_BASES:
        metric_cols = [f"{base}_{w}" for w in windows
                       if f"{base}_{w}" in df.columns]
        if not metric_cols:
            continue

        # only:<metric> — just this metric (across both windows)
        X_only = _within_mouse_zscore_features(df, metric_cols, groups)
        for cv in ['kfold', 'lomo']:
            r2, _, _ = cv_ridge_r2(X_only, target, groups, cv=cv,
                                   alpha=alpha, n_splits=n_splits, seed=seed)
            rows.append({'model': f'only:{base}',
                         'metric': base, 'metric_label': label,
                         'cv': cv, 'r2': r2,
                         'n_features': X_only.shape[1]})

        # ko:<metric> — full minus this metric
        ko_cols = [c for c in full_cols if c not in metric_cols]
        if ko_cols:
            X_ko = _within_mouse_zscore_features(df, ko_cols, groups)
            for cv in ['kfold', 'lomo']:
                r2, _, _ = cv_ridge_r2(X_ko, target, groups, cv=cv,
                                       alpha=alpha, n_splits=n_splits,
                                       seed=seed)
                rows.append({'model': f'ko:{base}',
                             'metric': base, 'metric_label': label,
                             'cv': cv, 'r2': r2,
                             'n_features': X_ko.shape[1]})

    # ----- Best-single-neuron baseline (kept) -----
    if per_mouse_neuron_rates_late is not None:
        per_mouse_r2 = []
        for mid, sub in df.groupby('Mouse_ID'):
            if mid not in per_mouse_neuron_rates_late:
                continue
            X_n = per_mouse_neuron_rates_late[mid]
            y_m = target[sub.index.values]
            r2_kf = cv_best_neuron_baseline_r2(X_n, y_m,
                                                n_splits=n_splits, seed=seed)
            if not np.isnan(r2_kf):
                per_mouse_r2.append(r2_kf)
        if per_mouse_r2:
            rows.append({'model': 'best_single_neuron',
                         'cv': 'kfold-per-mouse',
                         'r2': float(np.nanmean(per_mouse_r2)),
                         'n_features': 1})

    # ----- Within-CV deltas -----
    cv_names = sorted({r['cv'] for r in rows})
    for cv_name in cv_names:
        full_in_cv = [r for r in rows if r['model'] == 'full'
                      and r['cv'] == cv_name]
        if not full_in_cv:
            continue
        full_r2 = full_in_cv[0]['r2']
        for r in rows:
            if r['cv'] != cv_name:
                continue
            r['delta_r2_vs_full'] = r['r2'] - full_r2
            if isinstance(r['model'], str) and r['model'].startswith('ko:'):
                r['unique_r2_vs_full'] = full_r2 - r['r2']

    # Suppress unused-variable warning from label_lookup; metric_label is
    # already pulled directly from the loop above. (Kept for callers that
    # may want to map base → label without iterating METRIC_BASES.)
    _ = label_lookup
    return rows


# ==========================================
# 7. Summary tables
# ==========================================

def print_correlation_summary(df, ucol, ulabel):
    print(f"\n{'═'*86}")
    print(f"  Correlations: Neural Metrics vs {ulabel}")
    print(f"{'═'*86}")
    print(f"  {'Metric':<28s} {'Win':>5s}  "
          f"{'Pearson':>9s}  {'q (BH)':>9s}  "
          f"{'Spearman':>9s}  {'q (BH)':>9s}")
    print(f"  {'-'*86}")
    rows = []
    pvalues = []
    for window in ['Full', 'Late']:
        for base, label in METRIC_BASES:
            cn = f"{base}_{window}"
            if cn not in df.columns:
                continue
            clean = df.dropna(subset=[ucol, cn])
            if len(clean) < 5:
                rows.append({'metric': label, 'window': window,
                             'pearson_r': np.nan, 'pearson_p': np.nan,
                             'spearman_r': np.nan, 'spearman_p': np.nan})
                pvalues.extend([np.nan, np.nan])
                continue
            rp, pp, rs, ps_ = pearson_spearman(clean[ucol].values,
                                                clean[cn].values)
            rows.append({'metric': label, 'window': window,
                         'pearson_r': rp, 'pearson_p': pp,
                         'spearman_r': rs, 'spearman_p': ps_})
            pvalues.extend([pp, ps_])

    qs = benjamini_hochberg(np.asarray(pvalues))
    for i, r in enumerate(rows):
        r['pearson_q'] = qs[2 * i]
        r['spearman_q'] = qs[2 * i + 1]
        sigp = '***' if r['pearson_q'] < 0.001 else '**' if r['pearson_q'] < 0.01 else '*' if r['pearson_q'] < 0.05 else ''
        sigs = '***' if r['spearman_q'] < 0.001 else '**' if r['spearman_q'] < 0.01 else '*' if r['spearman_q'] < 0.05 else ''
        rp = r['pearson_r']
        pp_q = r['pearson_q']
        rs = r['spearman_r']
        ps_q = r['spearman_q']
        rp_s = f"{rp:+.3f}" if not np.isnan(rp) else "   nan"
        rs_s = f"{rs:+.3f}" if not np.isnan(rs) else "   nan"
        pp_s = f"{pp_q:.2e}" if not np.isnan(pp_q) else "   nan"
        ps_s = f"{ps_q:.2e}" if not np.isnan(ps_q) else "   nan"
        print(f"  {r['metric']:<28s} {r['window']:>5s}  "
              f"{rp_s:>9s} {sigp:>3s}  {pp_s:>9s}  "
              f"{rs_s:>9s} {sigs:>3s}  {ps_s:>9s}")
    print("  q-values are Benjamini–Hochberg adjusted across the rows above.")
    return pd.DataFrame(rows)


def print_partial_correlation_summary(df, ucol, ulabel):
    print(f"\n{'═'*86}")
    print(f"  Partial correlations (Late window) vs {ulabel}")
    print("  Controls: stimulus = (Ori, Contrast, Disp);  +behaviour adds (Vel, Lick)")
    print("  Computed per-mouse, pooled by Fisher-z weighted average.")
    print(f"{'═'*86}")
    print(f"  {'Metric':<28s} {'Raw r':>8s}  {'r|stim':>9s}  {'p|stim':>9s}  "
          f"{'r|stim+beh':>11s}  {'p|stim+beh':>11s}")
    print(f"  {'-'*86}")
    rows = []
    for base, label in METRIC_BASES:
        cn = f"{base}_Late"
        if cn not in df.columns:
            continue
        clean = df.dropna(subset=[ucol, cn])
        if len(clean) < 5:
            continue
        rp, _, _, _ = pearson_spearman(clean[cn].values, clean[ucol].values)
        pc_stim = partial_correlation(df, cn, ucol,
                                       controls=('Orientation', 'Contrast', 'Dispersion'),
                                       per_mouse=True)
        pc_stim_beh = partial_correlation(df, cn, ucol,
                                           controls=('Orientation', 'Contrast', 'Dispersion'),
                                           extra=('Velocity', 'Lick_Rate'),
                                           per_mouse=True)
        rows.append({'metric': label, 'raw_r': rp,
                     'partial_r_stim': pc_stim['r'],
                     'partial_p_stim': pc_stim['p'],
                     'partial_r_stim_beh': pc_stim_beh['r'],
                     'partial_p_stim_beh': pc_stim_beh['p']})
        print(f"  {label:<28s} {rp:>+8.3f}  "
              f"{pc_stim['r']:>+9.3f}  {pc_stim['p']:>9.2e}  "
              f"{pc_stim_beh['r']:>+11.3f}  {pc_stim_beh['p']:>11.2e}")
    return pd.DataFrame(rows)


def plot_neural_psychometrics(df):
    set_plot_style()
    for base, label in METRIC_BASES:
        cn = f"{base}_Late"
        if cn not in df.columns:
            continue
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)
        sns.lineplot(data=df, x='Orientation', y=cn, ax=axes[0],
                     color='#222', marker='o', errorbar='se')
        axes[0].set_title('vs Orientation', fontsize=11, fontweight='bold')
        sns.lineplot(data=df, x='Contrast', y=cn, ax=axes[1],
                     color='#222', marker='o', errorbar='se')
        axes[1].set_title('vs Contrast', fontsize=11, fontweight='bold')
        try:
            axes[1].set_xscale('log')
        except Exception:
            pass
        sns.lineplot(data=df, x='Dispersion', y=cn, ax=axes[2],
                     color='#222', marker='o', errorbar='se')
        axes[2].set_title('vs Dispersion', fontsize=11, fontweight='bold')
        sns.lineplot(data=df, x='Orientation', y=cn, hue='Contrast', ax=axes[3],
                     palette='viridis', marker='o', errorbar='se')
        axes[3].set_title('Ori × Contrast', fontsize=11, fontweight='bold')
        axes[3].legend(title='Contrast', fontsize=8, title_fontsize=9, frameon=False)
        sns.lineplot(data=df, x='Orientation', y=cn, hue='Dispersion', ax=axes[4],
                     palette='magma', marker='o', errorbar='se')
        axes[4].set_title('Ori × Dispersion', fontsize=11, fontweight='bold')
        axes[4].legend(title='Dispersion', fontsize=8, title_fontsize=9, frameon=False)
        for ax in axes:
            ax.set_xlabel('Stimulus Feature', fontsize=9)
            ax.set_ylabel(label if ax == axes[0] else '', fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"Neural Psychometrics: {label} (1-2s)",
                     fontsize=14, fontweight='bold', y=1.05)
        path = f"psychometrics_neural_{base}.svg"
        plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        print(f"  Saved {path}")
        plt.close()


# ==========================================
# 8. Entry Point
# ==========================================

if __name__ == "__main__":
    mouse_list = [0, 1, 2, 3, 4, 5]
    results_df, per_mouse_neuron_rates = run_population_variance_pipeline(mouse_list)

    if results_df.empty:
        print("[!] No data extracted.")
    else:
        results_df.to_csv("heuristics_results.csv", index=False)
        print(f"\nSaved heuristics_results.csv  ({len(results_df)} trials)")

        plot_neural_psychometrics(results_df)

        all_corr = {}
        all_partial = {}
        all_cv = {}
        for ucol, ulab in UNCERTAINTY_TARGETS:
            if ucol not in results_df.columns or results_df[ucol].dropna().empty:
                print(f"\n[skip] {ucol} not available")
                continue
            print(f"\n{'═'*86}\n  TARGET: {ulab}\n{'═'*86}")

            corr_df = print_correlation_summary(results_df, ucol, ulab)
            all_corr[ucol] = corr_df
            plot_pooled(results_df, ucol, ulab,
                        suffix=ucol.lower())
            plot_per_mouse(results_df, ucol, ulab,
                           suffix=ucol.lower())

            partial_df = print_partial_correlation_summary(results_df, ucol, ulab)
            all_partial[ucol] = partial_df
            plot_partial_vs_raw(results_df, ucol, ulab, suffix=ucol.lower())
            plot_residuals(results_df, ucol, ulab, suffix=ucol.lower())

            cv_rows = run_cv_regression(
                results_df, ucol, ulab,
                per_mouse_neuron_rates_late=per_mouse_neuron_rates)
            all_cv[ucol] = cv_rows
            cv_df = pd.DataFrame(cv_rows)
            cv_df.to_csv(f"cv_regression_{ucol}.csv", index=False)
            plot_cv_regression_bar(cv_rows, ulab, suffix=ucol.lower())

            print("\n  CV regression R² (target z-scored within mouse):")
            for cv_name in ['kfold', 'lomo']:
                sub = cv_df[cv_df['cv'] == cv_name]
                if sub.empty:
                    continue
                full_r = sub[sub['model'] == 'full']['r2']
                full_val = float(full_r.iloc[0]) if not full_r.empty else np.nan
                print(f"\n    [{cv_name}]  full R² = {full_val:+.3f}")
                # Per-metric: show only/ko side-by-side, sorted by unique contribution
                only_rows = sub[sub['model'].astype(str).str.startswith('only:')]
                ko_rows = sub[sub['model'].astype(str).str.startswith('ko:')]
                if not only_rows.empty and not ko_rows.empty:
                    only_map = {r['metric']: r['r2'] for _, r in only_rows.iterrows()
                                if 'metric' in r and not pd.isna(r.get('metric'))}
                    ko_map = {r['metric']: r['r2'] for _, r in ko_rows.iterrows()
                              if 'metric' in r and not pd.isna(r.get('metric'))}
                    label_map = {r['metric']: r.get('metric_label', r['metric'])
                                 for _, r in only_rows.iterrows()
                                 if 'metric' in r and not pd.isna(r.get('metric'))}
                    metrics = sorted(set(only_map) | set(ko_map),
                                      key=lambda m: full_val - ko_map.get(m, full_val),
                                      reverse=True)
                    print(f"      {'metric':<32s} {'only':>8s}  {'ko':>8s}  "
                          f"{'unique (full-ko)':>16s}")
                    for m in metrics:
                        only_v = only_map.get(m, np.nan)
                        ko_v = ko_map.get(m, np.nan)
                        unique = (full_val - ko_v) if not np.isnan(ko_v) else np.nan
                        print(f"      {label_map.get(m, m):<32s} "
                              f"{only_v:>+8.3f}  {ko_v:>+8.3f}  {unique:>+16.3f}")
                # Best-single-neuron baseline if reported on this cv
                base_row = sub[sub['model'] == 'best_single_neuron']
                if not base_row.empty:
                    print(f"      {'best_single_neuron':<32s} "
                          f"R² = {float(base_row['r2'].iloc[0]):+.3f}")

        if all_corr:
            combined = pd.concat(
                {ucol: df for ucol, df in all_corr.items()},
                names=['target', 'idx']).reset_index(level='idx', drop=True).reset_index()
            combined.to_csv("correlation_summary.csv", index=False)
            print("\nSaved correlation_summary.csv")
        if all_partial:
            combined_p = pd.concat(
                {ucol: df for ucol, df in all_partial.items()},
                names=['target', 'idx']).reset_index(level='idx', drop=True).reset_index()
            combined_p.to_csv("partial_correlation_summary.csv", index=False)
            print("Saved partial_correlation_summary.csv")

        print("\nDone.")
