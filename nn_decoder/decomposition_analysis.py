# -*- coding: utf-8 -*-
"""Moment-error decomposition of decoded vs target distributional losses.

Asks the question: when the spatial and temporal architectures differ in their decoding
loss, is the difference primarily driven by the *mean* of the decoded
distribution (a stimulus-orientation estimate), the *variance* (the actual
uncertainty representation), or higher-order shape (skewness, residual)?

Loss conventions
----------------
For 91-bin distributional targets Q(theta) and L(theta):

  mean_error   = (mu_decoded   - mu_target)^2          [deg^2]
  var_error    = (var_decoded  - var_target)^2         [deg^4]
  skew_error   = (skew_decoded - skew_target)^2        [unitless]
  total_l2_sq  = sum((decoded - target)^2, axis=-1)    [probability^2]
  shape_residual_l2_sq = total_l2_sq - moment_explained_l2_sq

  moment_explained_l2_sq is the squared L2 distance between Gaussian
  reconstructions matched to (mu, var) of the decoded and target. This
  quantifies how much of the raw L2 loss the first two moments alone can
  account for; the residual captures higher-order shape disagreement
  (skewness, kurtosis, multimodality).

For 2-bin decision targets [P(Go), P(NoGo)]:

  pgo_bias       = (P_hat(Go)  - P(Go))^2
  entropy_error  = (H(P_hat)   - H(P))^2

All errors are reported per-trial; aggregation across trials and mice is the
caller's responsibility (see ``aggregate_per_mouse``).

This module is pure analysis — no plotting, no I/O side effects beyond
loading from .mat / .npy files via the load helpers. Call it from
``plot_decomposition.py`` for the visualisation.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.io as sio

import paths


# ----------------------------------------------------------------------
# Per-trial moment computations
# ----------------------------------------------------------------------

def trial_moments(p, grid):
    """Compute mean and variance of each row of ``p`` over ``grid``.

    Parameters
    ----------
    p : ndarray, shape (n_trials, n_bins)
        Discrete probability distributions (rows must sum to ~1).
    grid : ndarray, shape (n_bins,)
        Support values (e.g. orientation in degrees).

    Returns
    -------
    mu : ndarray, shape (n_trials,)
    var : ndarray, shape (n_trials,)
    """
    p = np.asarray(p, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"`p` must be 2-D (n_trials, n_bins); got shape {p.shape}.")
    if p.shape[1] != grid.shape[0]:
        raise ValueError(
            f"`p` has {p.shape[1]} bins but `grid` has {grid.shape[0]}; mismatch."
        )
    norm = p.sum(axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    p = p / norm
    mu = (p * grid[None, :]).sum(axis=1)
    var = (p * (grid[None, :] - mu[:, None]) ** 2).sum(axis=1)
    return mu, var


def trial_skewness(p, grid):
    """Standardised third moment of each row of ``p`` over ``grid``.

    Returns NaN for any row with variance below 1e-9 (degenerate, no shape).
    """
    p = np.asarray(p, dtype=float)
    grid = np.asarray(grid, dtype=float)
    mu, var = trial_moments(p, grid)
    sigma = np.sqrt(np.maximum(var, 1e-30))
    third = (p / np.where(p.sum(1, keepdims=True) > 0, p.sum(1, keepdims=True), 1.0)
             * (grid[None, :] - mu[:, None]) ** 3).sum(axis=1)
    skew = third / sigma ** 3
    skew = np.where(var > 1e-9, skew, np.nan)
    return skew


# ----------------------------------------------------------------------
# Distributional decomposition (Q and L; 91-D)
# ----------------------------------------------------------------------

def _gauss_template(mu, var, grid):
    """Bin a Gaussian density with parameters (mu, var) onto ``grid`` and
    normalise. Used to compute the moment-explained portion of the L2 loss.

    Vectorised across trials.
    """
    sigma = np.sqrt(np.maximum(var, 1e-30))
    z = (grid[None, :] - mu[:, None]) / sigma[:, None]
    p = np.exp(-0.5 * z ** 2)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return p / s


def decompose_distributional(decoded, target, grid):
    """Decompose decoded-vs-target loss into mean / variance / skewness errors
    and a shape residual, per trial.

    Returns a dict of ndarrays each of shape (n_trials,).
    """
    decoded = np.asarray(decoded, dtype=float)
    target = np.asarray(target, dtype=float)
    grid = np.asarray(grid, dtype=float)

    mu_d, var_d = trial_moments(decoded, grid)
    mu_t, var_t = trial_moments(target, grid)

    skew_d = trial_skewness(decoded, grid)
    skew_t = trial_skewness(target, grid)

    mean_error = (mu_d - mu_t) ** 2
    var_error = (var_d - var_t) ** 2
    skew_error = (skew_d - skew_t) ** 2  # NaN propagates if either trial is degenerate

    total_l2_sq = ((decoded - target) ** 2).sum(axis=1)

    # Moment-explained portion: the squared L2 distance between Gaussian
    # reconstructions of decoded and target with their respective (mu, var).
    g_d = _gauss_template(mu_d, var_d, grid)
    g_t = _gauss_template(mu_t, var_t, grid)
    moment_explained_l2_sq = ((g_d - g_t) ** 2).sum(axis=1)
    shape_residual_l2_sq = total_l2_sq - moment_explained_l2_sq

    return dict(
        mu_decoded=mu_d, mu_target=mu_t,
        var_decoded=var_d, var_target=var_t,
        skew_decoded=skew_d, skew_target=skew_t,
        mean_error=mean_error,
        var_error=var_error,
        skew_error=skew_error,
        total_l2_sq=total_l2_sq,
        moment_explained_l2_sq=moment_explained_l2_sq,
        shape_residual_l2_sq=shape_residual_l2_sq,
    )


# ----------------------------------------------------------------------
# Decision-posterior decomposition (2-D)
# ----------------------------------------------------------------------

def _binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def decompose_decision(decoded, target):
    """For 2-D decision targets [P(Go), P(NoGo)], return per-trial:
      pgo_bias       = (P_hat(Go)  - P(Go))^2
      entropy_error  = (H(P_hat)   - H(P))^2
    """
    decoded = np.asarray(decoded, dtype=float)
    target = np.asarray(target, dtype=float)
    if decoded.shape[1] != 2:
        raise ValueError(
            f"decompose_decision expects 2-bin targets; got shape {decoded.shape}."
        )
    pgo_d = decoded[:, 0]
    pgo_t = target[:, 0]
    pgo_bias = (pgo_d - pgo_t) ** 2
    entropy_error = (_binary_entropy(pgo_d) - _binary_entropy(pgo_t)) ** 2
    total_mse = ((decoded - target) ** 2).mean(axis=1)
    return dict(
        pgo_decoded=pgo_d, pgo_target=pgo_t,
        h_decoded=_binary_entropy(pgo_d), h_target=_binary_entropy(pgo_t),
        pgo_bias=pgo_bias,
        entropy_error=entropy_error,
        total_mse=total_mse,
    )


# ----------------------------------------------------------------------
# Per-mouse aggregation
# ----------------------------------------------------------------------

def aggregate_per_mouse(trial_df, group_keys=('mouse', 'arch', 'target', 'split')):
    """Mean-aggregate every numeric column of ``trial_df`` over the trial
    dimension, grouped by ``group_keys``."""
    keys = list(group_keys)
    numeric = trial_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in keys and c != 'trial']
    if not numeric:
        return trial_df[keys].drop_duplicates().reset_index(drop=True)
    return (
        trial_df.groupby(keys, as_index=False)[numeric].mean()
    )


# ----------------------------------------------------------------------
# Loading and extraction
# ----------------------------------------------------------------------

# Default grid for Q(theta) / L(theta) targets in this project.
DEFAULT_GRID_91 = np.arange(0, 91, 1, dtype=float)


def load_population_results(path):
    """Load a population_results_*.mat file as a plain Python dict.

    Returns the dict that scipy.io.loadmat(..., simplify_cells=True) gives,
    preserving the {'results': ..., 'config': ...} layout."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sio.loadmat(path, simplify_cells=True)


def load_recovery_cache(path):
    """Load a recovery_cache_fixed_*.npy file as a plain Python dict.

    Returns the dict produced by run_fixed_recovery.py: keys 'spat' and 'temp'
    map to {'mouse_X': {'Dist': ...}}; 'base_config' carries the run config.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def _decompose_dist_for_arch(arch_dict, target_type, grid):
    """Return a per-trial dict of decomposition components for one (mouse, arch)
    Dist sub-dict. Branches on target_type to call the right decomposer."""
    decoded = np.asarray(arch_dict['decoded'])
    target = np.asarray(arch_dict['target'])
    if target_type in ('Q', 'L'):
        return decompose_distributional(decoded, target, grid)
    if target_type == 'd':
        return decompose_decision(decoded, target)
    raise ValueError(f"Unknown target_type {target_type!r}; expected 'Q', 'L', or 'd'.")


def extract_decompositions_from_dict(
    mat_like,
    target_type,
    split,
    archs=('spat', 'temp', 'spat_shf', 'temp_shf'),
    grid=DEFAULT_GRID_91,
):
    """Walk a production-results dict (as returned by load_population_results)
    and return a long-form trial-level DataFrame of decomposition components.

    Parameters
    ----------
    mat_like : dict
        Must contain 'results' mapping mouse_id -> {'Dist': {...}}.
    target_type : {'Q', 'L', 'd'}
        Which loss decomposition to use. 'Q' and 'L' use the distributional
        decomposition over a 91-bin grid; 'd' uses the 2-bin decision
        decomposition.
    split : str
        Recorded in the output for downstream grouping (e.g.
        'stratified_balanced').
    archs : tuple of str
        Architecture sub-dicts to extract. Defaults to all four (the two
        production architectures and their shuffled controls).

    Returns
    -------
    pd.DataFrame with one row per (mouse, arch, trial). Columns include
    metadata (mouse, arch, target, split, trial) and the per-trial
    decomposition components for the chosen ``target_type``.
    """
    if 'results' not in mat_like:
        raise KeyError("`mat_like` is missing the 'results' key.")

    if target_type == 'd':
        # Decision-posterior columns; skewness undefined.
        component_keys = ('pgo_bias', 'entropy_error', 'total_mse',
                          'pgo_decoded', 'pgo_target', 'h_decoded', 'h_target')
    else:
        component_keys = ('mean_error', 'var_error', 'skew_error',
                          'total_l2_sq', 'moment_explained_l2_sq',
                          'shape_residual_l2_sq',
                          'mu_decoded', 'mu_target',
                          'var_decoded', 'var_target',
                          'skew_decoded', 'skew_target')

    rows = []
    for mouse_key, mdata in mat_like['results'].items():
        # mouse_key like 'mouse_0'
        try:
            mid = int(str(mouse_key).split('_')[-1])
        except ValueError:
            mid = mouse_key
        dist = mdata['Dist']
        for arch in archs:
            if arch not in dist:
                continue
            comp = _decompose_dist_for_arch(dist[arch], target_type, grid)
            n_trials = len(comp[component_keys[0]])
            base = dict(mouse=mid, arch=arch, target=target_type, split=split)
            for i in range(n_trials):
                row = dict(base)
                row['trial'] = i
                for k in component_keys:
                    row[k] = comp[k][i]
                rows.append(row)
    return pd.DataFrame(rows)


def extract_decompositions_from_recovery(
    recovery_dict,
    target_type,
    split,
    grid=DEFAULT_GRID_91,
    archs=('spat', 'temp'),
):
    """Walk a recovery_cache dict and return a long-form trial-level DataFrame.

    Adds a ``target_arch`` column to label the source ('spat' or 'temp') of
    the synthetic target distribution; this is the recovery-specific
    bookkeeping that the production-results path does not need.
    """
    rows = []
    for src in ('spat', 'temp'):
        if src not in recovery_dict:
            continue
        for mouse_key, mdata in recovery_dict[src].items():
            try:
                mid = int(str(mouse_key).split('_')[-1])
            except ValueError:
                mid = mouse_key
            dist = mdata['Dist']
            for arch in archs:
                if arch not in dist:
                    continue
                comp = _decompose_dist_for_arch(dist[arch], target_type, grid)
                if target_type == 'd':
                    component_keys = ('pgo_bias', 'entropy_error', 'total_mse',
                                      'pgo_decoded', 'pgo_target',
                                      'h_decoded', 'h_target')
                else:
                    component_keys = ('mean_error', 'var_error', 'skew_error',
                                      'total_l2_sq', 'moment_explained_l2_sq',
                                      'shape_residual_l2_sq',
                                      'mu_decoded', 'mu_target',
                                      'var_decoded', 'var_target',
                                      'skew_decoded', 'skew_target')
                n_trials = len(comp[component_keys[0]])
                base = dict(mouse=mid, arch=arch, target_arch=src,
                            target=target_type, split=split)
                for i in range(n_trials):
                    row = dict(base)
                    row['trial'] = i
                    for k in component_keys:
                        row[k] = comp[k][i]
                    rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Convenience driver: load all production files for a given split
# ----------------------------------------------------------------------

# Iteration tuples sourced from paths.FIT_BASENAMES so the canonical
# basenames live in exactly one place. Filtered to the three
# distributional targets used by the decomposition (choice/truechoice
# aren't decomposed here).
PRODUCTION_TARGETS = tuple(
    (t, stem) for t, stem in paths.FIT_BASENAMES['fixed'].items()
    if t in ('Q', 'L', 'd')
)

# Stimulus-only baseline files. When present, loaded as an additional
# architecture key 'stim_mean' alongside the production architectures.
STIM_MEAN_TARGETS = tuple(
    (t, stem) for t, stem in paths.FIT_BASENAMES['stim_mean'].items()
    if t in ('Q', 'L', 'd')
)


def load_all_production(split, directory='.'):
    """Load and decompose all three production target files for a split,
    plus the stimulus-only baseline files when present.

    Returns a single concatenated trial-level DataFrame across Q, L, d
    (and the stim_mean architecture when those files are on disk).
    Missing files are skipped with a printed warning so partial runs still
    produce a (smaller) frame.
    """
    parts = []
    for target_type, stem in PRODUCTION_TARGETS:
        path = os.path.join(directory, f"{stem}_{split}.mat")
        if not os.path.exists(path):
            print(f"[decomposition] skipping missing file: {path}")
            continue
        mat = load_population_results(path)
        df = extract_decompositions_from_dict(mat, target_type, split)
        parts.append(df)

    # Stim-mean baseline files have a single 'stim_mean' arch under Dist.
    for target_type, stem in STIM_MEAN_TARGETS:
        path = os.path.join(directory, f"{stem}_{split}.mat")
        if not os.path.exists(path):
            continue
        mat = load_population_results(path)
        df = extract_decompositions_from_dict(
            mat, target_type, split, archs=('stim_mean',),
        )
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_all_recovery(split, directory='.'):
    """Load and decompose all three recovery-cache .npy files for a split.

    Returns a long-form trial-level DataFrame with the recovery-specific
    `target_arch` column.
    """
    parts = []
    for target_type, recovery_label in (('Q', 'perception'),
                                         ('L', 'likelihood'),
                                         ('d', 'decision')):
        # Compose under the caller's `directory` override (default '.'),
        # but use the canonical recovery_cache filename from paths.
        path = os.path.join(directory, paths.recovery_cache(recovery_label).name)
        if not os.path.exists(path):
            print(f"[decomposition] skipping missing file: {path}")
            continue
        rec = load_recovery_cache(path)
        df = extract_decompositions_from_recovery(rec, target_type, split)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
