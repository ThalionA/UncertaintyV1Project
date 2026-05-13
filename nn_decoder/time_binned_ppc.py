# -*- coding: utf-8 -*-
"""
Time-binned Probabilistic Population Code (PPC).

Addresses the third sub-question in `UncertaintyV1 brainstorming.md`:
extend the standard time-averaged PPC to a per-time-bin PPC and ask
whether the time-integrated version is "trivially the same as" the
time-averaged one.

Three PPC variants are computed per trial, per mouse:

  TimeAvg
      Single PPC on time-averaged rates with a stationary template
      (the existing baseline; numerically equivalent to
      ``utils_v26.generate_PPC_targets``).

  TimeInt-stationary
      Sum of Poisson bin-wise log-likelihoods with the *same* template
      every bin → softmax over s.

  TimeInt-timevarying
      Sum of Poisson bin-wise log-likelihoods with a *bin-specific*
      template estimated from easy trials per bin.

Mathematical note (answers the brainstorm's "all linear ops?" worry):

  Under Poisson emissions with the same template at every bin,

      LL_int(s)  = Σ_t Σ_n  r_{t,n}·log f_n(s) - f_n(s)
                 = Σ_n  R_n·log f_n(s) - T·f_n(s)
                 = T · (Σ_n  r̄_n·log f_n(s) - f_n(s))
                 = T · LL_avg(s)

  so TimeInt-stationary = softmax(T · LL_avg). The "time-integrated"
  PPC under a stationary template is just a temperature-sharpened
  copy of TimeAvg — no new degrees of freedom. The non-trivial
  extension is therefore TimeInt-timevarying, which is not reducible
  to TimeAvg whenever bin-specific tuning differs from the
  trial-averaged tuning.

For each variant we record per-trial moments + KL(PPC || IO) against
the perceptual posterior and the marginal likelihood, then aggregate
per-mouse Pearson / Spearman correlations vs IO posterior mean/var,
likelihood mean/var, and decision entropy. Two windows are run: full
(0–2 s) and half (1–2 s), at 100 ms bins (downsampled from native 50
ms by averaging pairs via ``apply_temporal_binning``).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# `utils_v26` pulls in torch via sibling modules; import lazily inside the
# pipeline runner so this module remains importable for the synthetic tests.

S_GRID = np.arange(0, 91, 1)
NATIVE_BIN_MS = 50
TARGET_BIN_MS = 100


# ==========================================================================
# Per-bin tuning templates
# ==========================================================================

def fit_template_from_rates(rates_2d, trials, s_grid=S_GRID):
    """Fit a (len(s_grid), n_neurons) tuning template from time-averaged rates.

    Mirrors ``utils_v26.get_tuning_templates``: the empirical template at
    each unique stimulus orientation is the mean response over trials at
    max-contrast / min-dispersion; missing orientations are linearly
    interpolated onto ``s_grid``.
    """
    orientations = trials['orientation']
    contrasts = trials['contrast']
    dispersions = trials['dispersion']
    unique_oris = np.unique(orientations)
    n_neurons = rates_2d.shape[1]

    templates_emp = np.full((len(unique_oris), n_neurons), np.nan)
    for i, o in enumerate(unique_oris):
        ori_mask = (orientations == o)
        if not np.any(ori_mask):
            continue
        max_c = np.nanmax(contrasts[ori_mask])
        c_mask = ori_mask & (contrasts == max_c)
        min_d = np.nanmin(dispersions[c_mask])
        best_mask = c_mask & (dispersions == min_d)
        if np.any(best_mask):
            templates_emp[i, :] = np.nanmean(rates_2d[best_mask, :], axis=0)

    # Drop any orientation rows that ended up all-NaN before interpolating.
    valid_rows = ~np.all(np.isnan(templates_emp), axis=1)
    if valid_rows.sum() < 2:
        # Degenerate: return a flat template so downstream code still runs.
        flat = np.nanmean(rates_2d, axis=0)
        return np.clip(np.broadcast_to(flat, (len(s_grid), n_neurons)).copy(),
                       1e-6, None)
    f = interp1d(unique_oris[valid_rows], templates_emp[valid_rows], axis=0,
                 kind='linear', fill_value='extrapolate')
    return np.clip(f(s_grid), 1e-6, None)


def fit_templates_per_bin(activities, trials, s_grid=S_GRID):
    """Per-bin templates from easy trials.

    Parameters
    ----------
    activities : (n_trials, t_bins, n_neurons)
    trials     : dict with 'orientation', 'contrast', 'dispersion'

    Returns
    -------
    templates : (t_bins, len(s_grid), n_neurons)
    """
    n_trials, t_bins, n_neurons = activities.shape
    out = np.empty((t_bins, len(s_grid), n_neurons))
    for t in range(t_bins):
        out[t] = fit_template_from_rates(activities[:, t, :], trials, s_grid)
    return out


# ==========================================================================
# PPC distributions over s
# ==========================================================================

def _softmax_over_s(log_lik):
    """Stable softmax across the s-axis (last axis)."""
    log_lik = log_lik - np.nanmax(log_lik, axis=-1, keepdims=True)
    p = np.exp(log_lik)
    p = p / np.nansum(p, axis=-1, keepdims=True)
    return p


def ppc_time_avg(activities, template_stationary, prior=None):
    """TimeAvg PPC: Poisson log-likelihood from time-averaged rates and a
    stationary (s, n_neurons) template.

    Returns ``(likelihoods, posteriors)`` each of shape (n_trials, len(s_grid)).
    """
    r_mean = np.nanmean(activities, axis=1)           # (n_trials, n_neurons)
    f = template_stationary                            # (S, n_neurons)
    log_f = np.log(f)
    # broadcast: r_mean[:, None, :] * log_f[None, :, :]
    LL = np.nansum(r_mean[:, None, :] * log_f[None, :, :]
                   - f[None, :, :], axis=-1)           # (n_trials, S)
    L = _softmax_over_s(LL)
    if prior is None:
        return L, L.copy()
    log_post = LL + np.log(prior + 1e-12)[None, :]
    return L, _softmax_over_s(log_post)


def ppc_time_int_stationary(activities, template_stationary, prior=None):
    """TimeInt-stationary PPC: bin-wise Poisson log-likelihoods summed across
    bins with the same template every bin.

    Under exact Poisson assumptions this collapses to a temperature-sharpened
    TimeAvg (see module docstring). Implemented explicitly anyway so the
    equivalence can be verified numerically and so the code path matches the
    time-varying variant.
    """
    n_trials, t_bins, n_neurons = activities.shape
    f = template_stationary                            # (S, n_neurons)
    log_f = np.log(f)
    # r: (n_trials, t_bins, n_neurons); we need Σ_t Σ_n r·log f - f
    R = np.nansum(activities, axis=1)                  # (n_trials, n_neurons)
    # Σ_t f_n(s) = T · f_n(s) for stationary template.
    LL = (np.nansum(R[:, None, :] * log_f[None, :, :], axis=-1)
          - t_bins * np.nansum(f, axis=-1)[None, :])    # (n_trials, S)
    L = _softmax_over_s(LL)
    if prior is None:
        return L, L.copy()
    log_post = LL + np.log(prior + 1e-12)[None, :]
    return L, _softmax_over_s(log_post)


def ppc_time_int_timevarying(activities, templates_per_bin, prior=None):
    """TimeInt-timevarying PPC: bin-wise Poisson log-likelihoods summed with
    a *bin-specific* template.

    templates_per_bin : (t_bins, S, n_neurons)
    """
    n_trials, t_bins, n_neurons = activities.shape
    # Σ_t [Σ_n r_{trial,t,n} log f_t(s,n) - f_t(s,n)]
    log_f = np.log(templates_per_bin)                  # (t_bins, S, n_neurons)
    # r·log f term: einsum over (trials, bins, neurons) × (bins, S, neurons)
    term_r = np.einsum('itn,tsn->is', activities, log_f)
    term_f = np.sum(templates_per_bin, axis=(0, 2))    # (S,)  Σ_t Σ_n f_t(s,n)
    LL = term_r - term_f[None, :]                       # (n_trials, S)
    L = _softmax_over_s(LL)
    if prior is None:
        return L, L.copy()
    log_post = LL + np.log(prior + 1e-12)[None, :]
    return L, _softmax_over_s(log_post)


# ==========================================================================
# Distribution summaries
# ==========================================================================

def dist_mean_var(p, s_grid=S_GRID):
    denom = np.nansum(p, axis=-1, keepdims=True) + 1e-12
    mu = np.nansum(p * s_grid, axis=-1, keepdims=True) / denom
    var = np.nansum(p * (s_grid - mu) ** 2, axis=-1) / denom.squeeze(-1)
    return mu.squeeze(-1), var


def dist_entropy(p):
    p_safe = np.clip(p, 1e-12, 1.0)
    return -np.nansum(p_safe * np.log(p_safe), axis=-1)


def dist_kl(p, q):
    """KL(p || q), per-trial. Both arrays are (n_trials, S)."""
    p_safe = np.clip(p, 1e-12, 1.0)
    q_safe = np.clip(q, 1e-12, 1.0)
    return np.nansum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)


# ==========================================================================
# Pipeline
# ==========================================================================

def _prior_bimodal(s_grid=S_GRID, sigma=15.0):
    p = stats.norm.pdf(s_grid, 0, sigma) + stats.norm.pdf(s_grid, 90, sigma)
    return p / p.sum()


def run_mouse(activities, trials, targets_perc, targets_dec, targets_lik,
              s_grid=S_GRID, prior=None, time_window='full',
              target_bin_ms=TARGET_BIN_MS, native_bin_ms=NATIVE_BIN_MS):
    """Run the three PPC variants on one mouse and one window.

    Parameters
    ----------
    activities : (n_trials, n_native_bins, n_neurons)
        Native-resolution spike data already sliced to the trial window
        (xG ∈ [0, 2] s by the loader). The function does its own
        window + bin-size selection via ``apply_temporal_binning``.
    trials : dict
    targets_perc, targets_dec, targets_lik : (n_trials, ·) IO targets.
    """
    # Lazy import so the module is importable in test environments without
    # torch installed (utils_v26 pulls torch via siblings).
    from utils_v26 import apply_temporal_binning

    if prior is None:
        prior = _prior_bimodal(s_grid)

    binned = apply_temporal_binning(activities, time_window=time_window,
                                    bin_size_ms=target_bin_ms,
                                    base_bin_ms=native_bin_ms)
    n_trials, t_bins, n_neurons = binned.shape

    # Stationary template from time-averaged rates of easy trials.
    rates_2d = np.nanmean(binned, axis=1)
    template_stat = fit_template_from_rates(rates_2d, trials, s_grid)
    templates_tv = fit_templates_per_bin(binned, trials, s_grid)

    L_avg, P_avg = ppc_time_avg(binned, template_stat, prior=prior)
    L_int_s, P_int_s = ppc_time_int_stationary(binned, template_stat, prior=prior)
    L_int_v, P_int_v = ppc_time_int_timevarying(binned, templates_tv, prior=prior)

    # IO moments and decision entropy
    io_post_mu, io_post_var = dist_mean_var(targets_perc, s_grid)
    if targets_lik is not None:
        io_lik_mu, io_lik_var = dist_mean_var(targets_lik, s_grid)
    else:
        io_lik_mu = np.full(n_trials, np.nan)
        io_lik_var = np.full(n_trials, np.nan)
    dec_H = dist_entropy(targets_dec)

    out = {
        'n_trials': n_trials,
        't_bins': t_bins,
        'n_neurons': n_neurons,
        'time_window': time_window,
        'target_bin_ms': target_bin_ms,
        'distributions': {
            'TimeAvg':           {'L': L_avg,   'P': P_avg},
            'TimeInt_stat':      {'L': L_int_s, 'P': P_int_s},
            'TimeInt_timevary':  {'L': L_int_v, 'P': P_int_v},
        },
        'io': {
            'post': targets_perc,
            'lik':  targets_lik,
            'dec':  targets_dec,
            'post_mu': io_post_mu, 'post_var': io_post_var,
            'lik_mu':  io_lik_mu,  'lik_var':  io_lik_var,
            'dec_H':   dec_H,
        },
        'trials': trials,
    }
    return out


def per_trial_table(run_out, mouse_id):
    """Flatten a run_mouse result into a long-format DataFrame of per-trial
    PPC moments / KL against IO. One row per (variant, trial)."""
    rows = []
    s_grid = S_GRID
    io = run_out['io']
    trials = run_out['trials']
    for variant, dists in run_out['distributions'].items():
        L = dists['L']
        P = dists['P']
        l_mu, l_var = dist_mean_var(L, s_grid)
        p_mu, p_var = dist_mean_var(P, s_grid)
        kl_post = dist_kl(P, io['post'])
        kl_post_rev = dist_kl(io['post'], P)
        if io['lik'] is not None:
            kl_lik = dist_kl(L, io['lik'])
            kl_lik_rev = dist_kl(io['lik'], L)
        else:
            kl_lik = np.full(L.shape[0], np.nan)
            kl_lik_rev = np.full(L.shape[0], np.nan)
        n = L.shape[0]
        rows.append(pd.DataFrame({
            'Mouse_ID': mouse_id,
            'Variant':  variant,
            'Window':   run_out['time_window'],
            'Trial_Idx': np.arange(n),
            'Orientation': trials['orientation'],
            'Contrast':    trials['contrast'],
            'Dispersion':  trials['dispersion'],
            'PPC_Lik_Mu':  l_mu,  'PPC_Lik_Var':  l_var,
            'PPC_Post_Mu': p_mu,  'PPC_Post_Var': p_var,
            'PPC_Post_H':  dist_entropy(P),
            'KL_PostFromIO':     kl_post,
            'KL_IOPostFromPPC':  kl_post_rev,
            'KL_LikFromIO':      kl_lik,
            'KL_IOLikFromPPC':   kl_lik_rev,
            'IO_Post_Mu':  io['post_mu'],  'IO_Post_Var':  io['post_var'],
            'IO_Lik_Mu':   io['lik_mu'],   'IO_Lik_Var':   io['lik_var'],
            'IO_Dec_H':    io['dec_H'],
        }))
    return pd.concat(rows, ignore_index=True)


def correlation_summary(df):
    """Per-(Mouse, Variant, Window) Pearson/Spearman correlations between
    PPC summaries and IO summaries. Headline rows compare across variants."""
    pairs = [
        ('PPC_Post_Mu',  'IO_Post_Mu'),
        ('PPC_Post_Var', 'IO_Post_Var'),
        ('PPC_Lik_Mu',   'IO_Lik_Mu'),
        ('PPC_Lik_Var',  'IO_Lik_Var'),
        ('PPC_Post_H',   'IO_Dec_H'),
    ]
    rows = []
    for (mouse, variant, window), g in df.groupby(['Mouse_ID', 'Variant', 'Window']):
        for x, y in pairs:
            xv = g[x].values
            yv = g[y].values
            ok = np.isfinite(xv) & np.isfinite(yv)
            if ok.sum() < 5:
                r_p = r_s = np.nan
            else:
                r_p = stats.pearsonr(xv[ok], yv[ok])[0]
                r_s = stats.spearmanr(xv[ok], yv[ok])[0]
            rows.append({
                'Mouse_ID': mouse, 'Variant': variant, 'Window': window,
                'Pair': f'{x} ~ {y}',
                'pearson_r': r_p, 'spearman_r': r_s, 'n': int(ok.sum()),
            })
    return pd.DataFrame(rows)


# ==========================================================================
# Figures
# ==========================================================================

VARIANT_COLORS = {
    'TimeAvg':           '#4361ee',
    'TimeInt_stat':      '#7209b7',
    'TimeInt_timevary':  '#e63946',
}
IO_COLOR = '#2a9d8f'


def _pick_example_trial_indices(io_post_var, n=3):
    """Pick (low / mid / high) IO-posterior-variance trial indices."""
    finite = np.where(np.isfinite(io_post_var))[0]
    if len(finite) < n:
        return finite
    order = finite[np.argsort(io_post_var[finite])]
    return order[np.linspace(0, len(order) - 1, n).astype(int)]


def plot_example_distributions(run_outs, mouse_ids, window, out_path):
    """For each mouse: 3 example trials × {TimeAvg, TimeInt-stat,
    TimeInt-timevary, IO posterior}, overlaid on the s-axis."""
    n_mice = len(run_outs)
    fig, axes = plt.subplots(n_mice, 3, figsize=(11, 2.4 * n_mice),
                             squeeze=False, sharex=True)
    s_grid = S_GRID
    for i, (mid, res) in enumerate(zip(mouse_ids, run_outs)):
        io_var = res['io']['post_var']
        picks = _pick_example_trial_indices(io_var, n=3)
        labels = ['low IO var', 'mid IO var', 'high IO var']
        for j, ti in enumerate(picks):
            ax = axes[i][j]
            for variant, dists in res['distributions'].items():
                ax.plot(s_grid, dists['P'][ti], lw=1.4,
                        color=VARIANT_COLORS[variant], label=variant)
            ax.plot(s_grid, res['io']['post'][ti], lw=1.6, ls='--',
                    color=IO_COLOR, label='IO posterior')
            ax.axvline(res['trials']['orientation'][ti], color='#888',
                       lw=0.8, ls=':')
            ax.set_title(f"Mouse {mid} — {labels[j]} "
                         f"(ori={res['trials']['orientation'][ti]:.0f}°)",
                         fontsize=8)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)
            if i == n_mice - 1:
                ax.set_xlabel('Orientation s (°)', fontsize=8)
            if j == 0:
                ax.set_ylabel('p(s | r)', fontsize=8)
    axes[0][0].legend(fontsize=6, loc='upper right', framealpha=0.85)
    fig.suptitle(f'Time-binned PPC — example trial distributions '
                 f'(window={window}, 100ms bins)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hexbin_grid(df, window, out_path):
    """Hexbin scatter: PPC summary vs IO summary, one panel per variant ×
    pair. Pooled across mice for the given window."""
    sub = df[df['Window'] == window]
    variants = ['TimeAvg', 'TimeInt_stat', 'TimeInt_timevary']
    pairs = [
        ('PPC_Post_Var', 'IO_Post_Var', 'PPC post var', 'IO post var'),
        ('PPC_Post_H',   'IO_Dec_H',    'PPC post entropy', 'IO decision entropy'),
        ('PPC_Post_Mu',  'IO_Post_Mu',  'PPC post mean (°)', 'IO post mean (°)'),
    ]
    fig, axes = plt.subplots(len(pairs), len(variants),
                             figsize=(3.5 * len(variants), 3.2 * len(pairs)),
                             squeeze=False)
    for r, (xcol, ycol, xlab, ylab) in enumerate(pairs):
        for c, variant in enumerate(variants):
            ax = axes[r][c]
            d = sub[sub['Variant'] == variant][[xcol, ycol]].dropna()
            if len(d) < 5:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            x = d[xcol].values
            y = d[ycol].values
            ax.hexbin(x, y, gridsize=40, cmap='magma', mincnt=1,
                      linewidths=0.1, edgecolors='none')
            try:
                rp = stats.pearsonr(x, y)[0]
                rs = stats.spearmanr(x, y)[0]
                ax.text(0.04, 0.96, f"r={rp:+.2f}\nρ={rs:+.2f}",
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=8, color='w',
                        bbox=dict(facecolor='black', alpha=0.5, pad=2,
                                  edgecolor='none'))
            except Exception:
                pass
            if r == 0:
                ax.set_title(variant, fontsize=10,
                             color=VARIANT_COLORS[variant])
            if r == len(pairs) - 1:
                ax.set_xlabel(xlab, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'PPC vs IO — pooled across mice (window={window})',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_bars(summary, out_path):
    """Per-mouse mean ± SEM Pearson r, grouped by (variant, window, pair)."""
    pairs = sorted(summary['Pair'].unique())
    windows = sorted(summary['Window'].unique())
    variants = ['TimeAvg', 'TimeInt_stat', 'TimeInt_timevary']
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 4.2),
                             squeeze=False, sharey=True)
    bar_w = 0.28
    x_centres = np.arange(len(windows))
    for i, pair in enumerate(pairs):
        ax = axes[0][i]
        for j, variant in enumerate(variants):
            means, sems = [], []
            for w in windows:
                vals = summary[(summary['Pair'] == pair)
                               & (summary['Window'] == w)
                               & (summary['Variant'] == variant)]['pearson_r'].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals))
                            if len(vals) > 1 else 0.0)
            xpos = x_centres + (j - 1) * bar_w
            ax.bar(xpos, means, bar_w, yerr=sems, capsize=2,
                   color=VARIANT_COLORS[variant], label=variant,
                   edgecolor='k', linewidth=0.4)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x_centres)
        ax.set_xticklabels(windows, fontsize=9)
        ax.set_title(pair, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel('Pearson r (per-mouse mean ± SEM)', fontsize=9)
    axes[0][-1].legend(fontsize=7, loc='best', framealpha=0.85)
    fig.suptitle('Time-binned PPC — correlation with IO by variant × window',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kl_histograms(df, out_path):
    """KL(PPC_post || IO_post) across variants × windows."""
    windows = sorted(df['Window'].unique())
    variants = ['TimeAvg', 'TimeInt_stat', 'TimeInt_timevary']
    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 4),
                             squeeze=False, sharey=True)
    for k, w in enumerate(windows):
        ax = axes[0][k]
        for variant in variants:
            vals = df[(df['Window'] == w) & (df['Variant'] == variant)][
                'KL_PostFromIO'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < 5:
                continue
            # clip extreme tails for readability
            hi = np.nanpercentile(vals, 99)
            ax.hist(np.clip(vals, 0, hi), bins=60, alpha=0.55,
                    color=VARIANT_COLORS[variant], label=variant,
                    edgecolor='none')
        ax.set_title(f'window={w}', fontsize=10)
        ax.set_xlabel('KL(PPC_post || IO_post)', fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel('# trials (pooled across mice)', fontsize=9)
        ax.legend(fontsize=7, framealpha=0.85)
    fig.suptitle('KL divergence: PPC posterior vs IO posterior', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stationary_sanity(run_outs, mouse_ids, window, out_path):
    """Numerical sanity: TimeInt-stationary should equal softmax(T · log TimeAvg)
    distribution-wise. Plot per-trial entropies as a scatter; identity line."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for mid, res in zip(mouse_ids, run_outs):
        T = res['t_bins']
        P_avg = res['distributions']['TimeAvg']['P']
        P_int_s = res['distributions']['TimeInt_stat']['P']
        P_avg_pow = np.clip(P_avg, 1e-12, 1.0) ** T
        P_avg_pow = P_avg_pow / P_avg_pow.sum(axis=-1, keepdims=True)
        H_expected = dist_entropy(P_avg_pow)
        H_actual = dist_entropy(P_int_s)
        ax.scatter(H_expected, H_actual, s=6, alpha=0.4, label=f'Mouse {mid}')
    lo = 0
    hi = float(np.nanmax([ax.get_xlim()[1], ax.get_ylim()[1]]))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Entropy of softmax(T · log TimeAvg)')
    ax.set_ylabel('Entropy of TimeInt-stationary')
    ax.set_title(f'Stationary identity check (window={window})')
    ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# Entry point
# ==========================================================================

def main(mouse_ids=(0, 1, 2, 3, 4, 5), windows=('full', 'half'),
         out_dir='time_binned_ppc_out'):
    from utils_v26 import load_vr_export

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    prior = _prior_bimodal()
    all_trials = []
    runs_by_window = {w: ([], []) for w in windows}  # window -> (mouse_ids, run_outs)
    for window in windows:
        for mid in mouse_ids:
            print(f"[mouse {mid} | window {window}] loading…")
            try:
                acts, t_perc, t_dec, t_lik, trials = load_vr_export(mid)
            except Exception as exc:
                print(f"  skip mouse {mid}: {exc}")
                continue
            # loader returns (n_neurons, n_trials, t_bins); permute.
            acts = np.transpose(acts, (1, 2, 0))
            res = run_mouse(acts, trials, t_perc, t_dec, t_lik,
                            prior=prior, time_window=window)
            tbl = per_trial_table(res, mouse_id=mid)
            all_trials.append(tbl)
            runs_by_window[window][0].append(mid)
            runs_by_window[window][1].append(res)
            print(f"  trials={res['n_trials']} bins={res['t_bins']} "
                  f"neurons={res['n_neurons']}")
    if not all_trials:
        print("No mice ran (no data?)."); return
    df = pd.concat(all_trials, ignore_index=True)
    df.to_csv(os.path.join(out_dir, 'per_trial.csv'), index=False)
    summary = correlation_summary(df)
    summary.to_csv(os.path.join(out_dir, 'correlation_summary.csv'), index=False)

    # Figures
    for window, (mids, runs) in runs_by_window.items():
        if not runs:
            continue
        plot_example_distributions(
            runs, mids, window,
            os.path.join(fig_dir, f'fig1_examples_{window}.png'))
        plot_hexbin_grid(
            df, window,
            os.path.join(fig_dir, f'fig3_hexbin_{window}.png'))
        plot_stationary_sanity(
            runs, mids, window,
            os.path.join(fig_dir, f'fig5_stationary_sanity_{window}.png'))
    plot_correlation_bars(summary,
                          os.path.join(fig_dir, 'fig2_correlation_bars.png'))
    plot_kl_histograms(df, os.path.join(fig_dir, 'fig4_kl_histograms.png'))

    # Compact print: mean correlation per (Variant, Window, Pair) over mice.
    agg = (summary
           .groupby(['Window', 'Variant', 'Pair'])
           .agg(mean_r=('pearson_r', 'mean'),
                mean_rho=('spearman_r', 'mean'),
                n_mice=('pearson_r', 'count'))
           .reset_index())
    print("\n=== Correlation summary (per-mouse mean) ===")
    print(agg.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f"\nFigures saved to: {fig_dir}")


if __name__ == '__main__':
    main()
