# -*- coding: utf-8 -*-
"""
PCA latent-space comparison: IO perceptual posteriors vs likelihoods.

Three analyses, per-mouse and pooled across mice:

  1. Baseline similarity
     PCA the full set of per-trial IO perceptual posteriors and,
     separately, the per-trial IO marginal likelihoods. Quantify how
     similar the two latent spaces are: principal angles / subspace
     overlap, canonical correlations of the top-k PC scores, and
     cosine similarity of index-matched loadings. k is chosen by 90%
     cumulative explained variance; the per-distribution 90%-k is
     reported and a common k (= max of the two) is used for the
     subspace comparisons.

  2. Condition means vs trial residuals
     For posteriors and for likelihoods, fit two PCA models:
       Approach A — PCA on condition means (one row per stim cell).
       Approach B — PCA on trial residuals  (trial minus its cell mean).
     A "condition" is the joint (Orientation, Contrast, Dispersion)
     stim-cell — the repo convention (see
     ``population_metrics_vs_uncertainty.py`` ``_design_matrix`` /
     ``partial_correlation``, default ``design='joint'``). Pooled
     analyses use the (mouse, condition) cell so means/residuals are
     within-mouse, matching the repo's residualisation convention.

  3. Cross-distribution check
     Principal angles between the Approach-A (signal) subspace and the
     Approach-B (noise) subspace, computed for posteriors and for
     likelihoods. Whichever distribution shows the larger A-vs-B
     divergence is the one whose signal and noise axes decouple most.
     Small angles ⇒ signal & noise axes aligned (entangled); large
     angles ⇒ decoupled / orthogonal.

Outputs (numeric CSVs + figures) under ``pca_posterior_vs_likelihood_out/``.
Run on the real data exactly as ``time_binned_ppc.py`` does.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# utils_v26 pulls torch via siblings; import lazily inside main().

S_GRID = np.arange(0, 91, 1)
VAR_THRESHOLD = 0.90
MOUSE_IDS = (0, 1, 2, 3, 4, 5)
N_PERM = 2000          # Haar-random null draws for principal-angle p-values
N_LOADING_PCS = 3      # how many PCs to draw in the loading-curve figures
RNG_SEED = 0

# --------------------------------------------------------------------------
# Plot style — tidy axes, no top/right spines, consistent fonts.
# --------------------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'legend.frameon': False,
    'legend.fontsize': 8,
})


# ==========================================================================
# Linear-algebra helpers
# ==========================================================================

def fit_pca(X):
    """Center X (n_samples, n_features) and return
    (components [n_comp, n_feat], explained_variance_ratio, mean)."""
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    n_comp = min(n, p)
    pca = PCA(n_components=n_comp, svd_solver='full')
    pca.fit(X)
    return pca.components_, pca.explained_variance_ratio_, pca.mean_


def k_for_variance(evr, thresh=VAR_THRESHOLD):
    """Smallest k with cumulative explained variance >= thresh."""
    c = np.cumsum(evr)
    k = int(np.searchsorted(c, thresh) + 1)
    return min(k, len(evr))


def principal_angles_deg(VA, VB):
    """Principal angles (degrees, ascending) between column spaces of
    VA (n_feat, kA) and VB (n_feat, kB). Length = min(kA, kB)."""
    ang = subspace_angles(VA, VB)          # radians, descending
    return np.sort(np.degrees(ang))        # ascending for readability


def subspace_overlap(angles_deg):
    """Mean cos^2 of the principal angles in [0, 1].
    1 = identical subspace, 0 = orthogonal."""
    if len(angles_deg) == 0:
        return np.nan
    return float(np.mean(np.cos(np.radians(angles_deg)) ** 2))


def canonical_correlations(X, Y, k):
    """Canonical correlations between two same-row score matrices via
    QR + SVD (numerically stable, no sklearn iteration). Returns up to k
    values in [0, 1], descending."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(np.sort(s)[::-1][:k], 0.0, 1.0)


def matched_loading_cosine(VA, VB, k):
    """abs cosine between index-matched columns i of VA and VB, i<k.
    PCA sign is arbitrary so absolute value is taken."""
    out = []
    for i in range(min(k, VA.shape[1], VB.shape[1])):
        a = VA[:, i]
        b = VB[:, i]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        out.append(abs(float(a @ b) / denom))
    return np.array(out)


def _basis(components, k):
    """(n_feat, k) orthonormal-ish basis from PCA components_ (k, n_feat)."""
    return np.asarray(components[:k]).T


def _random_orthonormal(p, k, rng):
    """Haar-distributed (p, k) orthonormal frame in R^p."""
    A = rng.standard_normal((p, k))
    Q, _ = np.linalg.qr(A)
    return Q


def principal_angle_null(p, k, n=N_PERM, rng=None):
    """Mean principal angle (deg) and mean cos^2 overlap between two
    Haar-random k-dim subspaces of R^p, sampled n times. Returns
    {'mean_angle': arr[n], 'overlap': arr[n]}.

    Use as a reference distribution for 'are these two k-dim subspaces
    more aligned than two random k-dim subspaces would be'."""
    rng = np.random.default_rng(RNG_SEED if rng is None else rng)
    ang_mean = np.empty(n)
    over = np.empty(n)
    for i in range(n):
        Va = _random_orthonormal(p, k, rng)
        Vb = _random_orthonormal(p, k, rng)
        a = principal_angles_deg(Va, Vb)
        ang_mean[i] = np.mean(a)
        over[i] = subspace_overlap(a)
    return {'mean_angle': ang_mean, 'overlap': over}


def empirical_p_value(value, null, tail='lower'):
    """Empirical one-sided p (inclusive). tail='lower' tests value<null
    (e.g. mean angle smaller than chance); tail='upper' tests value>null
    (e.g. overlap larger than chance)."""
    n = len(null)
    if tail == 'lower':
        k = int(np.sum(null <= value))
    else:
        k = int(np.sum(null >= value))
    return (k + 1) / (n + 1)


def sign_align_to_ref(loading, ref):
    """Return loading flipped (sign-aligned) so that its inner product
    with ref is non-negative. PCA component signs are arbitrary, so this
    is needed before overlaying or averaging."""
    loading = np.asarray(loading)
    ref = np.asarray(ref)
    return loading if np.dot(loading, ref) >= 0 else -loading


def sign_align_self(loading):
    """Flip so the largest-|amplitude| index is positive — useful when
    no reference is available (e.g. first time a loading is drawn)."""
    loading = np.asarray(loading)
    idx = int(np.argmax(np.abs(loading)))
    return loading if loading[idx] >= 0 else -loading


# ==========================================================================
# Data assembly
# ==========================================================================

def condition_labels(trials):
    """Joint (Orientation, Contrast, Dispersion) stim-cell id per trial."""
    o = np.asarray(trials['orientation'])
    c = np.asarray(trials['contrast'])
    d = np.asarray(trials['dispersion'])
    return np.array([f"{a}|{b}|{e}" for a, b, e in zip(o, c, d)])


def condition_means_and_residuals(X, cond):
    """Given X (n_trials, n_feat) and per-trial condition labels, return
    (means [n_cond, n_feat], residuals [n_trials, n_feat], cond_order)."""
    cond = np.asarray(cond)
    uniq = np.unique(cond)
    means = np.zeros((len(uniq), X.shape[1]))
    resid = np.empty_like(X)
    for i, u in enumerate(uniq):
        m = cond == u
        mu = X[m].mean(axis=0)
        means[i] = mu
        resid[m] = X[m] - mu
    return means, resid, uniq


def load_all(mouse_ids):
    """Returns dict mouse_id -> {post, lik, cond}. Trials with any
    non-finite value in either posterior or likelihood are dropped so
    all models per mouse use the same trial set."""
    from utils_v26 import load_vr_export
    data = {}
    for mid in mouse_ids:
        try:
            _, t_perc, _, t_lik, trials = load_vr_export(mid)
        except Exception as exc:
            print(f"  skip mouse {mid}: {exc}")
            continue
        if t_lik is None:
            print(f"  skip mouse {mid}: no marginal likelihood in export")
            continue
        post = np.asarray(t_perc, dtype=float)
        lik = np.asarray(t_lik, dtype=float)
        cond = condition_labels(trials)
        ok = (np.isfinite(post).all(axis=1)
              & np.isfinite(lik).all(axis=1))
        data[mid] = {'post': post[ok], 'lik': lik[ok], 'cond': cond[ok]}
        print(f"  mouse {mid}: {ok.sum()}/{len(ok)} trials kept, "
              f"{len(np.unique(cond[ok]))} conditions")
    return data


# ==========================================================================
# Analysis 1 — baseline similarity (posterior PCA vs likelihood PCA)
# ==========================================================================

def baseline_similarity(post, lik, label):
    Vp_c, evr_p, _ = fit_pca(post)
    Vl_c, evr_l, _ = fit_pca(lik)
    k_post = k_for_variance(evr_p)
    k_lik = k_for_variance(evr_l)
    k = min(max(k_post, k_lik), Vp_c.shape[0], Vl_c.shape[0])

    Vp = _basis(Vp_c, k)
    Vl = _basis(Vl_c, k)
    ang = principal_angles_deg(Vp, Vl)
    overlap = subspace_overlap(ang)

    # CCA on the top-k PC scores of the (shared) trials.
    Sp = (post - post.mean(0)) @ Vp
    Sl = (lik - lik.mean(0)) @ Vl
    ccor = canonical_correlations(Sp, Sl, k)

    mcos = matched_loading_cosine(Vp, Vl, k)

    return {
        'scope': label,
        'k_post_90': k_post,
        'k_lik_90': k_lik,
        'k_common': k,
        'mean_principal_angle_deg': float(np.mean(ang)),
        'min_principal_angle_deg': float(np.min(ang)),
        'max_principal_angle_deg': float(np.max(ang)),
        'subspace_overlap_cos2': overlap,
        'mean_canonical_corr': float(np.mean(ccor)),
        'top_canonical_corr': float(np.max(ccor)),
        'mean_matched_loading_cos': float(np.mean(mcos)),
        '_angles': ang,
        '_ccor': ccor,
        '_mcos': mcos,
        '_evr_post': evr_p,
        '_evr_lik': evr_l,
        '_Vp': Vp,
        '_Vl': Vl,
    }


# ==========================================================================
# Analysis 2 & 3 — A (condition means) vs B (residuals); cross-distribution
# ==========================================================================

def ab_models(X, cond):
    """Fit Approach A (condition means) and B (residuals) PCA for one
    distribution. Returns dict with components/evr/k for both."""
    means, resid, _ = condition_means_and_residuals(X, cond)
    Va, evr_a, _ = fit_pca(means)
    Vb, evr_b, _ = fit_pca(resid)
    return {
        'A_components': Va, 'A_evr': evr_a, 'A_k90': k_for_variance(evr_a),
        'B_components': Vb, 'B_evr': evr_b, 'B_k90': k_for_variance(evr_b),
        'n_conditions': means.shape[0], 'n_trials': X.shape[0],
    }


def ab_divergence(model, label, dist_name):
    """Principal angles between the A (signal) and B (noise) subspaces."""
    ka, kb = model['A_k90'], model['B_k90']
    k = min(max(ka, kb),
            model['A_components'].shape[0],
            model['B_components'].shape[0])
    Va = _basis(model['A_components'], k)
    Vb = _basis(model['B_components'], k)
    ang = principal_angles_deg(Va, Vb)
    return {
        'scope': label,
        'distribution': dist_name,
        'k_A_90': ka,
        'k_B_90': kb,
        'k_common': k,
        'n_conditions': model['n_conditions'],
        'n_trials': model['n_trials'],
        'mean_principal_angle_deg': float(np.mean(ang)),
        'min_principal_angle_deg': float(np.min(ang)),
        'max_principal_angle_deg': float(np.max(ang)),
        'subspace_overlap_cos2': subspace_overlap(ang),
        '_angles': ang,
        '_Va': Va,
        '_Vb': Vb,
        '_A_evr': model['A_evr'],
        '_B_evr': model['B_evr'],
    }


# ==========================================================================
# Figures
# ==========================================================================

C_POST = '#2a9d8f'
C_LIK = '#e63946'
C_A = '#264653'        # signal / condition-mean
C_B = '#e9c46a'        # noise / residual


def _feature_axis(n_feat):
    """Return (x, xlabel) for plotting loadings. If n_feat matches the
    91-bin orientation grid we plot against degrees; otherwise we fall
    back to feature index."""
    if n_feat == len(S_GRID):
        return S_GRID, 'orientation (deg)'
    return np.arange(n_feat), 'feature index'


def _stylise(ax):
    ax.tick_params(direction='out', length=3)


def plot_scree(baseline, ab_post, ab_lik, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    curves = [
        ('posterior (all-trial)', baseline['_evr_post'], C_POST, '-', 1.8),
        ('likelihood (all-trial)', baseline['_evr_lik'], C_LIK, '-', 1.8),
        ('posterior A (cond means)', ab_post['_A_evr'], C_POST, '--', 1.4),
        ('posterior B (residuals)', ab_post['_B_evr'], C_POST, ':', 1.4),
        ('likelihood A (cond means)', ab_lik['_A_evr'], C_LIK, '--', 1.4),
        ('likelihood B (residuals)', ab_lik['_B_evr'], C_LIK, ':', 1.4),
    ]
    for name, evr, color, ls, lw in curves:
        c = np.cumsum(evr)
        ax.plot(np.arange(1, len(c) + 1), c, ls, color=color, lw=lw,
                label=name)
        # mark the k* where cumulative variance first hits VAR_THRESHOLD
        k = k_for_variance(evr)
        ax.plot([k], [c[k - 1]], 'o', color=color, ms=4.5,
                markeredgecolor='white', markeredgewidth=0.7, zorder=5)
    ax.axhline(VAR_THRESHOLD, color='0.4', lw=0.7, ls='-.', alpha=0.7)
    ax.text(25, VAR_THRESHOLD - 0.04, f'{int(VAR_THRESHOLD*100)}% var',
            fontsize=8, color='0.3', ha='right')
    ax.set_xlabel('# components')
    ax.set_ylabel('cumulative explained variance')
    ax.set_xlim(0.5, 25)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Scree — pooled (markers = k at 90% var)', loc='left')
    ax.legend(loc='lower right', ncol=2, columnspacing=1.0,
              handlelength=2.4)
    _stylise(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_alignment_heatmap(VA, VB, labels, title, out_path,
                            stats_line=None):
    """|VA^T VB| absolute-cosine grid between matched bases. Optionally
    annotate a single-line stats summary under the title."""
    M = np.abs(VA.T @ VB)
    kA, kB = M.shape
    fig_w = max(5.0, 0.32 * kB + 3.5)
    fig_h = max(4.4, 0.32 * kA + 3.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    im = ax.imshow(M, cmap='magma', vmin=0, vmax=1, aspect='auto')
    # annotate cells only when the grid is small enough to be legible.
    if max(kA, kB) <= 10:
        for i in range(kA):
            for j in range(kB):
                v = M[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if v < 0.55 else 'black')
    # tick density adapts to grid size; always label PC1, PC2, ...
    step_x = max(1, kB // 12)
    step_y = max(1, kA // 12)
    ax.set_xticks(np.arange(0, kB, step_x))
    ax.set_yticks(np.arange(0, kA, step_y))
    ax.set_xticklabels([f'PC{i+1}' for i in range(0, kB, step_x)],
                       fontsize=8)
    ax.set_yticklabels([f'PC{i+1}' for i in range(0, kA, step_y)],
                       fontsize=8)
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[0])
    full_title = title if stats_line is None else f'{title}\n{stats_line}'
    ax.set_title(full_title, loc='left', fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|cosine|')
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cca_bars(per_mouse_ccor, pooled_ccor, out_path,
                   pooled_p=None):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    max_k = max(len(v) for v in list(per_mouse_ccor.values()) + [pooled_ccor])
    # individual mice — pastel
    for mid, cc in per_mouse_ccor.items():
        ax.plot(np.arange(1, len(cc) + 1), cc, 'o-', ms=3, lw=0.9,
                alpha=0.45, label=f'mouse {mid}')
    ax.plot(np.arange(1, len(pooled_ccor) + 1), pooled_ccor, 's-',
            ms=6, lw=2.0, color='k', label='pooled')
    # across-mouse mean ± SEM
    M = max(len(v) for v in per_mouse_ccor.values())
    arr = np.full((len(per_mouse_ccor), M), np.nan)
    for i, cc in enumerate(per_mouse_ccor.values()):
        arr[i, :len(cc)] = cc
    mu = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(arr), axis=0))
    xs = np.arange(1, M + 1)
    ax.errorbar(xs, mu, yerr=sem, fmt='D--', color='0.25', ms=4,
                lw=1.2, capsize=3, label='mean ± SEM (mice)')
    ax.set_xlabel('canonical component')
    ax.set_ylabel('canonical correlation\n(posterior scores vs likelihood scores)')
    ax.set_ylim(0, 1.04)
    step = max(1, max_k // 20)
    ax.set_xticks(np.arange(1, max_k + 1, step))
    title = 'Analysis 1 — CCA of top-k PC scores (posterior ↔ likelihood)'
    if pooled_p is not None:
        # The p-value is the principal-angle null result for the pooled
        # subspace (NOT a CC-specific permutation); CC near 1 ⇔ small
        # principal angle, so this is the relevant reference.
        title += f'   (subspace-alignment perm-p = {pooled_p:.3g})'
    ax.set_title(title, loc='left')
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, -0.32))
    _stylise(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_signal_noise_summary(div_rows, null_pooled, out_path):
    """Bar chart: A-vs-B subspace overlap & mean angle, posterior vs
    likelihood, per scope. Adds per-mouse mean ± SEM and null bands."""
    df = pd.DataFrame(div_rows)
    scopes = list(dict.fromkeys(df['scope']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.38
    xc = np.arange(len(scopes))
    panels = [
        (axes[0], 'subspace_overlap_cos2',
         'A-vs-B subspace overlap',
         'overlap  (1 = identical subspace)',
         'overlap'),
        (axes[1], 'mean_principal_angle_deg',
         'A-vs-B mean principal angle',
         'degrees  (90 = orthogonal)',
         'mean_angle'),
    ]
    for ax, col, ttl, ylab, null_key in panels:
        for j, dist in enumerate(['posterior', 'likelihood']):
            vals = [df[(df['scope'] == s) & (df['distribution'] == dist)]
                    [col].values[0] for s in scopes]
            bar_color = C_POST if dist == 'posterior' else C_LIK
            ax.bar(xc + (j - 0.5) * width, vals, width,
                   color=bar_color, label=dist, edgecolor='k',
                   linewidth=0.4, alpha=0.9)
            # annotate value
            for xi, v in zip(xc, vals):
                ax.text(xi + (j - 0.5) * width, v + (0.015 if col.endswith('cos2')
                        else 1.5), f'{v:.2f}', ha='center', va='bottom',
                        fontsize=7, color=bar_color)
        # null shading from Haar-random k-dim subspaces (pooled k)
        if null_pooled is not None and null_key in null_pooled:
            null = null_pooled[null_key]
            lo, hi = np.percentile(null, [2.5, 97.5])
            med = np.median(null)
            ax.axhspan(lo, hi, color='0.85', alpha=0.6, zorder=0,
                       label='random k-dim null (95%)')
            ax.axhline(med, color='0.5', lw=0.7, ls=':', zorder=0)
        ax.set_xticks(xc)
        ax.set_xticklabels(scopes, rotation=15, fontsize=8)
        ax.set_title(ttl, loc='left')
        ax.set_ylabel(ylab)
        ax.legend(loc='best', framealpha=0.85)
        _stylise(ax)
    axes[1].axhline(90, color='0.4', lw=0.7, ls='--')
    # de-duplicate: only the first panel needs the null-band legend entry.
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    # keep only posterior/likelihood on the right panel to reduce clutter
    keep = [(h, l) for h, l in zip(h1, l1)
            if l in ('posterior', 'likelihood')]
    if keep:
        axes[1].legend([k[0] for k in keep], [k[1] for k in keep],
                       loc='lower right', framealpha=0.85)
    fig.suptitle('Analysis 3 — signal (condition-mean) vs noise '
                 '(residual) axis alignment', fontsize=11,
                 x=0.05, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# New: PC-loading curve plots
# --------------------------------------------------------------------------

def _plot_loading_overlay(ax, x, loads_by_label, ref=None, title=None,
                           xlabel=None, ylabel=None, show_zero=True):
    """Overlay multiple sign-aligned loadings on one axis.

    loads_by_label : iterable of (label, vector, color, linestyle).
    ref            : reference vector for sign alignment (or None).
    """
    if show_zero:
        ax.axhline(0, color='0.7', lw=0.6, ls='-')
    for label, vec, color, ls in loads_by_label:
        v = sign_align_to_ref(vec, ref) if ref is not None else sign_align_self(vec)
        ax.plot(x, v, ls, color=color, lw=1.6, label=label)
    if title:
        ax.set_title(title, loc='left', fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    _stylise(ax)


def plot_loadings_post_vs_lik(loadings_post, loadings_lik, kind, out_path):
    """3-panel figure: PC1/PC2/PC3 loadings as functions of orientation,
    with posterior and likelihood overlaid. `kind` ∈ {'A', 'B'}."""
    p = loadings_post.shape[1]
    x, xlab = _feature_axis(p)
    fig, axes = plt.subplots(1, N_LOADING_PCS, figsize=(11, 3.4),
                              sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        vp = loadings_post[i]
        vl = loadings_lik[i]
        # sign-align lik to post; sign-align post to itself first
        vp_a = sign_align_self(vp)
        cos = abs(np.dot(vp_a, vl)) / (np.linalg.norm(vp_a) *
                                        np.linalg.norm(vl) + 1e-12)
        _plot_loading_overlay(
            ax, x,
            [('posterior', vp_a, C_POST, '-'),
             ('likelihood', vl, C_LIK, '-')],
            ref=vp_a,
            title=f'PC{i+1}   |cos(post,lik)| = {cos:.2f}',
            xlabel=xlab,
            ylabel='loading (a.u.)' if i == 0 else None,
        )
    axes[0].legend(loc='upper right', fontsize=8)
    suptitle = {
        'A': 'Approach A — condition-mean PCA loadings (pooled)',
        'B': 'Approach B — trial-residual PCA loadings (pooled)',
        'all': 'All-trial PCA loadings (pooled)',
    }[kind]
    fig.suptitle(suptitle, fontsize=11, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def plot_loadings_A_vs_B(loadings_A, loadings_B, dist_name, out_path):
    """3-panel figure: PC1/PC2/PC3 loadings of A (signal) vs B (noise)
    for a single distribution."""
    p = loadings_A.shape[1]
    x, xlab = _feature_axis(p)
    fig, axes = plt.subplots(1, N_LOADING_PCS, figsize=(11, 3.4),
                              sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        va = sign_align_self(loadings_A[i])
        vb = loadings_B[i]
        cos = abs(np.dot(va, vb)) / (np.linalg.norm(va) *
                                      np.linalg.norm(vb) + 1e-12)
        _plot_loading_overlay(
            ax, x,
            [('A (cond-mean / signal)', va, C_A, '-'),
             ('B (residual / noise)', vb, C_B, '-')],
            ref=va,
            title=f'PC{i+1}   |cos(A,B)| = {cos:.2f}',
            xlabel=xlab,
            ylabel='loading (a.u.)' if i == 0 else None,
        )
    axes[0].legend(loc='upper right', fontsize=8)
    color = C_POST if dist_name == 'posterior' else C_LIK
    fig.suptitle(
        f'Signal vs noise loadings — {dist_name} (pooled)',
        fontsize=11, color=color, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def plot_loadings_per_mouse(loadings_by_mouse, kind, dist_name, out_path,
                             ref_per_pc=None):
    """Per-mouse small-multiples for the first 3 PCs (rows = mice,
    cols = PCs). `loadings_by_mouse` : dict mid -> (k, p) array; only
    first N_LOADING_PCS rows are used. Sign-aligns each row to ref_per_pc
    (typically the pooled loading) so curves stack cleanly."""
    mids = list(loadings_by_mouse.keys())
    if not mids:
        return
    p = next(iter(loadings_by_mouse.values())).shape[1]
    x, xlab = _feature_axis(p)
    fig, axes = plt.subplots(len(mids), N_LOADING_PCS,
                              figsize=(10, 1.5 * len(mids) + 0.6),
                              sharex=True, sharey=True)
    if len(mids) == 1:
        axes = np.array([axes])
    color = (C_POST if dist_name == 'posterior'
             else C_LIK if dist_name == 'likelihood'
             else '0.3')
    for r, mid in enumerate(mids):
        L = loadings_by_mouse[mid]
        for i in range(N_LOADING_PCS):
            ax = axes[r, i]
            ax.axhline(0, color='0.8', lw=0.5)
            if i < L.shape[0]:
                ref = ref_per_pc[i] if ref_per_pc is not None else None
                v = (sign_align_to_ref(L[i], ref) if ref is not None
                     else sign_align_self(L[i]))
                ax.plot(x, v, '-', color=color, lw=1.3)
            if r == 0:
                ax.set_title(f'PC{i+1}', loc='left', fontsize=9.5)
            if i == 0:
                ax.set_ylabel(f'mouse {mid}', fontsize=9)
            if r == len(mids) - 1:
                ax.set_xlabel(xlab, fontsize=8)
            _stylise(ax)
    label = {'A': 'Approach A (cond-mean)',
             'B': 'Approach B (residual)',
             'all': 'all-trial'}[kind]
    fig.suptitle(f'{label} loadings, per mouse — {dist_name}',
                 fontsize=11, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Across-mouse summary
# --------------------------------------------------------------------------

def across_mouse_summary(per_mouse_rows, null_dict):
    """Build a (metric, distribution) table of mean ± SEM and a
    one-sample Wilcoxon vs the Haar-null median.

    null_dict : {'mean_angle': np.ndarray, 'overlap': np.ndarray}
                — drawn at the relevant pooled k. Used to centre the
                Wilcoxon test against a chance reference.
    """
    df = pd.DataFrame(per_mouse_rows)
    rows = []
    for dist in df['distribution'].unique() if 'distribution' in df else [None]:
        sub = df if dist is None else df[df['distribution'] == dist]
        for col, null_key in [
            ('subspace_overlap_cos2', 'overlap'),
            ('mean_principal_angle_deg', 'mean_angle'),
        ]:
            if col not in sub:
                continue
            vals = sub[col].dropna().to_numpy()
            null_med = float(np.median(null_dict[null_key])) if null_key in null_dict else np.nan
            try:
                w_stat, w_p = wilcoxon(vals - null_med, zero_method='wilcox',
                                       alternative='two-sided')
            except Exception:
                w_stat, w_p = np.nan, np.nan
            rows.append({
                'distribution': dist,
                'metric': col,
                'n_mice': len(vals),
                'mean': float(np.mean(vals)),
                'sem': float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                       if len(vals) > 1 else 0.0,
                'null_median': null_med,
                'wilcoxon_stat': float(w_stat) if np.isfinite(w_stat) else np.nan,
                'wilcoxon_p_two_sided': float(w_p) if np.isfinite(w_p) else np.nan,
            })
    return pd.DataFrame(rows)


# ==========================================================================
# Entry point
# ==========================================================================

def main(mouse_ids=MOUSE_IDS, out_dir='pca_posterior_vs_likelihood_out'):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading…")
    data = load_all(mouse_ids)
    if not data:
        print("No mice loaded."); return

    # ---- Analysis 1: baseline similarity ---------------------------------
    base_rows = []
    per_mouse_ccor = {}
    for mid, d in data.items():
        r = baseline_similarity(d['post'], d['lik'], f'mouse_{mid}')
        base_rows.append(r)
        per_mouse_ccor[mid] = r['_ccor']

    # Pooled: stack all mice's trials.
    post_all = np.vstack([d['post'] for d in data.values()])
    lik_all = np.vstack([d['lik'] for d in data.values()])
    pooled_base = baseline_similarity(post_all, lik_all, 'pooled')
    base_rows.append(pooled_base)

    base_df = pd.DataFrame([{k: v for k, v in r.items()
                             if not k.startswith('_')} for r in base_rows])
    base_df.to_csv(os.path.join(out_dir, 'baseline_similarity.csv'),
                   index=False)

    # ---- Analysis 2 + 3: A vs B per distribution -------------------------
    # Pooled cells = (mouse, condition) so means/residuals stay within-mouse.
    def pooled_cond(d_map):
        cond = []
        for mid, d in d_map.items():
            cond.append(np.array([f"m{mid}|{c}" for c in d['cond']]))
        return np.concatenate(cond)

    pooled_cond_lbl = pooled_cond(data)

    div_rows = []
    ab_var_rows = []
    # store pooled AB models for figures
    pooled_ab = {}
    # also stash per-mouse AB model components so we can draw per-mouse
    # loading panels later (rows of components_, not the (n_feat, k) basis).
    per_mouse_ab_components = {'posterior': {}, 'likelihood': {}}
    for dist_name, get in [('posterior', 'post'), ('likelihood', 'lik')]:
        # per mouse
        for mid, d in data.items():
            m = ab_models(d[get], d['cond'])
            dv = ab_divergence(m, f'mouse_{mid}', dist_name)
            div_rows.append(dv)
            ab_var_rows.append({
                'scope': f'mouse_{mid}', 'distribution': dist_name,
                'A_k90': m['A_k90'], 'B_k90': m['B_k90'],
                'n_conditions': m['n_conditions'], 'n_trials': m['n_trials'],
            })
            per_mouse_ab_components[dist_name][mid] = {
                'A': m['A_components'], 'B': m['B_components'],
            }
        # pooled
        X_all = np.vstack([d[get] for d in data.values()])
        m = ab_models(X_all, pooled_cond_lbl)
        dv = ab_divergence(m, 'pooled', dist_name)
        # carry the raw PCA components_ on the divergence record so plots
        # downstream can access them by ['_A_components']/'B_components'.
        dv['_A_components'] = m['A_components']
        dv['_B_components'] = m['B_components']
        div_rows.append(dv)
        pooled_ab[dist_name] = dv
        ab_var_rows.append({
            'scope': 'pooled', 'distribution': dist_name,
            'A_k90': m['A_k90'], 'B_k90': m['B_k90'],
            'n_conditions': m['n_conditions'], 'n_trials': m['n_trials'],
        })

    div_df = pd.DataFrame([{k: v for k, v in r.items()
                            if not k.startswith('_')} for r in div_rows])
    div_df.to_csv(os.path.join(out_dir, 'cross_distribution_divergence.csv'),
                  index=False)
    pd.DataFrame(ab_var_rows).to_csv(
        os.path.join(out_dir, 'ab_pca_variance.csv'), index=False)

    # ---- Permutation (Haar-random) nulls --------------------------------
    # The null distribution for each comparison is "two random k-dim
    # subspaces of R^p", where k = the k_common used in the real
    # comparison and p = the feature dimensionality. This lets us assign
    # a p-value to 'subspace_overlap_cos2' (upper-tail) and
    # 'mean_principal_angle_deg' (lower-tail).
    print("Drawing Haar-random subspace nulls…")
    p_feat = pooled_base['_Vp'].shape[0]
    null_rows = []
    # Analysis 1: post-vs-lik pooled
    k1 = pooled_base['_Vp'].shape[1]
    null1 = principal_angle_null(p_feat, k1, n=N_PERM)
    null_rows.append({
        'analysis': 'post_vs_lik',
        'scope': 'pooled',
        'distribution': '-',
        'k': k1,
        'value_overlap': pooled_base['subspace_overlap_cos2'],
        'value_mean_angle_deg': pooled_base['mean_principal_angle_deg'],
        'null_median_overlap': float(np.median(null1['overlap'])),
        'null_median_mean_angle': float(np.median(null1['mean_angle'])),
        'p_overlap_greater': empirical_p_value(
            pooled_base['subspace_overlap_cos2'], null1['overlap'], 'upper'),
        'p_mean_angle_smaller': empirical_p_value(
            pooled_base['mean_principal_angle_deg'], null1['mean_angle'],
            'lower'),
    })
    # Analysis 3: A-vs-B per distribution, pooled
    null3 = {}
    for dist_name in ('posterior', 'likelihood'):
        dv = pooled_ab[dist_name]
        k3 = dv['_Va'].shape[1]
        nd = principal_angle_null(p_feat, k3, n=N_PERM)
        null3[dist_name] = nd
        null_rows.append({
            'analysis': 'A_vs_B',
            'scope': 'pooled',
            'distribution': dist_name,
            'k': k3,
            'value_overlap': dv['subspace_overlap_cos2'],
            'value_mean_angle_deg': dv['mean_principal_angle_deg'],
            'null_median_overlap': float(np.median(nd['overlap'])),
            'null_median_mean_angle': float(np.median(nd['mean_angle'])),
            'p_overlap_greater': empirical_p_value(
                dv['subspace_overlap_cos2'], nd['overlap'], 'upper'),
            'p_mean_angle_smaller': empirical_p_value(
                dv['mean_principal_angle_deg'], nd['mean_angle'], 'lower'),
        })
    perm_df = pd.DataFrame(null_rows)
    perm_df.to_csv(os.path.join(out_dir, 'permutation_nulls.csv'),
                   index=False)

    # ---- Across-mouse summary stats (Wilcoxon vs null median) ----------
    per_mouse_div_rows = [
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in div_rows if r['scope'].startswith('mouse_')
    ]
    # For Wilcoxon we use the average of the per-distribution null medians
    # (close enough — both use Haar-random k-dim subspaces of R^p).
    avg_null = {
        'overlap': np.concatenate([null3['posterior']['overlap'],
                                    null3['likelihood']['overlap']]),
        'mean_angle': np.concatenate([null3['posterior']['mean_angle'],
                                       null3['likelihood']['mean_angle']]),
    }
    across_df = across_mouse_summary(per_mouse_div_rows, avg_null)
    across_df.to_csv(os.path.join(out_dir, 'across_mouse_summary.csv'),
                     index=False)

    # ---- Figures ---------------------------------------------------------
    plot_scree(pooled_base, pooled_ab['posterior'], pooled_ab['likelihood'],
               os.path.join(fig_dir, 'fig1_scree_pooled.png'))

    # Analysis 1 alignment heatmap with stats line.
    a1_row = perm_df[perm_df['analysis'] == 'post_vs_lik'].iloc[0]
    stats1 = (f"mean angle = {a1_row['value_mean_angle_deg']:.1f}°  "
              f"(null {a1_row['null_median_mean_angle']:.1f}°,  "
              f"p={a1_row['p_mean_angle_smaller']:.3g});  "
              f"overlap cos² = {a1_row['value_overlap']:.2f}  "
              f"(p={a1_row['p_overlap_greater']:.3g})")
    plot_alignment_heatmap(
        pooled_base['_Vp'], pooled_base['_Vl'],
        ('posterior PC', 'likelihood PC'),
        'Analysis 1 — posterior vs likelihood loadings (pooled)',
        os.path.join(fig_dir, 'fig2_baseline_alignment_pooled.png'),
        stats_line=stats1)

    # Analysis 1 CCA bars (pooled top CC p-value reuses the same null
    # ordering: top CC saturates against random k-dim subspaces).
    plot_cca_bars(per_mouse_ccor, pooled_base['_ccor'],
                  os.path.join(fig_dir, 'fig3_cca_bars.png'),
                  pooled_p=float(a1_row['p_overlap_greater']))

    for dist_name, fname in [('posterior', 'fig4_posterior_AvsB_pooled.png'),
                              ('likelihood', 'fig5_likelihood_AvsB_pooled.png')]:
        dv = pooled_ab[dist_name]
        prow = perm_df[(perm_df['analysis'] == 'A_vs_B')
                       & (perm_df['distribution'] == dist_name)].iloc[0]
        sline = (f"mean angle = {prow['value_mean_angle_deg']:.1f}°  "
                 f"(null {prow['null_median_mean_angle']:.1f}°,  "
                 f"p={prow['p_mean_angle_smaller']:.3g});  "
                 f"overlap cos² = {prow['value_overlap']:.2f}  "
                 f"(p={prow['p_overlap_greater']:.3g})")
        plot_alignment_heatmap(
            dv['_Va'], dv['_Vb'],
            (f'{dist_name} A (cond-mean) PC',
             f'{dist_name} B (residual) PC'),
            f'Analysis 3 — {dist_name} signal vs noise (pooled)',
            os.path.join(fig_dir, fname),
            stats_line=sline)

    plot_signal_noise_summary(
        [{k: v for k, v in r.items() if not k.startswith('_')}
         for r in div_rows],
        null_pooled={
            'overlap': avg_null['overlap'],
            'mean_angle': avg_null['mean_angle'],
        },
        out_path=os.path.join(fig_dir, 'fig6_signal_noise_summary.png'))

    # ---- New: PC-loading curve figures ----------------------------------
    # Approach A (cond-mean) loadings, posterior & likelihood overlaid.
    plot_loadings_post_vs_lik(
        pooled_ab['posterior']['_A_components'],
        pooled_ab['likelihood']['_A_components'],
        kind='A',
        out_path=os.path.join(fig_dir, 'fig7_loadings_A_pooled.png'))
    # Approach B (residual) loadings, posterior & likelihood overlaid.
    plot_loadings_post_vs_lik(
        pooled_ab['posterior']['_B_components'],
        pooled_ab['likelihood']['_B_components'],
        kind='B',
        out_path=os.path.join(fig_dir, 'fig8_loadings_B_pooled.png'))
    # A vs B side-by-side per distribution.
    plot_loadings_A_vs_B(
        pooled_ab['posterior']['_A_components'],
        pooled_ab['posterior']['_B_components'],
        dist_name='posterior',
        out_path=os.path.join(fig_dir,
                              'fig9_loadings_AvsB_posterior_pooled.png'))
    plot_loadings_A_vs_B(
        pooled_ab['likelihood']['_A_components'],
        pooled_ab['likelihood']['_B_components'],
        dist_name='likelihood',
        out_path=os.path.join(fig_dir,
                              'fig10_loadings_AvsB_likelihood_pooled.png'))

    # Per-mouse small multiples — sign-aligned to the pooled reference.
    for dist_name in ('posterior', 'likelihood'):
        for kind in ('A', 'B'):
            ref = pooled_ab[dist_name][f'_{kind}_components']
            per_mouse_components = {
                mid: per_mouse_ab_components[dist_name][mid][kind]
                for mid in sorted(per_mouse_ab_components[dist_name].keys())
            }
            plot_loadings_per_mouse(
                per_mouse_components, kind=kind, dist_name=dist_name,
                out_path=os.path.join(
                    fig_dir,
                    f'fig11_loadings_{kind}_per_mouse_{dist_name}.png'),
                ref_per_pc=ref)

    # ---- Console summary -------------------------------------------------
    pd.set_option('display.width', 140)
    print("\n=== Analysis 1: baseline posterior-PCA vs likelihood-PCA ===")
    print(base_df[['scope', 'k_post_90', 'k_lik_90', 'k_common',
                   'mean_principal_angle_deg', 'subspace_overlap_cos2',
                   'mean_canonical_corr', 'mean_matched_loading_cos']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print("\n=== Analysis 2/3: A (cond-mean) vs B (residual) divergence ===")
    print(div_df[['scope', 'distribution', 'k_A_90', 'k_B_90', 'k_common',
                  'mean_principal_angle_deg', 'subspace_overlap_cos2']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    pooled = div_df[div_df['scope'] == 'pooled'].set_index('distribution')
    pa = pooled.loc['posterior', 'mean_principal_angle_deg']
    la = pooled.loc['likelihood', 'mean_principal_angle_deg']
    bigger = 'posterior' if pa > la else 'likelihood'
    print(f"\n=== Analysis 3 verdict (pooled) ===")
    print(f"  posterior  A-vs-B mean angle = {pa:.2f}°  "
          f"(overlap {pooled.loc['posterior','subspace_overlap_cos2']:.3f})")
    print(f"  likelihood A-vs-B mean angle = {la:.2f}°  "
          f"(overlap {pooled.loc['likelihood','subspace_overlap_cos2']:.3f})")
    print(f"  → signal/noise axes decouple MORE in the {bigger}.")

    print("\n=== Permutation (Haar-random) nulls ===")
    print(perm_df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print("\n=== Across-mouse summary (Wilcoxon vs random-subspace null) ===")
    print(across_df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

    print(f"\nFigures → {fig_dir}")


if __name__ == '__main__':
    main()
