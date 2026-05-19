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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# utils_v26 pulls torch via siblings; import lazily inside main().

S_GRID = np.arange(0, 91, 1)
VAR_THRESHOLD = 0.90
MOUSE_IDS = (0, 1, 2, 3, 4, 5)


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


def plot_scree(baseline, ab_post, ab_lik, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    curves = [
        ('posterior (all-trial)', baseline['_evr_post'], C_POST, '-'),
        ('likelihood (all-trial)', baseline['_evr_lik'], C_LIK, '-'),
        ('posterior A (cond means)', ab_post['_A_evr'], C_POST, '--'),
        ('posterior B (residuals)', ab_post['_B_evr'], C_POST, ':'),
        ('likelihood A (cond means)', ab_lik['_A_evr'], C_LIK, '--'),
        ('likelihood B (residuals)', ab_lik['_B_evr'], C_LIK, ':'),
    ]
    for name, evr, color, ls in curves:
        c = np.cumsum(evr)
        ax.plot(np.arange(1, len(c) + 1), c, ls, color=color, lw=1.6,
                label=name)
    ax.axhline(VAR_THRESHOLD, color='k', lw=0.7, ls='-.', alpha=0.6)
    ax.text(1, VAR_THRESHOLD + 0.01, f'{int(VAR_THRESHOLD*100)}%',
            fontsize=8)
    ax.set_xlabel('# components')
    ax.set_ylabel('cumulative explained variance')
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 1.02)
    ax.set_title('Scree — pooled')
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alignment_heatmap(VA, VB, labels, title, out_path):
    """|VA^T VB| absolute-cosine grid between matched bases."""
    M = np.abs(VA.T @ VB)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap='magma', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel(f'{labels[1]} component')
    ax.set_ylabel(f'{labels[0]} component')
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label='|cosine|', fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cca_bars(per_mouse_ccor, pooled_ccor, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    max_k = max(len(v) for v in list(per_mouse_ccor.values()) + [pooled_ccor])
    x = np.arange(max_k)
    for mid, cc in per_mouse_ccor.items():
        ax.plot(np.arange(len(cc)), cc, 'o-', ms=3, lw=0.8, alpha=0.5,
                label=f'mouse {mid}')
    ax.plot(np.arange(len(pooled_ccor)), pooled_ccor, 's-', ms=6, lw=2.2,
            color='k', label='pooled')
    ax.set_xlabel('canonical component')
    ax.set_ylabel('canonical correlation (post scores vs lik scores)')
    ax.set_ylim(0, 1.02)
    ax.set_title('Analysis 1 — CCA of top-k PC scores')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signal_noise_summary(div_rows, out_path):
    """Bar chart: A-vs-B subspace overlap & mean angle, posterior vs
    likelihood, per scope."""
    df = pd.DataFrame(div_rows)
    scopes = list(dict.fromkeys(df['scope']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.38
    xc = np.arange(len(scopes))
    for ax, col, ttl, ylab in [
        (axes[0], 'subspace_overlap_cos2',
         'A-vs-B subspace overlap (mean cos²)', 'overlap (1=aligned)'),
        (axes[1], 'mean_principal_angle_deg',
         'A-vs-B mean principal angle', 'degrees (90=orthogonal)'),
    ]:
        for j, dist in enumerate(['posterior', 'likelihood']):
            vals = [df[(df['scope'] == s) & (df['distribution'] == dist)]
                    [col].values[0] for s in scopes]
            ax.bar(xc + (j - 0.5) * width, vals, width,
                   color=C_POST if dist == 'posterior' else C_LIK,
                   label=dist, edgecolor='k', linewidth=0.4)
        ax.set_xticks(xc)
        ax.set_xticklabels(scopes, rotation=20, fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(fontsize=8)
    if 'mean_principal_angle_deg':
        axes[1].axhline(90, color='gray', lw=0.7, ls='--')
    fig.suptitle('Analysis 3 — signal (cond-mean) vs noise (residual) '
                 'axis alignment', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
        # pooled
        X_all = np.vstack([d[get] for d in data.values()])
        m = ab_models(X_all, pooled_cond_lbl)
        dv = ab_divergence(m, 'pooled', dist_name)
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

    # ---- Figures ---------------------------------------------------------
    plot_scree(pooled_base, pooled_ab['posterior'], pooled_ab['likelihood'],
               os.path.join(fig_dir, 'fig1_scree_pooled.png'))
    plot_alignment_heatmap(
        pooled_base['_Vp'], pooled_base['_Vl'],
        ('posterior', 'likelihood'),
        'Analysis 1 — posterior vs likelihood loadings (pooled)',
        os.path.join(fig_dir, 'fig2_baseline_alignment_pooled.png'))
    plot_cca_bars(per_mouse_ccor, pooled_base['_ccor'],
                  os.path.join(fig_dir, 'fig3_cca_bars.png'))
    plot_alignment_heatmap(
        pooled_ab['posterior']['_Va'], pooled_ab['posterior']['_Vb'],
        ('posterior A (cond-mean)', 'posterior B (residual)'),
        'Analysis 3 — posterior signal vs noise (pooled)',
        os.path.join(fig_dir, 'fig4_posterior_AvsB_pooled.png'))
    plot_alignment_heatmap(
        pooled_ab['likelihood']['_Va'], pooled_ab['likelihood']['_Vb'],
        ('likelihood A (cond-mean)', 'likelihood B (residual)'),
        'Analysis 3 — likelihood signal vs noise (pooled)',
        os.path.join(fig_dir, 'fig5_likelihood_AvsB_pooled.png'))
    plot_signal_noise_summary(
        [{k: v for k, v in r.items() if not k.startswith('_')}
         for r in div_rows],
        os.path.join(fig_dir, 'fig6_signal_noise_summary.png'))

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
    print(f"\nFigures → {fig_dir}")


if __name__ == '__main__':
    main()
