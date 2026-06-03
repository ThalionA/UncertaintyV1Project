# -*- coding: utf-8 -*-
"""Per-trial spatial-vs-temporal loss scatter, per loss function, coloured by
stimulus / uncertainty features.

For each loss function it draws one figure: per-trial PPC (spatial) loss on x,
SBC (temporal) loss on y, with the y=x diagonal. Points below the diagonal are
trials the temporal decoder fits better. The figure has one subpanel per
feature, the SAME scatter recoloured by that feature, so you can read whether
the spat/temp advantage is structured by orientation, contrast, dispersion or
perceptual uncertainty.

Two value modes (both written):
  * ``raw``  — the loss under each decoder's OWN training metric, per trial.
  * ``norm`` — each trial's loss divided by that mouse's shuffle-decoder mean
               (per arch), so 1.0 = chance; removes per-mouse scale so trials
               pool cleanly across mice.

Scoring uses each loss's own metric (``nn_classifier._batched_fit_loss``); pass
``--metric PCA`` to score every loss on the common PCA-distance yardstick
instead.

Features (per trial, row-aligned with the decoded posteriors):
  orientation, contrast, dispersion (from ``trials``), and perceptual
  uncertainty = entropy of the target posterior.

Usage
-----
    python loss_scatter_spat_temp.py --run-name loss_comparison_v1
    python loss_scatter_spat_temp.py --run-name loss_comparison_v1 \
        --target L --window full --bin 50 --split generalize_contrast --metric PCA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_loss_sweep as P
from nn_classifier import _batched_fit_loss

LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')


def _per_trial_loss(decoded, target, metric, pcs, evar):
    """(n_trials,) per-trial loss under ``metric`` (own loss, or 'PCA')."""
    dec = np.asarray(decoded, float)
    tgt = np.asarray(target, float)
    pcs_t = evar_t = None
    if metric == 'PCA':
        pcs_t = torch.tensor(np.asarray(pcs, float))
        evar_t = torch.tensor(np.asarray(evar, float))
    out = _batched_fit_loss(torch.tensor(dec), torch.tensor(tgt), metric,
                            pcs_t, evar_t)
    return out.cpu().numpy()


def _target_entropy(target):
    t = np.asarray(target, float)
    t = t / np.clip(t.sum(1, keepdims=True), 1e-12, None)
    return -np.sum(t * np.log(t + 1e-12), axis=1)


# (label, colourmap, log-colour) per feature. The extractor is applied per mouse.
FEATURES = [
    ('orientation', 'twilight', False, lambda md: np.asarray(md['trials']['orientation'], float)),
    ('contrast', 'viridis', True, lambda md: np.asarray(md['trials']['contrast'], float)),
    ('dispersion', 'plasma', False, lambda md: np.asarray(md['trials']['dispersion'], float)),
    ('perceptual uncertainty\n(target entropy)', 'cividis', False,
     lambda md: _target_entropy(md['Dist']['spat']['target'])),
]


def _collect(mat, metric, value):
    """Pool per-trial (spat, temp, features) across mice for one loss cell."""
    spat, temp = [], []
    feats = {lab: [] for lab, _, _, _ in FEATURES}
    for mid, md in mat['results'].items():
        d = md['Dist']
        pcs, evar = d.get('pcs'), d.get('explained_var')
        s = _per_trial_loss(d['spat']['decoded'], d['spat']['target'], metric, pcs, evar)
        t = _per_trial_loss(d['temp']['decoded'], d['temp']['target'], metric, pcs, evar)
        if value == 'norm':
            s_sh = _per_trial_loss(d['spat_shf']['decoded'], d['spat_shf']['target'], metric, pcs, evar)
            t_sh = _per_trial_loss(d['temp_shf']['decoded'], d['temp_shf']['target'], metric, pcs, evar)
            sm, tm = np.nanmean(s_sh), np.nanmean(t_sh)
            s = s / sm if sm > 0 else s
            t = t / tm if tm > 0 else t
        spat.append(s); temp.append(t)
        for lab, _, _, fn in FEATURES:
            feats[lab].append(np.asarray(fn(md), float))
    spat = np.concatenate(spat); temp = np.concatenate(temp)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    return spat, temp, feats


def _feature_groups(c, max_levels=6):
    """Split a feature vector into ordered groups for the density row.

    Categorical (<= max_levels unique) → one group per level; otherwise →
    quartile bins. Returns list of (label, mask, representative_value) in
    ascending value order, plus (vmin, vmax) for colour mapping.
    """
    c = np.asarray(c, float)
    uniq = np.unique(c[np.isfinite(c)])
    groups = []
    if uniq.size <= max_levels:
        for u in uniq:
            groups.append((f"{u:g}", c == u, float(u)))
    else:
        qs = np.nanpercentile(c, [0, 25, 50, 75, 100])
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        for i in range(4):
            lo_e, hi_e = qs[i], qs[i + 1]
            mask = (c >= lo_e) & (c <= hi_e if i == 3 else c < hi_e)
            groups.append((labels[i], mask, float((lo_e + hi_e) / 2)))
    vals = [g[2] for g in groups]
    return groups, (min(vals), max(vals))


def _density_panel(ax, spat, temp, c, cmap, logc, lo, hi):
    """Per-group 2D-KDE contour overlay: one contour set per feature level,
    coloured on the same scale as the scatter, so the cloud's SHIFT with the
    feature is visible. Faint hexbin backdrop gives the overall density."""
    from scipy.stats import gaussian_kde
    ax.hexbin(spat, temp, gridsize=30, extent=(lo, hi, lo, hi),
              cmap='Greys', mincnt=1, alpha=0.30, linewidths=0)
    groups, (vmin, vmax) = _feature_groups(c)
    cnorm = (matplotlib.colors.LogNorm(vmin=max(vmin, 1e-9), vmax=vmax)
             if logc else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cm = matplotlib.colormaps[cmap]
    gx, gy = np.mgrid[lo:hi:80j, lo:hi:80j]
    grid = np.vstack([gx.ravel(), gy.ravel()])
    for glab, mask, gval in groups:
        xs, ys = spat[mask], temp[mask]
        if xs.size < 15 or np.ptp(xs) < 1e-9 or np.ptp(ys) < 1e-9:
            continue
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
        except Exception:
            continue
        z = kde(grid).reshape(gx.shape)
        ax.contour(gx, gy, z, levels=[z.max() * 0.5], linewidths=1.8,
                   colors=[cm(cnorm(gval))])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('spatial (PPC) loss')
    ax.set_ylabel('temporal (SBC) loss')
    return matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)


def plot_loss(mat, loss, metric, value, out_dir: Path):
    spat, temp, feats = _collect(mat, metric, value)
    ok = np.isfinite(spat) & np.isfinite(temp)
    spat, temp = spat[ok], temp[ok]
    n = len(FEATURES)
    # Two rows: scatter (top) + per-feature density contours (bottom), same axes.
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.4), squeeze=False)
    lo = float(min(spat.min(), temp.min()))
    hi = float(max(np.percentile(spat, 99.5), np.percentile(temp, 99.5)))
    frac_temp = float(np.mean(temp < spat))   # below diagonal = SBC better
    for j, (lab, cmap, logc, _) in enumerate(FEATURES):
        c = feats[lab][ok]
        # Row 0 — scatter coloured by feature.
        ax = axes[0, j]
        norm = matplotlib.colors.LogNorm() if logc else None
        sc = ax.scatter(spat, temp, c=c, cmap=cmap, norm=norm, s=8,
                        alpha=0.6, linewidths=0)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel('spatial (PPC) loss')
        ax.set_ylabel('temporal (SBC) loss')
        ax.set_title(lab, fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # Row 1 — per-feature-level density contours. Its own tighter limits
        # (the density bulk sits near the origin; the scatter's wide range
        # would squash the contours), kept square so y=x stays meaningful.
        dlo = float(min(np.percentile(spat, 1), np.percentile(temp, 1)))
        dhi = float(max(np.percentile(spat, 92), np.percentile(temp, 92)))
        sm = _density_panel(axes[1, j], spat, temp, c, cmap, logc, dlo, dhi)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[1, j], fraction=0.046, pad=0.04)
        axes[1, j].set_title(f'{lab} — density by level', fontsize=9)
    unit = ('loss / shuffle (1 = chance)' if value == 'norm'
            else f'{metric} loss')
    fig.suptitle(f'{loss} — per-trial spatial vs temporal ({unit})   '
                 f'[{frac_temp*100:.0f}% of trials below diagonal → SBC better]'
                 f'\ntop: scatter coloured by feature   bottom: 50%-density '
                 f'contour per feature level (zoomed to the bulk)',
                 y=1.01, fontsize=12)
    fig.tight_layout()
    stem = f'13_scatter_spat_temp_{loss}_{value}'
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {stem}.png/.svg  ({frac_temp*100:.0f}% below diagonal)")


def main(run_name, target='Q', window='half', bin_ms=100,
         split='stratified_balanced', metric='own', losses=LOSSES,
         values=('raw', 'norm'), out_root='figures/loss_sweep_plots',
         results_root='results'):
    cell = f'{target}_{window}_{bin_ms}ms_{split}'
    out_dir = Path(out_root) / run_name / cell / 'scatter'
    out_dir.mkdir(parents=True, exist_ok=True)
    from scipy.io import loadmat
    print(f"Spat/temp loss scatter: {run_name} | {target} {window} {bin_ms}ms "
          f"{split} | metric={metric}")
    for loss in losses:
        slug = (f"{target}_{loss}_{window}_{bin_ms}ms"
                + ('_all' if loss == 'PCA' else ''))
        path = Path(results_root) / run_name / slug / f'{split}.mat'
        if not path.exists():
            print(f"  [skip] {loss}: missing {path}")
            continue
        mat = loadmat(str(path), simplify_cells=True)
        m = metric if metric != 'own' else loss
        for value in values:
            plot_loss(mat, loss, m, value, out_dir)
    print(f"Done. {out_dir.resolve()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--metric', default='own',
                    help="'own' (each loss's training metric, default) or a "
                         "fixed metric name, e.g. 'PCA'.")
    ap.add_argument('--losses', nargs='+', default=list(LOSSES))
    ap.add_argument('--values', nargs='+', default=['raw', 'norm'],
                    choices=['raw', 'norm'])
    a = ap.parse_args()
    main(run_name=a.run_name, target=a.target, window=a.window, bin_ms=a.bin_ms,
         split=a.split, metric=a.metric, losses=tuple(a.losses),
         values=tuple(a.values))
