# -*- coding: utf-8 -*-
"""DO the leading PCs really encode location and the trailing PCs width/shape?
(2026-06-05, Theo's challenge — show it, don't assert it; reworked 2026-06-10.)

The whole peakiness argument rests on a claim the report must demonstrate: the
target PCA basis is an ordered "where the bump sits" -> "how wide/spiky it is"
basis, so the evar weighting constrains location and frees shape. We give the
evidence on two target sets:

  * TOY targets — translated bumps of fixed width on the CIRCLE (the clean
    idealisation; PCA of a circular translation family is the Fourier basis, so
    PC0/PC1 are the fundamental cos/sin = a phase code, and the projection traces
    a full circle).
  * REAL V1 targets — the ideal-observer posteriors (Q, mouse 0). Orientation
    here is a **LINEAR 0-90 deg axis (NOT circular: 0 and 90 deg are maximally
    different)**, and many targets are **bimodal** (the IO is uncertain between
    0 and 90 deg) — so the real projection traces an arc, not a circle, coloured
    by a sequential (non-circular) map.

The figure (one per target set), 2x4 panels:
  a  explained-variance spectrum + the 90% location|shape split
  b  the first 7 PC waveforms (ordered low->high spatial frequency)
  c  THE DECISIVE TEST — per-PC *coefficient variance* when we sweep ONE factor:
     vary location (fixed width) -> variance concentrates in the LEADING PCs;
     vary width (fixed location) -> variance concentrates in the TRAILING PCs.
     ("coefficient variance for PC k" = Var over the sweep of <target, u_k>, i.e.
     how much PC k's projection moves as that factor changes; the factor's
     centre-of-mass over PCs is its 'address' in the basis.)
  d  the (PC0,PC1) projection coloured by location (a circle for the toy, an arc
     for the linear real axis) — location lives in the leading-PC plane
  e  moving along PC0 from the mean (+/- a*sigma0): what the leading axis *does*
  f  moving along PC1 from the mean (+/- a*sigma1)
  g  |coefficient| vs PC for a graded width sweep — narrower bump => more energy
     in the trailing PCs (the magnitude view that complements panel c's variance)
  h  reconstruction from the leading K PCs (K = 1,2,4,all): position needs few
     PCs, sharpness needs the trailing ones

Outputs (PNG+SVG) under figures/pc_geometry/:
  pc_location_vs_shape_toy.png   pc_location_vs_shape_real.png

Usage:  python diagnostics/pc_location_vs_shape.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

C = 91                                   # orientation bins (matches the pipeline)
LOC_COL = '#2171b5'                      # "vary location" — blue
WID_COL = '#e6550d'                      # "vary width"    — warm orange


def circular_bump(centre, width, C=C):
    """Wrapped-Gaussian bump on the ring (the toy's geometry)."""
    x = np.arange(C)
    d = np.minimum(np.abs(x - centre), C - np.abs(x - centre))
    p = np.exp(-0.5 * (d / width) ** 2)
    return p / p.sum()


def linear_bump(centre, width, C=C):
    """Gaussian bump on the LINEAR 0..C axis (no wrap — the real-V1 geometry)."""
    x = np.arange(C)
    p = np.exp(-0.5 * ((x - centre) / width) ** 2)
    return p / p.sum()


def dominant_freq(u):
    """Index of the peak of |FFT| (excluding DC) — the PC's spatial frequency."""
    amp = np.abs(np.fft.rfft(u - u.mean()))
    return int(np.argmax(amp[1:]) + 1)


def split_kloc(evar, frac=0.90):
    return int(np.searchsorted(np.cumsum(evar), frac)) + 1


def toy_target_set(n=4000, width=9.0, seed=0):
    """Translated bumps of FIXED width at locations uniform on the circle."""
    rng = np.random.default_rng(seed)
    locs = rng.uniform(0, C, size=n)
    T = np.stack([circular_bump(s, width) for s in locs])
    return T.astype(np.float64), locs


def coeff_variance_curves(pca, bump, width0, loc0, loc_range, wid_range):
    """Per-PC variance of the projection coefficient when we sweep ONE factor:
    location at fixed width, or width at fixed location. The factor's 'address' in
    PC space is where its variance curve concentrates."""
    locs = np.linspace(*loc_range, 200)
    widths = np.linspace(*wid_range, 200)
    T_loc = np.stack([bump(s, width0) for s in locs])
    T_wid = np.stack([bump(loc0, w) for w in widths])
    return pca.transform(T_loc).var(axis=0), pca.transform(T_wid).var(axis=0)


def fig_demo(T, locs, title, stem, out_dir, *, circular):
    pca = PCA().fit(T)
    pcs, evar = pca.components_, pca.explained_variance_ratio_
    sig = np.sqrt(pca.explained_variance_)            # coeff std per PC
    kloc = split_kloc(evar)
    coeffs = pca.transform(T)
    peak = locs if (locs is not None and circular) else np.argmax(T, axis=1)
    bump = circular_bump if circular else linear_bump
    loc_range = (0, C) if circular else (8, C - 8)    # linear: keep off the edges
    wid_range = (3.0, 20.0)
    cmap = 'hsv' if circular else 'viridis'           # circular vs sequential (0!=90 deg)
    x = np.arange(C)

    fig, axes = plt.subplots(2, 4, figsize=ps.figsize(4, 2))

    # (a) evar spectrum + location|shape split
    ax = axes[0, 0]
    k = np.arange(1, min(30, len(evar)) + 1)
    ax.semilogy(k, evar[:len(k)] + 1e-12, color='0.3', marker='o', ms=3, lw=1.6)
    ax.axvspan(0.5, kloc + 0.5, color=LOC_COL, alpha=0.12)
    ax.axvspan(kloc + 0.5, len(k) + 0.5, color=WID_COL, alpha=0.12)
    ax.axvline(kloc + 0.5, color='k', ls='--', lw=1.2)
    ax.text(kloc / 2 + 0.5, evar[0], f'location\n(≤PC{kloc}, ≥90%)', color=LOC_COL,
            ha='center', va='top', fontsize=8.5)
    ax.text((kloc + len(k)) / 2, evar[0], 'shape', color=WID_COL, ha='center',
            va='top', fontsize=8.5)
    ax.set_xlabel('PC index  k'); ax.set_ylabel('explained-variance ratio  evar$_k$')
    ax.set_title('Explained-variance spectrum')

    # (b) the first 7 PC waveforms (ordered low->high frequency)
    ax = axes[0, 1]
    npc = 7
    cols = plt.get_cmap('viridis')(np.linspace(0, 0.85, npc))
    off = 0.0
    for j in range(min(npc, len(pcs))):
        u = pcs[j] / np.abs(pcs[j]).max() * 0.4
        ax.plot(x, u + off, color=cols[j], lw=1.5)
        ax.text(C + 1, off, f'PC{j} (f={dominant_freq(pcs[j])})', color=cols[j],
                va='center', fontsize=7.5)
        off -= 1.0
    ax.set_yticks([]); ax.set_xlabel('orientation bin')
    ax.set_xlim(0, C + 24)
    ax.set_title('First 7 PCs: low→high frequency')

    # (c) decisive test: coefficient variance under a location vs width sweep
    ax = axes[0, 2]
    var_loc, var_wid = coeff_variance_curves(pca, bump, 9.0, C / 2, loc_range, wid_range)
    kk = np.arange(1, min(30, len(evar)) + 1)
    ax.plot(kk, var_loc[:len(kk)] / var_loc.sum() + 1e-9, color=LOC_COL, lw=2.2,
            marker='o', ms=3, label='vary location (fixed width)')
    ax.plot(kk, var_wid[:len(kk)] / var_wid.sum() + 1e-9, color=WID_COL, lw=2.2,
            marker='s', ms=3, label='vary width (fixed location)')
    ax.axvline(kloc + 0.5, color='k', ls='--', lw=1.2)
    ax.set_yscale('log'); ax.set_xlabel('PC index  k')
    ax.set_ylabel('coeff. variance over the sweep (norm.)')
    idx = np.arange(1, len(var_loc) + 1)
    com_l = (idx * var_loc).sum() / var_loc.sum()
    com_w = (idx * var_wid).sum() / var_wid.sum()
    ax.set_title(f'Decisive test: location PC#{com_l:.1f} vs width PC#{com_w:.1f}')
    ax.legend(fontsize=7.5, loc='upper right')

    # (d) (PC0,PC1) projection coloured by location — circle (toy) / arc (real)
    ax = axes[0, 3]
    sc = ax.scatter(coeffs[:, 0], coeffs[:, 1], c=peak, cmap=cmap, s=6, alpha=0.6)
    ax.set_xlabel('projection on PC0'); ax.set_ylabel('projection on PC1')
    ax.set_aspect('equal', 'box')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('target peak bin')
    ax.set_title('(PC0,PC1) ' + ('circle' if circular else 'arc') + ' = location')

    # (e,f) moving along PC0 / PC1 from the mean (+/- a*sigma): graded
    amps = np.linspace(-2.5, 2.5, 9)
    grad = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(amps)))
    for col_i, (pc_i, axe) in enumerate(((0, axes[1, 0]), (1, axes[1, 1]))):
        for a, gc in zip(amps, grad):
            axe.plot(x, pca.mean_ + a * sig[pc_i] * pcs[pc_i], color=gc, lw=1.3)
        axe.plot(x, pca.mean_, color='k', lw=1.6, ls='--', label='mean')
        axe.set_xlabel('orientation bin'); axe.set_yticks([])
        axe.set_title(f'mean + a·σ·PC{pc_i}  (a: −2.5→2.5)')
        if col_i == 0:
            axe.legend(fontsize=7.5, loc='upper right')

    # (g) width sweep -> trailing-PC energy (graded; the magnitude view of c)
    ax = axes[1, 2]
    widths = np.linspace(4.0, 18.0, 7)
    wcols = plt.get_cmap('YlOrRd')(np.linspace(0.35, 0.95, len(widths)))
    for w, wc in zip(widths, wcols):
        c = pca.transform(bump(C / 2, w)[None])[0]
        kk = np.arange(1, min(30, len(c)) + 1)
        ax.plot(kk, np.abs(c[:len(kk)]) + 1e-6, color=wc, lw=1.5, marker='o', ms=2)
    ax.axvline(kloc + 0.5, color='k', ls='--', lw=1.2)
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(widths.min(), widths.max()))
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label('bump width')
    ax.set_yscale('log'); ax.set_xlabel('PC index  k')
    ax.set_ylabel('|coefficient| (fixed location)')
    ax.set_title('Narrower bump → trailing-PC energy')

    # (h) reconstruction from leading K PCs: K = 1,2,4,all
    ax = axes[1, 3]
    tr = int(np.argmin(np.abs(peak - (C / 2 if not circular else peak.mean()))))
    Ks = [1, 2, 4, len(pcs)]
    kcols = plt.get_cmap('plasma')(np.linspace(0.1, 0.85, len(Ks)))
    ps.target_band(ax, x, T[tr], label='target')
    for K, kc in zip(Ks, kcols):
        rec = pca.mean_ + coeffs[tr, :K] @ pcs[:K]
        ax.plot(x, rec, color=kc, lw=1.6, label=f'{K if K < len(pcs) else "all"} PCs')
    ax.set_xlabel('orientation bin'); ax.set_yticks([])
    ax.set_title('Reconstruction: leading K PCs')
    ax.legend(fontsize=7.5, loc='best')

    ps.label_panels(axes)
    fig.suptitle(title)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, stem)
    print(f'  {stem}: kloc(90%)={kloc}, n_pc={len(evar)}, '
          f'location CoM PC#{com_l:.1f}, width CoM PC#{com_w:.1f}, '
          f'PC0/PC1 freq={dominant_freq(pcs[0])}/{dominant_freq(pcs[1])}')


def real_target_set(results_root, run, slug, split, mouse, arch):
    mat = Path(results_root) / run / slug / f'{split}.mat'
    d = sio.loadmat(str(mat), simplify_cells=True)['results'][f'mouse_{mouse}']
    return np.asarray(d['Dist'][arch]['target'], float)


def main(results_root, out_root):
    ps.apply()
    out_dir = Path(out_root)

    T, locs = toy_target_set()
    fig_demo(T, locs, 'PC basis: location vs shape — toy targets (circular)',
             'pc_location_vs_shape_toy', out_dir, circular=True)

    try:
        T_real = real_target_set(results_root, 'wm3', 'Q_PCA_half_100ms_all',
                                 'stratified_balanced', 0, 'spat')
        fig_demo(T_real, None, 'PC basis on real V1 targets (Q, mouse 0; linear 0–90°)',
                 'pc_location_vs_shape_real', out_dir, circular=False)
    except Exception as e:
        print(f'  [real] skipped: {e}')
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/pc_geometry')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
