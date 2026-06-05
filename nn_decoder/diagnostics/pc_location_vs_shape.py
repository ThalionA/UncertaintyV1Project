# -*- coding: utf-8 -*-
"""DO the leading PCs really encode location and the trailing PCs width/shape?
(2026-06-05, Theo's challenge — show it, don't assert it.)

The whole peakiness argument rests on a claim the report never demonstrates: that
the target PCA basis is an ordered "where the bump sits" -> "how wide/spiky it is"
basis, so the evar weighting constrains location and frees shape. That claim is
provable (the targets are translated bumps; PCA of a translation family is an
ordered low->high spatial-frequency basis, i.e. discrete Fourier modes, with power
= the bump's spectrum), but a derivation isn't evidence. This script gives the
evidence, two ways, on the toy target set and on the real V1 targets:

  1. WAVEFORMS + SPECTRUM. The PCs themselves, plotted: leading ones are smooth
     low-frequency modes, trailing ones high-frequency wiggles. evar collapses fast;
     the 90% line is the location|shape split the report uses.

  2. THE DECISIVE TEST — which PCs move when you vary each factor?
       * vary LOCATION at fixed width  -> coefficient variance concentrates in the
         LEADING PCs;
       * vary WIDTH at fixed location  -> coefficient variance concentrates in the
         TRAILING PCs.
     Plotted on one axis, the two curves separate at (about) the 90% split — direct
     proof that location lives in the high-evar PCs and width/shape in the low-evar
     ones. A companion panel shows the (PC0,PC1) projection tracing a CIRCLE as
     location sweeps (the leading PCs are the fundamental cos/sin = a phase code for
     position), and a reconstruction panel shows leading-K PCs recover position but
     not sharpness (sharpness needs the trailing PCs).

Outputs (PNG+SVG) under figures/pc_geometry/:
  pc_location_vs_shape_toy.png   full controlled demonstration on the toy targets
  pc_location_vs_shape_real.png  the same basis facts on the real V1 targets

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
    x = np.arange(C)
    d = np.minimum(np.abs(x - centre), C - np.abs(x - centre))
    p = np.exp(-0.5 * (d / width) ** 2)
    return p / p.sum()


def dominant_freq(u):
    """Index of the peak of |FFT| (excluding DC) — the PC's spatial frequency."""
    amp = np.abs(np.fft.rfft(u - u.mean()))
    return int(np.argmax(amp[1:]) + 1)


def split_kloc(evar, frac=0.90):
    return int(np.searchsorted(np.cumsum(evar), frac)) + 1


# ----------------------------------------------------------------------

def toy_target_set(n=4000, width=9.0, seed=0):
    """Translated bumps of FIXED width at locations uniform on the circle —
    exactly the toy's target family."""
    rng = np.random.default_rng(seed)
    locs = rng.uniform(0, C, size=n)
    T = np.stack([circular_bump(s, width) for s in locs])
    return T.astype(np.float64), locs


def coeff_variance_curves(pca, width0=9.0, loc0=C / 2):
    """Per-PC variance of the projection coefficient when we sweep ONE factor:
    (a) location at fixed width, (b) width at fixed location. The factor's
    'address' in PC space is where its curve peaks."""
    locs = np.linspace(0, C, 200, endpoint=False)
    T_loc = np.stack([circular_bump(s, width0) for s in locs])
    widths = np.linspace(3.0, 20.0, 200)
    T_wid = np.stack([circular_bump(loc0, w) for w in widths])
    var_loc = pca.transform(T_loc).var(axis=0)
    var_wid = pca.transform(T_wid).var(axis=0)
    return var_loc, var_wid


def fig_demo(T, locs, title, stem, out_dir, real=False):
    pca = PCA().fit(T)
    pcs, evar = pca.components_, pca.explained_variance_ratio_
    kloc = split_kloc(evar)
    coeffs = pca.transform(T)                       # centred projection coefficients
    peak = np.argmax(T, axis=1) if real else locs   # colour-by-location

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (A) evar spectrum + cumulative, with the 90% location|shape split
    ax = axes[0, 0]
    k = np.arange(1, min(40, len(evar)) + 1)
    ax.semilogy(k, evar[:len(k)] + 1e-12, color='0.3', marker='o', ms=3, lw=1.6)
    ax.axvspan(0.5, kloc + 0.5, color=LOC_COL, alpha=0.10)
    ax.axvspan(kloc + 0.5, len(k) + 0.5, color=WID_COL, alpha=0.10)
    ax.axvline(kloc + 0.5, color='k', ls='--', lw=1.2)
    ax.text(kloc / 2 + 0.5, evar[0], 'location\n(≥90% var)', color=LOC_COL,
            ha='center', va='top', fontsize=9)
    ax.text((kloc + len(k)) / 2, evar[0], 'shape', color=WID_COL, ha='center',
            va='top', fontsize=9)
    ax.set_xlabel('PC index  k'); ax.set_ylabel('explained-variance ratio  evar$_k$')
    ax.set_title('Spectrum collapses → loss weights only the leading PCs')

    # (B) PC waveforms: leading (smooth) vs trailing (wiggly)
    ax = axes[0, 1]
    x = np.arange(C)
    show = [(0, LOC_COL), (1, LOC_COL), (2, LOC_COL),
            (kloc, WID_COL), (min(kloc + 6, len(pcs) - 1), WID_COL)]
    off = 0.0
    for j, col in show:
        u = pcs[j]
        u = u / np.abs(u).max() * 0.4
        ax.plot(x, u + off, color=col, lw=1.6)
        ax.text(C + 1, off, f'PC{j} (freq {dominant_freq(pcs[j])})', color=col,
                va='center', fontsize=8)
        off -= 1.0
    ax.set_yticks([]); ax.set_xlabel('orientation bin')
    ax.set_title('PCs are ordered low→high spatial frequency')
    ax.set_xlim(0, C + 22)

    # (C) THE DECISIVE TEST: where does each factor's variation live?
    ax = axes[0, 2]
    var_loc, var_wid = coeff_variance_curves(pca)
    kk = np.arange(1, min(40, len(evar)) + 1)
    vl = var_loc[:len(kk)] / var_loc.sum()
    vw = var_wid[:len(kk)] / var_wid.sum()
    ax.plot(kk, vl + 1e-9, color=LOC_COL, lw=2.2, marker='o', ms=3,
            label='vary LOCATION (fixed width)')
    ax.plot(kk, vw + 1e-9, color=WID_COL, lw=2.2, marker='s', ms=3,
            label='vary WIDTH (fixed location)')
    ax.axvline(kloc + 0.5, color='k', ls='--', lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('PC index  k')
    ax.set_ylabel('coeff. variance across the sweep (norm.)')
    ax.set_title('Location → leading PCs;  width → trailing PCs')
    ax.legend(fontsize=8, loc='upper right')
    idx = np.arange(1, len(var_loc) + 1)
    com_loc = (idx * var_loc).sum() / var_loc.sum()
    com_wid = (idx * var_wid).sum() / var_wid.sum()
    ax.text(0.03, 0.03, f'centre-of-mass:  location PC#{com_loc:.1f}   |   '
            f'width PC#{com_wid:.1f}', transform=ax.transAxes, fontsize=8.5,
            va='bottom', ha='left')

    # (D) leading PCs are a phase code for location: (PC0,PC1) traces a circle
    ax = axes[1, 0]
    sc = ax.scatter(coeffs[:, 0], coeffs[:, 1], c=peak, cmap='hsv', s=6, alpha=0.6)
    ax.set_xlabel('projection on PC0'); ax.set_ylabel('projection on PC1')
    ax.set_aspect('equal', 'box')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('target peak location (bin)')
    ax.set_title('(PC0, PC1) = a circle parametrised by location')

    # (E) width sweep at fixed location, in PC space: energy moves to trailing PCs
    ax = axes[1, 1]
    loc0 = C / 2
    for w, a in [(4.0, 1.0), (9.0, 0.7), (16.0, 0.45)]:
        c = pca.transform(circular_bump(loc0, w)[None])[0]
        kk = np.arange(1, min(40, len(c)) + 1)
        ax.plot(kk, np.abs(c[:len(kk)]) + 1e-6,
                lw=1.8, marker='o', ms=2.5, label=f'width={w:.0f}', alpha=a)
    ax.axvline(kloc + 0.5, color='k', ls='--', lw=1.2)
    ax.set_yscale('log'); ax.set_xlabel('PC index  k')
    ax.set_ylabel('|coefficient|  (fixed location)')
    ax.set_title('Narrower bump → more energy in the trailing PCs')
    ax.legend(fontsize=8)

    # (F) reconstruction: leading-K PCs give position but not sharpness
    ax = axes[1, 2]
    tr = int(np.argmin(np.abs(peak - C / 2))) if real else \
        int(np.argmin(np.abs(locs - C / 2)))
    x = np.arange(C)
    ps.target_band(ax, x, T[tr], label='target')
    for K, col, lab in [(kloc, LOC_COL, f'leading {kloc} PCs (location only)'),
                        (len(pcs), '0.2', 'all PCs')]:
        rec = pca.mean_ + coeffs[tr, :K] @ pcs[:K]
        ax.plot(x, rec, color=col, lw=1.8, label=lab)
    ax.set_xlabel('orientation bin'); ax.set_yticks([])
    ax.set_title('Leading PCs fix position; sharpness needs the trailing PCs')
    ax.legend(fontsize=8)

    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, stem)
    print(f'  {stem}: kloc(90%)={kloc}, n_pc={len(evar)}, '
          f'PC0/PC1 freq={dominant_freq(pcs[0])}/{dominant_freq(pcs[1])}')


def real_target_set(results_root, run, slug, split, mouse, arch):
    mat = Path(results_root) / run / slug / f'{split}.mat'
    d = sio.loadmat(str(mat), simplify_cells=True)['results'][f'mouse_{mouse}']
    return np.asarray(d['Dist'][arch]['target'], float)


def main(results_root, out_root):
    ps.apply()
    out_dir = Path(out_root)

    # 1) toy targets — full control
    T, locs = toy_target_set()
    fig_demo(T, locs, 'Are the leading PCs location and the trailing PCs shape? — '
             'TOY targets (translated fixed-width bumps)',
             'pc_location_vs_shape_toy', out_dir, real=False)

    # 2) real V1 targets — does the same basis structure hold?
    try:
        T_real = real_target_set(results_root, 'wm3', 'Q_PCA_half_100ms_all',
                                 'stratified_balanced', 0, 'spat')
        fig_demo(T_real, None, 'The same basis structure on REAL V1 targets '
                 '(Q, mouse 0, spatial)', 'pc_location_vs_shape_real', out_dir,
                 real=True)
    except Exception as e:
        print(f'  [real] skipped: {e}')
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/pc_geometry')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
