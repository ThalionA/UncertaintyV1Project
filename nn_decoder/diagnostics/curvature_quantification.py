# -*- coding: utf-8 -*-
"""The peakiness mechanism, as NUMBERS — the quantification behind "cheaper to
sharpen" (2026-06-10, Theo's challenge: bring the quantification, not the curve
shapes).

The PCA loss is *exactly* a quadratic in the PC-projection coordinates,

    L(p, t) = sum_k evar_k * <p - t, u_k>^2 ,

so the curvature of the loss along PC direction k is *exactly* 2*evar_k (no fit,
no approximation), and the distribution-space gradient is *exactly*
2*sum_k evar_k * r_k * u_k with r_k = <p - t, u_k>. This script reads the real
saved bases/decoders (wm3 = the deployed evar run, 6 mice) and turns that into
the three numbers that answer "why does peakiness *increase*?" in the currency a
dynamics question deserves — forces and dimensions, not landscape shapes:

  (a) CURVATURE SPECTRUM. 2*evar_k vs PC. The IO targets live on a ~2-D manifold
      (participation ratio ~1.7; PC0-1 carry ~93% of target variance), so width/
      shape lives in the trailing PCs where evar collapses to ~1e-4 and then to
      MACHINE ZERO by ~PC22. The loss's curvature along "make it peakier" is not
      small — past ~PC15 it is zero to ten significant figures. The calibrated
      target is a knife-edge RIDGE in the width direction, not a basin.

  (b) ERROR vs RESTORING FORCE at the deployed evar decoder. The width-subspace
      squared error is ~11x the location error (it IS grossly miscalibrated), yet
      the loss gradient there is only ~2.5% of the location gradient (~40x weaker
      restoring force). The optimiser cannot feel the error it is creating.

  (c) THE FIX INSTALLS THE MISSING CURVATURE. PCA + lambda*Brier floors every
      weight: evar_k -> evar_k + lambda, so the width-PC curvature jumps from ~0
      to 2*lambda. The relaxation timescale on width drops from infinity to the
      finite 1/(2*eta*lambda) — width now relaxes to the target instead of
      integrating drift. That the predicted cure is exactly the missing curvature
      is the proof the diagnosis is right.

Outputs (PNG+SVG) under figures/pc_geometry/:
  curvature_quantification.png

Usage:  python diagnostics/curvature_quantification.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

RUN, SLUG, SPLIT = 'wm3', 'Q_PCA_half_100ms_all', 'stratified_balanced'
ARCH = 'spat'                 # spatial decoder = the loss acting alone (no entropy term)
LAMBDA_FIX = 0.1              # the deployed shape-fix weight (note Â§7)

LOC_FILL = '#c6dbef'          # constrained "location" subspace — cool
WID_FILL = '#fdd0a2'          # free "width/shape" subspace — warm (where peakiness hides)
NEUTRAL = '#2b2b2b'


def load_bases_and_residuals(results_root):
    """Per mouse: PCA basis (pcs, evar) and the deployed decoded/target arrays.

    Returns
    -------
    evars : list of (n_pc,) explained-variance spectra (one per mouse)
    resids : list of (n_trials, n_pc) decoded-minus-target PC residuals
    """
    mat = Path(results_root) / RUN / SLUG / f'{SPLIT}.mat'
    res = sio.loadmat(str(mat), simplify_cells=True)['results']
    evars, resids = [], []
    for mk in sorted(k for k in res if k.startswith('mouse_')):
        d = res[mk]['Dist']
        pcs = np.asarray(d['pcs'], float)               # (n_pc, n_cats), orthonormal rows
        evar = np.asarray(d['explained_var'], float)    # (n_pc,)
        dec = np.asarray(d[ARCH]['decoded'], float)
        tgt = np.asarray(d[ARCH]['target'], float)
        evars.append(evar)
        resids.append((dec - tgt) @ pcs.T)              # r_k = <dec-tgt, u_k>
    return evars, resids


def quantify(evars, resids):
    """Boil the bases/residuals down to the headline numbers + the spectrum."""
    ev = np.mean(np.stack(evars), axis=0)               # mean spectrum across mice
    # effective dimension of the target manifold (participation ratio), per mouse
    pr = np.mean([(e.sum() ** 2) / np.sum(e ** 2) for e in evars])

    out = {'ev_mean': ev, 'part_ratio': pr}
    # location block A = leading PCs to 90% variance (per mouse, then the modal K)
    Ks = [int(np.searchsorted(np.cumsum(e), 0.90) + 1) for e in evars]
    K = int(np.bincount(Ks).argmax())
    out['K'] = K
    out['cumvar_K'] = ev[:K].sum()

    # gradient (= restoring force) the optimiser feels: g_k = 2*evar_k*r_k.
    # Compare the WIDTH block (k>=K) to the LOCATION block (k<K), per mouse.
    err_ratio, force_ratio = [], []
    for e, r in zip(evars, resids):
        g = 2.0 * e[None, :] * r                         # (trials, n_pc)
        err = r ** 2                                     # squared residual per PC
        eA = err[:, :K].sum(1).mean();  eB = err[:, K:].sum(1).mean()
        fA = np.sqrt((g[:, :K] ** 2).sum(1)).mean()
        fB = np.sqrt((g[:, K:] ** 2).sum(1)).mean()
        err_ratio.append(eB / eA);  force_ratio.append(fB / fA)
    out['err_ratio'] = float(np.mean(err_ratio))         # width error / location error (~11)
    out['force_ratio'] = float(np.mean(force_ratio))     # width force / location force (~0.025)
    # where the spectrum hits machine zero (curvature numerically vanishes)
    out['pc_machine_zero'] = int(np.argmax(ev < 1e-12)) if np.any(ev < 1e-12) else len(ev)
    return out


def make_figure(q, out_dir):
    ev = q['ev_mean']
    curv = 2.0 * ev                                      # exact loss curvature along PC k
    K = q['K']
    n_pc = len(ev)
    pcs_x = np.arange(n_pc)
    fix_floor = 2.0 * LAMBDA_FIX                          # curvature the lambda floor installs

    ps.apply()
    fig, axes = ps.fig(3, 1)
    axA, axB, axC = axes

    # -- (a) the curvature spectrum: flat where width lives -----------------
    axA.axvspan(-0.5, K - 0.5, color=LOC_FILL, alpha=0.9, lw=0)
    axA.axvspan(K - 0.5, n_pc, color=WID_FILL, alpha=0.5, lw=0)
    floor = 1e-16
    axA.semilogy(pcs_x, np.clip(curv, floor, None), '-o', color=NEUTRAL,
                 ms=3.2, lw=1.4, zorder=5)
    axA.set_xlim(-0.5, 40)
    axA.set_ylim(floor, 5)
    axA.set_xlabel('principal component  $k$  (location $\\rightarrow$ width)')
    axA.set_ylabel('loss curvature along PC $k$   $= 2\\,\\mathrm{evar}_k$')
    axA.set_title('The loss is flat where width lives')
    axA.text(4.5, 2.6,
             f'location subspace\n(PC0–{K-1}, {q["cumvar_K"]*100:.0f}% of variance,\n'
             f'manifold rank $\\approx$ {q["part_ratio"]:.1f})',
             ha='left', va='top', fontsize=8.2, color='#08519c')
    axA.text(39, 4e-3, 'width / shape:\ncurvature $\\approx 0$',
             ha='right', va='bottom', fontsize=8.5, color='#a63603')
    axA.annotate('machine zero\n'f'by PC{q["pc_machine_zero"]}',
                 xy=(q['pc_machine_zero'], floor * 6), xytext=(26, 1e-9),
                 fontsize=8.2, color='#a63603', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#a63603', lw=1))

    # -- (b) error vs restoring force at the deployed evar decoder ----------
    groups = ['squared\nerror', 'restoring force\n(loss gradient)']
    loc_vals = [1.0, 1.0]                                 # location = reference
    wid_vals = [q['err_ratio'], q['force_ratio']]         # width, relative to location
    x = np.arange(2); w = 0.36
    axB.bar(x - w / 2, loc_vals, w, color=LOC_FILL, edgecolor='#08519c',
            label='location subspace', zorder=3)
    axB.bar(x + w / 2, wid_vals, w, color=WID_FILL, edgecolor='#a63603',
            label='width subspace', zorder=3)
    axB.set_yscale('log')
    axB.axhline(1.0, ls=':', lw=1, color='0.5', zorder=1)
    axB.set_xticks(x); axB.set_xticklabels(groups)
    axB.set_ylim(8e-3, 30)
    axB.set_ylabel('relative to location subspace ($=1$)')
    axB.set_title('Huge width error, almost no force on it')
    axB.annotate(f'{q["err_ratio"]:.0f}$\\times$', xy=(0 + w / 2, q['err_ratio']),
                 xytext=(0, 4), textcoords='offset points', ha='center',
                 fontsize=10, fontweight='bold', color='#a63603')
    axB.annotate(f'{q["force_ratio"]*100:.0f}%', xy=(1 + w / 2, q['force_ratio']),
                 xytext=(0, 4), textcoords='offset points', ha='center',
                 fontsize=10, fontweight='bold', color='#a63603')
    axB.legend(loc='upper right', fontsize=8)

    # -- (c) the fix installs the missing curvature ------------------------
    axC.axvspan(K - 0.5, n_pc, color=WID_FILL, alpha=0.4, lw=0)
    axC.semilogy(pcs_x, np.clip(curv, floor, None), '-o', color=ps.PCA_EVAR,
                 ms=3.2, lw=1.4, label='PCA (evar): $\\lambda=0$', zorder=5)
    axC.semilogy(pcs_x, curv + fix_floor, '-o', color=ps.SHAPE,
                 ms=3.2, lw=1.4, label=f'PCA+$\\lambda$Brier: $\\lambda={LAMBDA_FIX}$', zorder=6)
    axC.axhline(fix_floor, ls='--', lw=1, color=ps.SHAPE, zorder=2)
    axC.set_xlim(-0.5, 40)
    axC.set_ylim(floor, 5)
    axC.set_xlabel('principal component  $k$')
    axC.set_ylabel('loss curvature along PC $k$')
    axC.set_title('The fix floors the width curvature at $2\\lambda$')
    axC.text(39, fix_floor * 1.6, 'width curvature  $0 \\rightarrow 2\\lambda$',
             ha='right', va='bottom', fontsize=8.5, color='#005a32')
    axC.legend(loc='lower left', fontsize=8)

    ps.label_panels(axes)
    ps.save_fig(fig, out_dir, 'curvature_quantification')


def main(results_root, out_root):
    evars, resids = load_bases_and_residuals(results_root)
    q = quantify(evars, resids)
    print('=== peakiness quantification (wm3 evar run, spatial, 6 mice) ===')
    print(f'  target manifold effective rank (participation ratio): {q["part_ratio"]:.2f}')
    print(f'  location block = PC0..PC{q["K"]-1}  ({q["cumvar_K"]*100:.1f}% of variance)')
    print(f'  curvature along PC0 = {2*q["ev_mean"][0]:.3f};  '
          f'machine-zero by PC{q["pc_machine_zero"]}')
    print(f'  width-subspace squared error = {q["err_ratio"]:.1f}x the location error')
    print(f'  width-subspace restoring force = {q["force_ratio"]*100:.2f}% of the '
          f'location force ({1/q["force_ratio"]:.0f}x weaker)')
    make_figure(q, Path(out_root))
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/pc_geometry')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
