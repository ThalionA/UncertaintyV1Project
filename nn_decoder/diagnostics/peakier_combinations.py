# -*- coding: utf-8 -*-
"""Peaky-vs-broad targets × spatial/temporal × loss (2026-06-10 meeting:
"all combinations of peakier examples → temporal peakier / broader etc.").

Two complementary views, on the matched loss_comparison_v1 cell (Q/half/100ms):

  QUANT (peakier_combinations_quant)  — the calibration curve. Bin trials by how
    peaky the IO target is (broad → peaky, by target max-prob) and plot the mean
    DECODED peakiness in each bin, spatial and temporal panels, one line per loss,
    against the identity (perfect tracking). Over-sharpening = the vertical gap
    above identity; the slope says whether the decoder follows the target's
    peaky/broad axis at all or pins to a near-constant sharpness.

  GALLERY (peakier_combinations_gallery) — actual posteriors. Pick example trials
    spanning the target-peakiness spectrum (broad → peaky) and overlay, per arch
    (rows) × example (cols), the IO target with the PCA / CE / KL decoded curves.
    Shows how each architecture renders a broad vs a peaky target.

Targets are loss-invariant (set by the stimulus / IO), so trials picked by target
peakiness in the PCA cell are the same trials in every loss cell — checked at
load (allclose) before pulling each loss's decoded posterior at those indices.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  peakier_combinations_quant.png
  peakier_combinations_gallery.png

Usage:  python diagnostics/peakier_combinations.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
GALLERY_LOSSES = ['PCA', 'CE', 'KL']     # the pathology + two calibrated, for the curves


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _load_cell(results_root, run, split, loss):
    f = Path(results_root) / run / _slug(loss) / f'{split}.mat'
    if not f.is_file():
        return None
    return sio.loadmat(str(f), simplify_cells=True).get('results')


def collect_maxprob(results_root, run, split):
    """per (loss, arch): pooled per-trial decoded & target max-prob (peakiness)."""
    out = {}
    for loss in LOSSES:
        res = _load_cell(results_root, run, split, loss)
        if not isinstance(res, dict):
            continue
        for arch, _ in ARCHS:
            dec, tgt = [], []
            for mk in sorted(res):
                D = res[mk]['Dist'][arch]
                dec.append(np.asarray(D['decoded'], float).max(1))
                tgt.append(np.asarray(D['target'], float).max(1))
            out[(loss, arch)] = (np.concatenate(dec), np.concatenate(tgt))
    return out


def _binned(x, y, edges):
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    cx = 0.5 * (edges[:-1] + edges[1:])
    m = np.array([np.nanmean(y[idx == b]) if (idx == b).any() else np.nan
                  for b in range(len(edges) - 1)])
    s = np.array([np.nanstd(y[idx == b], ddof=1) / np.sqrt(max((idx == b).sum(), 1))
                  if (idx == b).sum() > 1 else 0.0 for b in range(len(edges) - 1)])
    return cx, m, s


def fig_quant(data, out_root):
    ps.apply()
    # common target-peakiness bins across everything (target is loss-invariant)
    all_tgt = np.concatenate([v[1] for v in data.values()])
    edges = np.quantile(all_tgt[np.isfinite(all_tgt)], np.linspace(0, 1, 8))
    edges[-1] += 1e-9
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1), sharex=True, sharey=True)
    for ax, (arch, alabel) in zip(axes, ARCHS):
        lo = min(edges[0], 0); hi = edges[-1]
        for loss in LOSSES:
            if (loss, arch) not in data:
                continue
            dec, tgt = data[(loss, arch)]
            cx, m, s = _binned(tgt, dec, edges)
            ax.errorbar(cx, m, yerr=s, color=LCOL[loss], lw=2, marker='o', ms=4,
                        capsize=2, label=loss)
        ax.plot([lo, hi], [lo, hi], color='k', ls='--', lw=1.4, label='identity (perfect)')
        ax.set_xlabel('IO target peakiness (max-prob)  →  peakier')
        ax.set_ylabel('decoded peakiness (max-prob)')
        ax.set_title(alabel, fontsize=9)
        if arch == 'spat':
            ax.legend(fontsize=7.5, loc='upper left')
    ps.label_panels(axes)
    fig.suptitle('Decoded vs IO-target peakiness — over-sharpening across the '
                 'broad→peaky axis  (6 mice, Q/half/100ms)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'peakier_combinations_quant')

    print('decoded peakiness at the BROADEST vs PEAKIEST target bin (mean):')
    for arch, alabel in ARCHS:
        print(f'  [{alabel}]  (IO target {0.5*(edges[0]+edges[1]):.3f} → '
              f'{0.5*(edges[-2]+edges[-1]):.3f})')
        for loss in LOSSES:
            if (loss, arch) not in data:
                continue
            dec, tgt = data[(loss, arch)]
            _, m, _ = _binned(tgt, dec, edges)
            g = np.isfinite(m)
            print(f'    {loss:12s} {m[g][0]:.3f} → {m[g][-1]:.3f}')


def fig_gallery(results_root, run, split, out_root, mouse='mouse_0', ncol=6):
    ps.apply()
    ref = _load_cell(results_root, run, split, 'PCA')
    if not isinstance(ref, dict) or mouse not in ref:
        print('gallery: reference cell/mouse missing — skipped.'); return
    tgt_ref = np.asarray(ref[mouse]['Dist']['spat']['target'], float)
    tgt_mp = tgt_ref.max(1)
    good = np.isfinite(tgt_mp)
    order = np.argsort(tgt_mp[good])
    sel = np.where(good)[0][order][np.linspace(0, order.size - 1, ncol).round().astype(int)]

    # pull decoded for each gallery loss/arch at the SAME trial indices (verify align)
    dec = {}            # (loss, arch) -> (ncol, 91)
    tgt = {}            # arch -> (ncol, 91)
    for loss in GALLERY_LOSSES:
        res = _load_cell(results_root, run, split, loss)
        if not isinstance(res, dict) or mouse not in res:
            continue
        for arch, _ in ARCHS:
            D = res[mouse]['Dist'][arch]
            t = np.asarray(D['target'], float)
            if not np.allclose(np.nan_to_num(t[sel]), np.nan_to_num(
                    np.asarray(ref[mouse]['Dist'][arch]['target'], float)[sel]),
                    atol=1e-6):
                print(f'  WARN {loss}/{arch}: target misaligned vs PCA — skipping')
                continue
            dec[(loss, arch)] = np.asarray(D['decoded'], float)[sel]
            tgt[arch] = t[sel]

    xb = np.linspace(0, 90, tgt_ref.shape[1])
    fig, axes = plt.subplots(len(ARCHS), ncol,
                             figsize=ps.figsize(ncol, len(ARCHS), panel_w=1.6, panel_h=1.7),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for r, (arch, alabel) in enumerate(ARCHS):
        for c in range(ncol):
            ax = axes[r, c]
            if arch in tgt:
                ax.fill_between(xb, tgt[arch][c], color='0.8', lw=0, zorder=0)
                ax.plot(xb, tgt[arch][c], color='k', lw=1.3, label='IO target')
            for loss in GALLERY_LOSSES:
                if (loss, arch) in dec:
                    ax.plot(xb, dec[(loss, arch)][c], color=LCOL[loss], lw=1.5,
                            label=loss)
            if r == 0:
                ax.set_title(f'target max-prob\n{tgt_mp[sel][c]:.3f}', fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f'{alabel}\nposterior', fontsize=8.5)
            if r == len(ARCHS) - 1:
                ax.set_xlabel('orientation (deg)', fontsize=8)
            ax.set_yticks([])
    axes[0, 0].legend(fontsize=6.5, loc='upper right')
    fig.suptitle(f'How spatial vs temporal render broad→peaky targets '
                 f'({mouse}, Q/half/100ms; columns = example trials)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'peakier_combinations_gallery')
    print(f'\ngallery trials (target max-prob): {np.round(tgt_mp[sel], 3)}')


def main(results_root, run, split, out_root):
    data = collect_maxprob(results_root, run, split)
    if not data:
        raise SystemExit('no loss cells found.')
    fig_quant(data, out_root)
    fig_gallery(results_root, run, split, out_root)
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
