# -*- coding: utf-8 -*-
"""λ_smooth sweep: find the operating point of the output-smoothness penalty
(2026-06-16 — method 2, the cluster sweep).

`smoothsweep_smooth<λ>` runs (PCA + Wasserstein, patience 15 like production) vs the
smooth=0 baseline `loss_comparison_v1`. Per loss × arch: decoded peakiness and raw
KL(decoded ‖ IO target) vs λ_smooth. Prior: U-shaped KL — KL falls as λ_smooth pulls
the posterior toward the broad target, then rises again once it over-smooths past it.

Outputs (PNG+SVG) under figures/peakiness_scatter/:  smooth_lambda_sweep.png

Usage:  python diagnostics/smooth_lambda_sweep.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one, _agg  # noqa: E402

LOSSES = ['PCA', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _smooth_of(name):
    if name == 'loss_comparison_v1':
        return 0.0
    m = re.search(r'smooth([0-9p]+)', name)
    return float(m.group(1).replace('p', '.')) if m else None


def _conditions(results_root):
    root = Path(results_root)
    conds = []
    if (root / 'loss_comparison_v1').is_dir():
        conds.append(('loss_comparison_v1', 0.0))
    drops = [(d.name, _smooth_of(d.name)) for d in root.glob('smoothsweep_smooth*') if d.is_dir()]
    conds += sorted(drops, key=lambda t: (t[1] if t[1] is not None else 0))
    return conds


def _stats(results_root, run, loss, arch, split):
    f = Path(results_root) / run / _slug(loss) / f'{split}.mat'
    if not f.is_file():
        return (np.nan, np.nan), (np.nan, np.nan), np.nan
    res = sio.loadmat(str(f), simplify_cells=True).get('results')
    pk, kl, tg = [], [], []
    for mk in sorted(res):
        D = res[mk]['Dist']
        dec, t = np.asarray(D[arch]['decoded'], float), np.asarray(D[arch]['target'], float)
        pk.append(float(np.nanmean(dec.max(1)))); tg.append(float(np.nanmean(t.max(1))))
        kl.append(_eval_one(dec, t, 'KL', D.get('pcs'), D.get('explained_var')))
    return _agg(pk), _agg(kl), float(np.nanmean(tg))


def fig_gallery(results_root, split, out_root, loss='PCA', mouse='mouse_0', ncol=5):
    """Example posteriors across λ_smooth (dark→light = 0→max) over the IO target —
    the spike → smooth → on-target progression, both archs."""
    ps.apply()
    conds = _conditions(results_root)
    THETA = np.arange(91.0)
    fig, axes = plt.subplots(len(ARCHS), ncol,
                             figsize=ps.figsize(ncol, len(ARCHS), panel_w=1.9, panel_h=1.5), sharex=True)
    axes = np.atleast_2d(axes)

    def _load(run, arch):
        f = Path(results_root) / run / _slug(loss) / f'{split}.mat'
        D = sio.loadmat(str(f), simplify_cells=True)['results'][mouse]['Dist'][arch]
        return np.asarray(D['decoded'], float), np.asarray(D['target'], float)

    for r, (arch, alabel) in enumerate(ARCHS):
        _, tgt = _load(conds[0][0], arch)
        sel = np.argsort(tgt.max(1))[np.linspace(0, tgt.shape[0] - 1, ncol).round().astype(int)]
        decs = [_load(run, arch)[0] for run, _ in conds]
        for c, tr in enumerate(sel):
            ax = axes[r, c]
            ax.fill_between(THETA, tgt[tr], color='0.85', lw=0, zorder=0)
            for li, (_run, lam) in enumerate(conds):
                ax.plot(THETA, decs[li][tr], color=plt.cm.viridis(li / max(len(conds) - 1, 1)),
                        lw=1.2, label=f'λ={lam:g}' if (r == 0 and c == 0) else None)
            ps.cap_posterior_ylim(ax, float(tgt[tr].max()), mult=3.0)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'tgt mp {tgt[tr].max():.3f}', fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f'{alabel}\n{loss} posterior', fontsize=8)
            if r == len(ARCHS) - 1:
                ax.set_xlabel('orientation (deg)', fontsize=8)
    axes[0, 0].legend(fontsize=6, loc='upper right', title='λ_smooth')
    fig.suptitle(f'{loss} posteriors across λ_smooth (dark→light = 0→max) over the IO target '
                 f'(grey, {mouse})', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'smooth_lambda_gallery')


def main(results_root, split, out_root):
    ps.apply()
    conds = _conditions(results_root)
    if not any(_smooth_of(r) and _smooth_of(r) > 0 for r, _ in conds):
        raise SystemExit('no smoothsweep_smooth* runs found — rsync them down first.')
    lams = [lam for _, lam in conds]
    x = np.arange(len(conds))
    xlabels = [f'{l:g}' for l in lams]

    # gather
    M = {(loss, arch): {'peak': [], 'kl': []} for loss in LOSSES for a, _ in ARCHS for arch in [a]}
    io_peak = []
    for run, _lam in conds:
        for loss in LOSSES:
            for arch, _ in ARCHS:
                pk, kl, tg = _stats(results_root, run, loss, arch, split)
                M[(loss, arch)]['peak'].append(pk)
                M[(loss, arch)]['kl'].append(kl)
                if np.isfinite(tg):
                    io_peak.append(tg)
    io_peak = float(np.mean(io_peak)) if io_peak else np.nan

    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), sharex=True)
    for r, (arch, alabel) in enumerate(ARCHS):
        for c, (key, mlabel) in enumerate([('peak', 'decoded peakiness (max-prob)'),
                                           ('kl', 'KL(decoded ‖ IO target)')]):
            ax = axes[r, c]
            for loss in LOSSES:
                ys = [m for m, _ in M[(loss, arch)][key]]
                es = [s for _, s in M[(loss, arch)][key]]
                ax.errorbar(x, ys, yerr=es, color=LCOL[loss], lw=2, marker='o', ms=4,
                            capsize=2, label=loss)
            if key == 'peak' and np.isfinite(io_peak):
                ax.axhline(io_peak, color='k', ls=':', lw=1.4, label='IO target')
            ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right')
            ax.set_xlabel(r'$\lambda_{smooth}$  (0 = baseline)')
            ax.set_ylabel(f'{alabel}\n{mlabel}' if c == 0 else mlabel, fontsize=8)
            if r == 0:
                ax.set_title(mlabel, fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7.5, loc='best')
    ps.label_panels(axes.ravel())
    fig.suptitle('Output-smoothness penalty sweep (PCA, Wasserstein; 6 mice, Q/half/100ms, early-stop) '
                 '— is KL U-shaped?', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'smooth_lambda_sweep')
    fig_gallery(results_root, split, out_root)

    print(f'λ_smooth: {xlabels}   (IO-target peakiness ≈ {io_peak:.3f})')
    for loss in LOSSES:
        for arch, alabel in ARCHS:
            pk = [m for m, _ in M[(loss, arch)]['peak']]
            kl = [m for m, _ in M[(loss, arch)]['kl']]
            kl_fin = [v for v in kl if np.isfinite(v)]
            best = lams[int(np.nanargmin(kl))] if kl_fin else np.nan
            print(f'  {loss:12s}{alabel:9s} peak ' + ' '.join(f'{v:.3f}' for v in pk)
                  + ' | KL ' + ' '.join(f'{v:.2f}' for v in kl) + f'  (min KL @ λ={best:g})')
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.split, a.out_root)
