# -*- coding: utf-8 -*-
"""λ_H sweep: how does the TEMPORAL decoder do and look as the entropy/sharpness
penalty rises? (2026-06-10 meeting deliverable — "use other loss functions for
training with a λ_H sweep and check how temporal does and looks".)

Reads the `lambdaH_sweep_entlam<λ>` runs produced by
`run_loss_comparison.py --entropy-lambdas ...` (each an isolated copy of the grid
at one λ_H). The penalty acts only on the temporal (sampling) model, so the
spatial cells are the built-in control: they should be ~flat across λ_H.

Three panels (PNG+SVG), x = λ_H on a categorical axis (so λ_H=0 sits cleanly):
  (a) TEMPORAL KL-skill vs λ_H per loss — skill = test KL / shuffle KL (chance=1).
      "How it does": does rising λ_H pull PCA/Wasserstein down toward/below 1?
  (b) TEMPORAL peakiness (decoded max-prob) vs λ_H per loss, with the IO-target
      peakiness reference. "How it looks": does λ_H broaden the over-sharp posteriors?
  (c) SPATIAL KL-skill vs λ_H per loss — the control; expected flat (λ_H-invariant).

n = 6 mice; per-mouse then averaged. Loss maths == training (cross_loss_eval).

Outputs under figures/peakiness_scatter/:  lambda_h_temporal_sweep.png

Usage:  python diagnostics/lambda_h_temporal_sweep.py   # after rsync-down
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

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}


def _lambda_of(name):
    """lambdaH_sweep_entlam0p003 -> 0.003 ; ..._entlam0 -> 0.0."""
    m = re.search(r'entlam([0-9p]+)', name)
    return float(m.group(1).replace('p', '.')) if m else None


def _find_cell(run_dir, loss, split):
    """The Q_<loss>_half_100ms cell (tolerant of a trailing _all suffix)."""
    for h in sorted(run_dir.glob(f'Q_{loss}_half_100ms*')):
        f = h / f'{split}.mat'
        if f.is_file():
            return f
    return None


def collect(results_root, prefix, split):
    """{lambda: {(loss, arch): {'skill': [per-mouse], 'peak': [per-mouse]}}}, plus
    the loss-invariant IO-target peakiness (per mouse)."""
    runs = sorted(Path(results_root).glob(f'{prefix}_entlam*'),
                  key=lambda p: (_lambda_of(p.name) if _lambda_of(p.name) is not None else 1e9))
    data, target_peak = {}, []
    for rd in runs:
        lam = _lambda_of(rd.name)
        if lam is None:
            continue
        for loss in LOSSES:
            f = _find_cell(rd, loss, split)
            if f is None:
                continue
            res = sio.loadmat(str(f), simplify_cells=True).get('results')
            if not isinstance(res, dict):
                continue
            for arch in ('spat', 'temp'):
                shf = arch + '_shf'
                sk, pk = [], []
                for mk in sorted(res):
                    D = res[mk]['Dist']
                    pcs, evar = D.get('pcs'), D.get('explained_var')
                    dec = np.asarray(D[arch]['decoded'], float)
                    tgt = np.asarray(D[arch]['target'], float)
                    r = _eval_one(dec, tgt, 'KL', pcs, evar)
                    s = (_eval_one(np.asarray(D[shf]['decoded'], float),
                                   np.asarray(D[shf]['target'], float), 'KL', pcs, evar)
                         if shf in D else np.nan)
                    if np.isfinite(r) and np.isfinite(s) and s > 0:
                        sk.append(r / s)
                    pk.append(float(np.nanmean(dec.max(1))))
                    if arch == 'temp' and loss == LOSSES[0]:
                        target_peak.append(float(np.nanmean(tgt.max(1))))
                data.setdefault(lam, {})[(loss, arch)] = {'skill': sk, 'peak': pk}
    return data, (np.nanmean(target_peak) if target_peak else np.nan)


def main(results_root, prefix, split, out_root):
    ps.apply()
    data, tgt_peak = collect(results_root, prefix, split)
    if not data:
        raise SystemExit(f'no {prefix}_entlam* runs found under {results_root} '
                         '(rsync them down first).')
    lams = sorted(data)
    x = np.arange(len(lams))
    xlabels = [f'{l:g}' for l in lams]

    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1))

    def _line(ax, arch, field):
        for loss in LOSSES:
            ys, es = [], []
            for lam in lams:
                m, s = _agg(data[lam].get((loss, arch), {}).get(field, []))
                ys.append(m); es.append(s)
            ax.errorbar(x, ys, yerr=es, color=LCOL[loss], lw=2, marker='o', ms=4,
                        capsize=2, label=loss)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right')
        ax.set_xlabel(r'entropy penalty $\lambda_H$')

    # (a) temporal KL-skill
    _line(axes[0], 'temp', 'skill')
    axes[0].axhline(1.0, color='k', ls='--', lw=1.1)
    axes[0].set_ylabel('temporal KL-skill (test / shuffle)')
    axes[0].set_title('How temporal does (skill; 1 = chance)', fontsize=9)
    axes[0].legend(fontsize=7.5, loc='best')

    # (b) temporal peakiness
    _line(axes[1], 'temp', 'peak')
    if np.isfinite(tgt_peak):
        axes[1].axhline(tgt_peak, color='k', ls=':', lw=1.4, label='IO target')
    axes[1].set_ylabel('temporal decoded peakiness (max-prob)')
    axes[1].set_title('How temporal looks (peakiness)', fontsize=9)

    # (c) spatial KL-skill — control (penalty is temporal-only → expect flat)
    _line(axes[2], 'spat', 'skill')
    axes[2].axhline(1.0, color='k', ls='--', lw=1.1)
    axes[2].set_ylabel('spatial KL-skill (test / shuffle)')
    axes[2].set_title('Spatial control (λ_H-invariant → flat)', fontsize=9)

    ps.label_panels(axes)
    fig.suptitle('λ_H sweep on the temporal decoder, per loss  '
                 f'(6 mice, Q/half/100ms; {len(lams)} λ_H values)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'lambda_h_temporal_sweep')

    # numeric: temporal KL-skill at min vs max λ_H
    print(f'λ_H values: {xlabels}')
    print('temporal KL-skill (test/shuffle) at min→max λ_H per loss:')
    for loss in LOSSES:
        lo = _agg(data[lams[0]].get((loss, 'temp'), {}).get('skill', []))[0]
        hi = _agg(data[lams[-1]].get((loss, 'temp'), {}).get('skill', []))[0]
        plo = _agg(data[lams[0]].get((loss, 'temp'), {}).get('peak', []))[0]
        phi = _agg(data[lams[-1]].get((loss, 'temp'), {}).get('peak', []))[0]
        print(f'  {loss:12s} skill {lo:.2f} → {hi:.2f}   peak {plo:.3f} → {phi:.3f}')
    print(f'IO target peakiness ≈ {tgt_peak:.3f}')
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', default='lambdaH_sweep')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.prefix, a.split, a.out_root)
