# -*- coding: utf-8 -*-
"""Example decoded posteriors: flat-evar (Brier) PCA vs evar-weighted PCA vs IO
target, on matched trials (2026-06-03, Theo's request).

Both PCA runs share the split seed (test_size=0.5, random_state=42), so trial
index i is the same physical trial in `loss_comparison_v1` (evar-weighted) and
`brier_ctrl_flatevar` (flat-evar). We load the FINAL decoded posteriors straight
from each run's `.mat` (no re-decode) and overlay, for a spread of example trials
chosen evenly across target peak position. Shows directly what flat-evar buys:
broad, target-tracking posteriors instead of the weighted decoder's spikes.

Output: figures/loss_sweep_plots/brier_ctrl_flatevar/example_posteriors/
  examples_<arch>_m<mouse>.png

Usage
-----
    python diagnostics/flat_evar_example_posteriors.py --arch temp
    python diagnostics/flat_evar_example_posteriors.py --arch spat --n 12 --mouse 0
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

WEIGHTED_RUN = 'loss_comparison_v1'
FLAT_RUN = 'brier_ctrl_flatevar'
SLUG = 'Q_PCA_half_100ms_all'


def _H(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _load(results_root, run, split, mouse, arch):
    m = sio.loadmat(f'{results_root}/{run}/{SLUG}/{split}.mat', simplify_cells=True)
    d = m['results'][f'mouse_{mouse}']['Dist'][arch]
    return np.asarray(d['decoded'], float), np.asarray(d['target'], float)


def main(results_root, split, mouse, arch, n, out_root):
    ps.apply()
    dec_w, tgt = _load(results_root, WEIGHTED_RUN, split, mouse, arch)
    dec_f, tgt_f = _load(results_root, FLAT_RUN, split, mouse, arch)
    assert np.allclose(tgt, tgt_f, atol=1e-6), 'targets not aligned across runs'

    # pick n trials spread evenly across target peak position (a representative
    # tour of the stimulus range), tie-broken to also vary target width.
    peak = np.argmax(tgt, axis=1)
    order = np.lexsort((_H(tgt), peak))           # by peak, then width
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]

    # y-limit at a few × the target scale so the broad shapes stay legible; the
    # evar-weighted spikes deliberately clip off the top (their height is in the
    # title as max-prob). Keeps the comparison about SHAPE, not spike height.
    ymax = 4.0 * float(np.median(tgt[picks].max(axis=1)))

    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.7 * nrow),
                             squeeze=False, sharex=True)
    x = np.arange(tgt.shape[1])
    for k, tr in enumerate(picks):
        ax = axes[k // ncol][k % ncol]
        ps.target_band(ax, x, tgt[tr])
        ax.plot(x, dec_w[tr], color=ps.PCA_EVAR, lw=1.8, label='PCA (evar-weighted)')
        ax.plot(x, dec_f[tr], color=ps.FLAT_EVAR, lw=1.8, label='PCA (flat-evar)')
        ax.set_ylim(0, ymax)
        ax.set_title(f'trial {tr}   max-prob: tgt {tgt[tr].max():.2f} / '
                     f'wt {dec_w[tr].max():.2f} / flat {dec_f[tr].max():.2f}',
                     fontsize=8.5)
        ax.set_yticks([])
        if k // ncol == nrow - 1:
            ax.set_xlabel('orientation bin', fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.suptitle(f'Example decoded posteriors — {arch.upper()}  '
                 '(grey = IO target; spikes clip)', y=1.01, fontsize=12)
    fig.tight_layout()
    out_dir = Path(out_root) / FLAT_RUN / 'example_posteriors'
    ps.save_fig(fig, out_dir, f'examples_{arch}_m{mouse}')

    # numeric summary across all trials
    print(f'  {arch} mean entropy — target {_H(tgt).mean():.2f}, '
          f'weighted {_H(dec_w).mean():.2f}, flat-evar {_H(dec_f).mean():.2f}')
    print(f'  {arch} mean max-prob — target {tgt.max(1).mean():.3f}, '
          f'weighted {dec_w.max(1).mean():.3f}, flat-evar {dec_f.max(1).mean():.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--mouse', type=int, default=0)
    ap.add_argument('--arch', default='temp', choices=('spat', 'temp'))
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(a.results_root, a.split, a.mouse, a.arch, a.n, a.out_root)
