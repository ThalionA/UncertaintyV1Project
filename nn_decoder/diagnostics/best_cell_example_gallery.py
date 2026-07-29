# -*- coding: utf-8 -*-
"""Example posteriors at the selected projection-loss cell, including the temporal decoder's
INDIVIDUAL per-time-bin posteriors, over a deliberately diverse set of trials.

Cell: projection-based loss, shape_lambda = 30 (lambda_Brier = 0.3), H=8, dropout 0, patience 0.

Each panel shows, on one probability axis:
  - the IO target posterior (grey fill)
  - the spatial decoder's posterior
  - the temporal decoder's trial-averaged posterior (bold)
  - the temporal decoder's 10 INDIVIDUAL per-time-bin posteriors (faint), from Dist['temp']
    ['decoded_samp'], shape (n_trials, 91, T). The trial-averaged temporal posterior is the mean
    of these, so the faint lines show what the Jensen average is hiding.

Trials are NOT chosen for looking good. They are selected to span the interesting regimes:
best/worst for each architecture, the largest and smallest disagreement between time bins, and
the broadest and sharpest IO target. The selection criterion is printed in each panel title.

"Bin disagreement" = mean_t KL(p_t || p_bar), the Jensen gap between the individual per-bin
posteriors and their average. Large = the bins genuinely disagree; ~0 = the temporal decoder is
emitting the same posterior every bin (i.e. not really sampling).

Outputs (PNG+SVG) under figures/hparam_summary/: best_cell_example_gallery
Usage:  python diagnostics/best_cell_example_gallery.py [--mouse mouse_0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from nn_classifier import fit_loss_per_trial  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from story_figures import _results  # noqa: E402

RUN, CELL, LOSS = 'hpsweep_v2', 'lam0p003_drop0_acttanh_h8_pat0_vf0p2_wd0p0001_shp30', 'PCA'
C_TGT, C_SPAT, C_TEMP = '0.72', '#d95f02', '#7570b3'


def _kl_rows(a, b):
    """KL(a || b) per row."""
    a = np.clip(a, 1e-12, None); b = np.clip(b, 1e-12, None)
    return (a * (np.log(a) - np.log(b))).sum(-1)


def main(results_root, out_root, mouse):
    ps.apply()
    res = _results(results_root, RUN, CELL, loss=LOSS)
    D = res[mouse]['Dist']
    tgt = np.asarray(D['spat']['target'], float)
    sp = np.asarray(D['spat']['decoded'], float)
    tp = np.asarray(D['temp']['decoded'], float)
    samp = np.asarray(D['temp']['decoded_samp'], float)          # (n_trials, 91, T)
    tr = res[mouse]['trials']
    ori = np.asarray(tr['orientation'], float)

    ok = np.isfinite(tgt).all(1) & np.isfinite(sp).all(1) & np.isfinite(tp).all(1)
    idx_all = np.where(ok)[0]

    # per-trial scores
    pcs = torch.tensor(np.asarray(D['pcs'], float)); ev = torch.tensor(np.asarray(D['explained_var'], float))
    kl_s = _kl_rows(tgt, sp)
    kl_t = _kl_rows(tgt, tp)
    pbar = samp.mean(-1)                                          # == tp up to fp
    jensen = np.array([_kl_rows(samp[i].T, np.tile(pbar[i], (samp.shape[-1], 1))).mean()
                       for i in range(samp.shape[0])])            # bin disagreement
    tgt_peak = tgt.max(1)

    def pick(score, largest, used):
        order = idx_all[np.argsort(score[idx_all])][::-1] if largest else idx_all[np.argsort(score[idx_all])]
        for i in order:
            if i not in used:
                return int(i)
        return int(order[0])

    used, sel = set(), []
    for lab, score, largest in [
        # the two architectures disagreeing maximally, one panel per direction
        ('spatial ≫ temporal', kl_t - kl_s, True),
        ('temporal ≫ spatial', kl_s - kl_t, True),
        ('best for spatial', kl_s, False),
        ('best for temporal', kl_t, False),
        ('worst for spatial', kl_s, True),
        ('worst for temporal', kl_t, True),
        ('bins most different', jensen, True),
        ('bins most similar', jensen, False),
        ('broadest IO target', tgt_peak, False),
        ('sharpest IO target', tgt_peak, True),
    ]:
        i = pick(score, largest, used)
        used.add(i); sel.append((lab, i))

    x = np.arange(91)
    ncol = 5
    fig, axes = plt.subplots(2, ncol, figsize=ps.figsize(ncol, 2), sharex=True, sharey=True)
    for ax, (lab, i) in zip(axes.ravel(), sel):
        ax.fill_between(x, tgt[i], color=C_TGT, lw=0, zorder=1)
        for b in range(samp.shape[-1]):                           # the individual time bins
            ax.plot(x, samp[i, :, b], color=C_TEMP, lw=0.7, alpha=0.30, zorder=2)
        ax.plot(x, sp[i], color=C_SPAT, lw=1.8, zorder=4)
        ax.plot(x, tp[i], color=C_TEMP, lw=2.0, zorder=5)
        ax.set_title(f'{lab}\ntrial {i}, ori {ori[i]:.0f}°\n'
                     f'KL spat {kl_s[i]:.2f} / temp {kl_t[i]:.2f}, bins {jensen[i]:.2f}',
                     fontsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel('orientation (deg)', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('probability', fontsize=8)
    axes[0][0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=C_TGT, ec='none', label='IO target'),
        Line2D([0], [0], color=C_SPAT, lw=1.8, label='spatial'),
        Line2D([0], [0], color=C_TEMP, lw=2.0, label='temporal (mean)'),
        Line2D([0], [0], color=C_TEMP, lw=0.7, alpha=0.5, label='temporal per-bin (10)')],
        fontsize=6, frameon=True)
    ps.label_panels(axes.ravel())
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'best_cell_example_gallery')

    print(f'cell {RUN}/{CELL}  mouse {mouse}')
    print(f'  bin disagreement (Jensen gap) over all trials: '
          f'median {np.median(jensen[idx_all]):.3f}, range {jensen[idx_all].min():.3f}-'
          f'{jensen[idx_all].max():.3f}')
    print(f'  per-bin peakiness {samp.max(1).mean():.3f}  vs trial-mean {tp.max(1).mean():.3f} '
          f'vs IO target {tgt.max(1).mean():.3f}')
    for lab, i in sel:
        print(f'    {lab:22s} trial {i:4d}  KL_spat {kl_s[i]:.3f}  KL_temp {kl_t[i]:.3f}  '
              f'bin-disagree {jensen[i]:.3f}')
    print(f'Done -> {Path(out_root).resolve()}/best_cell_example_gallery.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hparam_summary')
    ap.add_argument('--mouse', default='mouse_0')
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.mouse)
