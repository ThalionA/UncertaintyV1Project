# -*- coding: utf-8 -*-
"""hp_fig8 — round-trip / recovery control: over-sharpening is NOT overfitting.

Replot of the cached `roundtrip_loss_refit.py` run (nothing is retrained here;
this reads `results/roundtrip_refit/<cell>/mouse_<i>.npz`, 6 mice x 6 cells).

Design (from PROJECT_LOG 2026-07-08): 3 TARGET SOURCES
  {real IO posterior, Projection-fitted posterior (peaky), KL-fitted posterior
   (broad but, being a net's own output, exactly achievable)}
x 2 REFIT LOSSES {projection-based ('PCA'), KL}, Q / half / 100 ms,
entropy_lambda=0, patience=0 (full 200-epoch trajectory), monitored 20% val.

Four rows, spatial | temporal columns (sharey per row so the two archs are
directly comparable):
  a,b  final decoded peakiness by target source, with that cell's own target
       peakiness as the grey reference;
  c,d  val-curve upturn (final val fit-loss / its minimum over epochs) — the
       scale-free overfitting index; 1.0 = no upturn;
  e,f  decoded peakiness / target peakiness vs epoch (the over-sharpening
       trajectory), from the 21 weight snapshots;
  g,h  val fit-loss / its own minimum vs epoch (the overfitting trajectory).

All points/bands are mean +/- SEM over the 6 mice.

Usage:  python diagnostics/roundtrip_recovery_fig.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / 'results' / 'roundtrip_refit'

TARGETS = ['realIO', 'pcaFit', 'klFit']
TGT_LABEL = {'realIO': 'real IO', 'pcaFit': 'Projection-fit', 'klFit': 'KL-fit'}
TGT_LS = {'realIO': '-', 'pcaFit': '--', 'klFit': '-.'}
LOSSES = ['PCA', 'KL']
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
OFFSET = {'PCA': -0.22, 'KL': +0.22}
N_MICE = 6
# The val curves fall ~40x over the first ~15 epochs, which would squash the
# late upturn (the quantity of interest) to a flat line; clip to the tail.
UPTURN_YLIM = (0.95, 2.0)


# ----------------------------------------------------------------- loading
def load(target, loss, arch, mice=range(N_MICE)):
    """Stack the cached per-mouse curves for one (target source, refit loss, arch).

    Returns a dict of arrays with a leading mouse axis:
      epochs      (n_snap,)   weight-snapshot epochs (identical across mice)
      peak        (n_mice, n_snap)   decoded max-prob at each snapshot
      tgt         (n_mice,)          that mouse's TARGET max-prob
      val         (n_mice, n_ep)     per-epoch val fit-loss (training loss units)
      train       (n_mice, n_ep)     per-epoch train fit-loss
    """
    peak, tgt, val, train, eps = [], [], [], [], None
    for m in mice:
        f = CACHE / f'{target}_{loss}' / f'mouse_{m}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        eps = d[f'{arch}_snap_epochs'] if eps is None else eps
        peak.append(d[f'{arch}_peak'])
        tgt.append(float(d[f'{arch}_target_peak']))
        val.append(d[f'{arch}_val_fit'])
        train.append(d[f'{arch}_train_fit'])
    return dict(epochs=np.asarray(eps), peak=np.vstack(peak), tgt=np.asarray(tgt),
                val=np.vstack(val), train=np.vstack(train))


def _ms(a, axis=0):
    """mean, SEM over the mouse axis."""
    a = np.asarray(a, float)
    return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis) / np.sqrt(a.shape[axis])


# ------------------------------------------------------------------ figure
def figure(out_root, mice=range(N_MICE)):
    ps.apply()
    fig, ax = plt.subplots(4, 2, figsize=ps.figsize(2, 4), sharey='row',
                           constrained_layout=True)
    D = {(t, l, a): load(t, l, a, mice) for t in TARGETS for l in LOSSES
         for a, _ in ARCHS}

    for c, (arch, alab) in enumerate(ARCHS):
        # -- a,b  final decoded peakiness, vs its own target ---------------
        A = ax[0][c]
        for loss in LOSSES:
            m, s = zip(*[_ms(D[(t, loss, arch)]['peak'][:, -1]) for t in TARGETS])
            A.errorbar(np.arange(3) + OFFSET[loss], m, yerr=s, fmt='o', ms=6,
                       color=ps.color(loss), capsize=3, lw=1.6)
        tm, ts = zip(*[_ms(D[(t, 'PCA', arch)]['tgt']) for t in TARGETS])
        A.errorbar(np.arange(3), tm, yerr=ts, fmt='_', ms=22, mew=2.4,
                   color='0.45', capsize=3, lw=1.6, zorder=1)
        A.set_ylabel('decoded peakiness (max-prob)')

        # -- c,d  val-curve upturn index -----------------------------------
        B = ax[1][c]
        for loss in LOSSES:
            up = [D[(t, loss, arch)]['val'] for t in TARGETS]
            m, s = zip(*[_ms(v[:, -1] / np.nanmin(v, axis=1)) for v in up])
            B.errorbar(np.arange(3) + OFFSET[loss], m, yerr=s, fmt='o', ms=6,
                       color=ps.color(loss), capsize=3, lw=1.6)
        B.axhline(1.0, ls=':', lw=1.2, color=ps.CHANCE_GREY)
        B.set_ylabel('val fit-loss: final / minimum')

        for A2 in (A, B):
            A2.set_xticks(range(3))
            A2.set_xticklabels([TGT_LABEL[t] for t in TARGETS])
            A2.set_xlabel('target source')
            A2.set_xlim(-0.55, 2.55)

        # -- e,f  over-sharpening trajectory -------------------------------
        C = ax[2][c]
        for t in TARGETS:
            for loss in LOSSES:
                d = D[(t, loss, arch)]
                m, s = _ms(d['peak'] / d['tgt'][:, None])
                C.plot(d['epochs'], m, color=ps.color(loss), ls=TGT_LS[t], lw=2.0,
                       marker='o', ms=2.5)
                C.fill_between(d['epochs'], m - s, m + s, color=ps.color(loss),
                               alpha=0.12, lw=0)
        C.axhline(1.0, ls=':', lw=1.2, color=ps.CHANCE_GREY)
        C.set_ylabel('decoded peakiness / target peakiness')
        C.set_xlabel('epoch (weight snapshot)')

        # -- g,h  overfitting trajectory -----------------------------------
        E = ax[3][c]
        for t in TARGETS:
            for loss in LOSSES:
                v = D[(t, loss, arch)]['val']
                m, s = _ms(v / np.nanmin(v, axis=1, keepdims=True))
                x = np.arange(1, v.shape[1] + 1)
                E.plot(x, m, color=ps.color(loss), ls=TGT_LS[t], lw=2.0)
                E.fill_between(x, m - s, m + s, color=ps.color(loss),
                               alpha=0.12, lw=0)
        E.axhline(1.0, ls=':', lw=1.2, color=ps.CHANCE_GREY)
        E.set_ylabel('val fit-loss / its minimum')
        E.set_xlabel('epoch')
        E.set_ylim(*UPTURN_YLIM)

        for A2 in (A, B, C, E):
            A2.set_title(alab)

    handles = [
        Line2D([0], [0], color=ps.color('PCA'), lw=2.2, marker='o', ms=5),
        Line2D([0], [0], color=ps.color('KL'), lw=2.2, marker='o', ms=5),
        Line2D([0], [0], color='0.45', lw=1.6, marker='_', ms=12, mew=2.4,
               ls='none'),
        Line2D([0], [0], color='k', lw=1.8, ls=TGT_LS['realIO']),
        Line2D([0], [0], color='k', lw=1.8, ls=TGT_LS['pcaFit']),
        Line2D([0], [0], color='k', lw=1.8, ls=TGT_LS['klFit']),
        Line2D([0], [0], color=ps.CHANCE_GREY, lw=1.2, ls=':'),
    ]
    labels = [f'refit loss: {ps.loss_label("PCA")}', 'refit loss: KL',
              'target posterior peakiness',
              'target: real IO', 'target: Projection-fit', 'target: KL-fit',
              'reference (= target / no upturn)']
    fig.legend(handles, labels, loc='outside lower center', ncol=4,
               frameon=False, fontsize=8.5)
    fig.suptitle('Round-trip refit: target source x refit loss '
                 f'(Q half 100 ms, mean +/- SEM over {len(list(mice))} mice)')
    ps.label_panels(ax.ravel())
    ps.save_fig(fig, Path(out_root), 'hp_fig8_roundtrip_recovery')


# ----------------------------------------------------------------- numbers
def scorecard(mice=range(N_MICE)):
    for arch, alab in ARCHS:
        print(f'\n=== {alab} ===')
        print(f'{"cell":14s} {"peak":>14s} {"target":>14s} {"ratio/mouse":>14s} '
              f'{"ratio of means":>15s} {"val upturn":>14s} {"val/train":>14s}')
        for t in TARGETS:
            for loss in LOSSES:
                d = load(t, loss, arch, mice)
                pk, tg, v, tr = d['peak'][:, -1], d['tgt'], d['val'], d['train']
                up = v[:, -1] / np.nanmin(v, axis=1)
                vt = v[:, -1] / tr[:, -1]
                f = lambda a: '{:.3f}+-{:.3f}'.format(*_ms(a))  # noqa: E731
                print(f'{t + "_" + loss:14s} {f(pk):>14s} {f(tg):>14s} '
                      f'{f(pk / tg):>14s} {np.mean(pk) / np.mean(tg):15.2f} '
                      f'{f(up):>14s} {f(vt):>14s}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/hparam_summary')
    a = ap.parse_args()
    figure(a.out_root)
    scorecard()
    print(f'\nDone -> {(Path(a.out_root) / "hp_fig8_roundtrip_recovery.png").resolve()}')


if __name__ == '__main__':
    main()
