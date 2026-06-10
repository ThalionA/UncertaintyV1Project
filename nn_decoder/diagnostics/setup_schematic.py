# -*- coding: utf-8 -*-
"""Setup schematic for the PCA-peakiness note (2026-06-09; reworked 2026-06-10).

A data-flow picture placed up front so the reader meets the two decoders and the
losses before any result:

  (a) the data → two readouts. V1 activity is a (neurons × time-bins) matrix per
      trial (a stack over trials). The **spatial** decoder averages over time first
      → one MLP+softmax → one posterior. The **temporal** decoder runs the MLP+
      softmax on each time-bin → averages the T posteriors. The MLP is drawn as a
      small node diagram; both paths end in a decoded posterior over orientation.
  (b) the objective: decoded vs ideal-observer target, and what each loss
      penalises (PCA = evar-weighted PC-distance, blind to the width PCs; KL =
      full-distribution divergence).

Illustrative bumps, not model outputs. Output under figures/schematic/:
  setup_schematic.png

Usage:  python diagnostics/setup_schematic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

C = 91


def bump(centre, width, amp=1.0):
    x = np.arange(C)
    d = np.minimum(np.abs(x - centre), C - np.abs(x - centre))
    p = np.exp(-0.5 * (d / width) ** 2)
    return amp * p / p.max()


def _box(ax, x, y, w, h, text, fc='#eef2f7', ec='0.5', fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.010',
                                fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color='0.4'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=10, color=color, lw=1.3, zorder=2))


def _mlp(ax, x, y, w, h, layers=(4, 3, 4), color='#2e7d6b'):
    """Tiny MLP node-and-edge diagram inside [x,x+w]×[y,y+h] (axes coords)."""
    cols = np.linspace(x, x + w, len(layers))
    pts = [[(cx, yy) for yy in np.linspace(y + 0.12 * h, y + 0.88 * h, n)]
           for cx, n in zip(cols, layers)]
    for a in range(len(pts) - 1):
        for p in pts[a]:
            for q in pts[a + 1]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color='0.78', lw=0.4, zorder=2)
    for li, col_pts in enumerate(pts):
        fc = color if 0 < li < len(pts) - 1 else '0.55'
        ax.scatter([p[0] for p in col_pts], [p[1] for p in col_pts],
                   s=42, c=fc, edgecolors='0.25', linewidths=0.5, zorder=4)


def _tile(ax, x, y, w, h, label):
    """A (neurons × time) activity tile with two shadow copies behind = trials."""
    rng = np.random.default_rng(0)
    act = np.clip(np.stack([bump(40 + 6 * np.sin(t), 14) for t in range(7)]).T
                  + rng.normal(0, 0.18, (C, 7)), 0, None)
    for k in (2, 1):                                       # trial-stack shadows
        sub = ax.inset_axes([x + 0.012 * k, y + 0.016 * k, w, h])
        sub.imshow(act, aspect='auto', cmap='magma'); sub.set_xticks([]); sub.set_yticks([])
        for s in sub.spines.values():
            s.set_edgecolor('0.6'); s.set_linewidth(0.8)
    main = ax.inset_axes([x, y, w, h])
    main.imshow(act, aspect='auto', cmap='magma')
    main.set_xlabel('time-bins', fontsize=6.5, labelpad=1)
    main.set_ylabel('neurons', fontsize=6.5, labelpad=1)
    main.set_xticks([]); main.set_yticks([])
    ax.text(x + w / 2 + 0.02, y + h + 0.075, label, ha='center', va='bottom',
            fontsize=7.5, transform=ax.transAxes)


def _post(ax, x, y, w, h, centre, width, color, title):
    sub = ax.inset_axes([x, y, w, h])
    xx = np.arange(C)
    sub.fill_between(xx, bump(centre, width), color=color, alpha=0.85, lw=0)
    sub.set_xticks([]); sub.set_yticks([])
    sub.set_title(title, fontsize=7)


def fig_schematic(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1, panel_w=4.9, panel_h=3.6))

    # ---- (a) data -> two readouts ----
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    _tile(ax, 0.02, 0.40, 0.17, 0.20, 'V1 activity\n(neurons × time-bins) × N trials')

    # spatial path (top)
    ax.text(0.60, 0.96, 'spatial (PPC):  softmax( MLP( $\\overline{a}_t$ ) )',
            ha='center', fontsize=8.5, color=ps.SPATIAL, fontweight='bold')
    _box(ax, 0.26, 0.74, 0.135, 0.11, 'average\nover time', '#fff3e6')
    _mlp(ax, 0.45, 0.71, 0.13, 0.17)
    _box(ax, 0.61, 0.74, 0.085, 0.11, 'softmax', '#eef2f7')
    _post(ax, 0.70, 0.70, 0.26, 0.19, 40, 9, ps.SPATIAL, 'spatial posterior')
    _arrow(ax, 0.205, 0.55, 0.26, 0.79, ps.SPATIAL)
    _arrow(ax, 0.395, 0.795, 0.45, 0.795); _arrow(ax, 0.58, 0.795, 0.61, 0.795)
    _arrow(ax, 0.695, 0.795, 0.70, 0.795)

    # temporal path (bottom)
    ax.text(0.60, 0.015, 'temporal (SBC):  $\\overline{\\mathrm{softmax}( MLP( a_t ) )}$',
            ha='center', fontsize=8.5, color=ps.TEMPORAL, fontweight='bold')
    _mlp(ax, 0.26, 0.10, 0.13, 0.17)
    _box(ax, 0.45, 0.13, 0.115, 0.11, 'softmax\nper time-bin', '#eef2f7')
    _box(ax, 0.61, 0.13, 0.085, 0.11, 'average\nposteriors', '#f0e9f5')
    _post(ax, 0.70, 0.09, 0.26, 0.19, 40, 12, ps.TEMPORAL, 'temporal posterior')
    _arrow(ax, 0.205, 0.45, 0.26, 0.21, ps.TEMPORAL)
    _arrow(ax, 0.39, 0.185, 0.45, 0.185); _arrow(ax, 0.565, 0.185, 0.61, 0.185)
    _arrow(ax, 0.695, 0.185, 0.70, 0.185)
    ps.panel_label(ax, 'a')

    # ---- (b) decoded vs target + the losses ----
    ax = axes[1]
    x = np.arange(C)
    tgt = bump(46, 9, 0.9); tgt = tgt / tgt.sum() * 8
    ps.target_band(ax, x, tgt, label='IO target')
    dec = bump(45, 4, 1.0)
    for c in (28, 60):
        dec = dec + 0.4 * bump(c, 3)
    dec = dec / dec.sum() * 8
    ax.plot(x, dec, color=ps.PCA_EVAR, lw=1.8, label='decoded (peaky)')
    ax.set_xlabel('orientation bin'); ax.set_ylabel('probability'); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)
    ax.text(0.03, 0.78,
            'PCA loss  $=\\sum_k \\mathrm{evar}_k(\\langle\\hat p,u_k\\rangle-\\langle t,u_k\\rangle)^2$\n'
            'weights the location PCs, blind to the width PCs\n'
            'KL  $=\\sum_k t_k\\log(t_k/\\hat p_k)$  — full distribution',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    ps.panel_label(ax, 'b')

    fig.suptitle('Decoding setup: V1 activity → two readouts → decoded vs IO target', y=1.0)
    ps.save_fig(fig, out_dir, 'setup_schematic', layout=None)  # manual layout (insets)


def main(out_root):
    ps.apply()
    fig_schematic(Path(out_root))
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/schematic')
    a = ap.parse_args()
    main(a.out_root)
