# -*- coding: utf-8 -*-
"""Setup schematic for the PCA-peakiness note (2026-06-09).

A visual demonstration of the decoding problem placed up front in the note, so
the reader meets the two decoders and the losses before any result:

  (a) the two readout hypotheses / forward passes — spatial (average activity over
      time, then one softmax) vs temporal (softmax each time-bin, then average);
  (b) the training objective — decoded vs ideal-observer target posterior, and what
      each loss penalises (PCA = variance-weighted PC-distance that weights the
      location PCs and ignores the width PCs; KL = full-distribution divergence).

Schematic (illustrative bumps, not model outputs). Output under figures/schematic/:
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


def _box(ax, x, y, w, h, text, fc='#eef2f7', ec='0.5', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=3)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=11, color='0.35', lw=1.3, zorder=1))


def fig_schematic(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1, panel_w=4.6, panel_h=3.4))

    # ---- (a) the two forward passes ----
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    # V1 activity tile (neurons x time)
    rng = np.random.default_rng(0)
    act = np.clip(np.stack([bump(40 + 6 * np.sin(t), 14) for t in range(6)]).T
                  + rng.normal(0, 0.18, (C, 6)), 0, None)
    axt = ax.inset_axes([0.0, 0.36, 0.20, 0.30])
    axt.imshow(act, aspect='auto', cmap='magma'); axt.set_xticks([]); axt.set_yticks([])
    axt.set_title('V1 activity\n(neurons × time)', fontsize=7.5)

    x = np.arange(C)
    # spatial path (top)
    _box(ax, 0.27, 0.70, 0.18, 0.13, 'average\nover time', '#fff3e6')
    _box(ax, 0.50, 0.70, 0.14, 0.13, 'MLP +\nsoftmax', '#eef2f7')
    axs = ax.inset_axes([0.70, 0.66, 0.27, 0.21])
    axs.fill_between(x, bump(40, 9), color=ps.SPATIAL, alpha=0.85, lw=0)
    axs.set_xticks([]); axs.set_yticks([]); axs.set_title('spatial posterior', fontsize=7.5)
    _arrow(ax, 0.21, 0.74, 0.27, 0.76); _arrow(ax, 0.45, 0.76, 0.50, 0.76)
    _arrow(ax, 0.64, 0.76, 0.70, 0.76)
    ax.text(0.62, 0.92, 'spatial:  softmax(MLP( mean$_t$ activity ))',
            ha='center', fontsize=8.5, color=ps.SPATIAL, fontweight='bold')

    # temporal path (bottom)
    _box(ax, 0.27, 0.16, 0.18, 0.13, 'MLP + softmax\nper time-bin', '#eef2f7')
    _box(ax, 0.50, 0.16, 0.14, 0.13, 'average\nposteriors', '#f0e9f5')
    axtm = ax.inset_axes([0.70, 0.12, 0.27, 0.21])
    axtm.fill_between(x, bump(40, 12), color=ps.TEMPORAL, alpha=0.85, lw=0)
    axtm.set_xticks([]); axtm.set_yticks([]); axtm.set_title('temporal posterior', fontsize=7.5)
    _arrow(ax, 0.21, 0.46, 0.27, 0.24); _arrow(ax, 0.45, 0.22, 0.50, 0.22)
    _arrow(ax, 0.64, 0.22, 0.70, 0.22)
    ax.text(0.62, 0.02, 'temporal:  mean$_t$ softmax(MLP( activity$_t$ ))',
            ha='center', fontsize=8.5, color=ps.TEMPORAL, fontweight='bold')
    ps.panel_label(ax, 'a')

    # ---- (b) the losses ----
    ax = axes[1]
    ps.target_band(ax, x, bump(46, 9, 0.9) / bump(46, 9, 0.9).sum() * 8, label='IO target')
    dec = bump(45, 4, 1.0)
    for c in (28, 60):
        dec = dec + 0.4 * bump(c, 3)
    dec = dec / dec.sum() * 8
    ax.plot(x, dec, color=ps.PCA_EVAR, lw=1.8, label='decoded (peaky)')
    ax.set_xlabel('orientation bin'); ax.set_ylabel('probability')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_yticks([])
    ax.text(0.03, 0.80,
            'PCA loss  $=100\\sum_k \\mathrm{evar}_k(\\langle\\hat p,u_k\\rangle-\\langle t,u_k\\rangle)^2$\n'
            'weights location PCs, ignores width PCs\n'
            'KL  $=\\sum_k t_k\\log(t_k/\\hat p_k)$  — full distribution',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    ps.panel_label(ax, 'b')

    fig.suptitle('Decoding setup: two readouts and the training losses', y=1.0)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, 'setup_schematic')


def main(out_root):
    ps.apply()
    fig_schematic(Path(out_root))
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/schematic')
    a = ap.parse_args()
    main(a.out_root)
