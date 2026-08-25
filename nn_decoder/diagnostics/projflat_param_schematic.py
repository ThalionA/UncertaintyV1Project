# -*- coding: utf-8 -*-
"""Parameter counts worked through, with a schematic of each decoder
(2026-08-04, Theo's request — after I mis-stated them by counting only W_in).

THE POINT: with 91 output bins, adding 8 hidden units REDUCES capacity. The
"linear" decoder is a full n_neurons -> 91 map; the H=8 decoder factors the same
map through an 8-dimensional bottleneck. At raw input the linear model therefore
has ~6x MORE parameters and ~10x higher rank — which is why it is the one that
overfits into a catastrophic tail (see projflat_fig9_tail_diagnosis).

Architecture (nn_classifier.SimpleFlexibleNNClassifier):
  no hidden : Linear(n_in -> 91)                     params = 91*n_in + 91
  H hidden  : Linear(n_in -> H) + Linear(H -> 91)    params = H*n_in + H + 91*H + 91

Both architectures are used identically by the spatial and temporal decoders — the
parameter count is the SAME for both. They differ only in WHERE the average over
time bins happens: spatial averages the INPUT then decodes once; temporal decodes
every bin then averages the 91-D OUTPUTS.

Real dimensions in this dataset: n_neurons = 65-153 (mean 108.5), ~291 training
trials per mouse after the 20% validation carve.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_param_schematic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

N_MEAN = 108.5          # mean neurons across the 6 mice (65,105,74,153,142,112)
N_TRAIN = 291           # mean training trials after the 20% val carve
N_OUT = 91              # orientation bins
H = 8


def p_linear(n_in):
    return N_OUT * n_in + N_OUT


def p_hidden(n_in, h=H):
    return h * n_in + h + N_OUT * h + N_OUT


def rank_linear(n_in):
    return min(n_in, N_OUT)


def rank_hidden(n_in, h=H):
    return min(n_in, h, N_OUT)


def draw_model(ax, n_in, hidden, title, in_label, rank_note=''):
    """Layer boxes + arrows + the arithmetic underneath."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title(title, fontsize=8.5, fontweight='bold')
    boxes = ([(1.2, in_label, ps.SPATIAL)] +
             ([(4.6, f'hidden\n{hidden}', ps.FLAT_EVAR)] if hidden else []) +
             [(8.0, f'output\n{N_OUT} bins', ps.TEMPORAL)])
    for xc, lab, colr in boxes:
        h_box = 4.6 if 'output' in lab else (2.2 if 'hidden' in lab else 4.0)
        ax.add_patch(Rectangle((xc - 0.75, 5.6 - h_box / 2), 1.5, h_box,
                               facecolor=colr, alpha=0.30, edgecolor='k', lw=1.0))
        ax.text(xc, 5.6, lab, ha='center', va='center', fontsize=7.5)
    xs = [b[0] for b in boxes]
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 0.8, 5.6), (x1 - 0.8, 5.6),
                                     arrowstyle='-|>', mutation_scale=12,
                                     lw=1.3, color='0.25'))
    # arithmetic
    if hidden:
        terms = (f'W₁ {hidden}×{n_in} = {hidden * n_in:,}\n'
                 f'b₁ {hidden} = {hidden}\n'
                 f'W₂ {N_OUT}×{hidden} = {N_OUT * hidden:,}\n'
                 f'b₂ {N_OUT} = {N_OUT}')
        tot = p_hidden(n_in, hidden); rk = rank_hidden(n_in, hidden)
    else:
        terms = (f'W {N_OUT}×{n_in} = {N_OUT * n_in:,}\n'
                 f'b {N_OUT} = {N_OUT}\n(single layer:\nW_in IS W_out)')
        tot = p_linear(n_in); rk = rank_linear(n_in)
    ax.text(0.4, 2.6, terms, fontsize=6.6, va='top', family='monospace')
    ax.text(9.6, 2.6, f'TOTAL {tot:,.0f}\n{tot / N_TRAIN:.1f} per train trial\nrank ≤ {rk:.0f}{rank_note}',
            fontsize=7.2, va='top', ha='right', fontweight='bold')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()
    ps.apply()

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    n = int(round(N_MEAN))
    draw_model(fig.add_subplot(gs[0, 0]), n, None,
               'A  linear, raw input', f'input\n{n} neurons',
               rank_note='\n(84 mean over mice:\nn ranges 65-153)')
    draw_model(fig.add_subplot(gs[0, 1]), n, H,
               'B  8 hidden, raw input', f'input\n{n} neurons')
    draw_model(fig.add_subplot(gs[0, 2]), 3, None,
               'C  linear, 3 neural PCs', 'input\n3 PCs')
    draw_model(fig.add_subplot(gs[0, 3]), 3, H,
               'D  8 hidden, 3 neural PCs', 'input\n3 PCs')

    # ---- params vs input dimensionality, with the crossover ----
    ax = fig.add_subplot(gs[1, :2])
    ks = np.arange(1, 160)
    ax.plot(ks, [p_linear(k) for k in ks], color=ps.PCA_EVAR, lw=2, label='linear (no hidden)')
    ax.plot(ks, [p_hidden(k) for k in ks], color=ps.FLAT_EVAR, lw=2, label='8 hidden units')
    kx = 736 / 83
    ax.axvline(kx, color='0.4', ls='--', lw=1.2)
    ax.text(kx * 1.06, 6e3, f'crossover\nk ≈ {kx:.1f}', fontsize=7, color='0.35')
    for k, mk in [(3, 'o'), (5, 's'), (10, '^'), (n, 'D')]:
        ax.plot(k, p_linear(k), mk, color=ps.PCA_EVAR, ms=6, mec='k', mew=0.5, zorder=5)
        ax.plot(k, p_hidden(k), mk, color=ps.FLAT_EVAR, ms=6, mec='k', mew=0.5, zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('input dimensionality (neural PCs, or raw neurons)', fontsize=8)
    ax.set_ylabel('total parameters', fontsize=8)
    ax.set_title('Parameters vs input width — the hidden layer only ADDS parameters below k ≈ 9',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=True)
    ax.grid(alpha=0.25, lw=0.5)

    # ---- effective rank ----
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(ks, [rank_linear(k) for k in ks], color=ps.PCA_EVAR, lw=2, label='linear (no hidden)')
    ax.plot(ks, [rank_hidden(k) for k in ks], color=ps.FLAT_EVAR, lw=2, label='8 hidden units')
    for k, mk in [(3, 'o'), (5, 's'), (10, '^'), (n, 'D')]:
        ax.plot(k, rank_linear(k), mk, color=ps.PCA_EVAR, ms=6, mec='k', mew=0.5, zorder=5)
        ax.plot(k, rank_hidden(k), mk, color=ps.FLAT_EVAR, ms=6, mec='k', mew=0.5, zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('input dimensionality', fontsize=8)
    ax.set_ylabel('effective rank of the input→output map', fontsize=8)
    ax.set_title('Effective rank — the 8-unit layer caps it at 8 however wide the input',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=True)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle('Parameter counts per decoder. With 91 output bins an 8-unit hidden layer is a rank BOTTLENECK, '
                 'not extra capacity:\nat raw input the LINEAR model has ~6× more parameters (9,964 vs 1,695) and '
                 '~10× the rank. Markers = the four input widths actually run (3, 5, 10 PCs, raw).\n'
                 'Spatial and temporal decoders share the architecture and count — they differ only in whether the '
                 'time-average is taken on the input or the output.', y=1.05, fontsize=8.2)
    ps.save_fig(fig, a.out_root, 'projflat_fig10_param_schematic', layout=None)
    print('  -> projflat_fig10_param_schematic.png/.svg')
    print(f"\n{'config':26s}{'formula':34s}{'total':>9s}{'/trial':>8s}{'rank':>6s}")
    for lab, k, hid in [('linear, raw', n, None), ('8 hidden, raw', n, H),
                        ('linear, 3 PCs', 3, None), ('8 hidden, 3 PCs', 3, H),
                        ('linear, 10 PCs', 10, None), ('8 hidden, 10 PCs', 10, H)]:
        if hid:
            f = f'{hid}×{k}+{hid}+{N_OUT}×{hid}+{N_OUT}'
            t, rk = p_hidden(k, hid), rank_hidden(k, hid)
        else:
            f = f'{N_OUT}×{k}+{N_OUT}'
            t, rk = p_linear(k), rank_linear(k)
        print(f'{lab:26s}{f:34s}{t:9,.0f}{t / N_TRAIN:8.1f}{rk:6.0f}')


if __name__ == '__main__':
    main()
