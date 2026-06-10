# -*- coding: utf-8 -*-
"""Ideal-observer model schematic (theory -> process -> fitting).

A single composed figure telling the whole IO story the v2 two-stage fit
implements (see `documents/ideal_observer_methods_v3.tex`):

  1. Generative model  — the world (category c, orientation s, contrast/dispersion
     c,d) produces a noisy internal measurement m ~ VonMises(s, kappa(c,d)); the
     task prior p(s) is bimodal at the 0 deg / 90 deg targets.
  2. Observer inference — Bayes update m -> posterior p(s|m); a utility function
     U(A,s) maps the posterior to expected utilities EU(Go), EU(NoGo); their
     difference is the decision variable DV(m), which drives the observable
     behaviour: the binary choice AND the continuous kinematic confidence proxies
     (running velocity, anticipatory licks).
  3. Fitting & inversion — a two-stage hierarchical fit (Stage 1: sensory +
     kinematic-emission params from kinematics alone, via BADS, 5-fold CV;
     Stage 2: a 4-param choice psychometric on the log posterior odds g(m),
     velocity-conditioned). After fitting, a marginalised trial-by-trial inversion
     exports the perceptual posterior Q(theta), the likelihood L(theta) and the
     decision posterior [P(Go), P(NoGo)] — the ground-truth targets the neural
     decoder is trained against — plus the perceptual / decision uncertainty
     read-outs.

Illustrative curves, not model outputs. PNG (<=1600 px) + SVG under
figures/schematic/:  io_model_schematic.{png,svg}

Usage:  python diagnostics/io_schematic.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

# ----------------------------------------------------------------------
# palette — the model schematic, not the loss palette
# ----------------------------------------------------------------------
ENV_EC    = '#1f3b73'   # navy  — solid "environment / world" frames
BRAIN_EC  = '#6a3d9a'   # purple (dashed) — "brain" frames
REGION_FC = '#fbfcff'   # near-white region fill
C_COL     = '#2e9e5b'   # green  — category node / decision-value node
S_COL     = '#e6720d'   # orange — orientation / posterior node
M_COL     = '#5b6470'   # grey   — internal measurement
BOX_FC    = '#eef2f7'   # generic box fill
UTIL_FC   = '#fff3e6'   # warm box fill (utility / value)
STAGE1_FC = '#e8f1fb'   # stage-1 fit box
STAGE2_FC = '#f0e9f7'   # stage-2 fit box
INV_FC    = '#e9f6ee'   # inversion box

GRID = np.linspace(0, 90, 181)


# ----------------------------------------------------------------------
# small numeric shapes for the inset mini-plots
# ----------------------------------------------------------------------
def _bump(centre, width, amp=1.0):
    return amp * np.exp(-0.5 * ((GRID - centre) / width) ** 2)


def _prior():               # bimodal task prior at the 0/90 targets
    p = _bump(4, 11) + _bump(86, 11)
    return p / p.max()


def _vonmises(centre=38, kappa=6):   # p(m|s): a single concentrated bump
    p = np.exp(kappa * np.cos(np.deg2rad(2 * (GRID - centre))))
    return p / p.max()


def _posterior():           # p(s|m): prior-shaped, peak near the measurement
    p = _bump(33, 9) * (0.55 + 0.9 * _prior())
    return p / p.max()


def _logistic(k=0.16, x0=45):
    return 1.0 / (1.0 + np.exp(-k * (GRID - x0)))


# ----------------------------------------------------------------------
# drawing primitives (all in data coords; aspect is equal)
# ----------------------------------------------------------------------
def region(ax, x, y, w, h, title, ec, dashed=False, title_xy='tl'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.18',
                                fc=REGION_FC, ec=ec, lw=2.0, zorder=1,
                                linestyle='--' if dashed else '-'))
    if title_xy == 'tl':
        ax.text(x + 0.18, y + h - 0.02, title, ha='left', va='top',
                fontsize=12, fontweight='bold', color=ec, zorder=6)


def subframe(ax, x, y, w, h, label, ec, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.12',
                                fc='none', ec=ec, lw=1.4, zorder=2,
                                linestyle=(0, (5, 3)) if dashed else '-'))
    ax.text(x + w - 0.1, y + h - 0.08, label, ha='right', va='top',
            fontsize=8.5, style='italic', color=ec, zorder=6)


def box(ax, cx, cy, w, h, text, fc=BOX_FC, ec='0.5', fs=8.0, lw=1.1, weight='normal'):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, zorder=5,
            fontweight=weight)


def node(ax, cx, cy, text, color, r=0.30, fs=11):
    ax.add_patch(Circle((cx, cy), r, fc='white', ec=color, lw=2.2, zorder=5))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=color, zorder=6)


def arrow(ax, p0, p1, color='0.35', lw=1.6, style='-|>', rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, zorder=3,
                                 linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=2, shrinkB=2))


def label(ax, x, y, text, fs=7.8, color='0.25', ha='center', va='center', it=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=6,
            style='italic' if it else 'normal')


def mini(ax, x, y, w, h, kind, color, title=None):
    """Mini distribution/plot, placed in data coords, no ticks."""
    sub = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=4)
    if kind == 'prior':
        sub.fill_between(GRID, _prior(), color='#3b78c4', alpha=0.45, lw=0)
        sub.plot(GRID, _prior(), color='#3b78c4', lw=1.1)
    elif kind == 'vonmises':
        sub.plot(GRID, _vonmises(), color=color, lw=1.6)
    elif kind == 'posterior':
        sub.fill_between(GRID, _posterior(), color=color, alpha=0.30, lw=0)
        sub.plot(GRID, _posterior(), color=color, lw=1.6)
    elif kind == 'logistic':
        sub.plot(GRID, _logistic(), color=color, lw=1.7)
        sub.axvline(45, color='0.6', ls=':', lw=0.8)
        sub.set_ylim(-0.05, 1.05)
    elif kind == 'eu':
        sub.bar([0, 1], [0.82, 0.34], color=[C_COL, '#9bb8a8'],
                width=0.62, edgecolor='0.3', lw=0.6)
        sub.set_xticks([0, 1]); sub.set_xticklabels(['Go', 'NoGo'], fontsize=6.5)
        sub.set_ylim(0, 1.0)
        sub.tick_params(length=0, labelbottom=True)
        sub.set_yticks([])
        sub.set_title('EU', fontsize=6.8, pad=1)
        return sub
    elif kind == 'gauss':
        g = _bump(45, 13)
        sub.fill_between(GRID, g, color=color, alpha=0.25, lw=0)
        sub.plot(GRID, g, color=color, lw=1.4)
    elif kind == 'target':
        t = _posterior()
        sub.fill_between(GRID, t, color='0.72', alpha=0.85, lw=0)
        sub.plot(GRID, t, color='0.45', lw=1.2)
    sub.set_xticks([]); sub.set_yticks([])
    for s in sub.spines.values():
        s.set_visible(False)
    if title and kind != 'eu':
        sub.set_title(title, fontsize=7, pad=2)
    return sub


# ----------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------
def build(out_dir):
    ps.apply()
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.set_aspect('equal'); ax.axis('off')

    fig.text(0.5, 0.975, 'Ideal-observer model: from stimulus to trial-by-trial perceptual posterior',
             ha='center', va='top', fontsize=15, fontweight='bold')

    # =================================================================
    # ROW 1 LEFT — Generative model
    # =================================================================
    region(ax, 0.2, 4.95, 6.35, 3.45, '1 · Generative model', ENV_EC)

    # -- environment subframe --
    subframe(ax, 0.55, 6.35, 5.65, 1.75, 'environment', ENV_EC)
    node(ax, 1.3, 7.55, 'c', C_COL)
    label(ax, 1.95, 7.55, 'stimulus category\n(horizontal / vertical)', ha='left', fs=7.4)
    node(ax, 1.3, 6.75, 's', S_COL)
    label(ax, 1.95, 6.75, 'orientation s\n(0–90°)', ha='left', fs=7.4)
    arrow(ax, (1.3, 7.25), (1.3, 7.05), color=C_COL)
    node(ax, 5.6, 7.45, 'c,d', M_COL, fs=9)
    label(ax, 5.6, 6.92, 'contrast · dispersion\n(stimulus strength)', fs=7.0)
    mini(ax, 3.35, 6.52, 1.25, 0.62, 'prior', '#3b78c4', title='prior p(s)')

    # -- brain subframe --
    subframe(ax, 0.55, 5.1, 5.65, 1.1, 'brain', BRAIN_EC, dashed=True)
    box(ax, 1.4, 5.6, 0.62, 0.5, 'm', fc='white', ec=M_COL, fs=12, lw=2.0, weight='bold')
    label(ax, 2.5, 5.6, 'internal\nmeasurement', ha='left', fs=7.4)
    mini(ax, 4.05, 5.28, 1.25, 0.6, 'vonmises', M_COL, title='p(m|s) = VonMises(s, κ)')
    # generative arrows into m
    arrow(ax, (1.3, 6.45), (1.4, 5.88), color=S_COL, rad=0.0)      # s -> m
    arrow(ax, (5.5, 7.12), (1.78, 5.7), color=M_COL, rad=0.22)     # (c,d) -> m
    label(ax, 3.2, 4.78, 'κ(c,d) = (κ$_{min}$+κ$_{amp}$)·c$^{\\,p_c}$·e$^{-p_d d}$   (isotropic precision)',
          fs=7.4, color='0.3')

    # =================================================================
    # ROW 1 RIGHT — Observer inference
    # =================================================================
    region(ax, 6.85, 4.95, 6.95, 3.45, '2 · Observer inference', BRAIN_EC)

    # behaviour (environment) strip at the top
    subframe(ax, 7.2, 7.45, 6.3, 0.6, '', ENV_EC)
    label(ax, 7.35, 7.97, 'behaviour (observed)', fs=8, color=ENV_EC, ha='left', it=True)
    box(ax, 9.45, 7.74, 2.25, 0.4, 'choice  (Go / NoGo)', fc='white', ec=ENV_EC, fs=8)
    box(ax, 12.15, 7.74, 1.95, 0.4, 'confidence\n(velocity · licks)', fc='white', ec=ENV_EC, fs=7.4)

    # brain inference chain
    subframe(ax, 7.2, 5.05, 6.3, 2.32, 'brain', BRAIN_EC, dashed=True)
    node(ax, 7.7, 5.5, 'm', M_COL, r=0.27, fs=10)
    label(ax, 7.7, 5.13, 'likelihood', fs=6.8, color=M_COL)
    node(ax, 7.7, 6.4, 's|m', S_COL, r=0.30, fs=9)
    arrow(ax, (7.7, 5.8), (7.7, 6.1), color=S_COL)
    label(ax, 8.18, 5.95, '× prior\np(s)', fs=6.8, ha='left')
    mini(ax, 8.45, 5.18, 1.1, 0.58, 'posterior', S_COL, title='posterior p(s|m)')

    # value chain: posterior -> utility -> EU -> DV
    arrow(ax, (8.0, 6.4), (9.45, 6.4), color='0.4')
    label(ax, 8.7, 6.62, '× utility', fs=6.8)
    box(ax, 10.4, 6.4, 1.95, 1.0,
        'utility U(A, s)\n[R$_{hit}$, R$_{miss}$, R$_{CR}$, R$_{FA}$]\n= [1, 0, .1, −.2]',
        fc=UTIL_FC, ec='#c98a3a', fs=7.3)
    mini(ax, 11.55, 6.05, 0.66, 0.72, 'eu', C_COL)
    arrow(ax, (12.25, 6.4), (12.6, 6.4), color='0.4')
    node(ax, 12.95, 6.4, 'ΔEU', C_COL, r=0.30, fs=8)
    label(ax, 11.95, 5.55, 'DV(m) = EU(Go) − EU(NoGo)', fs=6.9, ha='center', color=C_COL)

    # outputs of inference: choice from log-odds, confidence from DV
    arrow(ax, (7.95, 6.62), (9.15, 7.45), color='0.4', rad=0.14)      # posterior -> choice
    label(ax, 8.32, 7.18, 'log-odds g(m)\n→ P(Go | g(m))', fs=6.7, ha='left')
    arrow(ax, (12.95, 6.7), (12.25, 7.45), color='0.4', rad=-0.08)    # DV -> confidence
    label(ax, 12.46, 6.98, 'y = β·DV\n+ α + ε', fs=6.7, ha='left')

    # =================================================================
    # ROW 2 — Fitting & inversion (full width)
    # =================================================================
    region(ax, 0.2, 0.3, 13.6, 4.25, '3 · Fitting & trial-by-trial inversion', ENV_EC)

    yb = 3.05   # main pipeline row centre
    # observed behaviour (input to the fit)
    box(ax, 1.45, yb, 1.9, 1.15,
        'observed behaviour\n\nvelocity · licks\nchoices',
        fc='white', ec=ENV_EC, fs=8)

    # Stage 1
    box(ax, 4.55, yb, 2.5, 1.45,
        'Stage 1 — sensory + emission\nfrom kinematics alone\n'
        '{κamp, p$_c$, p$_d$, β, α, σ}\nBADS · hierarchical · 5-fold CV',
        fc=STAGE1_FC, ec='#3b6fb0', fs=7.8)
    # Stage 2
    box(ax, 7.65, yb, 2.5, 1.45,
        'Stage 2 — choice psychometric\non log-odds g(m) = log $\\frac{P(Go|m)}{P(NoGo|m)}$\n'
        '{α$_r$, β$_r$, γ$_r$, δ$_r$}\nvelocity-conditioned · 5-fold CV',
        fc=STAGE2_FC, ec='#7a52a8', fs=7.8)
    # Inversion
    box(ax, 10.85, yb, 2.55, 1.45,
        'Marginalised inversion\nintegrate over p(m | s, y)\n'
        '→ trial-by-trial targets',
        fc=INV_FC, ec='#2e9e5b', fs=8.0)

    arrow(ax, (2.42, yb), (3.28, yb), color='0.4')
    arrow(ax, (5.82, yb), (6.38, yb), color='0.4')
    arrow(ax, (8.92, yb), (9.55, yb), color='0.4')

    # outputs column (right edge)
    label(ax, 12.55, 3.98, 'targets', fs=8.5, color='#2e9e5b', it=True)
    mini(ax, 11.95, 3.2, 1.2, 0.6, 'target', '0.5', title='Q(θ)  posterior')
    label(ax, 12.55, 2.66, 'L(θ) likelihood', fs=6.9, ha='center')
    label(ax, 12.55, 2.44, '[P(Go), P(NoGo)]', fs=6.9, ha='center')
    arrow(ax, (12.13, 3.32), (12.4, 3.5), color='0.4', rad=-0.1)

    # downstream uses (bottom band)
    box(ax, 4.0, 1.15, 4.4, 0.95,
        'neural decoder target\nV1 activity → Q(θ)   (spatial / temporal read-outs)',
        fc='white', ec='0.45', fs=8.2)
    box(ax, 10.0, 1.15, 5.3, 0.95,
        'uncertainty read-outs\n'
        'perceptual  U$_{perc}$ = SD[Q(θ)]      decision  U$_{dec}$ = H[P(Go)]',
        fc='white', ec='0.45', fs=8.2)
    arrow(ax, (10.4, 2.33), (5.0, 1.63), color='0.5', rad=-0.16)
    arrow(ax, (10.9, 2.33), (10.0, 1.63), color='0.5', rad=0.03)

    ps.save_fig(fig, out_dir, 'io_model_schematic', layout=None)


def main(out_root):
    build(Path(out_root))
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/schematic')
    a = ap.parse_args()
    main(a.out_root)
