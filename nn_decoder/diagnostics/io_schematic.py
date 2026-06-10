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
        for s in sub.spines.values():
            s.set_visible(False)
        sub.set_title('EU', fontsize=6.8, pad=1)
        return sub
    elif kind == 'decision':                         # decision posterior [P(Go), P(NoGo)]
        sub.bar([0, 1], [0.68, 0.32], color=['#3b78c4', '#c3ccd6'],
                width=0.6, edgecolor='0.3', lw=0.6)
        sub.set_xticks([0, 1]); sub.set_xticklabels(['Go', 'NoGo'], fontsize=7.0)
        sub.set_ylim(0, 1.0)
        sub.tick_params(length=0, labelbottom=True)
        sub.set_yticks([])
        for s in sub.spines.values():
            s.set_visible(False)
        if title:
            sub.set_title(title, fontsize=7.4, pad=3)
        return sub
    elif kind == 'likelihood':                       # L(theta): evidence, no task prior
        g = _bump(34, 12)
        sub.fill_between(GRID, g, color=color, alpha=0.28, lw=0)
        sub.plot(GRID, g, color=color, lw=1.6)
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
    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 10.5); ax.set_aspect('equal'); ax.axis('off')

    ax.text(7.75, 10.2, 'Ideal-observer model: from stimulus to trial-by-trial perceptual posterior',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # =================================================================
    # ROW 1 LEFT — Generative model
    # =================================================================
    region(ax, 0.4, 6.0, 7.0, 3.8, '1 · Generative model', ENV_EC)

    # -- environment subframe --
    subframe(ax, 0.8, 7.85, 6.35, 1.6, 'environment', ENV_EC)
    node(ax, 1.7, 8.9, 'c', C_COL)
    label(ax, 2.45, 8.9, 'stimulus category\n(horizontal / vertical)', ha='left', fs=7.6)
    node(ax, 1.7, 8.15, 's', S_COL)
    label(ax, 2.45, 8.15, 'orientation s\n(0–90°)', ha='left', fs=7.6)
    arrow(ax, (1.7, 8.6), (1.7, 8.45), color=C_COL)
    node(ax, 6.45, 8.7, 'c,d', M_COL, fs=9)
    label(ax, 6.45, 8.08, 'contrast · dispersion\n(stimulus strength)', fs=7.2)
    mini(ax, 4.15, 8.0, 1.45, 0.72, 'prior', '#3b78c4', title='prior p(s)')

    # -- brain subframe --
    subframe(ax, 0.8, 6.25, 6.35, 1.4, 'brain', BRAIN_EC, dashed=True)
    box(ax, 1.95, 6.95, 0.66, 0.52, 'm', fc='white', ec=M_COL, fs=12, lw=2.0, weight='bold')
    label(ax, 2.85, 6.95, 'internal\nmeasurement', ha='left', fs=7.5)
    mini(ax, 4.6, 6.55, 1.45, 0.72, 'vonmises', M_COL, title='p(m|s) = VonMises(s, κ)')
    # generative arrows into m
    arrow(ax, (1.7, 7.85), (1.95, 7.25), color=S_COL, rad=0.0)     # s -> m
    arrow(ax, (6.25, 8.4), (2.35, 7.1), color=M_COL, rad=0.22)     # (c,d) -> m
    label(ax, 3.95, 6.12, 'κ(c,d) = (κ$_{min}$+κ$_{amp}$)·c$^{\\,p_c}$·e$^{-p_d d}$   (isotropic precision)',
          fs=7.5, color='0.3')

    # =================================================================
    # ROW 1 RIGHT — Observer inference
    # =================================================================
    region(ax, 7.8, 6.0, 7.3, 3.8, '2 · Observer inference', BRAIN_EC)

    # behaviour (environment) strip at the top
    subframe(ax, 8.15, 9.0, 6.6, 0.55, '', ENV_EC)
    label(ax, 8.3, 9.47, 'behaviour (observed)', fs=8, color=ENV_EC, ha='left', it=True)
    box(ax, 10.5, 9.27, 2.5, 0.4, 'choice  (Go / NoGo)', fc='white', ec=ENV_EC, fs=8)
    box(ax, 13.45, 9.27, 2.4, 0.4, 'confidence\n(velocity · licks)', fc='white', ec=ENV_EC, fs=7.4)

    # brain inference chain
    subframe(ax, 8.15, 6.2, 6.6, 2.65, 'brain', BRAIN_EC, dashed=True)
    node(ax, 8.85, 6.7, 'm', M_COL, r=0.28, fs=10)
    label(ax, 8.85, 6.32, 'likelihood', fs=6.9, color=M_COL)
    node(ax, 8.85, 7.85, 's|m', S_COL, r=0.32, fs=9)
    arrow(ax, (8.85, 6.98), (8.85, 7.53), color=S_COL)
    label(ax, 8.5, 7.25, '× prior\np(s)', fs=6.9, ha='right')
    mini(ax, 9.55, 6.42, 1.45, 0.72, 'posterior', S_COL, title='posterior p(s|m)')

    # value chain: posterior -> utility -> EU -> DV
    arrow(ax, (9.18, 7.85), (10.65, 7.85), color='0.4')
    label(ax, 9.9, 8.1, '× utility', fs=7.0)
    box(ax, 11.8, 7.85, 2.1, 1.0,
        'utility U(A, s)\n[R$_{hit}$, R$_{miss}$, R$_{CR}$, R$_{FA}$]\n= [1, 0, .1, −.2]',
        fc=UTIL_FC, ec='#c98a3a', fs=7.4)
    mini(ax, 13.0, 7.5, 0.7, 0.72, 'eu', C_COL)
    arrow(ax, (13.72, 7.85), (13.97, 7.85), color='0.4')
    node(ax, 14.25, 7.85, 'ΔEU', C_COL, r=0.28, fs=8)
    label(ax, 12.3, 6.95, 'DV(m) = EU(Go) − EU(NoGo)', fs=7.0, ha='center', color=C_COL)

    # outputs of inference: choice from log-odds, confidence from DV
    arrow(ax, (9.0, 8.15), (10.25, 9.0), color='0.4', rad=0.15)       # posterior -> choice
    label(ax, 9.1, 8.62, 'log-odds g(m)\n→ P(Go | g(m))', fs=6.8, ha='left')
    arrow(ax, (14.25, 8.13), (13.5, 9.0), color='0.4', rad=-0.08)     # DV -> confidence
    label(ax, 13.6, 8.5, 'y = β·DV\n+ α + ε', fs=6.8, ha='left')

    # =================================================================
    # ROW 2 — Fitting & inversion (full width)
    # =================================================================
    region(ax, 0.4, 0.4, 14.7, 5.1, '3 · Fitting & trial-by-trial inversion', ENV_EC)

    yA = 4.4   # pipeline row centre
    box(ax, 2.0, yA, 2.6, 1.15,
        'observed behaviour\n\nvelocity · licks\nchoices',
        fc='white', ec=ENV_EC, fs=8)
    box(ax, 5.7, yA, 3.15, 1.3,
        'Stage 1 — sensory + emission\nfrom kinematics alone\n'
        '{κamp, p$_c$, p$_d$, β, α, σ}\nBADS · hierarchical · 5-fold CV',
        fc=STAGE1_FC, ec='#3b6fb0', fs=7.9)
    box(ax, 9.6, yA, 3.15, 1.3,
        'Stage 2 — choice psychometric\non log-odds g(m) = log $\\frac{P(Go|m)}{P(NoGo|m)}$\n'
        '{α$_r$, β$_r$, γ$_r$, δ$_r$}\nvelocity-conditioned · 5-fold CV',
        fc=STAGE2_FC, ec='#7a52a8', fs=7.9)
    box(ax, 13.3, yA, 2.7, 1.3,
        'Marginalised inversion\nintegrate over p(m | s, y)\n'
        '→ trial-by-trial targets',
        fc=INV_FC, ec='#2e9e5b', fs=8.0)
    arrow(ax, (3.3, yA), (4.12, yA), color='0.4')
    arrow(ax, (7.28, yA), (8.02, yA), color='0.4')
    arrow(ax, (11.18, yA), (11.95, yA), color='0.4')

    # -- the three IO-derived targets --
    subframe(ax, 4.5, 1.85, 10.6, 1.6, '', '#2e9e5b')
    label(ax, 4.72, 3.32, 'IO-derived targets', fs=8.5, color='#2e9e5b', ha='left', it=True)
    mini(ax, 5.55, 2.0, 1.7, 0.82, 'posterior', S_COL, title='perceptual posterior  Q(θ)')
    mini(ax, 8.95, 2.0, 1.7, 0.82, 'likelihood', '#3b78c4', title='perceptual likelihood  L(θ)')
    mini(ax, 12.5, 2.0, 1.5, 0.82, 'decision', None, title='decision posterior')
    arrow(ax, (12.7, 3.75), (10.6, 3.48), color='0.4', rad=0.12)     # inversion -> targets

    # downstream uses (bottom band)
    box(ax, 4.35, 1.0, 5.4, 0.95,
        'neural decoder target\nV1 activity → IO target   (spatial / temporal read-outs)',
        fc='white', ec='0.45', fs=8.2)
    box(ax, 11.3, 1.0, 6.4, 0.95,
        'uncertainty read-outs\n'
        'perceptual  U$_{perc}$ = SD[Q(θ)]      ·      decision  U$_{dec}$ = H[P(Go), P(NoGo)]',
        fc='white', ec='0.45', fs=8.0)
    arrow(ax, (6.4, 1.85), (4.7, 1.5), color='0.5', rad=-0.12)
    arrow(ax, (12.5, 1.85), (11.2, 1.5), color='0.5', rad=0.06)

    ps.save_fig(fig, out_dir, 'io_model_schematic', layout=None)


def main(out_root):
    build(Path(out_root))
    print(f'Done. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default='figures/schematic')
    a = ap.parse_args()
    main(a.out_root)
