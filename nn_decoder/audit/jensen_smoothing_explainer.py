# -*- coding: utf-8 -*-
"""Pedagogical figures for the Monte-Carlo smoothing principle.

The shuffle-control asymmetry (``spat_shf > temp_shf``) is rigorously
diagnosed in ``shuffle_asymmetry_diagnostic.py``. This script is the
teaching companion: a small set of slide-ready figures that explain
*why* averaging T post-softmax distributions is fundamentally different
from a single softmax of the averaged logits, and why that produces
lower trial-to-trial variance.

Six figures, each self-contained for a PPT slide:

  fig_01_setup.png            : the two pipelines side by side as a cartoon.
  fig_02_single_trial.png     : on one trial's T per-bin logits, show that
                                softmax(mean(z_t)) differs from
                                mean_t(softmax(z_t)).
  fig_03_2class_geometry.png  : the same comparison on a 2-class problem,
                                where probabilities live on a line and
                                Jensen's inequality is *visually* obvious.
  fig_04_noise_drives.png     : sweep within-trial noise; SBC broadens
                                with noise, PPC does not. Trial-to-trial
                                variance flips with noise level.
  fig_05_T_law.png            : sweep T (time bins); SBC's trial-to-trial
                                variance shrinks roughly as ``1/T``,
                                PPC is invariant to T.
  fig_06_real_overlay.png     : 8 real trials from the saved .mat.
                                Spat_shf predictions cluster around their
                                trial means; temp_shf predictions cluster
                                around the global marginal.

Figures land in ``nn_decoder/figures/shuffle_asymmetry/`` next to the
rigorous-proof figures. Standalone usage:

    python audit/jensen_smoothing_explainer.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import scipy.io as sio

HERE = Path(__file__).resolve().parent
NN_DECODER = HERE.parent
if str(NN_DECODER) not in sys.path:
    sys.path.insert(0, str(NN_DECODER))
from paths import figures_dir
from figsave import save_fig

# Consistent colour palette across all figures.
PPC_C = '#1f77b4'      # spatial (PPC)
SBC_C = '#d62728'      # temporal (SBC / sampling)
MARG_C = '#444444'     # the training marginal

FIGS_SUBDIR = 'shuffle_asymmetry'


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _random_mlp(n_in, hidden, n_out, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, hidden))
    b1 = rng.normal(0.0, 0.1, size=(hidden,))
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, n_out))
    b2 = rng.normal(0.0, 0.1, size=(n_out,))

    def f(z):
        h = np.tanh(z @ W1 + b1)
        return h @ W2 + b2
    return f


def _synthetic_trials(n_trials=200, T=10, n_neurons=50, n_cats=91,
                       sigma_signal=0.5, sigma_bin=1.0, seed=0,
                       zero_mean_bin_noise=False):
    """Return ``(x, mlp)`` for synthetic time-binned activity.

    ``x`` has shape ``(n_trials, T, n_neurons)``; ``mlp`` is a fixed
    random network mapping ``n_neurons -> n_cats``.

    ``zero_mean_bin_noise=True`` constructs ε_t so that
    ``sum_t ε_t == 0`` per trial. This makes ``mean_t(x_t) == s`` exactly
    regardless of σ_bin or T, which isolates the SBC smoothing effect
    from changes in the PPC input — the cleanest pedagogical setup.
    """
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, sigma_signal, size=(n_trials, n_neurons))
    eps = rng.normal(0.0, sigma_bin, size=(n_trials, T, n_neurons))
    if zero_mean_bin_noise:
        eps = eps - eps.mean(axis=1, keepdims=True)
    x = s[:, None, :] + eps
    mlp = _random_mlp(n_neurons, 32, n_cats, seed=seed + 1)
    return x, mlp


def _ppc_sbc_predictions(x, mlp):
    """Compute (p_ppc, p_sbc, logits_per_bin) for the two pipelines.

    PPC averages INPUTS first, then runs ONE forward pass:
        p_ppc = softmax(mlp(mean_t(x_t)))
    SBC runs T forward passes, then averages OUTPUTS:
        p_sbc = mean_t(softmax(mlp(x_t)))
    """
    x_mean = x.mean(axis=1)                          # (n_trials, n_neurons)
    logits_ppc = mlp(x_mean)                         # (n_trials, n_cats)
    p_ppc = softmax(logits_ppc)
    logits_per_bin = mlp(x)                          # (n_trials, T, n_cats)
    p_sbc = softmax(logits_per_bin, axis=-1).mean(axis=1)
    return p_ppc, p_sbc, logits_per_bin


# ----------------------------------------------------------------------
# Figure 1 — setup cartoon
# ----------------------------------------------------------------------

def fig_setup(out_dir):
    """Schematic showing the two forward passes side by side."""
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.set_axis_off()

    def box(x, y, w, h, text, fc='white', ec='0.3', fontsize=11, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                     facecolor=fc, edgecolor=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='-|>', mutation_scale=14,
                                       color='0.3', lw=1.4))

    # Inputs (shared).
    ax.text(0.4, 7.4, 'Input: $x_t \\in \\mathbb{R}^{n\\,neurons}$ at $T$ time bins',
            fontsize=12, fontweight='bold')

    # Top row — PPC pipeline.
    ax.text(0.4, 6.0, 'PPC  (spatial)', fontsize=14, fontweight='bold',
            color=PPC_C)
    box(2.6, 5.5, 2.0, 1.2,
        'mean$_t$($x_t$)\n(average inputs)', fc='#e8f1f9', fontsize=11)
    box(5.4, 5.5, 1.8, 1.2, 'MLP', fc='white', fontsize=12, weight='bold')
    box(7.9, 5.5, 2.0, 1.2, 'softmax', fc='#fef0d8', fontsize=12, weight='bold')
    box(10.5, 5.5, 2.6, 1.2,
        '$p_{PPC}$\n$\\in \\mathbb{R}^{n\\,cats}$', fc='#e8f1f9', fontsize=11)
    arrow(4.6, 6.1, 5.4, 6.1)
    arrow(7.2, 6.1, 7.9, 6.1)
    arrow(9.9, 6.1, 10.5, 6.1)
    ax.text(7.0, 4.95, r'$p_{PPC} \;=\; \mathrm{softmax}(\mathrm{MLP}(\overline{x}))$',
            ha='center', fontsize=12, color=PPC_C, fontweight='bold')

    # Bottom row — Sampling pipeline.
    ax.text(0.4, 3.5, 'Sampling  (temporal)', fontsize=14, fontweight='bold',
            color=SBC_C)
    box(2.6, 1.9, 2.0, 1.2,
        '$x_1, x_2, \\ldots, x_T$\n(per-bin inputs)', fc='#fbe9eb', fontsize=10)
    box(5.4, 1.9, 1.8, 1.2, 'MLP\n(per bin)', fc='white', fontsize=11, weight='bold')
    box(7.9, 1.9, 2.0, 1.2, 'softmax\n(per bin)', fc='#fef0d8', fontsize=11, weight='bold')
    box(10.5, 1.9, 2.6, 1.2,
        'mean$_t(\\cdot)$ → $p_{SBC}$', fc='#fbe9eb', fontsize=11)
    arrow(4.6, 2.5, 5.4, 2.5)
    arrow(7.2, 2.5, 7.9, 2.5)
    arrow(9.9, 2.5, 10.5, 2.5)
    ax.text(7.0, 1.30, r'$p_{SBC} \;=\; \overline{\mathrm{softmax}(\mathrm{MLP}(x_t))}$',
            ha='center', fontsize=12, color=SBC_C, fontweight='bold')

    # Key takeaway.
    ax.text(7.0, 0.20,
            'Difference: WHERE the time-average happens. '
            'PPC averages BEFORE softmax; SBC averages AFTER.',
            ha='center', fontsize=11, color='0.25', fontweight='bold',
            bbox=dict(facecolor='#fff7d6', edgecolor='0.5', pad=4))

    fig.suptitle('Two forward passes, same MLP backbone',
                  fontsize=14, fontweight='bold')
    out = os.path.join(out_dir, 'fig_01_setup.png')
    save_fig(fig, os.path.dirname(out), os.path.splitext(os.path.basename(out))[0])
    return out


# ----------------------------------------------------------------------
# Figure 2 — single-trial side-by-side
# ----------------------------------------------------------------------

def fig_single_trial(out_dir, seed=2):
    """Pick one trial and visualise both predictions on the same axis.

    Top:    the T per-bin logits as transparent lines + their mean.
    Bottom: PPC prediction (softmax of mean logit, single sharp curve)
            and SBC prediction (mean of T per-bin softmaxes, broader).
    """
    n_cats = 91
    T = 10
    rng = np.random.default_rng(seed)
    # Hand-pick a logit vector with two clear modes, then jitter for T bins.
    grid = np.arange(n_cats)
    base = (3.0 * np.exp(-0.5 * ((grid - 30) / 6.0) ** 2)
            + 1.6 * np.exp(-0.5 * ((grid - 65) / 8.0) ** 2))
    z_t = base[None, :] + rng.normal(0.0, 1.3, size=(T, n_cats))
    z_mean = z_t.mean(axis=0)

    p_ppc = softmax(z_mean)
    p_per_bin = softmax(z_t, axis=-1)
    p_sbc = p_per_bin.mean(axis=0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                          gridspec_kw=dict(height_ratios=[1.0, 1.4]))
    # Top: logits.
    for t in range(T):
        ax_top.plot(grid, z_t[t], color='0.55', lw=0.8, alpha=0.5)
    ax_top.plot(grid, z_mean, color='k', lw=2.3,
                 label=r'$\overline{z} = \frac{1}{T}\sum_t z_t$ (PPC input)')
    ax_top.set_ylabel('logit value')
    ax_top.set_title('A. Per-bin logits for ONE trial (T = 10 bins)',
                      fontsize=12, fontweight='bold')
    ax_top.legend(fontsize=10, frameon=False, loc='upper right')
    ax_top.grid(alpha=0.25)

    # Bottom: the two predictions overlaid.
    for t in range(T):
        ax_bot.plot(grid, p_per_bin[t], color=SBC_C, lw=0.8, alpha=0.30)
    ax_bot.plot(grid, p_ppc, color=PPC_C, lw=2.8, zorder=5,
                 label=r'$p_{PPC} = \mathrm{softmax}(\overline{z})$ — sharp')
    ax_bot.plot(grid, p_sbc, color=SBC_C, lw=2.8, zorder=5,
                 label=r'$p_{SBC} = \overline{\mathrm{softmax}(z_t)}$ — broader')
    ax_bot.plot([], [], color=SBC_C, alpha=0.4, lw=0.8,
                 label='per-bin softmaxes (faded)')
    ax_bot.set_xlabel('Output category')
    ax_bot.set_ylabel('p(category)')
    ax_bot.set_title('B. Same trial — two predictions from the same backbone',
                      fontsize=12, fontweight='bold')
    ax_bot.legend(fontsize=10, frameon=False, loc='upper right')
    ax_bot.grid(alpha=0.25)

    # Annotation: entropy values.
    H_p = entropy(p_ppc); H_s = entropy(p_sbc)
    ax_bot.text(0.02, 0.96,
                 f'H($p_{{PPC}}$) = {H_p:.2f} nats\n'
                 f'H($p_{{SBC}}$) = {H_s:.2f} nats   (broader → larger H)',
                 transform=ax_bot.transAxes, va='top', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='0.5', pad=3))

    fig.suptitle('Softmax-of-mean vs mean-of-softmaxes — same mean, different shape',
                  fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(out_dir, 'fig_02_single_trial.png')
    save_fig(fig, os.path.dirname(out), os.path.splitext(os.path.basename(out))[0])
    return out


# ----------------------------------------------------------------------
# Figure 3 — 2-class geometry on the simplex
# ----------------------------------------------------------------------

def fig_2class_geometry(out_dir, T=8, sigma=2.5, mu=2.5, seed=11):
    """Toy 2-class problem: probabilities live on a 1-D simplex (a line).

    For each per-bin logit pair (z_0, z_1), the softmax is a single
    probability ``p_1 = sigmoid(z_1 - z_0)``. The PPC operation
    sigmoid(mean(z)) lands at one point; the SBC operation
    mean(sigmoid(z)) lands at another. The mean is placed at μ = 2.2
    where the sigmoid curves strongly, so the Jensen offset is large
    enough to see at a glance.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(mu, sigma, size=T)              # logit differences
    # Force zero-mean perturbation so sample mean = μ exactly.
    z = mu + (z - z.mean())
    p_bin = 1.0 / (1.0 + np.exp(-z))
    p_mean = float(p_bin.mean())                   # SBC
    p_of_mean = 1.0 / (1.0 + np.exp(-mu))          # PPC

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                     gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    # Left: sigmoid curve with the T samples + the two averages annotated.
    zz = np.linspace(-4, 8, 400)
    ax1.plot(zz, 1.0 / (1.0 + np.exp(-zz)), color='0.35', lw=2.4)
    # Per-bin samples on the curve.
    ax1.scatter(z, p_bin, s=85, color=SBC_C, zorder=5,
                 edgecolor='white', linewidth=1.2, alpha=0.85,
                 label=f'T={T} per-bin logits  →  sigmoids')
    # Vertical line at the mean.
    ax1.axvline(mu, color='0.5', ls=':', lw=1.2, zorder=1)
    # PPC: sit on the curve at z=μ.
    ax1.scatter([mu], [p_of_mean], s=320, color=PPC_C, marker='*',
                 zorder=7, edgecolor='white', linewidth=1.5,
                 label=r'PPC = sigmoid($\overline{z}$)  — on the curve')
    # SBC: average y-value of samples, plotted at z=μ.
    ax1.scatter([mu], [p_mean], s=220, color=SBC_C, marker='s',
                 zorder=7, edgecolor='white', linewidth=1.5,
                 label=r'SBC = $\overline{\mathrm{sigmoid}(z)}$  — average of $y$-values')
    # Visual annotation of the Jensen offset (vertical gap).
    ax1.annotate('', xy=(mu, p_of_mean), xytext=(mu, p_mean),
                  arrowprops=dict(arrowstyle='<|-|>', color='#444',
                                   lw=2.0, mutation_scale=14))
    ax1.text(mu + 0.30, 0.5 * (p_of_mean + p_mean),
              f'Jensen\noffset\n{p_of_mean - p_mean:+.3f}',
              fontsize=10.5, color='#222', fontweight='bold', va='center',
              bbox=dict(facecolor='#fff7d6', edgecolor='0.5', pad=3))
    # Chord between leftmost and rightmost samples — the SBC value lies
    # along that chord by convexity arguments.
    zlo, zhi = z.min(), z.max()
    plo, phi = 1.0 / (1.0 + np.exp(-zlo)), 1.0 / (1.0 + np.exp(-zhi))
    ax1.plot([zlo, zhi], [plo, phi], '--', color='0.55', lw=1.2,
              label='chord between extreme samples\n(SBC sits on / near this chord)')

    ax1.set_xlabel(r'$z = z_1 - z_0$  (logit difference)')
    ax1.set_ylabel(r'$p_1 = \mathrm{sigmoid}(z)$')
    ax1.set_title('A. Same mean, two different outputs',
                   fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, frameon=False, loc='lower right')

    # Right: same on the 1-D probability simplex (a line p in [0,1]).
    # All three quantities sit on a single horizontal line; the
    # left-right gap between the two markers IS the Jensen offset.
    ax2.hlines(0.5, 0, 1, color='0.55', lw=3)
    # Per-bin softmaxes as small dots on the line.
    ax2.scatter(p_bin, [0.5] * T, s=70, color=SBC_C, zorder=4,
                 edgecolor='white', linewidth=1.0, alpha=0.7,
                 label='per-bin softmaxes')
    # The two estimates as bigger markers, both on the line.
    ax2.scatter([p_of_mean], [0.5], s=380, color=PPC_C, marker='*',
                 zorder=6, edgecolor='white', linewidth=1.5,
                 label=r'PPC = sigmoid($\overline{z}$)')
    ax2.scatter([p_mean], [0.5], s=240, color=SBC_C, marker='s',
                 zorder=6, edgecolor='white', linewidth=1.5,
                 label=r'SBC = $\overline{\mathrm{sigmoid}(z)}$')
    # Horizontal double-arrow between the two estimates.
    ax2.annotate('', xy=(p_of_mean, 0.62), xytext=(p_mean, 0.62),
                  arrowprops=dict(arrowstyle='<|-|>', color='#222',
                                   lw=2.0, mutation_scale=14))
    ax2.text(0.5 * (p_of_mean + p_mean), 0.69,
              f'Jensen offset = {p_of_mean - p_mean:+.3f}',
              fontsize=10.5, color='#222', fontweight='bold', ha='center',
              bbox=dict(facecolor='#fff7d6', edgecolor='0.5', pad=3))
    ax2.axvline(0.5, color='0.7', ls='--', lw=1.2)
    ax2.text(0.5, 0.30, 'centre p = 0.5\n(broadest)', ha='center',
              fontsize=9, color='0.45')
    ax2.annotate('SBC is pulled\ntoward the centre',
                  xy=(p_mean, 0.5), xytext=(0.12, 0.30),
                  fontsize=10, color='#222', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='0.40', lw=1.4),
                  bbox=dict(facecolor='#fff7d6', edgecolor='0.5', pad=3))
    ax2.set_xlim(0, 1); ax2.set_ylim(0.15, 0.95)
    ax2.set_xlabel('p(class = 1)')
    ax2.set_yticks([])
    ax2.set_title('B. Where the two estimates sit on the simplex',
                   fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, frameon=False, loc='upper right')

    fig.suptitle('Jensen on a 2-class problem: SBC sits closer to the centre',
                  fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(out_dir, 'fig_03_2class_geometry.png')
    save_fig(fig, os.path.dirname(out), os.path.splitext(os.path.basename(out))[0])
    return out


# ----------------------------------------------------------------------
# Figure 4 — within-trial noise drives the smoothing
# ----------------------------------------------------------------------

def fig_noise_drives(out_dir, sigmas=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                      n_trials=400, T=10, n_neurons=60, seed=0):
    """Sweep within-trial noise; show what changes for each pipeline.

    Uses zero-mean per-trial noise so the PPC input is exactly the trial
    signal regardless of σ. This isolates the SBC smoothing effect:
    PPC entropy and trial-to-trial variance stay PERFECTLY flat (it
    never sees σ), while SBC broadens and its variance shrinks as σ
    grows.
    """
    H_ppc, H_sbc = [], []
    V_ppc, V_sbc = [], []
    for s in sigmas:
        x, mlp = _synthetic_trials(
            n_trials=n_trials, T=T, n_neurons=n_neurons,
            sigma_signal=0.5, sigma_bin=s, seed=seed,
            zero_mean_bin_noise=True)
        p_ppc, p_sbc, _ = _ppc_sbc_predictions(x, mlp)
        H_ppc.append(entropy(p_ppc).mean())
        H_sbc.append(entropy(p_sbc).mean())
        V_ppc.append(p_ppc.var(axis=0).sum())
        V_sbc.append(p_sbc.var(axis=0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # A: mean prediction entropy.
    axes[0].plot(sigmas, H_ppc, '-o', color=PPC_C, lw=2.4, ms=8,
                  label='PPC (softmax of mean)')
    axes[0].plot(sigmas, H_sbc, '-s', color=SBC_C, lw=2.4, ms=8,
                  label='SBC (mean of softmaxes)')
    axes[0].set_xlabel(r'Within-trial noise  $\sigma_{bin}$')
    axes[0].set_ylabel('Mean H(prediction)  (nats)')
    axes[0].set_title('A. SBC broadens with noise;  PPC does not',
                       fontsize=11.5, fontweight='bold')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=10, frameon=False)

    # B: trial-to-trial variance.
    axes[1].plot(sigmas, V_ppc, '-o', color=PPC_C, lw=2.4, ms=8,
                  label='PPC')
    axes[1].plot(sigmas, V_sbc, '-s', color=SBC_C, lw=2.4, ms=8,
                  label='SBC')
    axes[1].set_xlabel(r'Within-trial noise  $\sigma_{bin}$')
    axes[1].set_ylabel('Trial-to-trial variance\n'
                        r'($\sum_c \mathrm{var}_n(p_n[c])$)')
    axes[1].set_title('B. SBC variance shrinks with noise;  PPC is flat',
                       fontsize=11.5, fontweight='bold')
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=10, frameon=False)
    # Annotate the variance gap at the largest σ.
    sg = list(sigmas)[-1]
    ratio = V_ppc[-1] / V_sbc[-1] if V_sbc[-1] > 0 else np.nan
    axes[1].annotate(f'PPC / SBC = {ratio:.2f}×',
                      xy=(sg, V_ppc[-1]),
                      xytext=(sg * 0.55, max(V_ppc) * 0.78),
                      fontsize=11, color='0.20', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='0.40', lw=1.2))

    fig.suptitle(
        'Within-trial noise is the smoothing dial — SBC feels it, PPC is blind to it\n'
        '(noise constructed with $\\sum_t \\epsilon_t = 0$ so PPC\'s input is exactly the trial signal)',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(out_dir, 'fig_04_noise_drives.png')
    save_fig(fig, os.path.dirname(out), os.path.splitext(os.path.basename(out))[0])
    return out


# ----------------------------------------------------------------------
# Figure 5 — the 1/T law
# ----------------------------------------------------------------------

def fig_T_law(out_dir, T_grid=(1, 2, 4, 8, 16, 32, 64, 128),
               n_trials=600, n_neurons=60, sigma_bin=2.0, seed=0):
    """Trial-to-trial variance of SBC shrinks as T grows; PPC is flat.

    Uses zero-mean within-trial noise so PPC's input is exactly the
    trial signal for any T — its variance is the asymptotic ceiling
    ``Var(softmax(MLP(s)))``. SBC converges to a smaller asymptotic
    floor ``Var(E_ε[softmax(MLP(s + ε))])`` (a smoothed function of s),
    approaching it at a ``1/T`` Monte-Carlo rate.
    """
    V_ppc, V_sbc = [], []
    for T in T_grid:
        x, mlp = _synthetic_trials(
            n_trials=n_trials, T=T, n_neurons=n_neurons,
            sigma_signal=0.5, sigma_bin=sigma_bin, seed=seed,
            zero_mean_bin_noise=True)
        p_ppc, p_sbc, _ = _ppc_sbc_predictions(x, mlp)
        V_ppc.append(p_ppc.var(axis=0).sum())
        V_sbc.append(p_sbc.var(axis=0).sum())

    T_arr = np.array(T_grid, dtype=float)
    V_ppc = np.array(V_ppc); V_sbc = np.array(V_sbc)

    # Estimate the SBC floor with a very large T (dense MC).
    x_inf, mlp_inf = _synthetic_trials(
        n_trials=n_trials, T=512, n_neurons=n_neurons,
        sigma_signal=0.5, sigma_bin=sigma_bin, seed=seed + 99,
        zero_mean_bin_noise=True)
    _, p_sbc_inf, _ = _ppc_sbc_predictions(x_inf, mlp_inf)
    V_sbc_floor = float(p_sbc_inf.var(axis=0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Left: linear scale, with PPC ceiling and SBC floor annotated.
    ax = axes[0]
    ax.axhline(V_ppc[0], color=PPC_C, ls='--', lw=1.4, alpha=0.5,
                label='PPC = Var(softmax(MLP(s)))  (ceiling)')
    ax.axhline(V_sbc_floor, color=SBC_C, ls='--', lw=1.4, alpha=0.5,
                label=r'SBC floor = Var($E_\epsilon$[softmax(MLP(s+$\epsilon$))])')
    ax.plot(T_arr, V_ppc, '-o', color=PPC_C, lw=2.4, ms=8,
             label='PPC (flat in T)')
    ax.plot(T_arr, V_sbc, '-s', color=SBC_C, lw=2.4, ms=8,
             label='SBC (averages T softmaxes)')
    ax.set_xlabel('T  (number of time bins averaged)')
    ax.set_ylabel('Trial-to-trial variance of prediction')
    ax.set_title('A. SBC converges to a LOWER ceiling as T grows',
                  fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)

    # Right: log-log of (V_sbc - floor), shows 1/T convergence cleanly.
    ax = axes[1]
    excess = V_sbc - V_sbc_floor
    ax.plot(T_arr, excess, '-s', color=SBC_C, lw=2.4, ms=8,
             label=r'SBC variance excess  $V_{SBC}(T) - V_{SBC}(\infty)$')
    ref = excess[0] / T_arr * T_arr[0]
    ax.plot(T_arr, ref, '--', color='0.4', lw=1.4,
             label=r'$\propto 1/T$ reference')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('T  (number of time bins averaged)')
    ax.set_ylabel('Excess variance over SBC floor')
    ax.set_title('B. Monte-Carlo rate: excess variance falls as 1/T',
                  fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle('SBC as a Monte-Carlo estimator of a smoothed function',
                  fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(out_dir, 'fig_05_T_law.png')
    save_fig(fig, os.path.dirname(out), os.path.splitext(os.path.basename(out))[0])
    return out


# ----------------------------------------------------------------------
# Figure 6 — real-data per-trial overlay
# ----------------------------------------------------------------------

def _discover_mat():
    for name in ('population_results_fixed_hyperparams_JS_stratified_balanced.mat',
                 'population_results_fixed_hyperparams_stratified_balanced.mat'):
        p = NN_DECODER / name
        if p.exists():
            return p
    return None


def fig_real_overlay(out_dir, mouse_id=1, n_trials=8, seed=0):
    """Plot N real per-trial predictions from spat_shf vs temp_shf overlaid,
    on the same axis as the training-marginal target."""
    mat_path = _discover_mat()
    if mat_path is None:
        print('[fig_real_overlay] no population_results .mat found; skipping')
        return None
    d = sio.loadmat(str(mat_path), simplify_cells=True)
    key = f'mouse_{mouse_id}'
    if key not in d['results']:
        # Fall back to any mouse if the requested one is missing.
        key = next(k for k in d['results'] if k.startswith('mouse_'))
        mouse_id = int(key.split('_')[1])
    m = d['results'][key]
    p_s = np.asarray(m['Dist']['spat_shf']['decoded'], dtype=float)
    p_t = np.asarray(m['Dist']['temp_shf']['decoded'], dtype=float)
    target = np.asarray(m['Dist']['spat']['target'], dtype=float)
    marg = target.mean(axis=0)
    n_cats = p_s.shape[1]
    grid = np.arange(n_cats)

    rng = np.random.default_rng(seed)
    show = rng.choice(p_s.shape[0], size=n_trials, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for i, t in enumerate(show):
        lbl = f'{n_trials} sample trials' if i == 0 else None
        axes[0].plot(grid, p_s[t], color=PPC_C, alpha=0.55, lw=1.2,
                      label=lbl)
        axes[1].plot(grid, p_t[t], color=SBC_C, alpha=0.55, lw=1.2,
                      label=lbl)
    axes[0].plot(grid, p_s.mean(axis=0), color=PPC_C, lw=2.8,
                  label='mean over all trials')
    axes[1].plot(grid, p_t.mean(axis=0), color=SBC_C, lw=2.8,
                  label='mean over all trials')
    for ax, title in zip(axes,
                          ('spat_shf  (PPC) — per-trial predictions',
                           'temp_shf  (SBC) — per-trial predictions')):
        ax.plot(grid, marg, '--', color=MARG_C, lw=2.0,
                 label='target marginal (chance-optimal)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Output category')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, frameon=False, loc='upper right')
    axes[0].set_ylabel('p(category)')

    H_s = entropy(p_s).mean(); H_t = entropy(p_t).mean()
    V_s = p_s.var(axis=0).sum(); V_t = p_t.var(axis=0).sum()
    fig.suptitle(
        f'Real shuffled-target predictions  (mouse {mouse_id}, '
        f'n_trials = {p_s.shape[0]})\n'
        f'mean H: spat={H_s:.3f}  temp={H_t:.3f}    '
        f'trial-variance: spat={V_s:.4f}  temp={V_t:.4f}   '
        f'(PPC / SBC = {V_s / V_t:.2f}×)',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(out_dir, 'fig_06_real_overlay.png')
    save_fig(fig, os.path.dirname(out), os.path.splitext(os.path.basename(out))[0])
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: '
                             'nn_decoder/figures/shuffle_asymmetry/).')
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir else str(figures_dir(FIGS_SUBDIR))
    os.makedirs(out_dir, exist_ok=True)

    written = []
    print('[1/6] setup cartoon ...');           written.append(fig_setup(out_dir))
    print('[2/6] single-trial side-by-side ...'); written.append(fig_single_trial(out_dir))
    print('[3/6] 2-class simplex geometry ...'); written.append(fig_2class_geometry(out_dir))
    print('[4/6] noise sweep ...');             written.append(fig_noise_drives(out_dir))
    print('[5/6] T-law ...');                   written.append(fig_T_law(out_dir))
    print('[6/6] real-data overlay ...');       written.append(fig_real_overlay(out_dir))

    print(f'\nWritten {sum(1 for x in written if x)} figures to {out_dir}:')
    for p in written:
        if p:
            print(f'  {p}')
    return written


if __name__ == '__main__':
    main()
