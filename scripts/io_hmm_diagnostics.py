# -*- coding: utf-8 -*-
"""Diagnostic plots for a fitted IO-HMM on (real or synthetic) behaviour.

One figure per animal summarising what the fit found:

  (a) decoded-state ribbon  -- the Viterbi state path per session (one row per
      session), so you can see state dwell/switching structure across the data;
  (b) psychometric-by-state -- empirical P(go) vs stimulus angle split by the
      decoded state, with each state's fitted model curve overlaid (do the
      states actually behave differently in choices?);
  (c) velocity-by-state     -- the velocity distribution per decoded state with
      the fitted Gaussian markers (only when the fit used velocity);
  (d) transitions + occupancy -- the fitted A heatmap and state occupancy.

Kept out of the numpy-only ``io_hmm`` core (matplotlib import lives here).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_IO_HMM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "ideal_observer", "io_hmm")
if _IO_HMM not in sys.path:
    sys.path.insert(0, _IO_HMM)

import emissions as emissions_mod  # noqa: E402


def _state_runs(path: np.ndarray):
    """Yield (start, length, state) for contiguous runs in a 1-D path."""
    if len(path) == 0:
        return
    change = np.flatnonzero(np.diff(path)) + 1
    bounds = np.concatenate(([0], change, [len(path)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(a), int(b - a), int(path[a])


def _psych_curve(grids, stage1, state, free_vals, c=1.0, d=0.0, n=80):
    s = np.linspace(0.0, 90.0, n)
    conds = np.column_stack([s, np.full(n, c), np.full(n, d)])
    return s, emissions_mod.p_go_per_unique_condition(
        grids, stage1, state, dict(free_vals), conds)


def plot_animal_fit(animal, params, trials_list, paths, state_list, stage1,
                    grids, outpath, use_velocity=True):
    """Render the 4-panel diagnostic figure and save to ``outpath``."""
    K = len(state_list)
    names = [s.name for s in state_list]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(k) for k in range(K)]

    has_vel = use_velocity and all(t.has_velocity for t in trials_list) \
        and bool(params.vel_per_state)
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.9],
                          hspace=0.45, wspace=0.28)
    fig.suptitle(f"IO-HMM fit diagnostics — animal {animal} "
                 f"({len(trials_list)} sessions, "
                 f"{sum(t.n_trials for t in trials_list)} trials)",
                 fontsize=13, fontweight="bold")

    # (a) decoded-state ribbon, one row per session
    ax = fig.add_subplot(gs[0, :])
    for row, path in enumerate(paths):
        for start, length, k in _state_runs(np.asarray(path)):
            ax.broken_barh([(start, length)], (row + 0.1, 0.8),
                           facecolors=colors[k])
    ax.set_ylim(0, len(paths)); ax.set_xlim(0, max(len(p) for p in paths))
    ax.set_yticks(np.arange(len(paths)) + 0.5)
    ax.set_yticklabels([f"sess {i}" for i in range(len(paths))], fontsize=8)
    ax.set_xlabel("trial (within session)")
    ax.set_title("(a) decoded latent-state path per session")
    ax.legend(handles=[Patch(color=colors[k], label=names[k]) for k in range(K)],
              ncol=K, fontsize=8, loc="upper right")

    # pooled per-trial arrays + decoded state
    s_all = np.concatenate([t.s_deg for t in trials_list])
    ch_all = np.concatenate([t.choice for t in trials_list])
    z_all = np.concatenate([np.asarray(p) for p in paths])

    # (b) psychometric by decoded state (empirical points + model curve)
    ax = fig.add_subplot(gs[1, 0])
    edges = np.linspace(0, 90, 8)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for k in range(K):
        m = z_all == k
        if m.sum() < 10:
            continue
        bin_idx = np.digitize(s_all[m], edges) - 1
        pe = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            sel = bin_idx == b
            if sel.sum() >= 5:
                pe[b] = ch_all[m][sel].mean()
        ax.plot(centers, pe, "o", color=colors[k], ms=5)
        s_grid, p_model = _psych_curve(
            grids, stage1, state_list[k], params.psych_per_state.get(names[k], {}))
        ax.plot(s_grid, p_model, "-", color=colors[k], lw=1.8, label=names[k])
    ax.set_xlabel("stimulus angle (deg)"); ax.set_ylabel("P(go)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) psychometric by decoded state\n(points=empirical, line=model)")
    ax.legend(fontsize=8)

    # (c) velocity by decoded state (+ fitted markers), if available
    ax = fig.add_subplot(gs[1, 1])
    if has_vel:
        v_all = np.concatenate([t.velocity for t in trials_list])
        bins = np.linspace(np.nanmin(v_all), np.nanmax(v_all), 30)
        for k in range(K):
            m = z_all == k
            if m.sum() < 10:
                continue
            ax.hist(v_all[m], bins=bins, density=True, alpha=0.5,
                    color=colors[k], label=names[k])
            ax.axvline(params.vel_per_state[names[k]]["mu"], color=colors[k],
                       lw=2, ls="--")
        ax.set_xlabel("velocity (as fit)"); ax.set_ylabel("density")
        ax.set_title("(c) velocity by decoded state\n(dashed = fitted markers)")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "velocity channel not used", ha="center",
                va="center", fontsize=11, color="0.4")

    # (d) transition matrix
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(params.A, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{params.A[i, j]:.2f}", ha="center", va="center",
                    color="w" if params.A[i, j] < 0.6 else "k", fontsize=8)
    ax.set_title("(d) transition matrix A  (row=from, col=to)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # occupancy bar
    ax = fig.add_subplot(gs[2, 1])
    occ = np.array([(z_all == k).mean() for k in range(K)])
    ax.bar(range(K), occ, color=colors)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=45,
                                                ha="right", fontsize=8)
    ax.set_ylabel("fraction of trials"); ax.set_ylim(0, 1)
    ax.set_title("state occupancy")

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


__all__ = ["plot_animal_fit"]
