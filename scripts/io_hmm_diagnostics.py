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

    # (c) confidence coupling: velocity vs decision variable per decoded state,
    #     with each state's fitted v = beta_vel*DV + alpha_vel line.
    ax = fig.add_subplot(gs[1, 1])
    if has_vel:
        v_all = np.concatenate([t.velocity for t in trials_list])
        c_all = np.concatenate([t.c for t in trials_list])
        d_all = np.concatenate([t.d for t in trials_list])
        cond = np.column_stack([s_all, c_all, d_all]).astype(float)
        G_unique, G_idx = np.unique(cond, axis=0, return_inverse=True)
        for k in range(K):
            gm, dvm, pm = emissions_mod.precompute_state_terms(
                grids, stage1, state_list[k], G_unique)
            dvbar = (dvm * pm).sum(1)[G_idx]          # E[DV|cond] under state k
            sel = z_all == k
            if sel.sum() < 10:
                continue
            ax.scatter(dvbar[sel], v_all[sel], s=6, alpha=0.25, color=colors[k])
            vf = state_list[k].vel.resolve(params.vel_per_state.get(names[k], {}))
            xs = np.array([dvbar[sel].min(), dvbar[sel].max()])
            ax.plot(xs, vf["beta_vel"] * xs + vf["alpha_vel"], color=colors[k],
                    lw=2, label=f"{names[k]} (β={vf['beta_vel']:.2f})")
        ax.set_xlabel("E[DV | stimulus]  (decision variable)")
        ax.set_ylabel("velocity (z)")
        ax.set_title("(c) confidence coupling by state\n(line = fitted β_vel·DV+α_vel)")
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


__all__ = ["plot_animal_fit", "plot_group_summary", "write_group_table"]


# ---------------------------------------------------------------------------
# Cross-animal aggregation (works purely from the per-animal summary dicts, so
# it can be re-run from a saved fit_summary.json without refitting)
# ---------------------------------------------------------------------------


def _occ_matrix(summaries, names):
    """(n_animals, K) occupancy matrix in a consistent state order."""
    return np.array([[s["state_occupancy"].get(n, 0.0) for n in names]
                     for s in summaries])


def plot_group_summary(summaries, outpath):
    """Aggregate figure across animals from a list of per-animal summary dicts.

      (a) state occupancy per animal (stacked) -- between-animal heterogeneity;
      (b) group-mean transition matrix A;
      (c) fitted velocity marker mu per state across animals (consistency of the
          engagement marker), or fitted bias/slope if the fits were choice-only;
      (d) per-animal model fit quality (log-lik per trial).
    """
    if not summaries:
        raise ValueError("no summaries to aggregate")
    names = summaries[0]["state_names"]
    K = len(names)
    animals = [s["animal"] for s in summaries]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(k) for k in range(K)]
    has_vel = all(s.get("vel_per_state") for s in summaries)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    fig.suptitle(f"IO-HMM group summary — {len(animals)} animals, "
                 f"{sum(s['n_trials'] for s in summaries)} trials",
                 fontsize=13, fontweight="bold")

    # (a) occupancy per animal, stacked
    ax = fig.add_subplot(gs[0, 0])
    occ = _occ_matrix(summaries, names)
    bottom = np.zeros(len(animals))
    x = np.arange(len(animals))
    for k in range(K):
        ax.bar(x, occ[:, k], bottom=bottom, color=colors[k], label=names[k])
        bottom += occ[:, k]
    ax.set_xticks(x); ax.set_xticklabels(animals, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of trials"); ax.set_ylim(0, 1)
    ax.set_title("(a) state occupancy per animal")
    ax.legend(fontsize=8, ncol=min(K, 4), loc="upper center",
              bbox_to_anchor=(0.5, -0.18))

    # (b) group-mean transition matrix
    ax = fig.add_subplot(gs[0, 1])
    A_mean = np.mean([np.array(s["A"]) for s in summaries], axis=0)
    im = ax.imshow(A_mean, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{A_mean[i, j]:.2f}", ha="center", va="center",
                    color="w" if A_mean[i, j] < 0.6 else "k", fontsize=8)
    ax.set_title("(b) group-mean transition matrix A")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) per-state confidence coupling beta_vel across animals (else bias alpha)
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    if has_vel:
        label, title = "fitted confidence gain β_vel", \
            "(c) velocity-confidence coupling β_vel per state across animals"
        getval = lambda s, n: s["vel_per_state"].get(n, {}).get("beta_vel", np.nan)
    else:
        label, title = "fitted alpha (bias)", \
            "(c) fitted bias (alpha) per state across animals"
        getval = lambda s, n: s["psych_per_state"].get(n, {}).get("alpha", np.nan)
    for k, n in enumerate(names):
        vals = np.array([getval(s, n) for s in summaries], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        xj = k + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xj, vals, color=colors[k], s=28, alpha=0.8)
        ax.plot([k - 0.2, k + 0.2], [vals.mean()] * 2, color="k", lw=2)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(label); ax.set_title(title)
    ax.axhline(0, color="0.85", ls=":")

    # (d) fit quality per animal: log-lik per trial
    ax = fig.add_subplot(gs[1, 1])
    llpt = [s["final_log_lik"] / max(s["n_trials"], 1) for s in summaries]
    ax.bar(x, llpt, color="0.5")
    ax.set_xticks(x); ax.set_xticklabels(animals, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log-lik per trial"); ax.set_title("(d) fit quality per animal")

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_group_table(summaries, csv_path):
    """Tidy per-animal table + a group-mean row -> CSV."""
    import csv
    if not summaries:
        raise ValueError("no summaries to tabulate")
    names = summaries[0]["state_names"]
    header = (["animal", "n_sessions", "n_trials", "final_log_lik",
               "ll_per_trial"] + [f"occ_{n}" for n in names])
    rows = []
    for s in summaries:
        llpt = s["final_log_lik"] / max(s["n_trials"], 1)
        rows.append([s["animal"], s["n_sessions"], s["n_trials"],
                     round(s["final_log_lik"], 2), round(llpt, 4)]
                    + [round(s["state_occupancy"].get(n, 0.0), 4) for n in names])
    occ = _occ_matrix(summaries, names).mean(axis=0)
    mean_row = ["GROUP_MEAN", "", sum(s["n_trials"] for s in summaries), "",
                round(np.mean([r[4] for r in rows]), 4)] + [round(v, 4) for v in occ]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
        w.writerow(mean_row)
    return csv_path


def _main():
    """Re-aggregate from a saved fit_summary.json without refitting."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Aggregate IO-HMM fits across animals")
    ap.add_argument("--summary", required=True, help="path to fit_summary.json")
    ap.add_argument("--out", default=None, help="output dir (default: alongside summary)")
    args = ap.parse_args()
    with open(args.summary) as fh:
        summaries = json.load(fh)
    out = args.out or os.path.dirname(os.path.abspath(args.summary))
    fig = plot_group_summary(summaries, os.path.join(out, "group_summary.png"))
    tbl = write_group_table(summaries, os.path.join(out, "group_table.csv"))
    print(f"wrote {fig}\nwrote {tbl}")


if __name__ == "__main__":
    _main()
