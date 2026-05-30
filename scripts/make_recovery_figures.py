# -*- coding: utf-8 -*-
"""Generate parameter-recovery / diagnostic figures for the IO-HMM EM fit.

Reuses the exact scenarios from ``tests/test_io_hmm_fit.py`` so the figures
illustrate what the test suite asserts. Produces:

  fig1_recovery.png  -- 3-state distinct-fixed-psych scenario:
                        (a) true vs fitted transition matrix A,
                        (b) EM log-likelihood convergence across restarts,
                        (c) Viterbi-decoded vs true latent path,
                        (d) per-state psychometric curves (true vs fitted).
  fig2_free_psych.png -- Sensory + Disengaged joint fit: recovery of the
                        identifiable constant-rate bias (alpha) + A.

Run:  python scripts/make_recovery_figures.py [--outdir figures]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IO_HMM = os.path.join(REPO_ROOT, "ideal_observer", "io_hmm")
if IO_HMM not in sys.path:
    sys.path.insert(0, IO_HMM)

import io_core            # noqa: E402
import states as states_mod   # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod     # noqa: E402
import simulate as sim    # noqa: E402

PS = states_mod.PsychSpec


# --------------------------------------------------------------------------
# Shared helpers (mirrors tests/test_io_hmm_fit.py)
# --------------------------------------------------------------------------

def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _condition_grid():
    s_vals = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65], float)
    c_vals = np.array([0.5, 1.0])
    return np.array([[s, c, 0.0] for s in s_vals for c in c_vals], float)


def _sticky_A(K, p_self):
    A = np.full((K, K), (1.0 - p_self) / (K - 1))
    np.fill_diagonal(A, p_self)
    return A


def _simulate(state_list, true_psych, true_A, T, n_sessions, seed):
    grids = io_core.IOGrids.default()
    stage1 = _stage1()
    conds = _condition_grid()
    K = len(state_list)
    true_pi = np.ones(K) / K
    rng = np.random.default_rng(seed)
    trials_list, z_true_list = [], []
    for _ in range(n_sessions):
        tr, z = sim.simulate_sequence(state_list, stage1, grids, true_pi,
                                      true_A, true_psych, conds, T, rng)
        trials_list.append(tr)
        z_true_list.append(z)
    return grids, stage1, trials_list, z_true_list


def _psych_curve(grids, stage1, state, free, c=1.0):
    """True/fitted P(go) vs stimulus angle at fixed contrast c."""
    s_vals = np.linspace(20, 70, 60)
    conds = np.array([[s, c, 0.0] for s in s_vals], float)
    p_go = emissions_mod.p_go_per_unique_condition(
        grids, stage1, state, dict(free), conds)
    return s_vals, p_go


# --------------------------------------------------------------------------
# Figure 1: transition + path + psychometric recovery (3 distinct states)
# --------------------------------------------------------------------------

def figure_recovery(outdir):
    grids = io_core.IOGrids.default()
    bimodal = io_core.prior_bimodal(grids)
    state_list = [
        states_mod.IOState("Sensory",  bimodal, PS(alpha=0.0,  beta=3.0, gamma=0.0, delta=0.0)),
        states_mod.IOState("GoBias",   bimodal, PS(alpha=2.2,  beta=0.0, gamma=0.0, delta=0.0)),
        states_mod.IOState("NoGoBias", bimodal, PS(alpha=-2.2, beta=0.0, gamma=0.0, delta=0.0)),
    ]
    names = [s.name for s in state_list]
    K = len(state_list)
    true_A = _sticky_A(K, 0.92)
    true_psych = {n: {} for n in names}
    grids, stage1, trials_list, z_true_list = _simulate(
        state_list, true_psych, true_A, T=700, n_sessions=3, seed=0)

    params, history, allres = fit_mod.fit(
        trials_list, state_list, stage1, grids,
        n_restarts=3, max_iters=60, seed=1, return_all=True)

    paths = fit_mod.viterbi_paths(params, trials_list, state_list, stage1, grids)
    agree = np.mean([(p == z).mean() for p, z in zip(paths, z_true_list)])
    A_err = np.abs(params.A - true_A).mean()

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        f"IO-HMM parameter recovery  (3 distinct fixed-psych states; "
        f"mean |A_err| = {A_err:.3f}, Viterbi agreement = {agree:.1%})",
        fontsize=13, fontweight="bold")

    # (a) transition matrices: true vs fitted
    ax = fig.add_subplot(2, 3, 1)
    im = ax.imshow(true_A, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("(a) True transition A")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{true_A[i,j]:.2f}", ha="center", va="center",
                    color="w" if true_A[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(2, 3, 2)
    im = ax.imshow(params.A, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Fitted transition A")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{params.A[i,j]:.2f}", ha="center", va="center",
                    color="w" if params.A[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) EM log-likelihood convergence across restarts
    ax = fig.add_subplot(2, 3, 3)
    for r, (_, hist) in enumerate(allres):
        best = r == int(np.argmax([h[-1] for _, h in allres]))
        ax.plot(hist, marker="o", ms=2.5, lw=1.8 if best else 1.0,
                label=f"restart {r}" + (" (best)" if best else ""),
                color="C3" if best else None,
                alpha=1.0 if best else 0.6)
    ax.set_title("(b) EM log-likelihood")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("total log-lik")
    ax.legend(fontsize=7)

    # (c) Viterbi-decoded vs true path (first session, first 200 trials)
    ax = fig.add_subplot(2, 1, 2) if False else fig.add_subplot(2, 3, (4, 5))
    z_true = z_true_list[0][:200]
    z_dec = paths[0][:200]
    t = np.arange(len(z_true))
    ax.step(t, z_true, where="mid", color="0.3", lw=2.2, label="true state")
    ax.step(t, z_dec + 0.08, where="mid", color="C3", lw=1.4,
            label="Viterbi decoded")
    mism = np.where(z_true != z_dec)[0]
    ax.plot(mism, np.full_like(mism, -0.4, dtype=float), "|", color="red",
            ms=8, label="mismatch")
    ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(-0.7, K - 0.5)
    ax.set_xlabel("trial (session 0, first 200)")
    ax.set_title(f"(c) Latent-state decoding  (agreement {agree:.1%})")
    ax.legend(fontsize=7, loc="upper right", ncol=3)

    # (d) per-state psychometric curves (true vs fitted)
    ax = fig.add_subplot(2, 3, 6)
    for k, st in enumerate(state_list):
        s, p_true = _psych_curve(grids, stage1, st, true_psych[st.name])
        s, p_fit = _psych_curve(grids, stage1, st,
                                params.psych_per_state.get(st.name, {}))
        ax.plot(s, p_true, color=f"C{k}", lw=2, label=f"{st.name} (true)")
        ax.plot(s, p_fit, color=f"C{k}", lw=1, ls="--",
                label=f"{st.name} (fit)")
    ax.set_xlabel("stimulus angle (deg)"); ax.set_ylabel("P(go)")
    ax.set_title("(d) Psychometric curves  (c=1.0)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, ncol=1, loc="center left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, "fig1_recovery.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}  (|A_err|={A_err:.3f}, agreement={agree:.1%})")
    return agree, A_err


# --------------------------------------------------------------------------
# Figure 2: joint free-psych recovery (Sensory + Disengaged)
# --------------------------------------------------------------------------

def figure_free_psych(outdir):
    grids = io_core.IOGrids.default()
    bimodal = io_core.prior_bimodal(grids)
    state_list = [
        states_mod.IOState("Sensory",    bimodal, PS(alpha=0.0, beta=3.0, gamma=0.0, delta=0.0)),
        states_mod.IOState("Disengaged", bimodal, PS(beta=0.0, gamma=0.0, delta=0.0)),
    ]
    names = [s.name for s in state_list]
    K = len(state_list)
    true_A = _sticky_A(K, 0.92)
    true_alpha = -0.5
    true_psych = {"Sensory": {}, "Disengaged": {"alpha": true_alpha}}
    grids, stage1, trials_list, z_true_list = _simulate(
        state_list, true_psych, true_A, T=700, n_sessions=3, seed=0)

    params, history = fit_mod.fit(trials_list, state_list, stage1, grids,
                                  n_restarts=3, max_iters=60, seed=1)
    fit_alpha = params.psych_per_state["Disengaged"]["alpha"]
    A_err = np.abs(params.A - true_A).mean()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Joint recovery: latent dynamics + free constant-rate bias  "
        f"(alpha: true {true_alpha:+.2f} -> fit {fit_alpha:+.2f}; "
        f"|A_err| {A_err:.3f})", fontsize=12, fontweight="bold")

    # transition recovery scatter
    ax = axes[0]
    ax.scatter(true_A.ravel(), params.A.ravel(), s=60, color="C0", zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("true A entries"); ax.set_ylabel("fitted A entries")
    ax.set_title("(a) Transition recovery")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # alpha (constant P(go)) recovery
    ax = axes[1]
    ax.bar([0, 1], [1 / (1 + np.exp(-true_alpha)), 1 / (1 + np.exp(-fit_alpha))],
           color=["0.4", "C3"], width=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["true", "fitted"])
    ax.set_ylabel("Disengaged P(go) = sigmoid(alpha)")
    ax.set_title("(b) Constant-rate bias recovery")
    ax.set_ylim(0, 1)

    # log-likelihood
    ax = axes[2]
    ax.plot(history, marker="o", ms=3, color="C3")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("total log-lik")
    ax.set_title("(c) EM convergence")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "fig2_free_psych.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}  (alpha {true_alpha:+.2f}->{fit_alpha:+.2f})")
    return fit_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Generating figures into {args.outdir} ...")
    figure_recovery(args.outdir)
    figure_free_psych(args.outdir)
    print("done.")


if __name__ == "__main__":
    main()
