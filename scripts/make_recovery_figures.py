# -*- coding: utf-8 -*-
"""Parameter-recovery / identifiability figures for the IO-HMM EM fit.

Rather than one rigged scenario, these figures decompose recovery into the two
*distinct* identifiability questions the model faces, because they have very
different answers:

  fig1_psych_recovery.png  -- PSYCHOMETRIC recovery *in isolation* (state label
        known; single-state K=1 fits, lots of data). This is the optimistic
        "what is recoverable in principle" panel. alpha and the lapse gamma
        recover essentially exactly across a broad, *non-trivial* range; the
        slope beta recovers up to a saturation wall set by the IO evidence
        range (|g| ~ 5), beyond which the sigmoid is pinned and beta runs away.
        The example curves panel shows these are ordinary shallow / biased /
        lapsing psychometrics -- nothing degenerate.

  fig2_hmm_separability.png -- JOINT HMM recovery. The latent path (and hence
        per-state attribution) is only recoverable when states are
        behaviourally *separated*: choice-only emissions carry ~1 bit/trial, so
        Viterbi decoding is at chance until the per-state P(go) curves differ
        enough (here, a bias gap >~ 3). Permutation-corrected Viterbi agreement
        vs separation makes the threshold explicit; a well-separated 3-state
        scenario shows the full pipeline recovers A + path + psych once that
        bar is cleared. This separation requirement -- not any need for extreme
        params -- is why the headline test uses widely-spaced states, and is
        the concrete motivation for v0.5 velocity emissions (more bits/trial).

Run:  python scripts/make_recovery_figures.py [--outdir figures] [--quick]
"""

from __future__ import annotations

import argparse
import itertools
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
# Shared structure
# --------------------------------------------------------------------------

def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _conds():
    """Boundary-spanning grid; asymptotes present so lapses are identifiable."""
    s_vals = np.array([25, 28, 32, 36, 40, 45, 50, 54, 58, 62, 65], float)
    return np.array([[s, c, 0.0] for s in s_vals for c in (0.5, 1.0)], float)


def _grids_prior():
    grids = io_core.IOGrids.default()
    return grids, io_core.prior_bimodal(grids)


# --------------------------------------------------------------------------
# Isolated single-state psychometric recovery (label known)
# --------------------------------------------------------------------------

def _isolated_recover(spec_kwargs, free_key, true_val, fixed_full,
                      T, n_seeds):
    """Fit one free psych param on a K=1 (no-HMM) model; return per-seed fits."""
    grids, prior = _grids_prior()
    stage1, conds = _stage1(), _conds()
    st = states_mod.IOState("X", prior, PS(**spec_kwargs))
    full = {**fixed_full, free_key: true_val}
    fits = []
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        tr, _ = sim.simulate_sequence([st], stage1, grids, np.array([1.0]),
                                      np.array([[1.0]]), {"X": full},
                                      conds, T, rng)
        p, _ = fit_mod.fit([tr], [st], stage1, grids,
                           n_restarts=2, max_iters=60, seed=sd)
        fits.append(p.psych_per_state["X"][free_key])
    return np.array(fits)


def _sweep(free_key, spec_kwargs, fixed_full, true_grid, T, n_seeds):
    means, stds = [], []
    for v in true_grid:
        f = _isolated_recover(spec_kwargs, free_key, v, fixed_full, T, n_seeds)
        means.append(f.mean()); stds.append(f.std())
    return np.array(true_grid, float), np.array(means), np.array(stds)


def _psych_curve(state, free, c=1.0):
    grids, _ = _grids_prior()
    stage1 = _stage1()
    s = np.linspace(20, 70, 60)
    conds = np.array([[x, c, 0.0] for x in s], float)
    return s, emissions_mod.p_go_per_unique_condition(grids, stage1, state,
                                                      dict(free), conds)


def figure_psych(outdir, T, n_seeds):
    grids, prior = _grids_prior()
    stage1 = _stage1()

    # beta saturates against the |g|~5 IO evidence range; alpha & gamma do not.
    b_true, b_m, b_s = _sweep("beta", dict(alpha=0.0, gamma=0.0, delta=0.0),
                              {"alpha": 0.0, "gamma": 0.0, "delta": 0.0},
                              [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0], T, n_seeds)
    a_true, a_m, a_s = _sweep("alpha", dict(beta=1.0, gamma=0.0, delta=0.0),
                              {"beta": 1.0, "gamma": 0.0, "delta": 0.0},
                              [-2, -1, -0.5, 0, 0.5, 1, 2], T, n_seeds)
    g_true, g_m, g_s = _sweep("gamma", dict(alpha=0.0, beta=1.0, delta=0.0),
                              {"alpha": 0.0, "beta": 1.0, "delta": 0.0},
                              [0.0, 0.05, 0.1, 0.2, 0.3], T, n_seeds)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Psychometric recovery in isolation (state label known, "
                 f"K=1, T={T}, {n_seeds} seeds)", fontsize=13, fontweight="bold")

    def _identity(ax, lo, hi):
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1, label="identity")

    ax = axes[0, 0]
    ax.errorbar(b_true, b_m, yerr=b_s, fmt="o-", capsize=3, color="C0", zorder=3)
    _identity(ax, 0, 5.2)
    ax.axvspan(1.5, 5.3, color="red", alpha=0.07)
    ax.text(3.2, 0.6, "saturation:\n|g|~5 pins\nthe sigmoid", color="firebrick",
            fontsize=8, ha="center")
    ax.set_title("(a) slope beta -- recovers to ~1.5, then runs away")
    ax.set_xlabel("true beta"); ax.set_ylabel("recovered beta")
    ax.set_xlim(0, 5.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.errorbar(a_true, a_m, yerr=a_s, fmt="o-", capsize=3, color="C2", zorder=3)
    _identity(ax, -2.2, 2.2)
    ax.set_title("(b) bias alpha -- recovers across the range")
    ax.set_xlabel("true alpha"); ax.set_ylabel("recovered alpha")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.errorbar(g_true, g_m, yerr=g_s, fmt="o-", capsize=3, color="C4", zorder=3)
    _identity(ax, 0, 0.31)
    ax.set_title("(c) lapse gamma -- recovers across the range")
    ax.set_xlabel("true gamma"); ax.set_ylabel("recovered gamma")
    ax.legend(fontsize=8)

    # (d) example non-trivial curves: shallow / biased / lapsing, true vs fit
    ax = axes[1, 1]
    examples = [
        ("shallow (beta=0.5)", PS(alpha=0.0, gamma=0.0, delta=0.0),
         "beta", 0.5, {"alpha": 0.0, "gamma": 0.0, "delta": 0.0}, "C0"),
        ("biased (alpha=+1.5)", PS(beta=1.0, gamma=0.0, delta=0.0),
         "alpha", 1.5, {"beta": 1.0, "gamma": 0.0, "delta": 0.0}, "C2"),
        ("lapsing (gamma=0.2)", PS(alpha=0.0, beta=1.0, delta=0.0),
         "gamma", 0.2, {"alpha": 0.0, "beta": 1.0, "delta": 0.0}, "C4"),
    ]
    for label, spec, key, val, fixed, col in examples:
        st = states_mod.IOState(label, prior, spec)
        fit_seed = _isolated_recover(
            {k: getattr(spec, k) for k in ("alpha", "beta", "gamma", "delta")
             if getattr(spec, k) is not None},
            key, val, fixed, T, 1)[0]
        true_full = {**fixed, key: val}
        fit_full = {**fixed, key: fit_seed}
        s, p_true = _psych_curve(st, {k: true_full[k] for k in spec.free_params})
        _, p_fit = _psych_curve(st, {k: fit_full[k] for k in spec.free_params})
        ax.plot(s, p_true, color=col, lw=2.2, label=f"{label} true")
        ax.plot(s, p_fit, color=col, lw=1.1, ls="--", label="  recovered")
    ax.set_title("(d) example non-trivial curves: true vs recovered")
    ax.set_xlabel("stimulus angle (deg)"); ax.set_ylabel("P(go)")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7, ncol=1, loc="center left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, "fig1_psych_recovery.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------
# HMM separability + a well-separated full recovery
# --------------------------------------------------------------------------

def _perm_agreement(paths, z_list, K):
    best = 0.0
    for perm in itertools.permutations(range(K)):
        mp = np.array(perm)
        a = np.mean([(mp[p] == z).mean() for p, z in zip(paths, z_list)])
        best = max(best, a)
    return best


def _separability_point(gap, T, n_sessions, grids, prior, stage1, conds):
    states = [states_mod.IOState("Lo", prior, PS(beta=1.0, gamma=0.0, delta=0.0)),
              states_mod.IOState("Hi", prior, PS(beta=1.0, gamma=0.0, delta=0.0))]
    tp = {"Lo": {"alpha": -gap / 2}, "Hi": {"alpha": gap / 2}}
    A = np.array([[0.92, 0.08], [0.08, 0.92]]); pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(0)
    tl, zl = [], []
    for _ in range(n_sessions):
        tr, z = sim.simulate_sequence(states, stage1, grids, pi, A, tp,
                                      conds, T, rng)
        tl.append(tr); zl.append(z)
    p, _ = fit_mod.fit(tl, states, stage1, grids, n_restarts=3,
                       max_iters=60, seed=1)
    paths = fit_mod.viterbi_paths(p, tl, states, stage1, grids)
    return _perm_agreement(paths, zl, 2)


def figure_separability(outdir, quick):
    grids, prior = _grids_prior()
    stage1, conds = _stage1(), _conds()
    T, n_sessions = (600, 3) if quick else (900, 4)

    gaps = [1.0, 2.0, 3.0, 4.0, 6.0]
    agree = [_separability_point(g, T, n_sessions, grids, prior, stage1, conds)
             for g in gaps]

    # Well-separated 3-state full recovery (the headline pipeline works once
    # the separation bar is cleared).
    bimodal = prior
    state_list = [
        states_mod.IOState("Sensory",  bimodal, PS(alpha=0.0,  beta=3.0, gamma=0.0, delta=0.0)),
        states_mod.IOState("GoBias",   bimodal, PS(alpha=2.2,  beta=0.0, gamma=0.0, delta=0.0)),
        states_mod.IOState("NoGoBias", bimodal, PS(alpha=-2.2, beta=0.0, gamma=0.0, delta=0.0)),
    ]
    names = [s.name for s in state_list]
    K = len(state_list)
    A_true = np.full((K, K), 0.08 / (K - 1)); np.fill_diagonal(A_true, 0.92)
    pi = np.ones(K) / K
    rng = np.random.default_rng(0)
    tl, zl = [], []
    for _ in range(3):
        tr, z = sim.simulate_sequence(state_list, stage1, grids, pi, A_true,
                                      {n: {} for n in names}, conds, 700, rng)
        tl.append(tr); zl.append(z)
    params, _ = fit_mod.fit(tl, state_list, stage1, grids,
                            n_restarts=3, max_iters=60, seed=1)
    paths = fit_mod.viterbi_paths(params, tl, state_list, stage1, grids)
    full_agree = np.mean([(p == z).mean() for p, z in zip(paths, zl)])
    A_err = np.abs(params.A - A_true).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Joint HMM recovery is gated by behavioural separation "
                 "(choice-only ~1 bit/trial)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(gaps, np.array(agree) * 100, "o-", color="C3", zorder=3)
    ax.axhline(50, color="0.5", ls=":", label="chance (2 states)")
    ax.axvspan(0, 2.6, color="red", alpha=0.07)
    ax.text(1.5, 60, "states\nconfusable", color="firebrick", ha="center", fontsize=9)
    ax.set_xlabel("behavioural separation: bias gap  |alpha_Hi - alpha_Lo|")
    ax.set_ylabel("Viterbi agreement (perm-corrected, %)")
    ax.set_title("(a) latent path recovery vs state separation")
    ax.set_ylim(45, 100); ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(A_true.ravel(), params.A.ravel(), s=70, color="C0", zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("true A entries"); ax.set_ylabel("fitted A entries")
    ax.set_title(f"(b) well-separated 3-state fit: A recovers\n"
                 f"|A_err|={A_err:.3f}, Viterbi={full_agree:.0%}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "fig2_hmm_separability.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}  (sep agree={[round(a,2) for a in agree]}, "
          f"full |A_err|={A_err:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures"))
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds / shorter sessions for a fast preview")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    T, n_seeds = (3000, 4) if args.quick else (5000, 6)
    print(f"Generating figures into {args.outdir} (quick={args.quick}) ...")
    figure_psych(args.outdir, T=T, n_seeds=n_seeds)
    figure_separability(args.outdir, quick=args.quick)
    print("done.")


if __name__ == "__main__":
    main()
