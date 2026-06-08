# -*- coding: utf-8 -*-
"""Intuition figures for the shipped velocity channel (fit(..., use_velocity=True)).

Everything here runs the *production* EM, so the figures reflect the real
fitted model, not a standalone mock. Two states are made deliberately hard for
a choice-only HMM (small bias gap) but carry distinct velocity markers.

  fig4_velocity_intuition.png
    (a) the two states' psychometric curves nearly overlap -> choices barely
        separate them;
    (b) ... but their pre-RZ velocity distributions are well separated -> the
        extra channel carries the missing state information;
    (c) trial-by-trial Viterbi decode (one session): choice-only is near chance,
        choice+velocity tracks the true latent state;
    (d) dose-response: Viterbi agreement vs the velocity marker's d-prime --
        choice-only stays flat near chance, choice+velocity climbs to ceiling.

Run:  python scripts/velocity_intuition_figures.py [--outdir figures] [--quick]
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
BIAS_GAP = 1.0   # alpha = -/+ 0.5: states are behaviourally similar in choices


def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _conds():
    s_vals = np.array([30, 35, 40, 45, 50, 55, 60], float)
    return np.array([[s, c, 0.0] for s in s_vals for c in (0.5, 1.0)], float)


def _states():
    grids = io_core.IOGrids.default()
    bimodal = io_core.prior_bimodal(grids)
    # beta_vel=0 => stimulus-independent velocity baseline (the engagement-marker
    # special case of the confidence model), which is what these figures probe.
    VS0 = states_mod.VelocitySpec(beta_vel=0.0)
    return grids, [states_mod.IOState("Lo", bimodal, PS(beta=1., gamma=0., delta=0.), vel=VS0),
                   states_mod.IOState("Hi", bimodal, PS(beta=1., gamma=0., delta=0.), vel=VS0)]


def _simulate(dprime, T, n_sess, seed, sigma=1.0):
    grids, sl = _states()
    stage1 = _stage1()
    tp = {"Lo": {"alpha": -BIAS_GAP / 2}, "Hi": {"alpha": BIAS_GAP / 2}}
    vel = {"Lo": {"alpha_vel": 0.0, "sigma_vel": sigma},
           "Hi": {"alpha_vel": dprime * sigma, "sigma_vel": sigma}}
    A = np.array([[0.92, 0.08], [0.08, 0.92]])
    pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(seed)
    conds = _conds()
    tl, zl = [], []
    for _ in range(n_sess):
        tr, z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds, T,
                                      rng, vel_per_state=vel)
        tl.append(tr); zl.append(z)
    return grids, stage1, sl, tl, zl, vel


def _best_perm(paths, z_list, K):
    """Return (agreement, perm) maximising agreement over label permutations."""
    best_a, best_p = -1.0, tuple(range(K))
    for perm in itertools.permutations(range(K)):
        mp = np.array(perm)
        a = np.mean([(mp[p] == z).mean() for p, z in zip(paths, z_list)])
        if a > best_a:
            best_a, best_p = a, perm
    return best_a, np.array(best_p)


def _psych_curve(grids, stage1, state, alpha):
    s = np.linspace(28, 62, 60)
    conds = np.array([[x, 1.0, 0.0] for x in s], float)
    return s, emissions_mod.p_go_per_unique_condition(
        grids, stage1, state, {"alpha": alpha}, conds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    T, n_sess = (500, 2) if args.quick else (700, 3)
    n_restarts = 3
    dp_demo = 2.5

    # ---- data + fits at the demo d' (used for panels a, b, c) ----
    grids, stage1, sl, tl, zl, vel_true = _simulate(dp_demo, T, n_sess, seed=0)
    pc, _ = fit_mod.fit(tl, sl, stage1, grids, n_restarts=n_restarts,
                        max_iters=60, seed=1)
    pv, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                        n_restarts=n_restarts, max_iters=60, seed=1)
    paths_c = fit_mod.viterbi_paths(pc, tl, sl, stage1, grids)
    paths_v = fit_mod.viterbi_paths(pv, tl, sl, stage1, grids)
    ac, perm_c = _best_perm(paths_c, zl, 2)
    av, perm_v = _best_perm(paths_v, zl, 2)
    print(f"  demo d'={dp_demo}: choice-only={ac:.1%}  choice+vel={av:.1%}")

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle("Why a velocity channel rescues states the choices cannot "
                 f"(bias gap {BIAS_GAP}, demo d'={dp_demo})",
                 fontsize=13, fontweight="bold")

    # (a) psychometric curves: nearly overlapping
    ax = fig.add_subplot(2, 2, 1)
    for st, a, col in zip(sl, (-BIAS_GAP / 2, BIAS_GAP / 2), ("C0", "C1")):
        s, p = _psych_curve(grids, stage1, st, a)
        ax.plot(s, p, color=col, lw=2.2, label=f"{st.name} (alpha={a:+.1f})")
    ax.axvline(45, color="0.8", ls=":")
    ax.set_xlabel("stimulus angle (deg)"); ax.set_ylabel("P(go)")
    ax.set_title("(a) choices barely differ\n(psychometric curves nearly overlap)")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)

    # (b) velocity distributions: well separated
    ax = fig.add_subplot(2, 2, 2)
    v_all = np.concatenate([t.velocity for t in tl])
    z_all = np.concatenate(zl)
    bins = np.linspace(v_all.min(), v_all.max(), 40)
    for k, (name, col) in enumerate(zip(("Lo", "Hi"), ("C0", "C1"))):
        ax.hist(v_all[z_all == k], bins=bins, alpha=0.6, color=col,
                density=True, label=f"{name} (true state)")
    # fitted markers (perm-aligned to truth)
    inv = np.argsort(perm_v)
    for k, col in enumerate(("C0", "C1")):
        name = sl[inv[k]].name
        mu = pv.vel_per_state[name]["alpha_vel"]  # beta_vel=0 => baseline = mean
        ax.axvline(mu, color=col, lw=2, ls="--")
    ax.set_xlabel("pre-RZ velocity (standardised)"); ax.set_ylabel("density")
    ax.set_title("(b) ... but velocity separates them\n"
                 "(dashed = fitted markers)")
    ax.legend(fontsize=8)

    # (c) trial ribbon, one session
    ax = fig.add_subplot(2, 1, 2)
    z = zl[0][:160]
    dc = perm_c[paths_c[0]][:160]
    dv = perm_v[paths_v[0]][:160]
    t = np.arange(len(z))
    ax.step(t, z + 0.0, where="mid", color="0.25", lw=2.4, label="true state")
    ax.step(t, dc + 0.10, where="mid", color="C3", lw=1.1,
            label=f"choice only ({ac:.0%})")
    ax.step(t, dv - 0.10, where="mid", color="C2", lw=1.4,
            label=f"choice + velocity ({av:.0%})")
    mis_c = np.where(z != dc)[0]
    ax.plot(mis_c, np.full_like(mis_c, 1.45, dtype=float), "|", color="C3", ms=6)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Lo", "Hi"])
    ax.set_ylim(-0.4, 1.7)
    ax.set_xlabel("trial (session 0)")
    ax.set_title("(c) latent-state decoding: choice-only flips near chance; "
                 "choice+velocity tracks truth  (red ticks = choice-only errors)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(args.outdir, "fig4_velocity_intuition.png")
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"  wrote {p1}")

    # ---- (d) dose-response sweep over d' ----
    dprimes = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    if args.quick:
        dprimes = [0.0, 1.0, 2.0, 3.0]
    ag_c, ag_v = [], []
    for dp in dprimes:
        grids, stage1, sl, tl, zl, _ = _simulate(dp, T, n_sess, seed=0)
        pc, _ = fit_mod.fit(tl, sl, stage1, grids, n_restarts=n_restarts,
                            max_iters=60, seed=1)
        pv, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                            n_restarts=n_restarts, max_iters=60, seed=1)
        a_c, _ = _best_perm(fit_mod.viterbi_paths(pc, tl, sl, stage1, grids), zl, 2)
        a_v, _ = _best_perm(fit_mod.viterbi_paths(pv, tl, sl, stage1, grids), zl, 2)
        ag_c.append(a_c); ag_v.append(a_v)
        print(f"  d'={dp:.1f}  choice={a_c:.1%}  choice+vel={a_v:.1%}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dprimes, np.array(ag_c) * 100, "o-", color="C3",
            label="choice only")
    ax.plot(dprimes, np.array(ag_v) * 100, "s-", color="C2",
            label="choice + velocity")
    ax.axhline(50, color="0.5", ls=":", label="chance")
    ax.axvline(dp_demo, color="0.8", ls="--")
    ax.set_xlabel("velocity marker separation  d'  =  |mu_Hi - mu_Lo| / sigma")
    ax.set_ylabel("Viterbi agreement (perm-corrected, %)")
    ax.set_title(f"(d) dose-response (bias gap {BIAS_GAP}: choices alone ~chance)")
    ax.set_ylim(45, 102); ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(args.outdir, "fig5_velocity_doseresponse.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"  wrote {p2}")


if __name__ == "__main__":
    main()
