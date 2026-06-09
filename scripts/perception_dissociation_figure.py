# -*- coding: utf-8 -*-
"""Figure: free per-state perception (Phase 2) and the perception-vs-action
dissociation.

Produces ``figures/fig6_perception_dissociation.png`` with three panels:

  (a) MECHANISM. Sensory precision ``lambda_`` multiplies the Stage-1 kappa for
      *both* the generative ``p(m|s)`` and the inference posterior, so it
      reshapes the IO decision variable ``DV(m)`` -- sharper (lambda_ > 1) or
      blurrier (lambda_ < 1) perception. The choice psychometric, by contrast,
      only rescales an otherwise-fixed ``DV``.

  (b) LAMBDA DISSOCIATION. Two states with *identical* choice/action params
      (beta = beta_vel = 2) but different perception (Sharp lambda_=2.0 vs Blur
      lambda_=0.5). Free ``lambda_`` recovers the perceptual difference while the
      action gain ``beta_vel`` recovers to ~2 for both -- perception moves,
      action does not. Identifiable only because the velocity channel reads
      ``DV(m)`` independently of the choice psychometric.

  (c) PRIOR-WEIGHT DISSOCIATION (Phase 2b). A free *perceptual category bias*
      ``prior_weight`` (mass on the 0-deg Go centre) recovers and separates two
      states whose choice/action params match (Lo 0.25 vs Hi 0.75).

The recovery setups mirror the headline tests in
``tests/test_io_hmm_perception.py`` exactly, so the figure shows the same
quantities the suite asserts on. PNGs are git-ignored (regenerable).

    python scripts/perception_dissociation_figure.py            # full
    python scripts/perception_dissociation_figure.py --quick    # fewer iters
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
IO_HMM = os.path.join(REPO_ROOT, "ideal_observer", "io_hmm")
if IO_HMM not in sys.path:
    sys.path.insert(0, IO_HMM)

import io_core            # noqa: E402
import states as states_mod   # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod     # noqa: E402
import simulate as sim    # noqa: E402

PS = states_mod.PsychSpec
VS = states_mod.VelocitySpec
PC = states_mod.PerceptionSpec


def _stage1():
    return emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)


def _perm_agree(paths, z_list, K):
    return max(np.mean([(np.array(perm)[p] == z).mean()
                        for p, z in zip(paths, z_list)])
               for perm in itertools.permutations(range(K)))


# ---------------------------------------------------------------------------
# (a) lambda_ reshapes the IO decision variable
# ---------------------------------------------------------------------------

def panel_mechanism(ax):
    grids = io_core.IOGrids.default()
    st = states_mod.IOState("A", io_core.prior_bimodal(grids), PS())
    stage1 = _stage1()
    s_grid = np.linspace(0.0, 90.0, 25)
    conds = np.column_stack([s_grid, np.ones_like(s_grid), np.zeros_like(s_grid)])
    for lam, style in [(0.5, "--"), (1.0, "-"), (2.0, "-.")]:
        _g, dv, _pm = emissions_mod.precompute_state_terms(
            grids, stage1, st, conds, kappa_scale=lam)
        ax.plot(s_grid, dv, style, label=f"lambda_={lam:g}", lw=2)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("stimulus orientation (deg)")
    ax.set_ylabel("DV(m) = EU(Go) - EU(NoGo)")
    ax.set_title("(a) sensory precision reshapes DV(m)")
    ax.legend(frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# (b) lambda_ dissociation: perception moves, action does not
# ---------------------------------------------------------------------------

def _sim_lambda(grids, sl, T, n_sess, seed=0):
    stage1 = _stage1()
    tp = {"Sharp": {"beta": 2.0}, "Blur": {"beta": 2.0}}
    velp = {"Sharp": {"beta_vel": 2.0, "alpha_vel": 0.0, "sigma_vel": 0.5},
            "Blur": {"beta_vel": 2.0, "alpha_vel": 1.5, "sigma_vel": 0.5}}
    percp = {"Sharp": {"lambda_": 2.0}, "Blur": {"lambda_": 0.5}}
    A = np.array([[0.93, 0.07], [0.07, 0.93]]); pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(seed)
    conds = np.array([[s, 1.0, 0.0] for s in [25, 40, 55, 70]], float)
    tl, zl = [], []
    for _ in range(n_sess):
        tr, z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds, T,
                                      rng, vel_per_state=velp, perc_per_state=percp)
        tl.append(tr); zl.append(z)
    return stage1, tl, zl


def panel_lambda(ax, quick):
    grids = io_core.IOGrids.default()
    bim = io_core.prior_bimodal(grids)
    perc = PC(lambda_=None)
    sl = [states_mod.IOState("Sharp", bim, PS(alpha=0., gamma=0., delta=0.), vel=VS(), perc=perc),
          states_mod.IOState("Blur", bim, PS(alpha=0., gamma=0., delta=0.), vel=VS(), perc=perc)]
    T = 250 if quick else 350
    n_sess = 2 if quick else 3
    iters = 18 if quick else 30
    stage1, tl, zl = _sim_lambda(grids, sl, T, n_sess, seed=0)
    params, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                            n_restarts=2, max_iters=iters, seed=1)
    agree = _perm_agree(fit_mod.viterbi_paths(params, tl, sl, stage1, grids), zl, 2)

    lam = {n: params.perc_per_state[n]["lambda_"] for n in ("Sharp", "Blur")}
    bvel = {n: params.vel_per_state[n]["beta_vel"] for n in ("Sharp", "Blur")}
    true_lam = {"Sharp": 2.0, "Blur": 0.5}

    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [true_lam["Sharp"], true_lam["Blur"]], w,
           label="true lambda_", color="0.75")
    ax.bar(x + w / 2, [lam["Sharp"], lam["Blur"]], w,
           label="fitted lambda_", color="C0")
    for xi, n in zip(x, ("Sharp", "Blur")):
        ax.text(xi + w / 2, lam[n] + 0.05, f"{lam[n]:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(2.0, color="C3", lw=0.8, ls=":", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["Sharp", "Blur"])
    ax.set_ylabel("sensory precision lambda_")
    ax.set_title("(b) perception dissociates from action")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.02, 0.98,
            f"action gain beta_vel (true 2.0):\n  Sharp={bvel['Sharp']:.2f}  "
            f"Blur={bvel['Blur']:.2f}\nViterbi agreement={agree:.0%}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return dict(lam=lam, bvel=bvel, agree=agree)


# ---------------------------------------------------------------------------
# (c) prior_weight dissociation (Phase 2b)
# ---------------------------------------------------------------------------

def panel_prior(ax, quick):
    grids = io_core.IOGrids.default()
    bim = io_core.prior_bimodal(grids)
    perc = PC(lambda_=1.0, prior_strength=3.0, prior_weight=None,
              parameterized_prior=True)
    sl = [states_mod.IOState("Lo", bim, PS(alpha=0., gamma=0., delta=0.), vel=VS(), perc=perc),
          states_mod.IOState("Hi", bim, PS(alpha=0., gamma=0., delta=0.), vel=VS(), perc=perc)]
    stage1 = _stage1()
    tp = {"Lo": {"beta": 2.0}, "Hi": {"beta": 2.0}}
    velp = {"Lo": {"beta_vel": 2.0, "alpha_vel": 0.0, "sigma_vel": 0.5},
            "Hi": {"beta_vel": 2.0, "alpha_vel": 1.5, "sigma_vel": 0.5}}
    percp = {"Lo": {"prior_weight": 0.25}, "Hi": {"prior_weight": 0.75}}
    A = np.array([[0.93, 0.07], [0.07, 0.93]]); pi = np.array([0.5, 0.5])
    rng = np.random.default_rng(0)
    conds = np.array([[s, 1.0, 0.0] for s in [25, 40, 55, 70]], float)
    T = 250 if quick else 350
    n_sess = 2 if quick else 3
    iters = 18 if quick else 30
    tl, zl = [], []
    for _ in range(n_sess):
        tr, z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds, T,
                                      rng, vel_per_state=velp, perc_per_state=percp)
        tl.append(tr); zl.append(z)
    params, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=True,
                            n_restarts=2, max_iters=iters, seed=1)
    agree = _perm_agree(fit_mod.viterbi_paths(params, tl, sl, stage1, grids), zl, 2)
    pw = {n: params.perc_per_state[n]["prior_weight"] for n in ("Lo", "Hi")}
    true_pw = {"Lo": 0.25, "Hi": 0.75}

    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [true_pw["Lo"], true_pw["Hi"]], w,
           label="true prior_weight", color="0.75")
    ax.bar(x + w / 2, [pw["Lo"], pw["Hi"]], w,
           label="fitted prior_weight", color="C2")
    for xi, n in zip(x, ("Lo", "Hi")):
        ax.text(xi + w / 2, pw[n] + 0.02, f"{pw[n]:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(["Lo (NoGo bias)", "Hi (Go bias)"])
    ax.set_ylabel("perceptual category bias prior_weight")
    ax.set_ylim(0, 1)
    ax.set_title("(c) free prior-weight dissociation")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.02, f"Viterbi agreement={agree:.0%}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return dict(pw=pw, agree=agree)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures"))
    ap.add_argument("--quick", action="store_true",
                    help="fewer sessions/iters (faster, slightly noisier)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panel_mechanism(axes[0])
    rb = panel_lambda(axes[1], args.quick)
    rc = panel_prior(axes[2], args.quick)
    fig.tight_layout()
    path = os.path.join(args.outdir, "fig6_perception_dissociation.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"lambda_ panel : {rb}")
    print(f"prior panel   : {rc}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
