# -*- coding: utf-8 -*-
"""Proof-of-concept: do extra trial-by-trial channels (pre-RZ velocity, licks)
break the two IO-HMM identifiability walls documented in figures/README.md?

This is a *standalone* experiment -- it does NOT modify the production emission
module. It reuses io_core (g(m), p(m|s)) and the HMM core (hmm.viterbi runs on
any precomputed log-emission matrix), and simply *adds* a second channel's
log-likelihood to the choice channel. Two questions:

  (A) Slope wall (fig1a). A binary choice is a 1-bit quantisation of the graded
      decision variable d = alpha + beta*g(m); that is why beta saturates for
      beta >~ 1.5. A *graded* readout coupled to the same d does not saturate.
      We simulate choices + a linear-Gaussian velocity readout v ~ N(kappa*d,
      sigma) and compare beta recovery from choice-only vs choice+velocity.

  (B) Separability wall (fig2a). Choice-only emissions carry ~1 bit/trial, so
      Viterbi decoding is at chance until states' P(go) curves differ a lot
      (bias gap >~ 3). A per-state velocity *marker* (engagement: v ~ N(mu_k,
      sigma), stimulus-independent) adds state information that is independent
      of choice. We decode with KNOWN params (isolating the information content,
      not the M-step) and show the bias-gap threshold collapses as the velocity
      marker's d-prime grows.

Output: figures/fig3_velocity_emissions.png
Run:    python scripts/velocity_emission_demo.py [--outdir figures] [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
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
import hmm                # noqa: E402

PS = states_mod.PsychSpec
EPS = 1e-12


def _setup():
    grids = io_core.IOGrids.default()
    stage1 = emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)
    prior = io_core.prior_bimodal(grids)
    s_vals = np.array([25, 28, 32, 36, 40, 45, 50, 54, 58, 62, 65], float)
    conds = np.array([[s, c, 0.0] for s in s_vals for c in (0.5, 1.0)], float)
    state = states_mod.IOState("X", prior, PS(alpha=0.0, gamma=0.0, delta=0.0))
    g_m, _dv_m, p_m = emissions_mod.precompute_state_terms(grids, stage1, state, conds)
    return grids, conds, g_m, p_m


# --------------------------------------------------------------------------
# (A) Slope wall: choice-only vs choice + graded velocity readout
# --------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def slope_recovery(beta_true, g_m, p_m, n_trials, kappa=1.0, sigma=1.0,
                   rng=None):
    """Simulate (sample the latent m per trial so velocity sees the realised g)
    then recover beta three ways:

      choice      : choice channel only (reproduces the saturation wall).
      vel_free    : choice + velocity, velocity gain (kappa, sigma) FREE.
                    Velocity mean is kappa*beta*g, so only the product kappa*beta
                    is identified -- a free gain is degenerate with beta and does
                    not rescue the slope.
      vel_calib   : choice + velocity with the decision-coupling gain CALIBRATED
                    (kappa, sigma known). Velocity then directly observes beta*g
                    through a *linear* (non-saturating) link, so beta is
                    recovered even where the binary choice is saturated.
    """
    rng = rng or np.random.default_rng(0)
    n_cond, n_m = g_m.shape
    cond = rng.integers(0, n_cond, size=n_trials)
    # sample latent measurement index per trial from p(m | s)
    m_idx = np.array([rng.choice(n_m, p=p_m[c] / p_m[c].sum()) for c in cond])
    g_t = g_m[cond, m_idx]                      # realised decision evidence
    d_t = beta_true * g_t                        # alpha = 0
    choice = (rng.random(n_trials) < _sigmoid(d_t)).astype(float)
    vel = kappa * d_t + sigma * rng.standard_normal(n_trials)

    # marginal-over-m likelihoods (consistent with the production IO emission)
    pc = p_m[cond] / p_m[cond].sum(1, keepdims=True)   # (T, n_m)
    gc = g_m[cond]                                       # (T, n_m)

    def ll_choice(beta):
        pgo = np.clip((pc * _sigmoid(beta * gc)).sum(1), EPS, 1 - EPS)
        return (choice * np.log(pgo) + (1 - choice) * np.log(1 - pgo)).sum()

    def ll_vel(beta, kap, sig):
        comp = norm.pdf(vel[:, None], loc=kap * beta * gc, scale=sig)  # (T,n_m)
        pv = np.clip((pc * comp).sum(1), EPS, None)
        return np.log(pv).sum()

    b_choice = minimize_scalar(lambda b: -ll_choice(b),
                               bounds=(0.01, 15), method="bounded").x
    # free gain: fit (beta, kappa, log sigma) jointly
    res = minimize(lambda th: -(ll_choice(th[0]) + ll_vel(th[0], th[1],
                                                          np.exp(th[2]))),
                   x0=[min(beta_true, 2.0), 1.0, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-3, maxiter=2000))
    b_free = res.x[0]
    # calibrated gain: kappa, sigma known -> fit beta only
    b_calib = minimize_scalar(
        lambda b: -(ll_choice(b) + ll_vel(b, kappa, sigma)),
        bounds=(0.01, 15), method="bounded").x
    return b_choice, b_free, b_calib


# --------------------------------------------------------------------------
# (B) Separability wall: per-state velocity engagement marker
# --------------------------------------------------------------------------

def _perm_agree(path, z, K):
    return max(np.mean((np.array(perm)[path] == z))
               for perm in itertools.permutations(range(K)))


def separability(gap, dprime, g_m, p_m, n_trials, n_sess, sigma=1.0, rng=None):
    """Two states: bias +/-gap/2 (choice channel) and velocity means separated
    by `dprime` SDs (engagement channel). Decode with TRUE params under
    choice-only vs choice+velocity; return perm-corrected agreement for each."""
    rng = rng or np.random.default_rng(0)
    K = 2
    A = np.array([[0.92, 0.08], [0.08, 0.92]])
    pi = np.array([0.5, 0.5])
    log_pi, log_A = np.log(pi), np.log(A)
    n_cond, n_m = g_m.shape
    pc = p_m / p_m.sum(1, keepdims=True)
    # marginal P(go|cond) per state for the two biases
    alphas = np.array([-gap / 2, gap / 2])
    pgo = np.stack([np.clip((pc * _sigmoid(alphas[k] + 1.0 * g_m)).sum(1),
                            EPS, 1 - EPS) for k in range(K)], 1)   # (n_cond,K)
    mu_vel = np.array([0.0, dprime * sigma])                        # markers

    agree_c, agree_cv = [], []
    for _ in range(n_sess):
        # sample path
        z = np.empty(n_trials, int)
        z[0] = rng.integers(2)
        for t in range(1, n_trials):
            z[t] = rng.choice(2, p=A[z[t - 1]])
        cond = rng.integers(0, n_cond, size=n_trials)
        ch = (rng.random(n_trials) < pgo[cond, z]).astype(float)
        vel = mu_vel[z] + sigma * rng.standard_normal(n_trials)

        # choice log-emission (T,K)
        le_c = (ch[:, None] * np.log(pgo[cond])
                + (1 - ch)[:, None] * np.log(1 - pgo[cond]))
        # velocity log-emission with TRUE marker params
        le_v = norm.logpdf(vel[:, None], loc=mu_vel[None, :], scale=sigma)

        p_c, _ = hmm.viterbi(log_pi, log_A, le_c)
        p_cv, _ = hmm.viterbi(log_pi, log_A, le_c + le_v)
        agree_c.append(_perm_agree(p_c, z, K))
        agree_cv.append(_perm_agree(p_cv, z, K))
    return np.mean(agree_c), np.mean(agree_cv)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    grids, conds, g_m, p_m = _setup()
    n_seeds = 3 if args.quick else 6
    T_a = 3000 if args.quick else 6000

    # (A) slope recovery sweep
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    bc_m, bc_s, bf_m, bf_s, bk_m, bk_s = [], [], [], [], [], []
    for b in betas:
        bc, bf, bk = [], [], []
        for sd in range(n_seeds):
            c, f, k = slope_recovery(b, g_m, p_m, T_a,
                                     rng=np.random.default_rng(sd))
            bc.append(c); bf.append(f); bk.append(k)
        bc_m.append(np.mean(bc)); bc_s.append(np.std(bc))
        bf_m.append(np.mean(bf)); bf_s.append(np.std(bf))
        bk_m.append(np.mean(bk)); bk_s.append(np.std(bk))
        print(f"  beta={b:.1f}  choice={np.mean(bc):.2f}+/-{np.std(bc):.2f}  "
              f"vel(free gain)={np.mean(bf):.2f}+/-{np.std(bf):.2f}  "
              f"vel(calibrated)={np.mean(bk):.2f}+/-{np.std(bk):.2f}")

    # (B) separability vs bias gap for several velocity d-primes
    gaps = [0.5, 1.0, 2.0, 3.0, 4.0]
    T_b, n_sess = (500, 3) if args.quick else (800, 4)
    dprimes = [0.0, 1.0, 2.0]
    curves = {}
    for dp in dprimes:
        cv = []
        for g in gaps:
            _, a_cv = separability(g, dp, g_m, p_m, T_b, n_sess,
                                   rng=np.random.default_rng(0))
            cv.append(a_cv)
        curves[dp] = cv
        print(f"  velocity d'={dp}: choice+vel agreement vs gap = "
              f"{[round(x,2) for x in cv]}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Extra trial-by-trial channels break both IO-HMM "
                 "identifiability walls", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.errorbar(betas, bc_m, yerr=bc_s, fmt="o-", capsize=3, color="C0",
                label="choice only")
    ax.errorbar(betas, bf_m, yerr=bf_s, fmt="^-", capsize=3, color="C3",
                label="+ velocity, free gain (degenerate)")
    ax.errorbar(betas, bk_m, yerr=bk_s, fmt="s-", capsize=3, color="C2",
                label="+ velocity, calibrated gain")
    ax.plot([0, 5.2], [0, 5.2], "k--", lw=1, label="identity")
    ax.axvspan(1.5, 5.3, color="red", alpha=0.07)
    ax.set_xlim(0, 5.3); ax.set_ylim(0, 8)
    ax.set_xlabel("true beta"); ax.set_ylabel("recovered beta")
    ax.set_title("(a) slope wall: a calibrated graded readout of\n"
                 "d=alpha+beta*g recovers beta; a free-gain one does not")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for dp in dprimes:
        lbl = "choice only (d'=0)" if dp == 0 else f"choice + velocity (d'={dp})"
        ax.plot(gaps, np.array(curves[dp]) * 100, "o-", label=lbl)
    ax.axhline(50, color="0.5", ls=":", label="chance")
    ax.set_xlabel("choice separation: bias gap |alpha_Hi - alpha_Lo|")
    ax.set_ylabel("Viterbi agreement (perm-corrected, %)")
    ax.set_title("(b) separability wall: a per-state velocity marker\n"
                 "decodes states the choices cannot")
    ax.set_ylim(45, 102); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(args.outdir, "fig3_velocity_emissions.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
