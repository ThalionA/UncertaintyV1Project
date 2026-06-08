# -*- coding: utf-8 -*-
"""Generative simulation of IO-HMM sequences.

Used mainly for parameter recovery: draw a latent state path from a known
``pi`` / ``A``, draw trial conditions, then sample choices from the true
state-conditioned IO generative model (the same ``p_go`` path the emission
matrix uses). The simulator is the ground-truth counterpart to ``fit.fit``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

import emissions as emissions_mod
import states as states_mod
import io_core


def sample_state_path(pi: np.ndarray, A: np.ndarray, T: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Draw a Markov state path of length ``T`` from ``pi`` and ``A``."""
    K = len(pi)
    z = np.empty(T, dtype=np.int64)
    z[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        z[t] = rng.choice(K, p=A[z[t - 1]])
    return z


def simulate_sequence(state_list: Sequence[states_mod.IOState],
                      stage1: emissions_mod.Stage1Params,
                      grids: io_core.IOGrids,
                      pi: np.ndarray, A: np.ndarray,
                      psych_per_state: Mapping[str, Mapping[str, float]],
                      conditions: np.ndarray, T: int,
                      rng: np.random.Generator,
                      cond_probs: Optional[np.ndarray] = None,
                      vel_per_state: Optional[
                          Mapping[str, Mapping[str, float]]] = None,
                      utility: Optional[io_core.Utility] = None,
                      perc_per_state: Optional[
                          Mapping[str, Mapping[str, float]]] = None
                      ) -> tuple[emissions_mod.Trials, np.ndarray]:
    """Simulate one IO-HMM session.

    Parameters
    ----------
    conditions : (n_cond, 3) array
        Allowed ``(s_deg, c, d)`` rows; each trial draws one (uniformly by
        default, or with ``cond_probs``).
    T : int
        Number of trials.
    vel_per_state : optional
        ``{state.name: {free velocity params}}``. If given, velocity is sampled
        under the confidence model: per trial a latent measurement ``m`` is
        drawn from ``p(m|s,c,d)``, then choice ~ ``Bern(psi_k(g_k(m)))`` and
        velocity ~ ``N(beta_k*DV_k(m)+alpha_k, sigma_k)`` -- both from the *same*
        ``m``, matching the joint emission. Each state must carry a
        ``VelocitySpec``. If ``None``, choice is drawn from the marginal P(go)
        and no velocity is attached.

    Returns
    -------
    ``(trials, z_true)`` where ``trials`` is an ``emissions.Trials`` and
    ``z_true`` is the ground-truth state path (T,).
    """
    conditions = np.asarray(conditions, dtype=float)
    n_cond = conditions.shape[0]
    K = len(state_list)
    perc_per_state = perc_per_state or {}

    # Per-state IO terms over the condition rows (at each state's perception).
    g_m, dv_m, p_m = [], [], []
    for state in state_list:
        lam, prior = states_mod.perception_inputs(
            state, grids, dict(perc_per_state.get(state.name, {})))
        gm, dm, pm = emissions_mod.precompute_state_terms(
            grids, stage1, state, conditions, utility=utility,
            kappa_scale=lam, prior=prior)
        g_m.append(gm); dv_m.append(dm); p_m.append(pm)

    z = sample_state_path(np.asarray(pi, float), np.asarray(A, float), T, rng)
    cond_idx = rng.choice(n_cond, size=T, p=cond_probs)
    chosen = conditions[cond_idx]

    if vel_per_state is None:
        # Choice-only: draw from the marginal P(go) (equivalent in distribution
        # to drawing m then Bernoulli, but no shared-m velocity needed).
        p_go = np.empty((K, n_cond))
        for k, state in enumerate(state_list):
            full = state.psych.resolve(dict(psych_per_state.get(state.name, {})))
            p_go[k] = emissions_mod.p_go_from_terms(g_m[k], p_m[k], full)
        choice = (rng.random(T) < p_go[z, cond_idx]).astype(float)
        velocity = None
    else:
        # Confidence model: a shared latent m per trial drives both channels.
        n_m = grids.m_grid_deg.shape[0]
        choice = np.empty(T)
        velocity = np.empty(T)
        psych_full = [state.psych.resolve(dict(psych_per_state.get(state.name, {})))
                      for state in state_list]
        vel_full = [state.vel.resolve(dict(vel_per_state.get(state.name, {})))
                    for state in state_list]
        for t in range(T):
            k = int(z[t]); c = int(cond_idx[t])
            pm = p_m[k][c]
            m = int(rng.choice(n_m, p=pm / pm.sum()))
            psi = states_mod.psychometric(g_m[k][c, m], **psych_full[k])
            choice[t] = float(rng.random() < psi)
            vf = vel_full[k]
            mu = vf['beta_vel'] * dv_m[k][c, m] + vf['alpha_vel']
            velocity[t] = mu + vf['sigma_vel'] * rng.standard_normal()

    trials = emissions_mod.Trials(
        s_deg=chosen[:, 0].copy(),
        c=chosen[:, 1].copy(),
        d=chosen[:, 2].copy(),
        choice=choice,
        velocity=velocity,
    )
    return trials, z


__all__ = ['sample_state_path', 'simulate_sequence']
