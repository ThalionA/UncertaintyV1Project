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
                      cond_probs: Optional[np.ndarray] = None
                      ) -> tuple[emissions_mod.Trials, np.ndarray]:
    """Simulate one IO-HMM session.

    Parameters
    ----------
    conditions : (n_cond, 3) array
        Allowed ``(s_deg, c, d)`` rows; each trial draws one (uniformly by
        default, or with ``cond_probs``).
    T : int
        Number of trials.

    Returns
    -------
    ``(trials, z_true)`` where ``trials`` is an ``emissions.Trials`` and
    ``z_true`` is the ground-truth state path (T,).
    """
    conditions = np.asarray(conditions, dtype=float)
    n_cond = conditions.shape[0]
    K = len(state_list)

    # True P(go | state, condition) for every (state, unique condition).
    p_go = np.empty((K, n_cond))
    for k, state in enumerate(state_list):
        p_go[k] = emissions_mod.p_go_per_unique_condition(
            grids, stage1, state, dict(psych_per_state.get(state.name, {})),
            conditions
        )

    z = sample_state_path(np.asarray(pi, float), np.asarray(A, float), T, rng)
    cond_idx = rng.choice(n_cond, size=T, p=cond_probs)
    p_trial = p_go[z, cond_idx]
    choice = (rng.random(T) < p_trial).astype(float)

    chosen = conditions[cond_idx]
    trials = emissions_mod.Trials(
        s_deg=chosen[:, 0].copy(),
        c=chosen[:, 1].copy(),
        d=chosen[:, 2].copy(),
        choice=choice,
    )
    return trials, z


__all__ = ['sample_state_path', 'simulate_sequence']
