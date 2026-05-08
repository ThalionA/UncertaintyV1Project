# -*- coding: utf-8 -*-
"""Per-trial state-conditioned emission matrix for the IO-HMM.

Given trials, frozen Stage-1 sensory params, the IO-HMM states, and the
current free psychometric parameters per state, produces the (T, K) matrix
``log P(choice_t | trial_t, z_t = k)`` consumed by the HMM core.

Performance
-----------
Trials are grouped by unique ``(s, c, d)`` triples so the IO posterior
``P_z(s | m)`` and the marginalised ``P_z(go | s, c, d)`` are computed once
per (unique condition x state), then mapped back to per-trial entries via
an index. For typical sessions with ~30 unique conditions per mouse, this
is ~100x faster than per-trial recomputation. v0 emission is choice-only;
velocity is left to a future v0.5 once recovery on choices is solid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

import io_core
import states as states_mod

EPS = 1e-10


# ---------------------------------------------------------------------------
# Frozen Stage-1 inputs and per-session trial container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage1Params:
    """Frozen sensory params from v2 Stage 1.

    v0 emission is choice-only and only needs the kappa schedule. Velocity
    emission params live in Stage 1 too, but are unused here.
    """
    kappa_amp: float
    c_power: float
    d_power: float
    kappa_min: float = 1.0


@dataclass(frozen=True, eq=False)
class Trials:
    """Per-trial features and choices, all shape (T,).

    ``choice`` is 1 for Go (lick), 0 for NoGo. Floats are accepted (e.g.
    soft labels) but only the Bernoulli interpretation is used here.
    """
    s_deg: np.ndarray
    c: np.ndarray
    d: np.ndarray
    choice: np.ndarray

    def __post_init__(self):
        T = len(self.s_deg)
        for name in ('c', 'd', 'choice'):
            arr = getattr(self, name)
            if len(arr) != T:
                raise ValueError(
                    f"Trials field '{name}' has length {len(arr)} but s_deg "
                    f"has length {T}"
                )

    @property
    def n_trials(self) -> int:
        return len(self.s_deg)


# ---------------------------------------------------------------------------
# Unique-condition grouping
# ---------------------------------------------------------------------------


def unique_conditions(trials: Trials) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(G_unique, G_idx)``.

    ``G_unique`` has shape ``(n_conds, 3)`` with rows ``(s, c, d)``.
    ``G_idx`` has length ``T`` and maps each trial to its row in
    ``G_unique``. Ordering follows ``np.unique`` (lexicographic).
    """
    cond_mat = np.column_stack([trials.s_deg, trials.c, trials.d]).astype(float)
    G_unique, G_idx = np.unique(cond_mat, axis=0, return_inverse=True)
    return G_unique, G_idx


# ---------------------------------------------------------------------------
# Per-state P(go) over unique conditions
# ---------------------------------------------------------------------------


def p_go_per_unique_condition(grids: io_core.IOGrids, stage1: Stage1Params,
                              state: states_mod.IOState,
                              psych_values: Mapping[str, float],
                              G_unique: np.ndarray) -> np.ndarray:
    """Marginal ``P(go | s, c, d, state)`` for each unique condition row.

    Steps for each unique ``(s, c, d)``:

      1. ``kappa = kappa_for_trial(c, d; stage1)`` (orientation-independent).
      2. ``post = posterior_s_given_m(grids, kappa, state.prior)``.
      3. ``g_m  = log_posterior_odds(grids, post)``.
      4. ``p_go_m = psychometric(g_m; resolved psych params)``.
      5. ``p_m   = p_m_given_s(grids, kappa, s)``.
      6. ``P(go | cond, state) = sum_m p_go_m * p_m``.

    Returns shape ``(n_conds,)``, clipped into ``(EPS, 1 - EPS)`` for
    downstream log safety.
    """
    n_conds = G_unique.shape[0]
    p_go = np.empty(n_conds)

    full = state.psych.resolve(dict(psych_values))
    alpha, beta = full['alpha'], full['beta']
    gamma, delta = full['gamma'], full['delta']

    for j in range(n_conds):
        s_j, c_j, d_j = G_unique[j]
        kappa = float(io_core.kappa_for_trial(
            np.array([s_j]), np.array([c_j]), np.array([d_j]),
            kappa_amp=stage1.kappa_amp,
            c_power=stage1.c_power,
            d_power=stage1.d_power,
            kappa_min=stage1.kappa_min,
        ))
        post = io_core.posterior_s_given_m(grids, kappa=kappa, prior=state.prior)
        g_m = io_core.log_posterior_odds(grids, post)  # (n_m,)
        p_go_m = states_mod.psychometric(g_m, alpha=alpha, beta=beta,
                                         gamma=gamma, delta=delta)  # (n_m,)
        p_m = io_core.p_m_given_s(grids, kappa=kappa, s_deg=float(s_j))
        p_go[j] = float(np.sum(p_go_m * p_m))

    return np.clip(p_go, EPS, 1.0 - EPS)


# ---------------------------------------------------------------------------
# Per-trial log emission matrix
# ---------------------------------------------------------------------------


def log_emission_matrix(grids: io_core.IOGrids, stage1: Stage1Params,
                        state_list: Sequence[states_mod.IOState],
                        psych_per_state: Mapping[str, Mapping[str, float]],
                        trials: Trials) -> np.ndarray:
    """``log P(choice_t | trial_t, z_t = k)``. Shape ``(T, K)``.

    ``psych_per_state`` is ``{state.name: {free_param_name: value}}``. For
    states with no free params the entry can be ``{}`` or absent.
    """
    G_unique, G_idx = unique_conditions(trials)
    T = trials.n_trials
    K = len(state_list)

    log_emiss = np.empty((T, K))
    for k, state in enumerate(state_list):
        psych_vals = psych_per_state.get(state.name, {})
        p_go_unique = p_go_per_unique_condition(
            grids, stage1, state, psych_vals, G_unique
        )
        p_go_t = p_go_unique[G_idx]  # broadcast back to per trial
        # Bernoulli log-likelihood. choice in {0, 1}; floats also fine.
        log_emiss[:, k] = (trials.choice * np.log(p_go_t)
                           + (1.0 - trials.choice) * np.log(1.0 - p_go_t))
    return log_emiss


__all__ = [
    'Stage1Params',
    'Trials',
    'unique_conditions',
    'p_go_per_unique_condition',
    'log_emission_matrix',
]
