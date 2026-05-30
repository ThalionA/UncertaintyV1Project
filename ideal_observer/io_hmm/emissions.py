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


def precompute_state_terms(grids: io_core.IOGrids, stage1: Stage1Params,
                           state: states_mod.IOState,
                           G_unique: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cache the psych-independent emission terms for one state.

    For each unique ``(s, c, d)`` row, returns the IO log-posterior-odds
    ``g(m)`` and the generative ``p(m | s)`` on the measurement grid. Neither
    depends on the free psychometric params, so the EM loop can compute these
    once and reuse them across every iteration and restart.

    Returns ``(g_m, p_m)`` each of shape ``(n_conds, n_m)``.
    """
    n_conds = G_unique.shape[0]
    n_m = grids.m_grid_deg.shape[0]
    g_m = np.empty((n_conds, n_m))
    p_m = np.empty((n_conds, n_m))
    for j in range(n_conds):
        s_j, c_j, d_j = G_unique[j]
        kappa = float(io_core.kappa_for_trial(
            np.array([s_j]), np.array([c_j]), np.array([d_j]),
            kappa_amp=stage1.kappa_amp,
            c_power=stage1.c_power,
            d_power=stage1.d_power,
            kappa_min=stage1.kappa_min,
        )[0])
        post = io_core.posterior_s_given_m(grids, kappa=kappa, prior=state.prior)
        g_m[j] = io_core.log_posterior_odds(grids, post)  # (n_m,)
        p_m[j] = io_core.p_m_given_s(grids, kappa=kappa, s_deg=float(s_j))
    return g_m, p_m


def p_go_from_terms(g_m: np.ndarray, p_m: np.ndarray,
                    psych_full: Mapping[str, float]) -> np.ndarray:
    """``P(go | cond, state)`` from cached ``g(m)`` / ``p(m)`` and a *full*
    psych dict ``{alpha, beta, gamma, delta}``. Shape ``(n_conds,)``, clipped.
    """
    p_go_m = states_mod.psychometric(
        g_m, alpha=psych_full['alpha'], beta=psych_full['beta'],
        gamma=psych_full['gamma'], delta=psych_full['delta'],
    )  # (n_conds, n_m)
    p_go = np.sum(p_go_m * p_m, axis=1)
    return np.clip(p_go, EPS, 1.0 - EPS)


def p_go_per_unique_condition(grids: io_core.IOGrids, stage1: Stage1Params,
                              state: states_mod.IOState,
                              psych_values: Mapping[str, float],
                              G_unique: np.ndarray) -> np.ndarray:
    """Marginal ``P(go | s, c, d, state)`` for each unique condition row.

    For each unique ``(s, c, d)``: form the IO posterior under ``state.prior``,
    read off ``g(m) = log P(Go|m)/P(NoGo|m)``, push it through the state's
    resolved psychometric, and marginalise over the generative ``p(m | s)``:
    ``P(go | cond) = sum_m psi(g(m)) * p(m | s)``. Returns ``(n_conds,)``
    clipped into ``(EPS, 1 - EPS)``.

    Thin wrapper over ``precompute_state_terms`` + ``p_go_from_terms`` so the
    EM-cached fast path and this reference path stay numerically identical.
    """
    full = state.psych.resolve(dict(psych_values))
    g_m, p_m = precompute_state_terms(grids, stage1, state, G_unique)
    return p_go_from_terms(g_m, p_m, full)


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
    'precompute_state_terms',
    'p_go_from_terms',
    'p_go_per_unique_condition',
    'log_emission_matrix',
]
