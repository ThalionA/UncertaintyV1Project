# -*- coding: utf-8 -*-
"""Per-trial state-conditioned emission matrix for the IO-HMM.

Given trials, frozen Stage-1 sensory params, the IO-HMM states, and the
current free psychometric parameters per state, produces the (T, K) matrix
``log P(choice_t | trial_t, z_t = k)`` consumed by the HMM core.

Multi-channel emissions (v0.5)
------------------------------
Emissions factorise across channels conditional on the latent state: the joint
per-trial log-likelihood is the sum of the choice term and any additional
trial-by-trial channels. The first such channel is pre-reward-zone **velocity**,
modelled as a per-state Gaussian engagement marker ``v_t | z_t=k ~ N(mu_k,
sigma_k)`` (stimulus-independent; see ``log_velocity_matrix``). It is opt-in:
``log_emission_matrix`` adds it only when ``vel_per_state`` is supplied and
``Trials.velocity`` is present, so the choice-only path is unchanged.

Performance
-----------
Trials are grouped by unique ``(s, c, d)`` triples so the IO posterior
``P_z(s | m)`` and the marginalised ``P_z(go | s, c, d)`` are computed once
per (unique condition x state), then mapped back to per-trial entries via
an index. For typical sessions with ~30 unique conditions per mouse, this
is ~100x faster than per-trial recomputation. The velocity channel is
continuous, so it is evaluated per trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

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

    ``velocity`` is an optional continuous per-trial channel (e.g. pre-reward-
    zone running speed, ideally standardised). When present it enables the
    velocity emission; when ``None`` emissions are choice-only.
    """
    s_deg: np.ndarray
    c: np.ndarray
    d: np.ndarray
    choice: np.ndarray
    velocity: Optional[np.ndarray] = None

    def __post_init__(self):
        T = len(self.s_deg)
        for name in ('c', 'd', 'choice'):
            arr = getattr(self, name)
            if len(arr) != T:
                raise ValueError(
                    f"Trials field '{name}' has length {len(arr)} but s_deg "
                    f"has length {T}"
                )
        if self.velocity is not None and len(self.velocity) != T:
            raise ValueError(
                f"Trials field 'velocity' has length {len(self.velocity)} but "
                f"s_deg has length {T}"
            )

    @property
    def n_trials(self) -> int:
        return len(self.s_deg)

    @property
    def has_velocity(self) -> bool:
        return self.velocity is not None


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
                           state: states_mod.IOState, G_unique: np.ndarray,
                           utility: io_core.Utility | None = None,
                           kappa_scale: float = 1.0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache the (psych/velocity-independent) IO terms for one state.

    For each unique ``(s, c, d)`` row, returns the IO log-posterior-odds
    ``g(m)`` (choice), the utility-weighted decision variable ``DV(m)``
    (velocity), and the generative ``p(m | s)`` on the measurement grid. None
    depend on the free psychometric/velocity params, so the EM loop computes
    them once per (frozen-perception) restart and reuses them.

    ``kappa_scale`` (the per-state sensory-precision multiplier ``lambda_k``)
    multiplies the trial kappa for *both* the generative and the inference, so
    a state can have sharper/blurrier perception. With ``kappa_scale = 1`` and
    a fixed prior this is frozen; Phase 2 makes it free.

    Returns ``(g_m, dv_m, p_m)`` each of shape ``(n_conds, n_m)``.
    """
    n_conds = G_unique.shape[0]
    n_m = grids.m_grid_deg.shape[0]
    g_m = np.empty((n_conds, n_m))
    dv_m = np.empty((n_conds, n_m))
    p_m = np.empty((n_conds, n_m))
    for j in range(n_conds):
        s_j, c_j, d_j = G_unique[j]
        kappa = kappa_scale * float(io_core.kappa_for_trial(
            np.array([s_j]), np.array([c_j]), np.array([d_j]),
            kappa_amp=stage1.kappa_amp,
            c_power=stage1.c_power,
            d_power=stage1.d_power,
            kappa_min=stage1.kappa_min,
        )[0])
        post = io_core.posterior_s_given_m(grids, kappa=kappa, prior=state.prior)
        g_m[j] = io_core.log_posterior_odds(grids, post)        # (n_m,)
        dv_m[j] = io_core.decision_variable(grids, post, utility)  # (n_m,)
        p_m[j] = io_core.p_m_given_s(grids, kappa=kappa, s_deg=float(s_j))
    return g_m, dv_m, p_m


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
    g_m, _dv_m, p_m = precompute_state_terms(grids, stage1, state, G_unique)
    return p_go_from_terms(g_m, p_m, full)


# ---------------------------------------------------------------------------
# Per-trial log emission matrix
# ---------------------------------------------------------------------------


def joint_log_emission_state(g_m: np.ndarray, dv_m: np.ndarray, p_m: np.ndarray,
                             gidx: np.ndarray, choice: np.ndarray,
                             psych_full: Mapping[str, float],
                             velocity: Optional[np.ndarray] = None,
                             vel_full: Optional[Mapping[str, float]] = None
                             ) -> np.ndarray:
    """Per-trial ``log p(choice_t, velocity_t | cond_t, state)``. Shape ``(T,)``.

    Faithful joint emission marginalised over the latent measurement ``m``
    (paper Eqs. 11, 20). Cached per-condition arrays ``g_m``/``dv_m``/``p_m``
    (each ``(n_cond, n_m)``) are gathered to trials via ``gidx``.

    Choice-only (``velocity``/``vel_full`` is ``None``):
        ``P(go|cond) = sum_m psi(g(m)) p(m|cond)``; Bernoulli on choice. This is
        byte-for-byte the v0 choice emission.

    With velocity, choice and velocity are coupled through the shared ``m``:
        ``Z          = sum_m N(v; beta*DV(m)+alpha, sigma) p(m|cond)``     [vel marginal]
        ``post(m|v)  = N(...) p(m|cond) / Z``                              [vel-updated posterior]
        ``P(go|v)    = sum_m psi(g(m)) post(m|v)``
        ``log p      = log Z + choice*log P(go|v) + (1-choice)*log(1-P(go|v))``
    so velocity updates the choice via Bayes on ``m`` -- exactly the v2
    "metacognitive" coupling, and never as a direct linear predictor of choice.
    """
    G = g_m[gidx]            # (T, n_m)
    PM = p_m[gidx]           # (T, n_m)
    psi = states_mod.psychometric(
        G, alpha=psych_full['alpha'], beta=psych_full['beta'],
        gamma=psych_full['gamma'], delta=psych_full['delta'])  # (T, n_m)

    if velocity is None or vel_full is None:
        p_go = np.clip(np.sum(psi * PM, axis=1), EPS, 1.0 - EPS)
        return choice * np.log(p_go) + (1.0 - choice) * np.log(1.0 - p_go)

    DV = dv_m[gidx]          # (T, n_m)
    beta = float(vel_full['beta_vel'])
    alpha = float(vel_full['alpha_vel'])
    sigma = max(float(vel_full['sigma_vel']), EPS)
    v = np.asarray(velocity, dtype=float)[:, None]
    mu = beta * DV + alpha   # (T, n_m)
    log_N = -0.5 * np.log(2.0 * np.pi * sigma * sigma) - 0.5 * ((v - mu) / sigma) ** 2
    log_w = np.log(PM + EPS) + log_N                  # (T, n_m)
    log_Z = logsumexp(log_w, axis=1)                  # (T,)  velocity marginal log-lik
    post = np.exp(log_w - log_Z[:, None])             # (T, n_m) vel-updated posterior
    p_go = np.clip(np.sum(psi * post, axis=1), EPS, 1.0 - EPS)
    return log_Z + choice * np.log(p_go) + (1.0 - choice) * np.log(1.0 - p_go)


def log_emission_matrix(grids: io_core.IOGrids, stage1: Stage1Params,
                        state_list: Sequence[states_mod.IOState],
                        psych_per_state: Mapping[str, Mapping[str, float]],
                        trials: Trials,
                        vel_per_state: Optional[
                            Mapping[str, Mapping[str, float]]] = None,
                        utility: io_core.Utility | None = None) -> np.ndarray:
    """``log P(obs_t | trial_t, z_t = k)``. Shape ``(T, K)``.

    ``psych_per_state`` / ``vel_per_state`` are ``{state.name: {free_param:
    value}}``; the fixed entries come from the state's ``PsychSpec`` /
    ``VelocitySpec``. If ``vel_per_state`` is given, the joint
    choice+velocity emission is used (requires ``trials.velocity`` and each
    state to carry a ``VelocitySpec``); otherwise emissions are choice-only and
    identical to the v0 path.
    """
    G_unique, G_idx = unique_conditions(trials)
    T = trials.n_trials
    K = len(state_list)
    use_vel = vel_per_state is not None
    if use_vel and trials.velocity is None:
        raise ValueError("vel_per_state supplied but trials.velocity is None")

    log_emiss = np.empty((T, K))
    for k, state in enumerate(state_list):
        g_m, dv_m, p_m = precompute_state_terms(
            grids, stage1, state, G_unique, utility=utility)
        psych_full = state.psych.resolve(dict(psych_per_state.get(state.name, {})))
        if use_vel:
            if state.vel is None:
                raise ValueError(f"state '{state.name}' has no VelocitySpec")
            vel_full = state.vel.resolve(dict(vel_per_state.get(state.name, {})))
            log_emiss[:, k] = joint_log_emission_state(
                g_m, dv_m, p_m, G_idx, trials.choice, psych_full,
                velocity=trials.velocity, vel_full=vel_full)
        else:
            log_emiss[:, k] = joint_log_emission_state(
                g_m, dv_m, p_m, G_idx, trials.choice, psych_full)
    return log_emiss


__all__ = [
    'Stage1Params',
    'Trials',
    'unique_conditions',
    'precompute_state_terms',
    'p_go_from_terms',
    'p_go_per_unique_condition',
    'joint_log_emission_state',
    'log_emission_matrix',
]
