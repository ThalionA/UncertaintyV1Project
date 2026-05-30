# -*- coding: utf-8 -*-
"""EM (Baum-Welch) fitting for the IO-HMM.

Fits the latent-state dynamics and the free per-state psychometric entries
while keeping Stage-1 sensory params, the state priors, and the fixed
psychometric entries frozen. Concretely, EM updates:

  - ``pi`` : initial state distribution (K,)
  - ``A``  : transition matrix (K, K)
  - the *free* entries of each state's 4-param psychometric on g(m)

given one or more observed choice sequences (sessions). All three updates
share the same E-step posteriors.

E-step
------
For each sequence, build ``log_emiss`` and run the HMM core
(``hmm.log_forward`` / ``hmm.log_backward``) to obtain ``gamma`` (state
posteriors), ``xi`` (transition posteriors), and the sequence log-likelihood.

M-step
------
- ``pi`` and ``A`` have the standard closed-form Baum-Welch updates,
  accumulated across sequences.
- The free psychometric entries are fit by maximising the gamma-weighted
  expected complete-data log-likelihood. This objective decomposes over
  states (state k's emission depends only on state k's free params), so each
  state is optimised independently with ``scipy.optimize.minimize`` on a
  per-unique-condition aggregated objective.

Performance
-----------
``g(m)`` and ``p(m | s)`` do not depend on the free psychometric params, so
they are precomputed once per (state x unique condition) via
``emissions.precompute_state_terms`` and reused across all EM iterations and
restarts. Each emission rebuild and each optimizer evaluation then only
applies the cheap psychometric to cached arrays.

Because the v0 states are not exchangeable (each declares a distinct prior /
fixed-param pattern), there is no label-switching: a fitted ``IOHMMParams``
maps back to truth directly by state name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

import hmm
import emissions as emissions_mod
import states as states_mod
import io_core

NEG_INF = -np.inf
_EPS = 1e-12

# Bounds for the psychometric M-step. ``beta >= 0`` enforces a monotone-increasing
# psychometric in g(m); the finite caps keep the optimizer out of the
# quasi-separation regime (where the slope MLE diverges to +inf on
# near-separable gamma-weighted data) and avoid overflow.
_PSYCH_BOUNDS = {
    'alpha': (-15.0, 15.0),
    'beta': (0.0, 15.0),
    'gamma': (0.0, 0.49),
    'delta': (0.0, 0.49),
}


# ---------------------------------------------------------------------------
# Fitted-params container
# ---------------------------------------------------------------------------


@dataclass
class IOHMMParams:
    """Free IO-HMM parameters: initial dist, transitions, per-state psych.

    ``psych_per_state`` maps ``state.name -> {free_param_name: value}`` and
    contains only the *free* entries (fixed entries live in the state's
    ``PsychSpec``). For a state with no free params the entry may be ``{}``.

    ``vel_per_state`` maps ``state.name -> {'mu': float, 'sigma': float}`` for
    the optional Gaussian velocity channel. It is empty for choice-only fits.
    """
    pi: np.ndarray            # (K,)
    A: np.ndarray             # (K, K)
    psych_per_state: dict[str, dict[str, float]] = field(default_factory=dict)
    vel_per_state: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def log_pi(self) -> np.ndarray:
        with np.errstate(divide='ignore'):
            return np.log(self.pi)

    @property
    def log_A(self) -> np.ndarray:
        with np.errstate(divide='ignore'):
            return np.log(self.A)

    def copy(self) -> "IOHMMParams":
        return IOHMMParams(
            pi=np.array(self.pi, dtype=float),
            A=np.array(self.A, dtype=float),
            psych_per_state={k: dict(v) for k, v in self.psych_per_state.items()},
            vel_per_state={k: dict(v) for k, v in self.vel_per_state.items()},
        )


# ---------------------------------------------------------------------------
# Cached, frozen emission terms
# ---------------------------------------------------------------------------


@dataclass
class _Cache:
    """Psych-independent emission terms + per-sequence trial indexing."""
    g_m: list[np.ndarray]       # per state, (n_cond, n_m)
    p_m: np.ndarray             # (n_cond, n_m); shared across states
    gidx: list[np.ndarray]      # per sequence, (T,) -> row of the global conds
    choice: list[np.ndarray]    # per sequence, (T,)
    n_cond: int
    velocity: Optional[list[np.ndarray]] = None  # per sequence, (T,) or None


def _build_cache(trials_list, state_list, stage1, grids,
                 use_velocity: bool = False) -> _Cache:
    G_all, g_idx_global = _build_global_conditions(trials_list)
    p_m = None
    g_m = []
    for state in state_list:
        g_m_k, p_m_k = emissions_mod.precompute_state_terms(
            grids, stage1, state, G_all
        )
        g_m.append(g_m_k)
        if p_m is None:
            p_m = p_m_k  # p(m | s) is independent of the state's prior
    velocity = None
    if use_velocity:
        if any(t.velocity is None for t in trials_list):
            raise ValueError("use_velocity=True but some Trials lack velocity")
        velocity = [np.asarray(t.velocity, float) for t in trials_list]
    return _Cache(
        g_m=g_m, p_m=p_m, gidx=g_idx_global,
        choice=[t.choice for t in trials_list], n_cond=G_all.shape[0],
        velocity=velocity,
    )


def _p_go_cond(cache: _Cache, state_list, params: IOHMMParams) -> list[np.ndarray]:
    """P(go) per global unique condition, per state, under current psych."""
    out = []
    for k, state in enumerate(state_list):
        full = state.psych.resolve(params.psych_per_state.get(state.name, {}))
        out.append(emissions_mod.p_go_from_terms(cache.g_m[k], cache.p_m, full))
    return out


def _log_emiss_cached(cache: _Cache, seq_i: int,
                      p_go_cond: list[np.ndarray],
                      vel_per_state: Optional[Sequence[tuple[float, float]]] = None
                      ) -> np.ndarray:
    """(T, K) log-emission for one sequence from cached P(go), plus the
    Gaussian velocity channel when ``vel_per_state`` (per-state ``(mu, sigma)``)
    is supplied and the cache holds velocities."""
    gidx = cache.gidx[seq_i]
    ch = cache.choice[seq_i]
    K = len(p_go_cond)
    log_emiss = np.empty((len(ch), K))
    for k in range(K):
        p = p_go_cond[k][gidx]
        log_emiss[:, k] = ch * np.log(p) + (1.0 - ch) * np.log(1.0 - p)
    if vel_per_state is not None and cache.velocity is not None:
        v = cache.velocity[seq_i]
        for k in range(K):
            mu, sigma = vel_per_state[k]
            sigma = max(sigma, _EPS)
            log_emiss[:, k] += (-0.5 * np.log(2.0 * np.pi * sigma * sigma)
                                - 0.5 * ((v - mu) / sigma) ** 2)
    return log_emiss


# ---------------------------------------------------------------------------
# E-step (public reference path + cached internal path)
# ---------------------------------------------------------------------------


def e_step(params: IOHMMParams,
           state_list: Sequence[states_mod.IOState],
           stage1: emissions_mod.Stage1Params,
           grids: io_core.IOGrids,
           trials: emissions_mod.Trials,
           log_emiss: Optional[np.ndarray] = None
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Posteriors for one sequence (reference path via ``log_emission_matrix``).

    Returns ``(gamma, xi, log_lik)`` with shapes ``(T, K)``, ``(T-1, K, K)``
    and a scalar. ``log_emiss`` may be passed in if already computed.
    """
    if log_emiss is None:
        log_emiss = emissions_mod.log_emission_matrix(
            grids, stage1, state_list, params.psych_per_state, trials
        )
    return _posteriors_from_log_emiss(params, log_emiss)


def _posteriors_from_log_emiss(params: IOHMMParams, log_emiss: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, float]:
    log_pi, log_A = params.log_pi, params.log_A
    log_alpha, log_lik = hmm.log_forward(log_pi, log_A, log_emiss)
    log_beta = hmm.log_backward(log_A, log_emiss)
    gamma = hmm.posterior_states(log_alpha, log_beta)
    if log_emiss.shape[0] > 1:
        xi = hmm.posterior_transitions(log_alpha, log_beta, log_A, log_emiss)
    else:
        xi = np.zeros((0, log_emiss.shape[1], log_emiss.shape[1]))
    return gamma, xi, log_lik


# ---------------------------------------------------------------------------
# M-step: initial distribution and transitions (closed form)
# ---------------------------------------------------------------------------


def m_step_transitions(gammas: Sequence[np.ndarray],
                       xis: Sequence[np.ndarray],
                       K: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Baum-Welch updates for ``pi`` and ``A`` over sequences.

    ``pi`` is the mean of the first-step posteriors across sequences. ``A``
    is ``sum_seq sum_t xi[t,i,j]`` normalised by ``sum_seq sum_t gamma[:-1,i]``.
    Rows with zero expected outgoing mass fall back to uniform.
    """
    pi = np.zeros(K)
    for gamma in gammas:
        pi += gamma[0]
    pi /= len(gammas)
    pi = pi / (pi.sum() + _EPS)

    num = np.zeros((K, K))
    den = np.zeros(K)
    for gamma, xi in zip(gammas, xis):
        if xi.shape[0] > 0:
            num += xi.sum(axis=0)
            den += gamma[:-1].sum(axis=0)
    A = np.empty((K, K))
    for i in range(K):
        if den[i] > _EPS:
            A[i] = num[i] / den[i]
        else:
            A[i] = 1.0 / K
    A = A / A.sum(axis=1, keepdims=True)  # guard tiny normalisation drift
    return pi, A


# ---------------------------------------------------------------------------
# M-step: per-state psychometric (numerical, cached)
# ---------------------------------------------------------------------------


def m_step_velocity(gammas: Sequence[np.ndarray],
                    velocities: Sequence[np.ndarray],
                    state_list: Sequence[states_mod.IOState],
                    sigma_floor: float = 1e-2
                    ) -> dict[str, dict[str, float]]:
    """Closed-form weighted-Gaussian update for the velocity channel.

    For each state ``k``, ``mu_k`` and ``sigma_k`` are the gamma-weighted mean
    and standard deviation of the velocities, pooled across sequences -- the
    exact maximiser of the expected complete-data Gaussian log-likelihood, so
    EM stays monotone. States with negligible responsibility fall back to the
    global mean / std. ``sigma`` is floored to avoid variance collapse.
    """
    K = len(state_list)
    v_all = np.concatenate([np.asarray(v, float) for v in velocities])
    g_mean = float(v_all.mean())
    g_std = max(float(v_all.std()), sigma_floor)

    sw = np.zeros(K)
    swx = np.zeros(K)
    for gamma, v in zip(gammas, velocities):
        v = np.asarray(v, float)
        sw += gamma.sum(axis=0)
        swx += gamma.T @ v
    mu = np.where(sw > _EPS, swx / np.maximum(sw, _EPS), g_mean)

    swx2 = np.zeros(K)
    for gamma, v in zip(gammas, velocities):
        v = np.asarray(v, float)
        swx2 += (gamma * (v[:, None] - mu[None, :]) ** 2).sum(axis=0)
    var = np.where(sw > _EPS, swx2 / np.maximum(sw, _EPS), g_std ** 2)
    sigma = np.maximum(np.sqrt(var), sigma_floor)

    return {state.name: {'mu': float(mu[k]), 'sigma': float(sigma[k])}
            for k, state in enumerate(state_list)}


def _weighted_go_counts(gammas, choices, gidx_list, n_cond: int, k: int):
    """Gamma-weighted go / nogo mass per global unique condition for state k."""
    w_go = np.zeros(n_cond)
    w_nogo = np.zeros(n_cond)
    for gamma, choice, gidx in zip(gammas, choices, gidx_list):
        wk = gamma[:, k]
        np.add.at(w_go, gidx, wk * choice)
        np.add.at(w_nogo, gidx, wk * (1.0 - choice))
    return w_go, w_nogo


def _fit_psych_state(state: states_mod.IOState, current,
                     g_m_k: np.ndarray, p_m: np.ndarray,
                     w_go: np.ndarray, w_nogo: np.ndarray) -> dict[str, float]:
    """Maximise the gamma-weighted complete-data LL over a state's free params.

    Cached: each objective evaluation only applies the psychometric to the
    precomputed ``g_m_k`` and dots with ``p_m``. Warm-started from ``current``
    and guarded so the returned params never raise the objective above its
    warm-start value (protects EM monotonicity).
    """
    free_names = state.psych.free_params
    if not free_names:
        return {}
    x0 = np.array([float(current[name]) for name in free_names], dtype=float)
    bounds = [_PSYCH_BOUNDS[name] for name in free_names]
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    def obj(theta):
        free_vals = {name: float(theta[i]) for i, name in enumerate(free_names)}
        full = state.psych.resolve(free_vals)
        p_go = emissions_mod.p_go_from_terms(g_m_k, p_m, full)
        return float(-(w_go * np.log(p_go) + w_nogo * np.log(1.0 - p_go)).sum())

    f0 = obj(x0)
    res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)
    x_best = res.x if (np.isfinite(res.fun) and res.fun <= f0) else x0
    return {name: float(x_best[i]) for i, name in enumerate(free_names)}


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------


def random_init(state_list: Sequence[states_mod.IOState], rng: np.random.Generator,
                self_bias: float = 4.0) -> IOHMMParams:
    """Random starting point: Dirichlet pi/A (diagonal-biased) + random psych."""
    K = len(state_list)
    pi = rng.dirichlet(np.ones(K))
    A = np.empty((K, K))
    for i in range(K):
        conc = np.ones(K)
        conc[i] += self_bias  # gentle stickiness prior on the diagonal
        A[i] = rng.dirichlet(conc)
    psych_per_state: dict[str, dict[str, float]] = {}
    for state in state_list:
        vals: dict[str, float] = {}
        for name in state.psych.free_params:
            if name == 'beta':
                vals[name] = float(rng.uniform(0.5, 3.0))   # increasing psychometric
            elif name == 'alpha':
                vals[name] = float(rng.normal(0.0, 0.5))    # small bias
            else:  # gamma / delta lapse-like, kept small and positive
                vals[name] = float(rng.uniform(0.0, 0.1))
        psych_per_state[state.name] = vals
    return IOHMMParams(pi=pi, A=A, psych_per_state=psych_per_state)


def _random_vel_init(velocity_all: np.ndarray,
                     state_list: Sequence[states_mod.IOState],
                     rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Seed per-state velocity ``(mu, sigma)`` from the empirical distribution.

    Means are drawn at jittered empirical quantiles so the states start
    behaviourally distinct (breaking the symmetry that would make the velocity
    channel uninformative); sigma starts at the global std.
    """
    K = len(state_list)
    g_mean = float(velocity_all.mean())
    g_std = max(float(velocity_all.std()), 1e-2)
    qs = (np.arange(K) + 1.0) / (K + 1.0)
    centres = np.quantile(velocity_all, qs)
    jitter = rng.normal(0.0, 0.25 * g_std, size=K)
    mus = centres + jitter
    return {state.name: {'mu': float(mus[k]), 'sigma': float(g_std)}
            for k, state in enumerate(state_list)}


# ---------------------------------------------------------------------------
# Single EM run
# ---------------------------------------------------------------------------


def _run_em(init: IOHMMParams, cache: _Cache,
            state_list: Sequence[states_mod.IOState],
            max_iters: int, tol: float) -> tuple[IOHMMParams, list[float]]:
    """One EM run from a fixed init. Returns fitted params + log-lik history."""
    K = len(state_list)
    n_seq = len(cache.gidx)
    params = init.copy()
    history: list[float] = []
    prev_ll = NEG_INF

    use_velocity = cache.velocity is not None

    for _ in range(max_iters):
        # --- E-step over all sequences (cached emissions) ---
        p_go_cond = _p_go_cond(cache, state_list, params)
        vel_kvals = ([(params.vel_per_state[s.name]['mu'],
                       params.vel_per_state[s.name]['sigma'])
                      for s in state_list] if use_velocity else None)
        gammas, xis = [], []
        total_ll = 0.0
        for i in range(n_seq):
            log_emiss = _log_emiss_cached(cache, i, p_go_cond, vel_kvals)
            gamma, xi, ll = _posteriors_from_log_emiss(params, log_emiss)
            gammas.append(gamma)
            xis.append(xi)
            total_ll += ll
        history.append(total_ll)

        # --- M-step: transitions / initial ---
        pi, A = m_step_transitions(gammas, xis, K)

        # --- M-step: per-state psychometric ---
        psych_per_state: dict[str, dict[str, float]] = {}
        for k, state in enumerate(state_list):
            w_go, w_nogo = _weighted_go_counts(
                gammas, cache.choice, cache.gidx, cache.n_cond, k
            )
            psych_per_state[state.name] = _fit_psych_state(
                state, params.psych_per_state.get(state.name, {}),
                cache.g_m[k], cache.p_m, w_go, w_nogo
            )

        # --- M-step: per-state velocity (closed form) ---
        vel_per_state = (m_step_velocity(gammas, cache.velocity, state_list)
                         if use_velocity else {})

        params = IOHMMParams(pi=pi, A=A, psych_per_state=psych_per_state,
                             vel_per_state=vel_per_state)

        if len(history) > 1 and total_ll - prev_ll < tol:
            break
        prev_ll = total_ll

    return params, history


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_global_conditions(trials_list: Sequence[emissions_mod.Trials]
                             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Union of unique (s, c, d) across sequences + per-sequence trial index."""
    all_conds = np.column_stack([
        np.concatenate([t.s_deg for t in trials_list]),
        np.concatenate([t.c for t in trials_list]),
        np.concatenate([t.d for t in trials_list]),
    ]).astype(float)
    G_all = np.unique(all_conds, axis=0)
    key_to_row = {tuple(row): i for i, row in enumerate(G_all)}
    g_idx_global = []
    for t in trials_list:
        cond_t = np.column_stack([t.s_deg, t.c, t.d]).astype(float)
        idx = np.array([key_to_row[tuple(r)] for r in cond_t], dtype=np.int64)
        g_idx_global.append(idx)
    return G_all, g_idx_global


def fit(trials_list: Sequence[emissions_mod.Trials],
        state_list: Sequence[states_mod.IOState],
        stage1: emissions_mod.Stage1Params,
        grids: io_core.IOGrids,
        *,
        init: Optional[IOHMMParams] = None,
        max_iters: int = 100,
        tol: float = 1e-5,
        n_restarts: int = 5,
        seed: int = 0,
        use_velocity: bool = False,
        return_all: bool = False):
    """Fit the IO-HMM by EM with random restarts.

    Parameters
    ----------
    trials_list : sequence of ``Trials`` (or a single ``Trials``)
        One entry per session; all share the fitted parameters.
    state_list, stage1, grids :
        Frozen model structure (state priors / fixed psych, sensory schedule,
        discretisation grids).
    init :
        Optional fixed starting point used as the first restart; the remaining
        restarts are random.
    max_iters, tol :
        EM stopping controls (total log-lik improvement threshold).
    n_restarts :
        Number of starts (EM is non-convex); the best final log-lik wins.
    seed :
        Seed for the restart RNG.
    use_velocity :
        If True, add the per-state Gaussian velocity channel to the emission
        (requires every ``Trials`` to carry ``velocity``). The fitted
        ``vel_per_state`` is populated; the choice-only path is unchanged.

    Returns
    -------
    ``(best_params, best_history)`` or, if ``return_all``, also a list of
    ``(params, history)`` for every restart.
    """
    if isinstance(trials_list, emissions_mod.Trials):
        trials_list = [trials_list]
    rng = np.random.default_rng(seed)
    cache = _build_cache(trials_list, state_list, stage1, grids,
                         use_velocity=use_velocity)

    inits: list[IOHMMParams] = []
    if init is not None:
        inits.append(init.copy())
    while len(inits) < max(1, n_restarts):
        inits.append(random_init(state_list, rng))
    if use_velocity:
        v_all = np.concatenate(cache.velocity)
        for p in inits:
            if not p.vel_per_state:
                p.vel_per_state = _random_vel_init(v_all, state_list, rng)

    results = [_run_em(start, cache, state_list, max_iters, tol)
               for start in inits]
    best_idx = int(np.argmax([h[-1] for _, h in results]))
    best_params, best_history = results[best_idx]
    if return_all:
        return best_params, best_history, results
    return best_params, best_history


def viterbi_paths(params: IOHMMParams,
                  trials_list: Sequence[emissions_mod.Trials],
                  state_list: Sequence[states_mod.IOState],
                  stage1: emissions_mod.Stage1Params,
                  grids: io_core.IOGrids) -> list[np.ndarray]:
    """Most-likely state path per sequence under fitted ``params``.

    Uses the velocity channel iff ``params.vel_per_state`` is populated (and
    the trials carry velocity), matching whatever the fit used.
    """
    if isinstance(trials_list, emissions_mod.Trials):
        trials_list = [trials_list]
    vel = params.vel_per_state if params.vel_per_state else None
    paths = []
    for trials in trials_list:
        log_emiss = emissions_mod.log_emission_matrix(
            grids, stage1, state_list, params.psych_per_state, trials,
            vel_per_state=vel,
        )
        path, _ = hmm.viterbi(params.log_pi, params.log_A, log_emiss)
        paths.append(path)
    return paths


__all__ = [
    'IOHMMParams',
    'e_step',
    'm_step_transitions',
    'm_step_velocity',
    'random_init',
    'fit',
    'viterbi_paths',
]
