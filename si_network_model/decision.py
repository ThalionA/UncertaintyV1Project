"""Decision layer: cosine readout, the SI(t) evidence stream, and the DDM.

Two readout neurons -- Go-driving and NoGo-driving -- each carry a plastic
weight vector from V1. Their post-synaptic current is the cosine similarity
between the V1 trajectory r(t) and the weight vector (a dot product with
divisive normalisation). The bounded normalised difference of the two is the
moment-by-moment similarity index SI(t), the evidence stream fed to a
symmetric two-bound drift-diffusion accumulator.

Sign convention: SI > 0 favours **Go**. (This is the framework's SI with the
Go/NoGo order chosen so positive = Go = the active 'lick' response.)
"""

from __future__ import annotations

import numpy as np

from .config import Config


def cosine_trace(traj: np.ndarray, archetype: np.ndarray,
                 eps: float) -> np.ndarray:
    """Pointwise cosine similarity between r(t) and a trajectory archetype.

    ``traj`` is ``(..., n_steps, n_v1)``; ``archetype`` is the time-resolved
    archetype ``(n_steps, n_v1)`` -- the cosine compares r(t) to the
    archetype's value *at the same time t*. Returns ``(..., n_steps)``. Early
    in the trial r(t) is near zero, so the cosine is ~0 there (no evidence
    yet) -- the eps keeps it well defined.
    """
    dot = np.einsum("...tn,tn->...t", traj, archetype)
    norm_r = np.linalg.norm(traj, axis=-1)
    norm_a = np.linalg.norm(archetype, axis=-1)  # (n_steps,)
    return dot / (norm_r * norm_a + eps)


def si_trace(traj: np.ndarray, proto_go: np.ndarray, proto_nogo: np.ndarray,
             eps: float) -> np.ndarray:
    """Bounded prototype SI(t) in [-1, +1]; positive favours Go.

    Uses the framework's positivity-mapped cosine: each similarity is mapped
    to [0, 1] before the normalised difference is taken. ``proto_go`` and
    ``proto_nogo`` are trajectory archetypes ``(n_steps, n_v1)``.
    """
    s_go = (1.0 + cosine_trace(traj, proto_go, eps)) / 2.0
    s_nogo = (1.0 + cosine_trace(traj, proto_nogo, eps)) / 2.0
    return (s_go - s_nogo) / (s_go + s_nogo + eps)


def accumulate(si: np.ndarray, cfg: Config,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric two-bound DDM driven by the SI(t) stream.

    ``si`` is ``(B, n_steps)``. Integrates ``dX = gain * SI(t) dt + sigma dW``
    with absorbing bounds at +/- ``ddm_bound``. The first crossing sets the
    choice (upper = Go, lower = NoGo) and the response time; trials that never
    cross are decided by the sign of X at the 2 s deadline.

    Returns ``(choice, rt, x_trace)`` with ``choice`` in {1=Go, 0=NoGo},
    ``rt`` in seconds, ``x_trace`` of shape ``(B, n_steps)``.
    """
    b, n_steps = si.shape
    x = np.zeros(b)
    x_trace = np.empty((b, n_steps))
    crossed = np.zeros(b, dtype=bool)
    choice = np.full(b, -1, dtype=int)
    rt = np.full(b, cfg.t_trial, dtype=float)
    diff_sd = cfg.ddm_sigma * np.sqrt(cfg.dt)

    for t in range(n_steps):
        dx = (cfg.ddm_drift_gain * si[:, t] * cfg.dt
              + diff_sd * rng.standard_normal(b))
        x = x + dx
        # freeze trajectory of already-decided trials at the bound
        x = np.where(crossed, np.clip(x, -cfg.ddm_bound, cfg.ddm_bound), x)
        x_trace[:, t] = x

        hit_up = (~crossed) & (x >= cfg.ddm_bound)
        hit_dn = (~crossed) & (x <= -cfg.ddm_bound)
        now = (t + 1) * cfg.dt
        choice[hit_up] = 1
        rt[hit_up] = now
        choice[hit_dn] = 0
        rt[hit_dn] = now
        crossed |= hit_up | hit_dn

    # Deadline decisions: sign of the accumulator at t = t_trial.
    undecided = ~crossed
    choice[undecided] = (x[undecided] >= 0.0).astype(int)
    return choice, rt, x_trace


class DecisionLayer:
    """Two plastic readout neurons + the DDM accumulator, for one animal.

    Each readout neuron carries a *trajectory* archetype ``(n_steps, n_v1)`` --
    the framework's time-resolved prototype, learned by per-timestep
    reward-modulated Hebbian plasticity.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        # Small random initial archetypes -> SI(t) ~ 0 -> ~50/50 choices early
        # (the "RL agent acting randomly at the start" behaviour).
        shape = (cfg.n_steps, cfg.n_v1)
        self.proto_go = rng.normal(0.0, cfg.w_init_scale, shape)
        self.proto_nogo = rng.normal(0.0, cfg.w_init_scale, shape)

    def si_trace(self, traj: np.ndarray) -> np.ndarray:
        """Prototype SI(t) for a trajectory or a batch of trajectories."""
        return si_trace(traj, self.proto_go, self.proto_nogo, self.cfg.si_eps)

    def decide(self, traj: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map V1 trajectories to choices via SI(t) + the DDM.

        Accepts a single trajectory ``(n_steps, n_v1)`` or a batch
        ``(B, n_steps, n_v1)``; always returns batched arrays.
        """
        if traj.ndim == 2:
            traj = traj[None]
        si = self.si_trace(traj)
        return accumulate(si, self.cfg, rng)
