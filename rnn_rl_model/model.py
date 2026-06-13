"""The RNN + actor-critic model in a single torch graph.

Forward path, faithful to ``si_network_model`` but differentiable end to end:

    Gabor drive  ->  recurrent V1 dynamics  ->  cosine similarity to two
    (static) archetype vectors  ->  similarity index SI(t)  ->  Bernoulli Go/NoGo
    policy, with a value baseline.

What learns:

* the two **archetype vectors** ``w_go``, ``w_nogo`` (the V1->action mapping --
  the framework's "one plastic weight vector per response neuron"), a decision
  **bias** and a learnable decision **gain** ``beta``;
* a small **critic** (value baseline) over the similarity features;
* and, *only when* ``train_v1=True``, the recurrent weights ``W_rec`` -- so the
  recurrence is shaped end-to-end by reward before being frozen.

The Gabor feed-forward front end is fixed (the framework's "V1 is stimulus-driven
sensory cortex") and lives in the numpy ``V1Model``; this module consumes its
``jittered_drive`` and only owns the recurrence + readout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import RLConfig


def phi(x: torch.Tensor, sat: float) -> torch.Tensor:
    """V1 transfer function: rectified, smoothly saturating at ``sat``.

    Identical in form to ``si_network_model.v1_model._phi`` -- guarantees bounded,
    stable recurrent dynamics regardless of the (possibly trained) recurrent gain.
    """
    return sat * torch.tanh(torch.clamp(x, min=0.0) / sat)


class RecurrentReadoutAgent(nn.Module):
    """Recurrent V1 + cosine-archetype readout + actor-critic, for one animal."""

    def __init__(self, W_rec_init: np.ndarray, cfg: RLConfig, *,
                 train_v1: bool, seed: int):
        super().__init__()
        self.cfg = cfg
        b = cfg.base
        self.n_v1 = b.n_v1
        self.n_steps = b.n_steps
        self.eps = b.si_eps
        self._a = b.dt / b.tau
        self._sat = b.v1_sat
        self._rec_gain = b.rec_gain
        self._sensory_noise = b.sensory_noise
        self._within_noise = b.within_trial_noise
        self._shared_frac = b.shared_noise_frac

        g = torch.Generator().manual_seed(int(seed))

        # Recurrence: a Parameter when trained, a frozen buffer otherwise.
        W = torch.tensor(np.asarray(W_rec_init, dtype=np.float32))
        if train_v1:
            self.W_rec = nn.Parameter(W)
        else:
            self.register_buffer("W_rec", W)

        # Archetypes: small random init -> SI(t) ~ 0 -> ~50/50 choices early
        # (the RL agent acting randomly at the start).
        init = b.w_init_scale * torch.randn(2, self.n_v1, generator=g)
        self.w_go = nn.Parameter(init[0].clone())
        self.w_nogo = nn.Parameter(init[1].clone())
        self.bias = nn.Parameter(torch.zeros(()))
        # Learnable positive decision gain via softplus.
        beta0 = float(np.log(np.expm1(max(cfg.policy_beta_init, 1e-3))))
        self.beta_raw = nn.Parameter(torch.tensor(beta0))

        # Critic: linear value baseline over the two mean similarities.
        self.critic = nn.Linear(2, 1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------
    @property
    def beta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.beta_raw)

    @property
    def trains_v1(self) -> bool:
        return isinstance(self.W_rec, nn.Parameter)

    # ------------------------------------------------------------------
    def rollout(self, drive: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Recurrent V1 dynamics for a static drive ``(B, n_v1)`` -> ``(B, T, n_v1)``.

        Mirrors ``V1Model.simulate``: one across-trial input perturbation (held
        fixed over the trial -> bundle width) plus a per-timestep within-trial
        perturbation (-> the within-trial diffusion that produces Var_t[SI]).
        """
        B, n = drive.shape
        priv = np.sqrt(1.0 - self._shared_frac) * torch.randn(B, n, generator=gen)
        shared = np.sqrt(self._shared_frac) * torch.randn(B, 1, generator=gen)
        drive_n = drive + self._sensory_noise * (priv + shared)

        r = torch.zeros(B, n)
        traj = []
        for _ in range(self.n_steps):
            within = self._within_noise * torch.randn(B, n, generator=gen)
            rec = self._rec_gain * (r @ self.W_rec)
            r = r + self._a * (-r + phi(drive_n + rec + within, self._sat))
            traj.append(r)
        return torch.stack(traj, dim=1)

    # ------------------------------------------------------------------
    def similarities(self, traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pointwise cosine-derived similarities and SI(t).

        Returns ``(s_go, s_nogo, si)`` each ``(B, T)``: ``s_*`` are the
        positivity-mapped cosines in [0, 1]; ``si`` is the bounded similarity
        index in [-1, +1] (positive favours Go) -- the same form as
        ``si_network_model.decision.si_trace``.
        """
        norm_r = traj.norm(dim=-1)                          # (B, T)
        c_go = (traj @ self.w_go) / (norm_r * self.w_go.norm() + self.eps)
        c_nogo = (traj @ self.w_nogo) / (norm_r * self.w_nogo.norm() + self.eps)
        s_go = (1.0 + c_go) / 2.0
        s_nogo = (1.0 + c_nogo) / 2.0
        si = (s_go - s_nogo) / (s_go + s_nogo + self.eps)
        return s_go, s_nogo, si

    # ------------------------------------------------------------------
    def forward(self, drive: torch.Tensor, gen: torch.Generator) -> dict:
        """Full forward pass: rollout -> SI -> policy + value.

        Returns a dict with ``p_go`` (B,), ``value`` (B,), ``mean_si`` (B,),
        ``si`` (B, T) and ``traj`` (B, T, n_v1).
        """
        traj = self.rollout(drive, gen)
        s_go, s_nogo, si = self.similarities(traj)
        mean_si = si.mean(dim=1)
        logit = self.beta * mean_si + self.bias
        p_go = torch.sigmoid(logit)
        feats = torch.stack([s_go.mean(dim=1), s_nogo.mean(dim=1)], dim=1).detach()
        value = self.critic(feats).squeeze(-1)
        return {"p_go": p_go, "value": value, "mean_si": mean_si,
                "si": si, "traj": traj}

    # ------------------------------------------------------------------
    def archetype_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Frozen ``(w_go, w_nogo)`` as numpy, for the template-Delta-mu analysis."""
        with torch.no_grad():
            return (self.w_go.detach().cpu().numpy().copy(),
                    self.w_nogo.detach().cpu().numpy().copy())
