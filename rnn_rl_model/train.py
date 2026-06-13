"""Actor-critic training of one animal's V1->action mapping.

REINFORCE with a learned value baseline (a one-step actor-critic, since each
trial is a single-step episode): the agent samples a Go/NoGo action from its
policy, receives reward, and updates the archetypes (and, if ``train_v1``, the
recurrence) by policy gradient, advantage = reward - value.

The Gabor front end is the fixed numpy ``V1Model``; this loop only feeds its
``jittered_drive`` into the differentiable agent.
"""

from __future__ import annotations

import numpy as np
import torch

from si_network_model.config import (
    ARCHETYPE_PAIR,
    BOUNDARY_DEG,
    CONTRAST_DISPERSION_PAIRS,
    ORIENTATIONS_DEG,
    Config,
)
from si_network_model.stimuli import Condition
from si_network_model.v1_model import V1Model

from .config import RLConfig
from .model import RecurrentReadoutAgent

_GO_ORI = tuple(o for o in ORIENTATIONS_DEG if o < BOUNDARY_DEG)
_NOGO_ORI = tuple(o for o in ORIENTATIONS_DEG if o > BOUNDARY_DEG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_v1(cfg: RLConfig, seed: int) -> V1Model:
    """The fixed Gabor front end for one animal (shared by both V1 conditions)."""
    rng = np.random.default_rng(seed)
    v1 = V1Model(cfg.base, rng)
    v1.build_drive_grid()
    return v1


def batched_reward(action: np.ndarray, rewarded: np.ndarray,
                   base: Config) -> np.ndarray:
    """Vectorised version of ``si_network_model.plasticity.compute_reward``."""
    action = np.asarray(action).astype(int)
    rewarded = np.asarray(rewarded).astype(int)
    if base.reward_mode == "symmetric":
        return np.where(action == rewarded, base.reward_correct,
                        base.reward_error).astype(float)
    if base.reward_mode == "asymmetric":
        r = np.empty(len(action), dtype=float)
        r[(action == 1) & (rewarded == 1)] = base.reward_hit
        r[(action == 1) & (rewarded == 0)] = base.reward_false_alarm
        r[(action == 0) & (rewarded == 0)] = base.reward_correct_reject
        r[(action == 0) & (rewarded == 1)] = base.reward_miss
        return r
    raise ValueError(f"unknown reward_mode: {base.reward_mode!r}")


def sample_batch(cfg: RLConfig, rng: np.random.Generator,
                 n: int) -> tuple[list[Condition], np.ndarray]:
    """Draw ``n`` (Condition, rewarded_category) pairs: flat Go/NoGo prior."""
    cats = (rng.random(n) < 0.5).astype(int)  # 1 = Go, 0 = NoGo
    conds: list[Condition] = []
    for cat in cats:
        if rng.random() < cfg.archetype_fraction:
            ori = 0.0 if cat == 1 else 90.0
            contrast, dispersion = ARCHETYPE_PAIR
        else:
            pool = _GO_ORI if cat == 1 else _NOGO_ORI
            ori = float(rng.choice(pool))
            contrast, dispersion = CONTRAST_DISPERSION_PAIRS[
                rng.integers(len(CONTRAST_DISPERSION_PAIRS))]
        conds.append(Condition(float(ori), float(contrast), float(dispersion)))
    return conds, cats.astype(int)


def _renormalise_archetypes(agent: RecurrentReadoutAgent) -> None:
    """Project archetypes back to unit norm (cosine readout is scale-invariant).

    Keeps the cosine denominator well-conditioned without changing the policy.
    """
    with torch.no_grad():
        agent.w_go.div_(agent.w_go.norm() + agent.eps)
        agent.w_nogo.div_(agent.w_nogo.norm() + agent.eps)


# ---------------------------------------------------------------------------
# Train one agent
# ---------------------------------------------------------------------------


def train_agent(v1: V1Model, cfg: RLConfig, seed: int, *,
                train_v1: bool) -> tuple[RecurrentReadoutAgent, dict]:
    """Run the actor-critic curriculum for one (animal, V1-condition).

    Returns the trained agent and a history dict (per-batch reward / accuracy /
    mean-|SI|), the latter for learning-curve figures.
    """
    base = v1.cfg
    agent = RecurrentReadoutAgent(v1.W_rec, cfg, train_v1=train_v1, seed=seed)
    _renormalise_archetypes(agent)

    groups = [
        {"params": [agent.w_go, agent.w_nogo, agent.bias, agent.beta_raw],
         "lr": cfg.lr_readout},
        {"params": list(agent.critic.parameters()), "lr": cfg.lr_critic},
    ]
    if train_v1:
        groups.append({"params": [agent.W_rec], "lr": cfg.lr_v1,
                       "weight_decay": cfg.v1_weight_decay})
    opt = torch.optim.Adam(groups)

    rng = np.random.default_rng(seed + 11)
    gen = torch.Generator().manual_seed(int(seed) + 22)
    B = cfg.batch_size
    n_batches = max(1, cfg.n_train_trials // B)
    hist = {"reward": [], "acc": [], "abs_si": []}

    for _ in range(n_batches):
        conds, rewarded = sample_batch(cfg, rng, B)
        drive = torch.from_numpy(v1.jittered_drive(conds, rng).astype(np.float32))
        out = agent(drive, gen)
        p = out["p_go"].clamp(1e-6, 1.0 - 1e-6)

        u = torch.rand(B, generator=gen)
        action = (u < p).float()
        logp = action * torch.log(p) + (1.0 - action) * torch.log(1.0 - p)
        reward = torch.from_numpy(
            batched_reward(action.detach().numpy(), rewarded, base).astype(np.float32))

        value = out["value"]
        advantage = (reward - value).detach()
        policy_loss = -(advantage * logp).mean()
        value_loss = ((reward - value) ** 2).mean()
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()
        loss = (policy_loss + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
        opt.step()
        _renormalise_archetypes(agent)

        act_np = action.detach().numpy().astype(int)
        hist["reward"].append(float(reward.mean()))
        hist["acc"].append(float((act_np == rewarded).mean()))
        hist["abs_si"].append(float(out["mean_si"].abs().mean()))

    agent.eval()
    for pgrp in agent.parameters():
        pgrp.requires_grad_(False)
    return agent, {k: np.asarray(v) for k, v in hist.items()}
