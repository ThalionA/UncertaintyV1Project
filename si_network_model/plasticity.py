"""Reward-modulated plasticity: the framework's three-factor Hebbian rule.

After each trial the V1->decision synapses onto the *chosen* readout neuron
are updated, per timestep, by

    Delta archetype_chosen(t) = learning_rate * reward * r(t)

-- pre-synaptic factor r(t), post-synaptic gating to the chosen neuron, and
the reward as the third factor (the framework's literal rule, with the reward
itself as the modulatory signal). Rewarded outcomes potentiate, punished
outcomes depress.

Each archetype timestep is renormalised to unit length after every update
(the cosine readout is scale-invariant, so this is free). Over training each
archetype trajectory converges to the reward-weighted mean V1 trajectory for
its choice -- the prototype archetype.

Asymmetric payoffs are handled correctly by this rule: a NoGo choice that
earns zero either way (correct rejection = miss = 0) simply produces no
update -- the NoGo archetype freezes (a single-archetype strategy, framework
Fig. 5C). And unequal Go-side magnitudes (a small false-alarm cost vs a large
hit reward) net-potentiate the Go archetype -- an emergent Go-bias, the
framework's "prior as asymmetric weight scaling".
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .decision import DecisionLayer

_EPS = 1e-12


def compute_reward(choice: int, rewarded_category: int, cfg: Config) -> float:
    """Scalar reward for a trial outcome.

    ``choice`` and ``rewarded_category`` are 1 = Go, 0 = NoGo. In symmetric
    mode every correct choice is +1 and every error -1. In asymmetric mode the
    four Go/NoGo outcomes (hit / false alarm / correct rejection / miss) get
    distinct, water-task-like values.
    """
    if cfg.reward_mode == "symmetric":
        return cfg.reward_correct if choice == rewarded_category else cfg.reward_error
    if cfg.reward_mode == "asymmetric":
        if choice == 1 and rewarded_category == 1:
            return cfg.reward_hit
        if choice == 1 and rewarded_category == 0:
            return cfg.reward_false_alarm
        if choice == 0 and rewarded_category == 0:
            return cfg.reward_correct_reject
        return cfg.reward_miss
    raise ValueError(f"unknown reward_mode: {cfg.reward_mode!r}")


class Plasticity:
    """Three-factor reward-modulated Hebbian learner."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def update(self, layer: DecisionLayer, traj: np.ndarray,
               choice: int, reward: float) -> float:
        """Apply one plasticity step; return the reward signal used.

        ``traj`` is the trial's V1 trajectory ``(n_steps, n_v1)``. Only the
        *chosen* readout neuron's archetype is modified; each timestep is
        renormalised to unit length afterwards.
        """
        step = self.cfg.learning_rate * reward * traj  # (n_steps, n_v1)
        if choice == 1:
            a = layer.proto_go + step
            layer.proto_go = a / (np.linalg.norm(a, axis=1, keepdims=True) + _EPS)
        else:
            a = layer.proto_nogo + step
            layer.proto_nogo = a / (np.linalg.norm(a, axis=1, keepdims=True) + _EPS)
        return reward
