"""Configuration for the RNN+RL similarity-framework model.

``RLConfig`` *composes* the ``si_network_model`` base ``Config`` (so the
stimulus space, V1 architecture, noise model and ideal observers are shared and
identical to the Hebbian sibling) and adds the reinforcement-learning knobs.

The only structural choice that distinguishes this model from the Hebbian one is
the learner: an actor-critic policy-gradient agent whose policy reads a cosine
similarity index between the V1 population and two learned archetype vectors.
Whether the recurrent V1 weights ``W_rec`` are *also* trained (then frozen) is a
single switch, ``train_v1`` -- the fixed-vs-trained-V1 comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from si_network_model.config import Config, sample_animal_traits


@dataclass(frozen=True)
class RLConfig:
    """Immutable bundle: a base si ``Config`` plus the RL hyperparameters."""

    base: Config = field(default_factory=lambda: Config(n_v1=160))

    # ---- actor-critic ----
    lr_readout: float = 0.02     # archetypes + bias + policy gain
    lr_critic: float = 0.05      # value head
    lr_v1: float = 5e-3          # W_rec (only used when train_v1=True)
    v1_weight_decay: float = 1e-4  # keep a trained recurrence tame
    policy_beta_init: float = 4.0  # initial decision gain (softplus-parametrised, learnable)
    value_coef: float = 0.5
    entropy_coef: float = 0.01   # small entropy bonus -> sustained exploration early
    grad_clip: float = 5.0

    # ---- training schedule ----
    n_train_trials: int = 6000   # actor-critic trials per animal
    batch_size: int = 64
    archetype_fraction: float = 0.5  # share of training trials that are archetypes
    train_v1: bool = False       # fixed V1 (False) vs trained-then-frozen V1 (True)

    # ---- evaluation ----
    n_eval_per_cond: int = 40    # frozen-weight eval trials per condition

    # ---- cohort ----
    n_animals: int = 6
    master_seed: int = 20260613

    @property
    def n_v1(self) -> int:
        return self.base.n_v1

    @property
    def n_steps(self) -> int:
        return self.base.n_steps


def sample_rl_animal(base: RLConfig, rng: np.random.Generator) -> RLConfig:
    """Draw one animal's traits: si V1/jitter traits + RL learning-rate spread.

    The base ``Config`` traits (orientation-jitter gain, DDM diffusion, recurrent
    gain) are sampled by the si helper so this cohort has the same kind of
    between-animal spread in bundle width and lapse rate. The readout learning
    rate and decision gain get their own log-normal jitter so animals also differ
    in learning speed and decisiveness.
    """
    base_cfg = sample_animal_traits(base.base, rng)
    return replace(
        base,
        base=base_cfg,
        lr_readout=float(base.lr_readout * np.exp(rng.normal(0.0, 0.30))),
        policy_beta_init=float(base.policy_beta_init * np.exp(rng.normal(0.0, 0.20))),
    )


def animal_seeds(cfg: RLConfig) -> np.ndarray:
    """Deterministic per-animal seeds for a cohort."""
    base_rng = np.random.default_rng(cfg.master_seed)
    return base_rng.integers(1, 2 ** 31 - 1, cfg.n_animals)
