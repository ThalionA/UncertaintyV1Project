"""One simulated animal: V1 + decision layer + plasticity, with its own traits.

Each ``Animal`` is built from a single seed. The seed drives the Gabor RF
mosaic, the initial readout weights and all noise; a draw of trait parameters
(sensory noise, learning rate, DDM diffusion, recurrent gain) gives the cohort
genuine between-animal variability (Decision 5, UNDERSTANDING.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, sample_animal_traits
from .decision import DecisionLayer
from .plasticity import Plasticity
from .v1_model import V1Model


@dataclass
class Animal:
    """Container tying together one animal's V1, decision layer and learner.

    ``bundle_go`` / ``bundle_nogo`` hold the exemplar-trajectory bundle -- a
    reservoir-sampled memory of K experienced class-c trajectories, filled
    during training. They back the expected-exemplar / max-exemplar / vMF SI
    variants (the framework's "additional memory system" beyond the
    prototype-forming V1->motor synapses).
    """

    animal_id: int
    cfg: Config
    v1: V1Model
    decision: DecisionLayer
    plasticity: Plasticity
    seed: int
    bundle_go: np.ndarray | None = None    # (K, n_steps, n_v1)
    bundle_nogo: np.ndarray | None = None

    @classmethod
    def build(cls, animal_id: int, base_cfg: Config, seed: int) -> "Animal":
        """Instantiate an animal: sample traits, build V1 and the decision layer."""
        rng = np.random.default_rng(seed)
        cfg = sample_animal_traits(base_cfg, rng)
        v1 = V1Model(cfg, rng)
        v1.build_drive_grid()  # precompute drive-vs-orientation for jitter lookup
        decision = DecisionLayer(cfg, rng)
        plasticity = Plasticity(cfg)
        return cls(animal_id=animal_id, cfg=cfg, v1=v1,
                   decision=decision, plasticity=plasticity, seed=seed)

    @property
    def traits(self) -> dict[str, float]:
        """The four sampled trait values, for cross-animal analysis."""
        return {
            "ori_jitter_gain": self.cfg.ori_jitter_gain,
            "learning_rate": self.cfg.learning_rate,
            "ddm_sigma": self.cfg.ddm_sigma,
            "rec_gain": self.cfg.rec_gain,
        }
