"""Frozen-weight evaluation of a trained agent over all 81 conditions.

Produces the per-trial arrays the analysis and the RD-1/RD-2 readout tests
consume -- in the same shape as ``si_network_model``'s eval block:

    r_bar (n, n_v1), choice, mean_si, var_si, orientation, contrast, ...

Behaviour (``choice``, ``rt``) comes from the framework's two-bound DDM driven
by the SI(t) stream (reusing ``si_network_model.decision.accumulate``), so the
RL agent's policy and the framework's accumulator are reported consistently.
"""

from __future__ import annotations

import numpy as np
import torch

from si_network_model.decision import accumulate
from si_network_model.stimuli import Condition, enumerate_conditions
from si_network_model.v1_model import V1Model

from .config import RLConfig
from .model import RecurrentReadoutAgent

_EVAL_BATCH = 400


class AgentV1:
    """``V1Model``-compatible front end that rolls out the *agent's* recurrence.

    Exposes ``present(conditions, rng) -> (B, n_steps, n_v1)`` using the fixed
    Gabor drive plus the agent's (possibly trained) ``W_rec``. Lets the neural
    ideal observer be fitted on exactly the V1 code the agent reads.
    """

    def __init__(self, v1: V1Model, agent: RecurrentReadoutAgent):
        self.v1 = v1
        self.agent = agent

    def present(self, conditions: list[Condition],
                rng: np.random.Generator) -> np.ndarray:
        drive = torch.from_numpy(self.v1.jittered_drive(conditions, rng).astype(np.float32))
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 31 - 1)))
        with torch.no_grad():
            traj = self.agent.rollout(drive, gen)
        return traj.numpy()


def evaluate_agent(v1: V1Model, agent: RecurrentReadoutAgent, cfg: RLConfig,
                   seed: int) -> dict:
    """Balanced frozen-weight evaluation over every condition.

    Returns a dict of per-trial arrays; ``choice`` is the DDM readout, ``p_go``
    the policy probability, and ``r_bar``/``mean_si``/``var_si`` feed the RD
    tests.
    """
    base = v1.cfg
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(seed) + 7)
    conditions = enumerate_conditions()

    trials: list[Condition] = []
    cond_idx: list[int] = []
    for ci, cond in enumerate(conditions):
        trials.extend([cond] * cfg.n_eval_per_cond)
        cond_idx.extend([ci] * cfg.n_eval_per_cond)

    r_bar_parts, choice_parts, rt_parts = [], [], []
    pgo_parts, mean_si_parts, var_si_parts = [], [], []
    ori, con, dis = [], [], []

    for start in range(0, len(trials), _EVAL_BATCH):
        batch = trials[start:start + _EVAL_BATCH]
        drive = torch.from_numpy(v1.jittered_drive(batch, rng).astype(np.float32))
        with torch.no_grad():
            out = agent(drive, gen)
        si = out["si"].numpy()                       # (B, T)
        traj = out["traj"].numpy()
        choice, rt, _ = accumulate(si, base, rng)    # framework DDM behaviour

        r_bar_parts.append(traj.mean(axis=1).astype(np.float32))
        choice_parts.append(choice)
        rt_parts.append(rt)
        pgo_parts.append(out["p_go"].numpy())
        mean_si_parts.append(si.mean(axis=1))
        var_si_parts.append(si.var(axis=1))
        ori.extend(c.orientation_deg for c in batch)
        con.extend(c.contrast for c in batch)
        dis.extend(c.dispersion_deg for c in batch)

    return {
        "condition_idx": np.asarray(cond_idx),
        "orientation": np.asarray(ori, dtype=float),
        "contrast": np.asarray(con, dtype=float),
        "dispersion": np.asarray(dis, dtype=float),
        "r_bar": np.concatenate(r_bar_parts, axis=0),
        "choice": np.concatenate(choice_parts).astype(int),
        "rt": np.concatenate(rt_parts),
        "p_go": np.concatenate(pgo_parts),
        "mean_si": np.concatenate(mean_si_parts),
        "var_si": np.concatenate(var_si_parts),
    }
