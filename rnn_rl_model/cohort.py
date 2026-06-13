"""Cohort runner: every animal trained under both V1 conditions.

For each simulated animal a single fixed Gabor front end is built (shared RFs),
then two agents are trained on it: one with the recurrence frozen (``fixed``),
one with the recurrence trained end-to-end then frozen (``trained``). Both are
evaluated and scored identically, so the only difference between the two columns
is whether ``W_rec`` was task-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import fit_neural_io, summarise
from .config import RLConfig, animal_seeds, sample_rl_animal
from .evaluate import evaluate_agent
from .train import build_v1, train_agent

CONDITIONS = ("fixed", "trained")


@dataclass
class AnimalResult:
    animal_id: int
    cfg: RLConfig
    # per-condition payloads
    summary: dict = field(default_factory=dict)     # condition -> metrics dict
    eval: dict = field(default_factory=dict)         # condition -> eval arrays
    hist: dict = field(default_factory=dict)         # condition -> training history


def run_animal(animal_id: int, base_cfg: RLConfig, seed: int) -> AnimalResult:
    """Build one animal's V1 and train + evaluate both V1 conditions."""
    cfg = sample_rl_animal(base_cfg, np.random.default_rng(seed))
    v1 = build_v1(cfg, seed)
    res = AnimalResult(animal_id=animal_id, cfg=cfg)
    for cond in CONDITIONS:
        train_v1 = cond == "trained"
        agent, hist = train_agent(v1, cfg, seed, train_v1=train_v1)
        ev = evaluate_agent(v1, agent, cfg, seed + 1000)
        wgo, wnogo = agent.archetype_vectors()
        nio = fit_neural_io(v1, agent, cfg.base, seed + 2000)
        ev["io_log_odds"] = nio.log_odds(ev["r_bar"])
        ev["io_choice"] = nio.decide(ev["r_bar"]).astype(int)
        res.summary[cond] = summarise(ev, wgo, wnogo, nio, hist)
        res.eval[cond] = ev
        res.hist[cond] = hist
    return res


def run_cohort(base_cfg: RLConfig, verbose: bool = True) -> list[AnimalResult]:
    """Train + evaluate the whole cohort under both V1 conditions."""
    seeds = animal_seeds(base_cfg)
    results: list[AnimalResult] = []
    for i in range(base_cfg.n_animals):
        res = run_animal(i, base_cfg, int(seeds[i]))
        results.append(res)
        if verbose:
            f, t = res.summary["fixed"], res.summary["trained"]
            print(f"  animal {i:2d}: "
                  f"fixed acc={f['network_accuracy']:.3f} eff={f['decision_efficiency']:.3f} "
                  f"cosT={f['cos_readout_template']:.2f} | "
                  f"trained acc={t['network_accuracy']:.3f} eff={t['decision_efficiency']:.3f} "
                  f"cosT={t['cos_readout_template']:.2f}")
    return results


def cohort_table(results: list[AnimalResult]) -> dict:
    """Stack per-animal metrics into condition -> metric -> array, plus means."""
    keys = list(next(iter(results)).summary["fixed"].keys())
    table: dict = {}
    for cond in CONDITIONS:
        table[cond] = {k: np.array([r.summary[cond][k] for r in results])
                       for k in keys}
    return table
