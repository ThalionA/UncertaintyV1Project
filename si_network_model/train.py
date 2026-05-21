"""Curriculum training loop and cohort runner.

Phase 1 presents only the archetypes (0 deg / 90 deg at contrast 1,
dispersion 5) until rolling accuracy graduates the animal. Phase 2 presents
the full stimulus grid with sampling biased toward the archetypes, keeping a
flat Go/NoGo category prior throughout.

V1 is weight-independent, so V1 trajectories are simulated in batches; the
decision + plasticity step is then run trial-by-trial (weights change every
trial). After training, each animal is evaluated with frozen weights on a
balanced block spanning all 81 conditions, and a neural ideal observer is
fitted to its V1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent import Animal
from .config import (
    ARCHETYPE_PAIR,
    BOUNDARY_DEG,
    CONTRAST_DISPERSION_PAIRS,
    ORIENTATIONS_DEG,
    Config,
)
from .decision import accumulate
from .ideal_observer import NeuralIdealObserver, StimulusIdealObserver
from .plasticity import compute_reward
from .si_variants import VARIANT_NAMES, all_variant_si, estimate_kappa
from .stimuli import Condition, enumerate_conditions

_GO_ORIENTATIONS = tuple(o for o in ORIENTATIONS_DEG if o < BOUNDARY_DEG)
_NOGO_ORIENTATIONS = tuple(o for o in ORIENTATIONS_DEG if o > BOUNDARY_DEG)
_EVAL_BATCH = 600


# ---------------------------------------------------------------------------
# Trial sampling
# ---------------------------------------------------------------------------


def sample_trial(phase: int, cfg: Config,
                 rng: np.random.Generator) -> tuple[Condition, int]:
    """Draw one (Condition, rewarded_category) pair for the given phase.

    Category is always 50/50 (flat Go/NoGo prior). Phase 1 shows only
    archetypes; Phase 2 shows an archetype with probability
    ``archetype_fraction``, otherwise a random full-grid stimulus.
    """
    category = int(rng.random() < 0.5)  # 1 = Go, 0 = NoGo
    is_archetype = phase == 1 or rng.random() < cfg.archetype_fraction
    if is_archetype:
        orientation = 0.0 if category == 1 else 90.0
        contrast, dispersion = ARCHETYPE_PAIR
    else:
        pool = _GO_ORIENTATIONS if category == 1 else _NOGO_ORIENTATIONS
        orientation = float(rng.choice(pool))
        contrast, dispersion = CONTRAST_DISPERSION_PAIRS[
            rng.integers(len(CONTRAST_DISPERSION_PAIRS))]
    return Condition(orientation, float(contrast), float(dispersion)), category


# ---------------------------------------------------------------------------
# Per-chunk processing (batched V1, sequential decision + learning)
# ---------------------------------------------------------------------------


class _Reservoir:
    """Reservoir sampler: a uniform random K-sample of an online stream."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k = k
        self.rng = rng
        self.buf: list[np.ndarray] = []
        self.n = 0

    def offer(self, item: np.ndarray) -> None:
        self.n += 1
        if len(self.buf) < self.k:
            self.buf.append(item)
        else:
            j = int(self.rng.integers(self.n))
            if j < self.k:
                self.buf[j] = item

    def array(self) -> np.ndarray | None:
        return np.stack(self.buf) if self.buf else None


def _process_chunk(animal: Animal, conditions: list[Condition],
                   rewarded: list[int],
                   rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """Run V1 for a chunk of trials, then decide and learn online.

    Returns the per-trial record and the V1 trajectories ``(n, n_steps, n_v1)``.
    """
    cfg = animal.cfg
    trajs = animal.v1.present(conditions, rng)
    n = len(conditions)

    choice = np.empty(n, dtype=int)
    reward = np.empty(n)
    mean_si = np.empty(n)
    for k in range(n):
        traj = trajs[k]
        si = animal.decision.si_trace(traj)
        ch, _, _ = accumulate(si[None, :], cfg, rng)
        ch = int(ch[0])
        rw = compute_reward(ch, rewarded[k], cfg)
        animal.plasticity.update(animal.decision, traj, ch, rw)
        choice[k] = ch
        reward[k] = rw
        mean_si[k] = float(si.mean())

    correct = (choice == np.asarray(rewarded)).astype(int)
    return ({"choice": choice, "reward": reward, "mean_si": mean_si,
             "correct": correct}, trajs)


# ---------------------------------------------------------------------------
# Train one animal
# ---------------------------------------------------------------------------


def train_animal(animal: Animal, rng: np.random.Generator) -> dict:
    """Run the two-phase curriculum for one animal; return the trial log."""
    cfg = animal.cfg
    log: dict[str, list] = {k: [] for k in (
        "orientation", "contrast", "dispersion", "category",
        "choice", "correct", "reward", "mean_si", "phase")}

    def _record(conds, rewarded, out, phase):
        for c, rc, ch, cor, rw, ms in zip(
                conds, rewarded, out["choice"], out["correct"],
                out["reward"], out["mean_si"]):
            log["orientation"].append(c.orientation_deg)
            log["contrast"].append(c.contrast)
            log["dispersion"].append(c.dispersion_deg)
            log["category"].append(rc)
            log["choice"].append(ch)
            log["correct"].append(cor)
            log["reward"].append(rw)
            log["mean_si"].append(ms)
            log["phase"].append(phase)

    # Exemplar-trajectory reservoirs (filled from correct Phase-2 trials).
    res_rng = np.random.default_rng(int(rng.integers(2 ** 31 - 1)))
    bundle = {1: _Reservoir(cfg.n_exemplars, res_rng),
              0: _Reservoir(cfg.n_exemplars, res_rng)}

    # ---- Phase 1: archetypes until graduation or cap ----
    n_done = 0
    while n_done < cfg.phase1_cap:
        chunk = min(cfg.train_chunk, cfg.phase1_cap - n_done)
        trials = [sample_trial(1, cfg, rng) for _ in range(chunk)]
        conds = [t[0] for t in trials]
        rewarded = [t[1] for t in trials]
        out, _ = _process_chunk(animal, conds, rewarded, rng)
        _record(conds, rewarded, out, phase=1)
        n_done += chunk
        recent = np.asarray(log["correct"][-cfg.phase1_acc_window:])
        if (len(recent) >= cfg.phase1_acc_window
                and recent.mean() >= cfg.phase1_acc_threshold):
            break
    phase1_length = n_done

    # ---- Phase 2: full grid, archetype-biased ----
    n_done = 0
    while n_done < cfg.phase2_trials:
        chunk = min(cfg.train_chunk, cfg.phase2_trials - n_done)
        trials = [sample_trial(2, cfg, rng) for _ in range(chunk)]
        conds = [t[0] for t in trials]
        rewarded = [t[1] for t in trials]
        out, trajs = _process_chunk(animal, conds, rewarded, rng)
        _record(conds, rewarded, out, phase=2)
        # Reservoir-sample correct trials into the exemplar bundles.
        for k in range(chunk):
            if out["correct"][k]:
                bundle[rewarded[k]].offer(trajs[k].astype(np.float32))
        n_done += chunk

    animal.bundle_go = bundle[1].array()
    animal.bundle_nogo = bundle[0].array()
    result = {k: np.asarray(v) for k, v in log.items()}
    result["phase1_length"] = phase1_length
    return result


# ---------------------------------------------------------------------------
# Evaluate one animal (frozen weights, balanced over all 81 conditions)
# ---------------------------------------------------------------------------


def evaluate_animal(animal: Animal, rng: np.random.Generator) -> dict:
    """Frozen-weight evaluation block: balanced trials over all conditions.

    Computes SI(t) under all four bundle-similarity variants on the same
    trials; the ``prototype`` variant is the network's behavioural readout
    (columns ``choice``/``mean_si``/``si_full``), the others are stored
    alongside as ``*_{variant}`` for the variant comparison.
    """
    cfg = animal.cfg
    proto_go, proto_nogo = animal.decision.proto_go, animal.decision.proto_nogo
    bundle_go, bundle_nogo = animal.bundle_go, animal.bundle_nogo
    kappa = estimate_kappa(bundle_go, bundle_nogo)

    conditions = enumerate_conditions()
    trials: list[tuple[int, Condition]] = []
    for ci, cond in enumerate(conditions):
        trials.extend((ci, cond) for _ in range(cfg.n_eval_per_cond))

    cols = {k: [] for k in
            ("condition_idx", "orientation", "contrast", "dispersion")}
    r_bars: list[np.ndarray] = []
    rt_parts: list[np.ndarray] = []
    xfinal_parts: list[np.ndarray] = []
    var_choice = {v: [] for v in VARIANT_NAMES}
    var_mean_si = {v: [] for v in VARIANT_NAMES}
    var_si_full = {v: [] for v in VARIANT_NAMES}

    for start in range(0, len(trials), _EVAL_BATCH):
        batch = trials[start:start + _EVAL_BATCH]
        trajs = animal.v1.present([c for _, c in batch], rng)
        r_bars.append(trajs.mean(axis=1).astype(np.float32))
        si_by_variant = all_variant_si(trajs, proto_go, proto_nogo,
                                       bundle_go, bundle_nogo, kappa, cfg.si_eps)
        for variant in VARIANT_NAMES:
            si = si_by_variant[variant]
            choice, rt, x_trace = accumulate(si, cfg, rng)
            var_choice[variant].append(choice)
            var_mean_si[variant].append(si.mean(axis=1))
            var_si_full[variant].append(si.astype(np.float32))
            if variant == "prototype":      # the behavioural readout
                rt_parts.append(rt)
                xfinal_parts.append(x_trace[:, -1])
        for ci, cond in batch:
            cols["condition_idx"].append(ci)
            cols["orientation"].append(cond.orientation_deg)
            cols["contrast"].append(cond.contrast)
            cols["dispersion"].append(cond.dispersion_deg)

    out = {k: np.asarray(v) for k, v in cols.items()}
    out["r_bar"] = np.concatenate(r_bars, axis=0)
    out["rt"] = np.concatenate(rt_parts)
    out["x_final"] = np.concatenate(xfinal_parts)
    out["kappa_vmf"] = float(kappa)
    for variant in VARIANT_NAMES:
        out[f"choice_{variant}"] = np.concatenate(var_choice[variant])
        out[f"mean_si_{variant}"] = np.concatenate(var_mean_si[variant])
        out[f"si_full_{variant}"] = np.concatenate(var_si_full[variant])
    # The prototype variant is the network's behavioural readout.
    out["choice"] = out["choice_prototype"]
    out["mean_si"] = out["mean_si_prototype"]
    out["si_full"] = out["si_full_prototype"]
    out["var_si"] = out["si_full_prototype"].var(axis=1)
    return out


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------


@dataclass
class AnimalRun:
    """Everything produced for one animal."""

    animal_id: int
    traits: dict[str, float]
    train_log: dict
    eval: dict
    neural_io: NeuralIdealObserver
    phase1_length: int
    bundle_go: np.ndarray       # (K, n_steps, n_v1) exemplar trajectories
    bundle_nogo: np.ndarray


@dataclass
class CohortRun:
    """The full cohort plus the (animal-independent) stimulus ideal observer."""

    runs: list[AnimalRun]
    stimulus_io: StimulusIdealObserver
    cfg: Config


def animal_seeds(base_cfg: Config) -> np.ndarray:
    """Deterministic per-animal seeds for a cohort."""
    base_rng = np.random.default_rng(base_cfg.master_seed)
    return base_rng.integers(1, 2 ** 31 - 1, base_cfg.n_animals)


def train_one_animal(animal_id: int, base_cfg: Config, seed: int) -> AnimalRun:
    """Build, train, evaluate one animal and fit its neural ideal observer."""
    animal = Animal.build(animal_id, base_cfg, seed)
    train_log = train_animal(animal, np.random.default_rng(seed + 1))
    eval_data = evaluate_animal(animal, np.random.default_rng(seed + 2))
    neural_io = NeuralIdealObserver.fit(
        animal.v1, animal.cfg, np.random.default_rng(seed + 3))
    return AnimalRun(
        animal_id=animal_id, traits=animal.traits, train_log=train_log,
        eval=eval_data, neural_io=neural_io,
        phase1_length=train_log["phase1_length"],
        bundle_go=animal.bundle_go, bundle_nogo=animal.bundle_nogo)


def run_cohort(base_cfg: Config, verbose: bool = True) -> CohortRun:
    """Build, train and evaluate the whole cohort of simulated animals."""
    seeds = animal_seeds(base_cfg)
    runs: list[AnimalRun] = []
    for i in range(base_cfg.n_animals):
        run = train_one_animal(i, base_cfg, int(seeds[i]))
        runs.append(run)
        if verbose:
            acc = run.train_log["correct"][-200:].mean()
            print(f"  animal {i:2d}: phase1={run.phase1_length:4d} "
                  f"trials, final-200 train acc={acc:.3f}")
    return CohortRun(runs=runs, stimulus_io=StimulusIdealObserver(base_cfg),
                     cfg=base_cfg)
