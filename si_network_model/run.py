"""End-to-end orchestrator for the SI-framework network model.

Runs two cohorts -- one under symmetric reward, one under asymmetric
(water-task-like) reward -- then computes metrics and writes all figures:
the core network-vs-ideal-observer results, the SI-variant comparison, the
learned trajectory archetypes, and the symmetric-vs-asymmetric criterion shift.

Resumable: each animal is checkpointed (with the Config it was produced
under, so config changes invalidate it). The run works within a per-
invocation wall-time budget; re-invoke until it reports DONE.

Run from the repo root:  python -m si_network_model.run
"""

from __future__ import annotations

import dataclasses
import pickle
import time

import numpy as np

from .analysis import summarize_cohort
from .config import Config
from .ideal_observer import StimulusIdealObserver
from .paths import FIGURES_DIR, RESULTS_DIR, ensure_dirs
from .plots import make_all_figures
from .train import CohortRun, animal_seeds, train_one_animal

_CKPT_ROOT = RESULTS_DIR / "checkpoints"
_TIME_BUDGET_S = 33.0  # per-invocation wall-time budget for training animals


def _ckpt_dir(mode: str):
    d = _CKPT_ROOT / mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_valid(mode: str, i: int, cfg: Config):
    """Saved AnimalRun if its checkpoint exists and matches ``cfg``, else None."""
    path = _ckpt_dir(mode) / f"animal_{i}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except Exception:  # noqa: BLE001 -- corrupt -> retrain
        return None
    if not isinstance(data, dict) or data.get("cfg") != cfg:
        return None
    return data["run"]


def _train_cohort_pending(mode: str, cfg: Config, t0: float) -> bool:
    """Train any stale/missing animals for one cohort; True if all done."""
    seeds = animal_seeds(cfg)
    for i in range(cfg.n_animals):
        if _load_valid(mode, i, cfg) is not None:
            continue
        if time.time() - t0 > _TIME_BUDGET_S:
            done = sum(_load_valid(mode, j, cfg) is not None
                       for j in range(cfg.n_animals))
            print(f"  [{mode}] time budget reached: {done}/{cfg.n_animals} "
                  f"done -- re-run to continue")
            return False
        t1 = time.time()
        run = train_one_animal(i, cfg, int(seeds[i]))
        with open(_ckpt_dir(mode) / f"animal_{i}.pkl", "wb") as fh:
            pickle.dump({"cfg": cfg, "run": run}, fh)
        acc = run.train_log["correct"][-200:].mean()
        print(f"  [{mode}] animal {i:2d}: phase1={run.phase1_length:4d} | "
              f"train acc={acc:.3f} | {time.time() - t1:.1f}s")
    return True


def _assemble(mode: str, cfg: Config) -> CohortRun:
    runs = [_load_valid(mode, i, cfg) for i in range(cfg.n_animals)]
    return CohortRun(runs=runs, stimulus_io=StimulusIdealObserver(cfg), cfg=cfg)


def _save_results(cohort: CohortRun, summary: dict) -> str:
    """Save the symmetric cohort's per-animal arrays and metrics."""
    blob: dict[str, np.ndarray] = {}
    for run in cohort.runs:
        i = run.animal_id
        blob[f"train_correct_{i}"] = run.train_log["correct"]
        for key in ("orientation", "contrast", "dispersion", "choice",
                    "rt", "mean_si"):
            blob[f"eval_{key}_{i}"] = run.eval[key]
    per = summary["per_animal"]
    for metric in ("network_acc", "neural_io_acc", "efficiency", "kappa",
                   "lapse_rate", "si_spread", "si_logodds_r", "phase1_length"):
        blob[f"metric_{metric}"] = np.array([a[metric] for a in per])
    for variant, vals in summary["variant_summary"].items():
        blob[f"variant_{variant}_efficiency"] = np.array(vals["efficiency"])
    path = RESULTS_DIR / "cohort_results.npz"
    np.savez_compressed(path, **blob)
    return str(path)


def main() -> None:
    ensure_dirs()
    cfg_sym = Config()
    cfg_asym = dataclasses.replace(cfg_sym, reward_mode="asymmetric")
    print("=" * 70)
    print("SI-framework network model -- end-to-end run (two cohorts)")
    print(f"  {cfg_sym.n_animals} animals/cohort | V1 {cfg_sym.n_v1} neurons | "
          f"symmetric + asymmetric reward")
    print("=" * 70)

    t0 = time.time()
    if not _train_cohort_pending("symmetric", cfg_sym, t0):
        return
    if not _train_cohort_pending("asymmetric", cfg_asym, t0):
        return

    sym = _assemble("symmetric", cfg_sym)
    asym = _assemble("asymmetric", cfg_asym)
    sym_summary = summarize_cohort(sym)
    asym_summary = summarize_cohort(asym)

    print("-" * 70)
    print("SYMMETRIC COHORT")
    print(f"  network accuracy            : {sym_summary['cohort_network_acc']:.3f}")
    print(f"  neural-IO accuracy (optimal): {sym_summary['cohort_neural_io_acc']:.3f}")
    print(f"  decision efficiency         : {sym_summary['cohort_efficiency']:.3f}")
    print("  SI-variant decision efficiency:")
    for variant, vals in sym_summary["variant_summary"].items():
        print(f"    {variant:18s}: {vals['efficiency']:.3f}")
    per = sym_summary["per_animal"]
    kappa = np.array([a["kappa"] for a in per])
    lapse = np.array([a["lapse_rate"] for a in per])
    if np.std(kappa) > 0:
        print(f"  cross-animal r(kappa, lapse): "
              f"{np.corrcoef(kappa, lapse)[0, 1]:+.2f}")
    print("ASYMMETRIC COHORT")
    print(f"  network accuracy            : {asym_summary['cohort_network_acc']:.3f}")
    print("-" * 70)

    print(f"results archive: {_save_results(sym, sym_summary)}")
    fig_paths = make_all_figures(sym, sym_summary, asym)
    print(f"{len(fig_paths)} figures written to {FIGURES_DIR}:")
    for p in fig_paths:
        print(f"  {p.rsplit('/', 1)[-1]}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
