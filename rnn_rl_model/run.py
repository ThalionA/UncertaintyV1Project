"""End-to-end orchestrator: train the cohort, score it, make every figure.

    python -m rnn_rl_model.run            # full cohort
    python -m rnn_rl_model.run --quick    # small/fast smoke run

Outputs land in ``rnn_rl_model/figures/`` (PNG+SVG) and a results pickle +
summary JSON in ``rnn_rl_model/results/`` (both gitignored).
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np

from si_network_model.config import Config

from .cohort import CONDITIONS, cohort_table, run_cohort
from .config import RLConfig
from .paths import FIGURES_DIR, RESULTS_DIR, ensure_dirs
from .plots import make_all_figures
from .rd_adapter import run_rd_on_cohort


def make_config(quick: bool) -> RLConfig:
    if quick:
        return RLConfig(base=Config(n_v1=120, io_samples_per_cond=80),
                        n_train_trials=3200, n_eval_per_cond=25, n_animals=3)
    return RLConfig(base=Config(n_v1=160), n_train_trials=6000,
                    n_eval_per_cond=40, n_animals=6)


def _cohort_summary(results) -> dict:
    tab = cohort_table(results)
    keys = ["network_accuracy", "neural_io_accuracy", "decision_efficiency",
            "si_vs_logodds_r", "cos_readout_template", "cos_readout_whitened",
            "mean_abs_si", "lapse_rate"]
    out = {}
    for cond in CONDITIONS:
        out[cond] = {k: {"mean": float(np.nanmean(tab[cond][k])),
                         "sem": float(np.nanstd(tab[cond][k]) / np.sqrt(len(results)))}
                     for k in keys}
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    ensure_dirs()
    cfg = make_config(quick)
    print(f"RNN+RL similarity model — {'quick' if quick else 'full'} run "
          f"(n_animals={cfg.n_animals}, n_v1={cfg.n_v1}, "
          f"n_train={cfg.n_train_trials})")

    results = run_cohort(cfg)

    with open(RESULTS_DIR / "cohort.pkl", "wb") as f:
        pickle.dump(results, f)

    make_all_figures(results, cfg, FIGURES_DIR)
    rd = run_rd_on_cohort(results, FIGURES_DIR)

    summary = {"cohort": _cohort_summary(results), "rd": rd}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== cohort means (fixed | trained) ===")
    cs = summary["cohort"]
    for k in cs["fixed"]:
        print(f"  {k:22s} {cs['fixed'][k]['mean']:+.3f} | {cs['trained'][k]['mean']:+.3f}")
    print("\n=== RD readout tests (fixed | trained) ===")
    for k in rd["fixed"]:
        print(f"  {k:16s} {rd['fixed'][k]:+.4f} | {rd['trained'][k]:+.4f}")
    print(f"\nfigures -> {FIGURES_DIR}\nresults -> {RESULTS_DIR}\nDONE")


if __name__ == "__main__":
    main()
