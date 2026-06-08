# -*- coding: utf-8 -*-
"""Fit the IO-HMM to real VR mouse behaviour.

Ties together the real-data loader (``io_hmm.data_io``), the frozen Stage-1
sensory params (from the MATLAB ``IOResults.mat`` fit), the v0 four-state model,
and the EM fitter. Runs per animal, decodes the latent-state path, and writes a
JSON summary + an ``.npz`` of fitted arrays.

The real export (``data/VR_Decoder_Data_Export.mat``) and ``data/IOResults.mat``
are gitignored, so this is meant to be run wherever those live. Stage-1 params
default to the MATLAB fit; pass ``--kappa-amp/--c-power/--d-power`` to override
(or to run without IOResults.mat).

Examples
--------
    python scripts/fit_real_data.py --animal Cb15
    python scripts/fit_real_data.py --all --no-velocity
    python scripts/fit_real_data.py --animal Cb21 \
        --kappa-amp 22 --c-power 1.10 --d-power 0.035   # manual Stage-1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_IO_HMM = os.path.join(_REPO_ROOT, "ideal_observer", "io_hmm")
if _IO_HMM not in sys.path:
    sys.path.insert(0, _IO_HMM)

import io_core            # noqa: E402
import states as states_mod   # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod     # noqa: E402
import data_io            # noqa: E402


def _resolve_stage1(args, animal, animal_index=None):
    if args.kappa_amp is not None:
        return emissions_mod.Stage1Params(
            kappa_amp=args.kappa_amp, c_power=args.c_power,
            d_power=args.d_power, kappa_min=args.kappa_min), "manual"
    s1 = data_io.load_stage1(animal, animal_index=animal_index,
                             ioresults_path=args.ioresults,
                             kappa_min_default=args.kappa_min)
    return s1, "IOResults"


def _summarize(animal, params, trials_list, paths, state_list, history,
               use_velocity):
    K = len(state_list)
    names = [s.name for s in state_list]
    n_trials = int(sum(len(t.choice) for t in trials_list))
    occ = np.zeros(K)
    for p in paths:
        for k in range(K):
            occ[k] += int(np.sum(p == k))
    occ = (occ / occ.sum()).tolist()
    summary = {
        "animal": animal,
        "n_sessions": len(trials_list),
        "n_trials": n_trials,
        "use_velocity": use_velocity,
        "final_log_lik": float(history[-1]),
        "n_em_iters": len(history),
        "state_names": names,
        "state_occupancy": dict(zip(names, occ)),
        "pi": params.pi.tolist(),
        "A": params.A.tolist(),
        "psych_per_state": params.psych_per_state,
        "vel_per_state": params.vel_per_state,
    }
    return summary


def fit_animal(animal, args, grids, state_list, animal_index=None):
    trials_list = data_io.load_animal_trials(
        animal, export_path=args.export, by_session=not args.no_session,
        standardize_velocity=True, with_velocity=not args.no_velocity,
        min_trials=args.min_trials)
    stage1, s1_src = _resolve_stage1(args, animal, animal_index)
    print(f"[{animal}] {len(trials_list)} sessions, "
          f"{sum(len(t.choice) for t in trials_list)} trials | "
          f"Stage-1 ({s1_src}): kappa_amp={stage1.kappa_amp:.3g} "
          f"c_power={stage1.c_power:.3g} d_power={stage1.d_power:.3g}")

    params, history = fit_mod.fit(
        trials_list, state_list, stage1, grids,
        n_restarts=args.n_restarts, max_iters=args.max_iters, seed=args.seed,
        use_velocity=not args.no_velocity)
    paths = fit_mod.viterbi_paths(params, trials_list, state_list, stage1, grids)
    summary = _summarize(animal, params, trials_list, paths, state_list,
                         history, not args.no_velocity)
    print(f"[{animal}] LL={summary['final_log_lik']:.1f} "
          f"({summary['n_em_iters']} iters) | occupancy "
          + ", ".join(f"{n}={summary['state_occupancy'][n]:.2f}"
                      for n in summary["state_names"]))
    return summary, params, paths, trials_list, stage1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--animal", help="single animal tag, e.g. Cb15")
    src.add_argument("--all", action="store_true", help="every animal in export")
    ap.add_argument("--export", default=None, help="path to VR_Decoder_Data_Export.mat")
    ap.add_argument("--ioresults", default=None, help="path to IOResults.mat")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "ideal_observer",
                    "io_hmm", "real_fit_out"), help="output dir")
    # model / EM controls
    ap.add_argument("--no-velocity", action="store_true",
                    help="choice-only emissions (default uses velocity)")
    ap.add_argument("--no-session", action="store_true",
                    help="treat each animal as one sequence (default per-session)")
    ap.add_argument("--plots", action="store_true",
                    help="save a per-animal diagnostic figure to <out>")
    ap.add_argument("--prior-strength", type=float, default=3.0)
    ap.add_argument("--n-restarts", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--min-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    # manual Stage-1 override (bypasses IOResults)
    ap.add_argument("--kappa-amp", type=float, default=None)
    ap.add_argument("--c-power", type=float, default=1.0)
    ap.add_argument("--d-power", type=float, default=0.0)
    ap.add_argument("--kappa-min", type=float, default=1.0)
    args = ap.parse_args()

    grids = io_core.IOGrids.default()
    state_list = states_mod.default_v0_states(
        grids, bimodal_prior_strength=args.prior_strength,
        with_velocity=not args.no_velocity)

    animals = (data_io.list_animals(args.export) if args.all else [args.animal])
    os.makedirs(args.out, exist_ok=True)
    summaries = []
    for animal_index, animal in enumerate(animals):
        try:
            summary, params, paths, trials_list, stage1 = fit_animal(
                animal, args, grids, state_list, animal_index=animal_index)
        except Exception as exc:  # one bad animal shouldn't sink the batch
            print(f"[{animal}] SKIPPED: {exc}")
            continue
        summaries.append(summary)
        np.savez(os.path.join(args.out, f"fit_{animal}.npz"),
                 pi=params.pi, A=params.A,
                 paths=np.array(paths, dtype=object),
                 history=np.asarray(summary["final_log_lik"]))
        if args.plots:
            import io_hmm_diagnostics
            fig_path = os.path.join(args.out, f"fit_{animal}.png")
            io_hmm_diagnostics.plot_animal_fit(
                animal, params, trials_list, paths, state_list, stage1, grids,
                fig_path, use_velocity=not args.no_velocity)
            print(f"[{animal}] diagnostic figure -> {fig_path}")

    out_json = os.path.join(args.out, "fit_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nWrote {len(summaries)} animal fit(s) -> {out_json}")

    # Cross-animal aggregation (table always; figure with --plots)
    if summaries:
        import io_hmm_diagnostics
        tbl = io_hmm_diagnostics.write_group_table(
            summaries, os.path.join(args.out, "group_table.csv"))
        print(f"Wrote group table -> {tbl}")
        if args.plots:
            grp = io_hmm_diagnostics.plot_group_summary(
                summaries, os.path.join(args.out, "group_summary.png"))
            print(f"Wrote group figure -> {grp}")


if __name__ == "__main__":
    main()
