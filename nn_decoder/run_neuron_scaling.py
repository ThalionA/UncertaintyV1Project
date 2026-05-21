# -*- coding: utf-8 -*-
"""Entry point: run the neural-population scaling sweep.

For each subset size N it trains the decoder on (a) random N-neuron
subsets and (b) targeted top/bottom-N subsets by three neuron rankings
(orientation tuning, decoder weight magnitude, mean activity). Target-
shuffled controls are recorded at every N for free. See
``neuron_scaling.py`` for the method and its caveats.

First production scope (per the 2026-05-20 meeting decision): target Q
(perceptual posterior), stratified_balanced split, all six mice.

Examples
--------
    # Default scope — Q, stratified_balanced, mice 0-5:
    python run_neuron_scaling.py

    # Quick correctness pass before the full run (cheap):
    python run_neuron_scaling.py --mice 0 --n-draws 2 --step 30

    # Re-run a mouse whose per-mouse CSV already exists:
    python run_neuron_scaling.py --mice 3 --force

This is a long job: each fit trains 4 models x REP repetitions, and the
sweep is (1 full + n_draws x grid + 6 targeted curves x grid) fits per
mouse. Per-mouse CSVs are written as each mouse finishes, so the run is
resumable — re-invoking skips mice that already have output.
"""

from __future__ import annotations

import argparse

from neuron_scaling import DEFAULT_STRATEGIES, run_scaling


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target', default='Q',
                        help="Decoding target short-name (default: Q).")
    parser.add_argument('--split', default='stratified_balanced',
                        choices=['stratified_balanced',
                                 'generalize_contrast', 'generalize_dispersion'],
                        help="Train/test split (default: stratified_balanced).")
    parser.add_argument('--bin-size-ms', type=int, default=100,
                        choices=[50, 100, 250],
                        help="Temporal bin size (default: 100).")
    parser.add_argument('--mice', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5],
                        help="Mouse ids to process (default: 0 1 2 3 4 5).")
    parser.add_argument('--step', type=int, default=10,
                        help="Neuron-count grid step (default: 10).")
    parser.add_argument('--n-draws', type=int, default=10,
                        help="Random subsets per size, for the averaged "
                             "random curve (default: 10).")
    parser.add_argument('--seed', type=int, default=0,
                        help="Base RNG seed; offset per mouse (default: 0).")
    parser.add_argument('--strategies', nargs='+', default=list(DEFAULT_STRATEGIES),
                        choices=list(DEFAULT_STRATEGIES),
                        help="Selection strategies to run "
                             f"(default: all of {list(DEFAULT_STRATEGIES)}).")
    parser.add_argument('--out-dir', default=None,
                        help="Output directory "
                             "(default: nn_decoder/results/neuron_scaling/).")
    parser.add_argument('--force', action='store_true',
                        help="Recompute mice that already have a per-mouse CSV.")
    args = parser.parse_args(argv)

    print("=" * 64)
    print("NEURON-POPULATION SCALING SWEEP")
    print(f"  target={args.target}  split={args.split}  "
          f"bin_size_ms={args.bin_size_ms}")
    print(f"  mice={args.mice}  step={args.step}  n_draws={args.n_draws}")
    print(f"  strategies={args.strategies}")
    print("=" * 64)

    agg = run_scaling(
        mouse_ids=args.mice,
        target=args.target,
        split=args.split,
        bin_size_ms=args.bin_size_ms,
        step=args.step,
        n_draws=args.n_draws,
        seed=args.seed,
        strategies=tuple(args.strategies),
        out_dir=args.out_dir,
        force=args.force,
    )

    print(f"\nFinished: {len(agg)} rows across {agg['mouse'].nunique()} mice.")
    return agg


if __name__ == '__main__':
    main()
