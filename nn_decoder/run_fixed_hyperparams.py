# -*- coding: utf-8 -*-
"""Fixed-hyperparameter production run for the distributional targets
(Q, L) and the IO decision posterior (d).

Thin wrapper around ``training.run.run_config``. Per-target defaults
live in ``training.config._PRESETS`` and are documented there. Output
goes to ``results/<run_name>/<target>_<loss>_<window>_<bin>ms/<split>.mat``
plus a ``config.yaml`` provenance dump.

The legacy flat-named files (``population_results_fixed_hyperparams_<split>.mat``)
written by previous versions of this script remain on disk untouched
for backward-compat with analyses that haven't been updated yet to
read from the nested tree.
"""

from __future__ import annotations

from training import default_config_for_target, run_config


RUN_NAME = 'production_full_targets_v1'
TARGETS = ('Q', 'L', 'd')
BIN_SIZES_MS = (100,)   # add 50 once 100 ms baseline lands clean


def main(run_name: str = RUN_NAME,
         targets=TARGETS,
         bin_sizes_ms=BIN_SIZES_MS):
    print(f"Starting fixed-hyperparameter production run: {run_name}")
    print(f"  targets    : {targets}")
    print(f"  bin sizes  : {bin_sizes_ms} ms")
    for target in targets:
        for bs in bin_sizes_ms:
            cfg = default_config_for_target(
                target, run_name=run_name, bin_size_ms=bs,
            )
            run_config(cfg)


if __name__ == '__main__':
    main()
