# -*- coding: utf-8 -*-
"""Top-level orchestrator for the fresh production sweep.

Drives ALL six targets across both bin sizes (50 ms and 100 ms) and all
three production splits (stratified_balanced, generalize_contrast,
generalize_dispersion), under one shared ``run_name``.

Targets:
    Q           perceptual posterior (PCA-Euclid, 91-D)
    L           marginalised likelihood (PCA-Euclid, 91-D)
    d           IO decision posterior (MSE, 2-D)
    choice      animal's actual goChoice (CE, 2-D)
    stim_kernel Gaussian-smoothed delta at true theta (PCA-Euclid, 91-D)
    stim_cat    one-hot at true theta bin (CE, 91-D)

Output structure:
    results/<run_name>/
        Q_PCA_half_50ms/
            stratified_balanced.mat
            generalize_contrast.mat
            generalize_dispersion.mat
            config.yaml
        Q_PCA_half_100ms/
            ...
        L_PCA_half_50ms/
        L_PCA_half_100ms/
        d_MSE_half_50ms/
        d_MSE_half_100ms/
        choice_CE_half_50ms/
        choice_CE_half_100ms/
        stim_kernel_PCA_half_50ms/
        stim_kernel_PCA_half_100ms/
        stim_cat_CE_half_50ms/
        stim_cat_CE_half_100ms/

12 sub-directories total. Each holds 3 split .mat files plus config.yaml.

Total fits: 6 mice * 3 splits * 6 targets * 2 bin_sizes = 216 fits.
At ~1-2 minutes per fit on MPS this is roughly 3-7 hours wall-clock.
The driver runs sequentially; if any single (mouse, split) fails the
on_error='continue' default skips to the next mouse, so a transient
GPU failure doesn't abort the whole sweep.

Recommended workflow:
  1. Run per-target Optuna sweeps (``optuna_per_target.py --target=...``)
  2. Update ``training.config._PRESETS`` with the winners
  3. ``python run_production_sweep.py``  ← this script
  4. Update analysis modules to read from ``results/<run_name>/...``
  5. Re-run all comparison plots
"""

from __future__ import annotations

import argparse

from training import default_config_for_target, run_config


RUN_NAME_DEFAULT = 'production_full_targets_v1'
TARGETS = ('Q', 'L', 'd', 'choice', 'stim_kernel', 'stim_cat')
BIN_SIZES_MS = (50, 100)


def main(run_name: str = RUN_NAME_DEFAULT,
         targets=TARGETS,
         bin_sizes_ms=BIN_SIZES_MS):
    print(f"Production sweep: run_name={run_name!r}")
    print(f"  targets    : {targets}")
    print(f"  bin sizes  : {bin_sizes_ms} ms")
    print(f"  total dirs : {len(targets) * len(bin_sizes_ms)}")
    print(f"  total fits : {len(targets) * len(bin_sizes_ms) * 6 * 3} (6 mice * 3 splits)")

    for target in targets:
        for bs in bin_sizes_ms:
            cfg = default_config_for_target(
                target, run_name=run_name, bin_size_ms=bs,
            )
            run_config(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default=RUN_NAME_DEFAULT,
                        help='Output directory name under results/')
    parser.add_argument('--targets', nargs='+', default=list(TARGETS),
                        help='Subset of target types to run')
    parser.add_argument('--bin-sizes-ms', nargs='+', type=int,
                        default=list(BIN_SIZES_MS),
                        help='Bin sizes in ms')
    args = parser.parse_args()
    main(run_name=args.run_name, targets=tuple(args.targets),
         bin_sizes_ms=tuple(args.bin_sizes_ms))
