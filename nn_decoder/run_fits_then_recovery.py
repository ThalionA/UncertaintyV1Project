# -*- coding: utf-8 -*-
"""Chain: fixed-hyperparam fits -> Python parameter recovery.

Runs ``run_fixed_hyperparams.main()`` (fits Q, L, d across all three
splits under the current ``_PRESETS``) and then
``run_fixed_recovery.main()`` (spat/temp crossover on the stratified
balanced split using those fits as ground truth).

Each script remains independently runnable; this wrapper just
sequences them so a single Spyder cell or shell invocation covers
both stages.

Usage (Spyder)::

    %runcell -i run_fits_then_recovery.py

or from a shell::

    python run_fits_then_recovery.py
"""

from __future__ import annotations

from run_fixed_hyperparams import main as run_fits
from run_fixed_recovery import main as run_recovery


def main():
    print("=" * 60)
    print("STAGE 1/2: fixed-hyperparam fits")
    print("=" * 60)
    run_fits()

    print()
    print("=" * 60)
    print("STAGE 2/2: parameter recovery")
    print("=" * 60)
    run_recovery()


if __name__ == "__main__":
    main()
