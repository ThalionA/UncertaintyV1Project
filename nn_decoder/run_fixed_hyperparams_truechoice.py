# -*- coding: utf-8 -*-
"""True choice decoder — trained CE against the animal's actual goChoice.

Thin wrapper around ``training.run.run_config``. Per-target defaults
live in ``training.config._PRESETS['choice']`` and are documented there.
Output goes to ``results/<run_name>/choice_CE_<window>_<bin>ms/<split>.mat``
plus ``config.yaml``.

This is the corrected version — the historical
``run_fixed_hyperparams_choice.py`` (now in ``legacy/``) targeted the
IO's decision_posterior rather than the actual choice, which capped its
choice-prediction accuracy at the IO baseline. The ``'choice'`` target
in the new Config bypasses that mistake.
"""

from __future__ import annotations

from training import default_config_for_target, run_config


RUN_NAME = 'production_full_targets_v1'
BIN_SIZES_MS = (100,)


def main(run_name: str = RUN_NAME,
         bin_sizes_ms=BIN_SIZES_MS):
    print(f"Starting True Choice Decoder run: {run_name}")
    for bs in bin_sizes_ms:
        cfg = default_config_for_target(
            'choice', run_name=run_name, bin_size_ms=bs,
        )
        run_config(cfg)


if __name__ == '__main__':
    main()
