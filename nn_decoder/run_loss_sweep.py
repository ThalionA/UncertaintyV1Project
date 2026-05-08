# -*- coding: utf-8 -*-
"""Loss-function sweep on the Q target (perceptual posterior).

Replaces the previous two scripts:
  - ``run_fixed_hyperparams_loss_sweep.py`` (Wasserstein only)
  - ``run_kl_js_probe.py``                  (KL and JS only)

Both retired to ``legacy/``. Their outputs (the
``population_results_fixed_hyperparams_{Wasserstein,KL,JS}_<split>.mat``
files) remain on disk untouched.

This sweep is a methodological probe, not part of the production
production target list. It tests whether the trained Q decoder's
performance on choice prediction is loss-dependent. Confirmed result
(Session 2026-05-06): all four losses (PCA, KL, JS, Wasserstein) are
within ~0.06 NLL of each other on the lapse-corrected choice task,
and none beat the stim-mean baseline. Wasserstein is the worst because
it permits more peak-position drift; PCA-Euclidean is the best because
it weights leading PCs of per-condition Q's, which encode peak position.
"""

from __future__ import annotations

from training import default_config_for_target, run_config


RUN_NAME = 'production_full_targets_v1'
LOSSES = ('Wasserstein', 'KL', 'JS')   # PCA already covered by run_fixed_hyperparams.py
SPLITS = ('stratified_balanced',)      # in-distribution probe is sufficient


def main(run_name: str = RUN_NAME,
         losses=LOSSES,
         splits=SPLITS,
         bin_size_ms: int = 100):
    print(f"Starting Q-target loss sweep: {run_name}")
    print(f"  losses : {losses}")
    print(f"  splits : {splits}")
    for loss in losses:
        cfg = default_config_for_target(
            'Q', run_name=run_name, bin_size_ms=bin_size_ms,
            loss_func=loss,
        )
        run_config(cfg, splits=splits)


if __name__ == '__main__':
    main()
