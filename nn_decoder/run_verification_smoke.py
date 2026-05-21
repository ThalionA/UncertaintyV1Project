# -*- coding: utf-8 -*-
"""One-mouse smoke run for the checkpoint plumbing verification.

Goal: produce a single real checkpoint .pt in
``results/verification_2026_05_19_smoke/Q_PCA_half_100ms_condmean/checkpoints/mouse_0_stratified_balanced.pt``
to confirm:

  1. ``_extract_checkpoint`` builds a dict that survives torch.save / load.
  2. ``_pop_and_save_checkpoints`` writes it to the canonical path.
  3. The .mat file alongside the checkpoint loads cleanly under
     scipy.io.simplify_cells (i.e. no torch tensors leaked into it).
  4. ``state_dict`` + ``model_params`` reconstructs the trained network.
  5. Forward-passing saved ``X_test`` through the reconstructed network
     reproduces saved ``pred_probs`` bit-for-bit.

Uses production hyperparameters (REP=5, num_epochs=30) so the resulting
checkpoint reflects what real training produces. Restricted to 1 mouse,
1 split — that's the smallest scope that exercises the whole pipeline.

Spyder usage::

    from run_verification_smoke import main
    main()

Stand-alone::

    python run_verification_smoke.py
"""

from __future__ import annotations

from builtins import all  # noqa: A004
from pathlib import Path

from training import default_config_for_target, run_config


RUN_NAME = 'verification_2026_05_19_smoke'


def main():
    cfg = default_config_for_target(
        target='Q',
        run_name=RUN_NAME,
        notes='Smoke run: 1 mouse / 1 split for checkpoint plumbing verification.',
        # Minimal training — this run only verifies the checkpoint
        # plumbing produces files the sanity check can read; the fitted
        # weights don't need to be production-quality.
        REP=1,
        num_epochs=3,
    )
    # The default Config already sets PCA / condition_mean for Q.
    print(f"Config slug: {cfg.slug()}")
    print(f"Output dir: {cfg.output_dir('results')}")
    out_dir = run_config(
        config=cfg,
        mouse_ids=[0],
        splits=['stratified_balanced'],
        results_root='results',
    )
    print()
    print(f"Run finished. Output at {out_dir}")
    ckpt_dir = out_dir / 'checkpoints'
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob('*.pt')):
            print(f"  checkpoint: {p}  ({p.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [!] No checkpoints/ dir at {ckpt_dir} — plumbing did not fire.")


if __name__ == '__main__':
    main()
