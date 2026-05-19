# Audit — consumers of `KLs` / `Losses` and the entropy-penalty contamination

**Date:** 2026-05-19
**Trigger:** During planning for the checkpoint sanity-check, noticed that
`evaluate_model_entropy` returns `fit_loss + entropy_penalty` (the training
objective) at test time. The penalty is a training-time regulariser to push
per-bin sampling outputs toward sharpness and has no place in a held-out test
metric. The saved `KLs` field in every `.mat` produced by
`run_experiment.run_animal_decoder` therefore carries this contamination for
temporal (sampling) models. Spatial (PPC) models are unaffected — the PPC
branch of `custom_loss_all_H` hardcodes `penalty = 0.0`.

## Scope of contamination — production `KLs`

Two files in active `nn_decoder/` reference `KLs`:

1. `run_experiment.py:388` — **writer**. Returns
   `{'KLs': Losses, ...}` where `Losses[key]` is the per-trial array of
   `total_loss = fit_loss + λ·H(pred_probs)` values produced by
   `evaluate_model_entropy`.
2. `plot_post_fix_performance.py:89, 95–98, 143, 145–148` — **reader,
   contaminated**. Loads `v['KLs']` from `.mat`, then `kls['temp']`,
   `kls['temp_shf']`, `kls['spat']`, `kls['spat_shf']`. Computes
   `temp_mean / temp_shf_mean` as the headline normalised loss. The
   penalty contaminates both numerator and denominator, but it does not
   cancel because `H(pred_temp) ≠ H(pred_temp_shf)` in general — the
   unshuffled model learns to be sharper, so the shuffled control
   inherits a larger penalty term, biasing the reported ratio *downward*
   for temporal bars (i.e. temporal models look slightly better than
   they are).

Two legacy files (`legacy/decoder_recovery_v26.py`,
`legacy/plot_crossover_interactions.py`) reference `KLs` but are dead per
the audit plan.

## What is NOT contaminated

`generate_all_fixed_plots.py → decoder_plotting_utils.py` re-derives every
loss number from the saved `Dist[arch]['target']` and `Dist[arch]['decoded']`
arrays via `calc_pca_dist` (or its MSE/CE equivalent). The penalty term was
never written into those arrays — only into the scalar `KLs` field. Concrete
sites that re-derive cleanly:

- `decoder_plotting_utils.get_mouse_pca_losses` (line 101)
- `decoder_plotting_utils.plot_normalized_performance_with_lines` (lines
  297, 335)
- `decoder_plotting_utils.plot_temporal_dynamics` (lines 386, 387, 393,
  397, 407)
- `decoder_plotting_utils.plot_*_ambiguity_*` and mouse-scatter blocks
  (lines 823, 868)

Search confirms: ripgrep finds zero `KLs` references in
`decoder_plotting_utils.py`. Every plot reached through
`generate_all_fixed_plots.py` is unaffected.

## Other `custom_loss_all_H` callers

These callers correctly pass `entropy_lambda=0.0` when computing test/eval
loss, so they avoid the contamination at the source:

- `recovery_convergence_probe.py:216` — `custom_loss_all_H(p, y, 0.0, ...)`
- `optuna_per_target.py:292` — `custom_loss_all_H(p, y, 0.0, ...)`
- `legacy/optuna_universal_cv_v26.py:168, 211`
- `legacy/optuna_joint_v27.py:288`
- `legacy/optuna_phase2_sbc_lambda.py:161`

The unique offender is `nn_classifier.evaluate_model_entropy:185`, which
passes the actual `entropy_lambda` through to `custom_loss_all_H`. That is
the line `run_experiment.run_animal_decoder:350` calls to populate
`Losses[key]`, and it is the sole upstream source of the contamination.

## Magnitude

With production `entropy_lambda = 0.003`:
- 91-D angle targets (Q, L, stim_cat): penalty ≤ `0.003·log(91) ≈ 0.014`
- 2-D decision targets (d): penalty ≤ `0.003·log(2) ≈ 0.002`

Direction of bias on `plot_post_fix_performance.py` ratios depends on
`H(pred_temp_shf) − H(pred_temp)`. Likely positive (shuffled targets train
flatter predictors), making temporal bars look slightly better than truth
by a few percent for 91-D targets, well under 1% for 2-D targets.

## Why `plot_post_fix_performance.py` exists at all

Partial-migration artefact. The writer side was migrated to
`results/<run_name>/<slug>/<split>.mat` by `training/run.py`. The reader
side wasn't: `decoder_plotting_utils.load_results_dict` still reads the
old flat-named `population_results_*` files via
`paths.fit_path_from_stem`. `plot_post_fix_performance.py` was written as a
one-off bridge to read the new tree, duplicating loading and normalisation
logic that already lives in `decoder_plotting_utils.py`. The contamination
crept in via that duplication.

The proper consolidation (tasks #13 and #14):

1. Add `load_run_tree(run_name, slug, splits)` to
   `decoder_plotting_utils.py` — companion to `load_results_dict` — so
   the existing clean plotters can read the new layout.
2. Rewrite `plot_post_fix_performance.py` as a thin grid driver that
   calls the existing utils six times (one per target × loss × basis cell)
   and assembles them into the 2×3 grid. The duplicated loaders go away,
   and the contamination bug disappears for free because the existing
   utils recompute loss from the `Dist` arrays.

## Recommended cleanup order

1. (#13) Add `load_run_tree`.
2. (#14) Rewrite `plot_post_fix_performance.py` as a thin driver. Regenerate
   `spat_vs_temp_post_fix.png` from the existing `.mat` files; the new
   figure is the correct one.
3. (#9–11) Split `fit_loss` / `entropy_penalty` in `custom_loss_all_H` and
   `evaluate_model_entropy`. Save both separately in future `.mat`
   outputs. Future runs no longer carry the contamination at the source.
4. (#12) Add a GOTCHAS.md entry documenting the contamination period
   (everything before the #11 commit), noting which fields are affected
   (`KLs['temp']`, `KLs['temp_shf']`) and how to recover clean values
   from older files (`calc_pca_dist(Dist['temp']['target'],
   Dist['temp']['decoded'], pcs, evar)`).

## Backward compatibility

No existing `.mat` file needs to be re-run. The clean fit-loss for every
historical temporal model can always be recovered from the saved
`Dist[arch]['target']` and `Dist[arch]['decoded']` arrays, because the
penalty was only ever added to the scalar `KLs` field, never written into
the arrays themselves.
