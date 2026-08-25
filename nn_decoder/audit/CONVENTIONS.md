# nn_decoder Conventions

Recorded during the 2026-05-19 audit. The intent is that the next audit (or anyone reading this folder cold) doesn't have to re-derive these.

## Canonical layout

```
nn_decoder/
  *.py                       # Importable libraries + entry-point scripts.
  training/                  # The post-refactor orchestration API
                             # (`default_config_for_target`, `run_config`).
  audit/                     # This audit's report + conventions.
  legacy/                    # Archived code. Source-tree mirror of where it
                             # lived before archiving. Never deleted on first
                             # pass — kept for at least one full audit cycle.
  legacy/optuna_dbs_v27/     # Frozen Optuna study DBs + their plot SVGs
                             # from the pre-`optuna_per_target.py` workflow.

  figures/<analysis>/        # Canonical home for analysis figures + their
                             # accompanying numeric outputs (csv/npz).
                             # New plotting scripts MUST accept an --out-dir
                             # argument that defaults under figures/.
  results/<run_name>/<slug>/ # Canonical home for fit outputs (mat files)
                             # written by `training.run_config`. Pre-refactor
                             # `population_results_*.mat` may still live at
                             # nn_decoder/ root pending a separate audit pass
                             # (item 3.5).

  optuna_studies/            # Active per-target Optuna sweep outputs
                             # (`optuna_per_target.py`). The DBs here ARE
                             # the source of truth for the hyperparameters
                             # baked into `training/config.py` — do not
                             # delete without checking what's frozen.
```

## Versioning suffixes

**No `_v<n>` suffixes on canonical modules.** As of 2026-05-19, the three
remaining `_v26` modules have been renamed (`run_experiment_v26.py →
run_experiment.py`, `utils_v26.py → utils.py`, `neural_network_classifier_v26.py
→ nn_classifier.py`) as part of action plan P6. Filename versioning is git's
job; do not reintroduce numeric suffixes on top-level modules.

Legacy filenames inside `legacy/` that still carry `_v26`/`_v27` markers
(e.g. `decoder_sanity_check_v26.py`, `optuna_joint_v27.py`) were intentionally
left as-is — those filenames are themselves a snapshot of what the module was
called when it was archived, and any docstring/comment that references them
points to a real file.

## Data file locations

**`nn_decoder/paths.py` is the single source of truth for fit-output and
recovery-cache filenames.** Any caller that needs to load a
`population_results_*.mat` or `recovery_cache_*.npy` must route through:

- `paths.fit_path(protocol, target, split)` — resolve a fit `.mat`.
- `paths.fit_stem(protocol, target)` — canonical stem (no split, no ext).
- `paths.fit_path_from_stem(stem, split)` — escape hatch for tuple-iterating
  callers that already have a stem.
- `paths.recovery_cache(target)` — resolve a recovery `.npy`.
- `paths.FIT_BASENAMES` — `protocol → target → stem` mapping; iterate when
  you need to walk every (target, stem) pair.

Hardcoded basename strings anywhere in `nn_decoder/`, `tests/`, or `legacy/`
are a bug. The two directory anchors `paths.LEGACY_FITS` and
`paths.RECOVERY_CACHES` currently point at `nn_decoder/` itself; when items
3.5 / 3.6 ship they will be flipped to `data/processed/` subdirs and every
caller follows automatically.

## Rejections / false alarms

These looked like cleanup candidates but are intentional. The next audit
should skip them.

- **`decoder_plotting_utils.py` "dead" helpers (2026-08-25 audit, item X2)** —
  `add_stat_annotation`, `get_train_target_mean`, `get_mouse_trials`,
  `_per_mouse_cell_means`, `_stack_per_mouse_pivots`, `get_integrated_p_go`,
  `_load_io_stage2_params`, `_lapse_corrected_p_go_from_posteriors` have zero
  *external* importers but are all called internally by the module's exported
  plot functions. A dead-code grep for this module must include intra-module
  call sites, not just imports. NOT dead.

- **`compare_all_choice_methods.py` vs `compare_partial_corr_designs.py`** —
  Names rhyme; content is unrelated (one is loss-method comparison, the
  other is partial-corr design comparison). NOT a duplicate.

- **`optuna_studies/` directory** — Looks like off-path output, is actually
  the canonical home for production hyperparameter sweep state.

- **`nn_decoder/results/`** — Untracked because gitignored as bulk fit
  output. Sized in the hundreds of MB. Do not commit; do not move without
  updating `training/run.py`.

## Output-directory rule (post-2026-05-19)

Any new plotting or analysis script that writes to disk MUST accept an
`--out-dir` argument and default it under `figures/<analysis>/` (figures)
or `data/processed/<pipeline>/` (intermediate data per repo `CLAUDE.md`).
**No more bare-string `savefig("foo.svg")` or `to_csv("foo.csv")` in
scripts.** The audit identified ~50 root-level files generated by such
calls; do not regress.
