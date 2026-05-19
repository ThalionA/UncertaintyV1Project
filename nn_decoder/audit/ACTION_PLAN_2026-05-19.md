# nn_decoder Action Plan — 2026-05-19

Forward-looking plan for the items left after the first audit pass. Reorganised
around engineering principles instead of by file, because the surface-level
candidates cluster around a small number of root causes — fixing the root
causes makes most of the surface candidates trivial.

Companion docs in the same folder:
- `REPORT_2026-05-19.md` — original audit, items annotated ✅ where done.
- `CONVENTIONS.md` — agreed canonical layout, false-alarm list.

## Status

**Phase A: ✅ COMPLETE (2026-05-19).** P1 (paths.py) and P6 (_v26 rename) shipped
together. `nn_decoder/paths.py` is now the single source of truth for
`population_results_*` and `recovery_cache_*` basenames; 11 callers route
through it. `run_experiment_v26.py`, `utils_v26.py`,
`neural_network_classifier_v26.py` renamed to drop the suffix; 33 affected
files swept (non-legacy and legacy/). Validated by `tests/test_paths.py` (22
tests green) and `python -m compileall` across the whole package.

**Phase B: 🔜 NEXT — P3 (collapse `run_*.py` wrappers).** Five root entry
points (`run_fixed_hyperparams.py`, `run_fixed_hyperparams_truechoice.py`,
`run_loss_sweep.py`, `run_post_fix_loadings.py`, `run_production_sweep.py`)
collapse into a single `run_sweep.py` + preset registry. `run_fixed_recovery.py`
folds onto the same path.

**Phases C and D pending.** Phase C's data moves (items 3.5, 3.6) are now
reduced to flipping `paths.LEGACY_FITS` and `paths.RECOVERY_CACHES` (one line
each) plus a `git mv` — the heavy lifting was eliminated by P1.

---

## How to read this document

Each section names an engineering principle, names the symptoms in this repo
that violate it, and proposes a fix that addresses the principle rather than
just the symptom. The original numbered audit items (2.1, 3.5, 4.1, …) are
cross-referenced so you can tell which surface candidate each fix dispatches.

A **phase plan** at the bottom orders the work by prerequisite (you can't sanely
move data without a paths module to point at) and by risk (refactors before
data moves, so tests catch breakage before you've also shuffled the files
they read).

---

## P1. ✅ DONE — Single Source of Truth — data path resolution

### Symptom

Every script that touches a fit output or a recovery cache hardcodes its own
path string with its own default. The grep:

- `population_results_*` is referenced as a hardcoded basename in
  **14 files**: `compare_all_choice_methods.py`, `decoder_plotting_utils.py`,
  `decoder_residual_partial_corr.py`, `decomposition_analysis.py`,
  `generate_all_fixed_plots.py`, `io_coherence.py`, `plot_io_coherence.py`,
  `plot_loss_sweep_comparison.py`, `recovery_convergence_probe.py`,
  `run_experiment_v26.py`, `run_fixed_hyperparams.py`, `run_fixed_recovery.py`,
  `run_loss_sweep.py`, `stim_mean_baseline.py`.
- `recovery_cache_*` is referenced in **3 files**.
- Each reader has its own `directory='.'` or `directory=HERE` default that
  must be kept in sync if the data moves.

This is why audit items **3.5** (move 32 `population_results_*.mat`,
~600 MB) and **3.6** (move 3 `recovery_cache_*.npy`, ~222 MB) are flagged
"medium-large blast radius." They aren't intrinsically expensive — they're
expensive *because* the file location is a fact spread across 17 places.

### Principle

There must be exactly one place in the codebase that knows where data lives.
Every caller asks that place; nobody hardcodes a basename.

### Fix

Create `nn_decoder/paths.py` (a few dozen lines) that exports pure functions:

```python
# nn_decoder/paths.py
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DATA_PROCESSED = REPO_ROOT / 'data' / 'processed'
FIGURES = HERE / 'figures'
RESULTS = HERE / 'results'           # training/run.py writes here
LEGACY_FITS = DATA_PROCESSED / 'population_results_pre_refactor'
RECOVERY_CACHES = DATA_PROCESSED / 'recovery_caches'

def population_results_mat(base: str, split: str) -> Path:
    """Resolve a pre-refactor fit. base ∈ {fixed_hyperparams,
    stim_mean_Q, fixed_truechoice, ...}."""
    return LEGACY_FITS / f"{base}_{split}.mat"

def recovery_cache(target: str) -> Path:
    return RECOVERY_CACHES / f"recovery_cache_fixed_{target}.npy"

def figures_dir(analysis: str) -> Path:
    return FIGURES / analysis
```

Then in callers:

```python
# before:
sio.loadmat(os.path.join(directory, f'population_results_fixed_hyperparams_{split}.mat'))
# after:
sio.loadmat(paths.population_results_mat('fixed_hyperparams', split))
```

Once every caller routes through `paths.py`, items **3.5 and 3.6 reduce to
moving the files and updating one constant.** Future moves are equally
cheap. **This is the single highest-leverage refactor on the open list.**

Estimated touch: 17 files, mostly 1–3 lines each plus a new ~80-line module.

**Dispatches audit items**: 3.5, 3.6, partially 3.4, partially 4.1, lays the
groundwork for 4.2.

**Outcome (2026-05-19):** `nn_decoder/paths.py` shipped. `FIT_BASENAMES` dict
plus `fit_stem` / `fit_path` / `fit_path_from_stem` / `recovery_cache` /
`figures_dir` resolvers. `LEGACY_FITS` and `RECOVERY_CACHES` point at
`nn_decoder/` for now; flipping them is the entirety of items 3.5 + 3.6's
remaining work. 11 callers migrated, module-level export shapes
(`PRODUCTION_TARGETS`, `PRODUCTION_PREFIXES`, `TARGET_PREFIXES`) preserved
byte-for-byte. Pinned by `tests/test_paths.py`.

---

## P2. Separation of Concerns — split the megamodules

### Symptom

Three modules are >900 lines each and mix orthogonal responsibilities:

| File | Lines | Mixed responsibilities |
|------|-------|------------------------|
| `population_metrics_vs_uncertainty.py` | 1324 | metric computation, statistics (partial-corr, BH, ridge CV), plotting (5 plot kinds), CLI |
| `feature_ablation_analysis.py` | 1747 | GPR/ridge CV, cluster-robust stats, ablation orchestration, reporting, CLI |
| `decoder_plotting_utils.py` | 990 | results loading + distance computation + 9 distinct plot functions |
| `similarity_analysis.py` | 2141 | (haven't fully scoped — likely similar) |
| `pca_posterior_vs_likelihood.py` | 1009 | (item **2.2**: orphan, but if we keep, also mixed) |

A function in `population_metrics_vs_uncertainty.py` like
`run_cv_regression(df, ucol, ulabel, …)` knows how to z-score, fit ridge,
bootstrap, AND draw the bar plot. That's three reasons to change.

### Principle

A module should have one reason to change. Computation, IO, and rendering
are three reasons.

### Fix — proposed split per megamodule

**`decoder_plotting_utils.py` → split into 2 files** (mostly mechanical):

```
decoder_io.py
  load_results_dict, calc_pca_dist, get_mouse_pca_losses,
  get_mouse_trials, _cached_raw_trials, _load_io_stage2_params,
  get_integrated_p_go, _lapse_corrected_p_go_from_posteriors
decoder_plots.py
  set_style, add_stat_annotation, all plot_* fns, calculate_within_mouse_stats
```

Now `decoder_io` is testable without matplotlib in the dependency graph
(useful — currently `import decoder_plotting_utils` pulls matplotlib for
anyone who only wants the loaders).

**`population_metrics_vs_uncertainty.py` → split into 3 files**:

```
population_metrics.py
  entropy_np, compute_raw_metrics, compute_zscored_metrics,
  compute_trajectory_metrics, compute_population_geometry,
  run_population_variance_pipeline, _zscore_per_neuron
population_stats.py
  benjamini_hochberg, pearson_spearman, partial_correlation,
  _residualize, _stimulus_design_matrix, cv_ridge_r2,
  cv_best_neuron_baseline_r2, _r2_score
population_plots.py
  set_plot_style, _annotate_corr, plot_pooled, plot_per_mouse,
  plot_partial_vs_raw, plot_residuals, plot_cv_regression_bar
population_metrics_cli.py   (or just keep the __main__ in the metrics file)
  run_cv_regression, run_cv_regression_blocks (orchestrators)
```

**`feature_ablation_analysis.py` → split into 3 files** similarly: a
`feature_ablation_models.py` (regressors), `feature_ablation_stats.py`
(cluster-robust paired t, BH adjustment), and `feature_ablation_cli.py`
(orchestration + reporting).

### Why this is worth doing now

The split makes the next round of cleanups trivial:

- Item **3.2** (`population_metrics_vs_uncertainty.py` writes 46 root files)
  becomes "edit `population_plots.py` to accept `out_dir`" — one file,
  not 1324 lines.
- Items **4.3** (consolidate 7 plot_*.py) and the orphans 2.x become
  legible — you can tell at a glance whether a file is a library, a
  pipeline step, or a one-off.
- Test coverage gaps stop being structural: it's hard to test a 1324-line
  file; easy to test a 400-line one.

**Hard rule**: split modules; do NOT change logic in the same commit.
Pure code-motion commits are easy to review and easy to revert.

**Dispatches**: 3.2, contributes to 4.3, makes 2.x easier to evaluate.

---

## P3. Explicit > implicit — orchestration boundaries

### Symptom

There are 6 entry-point scripts at root that all wrap
`training.run_config`:

```
run_fixed_hyperparams.py             41 lines  -> distributional Q, L, d
run_fixed_hyperparams_truechoice.py  36 lines  -> CE on true goChoice
run_loss_sweep.py                    48 lines  -> KL/JS/W on Q
run_post_fix_loadings.py            169 lines  -> post-bug-fix retrain matrix
run_production_sweep.py              92 lines  -> all 6 targets × 2 bin sizes
run_fixed_recovery.py               (item 2.6, uses OLD `run_experiment_v26`
                                      API directly — not via training/)
```

The first 5 are configuration disguised as code. They differ only in which
target tags, splits, bin sizes, and loss functions they pass to
`run_config`. A reader has to open each one to find out what it does.

The 6th (`run_fixed_recovery.py`) is worse: it bypasses the `training/`
wrapper that the other 5 use, so any pre/post-fit invariant added to
`training.run_config` is silently skipped by recovery runs.

### Principle

If two scripts differ only in their arguments to a shared function, they
are not two scripts — they are one script with two configurations.
Configuration belongs in declarative files, not code.

### Fix

**Step 1 — collapse the 5 wrappers into a CLI + preset registry**:

```python
# nn_decoder/training/presets.py
PRESETS = {
    'production': SweepConfig(targets=ALL_SIX, bin_sizes_ms=(50, 100),
                              splits=ALL_THREE, loss='default'),
    'fixed_hyperparams': SweepConfig(targets=('Q', 'L', 'd'),
                                     bin_sizes_ms=(100,), splits=ALL_THREE),
    'truechoice': SweepConfig(targets=('choice',), bin_sizes_ms=(100,),
                              loss='CE'),
    'loss_sweep': SweepConfig(targets=('Q',), losses=('Wasserstein', 'KL', 'JS')),
    'post_fix_loadings': SweepConfig(...),  # whatever this one does
}

# nn_decoder/run_sweep.py  — the ONLY root-level training entry point
#   python run_sweep.py production
#   python run_sweep.py loss_sweep --run-name 2026-05-19-Q-loss
```

The 5 root wrappers go to `legacy/` once `run_sweep.py production` etc.
work. The preset file is the single place a new researcher reads to
understand "what sweeps exist."

**Step 2 — fold `run_fixed_recovery.py` into the same scheme** (item
**2.6**). Either add a `'recovery'` preset, or accept that recovery is
genuinely structurally different (loops over `target` to load from cache,
not from data) and put it in a `recovery/` subpackage with its own CLI.
Either way, eliminate the second-orchestration-path.

**Dispatches**: 2.6, 4.3 (partial — the run_*.py wrappers are part of the
plot/pipeline overlap), provides a home for the `optuna_per_target.py`
config-baking step (item 2.5).

---

## P4. Pipeline vs probe — give exploratory code a different home

### Symptom

Items 2.1–2.5 are five scripts that look like production code but turn
out to be one-off probes:

- `lapse_rate_analysis.py` (1.8 KB) — exploratory correlation analysis
- `pca_posterior_vs_likelihood.py` (42 KB) — one-shot PCA comparison
- `quick_diagonal_loss_check.py` (5.4 KB) — debug utility
- `recovery_convergence_probe.py` (20.7 KB) — diagnostic, mentioned in
  GOTCHAS but no caller
- `optuna_per_target.py` (19.9 KB) — sweep that produced the frozen
  hyperparameters now baked into `training/config.py`

You can't tell by looking which is which. A new contributor reading
`recovery_convergence_probe.py` reasonably assumes it's part of the
production pipeline because nothing flags it otherwise.

### Principle

The directory a file lives in should say what role it plays. Production
code, exploratory probes, and one-shot diagnostics have different
maintenance contracts, so they should live in different folders.

### Fix

Introduce a `nn_decoder/probes/` subdirectory (or reuse the existing
top-level `playground/`). Move each item 2.x into the appropriate place:

| File | Disposition |
|------|-------------|
| `lapse_rate_analysis.py` | `probes/` — kept, intent explicit |
| `pca_posterior_vs_likelihood.py` (item 2.2) | `probes/` if you'll revisit; else `legacy/`. Also fixes item **3.4** since `pca_posterior_vs_likelihood_out/` becomes `probes/pca_posterior_vs_likelihood/figures/`. |
| `quick_diagonal_loss_check.py` (2.3) | Fold into `recovery_convergence_probe.py` as a `--quick` flag, OR keep in `probes/` |
| `recovery_convergence_probe.py` (2.4) | `probes/`. Add a NOTES entry pointing to when to run it (currently only mentioned in GOTCHAS) |
| `optuna_per_target.py` (2.5) | Special case — this is a **config-generating** script (its DBs become `training/config.py` hyperparams). Move to `training/sweeps/` with a docstring explaining the dependency relationship |

Add a short `probes/README.md`: "Scripts in this folder are exploratory.
They are NOT part of the reproducible pipeline; they are kept because
they document an investigation."

**Dispatches**: 2.1, 2.2, 2.3, 2.4, 2.5, 3.4.

---

## P5. Make illegal states unrepresentable — filename templates

### Symptom (item 4.1)

`nn_decoder/legacy_k10/` and `nn_decoder/` (root) both contain
`heuristics_per_mouse_GV_likelihood_variance.svg`, `cv_regression_*.csv`,
etc. — 62 filename collisions. Memory says root is the k=5 set since
2026-05-19 and legacy_k10/ is the archived k=10 set. The two are NOT
interchangeable.

Today's safeguard is "remember which directory you're in." That is not a
safeguard.

### Principle

A file's name should fully identify its content. Any parameter that
changes the content must appear in the name.

### Fix

Rename templates to include `k`:

```
heuristics_per_mouse_GV_likelihood_variance__k5.svg
heuristics_per_mouse_GV_likelihood_variance__k10.svg
```

Then both sets can live in the same directory and overwriting is
impossible. Workflow:

1. Update the template in `population_metrics_vs_uncertainty.py`
   (will become `population_plots.py` after P2) — add a `k=` argument
   threaded down from `GV_FIXED_K`. Per the memory note,
   `GV_FIXED_K=5` is the committed value, but the template should
   reflect it explicitly.
2. Bulk-rename the existing root files to add `__k5` and the
   `legacy_k10/` files to add `__k10`.
3. Move both sets into `figures/population_metrics/` (this overlaps
   with item **3.2**; do it as one combined motion).
4. Drop the `legacy_k10/` directory.

**Generalise the rule**: any time you find yourself naming an output
directory or a subdirectory to disambiguate filenames, the filename is
underspecified. Fix the filename.

**Dispatches**: 4.1, contributes to 3.2.

---

## P6. ✅ DONE — Versioning is git's job, not the filename's

### Symptom (item 4.2)

`run_experiment_v26.py`, `utils_v26.py`, `neural_network_classifier_v26.py`
are the canonical, current modules. No `v25` exists at root; `v27` is
already in `legacy/`. The suffix is a tombstone of when the code was
being iterated rapidly.

### Principle

Filenames identify the role of the code. Versions identify the state of
the code, which is git's job. Mixing the two means every file rename
shows up in every diff for the rest of the project's life.

### Fix

Rename in one commit:

```
run_experiment_v26.py        -> run_experiment.py
utils_v26.py                 -> utils.py     (or split: data_io.py, etc.)
neural_network_classifier_v26.py -> nn_classifier.py
```

Update the 20-ish import sites in the same commit. This is high-touch
but mechanical and unambiguous: `git grep -l v26 -- '*.py'` enumerates
the work. Best done **alongside the P1 paths refactor** since both
require sweeping imports anyway.

Note that `utils_v26.py` is generically named — when you touch it,
consider whether it's actually one thing or several (it has the
`load_vr_export` function plus other things). If several, split per P2.

**Dispatches**: 4.2.

**Outcome (2026-05-19):** Renamed (`run_experiment_v26.py → run_experiment.py`,
`utils_v26.py → utils.py`, `neural_network_classifier_v26.py → nn_classifier.py`)
and swept across 33 affected files including `legacy/`. Legacy filenames that
themselves carry `_v26` (e.g. `decoder_sanity_check_v26.py`) were not renamed,
so docstring references to those legacy filenames remain accurate. The "split
`utils.py` if it does too much" recommendation is deferred to P2.

---

## P7. Provenance — outputs should be traceable to inputs

### Symptom

Several output files at root have no clear producer:

- `residuals_decision_entropy.svg`, `residuals_likelihood_variance.svg`,
  `residuals_perceptual_variance.svg` (item **3.9**) — no grep match for
  the filename pattern in any current `.py`.
- `correlation_summary.csv` *was* re-created during a recent
  `population_metrics_vs_uncertainty.py` run, but the file's mtime
  and the producer's mtime can disagree silently.

### Principle

For any file in the repo, you should be able to answer: "what code, run
at what configuration, produced this?" If you can't, the file is either
dead (delete) or a provenance bug (fix the producer).

### Fix

For each output directory under `figures/<analysis>/`, the producer
should additionally drop a `manifest.json` (or `_meta.json` — there's
already one at `figures/similarity_framework/_meta.json`) containing:

```json
{
  "produced_at": "2026-05-19T10:15:00Z",
  "producer": "population_metrics_vs_uncertainty.py",
  "producer_commit": "<git-sha>",
  "config": { ... whatever args/flags drove this run ... }
}
```

Three small wins:

1. The orphan `residuals_*.svg` files (item 3.9) can be diagnosed:
   manifest missing → delete; manifest present → look at the producer.
2. Future audits don't need to grep for filenames; they just read the
   manifest.
3. Reruns can detect "this manifest is older than the producer code"
   and warn.

**Dispatches**: 3.9, helps with future audits.

---

## P8. Tests at module boundaries

### Symptom

Test coverage today:
- ✅ Has tests: `feature_ablation_analysis`, `time_binned_ppc`,
  `stim_mean_baseline`, `plot_fano_factor`, `decoder_loadings_comparison`,
  `decoder_residual_partial_corr`, `decomposition_analysis`,
  `population_metrics_vs_uncertainty` (partial), `training/config`,
  `training/targets`, `io_coherence`, `archetype_similarity`,
  `neural_heuristics`, `pca_basis` (new).
- ❌ No direct tests: `decoder_plotting_utils`, `pca_posterior_vs_likelihood`,
  `similarity_analysis`, `recovery_convergence_probe`,
  `generate_all_fixed_plots`, the 5 `run_*.py` wrappers.

The pattern: anything that's a thin orchestration wrapper, or a >900-line
file mixing plots with compute, doesn't get tested.

### Principle

Each *concern* gets its own tests. Plotting helpers don't need pixel-level
tests; computation helpers do.

### Fix

After P2's splits, the `*_io.py`, `*_stats.py`, and `*_metrics.py` halves
of each split should get small synthetic-data unit tests (per CLAUDE.md
TDD policy). The `*_plots.py` halves don't need new tests — they're
called by the integration test that runs `generate_all_fixed_plots` end
to end on a stub dataset.

For the run_*.py wrappers, after they collapse into `run_sweep.py + presets`
under P3, the presets are pure data and the runner is exercised by
existing `training/config` tests + one smoke test per preset that asserts
the right config flows through.

**Dispatches**: lowers the cost of future refactors; covers no audit item
directly.

---

## Phase plan

Order matters: do refactors before data moves, so tests catch breakage
before the data also moved.

### Phase A — Foundations (refactors only, no data moves)

1. **P1: `paths.py` module.** ✅ DONE 2026-05-19. Defined; 11 active
   callers switched. Pinned by 22 tests.
2. **P2: Split the three megamodules.** ⏳ Pending. Pure code motion — no
   logic changes. Update imports. Verify all existing tests still pass.
3. **P6: Drop `_v26` suffixes.** ✅ DONE 2026-05-19. Bundled with P1 as
   originally suggested.

### Phase B — Orchestration

4. **P3: Collapse the 5 `run_*.py` wrappers into `run_sweep.py` + presets**.
   Migrate `run_fixed_recovery.py` (item 2.6) onto the same path.
   Archive the old wrappers to `legacy/`.

### Phase C — Move data and outputs

5. **P5 (bundled with item 3.2): Fix filename templates** for the
   `legacy_k10`/k5 collision, then move all `figures/population_metrics/`
   outputs there. Drop `legacy_k10/`.
6. **Items 3.5 + 3.6**: Move `population_results_*.mat` to
   `data/processed/population_results_pre_refactor/` and
   `recovery_cache_*.npy` to `data/processed/recovery_caches/`. Update
   the constants in `paths.py`. Total code change: ~5 lines.
7. **Item 3.4**: Resolve `pca_posterior_vs_likelihood_out/` per its P4
   disposition (probes/ or legacy/).

### Phase D — Triage and labelling

8. **P4: Move items 2.1–2.5 to `probes/` (or `legacy/`)** depending on
   disposition. Write `probes/README.md`.
9. **P7: Add `manifest.json` emission** to the producer wrappers
   under `figures/<analysis>/`. Solve item 3.9 by manifest-check.
10. **P8: Backfill tests** for the newly-split IO and stats modules
    from P2.

### Deferred / consider later

- **Item 4.3** (consolidate the 7 `plot_*.py` files) is mentioned in the
  original report. After P2 most of these will themselves be split, so
  re-examine whether consolidation is still desirable. Likely answer:
  no — different analyses, different inputs; the "7 files" appearance
  was inflated by the megamodules.

---

## Quick reference — audit items to phases

| Item | Concern | Phase | Principle |
|------|---------|-------|-----------|
| 2.1 | lapse_rate_analysis.py orphan | D | P4 |
| 2.2 | pca_posterior_vs_likelihood.py orphan | D | P4 |
| 2.3 | quick_diagonal_loss_check.py orphan | D | P4 |
| 2.4 | recovery_convergence_probe.py orphan | D | P4 |
| 2.5 | optuna_per_target.py orphan | D | P4 |
| 2.6 | run_fixed_recovery.py uses old API | B | P3 |
| 3.2 | population_metrics writes 46 root files | A then C | P2 then P1 |
| 3.4 | pca_posterior_vs_likelihood_out/ dual home | D | P4 |
| 3.5 | population_results_*.mat (600 MB) at root | C | P1 ✅ (move pending: flip `paths.LEGACY_FITS`) |
| 3.6 | recovery_cache_*.npy (222 MB) at root | C | P1 ✅ (move pending: flip `paths.RECOVERY_CACHES`) |
| 3.9 | residuals_*.svg orphans | D | P7 |
| 4.1 | legacy_k10 filename collision | C | P5 |
| 4.2 | _v26 filename suffixes | A | P6 ✅ DONE 2026-05-19 |
| 4.3 | 7 plot_*.py consolidation | (re-evaluate) | P2 follow-on |

---

## Sizing

Rough effort estimate (one focused session each):

- Phase A: ~3 sessions (paths module + 3 module splits + v26 rename)
  - P1 + P6 ✅ completed in one session on 2026-05-19.
  - P2 (three module splits) still pending — second Phase A session.
- Phase B: 1 session
- Phase C: 1 session (reduced from "potentially scary" to "trivial" by P1)
- Phase D: 1–2 sessions

After Phase A alone, `nn_decoder/` will read like a small,
well-organised package even before any further file moves. That's the
right place to pause and reassess.
