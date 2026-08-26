# Sharing the nn_decoder fitting core

What to hand a collaborator who wants to see how the decoder is fitted, without
the ~78k lines of analysis and plotting around it. Derived 2026-08-26 from the
actual import closure of the fit path, not from memory.

## The bundle

**Tier 1 — the fit path. 8 files, 3,478 lines.** This is the complete import
closure of `training.run_config` → `run_experiment.run_animal_decoder`; nothing
else in the repo is needed to fit a decoder.

| Lines | File | What it is |
|---|---|---|
| 969 | `nn_classifier.py` | The model, the six losses, `fit_model`, early stopping, restart selection. **The heart.** |
| 796 | `run_experiment.py` | `run_animal_decoder` — per-animal spine: load → split → z-score → PCA basis → shuffle control → train 4 arms → evaluate. |
| 700 | `training/config.py` | `Config` dataclass: every training decision in one place, with per-target presets, validation and the directory-slug convention. |
| 414 | `utils.py` | Data loading (`load_vr_export`), stratified/generalisation splits, synthetic target generators. |
| 304 | `training/run.py` | `run_config` driver: runs a Config over (mice × splits), writes shards + merged `.mat` + `config.yaml` provenance. |
| 155 | `paths.py` | Single source of truth for paths and canonical filenames. |
| 118 | `training/targets.py` | `make_target` — builds the supervised label array per target type. |
| 22 | `neural_dataset.py` | Thin `torch.utils.data.Dataset`. |

**Tier 2 — scoring. 2 files, 459 lines.** Not needed to *run* a fit
(`nn_classifier` carries its own torch PCA loss), but needed to judge one, and
required by the tests below.

| Lines | File |
|---|---|
| 268 | `decoder_metrics.py` |
| 191 | `pca_loss.py` (numpy twin of the training-side PCA loss; the tests pin their agreement) |

**Optional — `io_hmm_data.py` (686 lines).** Only if they want the IO-HMM
targets. It is lazily imported (`run_experiment.py:366`), so the core runs
without it.

**Tests — 11 files, 2,863 lines, and every one is data-free.**
`test_fit_model.py`, `test_loss_split.py`, `test_early_stopping.py`,
`test_training_config.py`, `test_training_targets.py`, `test_pca_loss.py`,
`test_shard_merge.py`, `test_checkpoint_plumbing.py`, `test_neural_pca.py`,
`test_paths.py`, `test_decoder_metrics_rows.py`.

## Why the tests are the best part of the bundle

Measured 2026-08-26: **212 tests, all synthetic, pass in 9.4 s with no data
file present.** A collaborator can clone the bundle and immediately run the
whole fitting core end-to-end. That matters because the alternative front door
is a 532 MB export they may not be able to have.

```bash
OMP_NUM_THREADS=1 python -m pytest tests/ -q
```

(`OMP_NUM_THREADS=1` is not optional on macOS — the suite segfaults under
multi-threaded BLAS. See `GOTCHAS.md`.)

What those tests actually pin, which doubles as a spec of the core: `fit_model`
is bit-equivalent to a reference training loop for **PCA/MSE/CE/JS/KL ×
spatial/temporal**, including trailing-minibatch weighting; the loss decomposes
exactly as `total = fit + penalty`; early stopping restores best weights and is
independent of the global RNG; restart selection honours val/train/fallback;
the numpy and torch PCA losses agree numerically; every `Config` field reaches
the trainer.

## Reading order

1. `training/config.py` — read the `Config` docstrings first. Every decision the
   pipeline makes is a documented field here, so this doubles as the methods section.
2. `run_experiment.py::run_animal_decoder` — the spine, top to bottom.
3. `nn_classifier.py::fit_model` — the training loop, then `custom_loss_all_H`
   for the six objectives.
4. `tests/test_fit_model.py` — the executable specification of 2 and 3.

## Two things to tell them explicitly

**The four arms.** Every fit trains *four* decoders, not one: spatial and
temporal readouts, each with a trial-shuffled control. The shuffle arm is the
null that normalises the reported losses.

**Two eps conventions, ~2× apart.** The training losses add a float32 eps
inside the log, which saturates at ≈15.9 nats per confidently-wrong bin; the
scoring/diagnostic side historically clipped at 1e-12, which does not. On
peaked posteriors against broad targets these disagree by a factor of ~1.9
(measured: 17.83 vs 9.36 mean per-trial KL). `decoder_metrics.kl_rows` takes an
explicit `eps_mode` for exactly this reason. Any KL they quote needs the
convention stated alongside it.

## Dependencies and data

`torch`, `numpy`, `scipy`, `sklearn`, `pyyaml`. Nothing exotic, no repo-internal
build step.

The only data dependency is `data/VR_Decoder_Data_Export.mat` (532 MB) — the
neural + ideal-observer export. Sharing it is a data-governance decision, not a
code one. If it cannot be shared, the tests still give them a fully runnable core,
and `utils.generate_PPC_targets` / `generate_SBC_targets` show how synthetic
targets are constructed.

## What NOT to send

Everything else — `diagnostics/` (94 scripts), the ~30 `plot_*.py`, the
`run_*.py` sweep drivers, `legacy/`. They are analysis and provenance for
particular runs, and they will bury the 3.5k lines that actually matter.
