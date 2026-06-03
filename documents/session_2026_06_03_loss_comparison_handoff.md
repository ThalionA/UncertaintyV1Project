# Session 2026-06-03 — Handoff: all-loss spat/temp comparison

Goal of the session: compare **all loss functions** for the V1 decoder — not
just the production PCA loss — for both architectures (spatial = PPC,
temporal = SBC), with early stopping, example trials with temporal bins,
per-mouse stats, and **test-time evaluation under every loss, not only PCA**.

Read top-to-bottom on a context reset. Everything below is on `main` (the repo
is trunk-only now — see CLAUDE.md; no branches).

---

## 0. TL;DR / state

- **The cluster run is the spine.** `run_loss_comparison.py` was extended from
  PCA/KL/JS to **all five real losses** (PCA, CE, KL, JS, Wasserstein; MSE
  dropped as degenerate) and runs **matched + early-stopped** across Q+L ×
  {50,100} ms × {full,half} × 3 splits = **40 slug dirs / 120 `.mat`**. It was
  ~24/40 slugs done at last check, still running in tmux `loss_cmp` on `gpu1`,
  resumable via `skip_existing`.
- **Local data**: 25 slug dirs rsync'd (all Q cells; only `L_*_full_50ms` of L
  so far — the run was mid-L when pulled).
- **Plotting is built and parameterised per cell.** One command,
  `python plot_all_cells.py --run-name loss_comparison_v1`, discovers every
  completed `(target, window, bin)` cell × split and emits the full figure
  suite. Re-runnable as the grid fills.
- **Headline finding**: the SBC>PPC architectural advantage is **real but small
  and metric-shaped** — strongest (and group-significant) under the *raw
  divergence metrics* (PCA, CE, KL), near-zero under Wasserstein, and limited at
  the group level by between-mouse inconsistency (3 of 6 mice carry it). The
  **PCA loss produces posteriors that are at chance under KL/CE** — a scale-free
  confirmation that it scores across-condition structure only.

---

## 1. The cluster run — `run_loss_comparison.py`

Matched comparison: one parameter set per (target, bin); only `loss_func`
varies; `entropy_lambda` pinned at 3e-3 across losses; early stopping on
(patience=15, min_epochs=20, cap 200, val_fraction=0.2); history + weight
snapshots exported. Each `.mat` already contains **both archs and their
shuffles** for all 6 mice (`Dist.{spat,spat_shf,temp,temp_shf}`).

Run / resume (gpu1, code pushed by rsync — the cluster copy is NOT a git
checkout, so `git pull` won't work there):

```bash
# laptop → push any code changes up first
rsync -avz nn_decoder/run_loss_comparison.py gpu1:~/UncertaintyV1/nn_decoder/

# gpu1
ssh gpu1 ; tmux attach -t loss_cmp        # or: tmux new -s loss_cmp
cd ~/UncertaintyV1/nn_decoder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/cluster-env/.venv/bin/python
$PY -u run_loss_comparison.py \
    --splits stratified_balanced generalize_contrast generalize_dispersion \
    2>&1 | tee loss_cmp_full.log
# resumes past finished (mouse, split) shards; Ctrl-B D to detach
```

Progress check: `ssh gpu1 'cd ~/UncertaintyV1/nn_decoder; ls
results/loss_comparison_v1 | wc -l'` (40 slug dirs when complete).

Pull down (laptop):
```bash
rsync -avz gpu1:~/UncertaintyV1/nn_decoder/results/loss_comparison_v1/ \
           nn_decoder/results/loss_comparison_v1/
```

---

## 2. Plotting — one capstone + five drivers

`results/` and `figures/` are gitignored; outputs stay local. All drivers take
`--target/--window/--bin/--split` and write to a **per-cell subdir**
`figures/loss_sweep_plots/<run>/<target>_<window>_<bin>ms[_<split>]/`.

**Capstone** — `plot_all_cells.py`: discovers cells, runs all suites per cell.
```bash
python plot_all_cells.py --run-name loss_comparison_v1
# subset/skip: --targets Q L  |  --skip scatter crossloss within_mouse spat_temp sweep
```

The five suites it calls:

1. **`plot_loss_sweep.py`** — the 9-figure cross-loss diagnostic suite
   (example posteriors, peakiness, per-bin SBC, cross-loss test-PCA bars,
   training/val curves with early-stop readout, weight norms, train-vs-test gap,
   and **posterior evolution for BOTH archs** — the SBC version overlays the
   per-time-bin posteriors under the time-averaged one). Pinned-cell globals
   `TARGET/WINDOW/BIN_MS/SPLIT` set by `generate_all`.

2. **`cross_loss_eval.py`** — test-time evaluation of every trained decoder
   under **every** loss (no retraining; re-applies `nn_classifier._batched_fit_loss`
   to saved decoded/target). Outputs:
   - `10_cross_loss_skill_{spat,temp}` — **skill = test_loss / shuffle_loss**
     (scale-free; <1 beats chance, 1 = shuffle). Green box = best training loss
     per eval metric.
   - `11_spat_vs_temp_diff` — skill difference spat−temp.
   - `12_spat_temp_paired_stats.csv` — per (train loss × eval metric) paired
     spat-vs-temp test (raw + skill, paired-t + Wilcoxon, n = mice).
   - `--value raw` for the legacy column-normalised absolute-loss view.

3. **`within_mouse_loss_plots.py`** — the full clean-run within-cell breakdown,
   **per loss** (PCA/CE/KL/JS/Wasserstein), reusing
   `generate_clean_run_plots.breakdown_one_cell` verbatim (that function was
   extracted from the PCA-gated driver so both share one implementation). 29
   figures per loss × {shuffle, variance, raw}: performance bars with
   within-mouse paired stats, per-mouse bars, orientation performance, ambiguity
   heatmaps, temporal dynamics, performance-vs-certainty, posterior examples,
   per-condition averages.

4. **`loss_scatter_spat_temp.py`** — per-trial **spatial-vs-temporal loss
   scatter**, one figure per loss, top row coloured by feature (orientation /
   contrast / dispersion / perceptual uncertainty = target entropy), bottom row
   per-feature-level **density contours** (zoomed to the bulk). `--metric own`
   (each loss's own metric) or `--metric PCA`; raw + shuffle-normalised.

5. **`plot_loss_spat_temp_comparison.py`** — the cross-loss spat/temp bars
   (clean-run style, all losses on/across one figure). Variants:
   - `14_..._{shuffle,raw}` — PCA distance, across-mouse, single axis,
     across-mouse paired-t.
   - `14_..._ownmetric_shuffle` — own metric, single axis (skill, unitless).
   - `14_..._ownmetric_facet_{shuffle,raw}` — own metric, **faceted per loss**
     (each panel its own raw scale), across-mouse paired-t. This is the
     own-metric/raw/across-mouse view.
   - `14b_..._{shuffle,raw}` and `14b_..._ownmetric_{shuffle,raw}` — **subpanels
     by loss**, each mouse a **trial-level** paired test (n = that mouse's
     trials).

---

## 3. Findings (Q / 100 ms / half / stratified unless noted)

1. **PCA loss scores at chance under KL/CE.** In the shuffle-normalised
   cross-loss skill matrix, PCA-trained (and Wasserstein-trained) decoders have
   skill ≥ 1 under the KL/CE metrics — no better than a label shuffle. They beat
   chance only on PCA and Wasserstein. KL/CE/JS are the generalists (skill
   0.4–0.9 under every metric). Direct, scale-free confirmation of the GOTCHAS
   note that the PCA loss measures across-condition variance only.

2. **Early stopping does not rescue the PCA loss.** A/B vs the non-early-stopped
   `new_loss_sweep_may27`: early stopping pushes PCA/Wasserstein *further* above
   chance under KL/JS (more miscalibrated); CE/KL ~unchanged; JS slightly
   improves. PPC overfits hard without ES (10–25% val degradation), SBC barely
   (<4%) — consistent across losses.

3. **Spat vs temp is real but metric-shaped and group-fragile.**
   - Skill (shuffle-normalised) spat−temp gap: largest under KL (+0.11..0.22),
     ~0 under CE, slightly reversed under Wasserstein.
   - Per-mouse **trial-level** tests: SBC<PPC is `**`/`***` for m0/m3/m4 (and
     m5 under KL/JS), null for m2 (low-loss outlier), reversed for m1 under KL.
   - The across-mouse group test only reaches significance for PCA (skill) — the
     between-mouse inconsistency (m1, m2, m5) caps n=6 power.
   - **Most sensitive group test: raw loss under own metric, faceted** — PCA,
     CE, and KL all significant; JS/Wasserstein not. Shuffle-normalisation adds
     between-mouse variance that washes out the CE/KL effect, so *raw* divergence
     loss is the cleaner group-level evidence.

4. **L weaker than Q.** On `L full 50 ms`, the SBC edge appears only for PCA on
   the stratified split and vanishes on the OOD splits — the dissociation is
   target-dependent.

---

## 4. Code changed this session (all on `main`)

Committed earlier: `run_loss_comparison.py` (extended to 5 losses, dc51bc8),
`cross_loss_eval.py` (created, 823d944; **then further modified** this session),
`generate_clean_run_plots.py` (`breakdown_one_cell` extracted, 384310f).

New / modified and committed in the handoff commit:
- `cross_loss_eval.py` — skill scores, spat/temp paired stats, `--target/...`
  cell flags + per-cell output, None-robustness for incomplete cells.
- `plot_loss_sweep.py` — temporal (SBC) posterior evolution with per-bin
  overlay; cell flags + per-cell output; OOD example-index bounds guard.
- `within_mouse_loss_plots.py` — per-loss full breakdown via `breakdown_one_cell`.
- `loss_scatter_spat_temp.py` — NEW, per-trial scatter + density companions.
- `plot_loss_spat_temp_comparison.py` — NEW, cross-loss spat/temp bars
  (PCA + own-metric, across-mouse + per-mouse trial-level, faceted variants).
- `plot_all_cells.py` — NEW capstone.

Memory: `memory/cross-loss-shuffle-eval.md` (the shuffle-normalised cross-loss
method + the PCA-at-chance finding).

---

## 5. Pending / next

- **Finish the cluster grid** (L cells beyond `full_50ms`, all OOD splits), rsync
  down, re-run `plot_all_cells.py`. Idempotent; just resume the same command.
- **L + OOD analysis**: with the full grid down, check whether the spat/temp and
  cross-loss-skill stories hold across bins/windows/splits, or are Q-specific.
- **Decision**: which loss to adopt as production. The cross-loss evidence
  favours **CE or KL** (generalists, calibrated under every metric) over PCA;
  but PCA is the historical basis and the stim_mean-baseline framing depends on
  it. Tie to the headline-framing decision in
  `documents/session_2026_05_16_handoff.md` §7.
- **n=6 ceiling**: the group spat/temp test is bottlenecked by between-mouse
  inconsistency, not trial count — no analysis fixes this.

## 6. Gotchas

- **Cluster has no git** — push code with `rsync`, not `git pull`.
- **`load_loss_sweep`/`cross_loss_eval` pin a single cell** via module globals
  `TARGET/WINDOW/BIN_MS/SPLIT`; the drivers set them — don't call the loaders
  bare expecting Q/100/half.
- **OOD splits have fewer test trials** (~287 vs 470) — any hardcoded example
  trial index must be bounds-guarded (fixed in `plot_example_posteriors`).
- **`decoded` is the held-out TEST set** (`run_experiment.py` ~L526), not train.
  The skill-matrix rows are the training *objective*; cell values are test loss.
- **MSE is excluded everywhere by design** (collapses to the marginal mean).
  Re-add only via an explicit flag/`--extra-mat`.
- **Own-metric raw can't share a y-axis** across losses (KL nats vs Wasserstein
  bins) — use the faceted figure (`14_..._ownmetric_facet_raw`).
- **Shuffle-normalisation adds between-mouse variance** — for group spat/temp
  tests, raw-under-own-metric is more sensitive than skill.
