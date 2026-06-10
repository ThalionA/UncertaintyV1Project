# PROJECT_LOG — single source of truth for cross-session tracking

**Start here on every new session.** This is the hub: an index of the living
documents, a rolled-up list of active open threads, and a reverse-chronological
session log. Detailed per-session handoffs live in their own files and are
linked from each entry (not duplicated).

**End of session ritual:** add a dated entry at the **top** of the Session log
below (newest first) — 3–6 lines + open items, linking any detailed handoff you
wrote. Fold persistent pitfalls into `GOTCHAS.md`, durable facts into
`memory/`. Keep this file the entry point.

---

## Active open threads (rolled up — prune as done)

**Live plan: [`nn_decoder/PLAN_2026-06-03_mate_followups.md`](nn_decoder/PLAN_2026-06-03_mate_followups.md)**
— the authoritative, prioritised roadmap after the 2026-06-03 Máté meeting
(supersedes the "next steps" tails of the recent handoffs). Top of it:

- **Tier A — DONE (2026-06-03).** A1 (`plot_weight_evolution_cell.py` figs D/E/F +
  best-val ★) and A2 (`posterior_pca_views.py`) built + run on `loss_comparison_v1`;
  meeting items 2–6 closed. Basis decision resolved → both (decoded + IO target).
- **Tier B (cluster)** — B1: hidden-width ablation `H∈{4,8,16,32,64}` (ladder **decided
  2026-06-09** — not going ≤2). Runner turnkey (`run_loss_comparison.py --hidden-sizes`);
  companion **`plot_overfit_vs_width.py` now written + smoke-tested** (reads each
  `hidden_ablation_h<H>` run's val−train total-loss gap at `best_epoch`, plots vs H per
  loss × spat/temp). **Launch queued — Theo runs it himself** (the agent's `ssh`/`rsync`
  to `gpu1` is blocked by the harness; exact rsync-up → tmux launch → rsync-down block is
  in the 2026-06-09 log entry). B2 (PPC `weight_decay` sweep) **deferred + NOT turnkey** —
  the runner has no `--weight-decay` flag (it's a fixed shared hyperparam), so B2 needs
  that flag added first.
- **Tier C (deferred)** — trained-as-target round-trip; stratified PCA basis;
  refill Wasserstein/JS gaps in `loss_sweep_h10_val_2026_05_27`; `pca_loss_demo`
  vs `diagnostics/loss_smoothness_demo` consolidation.

Other standing threads:
- **Finish the `loss_comparison_v1` cluster grid** (L cells beyond `full_50ms`,
  all OOD splits), rsync down, re-run `nn_decoder/plot_all_cells.py`. Resumable.
  [2026-06-03]
- **Decide the production loss**: cross-loss evidence favours CE/KL (calibrated
  generalists) over PCA (at chance under KL/CE); PCA is the historical basis and
  the stim_mean framing depends on it. Tie to headline framing. [2026-06-03]
- **Headline framing decision** (with Máté): (a) PPC-vs-SBC architectural — not
  recommended; (b) feature-ablation — narrower TV>order claim; (c) Similarity
  Framework — **now has decisive readout results** (2026-06-08): premise supported
  (choices read the template Δμ, not the whitened optimum, 6/6 mice), a measured
  discarded-covariance cost, Pred 16 partial (4/6), SBC wedge animal-dependent.
  See 2026-05-16, 2026-06-08. [open]
- **Task 2 done; Task 5 next**: residual-PCA retraining, if the residual
  partial-correlation result warrants (`documents/residual_partial_correlation.md`).
- **GPR + LOMO feature-ablation** run locally. [2026-05-16]
- **Methods-PDF rewrites** deferred until framing locked
  (`documents/methods_updates_required.md`).
- **n=6 ceiling**: group spat/temp test bottlenecked by between-mouse
  inconsistency, not trial count. [noted]
- **nn_decoder architecture cleanup — remaining items** (after the 2026-06-09 pass):
  #4 loss-sweep cluster dedup + the latent slug `pca_basis` gap (needs basis threaded
  through ~6 untested plotters; deferred — no condmean/residual runs exist); #9
  recovery/optuna shared `fit_pca_basis` reuse (optuna not installed locally → can't test
  `optuna_per_target`); `run_recovery_Q_spyder` to be `__main__`-guarded /
  `FORCE_RERUN=False` (runs destructively at import). [2026-06-09]

---

## Living documents (index)

Tracked-in-repo unless marked **[local]** (gitignored — present on Theo's
machine, absent on fresh clones / web sessions).

| Doc | What it is |
|---|---|
| `nn_decoder/PLAN_2026-06-03_mate_followups.md` | **Live roadmap** (Tier A/B/C + open decisions). The current "what next". |
| `GOTCHAS.md` **[local]** | Canonical list of methodological pitfalls. Add here, don't re-derive. |
| auto-memory (`~/.claude/projects/.../memory/`) | Cross-session facts: `MEMORY.md` index + `cross-loss-shuffle-eval.md`. User-level, outside the repo. |
| `CLAUDE.md` | Working agreement — **trunk-only, commit directly to `main`, no branches**. |
| `CLUSTER_RUNBOOK.md`, `SETUP_CLUSTER.md` | GPU cluster (`gpu1`, WSL2 on theo-desktop) setup + ops. |
| `README.md` | Repo overview. |
| `NOTES.md` **[local]** | **Archive** of pre-2026-05-17 session logs. Detail source for those dates; superseded by this hub going forward. |
| `documents/feature_catalog.md` | Per-trial feature inventory + the three Máté-question blocks. |
| `documents/residual_partial_correlation.md`, `documents/task2_residual_partial_corr_result.md` | Task 2: residualised partial-correlation analysis + result. |
| `documents/methods_updates_required.md` | Out-of-date sections of the methods PDF. |
| `documents/ideal_observer_methods_v3.tex`, `documents/*.tex`, `documents/methods_pdf.txt` | IO + methods manuscript sources. |
| `documents/Representation_of_Perceptual_Uncertainty_in_Mouse_V1.pdf`, `documents/mouse_uncertainty.pdf` | Manuscript / methods PDFs. |

Detailed handoffs **[local]**: `documents/session_2026_05_16_handoff.md` is
gitignored; the others (`documents/session_2026_06_03_*`,
`nn_decoder/HANDOFF_*`) are tracked. All are linked from the Session log below.

**Load-bearing code** (run tests after touching — see CLAUDE.md):
`nn_decoder/nn_classifier.py`, `run_experiment.py`, `training/config.py`,
`training/run.py`, `time_binned_ppc.py`.

---

## Session log (newest first)

### 2026-06-10 — Ideal-observer model schematic (theory → inference → fitting)
Built `nn_decoder/diagnostics/io_schematic.py` — one composed 3-region figure of the v2 two-stage
IO model, taking visual cues from Theo's PPT "Generative Model | Observer Inference" slide but
extended to this project's full pipeline. **Region 1 — generative model:** c→s, (c,d)→m ~
VonMises(s, κ(c,d)) with κ=(κmin+κamp)·c^pc·e^(−pd·d), bimodal task prior p(s) at 0/90°.
**Region 2 — observer inference:** Bayes m→p(s|m) (× prior); utility U(A,s)=[R_hit,R_miss,R_CR,R_FA]
=[1,0,.1,−.2] → EU(Go),EU(NoGo) → DV=ΔEU; *choice* branches from the log-odds g(m) (psychometric),
*kinematic confidence* (velocity, licks) branches from DV via y=β·DV+α+ε. **Region 3 — fitting &
inversion:** Stage 1 (sensory+emission {κamp,pc,pd,β,α,σ} from kinematics alone, BADS·hierarchical·
5-fold CV) → Stage 2 (4-param choice psychometric {α_r,β_r,γ_r,δ_r} on g(m), velocity-conditioned)
→ marginalised inversion (integrate over p(m|s,y)) → trial-by-trial Q(θ)/L(θ)/[P(Go),P(NoGo)]
targets → neural-decoder target + uncertainty read-outs U_perc=SD[Q], U_dec=H[P(Go)]. Reuses the
`peakiness_style` palette + `figsave.save_fig` (PNG ≤1600px **and** SVG; `layout=None` for the
hand-placed insets). Accurate to `documents/ideal_observer_methods_v3.tex` /
`wiki/Module_IdealObserver.md`. pyflakes clean. Figures gitignored (only the script is committed);
output `figures/schematic/io_model_schematic.{png,svg}`.
- **Open:** none — iterate on labels/colours if a specific slide/manuscript wants tweaks.

### 2026-06-10 — Removed the ×100 scalar from the PCA-loss calculation everywhere (Theo's request)
Dropped the arbitrary ×100 global scale from the PCA-weighted loss at all 9 sites — `pca_loss.pca_distance`
(covers `decoder_metrics.calc_pca_dist`), `decoder_metrics.variance_baseline`, `nn_classifier.custom_loss_all_H`
+ `fit_loss_per_trial` (the load-bearing training/eval branches), `optuna_per_target.marginal_baseline_loss`,
and the two toy scripts (`toy_peakiness_model`, `toy_width_matched`) — so the code matches the note's
equations. **Model-neutral:** the optimiser is always Adam (scale-invariant) ⇒ trained models unchanged;
existing runs on disk unaffected; skill/ratio metrics were already ×100-invariant. The `shape_lambda/100`
floor is **unchanged** (config knob = 100·λ, clean Brier weight λ = shape_lambda/100), so the width-matched
fix behaves identically. **Tests:** the numpy↔torch agreement test still passes; fixed the 4 hardcoded-scale
expectations (`test_pca_loss`: 80→0.8, 330→3.30; `test_variance_baseline`: two reference helpers). Full
suite: only the **pre-existing** `test_fit_model` 2-vs-3-tuple-unpack failure remains (unrelated). **Figures:**
raw-PCA-loss figures regenerated (y-scale ×100 smaller, shapes/ratios identical) — fig 12 why-overfitting,
fig 16 landscape, fig 25 per-mouse PCA raw row; fig 12 caption made value-free ("0.62→0.007" → "falling
toward zero"). Docstrings updated (pca_loss / config / run_experiment / nn_classifier / decoder_metrics).
Closes the open item from the overhaul session.

### 2026-06-10 — Peakiness note: 15-item figure + methods overhaul (Theo's worklist)
Worked through Theo's detailed worklist on the vault note `[[PCA-Peakiness-Mechanism]]` + the
generating code. **New real-data analyses:** `subspace_error_realdata.py` (toy fig-7 mechanism on
real V1 — PCA shape-error 38× KL's; §5); `uncertainty_scaling_realdata.py` (toy fig-8b noise sweep
on real data via per-trial dispersion/contrast — PCA over-confidence ratio 3.5×→5.8×; §6); the
hidden-width spat-vs-temp KL-skill axis in `plot_overfit_vs_width.py`; peakiness-vs-train/test-gap
scatter in `toy_width_matched._fig_why` (r=0.97). **Stats:** `spat_temp_performance.py` gained
per-mouse (n=trials, trial-level paired-t) + aggregate (n=6) figures for all losses & λ-variants,
AND the same under the **PCA metric** in 3 normalisations (raw/shuffle/variance, via
`decoder_metrics.variance_baseline`) showing the metric is blind per mouse; replaced the 3-image
fig 20. **PC-basis figure** (`pc_location_vs_shape.py`) reworked: verified real orientation is
**linear 0–90°** (199/200 edge-straddlers bimodal, not circular) → sequential cmap + arc; first-7-PCs
(killed a duplicate), coeff-variance methods, PC0/PC1 strips, 1/2/4/all-PC reconstruction. **Setup
schematic** redesigned (activity tile + trial-stack → MLP node-diagram → posteriors vs target).
**Methods** detailed (loss-basin $p_\gamma\propto t^\gamma$ family + R ratio; statement-(ii) evidence).
**Figure breathing:** `figsave.save_fig` now applies constrained layout by default (robust tight
fallback for the twinx ZeroDivisionError; `layout=None` opt-out), generous `peakiness_style.figsize`
margins — all multi-panel figures regenerated. **All 32 figures numbered Fig 1–30** (sequential;
cross-refs remapped). Commits `18d5a82 7b5fdbe b3aa282 f30176a 91e0938 5a964dd 2961e80 4774985` +
others; note is vault-side. **Open:** Theo proposed removing the ×100 scalar from the PCA loss
*calculation* (not just the note) — **verified safe** (optimiser is always Adam ⇒ global scale is
model-neutral; existing runs on disk unaffected; shape floor preserves the relative Brier weight),
**deferred** pending his go-ahead — touches load-bearing `nn_classifier.custom_loss_all_H` +
`pca_loss`, `decoder_metrics`, `optuna_per_target`, 2 toy scripts, and the numpy/torch agreement test.

### 2026-06-09 — B1 hidden-width ablation landed → §6 capacity + §9 width panels (vault note); figure-code audit triaged
B1 (hidden-width ablation, Q/half/100ms, H∈{4,8,16,32,64}, 5 losses, 6 mice) finished on gpu1 and
Theo rsync'd it down. Extended `plot_overfit_vs_width.py` with the **spat-vs-temp KL-skill-vs-H axis**
(`load_skill` reuses `cross_loss_eval._eval_one`; paired-t per width) and swapped the capacity-summary
2nd panel from train–val gap (Wasserstein's bin-unit scale squashed it) to **KL-skill**. **Headline
(strong):** decoded peakiness rises with H for PCA (0.11→0.21) and Wasserstein (0.13→0.25); CE/KL/JS
flat on the IO target at every width. **PCA's KL-skill crosses chance with capacity** — spatial 0.75
(H=4, *beats* chance) → 1.05 (H=8) → 1.29 (H=32) — so a tiny net stays below chance and capacity is
what tips PCA into worse-than-chance; calibrated losses beat chance (~0.55–0.64) at every width. The
best-val gap is high for PCA at all H (overfits even when tiny) — capacity controls how *far* it
over-sharpens, not whether. Spat-vs-temp: temporal PCA worse at every width (artefact is
capacity-independent); calibrated temporal≥spatial (small SBC edge) capacity-robust. **Vault note
`PCA-Peakiness-Mechanism`:** added the §6 "shrink the net → over-sharpening shrinks" subsection
(`peakiness_capacity_ablation.png`) + replaced the §9 placeholder with the spat/temp width panel
(`peakiness_spat_temp_width.png`); 30 figs synced, embeds resolve, LaTeX balanced.
**Figure-code audit (3 read-only Explore agents):** triaged — the flagged "critical shuffle-control
bug" (compare_loss_variants:125 / spat_temp_performance:118 score shuffled-decoded vs *real* target)
is a **FALSE POSITIVE**: verified `D[arch+'_shf']['target']` is allclose `D[arch]['target']`, so the
skill numbers stand — no re-run. `within_mouse_variants.py:128`'s `>1` guard is trial-level (hundreds),
not mouse-level — also fine. **Real fix:** `weight_evolution_variants.py` labeled variants λ=10 (config
units) → relabeled to **λ=0.1** + regenerated note **fig 26** (its legend visibly showed λ=10,
contradicting the §8 logit table). Lower-priority leftovers: duplication (`_entropy`×4, `split_kloc`×3,
`_paired_p`×3), toy ×100-scale wart.
- **Open / next:** (a) real-data **subspace-error decomposition** (toy fig 7 → real V1 from
  loss_comparison_v1) — central-mechanism comprehensiveness add, was mid-build when B1 landed; (b) the
  broader pass Theo asked for — nn_decoder architecture (the duplication cleanup), then IO observer,
  then IO HMM; (c) optional: extend the ablation to the L target.

### 2026-06-09 — PCA-peakiness report + figure-code audit and fixes
Reviewed `audit/PCA_PEAKY_POSTERIORS.md` → full report
`figures/loss_smoothness_demo/LOSS_SMOOTHNESS_REPORT.md` + generator
`diagnostics/loss_smoothness_demo.py`. Re-ran the demo: every headline number
still matches (Demo 1 entropies, Demo 2 KL≫JS≫PCA, Demo 2b collapse, basis
73%/PC8≈3.5e-5) **except** the JS too-sharp/too-broad width ratio, which is **4.6×**
now, not the 6× printed in the table (the figure's own annotation already said 4.6×)
— fixed. **Figure fixes:** (1) `fig7_bimodal_evolution` used `sharey=True`, so PCA's
0.5-tall spikes squashed KL/JS's recovered bimodal (~0.03 peak) into invisibility —
switched to per-panel y-scaling; verified KL/JS genuinely recover both modes
(H=3.90=target, dip at bin 47, far-mass 0.46). (2) Replaced the hardcoded
"(target 0.50)" with the measured far-mode mass 0.46 so KL/JS reads as *exact*
recovery. (3) Stripped the "Demo N —" prefixes from all figure suptitles (they said
"Demo 6/7" while the report calls the same panels "Demo 0/0b"). (4) Removed dead
`_normloss`. **Report refs:** many `nn_classifier.py`/`run_experiment.py` line
numbers were stale (PCA twin 208-210→192, mean-over-bins 164→146, entropy penalty
167→149, target-replicate 281-282→430); re-anchored on function names with corrected
hints, in both the report and the pointer file. Test `test_loss_smoothness_demo`
still passes. Files: `loss_smoothness_demo.py`, `LOSS_SMOOTHNESS_REPORT.md`,
`PCA_PEAKY_POSTERIORS.md`.

### 2026-06-09 — Spat-vs-temp performance with stats: audited engines, computed results, fixed λ-figure consistency
Theo wants spatial-vs-temporal **performance** compared **with stats** across everything (all
losses, variants, hidden-width). Audited the infra — **most exists**: `cross_loss_eval.py` →
`11_spat_vs_temp_diff` matrix + `12_spat_temp_paired_stats.csv` (paired-t + Wilcoxon, 5 losses ×
all metrics); `compare_loss_variants.py` → `spat_vs_temp_{PCA,KL}.png` + within-mouse +
`summary.csv` (paired-t, per-mouse dots). Ran both on on-disk `loss_comparison_v1` + `wm3*`.
**Results — one coherent story:** LOSSES (own-metric skill): temporal ≥ spatial in **4/5**
(PCA Δ+0.05 *p=0.036*, KL +0.06 p=0.081, JS +0.05 p=0.060, CE +0.00, Wass −0.02). VARIANTS
(KL-skill, spat/temp): evar **1.48 / 2.54** (temporal *worse* — the peakiness artifact, p=0.087)
→ flat **0.56 / 0.55** → shape λ=0.1 **0.64 / 0.53** → λ=0.3 **0.57 / 0.51** (*p=0.043*). So the
apparent **PPC>SBC under evar is a peakiness artifact that inverts** to the true small **SBC≥PPC**
once width is fixed. **Consistency fix (completing last turn's λ relabel):** `compare_loss_variants.py`
still labeled variants λ=10/30 (config units) → relabeled to clean **λ=0.01/0.1/0.3** (= config/100),
regenerated fig 16/18/22/23 + resynced — the note's *embedded* variant figures were inconsistent
with the relabeled text until now. Verified the within-mouse 1b figures use generic titles (no
stale λ), so fig 20 was already fine.
**Built this turn:** `spat_temp_performance.py` — unified spat-vs-temp performance bars (raw +
skill, KL & PCA metrics, per-mouse dots, paired-t, n=6) for the **losses** and **variants** groups
(reuses `_eval_one`); pyflakes clean; verified on disk (numbers match cross_loss_eval +
compare_loss_variants). Figs `figures/spat_temp/spat_temp_{losses,variants}_{KL,PCA}.png` + stats
CSVs. The 5-loss spat/temp **performance** figure is the genuinely new piece (the note shows losses
spat/temp only as *peakiness*, fig 9a-C). Headline under KL: CE/KL/JS temporal ≥ spatial (trend,
n=6), PCA temporal *worse* (artifact, p=0.007), Wasserstein worse-than-chance both.
**Note section DONE** — added **§9 "Spatial vs temporal performance: the architecture gap is a
calibration artefact"** to `[[PCA-Peakiness-Mechanism]]`: the new `peakiness_spat_temp_losses.png`
5-loss KL skill figure + a stats table + the honest caveats (metric-dependence, n=6 underpowering,
3/6 mice, evar artefact), referencing variant fig 18; conclusion bumped §9→§10 (no §9 cross-refs);
28 figs synced, all embeds resolve, LaTeX balanced. Added the sync MAP line.
- **Open / next:** (a) **width axis** — add spat-vs-temp KL-skill vs H (per loss, paired-t per H)
  to `plot_overfit_vs_width.py` reusing `_eval_one`, then a width panel/figure into §9; pending B1
  results. (b) optionally retire the spat/temp overlap in `compare_loss_variants` in favour of the
  unified `spat_temp_performance.py`.

### 2026-06-09 — Peakiness note: all-loss equations block, deeper Brier, dropped the ×100 (λ → Brier-weight units)
On Theo's asks while B1 runs on the cluster. **(1) Detailed equations** — added a "The losses,
precisely" block to §2 of `[[PCA-Peakiness-Mechanism]]` with full equations for every loss (PCA
evar-weighted, Brier/flat-evar, CE, forward KL, JS, Wasserstein-1 as L1-of-CDFs, the shape fix,
temporal entropy penalty), verified against `nn_classifier.py` (`custom_loss_all_H` + the
`KL_calc/JS_calc/Wasserstein_calc_1D/cross_entropy` helpers). Folded in three exact facts:
Brier = flat-evar by Parseval; CE = KL + H(t) ⇒ identical gradients (we report KL); only
Wasserstein uses bin order. **(2) Brier depth** — expanded §6 (strictly proper scoring rule;
penalises the whole distribution, not a privileged few; the λ→∞ limit of the shape fix).
**(3) Dropped the ×100** — the code scales L_PCA by 100 and the shape floor by λ/100 (they cancel
to `L_PCA + λ·Brier`). Removed it from the equations; **consequently λ is now the literal Brier
weight, so the sweet spot is λ=0.1 (was "λ=10")** — relabeled the sweep to {0, 0.01, 0.1, 0.3, ∞}
across the note (equation, §7/§8 prose+table, §9), the `diagnostics/lambda_sweep.py` LADDER +
docstring, regenerated `lambda_sweep.png` (curves/numbers unchanged: evar 0.231/0.346,
λ=0.1 → 0.066/0.060) and re-synced (27 figs). Added a reproduce note (config `--shape-lambda` =
100λ). Verified: λ labels all clean, no stray `/100`, LaTeX balanced (8 `$$`, `aligned` paired).
Note + attachments are vault-side; the only repo change is `lambda_sweep.py`.
- **Open / next:** (a) **§6 capacity ablation** pending B1 results — the `capacity_summary_spat`
  figure + the drafted "shrink the net → peakiness shrinks" subsection drop in once results rsync
  down; (b) **real-data-everywhere** (Theo's directive): build the real-data analogues of the two
  still-toy-only results — fig 7 subspace-error decomposition (decoded−target error by
  location/shape PC subspace) and fig 8b uncertainty-scaling (peakiness vs stimulus
  contrast/dispersion), both feasible from on-disk `loss_comparison_v1`; (c) the λ relabel reverts
  trivially to "λ=10 + a units note" if Theo prefers the familiar number.

### 2026-06-09 — Tier B triage for tomorrow's meeting: B1 plotter built, cluster launch queued for Theo
Answered "anything outstanding from previous meetings for tomorrow?" by cross-checking the
2026-06-03 Máté action items: **5/7 done (Tier A); 2 outstanding, both Tier B / cluster** —
(1) overfitting-vs-hidden-width and (2) weight regularisation. Decided **B1-only** for the
timeline; H-ladder fixed at `{4,8,16,32,64}` (not ≤2). **Wrote + smoke-tested
`nn_decoder/plot_overfit_vs_width.py`** — the missing companion plotter: per-loss facets of
the val−train **total-loss gap at `best_epoch`** vs H (spatial vs temporal, mean±sem + faint
per-mouse) plus a train/val-level companion; PNG+SVG via `figsave`; imports/`--help`/no-data
paths all verified. Confirmed the run is gap-capable (`run_loss_comparison` `VAL_FRACTION=0.2`,
`best_epoch` recorded in `history`). **Extended (same session) to also compute decoded
peakiness-vs-width** from the `.mat` posteriors (`Dist[arch]['decoded'].max(-1)` → mean
max-prob) + the IO-target line — the real-data analogue of the peakiness note's toy capacity
sweep (fig 8a/15); adds figs `capacity_summary_spat` (peakiness | gap, spatial = loss alone)
and `peakiness_vs_width_by_loss`. pyflakes clean. Found **B2 is not turnkey** — the runner has no
`--weight-decay` flag (weight_decay is a fixed shared hyperparam). **The agent could not run
the launch**: the harness auto-mode classifier blocks `ssh`/`rsync` to `gpu1` without an
explicit allow-rule (denied twice), so Theo runs it. Exact, verified block:
- **up:** `cd <repo>; rsync -avz nn_decoder/run_loss_comparison.py gpu1:~/UncertaintyV1/nn_decoder/`
- **launch** (tmux `hidden_abl`, `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `PY=~/cluster-env/.venv/bin/python`):
  `$PY -u run_loss_comparison.py --run-name hidden_ablation --hidden-sizes 4 8 16 32 64 --targets Q --bin-sizes-ms 100 --windows half --splits stratified_balanced 2>&1 | tee hidden_abl.log`  (Q-only probe; ≈150 fits — add `L` if Q lands with time)
- **down:** `rsync -avz 'gpu1:~/UncertaintyV1/nn_decoder/results/hidden_ablation_h*' nn_decoder/results/`
- **then:** `python plot_overfit_vs_width.py` (defaults Q/half/100ms → `figures/loss_sweep_plots/hidden_ablation/overfit_vs_width/`)
- **Open / next:** (a) Theo **launched B1** on `gpu1` (running — PCA+CE done at last ping); (b) after rsync-down, run `plot_overfit_vs_width.py`; (c) **fold the capacity result into the peakiness note** `[[PCA-Peakiness-Mechanism]]` — update *prepared* (new §6 control "shrink the net → peakiness shrinks", `capacity_summary_spat` figure + a `sync_peakiness_figs.sh` MAP line + caption drafted), pending results + Theo's sign-off; (d) **B2 still owed** — add a `--weight-decay` flag to `run_loss_comparison.py`, then the PPC sweep; (e) Monte-Carlo-in-sampling-models tangent parked in vault `ideas`. Commits `d7aa89c` (plotter v1) + this (peakiness extension + log); results/figures gitignored.

### 2026-06-09 — Peakiness note: full restructure for flow + figure overhaul + λ-sweep
Reworked the vault note `2026-06-03-PCA-Peakiness-Mechanism` end-to-end on Theo's
feedback (flow jumped around; figures inconsistent / over-titled / PPC-SBC labelled;
wanted a λ sweep). **(1) Restructure** — dissolved the separate Methods section and
reordered into a logical flow with methods *in place*: 1 puzzle → 2 setup (new
schematic) → 3 real-data observation (distributions) → 4 mechanism (PC-basis demo) →
5 toy proof → 6 controls → 7 fix + λ-sweep → 8 validation → 9 conclusion; no
forward-references (toy introduced before it's used). **(2) Figure system** — added
`peakiness_style.panel_label`/`label_panels` (bold a,b,c…) + `figsize()` for uniform
per-panel physical size; a 3-subagent pass added panel letters, trimmed every
exposition-y title to a short noun phrase (argument/numbers moved to captions),
repositioned legends off the data, and made panel sizes consistent across ~16 figures.
Wrote **panel-by-panel figure captions** in the note. **(3) spatial/temporal** — note
prose + all figures (the 2026-06-09 code rename + the two leftover `arch.upper()`
scripts). **(4) New visuals** — `setup_schematic.py` (two readouts + losses) and the
PC-basis demo serve comment "visual demos are good". **(5) λ-sweep** — new
`diagnostics/lambda_sweep.py`: peakiness + KL/PCA-skill across λ∈{0,1,10,30,∞},
spatial vs temporal, across-mice (mean±sem) AND within-mice (per-mouse lines); λ=30
in the tables. All numbers are the fixed-shuffle re-run values. Vault: 27 figures, 0
broken/orphan, LaTeX balanced. Commits `4628e1f`, `21daa3f`, `5ecf2fd` (note untracked).
- **Open:** none new; the note is the polished record.

### 2026-06-09 — Relabel "PPC"/"SBC" → "spatial"/"temporal" in all visible figure text
Renamed the architecture acronyms everywhere a human reads them — figure titles,
legends, axis/tick labels, text/annotate, math subscripts ($p_{PPC}$→$p_{\mathrm{spatial}}$),
plus comments/docstrings/console prints — across **42 non-legacy .py files** (mapping
fixed by README: PPC = spatial / Probabilistic Population Codes, SBC = temporal /
Sampling-Based Codes). Driven by a one-off scoped transform (uppercase-only, protected
glued identifiers/keys), then hand-reviewed every diff. **Left untouched** (by design):
code identifiers (`ppc_time_avg`, `generate_PPC_targets`, `PPC_C`), `model_type='ppc'/'sampling'`,
arch codes `'spat'/'temp'`, and result/CSV dict keys (`PPC_Post_Mu`, `'PPC_PCA'`,
`"M2-M1 (...SBC wedge)"`). Where a raw key was *displayed* (`plot_loss_sweep_comparison`
tick labels), added a `_pretty()` display-map instead of mutating the key. Caught & fixed
one regression (a changed `delta` dict key broke `test_rd2`). Tests: the 4 failing test
files all fail identically on clean `main` (pre-existing); my changes add **zero** new
failures. Key files: `audit/jensen_smoothing_explainer.py`, `time_binned_ppc.py`,
`plot_loss_sweep*.py`, `decoder_plotting_utils.py`, `similarity_readout_tests.py`,
core `nn_classifier.py`/`run_experiment.py`/`training/config.py` (comments/prints only).
**Open:** legacy/ deliberately skipped; manuscript/wiki docs still say PPC/SBC if a later
pass wants them aligned.

### 2026-06-09 — nn_decoder architecture cleanup pass (audit → 8 tested commits)
Ran a 10-agent read-only architecture audit of `nn_decoder` (liveness/dead-code map,
core depth/layering, shared-utility duplication, per-cluster duplication + legacy
candidates → synthesis), then executed the high-value findings as 8 commits on `main`
(no push; each verified). **Headline: a real load-bearing bug** — `time_binned_ppc.
run_mouse/main` did `from utils_v26 import …` (no such module; only a stale `.pyc`),
so both entry points raised `ModuleNotFoundError` on a clean checkout; the imports were
lazy so no test caught it. Repointed to the live `utils` + added a `run_mouse` smoke
test (`120f248`). The rest:
- Dead code / conservative legacy moves: removed the never-instantiated `NN_classifier`
  class, `legacy/{to_tensor,edit_top_mode}.py`, two dead `utils` helpers + two dead
  snippets; moved `compare_partial_corr_designs.py`→`legacy/`; `importorskip('ssm')`
  in test_glm_hmm (`2b03ea2`).
- `_batched_fit_loss` (underscore-private but imported by **6** scripts) → public
  `fit_loss_per_trial` + the missing direct numpy/torch agreement test; back-compat
  alias kept (`b87caa9`).
- `evaluate_model_entropy` signature narrowed (dropped unused `angles/circle_type/
  device`); Config's recorded-only knobs (`optimizer_type/momentum/weight_initialization`)
  **documented, not deleted** — 79 `config.yaml` provenance files depend on the schema
  (`b4c142a`).
- **`run_animal_decoder` god-function** → extracted tested `fit_pca_basis(...)` +
  `_decode_models_over_loader(...)` (kills the ~90%-duplicated held-out/full eval loops),
  pruned dead `training_params`. Verified **bit-identical** (20 arrays) on a seeded
  1-mouse/REP=1/2-epoch cell before vs after (`c152e95`).
- **`decoder_plotting_utils` god-module** (1756 ln) → pure scoring leaves
  (`calc_pca_dist/calc_fit_loss/variance_baseline/get_mouse_pca_losses/_normalize_mode`)
  extracted to new **`decoder_metrics.py`**, re-exported so all 14 importers are
  unchanged (`e31d943`).
- Similarity pillar: `similarity_m2_followup` now imports the shared `_cv_loglik` +
  WARM/COOL from `similarity_readout_tests` (closed the CV-scoring divergence hazard;
  dropped the dead `seed` arg) (`36c809d`).
- Figure-save: new dependency-light **`figsave.save_fig`** single sink (re-exported by
  `peakiness_style.save_fig`); fixed the PNG-only CLAUDE.md violators (loss_smoothness_demo,
  both `audit/` scripts, load-bearing `time_binned_ppc`) + 3 uncapped local `_save`
  helpers (`0d1f8f7`).

**Verification:** full suite **571 passed, 2 skipped, 4 failed** — the 4 failures are
**pre-existing** (fail identically on the pre-session base `bf6566a`; this session
touched none of those files: test_fit_model's 2-vs-3-tuple unpack, test_neuron_scaling's
`d` epoch-floor expectation, test_neural_heuristics' `reg` default, io_hmm's
`np.trapezoid` numpy-version gap). pyflakes clean on touched files. **Test-env trap
(now in GOTCHAS):** the torch suite needs `KMP_DUPLICATE_LIB_OK=TRUE` + single-thread or
it segfaults under the Anaconda libomp. **Data note:** an import-smoke accidentally
imported `run_recovery_Q_spyder` (runs at import, `FORCE_RERUN=True`), which regenerated
`recovery_cache_fixed_perception.npy`; the original was restored from its `.bak`.
- **Open / next:** (a) **#4 loss-sweep cluster dedup** (scoring/CLI/constant copy-paste
  across ~17 plotters) + the slug `pca_basis` gap — `plot_all_cells` discovery only
  matches `_all`, so condmean/residual runs are silently skipped, and fixing it correctly
  needs `pca_basis` threaded through ~6 *untested* downstream plotters. Deferred: it's
  **latent** (no condmean/residual runs exist; production basis is `all_trials`), and a
  partial fix would make those runs "discovered but unreadable". (b) **#9 recovery/optuna**
  shared-`fit_pca_basis` reuse — `optuna_per_target` can't be imported/tested locally
  (optuna not installed); `recovery_convergence_probe` could reuse it. (c) `run_recovery_
  Q_spyder` should be `__main__`-guarded with `FORCE_RERUN=False` (legacy candidate). (d)
  long-tail PNG-only one-offs (plot_kl_js_*, pca_posterior_vs_likelihood, plot_neuron_scaling,
  plot_post_fix_performance, decoder_loadings_comparison) left as-is — recorded one-offs
  that won't re-run, so rerouting their save code is no-op churn.

### 2026-06-09 — Resolved the Cb17/Cb22 within-trial-variance signal (SBC-wedge follow-up)
Chased down whether the 2/6-mouse within-trial-variance choice signal (RD-2 M2−M1>0) is
genuine within-trial sampling (SBC) or the variance substituting for a weak trial-mean
readout. New module `nn_decoder/similarity_m2_followup.py` (+`tests/test_similarity_m2_followup.py`)
runs a discriminating control battery (C1 variance-specificity, **C2 survives-strong-mean
ladder**, C3 directional-vs-uncertainty, C4 confidence link, C5 within-trial dynamics, C6
window, C7 unbounded-SI, C8 permutation-null+bootstrap-CI) on all 6 mice + the network. A
**7-agent adversarial workflow** verified it (code leakage-audited **sound**; decisive split
independently re-implemented; permutation null + bootstrap CIs added on the completeness
critic's recommendation). **Verdict:**
- **Cb22 = artifact (B)** — M2−M1 collapses to negative under any strong readout (−0.004 over
  IO log-odds), below the within-condition shuffle null (z=−2.3). Variance was collinear with
  the IO/stimulus readout.
- **Cb17 = genuine within-trial signal but NOT SBC** — survives the IO log-odds (+0.017, z=10.5
  vs shuffle null, p≤0.003, independently reproduced), the whitened decoder, the maximal static
  + full-difficulty control, and is present in the clean unbounded SI. So the wedge *as literally
  written* fails in Cb17.
- **No mouse shows the SBC mechanism** — in all 6, Var_t[SI] tracks **confidence/decisiveness**
  (partial-corr with IO decision entropy negative 6/6, bootstrap CI excludes 0; predicts
  correctness +ve 5/6), the *wrong sign* for SBC posterior-width.
**So the anti-SBC wedge is REFINED, not vindicated:** "within-trial variance carries no choice
info" is false in Cb17, but the narrower bootstrap-robust claim ("no posterior-width SBC
signature") holds in all 6. Cb17 exposes a genuine within-trial **dynamics/decisiveness** term
neither the framework's trial-mean accumulator nor SBC predicts — a single-animal lead (n=6 →
no population claim). Conjecture gained §Follow-up + verdict figure; Finding 4/Net revised.
**Pitfall logged (GOTCHAS):** a saturated control base can manufacture a positive ΔLL via a
suppressor effect (proven by Cb21) — trust the clean single-readout C2[+IO] permutation-null
discriminator, not the saturated base.
- **Open / next:** (a) re-run C4 against an *animal-internal* uncertainty proxy (confidence/
  lapse), not just the IO; (b) detrended C6 (regress out conditional-mean trajectory, not
  truncate bins); (c) the Cb17 dynamics lead needs more sessions/animals to support any claim.

### 2026-06-08 — Similarity Framework: theory tightened + two decisive readout tests (real data + network control)
Revived and hardened the **Similarity Framework** pillar (vault conjecture
`Conjectures/Similarity Framework.md`, the `si_network_model/`, and the dormant
real-data analyses that had been *run* — 103 per-session figs — but never
interpreted). New module `nn_decoder/similarity_readout_tests.py` (+`tests/
test_similarity_readout_tests.py`, 8 sensitivity/specificity tests; 42 similarity
tests total pass; pyflakes clean) implements two cross-validated, confound-controlled
tests, run on Cb15–Cb25 **and** on the `si_network_model` symmetric cohort as a
positive control:
- **RD-1 — template (Δμ) vs whitened LDA (Σ⁻¹Δμ) shrinkage sweep.** Whitening lifts
  *stimulus* decoding in **6/6 mice (+0.040 AUC)** — exploitable covariance exists.
- **RD-2 — within-condition nested choice models** (M0 stim → +mean SI → +Var_t[SI]
  → +whitened proj), held-out ΔLL: **M3−M1 ≈ 0 in all 6** (choices read the template,
  not the optimum → *premise supported*); **M1−M0 > 0 in 4/6** (Pred 16, partial);
  **M2−M1 ≈ 0 in 4/6 but +0.017/+0.011 in Cb17/Cb22** (a partial, animal-specific
  SBC signature). Network control: M1−M0 huge, M2−M1 = M3−M1 = 0 in all 10; RD-1
  Δstim **−0.02** (info-limiting noise → cosine already optimal → the REPORT's 96%).
**Headline:** the brain reads the un-whitened template direction; the
"cosine-discards-covariance" cost is now *measured* on data (+0.04 stimulus-AUC the
choices ignore), larger than in the model. **Theory upgrades in the conjecture:**
(1) only the vMF SI is a literal LLR — the prototype normalised-difference is a
monotone distortion (trial-dependent denominator); (2) the framework is a
*constrained-linear* claim (template vs whitened), subsuming the cosine-vs-Euclidean
& cosine-vs-Mahalanobis falsifiers and linking to RSA; (3) **Prediction 9 was
choice-probability-confounded** — replaced by the M2−M1 contrast. Conjecture gained
§*Empirical Findings* + 3 embedded figures; Falsifiers/Counterarguments/DevLog
updated; `si_network_model/REPORT.md` cross-references its new control role.
Figures: `nn_decoder/figures/similarity_framework/F_readout_tests/` (PNG+SVG, <1500px)
+ copies in `Conjectures/attachments/readout_*`.
- **Open / next:** (a) the Cb17/Cb22 M2 signal — is it literal within-trial sampling
  or the variance substituting for a weak mean readout? (control: does Var add over
  *stim alone*, M2 vs M0; and is it specifically variance vs any 2nd neural feature).
  (b) hybrid direction+magnitude readout (cosine discards ‖r‖, and contrast is
  informative here). (c) the still-deferred `similarity_fit.py` cross-modal DDM
  (Preds 7,8). (d) optionally fold the existing 103 per-session figs (variants-compare,
  κ↔lapse) into §Empirical Findings.

### 2026-06-05 — Code soundness audit of the whole peakiness pipeline + fixes
Ran a 16-agent soundness audit (review → adversarial-verify → synthesize) over all
code behind the note figures (toy + real-data + core loss/forward/run math). **Verdict:
sound-with-minor-issues** — core math/metrics/toy verified correct to machine precision
(PCA loss, flat_evar=Brier, shape_lambda=PCA+λ·Brier, forward-KL, PPC/SBC Jensen, basis
fit no-leakage, toy spectral test non-circular). Findings fixed:
- **MAJOR — shuffle-control index bug** (`run_experiment.py:344`): `repeat(perm,T)+tile(arange(T))`
  was wrong for the trial-major target layout (added bin offset to the trial index instead
  of `*T`) → not a real permutation (~48/470 trials used, bins blended across trials). Fixed
  to `repeat(perm,T)*T + tile(arange(T))`. **Re-ran the full wm3 family** (5 cells × 6 mice,
  fixed shuffle): peakiness & PCA-skill unchanged; evar **KL-skill 1.31/2.23 → 1.48/2.54**
  (strengthens "worse than chance"); shape KL-skill ~same. Updated all §7 tables + figures.
- **Trajectory/weight grid truncation** (`decode_entropy_trajectory._interp_mean`,
  `weight_evolution_variants.collect`): averaged on the *shortest* mouse's snapshot grid,
  hiding PCA's late descent and making evar≈shape ‖W_out‖ look equal. Fixed to UNION grid
  (single-mouse tail trimmed). Corrected fig2 (PCA descends through ep180) and the logit
  table (evar ‖W_out‖ 9.9 vs shape **12.6** — shape *larger*, conclusion strengthens).
- **`_batched_fit_loss`** now raises on PCA-without-basis (was silent CE fall-through).
- Note wording fixes: "~300× total" → shape-subspace ~300× (total ~67×); "×50" → ~46×;
  landscape R table flagged single-mouse + cross-mouse means added; §8.4 mean-centring note.
138 relevant tests pass. Commit `873c3a6` (code); note + figures updated and re-synced (28
figs, all <2000px). **No conclusion overturned; several strengthened.**
- **Open:** the bug fix is in code for all future runs; if other cells (loss_comparison_v1,
  OOD splits) get re-used for skill numbers, re-run them too (only wm3 was refreshed here).

### 2026-06-05 — Máté's bulk-vs-tail challenge: audit + answer
Máté (Slack) questioned whether the spatial PCA peakiness is a real effect or just a
right tail / unfortunate example trials, noting the max(P)/H(P) bulk looks aligned with
CE/KL/JS; he asked for a per-trial temporal-vs-spatial peakiness scatter coloured by
PCA-loss(spat−temp). **Answer: it is a genuine BULK shift, not a tail artifact.** Ran a
6-agent verification workflow (independent recompute + 2 adversarial refuters arguing
Máté's case — both concluded the bulk moves) and built two diagnostics:
`peakiness_distributions.py` (full per-trial dists, all 5 losses × both arches) and
`peakiness_scatter_spat_temp.py` (his requested scatter + representative top-left
examples). **Key numbers (pooled 2186 trials × 6 mice, Q half 100ms):** spatial PCA
**median** max(P) 0.135 vs CE/KL/JS median 0.043 (3.2×) and target 0.039; KS 0.48,
overlap 0.52; PCA p25 (0.058) already = calibrated median; 71% of PCA-spatial trials
above the calibrated p75; ~22% genuinely IO-like (his top-left subset, real but
minority). The H(P) view understates it (spatial posteriors multimodal → entropy stays
up). Examples are NOT cherry-picked (selection sorts by target position, not peakiness;
extremes ~0.9 never shown). spat≠temp under PCA confirmed (KS 0.39) but reframed: it's
two miscalibrations (spat 3.5×, temp 8×), not a clean architectural readout. **All vault
peakiness numbers independently verified within ~10%** (the median makes the SBC claim
stronger, not weaker). Vault §3 gained a "bulk or tail?" subsection + median in the
table + 3 figures (fig9a/9b/9c). Commit `f70733a`. Drafted a Slack reply for Theo.
- **Open:** none new; this hardens the headline against the exact critique.

### 2026-06-05 — Peakiness figure suite: shared visual language + declutter
Assessed whether the PCA-peakiness answer is meeting-ready (it is — the vault note
`2026-06-03-PCA-Peakiness-Mechanism.md` is a complete, consensus chain; only the
two §10 confirmatory checks remain open), then **restyled the whole 26-figure
suite** so it reads as one story. New `nn_decoder/peakiness_style.py`: one palette
(**warm = pathological PCA-evar everywhere; cool = the calibrated losses / fixes**),
one `save_fig` that caps PNGs ≤1600 px (previewable), one `apply()` taming the
seaborn *talk* context, and `target_band/target_line/chance_line` primitives.
Refactored all 8 plotters onto it (`decode_entropy_trajectory`,
`posterior_trajectory_landscape`, `toy_peakiness_model`, `toy_shortcut_dynamics`,
`toy_width_matched`, `flat_evar_example_posteriors`, `compare_loss_variants`,
`weight_evolution_variants`): killed the per-script colour/`_save` drift, collapsed
two-line argue-in-title headers to single lines, fixed the fig11 dual-encoding
(epoch-colorbar vs loss-colour → one colour/loss, open=start/filled=end), shared
the fig26 logit y-axis, and shrank the 3 over-2000px figures under the preview cap.
Regenerated everything (numbers reproduce exactly) and re-synced PNGs into the
vault attachments via `sync_peakiness_figs.sh`. Commit `8f21f27`.
Then **trimmed the vault note** (~30% less prose: cut the repeated mechanism
restatements, tightened the TL;DR / within-mouse / dynamics paragraphs) and added a
concrete **§8 Methods** (toy-model world + architecture + losses + training; real-data
decoders/variants; metrics & skill/stats; landscape probe; weight/logit analysis).
Reflowed paragraphs to single lines (Obsidian soft-wraps; no hard mid-line breaks).
Finally **pruned to a coherent, both-arch set (26→18 logical figs)**: cut morph,
dynamics-decomp, PCA-metric bars, overlay examples, weight-norm + weight-scatter
(fig 4/12/17/19/24/25); example posteriors now shown for BOTH PPC and SBC (flat-vs-evar
and the variant grid each in both arches); and **fixed the within-mouse figure** — the
old per-mouse skill-lines replaced by the clean-run **1b** per-mouse PPC-vs-SBC bars
(`plot_per_mouse_performance_with_stats`, now emits PNG too) for all 3 variants. Commit
`4fe0000`. Finally **expanded §8 Methods into 11 equation-rich subsections** (toy
generative model, the two forward passes, all loss equations incl. the evar-floor =
PCA+λ·Brier identity, PCA basis, subspace decomposition, peakiness, landscape probe,
shuffle-skill, weight/logit stats) — every formula pulled from source; LaTeX delimiters
verified balanced. **Then proved the load-bearing assumption** (Theo: "how do we *know*
leading PCs = location, trailing = shape?") with `diagnostics/pc_location_vs_shape.py`:
the target PCA basis is an ordered low→high spatial-frequency basis (PC0/PC1 = fundamental
cos/sin = phase code for location → (PC0,PC1) traces a circle); the decisive sweep test —
vary location at fixed width → leading PCs move (CoM PC#2–4), vary width at fixed location
→ trailing PCs move (CoM PC#5.5 toy, **#22 real**). Two figures (toy + real V1) added to a
new §2 subsection + §8.5 pointer. Commits `3891d9e`, regen/sync updated.
- **Open / next:** unchanged — §9 confirmatory checks (real-data peakiness-vs-
  uncertainty scaling; trained-as-target round-trip) and the production-loss
  decision (PCA+shape λ=10 vs a proper divergence). If the note still reads long,
  the remaining lever is figure count (26) — offered to consolidate.

### 2026-06-03 — Real-data 3-variant comparison (evar / flat / shape) + λ sweep
Implemented `shape_lambda` Config field (width-matched PCA loss = PCA + λ·Brier,
applied as evar+λ/100 floor like flat_evar; `run_loss_comparison.py --shape-lambda`).
Fresh matched runs `wm3{,_flatevar,_shape1,_shape10,_shape30}` (Q/100ms/half/balanced,
6 mice, shared seed). `compare_loss_variants.py` scores held-out posteriors under a
COMMON metric (true-evar PCA loss + KL), shuffle-normalised (skill=loss/shuffle),
across + within mice (paired-t, n=6). **Headline (target max-prob 0.059):**
- Peakiness: evar 0.22/0.39 → flat 0.053 → shape λ=10 hits target 0.066/0.059.
- **PCA metric is BLIND**: PCA-skill flat ~0.41–0.49 across EVERY variant & λ — it
  scores the peaky evar decoder identically to the calibrated ones.
- **KL reveals it**: evar is WORSE than chance (skill 1.31 spat / 2.23 temp);
  flat 0.60/0.57; shape λ=10 0.64/0.53. The big SBC-worse-than-PPC gap (KL raw
  p=0.008) is an evar **peakiness artifact** — it vanishes for flat/shape (PPC≈SBC).
- **Fix lands**: shape λ=10 = target peakiness, beats chance on KL (temp even ≤ flat),
  costs nothing on PCA (0.49/0.41), keeps PCA's location weighting. As λ↑ it converges
  toward flat-evar; λ=10 is the sweet spot. Figures in `figures/loss_variants/`.

### 2026-06-03 — Width-matched fix + WHY it keeps climbing (overfitting)
`diagnostics/toy_width_matched.py`. **(1) Fix:** width-matched loss `PCA_evar +
λ·Brier` (floor the per-PC weights so the shape subspace isn't free) parks at the
target (max-prob 0.049 vs 0.044) with PCA's location emphasis kept — phase portrait
shows it stops where plain PCA climbs to 0.26. **(2) Why it keeps climbing:** logged
train vs test weighted-PCA loss — train keeps falling (0.62→0.007) while TEST bottoms
at epoch ~92 then RISES (0.027→0.037); peakiness tracks the train–test gap. So the
over-sharpening is **overfitting** into the loss-blind subspace (fits noise as
overconfident sharp predictions). Early stopping at best-test caps it (0.157 vs 0.260)
but it's still 3.6× target and a weak signal (the weighted loss barely sees the shape
subspace); lower LR (1e-3) is the same drift, slower. Updated the vault report
(figs 13–15) and corrected the "not an optimisation problem" wording → it's
overfitting into a loss-blind subspace (rugged-landscape tricks miss; early stopping
partially helps; the loss fix is complete).
- **Next:** port the width-matched loss to `run_experiment` (a `shape_lambda` Config
  field, default 0 = no-op) and tune λ on one real cell.

### 2026-06-03 — Toy learning dynamics: the "shortcut" + one-way drift
`diagnostics/toy_shortcut_dynamics.py` instruments toy training, logging
location-subspace error vs shape-subspace error vs peakiness per epoch. Answers
Theo's dynamics question. **Two phases:** (1) shortcut, shared by all losses —
location error plummets ~10× in the first ~30 epochs and peakiness rises in
lockstep (from uniform, building a peak at the mode is the fast way to cut the
location-dominated loss); (2) divergence — KL/flat-L2 then drive the SHAPE error
down (fix width, peakiness halts at target), but under PCA the shape error
**grows monotonically ×50** over training (overfitting into the free subspace),
so peakiness ratchets up unbounded. Phase portrait (location-error × peakiness):
all dive left together, then KL/flat-L2 park at target while PCA climbs the
peakiness axis with location pinned. So it does NOT get stuck at a fixed width —
it gets stuck *climbing* (no gradient opposes further sharpening). Vault report
gained a dynamics subsection (figs 11–12).

### 2026-06-03 — Toy model PROVES (and corrects) the peakiness mechanism
Built `diagnostics/toy_peakiness_model.py` — fully synthetic (noisy population code
→ posterior, broad fixed-width targets, known location uncertainty), same MLP under
PCA / flat-L2 / KL. Reproduces it (PCA max-prob 0.26 vs target 0.044; flat-L2 & KL
match). **Decisive spectral test:** decoded-vs-target error split into location
subspace (top PCs) vs shape subspace — all three match location equally, PCA's
shape-subspace error ~300× larger than KL. So the precise, proven mechanism: the
evar weighting **constrains only the location subspace and leaves width/shape
unconstrained; the net fills that free subspace with spiky junk** while getting
location right. This **corrects** the earlier "loss minimiser is a spike" wording —
the weighted-L2 minimiser is the broad conditional mean; peakiness is the learned
solution in the loss-blind subspace. Sweeps: grows with capacity & input noise;
flat-L2/KL flat at ~0. Confirmed `pca_basis: all_trials` in both runs (not
condition_mean). Full report (8 figures) in the vault:
`ResearchVault/Projects/Uncertainty/2026-06-03-PCA-Peakiness-Mechanism.md`.
- **Spatial (clean, no entropy penalty):** since SBC has a per-bin entropy
  penalty and PPC has none, added `--metric maxprob` to `decode_entropy_trajectory`
  and showed the PPC decoder over-sharpens under PCA (max-prob 0.21 vs target 0.06,
  3.6×) and flat-evar fixes it (0.053) — so the loss is the cause; the entropy
  penalty is only a ~2× amplifier (SBC 6×). Vault report gained a spatial section
  (figs 9–10).
- **Next:** prototype the width-matched loss (keep evar + constrain shape subspace)
  in the toy first; trained-as-target round-trip; real-data peakiness-vs-contrast.

### 2026-06-03 — Flat-evar control CONFIRMS the driver; trajectory + landscape viz
Three follow-ups (Theo). **(1) Flat-evar control:** added `flat_evar` Config flag
(`run_experiment` flattens explained_variance → PCA loss = unweighted L2/Brier;
`run_loss_comparison.py --flat-evar`, isolated under `<run>_flatevar`). Ran PCA
Q/half/100ms/balanced/6 mice. **Decisive: evar-weighted PCA → H 3.24 (peaky);
flat-evar PCA → H 4.03, halting at the CE/KL/JS calibrated line (3.95).** The evar
weighting *is* the cause. **(3) Visualisations:** new
`diagnostics/posterior_trajectory_landscape.py` — posterior morphing across
training (PCA spikes & diverges from target; KL converges) + loss-vs-width
landscape. The landscape gives the quantitative driver: all losses min at γ=1,
but the basin is ASYMMETRIC — sharp/broad cost ratio PCA 0.69 & Wasserstein 0.80
(<1, drift peaky) vs CE/KL 1.64 (restoring), PCA-flat 1.21. `decode_entropy_
trajectory.py` gained an overlay mode (`--compare-runs`). **(2) Mitigations:**
more-inits / basin-hopping / perturbations do NOT help — this is loss
mis-specification (asymmetric, width-blind objective), not optimisation
difficulty; fixes are loss-side (flat evar, proper divergence, or width/entropy-
matching term). Corrects my earlier "shallow basin" guess → it's asymmetry.
- Touched core modules (config.py, run_experiment.py); 35/35 relevant tests pass
  + the flat-evar run completed end-to-end. `flat_evar` defaults False (no-op).
- Figures: `figures/loss_sweep_plots/{loss_comparison_v1,brier_ctrl_flatevar}/`
  (entropy_trajectory/, trajectory_landscape/). Results gitignored.
- **Open:** the trained-as-target round-trip is the remaining confirmatory check.

### 2026-06-03 — Why PCA goes peaky: decoded-entropy trajectory
Máté/Theo pushed on *what favours* peaky distributions (the loss optimum is the
broad target, so "doesn't punish broad" can't be the whole story). Built
`diagnostics/decode_entropy_trajectory.py` — re-decodes X_test at each weight
snapshot, tracks mean decoded entropy vs epoch. **Decisive result:** all losses
start near-uniform; **CE/KL/JS halt exactly at the IO target entropy and stay
flat (target is a stable attractor); PCA sails through the target and keeps
falling monotonically (H 4.48→3.0, still descending at the deployed epoch).**
- Corrected an earlier claim: peakiness is **not** an early-stopping artifact —
  ES *limits* it; it's a genuine attractor. PCA's `evar`-weighted L2 puts its
  over-sharpening penalty in the ~0-weight trailing PCs, so unlike KL/JS/CE it has
  no restoring force at the target.
- **Open / next:** the exact micro-gradient pulling PCA past the target isn't
  pinned. Decisive control = **flat-`evar` (unweighted Brier) L2** should behave
  like CE/KL — one flag in the loss; plus the trained-as-target round-trip.
  Figures: `figures/loss_sweep_plots/loss_comparison_v1/entropy_trajectory/`.

### 2026-06-03 — Tier A executed (weight + posterior-PCA diagnostics)
Ran the whole of Tier A from the live plan, all on existing `loss_comparison_v1`
data (no cluster). Data audit first: the checkpoints carry full weight tensors at
init (epoch 0) + every 10 epochs, plus train/val curves and decoded+target
posteriors — so meeting items 2–6 needed no re-run.
- **A1** — extended `plot_weight_evolution_cell.py` with figs D (fan-in-normalised
  `‖W_in‖/√N_in`, `‖W_out‖/√H`), E (weight mean±std vs snapshot epoch + init ref),
  F (init-vs-final histograms, **all 4 param groups** W_in/b_in/W_out/b_out per a
  follow-up Máté ask), and a best-val ★ on fig A. Ran both archs × 6 mice.
  Also answered Máté: **softmax temperature is not a free parameter** —
  `forward` returns raw logits, `F.softmax` has fixed T=1; sharpness is set only
  by the `W_out`/`b_out` logit scale (biases init to exactly 0). See memory
  `loss-comparison-v1-checkpoints`.
- **A2** — new `posterior_pca_views.py`: PC1/PC2 scatter (decoded + IO-target
  overlay), `mean + a·σ·PC` reconstruction strips, all-losses shared-target-basis
  panel. Ran all losses/archs/bases.
- **Findings:** weight **mean stays ≈0**; norm growth is **pure std broadening**;
  init is standard **Xavier-uniform** (not weird) — meeting item 5 closed.
  Target-basis **PC1≈91% var = peak position, PC2 = width**; the decoded-PCA basis
  is spiky; PCA is visibly more dispersed in latent space than CE/KL/JS/Wass.
- **B1 made turnkey:** added `--hidden-sizes` to `run_loss_comparison.py` (isolates
  each width under `run_name_h<H>`); 35/35 relevant tests pass (config/early/pca).
- Vault synced (ticked items 2–6 in `UncertaintyV1-Tasks`, logged). Figures under
  `figures/loss_sweep_plots/loss_comparison_v1/weight_evolution/` and
  `figures/posterior_pca/loss_comparison_v1/` (both gitignored).
- **Open / next:** Tier B on the cluster — run the hidden-width ablation + PPC
  `weight_decay` sweep, then write a small `plot_overfit_vs_width.py`. Open
  decision: H-ladder span (go ≤2 to force underfitting?). Tier C unchanged.
→ live plan: `nn_decoder/PLAN_2026-06-03_mate_followups.md`

### 2026-06-03 — Máté meeting → revised plan
Meeting follow-ups routed into a prioritised roadmap: Tier A (local — weight-norm
diagnostic refit, posterior PCA views), Tier B (cluster — hidden-width ablation,
PPC weight_decay sweep), Tier C (deferred). This is the current "what next".
→ live plan: `nn_decoder/PLAN_2026-06-03_mate_followups.md`

### 2026-06-03 — All-loss spat/temp comparison & cross-loss test-time eval
Compared **all five losses** (PCA, CE, KL, JS, Wasserstein; MSE dropped) for
PPC vs SBC. Extended `run_loss_comparison.py` to all losses (matched,
early-stopped, Q+L × {50,100}ms × {full,half} × 3 splits = 40 slugs); built the
`plot_all_cells.py` capstone + five per-cell drivers (`cross_loss_eval.py`
skill matrices + paired stats, `within_mouse_loss_plots.py`,
`loss_scatter_spat_temp.py`, `plot_loss_spat_temp_comparison.py`, temporal
posterior evolution in `plot_loss_sweep.py`). **Findings:** PCA loss is at
chance under KL/CE (scale-free, via shuffle-normalised skill); CE/KL are
calibrated generalists; SBC>PPC is real but small and metric-shaped (group-sig
only for raw loss under PCA/CE/KL), limited by between-mouse inconsistency
(3/6 mice carry it). Open: finish grid; pick production loss.
→ full: `documents/session_2026_06_03_loss_comparison_handoff.md`

### 2026-06-03 — PCA-loss demo + weight-evolution diagnostic + talk plan
Pedagogical "what the PCA loss sees" suite (`nn_decoder/pca_loss_demo.py`) and
output-weight-norm trajectories per cell (`nn_decoder/plot_weight_evolution_cell.py`),
both on `loss_comparison_v1`. New empirical result from weight trajectories
**complicates the weight-magnitude hypothesis** and **inverts
entropy-reg-as-weight-reg**. Serves the 2026-05-27 meeting threads (PPC
peakiness; honest spat-vs-temp).
→ full: `nn_decoder/HANDOFF_loss_demo_weight_evolution.md`

### 2026-06-02 — KL/JS entropy-λ sweep plotting
Built the KL/JS sweep plotting (`plot_kl_js_sweep.py`, `plot_kl_js_training.py`):
peakiness, knob sweeps, matched example posteriors, per-bin posteriors, held-out
fit-loss; spat|temp side by side. Identified that training-curve/weight plots
need a history-tracked re-run. Motivated the matched, full-export
`run_loss_comparison.py` (→ became the 06-03 grid).
→ full: `nn_decoder/HANDOFF_kl_js_plots.md`

### 2026-05-17 — weight_decay fix + residual-PCA basis + weight saving + loadings
Fixed the hardcoded `weight_decay=3e-4` bug; added `pca_basis ∈
{condition_mean, residual}`; trained-weight saving in `run_animal_decoder`;
`decoder_loadings_comparison.py` scaffold; spatial-vs-temporal performance plot
across targets; runner extension for L/d (+ a builtin-shadowing bug).
→ full: `NOTES.md` § "Session 2026-05-17"

### 2026-05-16 — Mate's-question feature ablation + framing audit
Implemented `feature_ablation_analysis.py` (CV regression, rate vs
temporal-variance vs order-sensitive blocks, cluster-robust paired stats,
Musall-style plots). Honest result: temporal-variance > order-sensitive is
robust; rate vs temporal-variance does not reach significance at n=6. Deep
audit concluding the PPC/SBC architectural framing is loose and the PCA loss is
structurally biased toward stim_mean.
→ full: `documents/session_2026_05_16_handoff.md` (pointer entry in `NOTES.md`)

### 2026-05-06 — Decomposition + IO coherence + stim-mean baseline + loss sweep
Decomposition analysis, IO coherence, stim_mean baseline, first loss sweep;
headline numbers recorded for the meeting.
→ full: `NOTES.md` § "Session 2026-05-06"

### 2026-05-05 (later) — Fano factor extracted to its own script + grand averages
→ full: `NOTES.md` § "Session 2026-05-05 (later)"

### 2026-05-05 — CASCADE-rate Fano caveat + per-condition window FF
→ full: `NOTES.md` § "Session 2026-05-05"

### 2026-05-04 — Basic neural features vs IO uncertainty refactor
→ full: `NOTES.md` § "Session 2026-05-04"

### 2026-05-03 — Fixed-hyperparam parameter recovery
→ full: `NOTES.md` § "Session 2026-05-03"

### (pre-2026-05-03) — IO methods audit + ANN decoder hyperparameter search
Confirmed Stage-2 IO fairness; ANN decoder Optuna hyperparameter search.
→ full: `NOTES.md` (top section)
