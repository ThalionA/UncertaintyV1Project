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
- **Tier B (cluster, batch together)** — B1: hidden-width ablation `H∈{4,8,16,32,64}`
  — runner is **turnkey** (`run_loss_comparison.py --hidden-sizes ...`); still need
  to run it + write `plot_overfit_vs_width.py`. B2: PPC `weight_decay` sweep (PPC
  alone may want 1e-3/1e-2). *Open decision:* H ladder span (go ≤2 to force underfitting?).
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
  Framework. See 2026-05-16. [open]
- **Task 2 done; Task 5 next**: residual-PCA retraining, if the residual
  partial-correlation result warrants (`documents/residual_partial_correlation.md`).
- **GPR + LOMO feature-ablation** run locally. [2026-05-16]
- **Methods-PDF rewrites** deferred until framing locked
  (`documents/methods_updates_required.md`).
- **n=6 ceiling**: group spat/temp test bottlenecked by between-mouse
  inconsistency, not trial count. [noted]

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
