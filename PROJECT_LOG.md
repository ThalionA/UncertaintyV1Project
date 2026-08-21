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

- **IO-HMM targets — six-mouse paired comparison COMPLETE (2026-08-19).** `io_hmm_v2` vs
  `io_hmm_v2_exportref`, 40 cells × 6 mice each; crossmouse synthesis + resolved priors in
  PREDICTIONS (2026-08-18 entry). Headline: mouse-0 conclusions were a broad-target special case —
  over-sharpening is graded by target concentration (mice 4–5 go to the ŝ clamp), H8 only. Next:
  by-state decoder split (gamma in shards), like-for-like check on the 'worse skill on new targets'
  result. State figure suite in `figures/io_hmm_by_state/`. [2026-08-19]

- **projflat_v1 (projection loss + flat/MSE) — COMPLETE and analysed (2026-08-04).** 70 cells + 2 local
  (`run_projflat_v1.py`; Q/100ms/second-half/tanh/patience 20). Headline: at lambda_H=0 flat/MSE matches the KL
  reference on BOTH architectures (spatial 0.93x target / 0.699; temporal 0.90x / 0.592, 6/6 mice), while the
  matched evar control is 2.25x/4.27x over-sharpened. Dim reduction (3/5/10 neural PCs) kills overfitting
  (~4-5x -> ~1.2) at ~0.1 cost in normalised loss. Reporters: `diagnostics/projflat_{report,config_axes,
  trainval_curves,tail_diagnosis,param_schematic,trial_explorer,spat_vs_temp,spat_vs_temp_bymouse,posteriors}.py`.
  **Not run:** grid x PC-ladder cross; evar controls at lambda_H=1e-2. [2026-08-04]
- **Temporal decoder CANNOT sample (2026-08-04)** — law-of-total-variance test on real data: bins can be made
  sharp (within-bin SD 4.9 deg vs Q's 19.3) but across-bin scatter caps at ~47% of sigma_Q *on train too*, so
  forcing sharp bins makes the posterior 7.5x worse than chance (6/6). V1 uncertainty here is within-instant
  (PPC), not across-time (SBC). Only untested escape: a decoder with explicit sampling dynamics.
  `playground/moment_sampling_test.py`. [2026-08-04]

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
- **DeepSets trialwise-uncertainty analysis — COMPLETE (2026-07-12).** Leakage-safe, parameter-matched
  mean / moments / DeepSets decoders, four losses and within-condition target-shuffle null; synthetic
  validation (10 datasets) recovers pure variance codes. Real Q/half/100ms grid (6 mice, 150 shards):
  DeepSets beats moments on raw KL 6/6, but **no model beats its within-condition null** and width/entropy
  gains are null; condition-mean oracle is ~20x better. Conclusion: no evidence unordered within-trial V1
  variability carries trial-specific IO uncertainty beyond condition. Full report:
  `documents/DEEPSETS_UNCERTAINTY_METHOD.md`.
- **Wide 6-axis hyperparam sweep — RUNNING on gpu1 (~75/123), analysis suite built (2026-07-08).**
  `nn_decoder/run_hyperparam_sweep.py` (123 cells, Q/half/100ms, 6 mice; full export for spat/temp/per-bin **and shuffle**).
  Done: PCA/KL/JS complete, Wasserstein ~20/22; **2-D grids + Wasserstein tail still to land**. Matched-axis analysis
  (`figures/hpsweep_shuffle/`): `shuffle_trainval_curves.py` (real-vs-shuffle train/val ÷ predict-mean + chance line),
  `shuffle_gap_vs_reg.py` (train–val gap vs hparams), **`peakiness_vs_hparams.py`** (the "how to stop overfitting" answer),
  `subspace_error_realdata.py --weight evar`. **Headline: generic knobs don't fix PCA over-sharpening — only early-stop caps
  it (still ~4.5× target); the fix is loss-side.** Meeting items #2/#4/#7/#8 done; **#6 no-hidden-layer, #3 Gaussian dropout,
  #2 loss-side need runs.** Monitor gotcha: `ls|wc -l` maxes at **57** dirs (cells share a dir across losses), not 123 —
  use the `[i/123]` log line or count `stratified_balanced.mat`. [2026-07-08]
- **2026-06-10 meeting follow-ups** — **essentially complete** (vault report
  `Projects/Uncertainty/2026-06 Loss, Orientation & Temporal-Sampling Analyses`, §1–8): shuffle-nulls, orientation,
  peaky-broad, λ_H/sampling (temporal decoder doesn't sample), dropout-vs-early-stop, averaging+smoothness (λ_smooth≈0.3
  = the fix), spat/temp head-to-head (+M2 leave-out, n_neurons), train–val gap (static offset; dropout barely closes).
  **Residual:** smoothness vs `shape_lambda`-Brier production-fix head-to-head; Mouse-2 "what's different about M2?"
  exploratory (deferred). [2026-06-17]
- **Similarity Framework — generative support now from TWO learning rules.**
  `si_network_model` (Hebbian) and `rnn_rl_model` (actor-critic RL; new 2026-06-13)
  both land on the template-`Δμ` readout at ~0.96–0.99 efficiency, `r≈0.85`, with
  RD-2 M3−M1≈0. Robust to learning rule. Consider folding into the vault note as a
  second positive control. [2026-06-13]
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
| `documents/DEEPSETS_UNCERTAINTY_METHOD.md` | Full method, synthetic validation and six-mouse result for the mean/moments/DeepSets uncertainty analysis. |
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

### 2026-08-19 — six-mouse paired comparison landed: mouse-0 headlines were a broad-target special case

> **⚠ VOID 2026-08-21 — every cell in this entry trained with an unintended `weight_decay=1.2e-4`,
> which annihilates the flat-projection cells (‖W_in‖ 0.65 vs projflat_v1's 6.50) and shrinks the evar
> ones. Re-running as `io_hmm_v3` with wd=0. Findings below are not citable. MISTAKES 2026-08-21.**
Both arms finished overnight (`io_hmm_v2` HMM targets, `io_hmm_v2_exportref` old export Q; 40 cells × 6
mice each, zero errors; verified from saved shards: 72/91 bins, alignment 0.975–0.993). The **s_hat
ground-truth gate passed all six mice in both arms** (worst recovery 0.05% across the 6–28° old-width
range) — first time it was run beyond mouse 0. Synthesis `diagnostics/io_hmm_vs_export_crossmouse.py`
→ `figures/io_hmm_vs_export_v2/crossmouse/` (dumbbells, headline-survival grid, disagreement map);
adversarial recheck re-derived every median independently (0 discrepancies) and byte-verified CSV
provenance against fresh v2 reruns. **Priors resolved (PREDICTIONS 2026-08-18):** (b) ⚠ invalidated —
evar-proj H8 over-sharpening is tamed in mice 0–3 (→0.57–0.97) but hits the ŝ clamp in the temporal
decoder of mice 4–5; it is graded by the NEW target's concentration (ρ +0.77 with marginal R), H8 only,
the linear decoder never tamed (3.3→4.0). (c) ✓ but like-for-like: KL/JS are λ_H-inert in BOTH arms
(≤4% vs evar 53–359%) — a property of calibrated losses, not the targets. (d) ↔ — KL-vs-projection
disagreement is real for the evar family (6/6 by sign spat), JS does not replicate. (e) ✓ — m2, m4, m5
each break a headline. Across the board projection skill worsens old→new in 5–6/6 mice and even
KL-trained decoders' KL skill worsens in 4–5/6: decoders beat the predict-mean null by LESS on the
broad HMM targets; flat-projection cells sit on the null. **Earlier in the day:** six-mouse state figure
suite (`figures/io_hmm_by_state/`, 22 figs; states differ in concentration not location; one
"engaged" state per mouse matches across animals by function; stimulus barely predicts state, previous
state + running speed do) and a decision-target semantics fix (`PS_Go_G_tr` is Go-category BELIEF,
`PS_choice_G_tr` is the psych curve — `targets_dec` now uses the latter).
**Open:** (a) by-state decoder split (gamma is in the v2 shards; no retraining); (b) the JS/KL "worse on
new targets" result deserves the like-for-like treatment before it becomes a claim — null compression on
broad targets is a candidate confound; (c) collaborator Qs still open: choice-coding mismatches, the
beyond-90° mass; (d) promote the `over-claimed` ledger rule (6 entries) to CLAUDE.md if Theo agrees.

### 2026-08-12 — meeting items 5 + 3: per-mouse spat/temp scatter & density; reduced-rank (rr8) support
Worked the 2026-08-05 meeting to-do list (Máté/Nathalie/Ishan). **Item 6 (use the collaborator's IO-HMM
posteriors) is dropped for now** — Ishan hasn't shared the full pkl, so item 1 (stimulus-dependent
GLM-HMM/IO-HMM states) is presumed blocked with it.

**Item 5 — per-mouse spatial-vs-temporal figures** (`nn_decoder/scatter_spat_temp_by_mouse.py`, new;
`1ef9c10`, `de20b9d`). Square log-log per-trial scatter, one panel per mouse, all 4 projflat configs
(n=2186, 6 mice), dots coloured by the **average difference between the individual temporal bins and the
full temporal model**. Verified from the data that the temporal `decoded` **is** the arithmetic mean over
`decoded_samp` (max abs diff 1.4e-8), so the "full model" is exactly the Jensen average; the default
colour is mean_bin KL(bin ‖ average) in bits (`--colour l1|gain` for alternatives). Second view added:
`--mode density` draws a filled log-space KDE plus one 50%-of-peak contour per spread tertile.
Each cell is scored under **its own** training weighting (per-cell `explained_var`: eigenvalue spectrum
for EVAR cells, uniform 1/91 = MSE for `flat_evar=1`), matching
`diagnostics/projflat_spat_vs_temp_bymouse.py` — compare within a config, not across.
**A first read of the scatter was wrong and the density view caught it:** high-spread trials sit
up-and-right, which is movement ALONG the diagonal = harder trials, NOT "temporal worse". The
perpendicular measure `d = median log10(temp/spat)`, per mouse, gives high-minus-low tertile
**−0.067 (1/6 mice +) lin-flat · +0.025 (3/6, null) h8-flat · −0.142 (1/6) lin-EVAR · +0.195 (6/6) h8-EVAR** —
so bin disagreement specifically penalises the temporal decoder in **exactly one of four configs**, and
runs backwards in both linear ones. Trap logged in `GOTCHAS.md`. Caveat recorded in the code: the
grouping variable is derived from the temporal decoder's own output and re-enters the outcome, so this is
descriptive, not an independent test — a target-side/stimulus-side grouping variable is needed for that.

**Item 3 — reduced-rank regression** (`59fae2c`, `9925500`). `hidden_sizes=[H]` + `activation='identity'`
= `Linear(n,H) → Linear(H,91)`, an affine **logit** map of rank ≤ H. Measured: `lin` rank 40 / 3731 params,
`rr8` rank 8 / 1147 params, `h8` rank 8 + tanh — so rr8-vs-lin isolates the rank bottleneck and rr8-vs-h8
the non-linearity. **No Config change was needed** (`activation_function` was already a field, already in
`to_legacy_dict`/`model_params`/checkpoints). **Trap closed:** `activation='identity'` previously hit the
`.get(..., nn.ReLU())` fallback and trained ReLU *silently* — an rr8 cell would have been a mislabelled
ReLU net that looked right in every diagnostic; unknown names now raise (no config.yaml on disk is
rejected). The two sweep orchestrators' hand-copied `VALID_ACTIVATIONS` mirrors had already drifted and
would have rejected `identity`; both now import it from `nn_classifier`. 5 new tests; 112 pass on the
architecture files, 127 + 1 skipped on the standard core filter. Note the tests use **additivity**, not
`f(2x)==2f(x)` — biases init to zero and ReLU is positively homogeneous, so ReLU passes the homogeneity
check exactly (0.00e+00) and a homogeneity-only test would bless the bug.
`run_projflat_v1.py` gains 3 cells/mouse (77 total, was 74): `rr8_raw_l0_d0_w0`, `rr8_raw_KLref`,
`rr8_raw_EVAR` — raw input only (`n_neural_pcs` is input-side rank, rr8 is map-side; crossing confounds)
and wd=0 only (L2 on {W1,W2} penalises the **nuclear** norm of the product, so rr8 and lin are not matched
regularisation at the same wd).

**Item 3 RESULTS — rr8 ran on gpu1 and answers the meeting question** (`633edef`;
`diagnostics/projflat_rank_vs_nonlinearity.py`, new). Launched by the agent (tmux `rr8`), 3 cells × 6 mice
= **360 fits, 0 errors**; results pulled to `results/projflat_v1/rr8_*`. Peakiness (decoded peak / IO
target peak, 1.0 = on target):

| | lin (full rank) | → rr8 (rank 8, no NL) | → h8 (rank 8 + tanh) | consistency |
|---|---|---|---|---|
| SPAT EVAR | 5.91 | 4.07 | 2.25 | rank **6/6**, tanh **6/6** |
| TEMP EVAR | 5.72 | 5.29 | 4.27 | rank **2/6**, tanh **6/6** |

So on the spatial decoder under the severe weighting, **rank and tanh each remove ~37% of lin's excess
over 1.0** — comparable contributions, neither factor alone explains the over-sharpening. The **tanh step
is 6/6 mice in every condition; the rank step is not** (2/6 for EVAR/temporal, where the mean is carried by
two animals) — the figure plots per-mouse lines so that shows rather than hides. rr8 also roughly **halves
overfitting** vs lin (EVAR temp 12.39→4.36, spat 9.55→4.52) at essentially unchanged projection loss.
**Cross-metric disagreement again:** projection loss — the *training* metric — spans only 0.46–0.66 across
arches whose peakiness spans 0.86–5.91, i.e. it is nearly blind to a 6× difference in over-sharpening.
No cell tripped the collapsed/suppressed guard. (`|W_in|` is deliberately not compared across arches:
lin's `W_in` **is** the whole 91×N map while rr8/h8's is 8×N.)

**Cluster access note:** `nn_decoder/CLUSTER_LAUNCH.md` said "the agent cannot ssh/rsync gpu1" — false
since 2026-08-08, now corrected (`dbb2b6b`). Two traps added there: the **remote git HEAD lies** about what
code will run (gpu1 was 10+ commits behind while several of its *files* were already current, because past
sessions rsync'd source up without committing there — grep the symbol on the remote file instead), and
macOS has no `timeout`. Recorded launch discipline: remote `--dry-run` → `--smoke` into a separate tree →
**verify the changed property from the saved `.mat`** → full tmux run. Step 3 is what proved the cells were
genuinely rank-8 (composite `W_out@W_in` rank exactly 8, all 4 arches) rather than a silent-ReLU net.

**Projection loss for all 9 configs, spatial vs temporal — and a median cross-check that overturns the
mean** (`24f238b`, `diagnostics/projflat_spat_vs_temp_bymouse.py`). Added rr8 + the KL-trained anchors to
both deliverables (per-mouse within-animal bars; across-mice bars with the paired t over n=6 and per-mouse
points), ordered lin→rr8→h8 within each weighting block. Verified from stored `explained_var` that flat
cells carry uniform 1/91 (= MSE) while evar **and KLref** carry the eigenvalue spectrum; KLref is labelled
as scored on a metric it wasn't trained on. **New `--trial-stat {mean,median}`, and the choice changes the
conclusion:**

| per-mouse summary | result |
|---|---|
| **mean** | temporal beats spatial in **all 9** configs, p 0.0066–0.058, 5/6–6/6 mice |
| **median** | **7 of 9 ns**, and the two survivors point in OPPOSITE directions (h8-flat temporal better p=0.025 6/6; rr8-EVAR temporal **worse** p=0.027 1/6) |

So **the temporal decoder's projection-loss advantage is a TAIL effect, not a typical-trial effect** —
Jensen-averaging ten per-bin posteriors shrinks extreme predictions and removes the catastrophic trials
that dominate a mean. This is the *same* tail behind the retired "linear + flat/MSE + raw SPATIAL is worse
than chance" claim, and the mean bars reproduce that artefact exactly (lin-flat spatial **1.156 mean vs
0.452 median**). Trap logged in `GOTCHAS.md`; median pairs with Wilcoxon + bootstrap SE, mean keeps paired
t + SEM.

**Open items / next steps**
- **Re-examine earlier spatial-vs-temporal conclusions that rest on means** — this cuts across the projflat
  story, not just the rr8 cells. Report both statistics; if they disagree the claim is about variance /
  robustness, not typical decoding accuracy.
- The projflat diagnostics hardcode arch token lists (`projflat_report.py`, `projflat_config_axes.py`,
  `projflat_spat_vs_temp.py`, `projflat_trial_explorer.py`) — they still **skip** rr8 until each gains the
  token. The new `projflat_rank_vs_nonlinearity.py` and `scatter_spat_temp_by_mouse.py` do carry it.
- **gpu1 has uncommitted local edits** on `diagnostics/predict_mean_baseline.py`,
  `diagnostics/subspace_error_realdata.py`, `DATA_MAP.md` — deliberately not overwritten during the rr8
  sync (which used `--backup --suffix=.pre_rr8_20260812`). Decide whether that work is worth keeping; the
  remote repo is otherwise far behind local.
- Meeting items still untouched: **lazy learning through initialisation** (weights barely moving from init;
  `loss_comparison_v1` already saves epoch-0 + snapshot weights, so this needs no re-run) and the
  **minimum-norm solution** (closed form; no analytic RRR/min-norm fit exists anywhere in the repo — new
  code, would sit beside `stim_mean_baseline.py`, which already exposes itself as an extra `Dist` arch key).
- For a non-circular test of the bin-spread effect, group trials by a target-side or stimulus-side variable
  instead of by the temporal decoder's own output.

### 2026-08-08 — collaborator's IO-HMM refit wired in; old-vs-new posterior comparison; mouse-0 sweep launched on gpu1
The new `data/fitted_data_and_posteriors.pkl` (ideal-observer **HMM**, via Slack) replaces the export's
91-bin Q as the perception target. **The local copy is TRUNCATED at 33 MiB** (Slack download cut off;
stream wants ~55 MB more mid-array) — mouse 0's per-trial fields recover fully via `pickle._Unpickler`
memo salvage; **mice 1–5 need Theo to re-download**. Support is **72 circular bins × 2.5° over
[0°,180°)**; trials align to the export as an in-order barcode subsequence (mouse 0: 940/945, dropped
[215,368,568,809,944]; **12 genuine choice-coding mismatches on easy trials — ask the collaborator**).
Wired end-to-end (committed `b661bec`): `io_hmm_data.py` loader (jax-free stubs, alignment +
choice-agreement floor 0.95, partial-recovery opt-in), Config `target_source`/`io_hmm_pkl_path`/
`io_hmm_allow_partial` (no-op defaults; legacy `'real'` constant replaced), `run_experiment` override +
`target_provenance` in every save, `run_io_hmm_v1.py` (24 cells: PCA/PCA-flat/KL/JS × H8/lin ×
λ_H {0,1e-3,3e-3}, projflat conventions). Tests green (`OMP_NUM_THREADS=1`), real smoke trains 72-bin
posteriors both architectures. **Old-vs-new comparison** (`diagnostics/compare_io_hmm_vs_export_posteriors.py`,
`figures/io_hmm_vs_export/`): new posteriors are near-uniform and near-constant-width (SD 24–25° on
~every trial vs old 12–33°; entropy r=0.52, means r=0.73 compressed to 30–60°; median TV 0.32;
**median 48% of mass beyond 90°** — folded for comparison; decoder target keeps raw 72). Old
low-contrast posteriors are bimodal-U; new are flat-with-tilt. **Width-sensitive machinery has little
to grip on these targets.** Priors registered in `PREDICTIONS.md` (2026-08-08 entry).
**Launched on gpu1** (ssh now allowed — see GOTCHAS/memory update): tmux `iohmm`, mouse 0 × 24 cells,
`--allow-partial`, log `nn_decoder/iohmm_v1_m0.log`, results `results/io_hmm_v1/`.
**Mouse-0 landed same day** (~1h50, 24/24 cells; results rsynced down; scorecard
`diagnostics/io_hmm_v1_scorecard.py` → `figures/io_hmm_v1/scorecard_m0.png`). **Headline
surprises (PREDICTIONS resolved, provisional n=1; ⚠ the peakiness claims here were CORRECTED
2026-08-12 as a bin-width artefact — see that entry and GOTCHAS):** PCA's over-sharpening attractor
looked largely absent on the broad targets (peakiness 0.78–1.05 at λ_H=0; only linear-temporal at 2.5×);
capacity matters again (linear spatial KL fit 2× worse than H8, reversing projflat); KL/JS
calibrated (0.84–0.93) and beat both nulls; λ_H exactly inert for KL/JS but destabilises
PCA-temporal non-monotonically; the null hierarchy compresses (shuffle ≈ predict-mean).
**Open:** (a) full pkl re-download → launch mice 1–5 (drop `--allow-partial`); (b) projection-loss
judging + train/val-history check on mouse 0 (both-metrics rule); (c) collaborator Qs: choice-coding mismatches + is the [0,180)
smear meaningful (fold-support knob if wanted); (d) time-resolved `PS_x_G_tr` (n,~202,72) arrives with
the full file — a natural temporal-decoder target follow-up.

### 2026-07-29 — the IO's "noise variance" is closed-form, and it is NOT the shape term
Two pieces. **(1) Best-cell head-to-head under three normalisations**
([`diagnostics/spat_temp_best_cell.py`](nn_decoder/diagnostics/spat_temp_best_cell.py), one figure
per scoring metric, 3 norms × {across animals, within animal}, stats on the panels). Headline: the
**shuffle normalisation flips the sign and manufactures significance** — KL Δ(temp−spat) is −0.015
(p=0.32) raw and −0.024 (p=0.20) under predict-mean, but **+0.059 (p=0.0065)** divided by the
shuffle. Cause measured here: the shuffle control is architecturally biased, the temporal shuffle
decoder's loss being 1.10× (projection) to 1.13× (KL) *lower* than the spatial one, because
Jensen-averaging ten per-bin posteriors helps a no-information decoder too. **Do not use the
shuffle null for spatial-vs-temporal.** On the two defensible nulls the architectures remain
indistinguishable. Note for readers: within-animal t/p are identical down a normalisation column
(the divisor is one per-mouse scalar on both arches); only the across-animal test changes.

**(2) The IO noise variance** ([`diagnostics/io_noise_variance.py`](nn_decoder/diagnostics/io_noise_variance.py),
`figures/io_noise/`). `post_s_marginal` is a marginal over the latent measurement m and the export
ships the weights (`IO.animals[i].inferred.m_posteriors`, 181/trial), so the mixture comes apart
**exactly** — no resampling. Reconstruction matches the export to 1e-11; every decoder target is an
IO row to 5e-8 (which also confirms animal i ↔ mouse_i). Results: within a trial the noise SD
tracks the target's own shape (peak 0.018 at the 90° prior mode, trough 0.0019 at the 44° boundary)
and is a **flat ~44% of the target across the support**; across trials it is **16.1 ± 1.1%** of the
target's second moment, flat in orientation and driven by κ alone (~0 below κ=0.1, ~0.25 above
κ=1). In the PC basis the **noise decays far more slowly than the signal** (evar 0.758/0.172/0.035
vs noise 0.325/0.221/0.148 on PC1–3), so `evar/noise` weighting flattens the loss in the *same
direction* as `shape_lambda` but far more weakly — effective PC count 1.65 → 2.54 vs 85.2. **Máté's
noise normalisation and the shape term are not equivalent.** Nothing in any loss was changed.

Open items / next steps:
- If the noise-normalised weights are worth a run, they are a ~6-cell probe (`evar/noise` with the
  numerical floor, vs `evar`, vs `shape30`) — but the effective-PC numbers predict it lands much
  closer to plain `evar` than to the cure.
- `m_posteriors` is a *posterior* over m (trial-specific within a stimulus cell; `conf_only` fit
  with a velocity confidence readout). Gaussian velocity links on `g(m)`, `|g(m)|`, `max_s p(s|m)`
  all failed to reproduce it. Harmless here, but ask Máté what the exact conditioning is.
- Contrast × dispersion is **not crossed** in this design — plot against κ, never one factor alone.
- Still unconsumed: the 2-D interaction grids of `hpsweep_v2` (~half the run) have no analysis.

### 2026-07-28 — λ_H cannot buy PEAKY BINS with a calibrated average: the ratio is pinned at ~1.15
`shapefix_v1` (6 cells, shape_lambda=30 × λ_H ∈ {0,3e-3,1e-2,3e-2,1e-1} + one with dropout 0.5,
early stopping, REP 3, 6 mice, ~18 min locally). The target state is the sampling-code signature —
sharp individual time bins whose Jensen average recovers the broad IO posterior.

| λ_H | per-bin | average | ratio | Jensen | projT | KL_T | projS | KL_S |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0614 | 0.0542 | 1.14 | 0.116 | **0.54** | **0.74** | 0.57 | 0.79 |
| 3e-3 | 0.0671 | 0.0582 | 1.16 | 0.149 | 0.55 | 0.76 | 0.57 | 0.79 |
| 1e-2 | 0.0750 | 0.0643 | 1.17 | 0.151 | 0.62 | 0.84 | 0.57 | 0.79 |
| 3e-2 | 0.0632 | 0.0565 | 1.13 | 0.075 | 0.86 | 1.17 | 0.57 | 0.79 |
| 1e-1 | 0.0347 | 0.0281 | 1.23 | 0.057 | 1.14 | 1.60 | 0.57 | 0.79 |

- **The answer is NO, for an informative reason.** The per-bin ÷ average ratio never moves —
  **1.13–1.23 across a 100× range of λ_H**. The knob raises or lowers the two *together* and never
  separates them. Worse, the **Jensen gap FALLS** at high λ_H (0.149 → 0.057): the bins become
  *more* alike, the opposite of a sampling code.
- **Mechanism.** Sharp bins with a broad average requires the bins to **disagree in location**.
  λ_H penalises per-bin entropy, but **nothing in the objective rewards bin-to-bin diversity**, so
  the cheapest response is to sharpen every bin toward the *same* place — which would sharpen the
  average too, and the shape term forbids that. The network's compromise is to not sharpen at all.
  **A real fix needs an explicit bin-diversity term, not an entropy penalty.**
- **Interpretation control.** With early stopping on, the temporal best epoch collapses
  **162 → 148 → 27 → 9 → 4** as λ_H rises while spatial holds at 83 (λ_H is temporal-only, so this
  doubles as a plumbing check). The apparent broadening at high λ_H is an **undertrained** decoder,
  not genuine broadening. Without early stopping (`hpsweep_v2`) the same knob trains to completion
  and drives temporal peakiness to 0.79, wrecking the average. **Neither regime reaches the target.**
- **λ_H = 0 is the best cell** (projection 0.54 / KL 0.74 temporal) — turning the entropy penalty
  off is a small improvement over the 3e-3 production value, consistent with GOTCHAS.
- Also this session: the softmax-Jacobian analysis (`diagnostics/jacobian_gate.py`). `dL/dz = p − t`
  for KL is an exact identity (verified to 6.7e-16); descent on an over-sharp decoder is strongly
  restoring for KL (−0.68) and JS (−0.64), barely restoring for projection (−0.12), and **actively
  sharpening for Wasserstein (+0.10)**. **Critical control: on a calibrated decoder all four losses
  are indistinguishable (−0.15…−0.17)** — so the softmax gate is a **ratchet, not a driver**: it
  explains why an over-sharp decoder cannot recover, not what pushed it out (that is the blindness).
  **This contradicts the mechanism note's "the width subspace has no restoring force"** — it is
  1.8% (spatial) / 0.07% (temporal) of KL's, not zero, and the gradient-energy centre-of-mass is
  essentially identical across losses (PC 12.2 vs 13.6), so it is *global attenuation with
  sharpness*, not a subspace-specific effect. §4½ of the vault note needs softening.

### 2026-08-04 — projflat_v1 scrutiny pass: an overfitting BUG fixed, the "worse than chance" result retired, and the linear model is the BIG one
Theo challenged three results in turn; two did not survive as stated. Everything below is on `projflat_v1` (70 cells + 2 added locally).
- **BUG FIXED (load-bearing readout): the overfitting ratio was read at the FINAL epoch, not the restored best epoch.** `fit_model` restores the best-val weights (`nn_classifier.py:676`) so every performance number comes from `best_epoch`, but `_overfit_ratio` used `[-1]` = `best_epoch + patience` — a different, further-overfit model. Inflated the ratios **14-68%** (h8 flat spat 4.85 -> **3.32**; h8 evar spat 6.88 -> 4.09; lin evar spat 13.49 -> 9.55). Config ordering preserved, so nothing qualitative flipped. `_overfit_ratio(at='best')` is now default and falls back to the last epoch when `best_epoch` is absent, so **every patience-0 hpsweep figure is bit-for-bit unchanged**.
- **"linear + flat/MSE + raw SPATIAL is worse than chance" is RETIRED as stated — it is a MEAN artefact of a 1% tail.** Median per-trial MSE/predict-mean is **0.43 (2.3x BETTER than chance)**; only 45% of trials are individually worse; p99 = 13.95 and the **worst 1% of trials carry 28% of all the squared error**. **Seed-stable**: means 1.156 / 1.268 / 1.264 across three independent restart draws (seed 0 was the mildest), medians 0.586/0.596/0.604 — so it is a property of the configuration, not the draw. Every capacity reduction removes the tail (hidden layer 10%, input PCA 7%, temporal average 10%). New `diagnostics/projflat_tail_diagnosis.py`.
- **PARAMETER COUNTS CORRECTED — the LINEAR model is ~6x BIGGER than H=8.** I had reported 5,915 vs 520 by counting only `W_in`. True totals at 108 neurons: **linear 9,919 (rank <= 91), H=8 1,691 (rank <= 8)**. With 91 output bins an 8-unit hidden layer is a **rank BOTTLENECK, not extra capacity** — so "linear" here means HIGHER capacity, which is exactly why it tails. The parameter ordering FLIPS below ~9 input dims (k=3: linear 364, H=8 851) but effective **rank** is 3 for both and both behave — **rank, not parameter count, predicts the tail**. Schematic + arithmetic: `diagnostics/projflat_param_schematic.py`.
- **The "linear" models are NOT linear regression / ridge.** They are `softmax(Wx+b)` — the parameterisation of multinomial logistic regression — fitted by **least squares** on a soft 91-D target, by Adam (lr 1e-3, <=200 ep, patience 20, 5 restarts on val), wd=0 in the main cells. So: GLM structure, non-GLM objective, and **non-convex** (softmax + squared error), which is why restarts exist. Corollary worth reusing: `KL_calc = cross_entropy - entropy(target)`, so the **KL cells ARE the proper GLM / soft-label maximum-likelihood fit** — the flat-vs-evar-vs-KL comparison at the linear architecture is literally "same GLM, three fitting criteria".
- **Ran 2 cells locally** (`--arms evarlam`): evar controls at lambda_H=3e-3, completing the flat-vs-evar contrast at lambda_H>0. Result: the SAME penalty produces opposite failure modes — flat/MSE temporal collapses (peaky 0.35-0.81x, overfit ~1.0, **1.4-1.8x worse than chance**, the early-stop bail) while evar temporal over-sharpens as intended (5.6-9.0x) and still scores ~0.55. Loss scale decides which.
- **Figures added** (all `figures/projflat/`): `fig9_tail_diagnosis`, `fig10_param_schematic`, `cfg_trainval[_pc-flat|_pc-evar|_lam0p003]` (train/val curves per config, best epoch marked — **val is FLAT while train falls, which is the visual answer to "how can everything overfit 3x and still beat the null"**), `cfg_{peakiness,overfitting,performance_shuffle}_{pc-flat,pc-evar,lam0p003}`, `projflat_trials_{lin,h8,lin_EVAR,h8_EVAR}` (per-trial spat-vs-temp scatter + exemplars + temporal-bin heatmaps), plus regenerated fig1-fig8.
- **Verified while bug-hunting** (all passed): the `*_shf` twin is trained on scrambled targets but evaluated against the TRUE test targets; spatial decodes are **bit-identical** across lambda_H (max|diff| 0.0), confirming the penalty is temporal-only and seeding is deterministic.
- **New GOTCHAS (5):** best-epoch overfitting ratio; a LOW overfit ratio can mean "fits train worse" not "generalises better"; the predict-mean null is 10-23x weaker under evar than under MSE (4-6.5x), so "beats chance" is not comparable across weightings; report median + top-1% error share before claiming a chance-normalised mean > 1; lambda_H under flat/MSE is a loss-scale mismatch not a sign bug.
- **Open / next:** the grid x PC-ladder cross was never run (regularisation swept at raw input only); evar controls at lambda_H=1e-2 exist in the runner spec but were not run; `plot_io_posteriors_by_contrast.py` (Theo's, untracked) still unrun/unverified.

### 2026-08-04 — CAN the temporal decoder sample? Moment-matching test on real data: NO (variance is within-instant, not across-time)
Theo asked whether the temporal (SBC) model can produce sharp individual-bin posteriors whose moments reproduce Q's mean+variance — genuine sampling. Built a standalone leakage-safe test (`playground/moment_sampling_test.py`, touches no load-bearing code) using the law of total variance: sigma2_Q = Var_t[mu_t] (across-bin scatter) + E_t[s2_t] (within-bin width). PPC puts it within-bin; SBC puts it across-bin. Trained the sampling decoder with a moment-matching objective (push scatter -> sigma2_Q, within -> 0, mean -> mu_Q) vs a standard KL head, 6 mice, held-out.
- **Answer: NO.** Bins CAN be made sharp (within-bin SD 4.9 deg vs Q's 19.3) but across-bin scatter caps at ~47% of sigma_Q (never >61%, and the cap holds ON TRAIN too, so it's representational not overfitting). Forcing sharp bins therefore wrecks the posterior: Jensen average ~8x over-peaked, KL(avg||Q) 7.5x worse than chance, 6/6. The standard KL objective (broad bins, ~zero scatter = PPC) reproduces Q at ~chance. So the posterior's variance can be carried WITHIN-instant (PPC) but not moved ACROSS-time as scattered samples.
- **Per-trial tracking weak:** held-out scatter-vs-sigma_Q r=0.35, slope 0.09 (falsifier r>0.4 & slope>0.5 not met in any mouse). Even when a couple of mice show r~0.58, scatter barely responds (slope ~0.1).
- **Scope/caveats:** this SBC readout is a shared per-bin MLP with NO explicit sampling dynamics; data is second-half/100ms; linear moments (which over-state scatter via outliers). A model with explicit temporal/latent sampling dynamics is untested and is the one way the negative could flip.
- Prediction registered before the run and scored in PREDICTIONS (sharpness confirmed, marginal-scatter match falsified — it fails at the easier bar; per-trial tracking weak as predicted). Figure `figures/projflat/moment_sampling_test.png` (gitignored); script committed.
- **Open/next:** if pursuing SBC seriously, the test to run is a decoder WITH sampling dynamics (recurrent/stochastic latent) — otherwise the within-vs-across-time partition is settled for the per-bin-readout family: V1 uncertainty here is within-instant.

### 2026-08-04 — projflat_v1 landed (70 cells): flat/MSE + patience 20 + lambda_H=0 MATCHES KL on both arches; lambda_H is the temporal killer
Ran overnight, all 70 cells down. New `diagnostics/projflat_report.py` (3 axes, dim x weighting + regularisation grids; 140-row metrics.csv). Judged on peakiness, overfitting AND performance-under-both-metrics.
- **Headline: at patience 20, raw input, lambda_H=0, flat/MSE lands on the KL reference for BOTH architectures** — spatial 0.93x target / KL loss 0.699 (5/6), temporal 0.90x / 0.592 (**6/6**). The matched evar control (same patience/input) is 2.25x/4.27x over-sharpened, KL 1.38/1.85 (3/6, 0/6). KL ref itself: 0.90x/0.86x, 0.668/0.567. So **removing the eigenvalue weighting (= plain MSE) makes the projection decoder calibrated, and early stopping keeps it alive.** This reverses the flatevar_v1 (patience 0) temporal-flat failure.
- **lambda_H is the temporal killer, isolated at dropout=0/wd=0:** temporal peaky/tgt 0.90 -> 0.40 -> 0.33 and KL loss 0.592 -> 1.558 -> 1.604 for lambda_H 0 / 3e-3 / 1e-2, beats-chance 6/6 -> 0/6 -> 0/6. Any entropy penalty pushes the flat temporal decoder below target — but [corrected 2026-08-04] this is a LOSS-SCALE MISMATCH, not the penalty broadening. Under flat/MSE fit-loss is ~3e-4 while lambda_H*H is ~45-150x bigger, so val fit-loss rises from epoch 0 and early-stopping saves a near-init (uniform) model. The SAME lambda_H at patience 0 sharpens per-bin posteriors to near-deltas (per-bin H 0.90) -> over-sharpened average (6.4x); the penalty works, early-stop timing sets the outcome. Working temporal config is lambda_H=0. GOTCHAS + diagnostics/projflat_lambda_diagnosis.py.
- **Input dimensionality (raw vs 3/5/10 PCs):** flat/MSE beats chance 6/6 at every k, both arches; dim reduction collapses overfitting from ~4-5x (raw) to ~1.1-1.4 (any PCs) for ~0.1 more normalised loss. evar stays over-sharpened at every k (1.8-3.5x) — shrinking input dims does NOT fix the weighting. Architecture (H=8 vs linear) barely matters on any axis.
- **The projection metric is blind, on real cells:** flat and evar sit on top of each other under the projection metric (~0.53 spat, ~0.46 temp) while KL separates them 2-3x. The both-metrics rule, vindicated (fig1 g/h).
- **Not isolated:** patience-20 vs patience-0 at lambda_H=0 (no such patience-0 cell exists). At lambda_H=3e-3 the regimes give opposite signs, so patience matters, but its clean isolation is untested.
- **Run design (committed `5670f4d`):** `run_projflat_v1.py` — Q/100ms/second-half/tanh/val0.2/patience20, flat_evar, 2 arch x 4 input x (lambda_H x dropout x wd grid) + 8 evar controls + 2 KL refs = 70 cells; wd ladder {0,1e-6,1e-5} (1e-4 excluded — annihilates). Prior registered + scored in PREDICTIONS.
- **Open / next:** the wd=1e-5 column starts to suppress even at lambda_H=0 (KL loss -> ~1.0, cells flagged); best operating point is lambda_H=0, wd in {0,1e-6}, dropout free. Posterior galleries + spat-vs-temp for projflat not yet made (the flatevar_* plotters are cell-name-specific). Figures: `figures/projflat/projflat_fig{1,2,3}`.

### 2026-08-03 — flatevar_v1 analysed: 30/36 cells VOID (weight decay annihilated the flat-evar arm); spatial fixed, temporal NOT
Built `diagnostics/flatevar_report.py` and ran it on the downloaded run. **The headline is a design error of mine, caught
by the reporter's own guards.**
- **`weight_decay=1e-4` is not scale-matched to flat weighting, and it kills the network.** Flat weighting puts 1/91 on
  every PC where evar puts ~0.5 on the leading location PCs, so the fit gradient is ~45x weaker while Adam's `wd*theta`
  is unchanged. Verified directly: `A_flat_base` has **||W_in|| = 0.0000** (evar baseline: 3.29) and emits the uniform
  posterior (peakiness 0.0116 = 1/91, normalised loss 1.55). **30 of 36 cells affected**, incl. the ENTIRE neural-PC
  ladder. `B_flat_linear` is suppressed not annihilated (||W_in|| 0.42 vs **23.0** for its own wd=0 twin).
  Mirror of the documented `shape_lambda` trap (that one *dilutes* wd up to 28x; this one *amplifies* it).
- **The tell:** peakiness identical to **4 d.p.** across widths 2/4/8/16/32 and dropout 0/0.25/0.5. Knobs don't agree to
  four decimals; dead decoders do. Both traps → GOTCHAS.
- **What survives (the `*_wd0` cells) — one real result and one clean falsification:**
  - **SPATIAL IS FIXED — and it reaches calibrated-divergence parity.** `A_flat_wd0` spatial **1.02x** target,
    normalised loss **0.948**. Against the wd-MATCHED evar baseline (hpsweep_v2 wd=0): over-sharpening 4.33x -> 1.02x
    (p=0.0106, **6/6 mice**), normalised loss 2.625 -> 0.948 (p=0.0371, **6/6 mice**). For scale, KL is 1.01x/0.938
    and JS 0.99x/0.841 — **flat weighting makes the projection loss behave like a calibrated divergence, spatially.**
    P1 confirmed for spatial (loss slightly worse than the predicted 0.72-0.85 — the lambda->inf limit is *not*
    better than shape_lambda=0.3). Caveat: only 2/6 mice are individually below chance (KL 3/6, JS 4/6).
  - **TEMPORAL IS NOT.** 6.41x (H=8) / 7.44x (linear) vs the predicted <=1.3x. **P2 falsified.**
    *[corrected same day: I first wrote "and the loss goes 8.18 -> 18.23, twice as bad" — that was my OWN
    weight-decay confound, reading flat@wd0 against evar@wd1e-4. `hpsweep_v2`'s matched evar pair shows removing wd
    takes the EVAR temporal loss 8.508 -> 17.093 by itself. Matched on wd, flat weighting leaves the temporal loss
    UNCHANGED (18.23 vs 17.09, p=0.37, 2/6 mice) while halving over-sharpening (12.93x -> 6.41x, p=0.0007, 6/6).]*
  - **Input-side PCA alone is not the fix** — `C_evar_npc16` (survived, evar-weighted) 4.5x target vs 6.14x full
    population. Bonus prior confirmed.
- **The confound I cannot yet remove:** `entropy_lambda` is temporal-only and is *also* unscaled against a 45x-weaker
  gradient, so under flat weighting it is relatively ~45x stronger — and it sharpens (`A_flat_lam1em2` temporal 14.6x).
  **The disentangling cell flat+wd0+λ_H=0 does not exist in v1**, because an OAT sweep varies one axis from a baseline
  that was itself broken. That is the point of v2.
- **Files:** `diagnostics/flatevar_report.py` (new — 4 figures + metrics.csv; `collapsed()` checks ||W||≈0 AND
  peakiness≈1/n_cats, `lobotomised()` catches partial suppression via the broader-than-target-yet-worse-than-chance
  quadrant; both mark points on the figures so a flat line through dead cells can't read as a finding).
  **`run_flatevar_v2.py`** (new — 25 cells, **wd=0 throughout including the references** so the contrast isn't
  confounded; decisive cell `F_flat_h8_lam0`; + an annihilation-threshold probe at wd=1e-6). Dry-run + smoke green.
- **Open / next:** **Theo launches `flatevar_v2 --arms core temporal`** (10 cells) — those two arms answer it. Then
  point `flatevar_report.py` at v2 (`RUN` constant). Meeting asks #4/#5/#6 still not started.

### 2026-07-29 — 2026-07-29 meeting: flat-evar + linear + neural-side PCA built and smoke-tested (launch queued for Theo)
Máté narrowed the programme to **the projection loss and its tradeoffs**. Transcribed today's note and the
previously-uncaptured 08/07 one into the vault, then built the run for asks #2 and #3.
- **`shape_lambda = ∞` is an existing knob** — `flat_evar=True` (uniform per-PC weights = unweighted Brier). It was
  last run **2026-06-03** (`brier_ctrl_flatevar`) at H=32, **pre** the restart-selection fix, scored on *entropy*
  (3.24→4.03 vs the calibrated 3.95). Never measured at H=8, under the fixed restart rule, on the current
  peakiness + chance-normalised-loss pair, and never with zero hidden units. So this is a re-measurement, not a repeat.
- **NEW `Config.n_neural_pcs`** (default `None` = no-op) — the meeting's "PCA on neural resp., decode from PCs".
  Projects the **input** activity onto its leading k PCs, **fit on training trials only** (same rule and same
  `train_indices` as the existing z-scoring, immediately above it), applied to train/val/test/full. Implemented on
  `activities_m_z` itself so all four X matrices and `input_size` follow automatically — the `neuron_subset`
  mechanism. Orthogonal to `pca_basis` (which is the *target* basis). Retained EVR recorded in provenance.
- **NEW `run_flatevar_v1.py`** — 36 cells / 4320 net trainings, ordered by decision value so any prefix is usable.
  Arms: **core** (evar vs flat × H=8 vs linear, + KL/JS references), **neural** (k ∈ {2,4,8,16,32,64} at H=8 and
  linear, + an evar-weighted control), **sweep** (flat-evar OAT over width/dropout/wd/λ_H/early-stop), **linear**
  (the same sweep with `hidden_sizes=[]`). `--arms` / `--only` / `--smoke` / `--dry-run`.
  **The decisive cell is `B_flat_linear`**: flat weighting with ZERO hidden units. `prodfix_v1` arm C showed the
  projection loss over-sharpens 5.6×/10.5× with no hidden layer, killing the capacity account — so if the evar
  weighting is the cause, removing it must fix it *even with no capacity*.
- **BUG FOUND AND FIXED (load-bearing): `savemat` cannot serialise `None`.** Adding an Optional Config field made
  **every run that left it at its default** crash at shard-save — after training completed. The partial write also
  left a truncated shard that the merge then failed to read, and resumability made the next run skip that mouse and
  fail again. Fixed with `training/run._matlab_safe_config` (None → `[]`). **Latent for `Config.seed` since
  2026-07-16**, unhit only because every run since set `seed=0`. Two GOTCHAS entries added (this, plus: a resumable
  runner's `--smoke` must write to a separate run root, or smoke shards get silently skipped-over by the real run —
  `run_flatevar_v1` writes to `<run>_smoke`).
- **Priors registered in `PREDICTIONS.md` before launch** (P1 flat-evar ≈ shape30; **P2 the decisive linear cell
  lands on target, ≥5/6 mice, ~80%** — falsified if it still over-sharpens ≥2×, which would break the loss-geometry
  account; P3 other knobs stop mattering; P4 neural-PC ladder saturates and never beats the full population, plus
  the evar-weighted input-PCA control should NOT fix over-sharpening).
- **Verification:** 161 tests pass (13 new in `tests/test_neural_pca.py` incl. a leakage pin that corrupts held-out
  trials and asserts training PC scores are unmoved, and a reshape round-trip; 2 new in `test_shard_merge.py`
  pinning the savemat fix). Both new code paths smoke-tested end to end on mouse 0 — neural PCA reported
  `65 neurons → 4 PCs, EVR 0.187`, and provenance round-trips (`n_neural_pcs` = `4` when set, `[]` when off).
- **Open / next:** **Theo launches `flatevar_v1` on gpu1** (agent ssh/rsync is blocked — launch block below). Then the
  analysis leg: the existing `diagnostics/prodfix_report.py` idiom extends to these cells, plus a new
  peakiness/normalised-loss-vs-k plotter for the neural-PC ladder. Meeting asks **#4 (temporal-bin heatmaps,
  spatial/temporal scatter, trials selected by across-bin difference)**, **#5 (noise variance vs stimulus features)**
  and **#6 (posteriors by contrast)** are **not started**.

```bash
cd ~/UncertaintyV1/nn_decoder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/cluster-env/.venv/bin/python
$PY run_flatevar_v1.py --dry-run
$PY run_flatevar_v1.py --smoke
$PY -u run_flatevar_v1.py --arms core 2>&1 | tee flatevar_core.log
$PY -u run_flatevar_v1.py 2>&1 | tee flatevar.log
```

### 2026-07-27 — Spatial vs temporal per manipulation: the SIGN FLIPS with the loss (6/6 mice each way)
New `diagnostics/spat_temp_manipulations.py` — head-to-head under all 10 manipulations, across animals (n=6, and
n=5 with M2 excluded) and within each animal. Values keyed by mouse id, so the pairing cannot silently misalign.
Figures `spat_temp_{across,within}_animals`.
- **The architecture answer is entirely loss-dependent.** Under the **projection-based** loss **spatial wins**
  (baseline Δ = −5.79, p=0.028, **6/6 mice**; no-hidden −3.39, p=0.005, 6/6). Under the **calibrated** losses
  **temporal wins** (KL Δ = +0.376, p=0.002, **6/6**; JS +0.276, p=0.005, 6/6). `shape λ=0.3` — the width fix —
  sits with the calibrated losses (Δ = +0.032, p=0.080 at n=6, **p=0.015 at n=5**).
- **This replicates and strengthens [[PCA-Peakiness-Mechanism]] §9** ("the PPC≫SBC gap is a calibration artefact"):
  the earlier evidence was Δ KL-skill +0.05–0.16 at p≈0.08; here it is 6/6 mice at p=0.002 in the cleaner regime
  (H=8, fixed restart rule, LOO null). **The apparent spatial advantage is an artefact of the over-sharpening, which
  the Jensen-averaged temporal arch is punished for hardest — and it REVERSES once the loss is calibrated.**
- **Mouse 2 changes nothing.** Every effect survives at n=5, several with *smaller* p. M2 is the best-performing
  animal (lowest normalised loss throughout) but moves in the same direction as the rest.
- **A p-value cautionary case:** `weight_decay=0.01` gives **Δ = 0.000 with p=0.011** — both arches have collapsed to
  the uniform decoder, so the difference is numerically tiny but sign-consistent. Effect size without a p is
  uninformative here, and so is a p without the effect size.
- **Dropout's "cure" is spatial-only:** at p=0.9 spatial peakiness lands on target (0.062) while temporal stays at
  **0.469, ~8× target** — another reason it is a lobotomy rather than a fix.
- **WITHIN-animal tests (n = that animal's trials, paired on per-trial KL — `per_trial_paired`).** Direction agrees
  with the across-animal result in every animal, but the *character* of the two effects is completely different:
  - **projection baseline: spatial better in 6/6 animals, and it is a UNIFORM shift** — dz −0.87 to −5.78, with
    **91–100% of individual trials** favouring spatial (p down to 1e-205).
  - **KL: temporal better in 6/6 animals, but the effect is SMALL and TAIL-DRIVEN** — dz only 0.20–0.37, and by
    trial-level sign it is near a coin flip (**45–58%** of trials favour spatial). Same for JS (dz 0.10–0.34).
    So the SBC edge is *consistent* across animals but *modest* within them — which matches the mechanism note's
    "modest, calibration-dependent temporal ≥ spatial edge" rather than overturning it.
  - `shape λ=0.3` is mixed/weak within animals, consistent with its small across-animal Δ.
  - **`weight_decay=0.01` is the degenerate case:** Δ ≈ 1e-5 yet **dz up to 2.42 and p = 1e-198**, because both
    arches have collapsed to the uniform decoder so the difference has almost no variance. **A standardised effect
    size can mislead exactly as badly as a p-value when the variance collapses** — read the raw Δ first.
- **Stats caveat:** the within-animal p-values (n = 326–470 trials) answer "is this reliable *within this animal*";
  they are NOT population evidence — averaging or pooling them across animals is pseudoreplication (GOTCHAS). The
  n=6 paired test is the generalisation claim. Across-animal: 20 tests, uncorrected; the 6/6 sign consistency is the
  robust statement (sign-test floor p=0.031).

### 2026-07-27 — prodfix_v1 landed: over-sharpening survives a LINEAR decoder (bias account confirmed); shape_lambda beats smooth_lambda; JS ≥ KL
14 cells, Q/half/100ms, H=8, 6 mice, restart-selection on val, seed 0, all pulled. Figures `figures/prodfix/`
(`diagnostics/prodfix_report.py`). Judged throughout on peakiness AND chance-normalised loss (LOO predict-mean).
- **ARM C — decisive, and it closes meeting item #6.** A decoder with **ZERO hidden units** (multinomial logistic
  regression) trained on the projection loss over-sharpens **5.6× (spatial) / 10.5× (temporal)** the IO target, in
  **6/6 mice** — and is statistically indistinguishable from the H=8 MLP (0.301 vs 0.325 spatial p=0.071; 0.569 vs
  0.643 temporal p=0.247). Removing the *entire* hidden layer changes peakiness by 7–11%. **The over-sharpening
  therefore cannot be capacity-driven overfitting — there is no hidden layer to overfit with.** KL and JS stay on
  target with zero hidden units (1.02–1.14× target, 0/6 mice >1.5×). This is the cleanest real-data confirmation of
  the loss-geometry (bias) account, exactly as [[PCA-Peakiness-Mechanism]] predicted from the softmax Jacobian.
- **ARM A — production fix: `shape_lambda` wins, `smooth_lambda` is a partial lobotomy.** Both land peakiness on
  target, but only shape also beats chance: shape30 (λ_Brier 0.3) peaky 0.057/0.058, normalised loss **0.78/0.75**;
  smooth0.3 — the previously recommended operating point — peaky 0.069/0.094 but normalised loss **1.11/1.44, i.e.
  WORSE than chance**; smooth1 only reaches 0.99/1.08. shape30 > smooth0.3 in **6/6 mice** (p=0.018 spat / 0.022 temp);
  shape30 > smooth1 (p=0.037/0.033). Exactly the weight_decay failure mode, caught because we now always plot both.
  **Caveat:** at shape_lambda=30, sum(evar)=28.3 so the Brier term is ~96% of the loss — the "fix" at that strength has
  essentially become unweighted Brier. The honest reading is *the more the projection weighting is diluted, the better
  it gets*, which argues against the projection loss rather than for a tuned hybrid.
- **KL vs JS — JS ≥ KL.** Spatial normalised loss 0.84 vs 0.94 (**6/6 mice**, p=0.001) and 2.6× less overfitting
  (val/train 10.0 vs 25.6); temporal a tie (0.57 vs 0.56, p=0.64, 3/6). Same calibration. Supports JS as production loss.
- **ARM B was a design error of mine — it does NOT test E1.** `pca_basis` only affects the PCA loss (documented in
  `training/config.py`), so `B_residual_kl` is a **bit-identical duplicate** of `A_reference_kl` (max|diff| = 0.0e+00 —
  which does at least verify `Config.seed` reproduces exactly). And the residual basis changes the loss *weighting*,
  not the target, so it cannot ask "is there trial-level signal beyond condition" for a calibrated loss.
  `B_residual_pca` is simply a worse projection decoder (peaky 0.344/0.741, normalised loss 2.77/9.90). **E1 still needs
  the DeepSets-style within-condition null, not a basis switch.**
- **Stats caveats:** n=6 paired t, ~8 tests, no multiple-comparison correction — the 6/6 sign consistency is the more
  robust statement (sign-test floor p=0.031).

### 2026-07-16 — Restart selection FIXED: restarts now chosen on held-out val, not training loss
The audit's sharpest finding, unfixed since 2026-05-16. `train_and_select_best_model` picked the restart with the
lowest TRAINING loss — i.e. systematically the most overfit one, and plausibly more so for richer objectives, which is
a direct confound for the "KL overfits most / projection-based least" ordering we explained mechanistically.
- **Fix:** selection now uses the **held-out validation fit-loss**. `Config.restart_selection` (`'val'` default,
  `'train'` reproduces history) → `to_legacy_dict` → `training_params`. Fit-loss only (`entropy_lambda=0`), matching
  the early-stopping signal, so the criterion isn't architecture-dependent (the entropy penalty is temporal-only).
  `fit_model` gained an opt-in `val_out` param exposing the val slice it actually used — return value unchanged, so
  all existing callers (incl. `optuna_per_target`) are untouched.
- **Blast radius is bounded:** with no val slice (`patience=0`, `monitor_val=False`, `val_frac=0`) there is nothing to
  select on, so those runs fall back to the old rule and are **bit-for-bit unchanged**. Runs *with* a val slice (both
  hpsweeps) will differ from their pre-fix counterparts — set `restart_selection='train'` to reproduce them.
- **Now measurable:** every run records `history['restart_scores']` = per-restart `(train, val)`. **The first smoke
  test already showed the rules disagreeing** (argmin train = rep0, argmin val = rep2).
- **CONTROL RESULT — the ordering SURVIVES; the confound is real but small.** Matched pair via the new `Config.seed`
  (both rules see *identical* restarts; only the winner differs), KL vs PCA × {val,train} × 6 mice, H=8, 200 ep:

  | val/train | rule=val | rule=train | rules disagreed |
  |---|---|---|---|
  | KL spatial | 25.56 | 25.56 | 0/6 mice |
  | KL temporal | 3.48 | 3.60 | 1/6 |
  | PCA spatial | 5.47 | 5.82 | **4/6** |
  | PCA temporal | 2.38 | 2.38 | **4/6** |

  **Ordering KL ≫ PCA holds under both rules** (25.6 vs 5.5 / 5.8 — a ~4.6× gap against a ≤6% selection effect), so the
  "KL overfits most / projection-based least" bias-vs-variance story is **not** an artefact of the selection rule.
  Direction is as predicted (val-selection gives slightly *lower* val/train, i.e. less-overfit restarts). Striking
  asymmetry: the rules disagree on **4/6 mice for PCA but 0–1/6 for KL** — consistent with PCA's restarts being closely
  bunched so the argmin flips easily, while KL's have a clear winner (plausible, not verified).
- **A bug the control caught in the fix itself:** `restart_selection` reached `to_legacy_dict` but was never threaded
  into `run_experiment`'s `training_params`, so it was **recorded in provenance while having no effect** — the first
  control silently compared 'val' against 'val' (identical metrics to 4 dp despite 4/6 disagreement gave it away).
  Fixed, and the control now reads `history['restart_selection']` back and refuses to report on a mismatch.
- **Tests:** 596 passed (4 new pinning val-selection, the `'train'` fallback, the no-val fallback, and `val_out`);
  same 3 pre-existing unrelated failures. GOTCHAS entry flipped from "left unfixed" to fixed.

### 2026-07-16 — Full audit of nn_decoder + vault (4 parallel auditors), then fixed the bugs and criticals
Ran four read-only audits (core training code / analysis+plotting layer / vault notes / statistical rigour) and
consolidated them into **[`documents/AUDIT_2026-07.md`](documents/AUDIT_2026-07.md)** — issues, consolidation
opportunities and ranked extension directions. Headline: **the training pipeline is leakage-clean where it counts;
the risk is concentrated in the analysis layer and the vault.**
- **Corrected the auditors three times** (they were wrong or overstated): (a) the `shape_lambda` missing
  renormalisation is real (`sum(evar)`=1+91λ, verified on disk as 28.3 at λ=0.3) but **does not invalidate the cure** —
  `(evar+λ)/(1+nλ) ∝ (evar+λ)`, so relative weighting is identical, and wd=0 (maximal dilution) leaves peakiness at
  0.232/0.692; (b) **`GOTCHAS` itself was stale** — it claimed `pca_basis` defaults to `condition_mean`, but all **401**
  configs use `all_trials`, so the "stim_mean tautology" applies to no run on disk (both the vault and stats auditors
  had aimed findings at it); (c) reconciled the two auditors on basis leakage — no *test* leakage, but the
  early-stopping epoch choice is measured in a basis fit partly on its own holdout.
- **Fixed my own bugs from the sweep sessions:** a **fabricated IO-target line** (`cure_comparison` drew a hardcoded
  0.059 when no data loaded); **zero missing-data guards** in all six hpsweep plotters (blank figure + a suptitle still
  asserting the conclusion — now all refuse, verified); **ddof=0** SEM (error bars 9.5% too small at n=6);
  `val_fraction` cells and the `_g` token drift; and **params/trial was 1.25× too low** because `monitor_val` removes
  20% of training trials (`nn_classifier.py:503`) — now **3.6–7.7** (was 2.9–6.1); log + memory corrected.
- **Fixed criticals:** leave-one-out predict-mean null (the old one was fit on the trials it scored —
  anti-conservative for "worse than chance"; **measured effect ~0.5%, flips nothing**); CE scorer eps aligned to the
  training-side float32 eps (they disagreed by up to 1.3 nats under a comment claiming they matched); the **100×**
  duplicate PCA loss in `time_binned_ppc` now delegates to `pca_loss.pca_distance` (also killing its silent MSE
  fallback); optional `Config.seed` (no-op default) since nothing seeded torch in production; import guard +
  `FORCE_RERUN=False` on `run_recovery_Q_spyder` (was moving a 116 MB cache and starting training on import).
- **Tests:** 592 passed; the 3 failures are pre-existing and unrelated (verified identical on a stashed clean tree):
  a `('d',100)` preset/test mismatch, a GV `reg` default drift, and `numpy.trapezoid` missing in this numpy.
- **Left for your call (each changes training or needs a run):** restart-selection-on-training-loss (**the sharpest
  finding — it biases toward the most overfit restart, more strongly for richer objectives, i.e. exactly the
  KL-overfits-most ordering we explained mechanistically; a `REP=1` control retires it**); the single-permutation
  shuffle null; training-side CE saturation at 15.94 nats; `shape_lambda` scale-matching; a 2-D-grid plotter
  (~half of `hpsweep_v2` still has no consumer).
- **Top extension directions** (in the audit doc): **E1** does V1 carry uncertainty *beyond condition* — the DeepSets
  null + condition-mean oracle say the trial-level signal is unestablished; **E2** PPC-vs-SBC is not diagnostic
  (chance floor >1 for Monte-Carlo reasons); **E6** the loss-geometry over-sharpening dissociation is the cleanest
  publishable claim today.

### 2026-07-16 — interrogation-DDM σ_v was NEVER identified: Prediction 18 retired, fit reparameterised
Found while building a Similarity-Framework teaching curriculum in the vault: `similarity_analysis._fit_interrogation_ddm`
fitted `{α, bias, σ_v, λ_L, λ_R}`, but the likelihood only ever sees `z = (α·c + bias)·√T / √(1 + T·σ_v²)` —
two compound quantities from three parameters. **σ_v is a flat ridge**: rescaling α and bias by `√(1+Tσ_v²)`
leaves negLL identical to **10 d.p.** across σ_v ∈ [1e-3, 1e2]. So the reported σ_v was the Nelder–Mead start
point, not the animal — on the real cohort it ran to **1.0e14 (Cb22)** and to **exactly 0 (Cb15/Cb17/Cb25)**.
`kappa_vs_ddm_sigma_v.png` — the conjecture's Prediction 18, billed as "the *proper* test" of bundle-width → `s_v`
— was therefore correlating κ against an optimiser artefact, and would have been driven entirely by the 1e14 point.

- **Fix:** fit the *identified* probit parameterisation `{slope, bias, λ_L, λ_R}` directly; report no σ_v or α.
  Same model, redundant parameter removed — and **strictly better**: held-out ΔLL ≥ 0 in 6/6 mice, +7.9 (Cb17),
  +5.8 (Cb21), **+30.6 (Cb22)** nats, because the 5-D simplex had been wandering the ridge. `bias` is now
  **probit-scale, not drift-scale** — not comparable to pre-2026-07-16 numbers.
- **Replacement figure:** `kappa_vs_ddm_slope.png` (+ per-session twin). Slope `A = α√T/√(1+Tσ_v²)`, so drift
  variability *attenuates* it ⇒ framework predicts κ↑ → slope↑. Ran it: **r = 0.27, p = 0.60, n = 6** — an honest
  null, and confounded with α anyway. Prediction 18 stays **untestable** on this dataset; it needs ≥2 interrogation
  times T (or RTs), which the fixed-2 s export does not have. This is a data-collection ask, not an analysis one.
- Files: `nn_decoder/similarity_analysis.py` (fit + 2 renderers + `__all__` + orchestrator job),
  `nn_decoder/session_similarity_analysis.py` (same, per-session), `tests/test_archetype_similarity.py`
  (2 tests updated to the new contract, +`test_interrogation_ddm_sigma_v_is_not_identified` pinning the flat ridge
  so σ_v cannot be silently re-added). 35 passed; 72 passed/1 skipped across the similarity suite. Both figures
  rendered on real data and eyeballed. Also fixed a `1/(1+exp(-x))` overflow at lapse→0 (now `scipy.special.expit`).
- **GOTCHAS** gained the identifiability trap **and** a correction: the "`xG` is position (not time)" line was
  **wrong** — the export README is explicit that `xG` is *time* (40×50 ms bins over the 2 s grating). This matters
  because every "within-trial" `Var_t[SI]` (the SBC wedge, RD-2 M2) is a *temporal* variance; a subagent reading
  only the code got it backwards this session.
- **Open / next:** the vault conjecture page `Conjectures/Similarity Framework.md` is the user's to write and was
  **left untouched** — its Prediction 18 row, the Prediction-4 naming-clarification paragraph, and the
  "Lowest-hanging next step" para all still endorse the retired σ_v test and need revising. Separately, two more
  live problems surfaced but are **not fixed**: (1) the Cb17 headline "p ≤ 0.003" **is** the add-one permutation
  floor 1/301, not a measurement, and its z = 10.5 extrapolates far past a 300-draw null's support; (2)
  `_compute_archetypes` builds archetypes in-sample (no train/test split, K can be 5) and `_zscore_neurons` is fit
  outside the CV loop. New vault curriculum: `ResearchVault/Lessons/Similarity Framework/2026-07-16_Similarity Framework — Lesson Plan.md`.

### 2026-07-12 — hpsweep_v2 landed (143 cells): shape_lambda CURES PCA (beats chance), weight_decay LOBOTOMISES it; +performance leg
Pulled all 143 v2 cells (20 GB), re-pointed the plotters onto a shared `diagnostics/hpsweep_spec.py`
(`--sweep {v1,v2}`, handles the PCA-only shape axis). Added the **third analysis leg (actual performance)**:
`performance_vs_hparams.py` = held-out **KL(decoded‖target) ÷ predict-mean** (calibrated scorer, <1 beats chance).
Findings (Q/half/100ms, 6 mice, H=8 base):
- **Re-basing worked mechanically** — params/trial now **3.6–7.7** (was ~14–30); baseline overfitting val/train
  *[params/trial corrected 2026-07-16: the original figures omitted the 20% `monitor_val` carve, understating by 1.25×;
  correlations unaffected]*
  **2–6** (was 5–60); capacity trend holds (`overfit_vs_capacity_v2` ρ 0.67 spat / 0.69 temp, p<1e-17). But it
  **did NOT fix peakiness** — PCA still over-sharpens at baseline (0.34/0.66) and is **worse than chance**
  (skill 2.5/8.6). Re-basing fixed variance, not bias — the decoupling holds at the new base.
- **shape_lambda = the real cure.** Peakiness → IO target (λ=0.1: 0.063/0.070; λ=0.3: 0.057/0.058) AND skill
  crosses **below chance** (0.78/0.75 at λ=0.3). A genuinely good decoder.
- **weight_decay = a FAKE cure (why checking performance mattered).** It drives peakiness DOWN too, but PAST the
  target to **0.011 = 1/91 = uniform**, and skill **plateaus at 1.59 (the uniform decoder) — never beats chance**.
  It trades over-sharpening for under-fitting: a dead decoder. Peakiness alone would have called it a cure.
- **Good decoders = KL/JS** (beat chance across all knobs); PCA worse-than-chance unless shape_lambda'd; width↑ and
  λ_H↑ make PCA worse; early-stop/dropout help but don't cross chance. Headline figure: `cure_comparison.png`.
- **Files (committed):** `diagnostics/hpsweep_spec.py` (new shared spec), `performance_vs_hparams.py` +
  `cure_comparison.py` (new), `{peakiness,overfitting,overfit_vs_capacity}_vs_hparams.py` (re-pointed to `--sweep`).
- **Open / next:** methodological headline — **peakiness (bias) and overfitting (variance) don't say if the decoder
  is GOOD; need chance-normalised skill under a calibrated metric** (weight_decay is the cautionary case). shape_lambda
  is the production over-sharpening fix. `shuffle_trainval_curves`/`shuffle_gap_vs_reg` not yet re-pointed to v2 (secondary).

### 2026-07-12 — DeepSets tests unordered temporal variability: synthetic positive, real trial-specific uncertainty null
- Built `deepsets_uncertainty.py` + resumable runner: parameter-matched mean, moments and DeepSets models; KL/JS/Brier/projection losses; nested train/val/test preprocessing; within-condition null; common metrics and mouse-level inference.
- Synthetic validation (10 datasets) behaved as designed: moments/DeepSets recover exact-mean-matched variance codes, all invariant models fail order-only width while an order oracle succeeds; projection loss again fails posterior shape.
- Full real Q/half/100ms grid (6 mice x 3 models x 4 losses x real/null): DeepSets raw KL < moments in 6/6, but real/null KL = 1.004 and width/entropy contrasts are null; no architecture extracts detectable trial-specific Q beyond condition.
- Condition-mean oracle KL ~0.014 versus neural 0.20-0.67, making stimulus identity versus trial residual explicit. Report: `documents/DEEPSETS_UNCERTAINTY_METHOD.md`; outputs/figures are gitignored.
- Verification: 96 focused tests pass; nine figures visually checked as PNG with paired SVG and longest side <=1600 px. Open: only pursue order-sensitive TCN/GRU and choice-readout tests if the question expands beyond unordered variability.

### 2026-07-08 — Overfitting deep-dive (it's capacity, not loss) + re-based sweep v2 (H=8, +weight_decay +shape_lambda)
Followed the peakiness thread into the **fit-loss overfitting** and found it's a **different object** from the
over-sharpening (corroborates [[roundtrip-loss-refit]] + the entry below), and traced its cause. Two new diagnostics
(committed, `figures/hpsweep_shuffle/`):
- **`overfitting_vs_hparams.py`** — val/train fit-loss ratio (log y) vs each hparam, one line per loss, same layout as
  `peakiness_vs_hparams`. **KL overfits the *most* (val/train ≈60 spatial), PCA the *least* (~5)** — the inverse of the
  peakiness ranking. dropout / smaller width / early-stop all shrink it (unlike peakiness, which only early-stop caps).
  Framing: **peakiness = bias (PCA over-sharpens systematically, invisible to its blind loss); fit-loss gap = variance
  (KL's rich full-distribution objective memorises)**. Anti-correlated across losses → bias–variance, not a contradiction.
- **`overfit_vs_capacity.py`** — val/train ratio vs **params-per-training-trial**, pooled over 6 mice × 5 widths × 4 losses.
  **Spearman ρ = 0.52 (spatial) / 0.59 (temporal), p<1e-9**, one rising trend across all losses/mice/widths → overfitting is
  set by **capacity-vs-data, not the loss**. Cause: **~5–8k params vs ~270–380 *fitted* training trials
  (~14–30× overparameterised at H=32)** *[corrected 2026-07-16: originally quoted 11–24× against ~350 trials, which
  omitted the 20% `monitor_val` validation carve — `nn_classifier.py:503` removes it from the training tensors]*.
  Stats caveat: pseudoreplication (6 mice reused) inflates the p; the within-mouse width trend is the genuine driver.
- **Re-basing (data-driven):** best-epoch val vs width shows **H=4–8 generalises ≥ H=32 for PCA/KL/JS** (KL strictly best at
  H=4) with far less overfitting — H=32 was pure overparameterisation. **`run_hyperparam_sweep_v2.py`** (`hpsweep_v2`):
  base **H=8**, width axis re-centred {2,4,8,16,32}, + **weight_decay** axis {0,1e-4,1e-3,1e-2,1e-1} (live in Adam,
  nn_classifier.py:805) + **shape_lambda** axis {0,1,3,10,30} (PCA-only loss-side cure, λ=shape/100). **143 cells, ~21–36 GB.**
  Dry-run + smoke (shape_lambda=10 PCA cell) green.
- **Open / next:** **Theo launches v2 on gpu1** (rsync + `$PY -u run_hyperparam_sweep_v2.py`, `--dry-run`/`--smoke` first;
  idempotent). Analysis scripts (`peakiness_vs_hparams` etc.) hardcode the v1 baseline (H=32, parent `hpsweep_wide`) — re-point
  to H=8 / `hpsweep_v2` when it lands. v1 `hpsweep_wide` left at 86/123 (OAT done, 2-D grids pending; superseded).

### 2026-07-08 — round-trip refit DISSOCIATES overfitting from PCA over-sharpening (Tier-C control, run locally)
New diagnostic `nn_decoder/diagnostics/roundtrip_loss_refit.py` (6 mice, Q/half/100ms, full 200-ep trajectory,
`patience=0` + `monitor_val` + `val_frac=0.2`, snapshots every 10, **`entropy_lambda=0`**). Matrix = 3 target sources
{real IO, PCA-fitted (peaky), KL-fitted (broad, achievable)} × 2 refit losses {PCA, KL}, reusing the existing
`target_source='recovery_spat'` machinery (base `full_decoded` → new target). Readout = decoded max-prob vs epoch +
train/val fit curves; figs `figures/roundtrip_refit/` (gitignored), per-(cell,mouse) caches `results/roundtrip_refit/`.
**Scorecard (spat, final maxprob ÷ target):** realIO_PCA **5.9×**, realIO_KL 1.05×; pcaFit_PCA **1.03×**, pcaFit_KL 0.99×;
klFit_PCA **4.33×**, klFit_KL 0.98×. Temp mirrors (5.7/0.87/1.06/0.86/4.44/0.91).
**Two findings.** (1) The round-trip **fixes overfitting**: only the real-IO row shows the train↓/val↑ upturn
(starkest realIO_KL val 0.35→0.58); on *achievable* fitted targets val plateaus, no upturn — the gap is an
unachievable/noisy-target artefact. (2) It does **NOT** fix **over-sharpening**: `klFit_PCA` over-sharpens a clean broad
achievable target **4.33× with essentially no train-val gap** → over-sharpening is pure loss mis-specification
(evar-blind shape subspace), separable from overfitting. **Corrects** the earlier "peakiness tracks the train–test gap /
it's overfitting" framing in [[loss-comparison-v1-checkpoints]]. `pcaFit_PCA=1.03×` shows a **bounded attractor** (PCA
faithfully reproduces an already-sharp achievable target), not unbounded drift. Cure stays **loss-side** (all `*_KL`≈1.0×);
target-side achievability alone does nothing for sharpness. Prior mis-scored: predicted pcaFit_PCA would climb past target
(it sits on it). **Open:** fold into the loss-side headline; no Config change needed (recovery machinery sufficed).

### 2026-07-08 — hpsweep analysis: "how to stop overfitting" answered (loss-side, not regularisation) + 4 matched-axis diagnostics + evar-weighted subspace + 2026-06-18 meeting note
`hpsweep_wide` resumed on gpu1 after a partial local run; ~75/123 cells down (PCA/KL/JS complete, Wasserstein ~20, 2-D
grids pending). Built four matched-axis diagnostics on it (all `figures/hpsweep_shuffle/`, PNG+SVG):
- **`shuffle_trainval_curves.py`** — real vs shuffle train/val, **÷ predict-mean** (meeting #8) with the **chance line at 1.0**
  (#7). Real decoders beat the null; shuffle controls overfit above it (val 1.1–2.5×); KL-spatial real creeps *above* 1.0 at
  200 ep (no-early-stop overfitting). First-ever look at the shuffle nets' training dynamics.
- **`peakiness_vs_hparams.py`** — decoded peakiness (the overfitting that MATTERS; the fit-loss gap is blind to it) vs each
  swept knob, 4 losses. **Answer to "how to stop overfitting": generic knobs fail** — λ_H makes temporal *worse* (sharpening),
  dropout ~flat till p=0.5 then falls steeply *[corrected 2026-07-27: NOT "flat till 0.9" — spatial peakiness goes
  0.34/0.35/0.35/0.27/0.14/**0.062** across p=0/.1/.25/.5/.75/.9, i.e. it reaches the IO target at p=0.9. But its
  normalised loss is 1.14, so it never beats chance: dropout is a **lobotomy**, not an ineffective knob]*,
  width caps only, activation nil; **only early-stopping bites (temporal 0.72→0.27) but stays ~4.5×
  target. KL/JS sit on target unconditionally.** Cure is loss-side (calibrated loss, or `λ·Brier`/`smooth_lambda≈0.3`) —
  confirms [[PCA-Peakiness-Mechanism]] at scale.
- **`shuffle_gap_vs_reg.py`** — train–val fit-loss gap vs hparams (real gap flat = static offset; only the shuffle gap moves).
- **`subspace_error_realdata.py --weight evar`** (meeting #2) — per-PC **loss contribution** (evar×error): PCA's shape error,
  38× KL's raw, collapses to ~2× / ~1% of the loss → the loss is **blind** to it. Non-destructive (default Fig 9 unchanged).
Filed the **2026-06-18 meeting note** in the vault (`Uncertainty Meetings/2026-06-18-Uncertainty-Meeting.md`). Meeting
analysis items #2/#4/#7/#8 done; the predict-mean null is fit+evaluated on the same held-out targets (mild in-sample
optimism — the established `predict_mean_baseline.py` convention, conservative direction).
- **Files (committed):** `diagnostics/{shuffle_trainval_curves,shuffle_gap_vs_reg,peakiness_vs_hparams}.py` (new),
  `diagnostics/subspace_error_realdata.py` (+`--weight evar`). Figures gitignored.
- **Open / next (all need RUNS):** #6 no-hidden-layer decoder (needs `nn_classifier` to allow 0 hidden layers) → peakiness;
  #3 multiplicative Gaussian dropout; #2 loss-side (per-PC eigenvalue-normalised *training* loss). Plus finish the 2-D grids +
  Wasserstein tail then regenerate; Mouse-2 and peakiness-by-orientation still open from 2026-06-10.

### 2026-06-18 — Shuffle decoders now save train/val curves; built the wide 6-axis hyperparam sweep (123 cells, queued for gpu1)
Q: "have we ever made train-val curves for the shuffle decoders?" → **no, and the data was discarded** — `run_experiment`
trained the shuffle nets (`*_shf`) but dropped their per-epoch history at the call site (`_`), and every train-val plotter
loads real archs only. **Fix (committed):** capture `history_*_shf` and attach a history-only sidecar
`Checkpoints['spat_shf'/'temp_shf']['history']` (per-epoch train+val curves + weight snapshots; final shuffle weights were
already in `Weights['*_shf']`). No-op when tracking off → production unchanged; the `.pt` saver and `recovery_sanity_check`
both tolerate the new keys (latter filters to `REAL_ARCHES`). Then built **`run_hyperparam_sweep.py`** — a "Wide" 6-axis
sweep: loss×{λ_H, dropout, activation, width, early-stop} one-at-a-time **under every loss** + width×dropout & patience×dropout
2-D grids, full export (history + val curves + snapshots) for **spat, temp, per-bin AND shuffle**. Fixed scope Q/half/100ms,
6 mice, REP 5; baseline = patience-0 + `monitor_val` (full 200-ep trajectories, no ES truncation). Dedup → **123 unique cells**
(~22–37 GB, ~1.5–2.5 days). Added `gelu`+`elu` to the `nn_classifier` activation registry (was relu/tanh/sigmoid with a
**silent-ReLU fallback** → orchestrator validates names). Dry-run + 1-mouse/3-epoch smoke verified: all 4 archs (incl `*_shf`)
save `train_fit`+`val_fit`+snapshots.
- **Files:** `run_experiment.py` (shuffle history capture), `nn_classifier.py` (+gelu/elu), `run_hyperparam_sweep.py` (new).
  Focused tests green (113 passed, 1 skipped) with `OMP_NUM_THREADS=1`.
- **Open / next:** **Theo launches on gpu1** (agent ssh/rsync blocked) — rsync code up, `$PY -u run_hyperparam_sweep.py`
  (`--dry-run`/`--smoke` first). Idempotent (per-mouse shards resume). After rsync-down, train-val plotters can finally read
  the shuffle curves at `Checkpoints['*_shf']['history']`; per-cell→hyperparams in `results/hpsweep_wide/MANIFEST.csv`.

### 2026-06-17 — More meeting feedback: "skill"→normalised loss, both-M2, real DECODERS (no free-fit), bins-by-condition
Round of live figure feedback. **(1) "Stop calling it skill" →** renamed the shuffle-normalised metric to **"normalised loss"** in all
displayed figure text + report + deck (`cross_loss_eval`, `spat_temp_per_animal`; code keys/`is_skill`/CSV dict-key stay).
**(2) Both cohorts:** rewrote `spat_temp_cross_loss_m2.py` to show the spat−temp train×eval diff **with AND without Mouse 2**
(2×2: normalised|raw × all-6|M2-excl). Robust: projection-based under KL Δ=−0.83** (all 6) / −0.91* (M2-excl). **(3) Dropped
the free-fit-from-a-spike** (Theo: "I don't know what the fuck this is") — `location_sharpness_grid.py` now uses **real data
only**: sharpen/broaden by raising to a power (`P^(1/T)`) + shift on **all** real posteriors (the "why 90? all!" fix), and an
examples gallery of the **real trained decoders' decoded posteriors** vs the IO target. On the trained decoders **both
projection-based AND Wasserstein over-sharpen** (jagged spikes; matches per-PC §1.4) — cleaner than the free-fit (where only
projection-based collapsed). **(4) New `temporal_bin_by_condition.py`:** does the temporal-bin similarity depend on the
stimulus? Yes — location dispersion rises with stimulus dispersion (CE/KL/JS 14→17°), width dispersion is U-shaped in
orientation (largest at 0/90° refs). **(5) Twin-axis fix:** in `temporal_bin_examples`, only the IO target on the LEFT axis;
the time-average AND the 10 bins on the RIGHT axis (both decoded → same scale).
- **Files:** `cross_loss_eval.py`, `diagnostics/{location_sharpness_grid,spat_temp_cross_loss_m2,spat_temp_per_animal,
  temporal_bin_similarity,temporal_bin_by_condition}.py`. Report §1/§2/§4 + TL;DR + Reproduce all reconsolidated; deck rebuilt
  (recovery slide dropped, condition slide added). Tests 113 pass. Figures/report/deck vault-side / gitignored.

### 2026-06-17 — Item-1 redone on REAL posteriors + no takeaway boxes on figures (Theo feedback)
Two pieces of pushback. **(1) No takeaway boxes:** Theo doesn't want prose conclusion/annotation boxes overlaid on
the plotted data ("Stop putting takeaway boxes on top of the figures!!"). Removed them — the location_sharpness
figures are now legend-only, the conclusion lives in the (concise) title/caption, and the **deck's takeaways moved to
speaker notes** (each content slide = accent bar + title + one big figure, nothing over the data). Saved as feedback
memory [[no-takeaway-boxes-on-figures]]. **(2) Sharpen/broaden + location on REAL posteriors:** rewrote
`diagnostics/location_sharpness_grid.py` from synthetic Gaussian bumps to the **real IO perceptual posteriors**
(`Dist['spat']['target']`, 90 sampled across 6 mice) using each mouse's **real rank-6 PCA basis** (only ~6 of 91 PCs
carry variance → the projection-based loss genuinely sees ~6 dims, ~85 free). Four box-free figures: sweeps
(temperature sharpen/broaden + shift), joint landscape, recovery, and an examples gallery.
- **Result (sharper than the synthetic version).** Sweeps: KL/JS/CE penalise sharpening > broadening; projection-based
  & Wasserstein ≈ symmetric. **Recovery/examples: only the projection-based loss collapses to a spike** —
  over-sharpens broad/bimodal real posteriors **~5.2×** (peakiness), while KL/CE/JS **and Wasserstein** recover the
  true shape (1.00×). Wasserstein's free-fit optimum *is* the real posterior (CDF-matching), so its trained-decoder
  over-sharpening (per-PC §1.4) is a network effect, not loss-intrinsic — the projection-based loss is the one whose
  *geometry* is blind. **Stats-rigor catch:** first scored recovery by 2nd-moment width and got "all losses recover"
  (wrong) — the over-sharpening is a high-frequency SPIKE on a correct broad pedestal, invisible to variance; rescored
  by **peakiness (max-prob)**, which the spike does move. (Same blindness the loss has.)
- **Files:** `diagnostics/location_sharpness_grid.py` (full real-data rewrite); report §1 + TL;DR rewritten to real
  framing (+ examples figure); deck rebuilt (notes + bigger figures). Figures/report/deck vault-side / gitignored.

### 2026-06-17 — 2026-06-18 deliverable polish: report reflow + loss relabel (PCA → "Projection-based")
Two follow-ups on the meeting deliverable. **(1) Report reflow:** the report had hard-wrapped prose (mid-sentence
newlines that render as visible breaks under Obsidian "Strict line breaks"); reflowed every paragraph/bullet/blockquote
to single lines to match the vault convention (329→168 lines, content byte-identical bar whitespace; fixed one
wrap artifact `run_experiment. fit_pca_basis`). **(2) Loss relabel:** renamed the **PCA loss → "Projection-based"** in
all displayed figure text (Theo's call; keeps CE/KL/JS/Wasserstein acronyms). Central display map in
`peakiness_style.py` (`LOSS_LABEL` + `loss_label()`/`loss_labels()`); the code/.mat keys stay `'PCA'` so no data
changes — same pattern as the PPC/SBC→spatial/temporal relabel. Wrapped every legend/tick site across 10 figure
scripts + `cross_loss_eval.py` (now imports `ps`); regenerated all 14 figures, re-synced to
`attachments/2026-06-18-meeting/`. Report prose + the lean deck (`_make_deck.py`) relabelled to match (protected
`PCA basis`, `[[PCA-Peakiness-Mechanism]]`, `--losses PCA`, `per-PC`). Focused tests green (113 passed); all figure
types (matrix ticks, legends, annotations, bars) verified to render "Projection-based" without crowding.
- **Files:** `peakiness_style.py` (+map), `cross_loss_eval.py` (+import, tick labels), 9 `diagnostics/*` plotters
  (legend/tick relabel). Figures/report/deck are gitignored / vault-side.

### 2026-06-17 — 2026-06-18 meeting prep: six-ask figure set + vault report + lean PPT
Produced the figures, a new vault report, and a figure-first deck for the six asks ahead of the 2026-06-18 meeting.
Vault report: ResearchVault `Projects/Uncertainty/2026-06-18 Meeting — Loss Geometry, Spatial-Temporal, Dropout &
Temporal Sampling.md` (14 figs, methods, paired-t stats, caveats; cross-links the prior `2026-06 Loss…` report +
`[[PCA-Peakiness-Mechanism]]`). Deck: `nn_decoder/figures/meeting_2026_06_18_deck.pptx` (13 slides, gitignored;
built by the throwaway `figures/_make_deck.py`, also gitignored). Figures synced to
`attachments/2026-06-18-meeting/mtg0618_*`.
- **New code (committed):** `diagnostics/location_sharpness_grid.py` (item 1 — synthetic location×sharpness probe,
  all 5 losses: independent sweeps + joint 2-D landscape + direct-fit recovery; reuses `loss_smoothness_demo`);
  `diagnostics/spat_temp_cross_loss_m2.py` (item 2 — spat−temp diff over the train×eval matrix, M2-excluded, skill+raw,
  paired-t stars); `diagnostics/temporal_bin_similarity.py` (item 4 — between-bin location/width dispersion vs λ_H +
  TWIN-axis unclipped per-bin gallery). `cross_loss_eval.py` gained a non-breaking **`--exclude <mouse>`** (drops an
  animal from `build_matrix` / paired stats / diff matrix) + `_stars()` paired-t annotations on the diff matrix.
  Items 3/5/6 were refresh-only (ran existing `dropout_*`, `uncertainty_scaling_realdata`, `predict_mean_baseline`;
  numbers reproduce the prior report exactly).
- **Key results.** (1) Loss geometry: location recovered by all losses; width is where they split — KL/JS/CE punish
  too-sharp (KL 22×, CE≡KL gradient), PCA/Wass ~symmetric (1.5×/1.1×) and a free fit **collapses to spikes even at
  50k steps**; real-V1 per-PC error PCA shape-subspace **38× KL's** (`subspace_error_realdata`). (2) PCA spat≫temp
  **only under a calibrated metric** (KL Δskill −0.91*, JS −0.65**, M2-excl n=5); own-metric a wash (+0.06). (3)
  early-stop fixes PCA peakiness, dropout doesn't; train–val gap a static offset. (4) bins are broad copies (mean
  per-bin width ≈ target 19°, location spread ~15°<target, ~flat in λ_H; only PCA sharpens bins 20°→15°). (5)/(6)
  reproduce prior report. **Gotcha (minor):** in `optimise_single` direct fits, JS needs ~8k steps to reach a broad
  target (weak bounded gradient); PCA/Wass **never** do (no width gradient) — so use ≥8k steps + a uniform init for
  the location control.
- **Tests:** focused suite green (113 passed, 1 skipped) with `OMP_NUM_THREADS=1`; new diagnostics are plotting-only.
- **Open / next:** the deck has no LibreOffice render here (verified structurally — 13 slides, all images fit); if a
  branded template is wanted, restyle `_make_deck.py`. Mouse-2 "what's different about M2?" still deferred.

### 2026-06-17 — Spat/temp head-to-head per animal (+M2 leave-out, n_neurons); train–val gap with dropout (`monitor_val`)
Worked the three remaining non-Mouse-2 meeting asks; all local, no cluster.
**Asks 1+2 — `diagnostics/spat_temp_per_animal.py` (committed 1c26a3f earlier):** paired (per-animal) spat-vs-temp KL-skill,
all 5 losses, all-mice vs Mouse-2-excluded, + neuron counts. **PCA is the only loss where spatial decisively beats
temporal** (1.34 vs 2.17, p=0.01); calibrated losses (CE/KL/JS) a wash/lean temporal (Δ +0.05–0.16, p≈0.07–0.09);
Wasserstein temporal-better n.s. **Robust to dropping Mouse 2** (PCA Δ −0.83→−0.91, same p). **Neuron count (65–153)
doesn't significantly predict skill** (spatial r=−0.67 p=0.14; temporal r=−0.34 p=0.51; n=6 underpowered; M2=74, mid-low).
**Ask 3 — train–val gap with dropout.** Built the **`monitor_val`** knob (commit 7ee15b4, 113 tests pass): carve+log a val
curve when `(patience>0 OR monitor_val)`, so patience-0 runs get a val curve *without* early-stopping (breaks the
val⟺early-stop coupling). Trained PCA locally, dropout∈{0,0.25,0.5}, patience 0, 200 ep, monitor_val; new
`diagnostics/dropout_trainval_curves.py`. **The gap is a *static offset*, not progressive overfitting** — val-PCA-loss
plateaus by ~ep 20 and never climbs (val 3–5× train spatial, ~2× temporal). **Dropout barely closes it** (spatial
0.0050→0.0044 ~12%, via val improving; temporal flat). Punchline: this val-loss curve is *not where the over-sharpening
lives* (peakiness is loss-blind), so early-stop fixes peakiness despite a flat val-loss while dropout fixes neither —
**train–val gap and over-sharpening are different objects**. Prior ↔ (over-predicted the gap shrink).
- **Files:** `diagnostics/dropout_trainval_curves.py` (new); `nn_classifier.py`/`config.py`/`run_experiment.py`/
  `run_loss_comparison.py` (`monitor_val`, 7ee15b4); `PREDICTIONS.md` (↔ resolution); vault report §7+§8 + figures
  `mtg0610_spat_temp_per_animal`, `mtg0610_dropout_trainval` + TL;DR/open-questions.
- **Open / next:** smoothness vs `shape_lambda`-Brier head-to-head (production PCA fix); the "both" cell (early-stop +
  dropout) if a combined regulariser is wanted; Mouse-2 "what's different about M2?" (exploratory) deferred.

### 2026-06-16 — λ_smooth sweep: PCA fixed at λ≈0.3 (lands on target); Wasserstein doesn't respond; no U-shape in range
Theo ran the smoothness sweep on gpu1 (`smoothsweep_smooth{0.01..1}`, PCA+Wasserstein, early-stop regime) + rsync'd
down. New `diagnostics/smooth_lambda_sweep.py` (peakiness + raw KL vs λ_smooth, vs the smooth=0 baseline
`loss_comparison_v1`). **PCA:** KL falls monotonically (temporal 1.75→0.48, spatial 1.10→0.49); peakiness lands ON
the IO target at **λ_smooth ≈ 0.3** (temporal 0.075, spatial 0.062; target 0.059) — the operating point. **No
U-shape in [0,1]** — KL min at the λ=1 boundary (mild over-broadening there is still lowest-KL), so the
over-smoothing→KL-rise onset is beyond λ=1 (prediction ↔: right direction, onset over-predicted). **Wasserstein
barely responds** at λ≤1 (KL ~flat ≈1.3) — NOT for lack of spikes (the gallery shows it's spiky) but a loss-scale
mismatch: its fit-loss is ~2800× PCA's (≈13.6 vs ≈0.005), so the same λ_smooth is negligible against it (a
loss-general smoothness term should scale λ_smooth by the fit-loss magnitude). **Production recommendation: λ_smooth ≈ 0.3 for the PCA loss.**
- **Files:** `diagnostics/smooth_lambda_sweep.py` (5216d70); vault report §6 + figure `mtg0610_smooth_sweep`;
  `PREDICTIONS.md` (↔ resolution).
- **Open / next:** smoothness vs `shape_lambda`-Brier head-to-head as the production PCA fix; extend the sweep past
  λ=1 only if the KL-upturn bracket is wanted. Mouse-2 deferred.

### 2026-06-16 — Anti-overfitting methods 1 & 2: averaging fails, output-smoothness penalty works (best fix yet)
Explored two more regularisers (after dropout/early-stop), motivated by the loss-geometry account.
**(1) Weight/output averaging** (`diagnostics/weight_averaging_test.py`, from saved snapshots, no retraining):
SWA + output-avg on `noreg` (200 ep, 21 snapshots) — **no effect on PCA** (peakiness 0.347→0.347 spat / 0.715→0.716
temp; final/SWA/output-avg posteriors superimposed). The over-sharpening is a monotonic, *stable* drift (spikes at
the same bins each epoch), immune to averaging; minor generic benefit for milder losses. Prior ✓ (commit 25ea274).
**(2) Output-smoothness penalty** — NEW `Config.smooth_lambda` → `_batched_total_loss` (λ·Σ(Δp)² Dirichlet energy of
the decoded posterior; training-only, both archs, no-op default; runner `--smooth-lambda`; commit e911705, 113 tests
pass). **It works — the best fix tried.** Local PCA λ_smooth=0.1 vs `noreg`: peakiness 0.347→**0.087** (spat) /
0.715→**0.133** (temp), KL 1.30→**0.65** / 4.61→**1.04** (IO target 0.059) — **better calibration than early
stopping** (temporal KL 1.04 vs 1.75). It's the smoothness-domain sibling of `shape_lambda`'s Brier term —
constrains the loss-blind high-frequency subspace. Prior ✓ (demo + scoring 86167bc).
**The loss-geometry model is now 5-for-5:** λ_H ✗ (sharpens); dropout / averaging / smaller-H — no help; smoothness
/ Brier + early-stop — work. Only terms constraining the loss-blind subspace fix the over-sharpening.
- **Files:** `diagnostics/weight_averaging_test.py`, `smoothness_penalty_demo.py`; smooth_lambda impl across
  `nn_classifier`/`config`/`run_experiment`/`run_loss_comparison`; vault report §6 + 2 figures; `PREDICTIONS.md` (2×✓).
- **Open / next:** λ_smooth **sweep** on the cluster (operating point + over-smoothing onset; only λ=0.1 tested) →
  then smoothness vs Brier as the production PCA fix. Mouse-2 deferred.

### 2026-06-16 — Dropout vs early stopping: early-stop tames PCA over-sharpening, dropout doesn't (prior ✓)
Completed the 2026-06-10 dropout task. The `dropout` knob landed 2026-06-15 (Config.dropout → MLP, runner
`--dropout`; no-op default; 113 tests pass — commit 5b54bef). Theo ran `noreg` / `dropreg_drop{0.1,0.25,0.5}` on
gpu1 (patience 0, 200 ep) + the existing `loss_comparison_v1` as the early-stop baseline, rsync'd down. New
`diagnostics/dropout_vs_earlystop.py` compares peakiness + **raw** KL(decoded‖target) across conditions.
**Stats-rigor catch:** shuffle-normalised KL-*skill* is confounded here (the shuffle is retrained per regime, so
its scale shifts → non-comparable), so used raw KL + peakiness (regime-independent). **Result (prior ✓):** early
stopping ~halves PCA peakiness (temporal 0.72→0.36) and raw KL (4.6→1.75); **dropout leaves PCA unchanged** (~0.72
at every p) and slightly worsens spatial-PCA KL (1.30→1.67); CE/KL/JS already calibrated. Confirms the registered
prior — PCA's over-sharpening is a loss-geometry drift (caught by stopping early), not capacity/overfitting
(untouched by dropout); the *same* model the λ_H sweep ✗'d. **Train–val gap not measurable:** the codebase couples
the val split to early stopping (`patience>0` ⇒ no val curve at patience 0) — needs a `monitor_val` knob to decouple.
- **Files:** `diagnostics/dropout_vs_earlystop.py` (new); `PREDICTIONS.md` (✓ entry); vault report §5 + figure synced.
- **Open / next:** (a) optional `monitor_val` knob → re-run for the train–val gap + the "both" (early-stop+dropout)
  cell; (b) Mouse-2 follow-ups still deferred.

### 2026-06-14 — λ_H sweep, CORRECTED framing: peaky bins vs broad average (the real question)
Theo flagged that the entry below misframed the sweep: penalising entropy *obviously* sharpens — that was never the
question. The real point is the SBC decomposition — can the temporal decoder produce **peaky per-bin (instantaneous)
samples** while its **time-averaged** posterior stays **broad/calibrated** (≈ IO target), even under CE/KL? Each
`.mat` saves the per-bin distributions (`Dist['temp']['decoded_samp']` (n,91,10); verified mean-over-bins ==
`decoded`). New `diagnostics/lambda_h_perbin_vs_avg.py` measures per-bin H vs time-avg H vs IO-target H (≈3.71 nats)
across λ_H. **Answer:** **CE/KL are inert** — per-bin H stays ≈3.65 (≈ target) at every λ_H, sampling spread
(H_avg−H_bin) flat ~0.26 → the 10 bins are broad copies of the average, no sampling. **JS is the only
info-theoretic loss that tolerates λ_H** — per-bin 3.71→3.28 and spread 0.23→0.37 at λ_H=0.1 while the average stays
≈ target (bounded/symmetric → doesn't fight the penalty to a standstill). **PCA/Wasserstein** get peaky bins
(per-bin H 1.4/2.8) but an **uncalibrated** average (1.9/3.2 vs target 3.7). So λ_H alone can't buy "peaky samples +
broad calibrated average" under CE/KL; you'd need a mechanism that pins the average while diversifying per-bin peak
*locations*. `PREDICTIONS.md` corrected (the registered prior scored the wrong quantity).
- **Files:** `diagnostics/lambda_h_perbin_vs_avg.py` (new). The prior `lambda_h_temporal_sweep.py` (time-avg
  skill/peakiness) is still valid for the *calibration* view but does not address the per-bin question.
- **Open / next:** (a) optional — does JS's growing per-bin sharpness sit at *varied* orientations across bins (true
  sampling) or the same one? (per-bin argmax spread). (b) **#4 dropout** still owed. (c) Mouse-2 deferred.

### 2026-06-14 — λ_H sweep landed: it SHARPENS (prior ✗) — temporal calibration worsens with λ_H
`lambdaH_sweep` (6 mice, Q/half/100ms, λ_H∈{0,1e-3,3e-3,1e-2,3e-2,0.1}) rsync'd down; ran new
`diagnostics/lambda_h_temporal_sweep.py` (temporal KL-skill + peakiness vs λ_H per loss + spatial control).
**Result violated my pre-launch prior (`PREDICTIONS.md`, ✗ direction):** the entropy penalty is `+λ·H(pred)`
(minimised → H↓ → *sharper*), so raising λ_H sharpens, not broadens. Temporal PCA KL-skill rose **monotonically
2.29→3.41** (worse) + peakiness 0.35→0.49; Wasserstein 1.37→1.72; JS over-confident only at λ_H=0.1; **CE/KL
immune** (~0.5, flat). **λ_H=0 best** for the peaky losses. Validated real: spatial control flat (CE/KL spread
0.02; PCA/Wass wobble, no trend), `entlam0p003`≈`loss_comparison_v1` (2.20 vs 2.17). Matches the standing
GOTCHAS "SBC sharpness-commitment" note (now sweep-confirmed). **Meeting takeaway:** the temporal decoder does
not want SBC's sharpness commitment — the data reject it.
- **Files:** `diagnostics/lambda_h_temporal_sweep.py` (new), `PREDICTIONS.md` (new), `GOTCHAS.md` (appended).
  Sweep results gitignored.
- **Open / next:** (a) optional example-posterior gallery (PCA-temporal sharpening across λ_H); (b) **#4 dropout**
  still owed; (c) Mouse-2 deferred.

### 2026-06-14 — Meeting 2026-06-10 follow-ups: orientation, shuffle nulls, peaky/broad, λ_H sweep runner
Worked the 2026-06-10 Máté/Nathalie/Ishan meeting tasks (captured + routed in ResearchVault
`2026-06-10-Uncertainty-Meeting`; Mouse-2 tasks deferred at Theo's request). Three **local** analyses on
`loss_comparison_v1` (6 mice, Q/half/100ms), each PNG+SVG under `figures/peakiness_scatter/`:
- **Peakiness vs orientation** (`diagnostics/uncertainty_scaling_realdata.py`, now 2×3: peakiness + over-conf
  ratio × disp/contrast/**orientation**). IO is U-shaped in orientation (peaky at 0/90° refs, broad/bimodal at the
  45° boundary); over-sharpening is orientation-structured, worst at the boundary. **Rigor caveat:** the ratio
  spike at 45° is partly the small IO denominator — raw decoded peakiness *does* dip there (PCA 0.30→0.11), so the
  honest claim is "structured over-sharpening", not "ignores the boundary". Disp panel reproduces the 3.5×→5.8×.
- **Shuffle control / three nulls** (NEW `diagnostics/predict_mean_baseline.py`): predict-mean, kill-weights
  (bias-only `softmax(model(0))`, exact since W_in=0 kills the input for both archs), shuffle-fit. Ordering
  **predict-mean (strictest) < kill-weights < shuffle-fit** — shuffle is a *looser* null (overfits scrambled labels
  + emits misplaced peaks). Under KL, trained **PCA 2.2×/3.4× and Wasserstein 2.7×/2.0× WORSE than predict-mean**
  (spat/temp) → below chance; CE/KL/JS beat all three. Corroborates `cross_loss_eval` skill with simpler nulls;
  **lead with predict-mean** (see memory [[shuffle-control-nulls]]).
- **Peaky/broad × spat/temp** (NEW `diagnostics/peakier_combinations.py`, quant + gallery): PCA/Wass over-sharpen
  broad targets 6–12×; **temporal PCA peakier than spatial** (0.23 vs 0.11 at broadest targets — the "temporal
  peakier" note), Wasserstein the reverse; CE/KL/JS track identity. Gallery: PCA = jagged hi-freq spikes both archs.
**#7 (next-meeting deliverable):** `run_loss_comparison.py` gained `--entropy-lambdas` (λ_H sweep → isolated
`_entlam<λ>` runs, mirrors `--evar-alpha`; no-op by default; smoke-tested). Theo launched `lambdaH_sweep` on gpu1
(`--entropy-lambdas 0 0.001 0.003 0.01 0.03 0.1 --targets Q --bin-sizes-ms 100 --windows half`).
- **Files:** `diagnostics/uncertainty_scaling_realdata.py` (M), `diagnostics/predict_mean_baseline.py` (new),
  `diagnostics/peakier_combinations.py` (new), `run_loss_comparison.py` (M — also folds the prior uncommitted
  singular `--entropy-lambda` WIP). Figures gitignored.
- **Open / next:** (a) **#4 dropout vs early stopping** — not started; load-bearing (`config` dropout knob default
  0.0 → `nn_classifier` MLP → `run_experiment` → runner flag) + full tests + launch block. (b) **#7 plotter**
  "temporal does/looks vs λ_H" — after `lambdaH_sweep` rsyncs down (reuses the peakiness collectors +
  `cross_loss_eval` skill). (c) `lambdaH_sweep` running on gpu1. (d) Mouse-2 tasks deferred. **Prior:** rising λ_H
  should pull temporal PCA's KL-skill toward 1 (optimum ~0.03–0.1); spatial λ_H-invariant (sanity check).
- **Tests:** focused suite green (113 passed) **with `OMP_NUM_THREADS=1`** — numpy PCA segfaults multi-threaded on
  macOS (see GOTCHAS). New diagnostics are plotting-only (not load-bearing).

### 2026-06-13 — New `rnn_rl_model/`: RL (actor-critic) sibling of the SI network model
Built a second generative test of the Similarity Framework: same grating Go/NoGo
pipeline as `si_network_model` (recurrent V1 → cosine-to-archetypes SI(t) → DDM),
but the V1→action mapping is learned by **actor-critic policy gradient** instead
of Hebbian, and the recurrent V1 is run **fixed** *or* **trained-end-to-end-then-
frozen** (the only difference between conditions is `W_rec`). One unified torch
graph; reuses the Hebbian sibling's stimuli/V1/IO/DDM. Design forks (sibling
package / both V1 modes / actor-critic) were resolved with Theo up front.
- **Result (cohort n=6).** The RL agent reproduces the framework's full RD
  signature: readout aligns with the **template `Δμ` (cos 0.88 fixed / 0.93
  trained)** not the whitened optimum (0.36 / 0.01); efficiency 0.99/0.95;
  `r(SI, IO log-odds)`=0.85. RD-2 **M3−M1 ≈ 0** (whitening adds nothing to choice)
  and **M2−M1 ≈ 0** (no SBC) in both modes. Task-training the recurrence makes V1
  covariance **stimulus-exploitable (RD-1 Δstim +0.033, like real mice's +0.04)
  while choices stay template** — the real-V1 dissociation, generated. The
  signature is thus robust to the learning rule, not just Hebbian.
  (M1−M0 is large by construction — SI *is* the policy input — so it only
  confirms the test fires.)
- **Files.** `rnn_rl_model/{config,model,train,evaluate,analysis,cohort,plots,
  rd_adapter,run}.py` + `tests/` (10 pass) + `README.md`/`UNDERSTANDING.md`.
  Small behaviour-preserving seam added to `si_network_model/v1_model.py`
  (`jittered_drive`); its 60 tests still pass. `results/` gitignored.
- **Next.** Worth adding as a second positive control to the Similarity-Framework
  vault note (Theo's synthesis call). Possible extensions: 2AFC lick-L/R; block
  prior; longitudinal κ over RL training; sweep `lr_v1` to map how much
  recurrence-training is needed before RD-1 Δstim turns positive.

### 2026-06-10 — evar^α (soft fix) fails + PC0/1 "two widths" refinement
Two follow-ups to the peakiness work. **(1) evar^α — negative result.** Added an `evar_alpha`
Config knob (default 1.0 = no-op; carried through `to_legacy_dict` → `fit_pca_basis`, `--evar-alpha`
CLI), the *multiplicative* cousin of `shape_lambda`: `evar_k → evar_k^α` renormalised (α=1 plain PCA,
α=0 ≡ flat-evar). Swept α∈{0.5,0.3,0.15} on wm3 (6 mice, PCA loss), scored vs the additive λ=0.1
benchmark (`diagnostics/evar_alpha_sweep.py`). **It does not work:** spatial peakiness stays ~0.20–0.24
(target 0.059) and KL-skill stays *worse than chance* (1.39–1.52) for every α>0; only α=0 (= flat) calibrates.
**Mechanism:** renormalised powers preserve the PC ordering, so the deep-tail high-freq PCs where peakiness
lives (PC22+, see below) keep weight ~1e-3 until α→0, vs the additive floor which puts an absolute 0.1 on
*every* PC. The restoring force must reach the **finest** directions — only the additive Brier floor does that
without flattening location. So the additive λ·Brier (§7) stays the recommended fix; the soft knob has no
useful operating point. **(2) PC0/1 stepping = "two widths".** Measured (real targets, mouse 0): stepping
PC0/1/2 sweeps the peak across ~88–90/91 bins (= location) and they're frequency-1 modes → smooth by
construction. The secondary envelope-width change PC0 induces is *coarse* (low-freq) width, which the loss
*does* constrain; the **fine sharpness/peakiness is high-frequency** and lives in the trailing PCs (PC22 ≈ 41
cycles) where evar≈0. So leading PCs = location + coarse width (constrained); trailing = fine sharpness (free)
— which is *why* over-confidence shows up as jagged high-freq spikes, not smooth narrowing. Committed
`134b92d` (plumbing + sweep). Vault note edits (negative-result paragraph in §7, two-widths refinement in §4)
**offered, not yet applied** — awaiting Theo's go.
- **Files:** `training/config.py`, `run_experiment.py`, `run_loss_comparison.py`, `diagnostics/evar_alpha_sweep.py` (new).
- **Open:** fold the two findings into the vault note if wanted; pre-existing `test_fit_model.py::
  test_train_and_select_best_model_uses_fit_model` failure (2-tuple unpack) flagged as a separate task — unrelated.

### 2026-06-10 — Peakiness "on-paper" derivation + quantification figure (Fig D)
Máté wasn't buying the asymmetric-basin curves (Fig 16) as the *why* for the increasing
peakiness, so reframed the argument in the currency a dynamics question wants — forces and
dimensions, not landscape shapes. Key identity: the PCA loss is **exactly** quadratic in PC-
projection space, so curvature along PC k is **exactly** `2·evar_k`. New `nn_decoder/diagnostics/
curvature_quantification.py` reads the real `wm3` bases/decoders (6 mice) and prints + plots the
three headline numbers: (a) IO-target manifold **effective rank ≈1.7** (PC0–2 = 97% var), curvature
**machine-zero by PC20** → a knife-edge ridge in the width direction, not a basin; (b) at the deployed
evar decoder the width-subspace error is **17× the location error** yet draws only **2% of the location
gradient** (~56× weaker restoring force) — the optimiser can't feel the error it's creating; (c) the
λ·Brier fix floors width curvature at `2λ`. Wrote the full derivation (no-restoring-force → frozen, not
peaky → softmax/shared-weight drift with a sharpening sign → free integrator → the cure is the missing
curvature) into the vault note as **§4½ + Fig D**, plus a TL;DR pointer and a Reproduce line; figure
synced to the vault. Committed `curvature_quantification.py` + sync map (`f23b0ce`).
- **Files:** `nn_decoder/diagnostics/curvature_quantification.py` (new), `sync_peakiness_figs.sh`;
  vault `Projects/Uncertainty/PCA-Peakiness-Mechanism.md` (§4½, Fig D, TL;DR, Reproduce).
- **Open:** none required. Optional follow-ups if Máté wants more: an `evar^α` (α<1) soft-compression
  variant as a one-knob alternative to the additive λ floor; the trained-as-target round-trip (§10).

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
→ marginalised inversion (integrate over p(m|s,y)). The **three IO-derived targets are shown
explicitly** as a dedicated "IO-derived targets" panel strip — perceptual posterior Q(θ),
perceptual likelihood L(θ), decision posterior [P(Go),P(NoGo)] — feeding the neural-decoder target
and the uncertainty read-outs U_perc=SD[Q], U_dec=H[P(Go),P(NoGo)]. Reuses the `peakiness_style`
palette + `figsave.save_fig` (PNG ≤1600px **and** SVG; `layout=None` for the hand-placed insets).
After Theo's feedback the layout was **de-densified** (15.5×10.5 canvas, generous inter-region gaps,
in-axes title to kill the top whitespace). Accurate to `documents/ideal_observer_methods_v3.tex` /
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
