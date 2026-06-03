# Revised plan — next steps after the 2026-06-03 Máté meeting

_Date: 2026-06-03. Branch: `main` (trunk-only). Supersedes the "Next steps" tails
of [`HANDOFF_kl_js_plots.md`](HANDOFF_kl_js_plots.md) and
[`HANDOFF_loss_demo_weight_evolution.md`](HANDOFF_loss_demo_weight_evolution.md)._

Companion to the vault note `2026-06-03-Uncertainty-Meeting` (the handwritten
"Meeting Máté" list) and its task routing in `UncertaintyV1-Tasks`.

---

## 0. State of play — what is already done

Settled by the prior two handoffs + the `loss_comparison_v1` run + the
`LOSS_SMOOTHNESS_REPORT.md` (all committed, working tree clean):

| Question | Verdict | Where |
|---|---|---|
| PPC peakiness — loss-driven? | **Yes** | smoothness report (width gradient ≈ 0 for PCA); `pca_loss_demo.py` on real posteriors |
| Peakiness — overfitting/early-stopping controlled? | **Yes** | `loss_comparison_v1` ran `patience=15`; train+val curves saved |
| Peakiness — weight-magnitude driven? | **No (surprising)** | `plot_weight_evolution_cell.py` #1 — PCA does *not* grow `‖W_out‖` most; Wasserstein does |
| "Entropy-reg acts as weight-reg"? | **Inverted** | #2 — SBC `‖W_out‖` ≥ PPC at matched epochs, for every loss |
| Matched loss comparison (PCA/CE/KL/JS/Wass) | **Run & on disk** | `results/loss_comparison_v1/` — Q&L × {half,full} × {50,100}ms × 6 mice |

**Plotting infra that exists:** `plot_kl_js_sweep.py`, `plot_kl_js_training.py`,
`pca_loss_demo.py`, `plot_weight_evolution_cell.py`, `loss_scatter_spat_temp.py`,
`pca_posterior_vs_likelihood.py` (PCA on *IO* posteriors/likelihoods),
`dim_reduction_explore.py` (PCA on *population activity*).

---

## 1. The new asks (Máté, 2026-06-03) → data reality

I inspected the `loss_comparison_v1` checkpoints. Each `checkpoints/mouse_*_<split>.pt`
holds, per arch (`spat`=PPC, `temp`=SBC):
- per-epoch `train_*`/`val_*` loss curves (fit, entropy, total, pca_yardstick);
- `weight_norms` (epochs × 4) = L2 of `[W_in, b_in, W_out, b_out]`;
- **`state_dicts` — full weight tensors snapshotted every 10 epochs, including
  epoch 0 (init)**. `layers.0.weight` = `W_in` (H=32 × N_in=105 neurons),
  `layers.1.weight` = `W_out` (91 cats × 32);
- `pred_probs` = decoded posteriors (spat `(n,1,91)`, temp `(n,T,91)`), also in the `.mat`.

**Consequence: items 2–7 below need no cluster time — they are new analysis on
data already on disk.** Only item 1 ("fewer hidden units") needs fresh fits.

| # | Meeting item | Feasible from disk? | Maps to |
|---|---|---|---|
| 1 | Overfitting comparison for **fewer hidden units** | train/val gap at H=32 **yes**; other H **needs re-run** | new ablation run + extend `plot_kl_js_training` curves |
| 2 | Posterior as **mean + PC1·a** reconstruction | **yes** | new script (decoded `pred_probs` of `loss_comparison_v1`) |
| 3 | **PC1/PC2 scatter of all posteriors** | **yes** | new script / extend `pca_loss_demo` geometry panel to all trials |
| 4 | **Normalise `‖W‖` by N input neurons** | **yes** | extend `plot_weight_evolution_cell.py` |
| 5 | **Why weight norms differ from init** / weird init? | **yes** (epoch-0 snapshot present) | extend `plot_weight_evolution_cell.py` |
| 6 | **mean + std of weights** instead of L2 norm | **yes** (snapshot tensors) | extend `plot_weight_evolution_cell.py` |
| 7 | Weight **regularisation** | partial | `weight_decay` sweep on PPC (cluster) |

Tangent — "Monte Carlo integrals in sampling models" — parked in the vault
`ideas.md`, not on this critical path.

---

## 2. Revised plan, prioritised

### Tier A — zero-cost wins on existing data (do first, all local)

**A1. Refit the weight-norm diagnostic to answer items 4–6 in one pass.**
Extend `plot_weight_evolution_cell.py` (it already loads the 6+ snapshots) to add:
- `‖W_in‖ / √N_in` and `‖W_out‖ / √H` (norm normalised by fan-in) overlaid by loss
  — the meeting's "normalise L2 by N input neurons". Compare raw vs normalised so
  the rescaling is visible.
- **mean ± std of each weight tensor vs snapshot-epoch** (replacing/augmenting the
  single L2 line) — distinguishes "norm grew because the distribution shifted/spread"
  from "a few weights blew up". Add a per-layer histogram of weights at init vs final.
- **init (epoch 0) reference line / Δ-from-init** on every curve — directly answers
  "why are the norms different from the beginning?". Spot-check from one cell:
  spat `W_in` std 0.122→0.157, temp 0.120→0.217 over training; `‖W_in‖/√N_in ≈ 0.69`
  at init. So weights **do** drift up from a clean, near-zero-mean init —
  **the init is not weird**, but confirm across cells/mice before claiming it.
- Keep the as-deployed ★ at `best_epoch` (carried over from the prior handoff's
  open "mark best-val epoch" item — fold it in here).

**A2. Posterior PCA visualisations (items 2 & 3).** New script, e.g.
`posterior_pca_views.py`, operating on the decoded `pred_probs` of a chosen
`loss_comparison_v1` cell (per loss, both archs):
- PCA the full set of decoded trial posteriors; **PC1/PC2 scatter of all trials**,
  coloured by a stimulus feature (target mean / contrast / dispersion).
- **mean-posterior + a·PC1 reconstruction strip**: render `mean ± {−2,−1,0,1,2}·σ₁·PC1`
  as actual posterior curves, to show what the leading axis of variation *is*
  (peak position vs width). Repeat for PC2.
- Overlay the **IO target** posteriors' PC1/PC2 for the same trials (reuse the
  basis-comparison machinery in `pca_posterior_vs_likelihood.py`) so decoder-vs-IO
  latent geometry is directly comparable. This also feeds the still-open vault task
  "Project neural posterior onto average posterior / check PCs remain the same".

_Decision needed (pick before coding A2):_ fit the PCA basis on **decoded
posteriors** (shows how the *decoder* spreads trials) or on **IO targets** and
project decoded onto it (shows decoder deviation from the ideal axis). Recommend
**both**, side by side — it's cheap and the contrast is the point.

### Tier B — needs the cluster (queue together)

**B1. Hidden-width ablation (item 1).** Re-run the matched comparison at a small
ladder of hidden sizes (e.g. `H ∈ {4, 8, 16, 32, 64}`) with `--track-history`
on, `stratified_balanced`, Q & L, 100ms/half, all 6 mice. Then plot the
**train–val gap vs H** per loss/arch — the explicit "overfitting for fewer hidden
n" comparison. The H=32 train/val curves already on disk are the anchor point.
Reuses the May-27 `[10]`-hidden ablation as a sanity cross-check.

**B2. PPC `weight_decay` sweep (item 7).** Already queued in the vault tasks
(current 1.2e-4 was tuned for SBC+PCA jointly; PPC alone may want 1e-3/1e-2).
Run alongside B1 since both are PPC-capacity questions; report peakiness + fit-loss
vs `weight_decay`.

**B3. (Carry-over) tracked entropy-λ re-run** `kl_js_entropy_sweep_tracked` —
only if the λ training/weight curves are still wanted for the talk; otherwise drop,
since `loss_comparison_v1` already carries tracked curves at the pinned λ=3e-3.

### Tier C — deferred / open (unchanged from prior handoffs)

- Trained-as-target round-trip (can the net represent its own smoothed output?).
- Stratified PCA basis + stratified test/plot reweighting.
- Refill Wasserstein/JS gaps in `loss_sweep_h10_val_2026_05_27` (smaller MICE
  batches + stagger to dodge CUDA OOM).
- Repo hygiene: `pca_loss_demo.py` vs `diagnostics/loss_smoothness_demo.py` overlap
  — consolidate only when the talk figures are locked.

---

## 3. Suggested order of execution

1. **A1** then **A2** (local, same afternoon — they close 5 of the 7 meeting items).
2. Kick **B1 + B2** to the cluster as one batch while A-tier plots are reviewed.
3. Fold A/B results into the talk deck arc (slide 8 "weight evolution" gains the
   normalised-norm + mean/std panels; a new posterior-geometry slide from A2).
4. Revisit Tier C only if a talk slide or reviewer question demands it.

## 4. Open decisions for the owner

- A2 basis choice (decoded vs IO target vs both — recommended both).
- B1 hidden-size ladder — is `{4,8,16,32,64}` the right span, or go lower (≤2) to
  force the underfitting regime?
- Whether B3 is worth the disk/GPU now or is subsumed by `loss_comparison_v1`.
