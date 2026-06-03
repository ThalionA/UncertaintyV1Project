# Handoff — PCA-loss demo, weight-evolution diagnostic & talk plan

_Date: 2026-06-03. Committed directly to `main` (no branches — see CLAUDE.md)._

Companion to [`HANDOFF_kl_js_plots.md`](HANDOFF_kl_js_plots.md) and the
`diagnostics/loss_smoothness_demo.py` report. Everything here is built on the
**matched, early-stopped, `entropy_lambda`-pinned `loss_comparison_v1`** run
(PCA/CE/KL/JS/Wasserstein, Q & L × {half,full} × {50,100ms} × 6 mice, full
history + checkpoints).

## TL;DR

Two new scripts and a slide plan, all serving the 2026-05-27 meeting's two
threads (PPC peakiness; honest spat-vs-temp). Plus one genuinely new empirical
result from the weight trajectories that **complicates the "weight-magnitude"
hypothesis** and **inverts the "entropy-reg-as-weight-reg" hypothesis**.

## What was built (this branch)

| File | What it does | Outputs (gitignored) |
|---|---|---|
| `pca_loss_demo.py` | Pedagogical "what the PCA loss sees" suite on a real trial: candidate gallery with all 5 losses per posterior, summary heatmap, fitted-by-loss (spat+temp), hand-picked example trials per loss, PC decomposition, directional sensitivity, PC1–PC2 geometry | `figures/loss_sweep_plots/loss_comparison_v1/pca_loss_demo/*` |
| `plot_weight_evolution_cell.py` | Output-weight-norm trajectories for ONE cell of a flat run; overlaid by loss + per-param small multiples (relabelled `W_in/b_in/W_out/b_out`) + spat-vs-temp; **both archs**, shared y-axes | `figures/loss_sweep_plots/loss_comparison_v1/weight_evolution/*` |

Reproduce:
```bash
cd nn_decoder
python pca_loss_demo.py                              # Q half 100ms, mouse 0
python plot_weight_evolution_cell.py                 # Q half 100ms balanced, both archs
# flags: --target L --window full --bin 50 --mouse N --split ...
```

## Key empirical result — weight evolution (Q, half, 100ms, balanced)

Reading `‖W_out‖` (output-layer weight norm; sets logit scale ⇒ posterior
peakiness) vs epoch, across-mouse mean, 6 mice:

1. **PCA does NOT grow the output weights most.** Among PPC, **Wasserstein**
   climbs highest (~14), JS next; PCA/CE/KL cluster lower and PCA's `‖W_out‖`
   *declines* after ~epoch 75. ⇒ The "PPC is peaky because PCA inflates the
   output weights" story does **not** hold. PCA's peakiness is better explained
   by loss *geometry* (no width gradient — see `LOSS_SMOOTHNESS_REPORT.md`) than
   by weight magnitude.

2. **"Entropy-reg acts as weight-reg" is backwards.** For **every** loss the
   **SBC (temp) `‖W_out‖` ≥ PPC (spat)** at matched epochs — not smaller. The
   per-bin entropy penalty *rewards sharp per-bin softmaxes*, which needs
   *larger* logits ⇒ larger `W_out`. The SBC trial posterior looks smooth only
   via Jensen time-averaging, not because its weights are smaller. This is a
   clean (slightly counter-intuitive) answer to the meeting's open question.

**Caveats to keep on any slide:** (a) early stopping restores the *best-val*
weights, so the deployed `W_out` is at `best_epoch`, earlier than the curve end
— add a ★ marker if you want as-deployed values; (b) `‖W_out‖` is a whole-tensor
summary (logit scale, not which units); (c) one cell only — confirm on others
before generalising.

## How this maps to the 2026-05-27 meeting

| Meeting item | Status | Evidence |
|---|---|---|
| Peakiness: loss-driven? | **Resolved — yes** | `LOSS_SMOOTHNESS_REPORT.md`; `pca_loss_demo` (width gradient ≈ 0 for PCA) |
| Peakiness: capacity? | Partial | May-27 `[10]`-hidden ablation |
| Peakiness: overfitting? | **Resolved — early stopping done & used** | `loss_comparison_v1` ran `patience=15` |
| Weight-magnitude driven? | **Resolved — no (surprising)** | `plot_weight_evolution_cell.py` finding #1 |
| Entropy-reg = weight-reg? | **Resolved — inverted** | finding #2 |
| Honest spat-vs-temp | Partial | `loss_scatter_spat_temp.py` (not mine), `cross_loss_eval` `11/12` |
| Trained-as-target round-trip | **Open** | deferred |
| Stratified PCA basis + plotting | **Open** | deferred |

## Talk plan (condensed)

Build the deck on `loss_comparison_v1`. Arc:
1. Frame: the meeting's two questions.
2. Peakiness tracks the **loss** (CE/KL/JS match target; PCA ~8×; MSE collapses)
   — `5_peakiness_spat_vs_temp` regen on this run.
3. **Why:** width-vs-shift asymmetry (KL 22× / JS 6× / PCA 1.5×) —
   `loss_smoothness_demo/fig9`; geometry on real data — `pca_loss_demo` geometry/decomposition.
4. SBC smoothness is mostly **Jensen** — `3_per_bin_sbc_peakiness`, `shuffle_asymmetry`.
5. Entropy-λ is the trigger — `kl_js_entropy_sweep_v1` figs + `fig4_temporal_training_outcome`.
6. Correct shape is **~free** — `cross_loss_eval` / `4_fit_loss`.
7. Overfitting **controlled** (early stopping done) — `plot_kl_js_training` curves.
8. **Weight evolution** (this session) — the two findings above.
9. Honest spat-vs-temp — `loss_scatter_spat_temp`, `11_spat_vs_temp_diff`.
10. Decisions (adopt KL/JS for width? `evar^p` flatten? entropy-match term?) + open items.

Full slide-by-slide plan is in the session transcript; lift into the vault note
if you want it persisted.

## Overlap / dedup flag (repo hygiene)

`pca_loss_demo.py` (this session) and `diagnostics/loss_smoothness_demo.py`
(pre-existing) make **the same core argument** ("PCA = variance-weighted L2, no
width gradient ⇒ peaky"). Keep both only because they are complementary:
`loss_smoothness_demo` = synthetic, with gradient-descent confirmation +
width/shift asymmetry; `pca_loss_demo` = the same on **real posteriors** + the
geometry/Mahalanobis framing + fitted-by-loss / example-trial figures. If
consolidating later, fold `pca_loss_demo`'s real-data figures in and drop the
duplicated synthetic-candidate scorecard.

## Working-tree state NOT in this branch (needs owner triage)

These predate this session and were left untouched — decide where they belong:
- Modified: `cross_loss_eval.py`, `plot_loss_sweep.py`, `within_mouse_loss_plots.py`
- Untracked: `loss_scatter_spat_temp.py`, `plot_all_cells.py`,
  `plot_loss_spat_temp_comparison.py`

## Next steps

- Mark best-val (as-deployed) epoch on the weight-evolution curves.
- Regenerate weight-evolution + peakiness for the other cells you'll show
  (`full`/100ms, L target).
- Tracked entropy-λ re-run (`kl_js_entropy_sweep_tracked`) for its training/weight curves.
- Meeting's still-open diagnostics: trained-as-target round-trip; stratified PCA basis.
