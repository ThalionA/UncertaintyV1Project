# Predictions ledger

Register a falsifiable prior **before** the outcome (no hindsight); resolve after; keep the large errors. Newest at top. Classes: `✓ confirmed` · `↔ magnitude` · `✗ direction` · `⚠ invalidated`.

---

## 2026-06-14 — λ_H sweep: effect on the temporal decoder  ✗ direction (+ ⚠ invalidated assumption)
**Registered (pre-launch, 2026-06-14, in-session before the gpu1 run):** rising λ_H broadens temporal posteriors and pulls temporal-PCA KL-skill down toward 1, with an optimum λ_H≈0.03–0.1; CE/KL flat; spatial λ_H-invariant. Confidence ~60%.
**Outcome (6 mice, λ_H∈{0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1}):** temporal-PCA KL-skill rose **monotonically 2.29→3.41 (worse)** and peakiness **0.35→0.49 (peakier)**; Wasserstein 1.37→1.72; JS over-confident only at λ_H=0.1; CE/KL immune (~0.5, flat); spatial flat (✓ that sub-prediction held). Validated real, not a bug: spatial control flat, and `entlam0p003` ≈ `loss_comparison_v1` (2.20 vs 2.17).
**Why wrong:** the entropy term is `total = fit_loss + λ·H(pred)`, minimised → drives H *down* → it **sharpens** (penalises entropy), opposite to the max-entropy / broadening regulariser I assumed. The "spatial flat" part was correct.
**Lesson:** check the **sign** of a penalty/loss term — or the relevant `GOTCHAS.md` entry — before registering a prior about its direction. GOTCHAS already said "the SBC entropy penalty enforces sample-sharpness; the data don't want this commitment."
**Correction (same day, after Theo flagged it):** the prior itself *misframed the experiment* — penalising entropy obviously sharpens; that was never the question. The real question: can **peaky per-bin (instantaneous) samples coexist with a broad, calibrated time-average** under the information-theoretic losses? Re-analysed with `diagnostics/lambda_h_perbin_vs_avg.py` (per-bin vs time-avg entropy): **CE/KL inert** (per-bin H ≈3.65 ≈ IO target across all λ_H; sampling spread flat ~0.26 → bins are broad copies of the average); **JS is the one info-theoretic loss that tolerates λ_H** (per-bin 3.71→3.28, spread 0.23→0.37, average still ≈ target at λ_H=0.1); PCA/Wasserstein get peaky bins but an uncalibrated average. **Deeper lesson (the real one):** understand what an experiment is *testing* before forming a prior on its outcome — I predicted the time-averaged direction, which was not the question being asked.
