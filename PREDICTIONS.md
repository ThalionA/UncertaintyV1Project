# Predictions ledger

Register a falsifiable prior **before** the outcome (no hindsight); resolve after; keep the large errors. Newest at top. Classes: `✓ confirmed` · `↔ magnitude` · `✗ direction` · `⚠ invalidated`.

---

## 2026-06-14 — λ_H sweep: effect on the temporal decoder  ✗ direction (+ ⚠ invalidated assumption)
**Registered (pre-launch, 2026-06-14, in-session before the gpu1 run):** rising λ_H broadens temporal posteriors and pulls temporal-PCA KL-skill down toward 1, with an optimum λ_H≈0.03–0.1; CE/KL flat; spatial λ_H-invariant. Confidence ~60%.
**Outcome (6 mice, λ_H∈{0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1}):** temporal-PCA KL-skill rose **monotonically 2.29→3.41 (worse)** and peakiness **0.35→0.49 (peakier)**; Wasserstein 1.37→1.72; JS over-confident only at λ_H=0.1; CE/KL immune (~0.5, flat); spatial flat (✓ that sub-prediction held). Validated real, not a bug: spatial control flat, and `entlam0p003` ≈ `loss_comparison_v1` (2.20 vs 2.17).
**Why wrong:** the entropy term is `total = fit_loss + λ·H(pred)`, minimised → drives H *down* → it **sharpens** (penalises entropy), opposite to the max-entropy / broadening regulariser I assumed. The "spatial flat" part was correct.
**Lesson:** check the **sign** of a penalty/loss term — or the relevant `GOTCHAS.md` entry — before registering a prior about its direction. GOTCHAS already said "the SBC entropy penalty enforces sample-sharpness; the data don't want this commitment." Science: the temporal decoder does **not** want the sharpness commitment — λ_H=0 is best for the peaky losses (PCA/Wasserstein), CE/KL are robust to it.
