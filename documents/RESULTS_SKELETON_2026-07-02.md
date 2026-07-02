# Results skeleton — 2026-07-02

_Scaffold for the Results section under the recommended framing
(`PAPER_STRATEGY_2026-07-02.md` §2): "Mouse V1 represents perceptual uncertainty
by a template-similarity readout." Each subsection lists the claim, the figure, the
key numbers (from `PROJECT_LOG.md` / vault notes), and the generating script so the
prose can be written and every number re-verified. Numbers are transcribed from the
log; **re-run the cited script before quoting any of them in the manuscript.**_

---

## R1. A behavioural task and ideal observer that isolate perceptual uncertainty

**Claim.** Mice perform an orientation Go/NoGo categorisation across a graded
uncertainty space (orientation × contrast × dispersion); a two-stage ideal observer
(IO) fit to kinematics captures choice behaviour and yields a trial-by-trial
perceptual posterior we can use as ground truth for uncertainty.

- **Figure:** Fig 1 (schematic) + Fig 2 (behaviour/IO validation).
- **Key content:** 9 orientations (0–90°), 4 contrasts (1/25/50/100%), 4 dispersions
  (5/30/45/90°); decision boundary 45°. IO conditioned on velocity + licks; Full vs
  Reduced IO; GLM family as baselines/ceiling; NLL + AUC over the corridor.
- **Provenance:** Methods PDF; `diagnostics/io_schematic.py`. Behavioural IO/GLM
  choice + stimulus-decoding numbers to be pulled from the run behind the Methods
  pipeline. **Gap:** this results figure/text does not exist yet — needs writing.

## R2. A calibrated neural decoder recovers the perceptual posterior from V1

**Claim.** We decode the IO perceptual posterior from V1 population activity with an
MLP. The choice of loss is not cosmetic: the standard PCA-weighted loss manufactures
false confidence, so we validate and select a calibrated decoder before reading any
uncertainty off it.

- **Figure:** Fig 3 (decoder validation + peakiness fix).
- **Key numbers:**
  - PCA loss over-sharpens: decoded max-prob ≫ IO target; e.g. spatial PCA median
    max(P) 0.135 vs calibrated 0.043 and target 0.039 (pooled 2186 trials × 6 mice).
  - Mechanism: PCA loss constrains only the location subspace; shape-subspace error
    ~300× KL's (toy), real-data trailing-PC (PC22) shape error confirmed.
  - Fix: additive Brier / output-smoothness penalty lands on target. λ_smooth≈0.3 →
    peakiness temporal 0.075 / spatial 0.062 vs target 0.059; KL falls to ~0.48/0.49.
  - Intervention ladder is 5-for-5: λ_H sharpens (✗); dropout/averaging/smaller-H no
    help; smoothness/Brier + early-stop work.
- **Provenance:** vault `PCA-Peakiness-Mechanism`; `diagnostics/smooth_lambda_sweep.py`,
  `dropout_vs_earlystop.py`, `peakier_combinations.py`, `curvature_quantification.py`,
  `peakiness_distributions.py`.
- **Decision needed:** lock ONE production loss (CE/KL or PCA+smoothness λ≈0.3) and
  regenerate the decoded-uncertainty figures with it (strategy §4 gap 6).

## R3. V1 choices read the un-whitened template direction, not the whitened optimum

**Claim (headline).** The behavioural readout aligns with the template difference
Δμ, not the covariance-whitened optimum Σ⁻¹Δμ — even though exploitable covariance
demonstrably exists in the population.

- **Figure:** Fig 4 (similarity readout, RD-1 + RD-2).
- **Key numbers:**
  - RD-1: whitening lifts *stimulus* decoding in **6/6 mice (+0.040 AUC)** →
    exploitable covariance exists.
  - RD-2 nested choice models (held-out ΔLL): **M3−M1 ≈ 0 in all 6** (choices read
    the template, not the whitened optimum → premise supported); M1−M0 > 0 in 4/6
    (Pred 16, partial); M2−M1 ≈ 0 in 4/6 but +0.017/+0.011 in Cb17/Cb22
    (partial, animal-specific).
- **Provenance:** `nn_decoder/similarity_readout_tests.py`; vault
  `Conjectures/Similarity Framework` §Empirical Findings.

## R4. Two generative models reproduce the readout signature under different learning rules

**Claim.** A Hebbian V1 network and an actor-critic RL agent both land on the
template-Δμ readout, showing the signature is a robust consequence of learning a
similarity readout, not an artifact of one model class.

- **Figure:** Fig 5 (generative positive controls).
- **Key numbers:** template cos 0.88 (fixed) / 0.93 (trained); efficiency 0.95–0.99;
  r(SI, IO log-odds) = 0.85; RD-2 M3−M1 ≈ 0 and M2−M1 ≈ 0 in both. Task-training the
  recurrence makes covariance stimulus-exploitable (RD-1 Δstim +0.033, ≈ real mice's
  +0.04) while choices stay template.
- **Provenance:** `si_network_model/REPORT.md`, `rnn_rl_model/`.

## R5. The apparent spatial-vs-temporal architecture gap is a calibration artifact

**Claim (honest supporting result).** The PPC-vs-SBC performance gap is driven by the
PCA loss's over-sharpening; once width is matched, spatial ≈ temporal, and no mouse
shows a population posterior-width sampling signature. This supports a simple
template readout over an elaborate sampling code.

- **Figure:** Fig 6 (architecture as artifact).
- **Key numbers:**
  - Under evar-PCA the SBC-worse-than-PPC KL gap is large (raw p=0.008) but
    **vanishes** for flat/shape losses (PPC≈SBC) — a peakiness artifact.
  - Per-animal: PCA is the only loss where spatial decisively beats temporal
    (1.34 vs 2.17, p=0.01); calibrated losses a wash/lean temporal; robust to
    dropping Mouse 2.
  - `similarity_m2_followup`: Cb22 within-trial signal = artifact; Cb17 = genuine
    within-trial dynamics but NOT SBC posterior-width; **no mouse shows the SBC
    mechanism** (Var_t[SI] tracks decisiveness, wrong sign for SBC).
- **Provenance:** vault §9 of peakiness note; `cross_loss_eval.py`,
  `spat_temp_per_animal.py`, `compare_loss_variants.py`, `similarity_m2_followup.py`.

## R6 (supplementary). Feature ablation and negative controls

- Temporal-variance > order-sensitive features is robust; rate vs temporal-variance
  not significant at n=6 (`feature_ablation_analysis.py`).
- λ_H entropy penalty sharpens rather than broadens (negative result); CE/KL immune
  (`diagnostics/lambda_h_temporal_sweep.py`, `lambda_h_perbin_vs_avg.py`).
- Predict-mean / kill-weights / shuffle nulls order correctly; PCA below chance
  (`diagnostics/predict_mean_baseline.py`).

---

## Cross-cutting caveats to carry into every subsection

- **n=6** — scope all population claims; separate 6/6-robust (R3 M3−M1) from
  partial/animal-specific (M1−M0 4/6; R5 Cb17).
- **One production loss** — R2's locked loss must be the loss used everywhere
  downstream, or the paper is internally inconsistent.
- **Every number above is transcribed from the log** — regenerate from the cited
  script before it enters the manuscript.
