# Paper progress strategy — 2026-07-02

_A plan to turn the existing body of analysis into a submittable manuscript.
Read alongside `PROJECT_LOG.md` (the result inventory) and the Methods PDF
`documents/Representation_of_Perceptual_Uncertainty_in_Mouse_V1.pdf`._

---

## TL;DR — the one thing blocking the paper

The science is largely done. What is missing is **a paper**, and the reason it
does not exist yet is a decision, not a result: **the headline framing is still
open** (it has been carried as `[open]` in `PROJECT_LOG.md` since 2026-05-16).
Every downstream step — which figures make the main text, what the abstract
claims, which analyses are "supporting" vs "headline" — is blocked on that one
choice.

**Concrete state:**

- **Methods** — written and committed (the PDF). Covers stimulus generation, the
  behavioural task, the two-stage Ideal-Observer model, the kinematics-based
  choice/stimulus decoding (IO + GLM family), and the neural-decoder loss
  functions (PCA, CE, KL, JS, Wasserstein). This is the most finished part of the
  paper.
- **Results / Introduction / Discussion** — **do not exist as narrative.** All
  findings live in ResearchVault notes and in `nn_decoder/diagnostics/*.py`.
- **Figures** — most main-text figures already have generating code; they are not
  yet assembled into a paper figure set or a figure-numbering scheme.

**Recommendation:** make the framing decision now (see §2), then execute the
figure-by-figure plan in §3. The first ~2 weeks of work are almost entirely
**writing and assembly from results already on disk** — no new cluster runs are
on the critical path.

---

## 1. What we have (result inventory, rolled up)

Three mature, largely independent result-clusters exist. Each is publishable-grade
in isolation; the strategic question is how to combine them.

| Cluster | One-line claim | Maturity | Home |
|---|---|---|---|
| **Similarity Framework** | Mouse V1 choices read the un-whitened template direction (Δμ), not the covariance-whitened optimum, in 6/6 mice; the discarded covariance carries +0.04 stimulus-AUC the choices ignore. Reproduced by two generative models (Hebbian + actor-critic RL). | **Decisive** (2026-06-08/13) | vault `Conjectures/Similarity Framework`; `nn_decoder/similarity_readout_tests.py`, `si_network_model/`, `rnn_rl_model/` |
| **PCA-peakiness mechanism** | The PCA decoder loss over-sharpens posteriors because it constrains only the location subspace and leaves the high-frequency width subspace free; calibrated losses (CE/KL) don't. The fix is an additive Brier / output-smoothness penalty (λ_smooth≈0.3). Loss-geometry model is 5-for-5 on interventions. | **Very complete** (methods-grade) | vault `PCA-Peakiness-Mechanism`; `nn_decoder/diagnostics/` (toy + real-data) |
| **Spatial vs temporal (PPC vs SBC)** | The apparent architecture gap is largely a **calibration artifact** of the PCA loss; once width is matched, PPC≈SBC. Residual SBC signal is small, metric-shaped, and carried by 3/6 mice. No population sampling-code claim survives. | Honest, mostly negative | vault §9 of peakiness note; `cross_loss_eval.py`, `spat_temp_per_animal.py`, `similarity_m2_followup.py` |

Supporting pieces: behavioural IO/kinematics choice prediction and stimulus
decoding (the Methods pipeline), feature-ablation (temporal-variance > order,
n=6-limited), and the IO model schematic.

---

## 2. Framing decision (the gating step)

The log records three candidate framings. My assessment of each:

- **(a) "PPC vs SBC architecture" (the framing the Methods PDF is built around).**
  **Do not lead with this.** Our own strongest result is that the architecture gap
  is largely a calibration artifact and the residual sampling-code signal does not
  survive at the population level (`similarity_m2_followup`: no mouse shows the SBC
  posterior-width mechanism). Leading with a claim we then dismantle is a weak paper.
- **(b) Feature ablation (temporal-variance > order-sensitive).** Real but narrow,
  and n=6-underpowered on the rate-vs-variance contrast. A supporting result, not a
  headline.
- **(c) Similarity Framework.** The only cluster with a **decisive, positive,
  multi-mouse mechanistic claim** plus two independent generative positive controls.
  This is the natural headline.

### Recommended framing

**Primary paper — biology.** _"Mouse V1 represents perceptual uncertainty by a
template-similarity readout."_

- **Headline claim:** choices read the un-whitened template direction; the
  covariance the optimal readout would exploit is measurably present (+0.04
  stimulus-AUC) but behaviourally discarded — a specific, falsifiable statement
  about *how* uncertainty is read out, supported in 6/6 mice and reproduced by two
  learning rules.
- **Rigor spine:** to trust any posterior/uncertainty readout, we need a decoder
  that doesn't manufacture confidence. The PCA-peakiness mechanism + the
  calibrated-loss choice (CE/KL, or PCA+smoothness) becomes a **Methods-validation
  figure** — it justifies the decoder and pre-empts the obvious reviewer question
  "how do you know the decoded uncertainty is real and not a loss artifact?"
- **Honest architecture result:** spatial-vs-temporal enters as a supporting
  section showing the architecture gap is a calibration artifact — which *supports*
  the template-readout story (a simple linear/template readout, not an elaborate
  sampling code, is what the data want).

**Why this ordering wins:** it turns our most complete negative/methods result (the
peakiness work) into the credibility foundation for our most complete positive
result (the similarity readout), instead of making them compete for the headline.

### Viable spin-off (do not let it block the biology paper)

**Methods paper — "Decoding posteriors from neural activity: how the PCA loss
manufactures false confidence, and how to fix it."** The peakiness cluster is
strong enough to stand alone (toy proof + real-data + a 5-for-5 intervention
ladder + a working fix). If the biology paper's reviewers want the peakiness detail
trimmed, it has a home. Recommend: **write the biology paper first**, carve the
methods paper out later only if the peakiness figure overflows.

> **Decision owner:** this framing call is Theo's / Máté's, not the agent's. §3
> assumes the recommended framing; if (a) or a methods-first ordering is chosen,
> the figure list reorders but most of the underlying assets are the same.

---

## 3. Proposed main-text figure plan (recommended framing)

Each figure lists its **provenance** (existing script / vault note) and **status**.
"Assemble" = code exists, needs paper-quality composition + caption. "Write" =
result exists, needs prose.

| Fig | Content | Provenance | Status |
|---|---|---|---|
| **1** | Task + IO model schematic: stimulus space (θ, contrast, dispersion), Go/NoGo, the two-stage IO, the three IO-derived targets (perceptual posterior, likelihood, decision posterior). | `diagnostics/io_schematic.py`; Methods PDF | Assemble |
| **2** | Behaviour + IO validation: psychometrics, kinematics (velocity/licks), IO vs GLM choice prediction (NLL/AUC over the corridor). | Methods pipeline (behavioural IO/GLM) | **Write + assemble** (result exists in Methods framing; needs a results figure) |
| **3** | The neural decoder and its validation: decoded posteriors vs IO target; the PCA-peakiness problem (over-sharpening) and the calibrated-loss / smoothness fix (λ≈0.3 lands on the IO target). Establishes the decoder is trustworthy. | `PCA-Peakiness-Mechanism` note; `diagnostics/smooth_lambda_sweep.py`, `dropout_vs_earlystop.py`, `peakier_combinations.py` | Assemble (condense a very rich note to 1 main + 1 supp figure) |
| **4** | **Headline — similarity readout.** RD-1 whitening lifts stimulus decoding (+0.04 AUC, 6/6) → exploitable covariance exists; RD-2 nested choice models: M3−M1≈0 in 6/6 (choices read the template, not the whitened optimum). | `nn_decoder/similarity_readout_tests.py`; vault `Similarity Framework` §Empirical Findings | Assemble |
| **5** | Generative positive controls: Hebbian (`si_network_model`) and actor-critic RL (`rnn_rl_model`) both reproduce the full RD signature (template cos 0.88–0.93, efficiency ~0.95–0.99, M3−M1≈0). Robust to learning rule. | `si_network_model/REPORT.md`, `rnn_rl_model/` | Assemble |
| **6** | Honest architecture section: spatial vs temporal is a calibration artifact (PPC≈SBC once width-matched); no population posterior-width SBC signature (`similarity_m2_followup` verdict). | vault §9; `cross_loss_eval.py`, `spat_temp_per_animal.py`, `similarity_m2_followup.py` | **Write** (numbers exist; needs the honest-framing prose) |

Supplementary: feature ablation; λ_H / entropy-penalty negative result; per-animal
spat/temp; weight-evolution diagnostics; the full peakiness intervention ladder.

---

## 4. Gap analysis — what is genuinely missing before a draft

Ordered by whether it blocks a *first complete draft*.

**Blocks the draft (must do):**
1. **Framing sign-off** (§2) — Theo/Máté. Everything waits on this.
2. **Results narrative** — there is no Results text anywhere. This is the single
   largest writing task. All the numbers exist in the vault; they need to become
   prose organised around Figs 2–6.
3. **Introduction + Discussion** — do not exist. The Discussion needs to state
   plainly what the template-readout result does and does not claim (n=6, the
   partial/animal-specific SBC signal, the measured-but-discarded covariance).
4. **Figure assembly** — most scripts emit per-analysis figures, not composed
   multi-panel paper figures with a shared visual language. `peakiness_style.py` /
   `figsave.py` already give a house style; extend it to the similarity + behaviour
   figures and compose the six main figures.

**Strengthens but does not block:**
5. **n=6 ceiling** — the recurring statistical limit. Not fixable by analysis;
   must be stated honestly and the claims scoped to it. Worth a power/consistency
   supplementary (which results are population-robust vs 3/6-carried).
6. **Production-loss lock-in** — decide CE/KL vs PCA+smoothness(λ≈0.3) as *the*
   decoder for the paper, and regenerate the headline decoded-uncertainty figures
   with that one loss so the paper is internally consistent. (Currently results are
   spread across losses for the mechanism story.)
7. **Methods PDF updates** — `documents/methods_updates_required.md` (local) lists
   out-of-date sections; reconcile once framing is locked.

**Explicitly deferred (do not let these block submission):**
- Trained-as-target round-trip; stratified PCA basis; Wasserstein/JS gap refills;
  Mouse-2 "what's different" exploratory; cross-modal DDM (Similarity Preds 7,8);
  the λ_smooth over-smoothing onset beyond λ=1. All are Tier-C / reviewer-driven.

---

## 5. Sequenced execution plan

The critical path is **writing and assembly**, not computation. Suggested order:

1. **Lock the framing** (§2). Blocking; owner decision.
2. **Lock the production decoder loss** (gap 6) — one loss for all headline
   uncertainty figures. Local, ~1 session; regenerate Fig 3–4 inputs.
3. **Draft Results around Figs 2–6** from the vault numbers. Largest task; pure
   writing. Can start the moment framing is locked.
4. **Assemble the six main figures** onto a shared style (extend `peakiness_style`
   / `figsave`). Parallelisable with step 3.
5. **Draft Introduction + Discussion.** Discussion must scope claims to n=6 and
   state the honest architecture result.
6. **Reconcile the Methods PDF** with the locked framing (gap 7).
7. **Internal review pass** — check every in-text number against its generating
   script (the vault notes already cross-reference scripts; verify nothing drifted).
8. **Supplementary** — assemble the deferred/supporting figures.

Only step 2 (optionally) touches compute, and it is local. No cluster run is on the
critical path to a first complete draft.

---

## 6. Risks to the headline claim (address in Discussion, not by hiding)

- **n=6.** Several contrasts are population-underpowered; the template-readout
  result (M3−M1≈0, 6/6) is the robust one and should carry the headline. Secondary
  contrasts (M1−M0 in 4/6; M2−M1 animal-specific) must be reported as partial.
- **The "discarded covariance" cost is modest (+0.04 AUC)** and measured, not huge.
  Frame as "the brain leaves recoverable information on the table," not "the readout
  is far from optimal."
- **The similarity readout is a constrained-linear claim** (template vs whitened),
  not a claim that V1 is literally computing a vMF LLR — the conjecture already
  makes this precise; the paper must too.
- **The peakiness result could read as "our tool is broken."** Framed correctly it
  is the opposite — it is *why the reader should trust our uncertainty readouts*.
  Lead Fig 3 with the fix working, not with the failure.

---

## 7. Immediate next action

The highest-leverage next step that does not require the framing decision is
**drafting the Results skeleton** (section headings + figure callouts + the key
number for each, pulled from the vault) so that the moment framing is signed off,
writing can proceed against a concrete scaffold. That skeleton is the natural
follow-up to this document.
