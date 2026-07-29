# Presentation outline — over-sharpening is loss geometry, and it flips the PPC/SBC conclusion

Prepared 2026-07-28; **2026-07-08 meeting folded in 2026-07-29** after the handwritten
page surfaced. Covers the action items from **2026-06-18** and **2026-07-08** and turns
the work since into a figure-first talk. Depth before breadth: **three results, fully
defensible**, bracketed by a status slide, a methods-credibility slide, an honest-limits
slide, and a closing slide of questions for Máté.

**If you read one section, read §0.2** — two of the four 08/07 asks are yours and neither
is started, which changes how the talk has to open.

---

## Part 0 — Todo audit

### 0.1 Meeting 2026-06-18 (Máté, Nathalie, Ishan) — vault note `Uncertainty Meetings/2026-06-18-Uncertainty-Meeting.md`

| # | Action item | Status | Where it landed |
|---|---|---|---|
| 1 | Upload ppt | ✅ done (Theo) | — |
| 2 | Normalise per-PC loss to its eigenvalue | ⚠️ **half** | Diagnostic done (`subspace_error_realdata.py --weight evar`): PCA's raw shape error is 38× KL's, but weighted by explained variance it collapses to ~2× and ~1% of the loss → **the loss is blind to it**. The *training-side* eigenvalue-normalised loss was never run as asked; `shape_lambda` (weights `evar + λ`, i.e. eigenvalue weighting progressively **diluted**) was swept instead and is the cure. |
| 3 | Increase dropout, incl. multiplicative Gaussian | ⚠️ **Bernoulli only** | p ∈ {0,.1,.25,.5,.75,.9}. Spatial peakiness reaches target only at p=0.9 (0.062) — but normalised loss 1.14 (worse than chance) and temporal stays 0.469 ≈ 8× target. **Dropout is a lobotomy, not a fix.** Multiplicative Gaussian dropout **not run**. |
| 4 | "How else can we solve overfitting?" | ✅ done | Three diagnostics: `peakiness_vs_hparams` (bias), `overfitting_vs_hparams` + `overfit_vs_capacity` (variance), `cure_comparison`. Answer: generic knobs fix variance and not bias; **the bias fix is loss-side**. |
| 5 | Over-fitting tests in validation, with synthetic/fitted targets | ✅ done — **and it is one of the two strongest results** | `roundtrip_loss_refit.py`. See Slide 4. |
| 6 | Other architectures → peakiness? incl. **no hidden layer** | ✅ done — **decisive** | prodfix ARM C. See Slide 3. |
| 7 | Show chance level on train–val curves | ✅ done | `shuffle_trainval_curves.py`, chance line at 1.0. |
| 8 | Use the predict-mean posterior as the normalisation | ✅ done | Used everywhere now; upgraded to a **leave-one-out** predict-mean in the July audit (the old one was fit on the trials it scored). |
| 9 | Encoding approach, posteriors / stimulus → neural responses (**Ishan**) | ❌ **not built by anyone** | Logged as unbuilt extension **E4** in `documents/AUDIT_2026-07.md:255-260`. Re-raised on 08/07 (see §0.2 D). **Do not let this read as done on a slide** — the tick below belongs to item 10 only. |
| 10 | Send neural data to Ishan + example plots + readme | ✅ done (Theo) | `documents/vr_export_handoff/` — but see §0.2 for the missing `fig4` and the fact that the handoff is untracked. |

### 0.2 Meeting 2026-07-08 (Máté, Ishan — **no Nathalie**) — vault note `Uncertainty Meetings/2026-07-08-Uncertainty-Meeting.md`

Transcribed 2026-07-29 from the handwritten page. Deck presented:
`ppt updates/20260708_ProjectUpdate.pptx` (5 slides: title, the H=32 baseline,
Overfitting, Peakiness). "27/07 back" at the foot of the page = the return-from-holiday
marker, so **this is the first meeting since**.

**Read this section before writing a single slide: two of the four items are yours, and
neither has been started.** All four verdicts below were checked against the repo and
adversarially re-checked.

| | Action item | Status | Evidence |
|---|---|---|---|
| **A** | Ishan implemented IO-HMM; validation still to be done | *Ishan's task* — **unverifiable from this side** | No Ishan-authored code, data or output exists in either tree (`git log --all -i --grep=ishan` → 0 commits). **And "IO-HMM" is ambiguous three ways** (below). |
| **B** | Compare posteriors with Ishan's — *"if he ever sends them"* | 🚫 **blocked, and long overdue** | Nothing has arrived. "Ask Ishan for his IO outputs" was ticked on **2026-05-06** with "Compare" left unticked; **2025-12-15** already records *"Disparity between Ishan and me"* — a comparison ran in **Dec 2025**, found a mismatch, and was never closed. |
| **C** | Noise variance, weighted inversely → per trial on my posteriors | ❌ **not started, under every reading** | No per-trial loss weighting exists repo-wide (`sample_weight`, `trial_weight`, heteroscedastic, `logvar`, aleatoric → nothing); `evar_alpha` only ever swept on α ∈ [0,1], never negative. |
| **D** | Do encoding from posterior to neural activity | ❌ **not started** | Every fitted model in both trees has neural activity as the *predictor*. Logged as unbuilt extension **E4**, `documents/AUDIT_2026-07.md:255-260` — which closes *"Ishan is already assigned the encoding direction"*, so **who owns it is itself unresolved**. |

**A — the name collision you must not walk into.** Theo's own repo contains
`ideal_observer/io_hmm/` — *"IO-HMM: Hidden Markov Model with state-indexed **Ideal
Observer** emissions"*, **already fitted to all 6 mice** (`real_fit_out/fit_Cb*.npz`) —
*and* `glm_hmm/`, the behavioural GLM-HMM. Ishan's remit has covered the ideal observer
(shared IO repo, 2025-11-07) *and* the GLM-HMM (2026-02-18, 2026-03-25). So "Ishan
implemented IO-HMM" has three referents. **One email settles it; a wrong guess wastes a
session.**

**B — the work is nearly free once the file arrives.** `nn_decoder/cross_loss_eval.py`
already scores arbitrary saved posterior pairs under KL/JS/Wasserstein with shuffle
normalisation and paired within-animal stats; `plot_spat_temp_divergence.py` is the
ready-made side-by-side (Δ-ranked deciles, MAP-aligned, per-mouse grid). Swap the
spat/temp axis for Theo/Ishan. **Trap:** Theo's targets are on a folded **91-point 0–90°**
grid, the measurement grid is **181-bin 0–180°** (`vr_export_handoff/README.md:156,177`)
— regrid before any divergence is computed, or the number is meaningless.

**C — do not guess what he meant; the readings imply opposite experiments.**

1. **Per-trial inverse-noise weighting of the loss** (weight trial *n* by 1/σ²ₙ). Best fit
   to the words — the sub-arrow says *calculate per trial*, i.e. a scalar per trial. It is
   the natural upgrade to the 18/06 predict-mean normalisation, which is currently a
   *single per-mouse denominator* (`decoder_plotting_utils.py:286-296`). Given the same-day
   round-trip finding that the train–val gap is an unachievable/noisy-target artefact,
   down-weighting high-noise trials is the textbook GLS response. **Genuinely unbuilt.**
2. **Per-PC inverse weighting — invert the noise-variance spectrum.** Weakest fit to "per
   trial", strongest repo support: `pca_basis='residual'` (`run_experiment.py:129-207`)
   fits the PCA on per-trial (target − within-cell mean), so its `explained_variance`
   **is** a per-PC noise variance — and it already runs as `prodfix_v1` arm B. Would be a
   two-line change in the same weight-transform block as `flat_evar`/`shape_lambda`.
   **But note the sign:** inverse-*noise* weighting further down-weights the trailing shape
   PCs — the *opposite* direction to the known cure — whereas inverse-*eigenvalue*
   weighting pushes toward it. So the tempting "he's restating the 18/06 eigenvalue ask"
   story is weaker than it looks; both the vault gloss of that item and Máté's 2024-02-13
   spec ("weight *is* variance along each PC") point at **forward** weighting.
3. **Precision weighting in the cue-combination sense** — combine per-bin posteriors
   weighted by 1/σ² instead of the current equal-weight Jensen average. This is Máté's
   home turf and a computational neuroscientist's default reading of the phrase. It has
   *zero* footprint in this project, which is weak grounds for ranking it third. **Do not
   discount it.**

   **Stats trap if you pursue (1) via κ:** `ideal_observer/io_hmm/io_core.py:89`
   (`kappa_for_trial`) is a deterministic function of (contrast, dispersion), so it is
   constant within a stimulus cell. Anything built on it is per-**condition**, n = conditions
   — not per-trial. A real trial-to-trial noise variance has to come from the residual side.

**D — data ready, model absent, ownership unclear.** `Gspk`/`Cspk` are already
row-aligned to `post_s_marginal` (join verified, recipe at
`vr_export_handoff/README.md:189-202`). Three tiers: (i) hours — per-neuron ridge/Poisson
`rᵢ ≈ f(posterior features)`, scored by cross-validated held-out R²; (ii) **build the null
first or it is vacuous** — a per-condition-mean encoder (clone
`stim_mean_baseline.compute_condition_means`, swap the target for activity), because E1 and
the DeepSets result both say condition explains nearly everything; (iii) what the audit
actually asks — tuning curves + noise covariance → p(r|s), invert, compare the implied
posterior to the IO's, **with no trained readout in the loop**, which is the point given
that this project's headline is that decoded posterior width is a property of the loss.
Nearest existing code is **off-main and synthetic**: `nn_decoder/simulation/generative.py`
on branch `origin/claude/spatiotemporal-neural-decoder-I4435` already does posterior →
linear-PPC Poisson counts / SBC samples — worthless as a result, valuable as a synthetic
ground truth with known p(r|s) to validate an estimator on before touching six mice.

### 0.3 Residual from 2026-06-10, still open

- **"What's different about Mouse 2?"** — the *exploratory* question is untouched.
  The *robustness* half is done: every headline is now reported at n=6 and n=5 (M2
  excluded), and M2 changes nothing (several effects get *smaller* p at n=5). M2 is
  the best-performing animal but moves with the rest.
- Peakiness-by-orientation: done 2026-06-14 (IO is U-shaped in orientation;
  over-sharpening is orientation-structured, worst at the 45° boundary — with the
  caveat that the *ratio* spike there is partly a small IO denominator).

---

## Part 1 — The talk

**One-sentence spine:** *The projection-based loss trains systematically over-confident
decoders; its own metric cannot see this; the cause is loss geometry rather than network
capacity; and once the loss is calibrated, the spatial-vs-temporal conclusion reverses.*

11 slides + 5 backup. Every number below is on disk; every figure has a PNG+SVG pair.

**The framing problem, and the fix.** Everything in Slides 2–7 answers the **18/06** asks.
Of the **08/07** asks, two are yours and neither is started. That is defensible — the
loss-geometry thread produced the strongest results the project has — but only if *you*
say it in the first minute rather than letting Máté find it in the last. Hence Slide 0.

---

### Slide 0 — Where the 08/07 asks stand (30 seconds, then move on)

Four rows, verbatim from the handwritten page, with an honest mark against each:

| Ask | Status |
|---|---|
| Ishan's IO-HMM validation | his side; nothing received |
| Compare posteriors with Ishan's | **blocked** — still nothing, and the Dec-2025 disparity was never closed |
| Noise variance, inversely weighted, per trial | **not done** — and I need one sentence from you to know which of two opposite experiments you meant |
| Encoding: posterior → neural activity | **not done** — and I need to know whether it is mine or Ishan's |

Then the pivot, said plainly: *"What I did instead, between the 12th and the 27th, was
finish the loss-geometry thread. It changed the spatial-vs-temporal answer, so I think it
was the right trade — here is the case."*

**Why this slide earns its place:** it converts your two weakest points into two questions
*you* are asking *him*, at the moment you have the floor. Both genuinely need his answer
(see §0.2 C and D), so this is not a rhetorical device — it is the actual blocker.

---

### Slide 1 — Setup and the two metrics (no result yet)

**Purpose:** make sure the two axes used for the rest of the talk are unarguable
*before* any claim rides on them.

Define, on one slide:

- **Decoded peakiness** = max-probability of the decoded posterior, averaged over
  held-out trials. **The IO target** is the same statistic on the ideal-observer
  posterior for those trials (≈0.059 spatial, ≈0.062 temporal, Q/half/100 ms).
  Peakiness ÷ target = **over-sharpening** (a *bias*).
- **Normalised loss** = held-out KL(decoded ‖ target) ÷ the **leave-one-out
  predict-mean** decoder (predict the marginal mean posterior on every trial).
  **<1 beats chance.** This is meeting item #8, and the LOO version is post-audit.
- **Overfitting** = validation ÷ training fit-loss (a *variance* quantity), plotted
  separately and never conflated with peakiness.

Say out loud: *"Peakiness alone never decides whether a decoder is good — weight decay
is the counter-example, and it is on Slide 5."*

---

### Slide 2 — Diagnosis (4 panels) — `figures/prodfix/story_fig1_diagnosis.png`

Spatial decoder, 6 mice, mean±sem. Four panels, left to right:

| Panel | What is plotted | The number |
|---|---|---|
| **a** | decoded peakiness per loss, dotted line = IO target | projection **0.325** vs target **0.059** (5.5×); KL 0.060; JS 0.059 |
| **b** | normalised loss ÷ predict-mean, **same decoders scored under two metrics** (solid = projection metric, hatched = KL metric), dotted = chance | under its **own** metric the projection decoder looks fine (≈0.62, better than chance); under **KL** it is **≈2.4, i.e. worse than chance**. KL 0.94, JS 0.84 under KL. |
| **c** | peakiness, H=8 (solid) vs **zero hidden units** (hatched) | projection 0.325 → **0.301**; KL/JS both stay ≈0.06 |
| **d** | over-sharpening (peakiness ÷ target, y) vs overfitting (val ÷ train loss, x), log–log | projection: **5.5× over-sharpened at val/train ≈5.5**; KL: **1.0× at ≈25.6**; JS 1.0× at ≈10.0 |

**The claims, in order:** (a) the failure exists; (b) *the loss cannot see it* — this
is the crux, and it is why no amount of tuning that objective would have surfaced it;
(c) it is not capacity; (d) **bias and variance are anti-correlated across losses** —
KL overfits the most and over-sharpens the least, and vice versa. They are two
different objects, so "regularise harder" was always going to fail.

**Anticipated challenges**

- *"max-prob is a crude sharpness measure."* Agree — it is a scalar summary. Two
  backups: the full decoded-vs-target posterior galleries, and the per-PC shape-error
  decomposition, which localises the error in the shape subspace rather than location.
- *"You are scoring a decoder under a metric it was not trained for — of course it loses."*
  That is the point, and it is symmetric: panel b shows KL/JS decoders scored under
  **both** metrics and they are fine under both. Only the projection loss is
  metric-dependent.
- *"Is 0.059 the right target?"* It is the ideal-observer posterior for those trials —
  the same object the decoder is trained to reproduce. If the IO is wrong, the whole
  target is wrong, not just the peakiness.

---

### Slide 3 — Not capacity: a decoder with **zero hidden units** — `figures/prodfix/prodfix_fig1_nohidden.png`

**This closes meeting item #6, and it is the cleanest single experiment in the set.**

Multinomial logistic regression (no hidden layer at all) trained on the projection loss:

- over-sharpens **5.6× (spatial) / 10.5× (temporal)** the IO target, in **6/6 mice**;
- **statistically indistinguishable from the H=8 MLP** — 0.301 vs 0.325 spatial
  (p=0.071), 0.569 vs 0.643 temporal (p=0.247). Deleting the entire hidden layer moves
  peakiness by 7–11%;
- KL and JS with zero hidden units stay **on target** (1.02–1.14×, 0/6 mice >1.5×).

**Say:** *"There is no hidden layer left to overfit with. The over-sharpening is a
property of the objective, not of the model class."* This is the real-data confirmation
of what the softmax-Jacobian argument in `PCA-Peakiness-Mechanism` predicted.

**Anticipated challenges**

- *"A linear decoder can still overfit."* Yes — and it does, on the *fit-loss* (that is
  the variance axis). The claim is specifically that the *bias* survives, and it does,
  at the same magnitude, with capacity removed.
- *"p=0.071 is not 'indistinguishable'."* Correct — quote the effect size first: the
  difference is 7% of a 5.6× effect. The honest phrasing is "removing all capacity
  changes peakiness by <10%, while the over-sharpening is >450%".

---

### Slide 4 — Not target-noise either: the round-trip control — `figures/roundtrip_refit/peakiness_grid_spat.png`

**This is meeting item #5 (over-fitting tests in validation with fitted targets), and it
is the strongest logical move in the talk** — it separates two explanations that had
been conflated for months.

**Design (2×3):** three target sources × two refit losses. Targets = {real IO;
PCA-decoder output (already peaky); KL-decoder output (broad, calibrated)}, i.e. rows
2–3 are targets that are **achievable by construction** — a network produced them.
Read the decoded max-prob against the dashed target line over 200 epochs, 6 mice.

| Target | refit with projection loss | refit with KL |
|---|---|---|
| real IO | **0.351 vs 0.059 → 5.9×** | 0.062 vs 0.059 → 1.05× |
| PCA-fitted (peaky) | 0.217 vs 0.211 → **1.03×** | 0.209 vs 0.211 → 0.99× |
| KL-fitted (broad, achievable) | **0.243 vs 0.056 → 4.33×** | 0.055 vs 0.056 → 0.98× |

Temporal mirrors it (5.7 / 0.87 / 1.06 / 0.86 / 4.44 / 0.91).

**Two dissociated findings:**

1. On the *train–val curves* (companion figure `trainval_grid_spat.png`), **only the
   real-IO row shows the train↓/val↑ upturn**. On achievable fitted targets the
   validation loss plateaus — **the overfitting was an unachievable/noisy-target
   artefact**.
2. **`klFit_PCA` over-sharpens a clean, broad, achievable target 4.33× with essentially
   no train–val gap.** Over-sharpening therefore is **not** overfitting. It is loss
   mis-specification.

And `pcaFit_PCA = 1.03×` shows a **bounded attractor**, not unbounded drift: given an
already-sharp achievable target, the projection loss reproduces it faithfully.

**Say:** *"This retires my own earlier framing that peakiness tracked the train–test gap."*
Saying that before Máté does is worth more than the result.

**Anticipated challenges**

- *"The fitted targets come from your own decoders — circular?"* They are used only as
  *targets*, and the decisive cell (`klFit_PCA`) refits with a **different** loss than
  the one that generated the target. The point is achievability, not provenance.
- *"Different targets have different peakiness, so ratios aren't comparable."* Each cell
  is scored against **its own** target (the dashed line in each panel), which is why the
  ratio is the readout rather than the raw value.

---

### Slide 5 — The cure, and three fake cures — `figures/prodfix/story_fig2_cure.png`

**The methodological headline of the whole project.** Four panels: peakiness (top) and
normalised loss ÷ predict-mean (bottom) × spatial (left) / temporal (right), every
candidate intervention on a shared x-axis.

| Intervention | peakiness (spat/temp) | normalised loss (spat/temp) | verdict |
|---|---|---|---|
| none (baseline) | 0.325 / 0.64 | 2.4 / 8.6 | broken |
| no hidden layer | 0.301 / 0.567 | 2.7 / 6.2 | not capacity |
| dropout 0.9 | **0.062** / 0.469 | 1.14 / 5.0 | **lobotomy** |
| weight decay 0.01 | **0.011 = 1/91 = uniform** | 1.59 / 1.8 | **dead decoder** |
| smooth λ=0.3 | 0.069 / 0.094 | **1.11 / 1.44 (worse than chance)** | **fake cure** |
| smooth λ=1 | 0.055 / 0.068 | 0.99 / 1.08 | marginal |
| shape λ=0.3 (λ·Brier) | 0.057 / 0.058 | **0.78 / 0.75** | ✅ real cure |
| use KL | 0.060 / 0.050 | 0.94 / 0.56 | ✅ |
| use JS | 0.059 / 0.050 | 0.84 / 0.57 | ✅ |

- **shape λ=0.3 beats smooth λ=0.3 in 6/6 mice** (p=0.018 spatial, 0.022 temporal), and
  beats smooth λ=1 (p=0.037/0.033). **smooth λ=0.3 was the previously recommended
  operating point** — it lands peakiness on target and is *worse than chance*. It only
  got caught because both panels are now always plotted.
- **JS ≥ KL:** spatial normalised loss 0.84 vs 0.94, **6/6 mice, p=0.001**, with 2.6×
  less overfitting (val/train 10.0 vs 25.6); temporal a tie (0.57 vs 0.56, p=0.64, 3/6).
  Supports JS as the production loss.

**Volunteer the caveat before it is found:** at shape λ=0.3 the per-PC weights are
`evar + λ` over 91 PCs, so `sum = 1 + 91×0.3 = 28.3` — **the Brier term is ~96% of the
loss**. The "fix" at that strength is essentially unweighted Brier. The honest reading:
*the more the projection weighting is diluted, the better it gets* — which is an argument
**against the projection loss**, not for a tuned hybrid. (Note the weights are only
unnormalised in scale: `(evar+λ)/(1+nλ) ∝ (evar+λ)`, so relative weighting is unaffected.)

**Say:** *"Peakiness alone would have called weight decay a cure. It drives peakiness
below the target, to the uniform decoder, and the skill plateaus at 1.59 — it trades
over-sharpening for under-fitting. That is why every candidate is now judged twice."*

---

### Slide 6 — The consequence: spatial vs temporal **reverses with the loss** — `figures/prodfix/spat_temp_across_animals.png`

Two panels (Δ normalised loss; Δ peakiness), spatial − temporal, per manipulation,
paired t over mice, grey = n=6, red = **M2 excluded, n=5**.

| Loss | Δ (spatial − temporal) | p | mice |
|---|---|---|---|
| projection baseline | **−5.79 (spatial better)** | 0.028 | 6/6 |
| no hidden layer | −3.39 | 0.005 | 6/6 |
| **KL** | **+0.376 (temporal better)** | 0.002 | 6/6 |
| **JS** | +0.276 | 0.005 | 6/6 |
| shape λ=0.3 | +0.032 | 0.080 (n=6) / **0.015 (n=5)** | — |

**The claim:** *the apparent spatial (PPC) advantage is an artefact of the
over-sharpening.* The Jensen-averaged temporal architecture is punished hardest by a
loss that rewards sharpness, and **the ordering reverses once the loss is calibrated**.
The width fix (shape λ=0.3) sits with the calibrated losses.

This replicates and strengthens `PCA-Peakiness-Mechanism` §9, where the same claim rested
on Δ KL-skill +0.05–0.16 at p≈0.08. Here it is 6/6 mice at p=0.002 in a cleaner regime
(H=8, fixed restart rule, LOO null).

**Mouse 2 changes nothing** — every effect survives at n=5, several with smaller p.

**Anticipated challenges — this slide will take the most fire**

- *"n=6, 20 uncorrected tests."* State it first. The robust statement is the **6/6 sign
  consistency** (sign-test floor p=0.031), not any individual p.
- *"Effect sizes are tiny for KL (+0.376 vs −5.79)."* True, and the within-animal figure
  (Slide 7) shows exactly *how* different the two effects are. Do not oversell the
  temporal edge: the honest claim is **"modest, calibration-dependent temporal ≥ spatial",
  and "the spatial advantage is not real"** — the second is the strong half.
- *"weight decay shows Δ=0.000 with p=0.011."* Volunteer it as the cautionary case: both
  architectures have collapsed to the uniform decoder, so the difference is numerically
  tiny but sign-consistent. **An effect size without a p is uninformative here, and so is
  a p without the effect size.**

---

### Slide 7 — Within-animal: the two effects have different *character* — `figures/prodfix/spat_temp_within_animal_tests.png`

10 panels (one per manipulation), 6 bars each = per-animal Δ per-trial KL, paired on
trials, M2 in red.

- **Projection baseline: spatial better in 6/6 animals, and it is a UNIFORM shift** —
  dz −0.87 to −5.78, with **91–100% of individual trials** favouring spatial (p to 1e-205).
- **KL: temporal better in 6/6 animals, but SMALL and TAIL-DRIVEN** — dz only 0.20–0.37,
  and by trial-level sign it is near a coin flip (**45–58%** of trials favour spatial).
  Same for JS (dz 0.10–0.34).
- **weight decay 0.01 is the degenerate case:** Δ ≈ 1e-5 yet **dz up to 2.42, p = 1e-198**,
  because both architectures collapsed to uniform so the difference has almost no
  variance. **A standardised effect size can mislead exactly as badly as a p-value.**

**Say the stats caveat unprompted:** these within-animal p-values (n = 326–470 trials)
answer *"is this reliable within this animal"*. They are **not** population evidence —
pooling or averaging them across animals would be pseudoreplication. **The n=6 paired
test on Slide 6 is the generalisation claim.**

---

### Slide 8 — Why you should believe the numbers (methods credibility)

Do not skip this in front of Máté. Four things, one line each:

1. **Restart selection was biased and is now fixed.** `train_and_select_best_model`
   picked the restart with the lowest *training* loss — systematically the most overfit
   one, plausibly more so for richer objectives, i.e. a direct confound for the
   "KL overfits most" ordering. Now selects on **held-out validation** fit-loss.
   **Matched-seed control (identical restarts, only the winner differs):** the
   KL ≫ PCA overfitting ordering **survives** (KL spatial val/train 25.56 under both
   rules; PCA 5.47 vs 5.82 — a 4.6× gap against a ≤6% selection effect). Striking
   asymmetry: the rules disagree on **4/6 mice for PCA, 0–1/6 for KL**.
2. **Leave-one-out predict-mean null** — the old null was fit on the trials it scored
   (anti-conservative for "worse than chance"). Measured effect ~0.5%; **flips nothing**.
3. **A full four-way audit** (`documents/AUDIT_2026-07.md`) found and fixed my own bugs:
   a **fabricated IO-target line** drawn from a hardcoded constant when no data loaded;
   missing-data guards in all six sweep plotters (blank figure + a title still asserting
   the conclusion); ddof=0 SEM (error bars 9.5% too small at n=6); params/trial
   understated 1.25× because `monitor_val` removes 20% of training trials.
4. **The pipeline is leakage-clean where it counts;** the residual is that the
   early-stopping epoch is chosen in a PCA basis fit partly on its own holdout — stated,
   not hidden.

---

### Slide 9 — What is *not* established (own the limits)

- **Does V1 carry uncertainty beyond stimulus condition? Unestablished.** Parameter-matched
  mean / moments / DeepSets decoders, 4 losses, 6 mice, with a within-condition target
  shuffle: DeepSets beats moments on raw KL 6/6, but **real/null KL = 1.004 — no model
  beats its own within-condition null**, and width/entropy gains are null. The
  condition-mean oracle reaches KL ≈0.014 against 0.20–0.67 for the neural decoders.
  Synthetic validation (10 datasets) *does* recover pure variance codes, so the test has
  power. Conclusion: **no evidence for trial-specific uncertainty beyond condition** in
  unordered within-trial variability.
- **PPC vs SBC is not architecturally diagnostic** — the chance floor exceeds 1 for
  Monte-Carlo reasons. Slide 6 is a statement about *the loss*, not a vote for SBC.
- **Not run, from 18/06:** multiplicative Gaussian dropout (#3); the training-side per-PC
  eigenvalue-normalised loss as literally asked (#2).
- **Not run, from 08/07:** the per-trial inverse-noise weighting (C) and the encoding
  direction (D). Both are on Slide 10 as questions rather than apologies.
- **The encoding direction is genuinely missing from the project**, not just from this
  quarter — no fitted model in either tree runs posterior → activity. Worth saying
  out loud that the whole project is currently one-directional.
- **Mouse 2** "what's different" remains exploratory; every result is reported with it
  excluded and nothing changes.

---

### Slide 10 — Two questions and one request (close on these)

Do not close on "any questions". Close on the three things only he can unblock:

1. **"Per trial or per PC?"** — for the noise-variance ask. If per PC, do you mean
   inverting the `explained_variance` from the residual (within-condition) basis we
   already run as `prodfix_v1` arm B? **The two readings point in opposite directions**:
   inverse-*noise* weighting pushes further from the known cure, inverse-*eigenvalue*
   weighting pushes toward it. One sentence from him decides which experiment gets built.
2. **"Is the encoding model mine or Ishan's?"** — the 18/06 note assigns it to Ishan, the
   08/07 note re-raises it without reassigning, and the July audit assumed Ishan. If it is
   yours, the first deliverable is the condition-mean encoder *null*, not the encoder.
3. **"Which HMM did Ishan build, and on what grid?"** — "IO-HMM" collides with your own
   `ideal_observer/io_hmm/`. And whatever arrives has to be regridded from 181-bin 0–180°
   to the folded 91-point 0–90° before any divergence is computed. Ask him now, in the
   meeting, with Ishan in the room if he is there.

---

## Part 2 — Backup slides (have them, do not present them)

| Backup | Figure | Use when asked |
|---|---|---|
| B1 | `hpsweep_shuffle/peakiness_vs_hparams_v2.png` | "Did you try tuning?" — 6 axes, generic knobs fail on bias |
| B2 | `hpsweep_shuffle/overfit_vs_capacity_v2.png` | "Isn't this just capacity?" — ρ 0.67/0.69, p<1e-17, but it is the *variance* axis |
| B3 | `hpsweep_shuffle/shuffle_trainval_grid_*.png` | items #7/#8 — real vs shuffle train/val ÷ predict-mean with the chance line |
| B4 | `peakiness_scatter/subspace_error_realdata_evarweighted.png` | item #2 — the loss is blind: 38× raw shape error → ~2× and ~1% of the loss once evar-weighted |
| B5 | `roundtrip_refit/trainval_grid_spat.png` | "show me the actual val curves" for Slide 4 |

---

## Part 3 — Before the meeting

**Do now (under an hour, and item 1 is two minutes):**

1. **Email Ishan.** Three questions: which HMM did you build (input–output? the
   ideal-observer IO-HMM? the GLM-HMM?), on what orientation grid, and can you send the
   posteriors. This has been outstanding since May, and the underlying disparity since
   December — arriving at the meeting having *just* chased it is worth more than another
   result. **2 minutes.**
2. **Regenerate `fig4_behaviour_and_io`.** It is defined and called in
   `documents/vr_export_handoff/make_validation_figures.py` and advertised in the README,
   but only fig1–fig3 exist on disk — and fig4 is the one showing the IO posteriors, i.e.
   exactly what Ishan needs in order to validate against them. One command. **10 minutes.**
   Separately: the entire `vr_export_handoff/` is untracked *and* unignored, with no
   `PROJECT_LOG` mention — the deliverable actually sent to a collaborator is the
   least-recorded thing in the repo. Decide track-or-ignore.
3. **Rehearse three numbers cold:** 5.6×/10.5× (linear decoder), 4.33× with no train–val
   gap (`klFit_PCA`), 6/6 at p=0.002 (KL temporal).
4. **Decide the headline sentence**; put it verbatim on Slide 1 and again on Slide 6.
   Recommended: *"The projection-based loss's over-confidence is a property of the
   objective, not the network — and it inverts the spatial-vs-temporal conclusion."*

**Optional, if there is cluster time before the meeting (~half a day):** run multiplicative
Gaussian dropout and the literal eigenvalue-normalised training loss, so 18/06 items #2 and
#3 close as *run* rather than *argued*. Both are predicted to behave like the other generic
knobs; running them removes the only "you didn't do what I asked" opening on the June list.

**Do not start item C or D before the meeting.** Both need a one-sentence answer from Máté
first (Slide 10), and under the wrong reading each is a wasted session pointing the opposite
way. Also stale, worth fixing when convenient: `GOTCHAS.md:29` still claims all 401 configs
use `pca_basis='all_trials'` — there are now 419, two of them `residual`.
