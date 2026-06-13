# An RL test of the Similarity-Index framework — and a two-model comparison

**Does an actor-critic reinforcement learner, whose policy reads a similarity
index between V1 and two learned archetypes, converge on the same template
readout as a Hebbian network — and does it matter whether the recurrent V1 is
task-trained?**

*RNN+RL similarity-framework model — full methods and results. UncertaintyV1
project, June 2026. Sibling of [`si_network_model`](../si_network_model)
([REPORT](../si_network_model/REPORT.md)).*

---

## Summary

We built a second, independent generative test of the **Similarity-Index (SI)
framework** on the UncertaintyV1 orientation Go/NoGo task. It shares the
Hebbian model's forward path — grating → recurrent V1 population `r(t)` → cosine
similarity to two learned archetypes → similarity index `SI(t)` → drift-diffusion
choice — but replaces the learning rule with an **actor-critic reinforcement
learner** (policy gradient with a value baseline), and runs the recurrent V1 in
two modes: **fixed** (the framework's "V1 is stimulus-driven sensory cortex") and
**trained-end-to-end-then-frozen**. The only structural difference between the
two modes is whether the recurrent weights `W_rec` are learned.

A cohort of six simulated animals learned the task from reward alone in each
mode. The main findings:

1. **The RL agent lands on the template direction, not the whitened optimum.**
   The learned archetype-difference `w_Go − w_NoGo` aligns with the class-mean
   difference `Δμ` at cosine **0.88** (fixed) / **0.93** (trained), and with the
   covariance-whitened optimum `Σ⁻¹Δμ` at only **0.36** / **0.01**. This is the
   framework's central premise — and it emerges from reward, with no template
   ever supplied.
2. **The readout is near-optimal.** Decision efficiency (network ÷ neural ideal
   observer) is **0.99** (fixed) / **0.95** (trained); the agent's mean SI tracks
   the neural-IO log-posterior-odds at **r = 0.85** (fixed).
3. **It reproduces the framework's full readout-test signature.** Run through the
   project's RD-1/RD-2 estimators exactly as the real Cb15–Cb25 V1 is, the RL
   cohort gives whitening-adds-nothing-to-choice (RD-2 M3−M1 ≈ 0) and
   no-within-trial-SBC-signal (M2−M1 ≈ 0) in both V1 modes.
4. **Task-training the recurrence reproduces real V1's signature covariance.** A
   trained V1 develops covariance that whitening *can* exploit for the stimulus
   (RD-1 Δstim-AUC **+0.033**, matching real mice's +0.04) — yet the agent's
   *choices* still read the template (M3−M1 ≈ 0). The exact dissociation seen on
   data, generated mechanistically.
5. **The framework's signature is robust to the learning rule.** Two very
   different learners — three-factor Hebbian and actor-critic RL — converge on
   the same template readout at ~0.96–0.99 efficiency with `r ≈ 0.85`. The
   signature is a property of the *architecture*, not the plasticity rule.

---

## 1. The question

The SI framework (see `ResearchVault/Conjectures/Similarity Framework.md`) holds
that, for a two-class discrimination, cortex does not place a learned
covariance-optimal linear boundary. Instead it stores two *archetypes* — one per
category — and decides by comparing the current V1 population trajectory to each
via a cosine similarity, integrating the moment-by-moment similarity index

```
SI(t) = ( s(r(t), a_Go(t)) − s(r(t), a_NoGo(t)) )
      / ( s(r(t), a_Go(t)) + s(r(t), a_NoGo(t)) )      in [−1, +1]
```

The framework's sharpest, falsifiable commitment is geometric: the readout reads
the **template direction `Δμ`** (the difference of class means), *not* the
Bayes-optimal covariance-whitened direction `Σ⁻¹Δμ`. The Hebbian sibling model
showed that a three-factor Hebbian rule lands on that template and reaches ~96 %
of optimal-readout accuracy. But Hebbian plasticity *mechanically* forms a
prototype — the rule's fixed point is the reward-weighted mean V1 pattern, which
*is* the template. So the Hebbian result, while a genuine demonstration, is
arguably built into the learning rule.

This model asks the harder question: does a learner with **no built-in prototype
bias** — an actor-critic agent that only ever sees a scalar reward and adjusts
its policy to maximise it — *also* converge on the template? And, since the
framework's premise is "V1 is fixed, only the readout learns", what happens if we
break that assumption and let the recurrent V1 be task-trained first? If the
template readout survives both a different learning rule and a task-shaped V1,
the framework's signature is robust in a way no single model can show.

---

## 2. Methods

### 2.1 Architecture

A single differentiable pipeline, implemented as one PyTorch graph:

```
grating image  ->  recurrent V1 r(t)  ->  cosine similarity to 2 archetypes  ->  SI(t)  ->  Go/NoGo
 (CreateGratings   (fixed Gabor RFs +      (two learned weight                  (actor-critic
  recipe)           recurrence; orient.     vectors = the V1->action            policy + value;
                    jitter = uncertainty)   mapping)                            DDM at eval)
```

The fixed Gabor feed-forward front end, the stimulus recipe, the ideal observers
and the drift-diffusion accumulator are **reused verbatim** from
`si_network_model`; this model owns the recurrence back end, the cosine readout,
and the reinforcement learner. The single change made to the shared code is a
behaviour-preserving seam (`V1Model.jittered_drive`) that exposes the
orientation-jittered Gabor drive *before* the recurrent dynamics, so the
differentiable recurrence can consume it (the Hebbian model's 60 tests are
unaffected).

### 2.2 Stimuli and perceptual uncertainty

Identical to the Hebbian model. Gratings are reproduced from the experiment's
`CreateGratings.m` as a Gaussian-weighted mixture of sine-wave orientation
components (`dispersion` = sigma over components, `contrast` = luminance
excursion). The grid is 9 orientations `{0,15,30,40,45,50,60,75,90}°` × 9
`(contrast, dispersion)` pairs (81 conditions); boundary 45° (Go below); the
archetypes are 0°/90° at contrast 1, dispersion 5°.

**Perceptual uncertainty** is injected as *information-limiting orientation
jitter*: on each trial the encoded orientation is the true orientation plus
Gaussian noise whose sd grows with low contrast and high dispersion. Because the
jitter lies along the task-relevant dimension, averaging across neurons cannot
remove it — it sets both stimulus difficulty and bundle width, and is the neural
mirror of the ideal observer's sensory precision `κ(contrast, dispersion)`. Small
isotropic across-trial (bundle-width) and per-timestep (within-trial diffusion)
noise terms are also present.

### 2.3 Recurrent V1 — fixed vs trained

The V1 layer is the Hebbian model's: ~160 Gabor rate units with diverse
preferred orientation, spatial frequency, phase, envelope and position; about
half phase-invariant complex cells. The feed-forward drive is the rectified
Gabor filter response with per-trial orientation jitter. The recurrent dynamics
are a rate system over the 2 s trial (`dt = 0.08 s`, 25 steps) with a saturating
rectifier `φ(x) = sat·tanh(relu(x)/sat)`:

```
r(t+dt) = r(t) + (dt/τ) [ −r(t) + φ( drive + g · r(t) W_rec + noise ) ]
```

`W_rec` is initialised from the framework's hand-built like-to-like-excitation /
broad-inhibition matrix, scaled to a fixed spectral radius. The two V1 modes
differ only in `W_rec`:

* **fixed** — `W_rec` is frozen (a buffer). Framework-faithful: V1 is
  stimulus-driven, only the readout learns.
* **trained** — `W_rec` is a trainable parameter, optimised end-to-end by the
  same actor-critic reward (with a small weight decay to keep the recurrence
  tame), then frozen for evaluation. This tests whether a task-shaped recurrence
  changes the readout geometry.

The saturating `φ` guarantees bounded, stable dynamics regardless of the (now
possibly trained) recurrent weights — a unit test checks a large drive stays
within `±sat`.

### 2.4 Decision layer: the cosine readout

Two readout neurons — Go-driving and NoGo-driving — each carry a **static
archetype weight vector** `w_c ∈ ℝ^{n_v1}` (the framework's "one plastic weight
vector per response neuron"). Per timestep the readout current is the cosine
similarity `s(r(t), w_c) = r(t)·w_c / (‖r(t)‖‖w_c‖)`, positivity-mapped to `[0,1]`
and contrasted into the bounded `SI(t) ∈ [−1,+1]` (positive favours Go) — the
same form as the Hebbian model's `decision.si_trace` (verified bit-for-bit by a
unit test). The archetypes are renormalised to unit length after each update;
since cosine is scale-invariant, this only conditions the denominator and does
not change the policy.

The decision variable is the trial-mean SI; the policy logit is `z = β · mean_t
SI(t) + bias`, with `β` a learnable positive decision gain (softplus). At
**evaluation**, the SI(t) stream is additionally run through the framework's
**two-bound drift-diffusion accumulator** (reused from `si_network_model`) to
emit the reported choice and response time — so the RL policy and the framework
accumulator are reported consistently.

### 2.5 The actor-critic learner

Each trial is a single-step episode. The agent samples a Go/NoGo action from its
Bernoulli policy `π(Go) = σ(z)`, receives reward `R` (symmetric: +1 correct, −1
error), and updates by policy gradient with a learned value baseline:

```
advantage   A   = R − V(features)          (V detached for the actor)
policy loss     = − A · log π(action)
value loss      = ( R − V )²
entropy bonus   = − c_H · H[π]             (sustained early exploration)
loss            = policy_loss + c_V · value_loss + entropy_bonus
```

The **critic** `V` is a linear value head over the two mean similarities
`[mean_t s_Go, mean_t s_NoGo]` (detached, so the value objective updates only the
critic and does not corrupt the representation). What learns: the archetypes
`w_Go`, `w_NoGo`, the `bias`, the gain `β`, and the critic — always; and `W_rec`
only in the trained-V1 mode. The fixed Gabor front end never learns. Optimisation
is Adam with separate learning rates for the readout, the critic and (when
applicable) the recurrence; gradients are norm-clipped.

Exploration is emergent: archetypes are initialised small and random, so early
`SI(t) ≈ 0`, the policy is ≈ 50/50, and the agent acts near-randomly until reward
shapes the archetypes — the RL analogue of the Hebbian model's "random at start".

### 2.6 Cohort

Six simulated animals per V1 mode. Each animal has its own seed (Gabor mosaic,
initial weights, noise) and a draw of trait parameters — the Hebbian model's V1
traits (orientation-jitter gain, DDM diffusion, recurrent gain) plus RL traits
(readout learning rate, decision gain) — giving genuine between-animal spread in
bundle width, decisiveness and learning speed. Training is 6000 trials with a
flat Go/NoGo prior and sampling biased toward the archetypes; after training each
animal is evaluated frozen on a balanced 81-condition block.

### 2.7 Ideal observers and metrics

* **Neural-IO** — the readout-efficiency ceiling and the network's fair
  benchmark: the optimal Gaussian class-conditional decoder of the agent's *own*
  V1 activity. It is fitted on whichever recurrence the agent uses (an `AgentV1`
  adapter rolls out the agent's `W_rec`), so each V1 mode is benchmarked against
  the best possible readout of exactly its own code.
* **Stimulus-IO** — the behavioural ceiling: a Bayesian observer with noisy
  direct access to the latent orientation, precision `κ(c,d)` (reused from the
  project's v2 ideal observer).
* **Template vs whitened** — the cosine of the learned `w_Go − w_NoGo` against
  `Δμ` (class-mean difference) and against `Σ⁻¹Δμ` (shrunk pooled within-class
  covariance), both computed on the held-out evaluation block.
* **Decision efficiency** = network accuracy ÷ neural-IO accuracy; **evidence
  quality** = Pearson `r(mean SI, neural-IO log-odds)`.

### 2.8 The readout-test bridge (RD-1 / RD-2)

Each agent's frozen-evaluation block is packaged into the exact array bundle the
project's `nn_decoder/similarity_readout_tests.py` consumes (mirroring its
`network_control_arrays` adapter), so the RL cohort becomes a second,
learning-rule-independent positive control alongside the Hebbian one:

* **RD-1** — cross-validated AUC of a shrinkage-interpolated readout
  `w(λ) = [(1−λ)Σ + λσ̄²I]⁻¹Δμ` for `λ ∈ [0,1]` (λ=1 template, λ=0 whitened LDA),
  scored separately on stimulus and on choice. Framework prediction: whitening
  may buy *stimulus* accuracy but adds ≈0 to *choice*.
* **RD-2** — within-condition nested choice models (all conditioned on signed
  contrast): **M0** stimulus only; **M1** + trial-mean SI; **M2** + within-trial
  SI variance (the SBC wedge); **M3** + whitened-`Δμ` projection (the premise).
  Reported as held-out ΔLL in nats/trial.

---

## 3. Results

### 3.1 The actor-critic learns the task from reward

![Figure 1 — learning curves](figures/fig1_learning_curves.png)

*Figure 1.* Reward-driven acquisition (cohort mean ± SEM). From near-chance
(archetypes random → `SI ≈ 0` → ≈50/50 policy), accuracy and mean reward climb as
policy gradient shapes the archetypes. Both V1 modes learn; the trained-V1 mode
(red) reaches a more decisive policy (higher `|SI|`, §3.4) but not higher
accuracy — the fixed Gabor V1 is already near the neural-IO ceiling, so training
the recurrence mostly buys decisiveness, not accuracy.

### 3.2 The learned readout is the template direction, not the whitened optimum

![Figure 2 — template vs whitened](figures/fig2_template_vs_whitened.png)

*Figure 2 (headline).* For every animal, in both V1 modes, the learned
archetype-difference `w_Go − w_NoGo` aligns strongly with the class-mean
difference `Δμ` (coloured bars) and weakly with the covariance-whitened optimum
`Σ⁻¹Δμ` (grey). Cohort means: template alignment **0.88 ± 0.02** (fixed) /
**0.93 ± 0.02** (trained); whitened alignment **0.36 ± 0.01** / **0.01 ± 0.01**.
The agent is given only scalar reward — never a template — yet policy gradient
converges its readout onto the template direction. This is the framework's
central premise, reproduced by a learner with no prototype bias.

### 3.3 The readout is near-optimal and its evidence is a log-posterior-odds

![Figure 3 — efficiency and SI vs optimal log-odds](figures/fig3_efficiency_and_si.png)

*Figure 3.* **Left:** per-animal network accuracy vs the optimal readout of the
same V1; points sit on or just below the identity line. Cohort decision
efficiency **0.987 ± 0.005** (fixed) / **0.954 ± 0.016** (trained). **Right:** the
agent's trial-mean SI against the neural-IO log-posterior-odds (example animal);
they correlate at **r = 0.85** (fixed) / **0.68** (trained), supporting the
framework's reading of `SI(t)` as a moment-by-moment log-likelihood ratio. The
trained-V1 relation is tighter-but-more-saturated (the trained recurrence makes
the representation more separable, so SI saturates earlier — high decisiveness,
slightly lower linear `r`).

### 3.4 Fixed vs trained-then-frozen V1

![Figure 5 — fixed vs trained summary](figures/fig5_fixed_vs_trained.png)

*Figure 5.* Cohort summary (mean ± SEM). Training the recurrence does **not**
raise accuracy or efficiency — the Gabor V1 is already near ceiling — but it
sharply raises **decisiveness** (mean `|SI|` 0.19 → 0.74) and tightens **template
alignment** (0.88 → 0.93). The interesting consequence is in the covariance
structure the trained V1 develops, which the readout tests expose next (§3.6).

### 3.5 Psychometric curves track the ideal-observer ceiling

![Figure 4 — psychometric](figures/fig4_psychometric.png)

*Figure 4.* P(Go) vs orientation, both V1 modes, against the stimulus-IO ceiling
(dashed). The agents produce the expected monotone psychometric — steep across
the 45° boundary, saturating at the easy ends — shadowing the optimal observer.

### 3.6 The readout-test signature — reproduced, and a real-V1 dissociation

![Figure RD-2 (fixed V1)](figures/rd2_nested_choice__fixed.png)

*Figure RD-2 (fixed V1).* Within-condition nested choice models, per animal
(bars) + cohort mean (thick line). **M1−M0 = +0.36 nats** (mean SI carries
choice beyond the stimulus) — large and consistent across all six animals;
**M2−M1 ≈ 0** (within-trial SI variance adds nothing — no SBC posterior-width
signal); **M3−M1 ≈ 0** (the whitened direction adds nothing beyond the template —
the premise). *Caveat:* M1−M0 is large **by construction** — the agent's policy
*is* `σ(β·mean SI)`, so mean SI predicts choice almost perfectly. It only
confirms the test fires (a positive control), exactly as the Hebbian cohort does;
it is **not** framework evidence. The decisive, non-circular cells are M3−M1 and
M2−M1.

![Figure RD-1 (fixed V1)](figures/rd1_template_vs_whitened__fixed.png)
![Figure RD-1 (trained V1)](figures/rd1_template_vs_whitened__trained.png)

*Figure RD-1 (fixed vs trained V1).* Cross-validated AUC vs shrinkage λ (λ=1
template, λ=0 whitened). **Fixed V1:** whitening barely changes stimulus AUC
(Δstim-AUC = −0.008) — the orientation-jitter noise is information-limiting, so a
whitened LDA cannot beat the template, exactly as in the Hebbian control. **Trained
V1:** whitening now *does* buy stimulus AUC (**Δstim-AUC = +0.033**), because the
task-trained recurrence develops exploitable covariance — and this matches the
real Cb15–Cb25 mice (+0.04). In **both** modes whitening *hurts* choice decoding
(Δchoice-AUC −0.034 / −0.054).

The crucial cell is the conjunction in the trained-V1 mode: **exploitable
covariance for the stimulus (RD-1 Δstim +0.033) coexisting with a template choice
readout (RD-2 M3−M1 = +0.004 ≈ 0)**. That is precisely the dissociation seen on
real V1 — covariance the brain *could* use for the stimulus but the animals'
*choices* ignore — and here it is generated mechanistically by task-training the
recurrence. (Real mice show a small within-trial signal in one animal, Cb17;
intriguingly the trained-V1 cohort shows a faint M2−M1 = +0.027, the only place
either model produces a non-zero temporal term.)

---

## 4. The two SI models compared

This model and `si_network_model` are deliberately matched: same task, same
fixed Gabor front end, same noise model, same ideal observers, same DDM, same
cosine SI readout, same RD-1/RD-2 readout tests. They differ in exactly one
component — the learner — plus this model's optional trained-V1 mode.

| | `si_network_model` | `rnn_rl_model` (this report) |
|---|---|---|
| Learning rule | three-factor reward-modulated Hebbian (`Δw ∝ pre·post·R`) | **actor-critic policy gradient** (`−A·∇log π`, value baseline) |
| Built-in prototype bias | **yes** — the rule's fixed point *is* the prototype | **no** — only a scalar reward signal |
| Recurrent V1 | fixed | **fixed _or_ trained-then-frozen** |
| Archetype | time-resolved trajectory `a_c(t)` | static weight vector `w_c` |
| Cohort | 10 animals × 2 reward modes | 6 animals × 2 V1 modes |

**Where they agree (the robust core):**

| Metric | Hebbian | RL (fixed V1) |
|---|---|---|
| Decision efficiency (network ÷ neural-IO) | 0.961 | **0.987** |
| Evidence quality `r(SI, IO log-odds)` | 0.84 | **0.85** |
| Reads the template (RD-2 M3−M1) | ≈ 0 | ≈ 0 |
| No SBC within-trial signal (RD-2 M2−M1) | ≈ 0 | ≈ 0 |
| Whitening on fixed V1 (RD-1 Δstim) | ≈ −0.02 | −0.008 |

Two learners that share *nothing* in their update rule — one with the prototype
baked into its fixed point, one that only ever sees scalar reward — converge on
the **same** near-optimal template readout. The framework's signature is
therefore a property of the **architecture** (cosine readout of a fixed V1, into
an accumulator), not of the plasticity rule. The Hebbian result alone could be
dismissed as "of course a Hebbian rule forms a prototype"; the RL result removes
that objection.

**What the RL model adds beyond the Hebbian one:**

1. **Learning-rule independence** of the template premise (above).
2. **A handle on V1's covariance.** The trained-V1 mode shows that whether RD-1's
   whitening buys *stimulus* AUC is controlled by how task-shaped the recurrence
   is: a hand-built like-to-like V1 has information-limiting noise (Δstim ≈ 0,
   like the Hebbian control), while a task-trained V1 develops the
   exploitable-but-choice-ignored covariance seen in real mice (Δstim ≈ +0.03).
   This identifies *recurrent learning in V1* as a candidate source of the
   real-data RD-1 effect — a prediction neither the Hebbian model nor the data
   alone could make.

---

## 5. Discussion

**The headline.** An actor-critic agent whose only learning signal is scalar
reward, and whose policy reads a cosine similarity to two archetypes, learns the
orientation Go/NoGo task and converges on the **template direction `Δμ`** — not
the covariance-whitened optimum — at ~0.95–0.99 of optimal-readout accuracy, with
its accumulated SI tracking the optimal log-posterior-odds at `r ≈ 0.85`. Because
this learner has no built-in prototype bias, the result is stronger evidence than
the Hebbian model that the framework's premise is an architectural consequence,
not a plasticity artefact.

**The trained-V1 result is the new science.** Relaxing the framework's "V1 is
fixed" assumption, and letting reward shape the recurrence, leaves the *choice*
readout template-aligned while making V1's covariance *stimulus*-exploitable —
reproducing, from first principles, the otherwise puzzling real-V1 dissociation
(whitening buys stimulus-AUC the animals' choices ignore). It suggests the
real-data RD-1 effect reflects task-driven recurrent structure in V1, with the
downstream readout still template-based.

**Honest accounting.** RD-2 M1−M0 is tautological here (SI *is* the policy
input), so it is reported only as a fire-check, never as evidence — the claims
rest on M3−M1 and M2−M1, which are not circular. Decision efficiency occasionally
sits at or just above 1 because the neural-IO's equal-covariance Gaussian is not
a strict ceiling on a near-separable trained V1.

---

## 6. Limitations

- **Rate-based, not spiking;** V1 noise is a phenomenological orientation-jitter
  model calibrated for a realistic difficulty gradient.
- **Go/NoGo only.** Lick-left/right 2AFC is a documented extension, not built.
- **Single-step RL.** Each trial is one episode; there is no within-trial credit
  assignment or temporal-difference bootstrapping beyond the value baseline.
- **The policy reads trial-mean SI** for the gradient (the differentiable
  surrogate for the DDM's first-passage choice probability); the DDM is applied
  only at evaluation for choice/RT. SI(t)'s temporal structure is therefore not
  load-bearing for learning.
- **Generative model + IO comparison, not a fit** to the real Cb15–Cb25 data.
- **Flat prior;** block-prior / volatility manipulations are not modelled.

---

## 7. Reproducing

```
python -m rnn_rl_model.run            # full cohort (n=6, both V1 modes)
python -m rnn_rl_model.run --quick    # fast smoke run (n=3)
python -m pytest rnn_rl_model/tests/ -q   # 10 ground-truth tests
```

Code is in `rnn_rl_model/` (see `README.md` for the quick reference and
`UNDERSTANDING.md` for the design decisions). Figures are written to `figures/`
(PNG+SVG); the cohort pickle and `summary.json` to `results/` (gitignored,
regenerable). The Hebbian sibling and its report are in `si_network_model/`.
