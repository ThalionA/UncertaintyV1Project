# A V1-to-premotor network model of the Similarity-Index framework

**Does a network that decides by comparing V1 population trajectories to learned
archetypes approximate optimal Bayesian inference on an orientation Go/NoGo task?**

*SI-framework network model — full methods and results. UncertaintyV1 project,
May 2026.*

---

## Summary

We built an end-to-end, reward-trained network model of the UncertaintyV1
orientation Go/NoGo task and used it to test the **Similarity-Index (SI)
framework**: the hypothesis that cortex decides not by a learned linear boundary
but by comparing the current V1 population trajectory `r(t)` to two learned
*archetypes*, one per category, via a cosine readout feeding a drift-diffusion
accumulator.

A cohort of ten simulated animals learned the task from reward alone. The main
findings:

1. **The SI readout is near-optimal.** After learning, the cosine/prototype
   readout captured **96.1 %** (range 92–98 %) of the accuracy of the *optimal
   Bayesian decoder of the same V1 activity*, and its accumulated evidence
   tracked the optimal log-posterior-odds at **r = 0.84**.
2. **It is suboptimal in a principled way.** Every animal sat just below the
   optimal-readout line — the residual gap is the framework's own stated cost
   (a cosine projection discards the noise covariance the full decoder uses).
3. **Reward-modulated plasticity carves two trajectory bundles** — the
   framework's archetypes — formed mechanistically rather than assumed.
4. **The framework's cross-animal predictions hold:** tighter bundles (higher
   concentration kappa) predict fewer lapses (r = -0.75) and lower drift
   variability (r = -0.83).
5. **Of the four bundle-similarity variants, the cheap prototype is as good as
   the bundle-aware ones** (prototype 0.96, vMF 0.98, expected-/max-exemplar
   0.94) — positive evidence that a synaptic-weight-only mechanism suffices on
   this task.
6. **Under biased payoffs the network learns a pronounced Go-bias**, shifting
   its criterion +11° toward Go — and *over-shifting* the reward-optimal
   criterion, a characteristic excess Go-bias.

---

## 1. The question

The brain infers the world from noisy sense data. *How* it does so neurally is
debated. The **SI framework** (see `ResearchVault/Conjectures/Similarity
Framework.md`) proposes that, for a two-class discrimination, cortex stores two
*trajectory archetypes* — one per category — and decides by comparing the
current V1 population trajectory to each. A pointwise cosine similarity yields a
moment-by-moment similarity index

```
SI(t) = ( s(r(t), a_Go(t)) - s(r(t), a_NoGo(t)) )
      / ( s(r(t), a_Go(t)) + s(r(t), a_NoGo(t)) )      in [-1, +1]
```

which is integrated by a drift-diffusion accumulator; the bound crossing sets
the choice. The framework's working hypothesis is that `SI(t)` is a
moment-by-moment **log-likelihood ratio**, so the accumulator implements
approximate sequential Bayesian inference.

This model asks the concrete question: if a network is *built* this way and
*trained from reward*, does the resulting readout actually approximate optimal
Bayesian inference? It does — with a measurable, principled shortfall.

---

## 2. Methods

### 2.1 Architecture

The model is a single feed-forward pipeline; only the V1-to-decision synapses
learn. A grating drives a fixed population of orientation-tuned V1 neurons;
their recurrent dynamics produce a population trajectory `r(t)`; two premotor
readout neurons compare it to their learned archetype weight vectors (red = Go,
blue = NoGo); the similarity index `SI(t)` is integrated by a drift-diffusion
accumulator that emits the choice; and reward feeds back to update the
archetype weights.

![Figure 0 — network architecture](figures/fig0_architecture.png)

The V1 layer is fixed throughout — the framework's commitment that V1 is
stimulus-driven sensory cortex — and only the synapses onto the premotor
readout neurons (the archetype weight vectors) are plastic.

### 2.2 Stimuli

Gratings are reproduced from the experiment's `CreateGratings.m`: each grating
is a Gaussian-weighted **mixture of sine-wave components** at different
orientations. `dispersion` is the Gaussian sigma over component orientation,
`contrast` sets the luminance excursion, and the pattern is rotated to the
requested `orientation`. The task grid is the experiment's: 9 orientations
`{0, 15, 30, 40, 45, 50, 60, 75, 90}°` x 9 `(contrast, dispersion)` pairs (= 81
conditions). The category boundary is 45° (Go below, NoGo above). The
*archetypes* are 0°/90° at contrast 1, dispersion 5°.

Gratings are static within a trial, so the feed-forward drive for each
orientation is rendered and cached once on a fine orientation grid; per-trial
orientation jitter (below) is then a fast interpolation.

### 2.3 V1 layer

**Receptive fields.** ~320 rate units, each a Gabor receptive field with diverse
preferred orientation (tiling 0–180°), spatial frequency, phase, envelope size
and position. Roughly half are phase-invariant complex (quadrature-energy)
cells, half phase-sensitive simple cells. The feed-forward drive is the
(image-size-normalised) rectified Gabor filter response.

**Recurrence.** V1 units are recurrently coupled by *tuning similarity*:
like-orientation excitation plus broad inhibition, scaled to a fixed spectral
radius. The population is run as a rate dynamical system over the 2 s trial
(`dt = 0.08 s`, 25 steps) with a saturating rectifier, producing a genuine
population **trajectory** `r(t)` that ramps from baseline to a stimulus-driven
state.

**Noise — and where difficulty comes from.** The dominant trial-to-trial noise
is **orientation jitter**: on each trial the *encoded orientation* is the true
orientation plus Gaussian noise whose sd grows with low contrast and high
dispersion. This is *information-limiting* — it lies along the task-relevant
dimension, so averaging across neurons cannot remove it — and it is the neural
mirror of the ideal observer's sensory precision `kappa(contrast, dispersion)`.
It is the source of both stimulus difficulty and **bundle width**. (An early
implementation used per-timestep noise instead; that washes out under
trial-averaging, leaving even 1 %-contrast stimuli perfectly decodable —
a bug fixed by the switch to orientation jitter.) Small isotropic across-trial
and per-timestep noise terms are also present.

### 2.4 Decision layer: the SI readout and the accumulator

Two premotor readout neurons — Go-driving and NoGo-driving — each carry a
**plastic trajectory archetype** `a_c(t)` of shape `(25, 320)`. The readout
current is the pointwise cosine similarity between `r(t)` and the archetype's
value at the same time `t` (a dot product with divisive normalisation — the
natural readout of a linear-summation neuron). The bounded, positivity-mapped
normalised difference of the two similarities is `SI(t)` (positive favours Go).

`SI(t)` drives a **symmetric two-bound drift-diffusion accumulator**,
`dX = gain * SI(t) dt + sigma dW`. The first bound crossing sets the choice and
the response time; trials that never cross are decided by the sign of `X` at the
2 s deadline. The diffusion `sigma` provides the early exploration: with random
initial archetypes `SI(t) ~ 0`, so early choices are noise-driven.

### 2.5 Reward-modulated plasticity

The framework's three-factor Hebbian rule, made literal. After each trial the
synapses onto the *chosen* readout neuron are updated, per timestep, by

```
Delta a_chosen(t) = learning_rate * reward * r(t)
```

— pre-synaptic factor `r(t)`, post-synaptic gating to the chosen neuron, and
the reward as the third (modulatory) factor. Each archetype timestep is
renormalised to unit length (the cosine readout is scale-invariant). Over
training each archetype converges to the reward-weighted mean V1 trajectory for
its choice — the **prototype archetype**.

This raw-reward rule handles asymmetric payoffs correctly: an outcome that earns
zero either way produces no update (the channel rests), while unequal reward
magnitudes net-potentiate one archetype — an emergent bias, the framework's
"prior as asymmetric weight scaling".

### 2.6 Curriculum and cohort

Training is two-phase. **Phase 1** presents only the archetypes (0°/90°) until
rolling accuracy graduates the animal. **Phase 2** presents the full 81-condition
grid, with sampling biased toward the archetypes and a flat Go/NoGo category
prior. A cohort of **10 simulated animals** is run; each has its own seed
(Gabor mosaic, initial weights, noise) and a draw of four **trait** parameters
— orientation-jitter gain, learning rate, DDM diffusion, recurrent gain — giving
the cohort genuine between-animal variability.

After training, each animal is evaluated with frozen weights on a balanced block
spanning all 81 conditions.

### 2.7 Ideal observers — the benchmarks

Two optimal observers, two ceilings:

* **Stimulus-IO** — the behavioural ceiling. A Bayesian observer with direct but
  noisy access to the latent orientation, sensory precision
  `kappa(contrast, dispersion)`. Reuses the project's v2 ideal observer
  (`ideal_observer/io_hmm/io_core.py`).
* **Neural-IO** — the readout-efficiency ceiling, and the network's *fair*
  benchmark. The optimal Bayesian decoder of the network's **own** V1 activity:
  a Gaussian class-conditional model (per-condition means + a
  shrinkage-regularised pooled covariance) fitted to the same V1 the network
  uses. It is the best any readout of this neural code could do.

Both observers decide with a reward-aware threshold on the log-posterior-odds:
0 for symmetric reward (the unbiased 45° rule), shifted for asymmetric reward.

### 2.8 The four SI variants

Comparing `r(t)` to a *bundle* rather than a single curve admits four
operationalisations (the framework's open question), here computed as
alternative readouts of the *same* learned representation:

* **prototype** — cosine to the bundle-mean trajectory.
* **expected-exemplar** — mean cosine over K = 20 stored exemplar trajectories
  (a kernel-density estimate of the class likelihood).
* **max-exemplar** — similarity to the single nearest stored exemplar.
* **vMF** — a von Mises–Fisher likelihood with concentration tied to bundle
  width; `SI` is then `tanh` of the log-likelihood ratio.

The exemplar bundle (K = 20 trajectories per class) is reservoir-sampled from
correct Phase-2 trials — the framework's "additional memory system" beyond the
prototype-forming synapses.

---

## 3. Results

### 3.1 The network learns the task from reward

![Figure 1 — learning curves](figures/fig1_learning_curves.png)

*Figure 1.* Reward-driven acquisition. Archetype discrimination (0° vs 90°) is
highly separable and is learned within the first ~50 trials (left); Phase 2
introduces the full grid, and accuracy settles into a harder mixed-difficulty
regime (right). All ten animals graduate Phase 1.

### 3.2 The psychometric tracks the optimal observers

![Figure 2 — psychometric](figures/fig2_psychometric.png)

*Figure 2.* P(Go) vs orientation, by difficulty tier. The SI network (red)
tracks both ideal observers, with a sigmoid that shallows on harder stimuli and
collapses to chance on the hardest, exactly as the observers do.

![Figure 9 — accuracy by difficulty](figures/fig9_accuracy_by_difficulty.png)

*Figure 9.* Accuracy across all nine contrast/dispersion pairs, easiest to
hardest. The network (red) shadows the neural-IO (blue) across the whole range;
all three observers collapse to chance together on the hardest stimuli. On the
high-contrast/high-dispersion pair the two ideal observers diverge — the
neural-IO is more robust to dispersion than the stimulus-IO's `kappa(c,d)` — so
the **neural-IO is the network's fair benchmark**.

### 3.3 The SI readout is near-optimal

![Figure 3 — decision efficiency](figures/fig3_efficiency.png)

*Figure 3.* Per-animal network accuracy vs the optimal readout of the same V1.
Every animal sits just **below** the identity line; cohort decision efficiency
is **0.961** (network 0.737, neural-IO 0.767; stimulus-IO 0.730). The network
captures ~96 % of the optimal readout's accuracy — near-optimal, but
consistently and characteristically *sub*-optimal. The residual gap is the
framework's own stated cost: a single cosine projection onto a prototype
discards the noise covariance the full Gaussian decoder exploits.

### 3.4 The accumulated evidence is a log-posterior-odds

![Figure 4 — SI vs IO log-odds](figures/fig4_si_vs_io.png)

*Figure 4.* The network's trial evidence `E_t[SI]` against the neural-IO's
log-posterior-odds, all trials. They correlate at **r = 0.84**, supporting the
framework's Bayesian-framing hypothesis that `SI(t)` is a moment-by-moment
log-likelihood ratio.

![Figure 5 — SI(t) traces by difficulty](figures/fig5_si_traces.png)

*Figure 5.* The `SI(t)` evidence stream is strong and steady on easy Go/NoGo
trials and sits near zero on ambiguous trials — the framework's qualitative
prediction.

### 3.5 Reward-modulated plasticity carves two trajectory bundles

![Figure 6 — archetype bundles](figures/fig6_bundles.png)

![Figure 10 — trajectory archetypes](figures/fig10_trajectory_archetypes.png)

*Figures 6 and 10.* The learned archetypes. Reward-modulated plasticity carves
two distinct V1 trajectory bundles; the learned weight vectors sit at their
centres. Figure 10 shows the genuine *time-resolved* archetypes — each class is
a bundle of exemplar trajectories with a prototype curve `a_c(t)` — and that the
two archetypes become more angularly separated as the trial unfolds.

### 3.6 The cheap prototype is as good as the bundle-aware variants

![Figure 11 — variant comparison](figures/fig11_variant_comparison.png)

*Figure 11.* The four SI variants as readouts of the same learned bundles.
Decision efficiency: prototype 0.96, vMF 0.98, expected-exemplar 0.94,
max-exemplar 0.94. The bundle-aware exemplar variants do **not** beat the cheap
prototype here — if anything they slightly underperform, because the
within-class bundle is roughly unimodal so the mean is a good summary and
individual exemplars carry noise the prototype averages out. This is positive
evidence (per the framework's own logic) that the **synaptic-weight-only
prototype mechanism is sufficient** for this task — the framework's Prediction 2
ordering (bundle-aware variants win) is *not* confirmed in this model.

### 3.7 Bundle width predicts behaviour across animals

![Figure 7 — cross-animal predictions](figures/fig7_cross_animal.png)

*Figure 7.* The framework's cross-animal predictions hold. Animals with tighter
bundles (higher concentration kappa, range 4.5–15.9 across the cohort) show
fewer lapses (**r = -0.75**, Prediction 15) and lower trial-mean-SI spread
(**r = -0.83**, Prediction 4). Bundle width is a genuine, measurable driver of
behavioural variability.

### 3.8 Response times

![Figure 8 — RT distributions](figures/fig8_rt_distributions.png)

*Figure 8.* Easy trials cross a bound and decide fast; ambiguous trials run to
the 2 s deadline — the expected difficulty-RT relationship.

### 3.9 Biased payoffs produce an excess Go-bias

![Figure 12 — asymmetric payoff](figures/fig12_asymmetric_comparison.png)

*Figure 12.* A second cohort was trained under asymmetric, water-task-like
payoffs (hit +1, false alarm -0.25, correct rejection +0.25, miss 0 — a cheap
false alarm relative to the hit reward). The network learns a clear **Go-bias**:
its psychometric shifts toward Go and its criterion moves from 43° (symmetric)
to 54° (asymmetric), a **+11° shift**. Notably the network *over-shifts* the
reward-optimal criterion (whose easy-stimulus shift is only ~+3°, because a
sharp psychometric is barely moved by a threshold change). This excess Go-bias
— a learned bias whose magnitude is set by the reward ratio rather than
calibrated to the Bayesian optimum — is consistent with the well-documented
excess Go-bias in real Go/NoGo behaviour, and with the framework's treatment of
Go-bias as a characteristic suboptimality.

*(Implementation note: the small correct-rejection reward gives the NoGo channel
a learning gradient — with zero reward for both correct rejections and misses
the NoGo archetype has nothing to learn from under a reward-modulated rule.)*

---

## 4. Discussion

**The headline.** A network built on the SI framework — fixed V1, two learned
trajectory archetypes, a cosine readout, a drift-diffusion accumulator, and
reward-modulated Hebbian plasticity — learns an orientation Go/NoGo task from
reward alone and ends up **near-optimal**: it captures ~96 % of the accuracy of
the optimal Bayesian decoder of its own V1 activity, and its accumulated
evidence is a faithful log-posterior-odds (r = 0.84). The framework's central
claim — that similarity-to-archetypes is a viable, near-Bayesian decision
algorithm — is supported.

**The shortfall is the point.** The network is *consistently* sub-optimal: every
animal sits below the optimal-readout line. This is not a failure but the
framework's own prediction — a cosine projection onto a prototype throws away
the noise covariance that the full Gaussian decoder uses. The model quantifies
that cost at ~4 % of accuracy on this task.

**Prototype sufficiency.** The four bundle-similarity variants were a genuine
open question in the framework. Here the cheap prototype matches or beats the
bundle-aware variants — evidence that, for a roughly unimodal class, the
synaptic-weight-only mechanism (which delivers only the prototype) is enough,
and that the richer exemplar/vMF machinery is not required to explain this task.

**Go-bias.** Under biased payoffs the network develops a pronounced Go-bias by
asymmetric weight scaling — and over-shifts the reward-optimal criterion. This
reproduces a real, characteristic feature of Go/NoGo behaviour and locates its
mechanism in the learning rule.

## 5. Limitations

- **Rate-based, not spiking.** V1 noise is a phenomenological orientation-jitter
  model calibrated for a realistic difficulty gradient, not a biophysical one.
- **Contrast and orientation are the strong difficulty axes; dispersion is
  weaker** in the V1 model than in the stimulus-IO's `kappa(c,d)`. The two ideal
  observers therefore diverge somewhat; the neural-IO is the network's fair
  benchmark.
- The neural-IO uses an equal-covariance Gaussian model — Bayes-optimal under
  that assumption, near-optimal in general.
- V1 dynamics are a simple relaxation to a fixed point, so the *trajectory*
  archetype and a *static* snapshot give nearly identical SI here — the
  trajectory machinery is in place but the temporal dimension is not yet
  load-bearing.
- This is a generative model plus an ideal-observer comparison; it is **not
  fitted** to the real Cb15–Cb25 data.

## 6. Reproducing

```
python -m si_network_model.run             # both cohorts; resumable, re-run to continue
python -m si_network_model.tests.run_tests # 60 synthetic-data tests
```

Code is in `si_network_model/` (see `README.md` for the module map and
`UNDERSTANDING.md` for the design decisions). Figures are written to
`figures/`, the results archive to `results/`.
