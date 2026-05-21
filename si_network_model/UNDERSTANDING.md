# UNDERSTANDING — SI-framework network model

*Current understanding at top. Edit log appended below.*

---

## Goal

Build and run an end-to-end network model — visual grating input → V1 population → higher
association/premotor accumulator → Go/NoGo choice — that learns by reward-modulated
plasticity, and test whether the **Similarity Index (SI) framework** readout it implements
can approximate **optimal Bayesian inference** on the UncertaintyV1 orientation Go/NoGo task.

Lives in `UncertaintyV1/si_network_model/`.

## Task being modelled

- Orientation Go/NoGo. Stimulus = oriented grating, parameterised by **(orientation,
  contrast, dispersion)**, matching `experiment_code/CreateGratings.m`.
- Orientation grid `[0,15,30,40,45,50,60,75,90]°`; decision boundary at **45°**
  (`< 45` → Go, `> 45` → NoGo, `== 45` → 50/50).
- 9 (contrast, dispersion) pairs from `NewExperimentTheo25_full2_v3.m`:
  `(1,5) (1,30) (1,90) (0.5,5) (0.5,45) (0.25,30) (0.25,90) (0.01,5) (0.01,45)`.
- **Archetypes** = `{0°, 90°}` at `(contrast 1, dispersion 5)`.
- Curriculum: archetypes only → then full orientation × (contrast,dispersion) grid.

## Resolved design decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Stimulus → V1 | **Render pixels + Gabor RFs.** Reproduce the `CreateGratings.m` mixed-orientation recipe as images; V1 drive = Gabor receptive-field response. Static grating per trial → render+cache the ~81 unique conditions once. |
| 2 | Scale | **Cohort of 10 simulated animals** (config). Fewer trials/animal (~4500: Phase 1 cap ~1200, Phase 2 ~3300) — archetype-biased sampling makes learning fast enough that fewer trials suffice. |
| 3 | Ideal observer | **Both.** Stimulus-IO (reuse `ideal_observer/io_hmm/io_core.py`, Bayesian over orientation, precision `kappa(c,d)`) = behavioural ceiling. Neural-IO = optimal Bayesian decoder of the network's own V1 activity `r(t)` = readout-efficiency ceiling. |
| 4 | Learning rule | **Three-factor reward-modulated Hebbian with RPE.** `Δw ∝ pre(V1 r) · post(readout) · (reward − expected)`. Correct → potentiate active-V1→chosen-neuron synapses; error → depress. Converges each weight vector to the reward-weighted mean V1 pattern = the **prototype** archetype. |
| 5 | Animal variation | **Seed + sampled traits.** Per-animal seed AND trait params drawn from a distribution: sensory-noise level, learning rate, exploration/diffusion temperature, recurrent gain. Gives spread in bundle width & lapse for cross-animal tests. |
| 6 | Decision readout | **Symmetric two-bound internal DDM.** `dX = SI(t)dt + σdW`, bounds ±B. Go-bound crossing → lick at that time (gives lick latency / RT); NoGo-bound crossing → withhold; no crossing by the 2 s deadline → sign of `X` at deadline. |
| 7 | Reward structure | **Both modes, switchable.** *Symmetric* (default): correct +1 / error −1 → reward-max policy is the unbiased 45° boundary → `io_core.py` is exactly optimal. *Asymmetric* (config): hit/FA/CR/miss water-task-like → reproduces Go-bias; IO extended to a reward-weighted Bayesian decision rule for a fair comparison. |

## Architecture (settled — framework-prescribed, not open)

- **V1 layer.** ~320 rate-based neurons (config), FIXED weights throughout. Diverse Gabor
  RFs: preferred orientation tiling [0,180°), diverse SF (around task cpd), phase, envelope
  size, RF centre; mix of simple (phase-sensitive) and complex (phase-invariant energy)
  cells. Recurrent connectivity by tuning similarity — like-to-like excitation + broad
  inhibition + divisive normalisation. Temporal dynamics over the 2 s trial
  (`dt ≈ 0.08 s`, ~25 steps) → genuine population trajectory `r(t)`. Sensory noise
  (private + shared), magnitude is a per-animal trait.
- **Decision layer.** Two readout neurons — Go-driving and NoGo-driving — each with a
  plastic weight vector from all V1 neurons. Readout current = cosine / normalised dot
  product `s(r(t), w_c)`. `SI(t)` = bounded normalised difference (framework form).
  Sign convention documented in code: positive `SI` favours Go.
- **Plasticity.** Decision-layer weights only. V1 weights never learn.
- **Exploration.** Emergent: random small initial weights → `SI(t) ≈ 0` → accumulator is
  pure diffusion → ~50/50 choices early (the "RL agent random at start"). Optional
  exploration-temperature trait.
- **Curriculum.** Phase 1: archetype trials only until rolling accuracy > 80 % over the
  last 200 trials (or a cap ~1200). Phase 2: full grid; errors keep updating weights.
- **Stimulus sampling.** Flat **category** prior — equal Go vs NoGo mass every phase. In
  Phase 2 the orientation sampling is **biased towards the archetypes** (0°, 90°
  oversampled relative to the 6 intermediate orientations; archetype share ≈ 0.5 of
  Phase-2 trials, a config knob). This is a sampling/curriculum choice, not an inference
  prior.
- **Prior (inference).** The ideal observers use a **flat prior over orientation**. With
  symmetric reward this keeps the 45° boundary exactly optimal. The archetype sampling
  bias does *not* change the optimal decision rule. Block-prior manipulations remain a
  documented extension, not in v1.

## Ideal observers

- **Stimulus-IO** — wraps `io_core.py`: `(s,c,d) → kappa → noisy measurement m → posterior
  over s → P(Go)`. Optimal given direct noisy latent access.
- **Neural-IO** — optimal Bayesian classifier of the network's own `r(t)`: estimate
  class-conditional `p(r | Go)`, `p(r | NoGo)` from the model's generative process, then
  `P(Go | r)`. Optimal readout of the V1 code the network actually has.
- Expected ordering: network ≤ neural-IO ≤ stimulus-IO. `network ≈ neural-IO` ⇒ the SI
  readout is an efficient readout; `neural-IO < stimulus-IO` ⇒ V1 information bottleneck.

## Deliverables

- `si_network_model/` package: `stimuli.py`, `v1_model.py`, `decision.py`,
  `plasticity.py`, `agent.py`, `ideal_observer.py`, `train.py`, `analysis.py`,
  `plots.py`, `run.py`, `config.py`, `README.md`.
- `si_network_model/tests/` — pytest, **written test-first** (TDD per repo `CLAUDE.md`),
  synthetic-data tests with known ground truth.
- `si_network_model/figures/` — learning curves, psychometric overlays (network vs
  neural-IO vs stimulus-IO), `SI(t)` traces by difficulty, learned prototype/bundle
  archetypes + per-animal κ, SI-vs-IO trial-by-trial agreement & decision-efficiency,
  cross-animal κ↔lapse / κ↔drift-variability, RT distributions.
- `si_network_model/results/` — generated arrays (this is model-generated data).
- Short findings note in `si_network_model/README.md` after the run.

## Conventions

- Work directly on `main` (no feature branch — per user preference). Commit at checkpoints.
- numpy / scipy / matplotlib only; no PyTorch (custom plasticity, no autograd needed).
- Every plot: axis labels with units, title, legend. Raw arrays saved beside figures.
- Verification step: synthetic-data tests with known ground truth; a recovery check that
  the plasticity rule recovers planted prototypes.

## Won't-Do (explicitly excluded from v1)

- Spiking neurons — rate-based only.
- Block-prior / volatility manipulations — flat prior only.
- Plasticity in V1 itself — V1 is fixed (framework commitment).
- Exemplar / vMF bundle variants as the *learning* mechanism — a Hebbian rule natively
  delivers only the prototype; exemplar variants are measured post-hoc, not learned.
- Calcium-imaging realism (deconvolution, GCaMP kinetics) — clean rate `r(t)`.
- Fitting to real Cb15–Cb25 data — this is a generative model + IO comparison, not a fit.
- DTW / warped temporal alignment — stimulus-locked only.

---

## Edit log

- **2026-05-19** — Initial version. Decisions 1–7 resolved via two rounds of clarifying
  questions. Decision 6 chosen as symmetric two-bound DDM; decision 7 user requested
  *both* reward modes (symmetric default + asymmetric config knob).
- **2026-05-19** — Post-plan tweaks: cohort raised 8 → 10 animals; trials/animal reduced
  (~4500); Phase-2 stimulus sampling biased towards archetypes with a flat Go/NoGo
  category prior; IO inference prior confirmed flat over orientation.
- **2026-05-20** — V1 noise model revised during the build. First implementation made
  difficulty come from per-timestep noise, which washes out under trial-averaging, so
  even 1%-contrast stimuli decoded perfectly. Replaced with **information-limiting
  orientation jitter**: per-trial noise in the *encoded orientation itself*, scaled by
  stimulus reliability (low contrast / high dispersion → larger jitter) — the neural
  mirror of the stimulus-IO's kappa(c,d). This gives a genuine difficulty gradient
  (hardest conditions near chance for all observers) and is the mechanistic source of
  bundle width. DDM drift gain raised to convert the (realistically small) SI into
  decisions. Calibrated outcome: decision efficiency network/neural-IO ≈ 0.97.
- **2026-05-20** — Build complete. `run.py` made resumable (per-animal pickle
  checkpoints embedding their Config) because the sandbox shell has a 45 s limit and
  cannot delete files. Final cohort run: network 0.756 / neural-IO 0.775 / efficiency
  0.977; SI-vs-IO log-odds r = 0.85; r(kappa,lapse) = -0.73, r(kappa,SI-spread) = -0.89.
  56 tests pass. numpy + matplotlib only (scipy/pytest unavailable in the sandbox;
  custom zero-dependency test runner). 9 figures + findings in README.md.
- **2026-05-20 (extension)** — Three follow-up requests from the user:
  (1) **Trajectory archetypes** — the prototype was a static vector; made it a genuine
  time-resolved trajectory `a_c(t)` learned by per-timestep Hebbian, plus a
  reservoir-sampled **exemplar bundle** (K = 20 per class).
  (2) **Four SI variants** — added `si_variants.py` (prototype / expected-exemplar /
  max-exemplar / vMF), compared as readouts of the same learned bundles. Result:
  prototype (0.96) and vMF (0.98) beat the exemplar variants (0.94) — the cheap
  prototype suffices.
  (3) **Asymmetric payoff** — second cohort under hit +1 / FA -0.25 / CR +0.25 /
  miss 0. This exposed a bug: the RPE baseline made CR=miss outcomes anti-train the
  NoGo archetype. **Fixed by switching plasticity from RPE to raw reward** (the
  framework's literal `Delta w ~ pre*post*R`); the user's added small CR reward gives
  the NoGo channel a learning gradient. Asymmetric cohort then learns and shows a
  pronounced (over-shooting) Go-bias.
  Two-cohort run: symmetric efficiency 0.961, SI-vs-IO r = 0.84, r(kappa,lapse) -0.75.
  60 tests pass. 12 figures. Full write-up in REPORT.md.
