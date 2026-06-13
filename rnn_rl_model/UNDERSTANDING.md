# UNDERSTANDING — RNN + RL similarity-framework model

*Current understanding at top. Edit log appended below.*

---

## Goal

A reinforcement-learning sibling of `si_network_model`. Same task and same
forward path — grating → recurrent V1 population `r(t)` → cosine similarity to
two learned archetypes → SI(t) → Go/NoGo — but the V1→action mapping is learned
by an **actor-critic** (policy gradient with a value baseline) instead of
three-factor Hebbian plasticity, and the recurrent V1 itself can optionally be
**trained end-to-end then frozen**.

The scientific question: does an RL agent whose policy reads a similarity index
between V1 and two archetypes still converge on the **template direction `Δμ`**
(not the covariance-whitened optimum `Σ⁻¹Δμ`) and reach **near-Bayesian decision
efficiency** — and does task-training the recurrence change that? If the
framework's signature (template readout; whitening adds nothing to *choice*) is
robust across two very different learning rules, that is strong evidence the
signature is a property of the *architecture*, not the learning rule.

Lives in `UncertaintyV1/rnn_rl_model/`. Reuses `si_network_model` for the
stimulus recipe, the fixed Gabor front end, the ideal observers and the DDM.

## Task

Identical to `si_network_model`: orientation Go/NoGo, boundary 45°, the 9×9
(orientation)×(contrast,dispersion) grid, archetypes `{0°,90°}` at
`(contrast 1, dispersion 5)`. Perceptual uncertainty = the same
**information-limiting orientation jitter** (per-trial noise in the encoded
orientation, scaled by stimulus reliability), the neural mirror of the
stimulus-IO's `κ(c,d)`. (Lick-left/right 2AFC is a documented extension; v1 is
Go/NoGo to match the real UncertaintyV1 task and reuse the whole stack.)

## Resolved design decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Where it lives | **New sibling package** `rnn_rl_model/`, reusing `si_network_model`'s stimuli, V1 front end, IO and DDM. Keeps the Hebbian model intact for a learning-rule contrast. |
| 2 | What is trainable | **Both, as a comparison.** `fixed` V1 = the hand-built Gabor + like-to-like recurrence, frozen (framework-faithful: only V1→action learns). `trained` V1 = the recurrence `W_rec` trained end-to-end by the same actor-critic, then frozen. The **only** difference between the two columns is `W_rec`. |
| 3 | Learner | **Actor-critic.** One-step episodes (each trial): sample a Go/NoGo action from the policy, get reward, update by policy gradient with advantage `reward − value`. A learned linear **critic** over the similarity features is the baseline (the dopamine-RPE-flavoured choice). |
| 4 | What the policy reads | The framework's **bounded prototype SI(t)** — pointwise cosine of `r(t)` against two **static** archetype vectors `w_go`, `w_nogo` (the literal "one plastic weight vector per response neuron"), positivity-mapped and contrasted. Policy logit = `β·mean_t SI(t) + bias`, `β` a learned positive decision gain. |
| 5 | What learns | `w_go`, `w_nogo`, `bias`, `β`, the critic — always; `W_rec` only in the `trained` condition. The fixed Gabor front end never learns (framework commitment: V1 is stimulus-driven sensory cortex). |
| 6 | Behaviour readout | At evaluation, the SI(t) stream is run through the framework's **two-bound DDM** (`si_network_model.decision.accumulate`) for choice + RT — so the RL policy and the framework accumulator are reported consistently. Learning uses the smooth `sigmoid(β·mean SI)` surrogate (the DDM's first-passage choice probability), which is differentiable. |
| 7 | Implementation | **One torch graph** (V1 recurrence + cosine readout + actor-critic). `fixed` vs `trained` = whether `W_rec` is a `Parameter` or a frozen buffer. Saturating `φ` keeps the (possibly trained) recurrence stable. |

## Architecture

- **V1 front end (fixed).** The numpy `si_network_model.V1Model`: diverse Gabor
  RFs, `jittered_drive` (a seam extracted so alternative recurrence back-ends can
  share it). Produces the static feed-forward drive with orientation jitter.
- **Recurrence (torch).** `r ← r + (dt/τ)(−r + φ(drive + g·r Wᵣ + noise))`, faithful
  to `V1Model.simulate`. `W_rec` initialised from the hand-built like-to-like
  matrix; trained in the `trained` condition.
- **Readout (torch).** Two static archetype vectors, renormalised to unit length
  each step (cosine is scale-invariant, so this only conditions the denominator).
  SI(t) is the bounded prototype index (matches `decision.si_trace` bit-for-bit —
  a unit test checks this).
- **Actor-critic (torch).** `policy_loss = −advantage·log π(a)`,
  `value_loss = (reward − V)²`, small entropy bonus for sustained early
  exploration. Advantage and the critic features are detached so the value head
  does not corrupt the representation.

## Ideal observers & metrics

- **Neural-IO** is fitted on the agent's *own* recurrence (an `AgentV1` adapter
  exposes a `present` that rolls out the agent's `W_rec`), so it is the optimal
  readout of exactly the V1 code each condition produces.
- **Template-vs-whitened**: cosine of the learned `w_go − w_nogo` against `Δμ`
  (class-mean difference) and against `Σ⁻¹Δμ` (shrunk pooled covariance).
- **Decision efficiency** = network accuracy / neural-IO accuracy;
  **evidence quality** = `r(mean SI, neural-IO log-odds)`.
- **RD-1/RD-2 bridge** (`rd_adapter.py`): the frozen-eval block is fed into the
  project's `similarity_readout_tests` exactly as the Hebbian cohort is, so the
  RL model becomes a second, learning-rule-independent positive control.

## Deliverables

- `rnn_rl_model/` package: `config`, `model`, `train`, `evaluate`, `analysis`,
  `cohort`, `plots`, `rd_adapter`, `run`, `paths`.
- `tests/` — synthetic ground-truth tests (SI matches the numpy reference;
  learns the real task; does *not* learn under uninformative reward;
  template-alignment sensitivity/specificity).
- `figures/` — learning curves; **template-vs-whitened** (headline); efficiency +
  SI-vs-log-odds; psychometric vs IO ceiling; fixed-vs-trained summary; the two
  RD figures per condition.
- `results/` — cohort pickle + `summary.json` (gitignored; regenerable).

## Conventions

- Work on `main`; commit at checkpoints (repo is trunk-only).
- torch for the model (already a repo dependency via `nn_decoder`); numpy/scipy
  for analysis; matplotlib for figures (PNG ≤1600 px + SVG).
- Reuse `si_network_model` rather than reimplement; the only change made there is
  the behaviour-preserving `V1Model.jittered_drive` seam (60 tests still pass).

## Won't-do (excluded from v1)

- Spiking neurons — rate-based only.
- 2AFC lick-left/right — Go/NoGo only (extension noted).
- Block-prior / volatility manipulations — flat prior.
- Fitting to real Cb15–Cb25 data — generative model + IO comparison, as with the
  Hebbian sibling.
- Q-learning / other RL algorithms — actor-critic only (the chosen variant).
- Exemplar / vMF bundle variants as the *learning* mechanism — the policy reads
  the prototype; bundle variants are a `si_network_model` concern.

---

## Edit log

- **2026-06-13** — Initial build. Decisions 1–7 resolved with the user up front
  (new sibling package; both fixed and trained V1; actor-critic). Unified torch
  model (V1 recurrence + cosine readout + actor-critic in one graph);
  fixed/trained = `W_rec` buffer vs Parameter. Reused the Hebbian sibling's
  stimuli/V1/IO/DDM; added a `V1Model.jittered_drive` seam (behaviour-preserving,
  60 si tests still pass). 10 package tests pass. See README for results.
