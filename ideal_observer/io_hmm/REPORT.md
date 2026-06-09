# IO-HMM Phase 2 — Free Per-State Perception and the Perception–Action Dissociation

*Report. Branch `claude/io-hmm-idea-iklbl`. Covers the v0.5 confidence-velocity
emission (Phase 1) and the free-perception extension (Phase 2a `lambda_`, Phase
2b parameterised prior, Phase 2c wiring into the default model and the real-data
fit).*

---

## 1. Summary

The IO-HMM models a mouse's behaviour on an orientation Go/NoGo task as a hidden
Markov chain over a small set of **normative internal states**, where each state
is an *ideal observer* with its own sensory, perceptual, and decisional
parameters. v0 distinguished states only by their **choice psychometric** (bias,
slope, lapses) and the shape of their task prior. The choice channel carries
~1 bit/trial, which caps what is recoverable: the psychometric slope `beta`
saturates, and latent-path attribution needs wide state separation.

Phase 1 added a faithful **confidence-velocity emission** — pre-reward-zone
running speed as a graded readout of the ideal observer's decision variable
`DV(m)`, marginalised over the latent sensory measurement `m` *jointly with
choice*. Phase 2 then opens the **perceptual front-end** of each state to EM:

- **`lambda_`** — a per-state *sensory precision* multiplier on the Stage-1
  `kappa`. It reshapes the inference posterior **and** the generative spread, so
  it is an observer who genuinely *sees* sharper or blurrier evidence.
- **`prior_strength` / `prior_weight`** — a per-state parameterised task prior
  (bimodal von Mises concentration and asymmetry). `prior_weight` is a
  *perceptual category bias* distinct from the decisional bias `alpha`.

The headline scientific result is a **perception–action dissociation**: two
states with *identical* choice and action parameters but different perception
are separated, and the perceptual parameter is recovered — **only because** the
velocity channel reads `DV(m)` independently of the choice psychometric. This
report documents the model, the identifiability structure, the recovery results
(Figure 6), and the wiring that lets the dissociation be run on real animals.

**Status:** 112 `io_hmm` tests green (109 prior + 3 new wiring tests). Code and
figure-generator pushed; figure regenerable via
`scripts/perception_dissociation_figure.py`.

---

## 2. Background and motivation

The behavioural model estimates the mouse's *internal uncertainty* from
observable behaviour under a normative (Bayesian ideal-observer) framework — see
`wiki/Module_IdealObserver.md` and `documents/ideal_observer_methods_v3.tex` for
the two-stage hierarchical fit that anchors the sensory/kinematic parameters
(Stage 1). The IO-HMM sits on top: it lets the *latent state* switch trial-to-
trial between ideal observers that differ in interpretable ways (engaged vs
disengaged vs naive, etc.).

Two identifiability constraints (quantified in `figures/README.md`) shaped the
design:

1. **Psychometric saturation (constraint 1).** Because the IO log-posterior-odds
   `g(m)` already span ≈ ±5 over the condition grid, `P(go) = sigma(alpha +
   beta*g)` pins near 0/1 for `beta >~ 1.5`; choices alone cannot recover steep
   slopes (logistic separation runs `beta` to the bound).
2. **Path separability (constraint 2).** Choice-only emissions carry ~1
   bit/trial, so the latent path — and hence per-state parameter attribution —
   is recoverable only when states are behaviourally well separated.

A second, graded channel (velocity) addresses both. Crucially, the production
velocity emission reads `DV(m)` computed from the **frozen** Stage-1 `kappa`, so
the confidence gain `beta_vel` is identified (no free sensory gain to be
degenerate with). That same property is what makes Phase 2 work: with velocity
present, a *perceptual* parameter that moves `DV(m)` becomes identifiable apart
from the choice psychometric.

---

## 3. Model and methods

### 3.1 Generative model (one ideal-observer state)

For a trial with stimulus orientation `s` (deg) and task covariates `c`
(contrast) and `d` (difficulty):

```
Sensory precision   kappa(s,c,d) = (kappa_min + kappa_amp) * c^c_power * exp(-d_power * d)
Per-state precision  kappa_eff    = lambda_ * kappa(s,c,d)            # Phase 2: lambda_ scales it
Measurement          m ~ VonMises_doubled(s, kappa_eff)              # p(m|s), 180-deg period
Task prior           prior(s)  = bimodal VonMises at {0, 90} deg, concentration prior_strength
                                 (asymmetric weight prior_weight on the 0-deg centre; Phase 2b)
Posterior            p(s|m)    ∝ p(m|s) * prior(s)
Log-odds             g(m)      = log P(Go|m) / P(NoGo|m)
Decision variable    DV(m)     = EU(Go|m) - EU(NoGo|m)
```

The expected-utility decision variable uses the task utility
`R_hit = 1, R_miss = 0, R_cr = 0.1, R_fa = -0.2` (Go orientations near 0 deg,
NoGo near 90 deg). `kappa_min = 1.0`, `prior_strength = 3.0`, `lambda_ = 1.0`
are the v0 defaults.

### 3.2 Joint choice + velocity emission

Both observable channels are functions of the *same* latent measurement `m`,
marginalised analytically (paper Eqs. 11, 20; `emissions.joint_log_emission_state`):

```
Choice psychometric  psi(g; alpha,beta,gamma,delta) = gamma + (1-gamma-delta)*sigmoid(alpha + beta*g)
Velocity readout     v | m ~ Normal(beta_vel*DV(m) + alpha_vel, sigma_vel^2)
```

- **Choice-only:** `P(go|cond) = sum_m psi(g(m)) * p(m|cond)`; Bernoulli on the
  choice. (Byte-for-byte the v0 emission.)
- **With velocity:** velocity Bayes-updates the posterior over `m` before the
  choice probability is formed —
  ```
  Z         = sum_m N(v; beta_vel*DV(m)+alpha_vel, sigma_vel) p(m|cond)
  post(m|v) = N(...) p(m|cond) / Z
  P(go|v)   = sum_m psi(g(m)) post(m|v)
  log p     = log Z + choice*log P(go|v) + (1-choice)*log(1-P(go|v))
  ```
  So velocity is **never a direct linear predictor of choice**; it enters only
  through the shared `m` (the "metacognitive" coupling). `beta_vel = 0` recovers
  the stimulus-independent engagement marker as a special case.

### 3.3 Per-state parameter blocks

| Block | Params | Meaning | Bounds |
|-------|--------|---------|--------|
| Psychometric | `alpha, beta, gamma, delta` | decisional bias, slope, lapses | slope/bias/lapse bounds |
| Velocity (confidence) | `beta_vel, alpha_vel, sigma_vel` | confidence gain, baseline, noise | `beta_vel` free ⇒ DV-coupled |
| **Perception (Phase 2)** | `lambda_` | sensory precision multiplier | `(0.1, 10.0)` |
| **Perception (Phase 2b)** | `prior_strength` | task-prior concentration | `(0.05, 20.0)` |
| **Perception (Phase 2b)** | `prior_weight` | prior asymmetry (mass on 0-deg Go centre) | `(0.02, 0.98)` |

Each entry is either **fixed** (frozen at a value) or **free** (`None`, fit by
EM). A state's *identity* is its pattern of fixed-vs-free entries.

Two perceptual axes differ in where they enter:

- **`lambda_`** scales `kappa_eff` for **both** `p(m|s)` (generative spread) and
  the inference posterior — a genuinely sharper/blurrier observer. It reshapes
  `g(m)`, `DV(m)`, and `p(m|s)` together.
- **The parameterised prior** (`prior_strength`, `prior_weight`) enters **only
  the inference posterior**, not `p(m|s)`. So it moves `g(m)`/`DV(m)` but not the
  generative spread. `prior_weight` shifts category mass toward Go (0 deg) or
  NoGo (90 deg) — a perceptual bias upstream of the decision.

### 3.4 Identifiability of the perceptual parameters

- `lambda_` and `prior_strength` are **both posterior-sharpness knobs** and only
  weakly separable (`lambda_` additionally moves `p(m|s)`; the prior does not).
  Fitting both free per state is therefore **fragile** — the suite asserts only
  monotone EM and parameter plumbing for the both-free case, not joint recovery.
- `prior_weight` (asymmetry) is distinct from both `lambda_` (symmetric
  sharpness) and the decisional bias `alpha`.
- **The velocity channel is the identifying instrument.** Because velocity reads
  `DV(m)` (which perception moves) independently of the choice psychometric (which
  only rescales an otherwise-fixed `g(m)`), a perceptual change is observable
  *separately* from an action change. Without velocity, free `lambda_` is not
  identifiable — which the API enforces (§3.6).

### 3.5 HMM and EM fitting

`K` states with initial distribution `pi` and transition matrix `A`. Fitting is
Baum–Welch EM (`fit.fit`): the E-step computes per-state joint-emission
log-likelihoods and posterior state responsibilities (forward–backward); the
M-step re-estimates `pi`, `A`, and each state's free parameters by maximising the
responsibility-weighted complete-data joint log-likelihood
(`fit._fit_state_joint`). The perception, psychometric, and velocity blocks are
optimised **together** per state, because they couple through the shared `m`:
the velocity channel and the choice channel both read terms that `lambda_`
reshapes. When `lambda_` (or a free prior param) is free, the IO terms
(`g, DV, p(m|s)`) are **recomputed inside the objective** as the parameter varies
— the compute cost of letting perception move; for frozen perception the cached
terms are reused. Updates are warm-started and guarded so EM stays monotone.
Latent paths are decoded by Viterbi (`fit.viterbi_paths`).

### 3.6 The v0 four-state model and the Phase-2c wiring

The default model (`states.default_v0_states`):

| State | Prior | Free choice | Free velocity | Free perception (`with_perception`) |
|-------|-------|-------------|---------------|-------------------------------------|
| Perfect | bimodal | `beta` | all three | `lambda_` |
| Thirsty | bimodal | `alpha, beta` | all three | `lambda_` |
| Disengaged | bimodal | `alpha` (`beta=0`) | `alpha_vel, sigma_vel` (`beta_vel=0`) | **frozen** |
| Naive | flat | `beta` | all three | `lambda_` |

The Phase-2c wiring adds a `with_perception` flag mirroring the existing
`with_velocity`:

- The **engaged** states (Perfect, Thirsty, Naive) receive free `lambda_`.
- **Disengaged stays frozen** (`lambda_ = 1`): with `beta = 0` (constant P(go))
  and decoupled velocity (`beta_vel = 0`), its `DV(m)` drives neither channel, so
  `lambda_` would be unidentifiable for it — the exact parallel to why its
  velocity is decoupled.
- `with_perception` **requires** `with_velocity`; requesting free perception
  without velocity raises `ValueError`, because `lambda_` is only identifiable
  through the velocity channel.

The real-data driver (`scripts/fit_real_data.py`) exposes this as
`--free-perception` (guarded against `--no-velocity`) and records
`perc_per_state` in each animal's JSON summary, so the dissociation can be fit
per animal.

---

## 4. Results

![Phase 2 perception dissociation](../../figures/fig6_perception_dissociation.png)

*Figure 6 — Free per-state perception and the perception–action dissociation.
Regenerate with `python scripts/perception_dissociation_figure.py`. The recovery
setups mirror the headline tests in `tests/test_io_hmm_perception.py` exactly.*

**(a) Mechanism — sensory precision reshapes `DV(m)`.** Sweeping `lambda_ ∈
{0.5, 1, 2}` for a fixed bimodal-prior observer: higher `lambda_` (sharper
evidence) makes `DV(m)` more extreme across the orientation grid; lower `lambda_`
flattens it. This is qualitatively different from the choice psychometric, which
only rescales an otherwise-fixed `DV`. Perception changes the *shape* of the
decision landscape; action changes how a *fixed* landscape maps to behaviour.

**(b) `lambda_` dissociates from action.** Two states with identical choice
(`beta = 2`) and action (`beta_vel = 2`) parameters but different perception
(Sharp `lambda_ = 2.0`, Blur `lambda_ = 0.5`), separable only via a velocity
baseline offset. Free `lambda_` recovers the perceptual difference
(**Sharp 1.64, Blur 0.53; ratio 3.1×**) while the action gain `beta_vel` recovers
to ≈2 for *both* states (**Sharp 1.99, Blur 1.89**) — perception moves, action
does not. Permutation-corrected Viterbi agreement **97%** confirms the latent
path is recovered. (The recovered Sharp `lambda_` sits below its true 2.0: at
`lambda_ >~ 2` the posterior is already near-saturated, the analogue of the
`beta` slope wall, so the *ordering and ratio* are recovered more tightly than
the absolute high-precision value.) The suite asserts `lambda_(Sharp) > 1.3`,
`lambda_(Blur) < 0.8`, `ratio > 2`, `|beta_vel − 2| < 0.8`, and agreement > 0.85.

**(c) Free perceptual category bias (`prior_weight`).** Two states with matched
choice/action but different prior asymmetry (Lo `prior_weight = 0.25` → NoGo
bias, Hi `0.75` → Go bias). Free `prior_weight` recovers and separates them
(**Lo 0.27, Hi 0.69**; Viterbi agreement **98%**) — a perceptual bias upstream
of the decision, distinct from the decisional bias `alpha`.

**Negative / fragility control.** Fitting `lambda_` **and** `prior_strength` free
per state is degenerate (both sharpen the posterior); the suite covers this case
only for EM monotonicity and parameter plumbing, **not** joint recovery, and the
default wiring frees only `lambda_` for this reason.

### Test suite

```
112 passed   (tests/test_io_hmm_*.py)
  - test_io_hmm_perception.py : mechanism, plumbing, lambda_ recovery,
    prior_weight recovery, both-free monotonicity
  - test_io_hmm_states.py     : +3 wiring tests (default frozen; engaged-free/
    Disengaged-frozen under with_perception; velocity-required guard)
```

---

## 5. Reproducing / running on real data

```bash
# Regenerate Figure 6 (runs the EM recovery fits; a few minutes)
python scripts/perception_dissociation_figure.py            # full
python scripts/perception_dissociation_figure.py --quick    # faster, slightly noisier

# Fit the four-state model with free perception on a real animal
python scripts/fit_real_data.py --animal Cb15 --free-perception
#   -> writes perc_per_state (per-state lambda_) into the JSON summary
#   --free-perception requires velocity (it errors if combined with --no-velocity)
```

`perc_per_state` in `<out>/fit_summary.json` then carries each engaged state's
fitted `lambda_`, i.e. the per-state sensory precision the dissociation reads.

---

## 6. Limitations and future work

- **`lambda_` ⊥ `prior_strength` degeneracy.** Both sharpen the posterior; only
  `lambda_` is wired into the default model. A joint free fit needs an extra
  identifying constraint (e.g. a channel sensitive to `p(m|s)` spread but not the
  prior) and is left out by design.
- **Velocity is the load-bearing channel.** All Phase-2 identifiability rests on
  the confidence-velocity emission; on animals/sessions with poor velocity signal
  the perceptual parameters will be weakly constrained.
- **Real-data validation pending.** The dissociation is demonstrated in recovery
  (simulation→fit). Running `--free-perception` across the cohort and checking
  whether engaged states actually differ in fitted `lambda_` is the natural next
  step, alongside the open review nits (naming/doc, Problems 3–5) and the
  free-perception fit-performance pass.

---

*Generated as part of the IO-HMM Phase 2 work. Figures live in `figures/`
(git-ignored, regenerable); see `figures/README.md` for the companion
identifiability figures (fig1–fig5).*
