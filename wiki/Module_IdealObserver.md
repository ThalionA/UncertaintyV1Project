# Module: Behavioural Modelling (Ideal Observer)

This module implements a normative framework to estimate the mouse's
internal uncertainty from its observable behaviour. The current
implementation is the **v2 two-stage hierarchical fit**, which strictly
separates the estimation of sensory and kinematic-emission parameters
(Stage 1) from the choice psychometric (Stage 2). See `documents/ideal_observer_methods_v3.tex` for the publication-ready methods write-up.

## Generative Model (one-line summary)

```
s --> p(m|s) = VonMises(s, kappa(c,d))   isotropic precision
m --> p(s|m) = Bayes update with bimodal task prior at 0/90 deg
m --> DV(m) = EU(Go|m) - EU(NoGo|m)      with utility R_hit=1, R_miss=0, R_cr=0.1, R_fa=-0.2
m --> y_vel  = beta_vel  * DV(m) + alpha_vel  + Normal(0, sigma_vel)
m --> y_lick = beta_lick * DV(m) + alpha_lick + Normal(0, sigma_lick)
```

Oblique-effect parameters (`rho_k`, `phi_pref`) are fixed to zero;
`kappa_min = 1.0`, `kappa_prior = 3.0`, `decision_beta = 1.0`.

## Two-Stage Hierarchical Fit

**Stage 1 — sensory + kinematic emissions (kinematics only).** Velocity is
always fit; licks are an optional channel (default: held out).
Optimisation is by BADS, hierarchical: pooled-data fit seeds the
per-animal fits. 5-fold CV per animal; per-fold parameters are used to
generate strict OOS predictions of velocity, licks, and the implicit
choice (`1[DV(m) > 0]`). A final all-data fit is warm-started from the
best-fold parameters.

**Stage 2 — choice psychometric on log posterior odds.** Four parameters
`(alpha_r, beta_r, gamma_r, delta_r)` define a logistic on the model's
log posterior odds `g(m) = log P(Go|m) / P(NoGo|m)`, conditioned on the
trial's *observed velocity* via a Bayes update on the latent
measurement. Velocity therefore never enters as a direct linear predictor of
choice. CV holds the four choice parameters strictly OOS while keeping
Stage 1 parameters at the pooled all-data fit.

## Marginalised Inversion

After fitting, the IO inversion exports two complementary trial-by-trial
distributions for each animal:

| Export | Symbol | Decoder use |
| :--- | :--- | :--- |
| `post_s_marginal` | `Q(theta)` | Perceptual posterior (with task prior) |
| `L_s_marginal` | `L(theta)` | Marginalised likelihood (no prior on s) |
| `decision_posterior` | `[P(Go), P(NoGo)]` | Soft binary decision posterior |

Each is computed by averaging the measurement-conditioned distribution
over the kinematic-conditioned posterior on `m`, with an underflow
safeguard that falls back to the generative prior when the kinematics
are astronomically unlikely under all `m`.

## Spatial Window for Kinematics

Pre-reward-zone kinematics are extracted from a configurable spatial
window. Production: `window_start = 30 vu`, `window_width = 10 vu` (so
`[30, 40)`, centred at 35 vu, sitting upstream of the 100–140 vu reward
zone). Both streams are robustly Z-scored per animal before fitting.

## Key Files

### Fitting & Estimation
- **`ideal_observer_hierarchical_fitting_v2.m`** — current production
  fit. Configuration block at the top (`config.fit_licks`,
  `config.fit_choice_psych`, window) controls Stage 1 likelihood content
  and whether Stage 2 runs.
- **`SpatiotemporalHierarchicalFitting_ModelComparison.m`** — older
  spatial-vs-temporal IO model comparison.
- **`Spatiotemporal_Decoding_Stimulus.m`** — decoding stimulus
  orientation directly from kinematic readouts (sanity check on the
  generative model).
- **`parameter_recovery_v2.m`** — parameter recovery on synthetic
  animals seeded from random and from fitted-real parameters.
- **`io_hmm/io_core.py`** — Python utilities for downstream consumption
  of IO exports.

### Prior sweep
- **`io_prior_sweep/run_real_prior_sweep.m`** — refits the IO under four
  prior configurations on real data.
- **`io_prior_sweep/run_recovery_prior_sweep.m`** — same sweep on the
  recovery (synthetic) cohort.
- **`io_prior_sweep/compare_real_priors.m`** and
  **`plot_recovery_results.m`** — comparison/visualisation drivers.

### Visualisation
- **`plot_IOresults_edit.m`** — population summary of fits, predicted
  vs. observed kinematics, posteriors, choice psychometrics.

## Inputs / Outputs
- **Input**: `vr_<sessionname>_light.mat` (per-trial position, velocity,
  licks, stimulus identity, etc.). Pooled across sessions per animal.
- **Output**: trial-by-trial `Q(theta)`, `L(theta)`, `[P(Go), P(NoGo)]`,
  fitted parameters, CV-OOS predictions of velocity/licks/choice,
  fitted choice psychometric. Exported into the `TrialTbl_Struct` /
  `NeuralStore` consumed by the [Neural Decoder](Module_NeuralDecoder.md)
  via `nn_decoder/utils_v26.py::load_vr_export`.
