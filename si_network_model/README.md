# SI-framework network model

An end-to-end, reward-trained network model of the UncertaintyV1 orientation
Go/NoGo task, built to test the **Similarity-Index (SI) framework**:

> Can a network that decides by comparing the V1 population trajectory `r(t)` to
> two learned *archetypes* (via a cosine readout feeding a drift-diffusion
> accumulator) learn the task from reward, and **approximate optimal Bayesian
> inference**?

**Short answer: yes.** After learning, the cosine/prototype readout captures
**96 %** of the accuracy of the optimal Bayesian decoder of the same V1
activity, and its accumulated evidence tracks the optimal log-posterior-odds at
**r = 0.84** — near-optimal, with a small principled shortfall.

**The full write-up — detailed methods, results, every figure, and the network
diagram — is in [`REPORT.md`](REPORT.md).** This file is the quick reference.

## Pipeline

```
grating image  ->  V1 population r(t)  ->  cosine readout -> SI(t)  ->  DDM  ->  Go/NoGo
 (CreateGratings   (fixed Gabor RFs +     (two plastic            (premotor   (choice +
  recipe)           recurrence;            archetype               accumulator) lick latency)
                    orientation jitter)    trajectories)
```

Only the V1->decision synapses learn (three-factor reward-modulated Hebbian);
V1 is fixed. A cohort of 10 simulated animals is trained per reward mode.

## Modules

| File | Role |
|---|---|
| `config.py` | All parameters; per-animal trait sampling |
| `stimuli.py` | Grating recipe (matches `CreateGratings.m`) |
| `v1_model.py` | Gabor RFs, recurrence, orientation-jitter front end -> `r(t)` |
| `decision.py` | Cosine readout, `SI(t)`, two-bound DDM |
| `plasticity.py` | Three-factor reward-modulated Hebbian rule |
| `si_variants.py` | The four bundle-similarity variants |
| `ideal_observer.py` | Stimulus-IO (reuses `io_core.py`) + neural-IO |
| `agent.py` / `train.py` | One animal; curriculum + cohort runner |
| `analysis.py` / `plots.py` | Metrics and the 12 figures |
| `run.py` | Resumable end-to-end orchestrator |

## Running it

```
python -m si_network_model.run             # two cohorts; resumable, re-run to continue
python -m si_network_model.tests.run_tests # 60 synthetic-data tests
```

`run.py` checkpoints each animal (with its Config, so config changes
invalidate it) and works within a per-invocation time budget — re-invoke until
it prints `DONE`. Outputs land in `figures/` and `results/`.

## Key results (symmetric-reward cohort)

| Result | Value |
|---|---|
| Network accuracy / neural-IO accuracy | 0.737 / 0.767 |
| **Decision efficiency** (network / neural-IO) | **0.961**  (range 0.92–0.98) |
| Network evidence vs optimal log-posterior-odds | Pearson **r = 0.84** |
| Cross-animal r(bundle kappa, lapse rate) | **-0.75** |
| Cross-animal r(bundle kappa, drift variability) | **-0.83** |
| SI variant efficiency (prototype / exp-exemplar / max-exemplar / vMF) | 0.96 / 0.94 / 0.94 / 0.98 |
| Asymmetric-payoff criterion shift (toward Go) | +11° (over-shoots the reward-optimal +3°) |

## Figures (`figures/`)

1–9 core results (learning, psychometric, efficiency, SI-vs-IO, SI traces,
bundles, cross-animal predictions, RTs, accuracy by difficulty); 10 the learned
trajectory archetypes; 11 the four-variant comparison; 12 the
symmetric-vs-asymmetric Go-bias.

## Caveats

Rate-based (not spiking); difficulty is a phenomenological orientation-jitter
model; contrast/orientation are the strong difficulty axes, dispersion weaker;
the neural-IO uses an equal-covariance Gaussian model; not fitted to real data.
See `REPORT.md` section 5 and `UNDERSTANDING.md`.
