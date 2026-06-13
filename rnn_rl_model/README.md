# RNN + RL similarity-framework model

A reinforcement-learning sibling of [`si_network_model`](../si_network_model).
Same grating Go/NoGo task and same forward path —

```
grating image  ->  recurrent V1 r(t)  ->  cosine similarity to 2 archetypes  ->  SI(t)  ->  Go/NoGo
 (CreateGratings   (fixed Gabor RFs +      (two learned weight                  (actor-critic
  recipe)           recurrence, orient.     vectors = the V1->action            policy + DDM)
                    jitter = uncertainty)   mapping)
```

— but the V1→action mapping is learned by an **actor-critic** (policy gradient
with a value baseline) instead of Hebbian plasticity, and the recurrent V1 can be
**trained end-to-end then frozen**.

> **Question.** Does an RL agent whose policy reads a cosine similarity to two
> learned archetypes still land on the **template direction `Δμ`** (not the
> covariance-whitened optimum `Σ⁻¹Δμ`) and reach near-Bayesian efficiency — and
> does task-training the recurrence change that?
>
> **Answer: yes, and the signature is robust to the learning rule.** An
> actor-critic agent reproduces the framework's full RD readout signature, just
> as the Hebbian model does. Task-training the recurrence makes the V1 covariance
> *stimulus*-exploitable (as in real V1) while the agent's *choices* stay
> template-aligned.

The **only** structural difference between the two columns below is whether the
recurrent weights `W_rec` are frozen (`fixed`) or task-trained-then-frozen
(`trained`). Everything else — Gabor front end, noise, readout, learner — is
identical. Full design rationale in [`UNDERSTANDING.md`](UNDERSTANDING.md).

## Running it

```
python -m rnn_rl_model.run            # full cohort (n=6)
python -m rnn_rl_model.run --quick    # fast smoke run (n=3)
python -m pytest rnn_rl_model/tests/ -q   # 10 ground-truth tests
```

Outputs: `figures/` (PNG+SVG) and `results/cohort.pkl` + `summary.json`
(gitignored, regenerable).

## Key results (cohort mean ± SEM, n = 6)

| Metric | fixed V1 | trained V1 | reading |
|---|---|---|---|
| Decision efficiency (network / neural-IO) | **0.987 ± 0.005** | 0.954 ± 0.016 | the cosine readout is near-optimal for its own V1 |
| Evidence quality `r(mean SI, IO log-odds)` | **0.852 ± 0.010** | 0.678 ± 0.030 | SI tracks the optimal log-posterior-odds |
| Readout vs **template** `Δμ` (cosine) | **0.881 ± 0.016** | **0.925 ± 0.023** | the learned readout *is* the template direction |
| Readout vs **whitened** `Σ⁻¹Δμ` (cosine) | 0.358 ± 0.008 | 0.008 ± 0.005 | … not the covariance-whitened optimum |

### RD-1 / RD-2 readout tests (the same estimators run on the real Cb15–Cb25 V1)

| Contrast | fixed V1 | trained V1 | framework prediction |
|---|---|---|---|
| RD-1 Δstim-AUC (whitening buys stimulus accuracy?) | −0.008 | **+0.033** | real V1: +0.04 — covariance exists & is stimulus-exploitable |
| RD-1 Δchoice-AUC (whitening buys *choice*?) | −0.034 | −0.054 | < 0 — whitening hurts the choice readout |
| RD-2 M1−M0 (mean SI adds choice info?) | +0.36 | +0.38 | > 0 — *sanity check*: SI drives the policy by construction |
| RD-2 M2−M1 (within-trial var adds? SBC wedge) | **≈ 0** | +0.027 | ≈ 0 — no SBC posterior-width signature |
| RD-2 M3−M1 (whitened adds beyond template? premise) | **≈ 0** | **≈ 0** | ≈ 0 — **choices read the template, not the optimum** |

The decisive cells are **M3−M1 ≈ 0** (whitening adds nothing to choice) and the
trained-V1 **RD-1 Δstim > 0 with M3−M1 still ≈ 0**: task-training the recurrence
reproduces real V1's *exploitable-but-unused-for-choice* covariance. (M1−M0 is
large by construction — the policy is `σ(β·mean SI)` — so it only confirms the
test fires, it is not framework evidence.)

## Figures (`figures/`)

1. `fig1_learning_curves` — actor-critic accuracy / reward over training.
2. `fig2_template_vs_whitened` — **headline**: learned readout aligns with `Δμ`,
   not `Σ⁻¹Δμ`, per animal, both V1 conditions.
3. `fig3_efficiency_and_si` — network-vs-neural-IO accuracy; SI vs optimal
   log-odds.
4. `fig4_psychometric` — P(Go) vs orientation against the stimulus-IO ceiling.
5. `fig5_fixed_vs_trained` — cohort summary. Trained V1 doesn't raise accuracy
   (the Gabor V1 is already near ceiling) — it raises **decisiveness** (mean |SI|
   0.19 → 0.74) and template alignment.
6. `rd{1,2}_*__{fixed,trained}` — the RD figures, one pair per V1 condition.

## Relationship to `si_network_model`

| | `si_network_model` | `rnn_rl_model` |
|---|---|---|
| Learning rule | three-factor Hebbian (RPE) | **actor-critic policy gradient** |
| Recurrent V1 | fixed | **fixed _or_ trained-then-frozen** |
| Reuses | — | its stimuli, V1 front end, IO, DDM |

Both land on the template readout at ~0.96–0.99 efficiency with `r ≈ 0.85` —
**the framework's signature is a property of the architecture, not the learning
rule.** Worth adding as a second positive control to the Similarity-Framework
vault note.

## Caveats

Rate-based; Go/NoGo only (2AFC is an extension); flat prior; a generative model +
IO comparison, not a fit to real data. `mean SI` predicts choice almost perfectly
*by construction* (it is the policy input), so RD-2 M1−M0 is a fire-check, not
evidence. Efficiency occasionally ≥ 1 because the neural-IO's equal-covariance
Gaussian is not a strict ceiling on a near-separable trained V1.
