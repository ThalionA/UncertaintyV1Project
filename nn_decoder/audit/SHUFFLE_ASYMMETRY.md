# Audit — spat_shf > temp_shf is an architectural artefact, not a finding

**Date:** 2026-05-26
**Trigger:** While building the cross-decoder ratio plot for the neuron-scaling
sweep (`nn_decoder/figures/neuron_scaling/aggregate_*_detail_ratio.png`), noticed
the shuffle-control ratio (`spat_shf / temp_shf`) sat consistently above 1.0
across every mouse, every N, on the Q-target run. If the shuffle controls
were unbiased, this ratio should be ~1 — both decoders fitting random noise.

## The mechanism

Both decoders share the same MLP backbone and training pipeline
(`run_experiment.run_animal_decoder`). They differ only in **where time-averaging
happens** inside the forward pass (`nn_classifier.get_model_probabilities`):

- **PPC (spat):** `softmax(MLP(mean_t(x_t)))` — one prediction per trial.
- **Sampling (temp):** `mean_t(softmax(MLP(x_t)))` — T per-bin forward passes,
  then the T predicted distributions are averaged.

These two pipelines are **not equivalent** under softmax. Averaging T post-softmax
distributions reduces trial-level prediction variance by roughly a `1/√T`
Monte-Carlo factor; averaging inputs *before* a single forward pass does not get
that bonus. Per-trial input variability flows one-to-one to per-trial output
variability in the PPC branch.

On target-shuffled controls, the optimal prediction is the training-set marginal
of the target. The sampling decoder's output-averaging pulls each trial's
prediction closer to that constant marginal than the PPC decoder's
single-pass output can. The result is a systematic per-trial JS-loss penalty on
spat_shf relative to temp_shf — **even though neither decoder has any real
signal to extract**.

## Empirical evidence

Decomposition of the spat_shf vs temp_shf gap, using the saved per-trial
predicted distributions in
`population_results_fixed_hyperparams_JS_stratified_balanced.mat` (n=6 mice,
target=Q):

| mouse | mean H(spat_shf) | mean H(temp_shf) | trial-var spat | trial-var temp | JS(mean_pred, marg) spat | JS(mean_pred, marg) temp |
|------:|-----------------:|-----------------:|---------------:|---------------:|-------------------------:|-------------------------:|
|     0 |            4.270 |            4.273 |        0.00064 |        0.00036 |                  0.00214 |                  0.00211 |
|     1 |            3.905 |            3.951 |        0.00575 |        0.00322 |                  0.00588 |                  0.00688 |
|     2 |            4.026 |            4.028 |        0.00518 |        0.00430 |                  0.00530 |                  0.00665 |
|     3 |            4.174 |            4.220 |        0.00406 |        0.00213 |                  0.00435 |                  0.00473 |
|     4 |            4.124 |            4.179 |        0.00522 |        0.00296 |                  0.00121 |                  0.00182 |
|     5 |            4.221 |            4.250 |        0.00211 |        0.00094 |                  0.00195 |                  0.00230 |

Reading:
1. **Per-trial entropy of the prediction is nearly identical** (gap ≤ 0.05 nats).
   Both models output similarly-broad distributions per trial — neither is "sharp".
2. **Trial-to-trial variance is ~2× larger for spat_shf in every mouse.**
   Spat_shf jiggles around the marginal more.
3. **Mean predictions of both models land equally close to the target marginal.**
   The miss is per-trial, not on average — exactly what the Monte-Carlo-smoothing
   explanation predicts.

Driver figure: `nn_decoder/figures/shuffle_asymmetry/empirical_breakdown.png`.

## Synthetic confirmation

`nn_decoder/audit/shuffle_asymmetry_diagnostic.py` reproduces the asymmetry from
architecture alone, no training involved:

- Closed-form (fixed random MLP, random per-bin Gaussian inputs):
  - H(PPC-style) ≈ H(SBC-style) (≤ 0.05 nats difference).
  - Trial-to-trial variance ratio PPC/SBC ≈ **1.92×**.
  - JS-to-marginal ratio PPC/SBC ≈ **1.83×**.
  - These numbers match the empirical magnitudes (≈ 2× variance, ≈ 1.5–2× JS).
  - Figure: `synthetic_jensen_demo.png`.
- Population-size scan (closed-form, no training): the architectural variance
  ratio is **roughly N-invariant** at ~1.8×.
  Figure: `synthetic_jensen_scan.png`.
- Training scan (trained on shuffled targets, requires torch): reproduces the
  *growth* of the asymmetry with N that the empirical data shows on top of the
  architectural floor. Training dynamics — both decoders fit more spurious
  structure as N grows, asymmetrically — explain the empirical drift from
  ratio ≈ 1.08 at N=10 to ratio ≈ 1.25 at full population.
  Figure: `synthetic_training_scan.png`.

## Practical implications

1. **Do not interpret `spat_shf` and `temp_shf` as a shared "chance level".**
   They are not directly comparable. The same is true of the `spat / temp`
   ratio — its baseline is shifted by the architectural Jensen factor, not 1.0.

2. **Comparing each decoder against its own shuffle is fair.** The "loss / own
   shuffle" normalisation cancels the architectural offset on both sides.
   `plot_chance_normalised_compare` in `plot_neuron_scaling.py`
   (`aggregate_*_chance_normalised.png`) implements this. On the Q-target
   scaling sweep, after this normalisation, **spatial and temporal extract
   nearly identical signal lift over their own chance levels at every N**.
   The apparent spat<temp gap on the raw ratio plot is almost entirely the
   architectural shuffle asymmetry, not a real signal-extraction difference.

3. **The neuron-scaling ratio plots (`aggregate_*_ratio.png`,
   `aggregate_*_detail_ratio.png`) need a caption acknowledging the floor is
   not at 1.0.** A reader who sees `spat/temp > 1` and concludes "temporal
   wins" overstates the gap.

4. **Anywhere downstream code computes `temp_mean / temp_shf_mean` and the
   spatial equivalent and reports them on the same axis, those numbers ARE
   directly comparable** — the architectural offset divides out within each
   decoder. `plot_post_fix_performance.py` (which uses `KLs[temp]`, contaminated
   pre-2026-05-19 by a separate issue tracked in `AUDIT_loss_consumers.md`)
   does this and is structurally on the right side of this question.

## Pedagogical companion

For a slide-deck-style walkthrough of the smoothing mechanism (six
standalone PNGs + speaker notes), see:

- `nn_decoder/audit/JENSEN_EXPLAINER.md` — talking points keyed to figures.
- `nn_decoder/audit/jensen_smoothing_explainer.py` — generates the figures.

The explainer figures isolate the smoothing effect using zero-mean
within-trial noise (so PPC's input is *exactly* the trial signal
regardless of σ or T), then sweep σ and T to show that PPC's behaviour
is invariant to within-trial noise structure while SBC inherits a
$1/T$ Monte-Carlo variance-reduction rate.

## Related entries
- `GOTCHAS.md` → "Architecture / framing" section already noted PPC and SBC
  differ only in pre/post-softmax time-averaging. This audit adds the
  quantitative behavioural consequence.
- `GOTCHAS.md` → "The SBC entropy penalty is asymmetric" — the entropy
  regulariser is *another* architectural asymmetry (training-time, sampling
  branch only) on top of the forward-pass asymmetry documented here.
- `AUDIT_loss_consumers.md` — separate, unrelated contamination in pre-2026-05-19
  `KLs[temp]` values from `entropy_lambda * H(pred)` being baked into the saved
  test loss.

## Reproducing

```bash
cd nn_decoder
python audit/shuffle_asymmetry_diagnostic.py
```

Writes the four figures into `nn_decoder/figures/shuffle_asymmetry/`. Requires
the saved per-trial predicted distributions in a `population_results_*.mat`
(the script auto-discovers the JS-loss / fixed-hyperparams variants under
`nn_decoder/`).
