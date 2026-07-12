# DeepSets analysis of trialwise perceptual uncertainty

**Status:** complete. Synthetic validation used ten independently generated datasets. The full real-data grid comprises 6 mice x 3 models x 4 losses x real/null fits, plus a leakage-safe condition-mean oracle. The GPU was occupied by the pre-existing hyperparameter sweep and rejected a second CUDA context, so the resumable real grid completed in non-overlapping CPU tmux jobs on `gpu1` without disturbing that sweep.

## 1. Question and inferential scope

The analysis asks whether the **unordered set of within-trial V1 population states** contains information about the trialwise ideal-observer perceptual posterior (Q(\theta)) beyond the trial-mean response.

It distinguishes three nested questions:

1. **Mean code:** is the trial-mean population response sufficient?
2. **Moment code:** do explicit per-neuron temporal variances add information?
3. **Learned set code:** do learned higher-order, order-invariant properties of the within-trial state set add information beyond mean and variance?

DeepSets is permutation-invariant. A positive result therefore supports useful **unordered temporal variability**, not temporal order, sampling dynamics, or necessity for behaviour. Necessity would require a causal perturbation that removes the relevant temporal feature while preserving mean activity.

## 2. Data and target

- Primary target: the 91-bin IO perceptual posterior (Q(\theta)), (\theta=0,\ldots,90^\circ).
- Neural input: V1 population activity during the late half of the 2 s open-loop stimulus epoch.
- Temporal resolution: ten non-overlapping 100 ms bins.
- Analysis unit: one mouse; six mice form the population sample.

The target remains an IO-inferred, velocity-conditioned construct rather than direct perceptual ground truth. Conclusions should therefore refer to information about **IO-inferred perceptual uncertainty**.

## 3. Leakage-safe splitting and preprocessing

Each mouse uses one fixed, reproducible split:

1. Outer 50/50 train-test split, stratified on the joint stimulus cell `(orientation, contrast, dispersion)`.
2. Inner validation set comprising 20% of the outer-training trials, also stimulus-stratified.
3. Per-neuron mean and standard deviation fitted only on inner-training trials, pooling their time bins.
4. Projection/PCA basis fitted only on inner-training (Q) targets using the all-trials basis.
5. The test set remains untouched until the best restart and epoch have been selected from validation loss.

Split indices, preprocessing statistics, PCA basis, random seeds and null permutations are saved with the result shards.

## 4. Models

All models receive an array (X\in\mathbb{R}^{T\times N}) and return logits over 91 orientation bins.

### Mean-only MLP

\[
\hat Q = \operatorname{softmax}\{f_\rho(T^{-1}\sum_t X_t)\}.
\]

This is the static trial-mean baseline.

### Moments MLP

\[
\hat Q = \operatorname{softmax}\{f_\rho([\operatorname{mean}_t X_t,
\operatorname{var}_t X_t])\}.
\]

Variance is the population variance (`unbiased=False`), matching the descriptive within-trial moment rather than the small-sample unbiased estimator.

### DeepSets

\[
\hat Q = \operatorname{softmax}\{\rho(T^{-1}\sum_t \phi(X_t))\}.
\]

`phi` is a two-layer tanh encoder and `rho` is a one-hidden-layer tanh readout. There are no positional embeddings, recurrent states or bin-specific parameters. Joint temporal-bin permutation therefore leaves the output invariant up to floating-point summation tolerance.

### Capacity matching

DeepSets defines the reference parameter budget. Integer hidden widths for the mean and moments MLPs are selected analytically to minimise the difference from this budget. For Mouse 0 (65 neurons), counts are 2,289 (mean), 2,311 (moments) and 2,291 (DeepSets): within 1%.

## 5. Training objectives

Four objectives are trained separately:

- Forward KL: (D_{KL}(Q\Vert\hat Q)).
- Jensen-Shannon divergence.
- Unweighted Brier score: (\sum_j(\hat Q_j-Q_j)^2). The code retains the `MSE` label for compatibility; mean-MSE differs only by the constant (1/91), but the Brier sum avoids an arbitrary 91-fold gradient-scale disadvantage.
- Projection-based loss: (\sum_k \lambda_k[(\hat Q-Q)^Tv_k]^2), where (v_k,\lambda_k) are fitted on inner-training targets only.

Cross-entropy and Wasserstein training were explicitly excluded. All models use Adam, gradient clipping, mini-batches of trials, relative validation improvement and early stopping. The real-data grid uses an epoch cap of 200 and three independently seeded restarts; the synthetic capability grid used a 120-epoch cap and two restarts to cover ten independently generated datasets. Restarts are selected by validation loss, never training or test loss.

## 6. Common held-out evaluation

Every fitted model is evaluated under the same metrics, irrespective of its training objective:

- forward KL;
- Jensen-Shannon divergence;
- Brier score;
- absolute posterior-mean error in degrees;
- absolute posterior-variance error in degrees squared;
- absolute entropy error in nats;
- projection loss as a secondary diagnostic.

The confirmatory real-data metric is held-out KL for the KL-trained models. Width and entropy errors prevent successful posterior-location decoding from masquerading as uncertainty recovery.

## 7. Trial-specific null

Whole (Q) rows are independently permuted within exact stimulus cells in the inner-training and validation partitions. The null therefore preserves:

- the condition-mean posterior;
- the distribution of targets within each condition;
- neural inputs and their temporal structure;
- train/validation/test sizes.

It destroys only the mapping between a particular neural trial and its trial-specific posterior deviation. Test targets retain their true alignment. Singleton cells remain fixed and their eligible fraction is reported; 96.0% of Mouse 0 inner-training trials were shuffle-eligible.

One null permutation is a matched descriptive baseline, not an empirical permutation distribution. Population inference rests on paired mouse effects rather than trial-level null p-values.

## 8. Synthetic validation

Ten independently generated datasets were used. Each had 600 trials, ten bins, 16 neural features, six posterior centres and five uncertainty levels. Targets were normalised Gaussian distributions over the 91-bin grid. A shared neural mean code represented posterior centre in all regimes, while the width code differed.

### Mean-coded uncertainty

Posterior width was encoded in the exact trial mean of a dedicated neural channel. Temporal noise was centred and RMS-normalised per trial so its variance did not covary with width.

Expected: all three architectures recover width.

### Variance-coded uncertainty

Posterior width scaled a dedicated channel's within-trial variance. Each noise sequence was centred to exactly zero and normalised to unit RMS before scaling; therefore realised temporal means—not merely expected means—were identical across uncertainty levels.

Expected: moments and DeepSets outperform mean-only; moments may be more sample-efficient.

### Order-coded uncertainty

Groups of five counterfactual trials used byte-identical unordered state sets and centres. Only a fixed temporal permutation differed across the five uncertainty levels.

Expected: all three invariant models remain at the width baseline. A separate order-sensitive positive control used four explicit order statistics and nearest-centroid cross-validation; it must recover the order label, showing that failure of invariant models is not a broken generator.

### Synthetic results

The preregistered qualitative architecture result was recovered for KL and JS:

| regime | KL-trained mean-only | moments | DeepSets | interpretation |
|---|---:|---:|---:|---|
| Mean-coded: KL | 0.081 | 0.052 | 0.020 | all recover; learned set transform is most flexible |
| Mean-coded: variance error | 27.0 | 22.3 | 12.1 deg² | all recover width |
| Variance-coded: KL | 0.228 | **0.065** | 0.115 | moments and DeepSets beat mean-only |
| Variance-coded: variance error | 79.0 | **21.6** | 50.1 deg² | explicit variance is most sample-efficient |
| Order-coded: KL | 0.23 | 0.23 | 0.23 | invariant models are indistinguishable |
| Order-coded: variance error | 79.9 | 79.1 | 80.3 deg² | all fail width as required |

Values are means over ten independently generated datasets. JS showed the same ordering. The order-positive-control accuracy was 0.88-0.99 against five-class chance of 0.20.

For KL training, both moments and DeepSets beat mean-only on variance-regime KL and width error in 10/10 datasets; moments beat DeepSets in 10/10, as expected for an oracle summary with better sample efficiency. In the order regime, width errors remained tightly clustered around 79-80 deg². Small KL differences there reflect recovery of the shared posterior-centre code, not recovery of the inaccessible order-coded width.

The loss comparison was itself diagnostic:

- Brier training learned the targets but was less sample-efficient at the shared 120-epoch synthetic budget.
- Projection-trained models performed poorly under common KL and width error, including in the mean-coded regime. This reproduces the known projection-loss blind-spot: low-variance posterior-shape directions exert too little restoring force. It must not be interpreted as an architectural inability.

Thus the synthetic validation establishes that the implementation can detect a pure variance code, that DeepSets is genuinely order-invariant, and that training loss can dominate apparent architectural conclusions.

## 9. Real-data statistical plan

The two prespecified KL-trained contrasts are computed once per mouse:

1. `DeepSets - mean`: does any unordered within-trial structure add information?
2. `DeepSets - moments`: do learned higher-order set properties add information beyond explicit variance?

Negative KL differences favour DeepSets. Raw and within-condition-null-normalised results are shown for every mouse. With (n=6), the two-sided exact Wilcoxon minimum p-value is 0.03125; animal consistency and effect size are more informative than threshold language. Other losses and metrics are robustness analyses and require multiplicity control if assigned formal p-values.

Trials, folds, losses and repeated stimulus splits are not independent population replicates.

## 10. Real-data results

### 10.1 Raw held-out architecture comparison

Under the prespecified KL training/evaluation, mean held-out KL across mice was:

| model | mean KL | mean width error | mean entropy error |
|---|---:|---:|---:|
| Mean-only | 0.352 | 401.8 deg² | 0.392 nats |
| Mean + variance | 0.367 | 392.1 deg² | 0.384 nats |
| DeepSets | **0.315** | 390.7 deg² | 0.382 nats |

DeepSets had lower raw KL than moments in 6/6 mice: mean paired difference -0.0523, paired t-test p=0.0146, exact two-sided Wilcoxon p=0.03125 (the n=6 floor). It beat mean-only in 5/6 mice: mean difference -0.0370, paired t p=0.052, Wilcoxon p=0.0625.

This is a real predictive advantage under raw held-out KL, but it does **not** by itself isolate trial-specific uncertainty. DeepSets is a richer invariant function class and can use condition-related distributions of within-trial states. Critically, DeepSets also beat moments in 6/6 **null-trained** models by nearly the same amount (mean null difference -0.0547), demonstrating that the raw gap does not depend on correct trial-to-target alignment.

There was no corresponding uncertainty-specific gain. DeepSets-minus-moments posterior-variance error was -1.39 +/- 14.91 deg² across mice (paired t p=0.929; Wilcoxon p=0.688), and entropy-error difference was -0.0019 +/- 0.0119 nats (paired t p=0.878; Wilcoxon p=0.563).

### 10.2 Within-condition null: the load-bearing result

The matched real/null KL ratios for KL-trained models were:

| model | mean real/null KL | range across mice | test against 1, paired t p | exact Wilcoxon p |
|---|---:|---:|---:|---:|
| Mean-only | 0.990 | 0.922-1.055 | 0.617 | 0.688 |
| Mean + variance | 0.993 | 0.974-1.008 | 0.191 | 0.219 |
| DeepSets | 1.004 | 0.969-1.066 | 0.791 | 1.000 |

None extracts detectable trial-specific (Q) information beyond the within-condition shuffle. Width and entropy real/null ratios were also approximately 1. The same null conclusion held under JS, Brier and projection training; no loss/model cell produced a consistent mouse-level improvement over its matched null.

Therefore the raw DeepSets advantage is best interpreted as an architectural or condition-decoding advantage, not evidence that unordered temporal variability represents trial-specific uncertainty.

### 10.3 Condition-mean oracle

The inner-training condition-mean predictor achieved KL 0.012-0.016 across mice (mean approximately 0.014), compared with 0.20-0.67 for KL-trained neural models. Exact stimulus condition almost determines the IO posterior; residual trialwise (Q) variation is small, and none of the tested neural summaries recovers it detectably.

This also shows why raw decoder performance is an inadequate uncertainty test: even the best neural model remains roughly twenty-fold above the condition oracle while real/null performance is at chance.

### 10.4 Loss robustness

DeepSets also had the lowest mean raw KL under JS and Brier training. Projection-trained outputs were substantially worse under common KL (DeepSets 0.707; mean 0.778; moments 0.608) and showed no real/null advantage. This matches the synthetic result and the established projection-loss blindness to posterior shape.

### 10.5 Scientific conclusion

The analysis validates that DeepSets can recover an unordered variance code when one exists, but finds no evidence that real V1 within-trial variability carries trial-specific IO uncertainty beyond stimulus condition. Temporal variability may still encode stimulus condition, posterior location, or features not captured by this IO target. Temporal order remains untested, and causal necessity is not addressed.

## 11. Outputs and reproducibility

- Core implementation: `nn_decoder/deepsets_uncertainty.py`
- Runner, aggregation and figures: `nn_decoder/run_deepsets_analysis.py`
- Tests: `tests/test_deepsets_uncertainty.py`
- Results: `nn_decoder/results/deepsets_uncertainty/` (gitignored)
- Figures: `nn_decoder/figures/deepsets_uncertainty/` (gitignored; every figure PNG+SVG)
- Mouse-level contrasts: `mouse_contrasts.csv`
- Prespecified group tests: `group_stats_primary.csv`

Canonical commands:

```bash
# Synthetic validation
python nn_decoder/run_deepsets_analysis.py synthetic --seeds 0:10

# Full real-data grid
python nn_decoder/run_deepsets_analysis.py real --mouse-ids 0 1 2 3 4 5

# Aggregate and render
python nn_decoder/run_deepsets_analysis.py plot
```

Each model/loss/null/mouse job writes an atomic `.npz` shard. Existing shards are skipped unless `--force` is supplied, making interrupted runs safely resumable.

Key figures:

- `synthetic_common_kl`: two common held-out metrics across regimes, models and losses.
- `synthetic_width_calibration`: target-versus-decoded posterior variance.
- `synthetic_order_oracle`: positive control for the order generator.
- `real_null_normalised_kl`: per-mouse real/null KL ratios.
- `real_headline_model_contrasts`: prespecified raw DeepSets contrasts.
- `real_vs_condition_mean_oracle`: neural decoders beside the stimulus oracle.
- `real_common_metric_scorecard`, `real_width_calibration`, `real_example_posteriors_mouse0`: robustness and qualitative diagnostics.

## 12. Interpretation boundaries

A real-data DeepSets gain over mean-only would show that unordered within-trial population-state variation contains useful information about IO-inferred (Q). A gain over moments would show that mean and per-neuron variance are insufficient summaries. Neither result would demonstrate temporal ordering, biological sampling, downstream use, or causal necessity.

The next extension, if warranted, is an order-sensitive but capacity-matched TCN/GRU compared directly with DeepSets under joint-bin permutation and population-cofluctuation nulls.
