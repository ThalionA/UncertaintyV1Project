# Module: Neural Decoding (PPC vs SBC)

This is the core computational module where we test the two competing
theories of neural uncertainty: **Probabilistic Population Codes (PPC)**
and **Sampling-Based Codes (SBC)**. Production decoders are trained
against three IO-derived distributional targets and a behavioural
choice control, all from the same shared MLP backbone.

## Targets

| Target | Shape | Source field | Loss |
| :--- | :--- | :--- | :--- |
| Perceptual posterior `Q(theta)` | 91-bin distribution | `post_s_marginal` | PCA-weighted Euclidean |
| Marginalised likelihood `L(theta)` | 91-bin distribution | `L_s_marginal` | PCA-weighted Euclidean |
| Decision posterior `[P(Go), P(NoGo)]` | 2-D soft probability | `decision_posterior` | MSE |
| Binary choice (control) | 2-D one-hot | `decision_posterior` (treated as label) | Cross-entropy |

The choice control bypasses the IO inversion entirely and predicts the
animal's report directly from the same population activity.

## Architectures (shared MLP backbone)

Both architectures share an MLP `f_phi` with:

- single hidden layer of **32 units**
- **`tanh`** activation
- **Xavier-uniform** weight init
- output layer projects to either 91 bins (distributional targets) or 2
  units (decision/choice), then `softmax`

They differ strictly in *when* temporal integration happens:

```
PPC:  P(theta | r_n)     = Softmax( f_phi( mean_t r_{n,t} ) )       (mean THEN softmax)
SBC:  P(theta | r_{n,t}) = Softmax( f_phi( r_{n,t} ) )                (per-bin softmax)
      Pbar_SBC(theta)    = (1/T) sum_t P(theta | r_{n,t})            (then average)
```

The SBC architecture additionally pays an instantaneous-entropy penalty
`lambda_H * mean_t H(P_{n,t})` with `lambda_H = 3e-3` (sharpness prior).
The PPC is exempt.

## Production Hyperparameters

These are the values currently used in `run_fixed_hyperparams.py` and
`run_fixed_hyperparams_choice.py`:

| | |
| :--- | :--- |
| `hidden_sizes` | `[32]` |
| `activation_function` | `tanh` |
| `weight_initialization` | `xavier_uniform` |
| `time_window` | `'half'` (1.0–2.0 s of the 2 s open-loop epoch) |
| `bin_size_ms` | `100` (downsampled from native 50 ms by averaging pairs) |
| `T` (bins/trial) | `10` |
| Optimiser | Adam, `lr = 1e-3`, `weight_decay = 3e-4` |
| Epochs | `30` |
| Gradient accumulation | `minibatch_size = 16` trials per optimiser step |
| Gradient clip | `||grad||_2 <= 1.0` |
| Random restarts | `REP = 5`, retain lowest-training-loss model |
| `entropy_lambda` | `3e-3` (SBC only) |
| Train/test split | 50/50, stratified by (orientation, contrast, dispersion), seed 42 |
| Z-scoring | per-neuron, training-fold-only stats, applied to all trials |

## Splits

- `stratified_balanced` — 50/50 stratified split on (orientation,
  contrast, dispersion) triplet.
- `generalize_contrast` — train on contrast-varying trials + 50% of
  high-contrast/low-dispersion baseline anchor; test on
  dispersion-varying trials + remaining 50% of baseline anchor.
- `generalize_dispersion` — reverse of the above.

## Shuffled Baseline

For every architecture, target type, and split we additionally train a
parallel "shuffled" model in which the trial-to-target mapping is
permuted **only on the training set**, with the same permutation applied
identically to all `T` time bins of a given trial (so within-trial
temporal structure is preserved). All loss metrics are normalised to
this baseline at the per-mouse aggregate level
(`mean(loss) / mean(loss_shuf)`), then averaged across mice — never
per-trial, because per-trial shuffled losses can be near-zero on the 2-D
MSE targets.

## Architecture-Level Recovery (Double Dissociation)

`decoder_recovery_v26.py` and `run_fixed_recovery.py` implement a
double-dissociation crossover:

1. Train production PPC and SBC base decoders on each real target.
2. Extract each base network's predictions across all trials
   (`full_decoded`).
3. Retrain *both* architectures from scratch against (a) PPC predictions
   and (b) SBC predictions, with identical hyperparameters and the same
   train/test split.
4. Score the resulting 2x2 (target architecture × decoding architecture)
   matrix on held-out trials.

A genuine architectural difference manifests as a double dissociation:
each architecture should recover its own counterpart's predictions
better than the opposing architecture's. Reporting uses the same loss
as the base decoder (PCA / MSE), normalised to the matched shuffled
baseline; KL and 1-D Wasserstein are exported as complementary
readouts.

## Key Files

### Production entry points
- **`run_fixed_hyperparams.py`** — runs the three distributional
  decoders (`perception`, `likelihood`, `decision`) across all three
  splits and all six animals. Saves `population_results_fixed_*_<split>.mat`.
- **`run_fixed_hyperparams_choice.py`** — runs the binary-choice
  control decoder across all three splits and animals.
- **`run_fixed_recovery.py`** — runs the architecture-level recovery
  crossover on whichever base configurations have been completed, plots
  the 2x2 loss matrix and scatter recoveries.

### Core pipeline
- **`run_experiment_v26.py`** — `run_animal_decoder(config, mouse_id)`.
  Handles loading, target routing, splitting, z-scoring, PCA basis
  extraction (training-only), training of `{PPC, SBC} × {real, shuffled}`,
  test-set evaluation, and `full_decoded` export for downstream
  recovery.
- **`neural_network_classifier_v26.py`** — `SimpleFlexibleNNClassifier`,
  divergence/loss implementations (KL, JS, Wasserstein, PCA, MSE, CE),
  forward routing for PPC vs SBC, and
  `train_and_select_best_model(REP, ...)`.
- **`utils_v26.py`** — IO export loading, stratified and generalisation
  splits, `apply_temporal_binning`, PCA/SBC synthetic-target generators.
- **`neural_dataset.py`**, **`to_tensor.py`** — PyTorch dataset / tensor
  glue.

### Sweep / Optuna
- **`run_experiment_v26.py`** still implements the four divergence
  losses (KL, JS, Wasserstein, PCA). The PCA-weighted loss was selected
  during the Optuna sweep (`optuna_universal_cv_v26.py`,
  `optuna_phase2_sbc_lambda.py`, `optuna_joint_v27.py`) for the
  distributional targets and is the production loss.

### Visualisation & deepdive
- **`generate_all_fixed_plots.py`** — single-shot driver for the
  publication figures.
- **`plot_decoder_results_v26.py`**, **`plot_normalized_bars_v26.py`**,
  **`plot_normalized_scatter_v26.py`** — summary loss plots.
- **`plot_decomposition.py`**, **`decomposition_analysis.py`** — neural
  variance decomposition.
- **`plot_fano_factor.py`**, **`population_metrics_vs_uncertainty.py`** —
  population-statistic readouts vs. uncertainty.
- **`plot_io_coherence.py`**, **`io_coherence.py`** — IO-vs-decoder
  coherence diagnostics.
- **`plot_ideal_observer.py`** — IO-side panels regenerated from the IO
  export.
- **`pca_visualisation_v26.py`** — uncertainty-manifold PCA.
- **`results_deepdive.py`**, **`plot_sample_dynamics.py`** — SBC sample
  dynamics, per-condition deepdive.
- **`normal_ppc_analysis.py`** — sanity check on the canonical PPC
  shape.

## Inputs / Outputs
- **Input**: deconvolved spike probabilities (from
  [Preprocessing](Module_Preprocessing.md)) plus IO targets (from
  [Ideal Observer](Module_IdealObserver.md)), all packaged in
  `data/VR_Decoder_Data_Export.mat` and loaded by
  `utils_v26.load_vr_export`.
- **Output**: per-mouse `population_results_fixed_*_<split>.mat`
  containing per-trial decoded distributions, shuffle controls, the
  full PCA basis used for the loss, and the trial metadata needed for
  downstream condition-stratified analysis.
