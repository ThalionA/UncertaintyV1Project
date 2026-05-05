# Pipeline Overview

The UncertaintyV1 pipeline is a multi-stage process that bridges
experimental biology and computational neuroscience.

## Data Flow

### Stage 1: Experimental Acquisition
- **Input**: Mouse behaviour in a virtual reality (VR) corridor.
- **Output**:
    - Raw 2-photon imaging TIFFs.
    - VR Logs (`.mat`) containing position, velocity, and licks.
    - Arduino logs for hardware synchronisation.
- **Tools**: MATLAB, ViRMEn, Psychtoolbox.

### Stage 2: Preprocessing & Data Fusion
- **Input**: Raw imaging and VR logs.
- **Process**:
    - Motion correction (NoRMCorre).
    - Segmentation (Cellpose).
    - Neuropil subtraction (FISSA).
    - Spike deconvolution (CASCADE, jGCaMP8-tuned).
    - Longitudinal ROI registration (`roi_reg`).
    - Alignment of neural data to VR behavioural timestamps.
- **Output**: `UnifiedSessionData` (structured MATLAB files containing
  aligned spikes, licks, velocity, and pupil data).
- **Primary Script**: `preprocessing/createUnifiedSessionData.m`.

### Stage 3: Behavioural Inference (Ideal Observer v2)
- **Input**: `UnifiedSessionData`.
- **Process**: Two-stage hierarchical IO fit. **Stage 1** fits
  isotropic sensory precision and kinematic emissions to velocity
  (always) and licks (optional, default off) using BADS, with a
  pooled-data fit seeding per-animal fits and 5-fold CV. **Stage 2**
  fits a four-parameter choice psychometric on the log posterior odds
  `g(m)`, conditioned on the trial's observed velocity via a Bayes
  update on `m` (so velocity never enters as a direct linear predictor
  of choice).
- **Output**: trial-by-trial perceptual posterior `Q(theta)`,
  marginalised likelihood `L(theta)`, decision posterior
  `[P(Go), P(NoGo)]`, fitted parameters, CV-OOS predictions of
  velocity/licks/choice, plus a fitted choice psychometric.
- **Primary Script**:
  `ideal_observer/ideal_observer_hierarchical_fitting_v2.m`.

### Stage 4: Latent State Discovery (GLM-HMM)
- **Input**: Behavioural logs.
- **Process**: Identifies discrete cognitive states (e.g., "Engaged",
  "Biased", "Disengaged") using a Hidden Markov Model with GLM
  observations.
- **Output**: State labels for every trial.
- **Usage**: Used to filter out trials where the mouse is not
  performing the task optimally, ensuring the neural decoder is trained
  on high-fidelity data.
- **Primary Script**: `glm_hmm/gonogo_glm_hmm_global_v4.py`.

### Stage 5: Neural Decoding (PPC vs SBC)
- **Input**: Aligned deconvolved spike probabilities (Stage 2) and IO
  targets (Stage 3).
- **Process**:
    - PyTorch ANNs decode three distributional targets — `Q(theta)`,
      `L(theta)`, and the soft 2-D decision posterior — from the same
      population activity. A behavioural choice control decodes the
      animal's binary report directly, bypassing the IO inversion.
    - Two architectures share the MLP backbone (32-unit hidden layer,
      `tanh`, Xavier init):
        - **Spatial (PPC)**: average across time, then MLP+softmax.
        - **Temporal (SBC)**: MLP+softmax per bin, then average; subject
          to an instantaneous-entropy sharpness prior.
    - Loss: PCA-weighted Euclidean for distributional targets (×100
      scaling), MSE for the 2-D decision posterior, cross-entropy for
      the choice control. KL/JS/Wasserstein retained as a sweep family.
    - Validation: 50/50 stratified split, training-fold-only z-scoring,
      trial-permuted shuffled baseline; OOD splits hold out one of
      contrast or dispersion variation.
    - Architecture-level recovery: each base decoder's predictions
      become synthetic targets for a 2x2 crossover that tests for a
      double dissociation between architectures.
- **Output**: per-mouse `population_results_fixed_*_<split>.mat` with
  decoded distributions, shuffle controls, PCA basis, and per-trial
  loss metrics.
- **Primary Scripts**:
  - `nn_decoder/run_fixed_hyperparams.py` — distributional decoders.
  - `nn_decoder/run_fixed_hyperparams_choice.py` — choice control.
  - `nn_decoder/run_fixed_recovery.py` — architecture-level recovery
    crossover.
  - `nn_decoder/run_experiment_v26.py` — single-animal pipeline driver.

## Core Connections

| From | To | Content |
| :--- | :--- | :--- |
| `experiment_code/` | `preprocessing/` | Raw `.mat` and TIFF files |
| `preprocessing/` | `ideal_observer/` | Aligned kinematics (`unifiedData`) |
| `preprocessing/` | `nn_decoder/` | Deconvolved spike probabilities |
| `ideal_observer/` | `nn_decoder/` | `Q(theta)`, `L(theta)`, decision posterior |
| `glm_hmm/` | `nn_decoder/` | Trial inclusion/exclusion filters |
| `nn_decoder/` (base) | `nn_decoder/` (recovery) | `full_decoded` predictions as synthetic targets |
