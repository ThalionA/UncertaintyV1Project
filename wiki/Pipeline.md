# Pipeline Overview

The UncertaintyV1 pipeline is a multi-stage process that bridges experimental biology and computational neuroscience.

## Data Flow

### Stage 1: Experimental Acquisition
- **Input**: Mouse behavior in a virtual reality (VR) corridor.
- **Output**: 
    - Raw 2-photon imaging TIFFs.
    - VR Logs (`.mat`) containing position, velocity, and licks.
    - Arduino logs for hardware synchronization.
- **Tools**: MATLAB, ViRMEn, Psychtoolbox.

### Stage 2: Preprocessing & Data Fusion
- **Input**: Raw imaging and VR logs.
- **Process**:
    - Motion correction (NoRMCorre).
    - Segmentation (Cellpose).
    - Neuropil subtraction (FISSA).
    - Spike deconvolution (CASCADE).
    - Alignment of neural data to VR behavioral timestamps.
- **Output**: `UnifiedSessionData` (structured MATLAB files containing aligned spikes, licks, velocity, and pupil data).
- **Primary Script**: `preprocessing/createUnifiedSessionData.m`.

### Stage 3: Behavioural Inference (Ideal Observer)
- **Input**: `UnifiedSessionData`.
- **Process**: Fits a generative model to the mouse's velocity and lick rates to infer the "Ideal Observer's" internal posterior distribution over stimulus orientation.
- **Output**: Trial-by-trial estimates of **Perceptual Uncertainty** (width of the posterior) and **Decision Uncertainty** (entropy of the choice).
- **Primary Script**: `ideal_observer/ideal_observer_hierarchical_fitting_v2.m`.

### Stage 4: Latent State Discovery (GLM-HMM)
- **Input**: Behavioral logs.
- **Process**: Identifies discrete cognitive states (e.g., "Engaged", "Biased", "Disengaged") using a Hidden Markov Model with GLM observations.
- **Output**: State labels for every trial.
- **Usage**: Used to filter out trials where the mouse is not performing the task optimally, ensuring the neural decoder is trained on high-fidelity data.
- **Primary Script**: `glm_hmm/gonogo_glm_hmm_global_v4.py`.

### Stage 5: Neural Decoding (PPC vs SBC)
- **Input**: Aligned spikes (from Stage 2) and Normative Uncertainty (from Stage 3).
- **Process**:
    - Trains PyTorch ANNs to decode uncertainty from population activity.
    - Compares two architectures:
        - **Spatial (PPC)**: Decodes from instantaneous population snapshots.
        - **Temporal (SBC)**: Decodes from temporal fluctuations (variance across samples).
- **Output**: Model performance metrics (KL Divergence, MSE), cross-validation results, and "Recovery" experiments.
- **Primary Script**: `nn_decoder/run_experiment_v26.py`.

## Core Connections

| From | To | Content |
| :--- | :--- | :--- |
| `experiment_code/` | `preprocessing/` | Raw `.mat` and TIFF files |
| `preprocessing/` | `ideal_observer/` | Aligned kinematics (`unifiedData`) |
| `preprocessing/` | `nn_decoder/` | Deconvolved spike probabilities |
| `ideal_observer/` | `nn_decoder/` | Normative uncertainty targets |
| `glm_hmm/` | `nn_decoder/` | Trial inclusion/exclusion filters |
