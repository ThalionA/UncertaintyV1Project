# Module: Behavioural Modelling (Ideal Observer)

This module implements a normative framework to estimate the mouse's internal uncertainty based on its observable behavior (kinematics).

## Key Files

### Fitting & Estimation
- **`ideal_observer_hierarchical_fitting_v2.m`**: Fits the generative model to behavioral data using Bayesian Adaptive Direct Search (BADS). It assumes the mouse is an "Ideal Observer" performing Bayesian inference on noisy orientation signals.
- **`SpatiotemporalHierarchicalFitting_ModelComparison.m`**: Compares different versions of the Ideal Observer (e.g., spatial vs. temporal integration of evidence) to see which best matches mouse behavior.

### Decoding Stimulus from Behavior
- **`Spatiotemporal_Decoding_Stimulus.m`**: Logic for decoding the actual stimulus orientation from the mouse's velocity and lick patterns.

### Visualisation
- **`velocity_heatmaps.m`**: Visualizes how mouse velocity profiles change as a function of stimulus uncertainty (dispersion/contrast).
- **`lick_heatmaps.m`**: Maps lick density across the corridor and during the grating presentation.

## Connections
- **Input**: Behavior from `unifiedData` (velocity, licks, stimulus identity).
- **Output**: Trial-by-trial **Normative Posteriors**. These serve as the "ground truth" labels for the [Neural Decoder](Module_NeuralDecoder.md).
