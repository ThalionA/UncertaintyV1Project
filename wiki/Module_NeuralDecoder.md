# Module: Neural Decoding (PPC vs SBC)

This is the core computational module where we test the two competing theories of neural uncertainty: **Probabilistic Population Codes (PPC)** and **Sampling-Based Codes (SBC)**.

## Key Files

### Core Pipeline
- **`run_experiment_v26.py`**: The main entry point for the decoding pipeline. It handles:
    - Loading aligned spikes and normative targets.
    - Training the two architecture types (PPC and SBC).
    - Managing hyperparameters and cross-validation splits.
    - Running "Recovery" experiments (using model predictions as new ground truth).
- **`neural_network_classifier_v26.py`**: Contains the PyTorch model definitions for the neural networks used in decoding.
- **`neural_dataset.py`**: Custom PyTorch `Dataset` class for handling the 3D neural activity tensors (Neurons x Trials x Time).
- **`utils_v26.py`**: Utility functions for z-scoring, temporal binning, and target generation (PPC/SBC specific mappings).

### Analysis & Plotting
- **`pca_visualisation_v26.py`**: Extracts the principal components of the population activity and visualizes the "uncertainty manifold."
- **`results_deepdive.py`**: Detailed post-hoc analysis of model performance across different stimulus conditions.
- **`plot_decoder_results_v26.py`**: Generates summary plots of KL divergence and decoding error.
- **`plot_sample_dynamics.py`**: Specifically analyzes the temporal dynamics of the SBC (sampling) models.

## Theoretical Comparison
- **Spatial Architecture (PPC)**: Assumes uncertainty is encoded in the pattern of activity across the population at a single moment.
- **Temporal Architecture (SBC)**: Assumes uncertainty is encoded in the variability of neural responses over time (samples).

## Connections
- **Input**: Deconvolved spikes (from [Preprocessing](Module_Preprocessing.md)) and Normative Uncertainty (from [Ideal Observer](Module_IdealObserver.md)).
- **Output**: Model comparisons that provide evidence for how V1 represents uncertainty.
