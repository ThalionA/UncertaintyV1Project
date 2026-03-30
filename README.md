# Representation of Perceptual Uncertainty in Mouse Primary Visual Cortex (V1)

This repository contains the experimental and computational pipeline investigating how perceptual and decision uncertainty are encoded in the primary visual cortex (V1) of mice. Specifically, we compare two leading theoretical frameworks: **Probabilistic Population Codes (PPC)** and **Sampling-Based Codes (SBC)**.

## Project Overview

The brain must infer the true state of the environment from noisy sensory signals. While it is widely accepted that the brain employs probabilistic inference, the exact neural algorithm remains debated. 

In this project, mice perform a virtual reality (VR) Go/No-Go visual discrimination task based on grating orientation. We extract trial-by-trial estimates of their internal uncertainty using a behavioural Ideal Observer model, and then train Artificial Neural Networks (ANNs) to decode this normative uncertainty directly from longitudinal 2-photon calcium imaging data (deconvolved to spikes) in V1.

### The Two Architectures Evaluated:
1. **Spatial Architecture (PPC):** Assumes uncertainty is represented by integrating population activity spatially across neurons and temporally across the stimulus epoch before generating a continuous probability distribution.
2. **Temporal Architecture (SBC):** Assumes the network approximates distributions sequentially via independent temporal samples, with uncertainty represented as variance across samples over time.

## Repository Structure

The codebase is organised modularly to separate experimental control, preprocessing, behavioural modelling, and neural decoding:

* **`experiment_code/`**: Custom ViRMEn-based VR engine (MATLAB) used for behavioural monitoring and rendering the linear corridor, alongside Arduino scripts for hardware control (rotary encoder, lick spout).
* **`preprocessing/`**: Pipeline for 2-photon calcium imaging data, primarily managed via the custom `rochefort_tools` suite. Includes wrappers for NoRMCorre (motion correction), Cellpose (ROI segmentation), FISSA (neuropil subtraction), and CASCADE (spike deconvolution optimised for jGCaMP8). Also utilises `roi_reg` for longitudinal ROI registration across days.
* **`glm_hmm/`**: Generalised Linear Model Hidden Markov Model implementation used to isolate engaged perceptual states from history-dependent or task-disengaged states.
* **`ideal_observer/`**: Generative behavioural modelling to extract trial-by-trial perceptual posteriors and decision uncertainty from continuous kinematic readouts (velocity and lick rates).
* **`nn_decoder/`**: The core PyTorch deep learning pipeline.
* **`documents/`**: Project notes, manuscript drafts, and figure assets.
* **`data/`**: *(Ignored in Git)* Local directory for `.mat` data dictionaries, raw TIFFs, and model checkpoints.

## Prerequisites and Dependencies

This project requires a hybrid MATLAB/Python computational environment.

### Python Environment
* **Deep Learning & Core Data Science**: `pytorch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
* **Behavioural Modelling**: `ssm` (State Space Models, required for GLM-HMM)
* **Calcium Preprocessing**: 
  * `rochefort_tools` (custom wrapper for NoRMCorre, FISSA, and Cellpose)
  * `cascade` (spike inference for jGCaMP8)
  * `roi_reg` (longitudinal ROI alignment)

### MATLAB Environment
* **Experimental Control & Stimulus Generation**: Base MATLAB, `ViRMEn` (Virtual Reality Mouse Engine), `Psychtoolbox`
* **Behavioural Optimisation**: `BADS` (Bayesian Adaptive Direct Search, used for fitting the Ideal Observer model)

## Data Availability

Due to file size constraints, raw calcium imaging TIFFs and the full aggregated `.mat` datasets are not stored in this repository.
