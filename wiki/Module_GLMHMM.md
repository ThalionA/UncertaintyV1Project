# Module: State Discovery (GLM-HMM)

The GLM-HMM module is used to identify latent cognitive states that influence the mouse's decision-making process.

## Key Files

### State Inference
- **`gonogo_glm_hmm_global_v4.py`**: Implementation using the `ssm` (State Space Models) library. It fits a Hidden Markov Model where the observations are Go/No-Go choices modeled by a Generalized Linear Model (GLM).
- **`VR_GoNoGo_GLMHMM_global.m`**: MATLAB wrapper for organizing data to be sent to the Python HMM script.

### Integration
- **`Integrate_GLMHMM_Analysis.m`**: Merges the inferred state labels back into the main project datasets.

## Why use GLM-HMM?
Mice are not always "engaged" in the task. They may have:
1.  **Engaged State**: Choice is driven by the visual stimulus.
2.  **Biased State**: Choice is driven by history (e.g., repeating the previous choice).
3.  **Disengaged State**: Mouse is lumping or not paying attention.

By identifying these states, we can filter our neural decoding analysis to only include "Engaged" trials, reducing noise and improving model interpretability.

## Connections
- **Input**: Choice behavior (Lick/No-Lick) and stimulus conditions.
- **Output**: State probabilities per trial, used as a filter for the [Neural Decoder](Module_NeuralDecoder.md).
