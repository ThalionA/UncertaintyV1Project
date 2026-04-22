# Module: Experimental Control

This module contains the code responsible for running the Virtual Reality (VR) environment and capturing mouse behavior.

## Key Files

### VR Experiment Scripts
- **`NewExperimentTheo25_full.m`**: The primary MATLAB script for the Go/No-Go visual discrimination task. It manages the trial structure, stimulus presentation (gratings), and reward delivery.
- **`VRHabituation.m`**: A simplified version of the corridor used for initial animal training and habituation to the treadmill.
- **`NewExperimentTheo25_full1_priorblocks.m`**: Version of the experiment with structured prior probability blocks to test expectation effects on uncertainty.

### Environment & Stimulus
- **`CreateGratings.m`**: Generates the visual grating stimuli with varying orientations and contrasts.
- **`createVRworlds_auto.m`**: Programmatically generates the 3D virtual corridor using the ViRMEn engine.

### Utilities
- **`changeExperimentCode.m`**: Utility to quickly swap experiment parameters across different versions.

## Connections
- **Hardware**: Communicates with Arduinos for lick detection (capacitive sensors) and rotary encoder (treadmill movement) readouts.
- **Data Output**: Saves trial-by-trial behavior into `vr_..._light.mat` files, which are later consumed by the [Preprocessing](Module_Preprocessing.md) module.
