# Module: Data Preprocessing

The preprocessing module transforms raw sensor data into a format suitable for neural decoding and behavioral modeling.

## Key Files

### Data Fusion
- **`createUnifiedSessionData.m`**: The central "aggregator." It performs the following:
    1.  Loads neural data (deltaF and spike probabilities).
    2.  Loads VR behavior logs.
    3.  Finds trial boundaries using pixel intensity (from the imaging metadata).
    4.  Processes pupil video to extract resampled pupil radius.
    5.  Bins neural and behavioral data into consistent time/position windows.
    6.  Saves a `unifiedData` structure.

### Spike Inference
- **`cascade_spkprediction.py`**: Python wrapper for the CASCADE spike deconvolution algorithm. It converts calcium fluorescence ($\Delta F/F$) into spike probabilities optimized for the jGCaMP8 indicator.

### Multi-Animal Analysis
- **`VR_multi_animal_analysis.m`**: High-level script to aggregate results across multiple animals and sessions.
- **`VR_multi_animal_plotting.m`**: Extensive plotting library for visualizing population-level behavioral and neural metrics.

## Connections
- **Input**: Raw `.mat` files from `experiment_code/` and imaging outputs (TIFFs processed via `rochefort_tools`).
- **Output**: The `unifiedData` structure is the foundation for the [Ideal Observer](Module_IdealObserver.md) and [Neural Decoder](Module_NeuralDecoder.md) modules.
