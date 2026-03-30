import os
import h5py
import numpy as np
import scipy.io as sio

import yaml
import sys

# Add the directory containing cascade2p to the Python path
cascade_dir = r"C:\Users\theox\Desktop\Experiments\Cascade"
if cascade_dir not in sys.path:
    sys.path.append(cascade_dir)
    
from cascade2p import cascade

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_session_metadata(sesname, meta_dir):
    # Construct the full path to the session's YAML file
    yaml_file = os.path.join(meta_dir, "sessions", "Theo", f"{sesname}.yaml")
    
    # Read the YAML file
    
    with open(yaml_file, 'r') as file:
        metadata = yaml.safe_load(file)
    
    
    longrecs = []
    
    # Extract longrecs from the first (and assumed only) FOV
    recordings = metadata.get("recordings", {})
    if recordings:
        first_animal = next(iter(recordings.values()))
        fovs = first_animal.get("fovs", [])
        if fovs:
            first_fov = fovs[0]
            rec_groups = first_fov.get("recGroups", {})
            for group in rec_groups.values():
                longrecs.extend(group.get("longrecs", []))
    
    return longrecs

def load_neurons_x_time(file_path):
    """
    Load data from a MATLAB file (.mat).
    Attempts HDF5 (MATLAB v7.3) first, falls back to scipy.io (v7/v6).
    """
    traces = None
    
    try:
        # Attempt 1: Try loading as v7.3 (HDF5)
        with h5py.File(file_path, 'r') as file:
            if 'deltaF' in file:
                traces = np.array(file['deltaF'])
            else:
                print(f"Warning: 'deltaF' not found in {file_path} (HDF5). Available keys: {list(file.keys())}")
                return None

    except OSError:
        # Attempt 2: h5py failed (likely v7.2 or older), try scipy.io
        try:
            mat_data = sio.loadmat(file_path)
            if 'deltaF' in mat_data:
                traces = mat_data['deltaF']
            else:
                print(f"Warning: 'deltaF' not found in {file_path} (scipy). Available keys: {list(mat_data.keys())}")
                return None
        except Exception as e:
            print(f"Error loading file {file_path}: Failed with both h5py and scipy.\nDetails: {e}")
            return None

    except Exception as e:
        print(f"Unexpected error loading {file_path}: {e}")
        return None

    # Common processing: Ensure dimensions are Neurons x Time
    # (h5py usually reads MATLAB arrays transposed, scipy reads them as-is)
    if traces.shape[0] > traces.shape[1]:
        traces = traces.T

    return traces

def process_recording(working_dir, sesname, recording, model_name):
    """Process a single recording and save spike probabilities."""
    input_file = os.path.join(working_dir, sesname, "CaActivity", f"{recording}deltaF_fissa.mat")
    output_file = os.path.join(working_dir, sesname, "CaActivity", f"{recording}deltaF_fissa_spkprob.mat")
    
    # Load traces
    traces = load_neurons_x_time(input_file)
    if traces is None:
        print(f"Failed to load traces for {sesname}/{recording}. Skipping.")
        return False

    # Predict spikes
    spike_prob = cascade.predict(model_name, traces)

    # Save predictions
    sio.savemat(output_file, {'spike_prob': spike_prob})
    print(f"Saved predictions for {sesname}/{recording}")
    return True

def main():
    # Set up file paths and parameters
    
    os.chdir(r"C:\Users\theox\Desktop\Experiments\Cascade")

    working_dir = r"C:\Users\theox\Desktop\Experiments\Working"
    
    settingsFile = r'C:\Users\theox\Desktop\Experiments\rochefort-tools-updated\settings_files\analysis_settings_TA.yaml'
    projectFile = r"C:\Users\theox\Desktop\Experiments\NRLabMetadata\projects\projects_TA.yaml"
    
    # Read settings and project files
    settingsInfo = read_yaml(settingsFile)
    projMD = read_yaml(projectFile)
    
    meta_dir = settingsInfo['root_metadata_dir']
    projName = 'VRGoNoGo'
    
    # Get session list from project metadata
    sessionList = projMD[projName]['sessions']

    # frame_rate = 32.6825  # Hz
    model_name = 'GC8m_EXC_30Hz_smoothing25ms_high_noise'
    # model_name = 'GCaMP6f_mouse_30Hz_smoothing200ms'

    # Download model if not already available
    cascade.download_model(model_name, verbose=1)

    for sesname in sessionList:
        longrecs = get_session_metadata(sesname, meta_dir)
        
        for recording in longrecs:
            success = process_recording(working_dir, sesname, recording, model_name)
            if success:
                print(f"Completed: {sesname}/{recording}")
            else:
                print(f"Failed: {sesname}/{recording}")
    
    print("All processing complete")

if __name__ == "__main__":
    main()