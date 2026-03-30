import os
import json
import itertools
import traceback

# Import your existing pipeline
from run_experiment_v26 import run_animal_decoder

TRACKER_FILE = 'experiment_tracker.json'

def flatten_config_for_legacy_pipeline(json_config):
    """
    Translates the nested JSON dictionary back into the flat dictionary 
    that run_animal_decoder() expects.
    """
    flat_config = {}
    flat_config.update(json_config['data_params'])
    flat_config.update(json_config['model_params'])
    flat_config.update(json_config['training_params'])
    return flat_config

def generate_master_grid():
    """Generates the combinatorial grid and incrementally updates the JSON tracker."""
    print("Generating/Updating Master Experiment Tracker...")
    
    # 1. Define the focused sweep space for the new batch
    grid_space = {
        'target_source': ['real'], 
        'time_window': ['full', 'half'], # Restricted to full and half
        'bin_size_ms': [50, 100, 200, 250],
        'split_type': ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion'], # New generalization splits
        'which_model': ['perception'],
        'hidden_sizes': [[16]],
        'activation_function': ['relu'],
        'weight_initialization': ['xavier_uniform'], 
        'custom_loss_func': ['PCA'], # Restricted to PCA
        'entropy_lambda': [0.01, 0.05, 0.1, 0.2], # Expanded lambda steps
        'learning_rate': [0.001],
        'optimizer_type': ['adam'],
        'momentum': [0.9],
        'num_epochs': [40],
        'minibatch_size': [32],
        'REP': [5]
    }
    
    # 2. Generate all unique combinations
    keys, values = zip(*grid_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 3. Load existing tracker to prevent overwriting
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            master_list = json.load(f)
        print(f"Loaded existing tracker with {len(master_list)} experiments.")
        next_id = max([exp['experiment_id'] for exp in master_list]) + 1 if master_list else 0
    else:
        master_list = []
        next_id = 0

    # Helper function to check if a configuration is already in the tracker
    def combo_exists(new_combo, existing_list):
        for exp in existing_list:
            flat_existing = flatten_config_for_legacy_pipeline(exp)
            match = True
            for k, v in new_combo.items():
                if flat_existing.get(k) != v:
                    match = False
                    break
            if match:
                return True
        return False

    # 4. Append ONLY missing combinations
    added_count = 0
    for combo in combinations:
        
        # Skip invalid combinations
        if combo['custom_loss_func'] == 'PCA' and combo['which_model'] == 'detection':
            continue
            
        # If it doesn't exist, build it and add it
        if not combo_exists(combo, master_list):
            c_type = 'perc_post' if combo['which_model'] == 'perception' else 'dec_post'
            
            config = {
                "experiment_id": next_id,
                "status": "pending",  
                
                "data_params": {
                    "target_source": combo['target_source'],
                    "time_window": combo['time_window'],
                    "bin_size_ms": combo['bin_size_ms'],
                    "split_type": combo['split_type']
                },
                "model_params": {
                    "which_model": combo['which_model'],
                    "certainty_type": c_type,
                    "hidden_sizes": combo['hidden_sizes'],
                    "activation_function": combo['activation_function'],
                    "weight_initialization": combo['weight_initialization']
                },
                "training_params": {
                    "custom_loss_func": combo['custom_loss_func'],
                    "entropy_lambda": combo['entropy_lambda'],
                    "learning_rate": combo['learning_rate'],
                    "optimizer_type": combo['optimizer_type'],
                    "momentum": combo['momentum'],
                    "num_epochs": combo['num_epochs'],
                    "minibatch_size": combo['minibatch_size'],
                    "REP": combo['REP']
                }
            }
            master_list.append(config)
            next_id += 1
            added_count += 1
            
    # 5. Save back to disk
    with open(TRACKER_FILE, 'w') as f:
        json.dump(master_list, f, indent=4)
        
    print(f"Successfully added {added_count} NEW pending configurations to the tracker.")
    
    
    
def run_pending_experiments():
    """Reads the tracker, runs pending jobs, and updates their status."""
    import scipy.io as sio 
    
    with open(TRACKER_FILE, 'r') as f:
        master_list = json.load(f)
        
    pending_experiments = [exp for exp in master_list if exp['status'] == 'pending']
    print(f"\nFound {len(pending_experiments)} pending experiments out of {len(master_list)} total.")
    
    for exp in pending_experiments:
        exp_id = exp['experiment_id']
        flat_config = flatten_config_for_legacy_pipeline(exp)
        
        print(f"\n" + "="*60)
        print(f"RUNNING EXPERIMENT {exp_id}")
        print(f"Target: {flat_config['target_source']} | Window: {flat_config['time_window']} | Bin: {flat_config['bin_size_ms']}ms | Split: {flat_config['split_type']}")
        print("="*60)
        
        # 1. Mark as running & save immediately (prevents cluster race conditions)
        exp['status'] = 'running'
        with open(TRACKER_FILE, 'w') as f: json.dump(master_list, f, indent=4)
            
        all_mice_results = {}
        success = True
        
        for mouse_id in [0, 1, 2, 3, 4, 5]:
            print(f"  --> Processing Mouse {mouse_id}...")
            try:
                # ---> THE CORE PIPELINE CALL <---
                animal_results = run_animal_decoder(flat_config, mouse_id)
                all_mice_results[f"mouse_{mouse_id}"] = animal_results
            except Exception as e:
                print(f"  [!] Failed for Mouse {mouse_id}: {e}")
                traceback.print_exc()
                success = False
                break 
                
        if success:
            # Save the results
            save_name = f"population_results_config_{exp_id}.mat"
            sio.savemat(save_name, {'results': all_mice_results, 'config': flat_config})
            print(f"Saved population results to {save_name}")
            
            # Mark as completed
            exp['status'] = 'completed'
        else:
            # Mark as failed so it doesn't block the queue
            exp['status'] = 'failed' 
            
        # 3. Save updated status back to JSON
        with open(TRACKER_FILE, 'w') as f:
            json.dump(master_list, f, indent=4)

if __name__ == "__main__":
    generate_master_grid()
    run_pending_experiments()