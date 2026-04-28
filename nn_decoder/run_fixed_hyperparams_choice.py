import os
import scipy.io as sio
import traceback

# Import the existing pipeline
from run_experiment_v26 import run_animal_decoder

def run_choice_analysis():
    print("Starting Choice Decoder (Detection) analysis...")
    
    splits = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']
    
    for split_type in splits:
        config = {
            "target_source": "real",
            "time_window": "half",
            "bin_size_ms": 100,
            "split_type": split_type,
            "which_model": "detection", # Direct Choice Model
            "hidden_sizes": [16],
            "activation_function": "tanh",
            "weight_initialization": "xavier_uniform",
            "custom_loss_func": "CE", # Cross Entropy for binary choice
            "entropy_lambda": 3e3,
            "learning_rate": 0.005,
            "optimizer_type": "adam",
            "momentum": 0.9,
            "num_epochs": 50,
            "minibatch_size": 16,
            "REP": 5
        }
        
        print(f"\n" + "="*60)
        print(f"RUNNING CHOICE DECODER SPLIT: {split_type}")
        print("="*60)
        
        all_mice_results = {}
        success = True
        
        for mouse_id in [0, 1, 2, 3, 4, 5]:
            print(f"  --> Processing Mouse {mouse_id}...")
            try:
                animal_results = run_animal_decoder(config, mouse_id)
                all_mice_results[f"mouse_{mouse_id}"] = animal_results
            except Exception as e:
                print(f"  [!] Failed for Mouse {mouse_id}: {e}")
                traceback.print_exc()
                success = False
                break 
                
        if success:
            save_name = f"population_results_fixed_choice_{split_type}.mat"
            sio.savemat(save_name, {'results': all_mice_results, 'config': config})
            print(f"Saved population results to {save_name}")
        else:
            print(f"Failed to complete split: {split_type}")

if __name__ == "__main__":
    run_choice_analysis()
