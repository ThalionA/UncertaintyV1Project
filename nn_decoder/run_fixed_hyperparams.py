import os
import scipy.io as sio
import traceback

# Import the existing pipeline
from run_experiment_v26 import run_animal_decoder

def run_fixed_analysis():
    print("Starting fixed hyperparameter analysis...")
    
    splits = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']
    
    # We now run for 3 target types: perceptual posteriors, likelihoods, and decision posteriors
    # Each tuple: (which_model, save_prefix, loss_function)
    # Perception & Likelihood use PCA loss (91D distributions)
    # Decision uses MSE loss (2D soft probability — PCA is meaningless on 2D)
    target_types = [
        ('perception', 'population_results_fixed_hyperparams', 'PCA'),
        ('likelihood', 'population_results_fixed_likelihood',  'PCA'),
        ('decision',   'population_results_fixed_decision',    'MSE'),
    ]
    
    for which_model, save_prefix, loss_func in target_types:
        for split_type in splits:
            config = {
                "target_source": "real",
                "time_window": "half",
                "bin_size_ms": 100,
                "split_type": split_type,
                "which_model": which_model,
                "hidden_sizes": [16],
                "activation_function": "tanh",
                "weight_initialization": "xavier_uniform",
                "custom_loss_func": loss_func,
                "entropy_lambda": 1e-4,
                "learning_rate": 0.005,
                "optimizer_type": "adam",
                "momentum": 0.9,
                "num_epochs": 50,
                "minibatch_size": 32,
                "REP": 5
            }
            
            print(f"\n" + "="*60)
            print(f"RUNNING: {which_model.upper()} | SPLIT: {split_type}")
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
                save_name = f"{save_prefix}_{split_type}.mat"
                sio.savemat(save_name, {'results': all_mice_results, 'config': config})
                print(f"Saved population results to {save_name}")
            else:
                print(f"Failed to complete {which_model} / {split_type}")

if __name__ == "__main__":
    run_fixed_analysis()
