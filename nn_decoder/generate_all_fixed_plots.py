import os
import decoder_plotting_utils as plot_utils

def generate_all_plots():
    print("Loading all results (Perception, Likelihood, Choice)...")
    
    splits = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']
    
    perc_results = plot_utils.load_results_dict("population_results_fixed_hyperparams", splits)
    lik_results = plot_utils.load_results_dict("population_results_fixed_likelihood", splits)
    dec_results = plot_utils.load_results_dict("population_results_fixed_decision", splits)
    choice_results = plot_utils.load_results_dict("population_results_fixed_choice", splits)
    
    if not perc_results:
        print("No Perception results found! Make sure run_fixed_hyperparams.py has completed successfully.")
        return
        
    # Collect all target-type results into a dict for multi-target plotting
    # Note: Decision uses MSE (not PCA), so the comparison plot shows
    # normalised loss within each target type's own metric
    all_target_results = {'Perception': perc_results}
    if lik_results:
        all_target_results['Likelihood'] = lik_results
    if dec_results:
        all_target_results['Decision'] = dec_results
    
    print("\n1. Generating Normalized Performance Bars...")
    plot_utils.plot_normalized_performance_with_lines(perc_results, splits)
    
    print("1b. Generating Per-Mouse Performance Bars...")
    plot_utils.plot_per_mouse_performance_with_stats(perc_results, splits)
    
    print("2. Generating Ambiguity Heatmaps...")
    for split in splits:
        if split in perc_results:
            plot_utils.plot_ambiguity_heatmaps(perc_results, split=split)
            
    print("3. Generating Orientation Performance Lines...")
    plot_utils.plot_orientation_performance(perc_results, splits)
    
    print("4. Generating Temporal Dynamics trajectories...")
    for split in splits:
        if split in perc_results:
            plot_utils.plot_temporal_dynamics(perc_results, split=split)
            
    print("5. Generating Neurometric vs Psychometric Curves (with Direct Choice)...")
    plot_utils.plot_neurometric_curves(perc_results, choice_results, splits)
    
    print("5b. Generating Per-Mouse Neurometric Curves...")
    plot_utils.plot_neurometric_curves_per_mouse(perc_results, choice_results, splits)
    
    print("6. Generating Within-Mouse Statistics...")
    plot_utils.calculate_within_mouse_stats(perc_results, splits)
    
    print("7. Generating Multi-Target Comparison (Perception vs Likelihood)...")
    plot_utils.plot_multi_target_comparison(all_target_results, splits)
    
    print("8. Generating Raw Posterior Examples...")
    plot_utils.plot_posterior_examples_and_averages(perc_results, splits=['stratified_balanced'])
    
    print("\nAll visualizations generated successfully! Check the current directory for the .svg files.")

if __name__ == "__main__":
    generate_all_plots()