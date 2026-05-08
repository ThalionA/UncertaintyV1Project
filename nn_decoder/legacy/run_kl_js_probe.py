"""Quick KL and JS loss probes on stratified_balanced only.

Mirrors run_fixed_hyperparams_loss_sweep.py but invokes the underlying
``run_loss_sweep`` helper with two extra losses (KL, JS) and only the
in-distribution split, since the Wasserstein result already covered the
OOD splits and showed loss has limited leverage.

Output:
    population_results_fixed_hyperparams_KL_stratified_balanced.mat
    population_results_fixed_hyperparams_JS_stratified_balanced.mat

Both filenames follow the same naming convention as the Wasserstein
sweep so ``compare_all_choice_methods`` can pick them up automatically.
Existing files are not overwritten.
"""

from run_fixed_hyperparams_loss_sweep import run_loss_sweep


def main():
    for loss in ('KL', 'JS'):
        print(f"\n=== Running loss = {loss} on stratified_balanced ===")
        run_loss_sweep(loss_func=loss,
                       splits=('stratified_balanced',))


if __name__ == "__main__":
    main()
