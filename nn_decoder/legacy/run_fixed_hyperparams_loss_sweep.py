"""Loss-function probe — does Wasserstein training change the picture?

Mirrors run_fixed_hyperparams.py structurally but trains only the
perceptual-posterior target Q(theta) under a single non-default loss
(Wasserstein-1 on the 91-bin orientation grid by default), across all
three production splits. Output filenames are tagged with the loss so
existing population_results_fixed_*.mat files are NOT overwritten.

Why this script exists
----------------------
The production decoder uses PCA-weighted Euclidean loss. PCA is fit on
per-condition averaged training Q's, so the leading PCs encode
between-condition structure. Variance-weighting the squared error means
matching those leading PCs dominates the loss; lower PCs (which contain
within-condition shape variation) are heavily down-weighted.

This biases the optimiser toward the per-condition mean — exactly what
the stim-mean baseline already is, by construction. Wasserstein-1 on a
1-D ordinal grid (orientation) is the natural alternative: it cares
about how far mass has to *travel* between distributions, which rewards
small angular shifts of the peak that PCA-weighted Euclidean smooths
over. If V1 trial-by-trial activity carries useful within-condition
signal that PCA-weighted training discards, Wasserstein should expose it.

How to read the output
----------------------
1. The per-mouse choice NLL through the IO Stage 2 lapse psychometric
   (compute via io_coherence on the Wasserstein-trained outputs).
   Compare against the stim-mean baseline.
2. If Wasserstein-trained PPC/SBC beat stim-mean on choice NLL, the
   loss was the bottleneck.
3. If they don't, V1 doesn't carry within-condition Q info regardless
   of loss — an information-limit result.

Computational note
------------------
6 mice * 3 splits * 4 archs (PPC, SBC, PPC-shf, SBC-shf) * REP=5 inits
* 50 epochs * 1 target. Same scale as one target's worth of the
production refit; figure on the order of an hour on GPU/MPS.
"""

import os
import scipy.io as sio
import traceback

# Reuse the production single-animal driver verbatim — it already
# handles every loss type via the loss_func config field, including
# Wasserstein. We just hand it a different config and a different
# save prefix.
from run_experiment import run_animal_decoder


# Loss being probed. Change to 'KL' or 'JS' to sweep additional losses;
# they map straight through to nn_classifier.custom_loss_all_H.
LOSS_FUNC = 'Wasserstein'


def run_loss_sweep(loss_func=LOSS_FUNC,
                   splits=('stratified_balanced', 'generalize_contrast',
                           'generalize_dispersion'),
                   mouse_ids=tuple(range(6)),
                   save_prefix=None):
    """Train PPC and SBC on the Q target across the given splits under
    the chosen loss. Saves per-split .mat files matching the production
    layout but with the loss name appended to the prefix.
    """
    if save_prefix is None:
        save_prefix = f'population_results_fixed_hyperparams_{loss_func}'

    print(f"Starting loss-sweep run with loss = {loss_func!r}")
    print(f"  splits     : {splits}")
    print(f"  mouse_ids  : {tuple(mouse_ids)}")
    print(f"  output tag : {save_prefix}_<split>.mat")

    for split_type in splits:
        config = {
            "target_source":         "real",
            "time_window":           "half",
            "bin_size_ms":           100,
            "split_type":            split_type,
            "which_model":           "perception",   # Q target only
            "hidden_sizes":          [32],
            "activation_function":   "tanh",
            "weight_initialization": "xavier_uniform",
            "custom_loss_func":      loss_func,
            "entropy_lambda":        3e-3,
            "learning_rate":         1e-3,
            "optimizer_type":        "adam",
            "momentum":              0.9,
            "num_epochs":            30,
            "minibatch_size":        16,
            "REP":                   5,
        }

        print("\n" + "=" * 60)
        print(f"RUNNING: Q ({loss_func}) | SPLIT: {split_type}")
        print("=" * 60)

        all_mice_results = {}
        success = True
        for mid in mouse_ids:
            print(f"  --> Processing Mouse {mid}...")
            try:
                animal_results = run_animal_decoder(config, mid)
                all_mice_results[f"mouse_{mid}"] = animal_results
            except Exception as exc:
                print(f"  [!] Failed for Mouse {mid}: {exc}")
                traceback.print_exc()
                success = False
                break

        if success:
            save_name = f"{save_prefix}_{split_type}.mat"
            sio.savemat(save_name, {'results': all_mice_results, 'config': config})
            print(f"Saved population results to {save_name}")
        else:
            print(f"Failed to complete {loss_func} / {split_type}; not writing.")


if __name__ == "__main__":
    run_loss_sweep()
