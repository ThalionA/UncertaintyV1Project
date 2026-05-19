"""Driver for the full set of fixed-hyperparam decoder plots.

Per-mouse plots always show every loaded mouse; aggregate (grand-average)
plots can be restricted to a subset via `aggregate_mouse_ids` (whitelist)
or `exclude_from_aggregate` (blacklist, convenient for "all but X"). The
underlying plotting helpers in `decoder_plotting_utils.py` are unchanged
— this driver just hands them filtered copies of the results dicts and
chdirs into `output_dir` so different aggregation runs don't overwrite
each other's SVGs (the helpers all save with hardcoded relative names).

Examples
--------
    # Default: every loaded mouse contributes to all plots
    generate_all_plots()

    # All animals but the third (id=2) in the aggregates,
    # but per-mouse plots still cover everyone
    generate_all_plots(exclude_from_aggregate=[2],
                        output_dir="FixedPlots_excl_2")

    # Only mice 0 and 4 contribute to the aggregates
    generate_all_plots(aggregate_mouse_ids=[0, 4],
                        output_dir="FixedPlots_only_0_4")
"""

import os
import copy

import decoder_plotting_utils as plot_utils


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _mouse_key(m):
    """Coerce a mouse identifier (int or str) into the canonical
    `mouse_<n>` key used by the .mat exports."""
    if isinstance(m, int):
        return f"mouse_{m}"
    if isinstance(m, str) and m.startswith("mouse_"):
        return m
    if isinstance(m, str) and m.isdigit():
        return f"mouse_{m}"
    raise ValueError(f"Cannot interpret mouse identifier: {m!r}")


def filter_results_by_mouse(results, include=None, exclude=None):
    """Return a new results dict with only the chosen mice in
    ``results[split]['results']``. The rest of the structure is shared
    by reference (no deep copy of the heavy arrays)."""
    if include is not None and exclude is not None:
        raise ValueError("Pass include OR exclude, not both.")
    if results is None:
        return None
    out = {}
    for split, split_data in results.items():
        if not isinstance(split_data, dict) or 'results' not in split_data:
            out[split] = split_data
            continue
        per_mouse = split_data['results']
        keys = set(per_mouse.keys())
        if include is not None:
            keep = {_mouse_key(m) for m in include} & keys
        elif exclude is not None:
            keep = keys - {_mouse_key(m) for m in exclude}
        else:
            keep = keys
        new_split = copy.copy(split_data)  # shallow
        new_split['results'] = {k: per_mouse[k] for k in per_mouse
                                if k in keep}
        out[split] = new_split
    return out


def _loaded_mouse_ids(results):
    """Sorted integer ids of mice present in any split of `results`."""
    ids = set()
    if results is None:
        return []
    for split_data in results.values():
        if isinstance(split_data, dict) and 'results' in split_data:
            for k in split_data['results'].keys():
                try:
                    ids.add(int(k.split('_')[1]))
                except (IndexError, ValueError):
                    continue
    return sorted(ids)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def generate_all_plots(aggregate_mouse_ids=None, exclude_from_aggregate=None,
                       output_dir="figures/decoder_fixed_plots", io_path=None,
                       include_stim_mean=True):
    """Run the full plotting suite.

    Parameters
    ----------
    aggregate_mouse_ids : iterable of int or None
        Whitelist for the aggregate (grand-average) plots. Per-mouse
        plots still show every loaded mouse.
    exclude_from_aggregate : iterable of int or None
        Blacklist for aggregate plots. Convenient for the common
        "all but X" case. Mutually exclusive with `aggregate_mouse_ids`.
    output_dir : str
        Directory the SVGs are written to. Created if missing. The
        plotting helpers in `decoder_plotting_utils` accept an
        `out_dir` kwarg and we pass `output_dir` through to each of
        them — no `chdir` involved. Use a fresh directory per
        aggregation choice if you want to keep multiple runs without
        clobbering each other.
    """
    if aggregate_mouse_ids is not None and exclude_from_aggregate is not None:
        raise ValueError("Pass aggregate_mouse_ids OR exclude_from_aggregate, not both.")

    print("Loading all results (Perception, Likelihood, Decision, Choice)...")
    splits = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']

    perc_results = plot_utils.load_results_dict("population_results_fixed_hyperparams", splits)
    lik_results = plot_utils.load_results_dict("population_results_fixed_likelihood", splits)
    dec_results = plot_utils.load_results_dict("population_results_fixed_decision", splits)
    choice_results = plot_utils.load_results_dict("population_results_fixed_choice", splits)

    # Optional: stim-mean baseline for the neurometric overlay. Only the
    # Q target's stim-mean files are used here (post_s_marginal -> P(Go))
    stim_mean_results = None
    if include_stim_mean:
        stim_mean_results = plot_utils.load_results_dict("population_results_stim_mean_Q", splits)
        if not stim_mean_results:
            print("[!] No stim-mean files found; neurometric plot will skip the stim-mean trace.")
            stim_mean_results = None

    if not perc_results:
        print("No Perception results found! Make sure run_fixed_hyperparams.py has completed successfully.")
        return

    loaded = _loaded_mouse_ids(perc_results)
    if aggregate_mouse_ids is not None:
        agg_ids = sorted(set(loaded) & set(aggregate_mouse_ids))
    elif exclude_from_aggregate is not None:
        agg_ids = sorted(set(loaded) - set(exclude_from_aggregate))
    else:
        agg_ids = list(loaded)

    if not agg_ids:
        print(f"[!] Aggregation set is empty (loaded mice: {loaded}). Nothing to do.")
        return

    print(f"Loaded mice (per-mouse plots): {loaded}")
    print(f"Aggregation subset (grand plots): {agg_ids}")
    if set(agg_ids) != set(loaded):
        excluded = sorted(set(loaded) - set(agg_ids))
        print(f"  → excluded from aggregates: {excluded}")

    # Filtered copies for aggregate plots. Per-mouse plots use the originals.
    if set(agg_ids) == set(loaded):
        agg_perc = perc_results
        agg_lik = lik_results
        agg_dec = dec_results
        agg_choice = choice_results
    else:
        agg_perc = filter_results_by_mouse(perc_results, include=agg_ids)
        agg_lik = filter_results_by_mouse(lik_results, include=agg_ids)
        agg_dec = filter_results_by_mouse(dec_results, include=agg_ids)
        agg_choice = filter_results_by_mouse(choice_results, include=agg_ids)

    all_target_results = {'Perception': agg_perc}
    if agg_lik:
        all_target_results['Likelihood'] = agg_lik
    if agg_dec:
        all_target_results['Decision'] = agg_dec

    os.makedirs(output_dir, exist_ok=True)
    out_abs = os.path.abspath(output_dir)
    print(f"\nWriting SVGs to {out_abs}")

    # ---- Aggregate plots use the filtered results ----
    print("\n1. Generating Normalized Performance Bars...")
    plot_utils.plot_normalized_performance_with_lines(agg_perc, splits, out_dir=output_dir)

    print("2. Generating Ambiguity Heatmaps...")
    for split in splits:
        if split in agg_perc:
            plot_utils.plot_ambiguity_heatmaps(agg_perc, split=split, out_dir=output_dir)

    print("3. Generating Orientation Performance Lines...")
    plot_utils.plot_orientation_performance(agg_perc, splits, out_dir=output_dir)

    print("4. Generating Temporal Dynamics trajectories...")
    for split in splits:
        if split in agg_perc:
            plot_utils.plot_temporal_dynamics(agg_perc, split=split, out_dir=output_dir)

    print("5. Generating Neurometric vs Psychometric Curves (with Direct Choice)...")
    plot_utils.plot_neurometric_curves(agg_perc, agg_choice, splits, out_dir=output_dir)
    if io_path is not None or stim_mean_results is not None:
        print("5a. Generating Neurometric Curves with lapse correction / stim-mean baseline...")
        agg_stim_mean = (filter_results_by_mouse(stim_mean_results, include=agg_ids)
                         if (stim_mean_results and set(agg_ids) != set(loaded))
                         else stim_mean_results)
        plot_utils.plot_neurometric_curves(
            agg_perc, agg_choice, splits,
            io_path=io_path, stim_mean_results=agg_stim_mean,
            out_dir=output_dir,
        )

    print("7. Generating Multi-Target Comparison (Perception vs Likelihood vs Decision)...")
    plot_utils.plot_multi_target_comparison(all_target_results, splits, out_dir=output_dir)

    print("8. Generating Raw Posterior Examples...")
    plot_utils.plot_posterior_examples_and_averages(agg_perc, splits=['stratified_balanced'], out_dir=output_dir)

    # ---- Per-mouse plots use ALL loaded mice ----
    print("\n1b. Generating Per-Mouse Performance Bars (all loaded mice)...")
    plot_utils.plot_per_mouse_performance_with_stats(perc_results, splits, out_dir=output_dir)

    print("5b. Generating Per-Mouse Neurometric Curves (all loaded mice)...")
    plot_utils.plot_neurometric_curves_per_mouse(perc_results, choice_results, splits, out_dir=output_dir)
    if io_path is not None or stim_mean_results is not None:
        print("5b. Generating Per-Mouse Neurometric (lapse / stim-mean overlays)...")
        plot_utils.plot_neurometric_curves_per_mouse(
            perc_results, choice_results, splits,
            io_path=io_path, stim_mean_results=stim_mean_results,
            out_dir=output_dir,
        )

    print("6. Generating Within-Mouse Statistics (all loaded mice)...")
    plot_utils.calculate_within_mouse_stats(perc_results, splits)

    print(f"\nAll visualizations generated. SVGs in {out_abs}")


if __name__ == "__main__":
    # All animals but the third (id=2) for the aggregates;
    # per-mouse plots still cover everyone
    generate_all_plots()
