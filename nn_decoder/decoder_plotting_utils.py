import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import functools

import paths
from pca_loss import pca_distance, fit_loss


@functools.lru_cache(maxsize=None)
def _cached_raw_trials(mouse_id):
    """Return the per-trial metadata dict (orientation, contrast,
    dispersion, choice, velocity, lick_rate, true_orientation) for a
    mouse, cached by mouse_id.

    The underlying VR_Decoder_Data_Export.mat file is ~558 MB; calling
    ``load_vr_export`` once per (mouse, split) means a 6-mouse, 3-split
    plot triggers 18 separate disk reads (~1 minute, sometimes longer).
    The plotting helpers in this module only need the lightweight
    ``raw_trials`` dict — caching just that keeps the cache footprint
    tiny (~6 small dicts) while collapsing the 18 reads to 6.

    Imported lazily so ``utils`` (which pulls torch at module top)
    isn't required when this module is imported for non-plot helpers.
    """
    from utils import load_vr_export
    _, _, _, _, raw_trials = load_vr_export(mouse_id)
    return raw_trials


def set_style():
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# HELPER FUNCTIONS (STATS)
# ==========================================

def add_stat_annotation(ax, x1, x2, y, h, p_val):
    """ Helper to draw statistical significance brackets and stars. """
    if p_val < 0.001: star = '***'
    elif p_val < 0.01: star = '**'
    elif p_val < 0.05: star = '*'
    else: star = 'ns'
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
    ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color='black', fontsize=12)

# ==========================================
# DATA LOADING & DISTANCE METRICS
# ==========================================

def load_results_dict(base_name=None, splits=None):
    """Load one .mat per split for a given basename stem.

    Behaviour note: pre-paths.py this function resolved filenames against
    the current working directory. It now resolves against
    ``paths.LEGACY_FITS`` (which equals ``nn_decoder/`` in Phase A and
    will become ``data/processed/population_results_pre_refactor/`` after
    Phase C). Callers running from the nn_decoder/ directory see no
    change; callers running from elsewhere now get the expected files
    instead of "not found" warnings.
    """
    if base_name is None:
        base_name = paths.fit_stem('fixed', 'Q')
    if splits is None:
        splits = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']

    results = {}
    for split in splits:
        path = paths.fit_path_from_stem(base_name, split)
        filename = str(path)
        if path.exists():
            try:
                mat = sio.loadmat(filename, simplify_cells=True)
                results[split] = mat
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
        else:
            print(f"Warning: {filename} not found.")
    return results


def load_run_tree(run_name, slug, splits=None, results_root=None):
    """Load one .mat per split from the nested results tree
    ``<results_root>/<run_name>/<slug>/<split>.mat``.

    Companion to :func:`load_results_dict`. ``load_results_dict`` reads
    the legacy flat-named files (``population_results_*_<split>.mat``)
    written by ``run_fixed_hyperparams.py``; ``load_run_tree`` reads the
    nested tree written by ``training/run.py``. Both return the same
    shape — ``{split: scipy.io.loadmat output}`` — so downstream plotters
    in this module work unchanged on either input.

    Parameters
    ----------
    run_name : str
        Subdirectory under ``results_root`` (e.g. ``'post_fix_loadings_2026_05_17'``).
    slug : str
        Subdirectory under ``<results_root>/<run_name>`` encoding the
        (target, loss, time_window, bin_size_ms) tuple
        (e.g. ``'d_MSE_half_100ms'``).
    splits : sequence of str, optional
        Splits to load. Defaults to the three production splits.
    results_root : str or Path, optional
        Root of the results tree. Defaults to ``paths.RESULTS``.

    Returns
    -------
    dict
        Mapping ``{split: mat_dict}`` where ``mat_dict`` has the same
        top-level keys (``results``, ``config``, etc.) as
        :func:`scipy.io.loadmat` produces. Splits with missing files are
        skipped with a printed warning.
    """
    from pathlib import Path
    if splits is None:
        splits = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']
    if results_root is None:
        results_root = paths.RESULTS

    slug_dir = Path(results_root) / run_name / slug
    out = {}
    for split in splits:
        path = slug_dir / f'{split}.mat'
        filename = str(path)
        if path.exists():
            try:
                mat = sio.loadmat(filename, simplify_cells=True)
                out[split] = mat
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
        else:
            print(f"Warning: {filename} not found.")
    return out


def calc_pca_dist(p, q, pcs, evar):
    """Plotting-layer wrapper around ``pca_loss.pca_distance``.

    A missing PCA basis yields an all-NaN per-trial array, so the
    per-mouse plotting loops in this module (which reduce with
    ``np.nanmean``) skip that cell instead of raising. This is the
    *only* place that tolerates a missing basis by design; the strict,
    raising contract lives in ``pca_loss.pca_distance`` and is what the
    training loss and the test suite use. Note this never substitutes a
    *different metric* (MSE/CE) — an absent basis becomes NaN, never
    another number.
    """
    if pcs is None or (isinstance(pcs, (list, np.ndarray)) and len(pcs) == 0):
        return np.full(np.asarray(p).shape[0], np.nan)
    return pca_distance(p, q, pcs, evar)


# The PCA/MSE/CE fit-loss dispatcher now lives in ``pca_loss``. It is
# re-exported here under its historical name for external importers
# (recovery_sanity_check, plot_post_fix_performance) and the test suite.
# Note: the 'PCA' branch raises on a missing basis (see pca_loss.py) —
# unlike ``calc_pca_dist`` above, which is the NaN-tolerant plotting
# wrapper. A 'PCA' fit-loss request always has a basis in correct
# configurations, so a raise here surfaces a genuine misconfiguration.
calc_fit_loss = fit_loss


def get_mouse_pca_losses(res_dict, arch_key, target_key='target'):
    all_losses = []
    for m_id, m_data in res_dict['results'].items():
        dist = m_data['Dist']
        pcs = dist.get('pcs', None)
        evar = dist.get('explained_var', None)
        loss = calc_pca_dist(dist[arch_key][target_key], dist[arch_key]['decoded'], pcs, evar)
        all_losses.append(loss)
    return np.concatenate(all_losses) if all_losses else np.array([])

def get_mouse_trials(res_dict, split_type='stratified_balanced'):
    trials = {'contrast': [], 'dispersion': [], 'orientation': [], 'choice': []}
    
    from utils import get_stratified_train_test_indices, get_generalization_split_indices

    for m_id, m_data in res_dict['results'].items():
        mouse_idx = int(m_id.split('_')[1])
        raw_trials = _cached_raw_trials(mouse_idx)
        
        stimulus_conditions_full = np.array(list(zip(raw_trials['orientation'], raw_trials['contrast'], raw_trials['dispersion'])))
        unique_stimulus_categories, trial_categories_all = np.unique(stimulus_conditions_full, axis=0, return_inverse=True)
        
        if split_type == 'stratified_balanced':
            _, test_indices = get_stratified_train_test_indices(trial_categories_all, test_size=0.5, random_state=42)
        elif split_type in ['generalize_contrast', 'generalize_dispersion']:
            _, test_indices = get_generalization_split_indices(raw_trials, split_type=split_type, random_state=42)
            
        test_choices = raw_trials['choice'][test_indices]
        
        trials['contrast'].append(m_data['trials']['contrast'])
        trials['dispersion'].append(m_data['trials']['dispersion'])
        trials['orientation'].append(m_data['trials']['orientation'])
        trials['choice'].append(test_choices)
    
    return {k: np.concatenate(v) for k, v in trials.items()}

# ==========================================
# 1. NORMALIZED PERFORMANCE BARS (ACROSS MICE)
# ==========================================

def plot_normalized_performance_with_lines(perception_results, splits, out_dir=".",
                                             normalize=True):
    """Population spat / temp-full / temp-bins performance bars.

    ``normalize=True`` (default) divides each mouse's loss by its own
    shuffle control (fraction of chance, 1.0 = shuffle baseline).
    ``normalize=False`` plots the raw PCA loss with no shuffle division
    and writes to a ``*_raw`` filename — useful for seeing absolute
    spatial-vs-temporal differences that the normalisation can mask.
    """
    set_style()
    suffix = "" if normalize else "_raw"
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 6), sharey=True)
    if len(splits) == 1: axes = [axes]

    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split not in perception_results: continue

        res_dict = perception_results[split]
        mouse_spat, mouse_temp, mouse_temp_bins = [], [], []

        for m_id, m_data in res_dict['results'].items():
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)

            # Raw Means - No Filtering
            v_spat = np.nanmean(calc_pca_dist(dist['spat']['target'], dist['spat']['decoded'], pcs, evar))
            v_spat_shf = np.nanmean(calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar))

            v_temp = np.nanmean(calc_pca_dist(dist['temp']['target'], dist['temp']['decoded'], pcs, evar))
            v_temp_shf = np.nanmean(calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar))

            # Bins
            temp_samp = dist['temp']['decoded_samp']
            target = dist['temp']['target']
            if temp_samp.ndim == 3:
                T = temp_samp.shape[2]
                target_expanded = np.repeat(target[:, :, np.newaxis], T, axis=2)
                v_temp_bins = np.nanmean(calc_pca_dist(target_expanded, temp_samp, pcs, evar))
            else:
                v_temp_bins = np.nan

            if normalize:
                n_spat = v_spat / v_spat_shf if v_spat_shf > 0 else np.nan
                n_temp = v_temp / v_temp_shf if v_temp_shf > 0 else np.nan
                n_temp_bins = v_temp_bins / v_temp_shf if v_temp_shf > 0 else np.nan
            else:
                n_spat, n_temp, n_temp_bins = v_spat, v_temp, v_temp_bins

            mouse_spat.append(n_spat)
            mouse_temp.append(n_temp)
            mouse_temp_bins.append(n_temp_bins)

            ax.plot([0, 1, 2], [n_spat, n_temp, n_temp_bins], marker='o', color='gray', alpha=0.4, linewidth=1.5)

        x = np.arange(3)
        means = [np.nanmean(mouse_spat), np.nanmean(mouse_temp), np.nanmean(mouse_temp_bins)]
        sems = [stats.sem(mouse_spat, nan_policy='omit'), stats.sem(mouse_temp, nan_policy='omit'), stats.sem(mouse_temp_bins, nan_policy='omit')]

        ax.bar(x, means, width=0.5, yerr=sems, capsize=5, color=['darkorange', 'steelblue', 'lightblue'], alpha=0.7)
        if normalize:
            ax.axhline(1.0, color='red', linestyle='--', lw=2, label='Shuffle Baseline')

        # Stats
        try:
            _, p_val = stats.ttest_rel(mouse_spat, mouse_temp)
            y_max = max(means[0] + sems[0], means[1] + sems[1])
            add_stat_annotation(ax, 0, 1, y_max + 0.05, 0.03, p_val)
        except Exception:
            pass

        ax.set_title(split)
        ax.set_xticks(x)
        ax.set_xticklabels(['Spatial\n(PPC)', 'Temporal\nFull (SBC)', 'Temporal\nBins Avg'])
        if idx == 0:
            ax.set_ylabel("Normalized PCA Loss (Raw Mean ± SEM)" if normalize
                           else "Raw PCA Loss (Mean ± SEM)")
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='lower left')

    scale = "Normalized to shuffle" if normalize else "RAW (not shuffle-normalized)"
    fig.suptitle(f"Population Performance with Statistics (N=6 Mice) — {scale}",
                  fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"1_Normalized_Performance_Bars_Paired_Stats{suffix}.svg"),
                 bbox_inches='tight')
    plt.close()

# ==========================================
# 1B. PER-MOUSE PERFORMANCE BARS (WITHIN-MOUSE)
# ==========================================

def plot_per_mouse_performance_with_stats(perception_results, splits, out_dir=".",
                                            normalize=True):
    """Per-mouse spatial-vs-temporal performance bars.

    ``normalize=True`` (default) divides each mouse's per-trial loss by
    its own shuffle mean; ``normalize=False`` plots the raw PCA loss
    (no shuffle division, no 1.0 baseline) into a ``*_raw`` filename.
    """
    set_style()
    suffix = "" if normalize else "_raw"
    for split in splits:
        if split not in perception_results: continue
        res_dict = perception_results[split]

        mice_ids = list(res_dict['results'].keys())
        n_mice = len(mice_ids)

        fig, ax = plt.subplots(figsize=(1.5 * n_mice + 1, 5))
        x = np.arange(n_mice)
        width = 0.35

        for i, (m_id, m_data) in enumerate(res_dict['results'].items()):
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)

            spat_trials = calc_pca_dist(dist['spat']['target'], dist['spat']['decoded'], pcs, evar)
            temp_trials = calc_pca_dist(dist['temp']['target'], dist['temp']['decoded'], pcs, evar)
            spat_shf_trials = calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar)
            temp_shf_trials = calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar)

            shf_spat_mean = np.nanmean(spat_shf_trials)
            shf_temp_mean = np.nanmean(temp_shf_trials)

            if normalize:
                norm_spat = spat_trials / shf_spat_mean
                norm_temp = temp_trials / shf_temp_mean
            else:
                norm_spat = spat_trials
                norm_temp = temp_trials

            mean_s, sem_s = np.nanmean(norm_spat), stats.sem(norm_spat, nan_policy='omit')
            mean_t, sem_t = np.nanmean(norm_temp), stats.sem(norm_temp, nan_policy='omit')

            ax.bar(x[i] - width/2, mean_s, width, yerr=sem_s, capsize=3, color='darkorange', label='Spatial (PPC)' if i==0 else "")
            ax.bar(x[i] + width/2, mean_t, width, yerr=sem_t, capsize=3, color='steelblue', label='Temporal (SBC)' if i==0 else "")

            # Trial-by-trial paired t-test
            valid_mask = ~np.isnan(norm_spat) & ~np.isnan(norm_temp)
            if np.sum(valid_mask) > 0:
                try:
                    diffs = norm_spat[valid_mask] - norm_temp[valid_mask]
                    if np.any(diffs != 0):
                        _, p_val = stats.ttest_rel(norm_spat[valid_mask], norm_temp[valid_mask])
                        y_max = max(mean_s + sem_s, mean_t + sem_t)
                        add_stat_annotation(ax, x[i] - width/2, x[i] + width/2, y_max + 0.05, 0.03, p_val)
                except Exception as e:
                    print(f"Stats warning for Mouse {m_id}: {e}")
                    pass

        if normalize:
            ax.axhline(1.0, color='red', linestyle='--', label='Shuffle Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([f"M{m.split('_')[1]}" for m in mice_ids])
        ax.set_ylabel("Normalized PCA Loss (Raw Mean ± SEM)" if normalize
                       else "Raw PCA Loss (Mean ± SEM)")
        scale = "" if normalize else " — RAW"
        ax.set_title(f"Intra-Mouse Model Comparison ({split}){scale}")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        ylim = ax.get_ylim()
        if normalize:
            ax.set_ylim(0, max(1.2, ylim[1] + 0.15))
        else:
            ax.set_ylim(0, ylim[1] * 1.12)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"1b_PerMouse_Performance_{split}{suffix}.svg"),
                     bbox_inches='tight')
        plt.close()

# ==========================================
# 2. AMBIGUITY HEATMAPS
# ==========================================

def plot_ambiguity_heatmaps(perception_results, split='stratified_balanced',
                              out_dir=".", normalize=True):
    """Decoder loss as a heatmap over stimulus contrast x dispersion.

    ``normalize=True`` (default) divides by the shuffle mean and fixes
    the colour scale to [0, 1]; ``normalize=False`` shows the raw PCA
    loss with an auto colour scale, written to a ``*_raw`` filename.
    """
    if split not in perception_results: return
    set_style()
    suffix = "" if normalize else "_raw"
    res_dict = perception_results[split]

    loss_word = 'PCA Loss' if normalize else 'RAW PCA Loss'
    archs = [('spat', f'Spatial (PPC) {loss_word}'),
             ('temp', f'Temporal (SBC) {loss_word}')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    trials = get_mouse_trials(res_dict, split)

    for ax, (arch_key, title) in zip(axes, archs):
        losses = get_mouse_pca_losses(res_dict, arch_key)
        shf_losses = get_mouse_pca_losses(res_dict, f"{arch_key}_shf")

        if normalize:
            plot_loss = losses / (np.nanmean(shf_losses) + 1e-10)
            vmin, vmax = 0, 1.0
        else:
            plot_loss = losses
            vmin, vmax = 0, None

        df = pd.DataFrame({
            'Contrast': np.round(trials['contrast'], 3),
            'Dispersion': np.round(trials['dispersion'], 3),
            'Loss': plot_loss
        })

        pivot = df.pivot_table(values='Loss', index='Dispersion', columns='Contrast', aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)

        sns.heatmap(pivot, cmap="YlGnBu", ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(title)

    scale = "" if normalize else " — RAW"
    fig.suptitle(f"Stimulus Ambiguity Failure Maps ({split}){scale}", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"2_Ambiguity_Heatmaps_{split}{suffix}.svg"),
                 bbox_inches='tight')
    plt.close()

# ==========================================
# 3. ORIENTATION PERFORMANCE (MEAN + SEM)
# ==========================================

def plot_orientation_performance(perception_results, splits, out_dir=".",
                                   normalize=True):
    """Decoder loss as a function of stimulus orientation.

    ``normalize=True`` (default) divides by the shuffle mean and fixes
    the y-axis to [0, 1.05]; ``normalize=False`` plots the raw PCA loss
    with an auto y-axis, written to a ``*_raw`` filename.
    """
    set_style()
    suffix = "" if normalize else "_raw"
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5), sharey=True)
    if len(splits) == 1: axes = [axes]

    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split not in perception_results: continue
        res_dict = perception_results[split]
        trials = get_mouse_trials(res_dict, split)

        for arch_key, color, label in [('spat', 'darkorange', 'Spatial (PPC)'), ('temp', 'steelblue', 'Temporal (SBC)')]:
            losses = get_mouse_pca_losses(res_dict, arch_key)
            shf_losses = get_mouse_pca_losses(res_dict, f"{arch_key}_shf")

            if normalize:
                plot_loss = losses / (np.nanmean(shf_losses) + 1e-10)
            else:
                plot_loss = losses

            df = pd.DataFrame({'Orientation': trials['orientation'], 'Loss': plot_loss})

            sns.lineplot(data=df, x='Orientation', y='Loss', ax=ax, color=color, label=label,
                         marker='o', estimator=np.mean, errorbar=('se', 1))

        if normalize:
            ax.axhline(1.0, color='red', linestyle='--', label='Shuffle')
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(split)
        ax.set_xlabel("Stimulus Orientation (deg)")

        if idx == 0:
            ax.set_ylabel("Normalized PCA Loss (Raw Mean ± SEM)" if normalize
                           else "Raw PCA Loss (Mean ± SEM)")
        else:
            ax.set_ylabel("")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if len(axes) > 0:
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    scale = "Normalized" if normalize else "RAW"
    fig.suptitle(f"Performance across Orientations ({scale}, Mean ± SEM)", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"3_Orientation_Performance{suffix}.svg"),
                 bbox_inches='tight')
    plt.close()

# ==========================================
# 4. TEMPORAL DYNAMICS (NORMALIZED)
# ==========================================

def plot_temporal_dynamics(perception_results, split='stratified_balanced', out_dir=".",
                             normalize=True):
    """Per-time-bin decoding trajectory for the temporal model, with the
    spatial and temporal-full averages overlaid.

    ``normalize=True`` (default) divides every loss by that mouse's
    shuffle mean (1.0 = shuffle); ``normalize=False`` plots the raw PCA
    loss with no shuffle division / no 1.0 line, into a ``*_raw`` file.
    """
    if split not in perception_results: return
    set_style()
    suffix = "" if normalize else "_raw"
    res_dict = perception_results[split]

    norm_bin_losses = []
    norm_spat_losses = []
    norm_temp_losses = []

    for m_id, m_data in res_dict['results'].items():
        dist = m_data['Dist']
        pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)

        # Per-mouse shuffle means as the normalisation baseline (or 1.0
        # when plotting raw loss).
        spat_shf = calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar)
        temp_shf = calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar)

        spat_denom = (np.nanmean(spat_shf) + 1e-10) if normalize else 1.0
        temp_denom = (np.nanmean(temp_shf) + 1e-10) if normalize else 1.0

        # Spatial average
        spat_raw = calc_pca_dist(dist['spat']['target'], dist['spat']['decoded'], pcs, evar)
        norm_spat_losses.append(np.nanmean(spat_raw) / spat_denom)

        # Temporal full average
        temp_raw = calc_pca_dist(dist['temp']['target'], dist['temp']['decoded'], pcs, evar)
        norm_temp_losses.append(np.nanmean(temp_raw) / temp_denom)

        # Temporal bins
        temp_samp = dist['temp']['decoded_samp']
        target = dist['temp']['target']

        if temp_samp.ndim == 3:
            T = temp_samp.shape[2]
            target_expanded = np.repeat(target[:, :, np.newaxis], T, axis=2)
            loss_t_raw = calc_pca_dist(target_expanded, temp_samp, pcs, evar) # Shape: (N_trials, T)

            # Mean across trials for this mouse, then divide by the
            # shuffle mean (or 1.0 for raw).
            mouse_bin_mean = np.nanmean(loss_t_raw, axis=0) # Shape: (T,)
            norm_bin_losses.append(mouse_bin_mean / temp_denom)
            
    if not norm_bin_losses: return
    
    # Calculate population stats ACROSS MICE (N=6)
    all_loss_t = np.vstack(norm_bin_losses)
    mean_loss_t = np.nanmean(all_loss_t, axis=0)
    sem_loss_t = stats.sem(all_loss_t, axis=0, nan_policy='omit')
    
    plt.figure(figsize=(8, 5))
    time_axis = np.arange(len(mean_loss_t)) * 100 # assuming 100ms bins
    
    # 1. Plot Bin-by-Bin Trajectory
    plt.plot(time_axis, mean_loss_t, color='steelblue', label='Temporal (SBC) Bin-by-Bin')
    plt.fill_between(time_axis, mean_loss_t - sem_loss_t, mean_loss_t + sem_loss_t, color='steelblue', alpha=0.3)
    
    # 2. Overlay Spatial Average
    spat_mean = np.nanmean(norm_spat_losses)
    spat_sem = stats.sem(norm_spat_losses, nan_policy='omit')
    plt.axhline(spat_mean, color='darkorange', label='Spatial (PPC) Average', linestyle='--')
    plt.fill_between(time_axis, spat_mean - spat_sem, spat_mean + spat_sem, color='darkorange', alpha=0.2)
    
    # 3. Overlay Temporal Full Average
    temp_full_mean = np.nanmean(norm_temp_losses)
    temp_full_sem = stats.sem(norm_temp_losses, nan_policy='omit')
    plt.axhline(temp_full_mean, color='darkblue', label='Temporal Full (SBC) Average', linestyle='-.')
    plt.fill_between(time_axis, temp_full_mean - temp_full_sem, temp_full_mean + temp_full_sem, color='darkblue', alpha=0.2)

    # 4. Overlay Shuffle Baseline — exactly 1.0 only in normalised mode.
    if normalize:
        plt.axhline(1.0, color='red', linestyle=':', label='Shuffle Baseline')

    scale = "" if normalize else " — RAW"
    plt.title(f"Temporal Dynamics of Decoding Uncertainty ({split}){scale}")
    plt.xlabel("Time in Window (ms)")
    plt.ylabel("Normalized PCA Loss (Mean ± SEM)" if normalize
                else "Raw PCA Loss (Mean ± SEM)")

    # Place legend outside so it doesn't cover the lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"4_Temporal_Dynamics_{split}{suffix}.svg"),
                 bbox_inches='tight')
    plt.close()


# ==========================================
# 4B. PERFORMANCE VS TARGET CERTAINTY
# ==========================================

def _posterior_entropy(p):
    """Shannon entropy (nats) per row of a probability array, ``-sum p log p``.
    Rows are renormalised defensively in case they don't sum to exactly 1."""
    p = np.asarray(p, dtype=float)
    row = p.sum(axis=1, keepdims=True)
    row = np.where(row > 0, row, 1.0)
    p = p / row
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def plot_performance_vs_certainty(perception_results, splits,
                                    n_bins=6, out_dir=".", normalize=True):
    """Decoder fit-loss as a function of how *certain* the IO target
    posterior is on each trial.

    Certainty is measured as ``1 - H(target)/log(n_cats)`` — the
    normalised (0..1) complement of the target posterior's Shannon
    entropy. 0 = target is uniform (maximally ambiguous stimulus), 1 =
    target is a delta (unambiguous). Trials are pooled across mice,
    binned into ``n_bins`` equal-width certainty bins, and the mean
    fit-loss per bin is plotted for spatial vs temporal. The question
    this answers: do the decoders do better on trials the ideal
    observer itself finds easy?

    ``normalize=True`` (default) divides each trial's loss by that
    mouse's shuffle mean; ``normalize=False`` plots the raw PCA loss
    (no shuffle division / no 1.0 line) into a ``*_raw`` filename.

    Only meaningful for the 91-D PCA-loss targets (Q, L) — the per-trial
    loss is the PCA-projected distance, recomputed via ``calc_pca_dist``.
    """
    suffix = "" if normalize else "_raw"
    for split in splits:
        if split not in perception_results:
            continue
        set_style()
        res_dict = perception_results[split]

        cert_all, spat_all, temp_all = [], [], []
        for m_id, m_data in res_dict['results'].items():
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)
            if pcs is None or (isinstance(pcs, np.ndarray) and len(pcs) == 0):
                # Non-PCA cell — certainty-vs-loss not defined here.
                continue

            target = dist['spat']['target']
            n_cats = target.shape[1]
            # Normalised certainty in [0, 1].
            certainty = 1.0 - _posterior_entropy(target) / np.log(n_cats)

            if normalize:
                spat_denom = np.nanmean(calc_pca_dist(
                    dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar)) + 1e-10
                temp_denom = np.nanmean(calc_pca_dist(
                    dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar)) + 1e-10
            else:
                spat_denom = temp_denom = 1.0

            spat_loss = calc_pca_dist(dist['spat']['target'],
                                        dist['spat']['decoded'], pcs, evar) / spat_denom
            temp_loss = calc_pca_dist(dist['temp']['target'],
                                        dist['temp']['decoded'], pcs, evar) / temp_denom

            cert_all.append(np.asarray(certainty, dtype=float))
            spat_all.append(np.asarray(spat_loss, dtype=float))
            temp_all.append(np.asarray(temp_loss, dtype=float))

        if not cert_all:
            continue
        cert = np.concatenate(cert_all)
        spat = np.concatenate(spat_all)
        temp = np.concatenate(temp_all)

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_idx = np.clip(np.digitize(cert, edges) - 1, 0, n_bins - 1)

        def _binned(vals):
            means = np.full(n_bins, np.nan)
            sems = np.full(n_bins, np.nan)
            for b in range(n_bins):
                v = vals[(bin_idx == b) & np.isfinite(vals)]
                if v.size:
                    means[b] = np.mean(v)
                    sems[b] = (np.std(v, ddof=1) / np.sqrt(v.size)
                                if v.size > 1 else 0.0)
            return means, sems

        spat_m, spat_s = _binned(spat)
        temp_m, temp_s = _binned(temp)

        plt.figure(figsize=(8, 5))
        plt.errorbar(centers, spat_m, yerr=spat_s, color='darkorange',
                      marker='o', capsize=4, label='Spatial (PPC)')
        plt.errorbar(centers, temp_m, yerr=temp_s, color='steelblue',
                      marker='o', capsize=4, label='Temporal (SBC)')
        if normalize:
            plt.axhline(1.0, color='red', linestyle=':', label='Shuffle baseline')
        plt.xlabel("Target certainty  (1 - H(target)/log N;  0 = ambiguous, 1 = certain)")
        plt.ylabel("Normalized PCA loss (Mean ± SEM)" if normalize
                    else "Raw PCA loss (Mean ± SEM)")
        scale = "" if normalize else " — RAW"
        plt.title(f"Decoder performance vs target certainty ({split}){scale}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"4B_Performance_vs_Certainty_{split}{suffix}.svg"),
                     bbox_inches='tight')
        plt.close()


# ==========================================
# 5 & 6. NEUROMETRIC VS PSYCHOMETRIC
# ==========================================

def get_integrated_p_go(decoded_posteriors, angles=np.arange(0, 91, 1), boundary=45.0):
    go_mask = angles < boundary
    p_go = np.sum(decoded_posteriors[:, go_mask], axis=1)
    return p_go

def _load_io_stage2_params(io_path):
    """Return {mouse_id: (alpha_r, beta_r, gamma_r, delta_r)} from IOResults.mat.
    Lazily imported so the rest of this module doesn't pay the io_coherence
    import cost when no lapse correction is requested."""
    from io_coherence import load_io_results
    animals = load_io_results(io_path)
    out = {}
    for mid, a in enumerate(animals):
        c = a['fit']['choice']
        out[mid] = (float(c['alpha_r']), float(c['beta_r']),
                    float(c['gamma_r']), float(c['delta_r']))
    return out


def _lapse_corrected_p_go_from_posteriors(posteriors, params,
                                           angles=np.arange(0, 91, 1),
                                           boundary=45.0):
    """For decoded Q (n_trials, n_bins), compute the IO-consistent P(Go)
    via the Stage 2 lapse psychometric. ``params`` is
    (alpha_r, beta_r, gamma_r, delta_r)."""
    from io_coherence import compute_log_posterior_odds, predict_choice_lapse
    g = compute_log_posterior_odds(posteriors, angles.astype(float))
    alpha_r, beta_r, gamma_r, delta_r = params
    return predict_choice_lapse(g, alpha_r, beta_r, gamma_r, delta_r)


def plot_neurometric_curves(perception_results, choice_results, splits,
                             *, io_path=None, stim_mean_results=None, out_dir="."):
    """Mouse psychometric vs decoder neurometric curves, pooled across mice.

    Default behaviour reproduces the original five-trace plot. Optional
    extras (back-compatible):
      io_path           : if provided, draw lapse-corrected PPC and SBC
                          neurometric traces by pushing g(Q-hat) through
                          each animal's IO Stage 2 psychometric.
      stim_mean_results : if provided (a dict in the same format as
                          perception_results, e.g. via
                          ``load_results_dict('population_results_stim_mean_Q')``),
                          draw the stim-mean baseline neurometric — both
                          the naive boundary integral and (when io_path is
                          also given) its lapse-corrected version.

    When either extra is provided the plot is saved to a separate filename
    so the original SVG is not overwritten.
    """
    set_style()
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), sharey=True)
    if len(splits) == 1: axes = [axes]

    io_params_by_mouse = _load_io_stage2_params(io_path) if io_path else None

    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split not in perception_results: continue

        perc_dict = perception_results[split]
        trials = get_mouse_trials(perc_dict, split)

        choices = (trials['choice'] > 0).astype(float)
        df_psycho = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': choices})
        sns.lineplot(data=df_psycho, x='Orientation', y='P_Go', ax=ax, color='black', label='Mouse Psychometric', errorbar=None, linewidth=3)

        for arch_key, color, label in [('spat', 'darkorange', 'Spatial (PPC) Posterior'), ('temp', 'steelblue', 'Temporal (SBC) Posterior')]:
            all_p_go = []
            all_p_go_lapse = []
            for m_id, m_data in perc_dict['results'].items():
                posteriors = m_data['Dist'][arch_key]['decoded']
                p_go = get_integrated_p_go(posteriors)
                all_p_go.append(p_go)
                if io_params_by_mouse is not None:
                    mid = int(m_id.split('_')[1])
                    if mid in io_params_by_mouse:
                        all_p_go_lapse.append(_lapse_corrected_p_go_from_posteriors(
                            posteriors, io_params_by_mouse[mid]))

            all_p_go = np.concatenate(all_p_go)
            naive_alpha = 0.45 if io_params_by_mouse is not None else 1.0
            naive_lw = 1.0 if io_params_by_mouse is not None else 1.8
            df_neuro = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': all_p_go})
            sns.lineplot(data=df_neuro, x='Orientation', y='P_Go', ax=ax,
                         color=color, label=label + ' (naive)', linestyle='--',
                         alpha=naive_alpha, linewidth=naive_lw)

            if all_p_go_lapse:
                all_p_go_lapse = np.concatenate(all_p_go_lapse)
                df_lapse = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': all_p_go_lapse})
                sns.lineplot(data=df_lapse, x='Orientation', y='P_Go', ax=ax,
                             color=color, label=label + ' + lapse',
                             linestyle='-', linewidth=2.4)

        # Stim-mean baseline neurometric
        if stim_mean_results is not None and split in stim_mean_results:
            sm_dict = stim_mean_results[split]
            sm_naive, sm_lapse = [], []
            for m_id, m_data in sm_dict['results'].items():
                if 'stim_mean' not in m_data['Dist']:
                    continue
                posteriors = m_data['Dist']['stim_mean']['decoded']
                sm_naive.append(get_integrated_p_go(posteriors))
                if io_params_by_mouse is not None:
                    mid = int(m_id.split('_')[1])
                    if mid in io_params_by_mouse:
                        sm_lapse.append(_lapse_corrected_p_go_from_posteriors(
                            posteriors, io_params_by_mouse[mid]))
            if sm_naive:
                df_sm = pd.DataFrame({'Orientation': trials['orientation'],
                                      'P_Go': np.concatenate(sm_naive)})
                naive_alpha = 0.45 if io_params_by_mouse is not None else 1.0
                naive_lw = 1.0 if io_params_by_mouse is not None else 1.8
                sns.lineplot(data=df_sm, x='Orientation', y='P_Go', ax=ax,
                             color='mediumseagreen', label='Stim-mean (naive)',
                             linestyle='--', alpha=naive_alpha, linewidth=naive_lw)
            if sm_lapse:
                df_sml = pd.DataFrame({'Orientation': trials['orientation'],
                                       'P_Go': np.concatenate(sm_lapse)})
                sns.lineplot(data=df_sml, x='Orientation', y='P_Go', ax=ax,
                             color='mediumseagreen', label='Stim-mean + lapse',
                             linestyle='-', linewidth=2.4)

        if choice_results and split in choice_results:
            choice_dict = choice_results[split]
            all_p_go_direct_spat, all_p_go_direct_temp = [], []

            for m_id, m_data in choice_dict['results'].items():
                spat_det = m_data['Dist']['spat']['decoded'][:, 0]
                temp_det = m_data['Dist']['temp']['decoded'][:, 0]
                all_p_go_direct_spat.append(spat_det)
                all_p_go_direct_temp.append(temp_det)

            all_p_go_direct_spat = np.concatenate(all_p_go_direct_spat)
            all_p_go_direct_temp = np.concatenate(all_p_go_direct_temp)

            df_direct_spat = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': all_p_go_direct_spat})
            df_direct_temp = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': all_p_go_direct_temp})

            sns.lineplot(data=df_direct_spat, x='Orientation', y='P_Go', ax=ax, color='red', label='Spatial Direct Choice', linestyle=':')
            sns.lineplot(data=df_direct_temp, x='Orientation', y='P_Go', ax=ax, color='purple', label='Temporal Direct Choice', linestyle=':')

        ax.set_title(split)
        ax.set_xlabel("Stimulus Orientation (deg)")
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.set_ylabel("P(Go)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax.get_legend().remove()

    suffix = ''
    if io_params_by_mouse is not None:
        suffix += '_lapse'
    if stim_mean_results is not None:
        suffix += '_stimmean'
    fig.suptitle("Neurometric vs Psychometric Curves" + (suffix.replace('_', ' ') if suffix else ''),
                 y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"5_Neurometric_Psychometric{suffix}.svg"), bbox_inches='tight')
    plt.close()

def plot_neurometric_curves_per_mouse(perception_results, choice_results, splits,
                                       boundary=45.0, *,
                                       io_path=None, stim_mean_results=None, out_dir="."):
    """Per-mouse neurometric vs psychometric curves.

    Same back-compatible extras as ``plot_neurometric_curves``: pass
    ``io_path`` to add lapse-corrected variants (Stage 2 push-through),
    and ``stim_mean_results`` to add the stim-mean baseline trace(s).
    Output filename includes a suffix when extras are provided so the
    original SVGs are not overwritten.
    """
    set_style()
    io_params_by_mouse = _load_io_stage2_params(io_path) if io_path else None

    for split in splits:
        if split not in perception_results: continue
        perc_dict = perception_results[split]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (m_id, m_data) in enumerate(perc_dict['results'].items()):
            if idx >= len(axes): break
            ax = axes[idx]
            trials = m_data['trials']
            mouse_idx = int(m_id.split('_')[1])

            from utils import get_stratified_train_test_indices, get_generalization_split_indices
            raw_trials = _cached_raw_trials(mouse_idx)

            stimulus_conditions_full = np.array(list(zip(raw_trials['orientation'], raw_trials['contrast'], raw_trials['dispersion'])))
            unique_cats, trial_cats = np.unique(stimulus_conditions_full, axis=0, return_inverse=True)

            if split == 'stratified_balanced':
                _, test_indices = get_stratified_train_test_indices(trial_cats, test_size=0.5, random_state=42)
            else:
                _, test_indices = get_generalization_split_indices(raw_trials, split_type=split, random_state=42)

            test_choices = (raw_trials['choice'][test_indices] > 0).astype(float)

            df_psycho = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': test_choices})
            sns.lineplot(data=df_psycho, x='Orientation', y='P_Go', ax=ax, color='black', label='Behavior', errorbar=None, linewidth=3)

            for arch_key, color, label in [('spat', 'darkorange', 'PPC'), ('temp', 'steelblue', 'SBC')]:
                posteriors = m_data['Dist'][arch_key]['decoded']
                p_go = get_integrated_p_go(posteriors, boundary=boundary)
                naive_alpha = 0.4 if io_params_by_mouse is not None else 1.0
                naive_lw = 0.9 if io_params_by_mouse is not None else 1.6
                df_neuro = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': p_go})
                sns.lineplot(data=df_neuro, x='Orientation', y='P_Go', ax=ax,
                             color=color, label=f'{label} (naive)',
                             linestyle='--', alpha=naive_alpha, linewidth=naive_lw)
                if io_params_by_mouse is not None and mouse_idx in io_params_by_mouse:
                    p_lapse = _lapse_corrected_p_go_from_posteriors(
                        posteriors, io_params_by_mouse[mouse_idx], boundary=boundary)
                    df_lapse = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': p_lapse})
                    sns.lineplot(data=df_lapse, x='Orientation', y='P_Go', ax=ax,
                                 color=color, label=f'{label}+lapse',
                                 linestyle='-', linewidth=2.0)

            if (stim_mean_results is not None and split in stim_mean_results
                    and m_id in stim_mean_results[split]['results']):
                sm_data = stim_mean_results[split]['results'][m_id]
                if 'stim_mean' in sm_data['Dist']:
                    posteriors = sm_data['Dist']['stim_mean']['decoded']
                    p_go = get_integrated_p_go(posteriors, boundary=boundary)
                    naive_alpha = 0.4 if io_params_by_mouse is not None else 1.0
                    naive_lw = 0.9 if io_params_by_mouse is not None else 1.6
                    df_sm = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': p_go})
                    sns.lineplot(data=df_sm, x='Orientation', y='P_Go', ax=ax,
                                 color='mediumseagreen', label='stim-mean (naive)',
                                 linestyle='--', alpha=naive_alpha, linewidth=naive_lw)
                    if io_params_by_mouse is not None and mouse_idx in io_params_by_mouse:
                        p_lapse = _lapse_corrected_p_go_from_posteriors(
                            posteriors, io_params_by_mouse[mouse_idx], boundary=boundary)
                        df_smlap = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': p_lapse})
                        sns.lineplot(data=df_smlap, x='Orientation', y='P_Go', ax=ax,
                                     color='mediumseagreen', label='stim-mean+lapse',
                                     linestyle='-', linewidth=2.0)

            if choice_results and split in choice_results and m_id in choice_results[split]['results']:
                c_data = choice_results[split]['results'][m_id]
                spat_det = c_data['Dist']['spat']['decoded'][:, 0]
                temp_det = c_data['Dist']['temp']['decoded'][:, 0]

                df_spat_dir = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': spat_det})
                df_temp_dir = pd.DataFrame({'Orientation': trials['orientation'], 'P_Go': temp_det})

                sns.lineplot(data=df_spat_dir, x='Orientation', y='P_Go', ax=ax, color='red', label='Spat Choice', linestyle=':')
                sns.lineplot(data=df_temp_dir, x='Orientation', y='P_Go', ax=ax, color='purple', label='Temp Choice', linestyle=':')

            ax.set_title(f"Mouse {mouse_idx}")
            ax.set_ylim(-0.05, 1.05)
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
            else:
                ax.get_legend().remove()

        fig.suptitle(f"Neurometric vs Psychometric (Split: {split})", y=1.02, fontsize=16)
        plt.tight_layout()
        suffix = ''
        if io_params_by_mouse is not None: suffix += '_lapse'
        if stim_mean_results is not None: suffix += '_stimmean'
        plt.savefig(os.path.join(out_dir, f"5_Neurometric_Psychometric_{split}_PerMouse{suffix}.svg"),
                    bbox_inches='tight')
        plt.close()

# ==========================================
# 6. MULTI-TARGET COMPARISON (Perception vs Likelihood)
# ==========================================

def plot_multi_target_comparison(all_target_results, splits, out_dir="."):
    """
    Compares decoder performance across different IO target types (e.g. Perception Posterior vs Likelihood vs Decision)
    for both Spatial (PPC) and Temporal (SBC) architectures.
    
    Each target type is normalized to its OWN shuffle baseline (so all values are relative to chance).
    Targets without PCA components (e.g. 2D Decision) automatically fall back to MSE distance.
    
    all_target_results: dict of {target_name: {split: mat_dict}} 
    """
    set_style()
    
    target_names = list(all_target_results.keys())
    if len(target_names) < 2:
        print("  [!] Need at least 2 target types for comparison. Skipping.")
        return
    
    def calc_dist(target, decoded, pcs, evar):
        """Generic distance: uses PCA if available, else MSE."""
        if pcs is not None and not (isinstance(pcs, np.ndarray) and len(pcs) == 0):
            return calc_pca_dist(target, decoded, pcs, evar)
        else:
            # MSE fallback for low-dimensional targets (e.g. 2D decision posterior)
            return np.mean((target - decoded)**2, axis=-1)
    
    for split in splits:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        
        for ax, (arch_key, arch_label) in zip(axes, [('spat', 'Spatial (PPC)'), ('temp', 'Temporal (SBC)')]):
            x_positions = np.arange(len(target_names))
            means, sems, all_mouse_vals = [], [], []
            
            for target_name in target_names:
                if split not in all_target_results[target_name]:
                    means.append(np.nan)
                    sems.append(np.nan)
                    all_mouse_vals.append([np.nan] * 6)
                    continue
                    
                res_dict = all_target_results[target_name][split]
                mouse_norms = []
                
                for m_id, m_data in res_dict['results'].items():
                    dist = m_data['Dist']
                    pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)
                    
                    v = np.nanmean(calc_dist(dist[arch_key]['target'], dist[arch_key]['decoded'], pcs, evar))
                    v_shf = np.nanmean(calc_dist(dist[f'{arch_key}_shf']['target'], dist[f'{arch_key}_shf']['decoded'], pcs, evar))
                    mouse_norms.append(v / v_shf if v_shf > 0 else np.nan)
                    
                means.append(np.nanmean(mouse_norms))
                sems.append(stats.sem(mouse_norms, nan_policy='omit'))
                all_mouse_vals.append(mouse_norms)
            
            # Bar plot
            colors = ['darkorange', 'forestgreen', 'mediumpurple', 'steelblue'][:len(target_names)]
            bars = ax.bar(x_positions, means, yerr=sems, capsize=5, color=colors, edgecolor='black', alpha=0.7, width=0.6)
            
            # Overlay individual mouse points with connecting lines
            for mouse_idx in range(6):
                mouse_vals = [all_mouse_vals[t][mouse_idx] if mouse_idx < len(all_mouse_vals[t]) else np.nan 
                              for t in range(len(target_names))]
                ax.plot(x_positions, mouse_vals, 'o-', color='grey', alpha=0.4, markersize=5, zorder=5)
            
            ax.axhline(1.0, color='red', linestyle='--', label='Shuffle Baseline')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(target_names, rotation=15, ha='right')
            ax.set_title(arch_label)
            if ax == axes[0]:
                ax.set_ylabel("Normalized Loss\n(Lower is better)")
                
        fig.suptitle(f"Target Comparison ({split})", fontsize=14, y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"6_MultiTarget_Comparison_{split}.svg"), bbox_inches='tight')
        plt.close()

# ==========================================
# 7. STATISTICS TEXT OUT
# ==========================================

def calculate_within_mouse_stats(perception_results, splits):
    print("\n--- POPULATION STATISTICS (RAW MEANS) ---")
    for split in splits:
        if split not in perception_results: continue
        res_dict = perception_results[split]
        
        spat_means = []
        temp_means = []
        
        for m_id, m_data in res_dict['results'].items():
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)
            
            spat_loss = calc_pca_dist(dist['spat']['target'], dist['spat']['decoded'], pcs, evar)
            temp_loss = calc_pca_dist(dist['temp']['target'], dist['temp']['decoded'], pcs, evar)
            
            spat_means.append(np.nanmean(spat_loss))
            temp_means.append(np.nanmean(temp_loss))
            
        t_stat, p_val = stats.ttest_rel(spat_means, temp_means)
        w_stat, p_val_w = stats.wilcoxon(spat_means, temp_means)
        
        print(f"Split: {split}")
        print(f"  Spatial (Mean ± SEM): {np.mean(spat_means):.4f} ± {stats.sem(spat_means):.4f}")
        print(f"  Temporal (Mean ± SEM): {np.mean(temp_means):.4f} ± {stats.sem(temp_means):.4f}")
        print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Wilcoxon test: w = {w_stat:.3f}, p = {p_val_w:.4f}\n")
        
        
        
# ==========================================
# 8. RAW POSTERIOR VISUALIZATIONS
# ==========================================

def plot_posterior_examples_and_averages(perception_results, splits=['stratified_balanced'], out_dir="."):
    set_style()
    s_grid = np.arange(0, 91, 1)

    for split in splits:
        if split not in perception_results: continue
        res_dict = perception_results[split]
        print(f"    -> Processing raw posteriors for {split}...")

        # ---------------------------------------------------------
        # PART A: PER-MOUSE EXAMPLES (BEST, WORST, RANDOM)
        # ---------------------------------------------------------
        for m_id, m_data in res_dict['results'].items():
            mouse_idx = m_id.split('_')[1]
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)
            trials = m_data['trials']

            target = dist['spat']['target'] # (N_trials, 91)
            spat_dec = dist['spat']['decoded']
            temp_dec = dist['temp']['decoded']
            temp_samp = dist['temp'].get('decoded_samp', None)

            # Calculate raw losses to identify the specific trials
            spat_loss = calc_pca_dist(target, spat_dec, pcs, evar)
            temp_loss = calc_pca_dist(target, temp_dec, pcs, evar)

            valid = ~np.isnan(spat_loss) & ~np.isnan(temp_loss)
            if not np.any(valid): continue

            spat_loss_v = np.where(valid, spat_loss, np.inf)
            temp_loss_v = np.where(valid, temp_loss, np.inf)

            diff_temp_better = spat_loss_v - temp_loss_v
            diff_spat_better = temp_loss_v - spat_loss_v

            # Find interesting indices
            idx_best_spat = np.argmin(spat_loss_v)
            idx_best_temp = np.argmin(temp_loss_v)
            idx_spat_wins = np.argmax(np.where(valid, diff_spat_better, -np.inf))
            idx_temp_wins = np.argmax(np.where(valid, diff_temp_better, -np.inf))
            idx_random = np.random.choice(np.where(valid)[0])

            indices = {
                'Best Spatial Performance': idx_best_spat,
                'Best Temporal Performance': idx_best_temp,
                'Spatial heavily outperforms Temporal': idx_spat_wins,
                'Temporal heavily outperforms Spatial': idx_temp_wins,
                'Randomly Chosen Trial': idx_random
            }

            fig, axes = plt.subplots(5, 2, figsize=(14, 18))
            fig.suptitle(f"Mouse {mouse_idx} - Posterior Examples ({split})", fontsize=18, y=1.02)

            for row, (name, idx) in enumerate(indices.items()):
                ax_1d, ax_2d = axes[row, 0], axes[row, 1]

                ori = np.round(trials['orientation'][idx], 1)
                c = np.round(trials['contrast'][idx], 3)
                d = np.round(trials['dispersion'][idx], 1)
                cond_str = f"Ori: {ori}°, Cont: {c}, Disp: {d}°"

                ax_1d.plot(s_grid, target[idx], 'k--', label='Target', lw=2.5)
                ax_1d.plot(s_grid, spat_dec[idx], color='darkorange', label='Spatial (PPC)', lw=2)
                ax_1d.plot(s_grid, temp_dec[idx], color='steelblue', label='Temporal (SBC)', lw=2)
                ax_1d.set_title(f"{name}  |  {cond_str}", fontsize=11)
                ax_1d.set_ylabel("Probability")
                
                # Keep y-limits tight to see the shape
                y_max = max(np.max(target[idx]), np.max(spat_dec[idx]), np.max(temp_dec[idx]))
                ax_1d.set_ylim(-0.01, max(0.05, y_max * 1.1))

                # Heatmap for Temporal Bins
                if temp_samp is not None and temp_samp.ndim == 3:
                    ts = temp_samp[idx] # Shape: (91, T_bins)
                    T = ts.shape[1]
                    im = ax_2d.imshow(ts, aspect='auto', origin='lower', extent=[0, T*100, 0, 90], cmap='viridis')
                    ax_2d.set_title("Temporal Dynamics (SBC Bins)", fontsize=11)
                    ax_2d.set_ylabel("Decoded Orientation (deg)")
                    if row == 4: ax_2d.set_xlabel("Time (ms)")
                else:
                    ax_2d.axis('off')

                if row == 0: ax_1d.legend(fontsize=9, loc='upper right')
                if row == 4: ax_1d.set_xlabel("Orientation (deg)")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"8_Examples_M{mouse_idx}_{split}.svg"), bbox_inches='tight')
            plt.close()

        # ---------------------------------------------------------
        # PART B: POPULATION AVERAGES BY CONDITION
        # ---------------------------------------------------------
        # Collect all trials across all mice
        all_target, all_spat, all_temp, all_bins = [], [], [], []
        all_ori, all_cont, all_disp = [], [], []

        for m_id, m_data in res_dict['results'].items():
            all_target.append(m_data['Dist']['spat']['target'])
            all_spat.append(m_data['Dist']['spat']['decoded'])
            all_temp.append(m_data['Dist']['temp']['decoded'])
            
            tsamp = m_data['Dist']['temp'].get('decoded_samp', None)
            if tsamp is not None and tsamp.ndim == 3: all_bins.append(tsamp)
            
            all_ori.append(m_data['trials']['orientation'])
            all_cont.append(m_data['trials']['contrast'])
            all_disp.append(m_data['trials']['dispersion'])

        T_full = np.concatenate(all_target, axis=0)
        S_full = np.concatenate(all_spat, axis=0)
        Tm_full = np.concatenate(all_temp, axis=0)
        O_full, C_full, D_full = np.concatenate(all_ori), np.concatenate(all_cont), np.concatenate(all_disp)
        
        min_T = min([b.shape[2] for b in all_bins]) if all_bins else 1
        B_full = np.concatenate([b[:, :, :min_T] for b in all_bins], axis=0) if all_bins else None

        for cond_name, cond_array in [('Orientation', O_full), ('Contrast', C_full), ('Dispersion', D_full)]:
            unique_vals = np.unique(cond_array)
            
            # If too many continuous values (e.g., orientation), bin them into ~10 plots to prevent crashes
            if len(unique_vals) > 12:
                bins = np.linspace(np.min(cond_array), np.max(cond_array), 10)
                idx_bins = np.digitize(cond_array, bins)
                plot_vals = bins
            else:
                idx_bins = None
                plot_vals = unique_vals

            fig, axes = plt.subplots(len(plot_vals), 2, figsize=(12, 3 * len(plot_vals)))
            if len(plot_vals) == 1: axes = np.array([axes])
            fig.suptitle(f"Average Posteriors across {cond_name} ({split})", fontsize=16, y=1.02)

            for row, val in enumerate(plot_vals):
                mask = (idx_bins == row + 1) if idx_bins is not None else (cond_array == val)
                title_val = f"~{np.round(val, 2)}" if idx_bins is not None else str(np.round(val, 3))
                
                ax_1d, ax_2d = axes[row, 0], axes[row, 1]

                if np.sum(mask) == 0:
                    ax_1d.axis('off'); ax_2d.axis('off'); continue

                avg_t = np.nanmean(T_full[mask], axis=0)
                avg_s = np.nanmean(S_full[mask], axis=0)
                avg_tm = np.nanmean(Tm_full[mask], axis=0)

                ax_1d.plot(s_grid, avg_t, 'k--', label='Target')
                ax_1d.plot(s_grid, avg_s, color='darkorange', label='Spatial')
                ax_1d.plot(s_grid, avg_tm, color='steelblue', label='Temporal')
                ax_1d.set_title(f"{cond_name}: {title_val}  (n={np.sum(mask)} trials)", fontsize=11)
                ax_1d.set_ylabel("Avg Probability")

                if B_full is not None:
                    avg_b = np.nanmean(B_full[mask], axis=0)
                    im = ax_2d.imshow(avg_b, aspect='auto', origin='lower', extent=[0, avg_b.shape[1]*100, 0, 90], cmap='viridis')
                    ax_2d.set_title("Average Temporal Dynamics", fontsize=11)
                    if row == len(plot_vals)-1: ax_2d.set_xlabel("Time (ms)")
                
                if row == 0: ax_1d.legend(fontsize=9, loc='upper right')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"9_Average_by_{cond_name}_{split}.svg"), bbox_inches='tight')
            plt.close()