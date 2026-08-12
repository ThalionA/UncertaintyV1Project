import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import functools

import paths
# Per-trial scoring metrics moved to decoder_metrics (loss maths out of the
# plotting file); re-exported here so existing `import decoder_plotting_utils
# as dpu` (dpu.calc_pca_dist, ...) and `from decoder_plotting_utils import
# calc_fit_loss` call sites keep working unchanged.
from decoder_metrics import (  # noqa: F401  (re-exported for back-compat)
    calc_pca_dist, calc_fit_loss, variance_baseline, get_mouse_pca_losses,
    _normalize_mode, _NORM_SUFFIX, _NORM_BASELINE_LABEL,
)


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
    # Never silently smooth an imshow. matplotlib defaults to 'antialiased', which
    # for a small array (per-bin posteriors are 91 orientations x ~10 time bins)
    # resolves to a 'hanning' resampling filter whenever the panel renders shorter
    # than ~3x the array's rows — blurring the orientation axis with no warning.
    # Set here as well as in peakiness_style.apply() because scripts that call
    # set_style() directly (e.g. the posterior heatmap in this module) never reach
    # that override. 2026-08-12.
    plt.rcParams['image.interpolation'] = 'nearest'

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


@functools.lru_cache(maxsize=None)
def get_train_target_mean(mouse_id: int, which_model: str, split_type: str):
    """Per-bin mean of training-set IO targets for one
    (mouse, target, split) tuple.

    Reloads raw IO targets via ``utils.load_vr_export`` and applies the
    same train/test split logic ``run_experiment.run_animal_decoder``
    uses (``random_state=42``), then returns the mean over training
    trials. Cached per ``(mouse_id, which_model, split_type)`` so the
    six plotters that need it pay the per-mouse VR-export load once.

    Returns ``None`` when the target type doesn't have a directly
    accessible raw IO source — currently anything other than
    ``'perception'`` / ``'likelihood'`` / ``'decision'`` (i.e.
    ``stim_kernel``, ``stim_cat``, ``true_choice``: those are
    synthesised inside ``run_experiment.py`` from
    ``trials['orientation']`` and would need re-synthesis with the
    training-time kernel/temperature params). Callers should fall back
    to the test-target mean when this returns ``None``.
    """
    from utils import (load_vr_export,
                       get_stratified_train_test_indices,
                       get_generalization_split_indices)

    _, targets_perc, targets_dec, targets_lik, trials = load_vr_export(mouse_id)

    if which_model == 'perception':
        raw_targets = targets_perc
    elif which_model == 'likelihood':
        raw_targets = targets_lik
    elif which_model == 'decision':
        raw_targets = targets_dec
    else:
        return None
    if raw_targets is None:
        return None

    stim_full = np.array(list(zip(trials['orientation'],
                                   trials['contrast'],
                                   trials['dispersion'])))
    _, trial_cats_all = np.unique(stim_full, axis=0, return_inverse=True)

    if split_type == 'stratified_balanced':
        train_idx, _ = get_stratified_train_test_indices(
            trial_cats_all, test_size=0.5, random_state=42)
    elif split_type in ('generalize_contrast', 'generalize_dispersion'):
        train_idx, _ = get_generalization_split_indices(
            trials, split_type=split_type, random_state=42)
    else:
        return None

    return np.asarray(raw_targets)[train_idx].mean(axis=0)


def _which_model_from_res(res_dict) -> str:
    """Best-effort ``which_model`` lookup from a loaded .mat res_dict.

    The config block sits at the top level of each ``sio.loadmat``
    output. ``''`` (empty string) for older .mats that pre-date the
    config-save change; downstream code treats that as "no IO target
    source available" and falls back to the test-target mean.
    """
    cfg = res_dict.get('config', {}) if isinstance(res_dict, dict) else {}
    if isinstance(cfg, dict):
        return cfg.get('which_model', '') or ''
    return ''


def _variance_denominator_per_mouse(m_id, m_data, which_model, split_type):
    """Per-mouse mean variance baseline — the denominator the
    ``variance``-normalised plots divide by.

    Mirrors the role of ``np.nanmean(calc_pca_dist(dist['<arch>_shf']...))``
    in the shuffle path, but is *architecture-independent* (it depends
    only on the targets, not the trained decoder). One number per
    (mouse, split). Falls back to the test-target mean as centroid when
    ``get_train_target_mean`` can't supply a training-set mean
    (synthesised targets — stim_kernel / stim_cat).
    """
    dist = m_data['Dist']
    pcs = dist.get('pcs', None)
    evar = dist.get('explained_var', None)
    test_target = np.asarray(dist['spat']['target'])  # spat == temp test target
    try:
        mouse_idx = int(str(m_id).split('_')[1])
    except (IndexError, ValueError):
        mouse_idx = None
    train_mean = None
    if mouse_idx is not None and which_model:
        train_mean = get_train_target_mean(mouse_idx, which_model, split_type)
    if train_mean is None:
        train_mean = np.nanmean(test_target, axis=0)
    per_trial = variance_baseline(test_target, train_mean, pcs, evar)
    return float(np.nanmean(per_trial))


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

    ``normalize`` selects the per-mouse denominator each loss is divided
    by before pooling across mice:

      * ``True`` / ``'shuffle'`` (historical default) — shuffle-trained
        decoder's loss (per arch). 1.0 = chance.
      * ``'variance'`` — variance baseline (marginal-mean loss; see
        :func:`variance_baseline`). One denominator per mouse, shared
        across spat / temp / temp-bins (it depends only on targets,
        not the trained decoder). 1.0 = predict the marginal mean.
      * ``False`` / ``'raw'`` — no division. Writes ``_raw`` filename.
    """
    set_style()
    mode = _normalize_mode(normalize)
    suffix = _NORM_SUFFIX[mode]
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 6), sharey=True)
    if len(splits) == 1: axes = [axes]

    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split not in perception_results: continue

        res_dict = perception_results[split]
        which_model = _which_model_from_res(res_dict)
        mouse_spat, mouse_temp, mouse_temp_bins = [], [], []

        for m_id, m_data in res_dict['results'].items():
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)

            # Raw Means - No Filtering
            v_spat = np.nanmean(calc_pca_dist(dist['spat']['target'], dist['spat']['decoded'], pcs, evar))
            v_temp = np.nanmean(calc_pca_dist(dist['temp']['target'], dist['temp']['decoded'], pcs, evar))

            # Bins
            temp_samp = dist['temp']['decoded_samp']
            target = dist['temp']['target']
            if temp_samp.ndim == 3:
                T = temp_samp.shape[2]
                target_expanded = np.repeat(target[:, :, np.newaxis], T, axis=2)
                v_temp_bins = np.nanmean(calc_pca_dist(target_expanded, temp_samp, pcs, evar))
            else:
                v_temp_bins = np.nan

            if mode == 'shuffle':
                v_spat_shf = np.nanmean(calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar))
                v_temp_shf = np.nanmean(calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar))
                n_spat = v_spat / v_spat_shf if v_spat_shf > 0 else np.nan
                n_temp = v_temp / v_temp_shf if v_temp_shf > 0 else np.nan
                n_temp_bins = v_temp_bins / v_temp_shf if v_temp_shf > 0 else np.nan
            elif mode == 'variance':
                # Single per-mouse denominator — target variance is the
                # same baseline for spatial and temporal architectures.
                v_denom = _variance_denominator_per_mouse(
                    m_id, m_data, which_model, split)
                n_spat = v_spat / v_denom if v_denom > 0 else np.nan
                n_temp = v_temp / v_denom if v_denom > 0 else np.nan
                n_temp_bins = v_temp_bins / v_denom if v_denom > 0 else np.nan
            else:  # raw
                n_spat, n_temp, n_temp_bins = v_spat, v_temp, v_temp_bins

            mouse_spat.append(n_spat)
            mouse_temp.append(n_temp)
            mouse_temp_bins.append(n_temp_bins)

            ax.plot([0, 1, 2], [n_spat, n_temp, n_temp_bins], marker='o', color='gray', alpha=0.4, linewidth=1.5)

        x = np.arange(3)
        means = [np.nanmean(mouse_spat), np.nanmean(mouse_temp), np.nanmean(mouse_temp_bins)]
        sems = [stats.sem(mouse_spat, nan_policy='omit'), stats.sem(mouse_temp, nan_policy='omit'), stats.sem(mouse_temp_bins, nan_policy='omit')]

        ax.bar(x, means, width=0.5, yerr=sems, capsize=5, color=['darkorange', 'steelblue', 'lightblue'], alpha=0.7)
        baseline_label = _NORM_BASELINE_LABEL[mode]
        if baseline_label is not None:
            ax.axhline(1.0, color='red', linestyle='--', lw=2, label=baseline_label)

        # Stats
        try:
            _, p_val = stats.ttest_rel(mouse_spat, mouse_temp)
            y_max = max(means[0] + sems[0], means[1] + sems[1])
            add_stat_annotation(ax, 0, 1, y_max + 0.05, 0.03, p_val)
        except Exception:
            pass

        ax.set_title(split)
        ax.set_xticks(x)
        ax.set_xticklabels(['Spatial', 'Temporal\nFull', 'Temporal\nBins Avg'])
        if idx == 0:
            ax.set_ylabel("Raw PCA Loss (Mean ± SEM)" if mode == 'raw'
                           else "Normalized PCA Loss (Raw Mean ± SEM)")
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='lower left')

    scale = {'shuffle': 'Normalized to shuffle',
             'variance': 'Normalized to variance baseline (marginal mean)',
             'raw': 'RAW (not normalized)'}[mode]
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

    ``normalize``: ``True`` / ``'shuffle'`` divides each mouse's
    per-trial loss by its own shuffle mean (per arch);
    ``'variance'`` divides by the shared variance baseline (one
    denominator per mouse, see :func:`variance_baseline`);
    ``False`` / ``'raw'`` returns raw per-trial losses with no division.
    """
    set_style()
    mode = _normalize_mode(normalize)
    suffix = _NORM_SUFFIX[mode]
    for split in splits:
        if split not in perception_results: continue
        res_dict = perception_results[split]
        which_model = _which_model_from_res(res_dict)

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

            if mode == 'shuffle':
                spat_shf_trials = calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar)
                temp_shf_trials = calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar)
                shf_spat_mean = np.nanmean(spat_shf_trials)
                shf_temp_mean = np.nanmean(temp_shf_trials)
                norm_spat = spat_trials / shf_spat_mean
                norm_temp = temp_trials / shf_temp_mean
            elif mode == 'variance':
                v_denom = _variance_denominator_per_mouse(
                    m_id, m_data, which_model, split)
                norm_spat = spat_trials / v_denom if v_denom > 0 else np.full_like(spat_trials, np.nan)
                norm_temp = temp_trials / v_denom if v_denom > 0 else np.full_like(temp_trials, np.nan)
            else:  # raw
                norm_spat = spat_trials
                norm_temp = temp_trials

            mean_s, sem_s = np.nanmean(norm_spat), stats.sem(norm_spat, nan_policy='omit')
            mean_t, sem_t = np.nanmean(norm_temp), stats.sem(norm_temp, nan_policy='omit')

            ax.bar(x[i] - width/2, mean_s, width, yerr=sem_s, capsize=3, color='darkorange', label='Spatial' if i==0 else "")
            ax.bar(x[i] + width/2, mean_t, width, yerr=sem_t, capsize=3, color='steelblue', label='Temporal' if i==0 else "")

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

        baseline_label = _NORM_BASELINE_LABEL[mode]
        if baseline_label is not None:
            ax.axhline(1.0, color='red', linestyle='--', label=baseline_label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"M{m.split('_')[1]}" for m in mice_ids])
        ax.set_ylabel("Raw PCA Loss (Mean ± SEM)" if mode == 'raw'
                       else "Normalized PCA Loss (Raw Mean ± SEM)")
        scale = " — RAW" if mode == 'raw' else ""
        ax.set_title(f"Intra-Mouse Model Comparison ({split}){scale}")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        ylim = ax.get_ylim()
        if mode == 'raw':
            ax.set_ylim(0, ylim[1] * 1.12)
        else:
            ax.set_ylim(0, max(1.2, ylim[1] + 0.15))

        plt.tight_layout()
        stem = os.path.join(out_dir, f"1b_PerMouse_Performance_{split}{suffix}")
        plt.savefig(stem + ".svg", bbox_inches='tight')
        # also emit a previewable PNG (longest side <=1600 px) alongside the SVG
        _w, _h = fig.get_size_inches()
        plt.savefig(stem + ".png", bbox_inches='tight', dpi=min(140, int(1600 / max(_w, _h))))
        plt.close()

# ==========================================
# 2. AMBIGUITY HEATMAPS
# ==========================================

def _per_mouse_cell_means(res_dict, arch_key, normalize, round_decimals=3,
                            split_type=None):
    """Per-mouse per-(contrast,dispersion) mean loss table.

    Returns a dict ``{mouse_id: pivot_df}`` where each pivot_df has
    Dispersion as index (descending), Contrast as columns. Used by
    plot_ambiguity_heatmaps to build a paired-across-mice t-test in
    each cell instead of treating all trials as exchangeable.

    ``normalize`` (``True`` / ``'shuffle'`` / ``'variance'`` /
    ``False`` / ``'raw'``): selects the per-mouse denominator each
    per-trial loss is divided by before binning into the (contrast,
    dispersion) cells. ``'variance'`` needs ``split_type`` so it can
    look up that mouse's train-set target mean.
    """
    mode = _normalize_mode(normalize)
    which_model = _which_model_from_res(res_dict)
    per_mouse = {}
    for m_id, m_data in res_dict['results'].items():
        dist = m_data['Dist']
        pcs = dist.get('pcs', None)
        evar = dist.get('explained_var', None)
        losses = calc_pca_dist(dist[arch_key]['target'],
                                dist[arch_key]['decoded'], pcs, evar)
        if mode == 'shuffle':
            shf = calc_pca_dist(dist[f'{arch_key}_shf']['target'],
                                  dist[f'{arch_key}_shf']['decoded'], pcs, evar)
            shf_mean = np.nanmean(shf)
            losses = losses / (shf_mean + 1e-10)
        elif mode == 'variance':
            v_denom = _variance_denominator_per_mouse(
                m_id, m_data, which_model, split_type or '')
            losses = losses / (v_denom + 1e-10)
        # mode == 'raw': leave losses unchanged.
        trials = m_data['trials']
        df = pd.DataFrame({
            'Contrast': np.round(trials['contrast'], round_decimals),
            'Dispersion': np.round(trials['dispersion'], round_decimals),
            'Loss': losses,
        })
        pivot = df.pivot_table(values='Loss', index='Dispersion',
                                columns='Contrast', aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)
        per_mouse[m_id] = pivot
    return per_mouse


def _stack_per_mouse_pivots(per_mouse_pivots):
    """Stack ``{mouse: pivot_df}`` into a single ``(n_mice, n_disp, n_cont)``
    ndarray. Aligns all mice to the union of Dispersion x Contrast levels;
    cells absent in a given mouse become NaN."""
    if not per_mouse_pivots:
        return None, None, None
    disp_levels = sorted({d for pv in per_mouse_pivots.values() for d in pv.index},
                         reverse=True)
    cont_levels = sorted({c for pv in per_mouse_pivots.values() for c in pv.columns})
    n_disp, n_cont = len(disp_levels), len(cont_levels)
    stack = np.full((len(per_mouse_pivots), n_disp, n_cont), np.nan)
    for i, (_m_id, pv) in enumerate(per_mouse_pivots.items()):
        pv2 = pv.reindex(index=disp_levels, columns=cont_levels)
        stack[i] = pv2.values
    return stack, disp_levels, cont_levels


def plot_ambiguity_heatmaps(perception_results, split='stratified_balanced',
                              out_dir=".", normalize=True,
                              annotate_p=0.05):
    """Decoder loss heatmaps over stimulus contrast x dispersion.

    Three-panel figure: spatial heatmap, temporal heatmap, and a
    per-(contrast, dispersion) spat-minus-temp **difference** heatmap.
    The difference panel is annotated with paired t-test stars: for
    each cell, the n=6 per-mouse cell means (spat vs temp) are compared
    with ``scipy.stats.ttest_rel``. No multiple-comparisons correction
    is applied — read individual stars as exploratory, not confirmatory.

    ``normalize`` (``True`` / ``'shuffle'`` / ``'variance'`` /
    ``False`` / ``'raw'``): selects the per-mouse denominator. The
    colour scale is fixed to ``[0, 1]`` for normalised modes (chance =
    1.0) and auto for raw. Variance mode shares one denominator per
    mouse across spat / temp (the target variance, not arch-specific).

    Parameters
    ----------
    annotate_p : float
        Cells with paired-t p < annotate_p are starred in the diff
        panel. Set to None or 0 to skip annotation.
    """
    if split not in perception_results: return
    set_style()
    mode = _normalize_mode(normalize)
    suffix = _NORM_SUFFIX[mode]
    res_dict = perception_results[split]

    spat_per_mouse = _per_mouse_cell_means(res_dict, 'spat', mode, split_type=split)
    temp_per_mouse = _per_mouse_cell_means(res_dict, 'temp', mode, split_type=split)
    spat_stack, disp_levels, cont_levels = _stack_per_mouse_pivots(spat_per_mouse)
    temp_stack, _, _ = _stack_per_mouse_pivots(temp_per_mouse)
    if spat_stack is None or temp_stack is None:
        return

    # Aggregate across mice for the per-cell display values + a paired
    # t-test (spat vs temp) per cell. With n=6 mice and many cells the
    # individual p-values are noisy — annotation is exploratory.
    spat_cell_mean = np.nanmean(spat_stack, axis=0)
    temp_cell_mean = np.nanmean(temp_stack, axis=0)
    diff_cell_mean = spat_cell_mean - temp_cell_mean
    n_disp, n_cont = spat_cell_mean.shape
    p_grid = np.full((n_disp, n_cont), np.nan)
    for i in range(n_disp):
        for j in range(n_cont):
            s = spat_stack[:, i, j]
            t = temp_stack[:, i, j]
            valid = np.isfinite(s) & np.isfinite(t)
            if valid.sum() >= 2:
                try:
                    _, p = stats.ttest_rel(s[valid], t[valid])
                    p_grid[i, j] = p
                except Exception:
                    pass

    loss_word = 'RAW PCA Loss' if mode == 'raw' else 'PCA Loss'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    vmin, vmax = (0, None) if mode == 'raw' else (0, 1.0)

    def _to_df(mat):
        return pd.DataFrame(mat, index=disp_levels, columns=cont_levels)

    sns.heatmap(_to_df(spat_cell_mean), cmap='YlGnBu', ax=axes[0],
                 vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Spatial {loss_word}')
    sns.heatmap(_to_df(temp_cell_mean), cmap='YlGnBu', ax=axes[1],
                 vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Temporal {loss_word}')

    # Symmetric diverging scale for the diff panel.
    finite_diff = diff_cell_mean[np.isfinite(diff_cell_mean)]
    diff_lim = float(np.nanmax(np.abs(finite_diff))) if finite_diff.size else 1.0
    if diff_lim == 0:
        diff_lim = 1.0
    sns.heatmap(_to_df(diff_cell_mean), cmap='RdBu_r', ax=axes[2],
                 vmin=-diff_lim, vmax=diff_lim, center=0.0)
    axes[2].set_title('Spatial − Temporal (paired t per cell, n=6 mice)')

    # Annotate paired-t significance into the diff panel.
    if annotate_p:
        for i in range(n_disp):
            for j in range(n_cont):
                p = p_grid[i, j]
                if not np.isfinite(p) or p >= annotate_p:
                    continue
                if p < 0.001: star = '***'
                elif p < 0.01: star = '**'
                else:           star = '*'
                axes[2].text(j + 0.5, i + 0.5, star,
                              ha='center', va='center',
                              color='black', fontsize=12, fontweight='bold')

    scale = {'shuffle': '', 'variance': ' — variance-normalised',
             'raw': ' — RAW'}[mode]
    fig.suptitle(f'Stimulus Ambiguity Failure Maps ({split}){scale}', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'2_Ambiguity_Heatmaps_{split}{suffix}.svg'),
                 bbox_inches='tight')
    plt.close()

# ==========================================
# 3. ORIENTATION PERFORMANCE (MEAN + SEM)
# ==========================================

def plot_orientation_performance(perception_results, splits, out_dir=".",
                                   normalize=True):
    """Decoder loss as a function of stimulus orientation.

    ``normalize`` (``True`` / ``'shuffle'`` / ``'variance'`` /
    ``False`` / ``'raw'``). The shuffle path keeps the historical
    pooled-across-mice denominator (one shuffle mean per arch). The
    variance path divides each mouse's per-trial losses by *its own*
    variance baseline before pooling — this is what actually reduces
    inter-mouse variance in the SEM bands.
    """
    set_style()
    mode = _normalize_mode(normalize)
    suffix = _NORM_SUFFIX[mode]
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5), sharey=True)
    if len(splits) == 1: axes = [axes]

    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split not in perception_results: continue
        res_dict = perception_results[split]
        which_model = _which_model_from_res(res_dict)
        trials = get_mouse_trials(res_dict, split)

        for arch_key, color, label in [('spat', 'darkorange', 'Spatial'), ('temp', 'steelblue', 'Temporal')]:
            if mode == 'variance':
                # Per-mouse normalisation, then concatenate.
                per_mouse_chunks = []
                for m_id, m_data in res_dict['results'].items():
                    dist = m_data['Dist']
                    pcs = dist.get('pcs', None)
                    evar = dist.get('explained_var', None)
                    loss = calc_pca_dist(dist[arch_key]['target'],
                                          dist[arch_key]['decoded'], pcs, evar)
                    v_denom = _variance_denominator_per_mouse(
                        m_id, m_data, which_model, split)
                    per_mouse_chunks.append(
                        loss / v_denom if v_denom > 0 else np.full_like(loss, np.nan))
                plot_loss = np.concatenate(per_mouse_chunks) if per_mouse_chunks else np.array([])
            else:
                losses = get_mouse_pca_losses(res_dict, arch_key)
                if mode == 'shuffle':
                    shf_losses = get_mouse_pca_losses(res_dict, f"{arch_key}_shf")
                    plot_loss = losses / (np.nanmean(shf_losses) + 1e-10)
                else:  # raw
                    plot_loss = losses

            df = pd.DataFrame({'Orientation': trials['orientation'], 'Loss': plot_loss})

            sns.lineplot(data=df, x='Orientation', y='Loss', ax=ax, color=color, label=label,
                         marker='o', estimator=np.mean, errorbar=('se', 1))

        baseline_label = _NORM_BASELINE_LABEL[mode]
        if baseline_label is not None:
            ax.axhline(1.0, color='red', linestyle='--', label=baseline_label)
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(split)
        ax.set_xlabel("Stimulus Orientation (deg)")

        if idx == 0:
            ax.set_ylabel("Raw PCA Loss (Mean ± SEM)" if mode == 'raw'
                           else "Normalized PCA Loss (Raw Mean ± SEM)")
        else:
            ax.set_ylabel("")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if len(axes) > 0:
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    scale = {'shuffle': 'Normalized',
             'variance': 'Variance-normalized',
             'raw': 'RAW'}[mode]
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

    ``normalize`` (``True`` / ``'shuffle'`` / ``'variance'`` /
    ``False`` / ``'raw'``). Variance mode uses one denominator per
    mouse (target variance, arch-independent) so the spat / temp / bins
    trajectories all sit on the same fraction-of-variance scale.
    """
    if split not in perception_results: return
    set_style()
    mode = _normalize_mode(normalize)
    suffix = _NORM_SUFFIX[mode]
    res_dict = perception_results[split]
    which_model = _which_model_from_res(res_dict)

    norm_bin_losses = []
    norm_spat_losses = []
    norm_temp_losses = []

    for m_id, m_data in res_dict['results'].items():
        dist = m_data['Dist']
        pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)

        # Per-mouse denominators. Shuffle: separate spat / temp shuffles
        # (decoder-trained on permuted (X, Y) pairs). Variance: one
        # number per mouse (target variance only, no arch dependence).
        # Raw: 1.0 / 1.0 (no division).
        if mode == 'shuffle':
            spat_shf = calc_pca_dist(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar)
            temp_shf = calc_pca_dist(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar)
            spat_denom = np.nanmean(spat_shf) + 1e-10
            temp_denom = np.nanmean(temp_shf) + 1e-10
        elif mode == 'variance':
            v_denom = _variance_denominator_per_mouse(
                m_id, m_data, which_model, split) + 1e-10
            spat_denom = v_denom
            temp_denom = v_denom
        else:  # raw
            spat_denom = 1.0
            temp_denom = 1.0

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
            # per-mouse denominator (or 1.0 for raw).
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
    plt.plot(time_axis, mean_loss_t, color='steelblue', label='Temporal Bin-by-Bin')
    plt.fill_between(time_axis, mean_loss_t - sem_loss_t, mean_loss_t + sem_loss_t, color='steelblue', alpha=0.3)
    
    # 2. Overlay Spatial Average
    spat_mean = np.nanmean(norm_spat_losses)
    spat_sem = stats.sem(norm_spat_losses, nan_policy='omit')
    plt.axhline(spat_mean, color='darkorange', label='Spatial Average', linestyle='--')
    plt.fill_between(time_axis, spat_mean - spat_sem, spat_mean + spat_sem, color='darkorange', alpha=0.2)
    
    # 3. Overlay Temporal Full Average
    temp_full_mean = np.nanmean(norm_temp_losses)
    temp_full_sem = stats.sem(norm_temp_losses, nan_policy='omit')
    plt.axhline(temp_full_mean, color='darkblue', label='Temporal Full Average', linestyle='-.')
    plt.fill_between(time_axis, temp_full_mean - temp_full_sem, temp_full_mean + temp_full_sem, color='darkblue', alpha=0.2)

    # 4. Overlay baseline line at 1.0 in normalised modes.
    baseline_label = _NORM_BASELINE_LABEL[mode]
    if baseline_label is not None:
        plt.axhline(1.0, color='red', linestyle=':', label=baseline_label)

    scale = {'shuffle': '', 'variance': ' — variance-normalised',
             'raw': ' — RAW'}[mode]
    plt.title(f"Temporal Dynamics of Decoding Uncertainty ({split}){scale}")
    plt.xlabel("Time in Window (ms)")
    plt.ylabel("Raw PCA Loss (Mean ± SEM)" if mode == 'raw'
                else "Normalized PCA Loss (Mean ± SEM)")

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

    ``normalize`` (``True`` / ``'shuffle'`` / ``'variance'`` /
    ``False`` / ``'raw'``). All three modes use per-mouse denominators
    so the pooled bin means aren't biased by inter-mouse scale.

    Only meaningful for the 91-D PCA-loss targets (Q, L) — the per-trial
    loss is the PCA-projected distance, recomputed via ``calc_pca_dist``.
    """
    mode = _normalize_mode(normalize)
    suffix = _NORM_SUFFIX[mode]
    for split in splits:
        if split not in perception_results:
            continue
        set_style()
        res_dict = perception_results[split]
        which_model = _which_model_from_res(res_dict)

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

            if mode == 'shuffle':
                spat_denom = np.nanmean(calc_pca_dist(
                    dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar)) + 1e-10
                temp_denom = np.nanmean(calc_pca_dist(
                    dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar)) + 1e-10
            elif mode == 'variance':
                v_denom = _variance_denominator_per_mouse(
                    m_id, m_data, which_model, split) + 1e-10
                spat_denom = v_denom
                temp_denom = v_denom
            else:  # raw
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
                      marker='o', capsize=4, label='Spatial')
        plt.errorbar(centers, temp_m, yerr=temp_s, color='steelblue',
                      marker='o', capsize=4, label='Temporal')
        baseline_label = _NORM_BASELINE_LABEL[mode]
        if baseline_label is not None:
            plt.axhline(1.0, color='red', linestyle=':', label=baseline_label)
        plt.xlabel("Target certainty  (1 - H(target)/log N;  0 = ambiguous, 1 = certain)")
        plt.ylabel("Raw PCA loss (Mean ± SEM)" if mode == 'raw'
                    else "Normalized PCA loss (Mean ± SEM)")
        scale = {'shuffle': '', 'variance': ' — variance-normalised',
                 'raw': ' — RAW'}[mode]
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
      io_path           : if provided, draw lapse-corrected spatial and temporal
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

        for arch_key, color, label in [('spat', 'darkorange', 'Spatial Posterior'), ('temp', 'steelblue', 'Temporal Posterior')]:
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

            for arch_key, color, label in [('spat', 'darkorange', 'spatial'), ('temp', 'steelblue', 'temporal')]:
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
    """Compare decoder performance across IO target types, clustered by
    target with spatial vs temporal paired side-by-side.

    For each split, three figures are written:

      * ``6_MultiTarget_Comparison_<split>.svg`` — shuffle-normalised
        loss (each mouse's loss divided by that mouse's own shuffle
        mean, per target+arch). Red dashed line at 1.0 marks chance.
      * ``6_MultiTarget_Comparison_<split>_varnorm.svg`` —
        variance-normalised loss (each mouse's loss divided by its
        own variance baseline — marginal-mean loss, see
        :func:`variance_baseline`). Per-target self-normalisation,
        so across-target magnitudes are directly comparable. Red
        dashed line at 1.0 marks "predict the marginal mean".
      * ``6_MultiTarget_Comparison_<split>_raw.svg`` — raw distance
        (PCA when available, MSE fallback for low-dim targets). No
        baseline line.

    Layout: x-axis groups one cluster per target; within each cluster a
    spatial bar (orange) and a temporal bar (steelblue) sit side by
    side. Per-mouse dots are jittered on each bar; faint lines connect
    each mouse's spat→temp pair within the same target. A paired t-test
    across mice (spat vs temp) is annotated above each cluster.

    Targets whose PCA basis is absent (e.g. 2-D decision) fall back to
    MSE distance automatically — same behaviour as the previous
    revision of this plot. The variance-normalised figure can only
    use PCA basis when one is present (low-dim targets fall back to
    the test-target variance for the denominator too).

    Parameters
    ----------
    all_target_results : dict
        ``{target_name: {split: mat_dict}}``
    splits : iterable of str
    out_dir : str
    """
    set_style()

    target_names = list(all_target_results.keys())
    if len(target_names) < 2:
        print("  [!] Need at least 2 target types for comparison. Skipping.")
        return

    def calc_dist(target, decoded, pcs, evar):
        if pcs is not None and not (isinstance(pcs, np.ndarray) and len(pcs) == 0):
            return calc_pca_dist(target, decoded, pcs, evar)
        return np.mean((target - decoded) ** 2, axis=-1)

    def _per_mouse_metric(res_dict, arch_key, mode, split_type=None):
        """Per-mouse mean distance for one (target, arch). Returns a
        list (length = n mice). ``mode`` is one of 'shuffle' /
        'variance' / 'raw'; 'variance' needs ``split_type`` to look up
        the per-mouse train-target mean."""
        which_model = _which_model_from_res(res_dict)
        out = []
        for m_id, m_data in res_dict['results'].items():
            dist = m_data['Dist']
            pcs, evar = dist.get('pcs', None), dist.get('explained_var', None)
            v = np.nanmean(calc_dist(dist[arch_key]['target'],
                                       dist[arch_key]['decoded'], pcs, evar))
            if mode == 'shuffle':
                v_shf = np.nanmean(calc_dist(dist[f'{arch_key}_shf']['target'],
                                               dist[f'{arch_key}_shf']['decoded'],
                                               pcs, evar))
                out.append(v / v_shf if v_shf > 0 else np.nan)
            elif mode == 'variance':
                if pcs is not None and not (isinstance(pcs, np.ndarray) and len(pcs) == 0):
                    v_denom = _variance_denominator_per_mouse(
                        m_id, m_data, which_model, split_type or '')
                else:
                    # Low-dim target fallback: variance of the test
                    # targets around their own mean in the MSE metric.
                    test_target = np.asarray(dist[arch_key]['target'])
                    centroid = np.nanmean(test_target, axis=0)
                    v_denom = float(np.nanmean(
                        np.mean((test_target - centroid) ** 2, axis=-1)))
                out.append(v / v_denom if v_denom > 0 else np.nan)
            else:  # raw
                out.append(v)
        return out

    arch_color = {'spat': 'darkorange', 'temp': 'steelblue'}
    arch_label = {'spat': 'Spatial', 'temp': 'Temporal'}

    def _make_figure(split, normalize, out_path):
        mode = _normalize_mode(normalize)
        # Per-(target, arch) per-mouse values.
        per_mouse = {}  # (target, arch) -> list of mouse means
        for target_name in target_names:
            if split not in all_target_results[target_name]:
                per_mouse[(target_name, 'spat')] = []
                per_mouse[(target_name, 'temp')] = []
                continue
            res_dict = all_target_results[target_name][split]
            per_mouse[(target_name, 'spat')] = _per_mouse_metric(
                res_dict, 'spat', mode, split_type=split)
            per_mouse[(target_name, 'temp')] = _per_mouse_metric(
                res_dict, 'temp', mode, split_type=split)

        x = np.arange(len(target_names))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(target_names)), 6))

        # Bars (mean ± SEM) for spat then temp at each cluster.
        for i, arch in enumerate(('spat', 'temp')):
            offset = (i - 0.5) * width
            means, sems = [], []
            for target_name in target_names:
                vals = [v for v in per_mouse[(target_name, arch)]
                         if np.isfinite(v)]
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)
            ax.bar(x + offset, means, width=width, yerr=sems, capsize=4,
                    color=arch_color[arch], edgecolor='black', alpha=0.85,
                    label=arch_label[arch], zorder=2)

        # Per-mouse paired dots with connecting line within each cluster.
        for ti, target_name in enumerate(target_names):
            sv = per_mouse[(target_name, 'spat')]
            tv = per_mouse[(target_name, 'temp')]
            n_mice = max(len(sv), len(tv))
            xs_spat = x[ti] - 0.5 * width
            xs_temp = x[ti] + 0.5 * width
            for mi in range(n_mice):
                s = sv[mi] if mi < len(sv) else np.nan
                t = tv[mi] if mi < len(tv) else np.nan
                ax.plot([xs_spat, xs_temp], [s, t],
                         '-', color='grey', alpha=0.35, lw=1.0, zorder=3)
                ax.plot([xs_spat], [s], 'o', color='black',
                         markersize=4, alpha=0.7, zorder=4)
                ax.plot([xs_temp], [t], 'o', color='black',
                         markersize=4, alpha=0.7, zorder=4)

        # Paired t-test annotation per cluster.
        for ti, target_name in enumerate(target_names):
            sv = np.asarray(per_mouse[(target_name, 'spat')], dtype=float)
            tv = np.asarray(per_mouse[(target_name, 'temp')], dtype=float)
            mask = np.isfinite(sv) & np.isfinite(tv)
            if mask.sum() < 2:
                continue
            try:
                _, p = stats.ttest_rel(sv[mask], tv[mask])
            except Exception:
                continue
            ymax = np.nanmax(np.concatenate([sv[mask], tv[mask]]))
            add_stat_annotation(ax,
                                  x[ti] - 0.5 * width,
                                  x[ti] + 0.5 * width,
                                  ymax * 1.05 if np.isfinite(ymax) else 0.05,
                                  ymax * 0.03 if np.isfinite(ymax) else 0.01,
                                  p)

        baseline_label = _NORM_BASELINE_LABEL[mode]
        if baseline_label is not None:
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.8,
                        label=baseline_label)
        if mode == 'shuffle':
            ax.set_ylabel('Normalised loss\n(loss / shuffle, lower = better)')
        elif mode == 'variance':
            ax.set_ylabel('Variance-normalised loss\n(loss / marginal-mean baseline)')
        else:
            ax.set_ylabel('Raw distance\n(PCA loss; MSE for 2-D targets)')

        ax.set_xticks(x)
        ax.set_xticklabels(target_names, rotation=15, ha='right')
        title_scale = {'shuffle': 'shuffle-normalised',
                       'variance': 'variance-normalised',
                       'raw': 'raw'}[mode]
        ax.set_title(f'Target comparison — {title_scale} ({split})')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    for split in splits:
        for mode in ('shuffle', 'variance', 'raw'):
            out_path = os.path.join(
                out_dir,
                f'6_MultiTarget_Comparison_{split}{_NORM_SUFFIX[mode]}.svg')
            _make_figure(split, normalize=mode, out_path=out_path)

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
                ax_1d.plot(s_grid, spat_dec[idx], color='darkorange', label='Spatial', lw=2)
                ax_1d.plot(s_grid, temp_dec[idx], color='steelblue', label='Temporal', lw=2)
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
                    ax_2d.set_title("Temporal Dynamics (temporal Bins)", fontsize=11)
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

        # Signed contrast: contrast * sign(45° - abs_from_go).
        # Sign is +1 on the Go side (abs_from_go < 45°) and -1 on the
        # NoGo side. abs_from_go = 45° is the categorical boundary, sign
        # 0 there (numpy convention) — those trials drop into the centre
        # bin and pool together.
        side_sign_full = np.sign(45.0 - O_full)
        signed_contrast_full = side_sign_full * C_full
        # Signed concentration: (5° / dispersion) * side_sign. Higher
        # absolute value = sharper stimulus (less dispersed). Convention
        # mirrors dim_reduction_explore.py for cross-figure consistency.
        with np.errstate(divide='ignore', invalid='ignore'):
            concentration_full = np.where(D_full > 0, 5.0 / D_full, np.nan)
        signed_concentration_full = side_sign_full * concentration_full

        for cond_name, cond_array in [('Orientation', O_full),
                                        ('Contrast', C_full),
                                        ('Dispersion', D_full),
                                        ('SignedContrast', signed_contrast_full),
                                        ('SignedConcentration', signed_concentration_full)]:
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