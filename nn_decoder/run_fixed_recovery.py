import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from run_experiment import run_animal_decoder

import paths
from pca_loss import pca_distance
from figsave import save_fig

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")

def load_base_predictions(base_file):
    """ Loads the base config. """
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file {base_file} not found.")
    
    data = sio.loadmat(base_file, simplify_cells=True)
    return data['results'], data['config']

def run_recovery_experiment(base_file, target_name):
    """ Runs (or loads cached) crossover experiments for a given base file """
    cache_file = str(paths.recovery_cache(target_name))
    
    if os.path.exists(cache_file):
        print(f"\nLoading cached recovery results from {cache_file}...")
        return np.load(cache_file, allow_pickle=True).item()

    results_real, base_cfg = load_base_predictions(base_file)
    
    t_win = base_cfg.get('time_window', 'half')
    b_sz = base_cfg.get('bin_size_ms', 100)
    
    print(f"\nStarting Recovery Experiment using {base_file} ({t_win.upper()} window, {b_sz}ms bins) as Ground Truth...")
    
    mouse_ids = [int(m.split('_')[1]) for m in results_real.keys()]
    
    target_types = ['spat', 'temp']
    recovery_results = {'base_config': base_cfg}

    for t_type in target_types:
        print(f"=== CROSSOVER BRANCH: Ground Truth = Fitted {t_type.upper()} ===")
        config = base_cfg.copy()
        config['target_source'] = f'recovery_{t_type}' 
        config['base_file_path'] = base_file
        
        session_results = {}
        for mid in mouse_ids:
            print(f"  Training Crossover Models for Mouse {mid}...")
            res = run_animal_decoder(config, mid)
            # run_animal_decoder now returns a 'Checkpoints' field holding
            # torch tensors (state_dict, X_test, pred_probs) for the
            # round-trip sanity check. The recovery cache is a numpy
            # .npy pickle and never needs those tensors — drop them so
            # the cache stays small and torch-free on load.
            res.pop('Checkpoints', None)
            session_results[f"mouse_{mid}"] = res

        recovery_results[t_type] = session_results

    np.save(cache_file, recovery_results)
    return recovery_results

# --- Helper Math ---
def calc_mse(p, q):
    return np.mean((p - q)**2, axis=1)

def calc_loss(arch_dist, loss_func, pcs=None, evar=None):
    """Per-trial loss for one architecture's predictions vs. its target.

    Returns the raw (un-normalized) per-trial loss array. Shuffle-baseline
    normalization is performed by the caller at the per-mouse aggregate level
    (mean(raw)/mean(shf)) — *not* per-trial — because for low-dim MSE targets
    the per-trial shuffled loss can be near-zero, making per-trial ratios blow
    up and dominate the mean. This matches the canonical pattern used in
    plot_normalized_bars_v26.py.

    Parameters
    ----------
    arch_dist : dict
        The per-architecture sub-dict, e.g. ``Dist['spat']`` with
        'decoded' and 'target' arrays.
    loss_func : {'PCA', 'MSE'}
        Which raw loss to compute.
    pcs, evar : ndarray, optional
        Principal components and their explained-variance ratios. Required for
        the PCA branch; they live at ``Dist['pcs']`` / ``Dist['explained_var']``,
        i.e. the *parent* of the per-arch sub-dict.
    """
    p = arch_dist['decoded']
    q = arch_dist['target']

    if loss_func == 'PCA':
        # Plotting-layer policy: a basis-less cell yields all-NaN so the
        # per-mouse np.nanmean aggregation skips it. pca_distance itself
        # raises on a missing basis (see nn_decoder/pca_loss.py).
        if pcs is None or (isinstance(pcs, (list, np.ndarray)) and len(pcs) == 0):
            return np.full(np.asarray(p).shape[0], np.nan)
        return pca_distance(p, q, pcs, evar)
    elif loss_func == 'MSE':
        return calc_mse(p, q)
    else:
        raise ValueError(f"Unknown loss metric requested: {loss_func}")

# --- Plotting Functions ---

def plot_recovery_matrix(recovery_results, target_name):
    """ Generates a Bar Matrix of Normalized Losses """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    sources = ['spat', 'temp']
    cfg = recovery_results['base_config']
    loss_func = cfg.get('custom_loss_func', 'PCA')
    
    # Organize data for grouped bar chart
    bar_labels = ['Target: Fitted spatial', 'Target: Fitted temporal']
    x = np.arange(len(bar_labels))
    width = 0.35
    
    means_spat_decoder = []
    sems_spat_decoder = []
    means_temp_decoder = []
    sems_temp_decoder = []

    # Normalization rule: aggregate (mean over trials) THEN ratio, per mouse,
    # then mean/SEM across mice. Per-trial element-wise ratios blow up when
    # the per-trial shuffled loss is near zero (common for low-dim MSE
    # targets like 2D decision posteriors) and inverted the recovery signal.
    for src in sources:
        spat_per_mouse = []
        temp_per_mouse = []
        for m_id in recovery_results[src].keys():
            parent_dist = recovery_results[src][m_id]['Dist']
            pcs = parent_dist.get('pcs', None)
            evar = parent_dist.get('explained_var', None)

            v_spat     = np.nanmean(calc_loss(parent_dist['spat'],     loss_func, pcs=pcs, evar=evar))
            v_spat_shf = np.nanmean(calc_loss(parent_dist['spat_shf'], loss_func, pcs=pcs, evar=evar))
            v_temp     = np.nanmean(calc_loss(parent_dist['temp'],     loss_func, pcs=pcs, evar=evar))
            v_temp_shf = np.nanmean(calc_loss(parent_dist['temp_shf'], loss_func, pcs=pcs, evar=evar))

            if v_spat_shf > 0:
                spat_per_mouse.append(v_spat / v_spat_shf)
            if v_temp_shf > 0:
                temp_per_mouse.append(v_temp / v_temp_shf)

        means_spat_decoder.append(np.nanmean(spat_per_mouse))
        sems_spat_decoder.append(stats.sem(spat_per_mouse, nan_policy='omit'))
        means_temp_decoder.append(np.nanmean(temp_per_mouse))
        sems_temp_decoder.append(stats.sem(temp_per_mouse, nan_policy='omit'))

    ax.bar(x - width/2, means_spat_decoder, width, yerr=sems_spat_decoder, color='darkorange', capsize=8, edgecolor='black', label='Spatial Decoder')
    ax.bar(x + width/2, means_temp_decoder, width, yerr=sems_temp_decoder, color='steelblue', capsize=8, edgecolor='black', label='Temporal Decoder')
    
    ax.axhline(1.0, color='red', linestyle='--', lw=2, label='Shuffled Baseline (Chance)')
    
    ax.set_ylabel(f"Normalized {loss_func} Loss\n(Fraction of Shuffle, Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_title(f"Loss Recovery Matrix ({target_name.upper()})\nLoss: {loss_func}")

    plt.tight_layout()
    
    out_dir = "Recovery_Plots_Fixed"
    os.makedirs(out_dir, exist_ok=True)
    save_fig(fig, out_dir, f"1_Loss_Matrix_{target_name}", layout=None)

def plot_recovery_scatter(recovery_results, target_name):
    """ Generates a 2x2 Hexbin Density Matrix of True vs Recovered Probabilities """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    sources = ['spat', 'temp']
    archs = ['spat', 'temp']
    arch_titles = {'spat': 'Spatial Arch', 'temp': 'Temporal Arch'}
    
    cfg = recovery_results['base_config']
    
    for row, src in enumerate(sources):
        for col, arch in enumerate(archs):
            ax = axes[row, col]
            
            all_targets = []
            all_decoded = []
            
            for m_id in recovery_results[src].keys():
                dist = recovery_results[src][m_id]['Dist'][arch]
                all_targets.append(dist['target'].flatten())
                all_decoded.append(dist['decoded'].flatten())
                
            all_targets = np.concatenate(all_targets)
            all_decoded = np.concatenate(all_decoded)
            
            hb = ax.hexbin(all_targets, all_decoded, gridsize=50, cmap='inferno', mincnt=1, bins='log')
            
            max_val = max(all_targets.max(), all_decoded.max())
            ax.plot([0, max_val], [0, max_val], color='cyan', linestyle='--', lw=2)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_decoded)
            r2 = r_value**2
            
            ax.text(0.05, 0.9, f"$R^2 = {r2:.3f}$", transform=ax.transAxes, 
                    fontsize=16, color='cyan', weight='bold', bbox=dict(facecolor='black', alpha=0.5))
            
            if row == 0: ax.set_title(arch_titles[arch])
            if col == 0: ax.set_ylabel(f"Ground Truth: {src.upper()}\nRecovered Probability")
            if row == 1: ax.set_xlabel("True (Target) Probability")
            
            ax.set_xlim([0, max_val])
            ax.set_ylim([0, max_val])
            
    cb = fig.colorbar(hb, ax=axes.ravel().tolist(), pad=0.02, shrink=0.8)
    cb.set_label('Log10(Count of Bins)')

    fig.suptitle(f"Scatter Recovery Matrix ({target_name.upper()})\nLoss: {cfg.get('custom_loss_func', 'PCA')}", fontsize=18)
    
    out_dir = "Recovery_Plots_Fixed"
    os.makedirs(out_dir, exist_ok=True)
    save_fig(fig, out_dir, f"2_Scatter_Matrix_{target_name}")

RUN_NAME = 'production_full_targets_alltrials_v1'   # match run_fixed_hyperparams.py
SLUG_BY_TARGET = {
    # PCA-loss slugs include the basis suffix (Config.slug()).
    # 'all_trials' -> '_all'; switched from '_condmean' on 2026-05-27
    # when all_trials became the default basis.
    'Q': 'Q_PCA_half_100ms_all',
    'L': 'L_PCA_half_100ms_all',
    'd': 'd_MSE_half_100ms',
}


def main(run_name: str = RUN_NAME,
         slug_by_target=None,
         split_type: str = 'stratified_balanced'):
    """Run the spat/temp crossover recovery for every (target) in
    ``slug_by_target`` against the fits in
    ``results/<run_name>/<slug>/<split_type>.mat``.

    Exists as an importable entry point so wrappers (e.g.
    run_fits_then_recovery.py) can chain fitting -> recovery in a single
    call. The script's __main__ guard preserves the old standalone
    invocation.
    """
    if slug_by_target is None:
        slug_by_target = SLUG_BY_TARGET

    target_types = [
        ('perception', 'Q', 'PCA'),
        ('likelihood', 'L', 'PCA'),
        ('decision',   'd', 'MSE'),
    ]

    for target_name, fit_target, expected_loss in target_types:
        base_file = str(paths.RESULTS / run_name / slug_by_target[fit_target] / f'{split_type}.mat')
        if not os.path.exists(base_file):
            print(f"[!] {base_file} not found. Has training completed for this slug?")
            continue
        try:
            base_cfg_check = sio.loadmat(base_file, simplify_cells=True)['config']
            actual_loss = base_cfg_check.get('custom_loss_func')
            if actual_loss != expected_loss:
                print(f"[!] {base_file} uses loss={actual_loss!r}, expected {expected_loss!r}. Skipping.")
                continue
            recov_data = run_recovery_experiment(base_file, target_name)
            print(f"Generating Recovery Plots for {target_name}...")
            plot_recovery_matrix(recov_data, target_name)
            plot_recovery_scatter(recov_data, target_name)
            print(f"Finished {target_name}.")
        except Exception as e:
            print(f"[!] Error processing {target_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll fixed recoveries complete! Check ./Recovery_Plots_Fixed/")


if __name__ == "__main__":
    main()
