"""Headline visualisations of the loss-sweep + stim-mean comparison.

Produces three figures, all written to ``Loss_Sweep_Plots/``:

1. ``method_bars_<split>.svg``  — choice NLL and AUROC per method, with
   per-mouse dots overlaid. One panel per metric, one figure per split.
2. ``moment_tradeoff.svg``      — mean error vs variance error scatter
   for the Q-shaped methods (PCA-trained spatial/temporal, Wasserstein spatial/temporal,
   stim_mean_Q). Per-mouse dots, per-method centroids labelled.
   Visualises why Wasserstein hurts choice NLL despite tightening
   variance error.
3. ``q_anatomy_<mouse>.svg``    — for a representative mouse, overlay
   decoded Q for PCA-trained spatial/temporal, Wasserstein-trained spatial/temporal, and
   stim_mean alongside the IO target on six trials spanning the stim
   space. Shows visually what each loss does to the decoded Q.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_coherence as ioc  # noqa: E402
import paths  # noqa: E402
from figsave import save_fig  # noqa: E402

GRID = np.arange(0, 91, 1, dtype=float)

# Default file locations resolve relative to this script so the entry
# points work the same whether invoked from the project root, from the
# nn_decoder/ directory, or via %runcell in IPython.
DEFAULT_IO_PATH = os.path.normpath(os.path.join(HERE, '..', 'data', 'IOResults.mat'))
DEFAULT_DIRECTORY = HERE
DEFAULT_OUT_DIR = os.path.join(HERE, 'Loss_Sweep_Plots')


# --- styling ---------------------------------------------------------

METHOD_COLORS = {
    'IO_baseline':      'black',
    'stim_mean_Q':      'mediumseagreen',
    'stim_mean_choice': 'olivedrab',
    'PPC_PCA':          'darkorange',
    'SBC_PCA':          'steelblue',
    'PPC_W':            'sandybrown',
    'SBC_W':            'lightblue',
    'PPC_KL':           'gold',
    'SBC_KL':           'cornflowerblue',
    'PPC_JS':           'peru',
    'SBC_JS':           'mediumpurple',
    'true_choice_PPC':  'firebrick',
    'true_choice_SBC':  'rebeccapurple',
}

# Panel A: Q-shaped methods — IO baseline, stim_mean_Q, all four loss
# variants of spatial and temporal. This panel is the loss-sweep + stim-baseline
# comparison for the Q decoders, all evaluated through the IO Stage 2
# lapse psychometric.
Q_METHOD_ORDER = [
    'IO_baseline',
    'stim_mean_Q',
    'PPC_PCA', 'PPC_KL', 'PPC_JS', 'PPC_W',
    'SBC_PCA', 'SBC_KL', 'SBC_JS', 'SBC_W',
]

# Panel B: choice-shaped methods — IO baseline, stim_mean_choice, and
# the true-choice spatial/temporal decoders trained CE against the animal's
# goChoice directly (no lapse correction; outputs are already P(Go)).
CHOICE_METHOD_ORDER = [
    'IO_baseline',
    'stim_mean_choice',
    'true_choice_PPC', 'true_choice_SBC',
]


def _pretty(method):
    """Display label for a method key — architecture acronyms are shown as
    spatial/temporal. The keys themselves stay PPC/SBC for data matching."""
    return method.replace('PPC', 'spatial').replace('SBC', 'temporal')


def _set_style():
    sns.set_context('talk')
    sns.set_style('ticks')


# --- (1) Per-split metric bar charts --------------------------------

def _bar_with_strip(ax, methods_present, sub, metric, ylabel):
    """Render one bar+strip panel for the given metric and method set."""
    xs = np.arange(len(methods_present))
    for i, m in enumerate(methods_present):
        vals = sub[sub['method'] == m][metric].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        color = METHOD_COLORS.get(m, 'grey')
        mean = float(np.mean(vals))
        sem = float(stats.sem(vals)) if len(vals) > 1 else np.nan
        ax.bar(i, mean, yerr=sem, color=color, edgecolor='black',
               capsize=4, alpha=0.9)
        rng = np.random.default_rng(i)
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, s=22, color='black',
                   alpha=0.6, zorder=3, linewidths=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([_pretty(m) for m in methods_present],
                        rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)


def plot_method_bars(df, out_dir):
    """Two-panel-per-metric layout per split:

      Panel A (top row): Q-shaped methods — IO baseline, stim_mean_Q,
        spatial and temporal under each loss (PCA, KL, JS, Wasserstein),
        evaluated through the Stage 2 lapse psychometric.
      Panel B (bottom row): choice-shaped methods — IO baseline,
        stim_mean_choice, true_choice spatial/temporal (no lapse correction;
        outputs are already calibrated P(Go)).

    Two columns: NLL (lower better), AUROC (higher better).
    """
    _set_style()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for split in sorted(df['split'].unique()):
        sub = df[df['split'] == split]
        q_present = [m for m in Q_METHOD_ORDER if m in sub['method'].unique()]
        c_present = [m for m in CHOICE_METHOD_ORDER if m in sub['method'].unique()]

        fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=False)

        # Row A: Q-shaped
        _bar_with_strip(axes[0, 0], q_present, sub, 'nll',
                        'Choice NLL  (lower = better)')
        _bar_with_strip(axes[0, 1], q_present, sub, 'auroc',
                        'Choice AUROC  (higher = better)')
        axes[0, 0].set_title('Q-shaped methods  (lapse-corrected through Stage 2)',
                              fontsize=12)
        axes[0, 1].set_title('Q-shaped methods  (lapse-corrected through Stage 2)',
                              fontsize=12)

        # Row B: choice-shaped
        if c_present:
            _bar_with_strip(axes[1, 0], c_present, sub, 'nll',
                            'Choice NLL  (lower = better)')
            _bar_with_strip(axes[1, 1], c_present, sub, 'auroc',
                            'Choice AUROC  (higher = better)')
            axes[1, 0].set_title('Choice-shaped methods  (already calibrated, no lapse)',
                                  fontsize=12)
            axes[1, 1].set_title('Choice-shaped methods  (already calibrated, no lapse)',
                                  fontsize=12)
        else:
            for a in axes[1]:
                a.text(0.5, 0.5, 'No choice-shaped files for this split',
                       ha='center', va='center', transform=a.transAxes,
                       fontsize=11, color='grey')
                a.set_xticks([]); a.set_yticks([])

        fig.suptitle(f'Choice prediction across methods  |  split = {split}',
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(out_dir, f'method_bars_{split}.svg')
        save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0], layout=None)
        written.append(path)
    return written


# --- (2) Moment trade-off scatter -----------------------------------

def plot_moment_tradeoff(df, out_dir):
    """Per-mouse mean_err vs var_err for Q-shaped methods, all splits.
    Each method gets a colour; per-mouse dots are translucent and a
    big centroid marks the per-method mean across mice and splits."""
    _set_style()
    os.makedirs(out_dir, exist_ok=True)
    sub = df.dropna(subset=['mean_err', 'var_err'])
    fig, ax = plt.subplots(figsize=(10, 8))
    for m in sub['method'].unique():
        s = sub[sub['method'] == m]
        color = METHOD_COLORS.get(m, 'grey')
        ax.scatter(s['mean_err'], s['var_err'], s=40, color=color,
                   alpha=0.45, label=None, edgecolor='none')
        # Centroid
        cx, cy = float(np.mean(s['mean_err'])), float(np.mean(s['var_err']))
        ax.scatter([cx], [cy], s=240, color=color, edgecolor='black',
                   linewidth=1.5, zorder=5, label=_pretty(m))
        ax.annotate(_pretty(m), (cx, cy), xytext=(8, 8), textcoords='offset points',
                    fontsize=9, color='black')
    ax.set_xscale('symlog', linthresh=10)
    ax.set_yscale('symlog', linthresh=1000)
    ax.set_xlabel(r'Mean error  $(\hat\mu - \mu)^2$  [deg$^2$]  '
                  '(lower = decoded Q peaks closer to target)')
    ax.set_ylabel(r'Variance error  $(\hat\sigma^2 - \sigma^2)^2$  [deg$^4$]  '
                  '(lower = decoded Q spread closer to target)')
    ax.set_title('Moment-error trade-off across Q-shaped methods\n'
                 'each dot = one (mouse, split); large markers = method centroid',
                 fontsize=12)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, 'moment_tradeoff.svg')
    save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0])
    return path


# --- (3) Per-trial Q-anatomy ----------------------------------------

def _pick_representative_trials(orientations, contrasts, dispersions, n=6, seed=1):
    rng = np.random.default_rng(seed)
    high_c = contrasts >= np.quantile(contrasts, 0.75)
    low_c = contrasts <= np.quantile(contrasts, 0.25)
    high_d = dispersions >= np.quantile(dispersions, 0.75)
    low_d = dispersions <= np.quantile(dispersions, 0.25)
    near = np.abs(orientations - 45) <= 10
    far = np.abs(orientations - 45) >= 35
    cells = [
        ('hi-C, lo-D, far', high_c & low_d & far),
        ('hi-C, lo-D, near', high_c & low_d & near),
        ('lo-C, lo-D, far', low_c & low_d & far),
        ('lo-C, hi-D, far', low_c & high_d & far),
        ('hi-C, hi-D, far', high_c & high_d & far),
        ('lo-C, hi-D, near', low_c & high_d & near),
    ]
    chosen = []
    for label, mask in cells:
        idx = np.where(mask)[0]
        if len(idx):
            chosen.append((label, int(rng.choice(idx))))
        if len(chosen) >= n:
            break
    return chosen


def plot_q_anatomy(mouse_id, split, io_path, directory, out_dir):
    _set_style()
    os.makedirs(out_dir, exist_ok=True)

    animals = ioc.load_io_results(io_path)
    a = animals[mouse_id]
    mask = ioc.get_trials_to_keep_mask(a['data']['trial_keys']['session_name'])
    Q_io = np.asarray(a['inferred']['post_s_marginal'])[mask]
    orientations = np.asarray(a['data']['orientation'])[mask]
    contrasts = np.asarray(a['data']['contrast'])[mask]
    dispersions = np.asarray(a['data']['dispersion'])[mask]
    choices = np.asarray(a['data']['choices'])[mask].astype(float)

    def _full(prefix, arch):
        path = os.path.join(directory, f'{prefix}_{split}.mat')
        if not os.path.exists(path):
            return None
        m = sio.loadmat(path, simplify_cells=True)
        try:
            return np.asarray(m['results'][f'mouse_{mouse_id}']['Dist'][arch]['full_decoded'])
        except KeyError:
            return None

    # Canonical stems sourced from paths.FIT_BASENAMES.
    Q_pca_ppc = _full(paths.fit_stem('fixed', 'Q'),                'spat')
    Q_pca_sbc = _full(paths.fit_stem('fixed', 'Q'),                'temp')
    Q_w_ppc   = _full(paths.fit_stem('loss_sweep', 'Wasserstein'), 'spat')
    Q_w_sbc   = _full(paths.fit_stem('loss_sweep', 'Wasserstein'), 'temp')
    Q_kl_ppc  = _full(paths.fit_stem('loss_sweep', 'KL'),          'spat')
    Q_kl_sbc  = _full(paths.fit_stem('loss_sweep', 'KL'),          'temp')
    Q_js_ppc  = _full(paths.fit_stem('loss_sweep', 'JS'),          'spat')
    Q_js_sbc  = _full(paths.fit_stem('loss_sweep', 'JS'),          'temp')
    Q_sm      = _full(paths.fit_stem('stim_mean', 'Q'),            'stim_mean')

    picks = _pick_representative_trials(orientations, contrasts, dispersions, n=6, seed=mouse_id)
    if not picks:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    axes = axes.flatten()
    for ax, (label, ti) in zip(axes, picks):
        ax.plot(GRID, Q_io[ti], color='black', linewidth=2.4, label='IO target')
        if Q_sm is not None:
            ax.plot(GRID, Q_sm[ti], color='mediumseagreen', linewidth=2.0,
                    linestyle='-', label='stim_mean')
        if Q_pca_ppc is not None:
            ax.plot(GRID, Q_pca_ppc[ti], color='darkorange', linewidth=1.4,
                    alpha=0.95, label='spatial PCA')
        if Q_pca_sbc is not None:
            ax.plot(GRID, Q_pca_sbc[ti], color='steelblue', linewidth=1.4,
                    alpha=0.95, label='temporal PCA')
        if Q_w_ppc is not None:
            ax.plot(GRID, Q_w_ppc[ti], color='sandybrown', linewidth=1.2,
                    linestyle=':', alpha=0.85, label='spatial W')
        if Q_w_sbc is not None:
            ax.plot(GRID, Q_w_sbc[ti], color='lightblue', linewidth=1.2,
                    linestyle=':', alpha=0.85, label='temporal W')
        if Q_kl_ppc is not None:
            ax.plot(GRID, Q_kl_ppc[ti], color='gold', linewidth=1.0,
                    linestyle='-.', alpha=0.85, label='spatial KL')
        if Q_kl_sbc is not None:
            ax.plot(GRID, Q_kl_sbc[ti], color='cornflowerblue', linewidth=1.0,
                    linestyle='-.', alpha=0.85, label='temporal KL')
        if Q_js_ppc is not None:
            ax.plot(GRID, Q_js_ppc[ti], color='peru', linewidth=1.0,
                    linestyle=(0, (3, 1, 1, 1)), alpha=0.85, label='spatial JS')
        if Q_js_sbc is not None:
            ax.plot(GRID, Q_js_sbc[ti], color='mediumpurple', linewidth=1.0,
                    linestyle=(0, (3, 1, 1, 1)), alpha=0.85, label='temporal JS')
        ax.axvline(45, color='grey', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(orientations[ti], color='red', linestyle=':', linewidth=1, alpha=0.7)
        c = 'Go' if choices[ti] > 0.5 else 'No-Go'
        ax.set_title(
            f'trial {ti}  |  {label}\n'
            f'theta={orientations[ti]:.0f}  c={contrasts[ti]:.2f}  '
            f'd={dispersions[ti]:.0f}  choice={c}',
            fontsize=10,
        )
        ax.set_xlabel('orientation theta [deg]')
        ax.set_ylabel('Q(theta)')
        ax.set_xlim(0, 90)
        ax.grid(linestyle='--', alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(
        f'Decoded Q anatomy across losses — mouse {mouse_id}, split = {split}',
        fontsize=14, y=1.06,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f'q_anatomy_mouse{mouse_id}_{split}.svg')
    save_fig(fig, out_dir, os.path.splitext(os.path.basename(path))[0], layout=None)
    return path


def main(io_path=None, directory=None, out_dir=None):
    if io_path is None:
        io_path = DEFAULT_IO_PATH
    if directory is None:
        directory = DEFAULT_DIRECTORY
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    # Build the per-mouse comparison frame via the existing comparison
    # script (so the table on disk and the plots come from the same
    # computation).
    import compare_all_choice_methods as cmp_mod
    df = cmp_mod.main(io_path=io_path, directory=directory)

    written = []
    written.extend(plot_method_bars(df, out_dir))
    written.append(plot_moment_tradeoff(df, out_dir))
    written.append(plot_q_anatomy(mouse_id=0, split='stratified_balanced',
                                   io_path=io_path, directory=directory,
                                   out_dir=out_dir))
    print(f'\n[loss_sweep_plots] wrote {len(written)} files to {out_dir}/')
    for p in written:
        print(f'  {p}')
    return written


if __name__ == '__main__':
    main()
