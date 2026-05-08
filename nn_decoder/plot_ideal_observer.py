# -*- coding: utf-8 -*-
"""
Ideal Observer (IO) plots across stimulus conditions.

Five focused diagnostics for the *real* IO outputs in `VR_Decoder_Data_Export.mat`
(no synthetic / decoded models — see decoder_sanity_check_v26.py for those):

  1. plot_io_posterior_shapes
       Mean posterior shape on the orientation grid, for the perceptual posterior
       Q(theta) and the marginal likelihood L(theta), split by Orientation,
       Contrast, and Dispersion. Reads off how the IO's belief shape changes
       with each stimulus feature.

  2. plot_io_uncertainty_tuning
       Tuning of the three uncertainty targets (Perceptual Variance, Likelihood
       Variance, Decision Entropy) across stimulus features, with the explicit
       Contrast x Dispersion interactions visualised.

  3. plot_io_psychometric
       IO P(Go) vs orientation split by Contrast (left) and Dispersion (right),
       overlaid with the animal's actual choice psychometric. The gap shows
       deviation between IO-optimal and behaviour.

  4. plot_io_map_bias
       Mean MAP of the perceptual posterior vs the true orientation, plus the
       signed deviation (MAP - true). The bimodal Go/No-Go prior should pull
       MAPs toward 0/90 deg, more strongly at low contrast.

  5. plot_io_lik_vs_post_kl
       KL divergence of L(theta) from Q(theta) — a "prior strength" diagnostic.
       Skipped if the export does not contain the marginal likelihood.

All five are produced as a Grand Average across mice and as one figure per
mouse, all written to `IO_Plots/` as SVGs.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from utils_v26 import load_vr_export, calculate_np_variance


S_GRID = np.arange(0, 91, 1)

# Palette choices: seaborn's mako / flare / crest are sequential and stay
# dark enough to remain legible on a white background. Each stimulus
# feature gets its own distinct hue family so panels are easy to tell
# apart at a glance.
ORI_CMAP = sns.color_palette("mako_r", as_cmap=True)    # blue → teal
CONT_CMAP = sns.color_palette("flare", as_cmap=True)    # orange → red
DISP_CMAP = sns.color_palette("crest_r", as_cmap=True)  # green → blue


def _cmap_color(cmap, i, n):
    """Sample a colormap at the i-th of n points, biased into the
    darker / more saturated middle so we never reach the very pale ends."""
    if n <= 1:
        return cmap(0.5)
    return cmap(0.15 + 0.7 * i / (n - 1))


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------

def set_style():
    sns.set_context("talk")
    sns.set_style("ticks")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _entropy_bits(d2):
    """Per-trial Bernoulli entropy of (P_Go, P_NoGo), normalised to [0, 1]."""
    p = np.clip(d2[:, 0], 1e-10, 1 - 1e-10)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return h / np.log(2.0)


def _kl_divergence(p, q, eps=1e-10):
    """Per-trial KL(p || q) for two distributions on the same grid."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.nansum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)


def _pooled_mean_sem(per_mouse_arr):
    """Across-mouse mean and SEM, omitting NaNs. Inputs shape (n_mice, n_x).
    Returns shape (n_x,) for both. With 1 mouse, SEM is NaN by construction
    (matches the sanity-check behaviour: per-mouse plots show no error bars)."""
    mean = np.nanmean(per_mouse_arr, axis=0)
    if per_mouse_arr.shape[0] > 1:
        sem = stats.sem(per_mouse_arr, axis=0, nan_policy='omit')
    else:
        sem = np.full_like(mean, np.nan, dtype=float)
    return mean, sem


def _errorbar_or_line(ax, x, mean, sem, **kw):
    """Plot mean +/- SEM if SEM is defined, else just a line."""
    if sem is not None and np.any(np.isfinite(sem)):
        ax.errorbar(x, mean, yerr=sem, capsize=3, **kw)
    else:
        ax.plot(x, mean, **{k: v for k, v in kw.items() if k != 'capsize'})


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_io_data(mouse_ids=range(6), filepath=None):
    """Build the per-mouse data dicts used by the plotting functions.

    Each entry has:
        mouse_id, trials, targets_perc, targets_lik, targets_dec,
        perc_var, lik_var, dec_ent, p_go (animal P(Go) per trial),
        animal_choice (binary).
    """
    all_data = []
    for mid in mouse_ids:
        try:
            (_, targets_perc, targets_dec, targets_lik,
             trials) = load_vr_export(mid, filepath)
        except Exception as e:
            print(f"  Skipping Mouse {mid}: {e}")
            continue
        perc_var = calculate_np_variance(targets_perc, S_GRID)
        lik_var = (calculate_np_variance(targets_lik, S_GRID)
                   if targets_lik is not None else None)
        dec_ent = _entropy_bits(targets_dec)
        all_data.append({
            'mouse_id': mid,
            'trials': trials,
            'targets_perc': targets_perc,
            'targets_lik': targets_lik,
            'targets_dec': targets_dec,
            'perc_var': perc_var,
            'lik_var': lik_var,
            'dec_ent': dec_ent,
            'io_p_go': targets_dec[:, 0],
            'animal_choice': np.asarray(trials['choice']),
        })
        print(f"  Loaded Mouse {mid}: {len(perc_var)} trials")
    return all_data


# ----------------------------------------------------------------------
# 1. Posterior shapes per condition
# ----------------------------------------------------------------------

def plot_io_posterior_shapes(data_list, unique_oris, unique_conts, unique_disps,
                             title_prefix="Grand Average"):
    """2 rows (Q, L) x 3 cols (split by Ori / Contrast / Dispersion).
    Each cell shows mean posterior curve on the s_grid, one line per category,
    pooled across the supplied mice. Likelihood row is skipped if no mouse has it."""
    has_lik = any(d['targets_lik'] is not None for d in data_list)
    n_rows = 2 if has_lik else 1

    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows),
                             sharex=True, squeeze=False)

    splits = [
        ('Orientation', unique_oris, 'orientation', ORI_CMAP),
        ('Contrast', unique_conts, 'contrast', CONT_CMAP),
        ('Dispersion', unique_disps, 'dispersion', DISP_CMAP),
    ]
    target_rows = [('Perceptual Posterior  Q(θ)', 'targets_perc')]
    if has_lik:
        target_rows.append(('Marginal Likelihood  L(θ)', 'targets_lik'))

    for r, (target_label, target_key) in enumerate(target_rows):
        for c, (split_label, vals, trial_key, cmap) in enumerate(splits):
            ax = axes[r, c]
            for i, val in enumerate(vals):
                pooled = []
                for d in data_list:
                    if d.get(target_key) is None:
                        continue
                    mask = (d['trials'][trial_key] == val)
                    if np.sum(mask) == 0:
                        continue
                    pooled.append(np.nanmean(d[target_key][mask], axis=0))
                if not pooled:
                    continue
                mean = np.nanmean(np.stack(pooled), axis=0)
                ax.plot(S_GRID, mean,
                        color=_cmap_color(cmap, i, len(vals)),
                        lw=2, label=f"{val}")
            if r == 0:
                ax.set_title(f"Split by {split_label}")
            if c == 0:
                ax.set_ylabel(f"{target_label}\nProbability")
            if r == n_rows - 1:
                ax.set_xlabel("|Δ from Go| (deg)")
            ax.legend(fontsize='x-small', title=split_label, loc='upper right')

    fig.suptitle(f"{title_prefix}: IO Posterior & Likelihood Shapes",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ----------------------------------------------------------------------
# 2. Tuning of the three uncertainty targets
# ----------------------------------------------------------------------

def plot_io_uncertainty_tuning(data_list, unique_oris, unique_conts, unique_disps,
                               title_prefix="Grand Average"):
    """3 rows (perc_var, lik_var, dec_ent) x 3 cols
    (Ori split by Cont, Ori split by Disp, Cont split by Disp).
    Lines = mean across mice of per-mouse means; error bars = SEM across mice."""
    n_mice = len(data_list)
    targets = [
        ('perc_var', 'Perceptual Variance (deg²)'),
        ('lik_var', 'Likelihood Variance (deg²)'),
        ('dec_ent', 'Decision Entropy (norm.)'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex='col')

    def _per_mouse_grouped(target_key, x_key, x_vals, hue_key, hue_vals):
        """Returns dict {hue_val: array of shape (n_mice, len(x_vals))}."""
        out = {h: np.full((n_mice, len(x_vals)), np.nan) for h in hue_vals}
        for i, d in enumerate(data_list):
            if d.get(target_key) is None:
                continue
            for h in hue_vals:
                hue_mask = (d['trials'][hue_key] == h)
                for j, xv in enumerate(x_vals):
                    m = hue_mask & (d['trials'][x_key] == xv)
                    if m.sum() > 0:
                        out[h][i, j] = np.nanmean(d[target_key][m])
        return out

    panel_specs = [
        ('orientation', unique_oris, 'contrast', unique_conts, CONT_CMAP, 'Contrast'),
        ('orientation', unique_oris, 'dispersion', unique_disps, DISP_CMAP, 'Dispersion'),
        ('contrast', unique_conts, 'dispersion', unique_disps, DISP_CMAP, 'Dispersion'),
    ]

    for r, (target_key, target_label) in enumerate(targets):
        for c, (xk, xv, hk, hv, cmap, hue_label) in enumerate(panel_specs):
            ax = axes[r, c]
            grouped = _per_mouse_grouped(target_key, xk, xv, hk, hv)
            for i, h in enumerate(hv):
                mean, sem = _pooled_mean_sem(grouped[h])
                _errorbar_or_line(ax, xv, mean, sem,
                                  marker='o', lw=2,
                                  color=_cmap_color(cmap, i, len(hv)),
                                  label=f"{h}")
            if r == 0:
                ax.set_title(f"{xk.title()} × {hue_label}")
            if c == 0:
                ax.set_ylabel(target_label)
            if r == 2:
                ax.set_xlabel(xk.title())
            if c == 0 or (target_key == 'lik_var' and not any(d.get('lik_var') is not None for d in data_list)):
                pass
            if xk == 'contrast':
                ax.set_xscale('log')
            ax.legend(title=hue_label, fontsize='x-small', loc='best')

    fig.suptitle(f"{title_prefix}: IO Uncertainty Tuning", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ----------------------------------------------------------------------
# 3. IO vs animal psychometric
# ----------------------------------------------------------------------

def plot_io_psychometric(data_list, unique_oris, unique_conts, unique_disps,
                         title_prefix="Grand Average"):
    """1 row x 2 cols: P(Go) vs Orientation, split by Contrast and by Dispersion.
    Solid line = IO P(Go); dashed line = animal P(Go); both pooled mean ± SEM."""
    n_mice = len(data_list)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    def _io_and_animal_per_mouse(hue_key, hue_vals):
        io_arr = {h: np.full((n_mice, len(unique_oris)), np.nan) for h in hue_vals}
        an_arr = {h: np.full((n_mice, len(unique_oris)), np.nan) for h in hue_vals}
        for i, d in enumerate(data_list):
            for h in hue_vals:
                hue_mask = (d['trials'][hue_key] == h)
                for j, o in enumerate(unique_oris):
                    m = hue_mask & (d['trials']['orientation'] == o)
                    if m.sum() > 0:
                        io_arr[h][i, j] = np.nanmean(d['io_p_go'][m])
                        an_arr[h][i, j] = np.nanmean(d['animal_choice'][m])
        return io_arr, an_arr

    for ax, (hk, hv, cmap, label) in zip(
            axes,
            [('contrast', unique_conts, CONT_CMAP, 'Contrast'),
             ('dispersion', unique_disps, DISP_CMAP, 'Dispersion')]):
        io_arr, an_arr = _io_and_animal_per_mouse(hk, hv)
        for i, h in enumerate(hv):
            color = _cmap_color(cmap, i, len(hv))
            io_mean, io_sem = _pooled_mean_sem(io_arr[h])
            an_mean, an_sem = _pooled_mean_sem(an_arr[h])
            _errorbar_or_line(ax, unique_oris, io_mean, io_sem,
                              marker='o', lw=2, color=color,
                              label=f"IO {label} {h}")
            _errorbar_or_line(ax, unique_oris, an_mean, an_sem,
                              marker='s', lw=2, color=color, ls='--',
                              alpha=0.6, label=f"Animal {label} {h}")
        ax.axvline(45, color='k', ls=':', alpha=0.4)
        ax.set_xlabel("|Δ from Go| (deg)")
        ax.set_title(f"P(Go) vs Orientation — split by {label}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize='xx-small', ncol=2, loc='best')
    axes[0].set_ylabel("P(Go)")
    fig.suptitle(f"{title_prefix}: IO vs Animal Psychometric", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ----------------------------------------------------------------------
# 4. MAP / bias of the perceptual posterior
# ----------------------------------------------------------------------

def plot_io_map_bias(data_list, unique_oris, unique_conts,
                     title_prefix="Grand Average"):
    """1 row x 2 cols: MAP of Q(θ) vs true orientation (split by Contrast),
    and signed error (MAP − true) vs true orientation (split by Contrast).
    Reveals prior pull toward 0/90 deg, especially at low contrast."""
    n_mice = len(data_list)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    map_arr = {c: np.full((n_mice, len(unique_oris)), np.nan) for c in unique_conts}
    err_arr = {c: np.full((n_mice, len(unique_oris)), np.nan) for c in unique_conts}

    for i, d in enumerate(data_list):
        for c in unique_conts:
            cmask = (d['trials']['contrast'] == c)
            for j, o in enumerate(unique_oris):
                m = cmask & (d['trials']['orientation'] == o)
                if m.sum() == 0:
                    continue
                # MAP per trial = argmax over s_grid
                trial_maps = S_GRID[np.argmax(d['targets_perc'][m], axis=1)]
                map_arr[c][i, j] = np.nanmean(trial_maps)
                err_arr[c][i, j] = np.nanmean(trial_maps - o)

    cmap = CONT_CMAP
    for i, c in enumerate(unique_conts):
        color = _cmap_color(cmap, i, len(unique_conts))
        m_mean, m_sem = _pooled_mean_sem(map_arr[c])
        e_mean, e_sem = _pooled_mean_sem(err_arr[c])
        _errorbar_or_line(axes[0], unique_oris, m_mean, m_sem,
                          marker='o', lw=2, color=color, label=f"Contrast {c}")
        _errorbar_or_line(axes[1], unique_oris, e_mean, e_sem,
                          marker='o', lw=2, color=color, label=f"Contrast {c}")

    # Diagonal on the MAP panel
    diag = np.array([unique_oris.min(), unique_oris.max()])
    axes[0].plot(diag, diag, ls='--', color='gray', lw=1.5, alpha=0.6,
                 label='Identity')
    axes[0].set_xlabel("True |Δ from Go| (deg)")
    axes[0].set_ylabel("Mean MAP of Q(θ) (deg)")
    axes[0].set_title("Perceptual MAP vs True Orientation")
    axes[0].legend(fontsize='x-small', loc='best')

    axes[1].axhline(0, color='gray', ls='--', lw=1.5, alpha=0.6)
    axes[1].set_xlabel("True |Δ from Go| (deg)")
    axes[1].set_ylabel("Signed error  (MAP − true, deg)")
    axes[1].set_title("Prior-Induced Bias")
    axes[1].legend(fontsize='x-small', loc='best')

    fig.suptitle(f"{title_prefix}: IO Perceptual MAP & Bias", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ----------------------------------------------------------------------
# 5. Likelihood vs Posterior KL — prior strength
# ----------------------------------------------------------------------

def plot_io_lik_vs_post_kl(data_list, unique_conts, unique_disps,
                           title_prefix="Grand Average"):
    """KL(L || Q) per trial, mean across trials by Contrast (split by Dispersion).
    Returns None if no mouse has the marginal likelihood available."""
    have_lik = [d for d in data_list if d['targets_lik'] is not None]
    if not have_lik:
        return None

    n_mice = len(have_lik)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    arr = {dv: np.full((n_mice, len(unique_conts)), np.nan) for dv in unique_disps}
    for i, d in enumerate(have_lik):
        kl_per_trial = _kl_divergence(d['targets_lik'], d['targets_perc'])
        for dv in unique_disps:
            dmask = (d['trials']['dispersion'] == dv)
            for j, c in enumerate(unique_conts):
                m = dmask & (d['trials']['contrast'] == c)
                if m.sum() > 0:
                    arr[dv][i, j] = np.nanmean(kl_per_trial[m])

    cmap = DISP_CMAP
    for i, dv in enumerate(unique_disps):
        color = _cmap_color(cmap, i, len(unique_disps))
        mean, sem = _pooled_mean_sem(arr[dv])
        _errorbar_or_line(ax, unique_conts, mean, sem,
                          marker='o', lw=2, color=color,
                          label=f"Dispersion {dv}")
    ax.set_xscale('log')
    ax.set_xlabel("Contrast")
    ax.set_ylabel("KL( L(θ)  ||  Q(θ) )  (nats)")
    ax.set_title(f"{title_prefix}: Prior Pull (Likelihood → Posterior KL)")
    ax.legend(fontsize='x-small', loc='best')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _generate_all(data_list, unique_oris, unique_conts, unique_disps,
                  title_prefix):
    figs = {}
    figs['1_Posterior_Shapes'] = plot_io_posterior_shapes(
        data_list, unique_oris, unique_conts, unique_disps, title_prefix)
    figs['2_Uncertainty_Tuning'] = plot_io_uncertainty_tuning(
        data_list, unique_oris, unique_conts, unique_disps, title_prefix)
    figs['3_Psychometric'] = plot_io_psychometric(
        data_list, unique_oris, unique_conts, unique_disps, title_prefix)
    figs['4_MAP_Bias'] = plot_io_map_bias(
        data_list, unique_oris, unique_conts, title_prefix)
    fig5 = plot_io_lik_vs_post_kl(
        data_list, unique_conts, unique_disps, title_prefix)
    if fig5 is not None:
        figs['5_Lik_vs_Post_KL'] = fig5
    return figs


def main(mouse_ids=range(6), output_dir="IO_Plots", filepath=None):
    set_style()
    print("Loading IO data...")
    all_data = load_io_data(mouse_ids, filepath=filepath)
    if not all_data:
        print("[!] No mice loaded.")
        return

    unique_oris = np.unique(np.concatenate([d['trials']['orientation'] for d in all_data]))
    unique_conts = np.unique(np.concatenate([d['trials']['contrast'] for d in all_data]))
    unique_disps = np.unique(np.concatenate([d['trials']['dispersion'] for d in all_data]))
    print(f"  Orientations: {unique_oris}")
    print(f"  Contrasts   : {unique_conts}")
    print(f"  Dispersions : {unique_disps}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving SVGs to ./{output_dir}/")

    # Grand average
    print("Building Grand Average figures...")
    grand_figs = _generate_all(all_data, unique_oris, unique_conts, unique_disps,
                                "Grand Average")
    for name, fig in grand_figs.items():
        out = os.path.join(output_dir, f"GrandAvg_{name}.svg")
        fig.savefig(out, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")

    # Per-mouse
    for d in all_data:
        mid = d['mouse_id']
        print(f"Building Mouse {mid} figures...")
        mouse_figs = _generate_all([d], unique_oris, unique_conts, unique_disps,
                                    f"Mouse {mid}")
        for name, fig in mouse_figs.items():
            out = os.path.join(output_dir, f"Mouse{mid}_{name}.svg")
            fig.savefig(out, format='svg', bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {out}")

    print("Done.")


if __name__ == "__main__":
    main()
