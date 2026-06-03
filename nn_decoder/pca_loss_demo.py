# -*- coding: utf-8 -*-
"""Didactic figure: what the PCA-weighted loss actually 'sees'.

The PCA fit-loss (``pca_loss.pca_distance``) projects both the predicted and
target posteriors onto a PC basis fit on the condition-averaged targets, then
penalises the squared projection difference *weighted by each PC's
explained-variance ratio*:

    loss = 100 * Σ_k  evar_k * (⟨pred, PC_k⟩ − ⟨target, PC_k⟩)²

Because ``evar`` is hugely concentrated on the first 2–3 components (for Q,
PC1 ≈ 0.91, PC2 ≈ 0.08, PC3 ≈ 0.01), the loss is almost entirely a function of
how well the prediction matches the target *along those few directions*. Any
discrepancy that lives in the remaining ~88 low-variance directions — a
high-frequency ripple, a small spurious bump — is essentially invisible to
the PCA loss even though it is plainly visible to the eye and to KL/CE/JS.

This script makes that concrete. For one example mouse it draws six figures
(all under ``figures/loss_sweep_plots/<run>/pca_loss_demo/``):

  * ``_gallery``  — the headline. A PC-loadings + scree header, then one row per
    candidate posterior (the real target, the real PPC fit, and hand-crafted
    perturbations) shown *next to* a bar chart of all five of its loss values.
    Read down a bar column to compare candidates within a loss; read across a
    row to compare losses for one shape.
  * ``_summary``  — candidate × loss heatmap (column-normalised colour).
  * ``_fitted``   — the SAME held-out trial decoded by models trained on each
    loss, for both architectures (spatial PPC and temporal SBC); shows how the
    training objective shapes the fitted posterior.
  * ``_examples_<loss>`` (one per loss) — five hand-picked example trials for
    that loss's own trained models: best spatial, best temporal, the two
    biggest spatial-vs-temporal gaps, and one where both architectures fit
    poorly; each shows the target with both fits and their per-trial losses.
  * ``_decomposition`` — per-PC loss budget (where the error lives vs what the
    loss charges).
  * ``_directional``   — equal-size perturbations along a high- vs low-variance
    PC direction; PCA is blind to one of them.
  * ``_geometry``      — the candidates in the PC1–PC2 plane with iso-PCA-loss
    ellipses (PCA loss ≈ a 2-D Mahalanobis distance).

All losses are computed via ``nn_classifier._batched_fit_loss`` — the exact
code path the training loop and ``cross_loss_eval.py`` use — so the numbers
here are the real metric values, not a re-implementation.

Usage
-----
    python pca_loss_demo.py                         # loss_comparison_v1, mouse_0
    python pca_loss_demo.py --run-name new_loss_sweep_may27 --mouse 2
    python pca_loss_demo.py --trial 137             # pin a specific target trial
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import decoder_plotting_utils as dpu
from nn_classifier import _batched_fit_loss


# Evaluation metrics, in display order. Keys are the loss_func_type strings
# _batched_fit_loss understands; values are (pretty label, unit).
LOSSES = {
    'PCA': ('PCA-weighted', '×100'),
    'KL': ('KL(target‖pred)', 'nats'),
    'CE': ('Cross-entropy', 'nats'),
    'JS': ('Jensen–Shannon', 'nats'),
    'Wasserstein': ('Wasserstein-1', 'bins'),
}


# ----------------------------------------------------------------------
# Candidate posteriors
# ----------------------------------------------------------------------

def _normalize(p):
    """Clip negatives and renormalise to a valid probability vector."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / p.size)


def _shift(p, k):
    """Shift a posterior by ``k`` bins (linear, edge-clamped), renormalised.
    Moves the peak -> large change along the dominant (low-frequency) PCs."""
    return _normalize(np.interp(np.arange(p.size) - k,
                                np.arange(p.size), p, left=0.0, right=0.0))


def _broaden(p, sigma):
    """Gaussian-smooth the posterior (wider = more uncertain-looking)."""
    x = np.arange(p.size)
    k = np.exp(-0.5 * ((x - x.mean()) / sigma) ** 2)
    k /= k.sum()
    return _normalize(np.convolve(p, k, mode='same'))


def _sharpen(p, power):
    """Raise to a power and renormalise (>1 sharpens the peak)."""
    return _normalize(np.asarray(p, dtype=float) ** power)


def _add_bump(p, center, width, weight):
    """Add a spurious Gaussian mode at ``center`` carrying ``weight`` mass."""
    x = np.arange(p.size)
    bump = np.exp(-0.5 * ((x - center) / width) ** 2)
    bump /= bump.sum()
    return _normalize((1 - weight) * _normalize(p) + weight * bump)


def _ripple(p, n_cycles, amp):
    """Multiply by a high-frequency cosine and renormalise. The perturbation
    lives almost entirely in the high-index (low-variance) PCs, so the PCA
    loss barely moves while KL/CE clearly register it."""
    x = np.arange(p.size)
    mod = 1.0 + amp * np.cos(2 * np.pi * n_cycles * x / p.size)
    return _normalize(np.asarray(p, dtype=float) * mod)


def _pc_reconstruct(p, pcs, n_keep):
    """Reconstruct ``p`` from only its first ``n_keep`` PCs, then renormalise.
    Matches the target along exactly the directions the PCA loss weights most,
    so PCA loss ≈ 0 — yet the discarded fine structure makes KL/CE non-zero."""
    proj = p @ pcs.T               # (n_components,)
    proj_kept = np.zeros_like(proj)
    proj_kept[:n_keep] = proj[:n_keep]
    return _normalize(proj_kept @ pcs)   # pcs orthonormal: inverse = transpose


def build_candidates(target, real_fit, pcs):
    """Return an ordered list of ``(name, posterior)`` candidates.

    The set is chosen to dissociate the losses: shifts/broadening/sharpening
    move mass along the dominant PCs (PCA-visible), while the ripple and the
    top-3 PC reconstruction perturb only low-variance directions (PCA-blind
    but KL/CE-visible)."""
    peak = int(np.argmax(target))
    n = target.size
    return [
        ('Target (reference)', _normalize(target)),
        ('Real fitted (PPC)', _normalize(real_fit)),
        ('Shift +8 bins', _shift(target, 8)),
        ('Broadened (σ=6)', _broaden(target, 6.0)),
        ('Sharpened (^2.5)', _sharpen(target, 2.5)),
        ('Spurious 2nd mode', _add_bump(target, (peak + n // 2) % n, 4.0, 0.30)),
        ('High-freq ripple', _ripple(target, n_cycles=12, amp=0.9)),
        ('Top-3 PC rebuild', _pc_reconstruct(target, pcs, n_keep=3)),
        ('Uniform', np.full(n, 1.0 / n)),
    ]


# ----------------------------------------------------------------------
# Loss computation
# ----------------------------------------------------------------------

def loss_value(pred, target, loss_key, pcs, evar):
    """Single-posterior loss via the canonical _batched_fit_loss path.
    ``pred`` is the candidate (the 'prediction'), ``target`` the IO target."""
    pred_t = torch.tensor(np.asarray(pred, float)[None, :])
    targ_t = torch.tensor(np.asarray(target, float)[None, :])
    pcs_t = evar_t = None
    if loss_key == 'PCA':
        pcs_t = torch.tensor(np.asarray(pcs, float))
        evar_t = torch.tensor(np.asarray(evar, float))
    per_trial = _batched_fit_loss(pred_t, targ_t, loss_key, pcs_t, evar_t)
    return float(per_trial.item())


def all_losses(candidates, target, pcs, evar):
    """Return ``{loss_key: [value per candidate]}``."""
    return {lk: [loss_value(p, target, lk, pcs, evar) for _, p in candidates]
            for lk in LOSSES}


# ----------------------------------------------------------------------
# Target selection
# ----------------------------------------------------------------------

def pick_trial(target_arr, spat_loss, requested):
    """Choose an example trial. If ``requested`` is given, use it. Otherwise
    pick an informative (clearly-peaked) target the spatial decoder fit
    reasonably well, so the real-fit candidate is a fair reference: take the
    most peaked third of trials, then the median spatial loss within it."""
    if requested is not None:
        return int(requested)
    peakiness = target_arr.max(axis=1)
    cand = np.where(peakiness >= np.nanpercentile(peakiness, 66))[0]
    if cand.size == 0:
        cand = np.arange(target_arr.shape[0])
    losses = np.where(np.isfinite(spat_loss[cand]), spat_loss[cand], np.inf)
    return int(cand[np.argsort(losses)[len(losses) // 2]])


def _loss_slug(target, loss, window, bin_ms):
    return f'{target}_{loss}_{window}_{bin_ms}ms' + ('_all' if loss == 'PCA' else '')


def load_loss_mouse_dist(run_name, target, window, bin_ms, split, mouse, loss):
    """Per-mouse ``Dist`` dict for one sibling loss run, or None if missing."""
    slug = _loss_slug(target, loss, window, bin_ms)
    tree = dpu.load_run_tree(run_name, slug, splits=[split])
    if split not in tree:
        return None
    res = tree[split]['results']
    mkey = f'mouse_{mouse}'
    return res[mkey]['Dist'] if mkey in res else None


def load_fits_by_loss(run_name, target, window, bin_ms, split, mouse, tr):
    """Load the real decoded posterior for trial ``tr`` from each sibling loss
    run, for both architectures. Returns ``{train_loss: {'spat', 'temp'}}``.

    The runs share the same split seed and mouse, so the test-trial ordering
    (hence ``tr``) is identical across losses — verified in main by checking the
    targets match. Missing runs are skipped."""
    fits = {}
    for tl in LOSSES:
        d = load_loss_mouse_dist(run_name, target, window, bin_ms, split,
                                 mouse, tl)
        if d is None:
            continue
        try:
            fits[tl] = {'spat': np.asarray(d['spat']['decoded'], float)[tr],
                        'temp': np.asarray(d['temp']['decoded'], float)[tr]}
        except (KeyError, IndexError):
            continue
    return fits


def per_trial_loss(decoded, target, loss_key, pcs, evar):
    """Per-trial loss for full (n_trials, n_cats) arrays via _batched_fit_loss."""
    dec = torch.tensor(np.asarray(decoded, float))
    tgt = torch.tensor(np.asarray(target, float))
    pcs_t = evar_t = None
    if loss_key == 'PCA':
        pcs_t = torch.tensor(np.asarray(pcs, float))
        evar_t = torch.tensor(np.asarray(evar, float))
    return _batched_fit_loss(dec, tgt, loss_key, pcs_t, evar_t).numpy()


def pick_example_trials(spat_loss, temp_loss):
    """Hand-pick five characteristic trials from per-trial spat/temp losses.
    Returns an ordered list of ``(label, trial_index)``."""
    valid = np.isfinite(spat_loss) & np.isfinite(temp_loss)
    BIG = np.inf
    s = np.where(valid, spat_loss, BIG)
    t = np.where(valid, temp_loss, BIG)
    diff_s = np.where(valid, temp_loss - spat_loss, -BIG)   # spatial wins
    diff_t = np.where(valid, spat_loss - temp_loss, -BIG)   # temporal wins
    both = np.where(valid, np.minimum(spat_loss, temp_loss), -BIG)
    return [
        ('Best spatial (PPC)', int(np.argmin(s))),
        ('Best temporal (SBC)', int(np.argmin(t))),
        ('Spatial >> Temporal', int(np.argmax(diff_s))),
        ('Temporal >> Spatial', int(np.argmax(diff_t))),
        ('Both poor', int(np.argmax(both))),
    ]


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

LOSS_COLOR = {'PCA': '#e6550d', 'KL': '#7b3294', 'CE': '#008837',
              'JS': '#c2a5cf', 'Wasserstein': '#a6611a'}


def _draw_dist(ax, s_grid, target, cand, name, color='#2c7fb8'):
    """One posterior panel: target (dashed) + candidate (filled)."""
    ax.plot(s_grid, target, 'k--', lw=1.8, label='Target')
    ax.plot(s_grid, cand, color=color, lw=2.0, label='Candidate')
    ax.fill_between(s_grid, cand, color=color, alpha=0.13)
    ax.set_title(name, fontsize=10.5)
    ax.set_ylim(-0.005, max(0.05, max(target.max(), cand.max()) * 1.18))
    ax.tick_params(labelsize=8)


def _draw_loss_bars(ax, vals_per_loss, maxv):
    """Horizontal bars of every loss for one candidate. Each bar is the value
    as a fraction of that loss's worst (largest) candidate, so the lengths are
    comparable across losses; the raw value is printed at the bar end."""
    keys = list(LOSSES)
    y = np.arange(len(keys))[::-1]            # PCA on top
    for yi, k in zip(y, keys):
        frac = vals_per_loss[k] / maxv[k] if maxv[k] > 0 else 0.0
        ax.barh(yi, frac, color=LOSS_COLOR[k], height=0.7)
        ax.text(min(frac, 1.0) + 0.02, yi, f'{vals_per_loss[k]:.2g}',
                va='center', ha='left', fontsize=8)
    ax.set_xlim(0, 1.32)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8.5)
    ax.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)


def plot_gallery_allloss(candidates, losses, s_grid, target, pcs, evar, out_dir):
    """The one-stop gallery: every candidate posterior shown *next to* all five
    of its loss values, with a PC-loadings + scree header so you can see which
    directions the PCA loss weights. Reading down the bar column compares
    candidates within a loss; reading across a row compares losses for one
    shape. This is where PCA's disagreement with the others is visible at a
    glance (e.g. the ripple: tiny PCA bar, large KL/CE bars)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    dpu.set_style()
    keys = list(LOSSES)
    maxv = {k: max(losses[k]) for k in keys}
    n = len(candidates)

    fig = plt.figure(figsize=(11.5, 3.0 + 1.5 * n))
    gs = GridSpec(n + 1, 2, figure=fig, width_ratios=[2.2, 1.0],
                  hspace=0.9, wspace=0.12,
                  height_ratios=[1.5] + [1] * n)

    # --- header: PC loadings + scree ---
    ax_pc = fig.add_subplot(gs[0, 0])
    pc_colors = ['#d7191c', '#fdae61', '#2b83ba']
    for k in range(3):
        ax_pc.plot(s_grid, pcs[k], color=pc_colors[k], lw=2.2,
                   label=f'PC{k+1}  ({evar[k]*100:.1f}% var)')
    ax_pc.axhline(0, color='0.7', lw=0.8)
    ax_pc.set_title('PC loadings — the directions the PCA loss weights',
                    fontsize=10.5)
    ax_pc.set_ylabel('Loading', fontsize=9)
    ax_pc.legend(fontsize=8, frameon=False)
    ax_pc.tick_params(labelsize=8)

    ax_sc = fig.add_subplot(gs[0, 1])
    kshow = min(8, len(evar))
    ax_sc.bar(np.arange(1, kshow + 1), np.asarray(evar[:kshow]) * 100,
              color='#555', width=0.7)
    ax_sc.set_title('Explained variance', fontsize=10.5)
    ax_sc.set_xlabel('PC index', fontsize=9)
    ax_sc.set_ylabel('% var', fontsize=9)
    ax_sc.set_xticks(np.arange(1, kshow + 1))
    ax_sc.tick_params(labelsize=7.5)

    # --- one row per candidate: distribution | all-loss bars ---
    for i, (name, cand) in enumerate(candidates):
        ax_d = fig.add_subplot(gs[i + 1, 0])
        _draw_dist(ax_d, s_grid, target, cand, name)
        if i == 0:
            ax_d.legend(fontsize=8, loc='upper right', frameon=False)
        if i == n - 1:
            ax_d.set_xlabel('Orientation (deg)', fontsize=9)
        ax_d.set_ylabel('Prob.', fontsize=8.5)

        ax_b = fig.add_subplot(gs[i + 1, 1])
        _draw_loss_bars(ax_b, {k: losses[k][i] for k in keys}, maxv)
        if i == 0:
            ax_b.set_title('loss value  (bar = vs worst candidate)',
                           fontsize=9)

    fig.suptitle('Each candidate posterior and all five of its loss values\n'
                 'bar = fraction of that loss\'s worst candidate; number = raw '
                 'value', fontsize=14, y=0.997)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_gallery.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print('  -> pca_loss_demo_gallery.png/.svg')


def plot_fitted_by_loss(fits, target, s_grid, pcs, evar, out_dir, info=''):
    """Same trial, real decoded posteriors from models trained on different
    losses — spatial (PPC) and temporal (SBC). Shows how the *training*
    objective shapes the fitted posterior on one held-out trial. ``fits`` is
    ``{train_loss: {'spat': vec, 'temp': vec}}``. Each panel is annotated with
    the fit's own-loss value and its PCA loss (the shared yardstick)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpu.set_style()
    train_losses = [l for l in LOSSES if l in fits]
    archs = [('spat', 'Spatial (PPC)', '#d95f02'),
             ('temp', 'Temporal (SBC)', '#1f78b4')]
    nrow, ncol = len(archs), len(train_losses)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.8 * nrow),
                             squeeze=False)
    for r, (arch, arch_lbl, color) in enumerate(archs):
        for c, tl in enumerate(train_losses):
            ax = axes[r][c]
            dec = fits[tl][arch]
            ax.plot(s_grid, target, 'k--', lw=1.8, label='Target')
            ax.plot(s_grid, dec, color=color, lw=2.0, label=arch_lbl)
            ax.fill_between(s_grid, dec, color=color, alpha=0.13)
            own = loss_value(dec, target, tl, pcs, evar)
            pca = loss_value(dec, target, 'PCA', pcs, evar)
            ax.text(0.96, 0.94,
                    f'{tl} loss: {own:.2g}\nPCA loss: {pca:.2g}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc='#fff3d6',
                              ec='#caa53d', lw=1.0))
            ax.set_ylim(-0.005, max(0.05, max(target.max(), dec.max()) * 1.2))
            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(f'trained on {tl}', fontsize=10.5)
            if r == nrow - 1:
                ax.set_xlabel('Orientation (deg)', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{arch_lbl}\nProbability', fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7.5, loc='upper left', frameon=False)
    fig.suptitle('Same trial, real decoded posterior by training loss '
                 f'{info}\n(dashed = IO target; box = own-loss / PCA-loss value)',
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_fitted.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print('  -> pca_loss_demo_fitted.png/.svg')


def plot_fitted_examples(dist, loss_key, s_grid, pcs, evar, out_dir, info=''):
    """Hand-picked example trials for ONE training loss. Using that loss's own
    trained spatial (PPC) and temporal (SBC) models, score every test trial and
    pick five characteristic ones — best spatial, best temporal, the two biggest
    spat-vs-temp gaps, and one where both architectures do poorly — then show
    target + both fits for each, annotated with the per-trial losses.

    ``dist`` is the per-mouse ``Dist`` for the ``loss_key`` run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpu.set_style()
    spat = np.asarray(dist['spat']['decoded'], float)
    temp = np.asarray(dist['temp']['decoded'], float)
    tgt = np.asarray(dist['spat']['target'], float)
    sl = per_trial_loss(spat, tgt, loss_key, pcs, evar)
    tl = per_trial_loss(temp, tgt, loss_key, pcs, evar)
    cats = pick_example_trials(sl, tl)

    n = len(cats)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.3 * n), squeeze=False)
    for r, (label, idx) in enumerate(cats):
        ax = axes[r][0]
        ax.plot(s_grid, tgt[idx], 'k--', lw=2.0, label='Target')
        ax.plot(s_grid, spat[idx], color='#d95f02', lw=2.0,
                label=f'Spatial (PPC)  [{sl[idx]:.2g}]')
        ax.plot(s_grid, temp[idx], color='#1f78b4', lw=2.0,
                label=f'Temporal (SBC)  [{tl[idx]:.2g}]')
        ax.fill_between(s_grid, spat[idx], color='#d95f02', alpha=0.08)
        ax.fill_between(s_grid, temp[idx], color='#1f78b4', alpha=0.08)
        ax.set_title(f'{label}   (trial {idx})', fontsize=11)
        ax.set_ylabel('Probability', fontsize=9)
        ax.legend(fontsize=8.5, loc='upper right', frameon=False,
                  title=f'{loss_key} loss [value]', title_fontsize=8)
        ax.set_ylim(-0.005,
                    max(0.05, max(tgt[idx].max(), spat[idx].max(),
                                  temp[idx].max()) * 1.18))
        ax.tick_params(labelsize=8)
        if r == n - 1:
            ax.set_xlabel('Orientation (deg)', fontsize=9)

    fig.suptitle(f'Hand-picked example trials — model trained on {loss_key} '
                 f'{info}\nselected & scored by the {loss_key} loss '
                 '(spatial vs temporal)', fontsize=13, y=1.0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_examples_{loss_key}.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(f'  -> pca_loss_demo_examples_{loss_key}.png/.svg  '
          f'(trials {[i for _, i in cats]})')


def plot_summary(candidates, losses, out_dir):
    """Heatmap of candidate × loss, each column min–max normalised so the
    *ranking* (not the incomparable raw scale) is what the colour shows.
    Raw values are printed in each cell."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpu.set_style()
    names = [n for n, _ in candidates]
    keys = list(LOSSES)
    raw = np.array([[losses[k][i] for k in keys]
                    for i in range(len(candidates))], dtype=float)

    # Per-column min-max normalisation to [0, 1]; target row (all ~0) anchors min.
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.0

    fig, ax = plt.subplots(figsize=(1.5 * len(keys) + 3.5,
                                    0.62 * len(names) + 2.2))
    im = ax.imshow(norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LOSSES[k][0] for k in keys], rotation=20, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Loss (column-normalised colour: green = best, red = worst)')
    ax.set_title('Same candidates, different losses — the ranking disagrees\n'
                 '(cell text = raw loss value; colour = rank within column)',
                 fontsize=12)
    # Box the best (lowest-loss) *non-trivial* candidate per loss — exclude the
    # Target reference (row 0, loss 0 under every metric) so the box shows where
    # the real candidates' rankings actually disagree across losses.
    masked = raw.copy()
    masked[0, :] = np.inf
    best = np.nanargmin(masked, axis=0)
    for j in range(len(keys)):
        for i in range(len(names)):
            ax.text(j, i, f'{raw[i, j]:.2g}', ha='center', va='center',
                    fontsize=8, color='black')
        ax.add_patch(matplotlib.patches.Rectangle(
            (j - 0.5, best[j] - 0.5), 1, 1, fill=False, edgecolor='#15561f',
            lw=2.4))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='rank within loss (0 = best)')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_summary.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print('  -> pca_loss_demo_summary.png/.svg')


def plot_pc_decomposition(candidates, target, pcs, evar, s_grid, out_dir,
                          which=('Shift +8 bins', 'High-freq ripple')):
    """Per-PC loss budget. For two contrasting candidates, show side by side:
      * top  — the candidate vs target in posterior space (what you see);
      * mid  — the *unweighted* squared projection error per PC (where the
               discrepancy actually lives);
      * bot  — the *evar-weighted* contribution 100·evar_k·Δ_k² (what the PCA
               loss charges; the bars sum to the PCA loss).
    The shift's error sits in PC1–2 and is fully charged; the ripple's error
    sits in high PCs that carry ~0 weight, so it is charged almost nothing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpu.set_style()
    cdict = dict(candidates)
    proj_t = np.asarray(target) @ pcs.T
    kshow = min(40, len(evar))
    ks = np.arange(1, kshow + 1)

    fig, axes = plt.subplots(3, len(which), figsize=(6.2 * len(which), 9.2),
                             gridspec_kw=dict(hspace=0.42, wspace=0.26))
    for c, name in enumerate(which):
        cand = cdict[name]
        proj = np.asarray(cand) @ pcs.T
        d2 = (proj - proj_t) ** 2
        contrib = 100 * evar * d2
        pca_total = contrib.sum()

        ax0, ax1, ax2 = axes[0, c], axes[1, c], axes[2, c]
        ax0.plot(s_grid, target, 'k--', lw=2.0, label='Target')
        ax0.plot(s_grid, cand, color='#2c7fb8', lw=2.2, label=name)
        ax0.fill_between(s_grid, cand, color='#2c7fb8', alpha=0.12)
        ax0.set_title(name, fontsize=12)
        ax0.set_ylabel('Probability', fontsize=9)
        ax0.set_xlabel('Orientation (deg)', fontsize=9)
        ax0.legend(fontsize=8.5, frameon=False)

        ax1.bar(ks, d2[:kshow], color='#888', width=0.8)
        ax1.set_title('Unweighted error per PC  (where the discrepancy lives)',
                      fontsize=10)
        ax1.set_ylabel('(Δproj)²', fontsize=9)
        ax1.set_xlabel('PC index', fontsize=9)

        ax2.bar(ks, contrib[:kshow], color='#d7191c', width=0.8)
        ax2.set_title(f'evar-weighted contribution  (Σ = PCA loss = '
                      f'{pca_total:.3g})', fontsize=10)
        ax2.set_ylabel('100·evar·(Δproj)²', fontsize=9)
        ax2.set_xlabel('PC index', fontsize=9)

    fig.suptitle('Per-PC loss budget — the PCA loss only charges the first few '
                 'PCs\nbig errors in high (low-variance) PCs cost almost nothing',
                 fontsize=14, y=0.995)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_decomposition.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print('  -> pca_loss_demo_decomposition.png/.svg')


def plot_directional_sensitivity(target, pcs, evar, out_dir, n_steps=16):
    """Move the posterior the same L2 distance along a low-frequency ('visible')
    vs a high-frequency ('blind') direction and plot how each loss responds.

    The directions are multiplicative modulations target·(1 + c·cos), so every
    candidate stays a valid non-negative distribution (no clipping artefacts).
    A low-frequency modulation (1 cycle) lands in PC1–2 — the directions the
    PCA loss weights; a high-frequency modulation (12 cycles) lands in the
    low-variance PCs it ignores. Plotted against the *actual* distance moved
    (‖cand − target‖₂), so the two panels are on a common x-axis. PCA flatlines
    along the blind direction while KL/CE/JS/W register it as readily as the
    visible one. Each loss is scaled by its own max along the visible sweep."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpu.set_style()
    target = _normalize(target)
    x = np.arange(target.size)
    dirs = [('visible direction  (1-cycle modulation → PC1–2)',
             np.cos(2 * np.pi * 1 * x / target.size)),
            ('blind direction  (12-cycle modulation → high PCs)',
             np.cos(2 * np.pi * 12 * x / target.size))]
    cs = np.linspace(0.0, 0.95, n_steps)
    # CE is dropped here: for a single target CE = H(target) + KL, so its
    # *response* to a perturbation is identical to KL (just offset by the
    # constant target entropy). Showing both would be redundant.
    keys = ['PCA', 'KL', 'JS', 'Wasserstein']
    palette = {'PCA': '#e6550d', 'KL': '#7b3294', 'JS': '#c2a5cf',
               'Wasserstein': '#a6611a'}

    # dist[label] -> actual ‖Δ‖₂ per step;  curves[(label, lk)] -> Δloss vs c=0.
    dist, curves = {}, {}
    for label, g in dirs:
        u = target * g                         # posterior-space direction
        d_arr = []
        per_loss = {lk: [] for lk in keys}
        base = {lk: loss_value(target, target, lk, pcs, evar) for lk in keys}
        for c in cs:
            cand = _normalize(target + c * u)  # = target·(1 + c·g) ≥ 0
            d_arr.append(float(np.linalg.norm(cand - target)))
            for lk in keys:
                per_loss[lk].append(
                    loss_value(cand, target, lk, pcs, evar) - base[lk])
        dist[label] = np.array(d_arr)
        for lk in keys:
            curves[(label, lk)] = np.array(per_loss[lk])

    vis_label = dirs[0][0]
    ref = {lk: max(curves[(vis_label, lk)].max(), 1e-12) for lk in keys}
    xmax = min(dist[dirs[0][0]].max(), dist[dirs[1][0]].max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (label, _) in zip(axes, dirs):
        for lk in keys:
            ax.plot(dist[label], curves[(label, lk)] / ref[lk], marker='o',
                    ms=4, color=palette[lk], label=LOSSES[lk][0], lw=2)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('L2 distance moved in posterior space  ||cand - target||',
                      fontsize=10)
        ax.set_xlim(0, xmax)
        ax.axhline(0, color='0.8', lw=0.8)
    axes[0].set_ylabel('loss  (relative to its max along the visible direction)',
                       fontsize=10)
    axes[1].legend(fontsize=9, frameon=False, loc='upper left')
    fig.suptitle('Same distance moved, different directions — PCA is blind to '
                 'one of them\nlow- vs high-frequency modulation of equal L2 '
                 'size; only PCA cares which', fontsize=14, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_directional.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print('  -> pca_loss_demo_directional.png/.svg')


def plot_projection_geometry(candidates, losses, target, pcs, evar, out_dir):
    """The candidates in the PC1–PC2 plane, with iso-PCA-loss ellipses.

    The PCA loss is ≈ 100·(evar₁·ΔPC1² + evar₂·ΔPC2²) — a Mahalanobis distance
    in just the top-2 PCs (here 99% of the variance). So a candidate's position
    in this plane nearly determines its loss, and anything that differs from the
    target only in the orthogonal (PC3+) directions lands on top of it. The
    ellipses are loss level-sets: narrow along PC1 (heavily weighted), wide
    along PC2 (lightly weighted)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpu.set_style()
    proj_t = np.asarray(target) @ pcs.T
    names = [n for n, _ in candidates]
    dx = np.array([(np.asarray(p) @ pcs.T)[0] - proj_t[0] for _, p in candidates])
    dy = np.array([(np.asarray(p) @ pcs.T)[1] - proj_t[1] for _, p in candidates])
    pca_vals = np.array(losses['PCA'])

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    # Frame to the data (equal aspect so the Mahalanobis geometry reads true).
    lim = 1.25 * max(np.abs(dx).max(), np.abs(dy).max(), 1e-3)

    # Iso-loss ellipses: 100·(evar0·x² + evar1·y²) = L. Narrow along PC1
    # (heavily weighted), wide along PC2 (lightly weighted).
    th = np.linspace(0, 2 * np.pi, 200)
    for L in [0.03, 0.1, 0.35, 1.0]:
        x_int = np.sqrt(L / (100 * evar[0]))   # PC1 intercept (stays in-frame)
        ax.plot(x_int * np.cos(th),
                np.sqrt(L / (100 * evar[1])) * np.sin(th), color='0.75',
                lw=1.0, zorder=1)
        ax.text(x_int, 0.004, f'PCA={L:g}', color='0.5', fontsize=8,
                ha='center', va='bottom', rotation=90)

    ax.axhline(0, color='0.85', lw=0.8)
    ax.axvline(0, color='0.85', lw=0.8)
    ax.scatter([0], [0], marker='*', s=300, color='k', zorder=6)
    cmap = plt.get_cmap('tab10')
    for i, name in enumerate(names):
        ax.scatter(dx[i], dy[i], s=110, color=cmap(i % 10), zorder=5,
                   edgecolor='k', lw=0.6)
        ax.annotate(str(i + 1), (dx[i], dy[i]), ha='center', va='center',
                    fontsize=8, fontweight='bold', zorder=7,
                    color='white' if i % 10 not in (8,) else 'black')

    # Legend maps marker number -> candidate name + its full PCA loss.
    handles = [plt.Line2D([], [], marker='o', ls='', mfc=cmap(i % 10),
                          mec='k', ms=9,
                          label=f'{i+1}. {names[i]}  (PCA {pca_vals[i]:.2g})')
               for i in range(len(names))]
    ax.legend(handles=handles, fontsize=8.5, frameon=True, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), title='Candidate (PCA loss)')

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f'Δ along PC1  ({evar[0]*100:.1f}% var — heavily weighted)',
                  fontsize=10)
    ax.set_ylabel(f'Δ along PC2  ({evar[1]*100:.1f}% var — lightly weighted)',
                  fontsize=10)
    ax.set_title('PCA loss is a Mahalanobis distance in the top-2 PC plane\n'
                 'candidates differing only in PC3+ collapse onto the target',
                 fontsize=13)
    ax.set_aspect('equal')
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'pca_loss_demo_geometry.{ext}',
                    bbox_inches='tight', dpi=140)
    plt.close(fig)
    print('  -> pca_loss_demo_geometry.png/.svg')


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(run_name='loss_comparison_v1', target='Q', window='half', bin_ms=100,
         split='stratified_balanced', mouse=0, trial=None,
         example_losses=tuple(LOSSES), out_root='figures/loss_sweep_plots'):
    slug = f'{target}_PCA_{window}_{bin_ms}ms_all'
    tree = dpu.load_run_tree(run_name, slug, splits=[split])
    if split not in tree:
        raise SystemExit(f'No data for {run_name}/{slug}/{split}.mat')

    res = tree[split]['results']
    mouse_key = f'mouse_{mouse}'
    if mouse_key not in res:
        raise SystemExit(f'{mouse_key} not in {list(res.keys())}')
    dist = res[mouse_key]['Dist']
    pcs = np.asarray(dist['pcs'], dtype=float)            # (n_comp, n_cats)
    evar = np.asarray(dist['explained_var'], dtype=float)
    target_arr = np.asarray(dist['spat']['target'], dtype=float)
    spat_dec = np.asarray(dist['spat']['decoded'], dtype=float)

    spat_loss = dpu.calc_pca_dist(target_arr, spat_dec, pcs, evar)
    tr = pick_trial(target_arr, spat_loss, trial)
    tgt = target_arr[tr]
    fit = spat_dec[tr]
    n_cats = tgt.size
    s_grid = np.arange(n_cats)

    print(f'PCA loss demo: {run_name} | {slug} | {split} | {mouse_key} | trial {tr}')
    print(f'  n_cats={n_cats}  evar[:3]={evar[:3].round(4)}  '
          f'(top-3 sum {evar[:3].sum()*100:.1f}%)')

    candidates = build_candidates(tgt, fit, pcs)
    tgt_n = _normalize(tgt)
    losses = all_losses(candidates, tgt_n, pcs, evar)

    out_dir = Path(out_root) / run_name / 'pca_loss_demo'

    # Main gallery: every candidate posterior next to ALL five of its losses,
    # under a PC-loadings + scree header.
    plot_gallery_allloss(candidates, losses, s_grid, tgt_n, pcs, evar, out_dir)

    # Overview heatmap of candidate × loss.
    plot_summary(candidates, losses, out_dir)

    # Same trial, real decoded posteriors from each TRAINING loss (spat & temp).
    fits = load_fits_by_loss(run_name, target, window, bin_ms, split, mouse, tr)
    if fits:
        plot_fitted_by_loss(fits, tgt_n, s_grid, pcs, evar, out_dir,
                            info=f'—  {mouse_key}, trial {tr}')
    else:
        print('  [skip] fitted-by-loss: no sibling loss runs found')

    # Per loss func: hand-picked example trials (best spat / best temp / the two
    # biggest spat-vs-temp gaps / both poor), scored by that loss's own models.
    for L in example_losses:
        d = load_loss_mouse_dist(run_name, target, window, bin_ms, split,
                                 mouse, L)
        if d is None:
            print(f'  [skip] examples({L}): run not found')
            continue
        plot_fitted_examples(d, L, s_grid, pcs, evar, out_dir,
                             info=f'—  {mouse_key}')

    # Mechanistic demonstrations of *why* the PCA loss behaves this way.
    plot_pc_decomposition(candidates, tgt_n, pcs, evar, s_grid, out_dir)
    plot_directional_sensitivity(tgt_n, pcs, evar, out_dir)
    plot_projection_geometry(candidates, losses, tgt_n, pcs, evar, out_dir)
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--mouse', type=int, default=0)
    ap.add_argument('--trial', type=int, default=None,
                    help='Pin a specific target trial for the candidate gallery '
                         '/ fitted figure (else auto-pick a clearly-peaked, '
                         'well-fit one).')
    ap.add_argument('--example-losses', nargs='+', default=list(LOSSES),
                    help='Which training losses get a hand-picked example-trials '
                         'figure (default: all).')
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(run_name=a.run_name, target=a.target, window=a.window, bin_ms=a.bin_ms,
         split=a.split, mouse=a.mouse, trial=a.trial,
         example_losses=tuple(a.example_losses), out_root=a.out_root)
