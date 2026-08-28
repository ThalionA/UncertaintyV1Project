# -*- coding: utf-8 -*-
"""IO posterior shape vs contrast, for one mouse, averaged over dispersions.

One figure:
  * top-left    mean Q(theta) per contrast, pooled over every trial (all
                orientations, all dispersions) — the literal "average posterior
                at this contrast".
  * top-middle  the design: trial counts per (contrast x dispersion) cell.
                Contrast and dispersion are **not crossed** in this experiment,
                so "averaged across dispersions" means a different dispersion
                mix at every contrast. The panel makes that visible.
  * top-right   posterior entropy H(Q) vs contrast, one line per dispersion —
                the uncertainty ordering without the pooling confound.
  * rows 2-4    mean Q(theta) per contrast, one panel per stimulus orientation
                (|delta from Go|), averaged over dispersions. The pooled panel
                smears across orientations; these do not.

Usage
-----
    python plot_io_posteriors_by_contrast.py                # mouse 1 (M1)
    python plot_io_posteriors_by_contrast.py --mouse 0
    python plot_io_posteriors_by_contrast.py --weight equal --target lik
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from figsave import save_fig
from utils import load_vr_export

S_GRID = np.arange(0, 91)
CONT_CMAP = sns.color_palette("flare", as_cmap=True)   # orange -> red, low -> high contrast
DISP_CMAP = sns.color_palette("crest_r", as_cmap=True)

TARGETS = {
    'perc': (1, r'Perceptual posterior  $Q(\theta)$'),
    'lik':  (3, r'Marginal likelihood  $L(\theta)$'),
}


def _cmap_color(cmap, i, n):
    """Sample a colormap, biased away from its washed-out low end."""
    return cmap(0.25 + 0.7 * (i / max(n - 1, 1)))


def _mean_posterior(post, mask_c, disp, u_disp, weight):
    """Mean posterior over the trials in ``mask_c``.

    ``weight='trial'`` pools trials directly (so a dispersion with 392 trials
    dominates one with 29). ``weight='equal'`` averages the per-dispersion means
    with equal weight, over whichever dispersions this contrast actually has.
    Returns ``(curve, n_trials)`` or ``(None, 0)``.
    """
    n = int(mask_c.sum())
    if n == 0:
        return None, 0
    if weight == 'trial':
        return post[mask_c].mean(axis=0), n
    per_disp = [post[mask_c & (disp == d)].mean(axis=0)
                for d in u_disp if (mask_c & (disp == d)).sum() > 0]
    return np.mean(np.stack(per_disp), axis=0), n


def make_figure(mouse_id, target='perc', weight='trial'):
    idx, target_label = TARGETS[target]
    out = load_vr_export(mouse_id)
    post, trials = out[idx], out[4]
    if post is None:
        raise SystemExit(f"mouse {mouse_id} has no '{target}' target in the export")

    ori = np.asarray(trials['orientation'])
    cont = np.asarray(trials['contrast'])
    disp = np.asarray(trials['dispersion'])
    u_ori, u_cont, u_disp = np.unique(ori), np.unique(cont), np.unique(disp)
    colors = {c: _cmap_color(CONT_CMAP, i, len(u_cont)) for i, c in enumerate(u_cont)}

    n_ori_rows = int(np.ceil(len(u_ori) / 3))
    fig, axes = plt.subplots(1 + n_ori_rows, 3, figsize=(12.5, 3.1 * (1 + n_ori_rows)))

    # --- top-left: pooled over everything -----------------------------------
    ax = axes[0, 0]
    for c in u_cont:
        curve, n = _mean_posterior(post, cont == c, disp, u_disp, weight)
        if curve is None:
            continue
        ax.plot(S_GRID, curve, color=colors[c], lw=2, label=f"{c:g}  (n={n})")
    ax.set_title("Pooled over all orientations", fontsize=10)
    ax.set_xlabel(r"$\theta$  |$\Delta$ from Go| (deg)")
    ax.set_ylabel("Probability")
    ax.legend(title="Contrast", fontsize='xx-small', title_fontsize='x-small')

    # --- top-middle: the design ---------------------------------------------
    ax = axes[0, 1]
    counts = np.array([[int(((cont == c) & (disp == d)).sum()) for d in u_disp]
                       for c in u_cont])
    im = ax.imshow(counts, cmap='Greys', aspect='auto')
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, counts[i, j], ha='center', va='center', fontsize=8,
                    color='white' if counts[i, j] > counts.max() * 0.6 else 'black')
    ax.set_xticks(range(len(u_disp)), [f"{d:g}" for d in u_disp])
    ax.set_yticks(range(len(u_cont)), [f"{c:g}" for c in u_cont])
    ax.set_xlabel("Dispersion (deg)")
    ax.set_ylabel("Contrast")
    ax.set_title("Trials per cell — contrast x dispersion\nis NOT crossed", fontsize=10)

    # --- top-right: entropy, contrast x dispersion --------------------------
    ax = axes[0, 2]
    ent = -np.sum(post * np.log(np.clip(post, 1e-12, None)), axis=1)
    for i, d in enumerate(u_disp):
        m = np.array([ent[(cont == c) & (disp == d)].mean()
                      if ((cont == c) & (disp == d)).sum() else np.nan
                      for c in u_cont])
        se = np.array([ent[(cont == c) & (disp == d)].std(ddof=1) /
                       np.sqrt(max(((cont == c) & (disp == d)).sum(), 1))
                       if ((cont == c) & (disp == d)).sum() > 1 else np.nan
                       for c in u_cont])
        ax.errorbar(u_cont, m, yerr=se, marker='o', lw=1.8, ms=4,
                    color=_cmap_color(DISP_CMAP, i, len(u_disp)), label=f"{d:g}")
    # reference lines: the prior's entropy (contrast -> 0 limit) and the uniform bound
    ax.axhline(ent[cont == u_cont[0]].mean(), color='0.45', ls='--', lw=0.9, zorder=0)
    ax.axhline(np.log(post.shape[1]), color='0.7', ls=':', lw=0.9, zorder=0)
    ax.annotate("prior", xy=(0.02, ent[cont == u_cont[0]].mean()),
                xycoords=('axes fraction', 'data'), fontsize=7, color='0.45',
                va='bottom')
    ax.annotate(f"uniform (ln {post.shape[1]})", xy=(0.02, np.log(post.shape[1])),
                xycoords=('axes fraction', 'data'), fontsize=7, color='0.6',
                va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel("Contrast")
    ax.set_ylabel("H  (nats)")
    ax.set_title("Entropy (mean +- SEM over trials)", fontsize=10)
    ax.legend(title="Dispersion", fontsize='xx-small', title_fontsize='x-small',
              loc='center left')

    # --- per-orientation panels ---------------------------------------------
    flat = axes[1:].ravel()
    for k, o in enumerate(u_ori):
        ax = flat[k]
        for c in u_cont:
            curve, n = _mean_posterior(post, (ori == o) & (cont == c),
                                       disp, u_disp, weight)
            if curve is None:
                continue
            ax.plot(S_GRID, curve, color=colors[c], lw=1.8, label=f"{c:g} (n={n})")
        ax.axvline(o, color='0.35', ls='--', lw=0.9, zorder=0)
        ax.set_title(rf"$|\Delta$go$|$ = {o:g}$^\circ$   (n={int((ori == o).sum())})",
                     fontsize=9)
        ax.legend(fontsize=5.5, loc='best', handlelength=1.0, frameon=False)
        if k % 3 == 0:
            ax.set_ylabel("Probability")
        if k >= len(u_ori) - 3:
            ax.set_xlabel(r"$\theta$ (deg)")
    for ax in flat[len(u_ori):]:
        ax.set_visible(False)

    wtxt = ("trial-weighted" if weight == 'trial'
            else "dispersions weighted equally")
    fig.suptitle(
        f"M{mouse_id} — IO {target_label} by contrast, averaged over dispersions "
        f"({wtxt})\n"
        f"dashed line = stimulus orientation; contrast and dispersion are not crossed "
        f"(see design panel)",
        fontsize=11)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mouse', type=int, default=1,
                   help='mouse index into the export (0-5); default 1 = M1')
    p.add_argument('--target', choices=('perc', 'lik'), default='perc',
                   help="'perc' = perceptual posterior Q (default), 'lik' = likelihood L")
    p.add_argument('--weight', choices=('trial', 'equal'), default='trial',
                   help='how dispersions are combined within a contrast')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures', 'io_posteriors_by_contrast')
    fig = make_figure(args.mouse, target=args.target, weight=args.weight)
    stem = f"M{args.mouse}_{args.target}_by_contrast_{args.weight}weighted"
    save_fig(fig, out_dir, stem)


if __name__ == '__main__':
    main()
