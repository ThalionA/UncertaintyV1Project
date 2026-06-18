# -*- coding: utf-8 -*-
"""Item 2 — spatial vs temporal across the full train-objective × eval-metric
grid, Mouse-2 excluded, paired-t stars, both raw and shuffle-normalised.

The meeting ask: "for all loss functions (both at training and evaluation),
show the temporal/spatial performance comparison excluding M2, with all stats
(paired t-tests), both raw and normalised losses."

"Both at training and evaluation" = the (training-loss × evaluation-metric)
matrix: every decoder, trained under one loss, scored under every loss. This is
exactly ``cross_loss_eval``'s object; here we collapse it to the spatial−temporal
*difference* and show it in two currencies side by side:

  - skill  (shuffle-normalised): Δ = skill_spat − skill_temp   (dimensionless)
  - raw    (absolute test loss):  Δ = (spat − temp)/temp [%]

Positive (green) = temporal more informative; negative (red) = spatial better.
Per-cell ``*`` p<0.05 / ``**`` p<0.01 are paired t-tests over mice (n=5 with M2
dropped), pairing spatial vs temporal within each animal. The diagonal (each
loss scored under its OWN training metric) is outlined — the key contrast is
PCA's diagonal/own-metric (≈0, looks calibrated) vs its KL/CE/JS columns (red:
spatial ≫ temporal, the over-sharpening surfacing only under a calibrated yard-
stick). Reuses the validated ``cross_loss_eval`` engine (same maths as training).

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  spat_temp_cross_loss_m2.png    two diff matrices (skill | raw), stars
The full per-cell stats are written by ``cross_loss_eval --exclude mouse_2``
to 12_spat_temp_paired_stats.csv (raw + skill, paired-t + Wilcoxon).

Run:  cd nn_decoder && OMP_NUM_THREADS=1 \
        python diagnostics/spat_temp_cross_loss_m2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps                                          # noqa: E402
import plot_loss_sweep as P                                          # noqa: E402
from cross_loss_eval import (                                        # noqa: E402
    build_matrix, _per_mouse_spat_temp, _paired, _stars, EVAL_LOSSES,
)


def diff_matrices(sweep, matrix, train_losses, eval_losses, exclude):
    """Return skill-diff, raw-%-diff arrays and matching star grids."""
    nL, nE = len(train_losses), len(eval_losses)
    Dsk = np.full((nL, nE), np.nan)
    Drw = np.full((nL, nE), np.nan)
    Ssk = [["" for _ in range(nE)] for _ in range(nL)]
    Srw = [["" for _ in range(nE)] for _ in range(nL)]
    for i, tl in enumerate(train_losses):
        for j, el in enumerate(eval_losses):
            ss = matrix["spat"][tl][el]["skill"][0]
            ts = matrix["temp"][tl][el]["skill"][0]
            sr = matrix["spat"][tl][el]["real"][0]
            tr = matrix["temp"][tl][el]["real"][0]
            if np.isfinite(ss) and np.isfinite(ts):
                Dsk[i, j] = ss - ts
            if np.isfinite(sr) and np.isfinite(tr) and tr > 0:
                Drw[i, j] = (sr - tr) / tr * 100.0
            srl, trl, ssl, tsl = _per_mouse_spat_temp(sweep[tl], el, exclude=exclude)
            Ssk[i][j] = _stars(_paired(ssl, tsl)[4])
            Srw[i][j] = _stars(_paired(srl, trl)[4])
    return Dsk, Drw, Ssk, Srw


def _draw(ax, D, S, train_losses, eval_losses, unit, title):
    vmax = np.nanmax(np.abs(D))
    im = ax.imshow(D, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(eval_losses)))
    ax.set_xticklabels(ps.loss_labels(eval_losses), rotation=20, ha="right")
    ax.set_yticks(range(len(train_losses)))
    ax.set_yticklabels(ps.loss_labels(train_losses))
    ax.set_xlabel("evaluation metric (held-out posteriors)")
    ax.set_ylabel("training objective")
    ax.set_title(title, fontsize=10)
    for i, tl in enumerate(train_losses):
        for j, el in enumerate(eval_losses):
            v = D[i, j]
            if np.isfinite(v):
                txt = (f"{v:+.2f}" if "skill" in unit else f"{v:+.0f}%") + S[i][j]
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.4")
            if tl == el:                                  # outline the diagonal
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="#222222", lw=2.0))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit)
    return im


def main(run, split, exclude, results_root, out_root):
    P.TARGET, P.WINDOW, P.BIN_MS, P.SPLIT = "Q", "half", 100, split
    sweep = P.load_loss_sweep(run, results_root=results_root)
    matrix, train_losses = build_matrix(sweep, exclude=exclude)
    eval_losses = list(EVAL_LOSSES)
    Dsk, Drw, Ssk, Srw = diff_matrices(sweep, matrix, train_losses,
                                       eval_losses, exclude)
    n = sum(1 for mid in sweep[train_losses[0]]["results"] if mid != exclude)

    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    _draw(axes[0], Dsk, Ssk, train_losses, eval_losses,
          "skill (spat − temp)", "Shuffle-normalised skill difference")
    _draw(axes[1], Drw, Srw, train_losses, eval_losses,
          "(spat − temp)/temp  [%]", "Raw test-loss difference")
    ps.label_panels(axes)
    fig.suptitle(f"Spatial vs temporal across train-objective × eval-metric "
                 f"({exclude} excluded, n={n}; green = temporal better, red = "
                 f"spatial; diagonal outlined; * p<.05 ** p<.01 paired-t)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ps.save_fig(fig, Path(out_root), "spat_temp_cross_loss_m2", layout=None)

    # console: the headline PCA contrast (diagonal vs KL column)
    di = train_losses.index("PCA")
    print(f"\nPCA spatial−temporal skill diff, n={n} (M2 excluded):")
    for el in eval_losses:
        j = eval_losses.index(el)
        own = " [own metric]" if el == "PCA" else ""
        print(f"  under {el:11s}: Δskill={Dsk[di, j]:+.2f}{Ssk[di][j]:2s}  "
              f"Δraw={Drw[di, j]:+6.0f}%{Srw[di][j]:2s}{own}")
    print(f"\nDone. {Path(out_root).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="loss_comparison_v1")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--exclude", default="mouse_2")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-root", default="figures/peakiness_scatter")
    a = ap.parse_args()
    main(a.run, a.split, a.exclude, a.results_root, a.out_root)
