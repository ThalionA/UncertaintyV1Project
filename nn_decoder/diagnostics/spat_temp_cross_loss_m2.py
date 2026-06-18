# -*- coding: utf-8 -*-
"""Item 2 — spatial vs temporal across the full train-objective × eval-metric
grid, shown WITH and WITHOUT Mouse 2, paired-t stars, raw and normalised loss.

The meeting ask: "for all loss functions (both at training and evaluation),
show the temporal/spatial performance comparison, with all stats (paired
t-tests), both raw and normalised losses, with and without Mouse 2."

"Both at training and evaluation" = the (training-loss × evaluation-metric)
matrix: every decoder, trained under one loss, scored under every loss. This is
``cross_loss_eval``'s object; here we collapse it to the spatial−temporal
*difference* and show it in two currencies × two cohorts:

  - normalised loss (test loss ÷ that arch's shuffle): Δ = norm_spat − norm_temp
  - raw test loss:                                     Δ = (spat − temp)/temp [%]
  columns = all 6 mice | Mouse 2 excluded (n=5).

Positive (green) = temporal more informative; negative (red) = spatial better.
Per-cell ``*`` p<0.05 / ``**`` p<0.01 are paired t-tests over mice, pairing
spatial vs temporal within each animal. The diagonal (each loss under its OWN
training metric) is outlined — the key contrast is the projection-based loss's
diagonal (≈0, looks calibrated) vs its KL/CE/JS columns (red: spatial ≫ temporal,
the over-sharpening surfacing only under a calibrated yardstick). Reuses the
validated ``cross_loss_eval`` engine. Conclusions live in the title/caption — no
text boxes over the data.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  spat_temp_cross_loss_m2.png    2×2 diff matrices (normalised|raw × all|no-M2)
Full per-cell stats: ``cross_loss_eval --exclude mouse_2`` (and without --exclude)
write 12_spat_temp_paired_stats.csv (raw + normalised, paired-t + Wilcoxon).

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
    """Return normalised-diff, raw-%-diff arrays and matching star grids."""
    nL, nE = len(train_losses), len(eval_losses)
    Dnorm = np.full((nL, nE), np.nan)
    Drw = np.full((nL, nE), np.nan)
    Snorm = [["" for _ in range(nE)] for _ in range(nL)]
    Srw = [["" for _ in range(nE)] for _ in range(nL)]
    for i, tl in enumerate(train_losses):
        for j, el in enumerate(eval_losses):
            ss = matrix["spat"][tl][el]["skill"][0]      # 'skill' = the normalised loss (code key)
            ts = matrix["temp"][tl][el]["skill"][0]
            sr = matrix["spat"][tl][el]["real"][0]
            tr = matrix["temp"][tl][el]["real"][0]
            if np.isfinite(ss) and np.isfinite(ts):
                Dnorm[i, j] = ss - ts
            if np.isfinite(sr) and np.isfinite(tr) and tr > 0:
                Drw[i, j] = (sr - tr) / tr * 100.0
            srl, trl, ssl, tsl = _per_mouse_spat_temp(sweep[tl], el, exclude=exclude)
            Snorm[i][j] = _stars(_paired(ssl, tsl)[4])
            Srw[i][j] = _stars(_paired(srl, trl)[4])
    return Dnorm, Drw, Snorm, Srw


def _draw(ax, D, S, train_losses, eval_losses, unit, title, vmax=None):
    is_pct = "%" in unit
    if vmax is None:
        vmax = np.nanmax(np.abs(D))
    im = ax.imshow(D, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(eval_losses)))
    ax.set_xticklabels(ps.loss_labels(eval_losses), rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(train_losses)))
    ax.set_yticklabels(ps.loss_labels(train_losses), fontsize=8)
    ax.set_xlabel("evaluation metric (held-out posteriors)", fontsize=8.5)
    ax.set_ylabel("training objective", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    for i, tl in enumerate(train_losses):
        for j, el in enumerate(eval_losses):
            v = D[i, j]
            if np.isfinite(v):
                txt = (f"{v:+.0f}%" if is_pct else f"{v:+.2f}") + S[i][j]
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.4")
            if tl == el:
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="#222222", lw=2.0))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit, fontsize=8)
    return im


def main(run, split, results_root, out_root):
    P.TARGET, P.WINDOW, P.BIN_MS, P.SPLIT = "Q", "half", 100, split
    sweep = P.load_loss_sweep(run, results_root=results_root)
    eval_losses = list(EVAL_LOSSES)
    cohorts = [(None, "all 6 mice"), ("mouse_2", "Mouse 2 excluded")]

    # compute both cohorts first so the colour scale can be shared per currency
    blocks = []
    for exc, label in cohorts:
        matrix, train_losses = build_matrix(sweep, exclude=exc)
        Dn, Dr, Sn, Sr = diff_matrices(sweep, matrix, train_losses, eval_losses, exc)
        n = sum(1 for mid in sweep[train_losses[0]]["results"] if mid != exc)
        blocks.append(dict(exc=exc, label=label, n=n, Dn=Dn, Dr=Dr, Sn=Sn, Sr=Sr,
                           train=train_losses))
    vmax_n = np.nanmax([np.nanmax(np.abs(b["Dn"])) for b in blocks])
    vmax_r = np.nanmax([np.nanmax(np.abs(b["Dr"])) for b in blocks])

    ps.apply()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.3))
    for col, b in enumerate(blocks):
        _draw(axes[0, col], b["Dn"], b["Sn"], b["train"], eval_losses,
              "normalised loss (spat − temp)",
              f"Shuffle-normalised loss — {b['label']} (n={b['n']})", vmax=vmax_n)
        _draw(axes[1, col], b["Dr"], b["Sr"], b["train"], eval_losses,
              "(spat − temp)/temp  [%]",
              f"Raw test loss — {b['label']} (n={b['n']})", vmax=vmax_r)
    ps.label_panels(axes)
    fig.suptitle("Spatial vs temporal across train-objective × eval-metric — with "
                 "& without Mouse 2.  Green = temporal better, red = spatial; "
                 "diagonal = own metric;  * p<.05 ** p<.01 (paired t)", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ps.save_fig(fig, Path(out_root), "spat_temp_cross_loss_m2", layout=None)

    # console: the headline projection-based contrast, both cohorts
    di = blocks[0]["train"].index("PCA")
    print("\nProjection-based spatial−temporal normalised-loss difference:")
    for b in blocks:
        print(f"  [{b['label']}, n={b['n']}]")
        for el in eval_losses:
            j = eval_losses.index(el)
            own = " [own metric]" if el == "PCA" else ""
            print(f"    under {ps.loss_label(el):14s}: Δ={b['Dn'][di, j]:+.2f}"
                  f"{b['Sn'][di][j]:2s}  Δraw={b['Dr'][di, j]:+6.0f}%{b['Sr'][di][j]:2s}{own}")
    print(f"\nDone. {Path(out_root).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="loss_comparison_v1")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-root", default="figures/peakiness_scatter")
    a = ap.parse_args()
    main(a.run, a.split, a.results_root, a.out_root)
