"""Publication-style schematic of the SI-framework network architecture.

Draws the full pipeline -- grating stimulus -> V1 population -> premotor
readout neurons -> drift-diffusion accumulator -> choice -- with the
reward-modulated plasticity loop. Run:

    python -m si_network_model.architecture_diagram

Produces figures/fig0_architecture.png.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from .config import Config  # noqa: E402
from .paths import FIGURES_DIR  # noqa: E402
from .stimuli import render_grating  # noqa: E402

C_GO = "#d62728"
C_NOGO = "#1f77b4"
C_INK = "#222222"
C_REW = "#2ca02c"
C_GREY = "#8a8a8a"


def _arrow(ax, x0, y0, x1, y1, color=C_INK, lw=2.4, rad=0.0, ls="-",
           scale=22, zorder=3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=scale, lw=lw,
        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        zorder=zorder))


def make_architecture_figure(path=None) -> str:
    """Render the architecture schematic; return the saved path."""
    cfg = Config()
    fig = plt.figure(figsize=(15.0, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "SI-framework network model — from grating to choice",
            ha="center", fontsize=13, weight="bold", color=C_INK)
    for x, label in [(0.085, "STIMULUS"), (0.345, "V1 POPULATION"),
                     (0.605, "PREMOTOR READOUT"), (0.80, "ACCUMULATOR"),
                     (0.945, "CHOICE")]:
        ax.text(x, 0.905, label, ha="center", fontsize=9.5, weight="bold",
                color=C_GREY)

    # ---- 1. stimulus grating (a real rendered grating) ----
    grating = render_grating(28.0, 1.0, 8.0, cfg)
    gax = fig.add_axes([0.022, 0.42, 0.125, 0.40])
    gax.imshow(grating, cmap="gray", vmin=0, vmax=1)
    gax.set_xticks([])
    gax.set_yticks([])
    for s in gax.spines.values():
        s.set_edgecolor(C_INK)
        s.set_linewidth(1.4)
    ax.text(0.085, 0.34, "oriented grating\norientation × contrast × dispersion",
            ha="center", va="top", fontsize=8.3, style="italic", color=C_INK)

    # ---- 2. V1 population ----
    rng = np.random.default_rng(7)
    n_show, cols = 27, 3
    vx, vy, vpref = [], [], []
    for i in range(n_show):
        vx.append(0.305 + (i % cols) * 0.030 + rng.uniform(-0.007, 0.007))
        vy.append(0.235 + (i // cols) * 0.066 + rng.uniform(-0.009, 0.009))
        vpref.append(rng.uniform(0, 180))
    vx, vy, vpref = np.array(vx), np.array(vy), np.array(vpref)
    go_xy, nogo_xy = (0.605, 0.66), (0.605, 0.40)

    # V1 -> readout connections; line weight encodes the learned archetype
    go_w = ((1 + np.cos(2 * np.deg2rad(vpref))) / 2) ** 1.6
    nogo_w = ((1 - np.cos(2 * np.deg2rad(vpref))) / 2) ** 1.6
    for i in range(n_show):
        ax.plot([vx[i], go_xy[0]], [vy[i], go_xy[1]], color=C_GO,
                lw=0.25 + 2.6 * go_w[i], alpha=0.20 + 0.4 * go_w[i], zorder=1)
        ax.plot([vx[i], nogo_xy[0]], [vy[i], nogo_xy[1]], color=C_NOGO,
                lw=0.25 + 2.6 * nogo_w[i], alpha=0.20 + 0.4 * nogo_w[i],
                zorder=1)
    ax.scatter(vx, vy, s=165, c=vpref, cmap="twilight", vmin=0, vmax=180,
               edgecolors=C_INK, linewidths=0.8, zorder=4)
    ax.text(0.345, 0.16, "~320 Gabor-tuned neurons\nrecurrent (tuning "
            "similarity) — fixed", ha="center", va="top", fontsize=8.3,
            style="italic", color=C_INK)
    # recurrence arc
    ax.add_patch(FancyArrowPatch((0.30, 0.80), (0.30, 0.27), arrowstyle="-|>",
                 mutation_scale=12, lw=1.6, color=C_GREY,
                 connectionstyle="arc3,rad=0.62", zorder=2))
    ax.text(0.225, 0.55, "recurrent", rotation=90, ha="center", va="center",
            fontsize=7.6, color=C_GREY, style="italic")
    # orientation colour key
    cax = fig.add_axes([0.285, 0.10, 0.115, 0.018])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 180), cmap="twilight"),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 90, 180])
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("preferred orientation (°)", fontsize=6.8)

    # ---- 3. premotor readout neurons ----
    for (x, y), col, lab in [(go_xy, C_GO, "Go"), (nogo_xy, C_NOGO, "NoGo")]:
        ax.scatter([x], [y], s=3000, c=col, edgecolors=C_INK, linewidths=1.7,
                   zorder=5)
        ax.text(x, y, lab, ha="center", va="center", color="white",
                fontsize=10.5, weight="bold", zorder=6)
    ax.text(0.605, 0.16, "2 readout neurons; the plastic weight\nvectors are "
            "the learned archetypes", ha="center", va="top", fontsize=8.3,
            style="italic", color=C_INK)
    ax.text(go_xy[0] + 0.052, go_xy[1] + 0.045, r"$\bar{a}_{Go}$",
            fontsize=10.5, color=C_GO, weight="bold")
    ax.text(nogo_xy[0] + 0.052, nogo_xy[1] - 0.065, r"$\bar{a}_{NoGo}$",
            fontsize=10.5, color=C_NOGO, weight="bold")

    # SI(t)
    _arrow(ax, 0.668, 0.53, 0.726, 0.53)
    ax.text(0.697, 0.595, r"$SI(t)$", ha="center", fontsize=12, color=C_INK,
            weight="bold")
    ax.text(0.697, 0.475, "cosine\ndifference", ha="center", va="top",
            fontsize=7.2, style="italic", color=C_GREY)

    # ---- 4. accumulator (drift-diffusion) ----
    dax = fig.add_axes([0.733, 0.40, 0.135, 0.34])
    r2 = np.random.default_rng(4)
    steps = 0.016 + 0.052 * r2.standard_normal(160)
    x_acc = np.cumsum(steps)
    hit = int(np.argmax(x_acc >= 1.0)) if np.any(x_acc >= 1.0) else len(x_acc) - 1
    tt = np.linspace(0, 1, hit + 1)
    x_acc = np.clip(x_acc[:hit + 1], -1.05, 1.0)
    dax.axhline(1.0, color=C_GO, lw=2.4)
    dax.axhline(-1.0, color=C_NOGO, lw=2.4)
    dax.axhline(0.0, color="#cccccc", lw=0.9, ls=":")
    dax.plot(tt, x_acc, color=C_INK, lw=1.7)
    dax.scatter([tt[-1]], [x_acc[-1]], s=55, color=C_GO, edgecolors=C_INK,
                zorder=5)
    dax.set_xlim(0, 1)
    dax.set_ylim(-1.22, 1.22)
    dax.set_xticks([])
    dax.set_yticks([])
    for s in dax.spines.values():
        s.set_visible(False)
    ax.text(0.868, 0.715, "Go bound", fontsize=6.9, color=C_GO, ha="left")
    ax.text(0.868, 0.305, "NoGo bound", fontsize=6.9, color=C_NOGO, ha="left")
    ax.text(0.80, 0.30, "leaky integration of SI(t)\nto a decision bound",
            ha="center", va="top", fontsize=8.3, style="italic", color=C_INK)

    # ---- 5. choice ----
    _arrow(ax, 0.873, 0.57, 0.918, 0.57)
    ax.text(0.948, 0.605, "Lick", ha="center", fontsize=11, weight="bold",
            color=C_GO)
    ax.text(0.948, 0.55, "(Go)", ha="center", fontsize=8, style="italic",
            color=C_GO)
    ax.text(0.948, 0.475, "No-lick", ha="center", fontsize=11, weight="bold",
            color=C_NOGO)
    ax.text(0.948, 0.42, "(NoGo)", ha="center", fontsize=8, style="italic",
            color=C_NOGO)

    # ---- stage arrow: grating -> V1 ----
    _arrow(ax, 0.16, 0.60, 0.255, 0.60)

    # ---- 6. reward / plasticity loop ----
    ax.add_patch(FancyArrowPatch((0.948, 0.36), (0.475, 0.52), arrowstyle="-|>",
                 mutation_scale=20, lw=2.2, color=C_REW,
                 linestyle=(0, (7, 3)), connectionstyle="arc3,rad=-0.42",
                 zorder=2))
    ax.text(0.66, 0.045, "reward  →  three-factor Hebbian plasticity  "
            "(updates the archetype weight vectors only)", ha="center",
            fontsize=8.6, color=C_REW, weight="bold", style="italic")

    out = str(path) if path else str(FIGURES_DIR / "fig0_architecture.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    print(make_architecture_figure())
