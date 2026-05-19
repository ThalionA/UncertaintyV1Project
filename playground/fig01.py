"""
Figure 1 — Bundle archetypes in V1 state space, in two prototype-separation regimes.

Each row is one regime:
  Top    — EASY:       well-separated prototypes (the cartoon V1).
  Bottom — REALISTIC:  close prototypes (real V1 — shared evoked-response,
                       small discriminative subspace).

Each column shows the same two views:
  Left   — bundle spaghetti, means, single trial, discriminative axis.
  Right  — covariance ellipses at four xG snapshots showing bundle width.

The framework's distinctive predictions live mostly in the bottom row —
in the realistic regime where discriminative info is a small slice of the
total response variance and bundle width starts to matter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

import framework_common as fc
from framework_common import (
    OUTDIR, T, N, DT, K, START, ou_traj,
    C_L, C_R, C_R_RESP, C_GRAY,
)


def _build_bundles(end_L, end_R, seed_offset=0):
    """Build mean trajectories + K spaghetti samples per side for a given
    prototype-endpoint pair."""
    mean_L = START + fc.RAMP[:, None] * (end_L - START)
    mean_R = START + fc.RAMP[:, None] * (end_R - START)
    trajs_L = np.stack([mean_L + ou_traj(N, seed=seed_offset + 100 + k)
                        for k in range(K)])
    trajs_R = np.stack([mean_R + ou_traj(N, seed=seed_offset + 200 + k)
                        for k in range(K)])
    return mean_L, mean_R, trajs_L, trajs_R


def _draw_panel_A(ax, mean_L, mean_R, trajs_L, trajs_R, end_L, end_R, title):
    # Test trial drawn from R class
    r_t = mean_R + ou_traj(N, sigma=0.25, tau=0.25, seed=42)

    for k in range(K):
        ax.plot(trajs_L[k, :, 0], trajs_L[k, :, 1],
                color=C_L, alpha=0.22, lw=0.75)
        ax.plot(trajs_R[k, :, 0], trajs_R[k, :, 1],
                color=C_R, alpha=0.22, lw=0.75)
    ax.plot(mean_L[:, 0], mean_L[:, 1], color=C_L, lw=2.4,
            label=r"$\bar{\mathbf{a}}_L(t)$")
    ax.plot(mean_R[:, 0], mean_R[:, 1], color=C_R, lw=2.4,
            label=r"$\bar{\mathbf{a}}_R(t)$")
    ax.plot(r_t[:, 0], r_t[:, 1], color=C_R_RESP, lw=1.7, zorder=4,
            label=r"$\mathbf{r}(t)$")
    ax.scatter(0, 0, c="k", s=40, marker="x", lw=2, zorder=6)
    ax.scatter(*START, c=C_GRAY, s=52, edgecolor="k", lw=1, zorder=6)
    ax.scatter(*mean_L[-1], c=C_L, s=180, marker="*",
               edgecolor="k", lw=1.0, zorder=7)
    ax.scatter(*mean_R[-1], c=C_R, s=180, marker="*",
               edgecolor="k", lw=1.0, zorder=7)
    # Discriminative axis
    disc_dir = end_R - end_L
    mid = 0.5 * (end_L + end_R)
    arrow_start = mid - 0.5 * disc_dir
    arrow_end = mid + 0.5 * disc_dir
    ax.annotate("", xy=arrow_end, xytext=arrow_start,
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.2,
                                linestyle=(0, (4, 2))))
    sep_deg = np.degrees(np.arccos(
        np.dot(end_L - START, end_R - START)
        / (np.linalg.norm(end_L - START) * np.linalg.norm(end_R - START))))
    ax.set_aspect("equal")
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.set_title(f"{title}\n(angular separation ≈ {sep_deg:.0f}°)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.18)


def _draw_panel_B(ax, mean_L, mean_R, trajs_L, trajs_R, title):
    for k in range(K):
        ax.plot(trajs_L[k, :, 0], trajs_L[k, :, 1],
                color=C_L, alpha=0.10, lw=0.6)
        ax.plot(trajs_R[k, :, 0], trajs_R[k, :, 1],
                color=C_R, alpha=0.10, lw=0.6)
    ax.plot(mean_L[:, 0], mean_L[:, 1], color=C_L, lw=1.8, alpha=0.85)
    ax.plot(mean_R[:, 0], mean_R[:, 1], color=C_R, lw=1.8, alpha=0.85)
    snapshots = [0.05, 0.15, 0.35, 0.70]
    for t_snap in snapshots:
        i = int(t_snap / DT)
        for trajs, color in [(trajs_L, C_L), (trajs_R, C_R)]:
            pts = trajs[:, i, :]
            mu = pts.mean(axis=0)
            cov = np.cov(pts, rowvar=False) + 1e-6 * np.eye(2)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            w, h = 2 * 1.5 * np.sqrt(np.maximum(eigvals, 0))
            el = Ellipse(mu, w, h, angle=angle, facecolor=color,
                         edgecolor=color, alpha=0.22, lw=1.2)
            ax.add_patch(el)
            ax.scatter(*mu, c=color, s=22, edgecolor="white", lw=0.7, zorder=5)
    ax.scatter(0, 0, c="k", s=40, marker="x", lw=2, zorder=6)
    ax.scatter(*START, c=C_GRAY, s=52, edgecolor="k", lw=1, zorder=6)
    handles = [
        Patch(facecolor=C_L, edgecolor=C_L, alpha=0.35,
              label=r"$\mathcal{A}_L$ — 1.5$\sigma$"),
        Patch(facecolor=C_R, edgecolor=C_R, alpha=0.35,
              label=r"$\mathcal{A}_R$ — 1.5$\sigma$"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.18)


fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0))
fig.suptitle("Figure 1 — Bundle archetypes in two prototype-separation regimes",
             fontsize=12.5, fontweight="bold", y=0.995)

regimes = [
    ("EASY (well-separated)", fc.END_L_EASY, fc.END_R_EASY, 0),
    ("REALISTIC (close prototypes — V1-like)", fc.END_L, fc.END_R, 1000),
]
for row_idx, (regime_name, end_L, end_R, seed_off) in enumerate(regimes):
    mean_L, mean_R, trajs_L, trajs_R = _build_bundles(end_L, end_R, seed_off)
    _draw_panel_A(axes[row_idx, 0], mean_L, mean_R, trajs_L, trajs_R,
                  end_L, end_R, f"A{row_idx+1}. Bundles — {regime_name}")
    _draw_panel_B(axes[row_idx, 1], mean_L, mean_R, trajs_L, trajs_R,
                  f"B{row_idx+1}. Covariance ellipses — {regime_name}")
    # Force shared limits for visual comparison within a row
    xs_full = list(axes[row_idx, 0].get_xlim()) + list(axes[row_idx, 1].get_xlim())
    ys_full = list(axes[row_idx, 0].get_ylim()) + list(axes[row_idx, 1].get_ylim())
    for ax in axes[row_idx]:
        ax.set_xlim(min(xs_full), max(xs_full))
        ax.set_ylim(min(ys_full), max(ys_full))

fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(f"{OUTDIR}/01_bundle_archetypes_state_space.png")
print("Figure 1 saved.")
plt.close()
