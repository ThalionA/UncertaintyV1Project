"""
Figure 2 — Cosine as projection onto S^{N-1}, in two prototype-separation regimes.

Each row is one regime (EASY top, REALISTIC bottom). Each row has:
  A. 3D state space — bundle spaghetti with magnitude-collapse demo
  B. Unit sphere — bundles become bands on S^2
  C. 1-D angular shadow θ(t) — the readout cosine actually sees

The framework's "discrimination reduces to angle on the sphere" story is
unambiguous in the easy regime (bundles on opposite sides of the sphere)
but visibly tighter in the realistic regime, where the two bundle bands
overlap angularly and bundle width starts to dominate discriminability.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import framework_common as fc
from framework_common import (
    OUTDIR, T, N, K, RAMP, ou_traj,
    C_L, C_R, C_R_RESP,
)


def _normalize_traj(traj):
    nrm = np.linalg.norm(traj, axis=1, keepdims=True) + 1e-9
    return traj / nrm


def _unit_sphere(ax, alpha=0.10, n_u=24, n_v=16, color="lightgray"):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)
    ax.plot_surface(x, y, z, color=color, alpha=0.04, linewidth=0)


def _angle_between(x, y):
    nx = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    ny = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)
    dot = np.clip(np.einsum("ij,ij->i", nx, ny), -1, 1)
    return np.arccos(dot)


def _build_3d_bundles(end_L_2d, end_R_2d, start_2d=fc.START, seed_offset=0):
    """Lift 2-D prototype endpoints into 3-D state space and build bundles."""
    z_offset_L = 0.6 if (end_L_2d[0] < 0) else 0.45
    z_offset_R = 0.9 if (end_R_2d[1] < 0) else 0.55
    end_L = np.array([end_L_2d[0], end_L_2d[1], z_offset_L])
    end_R = np.array([end_R_2d[0], end_R_2d[1], z_offset_R])
    start = np.array([start_2d[0], start_2d[1], 0.25])
    mean_L = start + RAMP[:, None] * (end_L - start)
    mean_R = start + RAMP[:, None] * (end_R - start)
    trajs_L = np.stack([mean_L + ou_traj(N, sigma=0.22, tau=0.30, dim=3,
                                         seed=seed_offset + 100 + k)
                        for k in range(K)])
    trajs_R = np.stack([mean_R + ou_traj(N, sigma=0.22, tau=0.30, dim=3,
                                         seed=seed_offset + 200 + k)
                        for k in range(K)])
    r_t = mean_R + ou_traj(N, sigma=0.25, tau=0.25, dim=3,
                           seed=seed_offset + 42)
    return mean_L, mean_R, trajs_L, trajs_R, r_t


def _draw_row(fig, gs_row, mean_L, mean_R, trajs_L, trajs_R, r_t, regime_label):
    # ---- Panel A: 3D state space
    axA = fig.add_subplot(gs_row[0], projection="3d")
    for k in range(K):
        axA.plot(trajs_L[k, :, 0], trajs_L[k, :, 1], trajs_L[k, :, 2],
                 color=C_L, alpha=0.20, lw=0.6)
        axA.plot(trajs_R[k, :, 0], trajs_R[k, :, 1], trajs_R[k, :, 2],
                 color=C_R, alpha=0.20, lw=0.6)
    axA.plot(*mean_L.T, color=C_L, lw=2.2)
    axA.plot(*mean_R.T, color=C_R, lw=2.2)
    axA.plot(*r_t.T, color=C_R_RESP, lw=1.4)
    axA.scatter(0, 0, 0, c="k", s=40, marker="x", zorder=10)
    axA.set_xlabel("PC 1", fontsize=8, labelpad=-8)
    axA.set_ylabel("PC 2", fontsize=8, labelpad=-8)
    axA.set_zlabel("PC 3", fontsize=8, labelpad=-8)
    axA.tick_params(labelsize=6, pad=-2)
    axA.set_title(f"A. State space  —  {regime_label}", fontsize=9.5, pad=2)
    axA.view_init(elev=22, azim=-58)

    # ---- Panel B: unit sphere
    axB = fig.add_subplot(gs_row[1], projection="3d")
    _unit_sphere(axB, alpha=0.18)
    th = np.linspace(0, 2 * np.pi, 100)
    axB.plot(np.cos(th), np.sin(th), np.zeros_like(th),
             color="gray", lw=0.5, alpha=0.4)
    for k in range(K):
        n_L = _normalize_traj(trajs_L[k])
        n_R = _normalize_traj(trajs_R[k])
        axB.plot(n_L[:, 0], n_L[:, 1], n_L[:, 2],
                 color=C_L, alpha=0.35, lw=0.6)
        axB.plot(n_R[:, 0], n_R[:, 1], n_R[:, 2],
                 color=C_R, alpha=0.35, lw=0.6)
    n_mean_L = _normalize_traj(mean_L)
    n_mean_R = _normalize_traj(mean_R)
    axB.plot(*n_mean_L.T, color=C_L, lw=2.4)
    axB.plot(*n_mean_R.T, color=C_R, lw=2.4)
    axB.scatter(*n_mean_L[-1], c=C_L, s=160, marker="*",
                edgecolor="k", lw=1.0, zorder=10)
    axB.scatter(*n_mean_R[-1], c=C_R, s=160, marker="*",
                edgecolor="k", lw=1.0, zorder=10)
    axB.set_xticks([]); axB.set_yticks([]); axB.set_zticks([])
    axB.set_title(f"B. Unit sphere  —  {regime_label}", fontsize=9.5, pad=2)
    axB.view_init(elev=22, azim=-58)
    axB.set_box_aspect((1, 1, 1))

    # ---- Panel C: angular shadow
    axC = fig.add_subplot(gs_row[2])
    ang_r_to_L = np.degrees(_angle_between(r_t, mean_L))
    ang_r_to_R = np.degrees(_angle_between(r_t, mean_R))
    sep = np.degrees(_angle_between(mean_L, mean_R)).mean()

    # Bundle angular widths (typical)
    ang_L_to_L = np.stack([np.degrees(_angle_between(trajs_L[k], mean_L))
                           for k in range(K)])
    ang_R_to_R = np.stack([np.degrees(_angle_between(trajs_R[k], mean_R))
                           for k in range(K)])
    width_L = np.mean(ang_L_to_L, axis=0) + np.std(ang_L_to_L, axis=0)
    width_R = np.mean(ang_R_to_R, axis=0) + np.std(ang_R_to_R, axis=0)

    axC.plot(T, ang_r_to_L, color=C_L, lw=2.0,
             label=r"$\angle$ to $\bar{\mathbf{a}}_L$")
    axC.plot(T, ang_r_to_R, color=C_R, lw=2.0,
             label=r"$\angle$ to $\bar{\mathbf{a}}_R$")
    axC.fill_between(T, 0, width_R, color=C_R, alpha=0.18,
                     label="R-bundle cone")
    axC.fill_between(T, np.maximum(0, np.mean(ang_r_to_L) - width_L),
                     np.mean(ang_r_to_L) + width_L,
                     color=C_L, alpha=0.13, label="L-bundle cone (around L mean)")
    axC.set_xlabel("time within trial (s)", fontsize=8)
    axC.set_ylabel("angle (deg)", fontsize=8)
    axC.set_ylim(-5, max(195, np.max(ang_r_to_L) * 1.05))
    axC.set_title(f"C. Angular shadow θ(t)  —  bundle sep ≈ {sep:.0f}°",
                  fontsize=9.5)
    axC.legend(fontsize=7, loc="center right", framealpha=0.9)
    axC.tick_params(labelsize=7)
    axC.grid(alpha=0.2)


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15, 10.5))
gs = GridSpec(2, 3, figure=fig, wspace=0.10, hspace=0.32,
              width_ratios=[1.0, 1.0, 0.9])

regimes = [
    (fc.END_L_EASY, fc.END_R_EASY, 0, "EASY"),
    (fc.END_L, fc.END_R, 1000, "REALISTIC"),
]
for row_idx, (end_L, end_R, seed_off, label) in enumerate(regimes):
    mean_L, mean_R, trajs_L, trajs_R, r_t = _build_3d_bundles(
        end_L, end_R, seed_offset=seed_off)
    _draw_row(fig, [gs[row_idx, 0], gs[row_idx, 1], gs[row_idx, 2]],
              mean_L, mean_R, trajs_L, trajs_R, r_t, label)

fig.suptitle(
    "Figure 2 — Cosine on $S^{N-1}$ in two prototype-separation regimes",
    fontsize=12.5, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/02_cosine_as_sphere_projection.png")
print("Figure 2 saved.")
plt.close()
