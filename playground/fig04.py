"""
Figure 4 — DDM decomposition, in two prototype-separation regimes.

Each row shows the canonical DDM-parameter sweep (drift v / diffusion s /
drift variability s_v / RT distributions) — but with the v range chosen
to reflect the typical drift produced under each prototype-separation
regime. In the easy regime, SI magnitude is high → v is large → drift
dominates and trials hit bound quickly. In the realistic regime, SI
magnitude is small → v is small → diffusion and s_v dominate, RT
distributions widen.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from framework_common import OUTDIR, C_R, C_EASY, C_AMB

T_DDM_MAX = 2.5
DT_DDM = 0.005
T_DDM = np.arange(0, T_DDM_MAX, DT_DDM)
N_DDM = len(T_DDM)
BOUND = 1.0


def _accum(v, s, n_trials=12, seed=0):
    rng = np.random.default_rng(seed)
    inc = v * DT_DDM + s * np.sqrt(DT_DDM) * rng.standard_normal((n_trials, N_DDM))
    X = np.cumsum(inc, axis=1)
    for k in range(n_trials):
        cross = np.where(np.abs(X[k]) >= BOUND)[0]
        if len(cross):
            X[k, cross[0] + 1:] = X[k, cross[0]]
    return X


def _rts(v, s, sv, n=4000, T_max=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n_steps = int(T_max / DT_DDM)
    rts = []
    for k in range(n):
        v_k = v + sv * rng.standard_normal() if sv > 0 else v
        inc = v_k * DT_DDM + s * np.sqrt(DT_DDM) * rng.standard_normal(n_steps)
        X = np.cumsum(inc)
        cr = np.where(np.abs(X) >= BOUND)[0]
        if len(cr):
            rts.append(cr[0] * DT_DDM)
    return np.array(rts)


def _draw_row(fig, gs_row, v_scale, regime_label):
    """v_scale: multiplier for the canonical drift values. Easy = 1.0;
    realistic = ~0.35."""
    # Panel A: drift sweep
    axA = fig.add_subplot(gs_row[0])
    drifts = [d * v_scale for d in (0.4, 1.5, 3.0)]
    colors_v = ["#cdd7e0", "#7fa5c5", C_R]
    for v, color in zip(drifts, colors_v):
        X = _accum(v=v, s=0.6, n_trials=12, seed=int(v * 1000) + 1)
        for k in range(12):
            axA.plot(T_DDM, X[k], color=color, alpha=0.5, lw=0.7)
        axA.plot([], [], color=color, lw=2.0, label=f"v={v:.2f}")
    axA.axhline(BOUND, color="k", lw=1.2)
    axA.axhline(-BOUND, color="k", lw=1.2)
    axA.set_xlim(0, T_DDM_MAX)
    axA.set_ylim(-1.2, 1.2)
    axA.set_xlabel("t (s)", fontsize=8)
    axA.set_ylabel("X(t)", fontsize=8)
    axA.set_title(f"A · drift ({regime_label})", fontsize=9)
    axA.legend(fontsize=7)
    axA.tick_params(labelsize=7)

    # Panel B: diffusion sweep
    axB = fig.add_subplot(gs_row[1])
    diffs = [0.3, 0.9, 1.6]
    colors_s = ["#bce5cf", "#5fc197", C_EASY]
    for s, color in zip(diffs, colors_s):
        X = _accum(v=1.5 * v_scale, s=s, n_trials=12, seed=int(s * 1000) + 2)
        for k in range(12):
            axB.plot(T_DDM, X[k], color=color, alpha=0.5, lw=0.7)
        axB.plot([], [], color=color, lw=2.0, label=f"s={s:.1f}")
    axB.axhline(BOUND, color="k", lw=1.2)
    axB.axhline(-BOUND, color="k", lw=1.2)
    axB.set_xlim(0, T_DDM_MAX)
    axB.set_ylim(-1.2, 1.2)
    axB.set_xlabel("t (s)", fontsize=8)
    axB.set_ylabel("X(t)", fontsize=8)
    axB.set_title(f"B · diffusion ({regime_label})", fontsize=9)
    axB.legend(fontsize=7)
    axB.tick_params(labelsize=7)

    # Panel C: drift variability sweep
    axC = fig.add_subplot(gs_row[2])
    rng = np.random.default_rng(0)
    levels = [(0.0, "tight  $s_v$=0", "#cdd7e0"),
              (0.6, "moderate  $s_v$=0.6", "#7fa5c5"),
              (1.5, "wide  $s_v$=1.5", C_R)]
    for s_v, lbl, color in levels:
        n_paths = 16
        v_k = 1.2 * v_scale + s_v * rng.standard_normal(n_paths)
        for k in range(n_paths):
            X = _accum(v=v_k[k], s=0.5, n_trials=1,
                       seed=int(s_v * 100) + 999 + k)
            axC.plot(T_DDM, X[0], color=color, alpha=0.5, lw=0.7)
        axC.plot([], [], color=color, lw=2.0, label=lbl)
    axC.axhline(BOUND, color="k", lw=1.2)
    axC.axhline(-BOUND, color="k", lw=1.2)
    axC.set_xlim(0, T_DDM_MAX)
    axC.set_ylim(-1.2, 1.2)
    axC.set_xlabel("t (s)", fontsize=8)
    axC.set_ylabel("X(t)", fontsize=8)
    axC.set_title(f"C · drift variability ({regime_label})", fontsize=9)
    axC.legend(fontsize=7, loc="lower right")
    axC.tick_params(labelsize=7)

    # Panel D: RT distributions easy vs ambig under this regime
    axD = fig.add_subplot(gs_row[3])
    rt_easy = _rts(v=3.0 * v_scale, s=1.0, sv=0.3, seed=11)
    rt_amb = _rts(v=0.6 * v_scale, s=1.2, sv=0.8, seed=22)
    bins = np.linspace(0, 2.5, 40)
    axD.hist(rt_easy, bins=bins, density=True, color=C_EASY, alpha=0.6,
             label="easy", edgecolor="white", lw=0.4)
    axD.hist(rt_amb, bins=bins, density=True, color=C_AMB, alpha=0.6,
             label="ambig", edgecolor="white", lw=0.4)
    axD.set_xlim(0, 2.5)
    axD.set_xlabel("RT (s)", fontsize=8)
    axD.set_ylabel("density", fontsize=8)
    axD.set_title(f"D · RT distribution ({regime_label})", fontsize=9)
    axD.legend(fontsize=7)
    axD.tick_params(labelsize=7)


fig = plt.figure(figsize=(16, 8.5))
gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.30)

# EASY regime: drifts unchanged (high v)
_draw_row(fig, [gs[0, j] for j in range(4)],
          v_scale=1.0, regime_label="EASY")
# REALISTIC regime: drifts scaled down (small angular separation → small v)
_draw_row(fig, [gs[1, j] for j in range(4)],
          v_scale=0.35, regime_label="REALISTIC")

fig.suptitle("Figure 4 — DDM decomposition in two prototype-separation regimes\n"
             "(realistic regime → smaller drift → s and $s_v$ have larger relative impact)",
             fontsize=11, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/04_ddm_decomposition.png")
print("Figure 4 saved.")
plt.close()
