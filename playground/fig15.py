"""
Figure 15 — Asymmetric strategy: predicted behavioural psychometric (Prediction 5).

If neural SI is asymmetric (one archetype weighted more than the other),
the behavioural psychometric P(Go) should be similarly asymmetric — the
animal calls more Gos than the symmetric reading of stimuli alone would
predict. Framework: asymmetric weighting at the readout stage produces
matched neural and behavioural asymmetry. Failure: behaviour is rescued
downstream — neural SI is asymmetric but the psychometric stays symmetric.

Layout: two rows (EASY top, REALISTIC bottom). Three columns: balanced
weighting / asymmetric (w_R = 0.7) / single-archetype. Each panel shows
both the resulting psychometric P(Go) and the mean neural SI by
signed_contrast on a twin axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import framework_common as fc
from framework_common import OUTDIR, RAMP, START, ou_traj


def _gen_trials_at_contrast(c, end_L, end_R, n=50, seed=0):
    """Generate n trial trajectories at signed_contrast c ∈ [-1, +1].
    c maps the trial mean to a weighted blend of the two archetype endpoints:
      c = +1  → R archetype
      c = -1  → L archetype
      c =  0  → midpoint (ambiguous)
    """
    rng = np.random.default_rng(seed)
    w_R = (c + 1) / 2.0  # weight on R
    end = w_R * end_R + (1.0 - w_R) * end_L
    mean = START + RAMP[:, None] * (end - START)
    # Higher |c| → tighter (clearer) trajectory; |c|≈0 → more variable
    sigma = 0.20 + 0.20 * (1 - abs(c))
    return np.stack([mean + ou_traj(fc.N, sigma=sigma, seed=seed + k)
                     for k in range(n)])


def _si_under_weights(trials, end_L, end_R, w_L, w_R):
    """Compute SI for each trial under given asymmetric weights."""
    mean_L = START + RAMP[:, None] * (end_L - START)
    mean_R = START + RAMP[:, None] * (end_R - START)
    si_list = []
    for tr in trials:
        cos_L = fc.cosine_sim(tr, mean_L)
        cos_R = fc.cosine_sim(tr, mean_R)
        sL = (1 + cos_L) / 2
        sR = (1 + cos_R) / 2
        si_t = (w_R * sR - w_L * sL) / (w_R * sR + w_L * sL + 1e-9)
        si_list.append(si_t.mean())  # trial-mean SI
    return np.array(si_list)


def _draw_panel(ax, end_L, end_R, w_L, w_R, panel_title):
    contrasts = np.linspace(-1.0, 1.0, 9)
    p_go = np.zeros(len(contrasts))
    mean_si = np.zeros(len(contrasts))
    for i, c in enumerate(contrasts):
        trials = _gen_trials_at_contrast(c, end_L, end_R, n=80,
                                         seed=int(1000 * (c + 2)))
        si_trials = _si_under_weights(trials, end_L, end_R, w_L, w_R)
        p_go[i] = float(np.mean(si_trials > 0))
        mean_si[i] = float(si_trials.mean())

    ax.plot(contrasts, p_go, "o-", color="black", lw=1.5, label="P(Go)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(Go)", fontsize=8)
    ax.axhline(0.5, color="gray", lw=0.4, ls=":", alpha=0.5)
    ax.axvline(0.0, color="gray", lw=0.4, ls=":", alpha=0.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)

    ax2 = ax.twinx()
    # Scale mean_si to its own max for visual comparability
    scale = max(abs(mean_si).max(), 1e-6)
    ax2.plot(contrasts, mean_si / scale, "s-", color="darkorange",
             lw=1.2, alpha=0.85, markersize=5, label="mean SI (rescaled)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("mean SI (scaled)", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.axhline(0, color="gray", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel("signed_contrast", fontsize=8)
    ax.set_title(panel_title, fontsize=9)


fig = plt.figure(figsize=(15, 8.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.45)

strategies = [
    (0.5, 0.5, "balanced  ($w_L = w_R$)"),
    (0.3, 0.7, "asymmetric  ($w_R = 0.7$)"),
    (0.0, 1.0, "single archetype  ($w_L = 0$)"),
]
regimes = [
    (fc.END_L_EASY, fc.END_R_EASY, "EASY"),
    (fc.END_L, fc.END_R, "REALISTIC"),
]
for row_idx, (end_L, end_R, reg_lbl) in enumerate(regimes):
    for col_idx, (w_L, w_R, strat_lbl) in enumerate(strategies):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        _draw_panel(ax, end_L, end_R, w_L, w_R,
                    f"{reg_lbl} · {strat_lbl}")

fig.suptitle("Figure 15 — Predicted psychometric under asymmetric strategy "
             "× prototype-separation regime\n"
             "(realistic regime → asymmetric weighting has larger psychometric impact)",
             fontsize=11, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/15_strategy_psychometric.png")
print("Figure 15 saved.")
plt.close()
