"""Figures for the RNN+RL similarity-framework model.

Every figure is written as a PNG (raster preview, longest side <= 1600 px) and
an SVG (vector, for the manuscript / vault), per the repo plotting convention.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from si_network_model.config import BOUNDARY_DEG, ORIENTATIONS_DEG  # noqa: E402
from si_network_model.ideal_observer import StimulusIdealObserver  # noqa: E402
from si_network_model.stimuli import enumerate_conditions  # noqa: E402

from .cohort import CONDITIONS, AnimalResult, cohort_table  # noqa: E402

_FIXED_C = "#4C72B0"   # cool: fixed V1
_TRAINED_C = "#C44E52"  # warm: trained V1
_CEIL_C = "#555555"
_COND_C = {"fixed": _FIXED_C, "trained": _TRAINED_C}


def _save(fig, out_dir: Path, stem: str, max_px: int = 1600) -> None:
    """Write ``stem.svg`` (full vector) and ``stem.png`` (longest side <= max_px)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    w_in, h_in = fig.get_size_inches()
    dpi = min(200, max_px / max(w_in, h_in))
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 -- learning curves
# ---------------------------------------------------------------------------


def fig_learning_curves(results: list[AnimalResult], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for cond in CONDITIONS:
        accs = np.stack([_smooth(r.hist[cond]["acc"]) for r in results])
        rew = np.stack([_smooth(r.hist[cond]["reward"]) for r in results])
        x = np.arange(accs.shape[1])
        for ax, data, lab in ((axes[0], accs, "accuracy"), (axes[1], rew, "mean reward")):
            m, se = data.mean(0), data.std(0) / np.sqrt(len(results))
            ax.plot(x, m, color=_COND_C[cond], label=f"{cond} V1")
            ax.fill_between(x, m - se, m + se, color=_COND_C[cond], alpha=0.2)
            ax.set_xlabel("training batch")
            ax.set_ylabel(lab)
    axes[0].axhline(0.5, ls=":", color=_CEIL_C, lw=1)
    axes[0].set_title("Actor-critic learning curves")
    axes[1].set_title("Reward")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "fig1_learning_curves")


def _smooth(a, k: int = 5):
    a = np.asarray(a, float)
    if len(a) < k:
        return a
    return np.convolve(a, np.ones(k) / k, mode="same")


# ---------------------------------------------------------------------------
# Fig 2 -- the headline: template vs whitened alignment
# ---------------------------------------------------------------------------


def fig_template_vs_whitened(results: list[AnimalResult], out_dir: Path) -> None:
    tab = cohort_table(results)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        t = tab[cond]["cos_readout_template"]
        w = tab[cond]["cos_readout_whitened"]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t, 0.4, color=_COND_C[cond], label=r"vs template $\Delta\mu$")
        ax.bar(x + 0.2, w, 0.4, color="#bbbbbb",
               label=r"vs whitened $\Sigma^{-1}\Delta\mu$")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{cond} V1")
        ax.set_xlabel("animal")
        ax.set_xticks(x)
    axes[0].set_ylabel("cosine alignment of learned readout")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("RL readout aligns with the template direction, not the whitened optimum",
                 fontsize=11)
    _save(fig, out_dir, "fig2_template_vs_whitened")


# ---------------------------------------------------------------------------
# Fig 3 -- decision efficiency + SI vs optimal log-odds
# ---------------------------------------------------------------------------


def fig_efficiency(results: list[AnimalResult], out_dir: Path) -> None:
    tab = cohort_table(results)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    for j, cond in enumerate(CONDITIONS):
        net = tab[cond]["network_accuracy"]
        io = tab[cond]["neural_io_accuracy"]
        x = np.arange(len(net)) + (j - 0.5) * 0.0
        ax.scatter(io, net, color=_COND_C[cond], label=f"{cond} V1", zorder=3)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    ax.plot([lo, 1], [lo, 1], ls="--", color=_CEIL_C, lw=1, label="efficiency = 1")
    ax.set_xlabel("neural-IO accuracy")
    ax.set_ylabel("network accuracy")
    ax.set_title("Decision efficiency")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ex = results[0]
    for cond in CONDITIONS:
        ev = ex.eval[cond]
        si, lo_ = ev["mean_si"], ev["io_log_odds"]
        m = np.isfinite(si) & np.isfinite(lo_)
        ax.scatter(lo_[m], si[m], s=4, alpha=0.25, color=_COND_C[cond],
                   label=f"{cond} (r={ex.summary[cond]['si_vs_logodds_r']:.2f})")
    ax.set_xlabel("neural-IO log-posterior-odds")
    ax.set_ylabel("agent mean SI")
    ax.set_title(f"SI tracks optimal evidence (animal {ex.animal_id})")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "fig3_efficiency_and_si")


# ---------------------------------------------------------------------------
# Fig 4 -- psychometric curves vs the ideal-observer ceilings
# ---------------------------------------------------------------------------


def fig_psychometric(results: list[AnimalResult], base_cfg, out_dir: Path) -> None:
    oris = np.array(ORIENTATIONS_DEG, float)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for cond in CONDITIONS:
        pgo = _psychometric_by_orientation(results, cond, oris)
        m, se = pgo.mean(0), pgo.std(0) / np.sqrt(len(results))
        ax.errorbar(oris, m, yerr=se, marker="o", color=_COND_C[cond],
                    label=f"{cond} V1", capsize=2)
    # stimulus-IO ceiling (animal-independent)
    sio = StimulusIdealObserver(base_cfg.base)
    conds = enumerate_conditions()
    io_pgo = sio.psychometric(conds)
    io_by_ori = np.array([io_pgo[[c.orientation_deg == o for c in conds]].mean()
                          for o in oris])
    ax.plot(oris, io_by_ori, ls="--", color=_CEIL_C, label="stimulus-IO ceiling")
    ax.axvline(BOUNDARY_DEG, ls=":", color="k", lw=1)
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel("P(Go)")
    ax.set_title("Psychometric curves")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "fig4_psychometric")


def _psychometric_by_orientation(results, cond, oris):
    out = np.full((len(results), len(oris)), np.nan)
    for i, r in enumerate(results):
        ev = r.eval[cond]
        for j, o in enumerate(oris):
            m = ev["orientation"] == o
            if m.any():
                out[i, j] = ev["choice"][m].mean()
    return out


# ---------------------------------------------------------------------------
# Fig 5 -- fixed vs trained summary
# ---------------------------------------------------------------------------


def fig_fixed_vs_trained(results: list[AnimalResult], out_dir: Path) -> None:
    tab = cohort_table(results)
    metrics = [
        ("network_accuracy", "accuracy"),
        ("decision_efficiency", "efficiency"),
        ("cos_readout_template", r"cos$(\hat w,\Delta\mu)$"),
        ("mean_abs_si", "mean |SI|"),
        ("lapse_rate", "lapse rate"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    x = np.arange(len(metrics))
    for j, cond in enumerate(CONDITIONS):
        vals = np.array([np.nanmean(tab[cond][k]) for k, _ in metrics])
        ses = np.array([np.nanstd(tab[cond][k]) / np.sqrt(len(results))
                        for k, _ in metrics])
        ax.bar(x + (j - 0.5) * 0.4, vals, 0.4, yerr=ses, capsize=3,
               color=_COND_C[cond], label=f"{cond} V1")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("cohort mean")
    ax.set_title("Fixed vs trained-then-frozen V1")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "fig5_fixed_vs_trained")


def make_all_figures(results: list[AnimalResult], base_cfg, out_dir: Path) -> None:
    fig_learning_curves(results, out_dir)
    fig_template_vs_whitened(results, out_dir)
    fig_efficiency(results, out_dir)
    fig_psychometric(results, base_cfg, out_dir)
    fig_fixed_vs_trained(results, out_dir)
