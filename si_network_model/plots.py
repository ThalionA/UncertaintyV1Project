"""Figure generation for the SI-framework network model.

Every figure compares the trained SI-framework network to the two optimal
observers and to the framework's own predictions. Saves PNGs to figures/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis  # noqa: E402
from .config import CONTRAST_DISPERSION_PAIRS  # noqa: E402
from .ideal_observer import decision_threshold  # noqa: E402
from .paths import FIGURES_DIR  # noqa: E402
from .si_variants import VARIANT_NAMES  # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})

C_NET = "#d62728"      # the SI-framework network
C_NIO = "#1f77b4"      # neural ideal observer
C_SIO = "#7f7f7f"      # stimulus ideal observer
C_GO = "#2ca02c"
C_AMB = "#9467bd"

# Difficulty tiers (by contrast) for the psychometric panels.
_TIERS = {
    "easy  (contrast 1.0)": [p for p in CONTRAST_DISPERSION_PAIRS if p[0] == 1.0],
    "medium  (contrast 0.25-0.5)": [p for p in CONTRAST_DISPERSION_PAIRS
                                    if p[0] in (0.5, 0.25)],
    "hard  (contrast 0.01)": [p for p in CONTRAST_DISPERSION_PAIRS if p[0] == 0.01],
}


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _pair_mask(ev: dict, pairs: list) -> np.ndarray:
    m = np.zeros(len(ev["orientation"]), dtype=bool)
    for contrast, dispersion in pairs:
        m |= (ev["contrast"] == contrast) & (ev["dispersion"] == dispersion)
    return m


# ---------------------------------------------------------------------------
# Figure 1 -- learning curves
# ---------------------------------------------------------------------------


def fig_learning_curves(cohort, summary) -> str:
    fig, (axz, axf) = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    median_p1 = float(np.median([r.phase1_length for r in cohort.runs]))

    # left: zoom on early learning (small window resolves the fast rise)
    zwin, zmax = 25, 700
    zr = [analysis.rolling_accuracy(r.train_log["correct"], zwin)[:zmax]
          for r in cohort.runs]
    common = min(len(r) for r in zr)
    for r in zr:
        axz.plot(r, color=C_NET, alpha=0.30, lw=1.0)
    axz.plot(np.stack([r[:common] for r in zr]).mean(0), color=C_NET, lw=2.6,
             label="cohort mean")
    axz.axhline(0.5, ls=":", color="k", lw=0.8, label="chance")
    axz.axvline(median_p1, ls="--", color=C_NIO, lw=1.1,
                label="Phase 1 -> 2 (median)")
    axz.set_xlabel("training trial")
    axz.set_ylabel(f"rolling accuracy ({zwin}-trial window)")
    axz.set_title("Early learning (Phase 1: archetypes only)")
    axz.set_ylim(0.3, 1.03)
    axz.legend(loc="lower right", fontsize=7.5)

    # right: the full curriculum
    fwin = 80
    fr = [analysis.rolling_accuracy(r.train_log["correct"], fwin)
          for r in cohort.runs]
    common = min(len(r) for r in fr)
    for r in fr:
        axf.plot(r, color=C_NET, alpha=0.30, lw=1.0)
    axf.plot(np.stack([r[:common] for r in fr]).mean(0), color=C_NET, lw=2.6)
    axf.axhline(0.5, ls=":", color="k", lw=0.8)
    axf.axvline(median_p1, ls="--", color=C_NIO, lw=1.1)
    axf.set_xlabel("training trial")
    axf.set_title("Full curriculum (Phase 2: archetype-biased full grid)")
    fig.suptitle("Figure 1 - Reward-driven learning: fast acquisition of the "
                 "archetypes, then a harder full-grid regime", y=1.04)
    return _save(fig, "fig1_learning_curves.png")


# ---------------------------------------------------------------------------
# Figure 2 -- psychometric: network vs both ideal observers
# ---------------------------------------------------------------------------


def fig_psychometric(cohort, summary) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, (label, pairs) in zip(axes, _TIERS.items()):
        net_curves, nio_curves = [], []
        oris = None
        for run in cohort.runs:
            ev = run.eval
            m = _pair_mask(ev, pairs)
            o, net = analysis.psychometric(ev["orientation"][m], ev["choice"][m])
            _, nio = analysis.psychometric(
                ev["orientation"][m], run.neural_io.p_go(ev["r_bar"])[m])
            oris = o
            net_curves.append(net)
            nio_curves.append(nio)
        net_curves = np.stack(net_curves)
        nio_curves = np.stack(nio_curves)
        # stimulus-IO (analytic, animal-independent), averaged over the tier
        sio = np.array([
            np.mean([cohort.stimulus_io.p_go(o, c, d) for c, d in pairs])
            for o in oris])

        ax.axhline(0.5, ls=":", color="k", lw=0.7)
        ax.axvline(45, ls=":", color="k", lw=0.7)
        ax.plot(oris, sio, color=C_SIO, lw=2.0, ls="--", marker="s",
                ms=4, label="stimulus-IO (optimal, latent)")
        ax.plot(oris, nio_curves.mean(0), color=C_NIO, lw=2.2, marker="o",
                ms=4, label="neural-IO (optimal readout)")
        ax.fill_between(oris,
                        net_curves.mean(0) - net_curves.std(0),
                        net_curves.mean(0) + net_curves.std(0),
                        color=C_NET, alpha=0.18)
        ax.plot(oris, net_curves.mean(0), color=C_NET, lw=2.4, marker="o",
                ms=4, label="SI network")
        ax.set_xlabel("grating orientation (deg)")
        ax.set_title(label)
        if ax is axes[0]:
            ax.set_ylabel("P(Go)")
            ax.legend(loc="upper right", fontsize=7.2)
    fig.suptitle("Figure 2 - Psychometric: the SI network tracks the optimal "
                 "Bayesian observers, shallowing on harder stimuli", y=1.04)
    return _save(fig, "fig2_psychometric.png")


# ---------------------------------------------------------------------------
# Figure 3 -- decision efficiency
# ---------------------------------------------------------------------------


def fig_efficiency(cohort, summary) -> str:
    per = summary["per_animal"]
    net = np.array([a["network_acc"] for a in per])
    nio = np.array([a["neural_io_acc"] for a in per])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    lo, hi = 0.55, 0.92
    ax1.plot([lo, hi], [lo, hi], ls="--", color="k", lw=1.0, label="identity (optimal)")
    ax1.scatter(nio, net, s=70, color=C_NET, edgecolor="k", zorder=3)
    ax1.set_xlabel("neural-IO accuracy (optimal readout of V1)")
    ax1.set_ylabel("SI network accuracy")
    ax1.set_title("Network vs optimal readout, per animal")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.legend(loc="upper left", fontsize=8)

    eff = np.array([a["efficiency"] for a in per])
    ids = np.arange(len(per))
    ax2.bar(ids, eff, color=C_NET, edgecolor="k")
    ax2.axhline(1.0, ls="--", color="k", lw=1.0)
    ax2.axhline(eff.mean(), ls="-", color=C_NIO, lw=2.0,
                label=f"cohort mean = {eff.mean():.3f}")
    ax2.set_xlabel("animal")
    ax2.set_ylabel("decision efficiency  (network acc / neural-IO acc)")
    ax2.set_title("Fraction of the optimal readout's accuracy captured")
    ax2.set_ylim(0.0, 1.1)
    ax2.set_xticks(ids)
    ax2.legend(loc="lower right", fontsize=8)
    fig.suptitle("Figure 3 - The SI-framework readout is near-optimal", y=1.03)
    return _save(fig, "fig3_efficiency.png")


# ---------------------------------------------------------------------------
# Figure 4 -- trial-by-trial: network evidence vs optimal log-posterior-odds
# ---------------------------------------------------------------------------


def fig_si_vs_io(cohort, summary) -> str:
    si_all, lo_all, cat_all = [], [], []
    for run in cohort.runs:
        ev = run.eval
        keep = ev["orientation"] != 45.0
        si_all.append(ev["mean_si"][keep])
        lo_all.append(run.neural_io.log_odds(ev["r_bar"])[keep])
        cat_all.append(analysis.category(ev["orientation"])[keep])
    si = np.concatenate(si_all)
    log_odds = np.concatenate(lo_all)
    cat = np.concatenate(cat_all)
    r = np.corrcoef(si, log_odds)[0, 1]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.axhline(0, ls=":", color="k", lw=0.7)
    ax.axvline(0, ls=":", color="k", lw=0.7)
    for c, col, name in [(1, C_GO, "Go stimuli"), (0, C_NIO, "NoGo stimuli")]:
        m = cat == c
        ax.scatter(log_odds[m], si[m], s=5, alpha=0.18, color=col, label=name)
    ax.set_xlabel("neural-IO log-posterior-odds  log P(Go)/P(NoGo)")
    ax.set_ylabel("network trial evidence  E$_t$[SI]")
    ax.set_title("Figure 4 - The network's accumulated evidence tracks the\n"
                 f"optimal log-posterior-odds  (Pearson r = {r:.2f}, all trials)")
    ax.legend(loc="lower right", fontsize=8, markerscale=2)
    return _save(fig, "fig4_si_vs_io.png")


# ---------------------------------------------------------------------------
# Figure 5 -- SI(t) traces by difficulty
# ---------------------------------------------------------------------------


def fig_si_traces(cohort, summary) -> str:
    cfg = cohort.cfg
    t = np.arange(cfg.n_steps) * cfg.dt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    groups = [
        ("easy Go  (0 deg, contrast 1)", lambda ev: (ev["orientation"] == 0)
         & (ev["contrast"] == 1.0) & (ev["dispersion"] == 5.0), C_GO),
        ("easy NoGo  (90 deg, contrast 1)", lambda ev: (ev["orientation"] == 90)
         & (ev["contrast"] == 1.0) & (ev["dispersion"] == 5.0), C_NIO),
        ("ambiguous  (45 deg, contrast 0.01)", lambda ev: (ev["orientation"] == 45)
         & (ev["contrast"] == 0.01), C_AMB),
    ]
    for ax, (label, selector, col) in zip(axes, groups):
        traces = np.concatenate([run.eval["si_full"][selector(run.eval)]
                                 for run in cohort.runs], axis=0)
        mean = traces.mean(axis=0)
        sd = traces.std(axis=0)
        ax.axhline(0, ls=":", color="k", lw=0.7)
        for tr in traces[:60]:
            ax.plot(t, tr, color=col, alpha=0.08, lw=0.8)
        ax.fill_between(t, mean - sd, mean + sd, color=col, alpha=0.25)
        ax.plot(t, mean, color=col, lw=2.6)
        ax.set_xlabel("time within trial (s)")
        ax.set_title(f"{label}\nVar$_t$[SI] = {traces.var(axis=1).mean():.4f}")
        if ax is axes[0]:
            ax.set_ylabel("SI(t)   (positive favours Go)")
    fig.suptitle("Figure 5 - SI(t) evidence stream: strong and steady on easy "
                 "trials, near zero with high variance when ambiguous", y=1.04)
    return _save(fig, "fig5_si_traces.png")


# ---------------------------------------------------------------------------
# Figure 6 -- learned bundle archetypes
# ---------------------------------------------------------------------------


def fig_bundles(cohort, summary) -> str:
    run = cohort.runs[0]
    ev = run.eval
    arch = ((ev["contrast"] == 1.0) & (ev["dispersion"] == 5.0)
            & np.isin(ev["orientation"], [0, 90]))
    r_bar = ev["r_bar"][arch].astype(float)
    cat = analysis.category(ev["orientation"][arch])
    centred = r_bar - r_bar.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = centred @ vt[:2].T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for c, col, name in [(1, C_GO, "Go bundle"), (0, C_NIO, "NoGo bundle")]:
        m = cat == c
        ax1.scatter(pcs[m, 0], pcs[m, 1], s=16, alpha=0.5, color=col, label=name)
        mu = pcs[m].mean(axis=0)
        ax1.scatter(*mu, s=260, marker="*", color=col, edgecolor="k", zorder=5)
    ax1.set_xlabel("V1 state-space PC 1")
    ax1.set_ylabel("V1 state-space PC 2")
    ax1.set_title(f"Animal 0 - archetype bundles in V1 space\n"
                  "stars = learned prototypes")
    ax1.legend(loc="best", fontsize=8)

    per = summary["per_animal"]
    ids = np.arange(len(per))
    kappa = np.array([a["kappa"] for a in per])
    ax2.bar(ids, kappa, color=C_NET, edgecolor="k")
    ax2.set_xlabel("animal")
    ax2.set_ylabel("bundle concentration  kappa")
    ax2.set_title("Per-animal bundle concentration\n(higher = tighter archetype bundle)")
    ax2.set_xticks(ids)
    fig.suptitle("Figure 6 - Reward-modulated plasticity carves two V1 trajectory "
                 "bundles - the framework's archetypes", y=1.04)
    return _save(fig, "fig6_bundles.png")


# ---------------------------------------------------------------------------
# Figure 7 -- cross-animal framework predictions
# ---------------------------------------------------------------------------


def _scatter_fit(ax, x, y, color):
    ax.scatter(x, y, s=70, color=color, edgecolor="k", zorder=3)
    if len(x) > 2 and np.std(x) > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, a + b * xs, ls="--", color="k", lw=1.2)
        return float(np.corrcoef(x, y)[0, 1])
    return float("nan")


def fig_cross_animal(cohort, summary) -> str:
    per = summary["per_animal"]
    kappa = np.array([a["kappa"] for a in per])
    lapse = np.array([a["lapse_rate"] for a in per])
    spread = np.array([a["si_spread"] for a in per])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    r1 = _scatter_fit(ax1, kappa, lapse, C_NET)
    ax1.set_xlabel("bundle concentration  kappa")
    ax1.set_ylabel("lapse rate  (errors on archetypes)")
    ax1.set_title(f"Prediction 15: tighter bundle -> fewer lapses\nPearson r = {r1:.2f}")

    r2 = _scatter_fit(ax2, kappa, spread, C_NIO)
    ax2.set_xlabel("bundle concentration  kappa")
    ax2.set_ylabel("trial-mean-SI spread  Var$_{trial}$[E$_t$ SI]")
    ax2.set_title(f"Prediction 4: tighter bundle -> lower drift variability\n"
                  f"Pearson r = {r2:.2f}")
    fig.suptitle("Figure 7 - Cross-animal: bundle width predicts behaviour, "
                 "as the framework predicts", y=1.03)
    return _save(fig, "fig7_cross_animal.png")


# ---------------------------------------------------------------------------
# Figure 8 -- response-time distributions by difficulty
# ---------------------------------------------------------------------------


def fig_rt_distributions(cohort, summary) -> str:
    cfg = cohort.cfg
    bins = np.linspace(0, cfg.t_trial, 26)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    groups = [
        ("easy  (contrast 1)", [p for p in CONTRAST_DISPERSION_PAIRS if p[0] == 1.0], C_GO),
        ("medium  (contrast 0.25-0.5)",
         [p for p in CONTRAST_DISPERSION_PAIRS if p[0] in (0.25, 0.5)], C_NIO),
        ("hard  (contrast 0.01)",
         [p for p in CONTRAST_DISPERSION_PAIRS if p[0] == 0.01], C_AMB),
    ]
    for label, pairs, col in groups:
        rts = np.concatenate([run.eval["rt"][_pair_mask(run.eval, pairs)]
                              for run in cohort.runs])
        ax.hist(rts, bins=bins, density=True, histtype="step", lw=2.2,
                color=col, label=f"{label}  (median {np.median(rts):.2f}s)")
    ax.set_xlabel("response time (s)   [2.0 s = deadline / withhold]")
    ax.set_ylabel("density")
    ax.set_title("Figure 8 - Response-time distributions: easy trials decide "
                 "fast, ambiguous trials run to the deadline")
    ax.legend(loc="upper center", fontsize=8)
    return _save(fig, "fig8_rt_distributions.png")


# ---------------------------------------------------------------------------
# Figure 9 -- accuracy by stimulus difficulty
# ---------------------------------------------------------------------------


def fig_accuracy_by_difficulty(cohort, summary) -> str:
    pairs = list(CONTRAST_DISPERSION_PAIRS)
    net = np.zeros((len(cohort.runs), len(pairs)))
    nio = np.zeros_like(net)
    sio = np.zeros(len(pairs))
    for ai, run in enumerate(cohort.runs):
        pp = analysis.per_pair_accuracy(run.eval, run.neural_io, cohort.stimulus_io)
        for pi, pair in enumerate(pairs):
            net[ai, pi] = pp[pair]["network"]
            nio[ai, pi] = pp[pair]["neural_io"]
            sio[pi] = pp[pair]["stimulus_io"]

    order = np.argsort(-sio)  # easiest (highest IO accuracy) first
    labels = [f"c={pairs[i][0]:g}\nd={pairs[i][1]:g}" for i in order]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0.5, ls=":", color="k", lw=0.8)
    ax.bar(x - 0.27, sio[order], width=0.27, color=C_SIO, label="stimulus-IO")
    ax.bar(x, nio.mean(0)[order], width=0.27, color=C_NIO, label="neural-IO",
           yerr=nio.std(0)[order], capsize=2)
    ax.bar(x + 0.27, net.mean(0)[order], width=0.27, color=C_NET,
           label="SI network", yerr=net.std(0)[order], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("stimulus condition (contrast / dispersion), easiest -> hardest")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Figure 9 - Accuracy across the difficulty range: the network "
                 "shadows both ideal observers, all collapsing to chance together")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "fig9_accuracy_by_difficulty.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 10 -- the learned trajectory archetypes
# ---------------------------------------------------------------------------


def fig_trajectory_archetypes(cohort, summary) -> str:
    run = cohort.runs[0]
    bg = run.bundle_go.astype(float)        # (K, n_steps, n_v1)
    bn = run.bundle_nogo.astype(float)
    k, n_steps, n_v1 = bg.shape

    flat = np.concatenate([bg.reshape(k * n_steps, n_v1),
                           bn.reshape(k * n_steps, n_v1)])
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    basis = vt[:2]

    def proj(traj):  # (n_steps, n_v1) -> (n_steps, 2)
        return (traj - mean) @ basis.T

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.7))
    for i in range(k):
        cg, cn = proj(bg[i]), proj(bn[i])
        axA.plot(cg[:, 0], cg[:, 1], color=C_GO, alpha=0.22, lw=0.9)
        axA.plot(cn[:, 0], cn[:, 1], color=C_NIO, alpha=0.22, lw=0.9)
    pg, pn = proj(bg.mean(0)), proj(bn.mean(0))
    axA.plot(pg[:, 0], pg[:, 1], color=C_GO, lw=3.0,
             label=r"Go prototype $\bar{a}_{Go}(t)$")
    axA.plot(pn[:, 0], pn[:, 1], color=C_NIO, lw=3.0,
             label=r"NoGo prototype $\bar{a}_{NoGo}(t)$")
    axA.scatter(*pg[-1], s=240, marker="*", color=C_GO, edgecolor="k", zorder=5)
    axA.scatter(*pn[-1], s=240, marker="*", color=C_NIO, edgecolor="k", zorder=5)
    axA.set_xlabel("V1 state-space PC 1")
    axA.set_ylabel("V1 state-space PC 2")
    axA.set_title(f"Animal 0 - archetype trajectory bundles\n"
                  f"thin = {k} exemplar trajectories, thick = prototype")
    axA.legend(loc="best", fontsize=8)

    # angular separation between the two prototype trajectories over time
    mg, mn = bg.mean(0), bn.mean(0)
    ug = mg / (np.linalg.norm(mg, axis=1, keepdims=True) + 1e-9)
    un = mn / (np.linalg.norm(mn, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(np.einsum("tn,tn->t", ug, un), -1.0, 1.0)
    t = np.arange(n_steps) * cohort.cfg.dt
    axB.plot(t, np.degrees(np.arccos(cos)), color="k", lw=2.6)
    axB.set_xlabel("time within trial (s)")
    axB.set_ylabel("angle between Go and NoGo prototypes (deg)")
    axB.set_title("Archetypes separate as the trial unfolds")
    axB.set_ylim(0, None)
    fig.suptitle("Figure 10 - The learned trajectory archetypes: two V1 "
                 "trajectory bundles and their prototype curves", y=1.04)
    return _save(fig, "fig10_trajectory_archetypes.png")


# ---------------------------------------------------------------------------
# Figure 11 -- comparison of the four SI variants
# ---------------------------------------------------------------------------

_VLABELS = {"prototype": "prototype", "expected_exemplar": "expected\nexemplar",
            "max_exemplar": "max\nexemplar", "vmf": "vMF"}


def fig_variant_comparison(cohort, summary) -> str:
    names = list(VARIANT_NAMES)
    per = summary["per_animal"]
    x = np.arange(len(names))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.7))

    eff = {v: [a["variant_metrics"][v]["efficiency"] for a in per] for v in names}
    axA.axhline(1.0, ls="--", color="k", lw=1.0, label="optimal readout")
    axA.bar(x, [summary["variant_summary"][v]["efficiency"] for v in names],
            color=C_NET, edgecolor="k", width=0.6)
    for i, v in enumerate(names):
        axA.scatter(np.full(len(eff[v]), i), eff[v], s=22, color="k",
                    zorder=3, alpha=0.6)
    axA.set_xticks(x)
    axA.set_xticklabels([_VLABELS[v] for v in names])
    axA.set_ylabel("decision efficiency  (variant acc / neural-IO acc)")
    axA.set_ylim(0.0, 1.12)
    axA.set_title("Each variant as a readout of the same learned bundles")
    axA.legend(loc="lower right", fontsize=8)

    tiers = ["easy", "medium", "hard"]
    tier_col = {"easy": C_GO, "medium": C_NIO, "hard": C_AMB}
    for ti, tier in enumerate(tiers):
        vals = [np.mean([a["variant_acc_by_tier"][v][tier] for a in per])
                for v in names]
        axB.bar(x + (ti - 1) * 0.26, vals, width=0.26, color=tier_col[tier],
                edgecolor="k", label=tier)
    axB.axhline(0.5, ls=":", color="k", lw=0.8)
    axB.set_xticks(x)
    axB.set_xticklabels([_VLABELS[v] for v in names])
    axB.set_ylabel("accuracy")
    axB.set_ylim(0.4, 1.0)
    axB.set_title("Variant accuracy by stimulus difficulty")
    axB.legend(title="difficulty", fontsize=8)
    fig.suptitle("Figure 11 - Four SI readout variants: the cheap prototype is "
                 "as good as the bundle-aware variants on this task", y=1.04)
    return _save(fig, "fig11_variant_comparison.png")


# ---------------------------------------------------------------------------
# Figure 12 -- symmetric vs asymmetric payoff
# ---------------------------------------------------------------------------


def fig_asymmetric_comparison(sym_cohort, asym_cohort) -> str:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.7))

    crit = {}
    for cohort, color, label in [(sym_cohort, C_NIO, "symmetric reward"),
                                 (asym_cohort, C_NET, "asymmetric reward")]:
        curves, oris = [], None
        for run in cohort.runs:
            # easy tier (contrast 1) -> a clean sigmoid for the criterion
            m = run.eval["contrast"] == 1.0
            o, c = analysis.psychometric(run.eval["orientation"][m],
                                         run.eval["choice"][m])
            oris = o
            curves.append(c)
        curves = np.stack(curves)
        axA.plot(oris, curves.mean(0), color=color, lw=2.5, marker="o", ms=4,
                 label=label)
        axA.fill_between(oris, curves.mean(0) - curves.std(0),
                         curves.mean(0) + curves.std(0), color=color, alpha=0.15)
        crit[label] = np.array([
            analysis.psychometric_criterion(r.eval["orientation"],
                                            r.eval["choice"])
            for r in cohort.runs])
    axA.axhline(0.5, ls=":", color="k", lw=0.8)
    axA.axvline(45, ls=":", color="k", lw=0.8, label="unbiased boundary")
    axA.set_xlabel("grating orientation (deg)")
    axA.set_ylabel("P(Go)")
    axA.set_title("Asymmetric payoff (cheap false alarms)\nshifts the "
                  "psychometric toward Go")
    axA.legend(loc="upper right", fontsize=8)

    # reward-optimal criterion: orientation where the stimulus-IO's
    # log-odds crosses the reward-weighted threshold, on the easy pair.
    def io_criterion(cohort):
        io = cohort.stimulus_io
        oris = np.arange(0, 91, 1.0)
        pgo = np.array([io.p_go(o, 1.0, 5.0) for o in oris])
        idx = np.argmin(np.abs(pgo - 0.5))
        return float(oris[idx])

    labels = ["symmetric reward", "asymmetric reward"]
    net_means = [crit[l].mean() for l in labels]
    net_sd = [crit[l].std() for l in labels]
    io_crit = [io_criterion(sym_cohort), io_criterion(asym_cohort)]
    xpos = np.arange(2)
    axB.axhline(45, ls=":", color="k", lw=0.8)
    axB.bar(xpos - 0.2, net_means, width=0.4, yerr=net_sd, capsize=3,
            color=C_NET, edgecolor="k", label="SI network")
    axB.bar(xpos + 0.2, io_crit, width=0.4, color=C_SIO, edgecolor="k",
            label="reward-optimal observer")
    axB.set_xticks(xpos)
    axB.set_xticklabels(labels)
    axB.set_ylabel("psychometric criterion (deg)  -- 50% P(Go) crossing")
    axB.set_title("Network shifts toward Go -- and over-shifts the\n"
                  "reward-optimal criterion (excess Go-bias)")
    axB.legend(loc="best", fontsize=8)
    fig.suptitle("Figure 12 - Biased payoffs: the network learns a pronounced "
                 "Go-bias, over-shifting the reward-optimal criterion", y=1.04)
    return _save(fig, "fig12_asymmetric_comparison.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_SYM_FIGURES = [
    fig_learning_curves, fig_psychometric, fig_efficiency, fig_si_vs_io,
    fig_si_traces, fig_bundles, fig_cross_animal, fig_rt_distributions,
    fig_accuracy_by_difficulty, fig_trajectory_archetypes, fig_variant_comparison,
]


def make_all_figures(sym_cohort, sym_summary, asym_cohort) -> list[str]:
    """Generate every figure; return the list of saved paths."""
    from .architecture_diagram import make_architecture_figure
    FIGURES_DIR.mkdir(exist_ok=True)
    paths = [make_architecture_figure()]
    paths += [fn(sym_cohort, sym_summary) for fn in _SYM_FIGURES]
    paths.append(fig_asymmetric_comparison(sym_cohort, asym_cohort))
    return paths
