"""Per-session archetype-similarity analysis.

Extends the similarity framework by running the *entire* analysis — archetype
extraction and per-trial SI comparison — on each session of each mouse
*separately*, then combining results up a three-level hierarchy:

    trial   -> session   (inside each per-session metric)
    session -> mouse     (trial-count weighted; see ``session_aggregation``)
    mouse   -> group     (trial-count weighted; weighted SEM, N = mice)

Why per session: the easy-trial archetypes are not guaranteed to be identical
from session to session. Extracting them per session keeps every trial's SI
referenced to archetypes from the *same* recording, and lets us measure how
stable the archetypes actually are across a mouse's sessions (see
``render_archetype_stability*`` and ``render_archetype_drift``).

Aggregation is **trial-count weighted** at both levels — sessions / mice with
more trials count proportionally more. The across-mice error bar is a weighted
SEM via Kish's effective sample size (``session_aggregation.weighted_mean_sem``).

Every renderer produces a per-mouse figure (the existing grid view) and, where
the user asked for it, a dedicated across-mice grand-average figure.

Run standalone::

    python session_similarity_analysis.py [--data PATH] [--out PATH] [--smoke]

Outputs land under ``--out / F_per_session_similarity/`` with a
``_meta_session.json`` summary.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, norm

from dim_reduction_explore import (
    MouseData,
    load_all_mice,
    _filter_mice,
    split_by_session,
    GROUPINGS_DISCRETE,
    DATA_PATH,
    compute_features,
    _grid_dims,
    _level_order,
    _palette,
    _union_levels,
    _finalize_grid,
    _save_fig,
)
from similarity_analysis import (
    _compute_archetypes,
    _compute_archetypes_exemplars,
    _archetype_similarity,
    _si_per_method,
    _per_animal_kappa,
    _per_animal_si_spread,
    _per_animal_lapse,
    _per_animal_velocity_drift,
    _per_animal_si_logistic,
    _per_animal_si_logistic_xG_binned,
    _fit_interrogation_ddm,
    _decision_entropy,
    SI_METHODS,
    SI_METHOD_LABELS,
)
import session_aggregation as agg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Archetypes are extracted from the first EASY_BLOCK_SIZE easy trials of a
# session (contrast = 1.0, dispersion = 5.0, cardinal stimulus). A *fixed
# absolute* easy definition — not per-session quantiles — so the archetype
# definition does not drift with each session's stimulus mix.
EASY_BLOCK_SIZE = 50

# Sessions with fewer than this many trials are skipped for the trial-hungry
# behavioural fits (DDM / logistic). The fit functions also self-guard.
MIN_SESSION_TRIALS = 20

XG_TARGET = 1.0  # fixed xG slice for the single-trial scatter

METHOD_COLORS = {
    "prototype": "#1f77b4",
    "expected_exemplar": "#2ca02c",
    "max_exemplar": "#ff7f0e",
    "vmf": "#d62728",
}
SIDE_COLORS = {"Go": "#2ca02c", "NoGo": "#d62728"}

__all__ = [
    "EASY_BLOCK_SIZE",
    "MIN_SESSION_TRIALS",
    "_archetype_stability",
    "build_session_cache",
    "render_archetype_stability",
    "render_archetype_stability_across_mice",
    "render_archetype_drift",
    "render_per_session_si_over_xG",
    "render_across_mice_si_over_xG",
    "render_per_session_kappa_vs_si_spread",
    "render_across_mice_kappa_vs_si_spread",
    "render_per_session_psychometric",
    "render_across_mice_psychometric",
    "render_per_session_kappa_vs_lapse",
    "render_per_session_velocity_accumulator",
    "render_per_session_ddm",
    "render_per_session_kappa_vs_ddm_slope",
    "render_per_session_si_choice_logistic",
    "render_per_session_si_choice_logistic_xG_binned",
    "render_per_session_variants_compare",
    "render_per_session_single_trial_scatter",
    "render_per_session_si_vs_confidence",
    "run_session_similarity_sweep",
]


# ---------------------------------------------------------------------------
# Archetype stability — pairwise cosine between session archetypes
# ---------------------------------------------------------------------------


def _archetype_stability(session_archetypes: list[dict[str, np.ndarray]]) -> dict:
    """Computes pairwise similarity between session archetypes.

    Returns scalar metrics (mean pairwise cosine) and per-xG traces.
    Cosine similarity is computed on flattened (n_neurons x n_xG) arrays,
    and also per xG bin.
    """
    n_sess = len(session_archetypes)
    res = {
        "Go": {"matrix": np.full((n_sess, n_sess), np.nan), "mean": np.nan, "xg_trace": None},
        "NoGo": {"matrix": np.full((n_sess, n_sess), np.nan), "mean": np.nan, "xg_trace": None},
    }

    for side in ["Go", "NoGo"]:
        valid_idx = [i for i, a in enumerate(session_archetypes) if side in a]
        if len(valid_idx) < 2:
            continue

        n_xG = session_archetypes[valid_idx[0]][side].shape[1]
        xg_sims = []
        flat_sims = []

        for i in range(len(valid_idx)):
            for j in range(i + 1, len(valid_idx)):
                a1 = session_archetypes[valid_idx[i]][side]
                a2 = session_archetypes[valid_idx[j]][side]

                # Flattened cosine sim
                c_flat = 1.0 - cosine_dist(a1.flatten(), a2.flatten())
                res[side]["matrix"][valid_idx[i], valid_idx[j]] = c_flat
                res[side]["matrix"][valid_idx[j], valid_idx[i]] = c_flat
                flat_sims.append(c_flat)

                # Per-xG cosine sim
                c_xg = np.zeros(n_xG)
                for x in range(n_xG):
                    norm1, norm2 = np.linalg.norm(a1[:, x]), np.linalg.norm(a2[:, x])
                    if norm1 > 0 and norm2 > 0:
                        c_xg[x] = np.dot(a1[:, x], a2[:, x]) / (norm1 * norm2)
                xg_sims.append(c_xg)

        res[side]["mean"] = float(np.mean(flat_sims))
        if xg_sims:
            res[side]["xg_trace"] = np.mean(xg_sims, axis=0)
            res[side]["xg_trace_sem"] = np.std(xg_sims, axis=0) / np.sqrt(len(xg_sims))

    return res


# ---------------------------------------------------------------------------
# Per-session cache — split each mouse, extract archetypes / SI once
# ---------------------------------------------------------------------------


def build_session_cache(mice: list[MouseData]) -> dict:
    """Split every mouse into sessions and pre-compute the per-session pieces.

    For each session we store: the per-session ``MouseData``, the prototype
    and exemplar archetypes (fixed easy-block definition), the feature dict,
    the SI tensor under every method, the bundle kappa per side, an
    ``usable`` flag (both Go and NoGo archetypes present), and the trial
    count. The cache is built once and shared by every renderer so the
    expensive archetype / SI work is not repeated.

    Returns ``{mouse_tag: [session_entry, ...]}`` in session order.
    """
    cache: dict[str, list[dict]] = {}
    for m in mice:
        entries: list[dict] = []
        for s in split_by_session(m):
            proto = _compute_archetypes(s, easy_block_size=EASY_BLOCK_SIZE)
            ex = _compute_archetypes_exemplars(s, easy_block_size=EASY_BLOCK_SIZE)
            usable = agg.session_usable(proto) and agg.session_usable(ex)
            entry = {
                "session": s,
                "tag": s.tag,
                "proto": proto,
                "ex": ex,
                "feats": compute_features(s),
                "usable": usable,
                "n_trials": int(s.n_trials),
                "si": {},
                "kappa": {},
            }
            if usable:
                for method in SI_METHODS:
                    try:
                        entry["si"][method] = _si_per_method(s, proto, ex, method=method)
                    except Exception:  # pragma: no cover - defensive
                        entry["si"][method] = np.full((s.n_trials, s.n_xG), np.nan)
                entry["kappa"] = _per_animal_kappa(s, ex)
            entries.append(entry)
        cache[m.tag] = entries
    return cache


def _usable(entries: list[dict]) -> list[dict]:
    """Subset of session entries that carry both archetypes."""
    return [e for e in entries if e["usable"]]


def _kappa_avg(entry: dict) -> float:
    """Sides-averaged bundle kappa for a usable session, NaN if unavailable."""
    kap = entry.get("kappa") or {}
    if "Go" in kap and "NoGo" in kap:
        return 0.5 * (float(kap["Go"]) + float(kap["NoGo"]))
    return float("nan")


def _stack_traces(xgs: list[np.ndarray], traces: list[np.ndarray]):
    """Stack per-item xG traces onto a common grid.

    If every xG grid is identical the traces are stacked directly; otherwise
    each trace is linearly interpolated (over its finite points) onto the
    densest grid. Returns ``(xG_ref, stacked)`` with ``stacked`` shape
    ``(n_items, n_ref)``.
    """
    if not traces:
        return np.array([]), np.empty((0, 0))
    same_len = len({len(x) for x in xgs}) == 1
    if same_len and all(np.allclose(x, xgs[0]) for x in xgs):
        return np.asarray(xgs[0], dtype=float), np.vstack(traces)
    ref = xgs[int(np.argmax([len(x) for x in xgs]))]
    out = []
    for x, t in zip(xgs, traces):
        finite = np.isfinite(t)
        if finite.sum() < 2:
            out.append(np.full(len(ref), np.nan))
        else:
            out.append(np.interp(ref, np.asarray(x)[finite], np.asarray(t)[finite],
                                 left=np.nan, right=np.nan))
    return np.asarray(ref, dtype=float), np.vstack(out)


def _mouse_color(i: int):
    """Stable per-mouse colour from a 20-way categorical map."""
    cmap = plt.get_cmap("tab20")
    return cmap(i % 20)


# ---------------------------------------------------------------------------
# Archetype stability renderers
# ---------------------------------------------------------------------------


def render_archetype_stability(mice: list[MouseData], cache: dict,
                               out_dir: Path) -> Path | None:
    """Per-mouse archetype-stability across sessions (xG traces + summary bar)."""
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Archetype stability across sessions (first {EASY_BLOCK_SIZE} easy trials)\n"
        "per-mouse: pairwise cosine between session archetypes", fontsize=11)

    summary = []  # (tag, go_mean, nogo_mean)
    for ax, m in zip(axes.ravel(), mice):
        archs = [e["proto"] for e in cache[m.tag]]
        if sum(1 for a in archs if agg.session_usable(a)) < 2:
            ax.set_axis_off()
            continue
        stab = _archetype_stability(archs)
        go_mean, nogo_mean = stab["Go"]["mean"], stab["NoGo"]["mean"]
        if np.isfinite(go_mean) and np.isfinite(nogo_mean):
            summary.append((m.tag, go_mean, nogo_mean))
        for side in ("Go", "NoGo"):
            tr = stab[side]["xg_trace"]
            if tr is None:
                continue
            ax.plot(m.xG, tr, color=SIDE_COLORS[side], lw=2,
                    label=f"{side} (avg r={stab[side]['mean']:.2f})")
            sem = stab[side].get("xg_trace_sem")
            if sem is not None:
                ax.fill_between(m.xG, tr - sem, tr + sem,
                                color=SIDE_COLORS[side], alpha=0.2)
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("xG (a.u.)", fontsize=8)
        ax.set_ylabel("pairwise cosine similarity", fontsize=8)
        n_sess = len(cache[m.tag])
        ax.set_title(f"{m.tag} ({n_sess} sessions)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_dir / "archetype_stability_curves"
    _save_fig(fig, out)

    if summary:
        fig_b, ax_b = plt.subplots(1, 1, figsize=(6 + len(summary) * 0.5, 5))
        fig_b.suptitle(
            f"Mean archetype stability across sessions "
            f"(first {EASY_BLOCK_SIZE} easy trials)", fontsize=11)
        tags = [d[0] for d in summary]
        x = np.arange(len(tags))
        w = 0.35
        ax_b.bar(x - w / 2, [d[1] for d in summary], w, label="Go archetype",
                 color=SIDE_COLORS["Go"], alpha=0.8)
        ax_b.bar(x + w / 2, [d[2] for d in summary], w, label="NoGo archetype",
                 color=SIDE_COLORS["NoGo"], alpha=0.8)
        ax_b.set_ylabel("mean pairwise cosine similarity")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(tags, rotation=30, ha="right")
        ax_b.set_ylim(0, 1.05)
        ax_b.legend()
        fig_b.tight_layout(rect=(0, 0, 1, 0.95))
        _save_fig(fig_b, out_dir / "archetype_stability_summary")
    return out


def render_archetype_stability_across_mice(mice: list[MouseData], cache: dict,
                                           out_dir: Path) -> Path | None:
    """Grand-average archetype stability across mice (trial-weighted).

    Each mouse contributes one stability curve and one scalar per side; mice
    are combined trial-weighted with a weighted SEM (N = mice).
    """
    per_mouse_scalar = {"Go": [], "NoGo": []}
    per_mouse_trace = {"Go": {"xgs": [], "traces": []},
                       "NoGo": {"xgs": [], "traces": []}}
    weights = {"Go": [], "NoGo": []}
    for m in mice:
        archs = [e["proto"] for e in cache[m.tag]]
        if sum(1 for a in archs if agg.session_usable(a)) < 2:
            continue
        stab = _archetype_stability(archs)
        mouse_trials = float(sum(e["n_trials"] for e in _usable(cache[m.tag])))
        for side in ("Go", "NoGo"):
            if stab[side]["xg_trace"] is None or not np.isfinite(stab[side]["mean"]):
                continue
            per_mouse_scalar[side].append(stab[side]["mean"])
            per_mouse_trace[side]["xgs"].append(m.xG)
            per_mouse_trace[side]["traces"].append(stab[side]["xg_trace"])
            weights[side].append(mouse_trials)
    if not weights["Go"] and not weights["NoGo"]:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    fig.suptitle("Archetype stability — across-mice grand average "
                 "(trial-weighted, error = weighted SEM across mice)", fontsize=11)
    ax_tr, ax_bar = axes[0, 0], axes[0, 1]

    bar_vals, bar_sem, bar_labels, bar_colors = [], [], [], []
    for side in ("Go", "NoGo"):
        if not weights[side]:
            continue
        xg_ref, stacked = _stack_traces(per_mouse_trace[side]["xgs"],
                                        per_mouse_trace[side]["traces"])
        gmean, gsem, _ = agg.mouse_to_group(stacked, np.array(weights[side]))
        ax_tr.plot(xg_ref, gmean, color=SIDE_COLORS[side], lw=2.4, label=side)
        band = np.where(np.isfinite(gsem), gsem, 0.0)
        ax_tr.fill_between(xg_ref, gmean - band, gmean + band,
                           color=SIDE_COLORS[side], alpha=0.22)
        smean, ssem, _ = agg.mouse_to_group(
            np.array(per_mouse_scalar[side]), np.array(weights[side]))
        bar_vals.append(float(smean))
        bar_sem.append(float(ssem) if np.isfinite(ssem) else 0.0)
        bar_labels.append(f"{side}\n(N={len(weights[side])} mice)")
        bar_colors.append(SIDE_COLORS[side])

    ax_tr.set_ylim(-0.2, 1.05)
    ax_tr.axhline(0, color="gray", lw=0.5, ls=":")
    ax_tr.set_xlabel("xG (a.u.)", fontsize=9)
    ax_tr.set_ylabel("grand-mean pairwise cosine similarity", fontsize=9)
    ax_tr.set_title("stability vs xG", fontsize=10)
    ax_tr.legend(fontsize=8)

    x = np.arange(len(bar_vals))
    ax_bar.bar(x, bar_vals, 0.6, yerr=bar_sem, color=bar_colors, alpha=0.85,
               capsize=5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("grand-mean pairwise cosine similarity", fontsize=9)
    ax_bar.set_title("xG-averaged stability", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out_dir / "archetype_stability_across_mice"
    _save_fig(fig, out)
    return out


def render_archetype_drift(mice: list[MouseData], cache: dict,
                           out_dir: Path) -> Path | None:
    """Archetype drift: stability as a function of session lag.

    For each mouse the pairwise-cosine matrix is collapsed by session lag
    (difference in session rank). If archetypes drift monotonically over
    days, stability should fall with lag. Per-mouse traces plus a
    trial-weighted across-mice grand average.

    Note: lag is the difference in *session rank* (recording order), not
    calendar days — the export's ``session`` field is an ordinal index.
    """
    per_mouse = {"Go": [], "NoGo": []}   # (tag, lags, mean, weight, mouse_idx)
    for mi, m in enumerate(mice):
        archs = [e["proto"] for e in cache[m.tag]]
        if sum(1 for a in archs if agg.session_usable(a)) < 2:
            continue
        stab = _archetype_stability(archs)
        mouse_trials = float(sum(e["n_trials"] for e in _usable(cache[m.tag])))
        for side in ("Go", "NoGo"):
            lags, mean, _sem, n_pairs = agg.stability_vs_lag(stab[side]["matrix"])
            if np.any(n_pairs > 0):
                per_mouse[side].append((m.tag, lags, mean, mouse_trials, mi))
    if not per_mouse["Go"] and not per_mouse["NoGo"]:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
    fig.suptitle("Archetype drift — stability vs session lag "
                 "(thin = mouse, thick = trial-weighted across-mice mean)",
                 fontsize=11)
    for ax, side in zip(axes[0], ("Go", "NoGo")):
        rows = per_mouse[side]
        if not rows:
            ax.set_axis_off()
            continue
        max_lag = max(int(r[1][-1]) for r in rows if len(r[1]))
        lag_axis = np.arange(1, max_lag + 1)
        mouse_curves, mouse_w = [], []
        for (tag, lags, mean, w, mi) in rows:
            curve = np.full(max_lag, np.nan)
            for lg, mu in zip(lags, mean):
                curve[int(lg) - 1] = mu
            ax.plot(lag_axis, curve, color=_mouse_color(mi), lw=0.9,
                    alpha=0.55, marker="o", markersize=3)
            mouse_curves.append(curve)
            mouse_w.append(w)
        gmean, gsem, _ = agg.mouse_to_group(np.vstack(mouse_curves),
                                            np.array(mouse_w))
        band = np.where(np.isfinite(gsem), gsem, 0.0)
        ax.plot(lag_axis, gmean, color=SIDE_COLORS[side], lw=2.6,
                marker="o", markersize=5, label="across-mice mean")
        ax.fill_between(lag_axis, gmean - band, gmean + band,
                        color=SIDE_COLORS[side], alpha=0.2)
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("session lag (Δ session rank)", fontsize=9)
        ax.set_ylabel("pairwise cosine similarity", fontsize=9)
        ax.set_title(f"{side} archetype  (N={len(rows)} mice)", fontsize=10)
        ax.set_xticks(lag_axis)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out_dir / "archetype_drift"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# SI over xG — per-session grid + across-mice grand average
# ---------------------------------------------------------------------------


def _mouse_level_traces(entries: list[dict], grouping: str, method: str,
                        levels: list):
    """Trial-weighted per-(grouping level) SI-vs-xG trace for one mouse.

    Returns ``{level: (trace, total_trials)}`` for levels with any data.
    ``trace`` is the trial-weighted mean over the mouse's sessions of each
    session's mean SI trace for that level.
    """
    bucket = {lv: {"traces": [], "weights": []} for lv in levels}
    for e in _usable(entries):
        si = e["si"][method]
        c = e["feats"][grouping]
        for lv in levels:
            mask = (c == lv)
            n = int(np.sum(mask))
            if n > 0:
                bucket[lv]["traces"].append(np.nanmean(si[mask], axis=0))
                bucket[lv]["weights"].append(float(n))
    out = {}
    for lv in levels:
        tr = bucket[lv]["traces"]
        if tr:
            w = np.array(bucket[lv]["weights"])
            out[lv] = (agg.session_to_mouse(np.vstack(tr), w), float(w.sum()))
    return out


def render_per_session_si_over_xG(mice: list[MouseData], cache: dict,
                                  grouping: str, out_dir: Path,
                                  *, method: str = "expected_exemplar"
                                  ) -> Path | None:
    """Per-mouse SI-vs-xG: thin = session means, thick = trial-weighted mouse mean."""
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Per-session SI over xG ({SI_METHOD_LABELS[method]}) | group: {grouping}\n"
        "thin = session, thick = trial-weighted mouse mean", fontsize=11)
    drew = False
    for ax, m in zip(axes.ravel(), mice):
        entries = _usable(cache[m.tag])
        levels = _level_order(grouping, compute_features(m)[grouping])
        pal = _palette(levels, grouping)
        if not entries or not levels:
            ax.set_axis_off()
            continue
        # thin per-session per-level traces
        for e in entries:
            si = e["si"][method]
            c = e["feats"][grouping]
            for lv in levels:
                mask = (c == lv)
                if np.sum(mask) > 0:
                    ax.plot(e["session"].xG, np.nanmean(si[mask], axis=0),
                            color=pal[lv], alpha=0.3, lw=0.8)
        # thick trial-weighted mouse mean
        mouse_traces = _mouse_level_traces(entries, grouping, method, levels)
        if not mouse_traces:
            ax.set_axis_off()
            continue
        drew = True
        for lv, (trace, _w) in mouse_traces.items():
            ax.plot(m.xG, trace, color=pal[lv], alpha=1.0, lw=2.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.autoscale(axis="y", tight=True)
        ax.set_xlabel("xG (a.u.)", fontsize=7)
        ax.set_ylabel("SI", fontsize=7)
        ax.set_title(m.tag, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    if not drew:
        plt.close(fig)
        return None
    union = _union_levels(mice, grouping)
    _finalize_grid(fig, cont=False, levels=union,
                   palette=_palette(union, grouping), label=grouping)
    out = out_dir / f"per_session_si__{method}__{grouping}"
    _save_fig(fig, out)
    return out


def render_across_mice_si_over_xG(mice: list[MouseData], cache: dict,
                                  grouping: str, out_dir: Path,
                                  *, method: str = "expected_exemplar"
                                  ) -> Path | None:
    """Across-mice grand-average SI-vs-xG per grouping level (trial-weighted)."""
    union = _union_levels(mice, grouping)
    if not union:
        return None
    pal = _palette(union, grouping)
    # collect per-mouse traces per level
    per_level = {lv: {"xgs": [], "traces": [], "weights": []} for lv in union}
    for m in mice:
        entries = _usable(cache[m.tag])
        if not entries:
            continue
        levels = _level_order(grouping, compute_features(m)[grouping])
        mouse_traces = _mouse_level_traces(entries, grouping, method, levels)
        for lv, (trace, w) in mouse_traces.items():
            if lv in per_level:
                per_level[lv]["xgs"].append(m.xG)
                per_level[lv]["traces"].append(trace)
                per_level[lv]["weights"].append(w)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.4))
    n_mice_used = 0
    drew = False
    for lv in union:
        d = per_level[lv]
        if not d["traces"]:
            continue
        n_mice_used = max(n_mice_used, len(d["traces"]))
        xg_ref, stacked = _stack_traces(d["xgs"], d["traces"])
        gmean, gsem, _ = agg.mouse_to_group(stacked, np.array(d["weights"]))
        ax.plot(xg_ref, gmean, color=pal[lv], lw=2.4, label=f"{lv}")
        band = np.where(np.isfinite(gsem), gsem, 0.0)
        ax.fill_between(xg_ref, gmean - band, gmean + band,
                        color=pal[lv], alpha=0.18)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("xG (a.u.)", fontsize=10)
    ax.set_ylabel("grand-mean SI", fontsize=10)
    ax.set_title(
        f"Across-mice SI over xG ({SI_METHOD_LABELS[method]}) | group: {grouping}\n"
        f"trial-weighted; band = weighted SEM across mice (N up to {n_mice_used})",
        fontsize=10)
    ax.legend(fontsize=8, title=grouping, loc="best")
    fig.tight_layout()
    out = out_dir / f"across_mice_si__{method}__{grouping}"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Generic two-level renderers — scatter (x vs y) and bars (scalar metrics)
# ---------------------------------------------------------------------------


def _scatter_two_level(mice: list[MouseData], cache: dict, out_dir: Path, *,
                       name: str, suptitle: str, panels: list,
                       min_trials: int = MIN_SESSION_TRIALS):
    """Generic per-session scatter renderer.

    ``panels`` is a list of ``(subtitle, xlabel, ylabel, point_fn)`` where
    ``point_fn(entry) -> (x, y)`` floats (NaN to drop). Produces two
    figures: ``{name}__per_session`` (sessions coloured by mouse + a
    trial-weighted mouse-mean X marker) and ``{name}__across_mice``
    (one trial-weighted point per mouse + Pearson r across mice).

    Returns ``(path_per_session, path_across_mice)``.
    """
    n_panels = len(panels)
    pcols = min(n_panels, 2)
    prows = int(np.ceil(n_panels / pcols))

    fig_s, axes_s = plt.subplots(prows, pcols, figsize=(5.6 * pcols, 4.6 * prows),
                                 squeeze=False)
    fig_a, axes_a = plt.subplots(prows, pcols, figsize=(5.6 * pcols, 4.6 * prows),
                                 squeeze=False)
    fig_s.suptitle(suptitle + "  — per session (point = session, X = mouse mean)",
                   fontsize=10)
    fig_a.suptitle(suptitle + "  — across mice (point = mouse, trial-weighted)",
                   fontsize=10)

    any_data = False
    for pi, (subtitle, xlabel, ylabel, point_fn) in enumerate(panels):
        ax_s = axes_s.ravel()[pi]
        ax_a = axes_a.ravel()[pi]
        mouse_x, mouse_y, mouse_w, mouse_tags = [], [], [], []
        for mi, m in enumerate(mice):
            entries = [e for e in _usable(cache[m.tag])
                       if e["n_trials"] >= min_trials]
            xs, ys, ws = [], [], []
            for e in entries:
                x, y = point_fn(e)
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(float(x))
                    ys.append(float(y))
                    ws.append(float(e["n_trials"]))
            if not xs:
                continue
            color = _mouse_color(mi)
            ax_s.scatter(xs, ys, color=color, s=28, alpha=0.6, label=m.tag)
            w = np.array(ws)
            mx = float(agg.session_to_mouse(np.array(xs), w))
            my = float(agg.session_to_mouse(np.array(ys), w))
            ax_s.scatter([mx], [my], color=color, s=130, marker="X",
                         edgecolors="black", zorder=5)
            mouse_x.append(mx)
            mouse_y.append(my)
            mouse_w.append(float(w.sum()))
            mouse_tags.append(m.tag)
            ax_a.scatter([mx], [my], color=color, s=90, edgecolors="black",
                         zorder=3, label=m.tag)
            ax_a.annotate(m.tag, (mx, my), fontsize=7, xytext=(4, 4),
                          textcoords="offset points")
        for ax, ttl in ((ax_s, subtitle), (ax_a, subtitle)):
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_title(ttl, fontsize=9)
        if len(mouse_x) >= 3:
            any_data = True
            r, p = pearsonr(mouse_x, mouse_y)
            ax_a.set_title(f"{subtitle}\nacross-mice Pearson r = {r:.2f}, "
                           f"p = {p:.3f}, N = {len(mouse_x)} mice", fontsize=9)
        elif mouse_x:
            any_data = True
    for ax in axes_s.ravel()[n_panels:]:
        ax.set_axis_off()
    for ax in axes_a.ravel()[n_panels:]:
        ax.set_axis_off()
    if axes_s.ravel()[0].get_legend_handles_labels()[0]:
        axes_s.ravel()[0].legend(fontsize=6, loc="best")
    if not any_data:
        plt.close(fig_s)
        plt.close(fig_a)
        return None, None
    fig_s.tight_layout(rect=(0, 0, 1, 0.93))
    fig_a.tight_layout(rect=(0, 0, 1, 0.91))
    out_s = out_dir / f"{name}__per_session"
    out_a = out_dir / f"{name}__across_mice"
    _save_fig(fig_s, out_s)
    _save_fig(fig_a, out_a)
    return out_s, out_a


def _bars_two_level(mice: list[MouseData], cache: dict, out_dir: Path, *,
                    name: str, suptitle: str, metric_fn, keys: list,
                    key_labels: dict | None = None, key_colors: dict | None = None,
                    ylabel: str = "value",
                    min_trials: int = MIN_SESSION_TRIALS):
    """Generic two-level renderer for per-session scalar metrics.

    ``metric_fn(entry) -> {key: float}`` is evaluated on usable sessions with
    at least ``min_trials`` trials. Values are aggregated session -> mouse
    (trial-weighted) and mouse -> group (trial-weighted, weighted SEM).
    Produces ``{name}__per_mouse`` (grouped bars, mouse x key, session
    points overlaid) and ``{name}__across_mice`` (grand mean +/- SEM per key).
    """
    key_labels = key_labels or {k: str(k) for k in keys}
    key_colors = key_colors or {k: _mouse_color(i) for i, k in enumerate(keys)}

    # per_mouse[tag][key] = trial-weighted mouse value; sessions[tag][key] = list
    per_mouse: dict[str, dict] = {}
    sessions: dict[str, dict] = {}
    mouse_weight: dict[str, float] = {}
    for m in mice:
        entries = [e for e in _usable(cache[m.tag])
                   if e["n_trials"] >= min_trials]
        if not entries:
            continue
        vals = {k: [] for k in keys}
        wts = {k: [] for k in keys}
        for e in entries:
            md = metric_fn(e)
            for k in keys:
                v = md.get(k, np.nan)
                if np.isfinite(v):
                    vals[k].append(float(v))
                    wts[k].append(float(e["n_trials"]))
        mvals = {}
        sess_vals = {}
        for k in keys:
            if vals[k]:
                mvals[k] = float(agg.session_to_mouse(
                    np.array(vals[k]), np.array(wts[k])))
                sess_vals[k] = list(vals[k])
            else:
                mvals[k] = np.nan
                sess_vals[k] = []
        if all(not np.isfinite(v) for v in mvals.values()):
            continue
        per_mouse[m.tag] = mvals
        sessions[m.tag] = sess_vals
        # Mouse -> group weight = total trials of this mouse's qualifying
        # sessions (these are whole-session scalar metrics, so the weight
        # is the session trial count, not a per-key sub-count).
        mouse_weight[m.tag] = float(sum(e["n_trials"] for e in entries))
    if not per_mouse:
        return None, None

    tags = list(per_mouse.keys())
    n_keys = len(keys)
    bar_w = 0.8 / n_keys

    # --- per-mouse figure ---
    fig_m, ax_m = plt.subplots(1, 1, figsize=(max(7, 1.4 * len(tags) + 3), 5.5))
    fig_m.suptitle(suptitle + "  — per mouse (bar = trial-weighted session mean)",
                   fontsize=10)
    x = np.arange(len(tags))
    for ki, k in enumerate(keys):
        offs = (ki - (n_keys - 1) / 2) * bar_w
        heights = [per_mouse[t][k] for t in tags]
        ax_m.bar(x + offs, heights, bar_w, color=key_colors[k],
                 label=key_labels[k], alpha=0.85)
        for xi, t in enumerate(tags):
            sv = sessions[t][k]
            if sv:
                jit = (np.random.default_rng(ki * 97 + xi).random(len(sv)) - 0.5)
                ax_m.scatter(np.full(len(sv), x[xi] + offs) + jit * bar_w * 0.7,
                             sv, s=10, color="black", alpha=0.45, zorder=4)
    ax_m.axhline(0, color="black", lw=0.7)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax_m.set_ylabel(ylabel, fontsize=9)
    ax_m.legend(fontsize=7, loc="best")
    ax_m.grid(alpha=0.2, axis="y")
    fig_m.tight_layout(rect=(0, 0, 1, 0.94))
    out_m = out_dir / f"{name}__per_mouse"
    _save_fig(fig_m, out_m)

    # --- across-mice figure ---
    fig_g, ax_g = plt.subplots(1, 1, figsize=(max(6, 1.3 * n_keys + 3), 5.2))
    weights = np.array([mouse_weight[t] for t in tags])
    g_means, g_sems = [], []
    for k in keys:
        col = np.array([per_mouse[t][k] for t in tags])
        gm, gs, _ = agg.mouse_to_group(col, weights)
        g_means.append(float(gm))
        g_sems.append(float(gs) if np.isfinite(gs) else 0.0)
    xk = np.arange(n_keys)
    ax_g.bar(xk, g_means, 0.6, yerr=g_sems, capsize=5,
             color=[key_colors[k] for k in keys], alpha=0.85)
    ax_g.axhline(0, color="black", lw=0.7)
    ax_g.set_xticks(xk)
    ax_g.set_xticklabels([key_labels[k] for k in keys], rotation=20,
                         ha="right", fontsize=8)
    ax_g.set_ylabel(ylabel, fontsize=9)
    ax_g.set_title(suptitle + f"\nacross-mice grand mean (trial-weighted), "
                   f"error = weighted SEM, N = {len(tags)} mice", fontsize=9)
    ax_g.grid(alpha=0.2, axis="y")
    fig_g.tight_layout()
    out_g = out_dir / f"{name}__across_mice"
    _save_fig(fig_g, out_g)
    return out_m, out_g


# ---------------------------------------------------------------------------
# kappa vs SI-spread
# ---------------------------------------------------------------------------


def render_per_session_kappa_vs_si_spread(mice: list[MouseData], cache: dict,
                                          out_dir: Path):
    """Bundle kappa vs trial-mean-SI spread, per SI method (per-session +
    across-mice scatter)."""
    def _make(method):
        def _fn(e):
            spread = _per_animal_si_spread(e["session"], e["proto"], e["ex"],
                                           method=method)
            return _kappa_avg(e), spread
        return _fn
    panels = [(SI_METHOD_LABELS[method], "bundle kappa (sides averaged)",
               "trial-mean-SI spread", _make(method))
              for method in SI_METHODS]
    return _scatter_two_level(
        mice, cache, out_dir,
        name="per_session_kappa_vs_si_spread",
        suptitle="Bundle kappa vs trial-mean-SI spread",
        panels=panels)


def render_per_session_kappa_vs_lapse(mice: list[MouseData], cache: dict,
                                      out_dir: Path):
    """Bundle kappa vs behavioural lapse rate (Go-side and NoGo-side)."""
    def _go(e):
        lg, _ln = _per_animal_lapse(e["session"])
        return _kappa_avg(e), lg

    def _nogo(e):
        _lg, ln = _per_animal_lapse(e["session"])
        return _kappa_avg(e), ln
    panels = [
        ("Go-side lapse  P(NoGo | strong Go)", "bundle kappa", "lapse rate", _go),
        ("NoGo-side lapse  P(Go | strong NoGo)", "bundle kappa", "lapse rate", _nogo),
    ]
    return _scatter_two_level(
        mice, cache, out_dir,
        name="per_session_kappa_vs_lapse",
        suptitle="Bundle kappa vs behavioural lapse rate",
        panels=panels)


def _session_ddm_fit(entry: dict) -> dict:
    """Interrogation-protocol probit fit for one session.

    Reports the identified parameterisation only — sigma_v is a flat likelihood
    ridge at a single interrogation time, so it is not fitted (see
    ``similarity_analysis._fit_interrogation_ddm``).
    """
    keys = ("slope", "bias", "lapse_L", "lapse_R")
    s = entry["session"]
    c = entry["feats"]["signed_contrast"]
    y = s.choice.astype(float)
    valid = np.isfinite(c) & np.isin(y, [0.0, 1.0])
    if valid.sum() < 30:
        return {k: float("nan") for k in keys}
    fit = _fit_interrogation_ddm(c[valid], y[valid])
    return {k: fit[k] for k in keys}


def render_per_session_kappa_vs_ddm_slope(mice: list[MouseData], cache: dict,
                                          out_dir: Path):
    """Bundle kappa vs the identified probit slope.

    Replaces the old kappa-vs-sigma_v panel (retired 2026-07-16): sigma_v is not
    identified at a single interrogation time. Slope A = alpha*sqrt(T) /
    sqrt(1 + T*sigma_v**2), so drift variability attenuates it and the framework
    predicts a positive kappa-slope correlation — confounded with alpha.
    """
    def _fn(e):
        return _kappa_avg(e), _session_ddm_fit(e)["slope"]
    panels = [("bundle kappa vs probit slope", "bundle kappa",
               "probit slope (alpha/sigma_v confounded)", _fn)]
    return _scatter_two_level(
        mice, cache, out_dir,
        name="per_session_kappa_vs_ddm_slope",
        suptitle="Bundle kappa vs identified probit slope "
                 "(sigma_v not identified at fixed T)",
        panels=panels)


# ---------------------------------------------------------------------------
# Behavioural scalar metrics — velocity drift, DDM, SI-choice logistic
# ---------------------------------------------------------------------------


def render_per_session_velocity_accumulator(mice: list[MouseData], cache: dict,
                                            out_dir: Path):
    """Per-session pre-RZ-velocity-vs-signed_contrast regression parameters."""
    def _fn(e):
        fit = _per_animal_velocity_drift(e["session"])
        return {"alpha": fit["alpha"], "bias": fit["bias"],
                "r_squared": fit["r_squared"]}
    return _bars_two_level(
        mice, cache, out_dir,
        name="per_session_velocity_accumulator",
        suptitle="Velocity-as-accumulator: pre-RZ velocity ~ signed_contrast",
        metric_fn=_fn, keys=["alpha", "bias", "r_squared"],
        key_labels={"alpha": "drift slope alpha", "bias": "bias",
                    "r_squared": "R^2"},
        ylabel="fit parameter value")


def render_per_session_ddm(mice: list[MouseData], cache: dict, out_dir: Path):
    """Per-session interrogation-protocol probit parameters (identified only)."""
    return _bars_two_level(
        mice, cache, out_dir,
        name="per_session_ddm",
        suptitle="Interrogation-protocol probit fit parameters "
                 "(sigma_v not identified at fixed T)",
        metric_fn=_session_ddm_fit,
        keys=["slope", "bias", "lapse_L", "lapse_R"],
        key_labels={"slope": "probit slope", "bias": "bias (probit)",
                    "lapse_L": "lapse_L", "lapse_R": "lapse_R"},
        ylabel="fit parameter value", min_trials=30)


def render_per_session_si_choice_logistic(mice: list[MouseData], cache: dict,
                                          out_dir: Path):
    """Per-session logistic choice prediction: LL(SI) - LL(signed_contrast)."""
    def _fn(e):
        out = {}
        for method in SI_METHODS:
            fit = _per_animal_si_logistic(e["session"], e["proto"], e["ex"],
                                          method=method)
            out[method] = fit["delta_ll"]
        return out
    return _bars_two_level(
        mice, cache, out_dir,
        name="per_session_si_choice_logistic",
        suptitle="Logistic choice prediction: LL(trial-mean SI) - LL(signed_contrast)",
        metric_fn=_fn, keys=list(SI_METHODS),
        key_labels=SI_METHOD_LABELS, key_colors=METHOD_COLORS,
        ylabel="delta log-likelihood (SI - stim)")


def render_per_session_si_choice_logistic_xG_binned(mice: list[MouseData],
                                                    cache: dict, out_dir: Path):
    """Per-session xG-binned logistic choice prediction."""
    def _fn(e):
        out = {}
        for method in SI_METHODS:
            fit = _per_animal_si_logistic_xG_binned(
                e["session"], e["proto"], e["ex"], method=method)
            out[method] = fit["delta_ll"]
        return out
    return _bars_two_level(
        mice, cache, out_dir,
        name="per_session_si_choice_logistic_xG_binned",
        suptitle="xG-binned logistic choice prediction: LL(SI bins) - LL(signed_contrast)",
        metric_fn=_fn, keys=list(SI_METHODS),
        key_labels=SI_METHOD_LABELS, key_colors=METHOD_COLORS,
        ylabel="delta log-likelihood (SI bins - stim)", min_trials=30)


def render_per_session_si_vs_confidence(mice: list[MouseData], cache: dict,
                                        out_dir: Path, *, proxy: str = "entropy"):
    """Per-session correlation of trial-mean SI with a confidence proxy."""
    if proxy not in ("entropy", "velocity"):
        raise ValueError(f"unknown proxy {proxy!r}")

    def _fn(e):
        s = e["session"]
        if proxy == "entropy":
            y = _decision_entropy(s.decision_posterior)
        else:
            y = s.velocity.astype(float)
        out = {}
        for method in SI_METHODS:
            tms = np.nanmean(e["si"][method], axis=1)
            v = np.isfinite(tms) & np.isfinite(y)
            if v.sum() >= 10:
                r, _ = pearsonr(tms[v], y[v])
                out[method] = r
            else:
                out[method] = np.nan
        return out
    return _bars_two_level(
        mice, cache, out_dir,
        name=f"per_session_si_vs_confidence__{proxy}",
        suptitle=f"Correlation of trial-mean SI with confidence proxy ({proxy})",
        metric_fn=_fn, keys=list(SI_METHODS),
        key_labels=SI_METHOD_LABELS, key_colors=METHOD_COLORS,
        ylabel=f"Pearson r (trial-mean SI vs {proxy})")


# ---------------------------------------------------------------------------
# SI-variant comparison
# ---------------------------------------------------------------------------


def _mouse_variant_traces(entries: list[dict]):
    """Trial-weighted mouse-mean SI trace per method at max |signed_contrast|."""
    out = {}
    for method in SI_METHODS:
        traces, weights = [], []
        for e in entries:
            c = e["feats"]["signed_contrast"]
            finite = np.isfinite(c)
            if finite.sum() == 0:
                continue
            uniq = np.unique(c[finite])
            lv = uniq[np.argmax(np.abs(uniq))]
            mask = (c == lv)
            n = int(np.sum(mask))
            if n >= 3:
                traces.append(np.nanmean(e["si"][method][mask], axis=0))
                weights.append(float(n))
        if traces:
            out[method] = (agg.session_to_mouse(np.vstack(traces),
                                                 np.array(weights)),
                           float(sum(weights)))
    return out


def render_per_session_variants_compare(mice: list[MouseData], cache: dict,
                                        out_dir: Path):
    """SI variants compared at max |signed_contrast| — per mouse + across mice."""
    rows, cols = _grid_dims(len(mice))
    fig_m, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                               squeeze=False)
    fig_m.suptitle("SI variants at max |signed_contrast| — per mouse "
                   "(trial-weighted session mean)", fontsize=11)
    across = {method: {"xgs": [], "traces": [], "weights": []}
              for method in SI_METHODS}
    drew = False
    for ax, m in zip(axes.ravel(), mice):
        entries = _usable(cache[m.tag])
        traces = _mouse_variant_traces(entries) if entries else {}
        if not traces:
            ax.set_axis_off()
            continue
        drew = True
        for method, (tr, w) in traces.items():
            ax.plot(m.xG, tr, color=METHOD_COLORS[method], lw=1.6,
                    label=SI_METHOD_LABELS[method])
            across[method]["xgs"].append(m.xG)
            across[method]["traces"].append(tr)
            across[method]["weights"].append(w)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("xG (a.u.)", fontsize=7)
        ax.set_ylabel("SI", fontsize=7)
        ax.set_title(m.tag, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    if not drew:
        plt.close(fig_m)
        return None, None
    handles = [Line2D([0], [0], color=METHOD_COLORS[k], lw=1.6,
                      label=SI_METHOD_LABELS[k]) for k in SI_METHODS]
    fig_m.legend(handles=handles, loc="upper center", ncol=len(SI_METHODS),
                 bbox_to_anchor=(0.5, 0.97), frameon=False, fontsize=8)
    fig_m.tight_layout(rect=(0, 0, 1, 0.91))
    out_m = out_dir / "per_session_variants_compare__per_mouse"
    _save_fig(fig_m, out_m)

    fig_a, ax_a = plt.subplots(1, 1, figsize=(8.0, 5.2))
    n_used = 0
    for method in SI_METHODS:
        d = across[method]
        if not d["traces"]:
            continue
        n_used = max(n_used, len(d["traces"]))
        xg_ref, stacked = _stack_traces(d["xgs"], d["traces"])
        gmean, gsem, _ = agg.mouse_to_group(stacked, np.array(d["weights"]))
        ax_a.plot(xg_ref, gmean, color=METHOD_COLORS[method], lw=2.4,
                  label=SI_METHOD_LABELS[method])
        band = np.where(np.isfinite(gsem), gsem, 0.0)
        ax_a.fill_between(xg_ref, gmean - band, gmean + band,
                          color=METHOD_COLORS[method], alpha=0.18)
    ax_a.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_a.set_xlabel("xG (a.u.)", fontsize=10)
    ax_a.set_ylabel("grand-mean SI", fontsize=10)
    ax_a.set_title("SI variants at max |signed_contrast| — across mice "
                   f"(trial-weighted, N up to {n_used} mice)", fontsize=10)
    ax_a.legend(fontsize=8)
    fig_a.tight_layout()
    out_a = out_dir / "per_session_variants_compare__across_mice"
    _save_fig(fig_a, out_a)
    return out_m, out_a


# ---------------------------------------------------------------------------
# Single-trial scatter in (sim_NoGo, sim_Go)
# ---------------------------------------------------------------------------


def render_per_session_single_trial_scatter(mice: list[MouseData], cache: dict,
                                            out_dir: Path):
    """Single-trial (sim_NoGo, sim_Go) cloud at a fixed xG, by signed_contrast.

    Per mouse: the trial cloud pooled across the mouse's sessions (each
    trial scored against *its own session's* archetypes, so the similarity
    coordinates are session-consistent). Across mice: per signed_contrast
    level, one trial-weighted mouse-mean (sim_NoGo, sim_Go) centroid per
    mouse.
    """
    grouping = "signed_contrast"
    union = _union_levels(mice, grouping)
    pal = _palette(union, grouping)
    rows, cols = _grid_dims(len(mice))
    fig_m, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                               squeeze=False)
    fig_m.suptitle(f"Single-trial scatter (sim_NoGo, sim_Go) at xG~{XG_TARGET:g} "
                   "— per mouse, sessions pooled", fontsize=11)
    centroids = {lv: {"x": [], "y": [], "w": []} for lv in union}
    drew = False
    for ax, m in zip(axes.ravel(), mice):
        entries = _usable(cache[m.tag])
        if not entries:
            ax.set_axis_off()
            continue
        per_level_pts = {lv: {"x": [], "y": []} for lv in union}
        for e in entries:
            s = e["session"]
            sim, names = _archetype_similarity(s, e["proto"])
            if "Go" not in names or "NoGo" not in names:
                continue
            gi, ni = names.index("Go"), names.index("NoGo")
            idx = int(np.argmin(np.abs(s.xG - XG_TARGET)))
            c = e["feats"][grouping]
            for lv in union:
                mask = (c == lv)
                if np.sum(mask) > 0:
                    per_level_pts[lv]["x"].extend(sim[mask, idx, ni].tolist())
                    per_level_pts[lv]["y"].extend(sim[mask, idx, gi].tolist())
        plotted = False
        for lv in union:
            xs = per_level_pts[lv]["x"]
            ys = per_level_pts[lv]["y"]
            if not xs:
                continue
            plotted = True
            ax.scatter(xs, ys, color=pal[lv], s=9, alpha=0.4, edgecolors="none")
            centroids[lv]["x"].append(float(np.nanmean(xs)))
            centroids[lv]["y"].append(float(np.nanmean(ys)))
            centroids[lv]["w"].append(float(len(xs)))
        if not plotted:
            ax.set_axis_off()
            continue
        drew = True
        lo, hi = -1.05, 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("sim to NoGo", fontsize=7)
        ax.set_ylabel("sim to Go", fontsize=7)
        ax.set_title(m.tag, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    if not drew:
        plt.close(fig_m)
        return None, None
    _finalize_grid(fig_m, cont=False, levels=union, palette=pal, label=grouping)
    out_m = out_dir / "per_session_single_trial_scatter__per_mouse"
    _save_fig(fig_m, out_m)

    fig_a, ax_a = plt.subplots(1, 1, figsize=(6.4, 6.0))
    for lv in union:
        d = centroids[lv]
        if not d["x"]:
            continue
        ax_a.scatter(d["x"], d["y"], color=pal[lv], s=70, edgecolors="black",
                     alpha=0.9, label=f"{lv:g}" if isinstance(lv, float) else str(lv))
    ax_a.plot([-1.05, 1.05], [-1.05, 1.05], "k--", lw=0.5, alpha=0.4)
    ax_a.set_xlim(-1.05, 1.05)
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_xlabel("sim to NoGo", fontsize=10)
    ax_a.set_ylabel("sim to Go", fontsize=10)
    ax_a.set_title(f"Across-mice (sim_NoGo, sim_Go) centroids at xG~{XG_TARGET:g}\n"
                   "one point = one mouse, coloured by signed_contrast", fontsize=10)
    ax_a.legend(fontsize=7, title=grouping, loc="best")
    fig_a.tight_layout()
    out_a = out_dir / "per_session_single_trial_scatter__across_mice"
    _save_fig(fig_a, out_a)
    return out_m, out_a


# ---------------------------------------------------------------------------
# Psychometric — per-session grid + across-mice grand average
# ---------------------------------------------------------------------------


def _fit_logistic(x: np.ndarray, y: np.ndarray):
    """1-D logistic MLE; returns (beta, b) or None if it cannot fit."""
    from scipy.optimize import minimize
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 20 or len(np.unique(y)) < 2:
        return None

    def neg_ll(params):
        beta, b = params
        z = beta * x + b
        return -float(np.sum(y * (-np.logaddexp(0.0, -z))
                             + (1 - y) * (-np.logaddexp(0.0, z))))
    res = minimize(neg_ll, x0=[0.0, 0.0], method="Nelder-Mead")
    if not res.success:
        return None
    return float(res.x[0]), float(res.x[1])


def _mouse_psychometric(entries: list[dict], grouping: str, levels: list):
    """Trial-weighted per-level P(Go) for one mouse, plus SI-derived predictions.

    The actual psychometric is computed per session then trial-weighted into
    a mouse curve. The SI predictions fit one logistic per method on the
    mouse's pooled (session-invariant) trial-mean SI vs choice, then average
    the predicted P(Go) per level. Returns ``(actual, n_per_level, pred)``
    where ``pred[method]`` is a per-level array.
    """
    n_lv = len(levels)
    pg_rows, w_rows = [], []
    pooled_si = {meth: [] for meth in SI_METHODS}
    pooled_choice, pooled_c = [], []
    for e in entries:
        c = e["feats"][grouping]
        y = e["session"].choice.astype(float)
        row_pg = np.full(n_lv, np.nan)
        row_w = np.zeros(n_lv)
        for i, lv in enumerate(levels):
            mask = (c == lv) & np.isfinite(y)
            if np.sum(mask) > 0:
                row_pg[i] = float(np.mean(y[mask]))
                row_w[i] = float(np.sum(mask))
        pg_rows.append(row_pg)
        w_rows.append(row_w)
        valid = np.isfinite(c) & np.isin(y, [0.0, 1.0])
        pooled_choice.append(y[valid])
        pooled_c.append(c[valid])
        for meth in SI_METHODS:
            tms = np.nanmean(e["si"][meth], axis=1)
            pooled_si[meth].append(tms[valid])

    pg_arr = np.array(pg_rows)
    w_arr = np.array(w_rows)
    actual = np.array([agg.session_to_mouse(pg_arr[:, i], w_arr[:, i])
                       for i in range(n_lv)])
    n_per = w_arr.sum(axis=0)

    pred = {}
    if pooled_c:
        c_all = np.concatenate(pooled_c)
        y_all = np.concatenate(pooled_choice)
        for meth in SI_METHODS:
            si_all = np.concatenate(pooled_si[meth])
            v = np.isfinite(si_all) & np.isfinite(y_all)
            fit = _fit_logistic(si_all[v], y_all[v])
            if fit is None:
                pred[meth] = np.full(n_lv, np.nan)
                continue
            beta, b = fit
            p_hat = 1.0 / (1.0 + np.exp(-(beta * si_all + b)))
            curve = np.full(n_lv, np.nan)
            for i, lv in enumerate(levels):
                mask = (c_all == lv) & v
                if np.sum(mask) > 0:
                    curve[i] = float(np.mean(p_hat[mask]))
            pred[meth] = curve
    else:
        pred = {meth: np.full(n_lv, np.nan) for meth in SI_METHODS}
    return actual, n_per, pred


def render_per_session_psychometric(mice: list[MouseData], cache: dict,
                                    grouping: str, out_dir: Path) -> Path | None:
    """Per-mouse psychometric: trial-weighted actual P(Go) + SI predictions."""
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle(f"Per-session psychometric | x = {grouping} | "
                 "actual (black, trial-weighted) vs SI predictions", fontsize=11)
    drew = False
    for ax, m in zip(axes.ravel(), mice):
        entries = _usable(cache[m.tag])
        levels = _level_order(grouping, compute_features(m)[grouping])
        if not entries or len(levels) < 2:
            ax.set_axis_off()
            continue
        actual, n_per, pred = _mouse_psychometric(entries, grouping, levels)
        ok = np.isfinite(actual) & (n_per > 0)
        if ok.sum() < 2:
            ax.set_axis_off()
            continue
        drew = True
        lv_num = np.array([float(x) for x in levels])
        ax.plot(lv_num[ok], actual[ok], "o-", color="black", lw=2.0,
                markersize=5, label="actual", zorder=5)
        for meth in SI_METHODS:
            cv = pred[meth]
            okp = np.isfinite(cv) & (n_per > 0)
            if okp.sum() >= 2:
                ax.plot(lv_num[okp], cv[okp], "-", color=METHOD_COLORS[meth],
                        lw=1.3, alpha=0.85, label=SI_METHOD_LABELS[meth])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.4)
        ax.axvline(45.0 if grouping == "abs_from_go" else 0.0,
                   color="gray", lw=0.5, ls=":", alpha=0.4)
        ax.set_xlabel(grouping, fontsize=7)
        ax.set_ylabel("P(Go)", fontsize=8)
        ax.set_title(m.tag, fontsize=8)
        ax.tick_params(labelsize=6)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=6, loc="best")
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    if not drew:
        plt.close(fig)
        return None
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"per_session_psychometric__{grouping}"
    _save_fig(fig, out)
    return out


def render_across_mice_psychometric(mice: list[MouseData], cache: dict,
                                    grouping: str, out_dir: Path) -> Path | None:
    """Across-mice grand-average psychometric (trial-weighted) + SI predictions."""
    union = _union_levels(mice, grouping)
    if len(union) < 2:
        return None
    n_lv = len(union)
    actual_rows, weight_rows = [], []
    pred_rows = {meth: [] for meth in SI_METHODS}
    for m in mice:
        entries = _usable(cache[m.tag])
        if not entries:
            continue
        actual, n_per, pred = _mouse_psychometric(entries, grouping, union)
        if np.sum(np.isfinite(actual) & (n_per > 0)) < 2:
            continue
        actual_rows.append(actual)
        weight_rows.append(n_per)
        for meth in SI_METHODS:
            pred_rows[meth].append(pred[meth])
    if not actual_rows:
        return None
    actual_arr = np.array(actual_rows)
    weight_arr = np.array(weight_rows)
    g_actual = np.full(n_lv, np.nan)
    g_sem = np.full(n_lv, np.nan)
    for i in range(n_lv):
        gm, gs, _ = agg.mouse_to_group(actual_arr[:, i], weight_arr[:, i])
        g_actual[i] = gm
        g_sem[i] = gs

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.4))
    lv_num = np.array([float(x) for x in union])
    ok = np.isfinite(g_actual)
    band = np.where(np.isfinite(g_sem), g_sem, 0.0)
    ax.plot(lv_num[ok], g_actual[ok], "o-", color="black", lw=2.4,
            markersize=6, label="actual (grand mean)", zorder=5)
    ax.fill_between(lv_num[ok], (g_actual - band)[ok], (g_actual + band)[ok],
                    color="black", alpha=0.15, zorder=2)
    for meth in SI_METHODS:
        pr = np.array(pred_rows[meth])
        g_pred = np.full(n_lv, np.nan)
        for i in range(n_lv):
            gm, _gs, _ = agg.mouse_to_group(pr[:, i], weight_arr[:, i])
            g_pred[i] = gm
        okp = np.isfinite(g_pred)
        if okp.sum() >= 2:
            ax.plot(lv_num[okp], g_pred[okp], "-", color=METHOD_COLORS[meth],
                    lw=1.6, alpha=0.85, label=SI_METHOD_LABELS[meth])
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.axvline(45.0 if grouping == "abs_from_go" else 0.0,
               color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel(grouping, fontsize=10)
    ax.set_ylabel("P(Go)", fontsize=10)
    ax.set_title(f"Across-mice psychometric | x = {grouping}\n"
                 f"trial-weighted; band = weighted SEM across mice "
                 f"(N = {len(actual_rows)} mice)", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = out_dir / f"across_mice_psychometric__{grouping}"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _build_jobs(mice: list[MouseData], cache: dict, sF: Path,
                f_groups: list[str]) -> list[tuple]:
    """Deterministic ordered list of ``(label, grouping, fn)`` renderer jobs.

    The order is fixed so the sweep can be run in slices (``job_start`` /
    ``job_count``) across separate short-lived processes and still cover
    every job exactly once.
    """
    jobs: list[tuple] = []
    jobs.append(("archetype_stability", None,
                 lambda: render_archetype_stability(mice, cache, sF)))
    jobs.append(("archetype_stability_across_mice", None,
                 lambda: render_archetype_stability_across_mice(mice, cache, sF)))
    jobs.append(("archetype_drift", None,
                 lambda: render_archetype_drift(mice, cache, sF)))
    for g in f_groups:
        for method in SI_METHODS:
            jobs.append((f"per_session_si__{method}", g,
                         lambda g=g, method=method:
                         render_per_session_si_over_xG(mice, cache, g, sF,
                                                       method=method)))
            jobs.append((f"across_mice_si__{method}", g,
                         lambda g=g, method=method:
                         render_across_mice_si_over_xG(mice, cache, g, sF,
                                                       method=method)))
    for g in ("signed_contrast", "abs_from_go"):
        jobs.append(("per_session_psychometric", g,
                     lambda g=g: render_per_session_psychometric(
                         mice, cache, g, sF)))
        jobs.append(("across_mice_psychometric", g,
                     lambda g=g: render_across_mice_psychometric(
                         mice, cache, g, sF)))
    jobs.append(("kappa_vs_si_spread", "signed_contrast",
                 lambda: render_per_session_kappa_vs_si_spread(mice, cache, sF)))
    jobs.append(("kappa_vs_lapse", "signed_contrast",
                 lambda: render_per_session_kappa_vs_lapse(mice, cache, sF)))
    jobs.append(("kappa_vs_ddm_slope", "signed_contrast",
                 lambda: render_per_session_kappa_vs_ddm_slope(mice, cache, sF)))
    jobs.append(("velocity_accumulator", "signed_contrast",
                 lambda: render_per_session_velocity_accumulator(mice, cache, sF)))
    jobs.append(("ddm", "signed_contrast",
                 lambda: render_per_session_ddm(mice, cache, sF)))
    jobs.append(("si_choice_logistic", "signed_contrast",
                 lambda: render_per_session_si_choice_logistic(mice, cache, sF)))
    jobs.append(("si_choice_logistic_xG_binned", "signed_contrast",
                 lambda: render_per_session_si_choice_logistic_xG_binned(
                     mice, cache, sF)))
    for proxy in ("entropy", "velocity"):
        jobs.append((f"si_vs_confidence__{proxy}", "signed_contrast",
                     lambda proxy=proxy: render_per_session_si_vs_confidence(
                         mice, cache, sF, proxy=proxy)))
    jobs.append(("variants_compare", "signed_contrast",
                 lambda: render_per_session_variants_compare(mice, cache, sF)))
    jobs.append(("single_trial_scatter", "signed_contrast",
                 lambda: render_per_session_single_trial_scatter(mice, cache, sF)))
    return jobs


def run_session_similarity_sweep(
    data_path: Path = DATA_PATH,
    fig_root: Path | None = None,
    *,
    smoke: bool = False,
    job_start: int = 0,
    job_count: int | None = None,
) -> dict:
    """Run the per-session similarity sweep, optionally as a slice of jobs.

    The full sweep is a deterministic ordered list of renderer jobs. To keep
    each process short-lived (the run environment reaps detached processes),
    it can be run in slices: ``job_start`` / ``job_count`` select
    ``jobs[job_start : job_start + job_count]``. The first slice
    (``job_start == 0``) writes a fresh ``_meta_session.json``; later slices
    append to it. ``meta["complete"]`` is True once the final slice has run.
    """
    if fig_root is None:
        fig_root = (Path(__file__).resolve().parent.parent
                    / "figures" / "similarity_framework")
    fig_root = Path(fig_root)
    sF = fig_root / "F_per_session_similarity"
    sF.mkdir(parents=True, exist_ok=True)
    meta_path = sF / "_meta_session.json"

    t0 = time.time()
    mice_all = load_all_mice(data_path)
    mice, _skipped = _filter_mice(mice_all)
    if not mice:
        raise RuntimeError("no mice survive filtering; check thresholds / data")
    if smoke:
        mice = mice[:2]
        f_groups = ["signed_contrast"]
    else:
        f_groups = list(GROUPINGS_DISCRETE)

    print(f"[F_SESS] building per-session cache for {len(mice)} mice", flush=True)
    cache = build_session_cache(mice)
    jobs = _build_jobs(mice, cache, sF, f_groups)
    total = len(jobs)
    end = total if job_count is None else min(total, job_start + job_count)
    print(f"[JOBS] total={total} running slice [{job_start}:{end}]", flush=True)

    if job_start <= 0 or not meta_path.exists():
        session_summary = [{
            "tag": m.tag,
            "n_sessions": len(cache[m.tag]),
            "n_usable_sessions": sum(1 for e in cache[m.tag] if e["usable"]),
            "n_trials": int(m.n_trials),
        } for m in mice]
        meta = {
            "data_path": str(data_path),
            "fig_root": str(fig_root),
            "smoke": smoke,
            "n_mice": len(mice),
            "n_sessions_total": sum(s["n_sessions"] for s in session_summary),
            "n_usable_sessions_total": sum(s["n_usable_sessions"]
                                           for s in session_summary),
            "easy_block_size": EASY_BLOCK_SIZE,
            "min_session_trials": MIN_SESSION_TRIALS,
            "aggregation": "trial-count weighted (session->mouse->group); "
                           "across-mice error = weighted SEM (Kish n_eff)",
            "sessions": session_summary,
            "jobs_total": total,
            "produced": [],
        }
    else:
        with open(meta_path) as f:
            meta = json.load(f)

    def _record(label, grouping, result, secs):
        paths = result if isinstance(result, (list, tuple)) else [result]
        for p in paths:
            entry = {"label": label, "grouping": grouping, "secs": secs}
            if isinstance(p, Exception):
                entry["error"] = repr(p)
            elif p is None:
                entry["skipped"] = "renderer returned None"
            else:
                entry["path"] = str(p)
            meta["produced"].append(entry)

    for idx in range(job_start, end):
        label, grouping, fn = jobs[idx]
        t1 = time.time()
        print(f"[F_SESS] [{idx}/{total}] {label}"
              + (f" group={grouping}" if grouping else ""), flush=True)
        try:
            out = fn()
        except Exception as e:  # pragma: no cover - defensive
            out = e
        _record(label, grouping, out, round(time.time() - t1, 1))

    meta["jobs_done"] = end
    meta["complete"] = bool(end >= total)
    if meta["complete"]:
        meta["n_errors"] = sum(1 for p in meta["produced"] if "error" in p)
    meta["elapsed_secs_last_slice"] = round(time.time() - t0, 1)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[chunk] jobs [{job_start}:{end}]/{total} done; "
          f"complete={meta['complete']}; "
          f"slice {meta['elapsed_secs_last_slice']}s", flush=True)
    if meta["complete"]:
        n_err = meta.get("n_errors", 0)
        print(f"[done] {len(meta['produced'])} outputs, {n_err} errors; "
              f"meta -> {meta_path}", flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description="Per-session similarity analysis")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="2 mice x 1 grouping for a fast sanity check")
    p.add_argument("--job-start", type=int, default=0,
                   help="index of the first renderer job to run")
    p.add_argument("--job-count", type=int, default=None,
                   help="number of renderer jobs to run (default: all remaining)")
    args = p.parse_args()
    run_session_similarity_sweep(data_path=args.data, fig_root=args.out,
                                 smoke=args.smoke, job_start=args.job_start,
                                 job_count=args.job_count)


if __name__ == "__main__":
    main()
