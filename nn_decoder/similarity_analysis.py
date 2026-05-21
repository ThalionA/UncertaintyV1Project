"""Archetype-similarity analysis for the V1 Go/NoGo decoder data.

Split out from ``dim_reduction_explore.py`` so that iterating on the
*similarity framework* renderers does not require regenerating the full
PCA / Isomap / dPCA / CEBRA / Procrustes sweep each time, and so that
similarity outputs live in their own ``figures/similarity_framework/``
subtree.

Conceptual notes preserved from the original Style F block:

  Idea: define "easy" Go and NoGo trials as those with strong sensory
  evidence (top-quartile contrast AND bottom-quartile dispersion). Average
  their neural activity within each side to get an "archetype trajectory"
  per side, shape (n_neurons, n_xG). For every trial, compute its cosine
  similarity to each archetype at every xG bin — this gives a
  (n_trials, n_xG, 2) tensor where axis 2 indexes {Go-arch, NoGo-arch}.

  Why this is useful: the 2D similarity space (sim_NoGo, sim_Go) is a
  semantically meaningful coordinate system that exists in every mouse —
  (1, 0) is "looks like an easy NoGo", (0, 1) is "looks like an easy
  Go", diagonal is "ambiguous". Lines in this space let you compare
  trajectory geometry across mice without needing alignment.

Run standalone:

  python similarity_analysis.py [--data PATH] [--out PATH] [--smoke]

Outputs land under ``--out / F_archetype_similarity/`` with a ``_meta.json``
summary at the root of ``--out``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Shared infrastructure: data loading, feature computation, palette/grid
# helpers and the trajectory-panel plotter all live in the legacy module.
# Importing them here pulls the heavy-import cost of dim_reduction_explore
# (sklearn etc.) but avoids duplicating ~600 lines of utility code.
from dim_reduction_explore import (
    DATA_PATH,
    GO_BOUNDARY_DEG,
    GROUPINGS_DISCRETE,
    MouseData,
    S_GRID,
    _annotate_panel,
    _filter_mice,
    _finalize_grid,
    _grid_dims,
    _is_discrete,
    _level_order,
    _palette,
    _plot_traj_panel,
    _save_fig,
    _union_levels,
    _zscore_neurons,
    compute_features,
    load_all_mice,
)

# Re-export the symbols the test suite consumes — so tests can use a
# single ``import similarity_analysis as sim`` and reach everything they
# need without crossing module boundaries.
__all__ = [
    "MouseData",
    "GO_BOUNDARY_DEG",
    "S_GRID",
    "compute_features",
    "ARCHETYPE_CONTRAST_Q",
    "ARCHETYPE_DISPERSION_Q",
    "ARCHETYPE_MIN_TRIALS",
    "SI_EPS",
    "SI_METHODS",
    "SI_METHOD_LABELS",
    "_compute_archetypes",
    "_compute_archetypes_exemplars",
    "_archetype_similarity",
    "_archetype_similarity_exemplars",
    "_archetype_similarity_vmf",
    "_aggregate_exemplars",
    "_kappa_banerjee",
    "_log_C_p",
    "_si_normalised",
    "_si_per_method",
    "_per_animal_kappa",
    "_per_animal_si_spread",
    "_psychometric",
    "_velocity_curve",
    "_decision_entropy",
    "_per_animal_lapse",
    "_per_animal_velocity_drift",
    "_per_animal_si_logistic",
    "_per_animal_si_logistic_xG_binned",
    "_fit_interrogation_ddm",
    "render_style_F_sim_over_xG",
    "render_style_F_archetype_traj",
    "render_style_F_overlay",
    "render_style_F_si_normalised_over_xG",
    "render_style_F_variants_compare",
    "render_style_F_single_trial_scatter",
    "render_style_F_kappa_vs_si_spread",
    "render_style_F_psychometric_overlay",
    "render_style_F_behaviour_overlay",
    "render_style_F_kappa_vs_lapse",
    "render_style_F_velocity_accumulator",
    "render_style_F_si_choice_logistic",
    "render_style_F_si_choice_logistic_xG_binned",
    "render_style_F_ddm_interrogation",
    "render_style_F_kappa_vs_ddm_sigma_v",
    "render_style_F_si_vs_confidence",
    "run_similarity_sweep",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHETYPE_CONTRAST_Q = 0.75      # top-quartile contrast counts as "easy"
ARCHETYPE_DISPERSION_Q = 0.25    # bottom-quartile dispersion counts as "easy"
ARCHETYPE_MIN_TRIALS = 5         # need at least this many easy trials per side

SI_EPS = 1e-9

SI_METHODS = ("prototype", "expected_exemplar", "max_exemplar", "vmf")
SI_METHOD_LABELS = {
    "prototype": "prototype (cos to mean)",
    "expected_exemplar": "expected exemplar (E_k cos)",
    "max_exemplar": "max exemplar (max_k cos)",
    "vmf": "vMF likelihood",
}


# ---------------------------------------------------------------------------
# Archetype computation: prototype (mean) and exemplar (bundle) forms
# ---------------------------------------------------------------------------


def _compute_archetypes(m: MouseData, easy_block_size: int | None = None) -> dict[str, np.ndarray]:
    """Single trajectory per side: mean of easy-trial trajectories.

    Returns a dict mapping {'Go', 'NoGo'} → (n_neurons, n_xG). A side is
    omitted if there are fewer than ``ARCHETYPE_MIN_TRIALS`` easy trials on it.
    Uses ``m.orientation`` (= abs_from_go) for side determination so that
    flipped Go/NoGo orientation mappings across mice work correctly.
    
    If ``easy_block_size`` is provided, it extracts archetypes from the first 
    ``easy_block_size`` easy trials (contrast=1.0, dispersion=5, stim in {0, 90})
    rather than using quantiles.
    """
    Az = _zscore_neurons(m.activity)
    
    if easy_block_size is not None:
        easy = (m.contrast == 1.0) & (m.dispersion == 5.0) & np.isin(m.true_orientation, [0.0, 90.0])
        easy_idx = np.where(easy)[0][:easy_block_size]
        easy_mask = np.zeros_like(easy, dtype=bool)
        easy_mask[easy_idx] = True
        easy = easy_mask
    else:
        cont_ok = np.isfinite(m.contrast)
        disp_ok = np.isfinite(m.dispersion)
        if cont_ok.sum() == 0 or disp_ok.sum() == 0:
            return {}
        c_thresh = np.quantile(m.contrast[cont_ok], ARCHETYPE_CONTRAST_Q)
        d_thresh = np.quantile(m.dispersion[disp_ok], ARCHETYPE_DISPERSION_Q)
        easy = (m.contrast >= c_thresh) & (m.dispersion <= d_thresh)
        
    is_go_side = m.orientation < GO_BOUNDARY_DEG
    archs = {}
    go_mask = easy & is_go_side
    nogo_mask = easy & (~is_go_side)
    if go_mask.sum() >= ARCHETYPE_MIN_TRIALS:
        archs["Go"] = Az[go_mask].mean(axis=0)
    if nogo_mask.sum() >= ARCHETYPE_MIN_TRIALS:
        archs["NoGo"] = Az[nogo_mask].mean(axis=0)
    return archs


def _archetype_similarity(m: MouseData,
                          archetypes: dict[str, np.ndarray]
                          ) -> tuple[np.ndarray, list[str]]:
    """Per-trial × per-xG cosine similarity to each archetype.

    Returns (sim, names) where sim has shape (n_trials, n_xG, n_archetypes).
    """
    Az = _zscore_neurons(m.activity)
    n_trials, n_neurons, n_xG = Az.shape
    names = list(archetypes.keys())
    if not names:
        return np.empty((n_trials, n_xG, 0)), names
    sim = np.zeros((n_trials, n_xG, len(names)))
    for ai, name in enumerate(names):
        A = archetypes[name]                                  # (n_neurons, n_xG)
        num = (Az * A[None, :, :]).sum(axis=1)                # (n_trials, n_xG)
        norm_t = np.linalg.norm(Az, axis=1)                   # (n_trials, n_xG)
        norm_a = np.linalg.norm(A, axis=0)                    # (n_xG,)
        denom = norm_t * norm_a[None, :]
        denom = np.where(denom > 0, denom, 1.0)
        sim[..., ai] = num / denom
    return sim, names


# ---------------------------------------------------------------------------
# Bounded SI + bundle / exemplar / vMF variants
# ---------------------------------------------------------------------------
#
# The framework defines
#
#     SI = (s_R − s_L) / (s_R + s_L) ∈ [−1, +1]
#
# but cosine on z-scored neural states can be negative on ambiguous trials,
# making the denominator s_R + s_L cross zero. We use the Wiener positivity
# map s_c → (1 + s_c) / 2 ∈ [0, 1] first, which keeps the denominator in
# (0, 2] and the SI in a stable [−1, +1] regardless of cosine sign.
#
# Four aggregation methods over the easy-trial bundle are exposed:
#   - "prototype"          : cosine against mean of easy-trial trajectories
#                            (the existing _compute_archetypes behaviour)
#   - "expected_exemplar"  : E_k[cos(r, a^k)]
#   - "max_exemplar"       : max_k cos(r, a^k)
#   - "vmf"                : log-likelihood ratio under a per-xG von Mises–
#                            Fisher fit (mean direction + Banerjee kappa),
#                            transformed to [−1, +1] by tanh((Δlog p)/2).
#
# All four return decision-aligned normalised SI of shape (n_trials, n_xG).


def _si_normalised(sim_R: np.ndarray, sim_L: np.ndarray) -> np.ndarray:
    """Bounded SI in [-1, +1] via the Wiener positivity map.

    Maps cosine-style similarities (possibly negative) into [0, 1] via
    ``(1 + s) / 2``, then computes the standard SI ratio. The denominator
    sR + sL after this map lies in (0, 2], so the ratio is numerically
    stable even when raw cosines cross zero on ambiguous trials.

    Parameters
    ----------
    sim_R, sim_L : np.ndarray
        Any-shape arrays of cosine-style similarities, typically in [-1, +1].

    Returns
    -------
    np.ndarray
        Same shape as inputs; values in [-1, +1].
    """
    sR_pos = (1.0 + sim_R) / 2.0
    sL_pos = (1.0 + sim_L) / 2.0
    return (sR_pos - sL_pos) / (sR_pos + sL_pos + SI_EPS)


def _compute_archetypes_exemplars(m: MouseData, easy_block_size: int | None = None) -> dict[str, np.ndarray]:
    """Individual easy-trial trajectories per side — the bundle, not the mean.

    Same "easy" filter as :func:`_compute_archetypes` (or exact block if ``easy_block_size``
    is provided), same minimum-trials requirement, but returns all N easy-trial trajectories 
    for each side.
    
    Returns a dict mapping {'Go', 'NoGo'} → (n_easy_trials, n_neurons, n_xG).
    """
    Az = _zscore_neurons(m.activity)
    
    if easy_block_size is not None:
        easy = (m.contrast == 1.0) & (m.dispersion == 5.0) & np.isin(m.true_orientation, [0.0, 90.0])
        easy_idx = np.where(easy)[0][:easy_block_size]
        easy_mask = np.zeros_like(easy, dtype=bool)
        easy_mask[easy_idx] = True
        easy = easy_mask
    else:
        cont_ok = np.isfinite(m.contrast)
        disp_ok = np.isfinite(m.dispersion)
        if cont_ok.sum() == 0 or disp_ok.sum() == 0:
            return {}
        c_thresh = np.quantile(m.contrast[cont_ok], ARCHETYPE_CONTRAST_Q)
        d_thresh = np.quantile(m.dispersion[disp_ok], ARCHETYPE_DISPERSION_Q)
        easy = (m.contrast >= c_thresh) & (m.dispersion <= d_thresh)

    is_go_side = m.orientation < GO_BOUNDARY_DEG
    exemplars: dict[str, np.ndarray] = {}
    go_mask = easy & is_go_side
    nogo_mask = easy & (~is_go_side)
    if go_mask.sum() >= ARCHETYPE_MIN_TRIALS:
        exemplars["Go"] = Az[go_mask]
    if nogo_mask.sum() >= ARCHETYPE_MIN_TRIALS:
        exemplars["NoGo"] = Az[nogo_mask]
    return exemplars


def _archetype_similarity_exemplars(
    m: MouseData, exemplars: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    """Per-trial × per-xG × per-side × per-exemplar cosine similarity.

    Different sides may have different exemplar counts; output is padded
    with NaN to a common K_max along the last axis so that downstream
    nanmean / nanmax aggregators handle the ragged shape transparently.

    Returns
    -------
    sim : np.ndarray
        Shape (n_trials, n_xG, n_sides, K_max).
    names : list[str]
        Side names matching ``sim``'s third axis.
    """
    Az = _zscore_neurons(m.activity)
    n_trials, n_neurons, n_xG = Az.shape
    names = list(exemplars.keys())
    if not names:
        return np.empty((n_trials, n_xG, 0, 0)), names
    K_max = max(ex.shape[0] for ex in exemplars.values())
    sim = np.full((n_trials, n_xG, len(names), K_max), np.nan)
    norm_t = np.linalg.norm(Az, axis=1)                       # (n_trials, n_xG)
    for si, name in enumerate(names):
        Ek = exemplars[name]                                  # (K, n_neurons, n_xG)
        K = Ek.shape[0]
        for k in range(K):
            A = Ek[k]                                         # (n_neurons, n_xG)
            num = (Az * A[None, :, :]).sum(axis=1)            # (n_trials, n_xG)
            norm_a = np.linalg.norm(A, axis=0)                # (n_xG,)
            denom = norm_t * norm_a[None, :]
            denom = np.where(denom > 0, denom, 1.0)
            sim[..., si, k] = num / denom
    return sim, names


def _aggregate_exemplars(sim_exemplars: np.ndarray, method: str) -> np.ndarray:
    """Collapse the K-axis of an exemplar-similarity tensor.

    Parameters
    ----------
    sim_exemplars : np.ndarray
        Shape (..., K) with NaN-padding for ragged K across sides.
    method : {'expected', 'max'}
        - ``'expected'``: NaN-safe mean over exemplars (E_k variant).
        - ``'max'``     : NaN-safe max over exemplars (nearest-exemplar).

    Returns
    -------
    np.ndarray
        Same shape minus the last axis.
    """
    if method == "expected":
        return np.nanmean(sim_exemplars, axis=-1)
    if method == "max":
        return np.nanmax(sim_exemplars, axis=-1)
    raise ValueError(f"unknown aggregation method: {method!r}")


def _kappa_banerjee(R_bar: np.ndarray, p: int) -> np.ndarray:
    """Banerjee et al. 2005 closed-form approximation to the vMF concentration MLE.

    For mean-resultant length ``R_bar`` ∈ [0, 1] in p-dim space,
    ``kappa ≈ R_bar (p − R_bar^2) / (1 − R_bar^2)``. Returns the array-shape
    of ``R_bar``. The input is clipped just below 1 to avoid division
    blow-ups when the bundle is degenerately tight.
    """
    R = np.clip(np.asarray(R_bar, dtype=float), 0.0, 1.0 - 1e-6)
    return R * (p - R ** 2) / (1.0 - R ** 2)


def _log_C_p(kappa: np.ndarray, p: int) -> np.ndarray:
    """Log of the vMF normalising constant in p dimensions.

    ``log C_p(κ) = (p/2 − 1) log κ − (p/2) log(2π) − log I_{p/2−1}(κ)``.

    Implemented via ``scipy.special.ive`` (exponentially-scaled modified
    Bessel) to stay finite for large κ in high p:
        log I_ν(κ) = log ive(ν, κ) + κ.
    """
    from scipy.special import ive  # local import — scipy is a heavy optional dep
    nu = p / 2.0 - 1.0
    k = np.clip(np.asarray(kappa, dtype=float), 1e-12, None)
    log_I = np.log(ive(nu, k) + 1e-300) + k
    return nu * np.log(k) - (p / 2.0) * np.log(2.0 * np.pi) - log_I


def _archetype_similarity_vmf(
    m: MouseData, exemplars: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    """Per-trial × per-xG vMF log-likelihood for each archetype side.

    For each side and each xG bin:
      1. unit-normalise the easy-trial exemplars (each (n_neurons,) slice)
      2. fit a vMF (mean direction = normalised mean resultant; concentration
         κ via Banerjee approximation)
      3. evaluate log p(r_hat | μ_c, κ_c) at every trial × xG.

    Returns
    -------
    log_p : np.ndarray
        Shape (n_trials, n_xG, n_sides). Full log-density including the
        log-normaliser, so log-LR with unequal κ_c is well-defined.
    names : list[str]
    """
    Az = _zscore_neurons(m.activity)                          # (n_trials, n_neurons, n_xG)
    n_trials, n_neurons, n_xG = Az.shape
    names = list(exemplars.keys())
    if not names:
        return np.empty((n_trials, n_xG, 0)), names

    norm_t = np.linalg.norm(Az, axis=1, keepdims=True) + SI_EPS  # (n_trials, 1, n_xG)
    r_hat = Az / norm_t                                          # unit-norm per (trial, xG)

    log_p = np.zeros((n_trials, n_xG, len(names)))
    for si, name in enumerate(names):
        Ek = exemplars[name]                                     # (K, n_neurons, n_xG)
        K = Ek.shape[0]
        norm_k = np.linalg.norm(Ek, axis=1, keepdims=True) + SI_EPS
        Ek_hat = Ek / norm_k                                     # (K, n_neurons, n_xG)
        resultant = Ek_hat.sum(axis=0)                           # (n_neurons, n_xG)
        R_len = np.linalg.norm(resultant, axis=0)                # (n_xG,)
        R_bar = R_len / K                                        # (n_xG,)
        mu = resultant / (R_len[None, :] + SI_EPS)               # (n_neurons, n_xG)
        kappa = _kappa_banerjee(R_bar, p=n_neurons)              # (n_xG,)
        dot = np.einsum("inj,nj->ij", r_hat, mu)                 # (n_trials, n_xG)
        log_C = _log_C_p(kappa, p=n_neurons)                     # (n_xG,)
        log_p[..., si] = kappa[None, :] * dot + log_C[None, :]
    return log_p, names


def _si_per_method(
    m: MouseData,
    archetypes_proto: dict[str, np.ndarray],
    exemplars: dict[str, np.ndarray],
    method: str,
) -> np.ndarray:
    """Compute decision-aligned, bounded SI per trial × xG under one method.

    Dispatches over {'prototype', 'expected_exemplar', 'max_exemplar', 'vmf'}.
    All four returns are bounded in [-1, +1] and share the same shape, so
    they're directly comparable in model-comparison plots.

    The vMF variant uses the posterior-form transformation
    ``SI = tanh((log p_R − log p_L) / 2)``, equivalent to
    ``(P_R − P_L)/(P_R + P_L)`` under a binary class prior, which is the
    natural bounded readout under the Bayesian framing.

    Returns NaN if the corresponding side is missing from the dicts.
    """
    if method == "prototype":
        sim, names = _archetype_similarity(m, archetypes_proto)
        if "Go" not in names or "NoGo" not in names:
            return np.full((m.n_trials, m.n_xG), np.nan)
        gi, ni = names.index("Go"), names.index("NoGo")
        return _si_normalised(sim[..., gi], sim[..., ni])
    if method in ("expected_exemplar", "max_exemplar"):
        sim_ex, names = _archetype_similarity_exemplars(m, exemplars)
        if "Go" not in names or "NoGo" not in names:
            return np.full((m.n_trials, m.n_xG), np.nan)
        gi, ni = names.index("Go"), names.index("NoGo")
        agg_key = "expected" if method == "expected_exemplar" else "max"
        agg = _aggregate_exemplars(sim_ex, agg_key)
        return _si_normalised(agg[..., gi], agg[..., ni])
    if method == "vmf":
        log_p, names = _archetype_similarity_vmf(m, exemplars)
        if "Go" not in names or "NoGo" not in names:
            return np.full((m.n_trials, m.n_xG), np.nan)
        gi, ni = names.index("Go"), names.index("NoGo")
        diff = log_p[..., gi] - log_p[..., ni]
        return np.tanh(0.5 * diff)
    raise ValueError(f"unknown SI method: {method!r}")


# ---------------------------------------------------------------------------
# Per-animal framework metrics: bundle κ, trial-mean-SI spread, psychometric,
# confidence-proxy. These power the framework-test renderers below.
#
# Note: ``_per_animal_si_spread`` was previously called ``_per_animal_sv`` and
# labelled as "DDM drift variability". The label was misleading — what we
# compute is the variance, across trials at the strongest evidence level, of
# the trial-averaged SI under the chosen method. It is a *neural summary
# statistic*, not a DDM-fit parameter. Connecting it to actual DDM $s_v$
# requires fitting a DDM to behaviour (see deferred ``similarity_fit.py``).
# ---------------------------------------------------------------------------


def _per_animal_kappa(
    m: MouseData,
    exemplars: dict[str, np.ndarray],
    *,
    xG_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Time-averaged vMF concentration κ per side.

    Fits a vMF (mean direction + Banerjee κ) at each xG bin from the
    easy-trial bundle, then averages κ over xG (within ``xG_range`` if
    given, else over all bins). Empty dict if exemplars missing.
    """
    if not exemplars:
        return {}
    n_neurons = m.activity.shape[1]
    if xG_range is not None:
        xG_mask = (m.xG >= xG_range[0]) & (m.xG <= xG_range[1])
    else:
        xG_mask = np.ones_like(m.xG, dtype=bool)
    out: dict[str, float] = {}
    for name, Ek in exemplars.items():
        norm_k = np.linalg.norm(Ek, axis=1, keepdims=True) + SI_EPS
        Ek_hat = Ek / norm_k
        resultant = Ek_hat.sum(axis=0)
        R_len = np.linalg.norm(resultant, axis=0)
        R_bar = R_len / Ek.shape[0]
        kappa = _kappa_banerjee(R_bar, p=n_neurons)
        out[name] = float(np.nanmean(kappa[xG_mask]))
    return out


def _per_animal_si_spread(
    m: MouseData,
    archetypes_proto: dict[str, np.ndarray],
    exemplars: dict[str, np.ndarray],
    *,
    method: str = "expected_exemplar",
    contrast_mask: np.ndarray | None = None,
) -> float:
    """Variance, across trials at a chosen contrast bin, of the trial-averaged $SI$.

    Returns $\\mathrm{Var}_{\\mathrm{trial}}\\,\\mathbb{E}_t[SI(t)]$ for trials
    in the chosen ``contrast_mask`` (default: trials at the maximum
    $|\\text{signed\\_contrast}|$ level). Returns NaN if there are not enough
    valid trials.

    This is a **neural summary statistic** — the spread of one number per
    trial (the trial-averaged SI) across trials with similar evidence
    strength. It is *not* the DDM drift-variability parameter $s_v$, which
    would require fitting a DDM to behavioural choice + RT data. The
    framework's claim that bundle width contributes to $s_v$ would
    properly be tested by fitting a DDM and asking whether neural κ
    predicts the fitted $s_v$; this function only tests the weaker neural-
    summary claim that bundle width predicts the spread of trial-averaged
    SI across max-contrast trials.
    """
    feats = compute_features(m)
    c = feats["signed_contrast"]
    finite = np.isfinite(c)
    if finite.sum() == 0:
        return float("nan")
    if contrast_mask is None:
        max_abs = float(np.max(np.abs(c[finite])))
        contrast_mask = (np.abs(np.abs(c) - max_abs) < 1e-9) & finite
    if contrast_mask.sum() < 3:
        return float("nan")
    si = _si_per_method(m, archetypes_proto, exemplars, method=method)
    trial_mean = np.nanmean(si[contrast_mask], axis=1)
    return float(np.nanvar(trial_mean))


def _velocity_curve(
    m: MouseData,
    feature: str = "signed_contrast",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-level pre-RZ velocity mean and SEM over an arbitrary x-axis feature.

    The velocity analogue of :func:`_psychometric`. ``m.velocity`` is the
    pre-reward-zone velocity (slower → anticipatory Go behaviour; faster →
    NoGo / withhold). Useful as a continuous behavioural readout
    alongside the binary choice psychometric.

    Returns ``(levels, mean_v, sem_v, n_per_level)``.
    """
    feats = compute_features(m)
    if feature not in feats:
        raise KeyError(f"feature {feature!r} not in compute_features output")
    c = feats[feature]
    levels = np.asarray(_level_order(feature, c), dtype=float)
    mean_v = np.full(len(levels), np.nan)
    sem_v = np.full(len(levels), np.nan)
    n_per = np.zeros(len(levels), dtype=int)
    v_ok = np.isfinite(m.velocity)
    for i, lv in enumerate(levels):
        mask = (c == lv) & v_ok
        n_per[i] = int(mask.sum())
        if n_per[i] > 0:
            vals = m.velocity[mask].astype(float)
            mean_v[i] = float(np.mean(vals))
            sem_v[i] = float(np.std(vals) / np.sqrt(max(1, n_per[i])))
    return levels, mean_v, sem_v, n_per


def _psychometric(
    m: MouseData,
    feature: str = "signed_contrast",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-level P(Go) over an arbitrary x-axis feature.

    Default ``feature='signed_contrast'`` reproduces the canonical psychometric.
    Pass ``feature='true_orientation'`` for the orientation psychometric.

    Returns ``(levels, p_go, n_per_level)``. NaN at levels with no trials.
    """
    feats = compute_features(m)
    if feature not in feats:
        raise KeyError(f"feature {feature!r} not in compute_features output")
    c = feats[feature]
    levels = np.asarray(_level_order(feature, c), dtype=float)
    p_go = np.full(len(levels), np.nan)
    n_per = np.zeros(len(levels), dtype=int)
    choice_ok = np.isfinite(m.choice)
    for i, lv in enumerate(levels):
        mask = (c == lv) & choice_ok
        n_per[i] = int(mask.sum())
        if n_per[i] > 0:
            p_go[i] = float(np.mean(m.choice[mask]))
    return levels, p_go, n_per


def _decision_entropy(decision_posterior: np.ndarray) -> np.ndarray:
    """Binary entropy of the IO model's decision posterior [P(Go), P(NoGo)].

    Returns shape ``(n_trials,)``, values in $[0, \\log 2]$.
    NaN-safe: trials with non-finite posteriors return NaN.
    """
    p = np.asarray(decision_posterior, dtype=float)[:, 0]
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    pv = np.clip(p[valid], 1e-12, 1.0 - 1e-12)
    out[valid] = -(pv * np.log(pv) + (1.0 - pv) * np.log(1.0 - pv))
    return out


# ---------------------------------------------------------------------------
# Per-animal behavioural readouts: lapse rate, velocity-as-accumulator,
# SI-logistic, interrogation-protocol DDM. These bridge the framework's
# neural-side predictions to behavioural readouts that don't require RT —
# UncertaintyV1's task is interrogation-protocol (fixed 2 s viewing window
# then commit), so we can't fit a standard first-passage DDM.
# ---------------------------------------------------------------------------


def _per_animal_lapse(
    m: MouseData,
    *,
    contrast_threshold: float = 0.5,
) -> tuple[float, float]:
    """Per-animal lapse rates at extreme |signed_contrast|.

    ``lapse_Go``  = P(NoGo | signed_contrast ≥ +threshold) — fraction of
                    strong-Go trials where the animal failed to lick.
    ``lapse_NoGo``= P(Go   | signed_contrast ≤ −threshold) — fraction of
                    strong-NoGo trials where the animal incorrectly licked.

    Returns NaN where a side has no trials above threshold.
    """
    feats = compute_features(m)
    c = feats["signed_contrast"]
    valid = np.isfinite(c) & np.isfinite(m.choice)
    go_strong = (c >= contrast_threshold) & valid
    nogo_strong = (c <= -contrast_threshold) & valid
    lapse_go = float(1.0 - np.mean(m.choice[go_strong])) if go_strong.sum() > 0 else float("nan")
    lapse_nogo = float(np.mean(m.choice[nogo_strong])) if nogo_strong.sum() > 0 else float("nan")
    return lapse_go, lapse_nogo


def _per_animal_velocity_drift(
    m: MouseData,
) -> dict[str, float]:
    """Per-animal regression of pre-RZ velocity on signed_contrast.

    Treats pre-RZ velocity as a *scalar* readout of the accumulator state
    at decision time. Returns drift sensitivity (slope), bias (intercept),
    R², and residual variance (a noise / drift-variability proxy).

    The framework's interpretation: high |signed_contrast| → fast decision-
    direction-dependent velocity (Go-evidence → slowing for the lick;
    NoGo-evidence → running through). The slope captures how strongly
    cortical evidence shapes the behavioural commitment.
    """
    feats = compute_features(m)
    c = feats["signed_contrast"]
    v = m.velocity.astype(float)
    valid = np.isfinite(c) & np.isfinite(v)
    if valid.sum() < 5:
        return {"alpha": float("nan"), "bias": float("nan"),
                "r_squared": float("nan"), "resid_var": float("nan")}
    c_v = c[valid]
    v_v = v[valid]
    # Simple least-squares fit
    alpha, bias = np.polyfit(c_v, v_v, 1)
    pred = alpha * c_v + bias
    ss_res = float(np.sum((v_v - pred) ** 2))
    ss_tot = float(np.sum((v_v - v_v.mean()) ** 2))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    resid_var = float(np.var(v_v - pred))
    return {"alpha": float(alpha), "bias": float(bias),
            "r_squared": r_squared, "resid_var": resid_var}


def _per_animal_si_logistic(
    m: MouseData,
    archetypes_proto: dict[str, np.ndarray],
    exemplars: dict[str, np.ndarray],
    *,
    method: str = "expected_exemplar",
) -> dict[str, float]:
    """Logistic regression of choice on integrated SI vs on signed_contrast.

    Fits two per-animal logistic models:
      P(Go) = σ(β_SI · ⟨SI⟩_t + b_SI)         — SI-only
      P(Go) = σ(β_stim · signed_contrast + b_stim) — stimulus-only

    Returns coefficients and log-likelihood ratios. The framework's claim
    is that integrated SI should predict choice *better* than the
    stimulus alone, because SI carries within-condition trial-to-trial
    fluctuations that the stimulus label doesn't.
    """
    if "Go" not in archetypes_proto or "NoGo" not in archetypes_proto:
        return {"beta_si": float("nan"), "beta_stim": float("nan"),
                "ll_si": float("nan"), "ll_stim": float("nan"),
                "delta_ll": float("nan"), "n_trials": 0}
    feats = compute_features(m)
    c = feats["signed_contrast"]
    si = _si_per_method(m, archetypes_proto, exemplars, method=method)
    trial_mean_si = np.nanmean(si, axis=1)
    y = m.choice.astype(float)
    valid = (np.isfinite(c) & np.isfinite(trial_mean_si)
             & np.isfinite(y) & np.isin(y, [0.0, 1.0]))
    n = int(valid.sum())
    if n < 20:
        return {"beta_si": float("nan"), "beta_stim": float("nan"),
                "ll_si": float("nan"), "ll_stim": float("nan"),
                "delta_ll": float("nan"), "n_trials": n}

    def _fit_logistic(x, y_):
        # Simple IRLS via scipy.optimize; numerically stable for small datasets.
        from scipy.optimize import minimize
        def neg_ll(params):
            beta, b = params
            z = beta * x + b
            # log-sigmoid stable form
            log_p1 = -np.logaddexp(0.0, -z)
            log_p0 = -np.logaddexp(0.0, z)
            return -float(np.sum(y_ * log_p1 + (1 - y_) * log_p0))
        res = minimize(neg_ll, x0=[0.0, 0.0], method="Nelder-Mead")
        return float(res.x[0]), float(res.x[1]), -float(res.fun)

    x_si = trial_mean_si[valid]
    x_stim = c[valid]
    y_v = y[valid]
    beta_si, b_si, ll_si = _fit_logistic(x_si, y_v)
    beta_stim, b_stim, ll_stim = _fit_logistic(x_stim, y_v)
    return {"beta_si": beta_si, "beta_stim": beta_stim,
            "ll_si": ll_si, "ll_stim": ll_stim,
            "delta_ll": ll_si - ll_stim, "n_trials": n}


def _per_animal_si_logistic_xG_binned(
    m: MouseData,
    archetypes_proto: dict[str, np.ndarray],
    exemplars: dict[str, np.ndarray],
    *,
    method: str = "expected_exemplar",
    n_bins: int = 5,
) -> dict[str, float]:
    """Logistic regression of choice on **xG-binned** SI (vs signed_contrast).

    Splits the within-trial SI trajectory into ``n_bins`` segments along xG,
    takes the mean SI in each, and uses those as features in a logistic
    regression of choice. Compares LL against the stim-only baseline.

    The framework's claim — neural SI carries within-condition information
    that signed_contrast doesn't — is really about SI(t), not its trial mean.
    Trial-mean SI throws away the within-trial structure that should
    discriminate choice within a fixed contrast level. xG-binned SI keeps
    early / mid / late SI as separate features.
    """
    if "Go" not in archetypes_proto or "NoGo" not in archetypes_proto:
        return {"ll_si": float("nan"), "ll_stim": float("nan"),
                "delta_ll": float("nan"), "n_trials": 0, "n_features": 0}
    feats = compute_features(m)
    c = feats["signed_contrast"]
    si = _si_per_method(m, archetypes_proto, exemplars, method=method)
    n_xG = si.shape[1]
    edges = np.linspace(0, n_xG, n_bins + 1, dtype=int)
    si_bins = np.stack(
        [np.nanmean(si[:, edges[i]:edges[i + 1]], axis=1) for i in range(n_bins)],
        axis=1)
    y = m.choice.astype(float)
    valid = (np.isfinite(c) & np.isfinite(si_bins).all(axis=1)
             & np.isfinite(y) & np.isin(y, [0.0, 1.0]))
    n = int(valid.sum())
    if n < 30:
        return {"ll_si": float("nan"), "ll_stim": float("nan"),
                "delta_ll": float("nan"), "n_trials": n, "n_features": n_bins}

    from scipy.optimize import minimize

    def _neg_ll_multivar(params, X, y_):
        beta = params[:-1]
        b = params[-1]
        z = X @ beta + b
        log_p1 = -np.logaddexp(0.0, -z)
        log_p0 = -np.logaddexp(0.0, z)
        return -float(np.sum(y_ * log_p1 + (1 - y_) * log_p0))

    def _neg_ll_univar(params, x, y_):
        return _neg_ll_multivar(np.array([params[0], params[1]]),
                                x[:, None], y_)

    X_si = si_bins[valid]
    x_stim = c[valid]
    y_v = y[valid]
    # SI-bins fit: n_bins + 1 free parameters
    x0_si = np.zeros(n_bins + 1)
    res_si = minimize(_neg_ll_multivar, x0_si, args=(X_si, y_v),
                      method="Nelder-Mead", options={"maxiter": 5000})
    res_stim = minimize(_neg_ll_univar, [0.0, 0.0], args=(x_stim, y_v),
                        method="Nelder-Mead")
    ll_si = -float(res_si.fun)
    ll_stim = -float(res_stim.fun)
    return {"ll_si": ll_si, "ll_stim": ll_stim,
            "delta_ll": ll_si - ll_stim, "n_trials": n, "n_features": n_bins}


def _fit_interrogation_ddm(
    signed_contrast: np.ndarray,
    choice: np.ndarray,
    *,
    T: float = 2.0,
) -> dict[str, float]:
    """Per-animal interrogation-protocol DDM fit from choice + signed_contrast.

    Model: drift v_i = α · c_i + N(0, σ_v²)  (across-trial drift variability)
           X(T) | v_i = N(v_i · T, T · σ²)   (within-trial diffusion)
           Choice = sign(X(T) + bias)
           P(Go | c) = λ_R + (1 − λ_L − λ_R) · Φ((α·c + bias)·√T / √(σ² + T·σ_v²))

    With σ fixed at 1 (overall scale absorbed into α), the identifiable
    parameters are α, bias, σ_v, λ_L, λ_R. Fitted by MLE on observed
    P(Go | c). Returns NaN dict on failure.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    valid = np.isfinite(signed_contrast) & np.isin(choice, [0.0, 1.0])
    if valid.sum() < 30:
        return {k: float("nan") for k in
                ("alpha", "bias", "sigma_v", "lapse_L", "lapse_R", "ll")}
    c = signed_contrast[valid]
    y = choice[valid].astype(float)

    def neg_ll(params):
        log_alpha, bias, log_sv, logit_lL, logit_lR = params
        alpha = np.exp(log_alpha)
        sigma_v = np.exp(log_sv)
        # Lapse cap = 0.9 (was 0.5) so strongly Go-biased animals — which can
        # genuinely lapse 60%+ on NoGo trials — don't push σ_v to zero just
        # because the lapse parameter saturates. With cap 0.5 the fits collapse
        # to λ_R = 0.5 + σ_v = 0 for animals like Cb22 / Cb25.
        lapse_L = 1.0 / (1.0 + np.exp(-logit_lL)) * 0.9
        lapse_R = 1.0 / (1.0 + np.exp(-logit_lR)) * 0.9
        denom = np.sqrt(1.0 + T * sigma_v ** 2)
        z = (alpha * c + bias) * np.sqrt(T) / denom
        p_clean = norm.cdf(z)
        p_go = lapse_R + (1.0 - lapse_L - lapse_R) * p_clean
        p_go = np.clip(p_go, 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p_go) + (1 - y) * np.log(1 - p_go)))

    x0 = [np.log(1.0), 0.0, np.log(0.5), -2.0, -2.0]
    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    if not res.success and res.nit < 100:
        return {k: float("nan") for k in
                ("alpha", "bias", "sigma_v", "lapse_L", "lapse_R", "ll")}
    log_alpha, bias, log_sv, logit_lL, logit_lR = res.x
    return {
        "alpha": float(np.exp(log_alpha)),
        "bias": float(bias),
        "sigma_v": float(np.exp(log_sv)),
        "lapse_L": float(0.9 / (1.0 + np.exp(-logit_lL))),
        "lapse_R": float(0.9 / (1.0 + np.exp(-logit_lR))),
        "ll": float(-res.fun),
    }


# ---------------------------------------------------------------------------
# Renderers — all output one PNG per grouping into ``out_dir``.
# ---------------------------------------------------------------------------


def render_style_F_sim_over_xG(mice: list[MouseData], grouping: str,
                               out_dir: Path) -> Path | None:
    """Decision-aligned similarity over xG: mean (sim_Go − sim_NoGo) per
    grouping level, one panel per mouse, shaded SEM."""
    if not _is_discrete(grouping):
        return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Style F — decision-aligned similarity (sim_Go − sim_NoGo) over xG "
        f"| groups: {grouping}", fontsize=11)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats[grouping]
        levels = _level_order(grouping, c)
        if len(levels) < 2:
            ax.text(0.5, 0.5, "<2 levels", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue
        archs = _compute_archetypes(m)
        if "Go" not in archs or "NoGo" not in archs:
            ax.text(0.5, 0.5, "no archetype\n(need easy Go + NoGo trials)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7)
            ax.set_axis_off()
            continue
        sim, names = _archetype_similarity(m, archs)
        gi = names.index("Go")
        ni = names.index("NoGo")
        diff = sim[..., gi] - sim[..., ni]
        pal = _palette(levels, grouping)
        for lv in levels:
            msk = (c == lv)
            n = int(msk.sum())
            if n == 0:
                continue
            mu = diff[msk].mean(axis=0)
            sem = diff[msk].std(axis=0) / max(1.0, np.sqrt(n))
            ax.plot(m.xG, mu, color=pal[lv], lw=1.5)
            ax.fill_between(m.xG, mu - sem, mu + sem, color=pal[lv], alpha=0.2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("xG (a.u.)", fontsize=7)
        ax.set_ylabel("sim_Go − sim_NoGo", fontsize=7)
        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m)

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    union = _union_levels(mice, grouping)
    pal_union = _palette(union, grouping)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    out = out_dir / f"sim_diff_over_xG__{grouping}"
    _save_fig(fig, out)
    return out


def render_style_F_archetype_traj(mice: list[MouseData], grouping: str,
                                  out_dir: Path) -> Path | None:
    """Condition-averaged trajectories in the (sim_NoGo, sim_Go) plane.
    Subpanels per mouse; one line per grouping level."""
    if not _is_discrete(grouping):
        return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Style F — trajectory in (sim_NoGo, sim_Go) plane | groups: {grouping}",
        fontsize=11)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats[grouping]
        levels = _level_order(grouping, c)
        if len(levels) < 2:
            ax.text(0.5, 0.5, "<2 levels", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue
        archs = _compute_archetypes(m)
        if "Go" not in archs or "NoGo" not in archs:
            ax.text(0.5, 0.5, "no archetype", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_axis_off()
            continue
        sim, names = _archetype_similarity(m, archs)
        gi = names.index("Go")
        ni = names.index("NoGo")
        pal = _palette(levels, grouping)
        for lv in levels:
            msk = (c == lv)
            if msk.sum() < 3:
                continue
            mu_go = sim[msk, :, gi].mean(axis=0)
            mu_nogo = sim[msk, :, ni].mean(axis=0)
            Y = np.column_stack([mu_nogo, mu_go])
            _plot_traj_panel(ax, Y, pal[lv], str(lv), n_components=2)
        # Diagonal for visual reference
        xl, yl = ax.get_xlim(), ax.get_ylim()
        lo = min(xl[0], yl[0])
        hi = max(xl[1], yl[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
        ax.set_xlabel("sim to NoGo archetype", fontsize=7)
        ax.set_ylabel("sim to Go archetype", fontsize=7)
        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m, extra="(circle=xG start, ^=end)")

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    union = _union_levels(mice, grouping)
    pal_union = _palette(union, grouping)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    out = out_dir / f"archetype_traj__{grouping}"
    _save_fig(fig, out)
    return out


def render_style_F_overlay(mice: list[MouseData], grouping: str,
                           out_dir: Path) -> Path | None:
    """All mice overlaid in (sim_NoGo, sim_Go). Each mouse uses its own
    archetypes, so coordinates are 'within-mouse normalised similarity' —
    semantically comparable (Go-side high, NoGo-side high) even though scale
    can differ slightly between mice."""
    if not _is_discrete(grouping):
        return None
    union = _union_levels(mice, grouping)
    if len(union) < 2:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))
    fig.suptitle(
        f"Style F — archetype-similarity overlay (all mice) | groups: {grouping}",
        fontsize=10)
    pal = _palette(union, grouping)

    mouse_markers = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h"]
    mouse_marker_map = {}
    for mi, m in enumerate(mice):
        archs = _compute_archetypes(m)
        if "Go" not in archs or "NoGo" not in archs:
            continue
        marker = mouse_markers[mi % len(mouse_markers)]
        mouse_marker_map[m.tag] = marker
        sim, names = _archetype_similarity(m, archs)
        gi = names.index("Go")
        ni = names.index("NoGo")
        feats = compute_features(m)
        c = feats[grouping]
        for lv in union:
            msk = (c == lv)
            if msk.sum() < 3:
                continue
            mu_go = sim[msk, :, gi].mean(axis=0)
            mu_nogo = sim[msk, :, ni].mean(axis=0)
            ax.plot(mu_nogo, mu_go, color=pal[lv], lw=1.0, alpha=0.6)
            ax.scatter(mu_nogo[0], mu_go[0], color=pal[lv], s=18,
                       marker=marker)
            ax.scatter(mu_nogo[-1], mu_go[-1], color=pal[lv], s=30,
                       marker=marker, edgecolors="black", linewidths=0.5)

    if not mouse_marker_map:
        plt.close(fig)
        return None
    xl, yl = ax.get_xlim(), ax.get_ylim()
    lo = min(xl[0], yl[0])
    hi = max(xl[1], yl[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("sim to NoGo archetype", fontsize=9)
    ax.set_ylabel("sim to Go archetype", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.10, right=0.72, top=0.92, bottom=0.10)
    mouse_handles = [Line2D([0], [0], marker=mk, color="k", lw=0, markersize=7,
                            label=tag)
                     for tag, mk in mouse_marker_map.items()]
    mouse_leg = fig.legend(handles=mouse_handles, loc="upper left",
                           bbox_to_anchor=(0.74, 0.92), frameon=False,
                           fontsize=8, title="mouse", title_fontsize=8)
    fig.add_artist(mouse_leg)
    lvl_handles = [Line2D([0], [0], marker="o", color=pal[lv], lw=0,
                          markersize=7, label=str(lv)) for lv in union]
    fig.legend(handles=lvl_handles, loc="lower left",
               bbox_to_anchor=(0.74, 0.06), frameon=False,
               fontsize=8, title=grouping, title_fontsize=8)
    out = out_dir / f"archetype_overlay__{grouping}"
    _save_fig(fig, out)
    return out


def render_style_F_si_normalised_over_xG(
    mice: list[MouseData], grouping: str, out_dir: Path, *,
    method: str = "prototype",
) -> Path | None:
    """Decision-aligned **normalised** SI over xG under one aggregation method.

    Mirrors :func:`render_style_F_sim_over_xG` (which plots the unnormalised
    difference s_Go − s_NoGo) but plots the framework's bounded SI =
    (s_Go − s_NoGo) / (s_Go + s_NoGo) under the Wiener positivity map.
    Parameterised by ``method`` ∈ ``SI_METHODS`` so the four variants
    (prototype / expected / max exemplar / vMF) share the same renderer.
    """
    if not _is_discrete(grouping):
        return None
    if method not in SI_METHODS:
        raise ValueError(f"unknown SI method: {method!r}")
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Style F — normalised SI over xG · method = {SI_METHOD_LABELS[method]} "
        f"| groups: {grouping}", fontsize=11)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats[grouping]
        levels = _level_order(grouping, c)
        if len(levels) < 2:
            ax.text(0.5, 0.5, "<2 levels", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue
        proto = _compute_archetypes(m)
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            ax.text(0.5, 0.5, "no archetype\n(need easy Go + NoGo trials)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7)
            ax.set_axis_off()
            continue
        try:
            si = _si_per_method(m, proto, ex, method=method)
        except Exception as e:  # pragma: no cover — surface a panel error
            ax.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_axis_off()
            continue
        pal = _palette(levels, grouping)
        for lv in levels:
            msk = (c == lv)
            n = int(msk.sum())
            if n == 0:
                continue
            mu = np.nanmean(si[msk], axis=0)
            sd = np.nanstd(si[msk], axis=0)
            sem = sd / max(1.0, np.sqrt(n))
            ax.plot(m.xG, mu, color=pal[lv], lw=1.5)
            ax.fill_between(m.xG, mu - sem, mu + sem, color=pal[lv], alpha=0.2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("xG (a.u.)", fontsize=7)
        ax.set_ylabel("normalised SI", fontsize=7)
        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m)

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    union = _union_levels(mice, grouping)
    pal_union = _palette(union, grouping)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    out = out_dir / f"SI_normalised__{method}__{grouping}"
    _save_fig(fig, out)
    return out


def render_style_F_variants_compare(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Overlay the four SI variants on one panel per mouse, at the highest
    |signed_contrast| level present in each mouse.

    The framework's empirical prediction is that the four variants (prototype,
    expected exemplar, max exemplar, vMF) diverge on broad-bundle trials —
    they collapse to the same trace when bundles are tight. A panel per
    mouse where the four lines are clearly separated is evidence for
    non-trivial bundle width.
    """
    # Hard-code signed_contrast: variant comparison is most diagnostic
    # at the strongest evidence levels, and we want a single common grouping
    # to keep the panels readable.
    if grouping != "signed_contrast":
        return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    fig.suptitle(
        "Style F — SI variants compared at max |signed_contrast| per mouse",
        fontsize=11)
    method_colors = {
        "prototype": "#1f77b4",
        "expected_exemplar": "#2ca02c",
        "max_exemplar": "#ff7f0e",
        "vmf": "#d62728",
    }
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats["signed_contrast"]
        finite = np.isfinite(c)
        if finite.sum() == 0:
            ax.set_axis_off()
            continue
        # Pick the level with maximum absolute signed contrast.
        unique_lv = np.unique(c[finite])
        if len(unique_lv) == 0:
            ax.set_axis_off()
            continue
        lv = unique_lv[np.argmax(np.abs(unique_lv))]
        msk = (c == lv)
        if msk.sum() < 3:
            ax.text(0.5, 0.5, "<3 trials at max |contrast|",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7)
            ax.set_axis_off()
            continue
        proto = _compute_archetypes(m)
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            ax.set_axis_off()
            continue
        for method in SI_METHODS:
            try:
                si = _si_per_method(m, proto, ex, method=method)
            except Exception:  # pragma: no cover
                continue
            mu = np.nanmean(si[msk], axis=0)
            # Per-method per-mouse rescaling so the four method traces share
            # a comparable visual scale within each panel. Different methods
            # produce SI on different absolute scales (vMF in particular
            # tends to saturate); this lets shape comparison dominate.
            finite = np.isfinite(mu)
            scale = (float(np.max(np.abs(mu[finite])))
                     if finite.any() else 0.0)
            mu_plot = mu / scale if scale > 0 else mu
            ax.plot(m.xG, mu_plot, color=method_colors[method], lw=1.5,
                    label=SI_METHOD_LABELS[method])
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("xG (a.u.)", fontsize=7)
        ax.set_ylabel("SI (per-method scaled to own |max|)", fontsize=7)
        ax.set_title(f"{m.tag}  signed_contrast = {lv:g}  n = {int(msk.sum())}",
                     fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08,
                        wspace=0.28, hspace=0.40)
    # Single legend across the figure.
    handles = [Line2D([0], [0], color=method_colors[m_], lw=1.5,
                      label=SI_METHOD_LABELS[m_]) for m_ in SI_METHODS]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=len(SI_METHODS),
               frameon=False, fontsize=8)
    out = out_dir / "SI_variants_compare__signed_contrast"
    _save_fig(fig, out)
    return out


def render_style_F_single_trial_scatter(
    mice: list[MouseData], grouping: str, out_dir: Path,
    *, xG_target: float = 1.0,
) -> Path | None:
    """Single-trial scatter in (sim_NoGo, sim_Go) at a fixed xG, by signed_contrast.

    Complements ``render_style_F_archetype_traj`` (which shows
    condition-averaged trajectories): this shows the actual trial cloud per
    condition, making *bundle width* visible as cloud thickness. The
    framework predicts that wider clouds → higher trial-mean-SI spread (and, if
    that translates to behaviour, eventually to DDM $s_v$).
    """
    if grouping != "signed_contrast":
        return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Style F — single-trial scatter (sim_NoGo, sim_Go) at xG≈{xG_target:g} "
        f"| groups: {grouping}", fontsize=11)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats[grouping]
        levels = _level_order(grouping, c)
        if len(levels) < 2:
            ax.set_axis_off()
            continue
        archs = _compute_archetypes(m)
        if "Go" not in archs or "NoGo" not in archs:
            ax.set_axis_off()
            continue
        sim, names = _archetype_similarity(m, archs)
        gi, ni = names.index("Go"), names.index("NoGo")
        idx = int(np.argmin(np.abs(m.xG - xG_target)))
        pal = _palette(levels, grouping)
        for lv in levels:
            msk = (c == lv)
            if msk.sum() == 0:
                continue
            ax.scatter(sim[msk, idx, ni], sim[msk, idx, gi],
                       color=pal[lv], s=10, alpha=0.45, edgecolors="none")
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        lo = min(xl[0], yl[0])
        hi = max(xl[1], yl[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("sim to NoGo", fontsize=7)
        ax.set_ylabel("sim to Go", fontsize=7)
        ax.set_title(f"{m.tag}  xG={m.xG[idx]:.2f}  n={int((c == c).sum())}",
                     fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    union = _union_levels(mice, grouping)
    pal_union = _palette(union, grouping)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    out = out_dir / f"single_trial_scatter__{grouping}__xG{xG_target:g}"
    _save_fig(fig, out)
    return out


def render_style_F_kappa_vs_si_spread(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Per-animal bundle concentration $\\kappa$ vs the variance, across
    trials at max $|\\text{signed\\_contrast}|$, of the trial-averaged SI
    under each SI aggregation method (2×2 grid).

    The framework's claim under test here is the **neural** version of the
    bundle-width prediction: tight bundles (high κ) should produce
    consistent trial-averaged SI across max-contrast trials → low variance.
    $\\kappa$ is a bundle property (constant across panels); the SI-spread
    measure depends on which SI variant is used.

    Note: this is *not* a test of DDM drift-variability $s_v$. DDM
    $s_v$ requires fitting a DDM to behaviour and asking whether the
    bundle κ predicts the fit. See the deferred ``similarity_fit.py``.
    """
    if grouping != "signed_contrast":
        return None
    # κ doesn't depend on method — compute it once per animal.
    animals = []
    for m in mice:
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in ex or "NoGo" not in ex:
            continue
        kap = _per_animal_kappa(m, ex)
        if not kap:
            continue
        kap_avg = 0.5 * (kap["Go"] + kap["NoGo"])
        animals.append((m, kap_avg))
    if len(animals) < 2:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), squeeze=False)
    fig.suptitle(
        "Style F — bundle κ vs trial-mean-SI spread  "
        "$\\mathrm{Var}_{\\mathrm{trial}}\\,\\mathbb{E}_t[SI]$  at max |signed_contrast|"
        "  (one panel per SI aggregation method)", fontsize=10)
    for ax, method in zip(axes.ravel(), SI_METHODS):
        rows = []
        for m, kap_avg in animals:
            proto = _compute_archetypes(m)
            ex = _compute_archetypes_exemplars(m)
            spread = _per_animal_si_spread(m, proto, ex, method=method)
            if not np.isfinite(spread):
                continue
            rows.append((m.tag, kap_avg, spread))
        if len(rows) < 2:
            ax.set_axis_off()
            continue
        tags = [r[0] for r in rows]
        kaps = np.array([r[1] for r in rows])
        spreads = np.array([r[2] for r in rows])
        ax.scatter(kaps, spreads, s=70, c="steelblue", edgecolors="black",
                   zorder=3)
        for tag, k, s in zip(tags, kaps, spreads):
            ax.annotate(tag, (k, s), fontsize=8, xytext=(5, 5),
                        textcoords="offset points")
        title = SI_METHOD_LABELS[method]
        if len(kaps) >= 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(kaps, spreads)
            title += f"\nPearson r = {r:.2f},  p = {p:.3f},  n = {len(kaps)}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("bundle κ  (Banerjee, sides averaged)", fontsize=8)
        ax.set_ylabel(r"trial-mean-SI spread $\mathrm{Var}_{\mathrm{trial}}\,\mathbb{E}_t[SI]$",
                      fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_dir / "kappa_vs_si_spread"
    _save_fig(fig, out)
    return out


def render_style_F_behaviour_overlay(
    mice: list[MouseData], grouping: str, out_dir: Path,
    *, behaviour: str = "choice",
) -> Path | None:
    """Per-mouse: a behavioural summary curve overlaid with mean neural SI
    under all four aggregation variants, x-axis = ``grouping`` feature.

    ``behaviour`` selects the primary-axis summary:
      - ``"choice"``   : P(Go) — the standard psychometric, bounded in [0, 1].
      - ``"velocity"`` : mean pre-RZ velocity ± SEM — a continuous behavioural
                         readout (anticipatory slowing on Go decisions).

    Each SI variant is rescaled to its own per-mouse max absolute value
    before overlay so the four method lines share a comparable y-scale on
    the twin axis. Output filename includes both ``behaviour`` and
    ``grouping`` so files don't collide.
    """
    if behaviour not in ("choice", "velocity"):
        raise ValueError(f"unknown behaviour: {behaviour!r}")
    # Quick sanity check that the feature exists for at least one mouse.
    if mice:
        sample_feats = compute_features(mice[0])
        if grouping not in sample_feats:
            return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    # Velocity is anti-correlated with Go-evidence (low velocity → confident Go
    # via anticipatory slowing). Negate it so both axes point in the same
    # decision direction as SI — high values = Go-evidence — and the visual
    # comparison with SI variants reads correctly.
    velocity_flipped = (behaviour == "velocity")
    behaviour_label = (
        "P(Go)" if behaviour == "choice"
        else "−(pre-RZ velocity)  (Go-evidence proxy)"
    )
    fig.suptitle(
        f"Style F — {behaviour_label} ↔ neural SI overlay (all variants, "
        f"each method scaled to own per-mouse max) | x = {grouping}",
        fontsize=11)
    method_colors = {
        "prototype": "#1f77b4",
        "expected_exemplar": "#2ca02c",
        "max_exemplar": "#ff7f0e",
        "vmf": "#d62728",
    }
    for ax, m in zip(axes.ravel(), mice):
        if behaviour == "choice":
            levels, primary_vals, n_per = _psychometric(m, feature=grouping)
            primary_sem = None
        else:
            levels, primary_vals, primary_sem, n_per = _velocity_curve(
                m, feature=grouping)
            if velocity_flipped:
                # Flip sign so high y = Go-evidence (same orientation as SI).
                primary_vals = -primary_vals
                # SEM is symmetric so its sign doesn't matter; keep magnitude.
        if (n_per > 0).sum() < 2:
            ax.set_axis_off()
            continue
        proto = _compute_archetypes(m)
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            ax.set_axis_off()
            continue
        feats = compute_features(m)
        c = feats[grouping]
        ok = (n_per > 0)
        # Primary axis: P(Go) [0, 1] or velocity (autoscale).
        ax.plot(levels[ok], primary_vals[ok], "o-", color="black",
                label=behaviour_label, lw=1.6, zorder=3)
        if primary_sem is not None:
            ax.fill_between(
                levels[ok],
                primary_vals[ok] - primary_sem[ok],
                primary_vals[ok] + primary_sem[ok],
                color="black", alpha=0.15, zorder=2)
        if behaviour == "choice":
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", lw=0.5, alpha=0.3, ls=":")
        ax.set_ylabel(behaviour_label, fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
        # Reference vertical: contrast → 0; abs_from_go → boundary at 45°
        if grouping == "abs_from_go":
            ax.axvline(GO_BOUNDARY_DEG, color="gray", lw=0.5, alpha=0.3, ls=":")
        else:
            ax.axvline(0.0, color="gray", lw=0.5, alpha=0.3, ls=":")
        # All four SI variants on twin axis, each scaled to own per-mouse max.
        ax2 = ax.twinx()
        for method in SI_METHODS:
            try:
                si = _si_per_method(m, proto, ex, method=method)
            except Exception:  # pragma: no cover
                continue
            mean_si_per_lv = np.full(len(levels), np.nan)
            for i, lv in enumerate(levels):
                mask = (c == lv)
                if mask.sum() > 0:
                    mean_si_per_lv[i] = float(np.nanmean(si[mask]))
            finite = np.isfinite(mean_si_per_lv)
            scale = (float(np.max(np.abs(mean_si_per_lv[finite])))
                     if finite.any() else 0.0)
            if scale > 0:
                mean_si_per_lv = mean_si_per_lv / scale
            ax2.plot(levels[ok], mean_si_per_lv[ok], "s-",
                     color=method_colors[method], lw=1.2, alpha=0.85,
                     markersize=4, label=SI_METHOD_LABELS[method])
        ax2.axhline(0, color="gray", lw=0.5, alpha=0.3, ls=":")
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_ylabel("mean SI (per-method scaled to own |max|)", fontsize=7)
        ax2.tick_params(axis="y", labelsize=6)
        ax.set_xlabel(grouping, fontsize=7)
        ax.set_title(m.tag, fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    fig.subplots_adjust(left=0.06, right=0.95, top=0.90, bottom=0.12,
                        wspace=0.55, hspace=0.45)
    handles = [Line2D([0], [0], color="black", lw=1.6, marker="o",
                      label=behaviour_label)]
    handles += [Line2D([0], [0], color=method_colors[m_], lw=1.2,
                       marker="s", label=SI_METHOD_LABELS[m_])
                for m_ in SI_METHODS]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(SI_METHODS) + 1,
               frameon=False, fontsize=8)
    out = out_dir / f"behaviour_overlay__{behaviour}__{grouping}"
    _save_fig(fig, out)
    return out


def render_style_F_psychometric_overlay(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Legacy thin wrapper: P(Go) overlay (behaviour='choice').

    Preferred entry point is :func:`render_style_F_behaviour_overlay` which
    also supports ``behaviour='velocity'``. Kept for backward compatibility.

    ``grouping`` is interpreted as the x-axis feature key from
    ``compute_features``. The two psychometric-relevant choices in
    UncertaintyV1 are ``"signed_contrast"`` (the standard contrast
    psychometric) and ``"abs_from_go"`` (the orientation-distance-from-Go
    psychometric); other features are accepted but may not be informative.

    Each SI variant is rescaled to its own per-mouse max absolute value
    before overlay so that the four method lines share a comparable y-scale
    on the twin axis. This is purely a visualisation choice; absolute
    magnitudes per method are still recoverable from the
    ``SI_normalised__{method}`` figures.

    Returns None if the grouping feature is not present in compute_features.
    """
    return render_style_F_behaviour_overlay(mice, grouping, out_dir,
                                            behaviour="choice")


def render_style_F_si_vs_confidence(
    mice: list[MouseData], grouping: str, out_dir: Path,
    *, method: str = "expected_exemplar", proxy: str = "entropy",
) -> Path | None:
    """Per-mouse: trial-mean $SI$ vs a behavioural / model-derived confidence proxy.

    ``proxy`` ∈
      - ``"entropy"``  : binary entropy of the IO model's ``decision_posterior``
                         ([P(Go), P(NoGo)]); high entropy = uncertain.
      - ``"velocity"`` : pre-reward-zone velocity; low → confident Go decision
                         (anticipatory slowing), high → confident NoGo.

    The framework predicts $|SI|$ should be monotonically related to
    confidence — high $|SI|$ trials should sit at the confident end of the
    proxy.
    """
    if grouping != "signed_contrast":
        return None
    if proxy not in ("entropy", "velocity"):
        raise ValueError(f"unknown confidence proxy: {proxy!r}")
    proxy_label = {
        "entropy": "IO decision-posterior entropy (high → uncertain)",
        "velocity": "pre-RZ velocity (low → confident Go, high → confident NoGo)",
    }[proxy]
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle(
        f"Style F — trial-mean SI vs {proxy} confidence proxy  "
        f"(method = {SI_METHOD_LABELS[method]})", fontsize=10)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats["signed_contrast"]
        proto = _compute_archetypes(m)
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            ax.set_axis_off()
            continue
        si = _si_per_method(m, proto, ex, method=method)
        trial_mean_si = np.nanmean(si, axis=1)
        if proxy == "entropy":
            y = _decision_entropy(m.decision_posterior)
        else:
            y = m.velocity.astype(float)
        valid = np.isfinite(trial_mean_si) & np.isfinite(y) & np.isfinite(c)
        if valid.sum() < 5:
            ax.set_axis_off()
            continue
        levels = _level_order("signed_contrast", c)
        pal = _palette(levels, "signed_contrast")
        for lv in levels:
            msk = (c == lv) & valid
            if msk.sum() == 0:
                continue
            ax.scatter(trial_mean_si[msk], y[msk],
                       color=pal[lv], s=8, alpha=0.4, edgecolors="none")
        ax.axvline(0, color="gray", lw=0.5, alpha=0.4, ls=":")
        ax.set_xlabel("mean SI per trial", fontsize=7)
        ax.set_ylabel(proxy, fontsize=7)
        ax.set_title(f"{m.tag}  n={int(valid.sum())}", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    union = _union_levels(mice, grouping)
    pal_union = _palette(union, grouping)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    fig.text(0.5, 0.015, proxy_label, ha="center", fontsize=7, style="italic")
    out = out_dir / f"si_vs_confidence__{proxy}__{method}__{grouping}"
    _save_fig(fig, out)
    return out


def render_style_F_kappa_vs_lapse(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Per-animal bundle κ vs behavioural lapse rate.

    The framework predicts that broad bundles (low κ) → more frequent
    misclassification on max-contrast trials → higher lapse rate. This is
    the cleanest *behavioural* test of bundle width that doesn't require
    DDM machinery. Two panels: Go-side lapse (P(NoGo | strong Go)) and
    NoGo-side lapse (P(Go | strong NoGo)).
    """
    if grouping != "signed_contrast":
        return None
    rows = []
    for m in mice:
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in ex or "NoGo" not in ex:
            continue
        kap = _per_animal_kappa(m, ex)
        if not kap:
            continue
        kap_avg = 0.5 * (kap["Go"] + kap["NoGo"])
        lapse_go, lapse_nogo = _per_animal_lapse(m)
        rows.append((m.tag, kap_avg, lapse_go, lapse_nogo))
    if len(rows) < 2:
        return None
    tags = [r[0] for r in rows]
    kaps = np.array([r[1] for r in rows])
    lg = np.array([r[2] for r in rows])
    ln = np.array([r[3] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), squeeze=False)
    fig.suptitle("Style F — bundle κ vs behavioural lapse rate "
                 "(framework predicts negative correlation)", fontsize=10)
    for ax, lapses, side, color in [
        (axes[0, 0], lg, "Go-side lapse  P(NoGo | strong Go)", "#d62728"),
        (axes[0, 1], ln, "NoGo-side lapse  P(Go | strong NoGo)", "#1f77b4"),
    ]:
        valid = np.isfinite(lapses)
        ax.scatter(kaps[valid], lapses[valid], s=70, c=color,
                   edgecolors="black", zorder=3)
        for tag, k, lp in zip(np.array(tags)[valid],
                              kaps[valid], lapses[valid]):
            ax.annotate(tag, (k, lp), fontsize=8, xytext=(5, 5),
                        textcoords="offset points")
        if valid.sum() >= 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(kaps[valid], lapses[valid])
            ax.set_title(f"{side}\nPearson r = {r:.2f}, p = {p:.3f}, "
                         f"n = {int(valid.sum())}", fontsize=9)
        else:
            ax.set_title(side, fontsize=9)
        ax.set_xlabel("bundle κ  (Banerjee, sides averaged)", fontsize=9)
        ax.set_ylabel("lapse rate", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0, color="gray", lw=0.4, ls=":", alpha=0.4)
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "kappa_vs_lapse"
    _save_fig(fig, out)
    return out


def render_style_F_velocity_accumulator(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Per-animal velocity-as-accumulator: pre-RZ velocity vs signed_contrast.

    Treats pre-RZ velocity as a scalar readout of the accumulator state at
    decision time. Per animal: scatter of velocity vs signed_contrast,
    regression line, R². The slope is a behavioural drift sensitivity,
    the residual variance an across-trial noise / drift-variability
    proxy. Title reports the per-animal fit parameters.
    """
    if grouping != "signed_contrast":
        return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle("Style F — pre-RZ velocity as scalar accumulator readout",
                 fontsize=11)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats["signed_contrast"]
        v = m.velocity.astype(float)
        valid = np.isfinite(c) & np.isfinite(v)
        if valid.sum() < 5:
            ax.set_axis_off()
            continue
        fit = _per_animal_velocity_drift(m)
        levels = _level_order("signed_contrast", c)
        pal = _palette(levels, "signed_contrast")
        for lv in levels:
            msk = (c == lv) & valid
            if msk.sum() == 0:
                continue
            ax.scatter(c[msk], v[msk], color=pal[lv], s=10,
                       alpha=0.5, edgecolors="none")
        if np.isfinite(fit["alpha"]):
            c_line = np.array([c[valid].min(), c[valid].max()])
            ax.plot(c_line, fit["alpha"] * c_line + fit["bias"],
                    color="black", lw=1.2)
        ax.set_xlabel("signed_contrast", fontsize=8)
        ax.set_ylabel("pre-RZ velocity", fontsize=8)
        ax.set_title(f"{m.tag}  α={fit['alpha']:.2f}  bias={fit['bias']:.2f}  "
                     f"R²={fit['r_squared']:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "velocity_accumulator__signed_contrast"
    _save_fig(fig, out)
    return out


def render_style_F_si_choice_logistic(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Per-animal logistic prediction of choice: integrated SI vs signed_contrast.

    For each animal and each SI variant, fit P(Go) = σ(β·x + b) twice:
    once with x = trial-mean SI, once with x = signed_contrast. Plot
    per-animal log-likelihood difference (LL_SI − LL_stim). A positive
    difference means neural SI predicts choice better than the stimulus
    label alone — the framework's prediction, since SI carries within-
    condition trial-to-trial variation that signed_contrast does not.
    """
    if grouping != "signed_contrast":
        return None
    rows = []
    for m in mice:
        proto = _compute_archetypes(m)
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            continue
        per_method = {}
        for method in SI_METHODS:
            fit = _per_animal_si_logistic(m, proto, ex, method=method)
            per_method[method] = fit
        rows.append((m.tag, per_method))
    if not rows:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    fig.suptitle("Style F — logistic choice prediction:  "
                 "LL(SI) − LL(signed_contrast)  per animal × method",
                 fontsize=10)
    method_colors = {
        "prototype": "#1f77b4",
        "expected_exemplar": "#2ca02c",
        "max_exemplar": "#ff7f0e",
        "vmf": "#d62728",
    }
    x_positions = np.arange(len(rows))
    bar_width = 0.18
    for mi, method in enumerate(SI_METHODS):
        deltas = [r[1][method]["delta_ll"] for r in rows]
        ax.bar(x_positions + (mi - 1.5) * bar_width, deltas,
               width=bar_width, color=method_colors[method],
               label=SI_METHOD_LABELS[method])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel(r"$\Delta$ log-likelihood  (SI − stim)", fontsize=9)
    ax.set_xlabel("animal", fontsize=9)
    ax.set_title("positive bar → SI predicts choice better than signed_contrast",
                 fontsize=8)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_dir / "si_choice_logistic"
    _save_fig(fig, out)
    return out


def render_style_F_si_choice_logistic_xG_binned(
    mice: list[MouseData], grouping: str, out_dir: Path,
    *, n_bins: int = 5,
) -> Path | None:
    """xG-binned version of ``render_style_F_si_choice_logistic``.

    Uses ``n_bins`` xG-binned SI features per trial (instead of trial-mean
    SI). The framework's prediction is properly about SI(t) — splitting
    the trial into bins preserves the within-trial information that the
    trial-mean version throws away.
    """
    if grouping != "signed_contrast":
        return None
    rows = []
    for m in mice:
        proto = _compute_archetypes(m)
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            continue
        per_method = {}
        for method in SI_METHODS:
            fit = _per_animal_si_logistic_xG_binned(
                m, proto, ex, method=method, n_bins=n_bins)
            per_method[method] = fit
        rows.append((m.tag, per_method))
    if not rows:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    fig.suptitle(f"Style F — xG-binned logistic choice prediction:  "
                 f"LL(SI bins) − LL(signed_contrast)  per animal × method  "
                 f"(n_bins = {n_bins})", fontsize=10)
    method_colors = {
        "prototype": "#1f77b4",
        "expected_exemplar": "#2ca02c",
        "max_exemplar": "#ff7f0e",
        "vmf": "#d62728",
    }
    x_positions = np.arange(len(rows))
    bar_width = 0.18
    for mi, method in enumerate(SI_METHODS):
        deltas = [r[1][method]["delta_ll"] for r in rows]
        ax.bar(x_positions + (mi - 1.5) * bar_width, deltas,
               width=bar_width, color=method_colors[method],
               label=SI_METHOD_LABELS[method])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel(r"$\Delta$ log-likelihood  (SI xG-bins − stim)", fontsize=9)
    ax.set_xlabel("animal", fontsize=9)
    ax.set_title("positive bar → xG-binned SI predicts choice better than signed_contrast",
                 fontsize=8)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_dir / "si_choice_logistic_xG_binned"
    _save_fig(fig, out)
    return out


def render_style_F_ddm_interrogation(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Per-animal interrogation-protocol DDM fit from choice + signed_contrast.

    Fits {α drift, bias, σ_v drift variability, λ_L, λ_R} per animal.
    Plots observed psychometric vs the fitted curve, with the recovered
    parameters in the panel title. This gives behavioural σ_v estimates
    that can later be correlated with neural κ — the proper DDM-side
    test of the framework's bundle-width prediction.
    """
    if grouping != "signed_contrast":
        return None
    rows_, cols_ = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows_, cols_, figsize=(3.6 * cols_, 3.2 * rows_),
                             squeeze=False)
    fig.suptitle("Style F — interrogation-protocol DDM fits "
                 "(per-animal {α, bias, σ_v, λ_L, λ_R})", fontsize=10)
    fits_by_animal = {}
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats["signed_contrast"]
        y = m.choice.astype(float)
        valid = np.isfinite(c) & np.isin(y, [0.0, 1.0])
        if valid.sum() < 30:
            ax.set_axis_off()
            continue
        fit = _fit_interrogation_ddm(c[valid], y[valid])
        fits_by_animal[m.tag] = fit
        # Observed psychometric
        levels, p_go, n_per = _psychometric(m)
        ok = (n_per > 0)
        ax.plot(levels[ok], p_go[ok], "o", color="black",
                markersize=6, label="observed")
        # Fitted curve
        if np.isfinite(fit["alpha"]):
            from scipy.stats import norm
            c_grid = np.linspace(levels.min(), levels.max(), 100)
            denom = np.sqrt(1.0 + 2.0 * fit["sigma_v"] ** 2)
            z = (fit["alpha"] * c_grid + fit["bias"]) * np.sqrt(2.0) / denom
            p_fit = fit["lapse_R"] + (1.0 - fit["lapse_L"] - fit["lapse_R"]) \
                * norm.cdf(z)
            ax.plot(c_grid, p_fit, color="darkorange", lw=1.4, label="DDM fit")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", lw=0.4, ls=":", alpha=0.5)
        ax.axvline(0.0, color="gray", lw=0.4, ls=":", alpha=0.5)
        ax.set_xlabel("signed_contrast", fontsize=7)
        ax.set_ylabel("P(Go)", fontsize=8)
        ax.set_title(
            f"{m.tag}  α={fit['alpha']:.2f}  σ_v={fit['sigma_v']:.2f}  "
            f"λ_L={fit['lapse_L']:.2f}  λ_R={fit['lapse_R']:.2f}",
            fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "ddm_interrogation_fits"
    _save_fig(fig, out)
    return out


def render_style_F_kappa_vs_ddm_sigma_v(
    mice: list[MouseData], grouping: str, out_dir: Path,
) -> Path | None:
    """Per-animal bundle κ vs *behavioural* σ_v from the interrogation-DDM fit.

    This is the proper test of the framework's bundle-width → DDM-$s_v$
    claim — the one the renamed ``kappa_vs_si_spread`` does *not* test.
    """
    if grouping != "signed_contrast":
        return None
    rows = []
    for m in mice:
        ex = _compute_archetypes_exemplars(m)
        if "Go" not in ex or "NoGo" not in ex:
            continue
        kap = _per_animal_kappa(m, ex)
        if not kap:
            continue
        kap_avg = 0.5 * (kap["Go"] + kap["NoGo"])
        feats = compute_features(m)
        c = feats["signed_contrast"]
        y = m.choice.astype(float)
        valid = np.isfinite(c) & np.isin(y, [0.0, 1.0])
        if valid.sum() < 30:
            continue
        fit = _fit_interrogation_ddm(c[valid], y[valid])
        if not np.isfinite(fit["sigma_v"]):
            continue
        rows.append((m.tag, kap_avg, fit["sigma_v"]))
    if len(rows) < 2:
        return None
    tags = [r[0] for r in rows]
    kaps = np.array([r[1] for r in rows])
    svs = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    fig.suptitle("Style F — bundle κ vs behavioural σ_v (interrogation-DDM fit)",
                 fontsize=10)
    ax.scatter(kaps, svs, s=80, c="steelblue", edgecolors="black", zorder=3)
    for tag, k, sv in zip(tags, kaps, svs):
        ax.annotate(tag, (k, sv), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    if len(rows) >= 3:
        from scipy.stats import pearsonr
        r, p = pearsonr(kaps, svs)
        ax.set_title(f"Pearson r = {r:.2f},  p = {p:.3f},  n = {len(rows)}",
                     fontsize=9)
    ax.set_xlabel("bundle κ  (Banerjee, sides averaged)", fontsize=9)
    ax.set_ylabel(r"DDM σ_v  (from interrogation-protocol fit)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "kappa_vs_ddm_sigma_v"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Standalone Style-F orchestrator
# ---------------------------------------------------------------------------


def run_similarity_sweep(
    data_path: Path = DATA_PATH,
    fig_root: Path | None = None,
    *,
    smoke: bool = False,
) -> dict:
    """Style-F-only orchestrator. Loads mice, runs every similarity renderer
    against every applicable grouping, writes figures + ``_meta.json``.
    """
    if fig_root is None:
        fig_root = (Path(__file__).resolve().parent
                    / "figures" / "similarity_framework")
    fig_root = Path(fig_root)
    fig_root.mkdir(parents=True, exist_ok=True)
    sF = fig_root / "F_archetype_similarity"
    sF.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mice_all = load_all_mice(data_path)
    mice, skipped = _filter_mice(mice_all)
    if not mice:
        raise RuntimeError("no mice survive filtering; check thresholds / data")
    if smoke:
        mice = mice[:1]
        f_groups = ["choice"]
    else:
        f_groups = GROUPINGS_DISCRETE

    meta = {
        "data_path": str(data_path),
        "fig_root": str(fig_root),
        "n_mice_loaded": len(mice_all),
        "n_mice_used": len(mice),
        "mice": [{"tag": m.tag, "n_trials": int(m.n_trials),
                  "n_neurons": int(m.n_neurons), "n_xG": int(m.n_xG)}
                 for m in mice],
        "skipped": skipped,
        "smoke": smoke,
        "produced": [],
    }

    def _record(label, grouping, path_or_err, secs):
        entry = {"label": label, "grouping": grouping, "secs": secs}
        if isinstance(path_or_err, Exception):
            entry["error"] = repr(path_or_err)
        elif path_or_err is None:
            entry["skipped"] = "renderer returned None"
        else:
            entry["path"] = str(path_or_err)
        meta["produced"].append(entry)

    for g in f_groups:
        # Legacy Style F renderers (unnormalised diff + 2D trajectories)
        for renderer, label in [
            (render_style_F_sim_over_xG, "sim_diff_over_xG"),
            (render_style_F_archetype_traj, "archetype_traj"),
            (render_style_F_overlay, "archetype_overlay"),
        ]:
            t1 = time.time()
            print(f"[F] {label} group={g}", flush=True)
            try:
                out = renderer(mice, g, sF)
            except Exception as e:
                out = e
            _record(label, g, out, round(time.time() - t1, 1))

        # Bounded SI under each aggregation variant.
        for method in SI_METHODS:
            label = f"SI_normalised__{method}"
            t1 = time.time()
            print(f"[F] {label} group={g}", flush=True)
            try:
                out = render_style_F_si_normalised_over_xG(
                    mice, g, sF, method=method)
            except Exception as e:
                out = e
            _record(label, g, out, round(time.time() - t1, 1))

        # Variants-compare overlay (only meaningful for signed_contrast).
        t1 = time.time()
        print(f"[F] SI_variants_compare group={g}", flush=True)
        try:
            out = render_style_F_variants_compare(mice, g, sF)
        except Exception as e:
            out = e
        _record("SI_variants_compare", g, out, round(time.time() - t1, 1))

        # Framework-validation plots (signed_contrast-only; others early-return).
        framework_renderers = [
            ("single_trial_scatter",
             lambda mm, gg, dd: render_style_F_single_trial_scatter(mm, gg, dd)),
            ("kappa_vs_si_spread",
             lambda mm, gg, dd: render_style_F_kappa_vs_si_spread(mm, gg, dd)),
        ]
        # si_vs_confidence: one figure per (method, proxy) combination
        # (4 methods × 2 proxies = 8 figures). ``_make_conf_fn`` binds the
        # loop variables properly — bare lambdas in a comprehension would
        # late-bind to whichever values the variables end up holding.
        def _make_conf_fn(_proxy, _method):
            return lambda mm, gg, dd: render_style_F_si_vs_confidence(
                mm, gg, dd, proxy=_proxy, method=_method)
        for proxy in ("entropy", "velocity"):
            for method in SI_METHODS:
                framework_renderers.append(
                    (f"si_vs_confidence__{proxy}__{method}",
                     _make_conf_fn(proxy, method)))
        for label, fn in framework_renderers:
            t1 = time.time()
            print(f"[F] {label} group={g}", flush=True)
            try:
                out = fn(mice, g, sF)
            except Exception as e:
                out = e
            _record(label, g, out, round(time.time() - t1, 1))

    # Behavioural / fitting renderers (signed_contrast only; not in f_groups
    # loop since they don't sweep groupings — one figure each).
    behavioural_renderers = [
        ("kappa_vs_lapse",
         lambda mm, dd: render_style_F_kappa_vs_lapse(mm, "signed_contrast", dd)),
        ("velocity_accumulator",
         lambda mm, dd: render_style_F_velocity_accumulator(mm, "signed_contrast", dd)),
        ("si_choice_logistic",
         lambda mm, dd: render_style_F_si_choice_logistic(mm, "signed_contrast", dd)),
        ("si_choice_logistic_xG_binned",
         lambda mm, dd: render_style_F_si_choice_logistic_xG_binned(
             mm, "signed_contrast", dd)),
        ("ddm_interrogation",
         lambda mm, dd: render_style_F_ddm_interrogation(mm, "signed_contrast", dd)),
        ("kappa_vs_ddm_sigma_v",
         lambda mm, dd: render_style_F_kappa_vs_ddm_sigma_v(mm, "signed_contrast", dd)),
    ]
    for label, fn in behavioural_renderers:
        t1 = time.time()
        print(f"[F] {label}", flush=True)
        try:
            out = fn(mice, sF)
        except Exception as e:
            out = e
        _record(label, "signed_contrast", out, round(time.time() - t1, 1))

    # Behaviour overlays run once per (behaviour, x-axis feature) pair —
    # P(Go) or mean pre-RZ velocity × signed_contrast or abs_from_go.
    # ``behaviour_overlay`` interprets ``grouping`` as the x-axis feature
    # rather than as a colour-grouping, so it sits outside ``f_groups``.
    for behaviour_kind in ("choice", "velocity"):
        for psy_feature in ("signed_contrast", "abs_from_go"):
            t1 = time.time()
            label = f"behaviour_overlay__{behaviour_kind}__{psy_feature}"
            print(f"[F] {label}", flush=True)
            try:
                out = render_style_F_behaviour_overlay(
                    mice, psy_feature, sF, behaviour=behaviour_kind)
            except Exception as e:
                out = e
            _record(label, psy_feature, out, round(time.time() - t1, 1))

    meta["elapsed_secs"] = round(time.time() - t0, 1)
    with open(fig_root / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    n_errors = sum(1 for p in meta["produced"] if "error" in p)
    print(f"\n[done] {len(meta['produced'])} entries; {n_errors} errors; "
          f"elapsed {meta['elapsed_secs']}s")
    print(f"[done] meta written to {fig_root / '_meta.json'}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_PATH,
                   help="path to NeuralStore .mat (default: dim_reduction_explore.DATA_PATH)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: ./figures/similarity_framework/)")
    p.add_argument("--smoke", action="store_true",
                   help="run one mouse × one grouping per renderer for a fast sanity check")
    args = p.parse_args()
    run_similarity_sweep(data_path=args.data, fig_root=args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
