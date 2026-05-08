# -*- coding: utf-8 -*-
"""
GLM-HMM Analysis for the Go/No-Go VR task — 3-stage hierarchical fitting
(Global → Animal → Session) with K-selection by BIC + LOSO held-out LL.

------------------------------------------------------------------------
CONVENTION TABLE (read this before changing anything below)

PREDICTORS (X columns), in PREDICTOR_COLS order:
    current_stim_signed  — + → Go-pole evidence on the current trial
    prev_stim_signed     — + → Go-pole evidence on the previous trial
    prev_choice          — ∈ {-1, +1}; +1 = previous Go
    prev_reward          — ∈ { 0, +1}; +1 = previous trial got reward
    bias                 — ≡ 1

INTERNAL TARGET (y, what ssm sees):
    y = 0 → Go (lick)
    y = 1 → No-Go
The raw extractor uses 1 = Go; we flip ONCE on the way into the GLM so that
ssm's logit weights params[k][0] read as
    log P(Go) − log P(No-Go) = params[k][0] @ x
and therefore ENGAGED states have a POSITIVE current_stim_signed weight.

EXPORTED CSV target column:
    choice_go = 1 → Go, 0 → No-Go (un-flipped — readable for downstream).

ANIMAL FLIP CONVENTION (per training history):
    Animals in GO_POLE_AT_90_DEG (Cb21, Cb22, Cb24) were trained with Go = 90°.
    Other animals (Cb15, Cb17, Cb25) were trained with Go = 0°.
    `compute_signed_stim` produces +1 at each animal's Go pole regardless.

The convention above is numerically identical to the legacy v4 dual-flip
(`flip_mapping` then unconditional `*= -1` plus `1 - y`). Tests in
tests/test_glm_hmm.py pin this equivalence.
------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ssm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

RNG_SEED = 42

# Animals trained with Go = 90°. Others have Go = 0°.
GO_POLE_AT_90_DEG = frozenset({"Cb21", "Cb22", "Cb24"})

PREDICTOR_COLS = [
    "current_stim_signed",
    "prev_stim_signed",
    "prev_choice",
    "prev_reward",
    "bias",
]

DO_SESSION_FITS = True
LOSO_AT_ALL_K = True

L2_REG = 1.5
NUM_ITERS_GLOBAL = 500
NUM_ITERS_ANIMAL = 100   # raised from 10; convergence-warning gates this
NUM_ITERS_SESSION = 100   # raised from 10
CONVERGENCE_TOL = 1e-3   # |Δ LL / |LL|| under this is "converged"
SIGMA_INIT = 0.5
TRANSITION_NOISE = 0.2
CANDIDATE_K = [2, 3, 4]
N_INITS = 10


# ---------------------------------------------------------------------------
# Sign-convention helpers (Item 1)
# ---------------------------------------------------------------------------

def compute_signed_stim(orientation, contrast, dispersion, animal_id):
    """Map (orientation, contrast, dispersion, animal) → signed scaled stimulus.

    Sign convention: positive output → evidence for the Go pole.

    Animals in GO_POLE_AT_90_DEG were trained with Go at 90°; their raw
    normalisation (orientation - 45)/45 is already +1 at the Go pole, so we
    keep the sign. Other animals have Go at 0° and we negate.

    Numerically equivalent to v4-legacy's chained `flip_mapping` then
    unconditional `*= -1`. See tests/test_glm_hmm.py::
    test_compute_signed_stim_replaces_legacy_dual_flip.
    """
    orientation = np.asarray(orientation, dtype=float)
    contrast = np.asarray(contrast, dtype=float)
    dispersion = np.asarray(dispersion, dtype=float)

    contrast_eff = np.where(np.isclose(contrast, 0.99), 1.0, contrast)
    disp_factor = 5.0 / dispersion
    norm = (orientation - 45.0) / 45.0
    sign = +1.0 if animal_id in GO_POLE_AT_90_DEG else -1.0
    return sign * norm * contrast_eff * disp_factor


# ---------------------------------------------------------------------------
# BIC (Item 5)
# ---------------------------------------------------------------------------

def count_glmhmm_params(K, input_dim):
    """Param count for a K-state input-driven binary GLM-HMM.

      K * input_dim   emission weights (one logit vector per state)
      K * (K - 1)     transition matrix (K rows × (K-1) free probs)
      (K - 1)         initial-state distribution (K-1 free probs)
    """
    return K * input_dim + K * (K - 1) + (K - 1)


def calculate_bic(model, X_data, y_data, K, input_dim):
    log_likelihood = model.log_likelihood(y_data, inputs=X_data)
    n_params = count_glmhmm_params(K, input_dim)
    n_samples = len(y_data)
    return -2.0 * log_likelihood + n_params * np.log(n_samples)


# ---------------------------------------------------------------------------
# Probe grid + functional alignment (Item 6)
# ---------------------------------------------------------------------------

def make_probe_grid():
    """36-point grid spanning natural input ranges, in PREDICTOR_COLS order.
    3 stim × 3 prev_stim × 2 prev_choice × 2 prev_reward × 1 bias = 36.
    """
    rows = []
    for s in [-1.0, 0.0, +1.0]:
        for ps in [-1.0, 0.0, +1.0]:
            for pc in [-1.0, +1.0]:
                for pr in [0.0, 1.0]:
                    rows.append([s, ps, pc, pr, 1.0])
    return np.asarray(rows, dtype=float)


def predict_pgo_on_grid(model, probe_grid):
    """P(Go) for each state at each probe point, shape (K, n_probe).

    ssm InputDrivenObservations parameterises class 0 (Go in our encoding) as
    the non-reference class relative to class C-1 (No-Go), so
        params[k][0] @ x = log P(Go) − log P(No-Go)
    and  P(Go) = sigmoid(params[k][0] @ x).
    """
    K = model.K
    P = np.zeros((K, probe_grid.shape[0]))
    for k in range(K):
        w = np.asarray(model.observations.params[k][0]).ravel()
        logit = probe_grid @ w
        P[k] = 1.0 / (1.0 + np.exp(-logit))
    return P


def match_states_functional(subject_model, reference_model, probe_grid):
    """Align subject states to reference by matching predicted P(Go|x_probe).

    Hungarian on L2 distance between (K, n_probe) probability matrices.
    Returns (subject_model_permuted, cost_matrix, margin) where
      cost_matrix[i, j] = ||p_subj[i] - p_ref[j]||₂  over the probe grid
      margin            = mean( min off-diagonal cost − chosen cost ),
                          larger = more confident assignment.
    """
    K = subject_model.K
    p_subj = predict_pgo_on_grid(subject_model, probe_grid)
    p_ref = predict_pgo_on_grid(reference_model, probe_grid)

    cost = np.linalg.norm(p_subj[:, None, :] - p_ref[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(K, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r
    subject_model.permute(perm)

    # Margin computed on the post-permutation cost matrix (diag = chosen).
    p_subj_after = predict_pgo_on_grid(subject_model, probe_grid)
    cost_after = np.linalg.norm(p_subj_after[:, None, :] - p_ref[None, :, :], axis=-1)
    diag = np.diag(cost_after)
    off = cost_after.copy()
    np.fill_diagonal(off, np.inf)
    if K > 1:
        min_off = off.min(axis=1)
        margin = float((min_off - diag).mean())
    else:
        margin = float("inf")
    return subject_model, cost, margin


def js_drift(subject_model, reference_model, probe_grid):
    """Per-state symmetric Jensen-Shannon between subject and reference P(Go).

    Assumes states are already aligned (run match_states_functional first).
    Returns array of shape (K,), each entry is the mean JS over probe points.
    """
    eps = 1e-12
    p_subj = np.clip(predict_pgo_on_grid(subject_model, probe_grid), eps, 1 - eps)
    p_ref = np.clip(predict_pgo_on_grid(reference_model, probe_grid), eps, 1 - eps)

    def _kl_bern(p, q):
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    m = 0.5 * (p_subj + p_ref)
    js = 0.5 * (_kl_bern(p_subj, m) + _kl_bern(p_ref, m))
    return js.mean(axis=1)


# ---------------------------------------------------------------------------
# Engagement sort
# ---------------------------------------------------------------------------

def sort_global_model_by_engagement(model):
    """Sort states so that state 0 has the largest |stim weight|."""
    stim_w = np.array([
        np.asarray(model.observations.params[k][0]).ravel()[0]
        for k in range(model.K)
    ])
    perm = np.argsort(np.abs(stim_w))[::-1]
    model.permute(perm)
    return model


# ---------------------------------------------------------------------------
# Fit wrappers with convergence diagnostics (Items 3, 7)
# ---------------------------------------------------------------------------

def _convergence_delta(lls):
    if lls is None or len(lls) < 2:
        return np.inf
    if abs(lls[-1]) < 1e-12:
        return abs(lls[-1] - lls[-2])
    return abs(lls[-1] - lls[-2]) / abs(lls[-1])


def _warn_if_not_converged(lls, label, tol=CONVERGENCE_TOL):
    delta = _convergence_delta(lls)
    if delta > tol:
        n_iters = len(lls) if lls is not None else 0
        warnings.warn(
            f"GLM-HMM EM did not converge for {label}: "
            f"final |ΔLL/|LL|| = {delta:.2e} > tol {tol:.0e} after {n_iters} iters",
            RuntimeWarning,
            stacklevel=2,
        )
    return delta


def fit_global_glmhmm(
    X, y, K,
    n_inits=N_INITS,
    num_iters=NUM_ITERS_GLOBAL,
    l2_reg=L2_REG,
    sigma_init=SIGMA_INIT,
    transition_noise=TRANSITION_NOISE,
    seed=None,
):
    """Stage-1 multi-init fit of the global GLM-HMM at K states.

    Returns (best_model, best_ll, best_traj). best_traj is the per-iter LL
    list of the best-LL init, suitable for convergence diagnostic plotting.
    """
    obs_dim = 1
    input_dim = X.shape[1]
    if seed is not None:
        np.random.seed(int(seed))

    # Single-state GLM baseline → warm-start for multi-state weights.
    glm_model = ssm.HMM(
        1, obs_dim, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=2),
        transitions="standard",
    )
    base_traj = glm_model.fit(
        y, inputs=X, method="em", num_iters=num_iters,
        observations_mstep_kwargs=dict(l2_penalty=l2_reg),
        verbose=0,
    )
    _warn_if_not_converged(base_traj, label="Stage-1 GLM baseline (K=1)")
    global_glm_weights = np.asarray(glm_model.observations.params[0][0]).copy()

    best_ll = -np.inf
    best_model = None
    best_traj = None
    for _ in range(n_inits):
        model = ssm.HMM(
            K, obs_dim, input_dim,
            observations="input_driven_obs",
            observation_kwargs=dict(C=2),
            transitions="standard",
        )
        for k in range(K):
            model.observations.params[k][0] = (
                global_glm_weights
                + np.random.normal(scale=sigma_init, size=global_glm_weights.shape)
            )
        init_trans = np.eye(K) + transition_noise * np.random.rand(K, K)
        init_trans /= init_trans.sum(axis=1, keepdims=True)
        model.transitions.log_Ps = np.log(init_trans)

        traj = model.fit(
            y, inputs=X, method="em", num_iters=num_iters,
            observations_mstep_kwargs=dict(l2_penalty=l2_reg),
            verbose=0,
        )
        ll = model.log_likelihood(y, inputs=X)
        if ll > best_ll:
            best_ll, best_model, best_traj = ll, model, traj

    _warn_if_not_converged(best_traj, label=f"Stage-1 global K={K}")
    return best_model, best_ll, best_traj


def fit_child_glmhmm(
    X, y, K, parent_model,
    num_iters=NUM_ITERS_ANIMAL,
    l2_reg=L2_REG,
    seed=None,
):
    """Stage-2/Stage-3 fit warm-started from `parent_model`. Returns (model, ll, traj)."""
    obs_dim = 1
    input_dim = X.shape[1]
    if seed is not None:
        np.random.seed(int(seed))

    model = ssm.HMM(
        K, obs_dim, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=2),
        transitions="standard",
    )
    model.transitions.log_Ps = np.asarray(parent_model.transitions.log_Ps).copy()
    for k in range(K):
        model.observations.params[k][0] = (
            np.asarray(parent_model.observations.params[k][0]).copy()
        )

    traj = model.fit(
        y, inputs=X, method="em", num_iters=num_iters,
        observations_mstep_kwargs=dict(l2_penalty=l2_reg),
        verbose=0,
    )
    ll = model.log_likelihood(y, inputs=X)
    return model, ll, traj


# ---------------------------------------------------------------------------
# LOSO cross-validation (Item 4)
# ---------------------------------------------------------------------------

def loso_animal_loglik(
    animal_session_data, parent_model, K,
    num_iters=NUM_ITERS_ANIMAL,
    l2_reg=L2_REG,
    seed=None,
):
    """Leave-one-session-out per animal.

    animal_session_data: dict[animal_id][session_id] = (X, y)
    Returns dict[animal_id][heldout_session_id] = per-trial held-out LL.
    """
    out = {}
    for animal_id, sessions in animal_session_data.items():
        out[animal_id] = {}
        sids = list(sessions.keys())
        for held in sids:
            train_sids = [sid for sid in sids if sid != held]
            X_train = np.concatenate([sessions[sid][0] for sid in train_sids], axis=0)
            y_train = np.concatenate([sessions[sid][1] for sid in train_sids], axis=0)
            model, _, _ = fit_child_glmhmm(
                X_train, y_train, K, parent_model,
                num_iters=num_iters, l2_reg=l2_reg, seed=seed,
            )
            X_held, y_held = sessions[held]
            ll = model.log_likelihood(y_held, inputs=X_held)
            out[animal_id][held] = float(ll) / max(len(y_held), 1)
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_predictor_sanity(data_transformed, out_dir):
    animals = sorted(data_transformed["animal"].unique())
    fig, axs = plt.subplots(
        len(animals), 4, figsize=(14, 2.0 * len(animals)),
        sharex="col", sharey="row", squeeze=False,
    )
    cols = ["current_stim_signed", "prev_stim_signed", "prev_choice", "prev_reward"]
    for i, ani in enumerate(animals):
        sub = data_transformed[data_transformed["animal"] == ani]
        for j, col in enumerate(cols):
            ax = axs[i, j]
            ax.hist(sub[col].dropna().values, bins=30,
                    color="steelblue", edgecolor="none")
            if i == 0:
                ax.set_title(col, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ani}\ncount", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Stage-0: per-animal predictor distributions (sanity)", y=1.0)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage0_predictor_sanity.png", dpi=140)
    plt.show()


def plot_k_selection(global_bics, global_lls, loso_results, out_dir):
    Ks = sorted(global_bics.keys())
    bics = [global_bics[K] for K in Ks]
    lls = [global_lls[K] for K in Ks]

    loso_mu, loso_sem = [], []
    for K in Ks:
        vals = []
        for _, by_sess in loso_results[K].items():
            vals.extend(by_sess.values())
        vals = np.asarray(vals, dtype=float)
        loso_mu.append(vals.mean())
        loso_sem.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    axs[0].plot(Ks, bics, "bo-", linewidth=2)
    axs[0].set_xlabel("Number of states K")
    axs[0].set_ylabel("BIC (lower is better)")
    axs[0].set_xticks(Ks); axs[0].set_title("BIC"); axs[0].grid(alpha=0.3)

    axs[1].plot(Ks, lls, "rs-", linewidth=2)
    axs[1].set_xlabel("Number of states K")
    axs[1].set_ylabel("Train log-likelihood")
    axs[1].set_xticks(Ks); axs[1].set_title("Train LL"); axs[1].grid(alpha=0.3)

    axs[2].errorbar(Ks, loso_mu, yerr=loso_sem, fmt="go-", linewidth=2, capsize=4)
    axs[2].set_xlabel("Number of states K")
    axs[2].set_ylabel("Held-out LL / trial")
    axs[2].set_xticks(Ks); axs[2].set_title("LOSO held-out LL")
    axs[2].grid(alpha=0.3)

    fig.suptitle("Stage-1 model selection (BIC + train LL + LOSO held-out)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage1_k_selection.png", dpi=140)
    plt.show()


def plot_global_weights_grid(global_models, out_dir):
    Ks = sorted(global_models.keys())
    max_K = max(Ks)
    fig, axs = plt.subplots(
        max_K, len(Ks), figsize=(4 * len(Ks), 2.6 * max_K),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax in axs.flat:
        ax.set_visible(False)
    for col, K in enumerate(Ks):
        model = global_models[K]
        for k in range(K):
            ax = axs[k, col]; ax.set_visible(True)
            w = np.asarray(model.observations.params[k][0]).ravel()
            ax.plot(range(len(PREDICTOR_COLS)), w, marker="o", color="k")
            ax.axhline(0, color="gray", linestyle=":")
            ax.set_xticks(range(len(PREDICTOR_COLS)))
            ax.set_xticklabels(PREDICTOR_COLS, rotation=45, fontsize=8)
            label = " (Engaged)" if k == 0 else ""
            ax.set_title(f"K={K} | State {k+1}{label}", fontsize=9)
            if col == 0:
                ax.set_ylabel("GLM weight\n(log-odds Go)", fontsize=8)
    fig.suptitle("Stage-1 global GLM weights across K (engaged = + stim weight)", y=1.01)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage1_global_weights.png", dpi=140)
    plt.show()


def plot_transition_matrices(global_models, out_dir):
    Ks = sorted(global_models.keys())
    fig, axs = plt.subplots(1, len(Ks), figsize=(4 * len(Ks), 3.2))
    if len(Ks) == 1:
        axs = [axs]
    for ax, K in zip(axs, Ks):
        T = np.exp(global_models[K].transitions.log_Ps)
        sns.heatmap(T, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f"K={K} transitions"); ax.set_xlabel("To state")
        ax.set_ylabel("From state")
    fig.suptitle("Stage-1 transition matrices")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage1_transitions.png", dpi=140)
    plt.show()


def plot_em_convergence(global_trajs, out_dir):
    Ks = sorted(global_trajs.keys())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for K in Ks:
        traj = global_trajs[K]
        if traj is None or len(traj) == 0:
            continue
        ax.plot(np.arange(1, len(traj) + 1), traj, marker=".", label=f"K={K}")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("Log-likelihood")
    ax.set_title("Stage-1 EM convergence (best init per K)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage1_em_convergence.png", dpi=140)
    plt.show()


def plot_animal_weights_overlay(global_model, animal_models, out_dir):
    K = global_model.K
    fig, axs = plt.subplots(K, 1, figsize=(10, 3.5 * K), sharex=True, squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(animal_models)))
    for k in range(K):
        ax = axs[k, 0]
        ref_w = np.asarray(global_model.observations.params[k][0]).ravel()
        ax.plot(range(len(PREDICTOR_COLS)), ref_w, "k--",
                linewidth=3, alpha=0.4, label="Global ref")
        for i, (animal, m) in enumerate(animal_models.items()):
            w = np.asarray(m.observations.params[k][0]).ravel()
            ax.plot(range(len(PREDICTOR_COLS)), w, marker="o",
                    color=colors[i], label=animal)
        ax.set_xticks(range(len(PREDICTOR_COLS)))
        ax.set_xticklabels(PREDICTOR_COLS, rotation=45, fontsize=8)
        ax.axhline(0, color="gray", linestyle=":")
        ax.set_title(f"State {k+1}" + (" (Engaged)" if k == 0 else ""))
        ax.set_ylabel("GLM weight (log-odds Go)")
        if k == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1), fontsize=8)
    fig.suptitle("Stage-2 GLM weights: animals vs global reference")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage2_animal_weights.png", dpi=140)
    plt.show()


def plot_pgo_curve_overlay(global_model, animal_models, out_dir):
    """Per-state predicted P(Go) vs current_stim_signed (history regressors at neutral)."""
    K = global_model.K
    n = 41
    stim_axis = np.linspace(-1.5, 1.5, n)
    base = np.zeros((n, len(PREDICTOR_COLS)))
    base[:, 0] = stim_axis
    base[:, 1] = 0.0
    base[:, 2] = 0.0
    base[:, 3] = 0.5
    base[:, 4] = 1.0

    fig, axs = plt.subplots(1, K, figsize=(4 * K, 3.5), sharey=True, squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(animal_models)))
    for k in range(K):
        ax = axs[0, k]
        ref_w = np.asarray(global_model.observations.params[k][0]).ravel()
        ax.plot(stim_axis, 1 / (1 + np.exp(-(base @ ref_w))),
                "k--", linewidth=2.5, alpha=0.6, label="Global ref")
        for i, (animal, m) in enumerate(animal_models.items()):
            w = np.asarray(m.observations.params[k][0]).ravel()
            ax.plot(stim_axis, 1 / (1 + np.exp(-(base @ w))),
                    color=colors[i], linewidth=1.5, label=animal)
        ax.set_xlabel("current_stim_signed (+ = Go)")
        if k == 0:
            ax.set_ylabel("P(Go)")
        ax.axhline(0.5, color="gray", linestyle=":")
        ax.axvline(0.0, color="gray", linestyle=":")
        ax.set_title(f"State {k+1}" + (" (Engaged)" if k == 0 else ""))
        if k == K - 1:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Stage-2 predicted P(Go) curves: animals vs global ref")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage2_pgo_curves.png", dpi=140)
    plt.show()


def plot_alignment_diagnostics(animal_costs, animal_margins, out_dir):
    animals = list(animal_costs.keys())
    n = len(animals)
    cols = min(n, 4); rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.6 * rows), squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    for i, ani in enumerate(animals):
        ax = axs[i // cols, i % cols]; ax.set_visible(True)
        cost = animal_costs[ani]
        sns.heatmap(cost, annot=True, fmt=".2f", cmap="viridis_r", cbar=False, ax=ax)
        for k in range(cost.shape[0]):
            ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=False,
                                       edgecolor="red", linewidth=2))
        ax.set_title(f"{ani} (margin={animal_margins[ani]:.3f})", fontsize=9)
        ax.set_xlabel("Reference state"); ax.set_ylabel("Subject state")
    fig.suptitle(
        "Stage-2 alignment cost (output-space L2)\n"
        "red = chosen assignment, larger margin = unambiguous"
    )
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage2_alignment_cost.png", dpi=140)
    plt.show()


def plot_drift_bar(animal_drifts, out_dir):
    animals = list(animal_drifts.keys())
    K = len(next(iter(animal_drifts.values())))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(animals)), 3.5))
    width = 0.8 / K
    for k in range(K):
        vals = [animal_drifts[a][k] for a in animals]
        x = np.arange(len(animals)) + (k - (K - 1) / 2) * width
        label = f"State {k+1}" + (" (Engaged)" if k == 0 else "")
        ax.bar(x, vals, width=width, label=label)
    ax.set_xticks(np.arange(len(animals)))
    ax.set_xticklabels(animals, rotation=30)
    ax.set_ylabel("Jensen–Shannon drift\n(animal vs global, P(Go) probe space)")
    ax.set_title("Stage-2 state drift per animal × state (lower = state preserved)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage2_drift.png", dpi=140)
    plt.show()


def plot_animal_transitions(animal_models, out_dir):
    animals = list(animal_models.keys())
    fig, axs = plt.subplots(1, len(animals), figsize=(3 * len(animals), 3))
    if len(animals) == 1:
        axs = [axs]
    for ax, ani in zip(axs, animals):
        T = np.exp(animal_models[ani].transitions.log_Ps)
        sns.heatmap(T, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(ani); ax.set_xlabel("To state")
    axs[0].set_ylabel("From state")
    fig.suptitle("Stage-2 transition matrices (animal level)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_stage2_animal_transitions.png", dpi=140)
    plt.show()


# ---------------------------------------------------------------------------
# Export aligned posteriors + weights
# ---------------------------------------------------------------------------

def export_aligned(
    data_transformed, animal_data, animal_models,
    animal_session_data, session_models,
    chosen_K, do_session_fits, out_dir,
):
    suffix = "_hierarchical.csv" if do_session_fits else "_animal_level.csv"

    export_df = data_transformed.copy()
    for k in range(chosen_K):
        export_df[f"prob_state_{k+1}"] = np.nan

    weight_rows = []
    if do_session_fits:
        for (animal_id, session_id), m in session_models.items():
            X_s, y_s = animal_session_data[animal_id][int(session_id)]
            posterior = m.expected_states(y_s, input=X_s)[0]
            mask = ((export_df["animal"] == animal_id)
                    & (export_df["session"] == session_id))
            idx = export_df.index[mask]
            for k in range(chosen_K):
                export_df.loc[idx, f"prob_state_{k+1}"] = posterior[:, k]
                w = np.asarray(m.observations.params[k][0]).ravel()
                for p_idx, p_name in enumerate(PREDICTOR_COLS):
                    weight_rows.append({
                        "animal": animal_id, "session": session_id,
                        "state": k + 1, "predictor": p_name,
                        "weight": float(w[p_idx]),
                    })
    else:
        for animal_id, m in animal_models.items():
            X_a, y_a = animal_data[animal_id]
            posterior = m.expected_states(y_a, input=X_a)[0]
            mask = (export_df["animal"] == animal_id)
            idx = export_df.index[mask]
            for k in range(chosen_K):
                export_df.loc[idx, f"prob_state_{k+1}"] = posterior[:, k]
                w = np.asarray(m.observations.params[k][0]).ravel()
                for p_idx, p_name in enumerate(PREDICTOR_COLS):
                    weight_rows.append({
                        "animal": animal_id, "session": "ALL",
                        "state": k + 1, "predictor": p_name,
                        "weight": float(w[p_idx]),
                    })

    # MATLAB integrator reads choice_go + signed-stim columns directly,
    # eliminating the duplicated per-animal flip logic on the MATLAB side.
    export_cols = [
        "animal", "session", "trial_in_session",
        "choice_go", "prev_reward",
        "current_stim_signed", "prev_stim_signed",
    ] + [f"prob_state_{k+1}" for k in range(chosen_K)]

    export_df[export_cols].to_csv(
        Path(out_dir) / f"GLM_HMM_states_aligned{suffix}", index=False
    )
    pd.DataFrame(weight_rows).to_csv(
        Path(out_dir) / f"GLM_HMM_weights{suffix}", index=False
    )
    print(f"Exported GLM_HMM_states_aligned{suffix} and GLM_HMM_weights{suffix}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(input_csv="GLMHmm_predictors_vr.csv", out_dir=".",
         seed=RNG_SEED, interactive=True):
    np.random.seed(seed)

    print("Loading Data...")
    data = pd.read_csv(input_csv)
    data["bias"] = 1.0

    print("Generating trial indices...")
    data["trial_in_session"] = data.groupby(["animal", "session"]).cumcount() + 1

    # Single signed-stim transformation per animal (Item 1).
    pieces = []
    for animal_id, group in data.groupby("animal"):
        g = group.copy()
        g["current_stim_signed"] = compute_signed_stim(
            g["current_stim_orientation"].values,
            g["current_contrast"].values,
            g["current_dispersion"].values,
            animal_id,
        )
        g["prev_stim_signed"] = compute_signed_stim(
            g["prev_stim_orientation"].values,
            g["prev_contrast"].values,
            g["prev_dispersion"].values,
            animal_id,
        )
        pieces.append(g)
    data_transformed = pd.concat(pieces, ignore_index=True)
    data_transformed = data_transformed.sort_values(
        ["animal", "session", "trial_in_session"]
    ).reset_index(drop=True)

    # choice_go = raw extractor convention (1 = Go). _y_internal flips for ssm.
    data_transformed["choice_go"] = data_transformed["current_choice"].astype(int)
    data_transformed["_y_internal"] = 1 - data_transformed["choice_go"]
    data_transformed["prev_choice"] = (
        data_transformed["prev_choice"].apply(lambda x: 1 if x == 1 else -1)
    )

    X_global = data_transformed[PREDICTOR_COLS].values.astype(float)
    y_global = data_transformed["_y_internal"].values.reshape(-1, 1).astype(int)

    animal_data = {}
    animal_session_data = {}
    for animal_id, group in data_transformed.groupby("animal"):
        animal_data[animal_id] = (
            group[PREDICTOR_COLS].values.astype(float),
            group["_y_internal"].values.reshape(-1, 1).astype(int),
        )
        sd = {}
        for session_id, sgroup in group.groupby("session"):
            sd[int(session_id)] = (
                sgroup[PREDICTOR_COLS].values.astype(float),
                sgroup["_y_internal"].values.reshape(-1, 1).astype(int),
            )
        animal_session_data[animal_id] = sd

    input_dim = len(PREDICTOR_COLS)

    # ---- Stage-0 ----
    plot_predictor_sanity(data_transformed, out_dir)

    # ---- Stage-1: K selection ----
    print("\n--- Stage-1: global multi-state search ---")
    global_models, global_bics, global_lls, global_trajs = {}, {}, {}, {}
    all_k_weights = []
    for K in CANDIDATE_K:
        model, ll, traj = fit_global_glmhmm(X_global, y_global, K, seed=seed)
        bic = calculate_bic(model, X_global, y_global, K, input_dim)
        model = sort_global_model_by_engagement(model)
        global_models[K] = model
        global_bics[K] = bic
        global_lls[K] = ll
        global_trajs[K] = traj
        for k_state in range(K):
            w = np.asarray(model.observations.params[k_state][0]).ravel()
            for p_idx, p_name in enumerate(PREDICTOR_COLS):
                all_k_weights.append({
                    "K_model": K, "state": k_state + 1,
                    "predictor": p_name, "weight": float(w[p_idx]),
                })
        print(f"  K={K}: LL={ll:.2f}, BIC={bic:.2f}")

    pd.DataFrame(all_k_weights).to_csv(
        Path(out_dir) / "GLM_HMM_global_weights_all_K.csv", index=False
    )

    # ---- LOSO at all candidate K ----
    print("\n--- LOSO cross-validation across all K ---")
    loso_results = {}
    for K in CANDIDATE_K:
        loso_results[K] = loso_animal_loglik(
            animal_session_data, global_models[K], K, seed=seed,
        )
    loso_rows = []
    for K, by_animal in loso_results.items():
        for animal_id, by_sess in by_animal.items():
            for sid, ll in by_sess.items():
                loso_rows.append({
                    "K": K, "animal": animal_id,
                    "heldout_session": sid, "heldout_ll_per_trial": ll,
                })
    pd.DataFrame(loso_rows).to_csv(Path(out_dir) / "GLM_HMM_loso_cv.csv", index=False)

    plot_k_selection(global_bics, global_lls, loso_results, out_dir)
    plot_global_weights_grid(global_models, out_dir)
    plot_transition_matrices(global_models, out_dir)
    plot_em_convergence(global_trajs, out_dir)

    best_K_bic = min(global_bics, key=global_bics.get)
    if interactive:
        try:
            user_input = input(
                f"\nEnter K {CANDIDATE_K}. [Enter] = best BIC ({best_K_bic}): "
            )
            chosen_K = int(user_input.strip()) if user_input.strip() else best_K_bic
            if chosen_K not in CANDIDATE_K:
                print(f"  {chosen_K} not in candidates → defaulting to {best_K_bic}")
                chosen_K = best_K_bic
        except (ValueError, EOFError):
            chosen_K = best_K_bic
    else:
        chosen_K = best_K_bic
    print(f"\n---> Proceeding with K = {chosen_K} <---")
    final_global_model = global_models[chosen_K]

    # ---- Stage-2: animal-level fits + functional alignment ----
    print(f"\n--- Stage-2: animal fits (K={chosen_K}) ---")
    probe_grid = make_probe_grid()
    animal_models, animal_costs, animal_margins, animal_drifts = {}, {}, {}, {}
    for animal_id, (X_a, y_a) in animal_data.items():
        m, _, traj = fit_child_glmhmm(
            X_a, y_a, chosen_K, final_global_model,
            num_iters=NUM_ITERS_ANIMAL, l2_reg=L2_REG, seed=seed,
        )
        _warn_if_not_converged(traj, label=f"Stage-2 animal {animal_id}")
        m, cost, margin = match_states_functional(m, final_global_model, probe_grid)
        drift = js_drift(m, final_global_model, probe_grid)
        animal_models[animal_id] = m
        animal_costs[animal_id] = cost
        animal_margins[animal_id] = margin
        animal_drifts[animal_id] = drift

    plot_animal_weights_overlay(final_global_model, animal_models, out_dir)
    plot_pgo_curve_overlay(final_global_model, animal_models, out_dir)
    plot_alignment_diagnostics(animal_costs, animal_margins, out_dir)
    plot_drift_bar(animal_drifts, out_dir)
    plot_animal_transitions(animal_models, out_dir)

    drift_rows = []
    for animal_id, drift in animal_drifts.items():
        for k_state, dval in enumerate(drift):
            drift_rows.append({
                "animal": animal_id, "state": k_state + 1,
                "js_drift": float(dval),
                "alignment_margin": float(animal_margins[animal_id]),
            })
    pd.DataFrame(drift_rows).to_csv(
        Path(out_dir) / "GLM_HMM_alignment_drift.csv", index=False
    )

    # ---- Stage-3: session-level fits ----
    session_models = {}
    if DO_SESSION_FITS:
        print(f"\n--- Stage-3: session fits (K={chosen_K}) ---")
        for animal_id, sess_dict in animal_session_data.items():
            for session_id, (X_s, y_s) in sess_dict.items():
                m, _, traj = fit_child_glmhmm(
                    X_s, y_s, chosen_K, animal_models[animal_id],
                    num_iters=NUM_ITERS_SESSION, l2_reg=L2_REG, seed=seed,
                )
                _warn_if_not_converged(
                    traj, label=f"Stage-3 {animal_id} session {session_id}"
                )
                m, _, _ = match_states_functional(m, final_global_model, probe_grid)
                session_models[(animal_id, session_id)] = m

    # ---- Export ----
    export_aligned(
        data_transformed, animal_data, animal_models,
        animal_session_data, session_models,
        chosen_K, do_session_fits=DO_SESSION_FITS, out_dir=out_dir,
    )
    print("\nPipeline complete.")
    if not interactive:
        plt.close("all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-HMM pipeline (3-stage hierarchical, vel-only Go/No-Go)."
    )
    parser.add_argument("--input", default="GLMHmm_predictors_vr.csv")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip the K prompt and use the best-BIC K.",
    )
    args = parser.parse_args()
    main(
        input_csv=args.input, out_dir=args.out_dir,
        seed=args.seed, interactive=not args.non_interactive,
    )
