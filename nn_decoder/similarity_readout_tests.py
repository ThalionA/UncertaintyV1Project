"""Decisive readout tests for the Similarity Framework.

Two cross-validated, confound-controlled tests that the framework's
``Similarity Framework.md`` flags as load-bearing but never implemented, here
*sharpened* so they actually discriminate the framework from its rivals:

RD-1 — **Template vs covariance-whitened direction** (the framework's premise).
  The archetype/cosine readout is *linear*: ``SI`` is monotone in
  ``r·(a_Go − a_NoGo)``, i.e. a projection onto the difference-of-class-means
  direction ``Δμ``.  The Bayes-optimal linear discriminant projects onto the
  *whitened* direction ``Σ⁻¹Δμ`` (Fisher/LDA).  So the framework's real claim is
  **not** "boundary vs no boundary" — the cosine readout *is* a boundary — but
  "the brain reads out the un-whitened template ``Δμ``, not the whitened optimal
  ``Σ⁻¹Δμ``".  We interpolate ``Σ(λ) = (1−λ)Σ_LW + λ·(trΣ/p)I`` (λ=1 → template,
  λ=0 → Ledoit–Wolf LDA) and ask, cross-validated, how *stimulus-side* and the
  animal's *choice* are decoded as a function of λ.  Framework prediction: the
  covariance buys stimulus-decoding accuracy (auc_stim rises as λ→0) but does
  *not* buy choice prediction (auc_choice does not ride it down) — a dissociation
  between what the population *could* support and what the animal *uses*.

RD-2 — **Within-condition nested choice model** (Predictions 16 + 9, the SBC
  wedge, made confound-free).  Everything is conditioned on ``signed_contrast``
  so "neural carries choice info" is tested *within* a stimulus level, immune to
  the trivial stimulus→choice coupling.  Nested cross-validated logistic models:

    M0: choice ~ signed_contrast
    M1:   + E_t[SI]          (trial-mean similarity index)   — Prediction 16
    M2:   + Var_t[SI]        (within-trial SI variance)       — SBC wedge
    M3:   + whitened-Δμ mean (Σ⁻¹Δμ trial-mean projection)   — RD-1 within-cond.

  Held-out ΔLL (nats/trial) is honest about overfitting (a noise feature gives
  ΔLL ≤ 0).  Framework predictions:  ΔLL(M1−M0) > 0  (neural SI adds within
  condition);  ΔLL(M2−M1) ≈ 0  (the within-trial wiggle is diffusion noise, not
  posterior samples — *against* SBC, which predicts > 0);  ΔLL(M3−M1) ≈ 0  (the
  whitened readout adds nothing the template didn't — the brain uses Δμ).

Why this fixes the conjecture's stated Prediction 9.  As written, Prediction 9
("within-trial residuals of r(t) are not choice-informative") is confounded by
choice probability — V1 covaries with choice at fixed stimulus under *every*
account, so a positive slope does not specifically support SBC.  The
distinguishing quantity is whether the within-trial *variance* adds choice
information *beyond the trial-mean readout*; that is M2−M1.

Both tests take plain arrays, so they run unchanged on the real Cb15–Cb25 data
and on the ``si_network_model`` cohort (a positive control where the
template-cosine mechanism is true by construction).

Run standalone:

  python similarity_readout_tests.py [--data PATH] [--out PATH] [--smoke]
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

from dim_reduction_explore import (
    DATA_PATH,
    GO_BOUNDARY_DEG,
    MouseData,
    _filter_mice,
    _save_fig,
    _zscore_neurons,
    compute_features,
    load_all_mice,
)
import similarity_analysis as sim

__all__ = [
    "class_direction",
    "shrunk_cov_ledoit_wolf",
    "whitened_direction",
    "rd1_template_vs_whitened",
    "rd2_nested_choice_models",
    "arrays_from_mouse",
    "run_readout_tests",
]

# Shrinkage grid for RD-1: λ=0 (full Ledoit–Wolf LDA) → λ=1 (pure template Δμ).
SHRINKAGE_GRID = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
N_FOLDS = 5
SEED = 0
EPS = 1e-9


# ---------------------------------------------------------------------------
# Linear-algebra core: template / whitened directions
# ---------------------------------------------------------------------------


def class_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-class-means ``Δμ = μ(y=1) − μ(y=0)`` for X (n, p)."""
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def shrunk_cov_ledoit_wolf(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf-shrunk pooled within-class covariance of X (n, p).

    Pools the within-class residuals (each class centred on its own mean) and
    applies analytic Ledoit–Wolf shrinkage toward a scaled identity, giving a
    well-conditioned estimate even when p ≳ n.
    """
    from sklearn.covariance import ledoit_wolf
    resids = []
    for cls in (0, 1):
        Xc = X[y == cls]
        if Xc.shape[0] >= 2:
            resids.append(Xc - Xc.mean(axis=0, keepdims=True))
    R = np.vstack(resids)
    Sigma, _ = ledoit_wolf(R)
    return Sigma


def whitened_direction(dmu: np.ndarray, Sigma: np.ndarray, lam: float) -> np.ndarray:
    """Unit-norm projection direction for shrinkage ``lam`` ∈ [0, 1].

    ``Σ(λ) = (1−λ)·Σ + λ·(trΣ/p)·I``; weight ``w(λ) = Σ(λ)⁻¹ Δμ`` normalised.
    λ=1 → ``Δμ`` (template / nearest-prototype); λ=0 → ``Σ⁻¹Δμ`` (LDA/Fisher).
    """
    p = dmu.shape[0]
    target = (np.trace(Sigma) / p) * np.eye(p)
    S = (1.0 - lam) * Sigma + lam * target
    w = np.linalg.solve(S, dmu)
    n = np.linalg.norm(w)
    return w / (n + EPS)


# ---------------------------------------------------------------------------
# RD-1 — template vs whitened: stimulus vs choice decoding across shrinkage
# ---------------------------------------------------------------------------


def rd1_template_vs_whitened(
    X_bar: np.ndarray,
    y_stim: np.ndarray,
    y_choice: np.ndarray,
    *,
    lambdas: np.ndarray = SHRINKAGE_GRID,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    """Cross-validated AUC for stimulus-side and choice vs shrinkage λ.

    For each fold the directions are fit on the *training* trials (Δμ and Σ from
    stimulus-side labels), then test trials are projected and scored against
    both the stimulus side and the animal's choice.  The direction is oriented
    so the training stimulus-AUC ≥ 0.5 (Δμ points Go-ward).

    Returns per-λ mean/SE AUC for stimulus and choice, plus the framework
    dissociation summary (Δauc from λ=1 to λ=0 for each target).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X_bar = np.asarray(X_bar, float)
    y_stim = np.asarray(y_stim).astype(int)
    y_choice = np.asarray(y_choice).astype(int)
    lambdas = np.asarray(lambdas, float)
    n_lam = lambdas.shape[0]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    auc_stim = np.full((n_folds, n_lam), np.nan)
    auc_choice = np.full((n_folds, n_lam), np.nan)

    for fi, (tr, te) in enumerate(skf.split(X_bar, y_stim)):
        # standardise on train stats
        mu = X_bar[tr].mean(axis=0, keepdims=True)
        sd = X_bar[tr].std(axis=0, keepdims=True) + EPS
        Xtr = (X_bar[tr] - mu) / sd
        Xte = (X_bar[te] - mu) / sd
        ys_tr = y_stim[tr]
        if len(np.unique(ys_tr)) < 2:
            continue
        dmu = class_direction(Xtr, ys_tr)
        Sigma = shrunk_cov_ledoit_wolf(Xtr, ys_tr)
        for li, lam in enumerate(lambdas):
            w = whitened_direction(dmu, Sigma, float(lam))
            # orient so train stim-projection separates Go-ward positive
            proj_tr = Xtr @ w
            if roc_auc_score(ys_tr, proj_tr) < 0.5:
                w = -w
            proj_te = Xte @ w
            ys_te = y_stim[te]
            yc_te = y_choice[te]
            if len(np.unique(ys_te)) == 2:
                auc_stim[fi, li] = roc_auc_score(ys_te, proj_te)
            if len(np.unique(yc_te)) == 2:
                auc_choice[fi, li] = roc_auc_score(yc_te, proj_te)

    def _mean_se(a):
        m = np.nanmean(a, axis=0)
        se = np.nanstd(a, axis=0) / np.sqrt(np.sum(np.isfinite(a), axis=0) + EPS)
        return m, se

    m_stim, se_stim = _mean_se(auc_stim)
    m_choice, se_choice = _mean_se(auc_choice)
    i1 = int(np.argmin(np.abs(lambdas - 1.0)))   # template
    i0 = int(np.argmin(np.abs(lambdas - 0.0)))   # whitened
    return {
        "lambdas": lambdas,
        "auc_stim_mean": m_stim, "auc_stim_se": se_stim,
        "auc_choice_mean": m_choice, "auc_choice_se": se_choice,
        # dissociation: covariance buys stimulus accuracy (>0) but not choice (~0)
        "d_auc_stim_whiten": float(m_stim[i0] - m_stim[i1]),
        "d_auc_choice_whiten": float(m_choice[i0] - m_choice[i1]),
        "auc_stim_template": float(m_stim[i1]),
        "auc_stim_whitened": float(m_stim[i0]),
        "auc_choice_template": float(m_choice[i1]),
        "auc_choice_whitened": float(m_choice[i0]),
    }


# ---------------------------------------------------------------------------
# RD-2 — within-condition nested choice models
# ---------------------------------------------------------------------------


def _cv_loglik(features: np.ndarray, y: np.ndarray, folds) -> float:
    """Mean held-out log-likelihood (nats/trial) of a logistic on ``features``.

    ``features`` is (n, k); an all-ones-intercept column is added by the model.
    Standardisation uses train-fold statistics.  Empty feature set (k=0) fits an
    intercept-only model (the base rate). Folds carry any seeding; lbfgs is
    deterministic, so the scoring itself takes no seed. Shared with
    ``similarity_m2_followup`` (which imports it) so the CV scoring can't drift
    between the two — a divergence hazard on the active Similarity pillar.
    """
    from sklearn.linear_model import LogisticRegression
    ll_sum, n_sum = 0.0, 0
    for tr, te in folds:
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2:
            continue
        if features.shape[1] == 0:
            p1 = float(np.clip(ytr.mean(), 1e-6, 1 - 1e-6))
            logp = yte * np.log(p1) + (1 - yte) * np.log(1 - p1)
        else:
            Xtr, Xte = features[tr], features[te]
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True) + EPS
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            clf.fit(Xtr, ytr)
            p = np.clip(clf.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
            logp = yte * np.log(p) + (1 - yte) * np.log(1 - p)
        ll_sum += float(np.sum(logp))
        n_sum += len(te)
    return ll_sum / (n_sum + EPS) if n_sum else float("nan")


def rd2_nested_choice_models(
    X_bar: np.ndarray,
    y_stim: np.ndarray,
    y_choice: np.ndarray,
    signed_contrast: np.ndarray,
    si_mean: np.ndarray,
    si_var: np.ndarray,
    *,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    """Nested CV logistic choice models; returns held-out LL and ΔLL per step.

    The whitened-projection feature (M3) is fit *inside* each fold (Σ⁻¹Δμ from
    the training stimulus labels) to avoid leakage; ``si_mean``/``si_var`` are
    fixed archetype-based features (stimulus-defined, not choice-defined) and
    ``signed_contrast`` is the stimulus covariate.
    """
    from sklearn.model_selection import StratifiedKFold

    X_bar = np.asarray(X_bar, float)
    y_stim = np.asarray(y_stim).astype(int)
    y = np.asarray(y_choice).astype(int)
    c = np.asarray(signed_contrast, float)
    si_mean = np.asarray(si_mean, float)
    si_var = np.asarray(si_var, float)

    valid = (np.isfinite(c) & np.isfinite(si_mean) & np.isfinite(si_var)
             & np.isfinite(y) & np.isin(y, [0, 1])
             & np.isfinite(X_bar).all(axis=1))
    idx = np.where(valid)[0]
    if idx.size < 40 or len(np.unique(y[idx])) < 2:
        return {"ll": {}, "delta": {}, "n_trials": int(idx.size)}

    X_bar, y_stim, y = X_bar[idx], y_stim[idx], y[idx]
    c, si_mean, si_var = c[idx], si_mean[idx], si_var[idx]

    # whitened-Δμ trial-mean projection, computed per-fold (leakage-free)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(X_bar, y))
    wproj = np.full(y.shape, np.nan)
    for tr, te in folds:
        ys_tr = y_stim[tr]
        if len(np.unique(ys_tr)) < 2:
            continue
        mu = X_bar[tr].mean(axis=0, keepdims=True)
        sd = X_bar[tr].std(axis=0, keepdims=True) + EPS
        Xtr = (X_bar[tr] - mu) / sd
        Xte = (X_bar[te] - mu) / sd
        dmu = class_direction(Xtr, ys_tr)
        Sigma = shrunk_cov_ledoit_wolf(Xtr, ys_tr)
        w = whitened_direction(dmu, Sigma, 0.0)
        if np.corrcoef(Xtr @ w, ys_tr)[0, 1] < 0:
            w = -w
        wproj[te] = Xte @ w

    c2 = c[:, None]
    si_m = si_mean[:, None]
    si_v = si_var[:, None]
    wp = wproj[:, None]

    M = {
        "M0_stim":            np.empty((len(y), 0)),
        "M1_+meanSI":         si_m,
        "M2_+varSI":          np.hstack([si_m, si_v]),
        "M3_+whitened":       np.hstack([si_m, wp]),
    }
    # all models also include signed_contrast (the stimulus covariate)
    ll = {}
    for k, extra in M.items():
        feats = np.hstack([c2, extra]) if extra.shape[1] else c2
        ll[k] = _cv_loglik(feats, y, folds)

    delta = {
        "M1-M0 (meanSI adds | Pred16)": ll["M1_+meanSI"] - ll["M0_stim"],
        "M2-M1 (varSI adds | SBC wedge)": ll["M2_+varSI"] - ll["M1_+meanSI"],
        "M3-M1 (whitened adds | premise)": ll["M3_+whitened"] - ll["M1_+meanSI"],
    }
    return {"ll": ll, "delta": delta, "n_trials": int(len(y))}


# ---------------------------------------------------------------------------
# Adapters: pull the array bundle from a MouseData
# ---------------------------------------------------------------------------


def arrays_from_mouse(m: MouseData, *, si_method: str = "prototype") -> dict | None:
    """Assemble (X_bar, y_stim, y_choice, signed_contrast, si_mean, si_var).

    ``X_bar`` is the z-scored, xG-averaged population (trial summary the
    accumulator integrates).  ``si_mean``/``si_var`` are the trial-mean and
    within-trial variance of the framework SI(t) under ``si_method`` (archetypes
    from easy, stimulus-defined trials).  Returns None if archetypes are missing.
    """
    proto = sim._compute_archetypes(m)
    ex = sim._compute_archetypes_exemplars(m)
    if "Go" not in proto or "NoGo" not in proto:
        return None
    Az = _zscore_neurons(m.activity)          # (n, p, T)
    X_bar = Az.mean(axis=2)                    # (n, p)
    feats = compute_features(m)
    y_stim = (m.orientation < GO_BOUNDARY_DEG).astype(int)
    si = sim._si_per_method(m, proto, ex, method=si_method)   # (n, T)
    return {
        "X_bar": X_bar,
        "y_stim": y_stim,
        "y_choice": m.choice.astype(int),
        "signed_contrast": feats["signed_contrast"],
        "si_mean": np.nanmean(si, axis=1),
        "si_var": np.nanvar(si, axis=1),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


WARM = "#c44e1e"   # template / pathological-optimal-discarded
COOL = "#2f6e8f"   # whitened / stimulus ceiling


def render_rd1(per_mouse: dict, out_dir: Path, stem: str = "rd1_template_vs_whitened"):
    """One panel per mouse + a cohort-mean panel: AUC vs shrinkage λ."""
    names = list(per_mouse.keys())
    n = len(names)
    ncol = min(4, n + 1)
    nrow = int(np.ceil((n + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.5 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    lambdas = per_mouse[names[0]]["lambdas"]

    def _panel(ax, r, title):
        ax.plot(r["lambdas"], r["auc_stim_mean"], "-o", ms=3, color=COOL,
                label="stimulus side")
        ax.fill_between(r["lambdas"], r["auc_stim_mean"] - r["auc_stim_se"],
                        r["auc_stim_mean"] + r["auc_stim_se"], color=COOL, alpha=.18)
        ax.plot(r["lambdas"], r["auc_choice_mean"], "-s", ms=3, color=WARM,
                label="choice")
        ax.fill_between(r["lambdas"], r["auc_choice_mean"] - r["auc_choice_se"],
                        r["auc_choice_mean"] + r["auc_choice_se"], color=WARM, alpha=.18)
        ax.axhline(0.5, ls=":", lw=.8, color="gray")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("λ  (0=LDA … 1=template)", fontsize=7.5)
        ax.set_ylabel("CV AUC", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xticks([0, 0.5, 1.0])

    # cohort mean (interp onto shared grid — same grid everywhere)
    coh_stim = np.nanmean([per_mouse[k]["auc_stim_mean"] for k in names], axis=0)
    coh_choice = np.nanmean([per_mouse[k]["auc_choice_mean"] for k in names], axis=0)
    _panel(axes[0], {"lambdas": lambdas, "auc_stim_mean": coh_stim,
                     "auc_stim_se": np.zeros_like(coh_stim),
                     "auc_choice_mean": coh_choice,
                     "auc_choice_se": np.zeros_like(coh_choice)},
           f"cohort mean (n={n})")
    axes[0].legend(fontsize=6.5, loc="lower left", framealpha=.6)
    for i, k in enumerate(names):
        _panel(axes[i + 1], per_mouse[k], k)
    for j in range(n + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("RD-1 — covariance-whitening (λ→0) improves *stimulus* decoding in every "
                 "mouse: the info is there to use.\nIts marginal choice gain is "
                 "stimulus→choice coupling; whether the brain uses the covariance is "
                 "tested confound-free in RD-2 (M3).",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out_dir / stem)


def render_rd2(per_mouse: dict, out_dir: Path, stem: str = "rd2_nested_choice"):
    """Per-mouse grouped bars of held-out ΔLL for the three nested steps."""
    names = [k for k in per_mouse if per_mouse[k].get("delta")]
    if not names:
        return
    keys = list(per_mouse[names[0]]["delta"].keys())
    short = ["M1−M0\nmean SI\n(Pred 16)", "M2−M1\nvar SI\n(SBC wedge)",
             "M3−M1\nwhitened\n(premise)"]
    colors = [COOL, WARM, "#7a5fa3"]
    n = len(names)
    x = np.arange(len(keys))
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, k in enumerate(names):
        vals = [per_mouse[k]["delta"][kk] for kk in keys]
        ax.bar(x + i * w, vals, w, label=k, alpha=.85)
    # cohort mean overlay
    coh = [np.nanmean([per_mouse[k]["delta"][kk] for k in names]) for kk in keys]
    for xi, (cx, cv, col) in enumerate(zip(x, coh, colors)):
        ax.plot([cx - 0.05, cx + 0.8 + 0.05], [cv, cv], color=col, lw=2.5,
                solid_capstyle="round", zorder=5)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("held-out ΔLL  (nats / trial)", fontsize=9)
    ax.set_title("RD-2 — within-condition nested choice models "
                 "(bars: mice;  thick line: cohort mean)\n"
                 "framework: M1−M0 > 0,  M2−M1 ≈ 0 (vs SBC > 0),  M3−M1 ≈ 0",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, ncol=2, framealpha=.6)
    fig.tight_layout()
    _save_fig(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Network-model positive control
# ---------------------------------------------------------------------------
#
# The si_network_model cohort is a system where the template-cosine mechanism is
# true *by construction* (choice is generated from the prototype mean-SI
# accumulator; V1 relaxes to a fixed point so there is no within-trial
# sampling).  Running the identical estimators there is the specificity anchor:
#   - M1−M0 should fire strongly (choice IS mean-SI driven),
#   - M2−M1 ≈ 0 (no sampling → within-trial variance is uninformative),
#   - M3−M1 ≈ 0 (template reader → whitening adds nothing),
#   - RD-1 Δstim ≈ 0 (the model's noise is information-limiting, so whitening
#     buys nothing and cosine is already near-optimal — this is *why* the
#     SI Network Model REPORT found ~96% decision efficiency).
# Contrast that with the real data, where whitening *does* buy stimulus accuracy
# yet choice still ignores it.


def network_control_arrays(checkpoint_dir: Path) -> list[dict]:
    """Load si_network_model eval blocks → the array bundle each test consumes.

    Requires the repository root on ``sys.path`` so the pickles' ``Config`` /
    ``AnimalRun`` classes resolve.  Returns [] if the checkpoints are absent.
    """
    import glob
    import pickle
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    pkls = sorted(glob.glob(str(Path(checkpoint_dir) / "animal_*.pkl")))
    out = []
    for pk in pkls:
        try:
            ev = pickle.load(open(pk, "rb"))["run"].eval
        except Exception:
            continue
        ori = ev["orientation"]
        out.append({
            "name": Path(pk).stem,
            "X_bar": ev["r_bar"].astype(float),
            "y_stim": (ori < 45).astype(int),
            "y_choice": ev["choice"].astype(int),
            "signed_contrast": ev["contrast"] * np.sign(45.0 - ori),
            "si_mean": ev["mean_si"].astype(float),
            "si_var": ev["var_si"].astype(float),
        })
    return out


def render_control_summary(rd1_real, rd2_real, rd1_net, rd2_net,
                           out_dir: Path, stem="control_real_vs_network"):
    """Two panels: RD-2 ΔLL contrasts and RD-1 Δauc, real data vs network."""
    d_keys = ["M1-M0 (meanSI adds | Pred16)", "M2-M1 (varSI adds | SBC wedge)",
              "M3-M1 (whitened adds | premise)"]
    d_short = ["M1−M0\nmean SI", "M2−M1\nvar SI\n(SBC)", "M3−M1\nwhitened\n(premise)"]

    def _coh(rd, key):
        v = [rd[k]["delta"][key] for k in rd if rd[k].get("delta")]
        return (np.nanmean(v), np.nanstd(v) / np.sqrt(max(len(v), 1)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(len(d_keys)); w = 0.36
    real_m = [_coh(rd2_real, k) for k in d_keys]
    net_m = [_coh(rd2_net, k) for k in d_keys]
    axL.bar(x - w / 2, [m[0] for m in real_m], w, yerr=[m[1] for m in real_m],
            label="real V1 (n=6)", color=WARM, capsize=3)
    axL.bar(x + w / 2, [m[0] for m in net_m], w, yerr=[m[1] for m in net_m],
            label="network model (n=10)", color=COOL, capsize=3)
    axL.axhline(0, color="k", lw=.8)
    axL.set_xticks(x); axL.set_xticklabels(d_short, fontsize=8.5)
    axL.set_ylabel("held-out ΔLL (nats/trial)", fontsize=9)
    axL.set_title("RD-2 nested choice models", fontsize=10)
    axL.legend(fontsize=7.5)

    # RD-1 Δstim / Δchoice
    def _coh1(rd, key):
        v = [rd[k][key] for k in rd]
        return (np.nanmean(v), np.nanstd(v) / np.sqrt(max(len(v), 1)))
    cats = ["Δstim\n(whiten−tmpl)", "Δchoice\n(whiten−tmpl)"]
    xk = np.arange(2)
    rkeys = ["d_auc_stim_whiten", "d_auc_choice_whiten"]
    real1 = [_coh1(rd1_real, k) for k in rkeys]
    net1 = [_coh1(rd1_net, k) for k in rkeys]
    axR.bar(xk - w / 2, [m[0] for m in real1], w, yerr=[m[1] for m in real1],
            label="real V1", color=WARM, capsize=3)
    axR.bar(xk + w / 2, [m[0] for m in net1], w, yerr=[m[1] for m in net1],
            label="network model", color=COOL, capsize=3)
    axR.axhline(0, color="k", lw=.8)
    axR.set_xticks(xk); axR.set_xticklabels(cats, fontsize=8.5)
    axR.set_ylabel("Δ CV AUC (λ=0 − λ=1)", fontsize=9)
    axR.set_title("RD-1 effect of whitening", fontsize=10)
    axR.legend(fontsize=7.5)
    fig.suptitle("Positive control: real V1 vs the template-cosine network model\n"
                 "real V1 has exploitable covariance (RD-1 Δstim>0) that choice ignores "
                 "(M3≈0) — the framework's discarded-covariance cost, on data",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_readout_tests(*, data_path: Path = DATA_PATH,
                      out_dir: Path | None = None, smoke: bool = False) -> dict:
    """Run RD-1 and RD-2 on the real data; write figures + a meta.json."""
    out_dir = Path(out_dir) if out_dir else (
        Path(__file__).resolve().parent / "figures" / "similarity_framework"
        / "F_readout_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    mice, _ = _filter_mice(load_all_mice(data_path))
    if smoke:
        mice = mice[:2]
    rd1: dict = {}
    rd2: dict = {}
    for m in mice:
        name = getattr(m, "name", getattr(m, "tag", "?"))
        bundle = arrays_from_mouse(m)
        if bundle is None:
            continue
        rd1[name] = rd1_template_vs_whitened(
            bundle["X_bar"], bundle["y_stim"], bundle["y_choice"])
        rd2[name] = rd2_nested_choice_models(
            bundle["X_bar"], bundle["y_stim"], bundle["y_choice"],
            bundle["signed_contrast"], bundle["si_mean"], bundle["si_var"])

    if rd1:
        render_rd1(rd1, out_dir)
    if rd2:
        render_rd2(rd2, out_dir)

    # network-model positive control (symmetric cohort)
    rd1_net: dict = {}
    rd2_net: dict = {}
    ckpt = (Path(__file__).resolve().parent.parent / "si_network_model"
            / "results" / "checkpoints" / "symmetric")
    for b in network_control_arrays(ckpt):
        rd1_net[b["name"]] = rd1_template_vs_whitened(
            b["X_bar"], b["y_stim"], b["y_choice"])
        rd2_net[b["name"]] = rd2_nested_choice_models(
            b["X_bar"], b["y_stim"], b["y_choice"],
            b["signed_contrast"], b["si_mean"], b["si_var"])
    if rd1_net and rd1:
        render_control_summary(rd1, rd2, rd1_net, rd2_net, out_dir)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return [None if not np.isfinite(x) else round(float(x), 5) for x in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    meta = {"rd1": _clean(rd1), "rd2": _clean(rd2),
            "rd1_network": _clean(rd1_net), "rd2_network": _clean(rd2_net),
            "n_mice": len(rd1), "n_network": len(rd1_net),
            "generated": time.strftime("%Y-%m-%d %H:%M")}
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    meta = run_readout_tests(data_path=args.data, out_dir=args.out, smoke=args.smoke)
    print(json.dumps({"n_mice": meta["n_mice"],
                      "rd2_delta_example": next(iter(meta["rd2"].values()))["delta"]
                      if meta["rd2"] else {}}, indent=2))


if __name__ == "__main__":
    main()
