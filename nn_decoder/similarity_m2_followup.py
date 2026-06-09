"""Follow-up battery: is the Cb17/Cb22 within-trial-variance choice signal a
genuine within-trial-sampling (SBC) effect, or the variance merely substituting
for a weak trial-mean readout?

Context.  In ``similarity_readout_tests.py`` RD-2, the nested step M2−M1 (does
the within-trial variance ``Var_t[SI]`` add held-out choice log-likelihood
*beyond* the trial-mean ``E_t[SI]`` and the stimulus?) is ≈0 in 4/6 mice and in
all 10 template-cosine network animals, but positive in **Cb17 (+0.017)** and
**Cb22 (+0.011)** — precisely the two mice where the mean readout (M1−M0) is
weakest.  Two readings:

  A (SBC / genuine sampling): r(t) traces a sequence of within-trial states whose
    spread carries choice-relevant uncertainty the trial-mean cannot capture.
  B (variance-as-alternative-summary): the template mean is just a poor readout in
    those mice, so *any* second neural feature helps; the variance is incidental.

This module runs the controls that discriminate A from B, all on one shared,
leakage-free CV scaffold so the numbers are directly comparable:

  C1  variance-specific?   — swap Var_t[SI] for generic 2nd features (‖r‖,
       whitened-mean, PC2-mean, mean SI under other archetype variants, mean SI²)
       and compare each one's unique ΔLL over M1.
  C2  survives a strong mean? — replace the weak template mean with progressively
       stronger means (whitened decoder, IO log-odds, all-of-them) and ask whether
       Var_t[SI] still adds.  B predicts the variance effect shrinks to 0 as the
       mean readout strengthens; A predicts it survives.
  C3  directional or uncertainty? — does Var_t[SI] predict choice *correctness*
       (SBC: posterior-width → more errors) rather than a directional Go-bias?
  C4  confidence link — partial correlation of Var_t[SI] with the IO decision
       entropy within signed-contrast level (SBC: variance ↔ posterior width).
  C5  within-trial dynamics — lag-1 autocorrelation / effective-sample-size of
       SI(t) across xG, real mice vs the fixed-point network (the substrate SBC
       needs; a fixed-point reader has none).
  C6  window robustness — recompute over stimulus-driven xG bins only.
  C7  mean–variance coupling — repeat with the *unbounded* raw template-projection
       variance (bounded SI mechanically suppresses variance near ±1).

Run:  python similarity_m2_followup.py [--out PATH]
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
    DATA_PATH, GO_BOUNDARY_DEG, _filter_mice, _save_fig,
    _zscore_neurons, compute_features, load_all_mice,
)
import similarity_analysis as sim
from similarity_readout_tests import (
    class_direction, shrunk_cov_ledoit_wolf, whitened_direction,
    network_control_arrays, EPS, N_FOLDS, SEED,
)

FOCUS = ("Cb17", "Cb22")          # the two mice with the M2 signal


# ---------------------------------------------------------------------------
# CV log-likelihood on an explicit feature matrix (shared scaffold)
# ---------------------------------------------------------------------------


def _cv_loglik(X: np.ndarray, y: np.ndarray, folds) -> float:
    """Mean held-out logistic LL (nats/trial).  k=0 → intercept-only base rate."""
    from sklearn.linear_model import LogisticRegression
    ll, n = 0.0, 0
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        if X.shape[1] == 0:
            p1 = float(np.clip(y[tr].mean(), 1e-6, 1 - 1e-6))
            lp = y[te] * np.log(p1) + (1 - y[te]) * np.log(1 - p1)
        else:
            mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + EPS
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            clf.fit((X[tr] - mu) / sd, y[tr])
            p = np.clip(clf.predict_proba((X[te] - mu) / sd)[:, 1], 1e-6, 1 - 1e-6)
            lp = y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)
        ll += float(np.sum(lp)); n += len(te)
    return ll / (n + EPS) if n else float("nan")


def _delta(feat: dict, base: list[str], add: str, y, folds) -> float:
    """Held-out ΔLL of adding feature ``add`` to the ``base`` model."""
    def _mat(keys):
        return (np.column_stack([feat[k] for k in keys]) if keys
                else np.empty((len(y), 0)))
    return _cv_loglik(_mat(base + [add]), y, folds) - _cv_loglik(_mat(base), y, folds)


# ---------------------------------------------------------------------------
# Per-fold (leakage-free) directional readouts filled into full-length arrays
# ---------------------------------------------------------------------------


def _fill_perfold(X_bar, y_stim, y, folds, kind: str) -> np.ndarray:
    """Whitened-Δμ ('whitened') or 2nd-PC ('pc2') trial-mean projection.

    Fit on each fold's *training* trials, applied to its test trials, so every
    trial's value comes from a model that never saw it.
    """
    out = np.full(y.shape, np.nan)
    for tr, te in folds:
        mu = X_bar[tr].mean(0, keepdims=True); sd = X_bar[tr].std(0, keepdims=True) + EPS
        Xtr = (X_bar[tr] - mu) / sd; Xte = (X_bar[te] - mu) / sd
        if kind == "whitened":
            if len(np.unique(y_stim[tr])) < 2:
                continue
            dmu = class_direction(Xtr, y_stim[tr])
            Sigma = shrunk_cov_ledoit_wolf(Xtr, y_stim[tr])
            w = whitened_direction(dmu, Sigma, 0.0)
            if np.corrcoef(Xtr @ w, y_stim[tr])[0, 1] < 0:
                w = -w
            out[te] = Xte @ w
        elif kind == "pc2":
            from sklearn.decomposition import PCA
            pcs = PCA(n_components=2, random_state=0).fit(Xtr)
            out[te] = Xte @ pcs.components_[1]
    return out


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def assemble(m_or_net, *, is_network=False) -> dict | None:
    """Build the full per-trial feature bundle + the CV folds.

    Works on a real ``MouseData`` or a network ``arrays`` dict.  All neural
    features are leakage-controlled; ``si_full`` (n, T) drives the within-trial
    statistics.
    """
    from sklearn.model_selection import StratifiedKFold

    if is_network:
        b = m_or_net
        X_bar = b["X_bar"]; y_stim = b["y_stim"]; y = b["y_choice"].astype(int)
        c = b["signed_contrast"]
        si = b["si_full"]                                   # (n, T) prototype SI
        raw_proj_var = np.nanvar(si, axis=1)                # network: reuse si
        io_logodds = np.full(len(y), np.nan)                # no IO for network
        H_dec = np.full(len(y), np.nan)
        correct = (y == y_stim).astype(float)
        rnorm = np.linalg.norm(X_bar, axis=1)
        si_methods = {"prototype": si}
        name = b["name"]
    else:
        m = m_or_net
        proto = sim._compute_archetypes(m); ex = sim._compute_archetypes_exemplars(m)
        if "Go" not in proto or "NoGo" not in proto:
            return None
        Az = _zscore_neurons(m.activity)                    # (n, p, T)
        X_bar = Az.mean(axis=2)
        feats = compute_features(m)
        c = feats["signed_contrast"]; y = m.choice.astype(int)
        y_stim = (m.orientation < GO_BOUNDARY_DEG).astype(int)
        si = sim._si_per_method(m, proto, ex, method="prototype")     # (n, T)
        si_methods = {"prototype": si,
                      "expected_exemplar": sim._si_per_method(m, proto, ex, method="expected_exemplar"),
                      "vmf": sim._si_per_method(m, proto, ex, method="vmf")}
        # unbounded raw SI analog = cos(r,Go) − cos(r,NoGo), direction-only (no
        # bounded-SI denominator) → isolates the bounding nonlinearity from any
        # genuine within-trial neural dispersion.
        scos, snames = sim._archetype_similarity(m, proto)  # (n, T, 2)
        gi, ni = snames.index("Go"), snames.index("NoGo")
        raw_si = scos[..., gi] - scos[..., ni]              # (n, T)
        raw_proj_var = np.nanvar(raw_si, axis=1)
        p_go = np.clip(feats["p_go"], 1e-6, 1 - 1e-6)
        io_logodds = np.log(p_go / (1 - p_go))
        H_dec = feats["H_dec"]
        correct = feats["correct"].astype(float)
        rnorm = np.linalg.norm(m.activity.mean(axis=2), axis=1)   # raw magnitude
        name = getattr(m, "name", getattr(m, "tag", "?"))

    valid = (np.isfinite(c) & np.isfinite(y) & np.isin(y, [0, 1])
             & np.isfinite(X_bar).all(axis=1) & np.isfinite(si).all(axis=1))
    idx = np.where(valid)[0]
    if idx.size < 60 or len(np.unique(y[idx])) < 2:
        return None

    def s(a):
        return np.asarray(a, float)[idx]
    X_bar = X_bar[idx]; y = y[idx].astype(int); y_stim = y_stim[idx].astype(int)
    si = si[idx]
    mean_si = np.nanmean(si, axis=1); var_si = np.nanvar(si, axis=1)
    # stimulus-driven window (drop first 25% of bins) for C6
    T = si.shape[1]; w0 = T // 4
    var_si_win = np.nanvar(si[:, w0:], axis=1)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_bar, y))
    whit = _fill_perfold(X_bar, y_stim, y, folds, "whitened")
    pc2 = _fill_perfold(X_bar, y_stim, y, folds, "pc2")

    feat = {
        "c": s(c), "y": y, "y_stim": y_stim,
        "mean_si": mean_si, "var_si": var_si, "var_si_win": var_si_win,
        "mean_si_sq": mean_si ** 2, "abs_mean_si": np.abs(mean_si),
        "rnorm": s(rnorm), "whitened": whit, "pc2": pc2,
        "io_logodds": s(io_logodds), "H_dec": s(H_dec),
        "correct": s(correct), "raw_proj_var": s(raw_proj_var),
    }
    for k, v in si_methods.items():
        feat[f"mean_si_{k}"] = np.nanmean(v[idx], axis=1)
    return {"name": name, "feat": feat, "folds": folds, "si": si,
            "n": int(idx.size)}


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------


def run_battery_on(bundle: dict, *, is_network=False) -> dict:
    feat = bundle["feat"]; folds = bundle["folds"]; y = feat["y"]
    out = {"name": bundle["name"], "n": bundle["n"]}

    # baseline M1−M0 and M2−M1
    out["M1-M0_meanSI"] = _delta(feat, ["c"], "mean_si", y, folds)
    out["M2-M1_varSI"] = _delta(feat, ["c", "mean_si"], "var_si", y, folds)

    # C1 — variance-specific? unique ΔLL of each candidate 2nd feature over M1
    base = ["c", "mean_si"]
    c1 = {}
    for k in ["var_si", "mean_si_sq", "rnorm", "whitened", "pc2"]:
        c1[k] = _delta(feat, base, k, y, folds)
    if not is_network:
        for k in ["mean_si_expected_exemplar", "mean_si_vmf"]:
            c1[k] = _delta(feat, base, k, y, folds)
    out["C1_second_feature_dLL"] = c1

    # C2 — does var_SI survive a stronger mean readout?
    c2 = {}
    c2["over_templateMean"] = _delta(feat, ["c", "mean_si"], "var_si", y, folds)
    c2["over_whitenedMean"] = _delta(feat, ["c", "whitened"], "var_si", y, folds)
    if not is_network:
        c2["over_IOlogodds"] = _delta(feat, ["c", "io_logodds"], "var_si", y, folds)
        c2["over_allStrong"] = _delta(
            feat, ["c", "mean_si", "whitened", "io_logodds"], "var_si", y, folds)
        # the maximal static + nonlinear-mean readout: every mean variant, the
        # whitened decoder, the IO log-odds, and mean nonlinearities. If var_SI
        # dies here, the M2 effect is fully explained by static/mean features.
        c2["over_maximal"] = _delta(
            feat, ["c", "mean_si", "mean_si_sq", "abs_mean_si",
                   "mean_si_vmf", "mean_si_expected_exemplar",
                   "whitened", "io_logodds"], "var_si", y, folds)
        # + IO decision entropy & |log-odds|: fully controls residual stimulus
        # *difficulty* (not just signed evidence). If var_SI dies here it was a
        # difficulty proxy, not a within-trial signal.
        feat["abs_io"] = np.abs(feat["io_logodds"])
        c2["over_maximal_plus_difficulty"] = _delta(
            feat, ["c", "mean_si", "mean_si_sq", "abs_mean_si",
                   "mean_si_vmf", "mean_si_expected_exemplar",
                   "whitened", "io_logodds", "abs_io", "H_dec"], "var_si", y, folds)
    else:
        c2["over_allStrong"] = _delta(
            feat, ["c", "mean_si", "whitened"], "var_si", y, folds)
        c2["over_maximal"] = _delta(
            feat, ["c", "mean_si", "mean_si_sq", "abs_mean_si", "whitened"],
            "var_si", y, folds)
    out["C2_varSI_survives_strong_mean"] = c2

    # C3 — directional or uncertainty? does var_SI predict *correctness*?
    c3 = {}
    if np.isfinite(feat["correct"]).all() and len(np.unique(feat["correct"])) == 2:
        cy = feat["correct"].astype(int)
        # control for |stimulus| (difficulty); var over (|c| + mean|SI|)
        absc = np.abs(feat["c"]); absmean = feat["abs_mean_si"]
        fc = {"absc": absc, "absmean": absmean, "var_si": feat["var_si"], "y": cy}
        foldsc = bundle["folds"]
        c3["varSI_adds_to_correctness"] = (
            _cv_loglik(np.column_stack([fc["absc"], fc["absmean"], fc["var_si"]]), cy, foldsc)
            - _cv_loglik(np.column_stack([fc["absc"], fc["absmean"]]), cy, foldsc))
        # sign of the within-sample correlation var_si vs correct (SBC: negative)
        m = np.isfinite(feat["var_si"]) & np.isfinite(cy)
        c3["corr_varSI_correct"] = float(np.corrcoef(feat["var_si"][m], cy[m])[0, 1])
    out["C3_directional_vs_uncertainty"] = c3

    # C4 — confidence link: partial corr(var_SI, IO entropy) within contrast level
    if not is_network and np.isfinite(feat["H_dec"]).any():
        c = feat["c"]; vs = feat["var_si"]; hd = feat["H_dec"]
        # residualise both on signed_contrast (categorical levels), then correlate
        rv, rh = vs.copy().astype(float), hd.copy().astype(float)
        for lv in np.unique(c[np.isfinite(c)]):
            mk = (c == lv) & np.isfinite(vs) & np.isfinite(hd)
            if mk.sum() >= 5:
                rv[mk] -= np.nanmean(vs[mk]); rh[mk] -= np.nanmean(hd[mk])
        mk = np.isfinite(rv) & np.isfinite(rh)
        out["C4_partialcorr_varSI_IOentropy"] = float(np.corrcoef(rv[mk], rh[mk])[0, 1])
    else:
        out["C4_partialcorr_varSI_IOentropy"] = float("nan")

    # C6 — stimulus-driven window robustness
    out["C6_varSI_window_dLL"] = _delta(feat, ["c", "mean_si"], "var_si_win", y, folds)

    # C7 — unbounded raw-projection variance (no mean–variance coupling)
    out["C7_rawProjVar_dLL"] = _delta(feat, ["c", "mean_si"], "raw_proj_var", y, folds)

    # C5 — within-trial temporal dynamics of SI(t)
    out["C5_dynamics"] = within_trial_dynamics(bundle["si"])

    # C8 — permutation null (C2[+IO]) + bootstrap CIs on C3/C4 (critic's gaps)
    if not is_network:
        out["C8_perm_boot"] = permutation_and_bootstrap(bundle)
    return out


def permutation_and_bootstrap(bundle: dict, *, n_perm=300, n_boot=1000,
                              seed=0) -> dict:
    """Calibrate the C2[+IO] var_SI ΔLL against a within-condition shuffle null,
    and bootstrap-CI the C3/C4 sign signatures (the critic's two key gaps).

    Permutation: shuffle ``var_si`` across trials *within each signed_contrast
    level* (breaks the var↔choice link, preserves the marginal and stimulus
    structure), recompute held-out ΔLL of +var_si over [c, IO log-odds].  Real
    ΔLL outside the null band ⇒ genuine signal, not a suppressor phantom.
    """
    feat = bundle["feat"]; folds = bundle["folds"]; y = feat["y"]
    rng = np.random.default_rng(seed)
    out = {}

    if np.isfinite(feat["io_logodds"]).any():
        base = ["c", "io_logodds"]
        real = _delta(feat, base, "var_si", y, folds)
        c = feat["c"]; v0 = feat["var_si"].copy()
        levels = np.unique(c[np.isfinite(c)])
        null = np.empty(n_perm)
        for p in range(n_perm):
            vp = v0.copy()
            for lv in levels:
                m = np.where(c == lv)[0]
                vp[m] = v0[rng.permutation(m)]
            f2 = dict(feat); f2["var_si_perm"] = vp
            null[p] = _delta(f2, base, "var_si_perm", y, folds)
        out["C2_IO"] = {
            "real_dLL": float(real),
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "p_right": float((np.sum(null >= real) + 1) / (n_perm + 1)),
            "z": float((real - np.mean(null)) / (np.std(null) + EPS))}

    # bootstrap CIs for C3 (corr var,correct) and C4 (partial r var,IO entropy)
    def _boot_corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        n = a.shape[0]
        if n < 30:
            return None
        bs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            bs[i] = np.corrcoef(a[idx], b[idx])[0, 1]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    out["C3_corr_ci"] = _boot_corr(feat["var_si"], feat["correct"])
    # C4: residualise var & IO-entropy on signed_contrast first
    if np.isfinite(feat["H_dec"]).any():
        c = feat["c"]; vs = feat["var_si"].astype(float); hd = feat["H_dec"].astype(float)
        rv, rh = vs.copy(), hd.copy()
        for lv in np.unique(c[np.isfinite(c)]):
            mk = (c == lv) & np.isfinite(vs) & np.isfinite(hd)
            if mk.sum() >= 5:
                rv[mk] -= np.nanmean(vs[mk]); rh[mk] -= np.nanmean(hd[mk])
        out["C4_partialcorr_ci"] = _boot_corr(rv, rh)
    return out


def within_trial_dynamics(si: np.ndarray) -> dict:
    """Lag-1 autocorrelation and effective-sample-size of SI(t) across xG.

    A fixed-point reader has SI(t) nearly constant after onset → ρ₁→1, ESS→1.
    Genuine within-trial dynamics → lower ρ₁, ESS>1 (more independent samples).
    """
    s = np.asarray(si, float)
    n, T = s.shape
    rhos = []
    for i in range(n):
        x = s[i]
        x = x[np.isfinite(x)]
        if x.size < 4 or np.std(x) < 1e-9:
            continue
        a, b = x[:-1], x[1:]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        rhos.append(np.corrcoef(a, b)[0, 1])
    rho = float(np.nanmedian(rhos)) if rhos else float("nan")
    ess = float(T * (1 - rho) / (1 + rho)) if np.isfinite(rho) and rho < 1 else float("nan")
    # mean within-trial first-difference variance (temporal roughness), normalised
    diffs = np.diff(s, axis=1)
    rough = float(np.nanmedian(np.nanvar(diffs, axis=1) / (np.nanvar(s, axis=1) + EPS)))
    return {"lag1_autocorr": rho, "ess_per_trial": ess, "rel_roughness": rough,
            "n_xG": int(T)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render(real: list[dict], net: list[dict], out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Panel A — C2: does var_SI survive stronger means? (focus mice + cohort)
    ax = axes[0, 0]
    keys = ["over_templateMean", "over_whitenedMean", "over_IOlogodds",
            "over_maximal", "over_maximal_plus_difficulty"]
    klab = ["+template\nmean", "+whitened\nmean", "+IO\nlog-odds",
            "+maximal\nstatic", "+full\ndifficulty"]
    x = np.arange(len(keys))
    for r in real:
        v = [r["C2_varSI_survives_strong_mean"].get(k, np.nan) for k in keys]
        style = dict(marker="o", lw=2) if r["name"] in FOCUS else dict(marker=".", lw=.8, alpha=.5)
        ax.plot(x, v, label=r["name"], **style)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(klab, fontsize=8)
    ax.set_ylabel("ΔLL of +Var$_t$[SI] (nats/trial)", fontsize=9)
    ax.set_title("C2 — does the variance survive a stronger mean readout?\n"
                 "B predicts → 0 as the mean strengthens", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)

    # Panel B — C1: variance-specific vs generic 2nd features (focus mice)
    ax = axes[0, 1]
    foc = [r for r in real if r["name"] in FOCUS]
    cand = ["var_si", "raw_proj_var?", "mean_si_sq", "rnorm", "whitened", "pc2",
            "mean_si_expected_exemplar", "mean_si_vmf"]
    # use actual keys present
    cand = ["var_si", "mean_si_sq", "rnorm", "whitened", "pc2",
            "mean_si_expected_exemplar", "mean_si_vmf"]
    xb = np.arange(len(cand)); w = 0.38
    for i, r in enumerate(foc):
        v = [r["C1_second_feature_dLL"].get(k, np.nan) for k in cand]
        ax.bar(xb + (i - .5) * w, v, w, label=r["name"])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(xb)
    ax.set_xticklabels([c.replace("mean_si_", "").replace("_", "\n") for c in cand],
                       fontsize=7, rotation=0)
    ax.set_ylabel("unique ΔLL over M1", fontsize=9)
    ax.set_title("C1 — is the variance special, or does any 2nd feature help?",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Panel C — C5: within-trial dynamics, real vs network
    ax = axes[1, 0]
    rn = [r["C5_dynamics"]["lag1_autocorr"] for r in real]
    nn = [r["C5_dynamics"]["lag1_autocorr"] for r in net]
    names = [r["name"] for r in real]
    cols = ["#c44e1e" if n in FOCUS else "#888" for n in names]
    ax.bar(range(len(rn)), rn, color=cols, label="real mice")
    ax.axhline(np.nanmean(nn), color="#2f6e8f", ls="--", lw=2,
               label=f"network mean ({np.nanmean(nn):.2f})")
    ax.set_xticks(range(len(rn))); ax.set_xticklabels(names, fontsize=8, rotation=45)
    ax.set_ylabel("lag-1 autocorr of SI(t)", fontsize=9)
    ax.set_title("C5 — within-trial dynamics (lower = more 'samples')\n"
                 "orange = Cb17/Cb22; dashed = fixed-point network", fontsize=9)
    ax.legend(fontsize=7)

    # Panel D — C3/C4 directional-vs-uncertainty + confidence link
    ax = axes[1, 1]
    names = [r["name"] for r in real]
    c3v = [r["C3_directional_vs_uncertainty"].get("varSI_adds_to_correctness", np.nan)
           for r in real]
    c4v = [r["C4_partialcorr_varSI_IOentropy"] for r in real]
    xb = np.arange(len(names)); w = 0.38
    ax.bar(xb - w / 2, c3v, w, label="C3 ΔLL var→correctness", color="#7a5fa3")
    ax.bar(xb + w / 2, c4v, w, label="C4 partial r(var,IO-entropy)", color="#3a9")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(xb); ax.set_xticklabels(names, fontsize=8, rotation=45)
    ax.set_title("C3/C4 — uncertainty signature?\n"
                 "SBC: var predicts errors (C3>0) & tracks IO entropy (C4>0)",
                 fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("M2 follow-up — is the Cb17/Cb22 within-trial-variance choice "
                 "signal genuine sampling (A) or a weak-mean substitute (B)?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, out_dir / "m2_followup_battery")


def render_verdict(real: list[dict], out_dir: Path):
    """Clean 3-panel verified verdict figure (the one embedded in the conjecture)."""
    names = [r["name"] for r in real]
    cols = ["#c44e1e" if n in FOCUS else "#888" for n in names]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A — C2[+IO] real ΔLL vs within-condition shuffle null
    real_d = [r["C8_perm_boot"]["C2_IO"]["real_dLL"] * 1000 for r in real]
    nmean = [r["C8_perm_boot"]["C2_IO"]["null_mean"] * 1000 for r in real]
    nsd = [r["C8_perm_boot"]["C2_IO"]["null_sd"] * 1000 for r in real]
    x = np.arange(len(names))
    a.bar(x, real_d, color=cols, zorder=3)
    a.errorbar(x, nmean, yerr=[2 * s for s in nsd], fmt="_", color="k", lw=1.5,
               capsize=4, label="shuffle null ±2σ", zorder=4)
    a.axhline(0, color="k", lw=.6)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=8, rotation=45)
    a.set_ylabel("var$_t$[SI] ΔLL over [c, IO log-odds] (×10³)", fontsize=8.5)
    a.set_title("A. Survives a strong readout? (vs within-condition shuffle null)\n"
                "only Cb17 clears the null (z=10.5); Cb22 dies (artifact)", fontsize=8.5)
    a.legend(fontsize=7)

    # Panel B — C4 partial corr(var, IO entropy) with bootstrap CI: wrong sign for SBC
    c4 = [r["C8_perm_boot"].get("C4_partialcorr_ci") for r in real]
    for i, (ci, col) in enumerate(zip(c4, cols)):
        if ci:
            b.plot([i, i], ci, color=col, lw=3, solid_capstyle="round")
            b.plot(i, (ci[0] + ci[1]) / 2, "o", color=col, ms=5)
    b.axhline(0, color="k", lw=.8)
    b.set_xticks(x); b.set_xticklabels(names, fontsize=8, rotation=45)
    b.set_ylabel("partial r(var$_t$SI, IO entropy) | contrast", fontsize=8.5)
    b.set_title("B. SBC content check (bootstrap 95% CI)\n"
                "negative in all 6 (CI excludes 0): variance tracks CONFIDENCE,\n"
                "the WRONG sign for SBC posterior-width", fontsize=8.5)

    # Panel C — survival ladder, Cb17 vs Cb22
    ladder = ["over_templateMean", "over_whitenedMean", "over_IOlogodds"]
    llab = ["+template\nmean", "+whitened\ndecoder", "+IO\nlog-odds"]
    xl = np.arange(len(ladder))
    for r in real:
        if r["name"] in FOCUS:
            v = [r["C2_varSI_survives_strong_mean"][k] * 1000 for k in ladder]
            c.plot(xl, v, "-o", lw=2.5, ms=6, label=r["name"],
                   color="#c44e1e" if r["name"] == "Cb17" else "#2f6e8f")
    c.axhline(0, color="k", lw=.8)
    c.set_xticks(xl); c.set_xticklabels(llab, fontsize=8)
    c.set_ylabel("var$_t$[SI] ΔLL (×10³)", fontsize=8.5)
    c.set_title("C. The two mice diverge under a strong mean\n"
                "Cb17 survives; Cb22 collapses → artifact", fontsize=8.5)
    c.legend(fontsize=8)

    fig.suptitle("Verified verdict — the apparent SBC signal is one artifact (Cb22) + one "
                 "genuine-but-non-SBC within-trial signal (Cb17); no posterior-width SBC signature in any mouse",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out_dir / "m2_verdict")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else (
        Path(__file__).resolve().parent / "figures" / "similarity_framework"
        / "F_readout_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    mice, _ = _filter_mice(load_all_mice(DATA_PATH))
    real = []
    for m in mice:
        b = assemble(m, is_network=False)
        if b:
            real.append(run_battery_on(b, is_network=False))

    ckpt = (Path(__file__).resolve().parent.parent / "si_network_model"
            / "results" / "checkpoints" / "symmetric")
    net = []
    for arr in network_control_arrays(ckpt):
        # attach si_full (prototype) for within-trial stats
        import pickle
        pk = ckpt / f"{arr['name']}.pkl"
        ev = pickle.load(open(pk, "rb"))["run"].eval
        arr["si_full"] = ev["si_full_prototype"].astype(float)
        b = assemble(arr, is_network=True)
        if b:
            net.append(run_battery_on(b, is_network=True))

    if real:
        render(real, net, out_dir)
        if all("C8_perm_boot" in r for r in real):
            render_verdict(real, out_dir)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    meta = {"real": _clean(real), "network": _clean(net),
            "generated": time.strftime("%Y-%m-%d %H:%M")}
    (out_dir / "_m2_followup_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    meta = run(out_dir=args.out)
    # compact console summary
    for r in meta["real"]:
        print(f"{r['name']:6} n={r['n']:4d}  M1-M0={r['M1-M0_meanSI']:+.4f}  "
              f"M2-M1={r['M2-M1_varSI']:+.4f}  "
              f"C2[+IO]={r['C2_varSI_survives_strong_mean'].get('over_IOlogodds'):+.4f}  "
              f"C2[+all]={r['C2_varSI_survives_strong_mean'].get('over_allStrong'):+.4f}  "
              f"C5_ac={r['C5_dynamics']['lag1_autocorr']:+.3f}")


if __name__ == "__main__":
    main()
