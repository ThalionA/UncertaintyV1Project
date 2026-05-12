# -*- coding: utf-8 -*-
"""Exploratory dim-reduction sweep over V1 population activity.

For every animal in the export, this script projects (n_trials × n_neurons × n_xG)
spike activity into low-D using a battery of linear and nonlinear methods, then
renders five families of figures:

  Style A   — single-trial 2D scatter (mean over xG); methods: PCA, Isomap
  Style A3D — single-trial 3D scatter (PCA only)
  Style B   — condition-averaged 2D trajectories over xG; methods: PCA, dPCA
  Style B3D — same in 3D
  Style C   — single-trial 2D trajectory lines (PCA only)
  Style D   — CEBRA multi-session shared 3D embedding (all mice in one space)
  Style E   — cross-mouse alignment: Procrustes overlay + task-axis regression

Styles A–D render one panel per mouse in a grid; Style E overlays all mice in
a single panel for direct comparison.

Group/colour variables include true orientation, signed contrast/dispersion
(value × sign(45° − stimulus)), choice, correct, P(Go), decision entropy, and
perceptual-posterior linear sd — both as continuous colour and as tertile
splits (lo / med / hi).

Outputs:
  figures/dim_reduction/<style>/<method>__<grouping>[_3d].{png,svg}
  figures/dim_reduction/_meta.json   (run config + per-mouse n_trials, n_neurons)

Run
---
Full sweep:
    python -m nn_decoder.dim_reduction_explore

Smoke test (one mouse, one combo each style, fastest path):
    python -m nn_decoder.dim_reduction_explore --smoke

Dependencies
------------
Required: numpy, scipy, matplotlib, scikit-learn.
Optional (gracefully skipped if missing): dPCA (for proper dPCA — falls back
to marginalisation if absent), cebra (for Style D — Style D is skipped
entirely if missing).

Notes
-----
* Z-scoring: neurons are z-scored across stacked (trials × xG) samples before
  DR. Intentional for population-geometry analysis; this is **not** the same
  z-scoring that breaks the IO fitter (per-animal z-scoring of synthetic
  velocity/licks). That constraint is about IO inputs, not neural pop DR.
* Within-mouse projections in A/B/C use that mouse's own PCA basis. Cross-mouse
  comparison lives in Style E (Procrustes alignment of cond-avg trajectories,
  task-axis regression projecting all mice into a shared task-defined space).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scipy.spatial import procrustes as _procrustes

# Optional deps
try:
    from dPCA import dPCA as _dPCA_lib  # noqa: F401
    _HAVE_DPCA = True
except Exception:
    _HAVE_DPCA = False

try:
    import cebra
    _HAVE_CEBRA = True
except Exception:
    _HAVE_CEBRA = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "VR_Decoder_Data_Export.mat"
FIG_ROOT = REPO_ROOT / "figures" / "dim_reduction"

S_GRID = np.arange(0, 91, 1, dtype=float)   # IO orientation grid
GO_BOUNDARY_DEG = 45.0

MIN_TRIALS = 50
MIN_NEURONS = 5

# Subsampling for Style C (single-trial trajectories) — keeps plots legible
STYLE_C_MAX_TRIALS_PER_LEVEL = 80

# Per-mouse trial cap for nonlinear methods (otherwise t-SNE/UMAP get slow)
NONLINEAR_TRIAL_CAP = 1500

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading — load .mat once, slice per animal
# ---------------------------------------------------------------------------

@dataclass
class MouseData:
    tag: str
    activity: np.ndarray          # (n_trials, n_neurons, n_xG)
    xG: np.ndarray                # (n_xG,)
    orientation: np.ndarray       # abs_from_go (categorical-ish but treat as float)
    true_orientation: np.ndarray  # stimulus
    contrast: np.ndarray
    dispersion: np.ndarray
    choice: np.ndarray            # goChoice in {0,1}
    velocity: np.ndarray
    lick_rate: np.ndarray
    post_s_marginal: np.ndarray   # (n_trials, len(S_GRID))
    decision_posterior: np.ndarray  # (n_trials, 2)  -> [P(Go), P(NoGo)]

    @property
    def n_trials(self) -> int:
        return self.activity.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.activity.shape[1]

    @property
    def n_xG(self) -> int:
        return self.activity.shape[2]


def _slice_gspk(raw: np.ndarray, xG_full: np.ndarray, xG_mask: np.ndarray,
                n_trials_expected: int, tag: str) -> np.ndarray:
    """Slice Gspk along the xG axis (detected by length match) and reorder to
    canonical (n_trials, n_neurons, n_xG).

    Why detection: utils_v26.py's loader does a hardcoded transpose(1, 0, 2)
    that assumes Gspk is stored as (n_neurons, n_trials, n_xG_full), but the
    current export comes through as (n_trials, n_neurons, n_xG_full). We trust
    nothing and identify axes by their lengths.
    """
    if raw.ndim != 3:
        raise ValueError(f"[{tag}] Gspk has ndim={raw.ndim}, expected 3")

    xG_full_len = int(xG_full.size)
    xG_axes = [i for i, s in enumerate(raw.shape) if s == xG_full_len]
    if not xG_axes:
        raise ValueError(
            f"[{tag}] no axis of Gspk{raw.shape} matches xG length {xG_full_len}"
        )
    if len(xG_axes) > 1:
        # tie-break: prefer the trailing axis (matlab convention)
        xG_axis = xG_axes[-1]
    else:
        xG_axis = xG_axes[0]

    sliced = np.take(raw, np.where(xG_mask)[0], axis=xG_axis)
    sliced = np.moveaxis(sliced, xG_axis, -1)  # xG is now last

    # Identify trials axis among the remaining two
    trial_axes = [i for i, s in enumerate(sliced.shape[:-1])
                  if s == n_trials_expected]
    if not trial_axes:
        raise ValueError(
            f"[{tag}] no axis of sliced Gspk{sliced.shape[:-1]} matches "
            f"trial-table count {n_trials_expected}"
        )
    trials_axis = trial_axes[0]
    if trials_axis != 0:
        sliced = np.swapaxes(sliced, 0, 1)
    return sliced  # (n_trials, n_neurons, n_xG)


def load_all_mice(filepath: Path = DATA_PATH) -> list[MouseData]:
    """Load the full export once and return a MouseData per animal tag."""
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find export at {filepath}")
    print(f"[load] reading {filepath}")
    mat = sio.loadmat(str(filepath), simplify_cells=True)
    neural_store = mat["NeuralStore"]
    trial_tbl = mat["TrialTbl_Struct"]
    animals = np.asarray(trial_tbl["animal"])

    out: list[MouseData] = []
    for i, ns in enumerate(neural_store):
        tag = ns.get("tag", f"Mouse_{i}") if isinstance(ns, dict) else f"Mouse_{i}"
        mask = (animals == tag)
        n_trials_expected = int(mask.sum())
        if n_trials_expected == 0:
            warnings.warn(f"[load] no trial-table rows match tag={tag}; skipping")
            continue

        xG = np.asarray(ns["xG"]).ravel()
        xG_mask = (xG >= 0.0) & (xG <= 2.0)
        raw = np.asarray(ns["Gspk"])
        try:
            activity = _slice_gspk(raw, xG, xG_mask, n_trials_expected, str(tag))
        except ValueError as e:
            warnings.warn(f"[load] {e}; skipping")
            continue

        def col(name):
            return np.asarray(trial_tbl[name])[mask]

        decp = np.asarray(trial_tbl["decision_posterior"])[mask]
        psm = np.asarray(trial_tbl["post_s_marginal"])[mask]

        md = MouseData(
            tag=str(tag),
            activity=activity,
            xG=xG[xG_mask],
            orientation=col("abs_from_go").astype(float),
            true_orientation=col("stimulus").astype(float),
            contrast=col("contrast").astype(float),
            dispersion=col("dispersion").astype(float),
            choice=col("goChoice").astype(float),
            velocity=col("preRZ_velocity").astype(float),
            lick_rate=col("preRZ_lick_rate").astype(float),
            post_s_marginal=psm.astype(float),
            decision_posterior=decp.astype(float),
        )
        # Alignment guarantee — activity rows must equal trial-table rows
        assert md.activity.shape[0] == md.true_orientation.shape[0], (
            f"[{tag}] activity rows={md.activity.shape[0]} != "
            f"trial-table rows={md.true_orientation.shape[0]}"
        )
        out.append(md)

    print(f"[load] loaded {len(out)} mice "
          f"(trials/units per mouse: "
          f"{[(m.tag, m.n_trials, m.n_neurons) for m in out]})")
    return out


# ---------------------------------------------------------------------------
# Grouping / uncertainty features
# ---------------------------------------------------------------------------

def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _linear_std(post: np.ndarray, grid: np.ndarray = S_GRID) -> np.ndarray:
    """sqrt of linear variance of post over grid (per row)."""
    denom = post.sum(axis=1, keepdims=True) + 1e-12
    mean = (post * grid[None, :]).sum(axis=1, keepdims=True) / denom
    var = (post * (grid[None, :] - mean) ** 2).sum(axis=1) / denom.squeeze(-1)
    return np.sqrt(np.clip(var, 0.0, None))


def _tertile_split(x: np.ndarray, labels=("lo", "med", "hi")) -> np.ndarray:
    """Return categorical tertile labels for finite x; NaN→empty string."""
    out = np.full(x.shape, "", dtype=object)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return out
    qs = np.nanquantile(x[finite], [1/3, 2/3])
    out[finite & (x <= qs[0])] = labels[0]
    out[finite & (x > qs[0]) & (x <= qs[1])] = labels[1]
    out[finite & (x > qs[1])] = labels[2]
    return out


def _ori_tertile(stim: np.ndarray) -> np.ndarray:
    """Tertile by |stimulus - 45|, so 'lo' = close to boundary (ambiguous),
    'hi' = far from boundary (clear Go or NoGo). Uses the metric that makes
    behavioural sense for a Go/NoGo around 45°."""
    return _tertile_split(np.abs(stim - GO_BOUNDARY_DEG))


def compute_features(m: MouseData) -> dict[str, np.ndarray]:
    """All grouping/colour vectors for one mouse, length n_trials.

    Signed variants follow ``value * sign(45 - stimulus)`` so positive means
    Go-side evidence, negative means NoGo-side. Stimuli exactly at the
    boundary contribute 0.
    """
    p_go = m.decision_posterior[:, 0] if m.decision_posterior.shape[1] >= 1 else np.full(m.n_trials, np.nan)
    H_dec = _binary_entropy(p_go)
    sd_perc = _linear_std(m.post_s_marginal, S_GRID)

    correct = ((m.choice == 1) & (m.true_orientation < GO_BOUNDARY_DEG)) | \
              ((m.choice == 0) & (m.true_orientation >= GO_BOUNDARY_DEG))

    side_sign = np.sign(GO_BOUNDARY_DEG - m.true_orientation)
    signed_contrast = m.contrast * side_sign
    signed_dispersion = m.dispersion * side_sign

    feats = {
        # Continuous task variables
        "true_orientation": m.true_orientation,
        "signed_contrast": signed_contrast,
        "signed_dispersion": signed_dispersion,
        "H_dec": H_dec,
        "sd_perc": sd_perc,
        "p_go": p_go,
        # Discrete groupings
        "choice": m.choice,
        "correct": correct.astype(float),
        # Tertile variants (categorical strings)
        "ori_tertile": _ori_tertile(m.true_orientation),
        "signed_contrast_tertile": _tertile_split(signed_contrast),
        "signed_dispersion_tertile": _tertile_split(signed_dispersion),
        "H_dec_tertile": _tertile_split(H_dec),
        "sd_perc_tertile": _tertile_split(sd_perc),
    }
    return feats


# ---------------------------------------------------------------------------
# DR fitters — return either (Z2d_per_trial,) or (Z_traj over xG,)
# ---------------------------------------------------------------------------

def _zscore_neurons(A: np.ndarray) -> np.ndarray:
    """A: (n_trials, n_neurons, n_xG) -> z-scored along (trial, xG) per neuron."""
    n_trials, n_neurons, n_xG = A.shape
    stacked = A.transpose(0, 2, 1).reshape(-1, n_neurons)  # (n_trials*n_xG, n_neurons)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(stacked)
    stacked_z = scaler.transform(stacked)
    return stacked_z.reshape(n_trials, n_xG, n_neurons).transpose(0, 2, 1)


def trial_mean_matrix(A: np.ndarray) -> np.ndarray:
    """A: (n_trials, n_neurons, n_xG) -> (n_trials, n_neurons), mean over xG."""
    return A.mean(axis=2)


def fit_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X)
    return pca.transform(X), pca


def fit_method(name: str, X: np.ndarray, n_components: int = 2,
               rng: int = RANDOM_STATE) -> np.ndarray:
    """Fit-transform one method on X = (n_samples, n_features)."""
    n = X.shape[0]
    if name == "PCA":
        return fit_pca(X, n_components)[0]
    if name == "Isomap":
        k = min(15, max(5, n // 20))
        return Isomap(n_components=n_components, n_neighbors=k).fit_transform(X)
    raise ValueError(f"unknown method {name}")


# ---------------------------------------------------------------------------
# dPCA (or fallback marginalisation)
# ---------------------------------------------------------------------------

def fit_dpca_condition_avg(A: np.ndarray, cond: np.ndarray,
                           n_components: int = 2) -> dict[str, np.ndarray]:
    """A: (n_trials, n_neurons, n_xG). cond: (n_trials,) discrete labels.

    Returns dict with keys "t" (time/xG marginal), "c" (condition marginal),
    "ct" (interaction), each shaped (n_components, n_levels, n_xG) projected
    from the condition-averaged tensor.

    If `dPCA` is available, use it; else fall back to a coarse marginalisation:
    subtract grand mean → marginalise → PCA each marginal.
    """
    levels = np.unique(cond)
    levels = levels[levels != ""]  # drop empty strings from tertile splits
    if len(levels) < 2:
        raise ValueError("dPCA needs >=2 condition levels")
    n_trials, n_neurons, n_xG = A.shape
    # Build (n_neurons, n_levels, n_xG) condition-averaged tensor.
    # A is (n_trials, n_neurons, n_xG) so A[idx].mean(axis=0) is already
    # (n_neurons, n_xG) — no transpose needed.
    R = np.full((n_neurons, len(levels), n_xG), np.nan)
    for li, lv in enumerate(levels):
        idx = (cond == lv)
        if idx.sum() == 0:
            continue
        R[:, li, :] = A[idx].mean(axis=0)
    # NaN-safe grand mean over (level, xG)
    grand = np.nanmean(R, axis=(1, 2), keepdims=True)
    Rc = R - grand
    # Replace lingering NaNs by 0 (rare conditions)
    Rc = np.nan_to_num(Rc, nan=0.0)

    if _HAVE_DPCA:
        try:
            dpca = _dPCA_lib.dPCA(labels="ct", n_components=n_components)
            dpca.protect = ["t"]
            Z = dpca.fit_transform(Rc)
            # Z is a dict like {"t": (k, n_levels, n_xG), "c": ..., "ct": ...}
            return {k: np.asarray(v) for k, v in Z.items()}
        except Exception as e:  # pragma: no cover
            warnings.warn(f"dPCA fit failed ({e}); falling back to marginalisation")

    # Fallback marginalisation:
    # t marginal: mean over levels then PCA over xG samples
    t_mean = Rc.mean(axis=1)                       # (n_neurons, n_xG)
    c_mean = Rc.mean(axis=2)                       # (n_neurons, n_levels)
    ct = Rc - t_mean[:, None, :] - c_mean[:, :, None]  # interaction

    def _project(M_neurons_by_samples: np.ndarray, k: int) -> np.ndarray:
        # PCA over samples (columns); return (k, n_samples)
        if M_neurons_by_samples.shape[1] < 2:
            return np.zeros((k, M_neurons_by_samples.shape[1]))
        pca = PCA(n_components=min(k, M_neurons_by_samples.shape[0],
                                   M_neurons_by_samples.shape[1]),
                  random_state=RANDOM_STATE)
        # transform samples into PC space: components_ @ centered_samples
        pca.fit(M_neurons_by_samples.T)
        Z = pca.transform(M_neurons_by_samples.T).T   # (k, n_samples)
        # pad if fewer components
        if Z.shape[0] < k:
            pad = np.zeros((k - Z.shape[0], Z.shape[1]))
            Z = np.vstack([Z, pad])
        return Z

    k = n_components
    Zt = _project(t_mean, k).reshape(k, 1, n_xG).repeat(len(levels), axis=1)
    Zc = _project(c_mean, k).reshape(k, len(levels), 1).repeat(n_xG, axis=2)
    Zct = _project(ct.reshape(n_neurons, len(levels) * n_xG), k).reshape(
        k, len(levels), n_xG)
    return {"t": Zt, "c": Zc, "ct": Zct, "_levels": levels}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

CONT_CMAP = "viridis"
CAT_CMAP = "tab10"

# Methods that can do single-trial scatter (Style A)
SCATTER_METHODS = ["PCA", "Isomap"]
# Methods for condition-averaged trajectories (Style B)
TRAJ_METHODS_B = ["PCA", "dPCA"]
# Methods for single-trial trajectories (Style C)
TRAJ_METHODS_C = ["PCA"]

# Discrete groupings (work for Style A, B, C and as alignment colour)
GROUPINGS_DISCRETE = [
    "choice", "correct",
    "ori_tertile",
    "signed_contrast_tertile", "signed_dispersion_tertile",
    "H_dec_tertile", "sd_perc_tertile",
]
# Continuous groupings (Style A scatter only)
GROUPINGS_CONTINUOUS = [
    "true_orientation",
    "signed_contrast", "signed_dispersion",
    "p_go", "H_dec", "sd_perc",
]
# Full Style A grouping list
GROUPINGS_A = GROUPINGS_CONTINUOUS + GROUPINGS_DISCRETE
# Style C is the heaviest plot — keep a sensible subset
GROUPINGS_C = ["choice", "correct", "ori_tertile",
               "signed_contrast_tertile", "H_dec_tertile", "sd_perc_tertile"]

# Tertile labels (lo/med/hi) — central source of truth
TERTILE_NAMES = {"ori_tertile", "signed_contrast_tertile",
                 "signed_dispersion_tertile", "H_dec_tertile", "sd_perc_tertile"}


def _grid_dims(n: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _is_discrete(name: str) -> bool:
    return name in set(GROUPINGS_DISCRETE)


def _level_order(name: str, values: np.ndarray) -> list:
    if name in TERTILE_NAMES:
        return [v for v in ("lo", "med", "hi") if v in np.unique(values)]
    uniq = np.unique(values[values != ""]) if values.dtype == object else np.unique(values)
    return list(uniq)


def _palette(levels: list) -> dict:
    cmap = plt.get_cmap(CAT_CMAP)
    return {lv: cmap(i % cmap.N) for i, lv in enumerate(levels)}


def _annotate_panel(ax, mouse: MouseData, extra: str = ""):
    txt = f"{mouse.tag}  n={mouse.n_trials}, units={mouse.n_neurons}"
    if extra:
        txt = f"{txt}  {extra}"
    ax.set_title(txt, fontsize=8)


def _save_fig(fig, out_path: Path, dpi=140):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layout helpers — keep legends/colorbars OUT of the data area
# ---------------------------------------------------------------------------

from matplotlib.lines import Line2D


def _union_levels(mice: list, grouping: str) -> list:
    """Canonical-ordered union of level labels across mice for a discrete grouping."""
    seen = []
    for m in mice:
        for lv in _level_order(grouping, compute_features(m)[grouping]):
            if lv not in seen:
                seen.append(lv)
    return seen


def _finalize_grid(fig, *, cont: bool, levels=None, palette=None,
                   norm=None, cmap=None, label: str = "",
                   right: float = 0.86, top: float = 0.93,
                   hspace: float = 0.35, wspace: float = 0.35):
    """Reserve space on the right of a subplot grid for a single shared
    legend (discrete) or colorbar (continuous). Avoids `tight_layout` because
    it interacts badly with 3D axes."""
    fig.subplots_adjust(left=0.06, right=right, top=top, bottom=0.08,
                        hspace=hspace, wspace=wspace)
    if cont:
        cax = fig.add_axes([right + 0.025, 0.18, 0.018, 0.62])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label)
    elif levels and palette:
        handles = [Line2D([0], [0], marker="o", color=palette[lv], lw=0,
                          markersize=7, label=str(lv)) for lv in levels]
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(right + 0.015, top),
                   frameon=False, fontsize=8, title=label or None,
                   title_fontsize=8)


def _finalize_single(fig, ax, *, cont: bool, levels=None, palette=None,
                     norm=None, cmap=None, label: str = ""):
    """Same idea for a single-panel figure (Style E)."""
    fig.subplots_adjust(left=0.10, right=0.78, top=0.92, bottom=0.10)
    if cont:
        cax = fig.add_axes([0.81, 0.18, 0.022, 0.62])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label)
    elif levels and palette:
        handles = [Line2D([0], [0], marker="o", color=palette[lv], lw=0,
                          markersize=7, label=str(lv)) for lv in levels]
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.80, 0.92),
                   frameon=False, fontsize=8, title=label or None,
                   title_fontsize=8)


# ---------------------------------------------------------------------------
# Style A — single-trial scatter
# ---------------------------------------------------------------------------

def render_style_A(mice: list[MouseData], method: str, grouping: str,
                   out_dir: Path, n_components: int = 2) -> Path | None:
    """Single-trial scatter. n_components=2 or 3."""
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    rows, cols = _grid_dims(len(mice))
    subplot_kw = {"projection": "3d"} if n_components == 3 else {}
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows),
                             squeeze=False, subplot_kw=subplot_kw)
    fig.suptitle(
        f"Style A{'3D' if n_components == 3 else ''} — single-trial scatter | "
        f"{method} | colour: {grouping}", fontsize=11)
    cont = not _is_discrete(grouping)
    if cont:
        all_vals = np.concatenate([compute_features(m)[grouping] for m in mice])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size == 0:
            plt.close(fig)
            return None
        norm = Normalize(vmin=np.nanpercentile(finite, 2),
                         vmax=np.nanpercentile(finite, 98))
        cmap = plt.get_cmap(CONT_CMAP)
    else:
        norm, cmap = None, None

    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats[grouping]
        X = trial_mean_matrix(_zscore_neurons(m.activity))

        try:
            Y = fit_method(method, X, n_components=n_components)
        except Exception as e:
            ax.text2D(0.5, 0.5, f"{method} failed:\n{e}",
                      ha="center", va="center",
                      transform=ax.transAxes, fontsize=7) if n_components == 3 else \
                ax.text(0.5, 0.5, f"{method} failed:\n{e}",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            ax.set_axis_off()
            continue

        if cont:
            args = (Y[:, 0], Y[:, 1], Y[:, 2]) if n_components == 3 else (Y[:, 0], Y[:, 1])
            ax.scatter(*args, c=c, s=6, alpha=0.55, cmap=cmap, norm=norm)
        else:
            levels = _level_order(grouping, c)
            pal = _palette(levels)
            for lv in levels:
                msk = (c == lv)
                args = (Y[msk, 0], Y[msk, 1], Y[msk, 2]) if n_components == 3 \
                       else (Y[msk, 0], Y[msk, 1])
                ax.scatter(*args, s=6, alpha=0.55, color=pal[lv])

        ax.set_xlabel(f"{method}-1", fontsize=7)
        ax.set_ylabel(f"{method}-2", fontsize=7)
        if n_components == 3:
            ax.set_zlabel(f"{method}-3", fontsize=7)
        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m)

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    union = _union_levels(mice, grouping) if not cont else None
    pal_union = _palette(union) if union else None
    _finalize_grid(fig, cont=cont, levels=union, palette=pal_union,
                   norm=norm, cmap=cmap, label=grouping)
    tag = "_3d" if n_components == 3 else ""
    out = out_dir / f"{method}__{grouping}{tag}"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Style B — condition-averaged trajectories over xG
# ---------------------------------------------------------------------------

def _cond_avg_traj_pca(m: MouseData, cond: np.ndarray, levels: list,
                       n_components: int = 2) -> dict:
    """Fit PCA on stacked condition-mean (xG × neurons) matrix.
    Return per-level (n_xG, n_components) trajectories."""
    Az = _zscore_neurons(m.activity)  # (n_trials, n_neurons, n_xG)
    n_trials, n_neurons, n_xG = Az.shape
    means = []
    valid_levels = []
    for lv in levels:
        msk = (cond == lv)
        if msk.sum() < 3:
            continue
        means.append(Az[msk].mean(axis=0).T)  # (n_xG, n_neurons)
        valid_levels.append(lv)
    if not means:
        return {}
    stacked = np.vstack(means)  # (len*n_xG, n_neurons)
    k = min(n_components, stacked.shape[1])
    Y = PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(stacked)
    if Y.shape[1] < n_components:
        Y = np.hstack([Y, np.zeros((Y.shape[0], n_components - Y.shape[1]))])
    out = {}
    for i, lv in enumerate(valid_levels):
        out[lv] = Y[i * n_xG:(i + 1) * n_xG]
    return out


def _plot_traj_panel(ax, Y, color, label, n_components):
    """Plot a single trajectory (with start/end markers) on a 2D or 3D axis."""
    if n_components == 3:
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color=color, lw=1.5, alpha=0.9, label=label)
        ax.scatter(Y[0, 0], Y[0, 1], Y[0, 2], color=color, s=30, marker="o")
        ax.scatter(Y[-1, 0], Y[-1, 1], Y[-1, 2], color=color, s=30, marker="^")
    else:
        ax.plot(Y[:, 0], Y[:, 1], color=color, lw=1.5, alpha=0.9, label=label)
        ax.scatter(Y[0, 0], Y[0, 1], color=color, s=30, marker="o")
        ax.scatter(Y[-1, 0], Y[-1, 1], color=color, s=30, marker="^")


def render_style_B(mice: list[MouseData], method: str, grouping: str,
                   out_dir: Path, n_components: int = 2) -> Path | None:
    if not _is_discrete(grouping):
        return None
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    rows, cols = _grid_dims(len(mice))
    subplot_kw = {"projection": "3d"} if n_components == 3 else {}
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                             squeeze=False, subplot_kw=subplot_kw)
    fig.suptitle(
        f"Style B{'3D' if n_components == 3 else ''} — condition-averaged "
        f"trajectory over xG | {method} | groups: {grouping}", fontsize=11)
    for ax, m in zip(axes.ravel(), mice):
        feats = compute_features(m)
        c = feats[grouping]
        levels = _level_order(grouping, c)
        if len(levels) < 2:
            ax.text(0.5, 0.5, "<2 levels", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue
        pal = _palette(levels)

        if method == "PCA":
            trajs = _cond_avg_traj_pca(m, c, levels, n_components=n_components)
            for lv, Y in trajs.items():
                _plot_traj_panel(ax, Y, pal[lv], str(lv), n_components)
            ax.set_xlabel("PC1", fontsize=7)
            ax.set_ylabel("PC2", fontsize=7)
            if n_components == 3:
                ax.set_zlabel("PC3", fontsize=7)
        elif method == "dPCA":
            try:
                Az = _zscore_neurons(m.activity)
                Z = fit_dpca_condition_avg(Az, c, n_components=n_components)
                ct = Z.get("ct", Z.get("c"))
                lv_used = Z.get("_levels", levels)
                for li, lv in enumerate(lv_used):
                    Y = ct[:, li, :].T   # (n_xG, k)
                    if Y.shape[1] < n_components:
                        Y = np.hstack([Y, np.zeros((Y.shape[0],
                                                    n_components - Y.shape[1]))])
                    color = pal.get(lv, plt.get_cmap(CAT_CMAP)(li % 10))
                    _plot_traj_panel(ax, Y, color, str(lv), n_components)
                ax.set_xlabel("dPC1 (cond×t)", fontsize=7)
                ax.set_ylabel("dPC2 (cond×t)", fontsize=7)
                if n_components == 3:
                    ax.set_zlabel("dPC3 (cond×t)", fontsize=7)
            except Exception as e:
                ax.text(0.5, 0.5, f"dPCA failed:\n{e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                ax.set_axis_off()
                continue
        else:
            ax.set_axis_off()
            continue

        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m, extra="(circle=xG start, ^=end)")

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    union = _union_levels(mice, grouping)
    pal_union = _palette(union)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    tag = "_3d" if n_components == 3 else ""
    out = out_dir / f"{method}__{grouping}{tag}"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Style C — single-trial trajectories over xG (PCA only)
# ---------------------------------------------------------------------------

def render_style_C(mice: list[MouseData], grouping: str,
                   out_dir: Path) -> Path | None:
    if not _is_discrete(grouping):
        return None
    rows, cols = _grid_dims(len(mice))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle(f"Style C — single-trial trajectories over xG | PCA | groups: {grouping}",
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
        pal = _palette(levels)
        Az = _zscore_neurons(m.activity)   # (n_trials, n_neurons, n_xG)
        n_trials, n_neurons, n_xG = Az.shape
        # Stack all (trial, xG) samples and fit PCA once
        X = Az.transpose(0, 2, 1).reshape(-1, n_neurons)
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
        # For each level, subsample trials and draw lines
        rng = np.random.RandomState(RANDOM_STATE)
        for lv in levels:
            msk = np.where(c == lv)[0]
            if msk.size == 0:
                continue
            if msk.size > STYLE_C_MAX_TRIALS_PER_LEVEL:
                msk = rng.choice(msk, STYLE_C_MAX_TRIALS_PER_LEVEL, replace=False)
            for ti in msk:
                Y = pca.transform(Az[ti].T)   # (n_xG, 2)
                ax.plot(Y[:, 0], Y[:, 1], color=pal[lv], lw=0.4, alpha=0.25)
            # condition mean trajectory on top
            mean_traj = pca.transform(Az[c == lv].mean(axis=0).T)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=pal[lv], lw=2.0,
                    alpha=0.95)

        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m)

    for ax in axes.ravel()[len(mice):]:
        ax.set_axis_off()

    union = _union_levels(mice, grouping)
    pal_union = _palette(union)
    _finalize_grid(fig, cont=False, levels=union, palette=pal_union,
                   label=grouping)
    out = out_dir / f"PCA__{grouping}"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Style D — CEBRA embeddings (per-mouse, label-supervised)
# ---------------------------------------------------------------------------

# Slightly aggressive defaults so the sweep is bearable; can be overridden.
CEBRA_KW = dict(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=3,
    max_iterations=2000,
    distance="cosine",
    conditional="discrete",
    device="cpu",
    verbose=False,
)


def _prepare_cebra_multisession(mice: list[MouseData], grouping: str):
    """Build per-mouse (X, labels) lists with a shared label encoding.

    Returns (X_list, y_list, info_list, union_levels, lv_to_int). info_list[i]
    has the originating MouseData plus the boolean valid-trial mask so we can
    map embeddings back to per-trial features for plotting.
    """
    union = _union_levels(mice, grouping)
    lv_to_int = {lv: i for i, lv in enumerate(union)}
    X_list, y_list, info_list = [], [], []
    for m in mice:
        feats = compute_features(m)
        c = feats[grouping]
        labels = np.array([lv_to_int.get(v, -1) for v in c])
        valid = labels >= 0
        if valid.sum() < 50:
            continue
        # Also require ≥2 distinct labels within this mouse (CEBRA needs
        # within-session contrast to learn)
        if len(np.unique(labels[valid])) < 2:
            continue
        X = trial_mean_matrix(_zscore_neurons(m.activity))
        X_list.append(X[valid].astype(np.float32))
        y_list.append(labels[valid].astype(np.int64))
        info_list.append((m, valid, labels))
    return X_list, y_list, info_list, union, lv_to_int


def render_style_D_cebra_shared(mice: list[MouseData], grouping: str,
                                out_dir: Path) -> Path | None:
    """Multi-session CEBRA: ONE model fit across all mice, shared 3D embedding.

    Produces two figure files:
      ``CEBRA_shared_subpanels__<grouping>`` — per-mouse subpanels of the
        shared embedding, each panel uses the same axes scale by construction.
      ``CEBRA_shared_overlay__<grouping>`` — all mice overlaid in a single 3D
        panel, distinguished by marker shape, coloured by label level.
    """
    if not _HAVE_CEBRA:
        raise RuntimeError(
            "cebra not installed — pip install cebra (PyTorch backend)."
        )
    if not _is_discrete(grouping):
        return None

    X_list, y_list, info_list, union, lv_to_int = _prepare_cebra_multisession(
        mice, grouping)
    if len(X_list) < 2 or len(union) < 2:
        return None

    try:
        model = cebra.CEBRA(**CEBRA_KW)
        model.fit(X_list, y_list)
    except Exception as e:
        raise RuntimeError(
            f"CEBRA multi-session fit failed ({e}); falling back to per-mouse "
            f"may be needed."
        )

    # Per-mouse embeddings in the shared latent
    embeddings = []
    for i, X in enumerate(X_list):
        try:
            Y = model.transform(X, session_id=i)
        except TypeError:
            # older cebra versions: positional session_id
            Y = model.transform(X, i)
        embeddings.append(np.asarray(Y))

    pal = _palette(union)

    # --- Figure 1: subpanels (one per mouse, shared coordinate frame) ---
    rows, cols = _grid_dims(len(X_list))
    fig = plt.figure(figsize=(3.6 * cols, 3.4 * rows))
    fig.suptitle(
        f"Style D — CEBRA multi-session shared embedding | groups: {grouping} "
        f"(n_mice={len(X_list)})", fontsize=11)
    axes_3d = [fig.add_subplot(rows, cols, i + 1, projection="3d")
               for i in range(rows * cols)]

    # Global axis limits across all mice so panels are directly comparable
    all_emb = np.vstack(embeddings)
    lims = [(all_emb[:, k].min(), all_emb[:, k].max()) for k in range(3)]

    for ax, (m, valid, labels), Y in zip(axes_3d, info_list, embeddings):
        labs = labels[valid]
        for lv in union:
            msk = (labs == lv_to_int[lv])
            if msk.sum() == 0:
                continue
            ax.scatter(Y[msk, 0], Y[msk, 1], Y[msk, 2], s=6, alpha=0.55,
                       color=pal[lv])
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_xlabel("CEBRA-1", fontsize=7)
        ax.set_ylabel("CEBRA-2", fontsize=7)
        ax.set_zlabel("CEBRA-3", fontsize=7)
        ax.tick_params(labelsize=6)
        _annotate_panel(ax, m)

    for ax in axes_3d[len(info_list):]:
        ax.set_axis_off()

    _finalize_grid(fig, cont=False, levels=union, palette=pal, label=grouping)
    out_sub = out_dir / f"CEBRA_shared_subpanels__{grouping}"
    _save_fig(fig, out_sub)

    # --- Figure 2: overlay (all mice in a single panel) ---
    fig2 = plt.figure(figsize=(9, 7))
    ax2 = fig2.add_subplot(111, projection="3d")
    fig2.suptitle(
        f"Style D — CEBRA multi-session overlay | groups: {grouping} "
        f"(n_mice={len(X_list)})", fontsize=10)

    mouse_markers = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h"]
    mouse_marker_map = {}
    for mi, ((m, valid, labels), Y) in enumerate(zip(info_list, embeddings)):
        marker = mouse_markers[mi % len(mouse_markers)]
        mouse_marker_map[m.tag] = marker
        labs = labels[valid]
        for lv in union:
            msk = (labs == lv_to_int[lv])
            if msk.sum() == 0:
                continue
            ax2.scatter(Y[msk, 0], Y[msk, 1], Y[msk, 2], s=10, alpha=0.5,
                        color=pal[lv], marker=marker)

    ax2.set_xlabel("CEBRA-1", fontsize=9)
    ax2.set_ylabel("CEBRA-2", fontsize=9)
    ax2.set_zlabel("CEBRA-3", fontsize=9)
    ax2.tick_params(labelsize=8)

    fig2.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.06)
    mouse_handles = [Line2D([0], [0], marker=mk, color="k", lw=0, markersize=7,
                            label=tag)
                     for tag, mk in mouse_marker_map.items()]
    mouse_leg = fig2.legend(handles=mouse_handles, loc="upper left",
                            bbox_to_anchor=(0.74, 0.92), frameon=False,
                            fontsize=8, title="mouse", title_fontsize=8)
    fig2.add_artist(mouse_leg)
    lvl_handles = [Line2D([0], [0], marker="o", color=pal[lv], lw=0,
                          markersize=7, label=str(lv)) for lv in union]
    fig2.legend(handles=lvl_handles, loc="lower left",
                bbox_to_anchor=(0.74, 0.06), frameon=False,
                fontsize=8, title=grouping, title_fontsize=8)
    out_ovl = out_dir / f"CEBRA_shared_overlay__{grouping}"
    _save_fig(fig2, out_ovl)

    return out_sub


# ---------------------------------------------------------------------------
# Style E — cross-mouse alignment
# ---------------------------------------------------------------------------

def _per_mouse_cond_trajs(mice: list[MouseData], grouping: str,
                          n_components: int = 2) -> tuple[dict, list]:
    """Per-mouse condition-averaged trajectories + list of levels common to ≥2 mice."""
    per_mouse = {}
    for m in mice:
        feats = compute_features(m)
        c = feats[grouping]
        levels = _level_order(grouping, c)
        if len(levels) < 2:
            continue
        trajs = _cond_avg_traj_pca(m, c, levels, n_components=n_components)
        if trajs:
            per_mouse[m.tag] = trajs
    # Common levels = those present in at least 2 mice
    count = {}
    for trajs in per_mouse.values():
        for lv in trajs:
            count[lv] = count.get(lv, 0) + 1
    common = [lv for lv, n in count.items() if n >= 2]
    # Preserve canonical tertile order
    if all(lv in ("lo", "med", "hi") for lv in common):
        common = [lv for lv in ("lo", "med", "hi") if lv in common]
    return per_mouse, common


def render_style_E_procrustes(mice: list[MouseData], grouping: str,
                              out_dir: Path, n_components: int = 2) -> Path | None:
    """Procrustes-align condition-averaged trajectories across mice on shared levels."""
    if not _is_discrete(grouping):
        return None
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    per_mouse, common = _per_mouse_cond_trajs(mice, grouping, n_components=n_components)
    if len(per_mouse) < 2 or len(common) < 2:
        return None

    # Pick reference = mouse with the most trials that *has* all common levels
    mice_by_tag = {m.tag: m for m in mice}
    candidates = [tag for tag, trajs in per_mouse.items()
                  if all(lv in trajs for lv in common)]
    if not candidates:
        candidates = list(per_mouse.keys())
    ref_tag = max(candidates, key=lambda t: mice_by_tag[t].n_trials)
    ref_trajs = per_mouse[ref_tag]
    ref_concat = np.vstack([ref_trajs[lv] for lv in common if lv in ref_trajs])

    # Procrustes-align each non-reference mouse
    aligned = {}
    ref_std_storage = None
    n_xG = next(iter(ref_trajs.values())).shape[0]
    for tag, trajs in per_mouse.items():
        lv_avail = [lv for lv in common if lv in trajs]
        if len(lv_avail) < 2:
            continue
        ref_match = np.vstack([ref_trajs[lv] for lv in lv_avail])
        mouse_match = np.vstack([trajs[lv] for lv in lv_avail])
        try:
            ref_s, mouse_s, _ = _procrustes(ref_match, mouse_match)
        except Exception:
            continue
        per_lv = {lv: mouse_s[i * n_xG:(i + 1) * n_xG]
                  for i, lv in enumerate(lv_avail)}
        aligned[tag] = per_lv
        if tag == ref_tag and ref_std_storage is None:
            ref_std_storage = {lv: ref_s[i * n_xG:(i + 1) * n_xG]
                               for i, lv in enumerate(lv_avail)}

    # Use the standardized form of the reference as the canonical frame
    if ref_std_storage is not None:
        aligned[ref_tag] = ref_std_storage

    if len(aligned) < 2:
        return None

    subplot_kw = {"projection": "3d"} if n_components == 3 else {}
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), subplot_kw=subplot_kw)
    fig.suptitle(
        f"Style E — Procrustes-aligned condition trajectories | groups: {grouping} "
        f"(ref={ref_tag}, n_mice={len(aligned)})", fontsize=10)
    pal = _palette(common)

    # Thin lines per mouse per level
    for tag, by_lv in aligned.items():
        for lv, Y in by_lv.items():
            if n_components == 3:
                ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color=pal[lv],
                        lw=0.8, alpha=0.35)
            else:
                ax.plot(Y[:, 0], Y[:, 1], color=pal[lv], lw=0.8, alpha=0.35)

    # Bold mean trajectory across mice per level
    for lv in common:
        stack = [by_lv[lv] for by_lv in aligned.values() if lv in by_lv]
        if len(stack) < 2:
            continue
        mean_traj = np.mean(np.stack(stack, axis=0), axis=0)
        _plot_traj_panel(ax, mean_traj, pal[lv], f"{lv} (mean)", n_components)

    ax.set_xlabel("Aligned-1", fontsize=9)
    ax.set_ylabel("Aligned-2", fontsize=9)
    if n_components == 3:
        ax.set_zlabel("Aligned-3", fontsize=9)
    ax.tick_params(labelsize=8)

    _finalize_single(fig, ax, cont=False, levels=common, palette=pal,
                     label=grouping)
    tag = "_3d" if n_components == 3 else ""
    out = out_dir / f"procrustes__{grouping}{tag}"
    _save_fig(fig, out)
    return out


# Fixed shared task-axis names used by render_style_E_taskaxis
TASK_AXES_DEFAULT = ["true_orientation", "p_go", "H_dec", "sd_perc",
                     "signed_contrast", "signed_dispersion"]


def _task_axis_projection(m: MouseData, axes_names: list,
                          ridge_alpha: float = 1.0) -> np.ndarray:
    """Return trial activity projected onto shared task axes.

    Per task variable y, fit Ridge(y ~ activity) and use the fitted coefficient
    as the encoding direction for that axis. Stack across axes → (n_neurons, k).
    Projection is then activity @ W → (n_trials, k). Standardising both X and
    W norms keeps mice on comparable scales.
    """
    feats = compute_features(m)
    X = trial_mean_matrix(_zscore_neurons(m.activity))   # (n_trials, n_neurons)
    W_cols = []
    used = []
    for name in axes_names:
        y = feats.get(name)
        if y is None:
            continue
        valid = np.isfinite(y)
        if valid.sum() < 30:
            continue
        reg = Ridge(alpha=ridge_alpha).fit(X[valid], y[valid])
        w = reg.coef_.copy().astype(float)
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            continue
        W_cols.append(w / norm)
        used.append(name)
    if not W_cols:
        return np.empty((m.n_trials, 0)), used
    W = np.stack(W_cols, axis=1)   # (n_neurons, k)
    Z = X @ W                      # (n_trials, k)
    return Z, used


def render_style_E_taskaxis(mice: list[MouseData], grouping: str,
                            out_dir: Path,
                            axes_names: list = None) -> Path | None:
    """Project each mouse onto shared task axes, scatter overlay coloured by `grouping`.

    Renders a single panel showing the first 3 task axes in 3D, all mice overlaid.
    """
    if axes_names is None:
        axes_names = TASK_AXES_DEFAULT

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(
        f"Style E — task-axis projection (all mice overlaid) | colour: {grouping}",
        fontsize=10)
    cont = not _is_discrete(grouping)
    used_axes = None
    norm = None
    cmap = plt.get_cmap(CONT_CMAP) if cont else None
    pal = None
    all_lvs = None
    if cont:
        all_vals = np.concatenate([compute_features(m)[grouping]
                                   for m in mice if grouping in compute_features(m)])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size == 0:
            plt.close(fig)
            return None
        norm = Normalize(vmin=np.nanpercentile(finite, 2),
                         vmax=np.nanpercentile(finite, 98))
    else:
        all_lvs = _union_levels(mice, grouping)
        pal = _palette(all_lvs)

    mouse_markers = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h"]
    mouse_marker_map = {}
    for mi, m in enumerate(mice):
        Z, used = _task_axis_projection(m, axes_names)
        if Z.shape[1] < 3:
            continue
        used_axes = used
        feats = compute_features(m)
        c = feats[grouping]
        marker = mouse_markers[mi % len(mouse_markers)]
        mouse_marker_map[m.tag] = marker
        if cont:
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=c, cmap=cmap, norm=norm,
                       s=8, alpha=0.5, marker=marker)
        else:
            for lv in _level_order(grouping, c):
                msk = (c == lv)
                ax.scatter(Z[msk, 0], Z[msk, 1], Z[msk, 2], s=8, alpha=0.5,
                           marker=marker, color=pal[lv])

    if used_axes is None or len(used_axes) < 3:
        plt.close(fig)
        return None

    ax.set_xlabel(f"axis: {used_axes[0]}", fontsize=8)
    ax.set_ylabel(f"axis: {used_axes[1]}", fontsize=8)
    ax.set_zlabel(f"axis: {used_axes[2]}", fontsize=8)
    ax.tick_params(labelsize=7)

    # Reserve right margin for legends/cbar
    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.06)

    # Mouse-marker legend (always shown — orthogonal to colour)
    mouse_handles = [Line2D([0], [0], marker=mk, color="k", lw=0, markersize=7,
                            label=tag)
                     for tag, mk in mouse_marker_map.items()]
    mouse_leg = fig.legend(handles=mouse_handles, loc="upper left",
                           bbox_to_anchor=(0.74, 0.92), frameon=False,
                           fontsize=8, title="mouse", title_fontsize=8)
    fig.add_artist(mouse_leg)

    if cont:
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.55])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                     label=grouping)
    else:
        lvl_handles = [Line2D([0], [0], marker="o", color=pal[lv], lw=0,
                              markersize=7, label=str(lv))
                       for lv in all_lvs]
        fig.legend(handles=lvl_handles, loc="lower left",
                   bbox_to_anchor=(0.74, 0.06), frameon=False,
                   fontsize=8, title=grouping, title_fontsize=8)

    out = out_dir / f"taskaxis__{grouping}"
    _save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _filter_mice(mice: list[MouseData]) -> tuple[list[MouseData], list[dict]]:
    keep, skipped = [], []
    for m in mice:
        if m.n_trials < MIN_TRIALS or m.n_neurons < MIN_NEURONS:
            skipped.append({"tag": m.tag, "n_trials": int(m.n_trials),
                            "n_neurons": int(m.n_neurons),
                            "reason": "below threshold"})
            continue
        keep.append(m)
    return keep, skipped


def run_sweep(smoke: bool = False, fig_root: Path = FIG_ROOT,
              data_path: Path = DATA_PATH) -> dict:
    t0 = time.time()
    mice_all = load_all_mice(data_path)
    mice, skipped = _filter_mice(mice_all)
    if not mice:
        raise RuntimeError("no mice survive filtering; check thresholds / data")

    if smoke:
        mice = mice[:1]
        a_methods = ["PCA"]
        a_groups = ["ori_tertile"]
        b_methods = ["PCA"]
        b_groups = ["choice"]
        c_groups = ["choice"]
        d_groups = ["choice"] if _HAVE_CEBRA else []
        e_groups = ["choice"]
    else:
        a_methods = list(SCATTER_METHODS)
        a_groups = GROUPINGS_A
        b_methods = TRAJ_METHODS_B
        b_groups = GROUPINGS_DISCRETE
        c_groups = GROUPINGS_C
        d_groups = GROUPINGS_DISCRETE if _HAVE_CEBRA else []
        e_groups = GROUPINGS_DISCRETE

    meta = {
        "data_path": str(data_path),
        "n_mice_loaded": len(mice_all),
        "n_mice_used": len(mice),
        "mice": [{"tag": m.tag, "n_trials": int(m.n_trials),
                  "n_neurons": int(m.n_neurons), "n_xG": int(m.n_xG)}
                 for m in mice],
        "skipped": skipped,
        "have_dpca": _HAVE_DPCA,
        "have_cebra": _HAVE_CEBRA,
        "smoke": smoke,
        "produced": [],
    }

    def _record(style, method, grouping, path_or_err, secs):
        entry = {"style": style, "method": method, "grouping": grouping, "secs": secs}
        if isinstance(path_or_err, Exception):
            entry["error"] = repr(path_or_err)
        elif path_or_err is None:
            entry["skipped"] = "renderer returned None"
        else:
            entry["path"] = str(path_or_err)
        meta["produced"].append(entry)

    # Style A — 2D scatter
    sA = fig_root / "A_single_trial_scatter"
    for method in a_methods:
        for g in a_groups:
            t1 = time.time()
            print(f"[A2D] method={method} group={g}", flush=True)
            try:
                out = render_style_A(mice, method, g, sA, n_components=2)
            except Exception as e:
                out = e
            _record("A", method, g, out, round(time.time() - t1, 1))

    # Style A3D — PCA only (Isomap/UMAP 3D is too cluttered for a subpanel grid)
    sA3 = fig_root / "A3D_single_trial_scatter"
    for g in a_groups:
        t1 = time.time()
        print(f"[A3D] method=PCA group={g}", flush=True)
        try:
            out = render_style_A(mice, "PCA", g, sA3, n_components=3)
        except Exception as e:
            out = e
        _record("A3D", "PCA", g, out, round(time.time() - t1, 1))

    # Style B — 2D trajectories
    sB = fig_root / "B_condition_avg_traj"
    for method in b_methods:
        for g in b_groups:
            t1 = time.time()
            print(f"[B2D] method={method} group={g}", flush=True)
            try:
                out = render_style_B(mice, method, g, sB, n_components=2)
            except Exception as e:
                out = e
            _record("B", method, g, out, round(time.time() - t1, 1))

    # Style B3D — 3D trajectories
    sB3 = fig_root / "B3D_condition_avg_traj"
    for method in b_methods:
        for g in b_groups:
            t1 = time.time()
            print(f"[B3D] method={method} group={g}", flush=True)
            try:
                out = render_style_B(mice, method, g, sB3, n_components=3)
            except Exception as e:
                out = e
            _record("B3D", method, g, out, round(time.time() - t1, 1))

    # Style C — single-trial trajectories (PCA, 2D)
    sC = fig_root / "C_single_trial_traj"
    for g in c_groups:
        t1 = time.time()
        print(f"[C] group={g}", flush=True)
        try:
            out = render_style_C(mice, g, sC)
        except Exception as e:
            out = e
        _record("C", "PCA", g, out, round(time.time() - t1, 1))

    # Style D — CEBRA multi-session shared embedding (one model across mice)
    sD = fig_root / "D_cebra"
    for g in d_groups:
        t1 = time.time()
        print(f"[D] CEBRA multi-session group={g}", flush=True)
        try:
            out = render_style_D_cebra_shared(mice, g, sD)
        except Exception as e:
            out = e
        _record("D", "CEBRA_shared", g, out, round(time.time() - t1, 1))

    # Style E — cross-mouse alignment
    sE = fig_root / "E_cross_mouse"
    for g in e_groups:
        # Procrustes 2D
        t1 = time.time()
        print(f"[E] procrustes 2d group={g}", flush=True)
        try:
            out = render_style_E_procrustes(mice, g, sE, n_components=2)
        except Exception as e:
            out = e
        _record("E", "procrustes2d", g, out, round(time.time() - t1, 1))
        # Procrustes 3D
        t1 = time.time()
        print(f"[E] procrustes 3d group={g}", flush=True)
        try:
            out = render_style_E_procrustes(mice, g, sE, n_components=3)
        except Exception as e:
            out = e
        _record("E", "procrustes3d", g, out, round(time.time() - t1, 1))

    # Style E task-axis — supports both discrete + continuous groupings
    for g in GROUPINGS_A:
        t1 = time.time()
        print(f"[E] taskaxis group={g}", flush=True)
        try:
            out = render_style_E_taskaxis(mice, g, sE)
        except Exception as e:
            out = e
        _record("E", "taskaxis", g, out, round(time.time() - t1, 1))

    meta["elapsed_secs"] = round(time.time() - t0, 1)
    fig_root.mkdir(parents=True, exist_ok=True)
    with open(fig_root / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[done] {len(meta['produced'])} entries; "
          f"{sum(1 for p in meta['produced'] if 'error' in p)} errors; "
          f"elapsed {meta['elapsed_secs']}s")
    print(f"[done] meta written to {fig_root / '_meta.json'}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="run one mouse × one combo per style (fast sanity check)")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--out", type=Path, default=FIG_ROOT)
    args = p.parse_args()
    run_sweep(smoke=args.smoke, fig_root=args.out, data_path=args.data)


if __name__ == "__main__":
    main()
