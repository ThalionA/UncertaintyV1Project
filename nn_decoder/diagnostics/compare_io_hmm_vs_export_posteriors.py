"""Old vs new ideal-observer posteriors, per matched trial, on a common x-axis.

Old: `post_s_marginal` from VR_Decoder_Data_Export.mat — 91 bins, linear 0-90 deg
(1 deg steps). New: `PS_stim_G_tr` from data/fitted_data_and_posteriors.pkl
(collaborator IO-HMM refit) — 72 circular bins x 2.5 deg spanning [0, 180).

X-axis matching: the new posterior is folded about 90 deg (mass at theta and
180-theta summed; orientation is pi-periodic and task stimuli span [0, 90]),
converted to a per-degree density, linearly resampled onto the old 1-deg grid and
renormalised. Both families are then probability vectors on the same 91-point
grid. Mass beyond 90 deg in the raw new posterior (which the old support cannot
represent) is reported separately before folding.

Trials are matched via the stimulus-condition barcode alignment (export trials
are an in-order subsequence of pkl trials). Prefers nn_decoder/io_hmm_data.py for
loading/alignment; falls back to a local partial-file recovery so the comparison
runs on the truncated Slack copy (mouse 0) before the full pkl arrives.

NATIVE vs FOLDED — which panel uses which support
-------------------------------------------------
Folding is lossy (a median ~48% of new-posterior mass sits beyond 90 deg) and it
*changes width estimates*, so the two supports are never silently mixed:

* **fig1, fig2, fig3** (target overlays, per-trial scatters): FOLDED. Both
  families on the 91-point 1-deg grid; the only way to draw them on one axis.
* **fig4** (width vs stimulus): NATIVE, and **referenced to each family's own
  ignorance level** — see the next section. The folded new width is over-plotted
  **dashed** as the explicit cross-check, so the size of the folding artefact is
  visible rather than assumed away.
* **fig5** (variance decomposition): NATIVE. Every summary used here is
  support-free by construction — the doubled-angle resultant vector (dimensionless,
  in the unit disc), differential entropy (nats, degrees as the unit, so the
  ``ln delta`` term absorbs the bin width), and per-trial TV to the condition-mean
  posterior (each family against its own condition mean, on its own support).
* **fig6** (localising the disagreement): TV distance and the mean-vs-mean scatter
  need a shared support and are FOLDED; the angular disagreement panel is NATIVE
  (a doubled-angle difference, which needs no common grid).

Raw Shannon entropy is *not* comparable across 72 vs 91 bins (uniform is ln 72
vs ln 91); differential entropy ``-sum p ln p + ln delta`` is, and is what fig4/5
plot.

WHY fig4 PLOTS NO ABSOLUTE WIDTHS OR ENTROPIES
----------------------------------------------
Bin-width invariance is not enough: the two families' native width statistics are
*incommensurable in level*, so drawing them against a shared "posterior width
(deg)" axis invites a comparison that means nothing. Two reasons:

1. Different estimators. The old family lives on a linear 0-90 support (ordinary
   SD); the new one is circular on [0,180) (doubled-angle dispersion). A uniform
   posterior — total ignorance — measures 26.3 deg on the old support and
   40.5 deg on the new. An old posterior at 28 deg is *wider* than ignorance; a
   new one at 36 deg is *narrower*. On one axis the new arm merely looks broader.
2. Different ignorance ceilings. A maximally-ignorant posterior has differential
   entropy ln 180 = 5.19 nats on the new support and ln 91 = 4.51 on the old, so
   the new arm sits ~0.7 nats higher for free.

fig4 therefore plots each family **relative to its own ignorance level**: width as
a fraction of the uniform-posterior width on that support (1 = as uncertain as
knowing nothing) and entropy as nats *below* uniform on that support (0 =
ignorance). Both references are drawn. The normalisation is affine, so the
``eta^2`` / spread-per-SD effect sizes in the panel titles — the actual
comparison, *modulation by stimulus* — are unchanged by it.

One wrinkle, stated rather than hidden: the Mardia circular SD ``sqrt(-2 ln R)``
used elsewhere (fig5, the decoder scorecards) DIVERGES as R -> 0, so a uniform
circular posterior has no finite width under it and cannot be used as a
denominator. fig4's width panels therefore use the bounded angular deviation
``sqrt(2(1-R))``, which agrees with the Mardia SD for concentrated posteriors and
stays finite at ignorance. Both are monotone in R; the Mardia values are still
printed to stdout as the raw native cross-check, and the old arm is additionally
drawn under the *same* axial estimator as the new one (dotted) so it is visible
that the estimator choice is not what separates the families.

Usage: python diagnostics/compare_io_hmm_vs_export_posteriors.py [--mouse-ids 0 ...]
Figures -> nn_decoder/figures/io_hmm_vs_export/ (png+svg via figsave.save_fig).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import types
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # nn_decoder/
from figsave import save_fig                   # noqa: E402
from utils import load_vr_export               # noqa: E402

REPO = _HERE.parent.parent
PKL_DEFAULT = REPO / "data" / "fitted_data_and_posteriors.pkl"
OUT_DIR = _HERE.parent / "figures" / "io_hmm_vs_export"

OLD_GRID = np.arange(91.0)                     # 1-deg bins, 0..90
NEW_GRID_FULL = np.arange(72) * 2.5            # circular, [0, 180)
FOLD_GRID = np.arange(37) * 2.5                # after folding, 0..90


# ---------------------------------------------------------------- loading ----
def _load_new_posteriors(pkl_path):
    """Return {mouse_id: {'ps': (n,72) rows sum 1, 'data': dict}} — via
    io_hmm_data if available, else local partial recovery."""
    try:
        import io_hmm_data                     # written by the wiring session
        mice = io_hmm_data.load_io_hmm_pkl(str(pkl_path), allow_partial=True)
        out = {}
        for m, entry in mice.items():
            d = entry["data"]
            ps = np.asarray(d["PS_stim_G_tr"], dtype=np.float64)
            if ps.shape[0] == 72:              # bins-first layout
                ps = ps.T
            out[m] = {"ps": ps / ps.sum(1, keepdims=True), "data": d}
        return out
    except ImportError:
        warnings.warn("io_hmm_data not importable — using local partial recovery")
        return _recover_partial(pkl_path)


def _recover_partial(pkl_path):
    class _StubBase:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def _stub_module(name):
        mod = types.ModuleType(name)

        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            cls = type(attr, (_StubBase,), {"__module__": name})
            setattr(mod, attr, cls)
            return cls

        mod.__getattr__ = __getattr__
        sys.modules[name] = mod

    _stub_module("datastructs")
    for name in ("jax", "jax._src"):
        sys.modules.setdefault(name, types.ModuleType(name))
    arr_mod = types.ModuleType("jax._src.array")

    def _reconstruct_array(fun, args, arr_state, *rest):
        v = fun(*args)
        v.__setstate__(arr_state)
        return v

    arr_mod._reconstruct_array = _reconstruct_array
    sys.modules["jax._src.array"] = arr_mod

    up = pickle._Unpickler(open(pkl_path, "rb"))
    try:
        up.load()
    except Exception:
        pass                                    # truncated copy — expected
    out = {}
    datas = [v for v in up.memo.values() if type(v).__name__ == "Data"]
    for m, dobj in enumerate(datas):
        d = dobj.__dict__
        if d.get("PS_stim_G_tr") is None:
            continue
        ps = np.asarray(d["PS_stim_G_tr"], dtype=np.float64)
        if ps.shape[0] == 72:
            ps = ps.T
        out[m] = {"ps": ps / ps.sum(1, keepdims=True), "data": d}
    return out


def _align(trials, data):
    """Greedy in-order barcode alignment; returns pkl index per export trial."""
    mo = np.asarray(trials["orientation"]).ravel().astype(float)
    md = np.asarray(trials["dispersion"]).ravel().astype(float)
    mc = np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3)
    po = np.asarray(data["orientation"]).astype(float)
    pdis = np.asarray(data["dispersion"]).astype(float)
    pc = np.round(np.asarray(data["contrast"]).astype(float), 3)
    idx, j = [], 0
    for i in range(len(mo)):
        s = (mo[i], md[i], mc[i])
        while j < len(po) and (po[j], pdis[j], pc[j]) != s:
            j += 1
        if j == len(po):
            raise RuntimeError(f"barcode alignment failed at export trial {i}")
        idx.append(j)
        j += 1
    return np.array(idx)


# ------------------------------------------------------------- x matching ----
def fold_and_resample(ps_new):
    """(n, 72) circular [0,180) -> mass>90 (n,), density on FOLD_GRID (n,37),
    and probability rows on OLD_GRID (n, 91)."""
    mass_beyond = ps_new[:, 37:].sum(1)
    folded = np.empty((ps_new.shape[0], 37))
    folded[:, 0] = ps_new[:, 0]
    folded[:, 36] = ps_new[:, 36]
    for k in range(1, 36):
        folded[:, k] = ps_new[:, k] + ps_new[:, 72 - k]
    # per-degree density: interior bins cover 2.5 deg; the 0 and 90 bins only
    # half of their circular width lands in [0, 90] (edge effect, second-order
    # for posteriors this broad)
    width = np.full(37, 2.5)
    width[[0, 36]] = 1.25
    dens = folded / width
    on_old = np.array([np.interp(OLD_GRID, FOLD_GRID, dens[i]) for i in range(len(dens))])
    on_old /= on_old.sum(1, keepdims=True)
    return mass_beyond, dens, on_old


def _moments(p, grid):
    mu = (p * grid).sum(1)
    sd = np.sqrt((p * (grid - mu[:, None]) ** 2).sum(1))
    return mu, sd


def _diff_entropy(p, delta):
    """Differential entropy in nats (grid-size independent): -sum p ln p + ln(delta)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    return h + np.log(delta)


# ------------------------------------------- native circular statistics ----
# Orientation is pi-periodic, so every circular quantity below is computed on the
# DOUBLED angle (phi = 2*theta, period 360 deg) and mapped back. This works
# unchanged on the new 72-bin [0,180) grid and on the old 91-bin [0,90] grid — no
# folding, no wrap pathology at ori 0 (where the new posterior straddles bin 71/0
# and a linear mean would return ~90 deg).

def _resultant(p, grid):
    """First circular moment on the doubled angle: (n,2) array [Re, Im].

    |z| = R in [0,1] is the concentration (1 = delta function, 0 = uniform or a
    perfectly balanced cardinal-U), arg(z)/2 is the preferred orientation. The
    vector itself is dimensionless and bin-count free, so it is the one location
    summary comparable across the two supports without folding.
    """
    phi = np.deg2rad(2.0 * np.asarray(grid, float))
    return np.stack([(p * np.cos(phi)).sum(1), (p * np.sin(phi)).sum(1)], axis=1)


def _circ_sd(p, grid):
    """Mardia circular SD of the doubled angle, expressed in orientation degrees.

    sqrt(-2 ln R)/2 rad; matches the ordinary SD for concentrated posteriors and
    diverges as R -> 0. The new posteriors sit at R ~ 0.2-0.3, i.e. in the regime
    where this estimator is heavy-tailed — that is why fig4/fig5 lead with the
    differential entropy and treat this as the intuitive-but-noisy companion.
    """
    r = np.clip(np.hypot(*_resultant(p, grid).T), 1e-12, None)
    return np.rad2deg(np.sqrt(-2.0 * np.log(r)) / 2.0)


def _axial_dispersion(p, grid):
    """BOUNDED axial dispersion in orientation degrees: sqrt(2(1-R))/2 rad.

    Mardia's angular deviation on the doubled angle. It agrees with ``_circ_sd``
    (and with an ordinary SD) for concentrated posteriors but, unlike
    ``sqrt(-2 ln R)``, stays FINITE as R -> 0: a uniform posterior measures
    40.5 deg on the new [0,180) grid and 24.7 deg on the old 0-90 one. That is
    what makes an ignorance-referenced width ratio definable at all — see the
    module docstring. Monotone in R, so it orders posteriors identically to
    ``_circ_sd``.
    """
    r = np.clip(np.hypot(*_resultant(p, grid).T), 0.0, 1.0)
    return np.rad2deg(np.sqrt(2.0 * (1.0 - r)) / 2.0)


def _circ_mean(p, grid):
    """Preferred orientation in [0, 180) from the doubled-angle resultant."""
    z = _resultant(p, grid)
    return (np.rad2deg(np.arctan2(z[:, 1], z[:, 0])) / 2.0) % 180.0


def _circ_diff(a, b):
    """Absolute pi-periodic difference between two orientations, in [0, 90]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)


# ------------------------------------------------- ignorance (uniform) levels ----
# Every width/entropy series in fig4 is expressed against the value a UNIFORM
# posterior takes on the SAME support with the SAME estimator, so the two families
# can be compared as modulation rather than as level. Computed here from the grids
# themselves rather than hard-coded, so they follow any change of support.
def _uniform_row(grid):
    return np.full((1, len(grid)), 1.0 / len(grid))


U_WIDTH_OLD_LIN = float(_moments(_uniform_row(OLD_GRID), OLD_GRID)[1][0])          # 26.27 deg
U_WIDTH_OLD_AX = float(_axial_dispersion(_uniform_row(OLD_GRID), OLD_GRID)[0])     # 24.66 deg
U_WIDTH_NEW_AX = float(_axial_dispersion(_uniform_row(NEW_GRID_FULL),
                                         NEW_GRID_FULL)[0])                        # 40.51 deg
H_UNIF_OLD = float(np.log(len(OLD_GRID) * 1.0))                                    # ln 91  = 4.51
H_UNIF_NEW = float(np.log(len(NEW_GRID_FULL) * 2.5))                               # ln 180 = 5.19


# ------------------------------------------------------------ effect size ----
def _eta2(x, factor):
    """Marginal (unadjusted) eta-squared: SS_between / SS_total for one factor."""
    x = np.asarray(x, float)
    _, inv = np.unique(factor, return_inverse=True)
    gm = x.mean()
    ssb = sum((inv == c).sum() * (x[inv == c].mean() - gm) ** 2 for c in np.unique(inv))
    sst = ((x - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


def _std_spread(x, factor):
    """(max - min of the level means) / pooled within-level SD — Cohen's-d units."""
    x = np.asarray(x, float)
    _, inv = np.unique(factor, return_inverse=True)
    mus = np.array([x[inv == c].mean() for c in np.unique(inv)])
    var = np.array([x[inv == c].var(ddof=1) for c in np.unique(inv) if (inv == c).sum() > 1])
    pooled = np.sqrt(var.mean()) if len(var) else np.nan
    return float((mus.max() - mus.min()) / pooled) if pooled else np.nan


def _var_components(x, labels):
    """One-way ANOVA variance components for x grouped by `labels`.

    x may be (n,) or (n, k) — for the 2-D resultant vector the two coordinates'
    sums-of-squares are pooled, which is the standard multivariate one-way split.
    Returns the *unbiased* between component (MS_b - MS_w)/n0 rather than
    SS_b/SS_total: with a median of 6 trials per condition the naive fraction
    credits a chunk of pure sampling noise to the stimulus.
    """
    x = np.asarray(x, float).reshape(len(x), -1)
    _, inv = np.unique(labels, return_inverse=True)
    cs = np.unique(inv)
    n, c = len(x), len(cs)
    gm = x.mean(0)
    ns = np.array([(inv == k).sum() for k in cs], float)
    ssb = float(sum((inv == k).sum() * ((x[inv == k].mean(0) - gm) ** 2).sum() for k in cs))
    ssw = float(sum(((x[inv == k] - x[inv == k].mean(0)) ** 2).sum() for k in cs))
    msw = ssw / (n - c)
    msb = ssb / (c - 1)
    n0 = (n - (ns ** 2).sum() / n) / (c - 1)
    between = max(0.0, (msb - msw) / n0)
    return dict(between=between, within=msw, frac_within=msw / (between + msw),
                naive_frac_within=ssw / (ssb + ssw), n_cond=c, n_trials=n, n0=n0)


def _within_cond_tv(p, labels):
    """Per-trial TV distance to that trial's condition-mean posterior (native support).

    A support-aware measure of how much the posterior actually reshapes from trial
    to trial *within* a fixed stimulus — the raw material any decoder would have to
    read out of the neural activity. Conditions with a single trial are dropped.
    """
    _, inv = np.unique(labels, return_inverse=True)
    out = []
    for k in np.unique(inv):
        m = inv == k
        if m.sum() < 2:
            continue
        out.append(0.5 * np.abs(p[m] - p[m].mean(0)).sum(1))
    return np.concatenate(out) if out else np.array([])


def _cond_labels(ori, ct, disp):
    """Full crossed stimulus barcode, one label per trial."""
    return np.array([f"{o:.0f}|{c:g}|{d:g}" for o, c, d in zip(ori, ct, disp)])


def _level_stats(x, factor, levels):
    """mean and SEM of x at each level of `factor` (NaN where the level is empty)."""
    mu = np.full(len(levels), np.nan)
    se = np.full(len(levels), np.nan)
    for i, lv in enumerate(levels):
        m = factor == lv
        if m.sum() == 0:
            continue
        mu[i] = x[m].mean()
        se[i] = x[m].std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else 0.0
    return mu, se


# ------------------------------------------------------------------ plots ----
def _panel(ax, old_p, new_dens_on_old, ori, label_old, label_new):
    ax.plot(OLD_GRID, old_p, lw=1.4, color="tab:blue", label=label_old)
    ax.plot(OLD_GRID, new_dens_on_old, lw=1.4, color="tab:orange", label=label_new)
    ax.axvline(ori, color="k", lw=0.7, ls=":")
    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=0)


def _native_panel(ax, old_row, new_row, ori, shade=True):
    """Draw both families on ONE 0-180 deg axis with NO mirroring.

    Each is a per-degree density on its own native bins — old: 91 bins x 1 deg
    over [0,90]; new: 72 bins x 2.5 deg over [0,180). Both integrate to 1 over
    their OWN domain, which is the honest statement: they are distributions over
    different supports, and the new one spreads the same unit mass over twice
    the range. The 90-180 band is shaded to mark where the old ideal observer
    has no support at all (not zero probability — undefined).
    """
    if shade:
        ax.axvspan(90, 180, color="0.92", zorder=0, lw=0)
    ax.plot(OLD_GRID, old_row, lw=1.4, color="tab:blue", label="old (native, 0-90)")
    # close the circle visually: the new support wraps 177.5 -> 0
    gx = np.concatenate([NEW_GRID_FULL, [180.0]])
    gy = np.concatenate([new_row / 2.5, [new_row[0] / 2.5]])
    ax.plot(gx, gy, lw=1.4, color="tab:orange", label="new (native, 0-180)")
    ax.axvline(ori, color="k", lw=0.7, ls=":")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(bottom=0)


def make_native_figures(mouse, old_p, ps_new, trials, out_dir):
    """Unmirrored counterparts of fig1/fig2 — no fold, no resampling."""
    ori = np.asarray(trials["orientation"]).ravel().astype(float)
    ct = np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3)

    # fig 1n — example trials (the trial closest to its condition mean)
    cts, oris = [1.0, 0.25, 0.01], [0.0, 40.0, 50.0, 90.0]
    fig, axes = plt.subplots(len(cts), len(oris), figsize=(12, 7.2), sharex=True)
    for r, c in enumerate(cts):
        for k, o in enumerate(oris):
            ax = axes[r, k]
            cand = np.where((ct == c) & (ori == o))[0]
            if len(cand) == 0:
                ax.set_axis_off()
                continue
            cm = ps_new[cand].mean(0)
            t = cand[np.argmin(np.abs(ps_new[cand] - cm).sum(1))]
            _native_panel(ax, old_p[t], ps_new[t], o)
            ax.set_title(f"ori {o:.0f}  ctr {c:g}  trial {t}", fontsize=8)
            if r == 0 and k == 0:
                ax.legend(fontsize=7, frameon=False)
            if r == len(cts) - 1:
                ax.set_xlabel("orientation (deg)", fontsize=8)
            if k == 0:
                ax.set_ylabel("prob / deg", fontsize=8)
    fig.suptitle(
        f"Mouse {mouse} — example matched trials, NO MIRRORING: each family on "
        "its own native support\n"
        "old 91 bins x 1 deg over [0,90] (blue) | new 72 bins x 2.5 deg over "
        "[0,180) (orange); shaded = outside the old model's support",
        fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig1n_example_trials_native", max_px=1560)

    # fig 2n — condition means, unmirrored
    u_oris = np.unique(ori)
    u_cts = sorted(np.unique(ct))[::-1]
    fig, axes = plt.subplots(len(u_cts), len(u_oris), figsize=(14, 7.4),
                             sharex=True, sharey="row")
    for r, c in enumerate(u_cts):
        for k, o in enumerate(u_oris):
            ax = axes[r, k]
            m = (ct == c) & (ori == o)
            if m.sum() == 0:
                ax.set_axis_off()
                continue
            _native_panel(ax, old_p[m].mean(0), ps_new[m].mean(0), o)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"ori {o:.0f}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"ctr {c:g}\nprob / deg", fontsize=8)
            if r == len(u_cts) - 1:
                ax.set_xlabel("orientation (deg)", fontsize=7)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(
        f"Mouse {mouse} — condition-mean posteriors, NO MIRRORING: each family "
        "on its own native support\n"
        "both integrate to 1 over their OWN domain, so the new family sits lower "
        "for spreading unit mass over twice the range; shaded = no old support",
        fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig2n_condition_means_native", max_px=1560)

    # what the reflection assumption would be asserting, quantified
    beyond = ps_new[:, 37:].sum(1)
    half = ps_new[:, 1:36]
    mirror = ps_new[:, 37:][:, ::-1]          # bins 71..37 <-> 1..35
    with np.errstate(invalid="ignore", divide="ignore"):
        asym = np.abs(half - mirror).sum(1) / (half + mirror).sum(1)
    print(f"  native view: mass beyond 90 deg median {np.median(beyond):.3f}; "
          f"reflection asymmetry |p(t)-p(180-t)| / total median {np.median(asym):.3f} "
          f"(0 = the two halves are exact mirrors, so folding would lose nothing)")
    return float(np.median(asym))


def make_figures(mouse, old_p, new_on_old, mass_beyond, trials, out_dir):
    ori = np.asarray(trials["orientation"]).ravel().astype(float)
    ct = np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3)
    disp = np.asarray(trials["dispersion"]).ravel().astype(float)

    mu_o, sd_o = _moments(old_p, OLD_GRID)
    mu_n, sd_n = _moments(new_on_old, OLD_GRID)
    h_o = _diff_entropy(old_p, 1.0)
    h_n = _diff_entropy(new_on_old, 1.0)
    tv = 0.5 * np.abs(old_p - new_on_old).sum(1)

    # fig 1 — example single trials: rows = contrast, cols = orientation
    cts = [1.0, 0.25, 0.01]
    oris = [0.0, 40.0, 50.0, 90.0]
    fig, axes = plt.subplots(len(cts), len(oris), figsize=(11, 7), sharex=True)
    for r, c in enumerate(cts):
        for k, o in enumerate(oris):
            ax = axes[r, k]
            cand = np.where((ct == c) & (ori == o))[0]
            if len(cand) == 0:
                ax.set_axis_off()
                continue
            # most representative trial: min TV distance to the condition mean
            # (single new-model trials swing wrong-way often enough that a random
            # pick misleads — see fig 3 spread)
            cell_mean = new_on_old[cand].mean(0)
            t = cand[np.argmin(np.abs(new_on_old[cand] - cell_mean).sum(1))]
            _panel(ax, old_p[t], new_on_old[t], o,
                   "old (91-bin Q)", "new (IO-HMM, folded)")
            ax.set_title(f"ori {o:.0f}  ctr {c:g}  trial {t}", fontsize=8)
            if r == 0 and k == 0:
                ax.legend(fontsize=7, frameon=False)
            if r == len(cts) - 1:
                ax.set_xlabel("orientation (deg)")
            if k == 0:
                ax.set_ylabel("prob / deg")
    fig.suptitle(f"Mouse {mouse} — example matched trials, old vs new posterior "
                 "(new folded about 90deg, resampled to 1deg grid)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig1_example_trials", max_px=1560)

    # fig 2 — condition means: rows = contrast, cols = orientation (all 9)
    u_oris = np.unique(ori)
    u_cts = sorted(np.unique(ct))[::-1]
    fig, axes = plt.subplots(len(u_cts), len(u_oris), figsize=(13.5, 7),
                             sharex=True, sharey="row")
    for r, c in enumerate(u_cts):
        for k, o in enumerate(u_oris):
            ax = axes[r, k]
            m = (ct == c) & (ori == o)
            if m.sum() == 0:
                ax.set_axis_off()
                continue
            _panel(ax, old_p[m].mean(0), new_on_old[m].mean(0), o, "old", "new")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"ori {o:.0f}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"ctr {c:g}\nprob / deg", fontsize=8)
            if r == len(u_cts) - 1:
                ax.set_xlabel("orientation (deg)", fontsize=7.5)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Mouse {mouse} — condition-mean posteriors, old vs new target "
                 "family (n trials per cell vary)\n"
                 "FOLDED: the new arm is folded about 90 deg and resampled onto the "
                 "old 1-deg grid, the only way to draw both on one axis",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig2_condition_means", max_px=1560)

    # fig 3 — per-trial summary scatters + mass beyond 90
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    ax = axes[0, 0]
    sc = ax.scatter(mu_o, mu_n, c=ori, cmap="viridis", s=8, alpha=0.6)
    ax.plot([0, 90], [0, 90], "k:", lw=0.8)
    ax.set_xlabel("old posterior mean (deg)")
    ax.set_ylabel("new posterior mean (deg)")
    r_mu = np.corrcoef(mu_o, mu_n)[0, 1]
    ax.set_title(f"means, r={r_mu:.2f}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="true ori (deg)")

    ax = axes[0, 1]
    sc = ax.scatter(sd_o, sd_n, c=np.log10(ct), cmap="magma", s=8, alpha=0.6)
    lims = [0, max(sd_o.max(), sd_n.max()) * 1.05]
    ax.plot(lims, lims, "k:", lw=0.8)
    ax.set_xlabel("old posterior SD (deg)")
    ax.set_ylabel("new posterior SD (deg)")
    r_sd = np.corrcoef(sd_o, sd_n)[0, 1]
    ax.set_title(f"widths, r={r_sd:.2f}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="log10 contrast")

    ax = axes[1, 0]
    sc = ax.scatter(h_o, h_n, c=disp, cmap="cividis", s=8, alpha=0.6)
    lims = [min(h_o.min(), h_n.min()) - 0.1, max(h_o.max(), h_n.max()) + 0.1]
    ax.plot(lims, lims, "k:", lw=0.8)
    ax.set_xlabel("old differential entropy (nats)")
    ax.set_ylabel("new differential entropy (nats)")
    r_h = np.corrcoef(h_o, h_n)[0, 1]
    ax.set_title(f"entropies, r={r_h:.2f}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="dispersion (deg)")

    ax = axes[1, 1]
    for c in u_cts:
        m = ct == c
        ax.hist(mass_beyond[m], bins=np.linspace(0, mass_beyond.max() * 1.02, 30),
                histtype="step", lw=1.3, label=f"ctr {c:g}", density=True)
    ax.set_xlabel("new-posterior mass beyond 90 deg (pre-fold)")
    ax.set_ylabel("trial density")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("mass the old support cannot represent", fontsize=9)

    fig.suptitle(f"Mouse {mouse} — per-trial old-vs-new summaries "
                 f"(median TV distance {np.median(tv):.3f})", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig3_trial_summaries", max_px=1560)

    return dict(r_mean=r_mu, r_sd=r_sd, r_entropy=r_h, tv_median=float(np.median(tv)),
                sd_old_median=float(np.median(sd_o)), sd_new_median=float(np.median(sd_n)),
                h_old_median=float(np.median(h_o)), h_new_median=float(np.median(h_n)),
                mass_beyond_median=float(np.median(mass_beyond)))


# -------------------------------------------- why the decoders differ ----
OLD_C, NEW_C = "tab:blue", "tab:orange"

# Concentration floor below which a posterior has no meaningful preferred
# orientation (fig6 lower-right gate). R = |first circular moment on the doubled
# angle|; a balanced cardinal-U or a flat posterior both sit at R ~ 0.
R_FLOOR = 0.1


def _levels(ct, disp):
    """Sorted factor levels + the OCCUPIED (contrast, dispersion) cells.

    The design is NOT crossed: only 8 of the 16 contrast x dispersion cells exist
    and contrast is confounded with dispersion (disp 5 only at ctr 0.25/1, disp 30
    only at ctr 0.01/0.5). Marginal per-factor effect sizes therefore each carry
    some of the other factor; the joint 8-cell factor is the unconfounded one.
    """
    u_ct = np.array(sorted(np.unique(ct)))
    u_disp = np.array(sorted(np.unique(disp)))
    cells = sorted({(float(c), float(d)) for c, d in zip(ct, disp)})
    return u_ct, u_disp, cells


def _fig_width_vs_stimulus(mouse, out_dir, ori, ct, disp, series, cellf, cells):
    """(a) Does posterior width track stimulus difficulty?

    NATIVE supports, every series divided/shifted by the value a UNIFORM posterior
    takes on THAT support with THAT estimator, so no absolute width or entropy is
    ever drawn on a shared axis (see the module docstring: the old family's uniform
    width is 26.3 deg and the new family's 40.5 deg, so raw degrees on one axis are
    meaningless). The ignorance level is drawn in every panel; the question the
    figure answers is how far each family MOVES away from it with the stimulus.
    """
    u_ct, u_disp, _ = _levels(ct, disp)
    u_ori = np.array(sorted(np.unique(ori)))
    factors = [("contrast", ct, u_ct, [f"{c:g}" for c in u_ct]),
               ("dispersion (deg)", disp, u_disp, [f"{d:g}" for d in u_disp]),
               ("orientation (deg)", ori, u_ori, [f"{o:.0f}" for o in u_ori]),
               ("contrast / dispersion cell", cellf, np.array([f"{c:g}/{d:g}" for c, d in cells]),
                [f"{c:g}\n{d:g}" for c, d in cells])]
    # (short name, series prefix, y label carrying the full normalisation,
    #  ignorance reference value, series suffixes to draw)
    measures = [
        ("posterior width", "wrel",
         "width / uniform width on own support\n(1 = as wide as ignorance)",
         1.0, ("old", "new", "new_fold", "old_ax")),
        ("differential entropy", "hrel",
         "differential entropy $-$ ignorance (nats)\n"
         f"(old $-$ ln 91 = {H_UNIF_OLD:.2f}; new $-$ ln 180 = {H_UNIF_NEW:.2f})",
         0.0, ("old", "new", "new_fold")),
    ]
    # every divisor / offset is named in the legend, since the y-axis has room for
    # the rule but not for three numbers
    styles = {
        "old": (f"old — native 0-90: linear SD $\\div$ {U_WIDTH_OLD_LIN:.1f} deg "
                "(entropy $-$ ln 91)", dict(color=OLD_C, ls="-", marker="o")),
        "new": (f"new — native [0,180) circular: axial $\\div$ {U_WIDTH_NEW_AX:.1f} deg "
                "(entropy $-$ ln 180)", dict(color=NEW_C, ls="-", marker="s")),
        "new_fold": (f"new — folded to 0-90: linear SD $\\div$ {U_WIDTH_OLD_LIN:.1f} deg "
                     "(fold artefact, cross-check)",
                     dict(color=NEW_C, ls="--", marker="^", alpha=0.6)),
        "old_ax": (f"old — axial estimator $\\div$ {U_WIDTH_OLD_AX:.1f} deg, as used for the "
                   "new arm (estimator cross-check; width panels only)",
                   dict(color=OLD_C, ls=":", marker="x", alpha=0.6)),
    }

    # NB the PNG lands at ~1612 px tall, a whisker over figsave's 1600 px aim:
    # bbox_inches="tight" grows the raster past figsize*dpi to fit the outside
    # legend. Still well inside the 2000 px reader limit and previews fine; the
    # 1600 px cap would need fixing in figsave, not hand-tuned here.
    fig, axes = plt.subplots(len(factors), 2, figsize=(10.5, 12.2), layout="constrained")
    for r, (fname, fvals, lv, ticks) in enumerate(factors):
        for c, (short, key, ylab, ref, arms) in enumerate(measures):
            ax = axes[r, c]
            xs = np.arange(len(lv))
            ax.axhline(ref, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=0)
            for arm in arms:
                lab, style = styles[arm]
                mu, se = _level_stats(series[f"{key}_{arm}"], fvals, lv)
                ax.errorbar(xs, mu, yerr=se, capsize=2, lw=1.4, ms=4, label=lab, **style)
            # eta^2 / spread-per-SD are affine-invariant, so these are identical to
            # the values the un-normalised series would give — the normalisation
            # changes the axis, never the modulation being reported.
            e_o = _eta2(series[f"{key}_old"], fvals)
            e_n = _eta2(series[f"{key}_new"], fvals)
            s_o = _std_spread(series[f"{key}_old"], fvals)
            s_n = _std_spread(series[f"{key}_new"], fvals)
            ax.set_title(f"{short} vs {fname}\n"
                         f"$\\eta^2$ old {e_o:.2f} / new {e_n:.2f}   "
                         f"spread/SD {s_o:.1f} / {s_n:.1f}", fontsize=8.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(ticks, fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel(fname, fontsize=8)
            ax.set_ylabel(ylab, fontsize=7.5)
            ax.margins(x=0.06)
    handles = [Line2D([], [], label=styles[a][0], **styles[a][1])
               for a in ("old", "new", "new_fold", "old_ax")]
    handles.append(Line2D([], [], color="0.55", lw=0.9, ls=(0, (4, 3)),
                          label="uniform posterior on the same support "
                                "(ignorance: 1.0 left, 0 nats right)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7.5,
               frameon=False)
    fig.suptitle(f"Mouse {mouse} — posterior width vs stimulus difficulty, mean $\\pm$ SEM "
                 "across trials\neach family on its NATIVE support and referenced to ITS OWN "
                 "ignorance level: the arms' absolute widths are\nincommensurable "
                 "(uniform = 26.3 deg old vs 40.5 deg new), so read MODULATION, not level",
                 fontsize=10)
    save_fig(fig, out_dir, f"m{mouse}_fig4_width_vs_stimulus", max_px=1560)


def _fig_variance_decomposition(mouse, out_dir, cond, comps, tv_old, tv_new,
                                tv_old_per, tv_new_per):
    """(b) Between- vs within-condition variance: what is there to decode? NATIVE."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), layout="constrained")

    # log y — the two components span >2 orders of magnitude between the arms, so
    # a stacked linear bar would render the new arm's entropy variance invisible
    handles = [Patch(facecolor="0.45", label="between-condition (stimulus-locked)"),
               Patch(facecolor="w", edgecolor="0.45", hatch="//",
                     label="within-condition (trial-specific)")]
    for ax, key, ylab, title in (
            (axes[0, 0], "resultant", "variance (unit-disc$^2$, log)",
             "posterior resultant vector (location + confidence)"),
            (axes[0, 1], "entropy", "variance (nats$^2$, log)",
             "differential entropy (posterior width)")):
        vals = []
        for i, arm in enumerate(("old", "new")):
            v = comps[key][arm]
            col = OLD_C if arm == "old" else NEW_C
            ax.bar(i - 0.19, v["between"], width=0.34, color=col)
            ax.bar(i + 0.19, v["within"], width=0.34, facecolor="w", edgecolor=col,
                   hatch="//", lw=1.2)
            vals += [v["between"], v["within"]]
            ax.text(i, max(v["between"], v["within"]) * 1.6,
                    f"{100 * v['frac_within']:.0f}% within", ha="center", va="bottom",
                    fontsize=8, color=col)
        ax.set_yscale("log")
        # bars on a log axis start at the axis floor, so put the floor well below
        # the smallest component or its bar renders as a sliver
        ax.set_ylim(min(vals) / 8, max(vals) * 12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["old (91-bin Q)", "new (IO-HMM)"], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")

    ax = axes[1, 0]
    keys = ["resultant", "entropy", "width"]
    labs = ["resultant\nvector", "differential\nentropy", "native width\n(deg)"]
    xs = np.arange(len(keys))
    for off, arm, col in ((-0.18, "old", OLD_C), (0.18, "new", NEW_C)):
        ax.bar(xs + off, [comps[k][arm]["frac_within"] for k in keys], width=0.34,
               color=col, label=f"{arm}")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("fraction of variance within condition", fontsize=8)
    ax.set_title("share of target variance the stimulus does NOT explain\n"
                 "(ANOVA components; dotted line = half)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    rng = np.random.default_rng(1)
    for i, (arm, per, allv, col) in enumerate((("old", tv_old_per, tv_old, OLD_C),
                                               ("new", tv_new_per, tv_new, NEW_C))):
        ax.bar(i, allv.mean(), yerr=allv.std(ddof=1) / np.sqrt(len(allv)), capsize=3,
               color=col, width=0.6, alpha=0.85)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(per)), per, s=7, color="k", alpha=0.35,
                   zorder=3, label="per-condition mean" if i == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["old (91-bin Q)", "new (IO-HMM)"], fontsize=8)
    ax.set_ylabel("mean TV to condition-mean posterior", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("trial-to-trial posterior reshaping within a fixed stimulus\n"
                 "(each family on its own support)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    n_cond = comps["entropy"]["old"]["n_cond"]
    fig.suptitle(f"Mouse {mouse} — where the target variance lives: {n_cond} occupied "
                 "(orientation x contrast x dispersion) conditions\n"
                 "solid = stimulus-locked variance; hatched = trial-specific, decodable "
                 "only if V1 activity carries it", fontsize=10)
    save_fig(fig, out_dir, f"m{mouse}_fig5_variance_decomposition", max_px=1560)


def _cell_grid(vals, ct, disp, u_ct, u_disp):
    """(n_ct, n_disp) mean of `vals` and the contributing counts.

    NaN entries in `vals` (trials excluded by a gate) are skipped, and the count
    reports the trials that actually contributed — so a gated panel never implies
    a bigger n than it used.
    """
    g = np.full((len(u_ct), len(u_disp)), np.nan)
    n = np.zeros_like(g)
    for i, c in enumerate(u_ct):
        for j, dd in enumerate(u_disp):
            m = (ct == c) & (disp == dd) & np.isfinite(vals)
            n[i, j] = m.sum()
            if m.sum():
                g[i, j] = vals[m].mean()
    return g, n


def _heat(ax, g, n, u_ct, u_disp, cmap, label, title):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.88")
    im = ax.imshow(np.ma.masked_invalid(g), cmap=cm, aspect="auto")
    ax.set_xticks(range(len(u_disp)))
    ax.set_xticklabels([f"{d:g}" for d in u_disp], fontsize=7.5)
    ax.set_yticks(range(len(u_ct)))
    ax.set_yticklabels([f"{c:g}" for c in u_ct], fontsize=7.5)
    ax.set_xlabel("dispersion (deg)", fontsize=8)
    ax.set_ylabel("contrast", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if np.isnan(g[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.45")
                continue
            # pick the annotation colour from the *rendered* cell luminance —
            # magma/viridis both run dark->light, so a value-threshold rule puts
            # white text on the brightest cells
            r, gr, b, _ = im.cmap(im.norm(g[i, j]))
            shade = "k" if (0.299 * r + 0.587 * gr + 0.114 * b) > 0.55 else "w"
            ax.text(j, i, f"{g[i, j]:.2f}\nn={int(n[i, j])}", ha="center", va="center",
                    fontsize=7, color=shade)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)


def _fig_tv_by_condition(mouse, out_dir, ori, ct, disp, tv, dang_gated, mu_o, mu_n, gate_kept):
    """(c) Localise the old-vs-new disagreement in stimulus space."""
    u_ct, u_disp, _ = _levels(ct, disp)
    u_ori = np.array(sorted(np.unique(ori)))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), layout="constrained")

    g, n = _cell_grid(tv, ct, disp, u_ct, u_disp)
    _heat(axes[0, 0], g, n, u_ct, u_disp, "magma", "mean TV distance",
          f"TV(old, new) per stimulus cell (folded grid)\n"
          f"$\\eta^2$ over the 8 occupied cells = "
          f"{_eta2(tv, np.array([f'{c:g}/{d:g}' for c, d in zip(ct, disp)])):.2f}")

    ax = axes[0, 1]
    for c in u_ct:
        m = ct == c
        mu, se = _level_stats(tv[m], ori[m], u_ori)
        ax.errorbar(np.arange(len(u_ori)), mu, yerr=se, capsize=2, lw=1.3, ms=4,
                    marker="o", label=f"ctr {c:g}")
    ax.set_xticks(range(len(u_ori)))
    ax.set_xticklabels([f"{o:.0f}" for o in u_ori], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("stimulus orientation (deg)", fontsize=8)
    ax.set_ylabel("TV(old, new)", fontsize=8)
    ax.set_title(f"disagreement vs orientation, split by contrast\n"
                 f"$\\eta^2$ orientation = {_eta2(tv, ori):.2f}, contrast = {_eta2(tv, ct):.2f}",
                 fontsize=9)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))          # headroom so the legend clears the data
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center")

    ax = axes[1, 0]
    sc = ax.scatter(mu_o, mu_n, c=tv, cmap="magma", s=9, alpha=0.75)
    ax.plot([0, 90], [0, 90], "k:", lw=0.8)
    ax.set_xlabel("old posterior mean (deg)", fontsize=8)
    ax.set_ylabel("new posterior mean, folded (deg)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"joint view: where the two families place the stimulus\n"
                 f"r = {np.corrcoef(mu_o, mu_n)[0, 1]:.2f}, colour = per-trial TV", fontsize=9)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("TV(old, new)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    g2, n2 = _cell_grid(dang_gated, ct, disp, u_ct, u_disp)
    _heat(axes[1, 1], g2, n2, u_ct, u_disp, "viridis", "mean |$\\Delta$ ori| (deg)",
          "preferred-orientation disagreement (NATIVE doubled-angle)\n"
          f"both posteriors oriented, R > {R_FLOOR:g} ({100 * gate_kept:.0f}% of trials); "
          "45 deg = unrelated")

    fig.suptitle(f"Mouse {mouse} — localising the old-vs-new disagreement in stimulus space "
                 f"(median TV {np.median(tv):.3f})\n"
                 "grey cells with a dash are unoccupied: contrast and dispersion are "
                 "confounded, only 8 of 16 combinations were run", fontsize=10)
    save_fig(fig, out_dir, f"m{mouse}_fig6_tv_by_condition", max_px=1560)


def make_mechanism_figures(mouse, old_p, ps_new, new_on_old, trials, out_dir):
    """Figures 4-6: why the two target families train different decoders.

    Returns a dict of the headline dimensionless numbers (also printed).
    """
    ori = np.asarray(trials["orientation"]).ravel().astype(float)
    ct = np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3)
    disp = np.asarray(trials["dispersion"]).ravel().astype(float)
    _u_ct, _u_disp, cells = _levels(ct, disp)
    cellf = np.array([f"{c:g}/{d:g}" for c, d in zip(ct, disp)])
    cond = _cond_labels(ori, ct, disp)

    series = {
        # raw native quantities — printed, and used by fig5's variance components
        "width_old": _moments(old_p, OLD_GRID)[1],                 # native linear SD, 0-90
        "width_new": _circ_sd(ps_new, NEW_GRID_FULL),              # native Mardia circular SD
        "width_new_fold": _moments(new_on_old, OLD_GRID)[1],       # folded cross-check
        "entropy_old": _diff_entropy(old_p, 1.0),
        "entropy_new": _diff_entropy(ps_new, 2.5),
        "entropy_new_fold": _diff_entropy(new_on_old, 1.0),
    }
    # fig4 series: each referenced to what a UNIFORM posterior scores on the SAME
    # support with the SAME estimator. Widths use the bounded angular deviation on
    # the circle (the Mardia SD above has no finite uniform value, so it cannot be
    # a denominator); entropies are shifted, not divided, because differential
    # entropy is a log-scale quantity.
    series.update({
        "wrel_old": series["width_old"] / U_WIDTH_OLD_LIN,
        "wrel_new": _axial_dispersion(ps_new, NEW_GRID_FULL) / U_WIDTH_NEW_AX,
        "wrel_new_fold": series["width_new_fold"] / U_WIDTH_OLD_LIN,
        "wrel_old_ax": _axial_dispersion(old_p, OLD_GRID) / U_WIDTH_OLD_AX,
        "hrel_old": series["entropy_old"] - H_UNIF_OLD,
        "hrel_new": series["entropy_new"] - H_UNIF_NEW,
        "hrel_new_fold": series["entropy_new_fold"] - H_UNIF_OLD,
    })

    print(f"\n=== mouse {mouse} — (a) width vs stimulus (NATIVE, ignorance-referenced) ===")
    print("  marginal eta^2 (SS_between/SS_total) and (range of level means)/pooled within SD;")
    print("  contrast x dispersion is NOT crossed (8/16 cells) so the two marginals share variance.")
    print("  eta^2/d are affine-invariant: identical for the raw and the ignorance-referenced")
    print("  series, so the figure's normalisation cannot inflate or deflate them.")
    eff = {}
    for fname, fvals in (("contrast", ct), ("dispersion", disp), ("orientation", ori),
                         ("ct/disp(8)", cellf)):
        for key in ("width", "entropy", "wrel", "hrel"):
            for arm in ("old", "new", "new_fold"):
                x = series[f"{key}_{arm}"]
                eff[f"{key}|{arm}|{fname}"] = (_eta2(x, fvals), _std_spread(x, fvals))
        print(f"  {fname:12s}"
              + "".join(
                  f"  {key}[{arm}] eta2={eff[f'{key}|{arm}|{fname}'][0]:.3f} "
                  f"d={eff[f'{key}|{arm}|{fname}'][1]:.2f}"
                  for key in ("wrel", "hrel") for arm in ("old", "new")))
    print("  folding cross-check (new): "
          + ", ".join(f"{f}: circSD eta2 {eff[f'width|new|{f}'][0]:.3f} -> folded SD "
                      f"{eff[f'width|new_fold|{f}'][0]:.3f}"
                      for f in ("contrast", "dispersion", "ct/disp(8)")))
    print("  estimator cross-check (new arm, plotted axial vs raw Mardia circSD): "
          + ", ".join(f"{f}: {eff[f'wrel|new|{f}'][0]:.3f} vs {eff[f'width|new|{f}'][0]:.3f}"
                      for f in ("contrast", "dispersion", "ct/disp(8)")))
    print(f"  median raw width: old {np.median(series['width_old']):.1f} deg (linear SD), "
          f"new {np.median(series['width_new']):.1f} deg (Mardia circSD), "
          f"new folded {np.median(series['width_new_fold']):.1f} deg  "
          "— NOT comparable across arms, see below")
    print(f"  median width / own ignorance width: old {np.median(series['wrel_old']):.3f} "
          f"(uniform = {U_WIDTH_OLD_LIN:.1f} deg), "
          f"new {np.median(series['wrel_new']):.3f} (uniform = {U_WIDTH_NEW_AX:.1f} deg), "
          f"old under the axial estimator {np.median(series['wrel_old_ax']):.3f} "
          f"(uniform = {U_WIDTH_OLD_AX:.1f} deg)")
    print(f"  median differential entropy below ignorance: "
          f"old {np.median(series['hrel_old']):+.2f} nats (raw {np.median(series['entropy_old']):.2f}, "
          f"ignorance ln 91 = {H_UNIF_OLD:.2f}), "
          f"new {np.median(series['hrel_new']):+.2f} nats "
          f"(raw {np.median(series['entropy_new']):.2f}, ignorance ln 180 = {H_UNIF_NEW:.2f})")

    _fig_width_vs_stimulus(mouse, out_dir, ori, ct, disp, series, cellf, cells)

    # ---- (b) between vs within condition -------------------------------------
    comps = {
        "resultant": {"old": _var_components(_resultant(old_p, OLD_GRID), cond),
                      "new": _var_components(_resultant(ps_new, NEW_GRID_FULL), cond)},
        "entropy": {"old": _var_components(series["entropy_old"], cond),
                    "new": _var_components(series["entropy_new"], cond)},
        "width": {"old": _var_components(series["width_old"], cond),
                  "new": _var_components(series["width_new"], cond)},
    }
    tv_old = _within_cond_tv(old_p, cond)
    tv_new = _within_cond_tv(ps_new, cond)
    keep = np.array([(cond == k).sum() >= 2 for k in np.unique(cond)])
    ukeep = np.unique(cond)[keep]
    tv_old_per = np.array([(0.5 * np.abs(old_p[cond == k] - old_p[cond == k].mean(0)).sum(1)).mean()
                           for k in ukeep])
    tv_new_per = np.array([(0.5 * np.abs(ps_new[cond == k] - ps_new[cond == k].mean(0)).sum(1)).mean()
                           for k in ukeep])

    ref = comps["entropy"]["old"]
    print(f"\n=== mouse {mouse} — (b) between- vs within-condition variance (NATIVE) ===")
    print(f"  {ref['n_cond']} conditions, {ref['n_trials']} trials, effective n0={ref['n0']:.1f}; "
          "unbiased components (MS_b - MS_w)/n0")
    for key, unit in (("resultant", "unit-disc^2"), ("entropy", "nats^2"), ("width", "deg^2")):
        for arm in ("old", "new"):
            v = comps[key][arm]
            print(f"  {key:9s} {arm:3s}  between={v['between']:.4g} {unit:11s} "
                  f"within={v['within']:.4g}  frac_within={v['frac_within']:.3f} "
                  f"(naive {v['naive_frac_within']:.3f})  sqrt(within)={np.sqrt(v['within']):.3g}")
    rb = comps["resultant"]
    eb = comps["entropy"]
    print(f"  stimulus-locked signal ratio old/new: resultant "
          f"{rb['old']['between'] / max(rb['new']['between'], 1e-12):.1f}x, "
          f"entropy {eb['old']['between'] / max(eb['new']['between'], 1e-12):.1f}x")
    print(f"  within-condition TV to the condition mean: old {tv_old.mean():.4f} "
          f"+- {tv_old.std(ddof=1) / np.sqrt(len(tv_old)):.4f}, "
          f"new {tv_new.mean():.4f} +- {tv_new.std(ddof=1) / np.sqrt(len(tv_new)):.4f} "
          f"({tv_new.mean() / tv_old.mean():.1f}x)")
    print("  ...split by contrast (old is a near-deterministic function of the stimulus "
          "when the stimulus is weak):")
    for c in sorted(np.unique(ct)):
        s = ct == c
        print(f"    ctr {c:<5g} n={int(s.sum()):4d}  old {_within_cond_tv(old_p[s], cond[s]).mean():.4f}"
              f"   new {_within_cond_tv(ps_new[s], cond[s]).mean():.4f}")

    _fig_variance_decomposition(mouse, out_dir, cond, comps, tv_old, tv_new,
                                tv_old_per, tv_new_per)

    # ---- (c) localise the disagreement ---------------------------------------
    tv = 0.5 * np.abs(old_p - new_on_old).sum(1)                    # folded
    dang = _circ_diff(_circ_mean(old_p, OLD_GRID), _circ_mean(ps_new, NEW_GRID_FULL))
    r_old = np.hypot(*_resultant(old_p, OLD_GRID).T)
    r_new = np.hypot(*_resultant(ps_new, NEW_GRID_FULL).T)
    # A preferred orientation is only meaningful when the posterior is actually
    # oriented: an old cardinal-U (mass split at 0 and 90) and a flat new posterior
    # both have R -> 0, and their "preferred orientations" are then arbitrary — the
    # ungated mean lands near 45 deg (chance) purely as an estimator artefact.
    ok = (r_old > R_FLOOR) & (r_new > R_FLOOR)
    dang_gated = np.where(ok, dang, np.nan)
    mu_o = _moments(old_p, OLD_GRID)[0]
    mu_n = _moments(new_on_old, OLD_GRID)[0]

    print(f"\n=== mouse {mouse} — (c) where the two families disagree ===")
    print(f"  TV(old,new) mean {tv.mean():.3f}, median {np.median(tv):.3f}; "
          f"eta^2 by ct/disp cell {_eta2(tv, cellf):.3f}, contrast {_eta2(tv, ct):.3f}, "
          f"dispersion {_eta2(tv, disp):.3f}, orientation {_eta2(tv, ori):.3f}")
    print("    cell            n     TV    R_old  R_new  |d ori| all / gated (n gated)")
    for c, dd in cells:
        m = (ct == c) & (disp == dd)
        mg = m & ok
        gv = f"{dang[mg].mean():5.1f}" if mg.sum() else "   --"
        print(f"    ctr {c:<5g} disp {dd:<3g} {int(m.sum()):4d}  {tv[m].mean():.3f}  "
              f"{r_old[m].mean():.3f}  {r_new[m].mean():.3f}   {dang[m].mean():5.1f} / {gv} "
              f"({int(mg.sum())})")
    print(f"  preferred-orientation disagreement: ungated mean {dang.mean():.1f} deg "
          f"(median {np.median(dang):.1f}); gated at R>{R_FLOOR:g} "
          f"{np.nanmean(dang_gated):.1f} deg (median {np.nanmedian(dang_gated):.1f}, "
          f"{100 * ok.mean():.0f}% of trials kept). 45 deg = unrelated.")

    _fig_tv_by_condition(mouse, out_dir, ori, ct, disp, tv, dang_gated, mu_o, mu_n,
                         float(ok.mean()))

    return dict(eta2_entropy_ctdisp_old=eff["entropy|old|ct/disp(8)"][0],
                eta2_entropy_ctdisp_new=eff["entropy|new|ct/disp(8)"][0],
                width_over_ignorance_old=float(np.median(series["wrel_old"])),
                width_over_ignorance_new=float(np.median(series["wrel_new"])),
                nats_below_ignorance_old=float(np.median(series["hrel_old"])),
                nats_below_ignorance_new=float(np.median(series["hrel_new"])),
                frac_within_entropy_old=comps["entropy"]["old"]["frac_within"],
                frac_within_entropy_new=comps["entropy"]["new"]["frac_within"],
                frac_within_resultant_old=comps["resultant"]["old"]["frac_within"],
                frac_within_resultant_new=comps["resultant"]["new"]["frac_within"],
                tv_within_old=float(tv_old.mean()), tv_within_new=float(tv_new.mean()),
                dang_median_gated=float(np.nanmedian(dang_gated)),
                dang_gate_kept=float(ok.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-ids", type=int, nargs="*", default=None)
    ap.add_argument("--pkl", type=Path, default=PKL_DEFAULT)
    ap.add_argument("--no-mechanism", action="store_true",
                    help="skip figs 4-6 (width-vs-stimulus, variance split, TV by condition)")
    args = ap.parse_args()

    new = _load_new_posteriors(args.pkl)
    mice = args.mouse_ids if args.mouse_ids is not None else sorted(new)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mouse in mice:
        if mouse not in new:
            print(f"mouse {mouse}: not recoverable from pkl copy — skipped")
            continue
        acts, targets_perc, _td, _tl, trials = load_vr_export(mouse)
        old_p = np.asarray(targets_perc, dtype=np.float64)
        old_p /= old_p.sum(1, keepdims=True)
        idx = _align(trials, new[mouse]["data"])
        ps_new = new[mouse]["ps"][idx]
        mass_beyond, _dens, new_on_old = fold_and_resample(ps_new)
        stats = make_figures(mouse, old_p, new_on_old, mass_beyond, trials, OUT_DIR)
        make_native_figures(mouse, old_p, ps_new, trials, OUT_DIR)
        n_drop = new[mouse]["ps"].shape[0] - len(idx)
        print(f"mouse {mouse}: {len(idx)} matched trials ({n_drop} pkl-only), "
              + ", ".join(f"{k}={v:.3f}" for k, v in stats.items()))
        if not args.no_mechanism:
            mech = make_mechanism_figures(mouse, old_p, ps_new, new_on_old, trials, OUT_DIR)
            print(f"\nmouse {mouse} headline (dimensionless): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in mech.items()))


if __name__ == "__main__":
    main()
