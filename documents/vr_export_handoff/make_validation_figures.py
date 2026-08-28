"""Validation figures for VR_Decoder_Data_Export.mat (collaborator hand-off).

Produces four sanity-check figures so a recipient can confirm the export is the
real dataset before relying on it:

  fig1_orientation_tuning   - V1 neurons are orientation-tuned (example cells +
                              population tuning heatmap, sorted by preferred ori)
  fig2_grating_response     - grating-phase PSTH over the 2 s window, split by
                              contrast, + the mean contrast-response function
  fig3_corridor_position    - corridor (xC) population activity tiles track
                              position (per-neuron heatmap, example mouse)
  fig4_behaviour_and_io     - behavioural psychometric (P(go) vs orientation)
                              + example ideal-observer posteriors vs true stimulus

Each figure is written as a PNG (longest side <=1600 px) and an SVG (full detail),
same basename, into ./figures/. No conclusion boxes are drawn on the data
(takeaways live in the README).

Run from anywhere:
    python documents/vr_export_handoff/make_validation_figures.py
"""
from __future__ import annotations

import os

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
EXPORT = os.path.join(_REPO, "data", "VR_Decoder_Data_Export.mat")
OUTDIR = os.path.join(_HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

CONTRASTS = np.array([0.01, 0.25, 0.5, 0.99])      # task contrast levels
C_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(CONTRASTS)))


def _save(fig, stem):
    """Write SVG (full) + PNG (longest side <= 1600 px), same basename."""
    fig.savefig(os.path.join(OUTDIR, stem + ".svg"), bbox_inches="tight")
    w_in, h_in = fig.get_size_inches()
    dpi = min(200, 1600.0 / max(w_in, h_in))       # cap longest side ~1600 px
    fig.savefig(os.path.join(OUTDIR, stem + ".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg  ({int(max(w_in, h_in) * dpi)} px long side)")


def _clean_contrast(c):
    c = np.round(np.asarray(c, float), 2)
    c[c > 0.9] = 1.0
    return c


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
print(f"Loading {EXPORT} ...", flush=True)
mat = sio.loadmat(EXPORT, simplify_cells=True)
NS = mat["NeuralStore"]                              # list, one dict per mouse
TBL = mat["TrialTbl_Struct"]                          # pooled struct-of-arrays
TAGS = [e["tag"] for e in NS]
print(f"  {len(NS)} mice: {TAGS}")


def grating_rate(e):
    """(n_trials, n_neurons) mean inferred-spike-prob over the 2 s grating window."""
    return np.nanmean(e["Gspk"], axis=2)             # Gspk is (trial, neuron, time)


# --------------------------------------------------------------------------- #
# FIG 1 - orientation tuning
# --------------------------------------------------------------------------- #
def fig1_tuning():
    ORIS = np.array([0, 15, 30, 40, 45, 50, 60, 75, 90])
    tunings, sems, rates = [], [], []                 # per-neuron tuning over ORIS
    for e in NS:
        R = grating_rate(e)                           # (trials, neurons)
        c = _clean_contrast(e["contrast"])
        stim = np.asarray(e["stimulus"], float)
        hi = c >= 1.0                                 # full contrast: cleanest drive
        for n in range(R.shape[1]):
            mean = np.full(len(ORIS), np.nan)
            sem = np.full(len(ORIS), np.nan)
            for j, o in enumerate(ORIS):
                vals = R[hi & (stim == o), n]
                if vals.size:
                    mean[j] = np.nanmean(vals)
                    sem[j] = np.nanstd(vals) / np.sqrt(max(1, np.isfinite(vals).sum()))
            if np.all(np.isfinite(mean)):
                tunings.append(mean); sems.append(sem)
                rates.append(np.nanmean(R[:, n]))
    T = np.array(tunings)                              # (N, 9)
    S = np.array(sems)
    rates = np.array(rates)
    pref = ORIS[np.argmax(T, axis=1)]
    depth = (T.max(1) - T.min(1)) / (T.max(1) + T.min(1) + 1e-9)
    print(f"  fig1: {T.shape[0]} neurons pooled; median tuning depth {np.median(depth):.2f}")

    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2], wspace=0.32)

    # A+B: example well-tuned cells spanning preferred orientations
    targets = [0, 30, 45, 60, 90]
    chosen = []
    cand0 = np.where((depth > 0.4) & (rates > np.percentile(rates, 50)))[0]
    for tg in targets:
        order_c = cand0[np.lexsort((-depth[cand0], np.abs(pref[cand0] - tg)))]
        for ci in order_c:
            if ci not in chosen:
                chosen.append(ci); break
    for ax_i, sl in enumerate([chosen[:3], chosen[3:5]]):
        ax = fig.add_subplot(gs[0, ax_i])
        for ci in sl:
            pk = T[ci].max()
            ax.plot(ORIS, T[ci] / pk, "-o", ms=3, lw=1.4, label=f"pref {pref[ci]}°")
            ax.fill_between(ORIS, (T[ci] - S[ci]) / pk, (T[ci] + S[ci]) / pk,
                            alpha=0.18, lw=0)
        ax.set_xlabel("grating orientation (°)")
        ax.set_ylabel("norm. response" if ax_i == 0 else "")
        ax.set_title("example tuned cells" if ax_i == 0 else "")
        ax.set_ylim(0, 1.15)
        ax.set_xticks([0, 45, 90]); ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    # C: population tuning heatmap, rows = neurons sorted by preferred orientation
    order = np.argsort(pref + 0.001 * np.argmax(T, axis=1))
    Tn = T / T.max(1, keepdims=True)                   # peak-normalise each cell
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(Tn[order], aspect="auto", cmap="magma",
                   extent=[ORIS[0], ORIS[-1], T.shape[0], 0], interpolation="nearest")
    ax.set_xlabel("grating orientation (°)")
    ax.set_ylabel("neuron (sorted by preferred ori)")
    ax.set_title(f"population tuning  (n={T.shape[0]} cells, 6 mice)")
    ax.set_xticks([0, 45, 90])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="norm. response")
    fig.suptitle("Fig 1 · Orientation tuning (grating-phase mean, high contrast)",
                 y=1.02, fontsize=11)
    _save(fig, "fig1_orientation_tuning")


# --------------------------------------------------------------------------- #
# FIG 2 - grating-phase response: onset PSTH transient + contrast-response
#         (CRF measured the only un-confounded way: onset window, at each
#          cell's preferred orientation, over driven cells. The full-window
#          grand mean is NOT monotonic in contrast - see README caveats.)
# --------------------------------------------------------------------------- #
def fig2_grating_response():
    ORIS = np.array([0, 15, 30, 40, 45, 50, 60, 75, 90])
    LEVELS = np.array([0.01, 0.25, 0.5, 1.0])          # uniform 4-level (snapped)
    xG = NS[0]["xG"]                                   # 2 s window, 40 x 50 ms bins
    onset = xG <= 0.30                                  # evoked-transient window

    psth_by_mouse = []                                  # (neurons, time) per mouse
    crf_pooled = {cl: [] for cl in LEVELS}              # driven-cell onset resp / contrast
    crf_by_mouse = []                                   # per-mouse mean CRF
    for e in NS:
        c = _clean_contrast(e["contrast"])              # snaps 0.99 -> 1.0
        stim = np.asarray(e["stimulus"], float)
        G = e["Gspk"]                                   # (trials, neurons, time)
        psth_by_mouse.append(np.nanmean(G, axis=0))     # PSTH, all trials
        Gon = np.nanmean(G[:, :, onset], axis=2)        # (trials, neurons) onset mean
        hi = c >= 1.0
        # preferred orientation per neuron from high-contrast onset tuning
        tune = np.full((Gon.shape[1], len(ORIS)), np.nan)
        for j, o in enumerate(ORIS):
            sel = hi & (stim == o)
            if sel.sum():
                tune[:, j] = np.nanmean(Gon[sel], axis=0)
        ok = np.isfinite(tune).any(1)
        pref = np.full(Gon.shape[1], np.nan)
        pref[ok] = ORIS[np.nanargmax(tune[ok], axis=1)]
        depth = (np.nanmax(tune, 1) - np.nanmin(tune, 1)) / \
                (np.nanmax(tune, 1) + np.nanmin(tune, 1) + 1e-9)
        driven = ok & (depth > 0.2)                     # responsive, tuned cells
        mouse_crf = {cl: [] for cl in LEVELS}
        for n in np.where(driven)[0]:
            row = []
            for cl in LEVELS:
                m = np.isclose(c, cl) & (stim == pref[n])
                row.append(np.nanmean(Gon[m, n]) if m.sum() >= 3 else np.nan)
            if np.all(np.isfinite(row)):
                for cl, v in zip(LEVELS, row):
                    crf_pooled[cl].append(v); mouse_crf[cl].append(v)
        crf_by_mouse.append([np.nanmean(mouse_crf[cl]) if mouse_crf[cl] else np.nan
                             for cl in LEVELS])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))

    # A: pooled grating-phase PSTH (mean +/- SEM over neurons) + per-mouse lines
    for p in psth_by_mouse:
        axA.plot(xG, np.nanmean(p, 0), color="0.78", lw=0.9)
    P = np.vstack(psth_by_mouse)
    mu, sem = np.nanmean(P, 0), np.nanstd(P, 0) / np.sqrt(P.shape[0])
    axA.plot(xG, mu, color="k", lw=2, label=f"pooled (n={P.shape[0]})")
    axA.fill_between(xG, mu - sem, mu + sem, color="k", alpha=0.2, lw=0)
    axA.axvspan(0, 0.30, color="orange", alpha=0.12, lw=0)
    axA.set_xlabel("time from grating onset (s)")
    axA.set_ylabel("mean inferred spike prob.")
    axA.set_title("grating-phase PSTH (grey = each mouse)")
    axA.legend(fontsize=8, frameon=False); axA.spines[["top", "right"]].set_visible(False)

    # B: contrast-response, onset window, at preferred orientation, driven cells
    mus = [np.nanmean(crf_pooled[cl]) for cl in LEVELS]
    sems = [np.nanstd(crf_pooled[cl]) / np.sqrt(len(crf_pooled[cl])) for cl in LEVELS]
    n_cells = len(crf_pooled[LEVELS[0]])
    axB.errorbar(range(len(LEVELS)), mus, yerr=sems, fmt="-o", color="crimson",
                 lw=2, ms=5, capsize=3, label=f"driven cells (n={n_cells})")
    axB.set_xticks(range(len(LEVELS)))
    axB.set_xticklabels([f"{c:g}" for c in LEVELS])
    axB.set_xlabel("contrast")
    axB.set_ylabel("onset response (0-0.3 s) at preferred ori")
    axB.set_title("contrast-response (saturating)")
    axB.legend(fontsize=8, frameon=False); axB.spines[["top", "right"]].set_visible(False)
    print(f"  fig2: onset CRF over {n_cells} driven cells = {np.round(mus, 4)}")
    fig.suptitle("Fig 2 · Grating-phase response: evoked transient & contrast tuning",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig2_grating_response")


# --------------------------------------------------------------------------- #
# FIG 3 - corridor (xC) position tuning, example mouse
# --------------------------------------------------------------------------- #
def fig3_corridor_position(tag="Cb15"):
    e = NS[TAGS.index(tag)]
    xC = e["xC"]
    M = np.nanmean(e["Cspk"], axis=0)                  # (neurons, position)
    # keep position bins visited on > half of trials (corridor length varies)
    occ = np.mean(np.isfinite(e["Cspk"][:, 0, :]), 0)
    keep = occ > 0.5
    M = M[:, keep]; xpos = xC[keep]
    M = np.where(np.isfinite(M), M, 0.0)
    peak = M.max(1, keepdims=True); peak[peak == 0] = 1
    Mn = M / peak
    order = np.argsort(np.argmax(Mn, axis=1))

    fig, (axT, axH) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), height_ratios=[1, 3], sharex=True,
        gridspec_kw=dict(hspace=0.08))
    axT.plot(xpos, np.nanmean(M, 0), color="k", lw=1.8)
    axT.set_ylabel("pop. mean\nspike prob.")
    axT.set_title(f"corridor position tuning · mouse {tag} "
                  f"({M.shape[0]} neurons)")
    axT.spines[["top", "right"]].set_visible(False)
    im = axH.imshow(Mn[order], aspect="auto", cmap="magma",
                    extent=[xpos[0], xpos[-1], M.shape[0], 0], interpolation="nearest")
    axH.set_xlabel("corridor position (cm)")
    axH.set_ylabel("neuron (sorted by peak position)")
    fig.colorbar(im, ax=axH, fraction=0.046, pad=0.02, label="norm. response")
    print(f"  fig3: {tag} {M.shape[0]} neurons, {keep.sum()} position bins kept")
    fig.suptitle("Fig 3 · Corridor (spatial) responses tile track position",
                 y=0.99, fontsize=11)
    _save(fig, "fig3_corridor_position")


# --------------------------------------------------------------------------- #
# FIG 4 - behaviour psychometric + ideal-observer posteriors
# --------------------------------------------------------------------------- #
def fig4_behaviour_and_io():
    animal = np.asarray(TBL["animal"]).astype(str)
    theta = np.asarray(TBL["theta_from_go"], float)   # signed angle from go boundary
    go = np.asarray(TBL["goChoice"], float)
    stim = np.asarray(TBL["stimulus"], float)
    disp = np.asarray(TBL["dispersion"], float)
    cont = _clean_contrast(TBL["contrast"])
    post = np.asarray(TBL["post_s_marginal"], float)  # (trials, 91), s = 0..90 deg
    s_grid = np.linspace(0, 90, post.shape[1])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # A: psychometric P(go) vs signed orientation from go boundary, by unique value
    uniq = np.array(sorted({int(t) for t in theta[np.isfinite(theta)]}))

    def pgo(mask, min_n=8):
        xs, ys = [], []
        for tv in uniq:
            sel = mask & (theta == tv)
            if sel.sum() >= min_n:
                xs.append(tv); ys.append(np.nanmean(go[sel]))
        return np.array(xs), np.array(ys)

    for tag in TAGS:
        x, y = pgo(animal == tag)
        axA.plot(x, y, "-", color="0.75", lw=1)
    x, y = pgo(np.ones_like(go, bool))
    axA.plot(x, y, "-o", color="crimson", lw=2, ms=4, label="pooled")
    axA.axvline(0, color="k", ls=":", lw=1)
    axA.text(2, 0.06, "go pole", fontsize=7, color="0.3")
    print("  fig4 psychometric (theta_from_go: n, P(go)):")
    for tv in uniq:
        sel = theta == tv
        print(f"    {tv:+4d}°  n={sel.sum():4d}  P(go)={np.nanmean(go[sel]):.2f}")
    axA.set_xlabel("signed orientation from go pole (°)")
    axA.set_ylabel("P(go)"); axA.set_ylim(-0.02, 1.02)
    axA.set_title("P(go) peaks at go pole, falls to nogo pole (±90°)\n(grey = each mouse)")
    axA.legend(fontsize=8, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)

    # B: example IO posteriors at increasing stimulus, low-dispersion high-contrast
    axB.set_prop_cycle(color=plt.cm.plasma(np.linspace(0.05, 0.85, 5)))
    for s_target in [0, 30, 45, 60, 90]:
        cand = np.where((stim == s_target) & (disp == disp.min()) & (cont >= 0.99))[0]
        if len(cand) == 0:
            cand = np.where(stim == s_target)[0]
        i = cand[0]
        ln, = axB.plot(s_grid, post[i], lw=1.8, label=f"stim {s_target}°")
        axB.axvline(s_target, color=ln.get_color(), ls=":", lw=1, alpha=0.7)
    axB.set_xlabel("orientation s (°)")
    axB.set_ylabel("ideal-observer posterior  P(s | data)")
    axB.set_title("example IO posteriors (dotted = true stimulus)")
    axB.set_xticks([0, 45, 90]); axB.legend(fontsize=7, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fig 4 · Behaviour & ideal-observer targets",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig4_behaviour_and_io")


if __name__ == "__main__":
    fig1_tuning()
    fig2_grating_response()
    fig3_corridor_position()
    fig4_behaviour_and_io()
    print(f"\nDone. Figures in {OUTDIR}")
