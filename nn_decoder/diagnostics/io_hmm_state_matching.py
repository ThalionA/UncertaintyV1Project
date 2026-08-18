"""IO-HMM latent states — can they be MATCHED across animals?

K differs by mouse (m0=3, m1=2, m2=2, m3=3, m4=3, m5=3) and the state indices
are ARBITRARY per fit (state 0 in m1 has nothing to do with state 0 in m3), so
states can only be matched by FUNCTION, never by index. Every (mouse, state)
with >= MIN_N hard-assigned trials (m5's 4-trial state is dropped and marked) is
described by a feature vector, and the rows are then compared across mice:

  features  [γ] = gamma-weighted over ALL trials, [hard] = over the state's
            argmax-gamma trials (each column of fig 3 carries its tag).
            R of P(stim|state) at contrast 1 [γ] (sensory sharpness) and its
            slope across log10 contrast [γ] (does the state USE contrast);
            occupancy = mean gamma; model discriminability [γ] = mean model
            P(choice=Go|state) (PS_choice_G_tr_by_state) on Go stimuli
            (ori<45) minus NoGo stimuli (ori>45; 45 is randomly rewarded and
            excluded); the same from the animal's choices on the state's hard
            trials [hard]; false-alarm rate [hard] = P(choice=Go | ori>45), the
            Go-bias / lapse-like signature; pre-RZ lick rate and running speed
            z-scored WITHIN mouse then averaged over hard trials (relative
            engagement); the state's own model params kappa_amp and lam (raw
            unconstrained values as pickled — the link functions are not in
            the pkl; ec/ed are shared across states within a mouse, so they
            are tabulated but NOT clustered on); dwell = mean run length of
            consecutive hard-state trials (z-scored / clustered on log10,
            because one state has a 193-trial mean run; raw value printed);
            model self-transition probability = diag of softmax(trans_logits)
            row-wise (rows = from; verified against the empirical hard-state
            transitions: max |model - empirical| <= 0.14 on rows with n >= 15,
            0.22 only on m5's 4-trial state).
  fig 3     (mouse,state) x feature heatmap, colour = z-score of each feature
            across all usable rows, raw value printed in the cell, n in the label.
  fig 4     sharpness x discriminability embedding (and sharpness x contrast
            slope), marker = mouse, colour = within-mouse running-speed z (the
            'engagement' proxy; lick-rate z is its near-mirror, r = -0.87 across
            rows, so a mean of the two would cancel), size ~ occupancy, dashed
            medians so quadrants read as sharp/broad x discriminating/guessing.
  fig 5     average-linkage dendrograms of the z-scored rows: absolute, and
            within-mouse-centred (each feature minus its mouse's mean over
            usable states) — 'do the states occupy the same absolute types?' vs
            'is there a consistent within-animal axis?'. Cuts at 2 and 3
            clusters reported on stdout, with the median raw feature of every
            cluster and whether a cluster holds exactly one state per mouse.
  fig 6     per mouse, model transition matrix (softmax rows) beside the
            empirical hard-state transition matrix, numbers in cells.

Data: ALL pkl trials, no neural-export alignment — this is a pure model /
behaviour analysis and state runs / transitions need the unbroken time-ordered
sequence (pkl trial index = time order). Posterior fields are read via
io_hmm_data._get_posterior_field (Data.<field> is None in the HMM layout).
Support: 72 circular bins x 2.5 deg; width = doubled-angle resultant R (0
uniform, 1 delta), never linear SD, never folded.
Field semantics (verified 2026-08-18 against the data, all six mice):
PS_choice_G_tr_by_state[z] IS the model's per-state P(choice = Go) — it mixes
to PS_choice_G_tr exactly (max err 1e-7) and its per-orientation mean tracks
the empirical psychometric curve; PS_Go_G_tr_by_state is NOT a choice
probability but the perceptual belief mass on the Go category (corr 0.9999
with the circular mass of PS_stim on ori<45 ∪ ori>135, mean ~0.5). An earlier
draft built 'model discriminability' and a 'wrong-side rate' from PS_Go —
that made the discriminability axis a copy of sharpness (rho 0.90 with R) and
the 'wrong-side' rate a Go-bias measure anti-correlated (-0.44) with the model
miss rate; both were replaced (P(choice=Go|state) and false-alarm rate).

Usage: python diagnostics/io_hmm_state_matching.py [--pkl PATH]
Figures -> figures/io_hmm_by_state/  (png+svg via figsave.save_fig)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from figsave import save_fig                          # noqa: E402
import io_hmm_data                                    # noqa: E402
from io_hmm_data import _get_posterior_field as gp    # noqa: E402

REPO = _HERE.parent.parent
GRID = io_hmm_data.GRID_DEG_IO          # 72 x 2.5 deg, [0,180)
OUT = _HERE.parent / "figures" / "io_hmm_by_state"
MIN_N = 15                              # per-state hard-trial floor for any statistic
PNG_PX = 1550                           # save_fig cap; bbox_inches='tight' overshoots the nominal size by ~1-3 %, keep PNGs <= 1600
MOUSE_MARKERS = ["o", "s", "^", "D", "v", "P"]
MOUSE_COLS = plt.cm.Dark2(np.arange(6))

# (key used for z-scoring / clustering, heat-map column label, cell format, key whose RAW value is printed in the cell)
FEATS = [
    ("R_c1",        "R at c=1\n(sharpness)\n[γ]",           "{:.2f}",  "R_c1"),
    ("R_slope",     "R slope\n/decade c\n[γ]",              "{:+.2f}", "R_slope"),
    ("occ",         "occupancy\n(mean γ)\n",                "{:.2f}",  "occ"),
    ("disc_model",  "model P(Go|z)\nGo − NoGo\n[γ]",        "{:+.2f}", "disc_model"),
    ("disc_choice", "choice P(Go)\nGo − NoGo\n[hard]",      "{:+.2f}", "disc_choice"),
    ("fa",          "false alarms\nP(Go|ori>45)\n[hard]",   "{:.2f}",  "fa"),
    ("lick_z",      "lick rate\nz in-mouse\n[hard]",        "{:+.2f}", "lick_z"),
    ("vel_z",       "velocity\nz in-mouse\n[hard]",         "{:+.2f}", "vel_z"),
    ("kappa_amp",   "kappa_amp\n(raw param)\n",             "{:+.2f}", "kappa_amp"),
    ("lam",         "lam\n(raw param)\n",                   "{:+.2f}", "lam"),
    ("dwell_log",   "dwell (trials)\nz on log10\n[hard]",   "{:.1f}",  "dwell"),
    ("self_T",      "self-trans.\nP (model)\n",             "{:.2f}",  "self_T"),
]
KEYS = [k for k, _, _, _ in FEATS]


# ----------------------------------------------------------------------------- helpers
def resultant(p):
    """Doubled-angle resultant length per row of (n,72). 0 = uniform, 1 = delta."""
    a = np.deg2rad(2 * GRID)
    return np.hypot((p * np.cos(a)).sum(1), (p * np.sin(a)).sum(1))


def softmax_rows(x):
    x = np.asarray(x, float); x = x - x.max(1, keepdims=True); e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def run_lengths(hard, z):
    """Lengths of maximal runs of value z in the time-ordered hard-state sequence."""
    is_z = np.r_[0, (hard == z).astype(int), 0]
    d = np.diff(is_z)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def empirical_T(hard, K):
    """Row-normalised hard-state transition counts and the from-state counts."""
    C = np.zeros((K, K))
    for a, b in zip(hard[:-1], hard[1:]):
        C[a, b] += 1
    n_from = C.sum(1)
    return C / np.maximum(n_from, 1)[:, None], n_from.astype(int)


def _param(pbs, z, key):
    """Scalar per-state model parameter, NaN if the state/key is missing or None."""
    if z >= len(pbs) or pbs[z] is None:
        return np.nan
    v = pbs[z].get(key)
    if v is None:
        return np.nan
    v = np.asarray(v, float)
    return float(v.ravel()[0]) if v.size == 1 else np.nan


def _wmean(x, w, msk):
    ww = w[msk]
    return float(np.average(x[msk], weights=ww)) if ww.sum() > 1e-9 else np.nan


# ----------------------------------------------------------------------------- data
def load(pkl):
    """All pkl trials per mouse (no export alignment — see module docstring)."""
    mice = io_hmm_data.load_io_hmm_pkl(str(pkl), allow_partial=False)
    out = {}
    for m in sorted(mice):
        e = mice[m]; K = int(e["n_states"]); d = e["data"]
        by = np.asarray(gp(e, "PS_stim_G_tr_by_state"), float)      # (K,72,n) bins-first
        by = np.transpose(by, (0, 2, 1)); by = by / by.sum(2, keepdims=True)
        gam = np.asarray(gp(e, "gamma"), float)
        hard = np.asarray(gp(e, "hard_state")).ravel().astype(int)
        assert np.array_equal(hard, gam.argmax(1)), f"m{m}: hard_state != argmax gamma"
        f = lambda k: np.asarray(d[k], float).ravel()               # noqa: E731
        pch_s = np.asarray(gp(e, "PS_choice_G_tr_by_state"), float)            # (K,n) model P(choice=Go | z)
        pch = np.asarray(gp(e, "PS_choice_G_tr"), float).ravel()
        assert pch_s.shape == (K, len(hard)) and np.abs((gam * pch_s.T).sum(1) - pch).max() < 1e-5, \
            f"m{m}: PS_choice_G_tr_by_state does not mix to PS_choice_G_tr under gamma"
        out[m] = dict(K=K, by=by, gamma=gam, hard=hard, pch_s=pch_s,
                      ori=f("orientation"), ct=np.round(f("contrast"), 3), choice=f("choices"),
                      lick=f("preRZ_lickrate"), vel=f("preRZ_velocity"),
                      pbs=e["params_by_state"],
                      TL=np.asarray(e["params"]["trans_logits"], float))
    return out


def build_rows(D):
    rows = []
    for m, M in D.items():
        K, g, hard = M["K"], M["gamma"], M["hard"]
        lick_z = (M["lick"] - M["lick"].mean()) / M["lick"].std()
        vel_z = (M["vel"] - M["vel"].mean()) / M["vel"].std()
        T = softmax_rows(M["TL"])
        go, nogo = M["ori"] < 45, M["ori"] > 45
        for z in range(K):
            w = g[:, z]; hz = hard == z; n = int(hz.sum())
            R = resultant(M["by"][z])
            lv = [(np.log10(c), _wmean(R, w, M["ct"] == c)) for c in np.unique(M["ct"])
                  if w[M["ct"] == c].sum() >= 3]
            r = dict(
                mouse=m, state=z, n=n, usable=n >= MIN_N, label=f"m{m}s{z}",
                R_all=_wmean(R, w, np.ones_like(hz)),
                R_c1=_wmean(R, w, M["ct"] == 1.0),
                R_slope=np.polyfit(*zip(*lv), 1)[0] if len(lv) >= 2 else np.nan,
                occ=float(w.mean()),
                disc_model=_wmean(M["pch_s"][z], w, go) - _wmean(M["pch_s"][z], w, nogo),
                disc_choice=(M["choice"][hz & go].mean() - M["choice"][hz & nogo].mean()
                             if (hz & go).any() and (hz & nogo).any() else np.nan),
                fa=float(M["choice"][hz & nogo].mean()) if (hz & nogo).any() else np.nan,
                lick_z=float(lick_z[hz].mean()) if n else np.nan,
                vel_z=float(vel_z[hz].mean()) if n else np.nan,
                kappa_amp=_param(M["pbs"], z, "kappa_amp"), lam=_param(M["pbs"], z, "lam"),
                ec=_param(M["pbs"], z, "ec"), ed=_param(M["pbs"], z, "ed"),
                dwell=float(run_lengths(hard, z).mean()) if n else np.nan,
                self_T=float(T[z, z]),
            )
            r["dwell_log"] = np.log10(r["dwell"]) if np.isfinite(r["dwell"]) else np.nan
            rows.append(r)
    return rows


def feature_matrix(rows, centre_within_mouse=False):
    """(n_usable, n_feat) z-scored matrix over usable rows, plus those rows.
    With centre_within_mouse each raw feature first has its mouse's mean over
    usable states subtracted (relative role of the state inside its animal)."""
    use = [r for r in rows if r["usable"]]
    X = np.array([[r[k] for k in KEYS] for r in use], float)
    if centre_within_mouse:
        mice = np.array([r["mouse"] for r in use])
        for m in np.unique(mice):
            X[mice == m] -= X[mice == m].mean(0, keepdims=True)
    Z = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    if np.isnan(Z).any():
        print(f"  ! {int(np.isnan(Z).sum())} NaN feature cells among usable rows -> filled with 0 (column mean)")
        Z = np.where(np.isnan(Z), 0.0, Z)
    return Z, use


# ----------------------------------------------------------------------------- figures
def fig_heatmap(rows):
    Z, use = feature_matrix(rows)
    ordered = sorted(rows, key=lambda r: (r["mouse"], r["state"]))
    nrow, ncol = len(ordered), len(KEYS)
    zmap = {r["label"]: Z[i] for i, r in enumerate(use)}
    Zfull = np.full((nrow, ncol), np.nan)
    for i, r in enumerate(ordered):
        if r["usable"]:
            Zfull[i] = zmap[r["label"]]
    fig, ax = plt.subplots(figsize=(13, 8.0))
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("0.85")
    vmax = np.nanmax(np.abs(Zfull))
    im = ax.imshow(np.ma.masked_invalid(Zfull), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i, r in enumerate(ordered):
        if not r["usable"]:
            ax.text(ncol / 2 - 0.5, i, f"n = {r['n']} < {MIN_N}: excluded from every statistic",
                    ha="center", va="center", fontsize=8, color="0.3", style="italic")
            continue
        for j, (_, _, fmt, dk) in enumerate(FEATS):
            v = r[dk]
            txt = "nan" if not np.isfinite(v) else fmt.format(v)
            zc = Zfull[i, j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if np.isfinite(zc) and abs(zc) > 1.6 else "black")
    # mouse group separators + labels
    ylab = []
    for i, r in enumerate(ordered):
        ylab.append(f"m{r['mouse']} s{r['state']}  (n={r['n']})")
        if i + 1 < nrow and ordered[i + 1]["mouse"] != r["mouse"]:
            ax.axhline(i + 0.5, color="k", lw=1.2)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(ylab, fontsize=8.5)
    for t, r in zip(ax.get_yticklabels(), ordered):
        t.set_color(MOUSE_COLS[r["mouse"]]); t.set_fontweight("bold" if r["usable"] else "normal")
    ax.set_xticks(range(ncol)); ax.set_xticklabels([lab for _, lab, _, _ in FEATS], fontsize=7.6)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0)
    for j in range(1, ncol):
        ax.axvline(j - 0.5, color="w", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("z-score of the feature across all usable (mouse,state) rows", fontsize=8.5)
    ax.set_title("IO-HMM (mouse,state) rows x function features — colour = z across rows, text = raw value; "
                 f"n = hard-assigned trials, states with n < {MIN_N} greyed out\n"
                 "[γ] = γ-weighted over all trials, [hard] = over the state's argmax-γ trials; "
                 "state indices are arbitrary per fit: matching across mice is by these features, not by index",
                 fontsize=9.5, pad=62)
    save_fig(fig, OUT, "fig3_state_feature_heatmap", max_px=PNG_PX)


def _place_labels(ax, pts, fontsize=8):
    """Greedy 4-direction label placement (NE/NW/SE/SW around each marker) that
    minimises overlap with markers and already-placed labels. pts = list of
    (x, y, marker_size_pts2, text, colour)."""
    fig = ax.figure; fig.canvas.draw()          # transforms need a layout pass
    k = fig.dpi / 72.0
    tr = ax.transData; ab = ax.get_window_extent()
    px = np.array([tr.transform((x, y)) for x, y, _, _, _ in pts])
    rad = np.array([np.sqrt(s_) / 2 * k for _, _, s_, _, _ in pts])
    mk_boxes = [(cx - r, cy - r, cx + r, cy + r) for (cx, cy), r in zip(px, rad)]
    placed = []

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    for i, (x, y, s_, text, col) in enumerate(pts):
        r = rad[i] + 3 * k
        lines = text.split("\n")
        w, h = 0.6 * fontsize * k * max(len(t) for t in lines), 1.15 * fontsize * k * len(lines)
        cx, cy = px[i]
        best = None
        for dx, dy, ha, va in [(1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
                               (1, -1, "left", "top"), (-1, -1, "right", "top")]:
            x0 = cx + r if dx > 0 else cx - r - w
            y0 = cy + r if dy > 0 else cy - r - h
            box = (x0 - 2 * k, y0 - 2 * k, x0 + w + 2 * k, y0 + h + 2 * k)     # padded
            cost = sum(overlap(box, b) for j, b in enumerate(mk_boxes) if j != i)
            cost += sum(overlap(box, b) for b in placed)
            out = (max(0, ab.x0 - box[0]) + max(0, box[2] - ab.x1) +
                   max(0, ab.y0 - box[1]) + max(0, box[3] - ab.y1))
            cost += 50 * out * h
            if best is None or cost < best[0]:
                best = (cost, box, dx, dy, ha, va)
        _, box, dx, dy, ha, va = best
        placed.append(box)
        ax.annotate(text, (x, y), fontsize=fontsize, xytext=(dx * r / k, dy * r / k), textcoords="offset points",
                    ha=ha, va=va, color=col, fontweight="bold", zorder=4, linespacing=0.95)


def _scatter_panel(ax, rows, xk, yk, xl, yl, cnorm, cmap):
    """One point per usable (mouse,state); states with n < MIN_N are omitted (see title)."""
    use = [r for r in rows if r["usable"] and np.isfinite(r[xk]) and np.isfinite(r[yk])]
    xs = np.array([r[xk] for r in use]); ys = np.array([r[yk] for r in use])
    pts = []
    for r in use:
        size = 60 + 900 * r["occ"]
        ax.scatter(r[xk], r[yk], marker=MOUSE_MARKERS[r["mouse"]], s=size,
                   c=[cmap(cnorm(r["vel_z"]))], edgecolor="k", lw=0.7, zorder=3)
        pts.append((r[xk], r[yk], size, f"{r['label']}\nn={r['n']}", MOUSE_COLS[r["mouse"]]))
    ax.margins(0.12)
    ax.axvline(np.nanmedian(xs), ls="--", color="0.5", lw=0.9)
    ax.axhline(np.nanmedian(ys), ls="--", color="0.5", lw=0.9)
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    _place_labels(ax, pts)


def fig_scatter(rows):
    use = [r for r in rows if r["usable"]]
    excl = [r for r in rows if not r["usable"]]
    vz = np.array([r["vel_z"] for r in use]); lz = np.array([r["lick_z"] for r in use])
    lim = max(0.3, np.abs(vz).max())
    cnorm = plt.Normalize(-lim, lim); cmap = plt.cm.coolwarm
    r_lv = np.corrcoef(lz, vz)[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9))
    xlab = "sensory sharpness: γ-weighted mean R of P(stim|state) at contrast 1\n(broad → sharp)"
    _scatter_panel(axes[0], rows, "R_c1", "disc_model", xlab,
                   "behavioural discriminability: model P(choice=Go|state), γ-weighted,\n"
                   "Go stimuli (ori<45) − NoGo stimuli (ori>45)   (guessing → discriminating)",
                   cnorm, cmap)
    axes[0].set_title("dashed = medians over usable rows, so quadrants read\nbroad/sharp × guessing/discriminating; "
                      "size ∝ occupancy (mean γ)", fontsize=9)
    _scatter_panel(axes[1], rows, "R_c1", "R_slope", xlab,
                   "contrast use: slope of γ-weighted state R vs log10 contrast (ΔR per decade)", cnorm, cmap)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("does the state USE contrast?\n(0 = same width at every contrast)", fontsize=9)
    # legends: mouse markers + engagement colourbar
    for m in range(6):
        axes[1].scatter([], [], marker=MOUSE_MARKERS[m], s=50, facecolor="w", edgecolor="k", label=f"m{m}")
    axes[1].legend(title="marker = mouse", fontsize=7.5, title_fontsize=8, loc="lower right", frameon=True)
    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("engagement proxy: pre-RZ running speed, z-scored WITHIN mouse, mean over hard trials\n"
                 f"(pre-RZ lick-rate z is its near-mirror across rows: r = {r_lv:+.2f})", fontsize=8)
    omitted = ", ".join(f"{r['label']} (n={r['n']})" for r in excl) or "none"
    fig.suptitle("IO-HMM states embedded by function — one point per (mouse,state) with n ≥ "
                 f"{MIN_N} hard trials (omitted: {omitted}); labels m<mouse>s<state>, n = hard trials "
                 "(indices arbitrary per fit)", fontsize=10)
    save_fig(fig, OUT, "fig4_state_embedding_sharpness_vs_discriminability", max_px=PNG_PX)


def _dendro_panel(ax, Z, use, title):
    L = linkage(Z, method="average", metric="euclidean")
    coph = cophenet(L, pdist(Z))[0]
    heights = L[:, 2]
    # colour threshold that yields 3 link colours; cut lines for k=2 and k=3
    cut2 = 0.5 * (heights[-1] + heights[-2]); cut3 = 0.5 * (heights[-2] + heights[-3])
    labels = [f"{r['label']} (n={r['n']})" for r in use]
    dn = dendrogram(L, ax=ax, labels=labels, leaf_rotation=90, leaf_font_size=8.5,
                    color_threshold=cut3, above_threshold_color="0.4")
    for t in ax.get_xticklabels():
        m = int(t.get_text()[1]); t.set_color(MOUSE_COLS[m]); t.set_fontweight("bold")
    ax.axhline(cut2, ls="--", color="0.3", lw=0.8)
    ax.text(0.995, cut2, "cut → 2 clusters ", fontsize=7, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(cut3, ls=":", color="0.3", lw=0.8)
    ax.text(0.995, cut3, "cut → 3 clusters ", fontsize=7, va="top", ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylabel("average-linkage distance (Euclidean, z-scored features)", fontsize=8.5)
    ax.set_title(f"{title}\ncophenetic corr = {coph:.2f}", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    return L


def report_clusters(L, use, tag):
    out = {}
    for k in (2, 3):
        cl = fcluster(L, k, criterion="maxclust")
        groups = {c: [r for r, c_ in zip(use, cl) if c_ == c] for c in np.unique(cl)}
        print(f"  [{tag}] cut into {k}:")
        for c, grp in groups.items():
            print(f"    cluster {c}: " + ", ".join(f"{r['label']}(n={r['n']})" for r in grp))
        # per-mouse split pattern
        pat = []
        for m in sorted({r["mouse"] for r in use}):
            cs = [int(c_) for r, c_ in zip(use, cl) if r["mouse"] == m]
            pat.append(f"m{m}:{'/'.join(map(str, cs))}")
        print("    per-mouse assignment (state order): " + "  ".join(pat))
        out[k] = dict(zip([r["label"] for r in use], cl.tolist()))
    return out


def fig_dendro(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    Za, use = feature_matrix(rows, centre_within_mouse=False)
    Zw, _ = feature_matrix(rows, centre_within_mouse=True)
    print("Hierarchical clustering of usable (mouse,state) rows, 'average' linkage on z-scored features:")
    La = _dendro_panel(axes[0], Za, use, "ABSOLUTE: features z-scored across all rows\n(same absolute functional type?)")
    ca = report_clusters(La, use, "absolute")
    Lw = _dendro_panel(axes[1], Zw, use, "WITHIN-MOUSE-CENTRED: each feature minus its mouse's mean, then z-scored\n(consistent within-animal axis?)")
    cw = report_clusters(Lw, use, "within-mouse-centred")
    fig.suptitle(f"Do (mouse,state) rows cluster across mice? leaves = m<mouse>s<state> (n = hard trials); "
                 f"{len(KEYS)} features (ec/ed excluded: shared within mouse)", fontsize=10)
    save_fig(fig, OUT, "fig5_state_dendrogram", max_px=PNG_PX)
    return ca, cw


def fig_transitions(D):
    mice = sorted(D)
    fig, axes = plt.subplots(2, len(mice), figsize=(14, 4.9))
    print("Transition structure — model softmax(trans_logits) rows vs empirical hard-state transitions:")
    for c, m in enumerate(mice):
        K = D[m]["K"]; T = softmax_rows(D[m]["TL"]); E, n_from = empirical_T(D[m]["hard"], K)
        for rrow, (M, name) in enumerate([(T, "model  P(z_t+1 | z_t)"), (E, "empirical (hard states)")]):
            ax = axes[rrow, c]
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
            for i in range(K):
                thin = rrow == 1 and n_from[i] < MIN_N        # too few exits to trust the row
                for j in range(K):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                            color="0.6" if thin else ("white" if M[i, j] > 0.6 else "black"),
                            style="italic" if thin else "normal")
            ax.set_xticks(range(K)); ax.set_xticklabels([f"s{z}" for z in range(K)], fontsize=8)
            ax.set_yticks(range(K))
            ax.set_yticklabels([f"s{z}" if rrow == 0 else f"s{z} (n={n_from[z]})" for z in range(K)], fontsize=8)
            ax.tick_params(length=0)
            if rrow == 0:
                ax.set_title(f"m{m}  K={K}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{name}\nfrom state", fontsize=8.5)
            if rrow == 1:
                ax.set_xlabel("to state", fontsize=8.5)
        print(f"  m{m}: model diag {np.round(np.diag(T), 2).tolist()}  empirical diag {np.round(np.diag(E), 2).tolist()}"
              f"  max|model−emp| {np.abs(T - E).max():.3f}  from-counts {n_from.tolist()}")
    fig.suptitle("Transition matrices per mouse — top: model softmax(trans_logits) row-wise (rows = from); "
                 f"bottom: empirical hard-state transitions (n = trials leaving each state; grey italics = n < {MIN_N})", fontsize=10)
    save_fig(fig, OUT, "fig6_transition_matrices", max_px=PNG_PX)


# ----------------------------------------------------------------------------- stdout
def print_table(rows):
    print("Feature table (n = hard-assigned trials; * = excluded, n < %d):" % MIN_N)
    hdr = f"{'row':7s}{'n':>5s}{'R_all':>7s}{'R_c1':>6s}{'Rslope':>7s}{'occ':>6s}{'discM':>7s}{'discC':>7s}{'FA':>6s}" \
          f"{'lickz':>7s}{'velz':>7s}{'k_amp':>7s}{'lam':>7s}{'ec':>7s}{'ed':>7s}{'dwell':>7s}{'selfT':>6s}"
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["mouse"], r["state"])):
        f = lambda k, w=6, d=2: f"{r[k]:>{w}.{d}f}" if np.isfinite(r[k]) else f"{'nan':>{w}s}"  # noqa: E731
        print(f"{r['label'] + ('*' if not r['usable'] else ''):7s}{r['n']:>5d}{f('R_all',7)}{f('R_c1')}{f('R_slope',7)}"
              f"{f('occ')}{f('disc_model',7)}{f('disc_choice',7)}{f('fa')}{f('lick_z',7)}{f('vel_z',7)}"
              f"{f('kappa_amp',7)}{f('lam',7)}{f('ec',7)}{f('ed',7)}{f('dwell',7,1)}{f('self_T')}")
    print("  R_all = gamma-weighted mean R over all contrasts (check vs the established per-state values); "
          "discM = model P(choice=Go|state) Go−NoGo (γ, PS_choice_G_tr_by_state); discC = same from choices on hard "
          "trials; FA = P(Go | ori>45) on hard trials; ec/ed identical across states within a mouse -> not clustered on.")
    use = [r for r in rows if r["usable"]]
    from scipy.stats import spearmanr
    a = np.array([[r["R_c1"], r["disc_model"], r["disc_choice"], r["fa"]] for r in use])
    print(f"  across usable rows: rho(discM, discC) = {spearmanr(a[:, 1], a[:, 2])[0]:+.2f}; "
          f"rho(R_c1, discM) = {spearmanr(a[:, 0], a[:, 1])[0]:+.2f}; rho(R_c1, FA) = {spearmanr(a[:, 0], a[:, 3])[0]:+.2f} "
          "(rows are 2-3 per mouse, not independent — descriptive only)")


def _desc(r):
    return (f"s{r['state']} (R={r['R_c1']:.2f}, disc={r['disc_model']:+.2f}, FA={r['fa']:.2f}, "
            f"occ={r['occ']:.2f}, dwell={r['dwell']:.0f})")


def verdicts(rows):
    """One line per mouse. Sharp/broad = max/min R at contrast 1 among usable states; a
    third state within TIE of the sharp or broad one is reported as a tie and described by
    what else separates it (false-alarm rate, occupancy, dwell). 'disc' = model
    P(choice=Go|state) Go−NoGo (γ-weighted)."""
    TIE, SPLIT = 0.05, 0.05
    print(f"Per-mouse verdict (sharp/broad by R at contrast 1 among usable states; tie = |ΔR| < {TIE}; "
          f"no clean split = total spread < {SPLIT}; disc = model P(choice=Go|state) Go−NoGo, FA = P(Go|ori>45) hard):")
    for m in sorted({r["mouse"] for r in rows}):
        use = [r for r in rows if r["mouse"] == m and r["usable"]]
        excl = [r for r in rows if r["mouse"] == m and not r["usable"]]
        note = f" [s{','.join(str(r['state']) for r in excl)} excluded, n<{MIN_N}]" if excl else ""
        if len(use) < 2:
            print(f"  m{m}: only one usable state{note}"); continue
        Rs = np.array([r["R_c1"] for r in use]); spread = Rs.max() - Rs.min()
        if spread < SPLIT:
            detail = ", ".join(_desc(r) for r in use)
            print(f"  m{m}: NO CLEAN SPLIT by sharpness — R@c1 spread {spread:.2f}: {detail}{note}")
            continue
        s, b = use[int(Rs.argmax())], use[int(Rs.argmin())]
        third = [r for r in use if r not in (s, b)]
        msg = f"  m{m}: sharp = {_desc(s)}, broad = {_desc(b)}"
        if third:
            t = third[0]
            if abs(t["R_c1"] - s["R_c1"]) < TIE:
                msg += f"; third {_desc(t)} TIES the sharp state in R"
            elif abs(t["R_c1"] - b["R_c1"]) < TIE:
                msg += f"; third {_desc(t)} TIES the broad state in R"
            else:
                msg += f"; third {_desc(t)} is intermediate in R"
        msg += f"; sharp state {'is' if s['disc_model'] > b['disc_model'] else 'is NOT'} the more discriminating{note}"
        print(msg)
    hi = [r for r in rows if r["usable"] and r["fa"] > 0.6]
    print("  States with false-alarm rate P(Go | ori>45) > 0.60 (Go-biased / lapse-like): " +
          (", ".join(f"{r['label']} (FA={r['fa']:.2f}, disc={r['disc_model']:+.2f}, dwell={r['dwell']:.0f}, "
                     f"vel z={r['vel_z']:+.2f})" for r in hi) or "none"))


def cluster_types(rows, cw, k=3):
    """Data-led matching: describe each within-mouse-centred cluster by the mean raw
    features of its members and list which state of each mouse falls in it."""
    use = [r for r in rows if r["usable"]]
    mice = sorted({r["mouse"] for r in use})
    print(f"Cluster-based state types (within-mouse-centred features, cut into {k}) — MEDIAN raw feature per cluster:")
    cols = ["R_c1", "R_slope", "occ", "disc_model", "disc_choice", "fa", "lick_z", "vel_z", "dwell", "self_T"]
    print(f"  {'cluster':8s}{'#':>3s}" + "".join(f"{c:>12s}" for c in cols) + "   members")
    for c in sorted(set(cw[k].values())):
        mem = [r for r in use if cw[k][r["label"]] == c]
        vals = [np.nanmedian([r[col] for r in mem]) for col in cols]
        print(f"  {c:<8d}{len(mem):>3d}" + "".join(f"{v:>12.2f}" for v in vals) + "   " + ", ".join(r["label"] for r in mem))
    print("  per-mouse membership (state -> cluster): " + "  ".join(
        f"m{m}:" + ",".join(f"s{r['state']}→{cw[k][r['label']]}" for r in use if r["mouse"] == m) for m in mice))
    for c in sorted(set(cw[k].values())):
        counts = {m: sum(1 for r in use if r["mouse"] == m and cw[k][r["label"]] == c) for m in mice}
        one_each = all(v == 1 for v in counts.values())
        print(f"  cluster {c} has exactly one state per mouse: {'YES' if one_each else 'no'} "
              f"(per-mouse counts {[counts[m] for m in mice]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(REPO / "data" / "fitted_data_and_posteriors_hmm.pkl"))
    a = ap.parse_args()
    D = load(a.pkl); OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows(D)
    print(f"Loaded {len(D)} mice, all pkl trials: " + ", ".join(f"m{m}: n={len(D[m]['hard'])} K={D[m]['K']}" for m in D))
    print_table(rows)
    use = [r for r in rows if r["usable"]]
    lz = np.array([r["lick_z"] for r in use]); vz = np.array([r["vel_z"] for r in use])
    print(f"  engagement components across usable rows: corr(lick z, velocity z) = {np.corrcoef(lz, vz)[0, 1]:+.2f}")
    fig_heatmap(rows)
    fig_scatter(rows)
    ca, cw = fig_dendro(rows)
    fig_transitions(D)
    verdicts(rows)
    cluster_types(rows, cw, k=3)
    cluster_types(rows, cw, k=2)


if __name__ == "__main__":
    main()
