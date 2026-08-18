"""IO-HMM latent states vs stimulus, time and behaviour — what drives state occupancy?

Companion to ``io_hmm_posteriors_by_state.py`` (which established that the states
differ in CONCENTRATION of P(stim | state), not location). Here the question is
what the states track: if they were purely internal (attention drifting) their
occupancy would be flat across stimulus conditions and structured only in time;
if the model uses the latent state to soak up contrast/dispersion effects,
occupancy will follow the stimulus grid.

Figures (all -> figures/io_hmm_by_state/, png+svg via figsave.save_fig):
  (a) fig_a_state_by_condition      gamma-weighted P(state | contrast x dispersion) heatmaps,
                                    rows = state, cols = mouse; Cramer's V of hard state vs
                                    condition in the column titles.
  (b) fig_b_state_by_orientation    gamma-weighted occupancy per orientation, one panel per
                                    mouse, one line per state (SEM bars, dashed = marginal).
  (c) fig_c_state_by_time           hard-state strip (alpha = gamma_max) + rolling (w=25)
                                    hard-state occupancy per mouse; mean dwell in the legend.
  (d) fig_d_state_by_behaviour      preRZ lick rate and velocity by hard state (violins, n
                                    labelled), P(correct) by state with Wilson CI (filled = all
                                    trials, hollow = contrast 1 only, to de-confound the stimulus).
  (e) fig_e_what_predicts_state     multinomial logistic regression of hard state on
                                    [contrast, dispersion, |ori-45|, previous hard state, lick
                                    rate, velocity]: McFadden pseudo-R2 in-sample (bars) and
                                    blocked 5-fold CV (dots), full model and each predictor alone.

Data: ALL pkl trials per mouse (no alignment to the neural export — none of these figures
need it), trial index = time order within the session. Posterior arrays are read via
``io_hmm_data._get_posterior_field`` (Data.<name> is None in the HMM layout). States with
hard n < 15 are greyed/suppressed everywhere (m5 state 1, n = 4). State indices are
arbitrary per mouse; colours are per index, not matched across mice.

Caveats printed with the numbers:
  * every p-value here treats trials as independent, but states are sticky (few
    transitions per session), so chi-square / Kruskal-Wallis p's are anti-conservative —
    read the effect sizes (Cramer's V, R2), not the p's;
  * the pkl ``correct`` field is the Go-STIMULUS category (1{ori<45}, coin-flip at 45),
    NOT accuracy — asserted in ``load()``; P(correct) here is derived from the task rule
    (choice == 1{ori<45}) on ori != 45 trials;
  * the per-state Params carry velocity-link terms (alpha_v/beta_v/gamma_v, different per
    state in every mouse; the *_nl lick terms are all 0), so velocity is plausibly an
    emission the state posterior was computed from — its state dependence in (d)/(e) is
    partly by construction, and 'behaviour predicts state' must not be read as causal.

Usage: python diagnostics/io_hmm_state_stimulus_dependence.py [--pkl PATH]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from figsave import save_fig            # noqa: E402
import io_hmm_data                      # noqa: E402

REPO = _HERE.parent.parent
OUT = _HERE.parent / "figures" / "io_hmm_by_state"
MIN_N = 15                              # states with fewer hard trials are suppressed
ROLL_W = 25                             # rolling-occupancy window (trials)
MIN_CELL_N = 10                         # fig a: 'rng' (occupancy range across condition cells) uses only cells with >= this many trials
CONTRASTS = [0.01, 0.25, 0.5, 1.0]
DISPERSIONS = [5.0, 30.0, 45.0, 90.0]
ORIS = [0.0, 15.0, 30.0, 40.0, 45.0, 50.0, 60.0, 75.0, 90.0]
STATE_COLS = plt.cm.tab10(np.arange(10))   # state z -> tab10[z], as in io_hmm_posteriors_by_state
GREY = "0.6"


# ----------------------------------------------------------------------------- data
def load(pkl):
    """All pkl trials per mouse, time order = pkl index. No export alignment."""
    mice = io_hmm_data.load_io_hmm_pkl(str(pkl), allow_partial=False)
    out = {}
    for m in sorted(mice):
        e = mice[m]; K = int(e["n_states"]); d = e["data"]
        gam = np.asarray(io_hmm_data._get_posterior_field(e, "gamma"), float)     # (n, K)
        hard = np.asarray(io_hmm_data._get_posterior_field(e, "hard_state")).ravel().astype(int)
        assert gam.shape[1] == K and (hard == gam.argmax(1)).all()
        f = lambda k: np.asarray(d[k]).ravel().astype(float)
        n_hard = np.bincount(hard, minlength=K)
        ori, choice = f("orientation"), f("choices")
        # NB: the pkl 'correct' field is the Go-STIMULUS category (1 for ori<45, coin-flip at 45,
        # 0 for ori>45) regardless of choice — measured 2026-08-18: it equals 1{ori<45} on every
        # off-boundary trial and averages 0.45-0.49 per mouse. It is NOT accuracy. Accuracy is
        # derived here from the task rule (Go iff ori<45); ori==45 trials have no correct answer.
        pkl_correct = f("correct")
        offb = ori != 45
        assert np.array_equal(pkl_correct[offb], (ori[offb] < 45).astype(float)), \
            "pkl 'correct' no longer equals 1{ori<45} — re-check its meaning before using it"
        acc = ((choice == 1) == (ori < 45)).astype(float)
        out[m] = dict(K=K, gamma=gam, hard=hard, n=len(hard), n_hard=n_hard,
                      ok=n_hard >= MIN_N,
                      ori=ori, ct=np.round(f("contrast"), 3), disp=f("dispersion"),
                      choice=choice, acc=acc, offb=offb,
                      lick=f("preRZ_lickrate"), vel=f("preRZ_velocity"))
    return out


# ----------------------------------------------------------------------------- helpers
def cramers_v(a, b):
    """Cramer's V, chi2 p, and dof for two integer-coded label vectors."""
    ua, ub = np.unique(a), np.unique(b)
    tab = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            tab[i, j] = np.sum((a == x) & (b == y))
    if min(tab.shape) < 2:
        return np.nan, np.nan, 0
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
    v = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    return v, p, dof


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h


def dwell_lengths(hard, K):
    """Run lengths of the hard-state sequence, grouped by state."""
    runs = {z: [] for z in range(K)}
    start = 0
    for t in range(1, len(hard) + 1):
        if t == len(hard) or hard[t] != hard[start]:
            runs[hard[start]].append(t - start); start = t
    return runs


def state_label(D, z):
    return f"state {z} (n={D['n_hard'][z]})" + ("" if D["ok"][z] else " suppressed")


# ----------------------------------------------------------------------------- (a)
def fig_a(Dall, mice):
    Kmax = max(Dall[m]["K"] for m in mice)
    fig, axes = plt.subplots(Kmax, len(mice), figsize=(14, 2.4 * Kmax + 0.9), squeeze=False)
    cmap = plt.cm.Blues.copy(); cmap.set_bad("0.92")
    stats_out = {}
    for j, m in enumerate(mice):
        D = Dall[m]; K = D["K"]
        cell_id = np.full(D["n"], -1)
        cells = []
        for ci, c in enumerate(CONTRASTS):
            for di, dd in enumerate(DISPERSIONS):
                msk = (D["ct"] == c) & (D["disp"] == dd)
                if msk.sum():
                    cell_id[msk] = len(cells); cells.append((ci, di, msk))
        keep = D["ok"][D["hard"]]                     # drop suppressed-state trials from the test
        V, p, dof = cramers_v(D["hard"][keep], cell_id[keep])
        stats_out[m] = (V, p, dof, len(cells))
        for z in range(Kmax):
            ax = axes[z, j]
            if z >= K:
                ax.set_axis_off(); continue
            grid = np.full((len(CONTRASTS), len(DISPERSIONS)), np.nan)
            for ci, di, msk in cells:
                grid[ci, di] = D["gamma"][msk, z].mean()
            occ_vals = np.array([grid[ci, di] for ci, di, msk in cells if msk.sum() >= MIN_CELL_N])   # populated cells only
            im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto",
                           origin="lower")
            for ci, di, msk in cells:
                v = grid[ci, di]
                dark = v > 0.6 and D["ok"][z]
                ax.text(di, ci + 0.08, f"{v:.2f}" if D["ok"][z] else "–", ha="center", va="center",
                        fontsize=6.5, color="w" if dark else "k")
                ax.text(di, ci - 0.3, f"n={msk.sum()}", ha="center", va="center", fontsize=5.4,
                        color="0.85" if dark else "0.35")
            if not D["ok"][z]:
                ax.add_patch(plt.Rectangle((-0.5, -0.5), 4, 4, fc="0.75", alpha=0.6, hatch="//", ec="none"))
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels([f"{d:g}" for d in DISPERSIONS], fontsize=6.5)
            ax.set_yticklabels([f"{c:g}" for c in CONTRASTS], fontsize=6.5)
            ax.tick_params(length=2)
            marg = D["gamma"][:, z].mean()
            ttl = (f"s{z} n={D['n_hard'][z]}  marg {marg:.2f}  rng {occ_vals.max() - occ_vals.min():.2f}"
                   if D["ok"][z] else f"s{z} n={D['n_hard'][z]} (suppressed)")
            ax.set_title(ttl, fontsize=7, color=STATE_COLS[z] if D["ok"][z] else GREY)
            if z == 0:   # mouse header (black, bold) above the coloured state line
                ax.text(0.5, 1.2, f"m{m}  K={K}\nCramer's V={V:.2f} (p={p:.0e})", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            if z == K - 1: ax.set_xlabel("dispersion (deg)", fontsize=7)
            if j == 0: ax.set_ylabel("contrast", fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.84, bottom=0.06, hspace=0.5, wspace=0.35)
    cax = fig.add_axes([0.925, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("gamma-weighted P(state | condition)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle("(a) State occupancy over the contrast x dispersion grid (8 of 16 cells occupied; grey = unused). "
                 "Numbers = mean gamma in cell; small n = trials in cell\n(very unbalanced: contrast 1 / dispersion 5 "
                 f"holds ~60% of trials; cells with n<{MIN_CELL_N} are noise). "
                 f"'rng' = max-min occupancy across cells with n>={MIN_CELL_N} (0 = flat = stimulus-independent);\n'marg' = marginal occupancy. "
                 "V = Cramer's V of hard state vs cell (states n>=15; p treats sticky trials as independent).",
                 fontsize=8.5)
    save_fig(fig, OUT, "fig_a_state_by_condition", layout=None)
    return stats_out


# ----------------------------------------------------------------------------- (b)
def fig_b(Dall, mice):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True, sharey=True)
    out = {}
    for ax, m in zip(axes.ravel(), mice):
        D = Dall[m]; K = D["K"]
        keep = D["ok"][D["hard"]]
        V, p, _ = cramers_v(D["hard"][keep], D["ori"][keep].astype(int))
        out[m] = (V, p)
        for z in range(K):
            if not D["ok"][z]:
                ax.plot([], [], color=GREY, label=state_label(D, z)); continue
            mu, se = [], []
            for o in ORIS:
                g = D["gamma"][D["ori"] == o, z]
                mu.append(g.mean() if g.size else np.nan)
                se.append(g.std(ddof=1) / np.sqrt(g.size) if g.size > 1 else np.nan)
            ax.errorbar(ORIS, mu, yerr=se, color=STATE_COLS[z], marker="o", ms=4, lw=1.4,
                        capsize=2, label=state_label(D, z))
            ax.axhline(D["gamma"][:, z].mean(), color=STATE_COLS[z], ls="--", lw=0.8, alpha=0.7)
        ax.axvline(45, color="0.5", ls=":", lw=0.8)
        ax.set_xticks(ORIS); ax.tick_params(labelsize=7)
        n_o = [int((D["ori"] == o).sum()) for o in ORIS]
        ax.set_title(f"m{m}  K={K}   Cramer's V(hard state, ori)={V:.2f} (p={p:.1e})\n"
                     f"n per ori: {min(n_o)}–{max(n_o)}", fontsize=8)
        ax.legend(fontsize=6.5, frameon=True, framealpha=0.85, edgecolor="none", loc="best")
        ax.set_ylim(0, 1)
    for ax in axes[-1]: ax.set_xlabel("orientation (deg); 45 = Go/NoGo boundary", fontsize=8)
    for ax in axes[:, 0]: ax.set_ylabel("gamma-weighted occupancy\n(mean gamma over trials at that ori)", fontsize=8)
    fig.suptitle("(b) State occupancy vs orientation (all trials, all contrasts).\n"
                 "Bars = SEM of gamma; dashed = marginal occupancy of that state.", fontsize=9)
    save_fig(fig, OUT, "fig_b_state_by_orientation", max_px=1560)
    return out


# ----------------------------------------------------------------------------- (c)
def fig_c(Dall, mice):
    fig = plt.figure(figsize=(13, 12.5))
    gs = fig.add_gridspec(3 * len(mice) - 1, 1, hspace=0.08,
                          height_ratios=([0.32, 1.0, 0.22] * len(mice))[:-1])   # strip, rolling, spacer
    out = {}
    for i, m in enumerate(mice):
        D = Dall[m]; K = D["K"]; n = D["n"]; hard = D["hard"]
        gmax = D["gamma"].max(1)
        rgba = STATE_COLS[hard].copy(); rgba[:, 3] = gmax
        ax_s = fig.add_subplot(gs[3 * i]); ax_r = fig.add_subplot(gs[3 * i + 1], sharex=ax_s)
        ax_s.imshow(rgba[None], aspect="auto", extent=(-0.5, n - 0.5, 0, 1), interpolation="nearest")
        ax_s.set_yticks([]); ax_s.tick_params(labelbottom=False, length=0)
        ax_s.set_ylabel(f"m{m}\nK={K}", fontsize=8, rotation=0, ha="right", va="center")
        for sp in ax_s.spines.values(): sp.set_visible(False)
        runs = dwell_lengths(hard, K)
        n_trans = int((hard[1:] != hard[:-1]).sum())
        p_stay = 1 - n_trans / (n - 1)
        p_indep = float((np.bincount(hard, minlength=K) / n) ** 2 @ np.ones(K))
        out[m] = dict(n_trans=n_trans, p_stay=p_stay, p_indep=p_indep,
                      dwell={z: (np.mean(runs[z]) if runs[z] else np.nan) for z in range(K)})
        kern = np.ones(ROLL_W) / ROLL_W
        x = np.arange(n)[ROLL_W // 2: n - (ROLL_W - 1 - ROLL_W // 2)]
        for z in range(K):
            r = np.convolve((hard == z).astype(float), kern, mode="valid")
            lab = f"state {z}: n={D['n_hard'][z]}, mean dwell {np.mean(runs[z]) if runs[z] else np.nan:.1f} tr ({len(runs[z])} runs)"
            ax_r.plot(x, r, color=STATE_COLS[z] if D["ok"][z] else GREY, lw=1.3, label=lab)
        ax_r.set_ylim(0, 1); ax_r.set_xlim(-0.5, n - 0.5)
        ax_r.set_ylabel(f"rolling P(hard state)\nw={ROLL_W}", fontsize=7)
        ax_r.tick_params(labelsize=7)
        ax_r.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                    title=f"{n_trans} transitions / {n - 1}; P(stay)={p_stay:.2f} vs {p_indep:.2f} if i.i.d.",
                    title_fontsize=6.5)
        if i < len(mice) - 1: ax_r.tick_params(labelbottom=False)
    ax_r.set_xlabel("trial index (time order within session)", fontsize=8)
    fig.suptitle("(c) Hard-state sequence over the session (strip: colour = state, alpha = gamma_max) and rolling occupancy of each state.\n"
                 "Legend: mean dwell = mean run length in trials; P(stay) = fraction of consecutive-trial pairs in the same state, vs the i.i.d. expectation sum(p_z^2).",
                 fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.72, top=0.94, bottom=0.04)
    save_fig(fig, OUT, "fig_c_state_by_time", layout=None)
    return out


# ----------------------------------------------------------------------------- (d)
def fig_d(Dall, mice):
    fig, axes = plt.subplots(len(mice), 3, figsize=(11.5, 2.05 * len(mice)))
    out = {}
    for i, m in enumerate(mice):
        D = Dall[m]; K = D["K"]; hard = D["hard"]
        res = {}
        for col, (key, name, short) in enumerate([("lick", "preRZ lick rate (Hz)", "lick rate"),
                                                  ("vel", "preRZ velocity", "velocity")]):
            ax = axes[i, col]
            groups = [D[key][hard == z] for z in range(K)]
            okg = [g for z, g in enumerate(groups) if D["ok"][z]]
            kw_p = stats.kruskal(*okg).pvalue if len(okg) >= 2 else np.nan
            res[key] = dict(median={z: float(np.median(g)) for z, g in enumerate(groups)}, kw_p=kw_p)
            for z, g in enumerate(groups):
                if g.size < 2: continue
                vp = ax.violinplot(g, positions=[z], widths=0.8, showmedians=True, showextrema=False)
                c = STATE_COLS[z] if D["ok"][z] else GREY
                for b in vp["bodies"]:
                    b.set_facecolor(c); b.set_edgecolor("k"); b.set_alpha(0.55 if D["ok"][z] else 0.3)
                vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.2)
            ax.set_xticks(range(K))
            ax.set_xticklabels([f"s{z}\nn={D['n_hard'][z]}" + ("" if D["ok"][z] else "\n(supp.)") for z in range(K)],
                               fontsize=6.5)
            ax.set_xlim(-0.6, K - 0.4); ax.tick_params(labelsize=7)
            ax.set_title(f"m{m}  {short} by hard state   KW p={kw_p:.0e}", fontsize=7.5)
            if col == 0: ax.set_ylabel(f"m{m}\n{name}", fontsize=7.5)
            else: ax.set_ylabel(name, fontsize=7.5)
        ax = axes[i, 2]
        pc = {}
        for z in range(K):
            c = STATE_COLS[z] if D["ok"][z] else GREY
            msk = (hard == z) & D["offb"]                       # ori != 45: only trials with a correct answer
            p, lo, hi = wilson(D["acc"][msk].sum(), msk.sum())
            m1 = msk & (D["ct"] == 1.0)
            p1, lo1, hi1 = wilson(D["acc"][m1].sum(), m1.sum())
            pc[z] = (p, msk.sum(), p1, m1.sum())
            ax.errorbar(z - 0.12, p, yerr=[[p - lo], [hi - p]], fmt="o", color=c, ms=6, capsize=3)
            ax.errorbar(z + 0.12, p1, yerr=[[p1 - lo1], [hi1 - p1]], fmt="o", color=c, ms=6, capsize=3,
                        mfc="w", mew=1.5)
            ax.text(z, 0.03, f"n={msk.sum()} | {m1.sum()}", ha="center", fontsize=5.5, color="0.4",
                    transform=ax.get_xaxis_transform())
        res["pcorrect"] = pc
        ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
        ax.set_ylim(0, 1.05); ax.set_xlim(-0.6, K - 0.4)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"s{z}\nn={D['n_hard'][z]}" for z in range(K)], fontsize=6.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("P(correct), Wilson 95% CI", fontsize=7.5)
        ax.set_title(f"m{m}  P(correct) by hard state: filled = all c | hollow = c 1", fontsize=7.5)
        out[m] = res
    for ax in axes[-1]: ax.set_xlabel("hard state", fontsize=8)
    fig.suptitle("(d) Behaviour by hard state (all pkl trials). Violins: distribution, black bar = median; grey = state with n<15 (no stats).\n"
                 "P(correct) = choice matches the rule (Go iff ori<45), ori=45 excluded; the pkl 'correct' field is the Go-stimulus "
                 "category, not accuracy.\nKruskal-Wallis / Wilson treat trials as independent (states are sticky). "
                 "Velocity is plausibly an IO-HMM emission (per-state *_v params): its state dependence is partly by construction.",
                 fontsize=8.5)
    save_fig(fig, OUT, "fig_d_state_by_behaviour", max_px=1560)
    return out


# ----------------------------------------------------------------------------- (e)
PRED_NAMES = ["contrast", "dispersion", "|ori-45|", "prev state", "lick rate", "velocity"]


def design(D):
    """Predictor blocks (each an (n-1, p_k) array) and the target, trial 1..n-1."""
    K = D["K"]; hard = D["hard"]
    prev = np.zeros((D["n"] - 1, K))
    prev[np.arange(D["n"] - 1), hard[:-1]] = 1
    blocks = {
        "contrast": D["ct"][1:, None],
        "dispersion": D["disp"][1:, None],
        "|ori-45|": np.abs(D["ori"][1:] - 45)[:, None],
        "prev state": prev[:, 1:],                       # drop-first one-hot of previous hard state
        "lick rate": D["lick"][1:, None],
        "velocity": D["vel"][1:, None],
    }
    y = hard[1:]
    return blocks, y


def mcfadden(X, y, cv_folds=5, C=1e4):
    """In-sample and blocked-CV McFadden pseudo-R2 of a multinomial logit y ~ X."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan, np.nan

    def fit(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=C, max_iter=5000)
            clf.fit(sc.transform(Xtr), ytr)
        return sc, clf

    def ll(clf, sc, X, y, classes_all):
        P = clf.predict_proba(sc.transform(X))
        col = {c: i for i, c in enumerate(clf.classes_)}
        out = np.full(len(y), np.log(1e-12))
        for i, yy in enumerate(y):
            if yy in col: out[i] = np.log(max(P[i, col[yy]], 1e-12))
        return out.sum()

    def ll_null(ytr, yte):
        p = np.array([(ytr == c).mean() for c in classes]); p = np.clip(p, 1e-12, 1)
        return sum(np.log(p[list(classes).index(yy)]) for yy in yte)

    sc, clf = fit(X, y)
    r2_in = 1 - ll(clf, sc, X, y, classes) / ll_null(y, y)
    # blocked (contiguous-in-time) folds
    n = len(y); edges = np.linspace(0, n, cv_folds + 1).astype(int)
    ll_m, ll_0 = 0.0, 0.0
    for k in range(cv_folds):
        te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True; tr = ~te
        if len(np.unique(y[tr])) < 2: continue
        sc_k, clf_k = fit(X[tr], y[tr])
        ll_m += ll(clf_k, sc_k, X[te], y[te], classes)
        ll_0 += ll_null(y[tr], y[te])
    r2_cv = 1 - ll_m / ll_0
    return r2_in, r2_cv


def fig_e(Dall, mice):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
    out = {}
    for ax, m in zip(axes.ravel(), mice):
        D = Dall[m]
        blocks, y = design(D)
        keep = D["ok"][y]                                # target restricted to states with n>=15
        y = y[keep]; blocks = {k: v[keep] for k, v in blocks.items()}
        groups = {"full": PRED_NAMES,
                  "stimulus": ["contrast", "dispersion", "|ori-45|"],
                  "behaviour": ["lick rate", "velocity"]}
        r = {g: mcfadden(np.hstack([blocks[k] for k in ks]), y) for g, ks in groups.items()}
        for k in PRED_NAMES:
            r[k] = mcfadden(blocks[k], y)
        out[m] = r
        names = list(groups) + PRED_NAMES
        cols = ["0.25", "#8172B2", "#64B5CD"] + ["#4C72B0"] * len(PRED_NAMES)
        xs = np.arange(len(names))
        ins = [r[k][0] for k in names]; cvs = [r[k][1] for k in names]
        ax.bar(xs, ins, color=cols, width=0.65, label="in-sample")
        ax.plot(xs, np.clip(cvs, -0.05, None), "o", color="k", ms=5, mfc="w", mew=1.4, label="blocked 5-fold CV")
        for xx, (a, b) in enumerate(zip(ins, cvs)):
            ax.text(xx, max(a, b, 0) + 0.035, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(xs); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.axvline(2.5, color="0.7", lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"m{m}  K={D['K']}  (n={len(y)} trials, states n>=15)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        if m == mice[0]: ax.legend(fontsize=7, frameon=False, loc="upper right")
    for ax in axes[:, 0]: ax.set_ylabel("McFadden pseudo-R2\n(1 - LL_model / LL_intercept-only)", fontsize=8)
    ymax = max(0.05, np.nanmax([v[0] for r in out.values() for v in r.values()]) * 1.15)
    axes[0, 0].set_ylim(-0.06, ymax)
    fig.suptitle("(e) What predicts the hard state? Multinomial logistic regression on standardised predictors: full model,\n"
                 "stimulus-only (contrast+dispersion+|ori-45|), behaviour-only (lick+velocity), then each predictor alone. "
                 "Bars = in-sample; hollow dots = blocked\n(contiguous-in-time) 5-fold CV, clipped at -0.05. "
                 "'prev state' = one-hot of the previous trial's hard state. Velocity is plausibly an IO-HMM emission (per-state *_v params).",
                 fontsize=8.5)
    save_fig(fig, OUT, "fig_e_what_predicts_state", max_px=1560)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(REPO / "data" / "fitted_data_and_posteriors_hmm.pkl"))
    a = ap.parse_args()
    Dall = load(a.pkl); OUT.mkdir(parents=True, exist_ok=True)
    mice = sorted(Dall)
    print("Data: all pkl trials, time order = pkl index; states with hard n<15 suppressed.")
    for m in mice:
        D = Dall[m]
        print(f"  m{m}: n={D['n']} K={D['K']} hard n per state={D['n_hard'].tolist()} suppressed={[z for z in range(D['K']) if not D['ok'][z]]}")

    print(f"\n(a) P(state | contrast x dispersion): Cramer's V of hard state vs cell (states n>=15), "
          f"occupancy range across cells with n>={MIN_CELL_N} per state, and by contrast (all dispersions)")
    ra = fig_a(Dall, mice)
    for m in mice:
        D = Dall[m]; V, p, dof, ncell = ra[m]
        rng = []
        for z in range(D["K"]):
            if not D["ok"][z]: rng.append("–"); continue
            occ = []
            for c in CONTRASTS:
                for dd in DISPERSIONS:
                    msk = (D["ct"] == c) & (D["disp"] == dd)
                    if msk.sum() >= MIN_CELL_N: occ.append(D["gamma"][msk, z].mean())
            occ_c = [D["gamma"][D["ct"] == c, z].mean() for c in CONTRASTS]
            rng.append(f"s{z}: cells(n>={MIN_CELL_N}) {min(occ):.2f}–{max(occ):.2f}, by contrast {'/'.join(f'{o:.2f}' for o in occ_c)}")
        print(f"  m{m}: V={V:.2f} p={p:.1e} ({ncell} cells)  " + "; ".join(rng))

    print("\n(b) state vs orientation: Cramer's V of hard state vs orientation (9 values)")
    rb = fig_b(Dall, mice)
    for m in mice:
        D = Dall[m]; V, p = rb[m]
        occ_o = {z: [D["gamma"][D["ori"] == o, z].mean() for o in ORIS] for z in range(D["K"]) if D["ok"][z]}
        print(f"  m{m}: V={V:.2f} p={p:.1e}  " + "; ".join(f"s{z} range {min(v):.2f}–{max(v):.2f}" for z, v in occ_o.items()))

    print("\n(c) state vs time: transitions, P(stay) vs i.i.d., mean dwell (trials) per state")
    rc = fig_c(Dall, mice)
    for m in mice:
        r = rc[m]
        print(f"  m{m}: {r['n_trans']} transitions / {Dall[m]['n'] - 1}; P(stay)={r['p_stay']:.2f} vs {r['p_indep']:.2f} i.i.d.; "
              "dwell " + ", ".join(f"s{z} {v:.1f}" for z, v in r["dwell"].items()))

    print("\n(d) behaviour by hard state: median lick rate / velocity per state, KW p; P(correct) all / contrast-1")
    rd = fig_d(Dall, mice)
    for m in mice:
        r = rd[m]; D = Dall[m]
        lick = ", ".join(f"s{z} {v:.2f}" for z, v in r["lick"]["median"].items() if D["ok"][z])
        vel = ", ".join(f"s{z} {v:.1f}" for z, v in r["vel"]["median"].items() if D["ok"][z])
        pc = ", ".join(f"s{z} {v[0]:.2f}/{v[2]:.2f}" for z, v in r["pcorrect"].items() if D["ok"][z])
        print(f"  m{m}: lick med [{lick}] KW p={r['lick']['kw_p']:.1e}; vel med [{vel}] KW p={r['vel']['kw_p']:.1e}; "
              f"P(correct) [{pc}]")

    print("\n(e) McFadden pseudo-R2 (in-sample / blocked-CV): full, stimulus-only, behaviour-only, then each predictor alone")
    re_ = fig_e(Dall, mice)
    for m in mice:
        r = re_[m]
        print(f"  m{m}: " + "  ".join(f"{k}={r[k][0]:.2f}/{r[k][1]:.2f}"
                                      for k in ["full", "stimulus", "behaviour"] + PRED_NAMES))
    print("\nNotes: (1) all p-values treat trials as independent; states are sticky, so they are anti-conservative — "
          "read V / R2, not p. (2) pkl 'correct' = Go-stimulus category, not accuracy (asserted in load(); "
          "P(correct) here is rule-derived, ori=45 excluded). (3) per-state Params carry velocity-link terms "
          "(alpha_v/beta_v/gamma_v), so velocity is plausibly an emission the state posterior was computed from — "
          "its state dependence is partly by construction; lick-rate terms (*_nl) are all 0 in the fits.")


if __name__ == "__main__":
    main()
