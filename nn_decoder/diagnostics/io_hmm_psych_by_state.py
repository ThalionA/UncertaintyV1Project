"""IO-HMM psychometric curves per latent state — all six mice.

Builds on diagnostics/io_hmm_posteriors_by_state.py (which established that the
states differ in CONCENTRATION of P(stim | state), not location).  Here the
question is behavioural: does the animal's choice function differ between
states, and does the model's own per-state P(Go) reproduce it?

Data: ALL pkl trials (no alignment to the neural export — these are pure
behaviour/model figures, so nothing is dropped; the export-aligned scripts lose
~5 trials per mouse).  Trial index = time order.  Posterior arrays are read via
io_hmm_data._get_posterior_field (Data.<name> is None in the HMM layout).

Definitions
  x-axis      orientation in deg (9 values); Go/NoGo boundary at 45 deg (ori<45 -> Go).
  empirical   P(Go) = fraction of choices==1.  Two flavours:
                hard   : trials with argmax gamma == z (counts, Wilson 95% CI)
                gamma  : sum(gamma_z * choice)/sum(gamma_z) over ALL trials
                         (Wilson CI on the Kish effective n = (sum w)^2/sum w^2)
  model       gamma-weighted mean of PS_choice_G_tr_by_state[z] per orientation
              (the model's own per-state P(choice = Go) on that state's actual
              trial mix); marginal = mean PS_choice_G_tr.
              NB (verified 2026-08-18): PS_Go_G_tr is NOT the choice probability —
              it is the perceptual posterior mass on the Go category (corr 1.000
              with the circular mass of PS_stim_G_tr on ori<45 ∪ ori>135) and sits
              far below the empirical Go rate; PS_choice_G_tr = sum_z gamma_z *
              PS_choice_G_tr_by_state[z] exactly and its per-orientation mean tracks
              the empirical curve.  Fig B therefore plots PS_choice (solid) as the
              psychometric curve and PS_Go (thin dotted) as the belief that feeds it.
  sharpness   doubled-angle resultant R of P(stim | state) on the native 72-bin
              circular support (0 uniform, 1 delta), gamma-weighted mean;
              "sharpest"/"broadest" state for fig (c) is defined by R at contrast 1
              (--sharp-by all uses the all-contrast R; the ordering differs for m0
              and m4 — stdout prints both).  A mouse whose sharpest and broadest
              usable states differ by ΔR < TIE_DR (0.05) has NO sharp/broad split
              (m5: 0.76 vs 0.74) and is flagged in figs C/C2 rather than read as one.
  caveats     fig D's pooled Spearman over (mouse,state) rows is descriptive only —
              rows are 2-3 per mouse (not independent); the within-mouse concordance
              (is the sharpest state also the most discriminating?) is what is reported.
              fig C2 hollows a point when either side of D has < MIN_N_D effective trials.
  discrim.    model-free discriminability D = P(Go | ori<=30) - P(Go | ori>=60);
              lapses = P(Go | ori=90) and 1 - P(Go | ori=0).
States with fewer than MIN_N hard-assigned trials are suppressed (grey note).

Figures -> figures/io_hmm_by_state/  (png+svg via figsave.save_fig)
  figA_psych_empirical_by_state   hard-state (top 2 rows) and gamma-weighted (bottom 2 rows)
  figB_psych_model_by_state       model per-state curves + faint gamma-weighted empirical
  figC_psych_contrast_sharp_vs_broad   4 contrasts x {sharpest, broadest} state per mouse
  figC2_discrim_vs_contrast       D vs contrast for the sharpest and broadest state
  figD_psych_scalars_vs_sharpness D and lapses vs state sharpness R
Usage: python diagnostics/io_hmm_psych_by_state.py [--pkl PATH] [--assign hard|gamma] [--sharp-by c1|all]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from figsave import save_fig            # noqa: E402
import io_hmm_data                      # noqa: E402

REPO = _HERE.parent.parent
GRID = io_hmm_data.GRID_DEG_IO          # 72 x 2.5 deg, [0,180)
OUT = _HERE.parent / "figures" / "io_hmm_by_state"
ORIS = np.array([0, 15, 30, 40, 45, 50, 60, 75, 90], float)
CTS = np.array([0.01, 0.25, 0.5, 1.0])
BOUNDARY = 45.0
MIN_N = 15                              # suppress states with fewer hard trials
MIN_N_POINT = 3                         # fig C: per-point effective n below which a point is hollow/unjoined
TIE_DR = 0.05                           # sharpest-vs-broadest R difference below which a mouse has NO sharp/broad split
MIN_N_D = 5                             # fig C2: effective n per side (ori<=30 / ori>=60) below which D is drawn hollow
STATE_COLS = plt.cm.tab10(np.arange(4))
CT_COLS = [plt.cm.magma(0.15 + 0.25 * j) for j in range(4)]   # 0.01 light -> 1 dark
MOUSE_MK = ["o", "s", "^", "D", "v", "P"]


# ----------------------------------------------------------------------------- helpers
def resultant(p):
    """Doubled-angle resultant length per row of (n,72). 0 = uniform, 1 = delta."""
    a = np.deg2rad(2 * GRID)
    return np.hypot((p * np.cos(a)).sum(1), (p * np.sin(a)).sum(1))


def wilson(p, n, z=1.96):
    """Wilson 95% interval (lo, hi) for a proportion p from n (may be effective n)."""
    n = np.asarray(n, float); p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / den
        h = z / den * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    lo, hi = c - h, c + h
    lo[n <= 0] = np.nan; hi[n <= 0] = np.nan
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def wpsych(x, choice, w, levels=ORIS):
    """Weighted P(Go) per level of x. Returns (p, n_eff, lo, hi); w = 0/1 gives counts."""
    p = np.full(len(levels), np.nan); n = np.zeros(len(levels))
    for k, lv in enumerate(levels):
        m = x == lv; ww = w[m]; s = ww.sum()
        if s > 0:
            p[k] = (ww * choice[m]).sum() / s
            n[k] = s ** 2 / (ww ** 2).sum()
    lo, hi = wilson(p, n)
    return p, n, lo, hi


def wmean(v, w):
    s = w.sum()
    return np.nan if s <= 0 else (w * v).sum() / s


def scalars(ori, choice, w):
    """Model-free discriminability and lapses from weighted P(Go)."""
    D = wmean(choice[ori <= 30], w[ori <= 30]) - wmean(choice[ori >= 60], w[ori >= 60])
    lapse_go = wmean(choice[ori == 90], w[ori == 90])          # Go on a clear NoGo
    lapse_nogo = 1 - wmean(choice[ori == 0], w[ori == 0])      # NoGo on a clear Go
    return D, lapse_go, lapse_nogo


# ----------------------------------------------------------------------------- load
def load(pkl, sharp_by="c1"):
    """All pkl trials per mouse; no export alignment (pure behaviour + model)."""
    mice = io_hmm_data.load_io_hmm_pkl(str(pkl), allow_partial=False)
    out = {}
    for m in sorted(mice):
        e = mice[m]; K = int(e["n_states"]); d = e["data"]
        g = lambda name: io_hmm_data._get_posterior_field(e, name)
        by = np.asarray(g("PS_stim_G_tr_by_state"), float)               # (K,72,n) bins-first
        by = np.transpose(by, (0, 2, 1)); by /= by.sum(2, keepdims=True)  # (K,n,72)
        gam = np.asarray(g("gamma"), float)                                # (n,K)
        n = gam.shape[0]
        ori = np.asarray(d["orientation"]).ravel().astype(float)
        ct = np.round(np.asarray(d["contrast"]).ravel().astype(float), 3)
        choice = np.asarray(d["choices"]).ravel().astype(float)
        assert ori.shape == (n,) and choice.shape == (n,), (m, ori.shape, gam.shape)
        assert set(np.unique(choice)) <= {0.0, 1.0}
        hard = gam.argmax(1)
        Rz = np.stack([resultant(by[z]) for z in range(K)])                # (K,n)
        R_all = np.array([wmean(Rz[z], gam[:, z]) for z in range(K)])
        R_c1 = np.array([wmean(Rz[z][ct == 1.0], gam[ct == 1.0, z]) for z in range(K)])
        occ = np.bincount(hard, minlength=K); ok = occ >= MIN_N
        Rsel = R_c1 if sharp_by == "c1" else R_all
        Rok = np.where(ok, Rsel, np.nan)
        dR = float(np.nanmax(Rok) - np.nanmin(Rok)) if ok.sum() >= 2 else np.nan
        out[m] = dict(K=K, n=n, ori=ori, ct=ct, choice=choice, gamma=gam, hard=hard,
                      Rsel=Rsel, dR=dR, tie=bool(np.isfinite(dR) and dR < TIE_DR),  # sharpest vs broadest within TIE_DR = no split
                      pch=np.asarray(g("PS_choice_G_tr"), float).ravel(),          # model P(choice=Go)
                      pch_by=np.asarray(g("PS_choice_G_tr_by_state"), float),      # (K,n)
                      pgo=np.asarray(g("PS_Go_G_tr"), float).ravel(),              # belief: P(Go category)
                      pgo_by=np.asarray(g("PS_Go_G_tr_by_state"), float),          # (K,n)
                      R_all=R_all, R_c1=R_c1, occ=occ, ok=ok,
                      sharp=int(np.argmax(np.where(ok, Rsel, -np.inf))),
                      broad=int(np.argmin(np.where(ok, Rsel, np.inf))),
                      sharp_alt=int(np.argmax(np.where(ok, R_all if sharp_by == "c1" else R_c1, -np.inf))),
                      broad_alt=int(np.argmin(np.where(ok, R_all if sharp_by == "c1" else R_c1, np.inf))))
    return out


def weights(D, z, assign):
    return (D["hard"] == z).astype(float) if assign == "hard" else D["gamma"][:, z]


def _style(ax, ylab=True, xlab=True):
    ax.axvline(BOUNDARY, color="0.6", lw=0.8, ls=":")
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]); ax.set_xlim(-4, 94); ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=7)
    if xlab: ax.set_xlabel("orientation (deg)", fontsize=8)
    if ylab: ax.set_ylabel("P(Go)", fontsize=8)


def _suppressed_note(ax, D):
    """Grey legend entry (no data drawn) for every state with n < MIN_N."""
    for z in range(D["K"]):
        if not D["ok"][z]:
            ax.plot([], [], color="0.6", lw=0, label=f"state {z}  n={D['occ'][z]} — suppressed (<{MIN_N})")


def _legend(ax, loc="best"):
    leg = ax.legend(fontsize=6, frameon=False, loc=loc, handlelength=1.5)
    for t in leg.get_texts():
        if "suppressed" in t.get_text(): t.set_color("0.45")


def _plot_curve(ax, p, n, lo, hi, color, label=None, alpha=1.0, lw=1.3, ms_scale=0.35, z=3,
                min_n=1, ci_alpha=None):
    """Line + CI whiskers + area-∝-n markers. Points with n < min_n are drawn hollow,
    without a CI, and are not joined by the line (they are noise, not evidence)."""
    ok = np.isfinite(p) & (n >= min_n); weak = np.isfinite(p) & (n < min_n)
    yerr = np.clip([p[ok] - lo[ok], hi[ok] - p[ok]], 0, None)   # fp guard: Wilson always contains p
    ax.plot(ORIS[ok], p[ok], "-", color=color, lw=lw, alpha=alpha, label=label, zorder=z)
    ax.errorbar(ORIS[ok], p[ok], yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=0,
                alpha=alpha if ci_alpha is None else ci_alpha, zorder=z)
    ax.scatter(ORIS[ok], p[ok], s=6 + ms_scale * n[ok], color=color, alpha=alpha,
               edgecolor="white", lw=0.4, zorder=z + 1)
    if weak.any():
        ax.scatter(ORIS[weak], p[weak], s=10, facecolor="none", edgecolor=color, lw=0.8,
                   alpha=alpha, zorder=z + 1)


# ----------------------------------------------------------------------------- figures
def fig_a(D, mice):
    """Empirical psych curves per state: hard-state (rows 0-1) and gamma-weighted (rows 2-3)."""
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 12.5), sharex=True, sharey=True)
    for blk, assign in enumerate(("hard", "gamma")):
        for i, m in enumerate(mice):
            ax = axes[2 * blk + i // 3, i % 3]; d = D[m]
            for z in range(d["K"]):
                if not d["ok"][z]: continue
                w = weights(d, z, assign)
                p, n, lo, hi = wpsych(d["ori"], d["choice"], w)
                lab = (f"state {z}  n={d['occ'][z]}" if assign == "hard"
                       else f"state {z}  Σγ={w.sum():.0f}")
                _plot_curve(ax, p, n, lo, hi, STATE_COLS[z], label=lab)
            p, n, lo, hi = wpsych(d["ori"], d["choice"], np.ones(d["n"]))
            _plot_curve(ax, p, n, lo, hi, "k", label=f"all trials n={d['n']}", lw=1.0,
                        ms_scale=0.12, z=5)
            _suppressed_note(ax, d)
            ax.set_title(f"m{m}  K={d['K']}  — " + ("hard state (argmax γ)" if assign == "hard"
                                                     else "γ-weighted (all trials, soft)"), fontsize=8.5)
            _legend(ax)
            _style(ax, ylab=(i % 3 == 0), xlab=(2 * blk + i // 3 == 3))
    fig.suptitle("Empirical psychometric curves per IO-HMM state — P(Go) vs orientation, all pkl trials\n"
                 "top two rows: trials hard-assigned to a state; bottom two rows: every trial weighted by γ(state). "
                 "Bars = Wilson 95% CI (effective n for γ); marker area ∝ n per point; dotted = 45° boundary",
                 fontsize=10)
    save_fig(fig, OUT, "figA_psych_empirical_by_state", max_px=1450)


def fig_b(D, mice, assign):
    """Model per-state psych curves (gamma-weighted PS_choice_by_state), empirical points faint;
    thin dotted = PS_Go_by_state, the perceptual belief in the Go category (not a choice prob)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex=True, sharey=True)
    calib, calib_go = {}, {}
    for i, m in enumerate(mice):
        ax = axes[i // 3, i % 3]; d = D[m]
        for z in range(d["K"]):
            if not d["ok"][z]: continue
            w = d["gamma"][:, z]
            pm = np.array([wmean(d["pch_by"][z][d["ori"] == o], w[d["ori"] == o]) for o in ORIS])
            pb = np.array([wmean(d["pgo_by"][z][d["ori"] == o], w[d["ori"] == o]) for o in ORIS])
            pe, n, lo, hi = wpsych(d["ori"], d["choice"], weights(d, z, assign))
            ax.plot(ORIS, pm, "-", color=STATE_COLS[z], lw=1.8, zorder=4,
                    label=f"state {z}  P(choice=Go | state)  Σγ={w.sum():.0f}")
            ax.plot(ORIS, pb, ":", color=STATE_COLS[z], lw=1.0, zorder=3, alpha=0.9)
            _plot_curve(ax, pe, n, lo, hi, STATE_COLS[z], alpha=0.4, lw=0.0, ms_scale=0.35, z=2)
            ok = np.isfinite(pe)
            calib[(m, z)] = np.average(np.abs(pm[ok] - pe[ok]), weights=n[ok])
            calib_go[(m, z)] = np.average(np.abs(pb[ok] - pe[ok]), weights=n[ok])
        pm_all = np.array([d["pch"][d["ori"] == o].mean() for o in ORIS])
        pe, n, lo, hi = wpsych(d["ori"], d["choice"], np.ones(d["n"]))
        ax.plot(ORIS, pm_all, "k--", lw=1.0, label="marginal model P(choice=Go)", zorder=4)
        ax.scatter(ORIS, pe, s=10, color="k", alpha=0.5, zorder=3, label="all-trial empirical")
        ax.plot([], [], ":", color="0.4", lw=1.0, label="dotted: belief P(Go category | state)")
        _suppressed_note(ax, d)
        ax.set_title(f"m{m}  K={d['K']}", fontsize=9)
        _legend(ax)
        _style(ax, ylab=(i % 3 == 0), xlab=(i // 3 == 1))
    fig.suptitle("Model per-state psychometric curves — γ-weighted mean of PS_choice_G_tr_by_state per orientation "
                 f"(solid) vs {assign}-assigned empirical P(Go) (faint points, Wilson 95% CI, area ∝ n)\n"
                 "thin dotted = PS_Go_G_tr_by_state (perceptual mass on the Go category — the belief, not the choice); "
                 "black dashed = marginal model; black dots = all-trial empirical",
                 fontsize=9.5)
    save_fig(fig, OUT, "figB_psych_model_by_state", max_px=1500)
    return calib, calib_go


def fig_c(D, mice, assign, sharp_by):
    """Contrast-resolved empirical psych curves for the sharpest and broadest state per mouse."""
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharex=True, sharey=True)
    Dc = {}
    for i, m in enumerate(mice):
        d = D[m]; r = i // 2
        for j, (tag, z) in enumerate((("sharpest", d["sharp"]), ("broadest", d["broad"]))):
            ax = axes[r, 2 * (i % 2) + j]
            w = weights(d, z, assign)
            for k, c in enumerate(CTS):
                msk = d["ct"] == c
                p, n, lo, hi = wpsych(d["ori"][msk], d["choice"][msk], w[msk])
                _plot_curve(ax, p, n, lo, hi, CT_COLS[k], label=f"c={c:g}  n={w[msk].sum():.0f}", ms_scale=0.6,
                            min_n=MIN_N_POINT, ci_alpha=0.45)
                Dc[(m, tag, c)] = scalars(d["ori"][msk], d["choice"][msk], w[msk])[0]
                # Kish effective n on each side of D (ori<=30 | ori>=60) so fig C2 can hollow under-powered points
                ww = w[msk]; oo = d["ori"][msk]
                neff = lambda sel: (ww[sel].sum() ** 2 / (ww[sel] ** 2).sum()) if ww[sel].sum() > 0 else 0.0  # noqa: E731
                Dc[(m, tag, c, "neff")] = min(neff(oo <= 30), neff(oo >= 60))
            nlab = f"n={d['occ'][z]}" if assign == "hard" else f"Σγ={w.sum():.0f}"
            tie = f"\n[ΔR={d['dR']:.2f} between usable states: NO sharp/broad split]" if d["tie"] else "\n"
            ax.set_title(f"m{m}  {tag} state {z}  R(c=1)={d['R_c1'][z]:.2f}  R(all)={d['R_all'][z]:.2f}  {nlab}{tie}",
                         fontsize=7.5, color=STATE_COLS[z], linespacing=1.1)
            _legend(ax)
            _style(ax, ylab=(2 * (i % 2) + j == 0), xlab=(r == 2))
            if d["sharp"] == d["broad"]:
                ax.text(0.5, 0.5, "only one usable state", transform=ax.transAxes, fontsize=7, color="0.45", ha="center")
    fig.suptitle(f"Contrast-resolved empirical psychometric curves ({assign}-assigned trials) — sharpest vs broadest "
                 f"state per mouse (state sharpness = γ-weighted mean R of P(stim | state) at contrast {'1' if sharp_by == 'c1' else 'all'})\n"
                 f"contrast 0.01 → 1 light → dark; Wilson 95% CI; marker area ∝ n; hollow = n<{MIN_N_POINT} (no CI, unjoined); "
                 f"states with n<{MIN_N} excluded\na mouse whose usable states differ by ΔR<{TIE_DR} has no sharp/broad split "
                 "(flagged in the panel title)", fontsize=10)
    save_fig(fig, OUT, "figC_psych_contrast_sharp_vs_broad", max_px=1500)
    return Dc


def fig_c2(D, mice, Dc):
    """Discriminability vs contrast, sharpest vs broadest state, per mouse."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
    for j, tag in enumerate(("sharpest", "broadest")):
        ax = axes[j]
        for i, m in enumerate(mice):
            z = D[m][tag[:5]]
            y = np.array([Dc[(m, tag, c)] for c in CTS])
            ne = np.array([Dc[(m, tag, c, "neff")] for c in CTS])
            weak = ne < MIN_N_D
            tie = "  [no split]" if D[m]["tie"] else ""
            ax.plot(np.arange(4), y, "-", color=STATE_COLS[z], lw=1.0, alpha=0.8, zorder=2)
            ax.scatter(np.arange(4)[~weak], y[~weak], marker=MOUSE_MK[i], color=STATE_COLS[z], s=40, zorder=3,
                       label=f"m{m} state {z} (R={D[m]['R_c1'][z]:.2f}, Σγ={D[m]['gamma'][:, z].sum():.0f}){tie}")
            ax.scatter(np.arange(4)[weak], y[weak], marker=MOUSE_MK[i], facecolor="none", edgecolor=STATE_COLS[z],
                       s=40, zorder=3)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{c:g}" for c in CTS])
        ax.set_xlabel("contrast"); ax.set_title(f"{tag} state per mouse (by R at contrast 1)", fontsize=9)
        ax.legend(fontsize=6.3, frameon=False)
        if j == 0: ax.set_ylabel("discriminability  P(Go|ori≤30) − P(Go|ori≥60)\n(γ-weighted over all trials)", fontsize=8.5)
    fig.suptitle("Is the broad state contrast-blind?  Model-free discriminability vs contrast — marker = mouse, colour = state index "
                 f"(arbitrary per mouse)\nhollow marker = fewer than {MIN_N_D} effective trials on one side (ori≤30 or ori≥60); "
                 f"[no split] = usable states within ΔR<{TIE_DR} of each other", fontsize=9)
    save_fig(fig, OUT, "figC2_discrim_vs_contrast", max_px=1500)


def fig_d(D, mice, assign):
    """Psych summary scalars per (mouse, state) vs state sharpness R."""
    rows = []
    for i, m in enumerate(mice):
        d = D[m]
        for z in range(d["K"]):
            if not d["ok"][z]: continue
            Dz, lg, ln = scalars(d["ori"], d["choice"], weights(d, z, assign))
            rows.append(dict(m=m, i=i, z=z, R=d["R_all"][z], R_c1=d["R_c1"][z], D=Dz,
                             lapse_go=lg, lapse_nogo=ln, n=d["occ"][z]))
    R = np.array([r["R"] for r in rows]); Dv = np.array([r["D"] for r in rows])
    rho, pval = spearmanr(R, Dv)
    # within-mouse check: rows are 2-3 per mouse (not independent), so the pooled p is not meaningful; count the mice
    # in which the sharpest usable state (by R_all) is also the most discriminating
    conc = []
    for m in mice:
        rm = [r for r in rows if r["m"] == m]
        if len(rm) >= 2:
            conc.append(max(rm, key=lambda r: r["R"])["z"] == max(rm, key=lambda r: r["D"])["z"])
    n_conc = int(sum(conc))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True)
    for ax, key, ylab in zip(axes, ("D", "lapse_go", "lapse_nogo"),
                             ("discriminability  P(Go|ori≤30) − P(Go|ori≥60)",
                              "Go lapse  P(Go | ori=90)", "NoGo lapse  1 − P(Go | ori=0)")):
        for r in rows:
            ax.scatter(r["R"], r[key], marker=MOUSE_MK[r["i"]], s=40 + 0.15 * r["n"], color=STATE_COLS[r["z"]],
                       edgecolor="k", lw=0.5, zorder=3)
            ax.annotate(f"m{r['m']}s{r['z']} ({r['n']})", (r["R"], r[key]), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points", color="0.3")
        ax.set_xlabel("state sharpness  R of P(stim | state), γ-weighted, all contrasts", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8); ax.tick_params(labelsize=7); ax.set_xlim(0, 0.7)
    axes[0].set_title(f"pooled ρ = {rho:.2f} over {len(rows)} rows (descriptive: 2-3 rows per mouse)\n"
                      f"within mouse: sharpest = most discriminating in {n_conc}/{len(conc)} mice", fontsize=8.5)
    axes[1].set_ylim(-0.02, 1.02); axes[2].set_ylim(-0.02, 1.02)
    for i, m in enumerate(mice):
        axes[1].scatter([], [], marker=MOUSE_MK[i], color="0.5", edgecolor="k", label=f"m{m}")
    for z in range(3):
        axes[1].scatter([], [], marker="o", color=STATE_COLS[z], label=f"state {z}")
    axes[1].legend(fontsize=6, frameon=False, ncol=2, loc="upper right")
    fig.suptitle(f"Psychometric summary scalars per (mouse, state), {assign}-assigned trials — "
                 f"marker = mouse, colour = state index (arbitrary per mouse),\narea ∝ hard n (in brackets); states with n<{MIN_N} excluded",
                 fontsize=10)
    save_fig(fig, OUT, "figD_psych_scalars_vs_sharpness", max_px=1500)
    return rows, rho, pval, n_conc, len(conc)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(REPO / "data" / "fitted_data_and_posteriors_hmm.pkl"))
    ap.add_argument("--assign", choices=("hard", "gamma"), default="gamma",
                    help="trial→state assignment for figs B/C/D (fig A shows both)")
    ap.add_argument("--sharp-by", choices=("c1", "all"), default="c1",
                    help="define sharpest/broadest state by R at contrast 1 (default) or all contrasts")
    a = ap.parse_args()
    D = load(a.pkl, a.sharp_by); OUT.mkdir(parents=True, exist_ok=True); mice = sorted(D)

    print(f"== all pkl trials, no export alignment; per-state sharpness R (γ-weighted; all contrasts | contrast 1), "
          f"hard occupancy, sharpest/broadest by R_{a.sharp_by} ==")
    for m in mice:
        d = D[m]
        alt = "" if (d["sharp_alt"], d["broad_alt"]) == (d["sharp"], d["broad"]) else \
              f"  [by R_{'all' if a.sharp_by == 'c1' else 'c1'}: s{d['sharp_alt']}/s{d['broad_alt']} — ordering DIFFERS]"
        print(f"m{m} K={d['K']} n={d['n']}  occ={d['occ'].tolist()}  R_all={np.round(d['R_all'], 2).tolist()}  "
              f"R_c1={np.round(d['R_c1'], 2).tolist()}  sharpest=s{d['sharp']} broadest=s{d['broad']} (ΔR={d['dR']:.2f}"
              f"{', NO sharp/broad split' if d['tie'] else ''}){alt}"
              + ("" if d["ok"].all() else f"  SUPPRESSED={[z for z in range(d['K']) if not d['ok'][z]]}"))

    fig_a(D, mice)
    print("== fig A: empirical P(Go) per orientation, hard | γ-weighted (rows = mouse/state) ==")
    print(f"{'':10s}" + " ".join(f"{o:>5.0f}" for o in ORIS))
    for m in mice:
        d = D[m]
        for z in range(d["K"]):
            if not d["ok"][z]: continue
            ph = wpsych(d["ori"], d["choice"], weights(d, z, "hard"))[0]
            pg = wpsych(d["ori"], d["choice"], weights(d, z, "gamma"))[0]
            print(f"m{m}s{z} hard  " + " ".join(f"{v:5.2f}" for v in ph))
            print(f"m{m}s{z} gamma " + " ".join(f"{v:5.2f}" for v in pg))

    calib, calib_go = fig_b(D, mice, a.assign)
    print(f"== fig B: n-weighted mean |model − {a.assign} empirical P(Go)| over orientations, per (mouse,state) ==")
    print("  PS_choice_by_state (psych curve): " + "  ".join(f"m{m}s{z}={v:.2f}" for (m, z), v in calib.items()))
    print("  PS_Go_by_state (belief, NOT choice): " + "  ".join(f"m{m}s{z}={v:.2f}" for (m, z), v in calib_go.items()))

    Dc = fig_c(D, mice, a.assign, a.sharp_by)
    fig_c2(D, mice, Dc)
    print(f"== fig C/C2: discriminability D by contrast ({a.assign}), sharpest | broadest state ==")
    print(f"{'':6s}" + "".join(f"{'c='+format(c,'g'):>8s}" for c in CTS) + "   |" + "".join(f"{'c='+format(c,'g'):>8s}" for c in CTS))
    for m in mice:
        d = D[m]
        print(f"m{m}  " + "".join(f"{Dc[(m,'sharpest',c)]:8.2f}" for c in CTS)
              + f"  s{d['sharp']}|" + "".join(f"{Dc[(m,'broadest',c)]:8.2f}" for c in CTS) + f"  s{d['broad']}")

    rows, rho, pval, n_conc, n_m = fig_d(D, mice, a.assign)
    print(f"== fig D: scalars per (mouse,state), {a.assign} — pooled Spearman rho(D, R_all) = {rho:.3f} over n={len(rows)} rows "
          f"(rows are 2-3 per mouse, NOT independent: p={pval:.3f} is anti-conservative, descriptive only); "
          f"within-mouse: sharpest state = most discriminating in {n_conc}/{n_m} mice ==")
    rho_c1, p_c1 = spearmanr([r["R_c1"] for r in rows], [r["D"] for r in rows])
    rho_lg, p_lg = spearmanr([r["R"] for r in rows], [r["lapse_go"] for r in rows])
    print(f"   (D vs R at contrast 1: rho = {rho_c1:.3f}, p={p_c1:.3f};  Go-lapse vs R_all: rho = {rho_lg:.3f}, p={p_lg:.3f})")
    print(f"{'row':8s}{'n':>5s}{'R_all':>7s}{'R_c1':>7s}{'D':>7s}{'lapseGo':>9s}{'lapseNoGo':>10s}")
    for r in rows:
        print(f"m{r['m']}s{r['z']:<5d}{r['n']:5d}{r['R']:7.2f}{r['R_c1']:7.2f}{r['D']:7.2f}{r['lapse_go']:9.2f}{r['lapse_nogo']:10.2f}")


if __name__ == "__main__":
    main()
