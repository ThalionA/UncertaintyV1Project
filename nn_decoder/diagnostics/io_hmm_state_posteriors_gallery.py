"""IO-HMM stimulus posteriors — how they look in general, and in each latent state.

Companion to ``diagnostics/io_hmm_posteriors_by_state.py`` (occupancy /
concentration summary + state-conditional condition means). This script is the
GALLERY view: what the ideal observer believes per mouse as a function of
stimulus (a), how wide each state's posterior is trial by trial (b), what single
trials actually look like (c), and whether "being in the sharp state" is what
makes a trial's marginal sharp (d).

Data: the complete 6-mouse HMM export ``data/fitted_data_and_posteriors_hmm.pkl``
read via ``io_hmm_data._get_posterior_field`` (posteriors live at entry level;
``Data.<name>`` is None in this layout). These are PURE model/behaviour figures,
so ALL pkl trials are used (no alignment to the neural export, which would drop
~5 trials per mouse; see ``io_hmm_posteriors_by_state.load`` for that path).

Support: native 72 x 2.5 deg circular bins on [0,180); every width statistic is
the doubled-angle resultant R = |sum_b p_b exp(2i theta_b)| (0 uniform, 1 delta)
— never a linear SD, never a fold. Trial index = time order in the session.
Sanity check printed at load: marginal == sum_z gamma_z P(stim|z) to ~1e-9.

Figures -> figures/io_hmm_by_state/  (png + svg via figsave.save_fig)
  gal_a_marginal_gallery           mouse x contrast; 9 orientation curves each
  gal_b_state_width_violins        per-trial R of P(stim|z), violin per state
  gal_c_single_trials_m1 / _m3     example trials, state x contrast grid
  gal_d_marginalR_vs_sharp_gamma   marginal R vs gamma(sharpest state), Spearman

Usage: python diagnostics/io_hmm_state_posteriors_gallery.py [--pkl PATH]
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
BIN = float(GRID[1] - GRID[0])          # 2.5 deg
OUT = _HERE.parent / "figures" / "io_hmm_by_state"
MIN_N = 15                              # per-state statistics need at least this many trials
CONTRASTS = (0.01, 0.25, 0.5, 1.0)
STATE_COLS = plt.cm.tab10(np.arange(10))                     # same as io_hmm_posteriors_by_state
CT_COLS = {c: plt.cm.magma(0.12 + 0.22 * j) for j, c in enumerate(CONTRASTS)}   # light->dark = low->high, kept off pale yellow


def _save(fig, stem):
    """figsave.save_fig with max_px=1540: bbox_inches='tight' grows the raster ~2-3% past figsize,
    so 1600 nominal lands at ~1615-1650 px; 1540 keeps the delivered PNG under 1600 on the long side."""
    save_fig(fig, OUT, stem, max_px=1540)


def resultant(p):
    """Doubled-angle resultant length per row of (..., 72). 0 = uniform, 1 = delta."""
    a = np.deg2rad(2 * GRID)
    return np.hypot((p * np.cos(a)).sum(-1), (p * np.sin(a)).sum(-1))


def load(pkl):
    """All pkl trials, per mouse. Posteriors normalised to sum 1 per trial; trials-first."""
    mice = io_hmm_data.load_io_hmm_pkl(str(pkl), allow_partial=False)
    out = {}
    for m in sorted(mice):
        e = mice[m]; K = int(e["n_states"])
        by = np.asarray(io_hmm_data._get_posterior_field(e, "PS_stim_G_tr_by_state"), float)  # (K,72,n)
        by = np.transpose(by, (0, 2, 1)); by = by / by.sum(2, keepdims=True)                   # (K,n,72)
        marg = np.asarray(io_hmm_data._get_posterior_field(e, "PS_stim_G_tr"), float).T       # (n,72)
        marg = marg / marg.sum(1, keepdims=True)
        gam = np.asarray(io_hmm_data._get_posterior_field(e, "gamma"), float)                 # (n,K)
        d = e["data"]
        mix_err = float(np.abs(np.einsum("nk,knb->nb", gam, by) - marg).max())
        hard = gam.argmax(1)
        n_hard = np.bincount(hard, minlength=K)
        Rz = resultant(by)                                                                    # (K,n)
        Rz_gw = np.array([np.average(Rz[z], weights=gam[:, z]) for z in range(K)])
        usable = n_hard >= MIN_N
        sharp = int(np.argmax(np.where(usable, Rz_gw, -np.inf)))
        out[m] = dict(K=K, by=by, marg=marg, gamma=gam, hard=hard, n_hard=n_hard, Rz=Rz,
                      Rmarg=resultant(marg), Rz_gw=Rz_gw, usable=usable, sharp=sharp, mix_err=mix_err,
                      ori=np.asarray(d["orientation"]).ravel().astype(float),
                      ct=np.round(np.asarray(d["contrast"]).ravel().astype(float), 3),
                      choice=np.asarray(d["choices"]).ravel().astype(float))
    return out


# --------------------------------------------------------------------------- (a)
def fig_marginal_gallery(D, mice):
    oris = sorted(np.unique(np.concatenate([D[m]["ori"] for m in mice])))
    ocol = {o: plt.cm.viridis(i / max(1, len(oris) - 1)) for i, o in enumerate(oris)}
    fig, axes = plt.subplots(len(mice), len(CONTRASTS), figsize=(13, 1.9 * len(mice)),
                             sharex=True, sharey=True)
    peaks = {}
    for r, m in enumerate(mice):
        for k, c in enumerate(CONTRASTS):
            ax = axes[r, k]
            for o in oris:
                msk = (D[m]["ct"] == c) & (D[m]["ori"] == o)
                if msk.sum() == 0: continue
                ax.plot(GRID, D[m]["marg"][msk].mean(0) / BIN, color=ocol[o], lw=1.2)
                peaks[(m, c, o)] = D[m]["marg"][msk].mean(0).max() / BIN
            ax.axvline(45, color="0.6", lw=0.6, ls=":")
            n_c = int((D[m]["ct"] == c).sum())
            ax.text(0.98, 0.95, f"n={n_c}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.35")
            ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180]); ax.tick_params(labelsize=7)
            if r == 0: ax.set_title(f"contrast {c:g}", fontsize=9)
            if k == 0: ax.set_ylabel(f"m{m} (K={D[m]['K']})\nP(stim | trial)  [prob/deg]", fontsize=7.5)
            if r == len(mice) - 1: ax.set_xlabel("orientation (deg)", fontsize=8)
    axes[0, 0].set_ylim(bottom=0)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, len(oris) - 1))
    cb = fig.colorbar(sm, ax=axes, ticks=range(len(oris)), fraction=0.02, pad=0.01)
    cb.ax.set_yticklabels([f"{o:g}" for o in oris]); cb.ax.tick_params(labelsize=7)
    cb.set_label("true orientation (deg); colour = orientation of the curve", fontsize=8)
    fig.suptitle("IO-HMM MARGINAL posterior P(stim | trial), condition mean per orientation — one curve per orientation, "
                 "rows = mouse, columns = contrast\n(dotted = 45 deg Go/NoGo boundary; shared y-axis; n = trials at that contrast; "
                 "the mean averages over trial-to-trial percept jitter, so it is broader than single trials — see gal_c)",
                 fontsize=10)
    _save(fig, "gal_a_marginal_gallery")
    return peaks


# --------------------------------------------------------------------------- (b)
def fig_state_width_violins(D, mice):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharey=True)
    ct_med = {}

    def _ct_medians(ax, pos, R, ct, key):
        # small squares to the right of each violin: median R at each contrast (explains bimodal violins)
        for c in CONTRASTS:
            s = ct == c
            if s.sum() < 5: ct_med[key + (c,)] = np.nan; continue
            ct_med[key + (c,)] = float(np.median(R[s]))
            ax.plot(pos + 0.33, ct_med[key + (c,)], "s", color=CT_COLS[c], ms=4, mec="w", mew=0.4, zorder=5)

    for ax, m in zip(axes.ravel(), mice):
        K = D[m]["K"]; pos = 0; ct = D[m]["ct"]
        vp = ax.violinplot([D[m]["Rmarg"]], positions=[pos], widths=0.75, showmedians=True, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor("0.6"); b.set_edgecolor("0.3"); b.set_alpha(0.6)
        vp["cmedians"].set_color("0.2")
        _ct_medians(ax, pos, D[m]["Rmarg"], ct, (m, "marg"))
        labels = [f"marginal\nn={len(D[m]['Rmarg'])}"]
        for z in range(K):
            pos += 1
            sel = D[m]["hard"] == z; n = int(sel.sum())
            labels.append(f"state {z}\nn={n}")
            if n < MIN_N:
                ax.scatter(np.full(n, pos) + np.random.default_rng(0).uniform(-0.15, 0.15, n), D[m]["Rz"][z][sel],
                           s=10, color="0.55", zorder=3)
                ax.text(pos, 0.98, f"n<{MIN_N}\n(suppressed)", ha="center", va="top", fontsize=7, color="0.45",
                        transform=ax.get_xaxis_transform())
                continue
            vp = ax.violinplot([D[m]["Rz"][z][sel]], positions=[pos], widths=0.75, showmedians=True, showextrema=False)
            for b in vp["bodies"]: b.set_facecolor(STATE_COLS[z]); b.set_edgecolor(STATE_COLS[z]); b.set_alpha(0.55)
            vp["cmedians"].set_color("k")
            ax.plot(pos, D[m]["Rz_gw"][z], "D", color=STATE_COLS[z], mec="k", ms=6, zorder=4)
            _ct_medians(ax, pos, D[m]["Rz"][z][sel], ct[sel], (m, z))
        ax.set_xticks(range(K + 1)); ax.set_xticklabels(labels, fontsize=7.5); ax.set_xlim(-0.5, K + 0.7)
        ax.set_ylim(0, 1); ax.set_title(f"m{m}  (K={K}, sharpest usable state = {D[m]['sharp']})", fontsize=9)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    for c in CONTRASTS:
        axes[0, 0].plot([], [], "s", color=CT_COLS[c], ms=4, label=f"median at contrast {c:g}")
    axes[0, 0].plot([], [], "D", color="0.5", mec="k", ms=6, label="gamma-weighted mean R (all trials)")
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper right")
    for ax in axes[:, 0]: ax.set_ylabel("per-trial resultant R of the posterior\n(0 uniform, 1 delta; native 72-bin support)", fontsize=8)
    for ax in axes[1]: ax.set_xlabel("posterior (violin = trials hard-assigned to that state)", fontsize=8)
    fig.suptitle("Width of the STATE-CONDITIONAL posterior P(stim | state = z), one violin per state (trials with argmax gamma = z; "
                 "black line = median; diamond = gamma-weighted mean R over ALL trials)\ngrey = marginal P(stim | trial) over all "
                 f"trials; small squares = median R per contrast (n >= 5); states with n < {MIN_N} shown as grey points only", fontsize=10)
    _save(fig, "gal_b_state_width_violins")
    return ct_med


# --------------------------------------------------------------------------- (c)
def pick_examples(D, m, gmin=0.9):
    """One trial per (usable state, contrast) cell: among trials with gamma_max > gmin & hard == z & ct == c,
    the one whose MARGINAL R is closest to that cell's median R (a typical trial, not an extreme). If the
    cell has no trial above gmin, the highest-gamma trial in it is used and flagged.
    Returns list of (z, c, idx, n_eligible, flagged)."""
    picks = []
    g = D[m]["gamma"]; gmax = g.max(1); R = D[m]["Rmarg"]
    for z in range(D[m]["K"]):
        if not D[m]["usable"][z]: continue
        for c in CONTRASTS:
            cell = np.flatnonzero((D[m]["hard"] == z) & (D[m]["ct"] == c))
            if cell.size == 0: picks.append((z, c, None, 0, True)); continue
            elig = cell[gmax[cell] > gmin]
            if elig.size:
                i = int(elig[np.argmin(np.abs(R[elig] - np.median(R[elig])))]); picks.append((z, c, i, int(elig.size), False))
            else:
                picks.append((z, c, int(cell[np.argmax(gmax[cell])]), 0, True))
    return picks


def fig_single_trials(D, m):
    picks = pick_examples(D, m)
    states = sorted({z for z, *_ in picks}); K = D[m]["K"]
    fig, axes = plt.subplots(len(states), len(CONTRASTS), figsize=(13, 2.9 * len(states)), sharex=True, squeeze=False)
    ymax = 0
    for (z, c, i, n_el, flagged) in picks:
        ax = axes[states.index(z), CONTRASTS.index(c)]
        if i is None:
            ax.text(0.5, 0.5, "no trial", transform=ax.transAxes, ha="center", va="center"); ax.set_axis_off(); continue
        for zz in range(K):
            if not D[m]["usable"][zz]: continue
            ax.plot(GRID, D[m]["by"][zz, i] / BIN, color=STATE_COLS[zz], lw=1.4)
        ax.plot(GRID, D[m]["marg"][i] / BIN, "k--", lw=1.1)
        ax.axvline(D[m]["ori"][i], color="0.4", lw=0.9, ls=":")
        gtxt = ", ".join(f"{D[m]['gamma'][i, zz]:.2f}" for zz in range(K))
        rtxt = "  ".join(f"z{zz} {D[m]['Rz'][zz, i]:.2f}" for zz in range(K) if D[m]["usable"][zz])
        ax.set_title(f"trial {i}  |  ori {D[m]['ori'][i]:g} deg, contrast {c:g}, {'Go' if D[m]['choice'][i] == 1 else 'NoGo'}\n"
                     f"gamma = ({gtxt});  cell has {n_el} trials >0.9" + ("  [NONE >0.9]" if flagged else "") + "\n"
                     f"R:  marginal {D[m]['Rmarg'][i]:.2f}   {rtxt}", fontsize=7.2)
        ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180]); ax.tick_params(labelsize=7)
        ymax = max(ymax, ax.get_ylim()[1])
    for r, z in enumerate(states):
        axes[r, 0].set_ylabel(f"hard state {z} (n={D[m]['n_hard'][z]})\nP(stim)  [prob/deg]", fontsize=8)
    for ax in axes[-1]: ax.set_xlabel("orientation (deg)", fontsize=8)
    for ax in axes.ravel(): ax.set_ylim(0, ymax)
    handles = [plt.Line2D([], [], color=STATE_COLS[zz], lw=1.6, label=f"P(stim | state {zz})") for zz in range(K) if D[m]["usable"][zz]]
    handles += [plt.Line2D([], [], color="k", ls="--", lw=1.2, label="marginal P(stim | trial)"),
                plt.Line2D([], [], color="0.4", ls=":", lw=1.0, label="true orientation")]
    fig.set_layout_engine("constrained")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f"m{m} (K={K}) single-trial posteriors — rows = hard-assigned state, columns = contrast\n"
                 "each panel = the trial whose marginal R is the median of its cell's gamma_max > 0.9 trials (a typical trial); "
                 "R = doubled-angle resultant on the native support", fontsize=10)
    _save(fig, f"gal_c_single_trials_m{m}")
    return picks


# --------------------------------------------------------------------------- (d)
def fig_marginalR_vs_sharp_gamma(D, mice):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, sharey=True)
    stats = {}
    for ax, m in zip(axes.ravel(), mice):
        zs = D[m]["sharp"]; g = D[m]["gamma"][:, zs]; R = D[m]["Rmarg"]
        rho, p = spearmanr(g, R)
        # what the state posterior alone predicts for sharpness: E_gamma[mean R of state]
        R_state_expected = D[m]["gamma"] @ D[m]["Rz_gw"]
        rho_exp = spearmanr(R_state_expected, R)[0]
        per_ct = {}
        for c in CONTRASTS:
            sel = D[m]["ct"] == c
            ax.scatter(g[sel] + np.random.default_rng(1).uniform(-0.012, 0.012, sel.sum()), R[sel], s=9, alpha=0.55,
                       color=CT_COLS[c], edgecolor="none", label=f"contrast {c:g} (n={sel.sum()})")
            per_ct[c] = spearmanr(g[sel], R[sel])[0] if sel.sum() > 3 and np.ptp(g[sel]) > 0 else np.nan
        stats[m] = dict(rho=rho, p=p, per_ct=per_ct, zs=zs, rho_exp=rho_exp)
        others = [(z, D[m]["Rz_gw"][z]) for z in range(D[m]["K"]) if D[m]["usable"][z] and z != zs]
        gap = D[m]["Rz_gw"][zs] - max(r for _, r in others) if others else np.nan
        tie = f"  TIE with s{max(others, key=lambda t: t[1])[0]} (gap {gap:.2f}): 'sharpest' ill-defined" \
            if others and gap < 0.05 else ""
        oth = ", ".join(f"s{z} {r:.2f}" for z, r in others) or "none"
        ax.set_title(f"m{m}: sharpest usable state = {zs} (n={D[m]['n_hard'][zs]}, mean R={D[m]['Rz_gw'][zs]:.2f})\n"
                     f"other usable states: {oth}{tie}\n"
                     f"Spearman rho = {rho:.2f} (n={len(R)} trials)   "
                     f"[vs sum_z gamma_z R_z: rho = {rho_exp:.2f}]", fontsize=7.6)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1); ax.grid(lw=0.3, alpha=0.5)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=1.8)
    for ax in axes[:, 0]: ax.set_ylabel("resultant R of the MARGINAL P(stim | trial)\n(per trial; 0 uniform, 1 delta)", fontsize=8)
    for ax in axes[1]: ax.set_xlabel("gamma of the sharpest state  (state posterior P(z = sharpest | trial))", fontsize=8)
    fig.suptitle("Is a trial's marginal sharp because the model puts it in the sharp state?  Marginal R vs gamma of the sharpest "
                 "usable state\n(sharpest = highest all-contrast gamma-weighted mean R; colour = contrast; x jittered +/-0.012; "
                 "per-contrast rho on stdout)\nwhere two states tie in R the x-axis is one arbitrary member of the tie — read the "
                 "bracketed rho vs sum_z gamma_z R_z instead",
                 fontsize=10)
    _save(fig, "gal_d_marginalR_vs_sharp_gamma")
    return stats


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(REPO / "data" / "fitted_data_and_posteriors_hmm.pkl"))
    a = ap.parse_args()
    D = load(a.pkl); OUT.mkdir(parents=True, exist_ok=True)
    mice = sorted(D)

    print("== load (all pkl trials) ==")
    for m in mice:
        print(f"m{m}: K={D[m]['K']} n={len(D[m]['hard'])} hard occ={D[m]['n_hard'].tolist()} "
              f"max|marg - sum_z gamma_z P(stim|z)|={D[m]['mix_err']:.1e}")

    print("\n== (a) marginal gallery: peak of the condition-mean marginal (prob/deg), ori 0 vs 45 vs 90, by contrast ==")
    peaks = fig_marginal_gallery(D, mice)
    for m in mice:
        row = "  ".join(f"c{c:g}: " + "/".join(f"{peaks.get((m, c, o), np.nan):.3f}" for o in (0.0, 45.0, 90.0)) for c in CONTRASTS)
        print(f"m{m}: {row}   (uniform = {1 / 180:.4f})")
    print("   single-trial marginal: fraction of trials whose posterior circular mean lies within +/-22.5 deg of the true "
          "orientation, by contrast 0.01/0.25/0.5/1 (chance = 0.25):")
    a2 = np.deg2rad(2 * GRID)
    for m in mice:
        mu = np.rad2deg(np.arctan2((D[m]["marg"] * np.sin(a2)).sum(1), (D[m]["marg"] * np.cos(a2)).sum(1))) / 2 % 180
        err = np.abs(((mu - D[m]["ori"]) + 90) % 180 - 90)
        print(f"   m{m}: " + "/".join(f"{(err[D[m]['ct'] == c] < 22.5).mean():.2f}" for c in CONTRASTS))

    print("\n== (b) per-trial R of P(stim|z) over hard-assigned trials: median [IQR] (n) | gw = gamma-weighted mean over all trials"
          " | median R at contrast 0.01/0.25/0.5/1 ==")
    ct_med = fig_state_width_violins(D, mice)

    def _cm(key):
        return "/".join("--" if np.isnan(ct_med.get(key + (c,), np.nan)) else f"{ct_med[key + (c,)]:.2f}" for c in CONTRASTS)

    for m in mice:
        rm = D[m]["Rmarg"]
        print(f"m{m}: marginal {np.median(rm):.2f} [{np.percentile(rm, 25):.2f},{np.percentile(rm, 75):.2f}] (n={len(rm)}) "
              f"by-contrast {_cm((m, 'marg'))}")
        for z in range(D[m]["K"]):
            sel = D[m]["hard"] == z; r = D[m]["Rz"][z][sel]
            tag = "" if D[m]["usable"][z] else "  SUPPRESSED (n<15)"
            print(f"      z{z}: {np.median(r):.2f} [{np.percentile(r, 25):.2f},{np.percentile(r, 75):.2f}] (n={sel.sum()}) | "
                  f"gw {D[m]['Rz_gw'][z]:.2f} | by-contrast {_cm((m, z))}{tag}")

    print("\n== (c) single-trial examples (state x contrast grid): trial idx, gamma_max, [n eligible in cell] ==")
    for m in (1, 3):
        picks = fig_single_trials(D, m)
        print(f"m{m}: " + ", ".join(f"z{z}/c{c:g}: t{i} g={D[m]['gamma'][i].max():.2f} [{n_el}]{'*' if fl else ''}"
                                    for z, c, i, n_el, fl in picks if i is not None)
              + "   (* = no trial with gamma_max>0.9 in that cell)")

    print("\n== (d) Spearman rho: marginal R vs gamma(sharpest usable state) — all trials, per contrast 0.01/0.25/0.5/1, "
          "and vs the state-expected sharpness sum_z gamma_z mean R_z ==")
    stats = fig_marginalR_vs_sharp_gamma(D, mice)
    for m in mice:
        s = stats[m]
        pc = "/".join("nan" if np.isnan(s["per_ct"][c]) else f"{s['per_ct'][c]:.2f}" for c in CONTRASTS)
        print(f"m{m}: sharpest z={s['zs']} (gw R={D[m]['Rz_gw'][s['zs']]:.2f}; other usable states "
              f"{[round(float(x), 2) for z, x in enumerate(D[m]['Rz_gw']) if D[m]['usable'][z] and z != s['zs']]})  "
              f"rho_all={s['rho']:.2f} (p={s['p']:.1e})  rho_by_contrast={pc}  rho_vs_state_expected_R={s['rho_exp']:.2f}")


if __name__ == "__main__":
    main()
