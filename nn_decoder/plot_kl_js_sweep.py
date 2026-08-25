# -*- coding: utf-8 -*-
"""Plot the KL/JS entropy-lambda sweep: posterior peakiness, knob sweeps,
example posteriors, and held-out fit-loss.

Self-contained — depends only on numpy / scipy / matplotlib (no repo-internal
imports), so it runs on a laptop after rsync'ing the results down from the
cluster. It DISCOVERS whatever .mat files exist under the run tree, so it works
whether all 48 configs completed or only a subset.

Expected layout (produced by run_kl_js_entropy_sweep.py):

    results/<run_name>/lam<λ>/<target>_<loss>_<window>_<bin>ms/<split>.mat

e.g. results/kl_js_entropy_sweep_v1/lam3e-03/Q_KL_half_100ms/stratified_balanced.mat

Each .mat has {'results': {mouse_<id>: {...}}, 'config': {...}}. Per mouse:
  Dist['temp']['decoded'] / ['spat']['decoded']  (n_trials, n_cats)  posteriors
  Dist['temp']['target']  / ['spat']['target']   (n_trials, n_cats)  IO targets
  fit_loss['temp'] / ['spat']                     (n_trials,)         held-out loss

Peakiness metrics (matching plot_loss_sweep.py):
  - max-prob  : max probability per trial (1 = delta spike, 1/n_cats = uniform)
  - norm-H    : Shannon entropy / log(n_cats), per trial (0 = delta, 1 = uniform)

Run:
    python plot_kl_js_sweep.py
    python plot_kl_js_sweep.py --run-name kl_js_entropy_sweep_v1 \
        --split stratified_balanced --arch temp --out-dir figures/kl_js_sweep
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figsave import save_fig


# ----------------------------------------------------------------------
# Metrics (row-wise over a (n_trials, n_cats) probability array)
# ----------------------------------------------------------------------
def _renorm(P):
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[None, :]
    row = P.sum(axis=-1, keepdims=True)
    return P / np.where(row > 0, row, 1.0)


def norm_entropy(P):
    """H(p)/log(n_cats) per row. 0 = delta spike, 1 = uniform."""
    P = _renorm(P)
    H = -np.sum(P * np.log(P + 1e-12), axis=-1)
    return H / np.log(P.shape[-1])


def max_prob(P):
    """Max probability per row — the 'how peaky' scalar."""
    return _renorm(P).max(axis=-1)


# ----------------------------------------------------------------------
# Discovery + loading
# ----------------------------------------------------------------------
SLUG_RE = re.compile(
    r"^(?P<target>[A-Za-z]+)_(?P<loss>PCA|MSE|CE|KL|JS|Wasserstein)"
    r"_(?P<window>full|half|last_quarter)_(?P<bin>\d+)ms")
LAM_RE = re.compile(r"^lam(?P<lam>[0-9.eE+-]+)$")


def discover(run_dir: Path, split: str):
    """Yield (lam, target, loss, window, bin_ms, mat_path) for every cell .mat
    found under run_dir/lam*/<slug>/<split>.mat."""
    for lam_dir in sorted(run_dir.glob("lam*")):
        lm = LAM_RE.match(lam_dir.name)
        lam = float(lm.group("lam")) if lm else float("nan")
        for slug_dir in sorted(lam_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            sm = SLUG_RE.match(slug_dir.name)
            if not sm:
                continue
            mat_path = slug_dir / f"{split}.mat"
            if mat_path.exists():
                yield (lam, sm.group("target"), sm.group("loss"),
                       sm.group("window"), int(sm.group("bin")), mat_path)


def _load_arch(dist, md, arch):
    """Stack one arch's decoded/target/decoded_samp across the single mouse's
    Dist. Returns (decoded, target, decoded_samp_or_None)."""
    if arch not in dist:
        return None
    a = dist[arch]
    dec = np.atleast_2d(np.asarray(a.get("decoded", [])))
    tgt = np.atleast_2d(np.asarray(a.get("target", [])))
    if dec.size == 0 or tgt.size == 0:
        return None
    samp = np.asarray(a.get("decoded_samp", []))
    # decoded_samp is (n_trials, n_cats, T) for temp; empty (0,n_cats,T) for spat.
    samp = samp if samp.ndim == 3 and samp.shape[0] == dec.shape[0] else None
    return dec, tgt, samp


def load_cell(mat_path: Path):
    """Load one cell's .mat -> per-mouse records carrying BOTH archs.

    Returns a list of per-mouse dicts:
      { mouse, temp:{decoded,target,samp,fit_loss}, spat:{...} }
    Trials are in the same seeded test-set order across every loss/config
    (test_size=0.5, random_state=42), so trial index `t` refers to the same
    physical trial in KL and JS cells of the same (target,window,bin) — that is
    what makes the matched-trial figure valid.
    """
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    results = mat.get("results", {})
    mice = []
    for mouse_key, md in results.items():
        if not isinstance(md, dict) or "Dist" not in md:
            continue
        dist = md["Dist"]
        fl = md.get("fit_loss", {})
        rec = {"mouse": mouse_key}
        ok = False
        for arch in ("temp", "spat"):
            loaded = _load_arch(dist, md, arch)
            if loaded is None:
                rec[arch] = None
                continue
            dec, tgt, samp = loaded
            floss = (np.ravel(np.asarray(fl[arch]))
                     if isinstance(fl, dict) and arch in fl else np.array([]))
            rec[arch] = {"decoded": dec, "target": tgt, "samp": samp,
                         "fit_loss": floss}
            ok = True
        if ok:
            mice.append(rec)
    return mice or None


def _stack(mice, arch, field):
    parts = [m[arch][field] for m in mice
             if m.get(arch) and m[arch].get(field) is not None
             and np.asarray(m[arch][field]).size]
    if not parts:
        return np.array([])
    return np.vstack(parts) if parts[0].ndim >= 2 else np.concatenate(parts)



def collect(run_dir: Path, split: str):
    """Build a list of per-cell records, each carrying BOTH archs + per-mouse
    records (for matched per-bin plots) and pooled metrics per arch."""
    cells = []
    for lam, target, loss, window, bin_ms, mat_path in discover(run_dir, split):
        mice = load_cell(mat_path)
        if mice is None:
            print(f"  [skip] {mat_path} — no usable Dist")
            continue
        rec = dict(lam=lam, target=target, loss=loss, window=window,
                   bin_ms=bin_ms, slug=mat_path.parent.name, mice=mice)
        for arch in ("temp", "spat"):
            dec = _stack(mice, arch, "decoded")
            tgt = _stack(mice, arch, "target")
            fl = _stack(mice, arch, "fit_loss")
            if dec.size:
                rec[arch] = dict(
                    decoded=dec, target=tgt, fit_loss=fl,
                    n_trials=dec.shape[0], n_cats=dec.shape[1],
                    dec_maxprob=max_prob(dec), dec_normH=norm_entropy(dec),
                    tgt_maxprob=max_prob(tgt), tgt_normH=norm_entropy(tgt))
            else:
                rec[arch] = None
        cells.append(rec)
        nt = rec["temp"]["n_trials"] if rec["temp"] else (
            rec["spat"]["n_trials"] if rec["spat"] else 0)
        has_samp = any(m.get("temp") and m["temp"].get("samp") is not None
                       for m in mice)
        print(f"  loaded {mat_path.parent.name} (λ={lam:g}) n_trials={nt} "
              f"per-bin={'yes' if has_samp else 'no'}")
    return cells


LOSS_COLORS = {"KL": "#1f77b4", "JS": "#2ca02c", "PCA": "#d62728",
               "MSE": "#9467bd", "CE": "#8c564b", "Wasserstein": "#ff7f0e"}


# ----------------------------------------------------------------------
# Figure 2 — sweep over knobs (lambda x bin x window)
# ----------------------------------------------------------------------
def fig_sweep_over_knobs(cells, arch, out_dir):
    has = [c for c in cells if c.get(arch)]
    if not has:
        print(f"  [skip] fig2 ({arch}) — no cells with this arch")
        return None
    targets = sorted({c["target"] for c in has})
    windows = sorted({c["window"] for c in has})
    bins = sorted({c["bin_ms"] for c in has})
    losses = sorted({c["loss"] for c in has})
    lams = sorted({c["lam"] for c in has})

    ncol = max(1, len(windows) * len(bins))
    fig, axes = plt.subplots(len(targets), ncol, figsize=(4.5 * ncol, 4 * len(targets)),
                             squeeze=False)
    tgt_ref = {}
    for t in targets:
        tt = np.concatenate([c[arch]["tgt_normH"] for c in has if c["target"] == t])
        tgt_ref[t] = float(np.mean(tt)) if tt.size else np.nan

    for ti, t in enumerate(targets):
        col = 0
        for win in windows:
            for bm in bins:
                ax = axes[ti][col]; col += 1
                for loss in losses:
                    xs, ys, es = [], [], []
                    for lam in lams:
                        sub = [c for c in has if c["target"] == t and c["loss"] == loss
                               and c["window"] == win and c["bin_ms"] == bm
                               and c["lam"] == lam]
                        if not sub:
                            continue
                        vals = np.concatenate([c[arch]["dec_normH"] for c in sub])
                        xs.append(lam); ys.append(vals.mean()); es.append(vals.std())
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3,
                                    color=LOSS_COLORS.get(loss), label=loss)
                if not np.isnan(tgt_ref[t]):
                    ax.axhline(tgt_ref[t], color="0.4", ls="--", lw=1.2,
                               label="target mean")
                ax.set_xscale("log")
                ax.set_xlabel("entropy_lambda"); ax.set_ylabel("mean decoded norm-H")
                ax.set_title(f"{t} | {win} {bm}ms", fontsize=9)
                ax.legend(fontsize=7)
    fig.suptitle(f"Decoded posterior entropy vs swept knobs — {arch} arch "
                 "(higher = less peaky; dashed = target)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"2_sweep_over_knobs_{arch}"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


# ----------------------------------------------------------------------
# Figure 3 — MATCHED example posteriors: same trial across KL/JS,
#            spatial and temporal side by side, with per-bin overlay.
# ----------------------------------------------------------------------
def _pick_matched_cells(cells, losses, target, window, bin_ms, lam):
    """Return {loss: cell} for one (target,window,bin,lam) config, only where
    every requested loss is present (so trials are matched across them)."""
    out = {}
    for loss in losses:
        sub = [c for c in cells if c["loss"] == loss and c["target"] == target
               and c["window"] == window and c["bin_ms"] == bin_ms
               and c["lam"] == lam]
        if sub:
            out[loss] = sub[0]
    return out


def _config_options(cells, losses):
    """All (target,window,bin,lam) configs where ALL `losses` exist — these are
    the configs in which trials can be matched across losses."""
    keys = defaultdict(set)
    for c in cells:
        keys[(c["target"], c["window"], c["bin_ms"], c["lam"])].add(c["loss"])
    return [k for k, ls in keys.items() if all(l in ls for l in losses)]


def _per_mouse_trial(cell, arch, global_tr):
    """Map a global (pooled) trial index back to (mouse_rec, local_idx) so we
    can fetch decoded_samp, which is stored per mouse."""
    off = 0
    for m in cell["mice"]:
        a = m.get(arch)
        if not a or not a.get("decoded", np.array([])).size:
            continue
        n = a["decoded"].shape[0]
        if global_tr < off + n:
            return m, global_tr - off
        off += n
    return None, None


def fig_matched_examples(cells, out_dir, target, lam, losses=("KL", "JS"),
                         window=None, bin_ms=None, n_examples=4, seed=0):
    """Matched example posteriors for ONE target at ONE entropy_lambda.

    Each trial is a row. Columns:
      [ spat decoded | temp time-avg | per-bin <loss1> | per-bin <loss2> | ... ]
    spat & temp overlay the matched losses (same trial, head-to-head); each
    per-bin column shows ONE loss's individual time-bin posteriors (faint) under
    their bold mean. y-limits are matched across each row so peakiness is
    visually comparable within a trial.
    """
    losses = [l for l in losses
              if any(c["loss"] == l and c["target"] == target and c["lam"] == lam
                     for c in cells)]
    if len(losses) < 1:
        print(f"  [skip] fig3 ({target} λ={lam:g}) — no requested losses")
        return None
    tcells = [c for c in cells if c["target"] == target and c["lam"] == lam]
    opts = [k for k in _config_options(tcells, losses)]   # (target,win,bin,lam)
    if window is not None and bin_ms is not None:
        # Caller pinned a (window, bin) — honour it exactly; skip if absent
        # (no silent fallback to a different window, which would mislabel).
        opts = [k for k in opts if k[1] == window and k[2] == bin_ms]
        if not opts:
            return None
    if not opts:
        opts = _config_options(tcells, losses[:1])
        if not opts:
            print(f"  [skip] fig3 ({target} λ={lam:g}) — no shared config")
            return None
    opts.sort(key=lambda k: (k[2] == 100, k[1] == "half"), reverse=True)
    _t, window, bin_ms, _lam = opts[0]
    chosen = _pick_matched_cells(tcells, losses, target, window, bin_ms, lam)

    ref = next(iter(chosen.values()))
    ref_arch = "temp" if ref.get("temp") else "spat"
    n_cats = ref[ref_arch]["n_cats"]
    n_trials = ref[ref_arch]["n_trials"]
    rng = np.random.default_rng(seed)
    trials = rng.choice(n_trials, size=min(n_examples, n_trials), replace=False)
    x = np.arange(n_cats)

    ncol = 2 + len(losses)   # spat, temp-avg, then one per-bin col per loss
    fig, axes = plt.subplots(len(trials), ncol,
                             figsize=(3.4 * ncol, 2.8 * len(trials)),
                             squeeze=False)
    for ri, tr in enumerate(trials):
        row_max = 0.0   # track for shared y-lim across this row

        # col 0 — spat decoded (overlaid)
        ax = axes[ri][0]; drawn = False
        for loss, c in chosen.items():
            a = c.get("spat")
            if not a:
                continue
            if not drawn:
                ax.fill_between(x, a["target"][tr], color="0.85", label="target")
                row_max = max(row_max, a["target"][tr].max())
                drawn = True
            ax.plot(x, a["decoded"][tr], color=LOSS_COLORS.get(loss), lw=1.6,
                    label=f"{loss} H={a['dec_normH'][tr]:.2f}")
            row_max = max(row_max, a["decoded"][tr].max())
        ax.set_title(f"Spatial — trial {tr}", fontsize=8)
        ax.set_ylabel("prob")
        if ri == 0:
            ax.legend(fontsize=6)

        # col 1 — temp time-avg decoded (overlaid)
        ax = axes[ri][1]; drawn = False
        for loss, c in chosen.items():
            a = c.get("temp")
            if not a:
                continue
            if not drawn:
                ax.fill_between(x, a["target"][tr], color="0.85", label="target")
                row_max = max(row_max, a["target"][tr].max())
                drawn = True
            ax.plot(x, a["decoded"][tr], color=LOSS_COLORS.get(loss), lw=1.6,
                    label=f"{loss} H={a['dec_normH'][tr]:.2f}")
            row_max = max(row_max, a["decoded"][tr].max())
        ax.set_title(f"Temporal time-avg — trial {tr}", fontsize=8)
        if ri == 0:
            ax.legend(fontsize=6)

        # cols 2.. — per-bin spread, ONE loss per column
        for ci, loss in enumerate(losses):
            ax = axes[ri][2 + ci]
            c = chosen.get(loss)
            m, li = _per_mouse_trial(c, "temp", tr) if c else (None, None)
            samp = m["temp"]["samp"] if (m and m.get("temp")) else None
            if samp is None:
                ax.text(0.5, 0.5, f"{loss}\nno per-bin data", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8,
                        color="0.5")
            else:
                perbin = samp[li]                      # (n_cats, T)
                T = perbin.shape[1]
                ax.fill_between(x, c["temp"]["target"][tr], color="0.9", zorder=0)
                for t in range(T):
                    ax.plot(x, perbin[:, t], color=LOSS_COLORS.get(loss),
                            alpha=0.30, lw=0.8)
                ax.plot(x, perbin.mean(axis=1), color=LOSS_COLORS.get(loss),
                        lw=2.0, label=f"mean of {T} bins")
                row_max = max(row_max, perbin.max())
                if ri == 0:
                    ax.legend(fontsize=6)
            ax.set_title(f"{loss} per-bin — trial {tr}", fontsize=8)

        # --- match y-limits across the whole row ---
        for a_ in axes[ri]:
            a_.set_ylim(0, row_max * 1.08 if row_max > 0 else 1)
            a_.set_xlabel("angle bin")

    long = {"Q": "perceptual posterior Q(θ)", "L": "likelihood L(θ)"}.get(
        target, target)
    fig.suptitle(
        f"Matched example posteriors — {long}  |  {window} {bin_ms}ms | "
        f"λ={lam:g}  (same trials; spat | temp-avg | per-bin per loss; "
        f"y-lim matched per row)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    stem = f"3_matched_examples_{target}_{window}_{bin_ms}ms_lam{lam:.0e}"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


# ----------------------------------------------------------------------
# Figure 6 — per-bin posteriors compared ACROSS lambda, matched trials.
# ----------------------------------------------------------------------
def fig_perbin_vs_lambda(cells, out_dir, target, loss,
                         window="half", bin_ms=100, n_examples=4, seed=0):
    """How the temp per-bin posteriors sharpen as entropy_lambda increases, for
    MATCHED trials. Rows = trials (same across all λ); columns = λ values. Each
    panel draws that trial's individual time-bin posteriors (faint) + their mean
    (bold) at that λ, over the broad target. Only meaningful for temporal."""
    sub = [c for c in cells if c["target"] == target and c["loss"] == loss
           and c["window"] == window and c["bin_ms"] == bin_ms and c.get("temp")]
    # keep only cells that actually carry per-bin samples
    sub = [c for c in sub
           if any(m.get("temp") and m["temp"].get("samp") is not None
                  for m in c["mice"])]
    lams = sorted({c["lam"] for c in sub})
    if len(lams) < 2:
        print(f"  [skip] fig6 ({target} {loss} {window} {bin_ms}ms) — need >=2 "
              f"lambdas with per-bin data (have {len(lams)})")
        return None
    by_lam = {lam: next(c for c in sub if c["lam"] == lam) for lam in lams}

    ref = by_lam[lams[0]]["temp"]
    n_cats, n_trials = ref["n_cats"], ref["n_trials"]
    rng = np.random.default_rng(seed)
    trials = rng.choice(n_trials, size=min(n_examples, n_trials), replace=False)
    x = np.arange(n_cats)
    col = LOSS_COLORS.get(loss)

    def _normH(p):
        p = p / max(p.sum(), 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum() / np.log(n_cats))

    # Pre-fetch per-(trial,lam) per-bin arrays so we can two-pass for ylim.
    perbin_cache = {}   # (tr, lam) -> (n_cats, T) or None
    for tr in trials:
        for lam in lams:
            m, li = _per_mouse_trial(by_lam[lam], "temp", tr)
            samp = m["temp"]["samp"] if (m and m.get("temp")) else None
            perbin_cache[(tr, lam)] = samp[li] if samp is not None else None

    fig, axes = plt.subplots(len(trials), len(lams),
                             figsize=(3.4 * len(lams), 2.8 * len(trials)),
                             squeeze=False, sharex=True)
    for ri, tr in enumerate(trials):
        # shared y-lim for this trial's row (across all lambdas)
        row_max = max([by_lam[lams[0]]["temp"]["target"][tr].max()]
                      + [perbin_cache[(tr, lam)].max()
                         for lam in lams if perbin_cache[(tr, lam)] is not None]
                      or [1.0])
        for ci, lam in enumerate(lams):
            ax = axes[ri][ci]
            c = by_lam[lam]
            ax.fill_between(x, c["temp"]["target"][tr], color="0.9", zorder=0)
            perbin = perbin_cache[(tr, lam)]
            if perbin is None:
                ax.text(0.5, 0.5, "no per-bin", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
            else:
                T = perbin.shape[1]
                for t in range(T):
                    ax.plot(x, perbin[:, t], color=col, alpha=0.30, lw=0.8)
                mean = perbin.mean(axis=1)
                ax.plot(x, mean, color=col, lw=2.0)
                # mean per-BIN entropy (avg over time bins) AND entropy of the
                # time-averaged posterior — both annotated.
                perbin_H = float(np.mean([_normH(perbin[:, t]) for t in range(T)]))
                avg_H = _normH(mean)
                ax.text(0.02, 0.95,
                        f"per-bin H̄={perbin_H:.2f}\navg H={avg_H:.2f}",
                        transform=ax.transAxes, fontsize=6.5, va="top")
            ax.set_ylim(0, row_max * 1.08)
            if ri == 0:
                ax.set_title(f"λ = {lam:g}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"trial {tr}\nprob", fontsize=8)
            if ri == len(trials) - 1:
                ax.set_xlabel("angle bin")

    long = {"Q": "perceptual posterior Q(θ)", "L": "likelihood L(θ)"}.get(
        target, target)
    fig.suptitle(
        f"Per-bin posteriors vs entropy_lambda — {long} | {loss} | "
        f"{window} {bin_ms}ms  (matched trials; faint=time-bins, bold=mean; "
        f"y-lim matched per row)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"6_perbin_vs_lambda_{target}_{loss}_{window}_{bin_ms}ms"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


# ----------------------------------------------------------------------
# Figure 7 — average per-bin entropy vs lambda (the #3 summary).
# ----------------------------------------------------------------------
def fig_avg_perbin_entropy_vs_lambda(cells, out_dir, window="half", bin_ms=100):
    """Mean per-bin posterior entropy (normalised, averaged over time bins and
    pooled over all trials/mice) as a function of entropy_lambda, one line per
    (target, loss). Shows directly how raising lambda sharpens the per-bin
    posteriors. temporal only (per-bin data does not exist for spatial)."""
    def _rowH(P):                      # P: (..., n_cats) -> normalised entropy
        P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
        return -np.sum(P * np.log(P + 1e-12), -1) / np.log(P.shape[-1])

    targets = sorted({c["target"] for c in cells})
    losses = sorted({c["loss"] for c in cells})
    rows = {}   # (target, loss) -> {lam: mean_perbin_H}
    for c in cells:
        if c["window"] != window or c["bin_ms"] != bin_ms or not c.get("temp"):
            continue
        vals = []
        for m in c["mice"]:
            samp = m.get("temp", {}).get("samp") if m.get("temp") else None
            if samp is None:
                continue
            # samp: (n_trials, n_cats, T) -> entropy per (trial,bin), then mean
            H = _rowH(np.transpose(samp, (0, 2, 1)))   # (n_trials, T)
            vals.append(H.ravel())
        if vals:
            rows.setdefault((c["target"], c["loss"]), {})[c["lam"]] = \
                float(np.concatenate(vals).mean())
    if not rows:
        print(f"  [skip] fig7 — no temp per-bin data at {window} {bin_ms}ms")
        return None

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5),
                             squeeze=False)
    for ti, t in enumerate(targets):
        ax = axes[0][ti]
        for loss in losses:
            d = rows.get((t, loss))
            if not d:
                continue
            xs = sorted(d)
            ax.plot(xs, [d[l] for l in xs], marker="o", ms=5,
                    color=LOSS_COLORS.get(loss), label=loss)
        ax.set_xscale("log")
        ax.set_xlabel("entropy_lambda")
        ax.set_ylabel("mean per-bin posterior entropy (normalised)")
        ax.set_title(f"{t} — temporal ({window} {bin_ms}ms)", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Average per-bin posterior entropy vs entropy_lambda "
                 "(higher λ -> sharper per-bin posteriors -> lower entropy)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = f"7_avg_perbin_entropy_vs_lambda_{window}_{bin_ms}ms"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


# ----------------------------------------------------------------------
# Figure 4 — held-out fit-loss; spat AND temp in the same subplot per target.
# ----------------------------------------------------------------------
def fig_fit_loss(cells, out_dir):
    """One subplot per target. Each loss is a colour; spat (solid) and temp
    (dashed) are overlaid in the SAME axes so the two architectures compare
    directly. y-lim is shared across all target subplots."""
    have = [c for c in cells
            if (c.get("spat") and c["spat"]["fit_loss"].size)
            or (c.get("temp") and c["temp"]["fit_loss"].size)]
    if not have:
        print("  [skip] fig4 — no fit_loss arrays present")
        return None
    targets = sorted({c["target"] for c in have})
    losses = sorted({c["loss"] for c in have})
    lams = sorted({c["lam"] for c in have})
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5),
                             squeeze=False, sharey=True)
    arch_style = {"spat": dict(ls="-", marker="o"),
                  "temp": dict(ls="--", marker="s")}
    for ti, t in enumerate(targets):
        ax = axes[0][ti]
        for loss in losses:
            for arch in ("spat", "temp"):
                xs, ys, es = [], [], []
                for lam in lams:
                    sub = [c for c in have if c["target"] == t
                           and c["loss"] == loss and c["lam"] == lam
                           and c.get(arch) and c[arch]["fit_loss"].size]
                    if not sub:
                        continue
                    v = np.concatenate([c[arch]["fit_loss"] for c in sub])
                    xs.append(lam); ys.append(v.mean())
                    es.append(v.std() / np.sqrt(len(v)))
                if xs:
                    ax.errorbar(xs, ys, yerr=es, capsize=3, ms=4,
                                color=LOSS_COLORS.get(loss),
                                label=f"{loss} {arch}", **arch_style[arch])
        ax.set_xscale("log"); ax.set_xlabel("entropy_lambda")
        if ti == 0:
            ax.set_ylabel("mean held-out fit_loss (±SEM)")
        ax.set_title(t, fontsize=11); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Held-out fit-loss per config — spat (solid) vs temp (dashed), "
                 "shared y-axis  (pooled over bin/window)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = "4_fit_loss"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


# ----------------------------------------------------------------------
# Figure 5 — decoded peakiness vs target: rows = metric (normH, maxprob),
#            columns = arch (spat | temp). Supersedes the old per-arch fig1.
# ----------------------------------------------------------------------
def fig_spat_temp_side_by_side(cells, out_dir):
    """Decoded-posterior peakiness distributions per loss vs the target.
    Rows are the two peakiness metrics (normalised entropy; max probability);
    columns are the two architectures (spatial | temporal) so both the
    metric-pair and the arch-pair are on one page."""
    archs = [a for a in ("spat", "temp") if any(c.get(a) for c in cells)]
    if not archs:
        print("  [skip] fig5 — no arch data")
        return None
    losses = sorted({c["loss"] for c in cells})
    metrics = [("normH", "decoded normalised entropy (0=spike, 1=uniform)"),
               ("maxprob", "decoded max probability (1=spike)")]
    fig, axes = plt.subplots(len(metrics), len(archs),
                             figsize=(6.5 * len(archs), 4.4 * len(metrics)),
                             squeeze=False)
    for mi, (metric, xlab) in enumerate(metrics):
        for ai, arch in enumerate(archs):
            ax = axes[mi][ai]
            has = [c for c in cells if c.get(arch)]
            for loss in losses:
                vals = [c[arch][f"dec_{metric}"] for c in has if c["loss"] == loss]
                if not vals:
                    continue
                v = np.concatenate(vals)
                ax.hist(v, bins=40, density=True, histtype="step", lw=2,
                        color=LOSS_COLORS.get(loss),
                        label=f"{loss} (mean {v.mean():.2f})")
            tgt = np.concatenate([c[arch][f"tgt_{metric}"] for c in has])
            ax.hist(tgt, bins=40, density=True, histtype="stepfilled", alpha=0.25,
                    color="0.5", label=f"target (mean {tgt.mean():.2f})")
            ax.set_xlabel(xlab); ax.set_ylabel("density")
            ax.set_title(f"{'spatial' if arch == 'spat' else 'temporal'}",
                         fontsize=11)
            ax.legend(fontsize=7)
    fig.suptitle("Decoded posterior peakiness vs target — "
                 "rows: entropy / max-prob,  columns: spatial / temporal")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = "5_peakiness_spat_vs_temp"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


def write_summary(cells, out_dir):
    import csv
    p = out_dir / "summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "loss", "window", "bin_ms", "lam", "arch",
                    "n_trials", "dec_normH_mean", "tgt_normH_mean",
                    "dec_maxprob_mean", "fit_loss_mean"])
        for c in sorted(cells, key=lambda c: (c["target"], c["loss"], c["bin_ms"],
                                              c["window"], c["lam"])):
            for arch in ("spat", "temp"):
                a = c.get(arch)
                if not a:
                    continue
                w.writerow([c["target"], c["loss"], c["window"], c["bin_ms"],
                            f"{c['lam']:g}", arch, a["n_trials"],
                            f"{a['dec_normH'].mean():.4f}",
                            f"{a['tgt_normH'].mean():.4f}",
                            f"{a['dec_maxprob'].mean():.4f}",
                            f"{a['fit_loss'].mean():.4f}" if a["fit_loss"].size else ""])
    return p


def main(run_name, split, results_root, out_dir, match_losses=("KL", "JS")):
    run_dir = Path(results_root) / run_name
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}\n"
                         f"(rsync the results down first.)")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {run_dir} (split={split}) ...")
    cells = collect(run_dir, split)
    if not cells:
        raise SystemExit("No cells loaded — check run-name/split and that "
                         "the .mat files were transferred.")
    print(f"Loaded {len(cells)} cell(s). Writing figures to {out_dir} ...")
    targets = sorted({c["target"] for c in cells})
    losses = sorted({c["loss"] for c in cells})
    lams = sorted({c["lam"] for c in cells})
    # (window, bin_ms) combos actually present — drives the per-window figures
    # so 'full' and 'half' are both plotted when both were run.
    wbins = sorted({(c["window"], c["bin_ms"]) for c in cells})

    # Knob sweeps (per arch). Peakiness distributions are in fig5 (both archs
    # + both metrics in one figure), which supersedes the old per-arch fig1.
    for arch in ("spat", "temp"):
        p = fig_sweep_over_knobs(cells, arch, out_dir)
        if p:
            print(f"  wrote {p.name}")
    # Fit-loss: spat + temp together in one figure.
    p = fig_fit_loss(cells, out_dir)
    if p:
        print(f"  wrote {p.name}")
    # Matched example posteriors — ONE figure per (target, window, bin, lambda).
    for tgt in targets:
        for (win, bm) in wbins:
            for lam in lams:
                p = fig_matched_examples(cells, out_dir, tgt, lam,
                                         losses=tuple(match_losses),
                                         window=win, bin_ms=bm)
                if p:
                    print(f"  wrote {p.name}")
    # Per-bin posteriors across lambda — one per (target, loss, window, bin).
    for tgt in targets:
        for loss in losses:
            for (win, bm) in wbins:
                p = fig_perbin_vs_lambda(cells, out_dir, tgt, loss,
                                         window=win, bin_ms=bm)
                if p:
                    print(f"  wrote {p.name}")
    # Average per-bin entropy vs lambda (summary) — one per (window, bin).
    for (win, bm) in wbins:
        p = fig_avg_perbin_entropy_vs_lambda(cells, out_dir, window=win, bin_ms=bm)
        if p:
            print(f"  wrote {p.name}")
    # Cross-arch summary.
    p = fig_spat_temp_side_by_side(cells, out_dir)
    if p:
        print(f"  wrote {p.name}")
    s = write_summary(cells, out_dir)
    print(f"  wrote {s.name}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-name", default="kl_js_entropy_sweep_v1")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--losses", nargs="+", default=["KL", "JS"],
                    help="Losses to overlay in the matched-example figure")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default=None,
                    help="default: figures/kl_js_sweep/<run-name>")
    args = ap.parse_args()
    out_dir = args.out_dir or f"figures/kl_js_sweep/{args.run_name}"
    main(args.run_name, args.split, args.results_root, out_dir,
         match_losses=args.losses)
