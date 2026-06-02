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


def load_cell(mat_path: Path, arch: str):
    """Load one cell's .mat -> dict with stacked decoded/target/fit_loss across
    mice. arch in {'temp','spat'}. Returns None if the file lacks the arch."""
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    results = mat.get("results", {})
    decoded, target, floss = [], [], []
    for mouse_key, md in results.items():
        if not isinstance(md, dict) or "Dist" not in md:
            continue
        dist = md["Dist"]
        if arch not in dist:
            continue
        a = dist[arch]
        dec = np.atleast_2d(np.asarray(a.get("decoded", [])))
        tgt = np.atleast_2d(np.asarray(a.get("target", [])))
        if dec.size == 0 or tgt.size == 0:
            continue
        decoded.append(dec)
        target.append(tgt)
        fl = md.get("fit_loss", {})
        if isinstance(fl, dict) and arch in fl:
            floss.append(np.ravel(np.asarray(fl[arch])))
    if not decoded:
        return None
    return {
        "decoded": np.vstack(decoded),
        "target": np.vstack(target),
        "fit_loss": np.concatenate(floss) if floss else np.array([]),
    }


def collect(run_dir: Path, split: str, arch: str):
    """Build a list of per-cell records with metrics computed."""
    cells = []
    for lam, target, loss, window, bin_ms, mat_path in discover(run_dir, split):
        data = load_cell(mat_path, arch)
        if data is None:
            print(f"  [skip] {mat_path} — no '{arch}' data")
            continue
        dec, tgt = data["decoded"], data["target"]
        rec = dict(
            lam=lam, target=target, loss=loss, window=window, bin_ms=bin_ms,
            slug=mat_path.parent.name, n_trials=dec.shape[0], n_cats=dec.shape[1],
            dec_maxprob=max_prob(dec), dec_normH=norm_entropy(dec),
            tgt_maxprob=max_prob(tgt), tgt_normH=norm_entropy(tgt),
            fit_loss=data["fit_loss"], decoded_arr=dec, target_arr=tgt,
        )
        cells.append(rec)
        print(f"  loaded {mat_path.parent.name} (λ={lam:g}) "
              f"n_trials={dec.shape[0]} n_cats={dec.shape[1]}")
    return cells


LOSS_COLORS = {"KL": "#1f77b4", "JS": "#2ca02c", "PCA": "#d62728",
               "MSE": "#9467bd", "CE": "#8c564b", "Wasserstein": "#ff7f0e"}


# ----------------------------------------------------------------------
# Figure 1 — peakiness of decoded posteriors vs their targets
# ----------------------------------------------------------------------
def fig_peakiness_vs_targets(cells, arch, out_dir):
    losses = sorted({c["loss"] for c in cells})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # pooled across all configs of each loss
    for metric, ax, lab in [("normH", axes[0], "normalised entropy  (0=spike, 1=uniform)"),
                            ("maxprob", axes[1], "max probability  (1=spike)")]:
        for loss in losses:
            dec = np.concatenate([c[f"dec_{metric}"] for c in cells if c["loss"] == loss])
            ax.hist(dec, bins=40, density=True, histtype="step", lw=2,
                    color=LOSS_COLORS.get(loss, None), label=f"{loss} decoded")
        # target reference (same regardless of loss) — pool all
        tgt = np.concatenate([c[f"tgt_{metric}"] for c in cells])
        ax.hist(tgt, bins=40, density=True, histtype="stepfilled", alpha=0.25,
                color="0.5", label="target")
        ax.set_xlabel(lab); ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle(f"Posterior peakiness vs targets — {arch} arch  "
                 f"(decoded should sit near the target if calibrated)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / f"1_peakiness_vs_targets_{arch}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------
# Figure 2 — sweep over knobs (lambda x bin x window)
# ----------------------------------------------------------------------
def fig_sweep_over_knobs(cells, arch, out_dir):
    targets = sorted({c["target"] for c in cells})
    windows = sorted({c["window"] for c in cells})
    bins = sorted({c["bin_ms"] for c in cells})
    losses = sorted({c["loss"] for c in cells})
    lams = sorted({c["lam"] for c in cells})

    ncol = max(1, len(windows) * len(bins))
    fig, axes = plt.subplots(len(targets), ncol, figsize=(4.5 * ncol, 4 * len(targets)),
                             squeeze=False)
    tgt_ref = {}  # per target, mean target normH
    for t in targets:
        tt = np.concatenate([c["tgt_normH"] for c in cells if c["target"] == t])
        tgt_ref[t] = float(np.mean(tt)) if tt.size else np.nan

    for ti, t in enumerate(targets):
        col = 0
        for win in windows:
            for bm in bins:
                ax = axes[ti][col]; col += 1
                for loss in losses:
                    xs, ys, es = [], [], []
                    for lam in lams:
                        sub = [c for c in cells if c["target"] == t and c["loss"] == loss
                               and c["window"] == win and c["bin_ms"] == bm
                               and c["lam"] == lam]
                        if not sub:
                            continue
                        vals = np.concatenate([c["dec_normH"] for c in sub])
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
    p = out_dir / f"2_sweep_over_knobs_{arch}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------
# Figure 3 — example decoded posteriors overlaid on targets
# ----------------------------------------------------------------------
def fig_example_posteriors(cells, arch, out_dir, n_examples=4, seed=0):
    # one row per loss; pick a representative config (prefer 100ms/half, mid lambda)
    losses = sorted({c["loss"] for c in cells})
    rng = np.random.default_rng(seed)

    def pick_cell(loss):
        cand = [c for c in cells if c["loss"] == loss]
        pref = [c for c in cand if c["bin_ms"] == 100 and c["window"] == "half"]
        pool = pref or cand
        lams = sorted({c["lam"] for c in pool})
        midlam = lams[len(lams) // 2]
        best = [c for c in pool if c["lam"] == midlam]
        return best[0] if best else pool[0]

    fig, axes = plt.subplots(len(losses), n_examples,
                             figsize=(3.2 * n_examples, 2.8 * len(losses)),
                             squeeze=False)
    for li, loss in enumerate(losses):
        c = pick_cell(loss)
        idx = rng.choice(c["decoded_arr"].shape[0],
                         size=min(n_examples, c["decoded_arr"].shape[0]), replace=False)
        x = np.arange(c["n_cats"])
        for j, tr in enumerate(idx):
            ax = axes[li][j]
            ax.fill_between(x, c["target_arr"][tr], color="0.8", label="target")
            ax.plot(x, c["decoded_arr"][tr], color=LOSS_COLORS.get(loss), lw=1.6,
                    label="decoded")
            ax.set_title(f"{loss} | {c['slug']} λ={c['lam']:g}\n"
                         f"trial {tr}: H={c['dec_normH'][tr]:.2f} "
                         f"(tgt {c['tgt_normH'][tr]:.2f})", fontsize=7)
            if j == 0:
                ax.set_ylabel("prob"); ax.legend(fontsize=6)
            ax.set_xlabel("bin")
    fig.suptitle(f"Example decoded posteriors vs targets — {arch} arch")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"3_example_posteriors_{arch}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------
# Figure 4 — held-out fit-loss per config
# ----------------------------------------------------------------------
def fig_fit_loss(cells, arch, out_dir):
    have = [c for c in cells if c["fit_loss"].size]
    if not have:
        print("  [skip] fig4 — no fit_loss arrays present")
        return None
    targets = sorted({c["target"] for c in have})
    losses = sorted({c["loss"] for c in have})
    lams = sorted({c["lam"] for c in have})
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5),
                             squeeze=False)
    for ti, t in enumerate(targets):
        ax = axes[0][ti]
        for loss in losses:
            xs, ys, es = [], [], []
            for lam in lams:
                sub = [c for c in have if c["target"] == t and c["loss"] == loss
                       and c["lam"] == lam]
                if not sub:
                    continue
                v = np.concatenate([c["fit_loss"] for c in sub])
                xs.append(lam); ys.append(v.mean()); es.append(v.std() / np.sqrt(len(v)))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3,
                            color=LOSS_COLORS.get(loss), label=loss)
        ax.set_xscale("log"); ax.set_xlabel("entropy_lambda")
        ax.set_ylabel("mean held-out fit_loss (±SEM)")
        ax.set_title(f"{t} — {arch}", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(f"Held-out fit-loss per config — {arch} arch "
                 "(pooled over bin/window)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / f"4_fit_loss_{arch}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def write_summary(cells, arch, out_dir):
    import csv
    p = out_dir / f"summary_{arch}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "loss", "window", "bin_ms", "lam", "n_trials",
                    "dec_normH_mean", "tgt_normH_mean", "dec_maxprob_mean",
                    "fit_loss_mean"])
        for c in sorted(cells, key=lambda c: (c["target"], c["loss"], c["bin_ms"],
                                              c["window"], c["lam"])):
            w.writerow([c["target"], c["loss"], c["window"], c["bin_ms"],
                        f"{c['lam']:g}", c["n_trials"],
                        f"{c['dec_normH'].mean():.4f}",
                        f"{c['tgt_normH'].mean():.4f}",
                        f"{c['dec_maxprob'].mean():.4f}",
                        f"{c['fit_loss'].mean():.4f}" if c["fit_loss"].size else ""])
    return p


def main(run_name, split, arch, results_root, out_dir):
    run_dir = Path(results_root) / run_name
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}\n"
                         f"(rsync the results down first.)")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {run_dir} (split={split}, arch={arch}) ...")
    cells = collect(run_dir, split, arch)
    if not cells:
        raise SystemExit("No cells loaded — check run-name/split/arch and that "
                         "the .mat files were transferred.")
    print(f"Loaded {len(cells)} cell(s). Writing figures to {out_dir} ...")
    for fn in (fig_peakiness_vs_targets, fig_sweep_over_knobs,
               fig_example_posteriors, fig_fit_loss):
        p = fn(cells, arch, out_dir)
        if p:
            print(f"  wrote {p.name}")
    s = write_summary(cells, arch, out_dir)
    print(f"  wrote {s.name}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-name", default="kl_js_entropy_sweep_v1")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--arch", default="temp", choices=["temp", "spat"],
                    help="temp = sampling/SBC, spat = PPC")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default=None,
                    help="default: figures/kl_js_sweep/<run-name>")
    args = ap.parse_args()
    out_dir = args.out_dir or f"figures/kl_js_sweep/{args.run_name}"
    main(args.run_name, args.split, args.arch, args.results_root, out_dir)
