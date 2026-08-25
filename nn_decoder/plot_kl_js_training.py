# -*- coding: utf-8 -*-
"""Plot training dynamics from the KL/JS sweep checkpoints: train-vs-validation
loss across epochs (with the early-stop point marked) and weight evolution.

REQUIRES the sweep to have been run with history tracking ON, i.e.:

    python run_kl_js_entropy_sweep.py --track-history --snapshot-every 10 ...

That writes per-epoch curves + (optionally) periodic weight snapshots into the
per-mouse checkpoint files:

    results/<run>/lam<λ>/<slug>/checkpoints/mouse_<id>_<split>.pt
        -> { 'temp': {... 'history': {...}}, 'spat': {... 'history': {...}} }

history keys (per arch, when tracking was on):
    train_fit_loss, train_total_loss, train_entropy_pen   (per epoch)
    val_fit_loss,   val_total_loss                        (per epoch, if val set)
    weight_norms        (per epoch: list of per-parameter L2 norms)
    snapshot_epochs, state_dicts   (weight snapshots every N epochs)
    early_stopped_epoch, best_epoch, epoch_cap, patience  (scalars; new)

If no history is present (sweep run without --track-history), this script says
so and exits — it cannot reconstruct curves that were never saved.

Self-contained: numpy / matplotlib / torch only (torch is needed to load .pt).

Run:
    python plot_kl_js_training.py
    python plot_kl_js_training.py --run-name kl_js_entropy_sweep_v1 \
        --split stratified_balanced --arch temp --mouse 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figsave import save_fig

SLUG_RE = re.compile(
    r"^(?P<target>[A-Za-z]+)_(?P<loss>PCA|MSE|CE|KL|JS|Wasserstein)"
    r"_(?P<window>full|half|last_quarter)_(?P<bin>\d+)ms")
LAM_RE = re.compile(r"^lam(?P<lam>[0-9.eE+-]+)$")
LOSS_COLORS = {"KL": "#1f77b4", "JS": "#2ca02c", "PCA": "#d62728",
               "MSE": "#9467bd", "CE": "#8c564b", "Wasserstein": "#ff7f0e"}


def discover_ckpts(run_dir: Path, split: str, mouse: int):
    """Yield (lam, target, loss, window, bin_ms, ckpt_path) for each cell that
    has a checkpoint file for the given mouse+split."""
    for lam_dir in sorted(run_dir.glob("lam*")):
        lm = LAM_RE.match(lam_dir.name)
        lam = float(lm.group("lam")) if lm else float("nan")
        for slug_dir in sorted(p for p in lam_dir.iterdir() if p.is_dir()):
            sm = SLUG_RE.match(slug_dir.name)
            if not sm:
                continue
            ck = slug_dir / "checkpoints" / f"mouse_{mouse}_{split}.pt"
            if ck.exists():
                yield (lam, sm.group("target"), sm.group("loss"),
                       sm.group("window"), int(sm.group("bin")), ck)


def load_history(ckpt_path: Path, arch: str):
    """Load one arch's history dict from a .pt checkpoint. Returns None if the
    file has no history (tracking was off) or no such arch."""
    import torch
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or arch not in ck:
        return None
    hist = ck[arch].get("history") if isinstance(ck[arch], dict) else None
    if not hist or "train_fit_loss" not in hist or not len(hist["train_fit_loss"]):
        return None
    return hist


def collect(run_dir: Path, split: str, arch: str, mouse: int):
    recs = []
    n_seen, n_hist = 0, 0
    for lam, target, loss, window, bin_ms, ck in discover_ckpts(run_dir, split, mouse):
        n_seen += 1
        hist = load_history(ck, arch)
        if hist is None:
            continue
        n_hist += 1
        recs.append(dict(lam=lam, target=target, loss=loss, window=window,
                         bin_ms=bin_ms, slug=ck.parent.parent.name, hist=hist))
        print(f"  history: {ck.parent.parent.name} (λ={lam:g}) "
              f"epochs={len(hist['train_fit_loss'])}")
    if n_seen and not n_hist:
        print(f"\n  Found {n_seen} checkpoint(s) but NONE contain a training "
              f"history.\n  -> The sweep was run WITHOUT --track-history, so "
              f"per-epoch curves\n     and weight snapshots were never saved. "
              f"Re-run with:\n       python run_kl_js_entropy_sweep.py "
              f"--track-history --snapshot-every 10 ...")
    return recs


# ----------------------------------------------------------------------
# Figure A — train vs validation loss across epochs, early-stop marked
# ----------------------------------------------------------------------
def fig_training_curves(recs, arch, out_dir, mouse):
    if not recs:
        return None
    # one panel per (target, loss) config at the preferred 100ms/half cell,
    # lines coloured by lambda. Keeps it readable.
    targets = sorted({r["target"] for r in recs})
    losses = sorted({r["loss"] for r in recs})
    fig, axes = plt.subplots(len(targets), len(losses),
                             figsize=(5.5 * len(losses), 4 * len(targets)),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ti, t in enumerate(targets):
        for li, loss in enumerate(losses):
            ax = axes[ti][li]
            sub = [r for r in recs if r["target"] == t and r["loss"] == loss]
            # prefer 100ms/half; fall back to whatever exists
            pref = [r for r in sub if r["bin_ms"] == 100 and r["window"] == "half"]
            pool = pref or sub
            lams = sorted({r["lam"] for r in pool})
            for r in sorted(pool, key=lambda r: r["lam"]):
                h = r["hist"]
                tr = np.asarray(h["train_fit_loss"], float)
                ep = np.arange(len(tr))
                frac = (lams.index(r["lam"]) / max(1, len(lams) - 1)) if len(lams) > 1 else 0.5
                col = cmap(frac)
                ax.plot(ep, tr, color=col, lw=1.6,
                        label=f"λ={r['lam']:g} train")
                if "val_fit_loss" in h and len(h["val_fit_loss"]):
                    val = np.asarray(h["val_fit_loss"], float)
                    ax.plot(np.arange(len(val)), val, color=col, lw=1.4, ls="--")
                # early-stop / best markers (scalars added by fit_model)
                if "best_epoch" in h and h["best_epoch"] >= 0:
                    be = int(h["best_epoch"])
                    yv = (np.asarray(h.get("val_fit_loss", tr), float)[be]
                          if len(h.get("val_fit_loss", [])) > be else tr[min(be, len(tr) - 1)])
                    ax.scatter([be], [yv], color=col, marker="*", s=90, zorder=5)
                if "early_stopped_epoch" in h:
                    ax.axvline(int(h["early_stopped_epoch"]), color=col,
                               lw=0.8, alpha=0.4)
            ax.set_title(f"{t} | {loss}  (mouse {mouse})", fontsize=9)
            ax.set_xlabel("epoch"); ax.set_ylabel("fit loss")
            ax.legend(fontsize=6, title="solid=train  dashed=val\n★=best  |=stop",
                      title_fontsize=6)
    fig.suptitle(f"Training vs validation loss across epochs — {arch} arch "
                 "(★ = restored-best epoch, vertical line = early stop)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"A_training_curves_{arch}_mouse{mouse}"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


# ----------------------------------------------------------------------
# Figure B — weight evolution
# ----------------------------------------------------------------------
def fig_weight_evolution(recs, arch, out_dir, mouse):
    have = [r for r in recs if r["hist"].get("weight_norms")]
    if not have:
        print(f"  [skip] weight evolution ({arch}) — no weight_norms recorded")
        return None
    targets = sorted({r["target"] for r in have})
    losses = sorted({r["loss"] for r in have})
    fig, axes = plt.subplots(len(targets), len(losses),
                             figsize=(5.5 * len(losses), 4 * len(targets)),
                             squeeze=False)
    for ti, t in enumerate(targets):
        for li, loss in enumerate(losses):
            ax = axes[ti][li]
            sub = [r for r in have if r["target"] == t and r["loss"] == loss]
            pref = [r for r in sub if r["bin_ms"] == 100 and r["window"] == "half"]
            pool = pref or sub
            # use the middle-lambda cell; plot per-parameter L2 norm over epochs
            lams = sorted({r["lam"] for r in pool})
            r = [x for x in pool if x["lam"] == lams[len(lams) // 2]][0]
            wn = np.asarray(r["hist"]["weight_norms"], float)  # (epochs, n_params)
            if wn.ndim == 2:
                for pi in range(wn.shape[1]):
                    ax.plot(np.arange(wn.shape[0]), wn[:, pi], lw=1.4,
                            label=f"param {pi}")
            if "early_stopped_epoch" in r["hist"]:
                ax.axvline(int(r["hist"]["early_stopped_epoch"]), color="0.4",
                           ls=":", lw=1)
            ax.set_title(f"{t} | {loss}  λ={r['lam']:g}", fontsize=9)
            ax.set_xlabel("epoch"); ax.set_ylabel("per-parameter L2 norm")
            ax.legend(fontsize=6)
    fig.suptitle(f"Weight evolution (per-parameter L2 norm vs epoch) — {arch} "
                 "arch (dotted = early stop)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"B_weight_evolution_{arch}_mouse{mouse}"
    save_fig(fig, out_dir, stem, layout=None)
    return out_dir / f"{stem}.png"


def main(run_name, split, arch, mouse, results_root, out_dir):
    run_dir = Path(results_root) / run_name
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir} (rsync results first).")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {run_dir} for {arch} histories (split={split}, mouse={mouse}) ...")
    recs = collect(run_dir, split, arch, mouse)
    if not recs:
        raise SystemExit("No training histories found — see message above.")
    print(f"Loaded {len(recs)} history record(s). Writing to {out_dir} ...")
    for fn in (fig_training_curves, fig_weight_evolution):
        p = fn(recs, arch, out_dir, mouse)
        if p:
            print(f"  wrote {p.name}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-name", default="kl_js_entropy_sweep_v1")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--arch", default="temp", choices=["temp", "spat"])
    ap.add_argument("--mouse", type=int, default=0)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or f"figures/kl_js_sweep/{args.run_name}/training"
    main(args.run_name, args.split, args.arch, args.mouse,
         args.results_root, out_dir)
