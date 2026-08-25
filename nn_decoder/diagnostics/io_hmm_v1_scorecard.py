"""Interim scorecard for the io_hmm_v1 sweep (IO-HMM 72-bin targets).

Per cell x architecture: decoded-vs-target peakiness (max-prob and entropy
ratios), mean KL(decoded||target), and two references — the cell's own shuffle
decoder and a predict-mean proxy (KL of the split-mean target against each
trial's target; the strict predict-mean null uses the train-half mean, this
uses the same-split mean, close at n=470).

Projection-loss evaluation (the second judging metric) needs the PCA basis and
is left to the full cross-loss analysis; this scorecard settles peakiness and
KL-side calibration only.

Usage: python diagnostics/io_hmm_v1_scorecard.py [--root results/io_hmm_v1]
                                                 [--mouse 0]
Figure -> figures/io_hmm_v1/scorecard_m<mouse>.{png,svg}; prints a summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from figsave import save_fig                   # noqa: E402
from decoder_metrics import kl_rows           # noqa: E402  (canonical KL — audit D1)

ARCHS = ("spat", "temp")
LOSS_ORDER = ("pca", "pcaflat", "kl", "js")
LAMBDA_ORDER = ("lh0", "lh1e-4", "lh3e-4", "lh1e-3", "lh3e-3")


def _entropy(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(p * np.log(np.clip(p, 1e-300, None))).sum(1)


def score_cell(mat_path, mouse):
    m = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    r = getattr(m["results"], f"mouse_{mouse}", None)
    if r is None:
        return None
    out = {}
    for arch in ARCHS:
        a = getattr(r.Dist, arch)
        shf = getattr(r.Dist, f"{arch}_shf")
        dec, tgt = np.asarray(a.decoded), np.asarray(a.target)
        mean_tgt = tgt.mean(0, keepdims=True)
        out[arch] = {
            "peak_ratio": float(dec.max(1).mean() / tgt.max(1).mean()),
            "entropy_gap": float((_entropy(tgt) - _entropy(dec)).mean()),
            "kl": float(kl_rows(dec, tgt).mean()),
            "kl_shuffle": float(kl_rows(np.asarray(shf.decoded), np.asarray(shf.target)).mean()),
            "kl_predmean": float(kl_rows(np.broadcast_to(mean_tgt, tgt.shape), tgt).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_HERE.parent / "results" / "io_hmm_v1")
    ap.add_argument("--mouse", type=int, default=0)
    args = ap.parse_args()

    cells = {}
    for shard in sorted(args.root.glob("*/Q_*/stratified_balanced.mat")):
        cell = shard.parent.parent.name
        s = score_cell(shard, args.mouse)
        if s is not None:
            cells[cell] = s
    if not cells:
        sys.exit(f"no scored cells under {args.root} for mouse {args.mouse}")

    print(f"{len(cells)} cells scored (mouse {args.mouse})")
    hdr = (f"{'cell':24s} " + "  ".join(
        f"{a}: peakR  KL  (shf/predM)" for a in ARCHS))
    print(hdr)
    for cell in sorted(cells):
        s = cells[cell]
        row = f"{cell:24s} "
        for a in ARCHS:
            v = s[a]
            row += (f"  {v['peak_ratio']:5.2f}  {v['kl']:6.3f} "
                    f"({v['kl_shuffle']:5.3f}/{v['kl_predmean']:5.3f})")
        print(row)

    # figure: one row per arch; x = cells grouped by loss family; two axes —
    # peakiness ratio (1 = calibrated) and KL vs the predict-mean reference
    def cell_sort_key(c):
        for i, l in enumerate(LOSS_ORDER):
            for j, lam in enumerate(LAMBDA_ORDER):
                if c.startswith(f"{l}_") and c.endswith(lam):
                    arch_j = 0 if "_h8_" in c else 1
                    return (i, arch_j, j)
        return (99, 99, 99)

    names = sorted(cells, key=cell_sort_key)
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for row, arch in enumerate(ARCHS):
        pk = [cells[c][arch]["peak_ratio"] for c in names]
        kl = [cells[c][arch]["kl"] for c in names]
        pm = [cells[c][arch]["kl_predmean"] for c in names]
        sh = [cells[c][arch]["kl_shuffle"] for c in names]
        ax = axes[row, 0]
        ax.bar(x, pk, color="tab:blue")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_ylabel(f"{arch}\npeakiness ratio (dec/tgt)")
        ax = axes[row, 1]
        ax.plot(x, kl, "o-", color="tab:red", label="KL(tgt||dec)")
        ax.plot(x, pm, "s--", color="0.4", label="predict-mean ref")
        ax.plot(x, sh, "^:", color="tab:orange", label="shuffle ref")
        ax.set_ylabel(f"{arch}\nmean KL (nats)")
        if row == 0:
            ax.legend(fontsize=7, frameon=False)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=75, fontsize=6, ha="right")
    fig.suptitle(f"io_hmm_v1 mouse {args.mouse} — peakiness and KL-side calibration "
                 "(projection-loss judging deferred to the full analysis)", fontsize=10)
    fig.tight_layout()
    out_dir = _HERE.parent / "figures" / "io_hmm_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir, f"scorecard_m{args.mouse}")


if __name__ == "__main__":
    main()
