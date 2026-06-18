# -*- coding: utf-8 -*-
"""Does the temporal bins' similarity (in mean-location or in width) depend on
the STIMULUS — orientation, contrast, dispersion?

Item 4 measured how similar the 10 per-time-bin posteriors are (between-bin
dispersion of the per-bin mean μ_t and per-bin width σ_t). Here we ask whether
that similarity is stimulus-dependent: per trial we compute

  between-bin LOCATION dispersion = std_t(μ_t)   (low = bins share a location)
  between-bin WIDTH    dispersion = std_t(σ_t)   (low = bins share a width)

then group trials by each stimulus variable (`trials.{dispersion,contrast,
orientation}`) and average. Reads `loss_comparison_v1` (temporal arch,
`decoded_samp`, 6 mice), all five losses.

Output (PNG+SVG) under figures/peakiness_scatter/:
  temporal_bin_by_condition.png   2×3 grid: (location | width dispersion) ×
                                   (dispersion | contrast | orientation), per loss

Run:  cd nn_decoder && OMP_NUM_THREADS=1 \
        python diagnostics/temporal_bin_by_condition.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps                                          # noqa: E402
from diagnostics.lambda_h_sampling_test import _moments              # noqa: E402
from diagnostics.uncertainty_scaling_realdata import (               # noqa: E402
    _trial_field, _binned_mean,
)

LOSSES = ["PCA", "CE", "KL", "JS", "Wasserstein"]
LCOL = {"PCA": ps.PCA_EVAR, "CE": ps.CE, "KL": ps.KL, "JS": ps.JS,
        "Wasserstein": ps.WASSERSTEIN}
VARS = [("disp", "stimulus dispersion (→ more uncertain)", 7),
        ("con",  "stimulus contrast (→ less uncertain)", 7),
        ("ori",  "stimulus orientation (deg; 0/90 = references)", 9)]
METRICS = [("loc", "between-bin LOCATION dispersion\nstd$_t(\\mu_t)$ (deg)"),
           ("wid", "between-bin WIDTH dispersion\nstd$_t(\\sigma_t)$ (deg)")]


def _slug(loss):
    return f"Q_{loss}_half_100ms" + ("_all" if loss == "PCA" else "")


def collect(results_root, run, split):
    """Per loss: per-trial bin-dispersion (location & width) + stimulus vars."""
    out = {}
    for loss in LOSSES:
        f = Path(results_root) / run / _slug(loss) / f"{split}.mat"
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get("results")
        if not isinstance(res, dict):
            continue
        loc, wid, disp, con, ori = [], [], [], [], []
        for mk in sorted(res):
            D = res[mk]["Dist"]["temp"]
            ds = np.asarray(D["decoded_samp"], float)        # (n, 91, T)
            tr = res[mk]["trials"]
            n = ds.shape[0]
            mu_t, var_t = _moments(ds, 1)                    # (n, T)
            sd_t = np.sqrt(np.clip(var_t, 0, None))
            loc.append(np.nanstd(mu_t, axis=1))             # (n,)
            wid.append(np.nanstd(sd_t, axis=1))             # (n,)
            disp.append(_trial_field(tr, "dispersion", n))
            con.append(_trial_field(tr, "contrast", n))
            ori.append(_trial_field(tr, "orientation", n))
        if loc:
            out[loss] = {"loc": np.concatenate(loc), "wid": np.concatenate(wid),
                         "disp": np.concatenate(disp), "con": np.concatenate(con),
                         "ori": np.concatenate(ori)}
    return out


def main(results_root, run, split, out_root):
    ps.apply()
    data = collect(results_root, run, split)
    if not data:
        raise SystemExit("no temporal cells found.")
    ntot = len(next(iter(data.values()))["loc"])

    fig, axes = plt.subplots(len(METRICS), len(VARS), figsize=(13.5, 7.0),
                             sharex="col")
    for r, (mkey, mlab) in enumerate(METRICS):
        for c, (ckey, xlab, nb) in enumerate(VARS):
            ax = axes[r, c]
            for loss in LOSSES:
                if loss not in data:
                    continue
                d = data[loss]
                cx, m, s = _binned_mean(d[ckey], d[mkey], nb)
                ax.errorbar(cx, m, yerr=s, color=LCOL[loss], lw=1.8, marker="o",
                            ms=4, capsize=2, label=ps.loss_label(loss))
            if r == len(METRICS) - 1:
                ax.set_xlabel(xlab, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(mlab, fontsize=8.5)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7, loc="best")
    ps.label_panels(axes.ravel())
    fig.suptitle("Do the temporal bins' similarity (mean-location & width) depend "
                 f"on the stimulus?  (loss_comparison_v1, temporal, 6 mice, "
                 f"{ntot} trials)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ps.save_fig(fig, Path(out_root), "temporal_bin_by_condition", layout=None)

    # console: range of each metric across each stimulus, per loss
    print(f"{ntot} trials. between-bin dispersion range (min→max across levels):")
    for mkey, mlab in METRICS:
        print(f"  [{mkey}]")
        for ckey, _, nb in VARS:
            line = []
            for loss in LOSSES:
                if loss not in data:
                    continue
                _, m, _ = _binned_mean(data[loss][ckey], data[loss][mkey], nb)
                line.append(f"{loss[:4]} {np.nanmin(m):.1f}-{np.nanmax(m):.1f}")
            print(f"    vs {ckey:5s}: " + "  ".join(line))
    print(f"\nDone. {Path(out_root).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="loss_comparison_v1")
    ap.add_argument("--split", default="stratified_balanced")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-root", default="figures/peakiness_scatter")
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
