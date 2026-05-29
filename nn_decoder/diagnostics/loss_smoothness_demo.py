# -*- coding: utf-8 -*-
"""Demonstration: why the PCA-weighted fit-loss does not encourage *smooth*
posteriors, while KL and JS do — and how each loss behaves on the temporal
(sampling) decoder, where a full-trial posterior is the **average of sharp
per-time-bin posteriors**.

This is a methodological probe / teaching demo. It is self-contained
(synthetic targets) but imports the **production** loss code so the result
reflects the real objective, not a re-implementation:

  - ``pca_loss.pca_distance``            (numpy PCA-weighted distance)
  - ``nn_classifier.{KL_calc, JS_calc, Wasserstein_calc_1D,
      entropy_calc, custom_loss_all_H}``  (torch divergences + objective)
  - ``sklearn.decomposition.PCA``         fit exactly as
      ``time_binned_ppc.fit_pca_basis`` / ``run_experiment.py`` do
      (condition-averaged broad target bumps).

Outputs (default ``figures/loss_smoothness_demo/``):
  fig1_basis_spectrum_and_pcs.png   PCA basis: evar decay + PC shapes
  fig2_direct_fit_overlay.png       best single posterior per loss vs target
  fig3_temporal_mixture.png         loss vs per-bin sharpness (KL≫JS≫PCA)
  fig4_temporal_training_outcome.png per-bin entropy + trial spread per loss
  metrics.csv                       numeric outputs

Run:  cd nn_decoder && python diagnostics/loss_smoothness_demo.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import PCA

# --- import the production loss code (this script lives in nn_decoder/diagnostics) ---
NN_DECODER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

from pca_loss import pca_distance                                  # noqa: E402
from nn_classifier import (                                        # noqa: E402
    KL_calc, JS_calc, Wasserstein_calc_1D, entropy_calc, custom_loss_all_H,
)

torch.manual_seed(0)
np.random.seed(0)

N_CATS = 91          # angle bins (matches the 91-bin orientation targets)
EPS = 1e-12


# ======================================================================
# Synthetic targets: circular (wrapped-Gaussian) bumps over N_CATS bins
# ======================================================================
def circ_dist(i, c, n=N_CATS):
    """Shortest circular distance between bin indices on a ring of length n."""
    d = np.abs(i - c)
    return np.minimum(d, n - d)


def bump(center, sigma, n=N_CATS):
    """Normalised wrapped-Gaussian bump centred on ``center`` (bins)."""
    idx = np.arange(n)
    p = np.exp(-0.5 * (circ_dist(idx, center, n) / sigma) ** 2)
    return p / p.sum()


def circ_std_bins(p, n=N_CATS):
    """Circular standard deviation (in bins) of a distribution over the ring."""
    ang = 2 * np.pi * np.arange(n) / n
    r = np.sqrt((p * np.cos(ang)).sum() ** 2 + (p * np.sin(ang)).sum() ** 2)
    r = min(max(r, 1e-9), 1 - 1e-9)
    return np.sqrt(-2 * np.log(r)) * n / (2 * np.pi)


def entropy_np(p):
    return float(-(p * np.log(p + EPS)).sum())


# ======================================================================
# PCA basis — fit on condition-averaged broad bumps (mirrors fit_pca_basis)
# ======================================================================
def fit_basis(broad_sigma, n_conditions=24):
    """One broad-bump target per stimulus condition; PCA() keeps all PCs.

    Leading PCs encode peak *position* (the across-condition shift); trailing
    PCs encode bump *shape/width*. Returns (pcs, evar) as float32 arrays.
    """
    centers = np.linspace(0, N_CATS, n_conditions, endpoint=False)
    cond_targets = np.stack([bump(c, broad_sigma) for c in centers])
    pca = PCA()
    pca.fit(cond_targets)
    return (pca.components_.astype(np.float32),
            pca.explained_variance_ratio_.astype(np.float32),
            cond_targets)


# ======================================================================
# Loss helpers — call the production functions
# ======================================================================
def all_losses(pred, target, pcs, evar):
    """Compute every loss for a single (pred, target) pair of length N_CATS.

    Returns a dict {name: float}. KL is forward KL D(target || pred) — the
    mass-covering direction the trainer uses (``KL_calc(X=pred, Y=target)``).
    """
    p = torch.tensor(pred, dtype=torch.float32).unsqueeze(0)
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    return {
        "PCA": float(pca_distance(pred[None, :], target[None, :], pcs, evar)[0]),
        "KL": float(KL_calc(p, t)[0]),
        "JS": float(JS_calc(p, t)[0]),
        "Wasserstein": float(Wasserstein_calc_1D(p, t)[0]),
    }


LOSS_ORDER = ["PCA", "KL", "JS", "Wasserstein"]
LOSS_COLORS = {"PCA": "#d62728", "KL": "#1f77b4", "JS": "#2ca02c",
               "Wasserstein": "#9467bd"}


# ======================================================================
# Demo 1 — direct fit: which losses prefer a SMOOTH posterior?
# ======================================================================
def optimise_single(target, loss_name, pcs_t, evar_t, init_logits,
                    steps=3000, lr=0.05, record_every=20):
    """Gradient-descend a free softmax posterior to minimise ``loss_name``
    against ``target`` (single posterior; model_type='ppc', no entropy term).

    Starts from ``init_logits`` (so we can probe whether a loss pulls an
    *over-confident* posterior back toward the broad target) and records the
    prediction entropy over training. Returns (final_pred, steps_axis, H_traj).
    """
    logits = init_logits.clone().detach().requires_grad_(True)
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    opt = torch.optim.Adam([logits], lr=lr)
    steps_axis, H_traj = [], []
    for s in range(steps + 1):
        pred = torch.softmax(logits, dim=-1).unsqueeze(0)
        if s % record_every == 0:
            steps_axis.append(s)
            H_traj.append(float(entropy_calc(pred.detach())[0]))
        opt.zero_grad()
        _, fit, _ = custom_loss_all_H(
            pred, t, entropy_lambda=0.0, model_type="ppc",
            pcs=pcs_t, explained_variance=evar_t, loss_func_type=loss_name)
        fit.backward()
        opt.step()
    return (torch.softmax(logits.detach(), dim=-1).numpy(),
            np.array(steps_axis), np.array(H_traj))


def demo1_direct_fit(pcs, evar, broad_sigma, out_dir, rows):
    """Honest framing: with the *full* PCA basis the weighted-L2 optimum is the
    target itself, so given unlimited steps every loss reaches it. The real
    decoder, however, has finite capacity/training — so what matters is the
    *restoring force* each loss applies to an over-confident posterior. We
    start from a sharp spike at the correct location and watch each loss try to
    broaden it back to the target."""
    pcs_t = torch.tensor(pcs)
    evar_t = torch.tensor(evar)
    target = bump(N_CATS // 2, broad_sigma)
    losses_fit = ["PCA", "KL", "JS"]

    # Over-confident initialisation: a sharp spike at the true peak location.
    init_logits = torch.tensor(np.log(bump(N_CATS // 2, 1.2) + EPS),
                               dtype=torch.float32)

    fits, trajs = {}, {}
    for name in losses_fit:
        fits[name], xs, H = optimise_single(target, name, pcs_t, evar_t, init_logits)
        trajs[name] = (xs, H)
        rows.append({
            "demo": "direct_fit_from_sharp", "loss": name,
            "init_entropy": float(entropy_calc(
                torch.softmax(init_logits, -1).unsqueeze(0))[0]),
            "final_entropy": entropy_np(fits[name]),
            "final_circ_std_bins": circ_std_bins(fits[name]),
            "target_entropy": entropy_np(target),
            "target_circ_std_bins": circ_std_bins(target),
        })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(N_CATS)
    init_p = torch.softmax(init_logits, -1).numpy()
    ax = axes[0]
    ax.fill_between(x, target, color="0.8",
                    label=f"target (broad, σ={broad_sigma:g}, H={entropy_np(target):.2f})")
    ax.plot(x, init_p, color="0.4", ls=":", lw=1.5,
            label=f"sharp init (H={entropy_np(init_p):.2f})")
    for name in losses_fit:
        ax.plot(x, fits[name], color=LOSS_COLORS[name], lw=2,
                label=f"{name} (H={entropy_np(fits[name]):.2f})")
    ax.set_xlabel("angle bin"); ax.set_ylabel("probability")
    ax.set_title("Posterior after equal training from a sharp start")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(entropy_np(target), color="C1", ls="--", lw=1.5, label="target H")
    for name in losses_fit:
        xs, H = trajs[name]
        ax.plot(xs, H, color=LOSS_COLORS[name], lw=2, label=name)
    ax.set_xlabel("optimisation step"); ax.set_ylabel("posterior entropy (nats)")
    ax.set_title("KL & JS rapidly broaden the spike;\n"
                 "PCA exerts almost no broadening force")
    ax.legend(fontsize=8)

    fig.suptitle("Demo 1 — restoring force toward a smooth posterior "
                 "(start = over-confident spike)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "fig2_direct_fit_overlay.png"), dpi=130)
    plt.close(fig)
    return fits, target


# ======================================================================
# Demo 2 — temporal mixture: full-trial posterior = mean of T sharp bins
# ======================================================================
def mixture_average(centers, sigma_bin):
    """Full-trial posterior = mean over time bins of per-bin sharp bumps."""
    per_bin = np.stack([bump(c, sigma_bin) for c in centers])  # (T, N_CATS)
    return per_bin, per_bin.mean(axis=0)


def demo2_temporal_mixture(pcs, evar, broad_sigma, out_dir, rows, T=12):
    target = bump(N_CATS // 2, broad_sigma)
    # Per-bin centres tile the target's support (inverse-CDF at T quantiles),
    # so as sigma_bin -> broad the mixture average -> target, and as
    # sigma_bin -> 0 it becomes a gappy comb of spikes.
    cdf = np.cumsum(target)
    q = (np.arange(T) + 0.5) / T
    centers = np.interp(q, cdf, np.arange(N_CATS))

    sigmas = np.linspace(0.6, broad_sigma * 1.4, 26)  # sharp -> broad
    curves = {k: [] for k in LOSS_ORDER}
    avg_entropy = []
    for s in sigmas:
        _, avg = mixture_average(centers, s)
        L = all_losses(avg, target, pcs, evar)
        for k in LOSS_ORDER:
            curves[k].append(L[k])
        avg_entropy.append(entropy_np(avg))
    curves = {k: np.array(v) for k, v in curves.items()}

    # record sharp vs broad endpoints + fractional rise
    for k in LOSS_ORDER:
        broad_v, sharp_v = curves[k][-1], curves[k][0]
        rows.append({
            "demo": "temporal_mixture", "loss": k,
            "loss_broad_perbin": broad_v, "loss_sharp_perbin": sharp_v,
            "frac_increase_sharp_over_broad":
                (sharp_v - broad_v) / (broad_v + EPS),
        })

    # ---- figure: 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    x = np.arange(N_CATS)

    # (a) illustration at HIGH sharpness (gappy average)
    s_sharp = sigmas[0]
    per_bin, avg = mixture_average(centers, s_sharp)
    ax = axes[0, 0]
    for pb in per_bin:
        ax.plot(x, pb, color="0.7", lw=0.8)
    ax.plot(x, avg, color="k", lw=2, label="trial posterior = mean of bins")
    ax.plot(x, target, color="C1", lw=2, ls="--", label="broad target")
    ax.set_title(f"Sharp per-bin posteriors (σ_bin={s_sharp:.1f})\n"
                 "average is a gappy 'comb'")
    ax.set_xlabel("angle bin"); ax.set_ylabel("probability")
    ax.legend(fontsize=7)

    # (b) illustration at LOW sharpness (smooth average ~ target)
    s_broad = sigmas[-1]
    per_bin, avg = mixture_average(centers, s_broad)
    ax = axes[0, 1]
    for pb in per_bin:
        ax.plot(x, pb, color="0.7", lw=0.8)
    ax.plot(x, avg, color="k", lw=2, label="trial posterior = mean of bins")
    ax.plot(x, target, color="C1", lw=2, ls="--", label="broad target")
    ax.set_title(f"Broad per-bin posteriors (σ_bin={s_broad:.1f})\n"
                 "average ≈ target")
    ax.set_xlabel("angle bin"); ax.set_ylabel("probability")
    ax.legend(fontsize=7)

    # (c) KL & JS vs sharpness — same axis (nats): KL suffers a lot, JS less
    ax = axes[1, 0]
    ax.plot(sigmas, curves["KL"], color=LOSS_COLORS["KL"], lw=2, marker="o", ms=3,
            label="KL (forward)")
    ax.plot(sigmas, curves["JS"], color=LOSS_COLORS["JS"], lw=2, marker="s", ms=3,
            label="JS")
    ax.invert_xaxis()  # sharper to the right
    ax.set_xlabel("per-bin width σ_bin  (sharper →)")
    ax.set_ylabel("loss (nats)")
    ax.set_title("KL vs JS as per-bin posteriors sharpen\n"
                 "KL explodes on the gappy average; JS bounded")
    ax.legend(fontsize=8)

    # (d) PCA & Wasserstein vs sharpness — y from 0 so flatness is visible
    ax = axes[1, 1]
    ax.plot(sigmas, curves["PCA"], color=LOSS_COLORS["PCA"], lw=2, marker="o", ms=3,
            label="PCA (weighted L2)")
    ax2 = ax.twinx()
    ax2.plot(sigmas, curves["Wasserstein"], color=LOSS_COLORS["Wasserstein"],
             lw=2, marker="^", ms=3, label="Wasserstein")
    ax.invert_xaxis()
    ax.set_ylim(bottom=0, top=max(curves["PCA"].max() * 3, EPS))
    ax2.set_ylim(bottom=0)
    ax.set_xlabel("per-bin width σ_bin  (sharper →)")
    ax.set_ylabel("PCA loss (native units)", color=LOSS_COLORS["PCA"])
    ax2.set_ylabel("Wasserstein", color=LOSS_COLORS["Wasserstein"])
    ax.set_title("PCA barely changes as per-bin posteriors sharpen\n"
                 "(leading position-PCs preserved by averaging)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    fig.suptitle("Demo 2 — temporal code: trial posterior = mean of T sharp "
                 "per-bin posteriors", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig3_temporal_mixture.png"), dpi=130)
    plt.close(fig)
    return sigmas, curves


# ======================================================================
# Demo 2b — training dynamics: optimise T per-bin logits under each loss
# ======================================================================
def optimise_perbin(target, loss_name, pcs_t, evar_t, T, entropy_lambda,
                    steps=4000, lr=0.05):
    """Optimise T per-bin posteriors via the real sampling objective.

    Mirrors training: ``custom_loss_all_H`` averages the T bins into the trial
    posterior, scores it against the (replicated) target, and adds the per-bin
    entropy penalty. Returns (per_bin (T,N_CATS), trial_avg (N_CATS))."""
    logits = (0.01 * torch.randn(T, N_CATS)).requires_grad_(True)
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0).repeat(T, 1)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = torch.softmax(logits, dim=-1)        # (T, N_CATS)
        total, _, _ = custom_loss_all_H(
            pred, t, entropy_lambda=entropy_lambda, model_type="sampling",
            pcs=pcs_t, explained_variance=evar_t, loss_func_type=loss_name)
        total.backward()
        opt.step()
    pred = torch.softmax(logits.detach(), dim=-1).numpy()
    return pred, pred.mean(axis=0)


def demo2b_training_outcome(pcs, evar, broad_sigma, out_dir, rows, T=12):
    pcs_t = torch.tensor(pcs)
    evar_t = torch.tensor(evar)
    target = bump(N_CATS // 2, broad_sigma)
    losses = ["PCA", "KL", "JS"]
    lambdas = [0.0, 3e-3]  # 3e-3 = production entropy_lambda

    res = {}  # (loss, lam) -> dict
    for lam in lambdas:
        for name in losses:
            per_bin, avg = optimise_perbin(target, name, pcs_t, evar_t, T, lam)
            mean_perbin_H = float(np.mean([entropy_np(p) for p in per_bin]))
            res[(name, lam)] = {
                "mean_perbin_entropy": mean_perbin_H,
                "trial_entropy": entropy_np(avg),
                "trial_circ_std_bins": circ_std_bins(avg),
            }
            rows.append({
                "demo": "temporal_training", "loss": name,
                "entropy_lambda": lam,
                "mean_perbin_entropy": mean_perbin_H,
                "trial_entropy": entropy_np(avg),
                "trial_circ_std_bins": circ_std_bins(avg),
            })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    width = 0.35
    xpos = np.arange(len(losses))
    target_H = entropy_np(target)
    alphas = {0.0: 0.55, 3e-3: 1.0}

    for j, (metric, title, ylab) in enumerate([
        ("mean_perbin_entropy", "Per-bin posterior entropy",
         "mean per-bin H (nats)"),
        ("trial_entropy", "Trial posterior entropy (mean over bins)",
         "trial H (nats)")]):
        ax = axes[j]
        for i, lam in enumerate(lambdas):
            vals = [res[(name, lam)][metric] for name in losses]
            ax.bar(xpos + (i - 0.5) * width, vals, width,
                   color=[LOSS_COLORS[n] for n in losses],
                   alpha=alphas[lam], edgecolor="k", linewidth=0.5)
        ax.axhline(target_H, color="C1", ls="--", lw=1.5)
        ax.set_xticks(xpos); ax.set_xticklabels(losses)
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(0, target_H * 1.18)

    # shared, non-overlapping legend (gray proxies for the two λ levels)
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(facecolor="0.5", alpha=alphas[0.0], edgecolor="k",
                       label="no entropy penalty (λ=0)"),
        mpatches.Patch(facecolor="0.5", alpha=alphas[3e-3], edgecolor="k",
                       label="production penalty (λ=3e-3)"),
        Line2D([0], [0], color="C1", ls="--", lw=1.5, label="target H"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Demo 2b — temporal training outcome: per-bin sharpness the "
                 "loss tolerates\nWith the production entropy penalty, PCA's trial "
                 "posterior collapses; KL & JS stay calibrated", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(os.path.join(out_dir, "fig4_temporal_training_outcome.png"), dpi=130)
    plt.close(fig)


# ======================================================================
# fig1 — basis spectrum + PC shapes
# ======================================================================
def fig1_basis(pcs, evar, cond_targets, broad_sigma, out_dir, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    k = min(12, len(evar))
    axes[0].bar(np.arange(k), evar[:k], color="0.4")
    axes[0].set_xlabel("principal component"); axes[0].set_ylabel("explained-variance ratio")
    axes[0].set_title(f"PCA basis spectrum (top {k})\n"
                      f"PC0–1 hold {evar[:2].sum()*100:.1f}% of variance")
    x = np.arange(N_CATS)
    for idx, lab in [(0, "PC0 (position)"), (1, "PC1 (position)"),
                     (min(8, len(pcs) - 1), f"PC{min(8,len(pcs)-1)} (shape/width)")]:
        axes[1].plot(x, pcs[idx], label=f"{lab}, evar={evar[idx]:.1e}")
    axes[1].set_xlabel("angle bin"); axes[1].set_ylabel("PC weight")
    axes[1].set_title("Leading PCs encode peak position;\n"
                      "width/shape lives in near-zero-evar PCs")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_basis_spectrum_and_pcs.png"), dpi=130)
    plt.close(fig)
    for i in range(k):
        rows.append({"demo": "basis", "pc_index": i, "explained_var_ratio": float(evar[i])})


# ======================================================================
def write_csv(rows, out_dir):
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    path = os.path.join(out_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def main(out_dir, broad_sigma=9.0, T=12):
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    pcs, evar, cond_targets = fit_basis(broad_sigma)
    print(f"PCA basis: {len(evar)} components; "
          f"top-2 evar = {evar[:2]}, sum top-2 = {evar[:2].sum():.4f}")

    fig1_basis(pcs, evar, cond_targets, broad_sigma, out_dir, rows)
    fits, target = demo1_direct_fit(pcs, evar, broad_sigma, out_dir, rows)
    print("Demo 1 (direct fit) posterior entropy [nats]:")
    print(f"  target = {entropy_np(target):.3f}")
    for name in ["PCA", "KL", "JS"]:
        print(f"  {name:11s} = {entropy_np(fits[name]):.3f}")

    sigmas, curves = demo2_temporal_mixture(pcs, evar, broad_sigma, out_dir, rows, T)
    print("\nDemo 2 (temporal mixture) loss at sharp vs broad per-bin:")
    for k in LOSS_ORDER:
        frac = (curves[k][0] - curves[k][-1]) / (curves[k][-1] + EPS)
        print(f"  {k:11s}: broad={curves[k][-1]:.4g}  sharp={curves[k][0]:.4g}  "
              f"(+{frac*100:.0f}%)")

    demo2b_training_outcome(pcs, evar, broad_sigma, out_dir, rows, T)

    path = write_csv(rows, out_dir)
    print(f"\nWrote figures + {os.path.basename(path)} to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="figures/loss_smoothness_demo")
    p.add_argument("--broad-sigma", type=float, default=9.0,
                   help="width (bins) of the broad target bump")
    p.add_argument("--n-bins", type=int, default=12,
                   help="T: number of time bins averaged into the trial posterior")
    args = p.parse_args()
    main(args.out_dir, broad_sigma=args.broad_sigma, T=args.n_bins)
