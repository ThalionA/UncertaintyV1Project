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
  fig1_basis_spectrum_and_pcs.png    PCA basis: evar decay + PC shapes
  fig2_direct_fit_overlay.png        best single posterior per loss vs target
  fig3_temporal_mixture.png          loss vs per-bin sharpness (KL≫JS≫PCA)
  fig4_temporal_training_outcome.png per-bin entropy + trial spread per loss
  fig5_target_gallery_fits.png       6 varied targets, each fit by each loss
  fig6_entropy_width_evolution.png   entropy & width vs optimisation step
  fig7_bimodal_evolution_gradient.png posterior evolving on a bimodal target
  fig8_loss_scorecard.png            hand-built candidates scored by each loss
  fig9_width_shift_asymmetry.png     loss surface vs width vs peak-shift error
  metrics.csv                        numeric outputs

The fig8/fig9 "scorecard" demos use NO fitting — they just evaluate each loss on
illustrative candidate posteriors, to show how the *loss itself* behaves (and
hence which way it would push the network's weights). The later demos confirm
that gradient descent follows those static surfaces.

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


# ======================================================================
# Extra target shapes for the richer galleries
# ======================================================================
def skewed_bump(center, sigma, skew=2.5, n=N_CATS):
    """Asymmetric wrapped bump: wider on one side (different sigma each side)."""
    idx = np.arange(n)
    d = (idx - center + n // 2) % n - n // 2          # signed circular offset
    w = np.where(d >= 0, sigma * skew, sigma)
    p = np.exp(-0.5 * (d / w) ** 2)
    return p / p.sum()


def bimodal_bump(c1, c2, s1, s2, ratio=0.5, n=N_CATS):
    """Two bumps — the case where PCA's collapse to one spike is starkest."""
    p = ratio * bump(c1, s1, n) + (1 - ratio) * bump(c2, s2, n)
    return p / p.sum()


def _argmax_center(p):
    return int(np.argmax(p))


# ======================================================================
# Demo 3 — gallery: many target shapes, each with its best fit per loss
# ======================================================================
def demo3_target_gallery(pcs, evar, out_dir, rows):
    """fig5: a 2x3 gallery of varied targets. Each panel starts from a sharp
    spike at the target's mode (the over-confident state a decoder defaults to)
    and overlays the posterior each loss reaches after equal training — so the
    *restoring force toward the target shape* is what differs, not the optimum.
    PCA stays spiky on every shape; KL/JS recover the shape."""
    pcs_t = torch.tensor(pcs)
    evar_t = torch.tensor(evar)
    gallery = {
        "narrow @45 (sigma=3)": bump(45, 3),
        "broad @45 (sigma=14)": bump(45, 14),
        "off-centre @20 (sigma=8)": bump(20, 8),
        "boundary @3 (wraps)": bump(3, 8),
        "skewed @50": skewed_bump(50, 6, skew=2.6),
        "bimodal 30/65": bimodal_bump(30, 65, 6, 6),
    }
    losses_fit = ["PCA", "KL", "JS"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    x = np.arange(N_CATS)
    for ax, (tname, target) in zip(axes.ravel(), gallery.items()):
        init_logits = torch.tensor(np.log(bump(_argmax_center(target), 1.2) + EPS),
                                   dtype=torch.float32)
        ax.fill_between(x, target, color="0.85", zorder=0,
                        label=f"target (H={entropy_np(target):.2f})")
        for name in losses_fit:
            fit, _, _ = optimise_single(target, name, pcs_t, evar_t, init_logits)
            ax.plot(x, fit, color=LOSS_COLORS[name], lw=1.8,
                    label=f"{name} (H={entropy_np(fit):.2f})")
            rows.append({
                "demo": "gallery", "target": tname, "loss": name,
                "final_entropy": entropy_np(fit),
                "final_circ_std_bins": circ_std_bins(fit),
                "target_entropy": entropy_np(target),
                "target_circ_std_bins": circ_std_bins(target),
            })
        ax.set_title(tname, fontsize=10)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("angle bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("probability")
    fig.suptitle("Demo 3 — gallery of targets & fits: PCA stays spiky on every "
                 "shape; KL/JS recover the target shape", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "fig5_target_gallery_fits.png"), dpi=130)
    plt.close(fig)


# ======================================================================
# Demo 4 — how smoothness evolves: entropy & width vs optimisation step
# ======================================================================
def _fit_metric_history(target, loss_name, pcs_t, evar_t, init_logits,
                        steps=3000, lr=0.05, every=15):
    """Record (step, entropy, circular-std) of the posterior along training."""
    logits = init_logits.clone().detach().requires_grad_(True)
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    opt = torch.optim.Adam([logits], lr=lr)
    xs, ents, widths = [], [], []
    for s in range(steps + 1):
        pred = torch.softmax(logits, dim=-1).unsqueeze(0)
        if s % every == 0:
            p = pred.detach().numpy()[0]
            xs.append(s); ents.append(entropy_np(p)); widths.append(circ_std_bins(p))
        opt.zero_grad()
        _, fit, _ = custom_loss_all_H(
            pred, t, entropy_lambda=0.0, model_type="ppc",
            pcs=pcs_t, explained_variance=evar_t, loss_func_type=loss_name)
        fit.backward(); opt.step()
    return np.array(xs), np.array(ents), np.array(widths)


def demo4_entropy_width_evolution(pcs, evar, broad_sigma, out_dir, rows):
    """fig6: posterior entropy & circular-width vs step, from a sharp start.
    PCA flatlines well below the target; KL/JS climb to the target reference."""
    pcs_t = torch.tensor(pcs)
    evar_t = torch.tensor(evar)
    target = bump(N_CATS // 2, broad_sigma)
    init_logits = torch.tensor(np.log(bump(N_CATS // 2, 1.2) + EPS),
                               dtype=torch.float32)
    t_ent, t_wid = entropy_np(target), circ_std_bins(target)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in ["PCA", "KL", "JS"]:
        xs, ents, widths = _fit_metric_history(target, name, pcs_t, evar_t,
                                               init_logits)
        ax1.plot(xs, ents, color=LOSS_COLORS[name], lw=2, label=name)
        ax2.plot(xs, widths, color=LOSS_COLORS[name], lw=2, label=name)
        rows.append({"demo": "evolution", "loss": name,
                     "final_entropy": float(ents[-1]),
                     "final_circ_std_bins": float(widths[-1]),
                     "target_entropy": t_ent, "target_circ_std_bins": t_wid})
    ax1.axhline(t_ent, color="C1", ls="--", lw=1.5, label="target")
    ax2.axhline(t_wid, color="C1", ls="--", lw=1.5, label="target")
    ax1.set_ylabel("posterior entropy (nats)")
    ax1.set_title("Entropy evolution: PCA stays collapsed below the target")
    ax2.set_ylabel("posterior circular std (bins)")
    ax2.set_title("Width evolution: KL/JS settle at the target width")
    for ax in (ax1, ax2):
        ax.set_xlabel("optimisation step"); ax.legend(fontsize=8)
    fig.suptitle("Demo 4 — how posterior smoothness evolves during fitting "
                 "(start = over-confident spike)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "fig6_entropy_width_evolution.png"), dpi=130)
    plt.close(fig)


# ======================================================================
# Demo 5 — evolution snapshots on a bimodal target (viridis = step)
# ======================================================================
def demo5_bimodal_evolution(pcs, evar, out_dir, rows):
    """fig7: overlay the posterior at many optimisation steps (viridis = step)
    on a bimodal target, starting from an over-confident spike on ONE mode (the
    decoder's default failure). This is the restoring-force test of Demo 1 applied
    to a two-mode target: PCA exerts no force to discover the missing mode and
    stays a single spike; KL grows the second mode fastest, JS slower. Far-mode
    mass (annotated) orders KL > JS > PCA, matching the rest of the report."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    pcs_t = torch.tensor(pcs)
    evar_t = torch.tensor(evar)
    target = bimodal_bump(30, 65, 6, 6)
    # Over-confident spike on the LEFT mode only; the far mode must be discovered.
    init_logits = torch.tensor(np.log(bump(30, 1.5) + EPS), dtype=torch.float32)
    snap_steps = [0, 20, 60, 150, 400, 1000, 2500, 5000, 8000]
    losses_fit = ["PCA", "KL", "JS"]
    far = slice(55, 76)   # far (right) mode support, for the mass annotation

    def run(loss_name):
        logits = init_logits.clone().detach().requires_grad_(True)
        t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        opt = torch.optim.Adam([logits], lr=0.05)
        snaps = {}
        for s in range(max(snap_steps) + 1):
            pred = torch.softmax(logits, dim=-1).unsqueeze(0)
            if s in snap_steps:
                snaps[s] = pred.detach().numpy()[0]
            opt.zero_grad()
            _, fit, _ = custom_loss_all_H(
                pred, t, entropy_lambda=0.0, model_type="ppc",
                pcs=pcs_t, explained_variance=evar_t, loss_func_type=loss_name)
            fit.backward(); opt.step()
        return snaps

    norm = Normalize(vmin=0, vmax=len(snap_steps) - 1)
    cmap = plt.get_cmap("viridis")
    x = np.arange(N_CATS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, name in zip(axes, losses_fit):
        snaps = run(name)
        for i, s in enumerate(snap_steps):
            ax.plot(x, snaps[s], color=cmap(norm(i)), lw=1.5)
        ax.plot(x, target, "r--", lw=1.6, label="target")
        final = snaps[max(snap_steps)]
        far_mass = float(final[far].sum())
        ax.set_title(f"{name} — final far-mode mass = {far_mass:.2f}  "
                     f"(target 0.50)")
        ax.set_xlabel("angle bin"); ax.legend(fontsize=8)
        rows.append({"demo": "bimodal_evolution", "loss": name,
                     "final_entropy": entropy_np(final),
                     "final_far_mode_mass": far_mass,
                     "target_entropy": entropy_np(target),
                     "target_far_mode_mass": float(target[far].sum())})
    axes[0].set_ylabel("probability")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_ticks(range(len(snap_steps)))
    cbar.set_ticklabels([str(s) for s in snap_steps])
    cbar.set_label("optimisation step")
    fig.suptitle("Demo 5 — bimodal target from a spike on one mode: PCA never "
                 "discovers the second mode; KL grows it fastest, JS slower",
                 fontsize=12)
    fig.savefig(os.path.join(out_dir, "fig7_bimodal_evolution_gradient.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Demo 6 — scorecard: hand-crafted candidates, just READ OFF the losses
# ======================================================================
# No fitting. For each target we draw a handful of illustrative candidate
# posteriors and tabulate what every loss assigns. This isolates the *shape of
# the loss* (hence the direction it would push weights) from optimiser dynamics.

def _normloss(L, ref):
    """Express each loss relative to its own 'exact match = 0' so the four
    very-differently-scaled losses can share one bar axis. ``ref`` is the loss
    dict of the worst candidate, used as the per-loss 100% reference."""
    return {k: (L[k] / ref[k] if ref[k] > EPS else 0.0) for k in L}


def demo6_scorecard(pcs, evar, out_dir, rows):
    """fig8: a grid of (target, candidates) cards. Each card overlays the target
    (grey fill) with 4-5 hand-built candidate posteriors and prints the loss each
    candidate gets under PCA / KL / JS / Wasserstein. Reading across a card shows
    which mistakes each loss cares about."""
    cards = {
        "broad unimodal (sigma=12)": {
            "target": bump(45, 12),
            "candidates": {
                "exact": bump(45, 12),
                "too sharp": bump(45, 2),
                "too broad": bump(45, 24),
                "shifted +8": bump(53, 12),
                "uniform": np.ones(N_CATS) / N_CATS,
            },
        },
        "narrow unimodal (sigma=4)": {
            "target": bump(45, 4),
            "candidates": {
                "exact": bump(45, 4),
                "too sharp": bump(45, 1.5),
                "too broad": bump(45, 12),
                "shifted +6": bump(51, 4),
                "uniform": np.ones(N_CATS) / N_CATS,
            },
        },
        "bimodal (30 / 65)": {
            "target": bimodal_bump(30, 65, 6, 6),
            "candidates": {
                "exact": bimodal_bump(30, 65, 6, 6),
                "one mode only": bump(30, 6),
                "broad covers both": bump(47, 18),
                "sharp bimodal": bimodal_bump(30, 65, 2, 2),
                "uniform": np.ones(N_CATS) / N_CATS,
            },
        },
        "skewed": {
            "target": skewed_bump(45, 6, skew=2.6),
            "candidates": {
                "exact": skewed_bump(45, 6, skew=2.6),
                "symmetric narrow": bump(45, 4),
                "symmetric broad": bump(45, 12),
                "shifted +6": skewed_bump(51, 6, skew=2.6),
                "uniform": np.ones(N_CATS) / N_CATS,
            },
        },
    }

    cand_colors = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#8c564b"]
    fig, axes = plt.subplots(len(cards), 2, figsize=(15, 4.2 * len(cards)),
                             gridspec_kw={"width_ratios": [1.05, 1.4]})
    x = np.arange(N_CATS)

    for r, (cname, card) in enumerate(cards.items()):
        target = card["target"]
        cands = card["candidates"]
        # losses for every candidate
        Ls = {n: all_losses(c, target, pcs, evar) for n, c in cands.items()}
        for n in cands:
            rows.append({"demo": "scorecard", "target": cname, "candidate": n,
                         "PCA": Ls[n]["PCA"], "KL": Ls[n]["KL"],
                         "JS": Ls[n]["JS"], "Wasserstein": Ls[n]["Wasserstein"],
                         "cand_entropy": entropy_np(cands[n])})

        # --- left: the posteriors ---
        axL = axes[r, 0]
        axL.fill_between(x, target, color="0.85", zorder=0, label="target")
        axL.plot(x, target, "k--", lw=1.2, zorder=1)
        for (n, c), col in zip(cands.items(), cand_colors):
            if n == "exact":
                continue
            axL.plot(x, c, color=col, lw=1.6, label=n)
        axL.set_title(f"{cname}", fontsize=10)
        axL.set_xlabel("angle bin"); axL.set_ylabel("probability")
        axL.legend(fontsize=7)

        # --- right: grouped bars, each loss normalised to its own worst case ---
        axR = axes[r, 1]
        names = [n for n in cands if n != "exact"]
        worst = {k: max(Ls[n][k] for n in names) for k in LOSS_ORDER}
        ncand = len(names)
        bw = 0.8 / len(LOSS_ORDER)
        for li, lname in enumerate(LOSS_ORDER):
            vals = [Ls[n][lname] / (worst[lname] + EPS) for n in names]
            axR.bar(np.arange(ncand) + li * bw, vals, bw,
                    color=LOSS_COLORS[lname], label=lname, edgecolor="k", lw=0.3)
            # annotate with the raw loss value
            for ci, n in enumerate(names):
                axR.text(ci + li * bw, vals[ci] + 0.02, f"{Ls[n][lname]:.2g}",
                         ha="center", va="bottom", fontsize=5.5, rotation=90)
        axR.set_xticks(np.arange(ncand) + bw * (len(LOSS_ORDER) - 1) / 2)
        axR.set_xticklabels(names, fontsize=8)
        axR.set_ylabel("loss (relative to each\nloss's worst candidate)")
        axR.set_ylim(0, 1.28)
        axR.set_title("how each loss scores the candidates "
                      "(raw value printed above bar)", fontsize=9)
        axR.legend(fontsize=7, ncol=4, loc="upper center")

    fig.suptitle("Demo 6 — loss scorecard: hand-built candidates, no fitting. "
                 "Each loss's bar height = its score relative to its own worst "
                 "candidate.\nKL/JS punish 'too sharp' & 'one mode only' hardest; "
                 "PCA is nearly flat across width errors (only peak position "
                 "moves it).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig8_loss_scorecard.png"), dpi=130)
    plt.close(fig)


# ======================================================================
# Demo 7 — the asymmetry that sets the gradient direction
# ======================================================================
def demo7_width_shift_asymmetry(pcs, evar, out_dir, rows):
    """fig9: for one broad target, sweep a candidate's WIDTH (left) and its
    PEAK POSITION (right) and plot each loss. The width panel exposes the key
    fact: KL is steeply asymmetric (under-coverage punished far more than
    over-coverage) while PCA is shallow and nearly symmetric — so PCA's
    width-direction gradient is weak, and whatever pressure exists (the entropy
    penalty) wins. The shift panel shows all losses agree peak position matters,
    which is why PCA is fine for position/choice tasks."""
    target_sigma = 12.0
    target = bump(45, target_sigma)

    sig_cands = np.linspace(2, 30, 29)
    shift_cands = np.arange(0, 26)

    width_curves = {k: [] for k in ["PCA", "KL", "JS"]}
    for s in sig_cands:
        L = all_losses(bump(45, s), target, pcs, evar)
        for k in width_curves:
            width_curves[k].append(L[k])
    shift_curves = {k: [] for k in ["PCA", "KL", "JS"]}
    for d in shift_cands:
        L = all_losses(bump(45 + d, target_sigma), target, pcs, evar)
        for k in shift_curves:
            shift_curves[k].append(L[k])

    for s, *_ in [(0,)]:
        pass
    for i, s in enumerate(sig_cands):
        rows.append({"demo": "width_sweep", "cand_sigma": float(s),
                     "PCA": width_curves["PCA"][i], "KL": width_curves["KL"][i],
                     "JS": width_curves["JS"][i]})
    for i, d in enumerate(shift_cands):
        rows.append({"demo": "shift_sweep", "cand_shift_bins": int(d),
                     "PCA": shift_curves["PCA"][i], "KL": shift_curves["KL"][i],
                     "JS": shift_curves["JS"][i]})

    fig, (axW, axS) = plt.subplots(1, 2, figsize=(14, 5.2))
    # --- width panel, normalised per loss so shapes are comparable ---
    for k in ["PCA", "KL", "JS"]:
        c = np.array(width_curves[k]); c = c / (c.max() + EPS)
        axW.plot(sig_cands, c, color=LOSS_COLORS[k], lw=2, marker="o", ms=3, label=k)
    axW.axvline(target_sigma, color="0.4", ls="--", lw=1.2)
    axW.text(target_sigma + 0.4, 0.9, "target\nwidth", fontsize=8, color="0.3")
    axW.axvspan(2, target_sigma, color="orange", alpha=0.07)
    axW.text(5.5, 0.5, "too sharp\n(under-cover)", fontsize=8, color="darkorange")
    axW.text(22, 0.5, "too broad\n(over-cover)", fontsize=8, color="0.3")
    axW.set_xlabel("candidate width  sigma_cand (bins)")
    axW.set_ylabel("loss (normalised per loss to its own max)")
    axW.set_title("Width error: KL is steeply asymmetric (under-coverage hurts "
                  "most);\nPCA is shallow & ~symmetric -> weak width gradient")
    axW.legend(fontsize=9)

    # --- shift panel, raw-ish (normalised per loss too) ---
    for k in ["PCA", "KL", "JS"]:
        c = np.array(shift_curves[k]); c = c / (c.max() + EPS)
        axS.plot(shift_cands, c, color=LOSS_COLORS[k], lw=2, marker="s", ms=3, label=k)
    axS.set_xlabel("candidate peak shift (bins off target)")
    axS.set_ylabel("loss (normalised per loss to its own max)")
    axS.set_title("Peak-position error: all losses rise together\n"
                  "(PCA's strong direction — why it is best for position/choice)")
    axS.legend(fontsize=9)

    # annotate the headline asymmetry ratio for KL vs PCA
    L_sharp = all_losses(bump(45, 2), target, pcs, evar)
    L_broad = all_losses(bump(45, 24), target, pcs, evar)
    txt = (f"too-sharp / too-broad loss ratio:\n"
           f"  KL  = {L_sharp['KL']/ (L_broad['KL']+EPS):5.1f}x  (punishes under-coverage)\n"
           f"  JS  = {L_sharp['JS']/ (L_broad['JS']+EPS):5.1f}x\n"
           f"  PCA = {L_sharp['PCA']/(L_broad['PCA']+EPS):5.1f}x  (nearly symmetric)")
    axW.text(0.98, 0.02, txt, transform=axW.transAxes, fontsize=8,
             ha="right", va="bottom", family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.suptitle("Demo 7 — what each loss's landscape looks like along the two "
                 "error axes (no fitting; just the loss surface)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "fig9_width_shift_asymmetry.png"), dpi=130)
    plt.close(fig)


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

    # ---- richer example galleries + evolution figures ----
    demo3_target_gallery(pcs, evar, out_dir, rows)
    demo4_entropy_width_evolution(pcs, evar, broad_sigma, out_dir, rows)
    demo5_bimodal_evolution(pcs, evar, out_dir, rows)
    print("\nDemo 3-5 (gallery + evolution) figures written.")

    # ---- loss-only scorecards (no fitting): how the LOSS itself behaves ----
    demo6_scorecard(pcs, evar, out_dir, rows)
    demo7_width_shift_asymmetry(pcs, evar, out_dir, rows)
    print("Demo 6-7 (loss scorecard + asymmetry) figures written.")

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
