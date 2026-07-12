# -*- coding: utf-8 -*-
"""Run and plot the mean/moments/DeepSets uncertainty analysis.

Examples
--------
Synthetic validation (10 independent datasets)::

    python run_deepsets_analysis.py synthetic --seeds 0:10

One-mouse real-data smoke::

    python run_deepsets_analysis.py real --mouse-ids 0 --smoke

Full resumable real-data grid::

    python run_deepsets_analysis.py real --mouse-ids 0 1 2 3 4 5

Aggregate and render existing shards::

    python run_deepsets_analysis.py plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from deepsets_uncertainty import (  # noqa: E402
    AnalysisConfig, LOSS_NAMES, MODEL_NAMES, condition_mean_prediction,
    config_dict, load_result, make_synthetic, posterior_metrics,
    prepare_arrays, save_result, train_model,
)
from figsave import save_fig  # noqa: E402
from utils import apply_temporal_binning, load_vr_export  # noqa: E402


DEFAULT_ROOT = HERE / "results" / "deepsets_uncertainty"
DEFAULT_FIG = HERE / "figures" / "deepsets_uncertainty"
MODEL_LABEL = {"mean": "Mean only", "moments": "Mean + variance",
               "deepsets": "DeepSets"}
MODEL_COLOUR = {"mean": "#777777", "moments": "#2878B5",
                "deepsets": "#D55E00"}


def save_analysis_fig(fig, out_dir, stem):
    """Safety margin for bbox_inches='tight', which can expand final pixels."""
    save_fig(fig, out_dir, stem, max_px=1500)


def parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        return list(range(int(a), int(b)))
    return [int(x) for x in spec.split(",") if x.strip()]


def order_oracle_cv_accuracy(features: np.ndarray, labels: np.ndarray,
                             seed: int, n_folds: int = 5) -> float:
    """Dependency-light nearest-centroid CV positive control for order code."""
    features = np.asarray(features, float)
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    correct = total = 0
    for fold in range(n_folds):
        tr, te = folds != fold, folds == fold
        mu, sd = features[tr].mean(0), features[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        xtr, xte = (features[tr] - mu) / sd, (features[te] - mu) / sd
        classes = np.unique(labels[tr])
        centroids = np.stack([xtr[labels[tr] == c].mean(0) for c in classes])
        pred = classes[np.argmin(((xte[:, None] - centroids[None]) ** 2).sum(2), axis=1)]
        correct += int(np.sum(pred == labels[te]))
        total += int(np.sum(te))
    return correct / max(total, 1)


def cfg_from_args(args) -> AnalysisConfig:
    if args.smoke:
        return AnalysisConfig(seed=args.seed, max_epochs=5, patience=3,
                              min_epochs=2, restarts=1, hidden_dim=16,
                              phi_hidden=8, phi_dim=8)
    return AnalysisConfig(seed=args.seed, max_epochs=args.max_epochs,
                          patience=args.patience, restarts=args.restarts)


def _job_path(root: Path, domain: str, identity: str, regime: str,
              model: str, loss: str, null: bool) -> Path:
    suffix = "null" if null else "real"
    return root / domain / identity / regime / f"{model}__{loss}__{suffix}.npz"


def _run_grid(prepared, root, domain, identity, regime, cfg,
              losses=LOSS_NAMES, include_null=False, force=False,
              extra_metadata=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{domain}/{identity}/{regime}] device={device} "
          f"train={len(prepared.X_train)} val={len(prepared.X_val)} "
          f"test={len(prepared.X_test)} neurons={prepared.X_train.shape[-1]}")
    for loss in losses:
        for model in MODEL_NAMES:
            for null in ([False, True] if include_null else [False]):
                path = _job_path(root, domain, identity, regime, model, loss, null)
                if path.exists() and not force:
                    print(f"  [skip] {path.relative_to(root)}")
                    continue
                print(f"  [fit] model={model} loss={loss} null={null}")
                result = train_model(model, loss, prepared, cfg, null=null,
                                     device=device)
                metadata = {
                    "domain": domain, "identity": identity, "regime": regime,
                    "model": model, "loss": loss, "null": null,
                    "config": config_dict(cfg),
                    "train_idx": prepared.train_idx.tolist(),
                    "val_idx": prepared.val_idx.tolist(),
                    "test_idx": prepared.test_idx.tolist(),
                }
                if extra_metadata:
                    metadata.update(extra_metadata)
                save_result(path, result, metadata)


def run_synthetic(args):
    cfg0 = cfg_from_args(args)
    root = Path(args.results_root)
    for seed in parse_seeds(args.seeds):
        cfg = AnalysisConfig(**{**config_dict(cfg0), "seed": seed})
        for regime in ("mean", "variance", "order"):
            X, Q, C, audit = make_synthetic(
                regime, seed, n_base=args.n_synthetic,
                t_bins=10, n_neurons=args.synthetic_neurons)
            prepared = prepare_arrays(X, Q, C, cfg)
            oracle_auc = np.nan
            if regime == "order":
                y = np.asarray(audit["u_idx"])
                x = np.asarray(audit["order_features"])
                oracle_auc = order_oracle_cv_accuracy(x, y, seed)
            # Mechanism audit is persisted once per dataset/regime.
            audit_path = root / "synthetic" / f"seed_{seed:03d}" / regime / "audit.npz"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(audit_path, **audit, oracle_order_accuracy=oracle_auc)
            _run_grid(prepared, root, "synthetic", f"seed_{seed:03d}", regime,
                      cfg, losses=args.losses, force=args.force,
                      extra_metadata={"oracle_order_accuracy": oracle_auc})
    aggregate(root)


def load_real_mouse(mouse_id: int):
    activity, q, _d, _l, trials = load_vr_export(mouse_id)
    x = np.transpose(activity, (1, 2, 0))
    x = apply_temporal_binning(x, time_window="half", bin_size_ms=100)
    c = np.column_stack([trials["orientation"], trials["contrast"],
                         trials["dispersion"]])
    return x.astype(np.float32), np.asarray(q, np.float32), c


def run_real(args):
    cfg = cfg_from_args(args)
    root = Path(args.results_root)
    for mouse_id in args.mouse_ids:
        X, Q, C = load_real_mouse(mouse_id)
        prepared = prepare_arrays(X, Q, C, cfg)
        _run_grid(prepared, root, "real", f"mouse_{mouse_id}", "Q_half_100ms",
                  cfg, losses=args.losses, include_null=True, force=args.force,
                  extra_metadata={"mouse_id": mouse_id})
        # Save the leakage-safe condition-mean baseline for this exact split.
        baseline_path = (root / "real" / f"mouse_{mouse_id}" /
                         "Q_half_100ms" / "condition_mean_baseline.npz")
        if args.force or not baseline_path.exists():
            pred = condition_mean_prediction(prepared.Q_train, prepared.C_train,
                                             prepared.C_test)
            save_result(baseline_path, {
                "pred": pred, "target": prepared.Q_test,
                "conditions": prepared.C_test, "pcs": prepared.pcs,
                "evar": prepared.evar,
            }, {"domain": "real", "identity": f"mouse_{mouse_id}",
                "regime": "Q_half_100ms", "model": "condition_mean",
                "loss": "none", "null": False, "mouse_id": mouse_id,
                "config": config_dict(cfg)})
    aggregate(root)


def aggregate(root: Path):
    root = Path(root)
    trial_rows = []
    summary_rows = []
    for path in sorted(root.glob("**/*.npz")):
        if path.name == "audit.npz":
            continue
        try:
            data, meta = load_result(path)
        except Exception as exc:
            print(f"[warn] skip unreadable {path}: {exc}")
            continue
        if "pred" not in data or "target" not in data:
            continue
        metrics = posterior_metrics(data["pred"], data["target"],
                                    data.get("pcs"), data.get("evar"))
        for key, value in meta.items():
            if key != "config" and np.isscalar(value):
                metrics[key] = value
        metrics["trial"] = np.arange(len(metrics))
        metrics["source"] = str(path.relative_to(root))
        trial_rows.append(metrics)
        numeric = metrics.select_dtypes(include=[np.number]).mean().to_dict()
        summary_rows.append({**{k: v for k, v in meta.items()
                                if k != "config" and np.isscalar(v)},
                             **numeric, "source": str(path.relative_to(root)),
                             "n_params": float(np.asarray(data.get("n_params", np.nan))),
                             "best_epoch": float(np.asarray(data.get("best_epoch", np.nan))),
                             "eligible_train": float(np.asarray(data.get("eligible_train", np.nan)))})
    root.mkdir(parents=True, exist_ok=True)
    trial_df = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    trial_df.to_csv(root / "trial_metrics.csv", index=False)
    summary_df.to_csv(root / "summary_metrics.csv", index=False)
    _write_mouse_contrasts(summary_df, root)
    print(f"[aggregate] {len(summary_df)} jobs -> {root / 'summary_metrics.csv'}")
    return trial_df, summary_df


def _write_mouse_contrasts(summary: pd.DataFrame, root: Path):
    """Persist mouse-level contrasts; never treat trials as group replicates."""
    if summary.empty or "domain" not in summary:
        return
    real = summary[(summary.domain == "real") &
                   summary.model.isin(MODEL_NAMES)].copy()
    if real.empty:
        return
    real["mouse"] = real["identity"].str.extract(r"(\d+)").astype(int)
    metrics = ["KL", "JS", "Brier", "mean_abs_deg",
               "variance_abs_deg2", "entropy_abs_nats"]
    rows = []
    for loss in sorted(real.loss.unique()):
        ss = real[real.loss == loss]
        for metric in metrics:
            for a, b, label in [("deepsets", "mean", "deepsets-minus-mean"),
                                ("deepsets", "moments", "deepsets-minus-moments")]:
                aa = ss[(ss.model == a) & (~ss["null"].astype(bool))].set_index("mouse")
                bb = ss[(ss.model == b) & (~ss["null"].astype(bool))].set_index("mouse")
                for mouse in aa.index.intersection(bb.index):
                    rows.append({"mouse": mouse, "loss": loss, "metric": metric,
                                 "contrast": label,
                                 "value": aa.loc[mouse, metric] - bb.loc[mouse, metric]})
            for model in MODEL_NAMES:
                rr = ss[(ss.model == model) & (~ss["null"].astype(bool))].set_index("mouse")
                nn = ss[(ss.model == model) & (ss["null"].astype(bool))].set_index("mouse")
                for mouse in rr.index.intersection(nn.index):
                    rows.append({"mouse": mouse, "loss": loss, "metric": metric,
                                 "contrast": f"{model}-real-over-null",
                                 "value": rr.loc[mouse, metric] / nn.loc[mouse, metric]})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(root / "mouse_contrasts.csv", index=False)

    # Formal group tests are restricted to the two prespecified KL-trained
    # architecture contrasts. Other loss/metric grids remain robustness views.
    group_rows = []
    primary = contrasts[(contrasts.loss == "KL") &
                        contrasts.contrast.str.startswith("deepsets-minus")]
    for (metric, contrast), g in primary.groupby(["metric", "contrast"]):
        values = g.value.dropna().values
        if len(values) < 2:
            continue
        t = stats.ttest_1samp(values, 0.0)
        try:
            w = stats.wilcoxon(values, alternative="two-sided",
                               method="exact") if np.any(values != 0) else None
        except ValueError:
            w = None
        group_rows.append({
            "loss": "KL", "metric": metric, "contrast": contrast,
            "n_mice": len(values), "mean": np.mean(values),
            "sem": stats.sem(values), "paired_t": t.statistic,
            "paired_t_p": t.pvalue,
            "wilcoxon": w.statistic if w is not None else np.nan,
            "wilcoxon_p": w.pvalue if w is not None else np.nan,
            "note": "mouse is inference unit; exact Wilcoxon floor at n=6 is 0.03125",
        })
    pd.DataFrame(group_rows).to_csv(root / "group_stats_primary.csv", index=False)


def plot_synthetic(root: Path, fig_root: Path):
    summary = pd.read_csv(root / "summary_metrics.csv")
    sub = summary[summary["domain"] == "synthetic"].copy()
    if sub.empty:
        print("[plot] no synthetic results")
        return
    regimes = [r for r in ("mean", "variance", "order") if r in set(sub.regime)]
    fig, axes = plt.subplots(2, len(regimes), figsize=(4.6 * len(regimes), 7.4),
                             sharex="col", sharey="row")
    metrics = [("KL", "Held-out KL(Q || prediction)"),
               ("variance_abs_deg2", "Absolute posterior-variance error (deg^2)")]
    for row, (metric, ylabel) in enumerate(metrics):
        for col, regime in enumerate(regimes):
            ax = axes[row, col]
            s = sub[sub.regime == regime]
            for li, loss in enumerate(sorted(s.loss.unique())):
                means, sems = [], []
                for model in MODEL_NAMES:
                    v = s[(s.loss == loss) & (s.model == model)][metric].values
                    means.append(np.mean(v) if len(v) else np.nan)
                    sems.append(stats.sem(v) if len(v) > 1 else 0)
                x = np.arange(3) + (li - (len(s.loss.unique()) - 1) / 2) * 0.045
                ax.errorbar(x, means, yerr=sems, marker="o", capsize=2, lw=1.2,
                            label=loss)
            ax.set_xticks(range(3), [MODEL_LABEL[m] for m in MODEL_NAMES], rotation=20)
            if row == 0:
                ax.set_title(f"{regime.capitalize()}-coded uncertainty")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=.2)
            if col == 0:
                ax.set_ylabel(ylabel + "\nmean +/- SEM across datasets")
    axes[0, -1].legend(title="Training loss", fontsize=8)
    fig.suptitle("Synthetic capability test: common held-out metric")
    save_analysis_fig(fig, fig_root, "synthetic_common_kl")

    # Width calibration under headline KL training.
    trials = pd.read_csv(root / "trial_metrics.csv")
    t = trials[(trials.domain == "synthetic") & (trials.loss == "KL")]
    fig, axes = plt.subplots(len(regimes), 3, figsize=(11, 3.1 * len(regimes)),
                             sharex=True, sharey=True)
    for ri, regime in enumerate(regimes):
        for mi, model in enumerate(MODEL_NAMES):
            ax = axes[ri, mi]
            s = t[(t.regime == regime) & (t.model == model)]
            ax.hexbin(s.target_variance_deg2, s.pred_variance_deg2,
                      gridsize=24, mincnt=1, cmap="viridis")
            lim = [0, max(s.target_variance_deg2.max(), s.pred_variance_deg2.max())]
            ax.plot(lim, lim, "k--", lw=.8)
            if ri == 0: ax.set_title(MODEL_LABEL[model])
            if mi == 0: ax.set_ylabel(f"{regime}\npredicted variance")
            if ri == len(regimes) - 1: ax.set_xlabel("target variance (deg^2)")
    fig.suptitle("Synthetic posterior-width calibration (KL-trained)")
    save_analysis_fig(fig, fig_root, "synthetic_width_calibration")

    # Mechanism and order-positive-control audit.
    audits = []
    for p in root.glob("synthetic/seed_*/order/audit.npz"):
        z = np.load(p)
        audits.append(float(z["oracle_order_accuracy"]))
    if audits:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.scatter(np.arange(len(audits)), audits, color="#4C78A8")
        ax.axhline(.2, color="k", ls="--", label="5-class chance")
        ax.set(xlabel="Synthetic dataset", ylabel="Order oracle CV accuracy",
               title="Order generator positive control")
        ax.legend()
        save_analysis_fig(fig, fig_root, "synthetic_order_oracle")


def plot_real(root: Path, fig_root: Path):
    summary = pd.read_csv(root / "summary_metrics.csv")
    sub = summary[(summary.domain == "real") &
                  (summary.model.isin(MODEL_NAMES))].copy()
    if sub.empty:
        print("[plot] no real results")
        return
    sub["mouse"] = sub["identity"].str.extract(r"(\d+)").astype(int)
    losses = sorted(sub.loss.unique())
    fig, axes = plt.subplots(1, len(losses), figsize=(4.2 * len(losses), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, loss in zip(axes, losses):
        s = sub[sub.loss == loss]
        for mi, model in enumerate(MODEL_NAMES):
            real = s[(s.model == model) & (~s["null"].astype(bool))].set_index("mouse")
            null = s[(s.model == model) & (s["null"].astype(bool))].set_index("mouse")
            common = real.index.intersection(null.index)
            skill = real.loc[common, "KL"] / null.loc[common, "KL"]
            x = np.full(len(skill), mi, float) + np.linspace(-.06, .06, len(skill))
            ax.scatter(x, skill, color=MODEL_COLOUR[model], alpha=.75)
            ax.plot([mi - .16, mi + .16], [skill.mean()] * 2,
                    color=MODEL_COLOUR[model], lw=3)
        ax.axhline(1, color="k", ls="--", lw=.8)
        ax.set_xticks(range(3), [MODEL_LABEL[m] for m in MODEL_NAMES], rotation=20)
        ax.set_title(f"Trained with {loss}")
        finite_skill = []
        for model in MODEL_NAMES:
            real = s[(s.model == model) & (~s["null"].astype(bool))].set_index("mouse")
            null = s[(s.model == model) & (s["null"].astype(bool))].set_index("mouse")
            common = real.index.intersection(null.index)
            finite_skill.extend((real.loc[common, "KL"] / null.loc[common, "KL"]).tolist())
        upper = max(1.1, np.nanmax(finite_skill) * 1.05) if finite_skill else 1.1
        ax.set_ylim(0.5, upper)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Held-out KL / within-condition-null KL\n(one point per mouse; <1 better)")
    fig.suptitle("Real V1: trial-specific Q decoding beyond stimulus condition")
    save_analysis_fig(fig, fig_root, "real_null_normalised_kl")

    # Prespecified paired model contrasts for the KL-trained headline.
    h = sub[(sub.loss == "KL") & (~sub["null"].astype(bool))]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), sharey=True)
    contrasts = [("deepsets", "mean", "DeepSets - mean"),
                 ("deepsets", "moments", "DeepSets - moments")]
    for ax, (a, b, title) in zip(axes, contrasts):
        aa = h[h.model == a].set_index("mouse")
        bb = h[h.model == b].set_index("mouse")
        common = aa.index.intersection(bb.index)
        delta = aa.loc[common, "KL"] - bb.loc[common, "KL"]
        ax.axhline(0, color="k", ls="--", lw=.8)
        ax.scatter(common, delta, color=MODEL_COLOUR[a], s=38)
        ax.plot(common, delta, color=MODEL_COLOUR[a], alpha=.35)
        ax.set(title=title, xlabel="Mouse", ylabel="Delta held-out KL (negative favours DeepSets)")
    fig.suptitle("Prespecified mouse-level contrasts (KL training)")
    save_analysis_fig(fig, fig_root, "real_headline_model_contrasts")

    # Put neural decoder losses beside the leakage-safe stimulus-condition oracle.
    # This exposes how little residual Q variation remains after condition is known.
    cond = summary[(summary.domain == "real") &
                   (summary.model == "condition_mean")].copy()
    if not cond.empty:
        cond["mouse"] = cond["identity"].str.extract(r"(\d+)").astype(int)
        neural = sub[(sub.loss == "KL") & (~sub["null"].astype(bool))]
        labels = [MODEL_LABEL[m] for m in MODEL_NAMES] + ["Condition mean"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for mouse in sorted(neural.mouse.unique()):
            vals = []
            for model in MODEL_NAMES:
                vals.append(float(neural[(neural.mouse == mouse) &
                                         (neural.model == model)].KL.iloc[0]))
            vals.append(float(cond[cond.mouse == mouse].KL.iloc[0]))
            ax.plot(range(4), vals, "-o", alpha=.65, label=f"Mouse {mouse}")
        ax.set_yscale("log")
        ax.set_xticks(range(4), labels, rotation=15)
        ax.set_ylabel("Held-out KL(Q || prediction), log scale")
        ax.set_title("Neural decoders versus stimulus-condition oracle")
        ax.grid(axis="y", alpha=.2)
        ax.legend(ncol=2, fontsize=8)
        save_analysis_fig(fig, fig_root, "real_vs_condition_mean_oracle")

    # Cross-loss common-metric scorecard, each real model divided by its own
    # matched within-condition null. Values <1 indicate trial-specific signal.
    metric_specs = [("KL", "KL / null KL"),
                    ("variance_abs_deg2", "Width error / null width error")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, (metric, title) in zip(axes, metric_specs):
        mat = np.full((len(MODEL_NAMES), len(losses)), np.nan)
        for i, model in enumerate(MODEL_NAMES):
            for j, loss in enumerate(losses):
                ss = sub[(sub.model == model) & (sub.loss == loss)]
                rr = ss[~ss["null"].astype(bool)].set_index("mouse")
                nn = ss[ss["null"].astype(bool)].set_index("mouse")
                common = rr.index.intersection(nn.index)
                if len(common):
                    mat[i, j] = np.mean(rr.loc[common, metric] / nn.loc[common, metric])
        im = ax.imshow(mat, vmin=.5, vmax=1.2, cmap="RdBu_r", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(losses)), losses)
        ax.set_yticks(range(3), [MODEL_LABEL[m] for m in MODEL_NAMES])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=.8, label="real / null")
    fig.suptitle("Common held-out metrics across training losses")
    save_analysis_fig(fig, fig_root, "real_common_metric_scorecard")

    # Width calibration for the prespecified KL-trained analysis.
    trials = pd.read_csv(root / "trial_metrics.csv")
    t = trials[(trials.domain == "real") & (trials.loss == "KL") &
               (~trials["null"].astype(bool)) & trials.model.isin(MODEL_NAMES)]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True, sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        ss = t[t.model == model]
        ax.hexbin(ss.target_variance_deg2, ss.pred_variance_deg2,
                  gridsize=28, mincnt=1, cmap="viridis")
        hi = max(ss.target_variance_deg2.max(), ss.pred_variance_deg2.max())
        ax.plot([0, hi], [0, hi], "w--", lw=1)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("Target Q variance (deg^2)")
    axes[0].set_ylabel("Decoded Q variance (deg^2)")
    fig.suptitle("Real V1 posterior-width calibration (KL-trained)")
    save_analysis_fig(fig, fig_root, "real_width_calibration")

    # Representative held-out posterior curves for Mouse 0, KL training.
    paths = {m: _job_path(root, "real", "mouse_0", "Q_half_100ms", m, "KL", False)
             for m in MODEL_NAMES}
    if all(p.exists() for p in paths.values()):
        loaded = {m: load_result(p)[0] for m, p in paths.items()}
        target = loaded["mean"]["target"]
        pm = posterior_metrics(target, target)
        order = np.argsort(pm.target_variance_deg2.values)
        picks = order[[len(order) // 10, len(order) // 2, 9 * len(order) // 10]]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
        grid = np.arange(target.shape[1])
        for ax, idx in zip(axes, picks):
            ax.plot(grid, target[idx], color="black", lw=2.2, label="IO target")
            for model in MODEL_NAMES:
                ax.plot(grid, loaded[model]["pred"][idx], lw=1.3,
                        color=MODEL_COLOUR[model], label=MODEL_LABEL[model])
            ax.set_title(f"target variance={pm.target_variance_deg2.iloc[idx]:.0f} deg^2")
            ax.set_xlabel("Orientation (deg)")
        axes[0].set_ylabel("Probability")
        axes[-1].legend(fontsize=8)
        fig.suptitle("Representative held-out Q predictions (Mouse 0; KL-trained)")
        save_analysis_fig(fig, fig_root, "real_example_posteriors_mouse0")


def plot_all(args):
    root, fig_root = Path(args.results_root), Path(args.fig_root)
    if not (root / "summary_metrics.csv").exists():
        aggregate(root)
    plot_synthetic(root, fig_root)
    plot_real(root, fig_root)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["synthetic", "real", "plot", "all"])
    p.add_argument("--results-root", default=str(DEFAULT_ROOT))
    p.add_argument("--fig-root", default=str(DEFAULT_FIG))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--losses", nargs="+", choices=LOSS_NAMES, default=list(LOSS_NAMES))
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", default="0:10")
    p.add_argument("--n-synthetic", type=int, default=600)
    p.add_argument("--synthetic-neurons", type=int, default=16)
    p.add_argument("--mouse-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(vars(args), indent=2))
    if args.mode in ("synthetic", "all"):
        run_synthetic(args)
    if args.mode in ("real", "all"):
        run_real(args)
    if args.mode in ("plot", "all"):
        plot_all(args)


if __name__ == "__main__":
    main()
