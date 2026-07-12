# -*- coding: utf-8 -*-
"""Leakage-safe DeepSets analysis for trialwise perceptual uncertainty.

The three decoders deliberately differ only in the temporal summary exposed to
the final posterior readout:

``mean``
    trial-mean population activity;
``moments``
    per-neuron temporal mean and population variance;
``deepsets``
    ``rho(mean_t phi(r_t))``, an order-invariant learned set summary.

DeepSets can test whether the *unordered set* of within-trial population states
contains information beyond the mean.  It cannot test temporal order and must
not be described as a sampling-code decoder.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


GRID_DEG = np.arange(91, dtype=np.float32)
MODEL_NAMES = ("mean", "moments", "deepsets")
LOSS_NAMES = ("KL", "JS", "MSE", "PCA")
EPS = 1e-8


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 42
    test_fraction: float = 0.5
    val_fraction: float = 0.2
    hidden_dim: int = 32
    phi_hidden: int = 16
    phi_dim: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    min_epochs: int = 20
    restarts: int = 3
    min_delta_fraction: float = 1e-4


@dataclass
class PreparedData:
    X_train: np.ndarray
    Q_train: np.ndarray
    C_train: np.ndarray
    X_val: np.ndarray
    Q_val: np.ndarray
    C_val: np.ndarray
    X_test: np.ndarray
    Q_test: np.ndarray
    C_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    pcs: np.ndarray
    evar: np.ndarray


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class MeanDecoder(nn.Module):
    def __init__(self, n_neurons: int, hidden_dim: int, n_bins: int = 91):
        super().__init__()
        self.readout = _MLP(n_neurons, hidden_dim, n_bins)

    def forward(self, x):
        return self.readout(x.mean(dim=1))


class MomentsDecoder(nn.Module):
    def __init__(self, n_neurons: int, hidden_dim: int, n_bins: int = 91):
        super().__init__()
        self.readout = _MLP(2 * n_neurons, hidden_dim, n_bins)

    def forward(self, x):
        summary = torch.cat([x.mean(dim=1), x.var(dim=1, unbiased=False)], dim=-1)
        return self.readout(summary)


class DeepSetsDecoder(nn.Module):
    def __init__(self, n_neurons: int, phi_hidden: int, phi_dim: int,
                 rho_hidden: int, n_bins: int = 91):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_neurons, phi_hidden), nn.Tanh(),
            nn.Linear(phi_hidden, phi_dim), nn.Tanh(),
        )
        self.rho = _MLP(phi_dim, rho_hidden, n_bins)
        for layer in self.phi.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.rho(self.phi(x).mean(dim=1))


def _nearest_mlp_width(input_dim: int, output_dim: int, target_params: int,
                       minimum: int = 2) -> int:
    # params = H*(input_dim + output_dim + 1) + output_dim
    h = round((target_params - output_dim) / (input_dim + output_dim + 1))
    return max(minimum, int(h))


def build_model(name: str, n_neurons: int, cfg: AnalysisConfig,
                n_bins: int = 91) -> nn.Module:
    """Build approximately parameter-matched models.

    DeepSets defines the reference budget.  Mean and moments widths are chosen
    analytically to be nearest to that budget; exact counts are saved with every
    result because integer hidden widths cannot generally match exactly.
    """
    ref = DeepSetsDecoder(n_neurons, cfg.phi_hidden, cfg.phi_dim,
                          cfg.hidden_dim, n_bins)
    budget = parameter_count(ref)
    if name == "deepsets":
        return ref
    if name == "mean":
        h = _nearest_mlp_width(n_neurons, n_bins, budget)
        return MeanDecoder(n_neurons, h, n_bins)
    if name == "moments":
        h = _nearest_mlp_width(2 * n_neurons, n_bins, budget)
        return MomentsDecoder(n_neurons, h, n_bins)
    raise ValueError(f"unknown model {name!r}; expected {MODEL_NAMES}")


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def loss_per_trial(logits: torch.Tensor, target: torch.Tensor, kind: str,
                   pcs: torch.Tensor | None = None,
                   evar: torch.Tensor | None = None) -> torch.Tensor:
    """Requested training objectives, with no silent fallback."""
    pred = probabilities(logits).clamp_min(torch.finfo(logits.dtype).eps)
    target = target.clamp_min(0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(EPS)
    if kind == "KL":
        return torch.sum(target * (torch.log(target.clamp_min(EPS)) - torch.log(pred)), dim=-1)
    if kind == "JS":
        m = 0.5 * (pred + target)
        kl_tm = torch.sum(target * (torch.log(target.clamp_min(EPS)) - torch.log(m)), dim=-1)
        kl_pm = torch.sum(pred * (torch.log(pred) - torch.log(m)), dim=-1)
        return 0.5 * (kl_tm + kl_pm)
    if kind == "MSE":
        # Proper multiclass Brier score.  This is the unweighted squared-error
        # objective; dividing by n_bins gives mean-MSE with the same optimum but
        # makes gradients 91x smaller for Q and confounds a shared-optimiser
        # loss comparison through Adam's epsilon / early stopping.
        return torch.sum((pred - target) ** 2, dim=-1)
    if kind == "PCA":
        if pcs is None or evar is None:
            raise ValueError("projection/PCA loss requires a training-fold basis")
        dp = pred @ pcs.T - target @ pcs.T
        return torch.sum(evar * dp.square(), dim=-1)
    raise ValueError(f"unknown loss {kind!r}; expected {LOSS_NAMES}")


def fit_projection_basis(q_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the all-trials projection basis on inner-training Q only."""
    pca = PCA().fit(np.asarray(q_train, dtype=np.float64))
    return (pca.components_.astype(np.float32),
            pca.explained_variance_ratio_.astype(np.float32))


def _joint_labels(conditions: np.ndarray) -> np.ndarray:
    _, labels = np.unique(np.asarray(conditions), axis=0, return_inverse=True)
    return labels


def _split(indices: np.ndarray, labels: np.ndarray, test_fraction: float,
           seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if (len(counts) > 0 and counts.min() >= 2 and
                          round(len(indices) * test_fraction) >= len(unique)) else None
    a, b = train_test_split(indices, test_size=test_fraction,
                            random_state=seed, shuffle=True, stratify=stratify)
    return np.sort(a), np.sort(b)


def prepare_arrays(X: np.ndarray, Q: np.ndarray, conditions: np.ndarray,
                   cfg: AnalysisConfig) -> PreparedData:
    """Outer test and inner validation splits, then train-only preprocessing."""
    X = np.asarray(X, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    conditions = np.asarray(conditions)
    if X.ndim != 3 or Q.ndim != 2 or len(X) != len(Q):
        raise ValueError("expected X=(trials,time,neurons), Q=(trials,bins)")
    all_idx = np.arange(len(X))
    labels = _joint_labels(conditions)
    outer_train, test_idx = _split(all_idx, labels, cfg.test_fraction, cfg.seed)
    inner_train_rel, val_rel = _split(
        np.arange(len(outer_train)), labels[outer_train], cfg.val_fraction,
        cfg.seed + 1)
    train_idx = outer_train[inner_train_rel]
    val_idx = outer_train[val_rel]

    z_mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
    z_std = X[train_idx].std(axis=(0, 1), keepdims=True)
    z_std = np.where(z_std < 1e-6, 1.0, z_std)
    Xz = (X - z_mean) / z_std
    pcs, evar = fit_projection_basis(Q[train_idx])
    return PreparedData(
        Xz[train_idx], Q[train_idx], conditions[train_idx],
        Xz[val_idx], Q[val_idx], conditions[val_idx],
        Xz[test_idx], Q[test_idx], conditions[test_idx],
        train_idx, val_idx, test_idx, z_mean, z_std, pcs, evar,
    )


def within_condition_shuffle(target: np.ndarray, conditions: np.ndarray,
                             seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Permute whole posterior rows within exact stimulus cells."""
    target = np.asarray(target)
    out = target.copy()
    perm = np.arange(len(target))
    rng = np.random.default_rng(seed)
    labels = _joint_labels(conditions)
    eligible = 0
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > 1:
            src = rng.permutation(idx)
            out[idx] = target[src]
            perm[idx] = src
            eligible += len(idx)
    return out, perm, eligible / max(len(target), 1)


def _to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def _mean_loss(model, X, Q, kind, pcs, evar, batch_size=256) -> float:
    model.eval()
    vals = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            vals.append(loss_per_trial(model(X[s:s + batch_size]), Q[s:s + batch_size],
                                       kind, pcs, evar))
    return float(torch.cat(vals).mean().item())


def train_model(name: str, loss_name: str, prepared: PreparedData,
                cfg: AnalysisConfig, null: bool = False,
                device: torch.device | None = None) -> dict:
    """Train restarts and select strictly on validation objective."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_train, q_val = prepared.Q_train, prepared.Q_val
    perm_train = np.arange(len(q_train))
    perm_val = np.arange(len(q_val))
    eligible_train = eligible_val = 1.0
    if null:
        q_train, perm_train, eligible_train = within_condition_shuffle(
            q_train, prepared.C_train, cfg.seed + 1001)
        q_val, perm_val, eligible_val = within_condition_shuffle(
            q_val, prepared.C_val, cfg.seed + 2001)

    Xtr = _to_tensor(prepared.X_train, device)
    Ytr = _to_tensor(q_train, device)
    Xva = _to_tensor(prepared.X_val, device)
    Yva = _to_tensor(q_val, device)
    Xte = _to_tensor(prepared.X_test, device)
    pcs = _to_tensor(prepared.pcs, device)
    evar = _to_tensor(prepared.evar, device)

    best = None
    restart_records = []
    for restart in range(cfg.restarts):
        run_seed = cfg.seed + 10000 * restart
        seed_everything(run_seed)
        model = build_model(name, prepared.X_train.shape[-1], cfg,
                            prepared.Q_train.shape[-1]).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(run_seed + 17)
        best_val = math.inf
        best_state = None
        best_epoch = -1
        stale = 0
        history = {"train": [], "val": []}
        for epoch in range(cfg.max_epochs):
            model.train()
            order = torch.randperm(len(Xtr), generator=generator).to(device)
            for s in range(0, len(order), cfg.batch_size):
                idx = order[s:s + cfg.batch_size]
                loss = loss_per_trial(model(Xtr[idx]), Ytr[idx], loss_name,
                                      pcs, evar).mean()
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            train_value = _mean_loss(model, Xtr, Ytr, loss_name, pcs, evar)
            val_value = _mean_loss(model, Xva, Yva, loss_name, pcs, evar)
            history["train"].append(train_value)
            history["val"].append(val_value)
            threshold = (cfg.min_delta_fraction * max(abs(best_val), 1e-12)
                         if np.isfinite(best_val) else 0.0)
            if best_state is None or val_value < best_val - threshold:
                best_val = val_value
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
                break
        model.load_state_dict(best_state)
        rec = {"restart": restart, "seed": run_seed, "val_loss": best_val,
               "best_epoch": best_epoch, "history": history,
               "state_dict": copy.deepcopy(model.state_dict()),
               "n_params": parameter_count(model)}
        restart_records.append(rec)
        if best is None or best_val < best["val_loss"]:
            best = rec

    model = build_model(name, prepared.X_train.shape[-1], cfg,
                        prepared.Q_train.shape[-1]).to(device)
    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = probabilities(model(Xte)).cpu().numpy()
    return {
        "pred": pred,
        "target": prepared.Q_test.copy(),
        "conditions": prepared.C_test.copy(),
        "best_val_loss": float(best["val_loss"]),
        "best_epoch": int(best["best_epoch"]),
        "best_restart": int(best["restart"]),
        "n_params": int(best["n_params"]),
        "history_train": np.asarray(best["history"]["train"], dtype=np.float32),
        "history_val": np.asarray(best["history"]["val"], dtype=np.float32),
        "perm_train": perm_train,
        "perm_val": perm_val,
        "eligible_train": float(eligible_train),
        "eligible_val": float(eligible_val),
        "pcs": prepared.pcs,
        "evar": prepared.evar,
        "z_mean": prepared.z_mean,
        "z_std": prepared.z_std,
    }


def posterior_metrics(pred: np.ndarray, target: np.ndarray,
                      pcs: np.ndarray | None = None,
                      evar: np.ndarray | None = None) -> pd.DataFrame:
    pred = np.clip(np.asarray(pred, float), EPS, None)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.clip(np.asarray(target, float), EPS, None)
    target /= target.sum(axis=1, keepdims=True)
    m = 0.5 * (pred + target)
    kl = np.sum(target * (np.log(target) - np.log(pred)), axis=1)
    js = 0.5 * (np.sum(target * (np.log(target) - np.log(m)), axis=1) +
                np.sum(pred * (np.log(pred) - np.log(m)), axis=1))
    brier = np.sum((pred - target) ** 2, axis=1)
    grid = GRID_DEG[:pred.shape[1]].astype(float)
    pred_mean = pred @ grid
    targ_mean = target @ grid
    pred_var = np.sum(pred * (grid[None, :] - pred_mean[:, None]) ** 2, axis=1)
    targ_var = np.sum(target * (grid[None, :] - targ_mean[:, None]) ** 2, axis=1)
    pred_ent = -np.sum(pred * np.log(pred), axis=1)
    targ_ent = -np.sum(target * np.log(target), axis=1)
    data = {
        "KL": kl, "JS": js, "Brier": brier,
        "mean_abs_deg": np.abs(pred_mean - targ_mean),
        "variance_abs_deg2": np.abs(pred_var - targ_var),
        "entropy_abs_nats": np.abs(pred_ent - targ_ent),
        "pred_mean_deg": pred_mean, "target_mean_deg": targ_mean,
        "pred_variance_deg2": pred_var, "target_variance_deg2": targ_var,
        "pred_entropy_nats": pred_ent, "target_entropy_nats": targ_ent,
    }
    if pcs is not None and evar is not None:
        dp = pred @ pcs.T - target @ pcs.T
        data["Projection"] = np.sum(evar * dp ** 2, axis=1)
    return pd.DataFrame(data)


def condition_mean_prediction(q_train: np.ndarray, c_train: np.ndarray,
                              c_test: np.ndarray) -> np.ndarray:
    global_mean = np.mean(q_train, axis=0)
    lookup = {}
    for row in np.unique(c_train, axis=0):
        mask = np.all(c_train == row, axis=1)
        lookup[tuple(row.tolist())] = np.mean(q_train[mask], axis=0)
    return np.stack([lookup.get(tuple(row.tolist()), global_mean) for row in c_test])


def atomic_save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem, suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_result(path: str | Path, result: dict, metadata: dict) -> None:
    arrays = {k: v for k, v in result.items()
              if isinstance(v, (np.ndarray, float, int, np.number))}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    atomic_save_npz(path, **arrays)


def load_result(path: str | Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files if k != "metadata_json"}
    meta = json.loads(str(z["metadata_json"]))
    return data, meta


def gaussian_posterior(mu: np.ndarray, sigma: np.ndarray,
                       grid: np.ndarray = GRID_DEG) -> np.ndarray:
    mu = np.asarray(mu, float)[:, None]
    sigma = np.asarray(sigma, float)[:, None]
    q = np.exp(-0.5 * ((grid[None, :] - mu) / sigma) ** 2)
    return (q / q.sum(axis=1, keepdims=True)).astype(np.float32)


def _centred_unit_rms(z: np.ndarray) -> np.ndarray:
    z = z - z.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(z ** 2, axis=1, keepdims=True))
    return z / np.maximum(rms, 1e-6)


def make_synthetic(regime: str, seed: int, n_base: int = 600,
                   t_bins: int = 10, n_neurons: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create distributional targets with controlled mean/variance/order codes."""
    rng = np.random.default_rng(seed)
    mu_levels = np.asarray([20, 30, 40, 50, 60, 70], float)
    sigma_levels = np.asarray([3, 5, 8, 12, 18], float)
    mu_idx = rng.integers(0, len(mu_levels), n_base)
    u_idx = rng.integers(0, len(sigma_levels), n_base)
    mu = mu_levels[mu_idx]
    sigma = sigma_levels[u_idx]
    q = gaussian_posterior(mu, sigma)
    X = np.zeros((n_base, t_bins, n_neurons), dtype=np.float32)

    # Smooth signed centre code, shared by all regimes and constant over time.
    angles = np.linspace(0.3, 2.7, min(4, n_neurons))
    centre = np.stack([np.sin(mu / 90 * np.pi * a) for a in angles], axis=1)
    X[:, :, :len(angles)] = centre[:, None, :]
    nuisance = _centred_unit_rms(rng.normal(size=(n_base, t_bins, n_neurons - len(angles))))
    X[:, :, len(angles):] = 0.15 * nuisance

    u_scaled = (u_idx - u_idx.mean()) / max(u_idx.std(), 1e-6)
    code_channel = min(4, n_neurons - 1)
    transition_stat = np.zeros(n_base, dtype=float)
    order_features = np.zeros((n_base, 4), dtype=float)
    if regime == "mean":
        X[:, :, code_channel] += u_scaled[:, None]
    elif regime == "variance":
        z = _centred_unit_rms(rng.normal(size=(n_base, t_bins, 1)))[:, :, 0]
        amp = 0.35 + 0.35 * u_idx
        X[:, :, code_channel] = amp[:, None] * z
    elif regime == "order":
        # A fixed multiset per trial, permuted according to uncertainty level.
        # Five counterfactual trials share the exact same state set and centre;
        # every invariant summary is therefore byte-identical within a group.
        perms = [
            np.arange(t_bins), np.arange(t_bins)[::-1],
            np.roll(np.arange(t_bins), 2),
            np.r_[np.arange(0, t_bins, 2), np.arange(1, t_bins, 2)],
            np.argsort(np.sin(np.arange(t_bins) * 1.7)),
        ]
        n_groups = max(1, n_base // len(sigma_levels))
        n_base = n_groups * len(sigma_levels)
        group_mu_idx = rng.integers(0, len(mu_levels), n_groups)
        mu_idx = np.repeat(group_mu_idx, len(sigma_levels))
        u_idx = np.tile(np.arange(len(sigma_levels)), n_groups)
        mu = mu_levels[mu_idx]
        sigma = sigma_levels[u_idx]
        q = gaussian_posterior(mu, sigma)
        centre = np.stack([np.sin(mu / 90 * np.pi * a) for a in angles], axis=1)
        base_group = _centred_unit_rms(
            rng.normal(size=(n_groups, t_bins, n_neurons))).astype(np.float32)
        ramp = np.linspace(-1.0, 1.0, t_bins, dtype=np.float32)
        base_group[:, :, code_channel] = ramp[None, :]
        X = np.empty((n_base, t_bins, n_neurons), dtype=np.float32)
        transition_stat = np.zeros(n_base, dtype=float)
        for i in range(n_base):
            g = i // len(sigma_levels)
            states = base_group[g].copy()
            states[:, :len(angles)] = centre[i][None, :]
            X[i] = states[perms[u_idx[i]]]
            transition_stat[i] = np.sum(np.diff(X[i], axis=0) ** 2)
            trace = X[i, :, code_channel]
            order_features[i] = [trace[0], trace[-1],
                                 np.dot(trace, np.linspace(-1, 1, t_bins)),
                                 transition_stat[i]]
    else:
        raise ValueError("regime must be 'mean', 'variance', or 'order'")

    conditions = np.column_stack([mu_idx, u_idx, np.zeros(n_base, dtype=int)])
    audit = {
        "mu": mu, "sigma": sigma, "u_idx": u_idx,
        "transition_stat": transition_stat,
        "order_features": order_features,
        "temporal_mean": X.mean(axis=1),
        "temporal_variance": X.var(axis=1),
    }
    return X, q, conditions, audit


def config_dict(cfg: AnalysisConfig) -> dict:
    return asdict(cfg)
