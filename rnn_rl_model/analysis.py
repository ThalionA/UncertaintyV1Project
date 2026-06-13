"""Scientific read-outs on a trained agent.

Three questions, all framework-facing:

1. **Template vs whitened.** Does the learned archetype-difference
   ``w_go - w_nogo`` point along the class-mean difference ``Delta mu`` (the
   template direction the framework predicts) rather than the covariance-whitened
   optimum ``Sigma^-1 Delta mu``? Reported as cosine alignment with each.

2. **Decision efficiency.** Network accuracy / neural-ideal-observer accuracy --
   how much of the optimal readout of its own V1 the agent captures.

3. **Evidence quality.** Pearson r between the agent's mean SI and the neural-IO
   log-posterior-odds (the quantity an optimal accumulator would integrate).

The fixed-vs-trained-V1 contrast is just these metrics computed for both
conditions.
"""

from __future__ import annotations

import numpy as np

from si_network_model.config import BOUNDARY_DEG
from si_network_model.ideal_observer import NeuralIdealObserver

from .evaluate import AgentV1


def _category_mask(eval_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """(go/nogo label, keep-mask) excluding the ambiguous 45-deg boundary."""
    ori = eval_dict["orientation"]
    keep = ori != BOUNDARY_DEG
    label = (ori < BOUNDARY_DEG).astype(int)  # 1 = Go
    return label, keep


def _shrunk_within_class_cov(X: np.ndarray, y: np.ndarray,
                             shrinkage: float = 0.10) -> np.ndarray:
    """Pooled within-class covariance, shrunk toward its diagonal."""
    centred = np.empty_like(X)
    for c in (0, 1):
        m = y == c
        centred[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    cov = centred.T @ centred / max(X.shape[0] - 2, 1)
    return (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def template_alignment(eval_dict: dict, w_go: np.ndarray,
                       w_nogo: np.ndarray) -> dict:
    """Cosine alignment of the learned readout with template vs whitened directions."""
    label, keep = _category_mask(eval_dict)
    X = eval_dict["r_bar"][keep]
    y = label[keep]
    dmu = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    Sigma = _shrunk_within_class_cov(X, y)
    w_white = np.linalg.solve(Sigma, dmu)
    w_diff = np.asarray(w_go) - np.asarray(w_nogo)
    return {
        "cos_readout_template": _cos(w_diff, dmu),
        "cos_readout_whitened": _cos(w_diff, w_white),
        "cos_template_whitened": _cos(dmu, w_white),
        "dmu_norm": float(np.linalg.norm(dmu)),
    }


def network_accuracy(eval_dict: dict) -> float:
    """Fraction-correct of the DDM choice over non-boundary trials."""
    label, keep = _category_mask(eval_dict)
    return float((eval_dict["choice"][keep] == label[keep]).mean())


def efficiency(eval_dict: dict, neural_io: NeuralIdealObserver) -> dict:
    """Network vs neural-IO accuracy and the SI-vs-optimal-log-odds correlation."""
    label, keep = _category_mask(eval_dict)
    r_bar = eval_dict["r_bar"]
    net_acc = float((eval_dict["choice"][keep] == label[keep]).mean())
    io_choice = neural_io.decide(r_bar)
    io_acc = float((io_choice[keep] == label[keep]).mean())
    log_odds = neural_io.log_odds(r_bar)
    si = eval_dict["mean_si"]
    finite = np.isfinite(si) & np.isfinite(log_odds)
    r = float(np.corrcoef(si[finite], log_odds[finite])[0, 1]) if finite.sum() > 2 else float("nan")
    return {
        "network_accuracy": net_acc,
        "neural_io_accuracy": io_acc,
        "decision_efficiency": net_acc / io_acc if io_acc > 0 else float("nan"),
        "si_vs_logodds_r": r,
    }


def fit_neural_io(v1, agent, base_cfg, seed: int) -> NeuralIdealObserver:
    """Fit the optimal Gaussian decoder on the agent's own V1 code."""
    return NeuralIdealObserver.fit(AgentV1(v1, agent), base_cfg,
                                   np.random.default_rng(seed))


def summarise(eval_dict: dict, w_go: np.ndarray, w_nogo: np.ndarray,
              neural_io: NeuralIdealObserver, hist: dict) -> dict:
    """Combine all metrics for one (animal, V1-condition) into a flat dict."""
    out = {}
    out.update(template_alignment(eval_dict, w_go, w_nogo))
    out.update(efficiency(eval_dict, neural_io))
    out["final_train_acc"] = float(np.mean(hist["acc"][-10:]))
    out["mean_abs_si"] = float(np.nanmean(np.abs(eval_dict["mean_si"])))
    # lapse: error rate on the easiest (max-contrast, low-dispersion) trials
    easy = (eval_dict["contrast"] >= 0.99) & (eval_dict["dispersion"] <= 5.0)
    label, _ = _category_mask(eval_dict)
    e = easy & (eval_dict["orientation"] != BOUNDARY_DEG)
    out["lapse_rate"] = float((eval_dict["choice"][e] != label[e]).mean()) if e.any() else float("nan")
    return out
