"""Analysis metrics for the SI-framework network model.

Pure functions that turn a trained cohort into the numbers the figures and the
findings need: learning curves, psychometrics, decision efficiency, SI-vs-IO
agreement, and the framework's cross-animal quantities (bundle concentration
kappa, lapse rate, trial-mean-SI spread).
"""

from __future__ import annotations

import numpy as np

from .config import ARCHETYPE_PAIR, BOUNDARY_DEG, CONTRAST_DISPERSION_PAIRS
from .ideal_observer import NeuralIdealObserver, StimulusIdealObserver
from .si_variants import VARIANT_NAMES

_ARCH_C, _ARCH_D = ARCHETYPE_PAIR

# Contrast difficulty tiers used for stratified analyses.
CONTRAST_TIERS: dict[str, tuple[float, ...]] = {
    "easy": (1.0,),
    "medium": (0.5, 0.25),
    "hard": (0.01,),
}


def category(orientation: np.ndarray) -> np.ndarray:
    """1 = Go (orientation < 45 deg), 0 = NoGo."""
    return (np.asarray(orientation) < BOUNDARY_DEG).astype(int)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


def rolling_accuracy(correct: np.ndarray, window: int = 120) -> np.ndarray:
    """Rolling-mean accuracy over a training trial sequence."""
    correct = np.asarray(correct, dtype=float)
    window = min(window, max(1, len(correct) // 3))
    kernel = np.ones(window) / window
    return np.convolve(correct, kernel, mode="valid")


def learning_timescale(correct: np.ndarray) -> float:
    """Trials to first reach 80 % rolling accuracy (a learning-speed proxy)."""
    roll = rolling_accuracy(correct, window=120)
    hit = np.where(roll >= 0.80)[0]
    return float(hit[0]) if len(hit) else float(len(roll))


# ---------------------------------------------------------------------------
# Psychometric curves
# ---------------------------------------------------------------------------


def psychometric(orientation: np.ndarray, p_go: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(Go) at each distinct orientation level."""
    levels = np.array(sorted(set(np.round(orientation, 3))))
    curve = np.array([p_go[orientation == o].mean() for o in levels])
    return levels, curve


def network_accuracy(eval_data: dict, choice: np.ndarray | None = None) -> float:
    """Fraction-correct over non-boundary evaluation trials."""
    ori = eval_data["orientation"]
    keep = ori != BOUNDARY_DEG
    ch = eval_data["choice"] if choice is None else choice
    return float((ch[keep] == category(ori)[keep]).mean())


def per_pair_accuracy(eval_data: dict, neural_io: NeuralIdealObserver,
                      stimulus_io: StimulusIdealObserver) -> dict:
    """Per (contrast, dispersion) pair: network / neural-IO / stimulus-IO accuracy."""
    ori = eval_data["orientation"]
    c = eval_data["contrast"]
    d = eval_data["dispersion"]
    cat = category(ori)
    net_choice = eval_data["choice"]
    nio_choice = neural_io.decide(eval_data["r_bar"])
    out = {}
    for contrast, dispersion in CONTRAST_DISPERSION_PAIRS:
        m = (c == contrast) & (d == dispersion) & (ori != BOUNDARY_DEG)
        sio = [
            (lambda pg, o: pg if o < BOUNDARY_DEG else 1.0 - pg)(
                stimulus_io.p_go(o, contrast, dispersion), o)
            for o in np.unique(ori[m])
        ]
        out[(contrast, dispersion)] = {
            "network": float((net_choice[m] == cat[m]).mean()),
            "neural_io": float((nio_choice[m] == cat[m]).mean()),
            "stimulus_io": float(np.mean(sio)),
            "jitter": float(contrast),
        }
    return out


# ---------------------------------------------------------------------------
# Decision efficiency and SI-vs-IO agreement
# ---------------------------------------------------------------------------


def decision_efficiency(eval_data: dict, neural_io: NeuralIdealObserver) -> dict:
    """Network accuracy vs the two optimal observers."""
    net = network_accuracy(eval_data)
    nio = network_accuracy(eval_data, choice=neural_io.decide(eval_data["r_bar"]))
    return {"network": net, "neural_io": nio,
            "efficiency": net / nio if nio > 0 else float("nan")}


def si_io_agreement(eval_data: dict, neural_io: NeuralIdealObserver) -> dict:
    """How well the network's evidence tracks the optimal log-posterior-odds.

    Compares the network's trial evidence E_t[SI] to the neural-IO's log-odds,
    and the network's choices to the neural-IO's choices, on non-boundary
    evaluation trials.
    """
    keep = eval_data["orientation"] != BOUNDARY_DEG
    si = eval_data["mean_si"][keep]
    io_log_odds = neural_io.log_odds(eval_data["r_bar"])[keep]
    r = float(np.corrcoef(si, io_log_odds)[0, 1])
    choice_match = float((eval_data["choice"][keep]
                          == neural_io.decide(eval_data["r_bar"])[keep]).mean())
    return {"si_logodds_r": r, "choice_agreement": choice_match}


# ---------------------------------------------------------------------------
# Framework cross-animal quantities
# ---------------------------------------------------------------------------


def _archetype_mask(eval_data: dict, go: bool) -> np.ndarray:
    """Boolean mask of archetype trials (c=1, d=5) for one category."""
    ori_target = 0.0 if go else 90.0
    return ((eval_data["contrast"] == _ARCH_C)
            & (eval_data["dispersion"] == _ARCH_D)
            & (eval_data["orientation"] == ori_target))


def bundle_concentration(eval_data: dict) -> float:
    """Bundle concentration kappa: tightness of the V1 r_bar bundle.

    Computed on archetype trials (fixed stimulus, so the spread is pure
    trial-to-trial bundle width). For each archetype class the trial-mean V1
    vectors are unit-normalised; the mean cosine to the class-mean direction
    gives a width, and kappa is its reciprocal. Higher kappa = tighter bundle.
    """
    widths = []
    for go in (True, False):
        r_bar = eval_data["r_bar"][_archetype_mask(eval_data, go)].astype(float)
        if len(r_bar) < 5:
            continue
        unit = r_bar / (np.linalg.norm(r_bar, axis=1, keepdims=True) + 1e-12)
        mean_dir = unit.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir) + 1e-12
        cos = np.clip(unit @ mean_dir, -1.0, 1.0)
        widths.append(float(np.mean(1.0 - cos)))
    width = float(np.mean(widths)) if widths else float("nan")
    return 1.0 / (width + 1e-9)


def lapse_rate(eval_data: dict) -> float:
    """Error rate on high-contrast stimuli at extreme (far-from-boundary)
    orientations -- the psychometric-asymptote lapse, the cleanest behavioural
    read-out of bundle width (framework Prediction 15).

    Measured on the full contrast-1.0 tier (dispersions 5/30/90) at the
    extreme orientations 0/15/75/90, so it has genuine cross-animal variance
    (pure archetypes are too easy -- every animal scores ~0 there)."""
    mask = ((eval_data["contrast"] == 1.0)
            & np.isin(eval_data["orientation"], [0.0, 15.0, 75.0, 90.0]))
    ch = eval_data["choice"][mask]
    cat = category(eval_data["orientation"][mask])
    return float(np.mean(ch != cat))


def trial_mean_si_spread(eval_data: dict) -> float:
    """Var_trial[E_t SI] on archetype trials -- the framework's neural proxy
    for across-trial drift variability s_v (framework Prediction 4)."""
    spreads = []
    for go in (True, False):
        si = eval_data["mean_si"][_archetype_mask(eval_data, go)]
        if len(si) >= 5:
            spreads.append(float(np.var(si)))
    return float(np.mean(spreads)) if spreads else float("nan")


# ---------------------------------------------------------------------------
# SI-variant comparison
# ---------------------------------------------------------------------------


def variant_metrics(eval_data: dict, neural_io: NeuralIdealObserver) -> dict:
    """Per-variant accuracy, decision efficiency and SI-vs-IO correlation."""
    ori = eval_data["orientation"]
    keep = ori != BOUNDARY_DEG
    cat = category(ori)
    io_choice = neural_io.decide(eval_data["r_bar"])
    io_acc = float((io_choice[keep] == cat[keep]).mean())
    io_log_odds = neural_io.log_odds(eval_data["r_bar"])[keep]
    out = {}
    for variant in VARIANT_NAMES:
        ch = eval_data[f"choice_{variant}"]
        acc = float((ch[keep] == cat[keep]).mean())
        si = eval_data[f"mean_si_{variant}"][keep]
        out[variant] = {
            "accuracy": acc,
            "efficiency": acc / io_acc if io_acc > 0 else float("nan"),
            "si_io_r": float(np.corrcoef(si, io_log_odds)[0, 1]),
        }
    return out


def variant_accuracy_by_tier(eval_data: dict) -> dict:
    """Per-variant accuracy within each contrast difficulty tier."""
    ori = eval_data["orientation"]
    cat = category(ori)
    contrast = eval_data["contrast"]
    out = {variant: {} for variant in VARIANT_NAMES}
    for tier, contrasts in CONTRAST_TIERS.items():
        m = np.isin(contrast, contrasts) & (ori != BOUNDARY_DEG)
        for variant in VARIANT_NAMES:
            ch = eval_data[f"choice_{variant}"]
            out[variant][tier] = float((ch[m] == cat[m]).mean())
    return out


# ---------------------------------------------------------------------------
# Psychometric criterion (for the symmetric-vs-asymmetric comparison)
# ---------------------------------------------------------------------------


def psychometric_criterion(orientation: np.ndarray, p_go: np.ndarray) -> float:
    """Orientation at which P(Go) crosses 0.5 (linear interpolation)."""
    levels, curve = psychometric(orientation, p_go)
    for i in range(len(levels) - 1):
        if (curve[i] - 0.5) * (curve[i + 1] - 0.5) <= 0:
            frac = (0.5 - curve[i]) / (curve[i + 1] - curve[i] + 1e-12)
            return float(levels[i] + frac * (levels[i + 1] - levels[i]))
    return float("nan")


# ---------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------


def summarize_cohort(cohort) -> dict:
    """Compute every per-animal and pooled metric used by the figures."""
    per_animal = []
    for run in cohort.runs:
        ev, nio = run.eval, run.neural_io
        eff = decision_efficiency(ev, nio)
        agree = si_io_agreement(ev, nio)
        per_animal.append({
            "animal_id": run.animal_id,
            "traits": run.traits,
            "phase1_length": run.phase1_length,
            "network_acc": eff["network"],
            "neural_io_acc": eff["neural_io"],
            "efficiency": eff["efficiency"],
            "si_logodds_r": agree["si_logodds_r"],
            "choice_agreement": agree["choice_agreement"],
            "kappa": bundle_concentration(ev),
            "lapse_rate": lapse_rate(ev),
            "si_spread": trial_mean_si_spread(ev),
            "learning_timescale": learning_timescale(run.train_log["correct"]),
            "variant_metrics": variant_metrics(ev, nio),
            "variant_acc_by_tier": variant_accuracy_by_tier(ev),
        })
    # Cohort-mean efficiency per SI variant.
    variant_summary = {
        variant: {
            "efficiency": float(np.mean([a["variant_metrics"][variant]["efficiency"]
                                         for a in per_animal])),
            "accuracy": float(np.mean([a["variant_metrics"][variant]["accuracy"]
                                       for a in per_animal])),
            "si_io_r": float(np.mean([a["variant_metrics"][variant]["si_io_r"]
                                      for a in per_animal])),
        }
        for variant in VARIANT_NAMES
    }
    return {
        "per_animal": per_animal,
        "variant_summary": variant_summary,
        "cohort_efficiency": float(np.mean([a["efficiency"] for a in per_animal])),
        "cohort_network_acc": float(np.mean([a["network_acc"] for a in per_animal])),
        "cohort_neural_io_acc": float(np.mean([a["neural_io_acc"] for a in per_animal])),
        "cohort_si_logodds_r": float(np.mean([a["si_logodds_r"] for a in per_animal])),
    }
