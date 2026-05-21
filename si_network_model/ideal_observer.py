"""Optimal Bayesian observers to compare the network against.

Two observers, two ceilings (Decision 3 in UNDERSTANDING.md):

* **Stimulus-IO** -- the behavioural ceiling. A Bayesian observer with direct
  but noisy access to the latent orientation, sensory precision
  ``kappa(contrast, dispersion)``. Reuses the project's v2 ideal observer
  (``ideal_observer/io_hmm/io_core.py``).

* **Neural-IO** -- the readout-efficiency ceiling. The optimal Bayesian
  decoder of the network's *own* V1 activity r(t): a Gaussian class-
  conditional model (per-condition means + a shrinkage-regularised pooled
  covariance), which is Bayes-optimal under the equal-covariance Gaussian
  assumption.

Both observers decide with a reward-aware threshold on the log-posterior-odds
(``decision_threshold``): 0 for symmetric reward (the unbiased 45-deg rule),
shifted for asymmetric reward so the comparison stays fair.
"""

from __future__ import annotations

import sys

import numpy as np

from . import stimuli
from .config import Config
from .paths import IO_HMM_DIR

# Reuse the project's v2 ideal observer core (numpy-only, no scipy).
if str(IO_HMM_DIR) not in sys.path:
    sys.path.insert(0, str(IO_HMM_DIR))
import io_core  # noqa: E402  (path shim must run first)


# ---------------------------------------------------------------------------
# Reward-aware decision threshold (shared by both observers)
# ---------------------------------------------------------------------------


def decision_threshold(cfg: Config) -> float:
    """Log-odds threshold above which the reward-maximising choice is Go.

    Bayesian decision theory: decide Go iff the expected reward of Go beats
    NoGo, which reduces to ``log P(Go)/P(NoGo) > log[(r_cr - r_fa)/(r_hit -
    r_miss)]``. For symmetric reward this is exactly 0 (the unbiased rule);
    for asymmetric water-task reward it shifts negative -> a Go-bias.
    """
    if cfg.reward_mode == "symmetric":
        r_hit = r_cr = cfg.reward_correct
        r_fa = r_miss = cfg.reward_error
    else:
        r_hit, r_fa = cfg.reward_hit, cfg.reward_false_alarm
        r_cr, r_miss = cfg.reward_correct_reject, cfg.reward_miss
    return float(np.log((r_cr - r_fa) / (r_hit - r_miss)))


# ---------------------------------------------------------------------------
# Stimulus-IO -- Bayesian over the latent orientation
# ---------------------------------------------------------------------------


class StimulusIdealObserver:
    """Optimal observer with noisy direct access to the latent orientation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.grids = io_core.IOGrids.default()
        self.prior = io_core.prior_flat(self.grids)
        self.threshold = decision_threshold(cfg)

    def kappa(self, contrast: float, dispersion: float) -> float:
        """Sensory precision for a (contrast, dispersion) pair."""
        return float(io_core.kappa_for_trial(
            0.0, contrast, dispersion,
            kappa_amp=self.cfg.io_kappa_amp,
            c_power=self.cfg.io_c_power,
            d_power=self.cfg.io_d_power,
            kappa_min=self.cfg.io_kappa_min,
        ))

    def _log_odds_grid(self, kappa: float) -> np.ndarray:
        """g(m) = log P(Go|m)/P(NoGo|m) over the measurement grid."""
        post = io_core.posterior_s_given_m(self.grids, kappa, self.prior)
        return io_core.log_posterior_odds(self.grids, post)

    def p_go(self, orientation: float, contrast: float, dispersion: float) -> float:
        """Analytic P(choose Go | stimulus), marginalising over measurements."""
        kappa = self.kappa(contrast, dispersion)
        g = self._log_odds_grid(kappa)
        p_m = io_core.p_m_given_s(self.grids, kappa, orientation)
        decide_go = g > self.threshold
        return float(np.sum(p_m[decide_go]))

    def psychometric(self, conditions: list[stimuli.Condition]) -> np.ndarray:
        """P(Go) for each condition, aligned to ``conditions``."""
        return np.array([
            self.p_go(c.orientation_deg, c.contrast, c.dispersion_deg)
            for c in conditions
        ])

    def accuracy(self, conditions: list[stimuli.Condition]) -> float:
        """Expected fraction correct over the given conditions (non-boundary)."""
        acc = []
        for c in conditions:
            if c.is_boundary:
                continue
            pgo = self.p_go(c.orientation_deg, c.contrast, c.dispersion_deg)
            acc.append(pgo if c.category == 1 else 1.0 - pgo)
        return float(np.mean(acc))


# ---------------------------------------------------------------------------
# Neural-IO -- optimal Gaussian decoder of the network's V1 activity
# ---------------------------------------------------------------------------


def _go_weight(condition: stimuli.Condition) -> float:
    """1.0 for a Go condition, 0.0 for NoGo, 0.5 for the 45-deg boundary."""
    if condition.is_boundary:
        return 0.5
    return 1.0 if condition.category == 1 else 0.0


class NeuralIdealObserver:
    """Optimal Bayesian classifier of trial-mean V1 activity r_bar.

    Built by sampling the network's own V1 responses, so it is the best
    possible readout of exactly the neural code the network has to work with.
    """

    def __init__(self, cond_means: np.ndarray, cov_inv: np.ndarray,
                 conditions: list[stimuli.Condition], cfg: Config):
        self.cfg = cfg
        self.conditions = conditions
        self.threshold = decision_threshold(cfg)
        self.go_weight = np.array([_go_weight(c) for c in conditions])
        # Precompute the linear-discriminant terms (shared-covariance Gaussian):
        #   log p(cond|x)  ==  x @ A_cond - 0.5 * b_cond  + const.
        self._A = cov_inv @ cond_means.T          # (n_v1, n_cond)
        self._b = np.einsum("cn,nc->c", cond_means, self._A)  # (n_cond,)

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, v1_model, cfg: Config,
            rng: np.random.Generator) -> "NeuralIdealObserver":
        """Sample V1 responses for every condition and fit the Gaussian model.

        Sampling goes through ``v1_model.present``, so the per-condition
        samples include the per-trial orientation jitter -- the neural-IO is
        therefore correctly limited by exactly the noise the network faces.
        """
        conditions = stimuli.enumerate_conditions()
        n_samp = cfg.io_samples_per_cond
        cond_means = np.empty((len(conditions), cfg.n_v1))
        centred = np.empty((len(conditions) * n_samp, cfg.n_v1))

        for i, cond in enumerate(conditions):
            r_bar = v1_model.present([cond] * n_samp, rng).mean(axis=1)
            mu = r_bar.mean(axis=0)
            cond_means[i] = mu
            centred[i * n_samp:(i + 1) * n_samp] = r_bar - mu

        # Pooled covariance, shrunk toward its diagonal for conditioning.
        cov = centred.T @ centred / (centred.shape[0] - 1)
        cov = ((1.0 - cfg.io_cov_shrinkage) * cov
               + cfg.io_cov_shrinkage * np.diag(np.diag(cov)))
        cov_inv = np.linalg.inv(cov)
        return cls(cond_means, cov_inv, conditions, cfg)

    # ------------------------------------------------------------------
    def _posterior(self, r_bar: np.ndarray) -> np.ndarray:
        """P(condition | r_bar) for a batch; flat prior over conditions."""
        disc = r_bar @ self._A - 0.5 * self._b[None, :]  # (B, n_cond)
        disc -= disc.max(axis=1, keepdims=True)
        post = np.exp(disc)
        return post / post.sum(axis=1, keepdims=True)

    def p_go(self, r_bar: np.ndarray) -> np.ndarray:
        """Optimal P(Go | r_bar) for a batch of trial-mean V1 vectors."""
        r_bar = np.atleast_2d(r_bar)
        p = self._posterior(r_bar) @ self.go_weight
        return np.clip(p, 0.0, 1.0)  # guard tiny floating-point overshoot

    def log_odds(self, r_bar: np.ndarray) -> np.ndarray:
        """Optimal log P(Go)/P(NoGo) given r_bar -- the quantity SI integrates."""
        p = np.clip(self.p_go(r_bar), 1e-9, 1.0 - 1e-9)
        return np.log(p / (1.0 - p))

    def decide(self, r_bar: np.ndarray) -> np.ndarray:
        """Reward-optimal Go/NoGo choice (1/0) given r_bar."""
        return (self.log_odds(r_bar) > self.threshold).astype(int)
