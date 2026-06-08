# -*- coding: utf-8 -*-
"""IO-HMM state definitions: prior over orientation + psychometric structure.

A state declares (a) its prior over the orientation grid (which feeds into
the IO posterior used to compute g(m)), and (b) which entries of the
4-param choice psychometric on g(m) are fixed vs. free. EM only updates
the free entries.

The 4-param psychometric is

    psi(g; alpha, beta, gamma, delta) =
        gamma + (1 - gamma - delta) * sigmoid(alpha + beta * g)

where g is the IO log posterior odds log P(Go | m) / P(NoGo | m).

v0 four-state spec
------------------

| State       | Prior              | Fixed psych              | Free psych  |
|-------------|--------------------|--------------------------|-------------|
| Perfect     | Bimodal kappa=3    | alpha=0, gamma=0, delta=0| beta        |
| Thirsty     | Bimodal kappa=3    | gamma=0, delta=0         | alpha, beta |
| Disengaged  | Bimodal kappa=3    | beta=0, gamma=0, delta=0 | alpha       |
| Naive       | Flat               | alpha=0, gamma=0, delta=0| beta        |

Identifying logic: Perfect vs Thirsty differ only in whether ``alpha`` is
free (bias allowed in Thirsty); Perfect/Thirsty vs Disengaged differ in
whether ``beta`` is free (Disengaged has stimulus-independent constant
P(go)); Perfect vs Naive differ only in the prior shape going into the IO
inference (so emissions diverge on stimuli where bimodal vs flat priors
push g(m) in different directions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import io_core  # flat import; ideal_observer/io_hmm must be on sys.path

PSYCH_PARAMS = ('alpha', 'beta', 'gamma', 'delta')
VEL_PARAMS = ('beta_vel', 'alpha_vel', 'sigma_vel')


# ---------------------------------------------------------------------------
# Psychometric structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PsychSpec:
    """Per-state 4-param psychometric structure.

    Each of (alpha, beta, gamma, delta) is either a fixed float or ``None``
    (meaning "free, fit by EM"). The identifying constraint of a state is
    the pattern of fixed entries.
    """
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    delta: Optional[float] = None

    @property
    def free_params(self) -> tuple[str, ...]:
        return tuple(name for name in PSYCH_PARAMS if getattr(self, name) is None)

    @property
    def n_free(self) -> int:
        return len(self.free_params)

    def resolve(self, free_values: dict[str, float]) -> dict[str, float]:
        """Combine fixed values from this spec with ``free_values``.

        ``free_values`` must contain a value for every free param of this
        spec; extra keys are ignored. Returns a complete
        ``{alpha, beta, gamma, delta}`` dict of floats.
        """
        out: dict[str, float] = {}
        for name in PSYCH_PARAMS:
            fixed = getattr(self, name)
            if fixed is not None:
                out[name] = float(fixed)
            else:
                if name not in free_values:
                    raise KeyError(
                        f"free psych param '{name}' missing from free_values "
                        f"(spec free_params={self.free_params})"
                    )
                out[name] = float(free_values[name])
        return out


# ---------------------------------------------------------------------------
# Velocity (confidence) emission structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VelocitySpec:
    """Per-state confidence-velocity structure (paper Eq. 10).

    Velocity is a graded readout of the decision variable ``DV(m)``:
    ``v ~ Normal(beta_vel * DV(m) + alpha_vel, sigma_vel^2)``, marginalised over
    the latent measurement ``m`` jointly with choice. Each of
    (beta_vel, alpha_vel, sigma_vel) is a fixed float or ``None`` (free, fit by
    EM). ``beta_vel`` is the *confidence gain*: ``beta_vel = 0`` decouples
    velocity from the stimulus (the old stimulus-independent engagement marker
    is exactly this special case, with alpha_vel the mean and sigma_vel the sd).
    """
    beta_vel: Optional[float] = None
    alpha_vel: Optional[float] = None
    sigma_vel: Optional[float] = None

    @property
    def free_params(self) -> tuple[str, ...]:
        return tuple(name for name in VEL_PARAMS if getattr(self, name) is None)

    @property
    def n_free(self) -> int:
        return len(self.free_params)

    def resolve(self, free_values: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in VEL_PARAMS:
            fixed = getattr(self, name)
            if fixed is not None:
                out[name] = float(fixed)
            else:
                if name not in free_values:
                    raise KeyError(
                        f"free velocity param '{name}' missing from free_values "
                        f"(spec free_params={self.free_params})"
                    )
                out[name] = float(free_values[name])
        return out


def default_velocity_spec() -> "VelocitySpec":
    """All three confidence params free (the full per-trial confidence model)."""
    return VelocitySpec()


# ---------------------------------------------------------------------------
# Psychometric function
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def psychometric(g: np.ndarray, alpha: float, beta: float,
                 gamma: float, delta: float) -> np.ndarray:
    """4-param psychometric P(go | g).

    ``gamma + (1 - gamma - delta) * sigmoid(alpha + beta * g)``. Returns an
    array of the same shape as ``g``, with values in
    ``[gamma, 1 - delta]``.
    """
    return gamma + (1.0 - gamma - delta) * _sigmoid(alpha + beta * g)


# ---------------------------------------------------------------------------
# IOState
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)  # eq=False: numpy prior breaks default __eq__
class IOState:
    """An IO-HMM state: name + prior over orientation + psychometric spec.

    The prior is given as a realised array on the orientation grid (rather
    than a callable) so that two states with the same prior share no
    redundant compute in emissions, and so the state object captures the
    actual numerical prior used.
    """
    name: str
    prior: np.ndarray  # shape (n_s,), normalised
    psych: PsychSpec
    description: str = ''
    vel: Optional[VelocitySpec] = None  # confidence-velocity spec (None = choice-only)

    def __post_init__(self):
        s = float(self.prior.sum())
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(
                f"state '{self.name}' prior not normalised (sum={s:.6f})"
            )


# ---------------------------------------------------------------------------
# v0 four-state factory
# ---------------------------------------------------------------------------


def default_v0_states(grids: io_core.IOGrids,
                      bimodal_prior_strength: float = 3.0,
                      with_velocity: bool = False) -> list[IOState]:
    """The four v0 states: Perfect, Thirsty, Disengaged, Naive.

    If ``with_velocity``, each state also carries a ``VelocitySpec``. Engaged
    states (Perfect / Thirsty / Naive) get the full confidence model (all three
    params free); Disengaged gets ``beta_vel = 0`` fixed -- its defining feature
    is that velocity is *decoupled* from decision confidence (a pure baseline).
    """
    bimodal = io_core.prior_bimodal(grids, prior_strength=bimodal_prior_strength)
    flat = io_core.prior_flat(grids)

    def vel(decoupled=False):
        if not with_velocity:
            return None
        return VelocitySpec(beta_vel=0.0) if decoupled else default_velocity_spec()

    return [
        IOState(
            name='Perfect',
            prior=bimodal,
            psych=PsychSpec(alpha=0.0, gamma=0.0, delta=0.0),
            vel=vel(),
            description='Bimodal prior; unbiased no-lapse psychometric (only beta free).',
        ),
        IOState(
            name='Thirsty',
            prior=bimodal,
            psych=PsychSpec(gamma=0.0, delta=0.0),
            vel=vel(),
            description='Bimodal prior; no-lapse, alpha and beta both free (bias allowed).',
        ),
        IOState(
            name='Disengaged',
            prior=bimodal,
            psych=PsychSpec(beta=0.0, gamma=0.0, delta=0.0),
            vel=vel(decoupled=True),
            description='Bimodal prior; beta=0 => constant P(go); velocity decoupled (beta_vel=0).',
        ),
        IOState(
            name='Naive',
            prior=flat,
            psych=PsychSpec(alpha=0.0, gamma=0.0, delta=0.0),
            vel=vel(),
            description='Flat prior; unbiased no-lapse psychometric (only beta free).',
        ),
    ]


__all__ = [
    'PSYCH_PARAMS',
    'VEL_PARAMS',
    'PsychSpec',
    'VelocitySpec',
    'default_velocity_spec',
    'IOState',
    'psychometric',
    'default_v0_states',
]
