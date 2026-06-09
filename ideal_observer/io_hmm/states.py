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
PERC_PARAMS = ('lambda_', 'prior_strength', 'prior_weight')


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
# Perceptual structure (Phase 2): per-state sensory precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionSpec:
    """Per-state perceptual parameters that reshape the IO inference itself.

    Two perceptual axes, each a fixed float (frozen) or ``None`` (free, fit by
    EM):

    - ``lambda_``: sensory-precision multiplier on the Stage-1 kappa
      (``kappa_eff = lambda_ * kappa(c, d)``). It scales precision for *both* the
      generative ``p(m|s)`` and the inference posterior (an observer that
      genuinely sees sharper/blurrier evidence), reshaping ``g(m)``, ``DV(m)``
      and ``p(m|s)`` together. ``lambda_ = 1.0`` is the v0 default.

    - the prior, when ``parameterized_prior`` is True: a bimodal von Mises at
      ``prior_centers`` with free concentration ``prior_strength`` and asymmetry
      ``prior_weight`` (mass on the first centre). The prior enters only the
      *inference* posterior (not ``p(m|s)``), so it moves ``g(m)`` / ``DV(m)``
      but not the generative spread. When ``parameterized_prior`` is False the
      state's fixed ``IOState.prior`` array is used and these fields are ignored.

    Identifiability note: ``prior_strength`` and ``lambda_`` are *both* posterior
    sharpness knobs and are only weakly separable (``lambda_`` also moves
    ``p(m|s)``; the prior does not), so fitting both free per state is fragile.
    ``prior_weight`` (asymmetry) is distinct from both ``lambda_`` (symmetric)
    and the decisional bias ``alpha``. The velocity channel (reading ``DV(m)``)
    is what makes the perceptual params identifiable apart from the choice
    psychometric. Because perception enters the posterior, the cached IO terms
    are recomputed as these params vary -- the cost of letting perception move.
    """
    lambda_: Optional[float] = 1.0
    prior_strength: Optional[float] = None
    prior_weight: Optional[float] = None
    prior_centers: tuple[float, float] = (0.0, 90.0)
    parameterized_prior: bool = False

    @property
    def free_params(self) -> tuple[str, ...]:
        names = []
        if self.lambda_ is None:
            names.append('lambda_')
        if self.parameterized_prior:
            if self.prior_strength is None:
                names.append('prior_strength')
            if self.prior_weight is None:
                names.append('prior_weight')
        return tuple(names)

    @property
    def n_free(self) -> int:
        return len(self.free_params)

    @property
    def is_frozen(self) -> bool:
        """True if no perceptual param is free (IO terms can be cached once)."""
        return self.n_free == 0

    def resolve(self, free_values: dict[str, float]) -> dict[str, float]:
        def pick(name, fixed):
            if fixed is not None:
                return float(fixed)
            if name not in free_values:
                raise KeyError(
                    f"free perception param '{name}' missing from free_values "
                    f"(spec free_params={self.free_params})")
            return float(free_values[name])

        out = {'lambda_': pick('lambda_', self.lambda_)}
        if self.parameterized_prior:
            out['prior_strength'] = pick('prior_strength', self.prior_strength)
            out['prior_weight'] = pick('prior_weight', self.prior_weight)
        return out


def default_perception_spec(parameterized_prior: bool = False) -> "PerceptionSpec":
    """Free ``lambda_`` (and, if ``parameterized_prior``, free prior too)."""
    if parameterized_prior:
        return PerceptionSpec(lambda_=None, prior_strength=None, prior_weight=None,
                              parameterized_prior=True)
    return PerceptionSpec(lambda_=None)


def perception_inputs(state: "IOState", grids: io_core.IOGrids,
                      perc_values: dict[str, float]
                      ) -> tuple[float, np.ndarray]:
    """Resolve a state's perception into ``(lambda_, prior_array)``.

    ``lambda_`` scales the trial kappa; ``prior_array`` is the parameterised
    bimodal prior when the spec declares one, else the state's fixed prior.
    """
    spec = state.perception
    resolved = spec.resolve(dict(perc_values))
    if spec.parameterized_prior:
        prior = io_core.prior_bimodal_weighted(
            grids, resolved['prior_strength'], resolved['prior_weight'],
            spec.prior_centers)
    else:
        prior = state.prior
    return resolved['lambda_'], prior


# A frozen, lambda_=1 perception spec -- the v0 default used when a state
# carries no PerceptionSpec (perception is fixed at the Stage-1 kappa).
_FROZEN_PERCEPTION = PerceptionSpec(lambda_=1.0)


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
    perc: Optional[PerceptionSpec] = None  # perceptual spec (None = frozen lambda_=1)

    @property
    def perception(self) -> PerceptionSpec:
        """The state's PerceptionSpec, defaulting to frozen lambda_=1."""
        return self.perc if self.perc is not None else _FROZEN_PERCEPTION

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
                      with_velocity: bool = False,
                      with_perception: bool = False) -> list[IOState]:
    """The four v0 states: Perfect, Thirsty, Disengaged, Naive.

    If ``with_velocity``, each state also carries a ``VelocitySpec``. Engaged
    states (Perfect / Thirsty / Naive) get the full confidence model (all three
    params free); Disengaged gets ``beta_vel = 0`` fixed -- its defining feature
    is that velocity is *decoupled* from decision confidence (a pure baseline).

    If ``with_perception``, the engaged states additionally get a free sensory
    precision ``lambda_`` (Phase 2): an observer that genuinely sees sharper /
    blurrier evidence, reshaping ``g(m)``, ``DV(m)`` and ``p(m|s)`` together.
    Disengaged keeps frozen perception (``lambda_ = 1``) -- with ``beta = 0``
    and decoupled velocity its ``DV(m)`` drives nothing, so ``lambda_`` would be
    unidentifiable for it (mirroring why its velocity is decoupled).

    ``with_perception`` requires ``with_velocity``: free ``lambda_`` is only
    separable from the choice psychometric through the velocity channel reading
    ``DV(m)`` (see ``PerceptionSpec`` identifiability note). Asking for free
    perception without velocity raises ``ValueError``.
    """
    if with_perception and not with_velocity:
        raise ValueError(
            "with_perception requires with_velocity: free lambda_ is only "
            "identifiable through the velocity channel reading DV(m)."
        )
    bimodal = io_core.prior_bimodal(grids, prior_strength=bimodal_prior_strength)
    flat = io_core.prior_flat(grids)

    def vel(decoupled=False):
        if not with_velocity:
            return None
        return VelocitySpec(beta_vel=0.0) if decoupled else default_velocity_spec()

    def perc(frozen=False):
        # Engaged states get free lambda_; Disengaged (frozen=True) stays at
        # lambda_=1 since its DV(m) drives nothing.
        if not with_perception or frozen:
            return None
        return default_perception_spec()

    return [
        IOState(
            name='Perfect',
            prior=bimodal,
            psych=PsychSpec(alpha=0.0, gamma=0.0, delta=0.0),
            vel=vel(),
            perc=perc(),
            description='Bimodal prior; unbiased no-lapse psychometric (only beta free).',
        ),
        IOState(
            name='Thirsty',
            prior=bimodal,
            psych=PsychSpec(gamma=0.0, delta=0.0),
            vel=vel(),
            perc=perc(),
            description='Bimodal prior; no-lapse, alpha and beta both free (bias allowed).',
        ),
        IOState(
            name='Disengaged',
            prior=bimodal,
            psych=PsychSpec(beta=0.0, gamma=0.0, delta=0.0),
            vel=vel(decoupled=True),
            perc=perc(frozen=True),
            description='Bimodal prior; beta=0 => constant P(go); velocity decoupled (beta_vel=0).',
        ),
        IOState(
            name='Naive',
            prior=flat,
            psych=PsychSpec(alpha=0.0, gamma=0.0, delta=0.0),
            vel=vel(),
            perc=perc(),
            description='Flat prior; unbiased no-lapse psychometric (only beta free).',
        ),
    ]


__all__ = [
    'PSYCH_PARAMS',
    'VEL_PARAMS',
    'PERC_PARAMS',
    'PsychSpec',
    'VelocitySpec',
    'PerceptionSpec',
    'default_velocity_spec',
    'default_perception_spec',
    'perception_inputs',
    'IOState',
    'psychometric',
    'default_v0_states',
]
