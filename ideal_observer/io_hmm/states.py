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
                      bimodal_prior_strength: float = 3.0) -> list[IOState]:
    """The four v0 states: Perfect, Thirsty, Disengaged, Naive."""
    bimodal = io_core.prior_bimodal(grids, prior_strength=bimodal_prior_strength)
    flat = io_core.prior_flat(grids)
    return [
        IOState(
            name='Perfect',
            prior=bimodal,
            psych=PsychSpec(alpha=0.0, gamma=0.0, delta=0.0),
            description='Bimodal prior; unbiased no-lapse psychometric (only beta free).',
        ),
        IOState(
            name='Thirsty',
            prior=bimodal,
            psych=PsychSpec(gamma=0.0, delta=0.0),
            description='Bimodal prior; no-lapse, alpha and beta both free (bias allowed).',
        ),
        IOState(
            name='Disengaged',
            prior=bimodal,
            psych=PsychSpec(beta=0.0, gamma=0.0, delta=0.0),
            description='Bimodal prior; beta=0 => stimulus-independent constant P(go) = sigmoid(alpha).',
        ),
        IOState(
            name='Naive',
            prior=flat,
            psych=PsychSpec(alpha=0.0, gamma=0.0, delta=0.0),
            description='Flat prior; unbiased no-lapse psychometric (only beta free).',
        ),
    ]


__all__ = [
    'PSYCH_PARAMS',
    'PsychSpec',
    'IOState',
    'psychometric',
    'default_v0_states',
]
