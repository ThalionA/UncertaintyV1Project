"""Grating stimulus generation -- faithful to experiment_code/CreateGratings.m.

The real experiment renders each grating as a Gaussian-weighted *mixture* of
sine-wave components at different orientations: ``dispersion`` is the Gaussian
sigma over component orientation, ``contrast`` sets the luminance excursion,
and the whole pattern is rotated to the requested ``orientation``.

CreateGratings.m builds the mixture in a base frame and then ``imrotate``s the
image by ``orientation - 90``. Rotating the image is equivalent to shifting
every component's angle by the same amount (rotation is a coordinate remap and
commutes with the pointwise ``sign`` threshold). We build the components
directly at the shifted angles -- mathematically identical, with no
interpolation artefacts and no scipy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    ARCHETYPE_ORIENTATIONS_DEG,
    ARCHETYPE_PAIR,
    BOUNDARY_DEG,
    CONTRAST_DISPERSION_PAIRS,
    ORIENTATIONS_DEG,
    Config,
)

_GREY = 0.5  # mid-grey background, as in CreateGratings.m


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One (orientation, contrast, dispersion) stimulus condition."""

    orientation_deg: float
    contrast: float
    dispersion_deg: float

    @property
    def key(self) -> tuple[float, float, float]:
        return (self.orientation_deg, self.contrast, self.dispersion_deg)

    @property
    def category(self) -> int:
        """1 = Go (orientation < 45 deg), 0 = NoGo (orientation > 45 deg)."""
        return 1 if self.orientation_deg < BOUNDARY_DEG else 0

    @property
    def is_boundary(self) -> bool:
        return self.orientation_deg == BOUNDARY_DEG

    @property
    def signed_orientation(self) -> float:
        """Orientation re-centred on the 45-deg boundary (negative = Go side)."""
        return self.orientation_deg - BOUNDARY_DEG


def enumerate_conditions() -> list[Condition]:
    """All 81 unique conditions: 9 orientations x 9 (contrast, dispersion)."""
    conds: list[Condition] = []
    for ori in ORIENTATIONS_DEG:
        for contrast, dispersion in CONTRAST_DISPERSION_PAIRS:
            conds.append(Condition(float(ori), float(contrast), float(dispersion)))
    return conds


def archetype_conditions() -> list[Condition]:
    """The Phase-1 archetypes: 0 deg / 90 deg at (contrast 1, dispersion 5)."""
    contrast, dispersion = ARCHETYPE_PAIR
    return [
        Condition(float(ori), float(contrast), float(dispersion))
        for ori in ARCHETYPE_ORIENTATIONS_DEG
    ]


# ---------------------------------------------------------------------------
# Grating rendering
# ---------------------------------------------------------------------------


def mixture_weights(dispersion_deg: float, n_components: int) -> np.ndarray:
    """Gaussian weights over the orientation-mixture components.

    Mirrors CreateGratings.m: components are spaced linearly over
    ``[-180, +180]`` degrees and weighted by ``exp(-angle^2 / (2 sigma^2))``
    with ``sigma = dispersion``. Small dispersion -> a single orientation;
    large dispersion -> a broad mixture.
    """
    angles = np.linspace(-180.0, 180.0, n_components)
    w = np.exp(-(angles ** 2) / (2.0 * dispersion_deg ** 2))
    return w / w.sum()


def render_grating(orientation_deg: float, contrast: float,
                   dispersion_deg: float, cfg: Config) -> np.ndarray:
    """Render one square-wave grating image, values in [0, 1].

    Returns an ``(img_size, img_size)`` array. A zero-contrast stimulus is a
    flat grey field; a contrast-1 stimulus spans [0, 1].
    """
    n = cfg.img_size
    coords = np.linspace(-0.5, 0.5, n)
    x, y = np.meshgrid(coords, coords)

    base_angles = np.linspace(-180.0, 180.0, cfg.n_components)
    w = mixture_weights(dispersion_deg, cfg.n_components)
    # Shift every component angle by (orientation - 90): equivalent to
    # imrotate-ing the finished image, per the module docstring.
    comp_angles = np.deg2rad(base_angles + orientation_deg - 90.0)

    tot = np.zeros((n, n))
    for angle, weight in zip(comp_angles, w):
        carrier = (2.0 * np.pi * cfg.grating_freq
                   * (np.cos(angle) * x + np.sin(angle) * y)
                   + cfg.grating_phase)
        tot += weight * np.sin(carrier)

    # Square-wave threshold (sign), then map {-1, +1} -> grey -/+ contrast*grey.
    square = np.where(tot >= 0.0, 1.0, -1.0)
    return _GREY + contrast * _GREY * square


# ---------------------------------------------------------------------------
# Cached bank of all condition images
# ---------------------------------------------------------------------------


class StimulusBank:
    """Renders every unique condition image once and caches it.

    Gratings are static within a trial, so the ~81 condition images are
    rendered a single time and reused across all trials and all animals.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cache: dict[tuple[float, float, float], np.ndarray] = {}

    def image(self, condition: Condition) -> np.ndarray:
        """Return the cached grating image for ``condition``."""
        img = self._cache.get(condition.key)
        if img is None:
            img = render_grating(
                condition.orientation_deg,
                condition.contrast,
                condition.dispersion_deg,
                self.cfg,
            )
            self._cache[condition.key] = img
        return img

    def prerender(self) -> None:
        """Eagerly render and cache all 81 conditions."""
        for cond in enumerate_conditions():
            self.image(cond)
