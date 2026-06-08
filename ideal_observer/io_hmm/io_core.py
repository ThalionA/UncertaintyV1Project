# -*- coding: utf-8 -*-
"""IO core: pure functions mirroring the v2 MATLAB ideal observer.

Conventions match ``ideal_observer_hierarchical_fitting_v2.m``:

  - Orientation grid ``s_grid_deg = arange(0, 91)`` (91 bins, deg).
  - Latent measurement grid ``m_grid_deg = arange(0, 181)`` (181 bins, deg).
  - Doubled-angle von Mises so that orientation has 180-deg period.
  - Sensory precision ``kappa(s, c, d) = (kappa_min + kappa_amp) *
    c**c_power * exp(-d_power * d)`` (isotropic / orientation-independent).
  - Decision boundary at s == 45 deg (s < 45 -> Go, s > 45 -> NoGo,
    s == 45 -> half each).

These functions are deterministic and pure: no fitting, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Modified Bessel I0 used to normalise the von Mises pdf. ``numpy.i0`` is the
# same function as ``scipy.special.i0`` and avoids a scipy dependency for
# this module (everything else here is numpy only).
from numpy import i0

EPS = 1e-12


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IOGrids:
    """Discretisation grids used by the v2 IO. Values match the MATLAB code."""

    s_grid_deg: np.ndarray  # shape (n_s,), default arange(0, 91)
    m_grid_deg: np.ndarray  # shape (n_m,), default arange(0, 181)
    boundary_deg: float = 45.0

    @classmethod
    def default(cls) -> "IOGrids":
        return cls(
            s_grid_deg=np.arange(0, 91, dtype=float),
            m_grid_deg=np.arange(0, 181, dtype=float),
            boundary_deg=45.0,
        )

    @property
    def s_grid_rad(self) -> np.ndarray:
        return np.deg2rad(self.s_grid_deg)

    @property
    def m_grid_rad(self) -> np.ndarray:
        return np.deg2rad(self.m_grid_deg)

    @property
    def is_go(self) -> np.ndarray:
        return self.s_grid_deg < self.boundary_deg

    @property
    def is_nogo(self) -> np.ndarray:
        return self.s_grid_deg > self.boundary_deg

    @property
    def is_boundary(self) -> np.ndarray:
        return self.s_grid_deg == self.boundary_deg


# ---------------------------------------------------------------------------
# von Mises (doubled-angle) and sensory precision
# ---------------------------------------------------------------------------


def vonmises_pdf_doubled(x_rad: np.ndarray, mu_rad: np.ndarray | float,
                         kappa: float | np.ndarray) -> np.ndarray:
    """von Mises pdf with 180-deg period (doubled-angle convention).

    Matches v2's ``pdfVonMises``: density is ``exp(k cos(2x - 2mu)) /
    (2 pi I0(k))``. Broadcasting is standard numpy: pass a row vector for
    one axis and a column for the other to get the cross product.
    """
    return np.exp(kappa * np.cos(2 * x_rad - 2 * mu_rad)) / (2 * np.pi * i0(kappa))


def kappa_for_trial(s_deg: np.ndarray, c: np.ndarray, d: np.ndarray,
                    kappa_amp: float, c_power: float, d_power: float,
                    kappa_min: float = 1.0) -> np.ndarray:
    """Per-trial sensory precision kappa.

    Mirrors ``get_kappa_for_trial`` in v2. Note the result is independent
    of orientation (isotropic). The ``s_deg`` argument is accepted for
    interface symmetry but unused.
    """
    del s_deg
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    return (kappa_min + kappa_amp) * (c ** c_power) * np.exp(-d_power * d)


# ---------------------------------------------------------------------------
# Priors over orientation
# ---------------------------------------------------------------------------


def prior_flat(grids: IOGrids) -> np.ndarray:
    """Uniform prior over the s grid."""
    p = np.ones_like(grids.s_grid_deg, dtype=float)
    return p / p.sum()


def prior_bimodal(grids: IOGrids, prior_strength: float = 3.0,
                  centers_deg: tuple[float, float] = (0.0, 90.0)) -> np.ndarray:
    """Bimodal von Mises prior, components at ``centers_deg``.

    Mirrors v2's ``'Bimodal'`` branch with ``prior_strength`` controlling
    each component's concentration (kappa).
    """
    s_rad = grids.s_grid_rad
    c1 = vonmises_pdf_doubled(s_rad, np.deg2rad(centers_deg[0]), prior_strength)
    c2 = vonmises_pdf_doubled(s_rad, np.deg2rad(centers_deg[1]), prior_strength)
    p = c1 + c2
    return p / (p.sum() + EPS)


def prior_unimodal(grids: IOGrids, center_deg: float, prior_strength: float) -> np.ndarray:
    s_rad = grids.s_grid_rad
    p = vonmises_pdf_doubled(s_rad, np.deg2rad(center_deg), prior_strength)
    return p / (p.sum() + EPS)


# ---------------------------------------------------------------------------
# Posterior over s given m and log posterior odds g(m)
# ---------------------------------------------------------------------------


def likelihood_m_given_s(grids: IOGrids, kappa: float) -> np.ndarray:
    """L[m, s] = p(m | s, kappa). Shape (n_m, n_s).

    Each column is the generative density of m given a fixed s, so columns
    sum (approximately) to one over the m grid.
    """
    m_col = grids.m_grid_rad[:, None]
    s_row = grids.s_grid_rad[None, :]
    return vonmises_pdf_doubled(m_col, s_row, kappa)


def posterior_s_given_m(grids: IOGrids, kappa: float, prior: np.ndarray) -> np.ndarray:
    """Post[m, s] = p(s | m). Shape (n_m, n_s). Each row sums to 1."""
    lik = likelihood_m_given_s(grids, kappa)  # (n_m, n_s)
    unnormalised = lik * prior[None, :]
    return unnormalised / (unnormalised.sum(axis=1, keepdims=True) + EPS)


def p_m_given_s(grids: IOGrids, kappa: float, s_deg: float) -> np.ndarray:
    """p(m | s, kappa) on the m grid, normalised to sum to 1.

    Used as the marginal generative for the per-trial likelihood
    ``sum_m p_m_given_s * P(choice | m, z)``.
    """
    s_rad = np.deg2rad(s_deg)
    p = vonmises_pdf_doubled(grids.m_grid_rad, s_rad, kappa)
    return p / (p.sum() + EPS)


def log_posterior_odds(grids: IOGrids, posterior_s_m: np.ndarray) -> np.ndarray:
    """g(m) = log P(Go | m) / P(NoGo | m). Shape (n_m,).

    Mirrors v2's ``g_all`` row construction in ``precompute_choice_inputs``.
    The 45-deg boundary contributes half to Go and half to NoGo.
    """
    p_go = (posterior_s_m[:, grids.is_go].sum(axis=1)
            + 0.5 * posterior_s_m[:, grids.is_boundary].sum(axis=1))
    p_go = np.clip(p_go, EPS, 1 - EPS)
    return np.log(p_go / (1 - p_go))


# ---------------------------------------------------------------------------
# Utility-weighted decision variable DV(m)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Utility:
    """Trial-outcome utilities for the go/nogo task (v2 / paper defaults).

    Used to form the decision variable ``DV(m) = EU(Go|m) - EU(NoGo|m)`` that
    drives the kinematic (velocity) emission. Matches the MATLAB
    ``get_utility_vectors`` constants and the paper's Eqs. 5-8.
    """
    R_hit: float = 1.0    # respond on a Go stimulus
    R_miss: float = 0.0   # withhold on a Go stimulus
    R_cr: float = 0.1     # withhold on a NoGo stimulus (correct rejection)
    R_fa: float = -0.2    # respond on a NoGo stimulus (false alarm)


def utility_difference(grids: IOGrids, utility: Utility | None = None) -> np.ndarray:
    """``U(Go, s) - U(NoGo, s)`` on the s-grid. Shape (n_s,).

    Step function with the 45-deg boundary handled by the same 50/50 split
    used for g(m): the boundary bin gets the average of the Go and NoGo
    utilities for each action.
    """
    u = utility or Utility()
    u_go = np.where(grids.is_go, u.R_hit,
                    np.where(grids.is_nogo, u.R_fa, 0.5 * (u.R_hit + u.R_fa)))
    u_nogo = np.where(grids.is_go, u.R_miss,
                      np.where(grids.is_nogo, u.R_cr, 0.5 * (u.R_miss + u.R_cr)))
    return u_go - u_nogo


def decision_variable(grids: IOGrids, posterior_s_m: np.ndarray,
                      utility: Utility | None = None) -> np.ndarray:
    """``DV(m) = EU(Go|m) - EU(NoGo|m) = sum_s p(s|m) [U(Go,s) - U(NoGo,s)]``.

    Shape (n_m,). ``posterior_s_m`` is ``p(s|m)`` with rows summing to 1
    (e.g. from ``posterior_s_given_m``). Mirrors v2's ``dv = eu_go - eu_nogo``.
    Positive when the posterior favours Go, negative when it favours NoGo.
    """
    diff = utility_difference(grids, utility)  # (n_s,)
    return posterior_s_m @ diff                # (n_m,)

