"""V1 layer: diverse Gabor receptive fields + recurrent dynamics.

The V1 layer is FIXED -- its feed-forward Gabor RFs and its recurrent weights
never learn (the SI framework's commitment: V1 is stimulus-driven sensory
cortex, only the V1->decision synapses are plastic). Each simulated animal
gets its own V1 instance, built from its seed, so the cohort has diverse RF
mosaics.

Pipeline per trial:
  grating image -> Gabor feed-forward drive (static) -> recurrent dynamics
  over the 2 s trial -> population trajectory r(t).
"""

from __future__ import annotations

import numpy as np

from .config import CONTRAST_DISPERSION_PAIRS, Config
from .stimuli import Condition, render_grating

# Orientation grid for the precomputed feed-forward drive lookup. It spans well
# beyond the task's 0-90 deg range so that large per-trial orientation jitter
# (on hard, low-contrast stimuli) is covered without extrapolation. 3-deg
# spacing is fine -- V1 tuning curves are smooth, so linear interpolation
# between nodes is accurate.
_ORI_GRID = np.arange(-90.0, 270.0, 3.0)


def _phi(x: np.ndarray, sat: float) -> np.ndarray:
    """V1 transfer function: rectified, saturating at ``sat``.

    Linear near zero, saturates smoothly -- guarantees bounded, stable
    recurrent dynamics regardless of the recurrent gain.
    """
    return sat * np.tanh(np.maximum(x, 0.0) / sat)


def orientation_jitter_sd(contrast: np.ndarray, dispersion: np.ndarray,
                          cfg: Config) -> np.ndarray:
    """Per-trial orientation-jitter sd (deg) as a function of stimulus reliability.

    Larger for low contrast and high dispersion -- the neural mirror of the
    stimulus-IO's sensory precision kappa(c, d).
    """
    contrast = np.asarray(contrast, dtype=float)
    dispersion = np.asarray(dispersion, dtype=float)
    return (cfg.ori_jitter_gain
            * contrast ** (-cfg.ori_jitter_c_power)
            * np.exp(cfg.ori_jitter_d_power * dispersion))


class V1Model:
    """A fixed V1 population for one animal."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        n = cfg.n_v1
        img = cfg.img_size

        # ---- diverse Gabor RF parameters ----
        # Preferred orientations tile [0, 180) deg with jitter for coverage.
        base = np.linspace(0.0, 180.0, n, endpoint=False)
        jitter = rng.uniform(-0.5, 0.5, n) * (180.0 / n)
        self.preferred_orientation_deg = (base + jitter) % 180.0
        sf = cfg.gabor_sf_mean * np.exp(rng.normal(0.0, cfg.gabor_sf_jitter, n))
        phase = rng.uniform(0.0, 2.0 * np.pi, n)
        sigma = rng.uniform(cfg.gabor_size_min, cfg.gabor_size_max, n)
        aspect = cfg.gabor_aspect * np.exp(rng.normal(0.0, 0.15, n))
        # RF centres tile the central 60% of the image.
        cx = rng.uniform(-0.3, 0.3, n)
        cy = rng.uniform(-0.3, 0.3, n)
        # ~half the cells are phase-invariant (complex) energy units.
        self.is_complex = rng.random(n) < cfg.frac_complex

        # ---- build quadrature Gabor kernels, (n_v1, img*img) ----
        coords = np.linspace(-0.5, 0.5, img)
        gx, gy = np.meshgrid(coords, coords)
        k_cos = np.empty((n, img * img))
        k_sin = np.empty((n, img * img))
        for i in range(n):
            th = np.deg2rad(self.preferred_orientation_deg[i])
            xr = (gx - cx[i]) * np.cos(th) + (gy - cy[i]) * np.sin(th)
            yr = -(gx - cx[i]) * np.sin(th) + (gy - cy[i]) * np.cos(th)
            env = np.exp(-(xr ** 2 / (2.0 * sigma[i] ** 2)
                           + yr ** 2 / (2.0 * (aspect[i] * sigma[i]) ** 2)))
            carrier = 2.0 * np.pi * sf[i] * xr + phase[i]
            kc = env * np.cos(carrier)
            ks = env * np.sin(carrier)
            # zero-mean (reject DC) and unit-norm (comparable scale across cells)
            kc -= kc.mean()
            ks -= ks.mean()
            kc /= np.linalg.norm(kc) + 1e-12
            ks /= np.linalg.norm(ks) + 1e-12
            k_cos[i] = kc.ravel()
            k_sin[i] = ks.ravel()
        self._k_cos = k_cos
        self._k_sin = k_sin

        # ---- recurrent connectivity by tuning (orientation) similarity ----
        self.W_rec = self._build_recurrent()

    # ------------------------------------------------------------------
    def _build_recurrent(self) -> np.ndarray:
        """Like-to-like excitation + broad inhibition, scaled for stability."""
        cfg = self.cfg
        pref = self.preferred_orientation_deg
        d = np.abs(pref[:, None] - pref[None, :])
        d = np.minimum(d, 180.0 - d)  # circular orientation distance, [0, 90]
        w_exc = cfg.rec_exc * np.exp(-(d ** 2) / (2.0 * cfg.rec_width_deg ** 2))
        w = w_exc - cfg.rec_inh  # broad uniform inhibition
        np.fill_diagonal(w, 0.0)
        # Scale to a fixed spectral radius (W is symmetric).
        max_eig = np.max(np.abs(np.linalg.eigvalsh(w)))
        return w * (cfg.rec_spectral_target / (max_eig + 1e-12))

    # ------------------------------------------------------------------
    def feedforward_drive(self, images: np.ndarray) -> np.ndarray:
        """Static Gabor drive for a batch of grating images.

        ``images`` has shape ``(B, img, img)``. Returns ``(B, n_v1)``,
        non-negative: simple cells half-wave rectify, complex cells take
        quadrature energy.
        """
        b = images.shape[0]
        flat = images.reshape(b, -1)
        flat = flat - flat.mean(axis=1, keepdims=True)  # remove DC
        # Normalise by sqrt(n_pixels) so the drive scale is image-size
        # independent (a mean overlap, not a sum). Keeps V1 below saturation.
        scale = np.sqrt(flat.shape[1])
        cos_r = (flat @ self._k_cos.T) / scale  # (B, n_v1)
        sin_r = (flat @ self._k_sin.T) / scale
        simple = np.maximum(cos_r, 0.0)
        complex_ = np.sqrt(cos_r ** 2 + sin_r ** 2)
        drive = np.where(self.is_complex[None, :], complex_, simple)
        return self.cfg.v1_ff_gain * drive

    # ------------------------------------------------------------------
    def simulate(self, ff_drive: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Run the recurrent dynamics given a static feed-forward drive.

        ``ff_drive`` is ``(B, n_v1)``. Returns the trajectory
        ``(B, n_steps, n_v1)``.

        Two noise sources: an across-trial input perturbation drawn once per
        trial (constant over the trial -- sets bundle width and stimulus
        difficulty) and a per-timestep within-trial perturbation (washes out
        under averaging -- the diffusion term).
        """
        cfg = self.cfg
        b, n = ff_drive.shape
        a = cfg.dt / cfg.tau

        # Across-trial input noise: one draw per trial, held fixed all trial.
        priv = np.sqrt(1.0 - cfg.shared_noise_frac) * rng.standard_normal((b, n))
        shared = np.sqrt(cfg.shared_noise_frac) * rng.standard_normal((b, 1))
        drive = ff_drive + cfg.sensory_noise * (priv + shared)

        r = np.zeros((b, n))
        traj = np.empty((b, cfg.n_steps, n))
        for t in range(cfg.n_steps):
            within = cfg.within_trial_noise * rng.standard_normal((b, n))
            rec = cfg.rec_gain * (r @ self.W_rec)
            r = r + a * (-r + _phi(drive + rec + within, cfg.v1_sat))
            traj[:, t, :] = r
        return traj

    # ------------------------------------------------------------------
    def run(self, images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Convenience: feed-forward drive + recurrent dynamics in one call."""
        return self.simulate(self.feedforward_drive(images), rng)

    # ------------------------------------------------------------------
    # Orientation-jitter front end
    # ------------------------------------------------------------------
    def build_drive_grid(self) -> None:
        """Precompute the feed-forward drive vs orientation for each (c, d) pair.

        Gratings are rendered (the CreateGratings.m recipe) on a fine
        orientation grid and passed through this animal's Gabor RFs once. Per
        trial the drive for a jittered orientation is then a cheap linear
        interpolation -- so orientation jitter is exact and fast.
        """
        self.ori_grid = _ORI_GRID
        self.drive_grid: dict[tuple[float, float], np.ndarray] = {}
        for contrast, dispersion in CONTRAST_DISPERSION_PAIRS:
            images = np.stack([
                render_grating(o, contrast, dispersion, self.cfg)
                for o in self.ori_grid
            ])
            self.drive_grid[(contrast, dispersion)] = self.feedforward_drive(images)

    def drive_for(self, contrast: float, dispersion: float,
                  orientations: np.ndarray) -> np.ndarray:
        """Feed-forward drive for arbitrary orientations (linear interpolation)."""
        grid = self.drive_grid[(contrast, dispersion)]
        step = self.ori_grid[1] - self.ori_grid[0]
        idx = (np.asarray(orientations, dtype=float) - self.ori_grid[0]) / step
        idx = np.clip(idx, 0.0, len(self.ori_grid) - 1 - 1e-6)
        lo = np.floor(idx).astype(int)
        frac = (idx - lo)[:, None]
        return (1.0 - frac) * grid[lo] + frac * grid[lo + 1]

    def present(self, conditions: list[Condition],
                rng: np.random.Generator) -> np.ndarray:
        """Simulate V1 for a batch of trials, applying per-trial orientation jitter.

        Each trial's encoded orientation is ``true orientation + N(0, sd)``
        with ``sd`` from :func:`orientation_jitter_sd`. Returns trajectories
        ``(B, n_steps, n_v1)``.
        """
        contrast = np.array([c.contrast for c in conditions])
        dispersion = np.array([c.dispersion_deg for c in conditions])
        true_ori = np.array([c.orientation_deg for c in conditions])
        sd = orientation_jitter_sd(contrast, dispersion, self.cfg)
        eff_ori = true_ori + sd * rng.standard_normal(len(conditions))

        drive = np.empty((len(conditions), self.cfg.n_v1))
        for cd_contrast, cd_dispersion in CONTRAST_DISPERSION_PAIRS:
            mask = (contrast == cd_contrast) & (dispersion == cd_dispersion)
            if mask.any():
                drive[mask] = self.drive_for(cd_contrast, cd_dispersion,
                                             eff_ori[mask])
        return self.simulate(drive, rng)
