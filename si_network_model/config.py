"""Central configuration for the SI-framework network model.

Every tunable parameter lives here so the model has no hidden constants. The
per-animal trait sampling (Decision 5 in UNDERSTANDING.md) is in
``sample_animal_traits``: each simulated animal gets its own seed *and* a
draw of trait parameters, producing the cross-animal spread in bundle width
and lapse rate that the framework's predictions are tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# Stimulus space -- matches experiment_code/NewExperimentTheo25_full2_v3.m
# ---------------------------------------------------------------------------

# The 9 grating orientations used in the real task (degrees).
ORIENTATIONS_DEG: tuple[int, ...] = (0, 15, 30, 40, 45, 50, 60, 75, 90)

# The 9 (contrast, dispersion) pairs. dispersion is in degrees and acts as the
# Gaussian sigma weighting the orientation-mixture components.
CONTRAST_DISPERSION_PAIRS: tuple[tuple[float, float], ...] = (
    (1.00, 5.0), (1.00, 30.0), (1.00, 90.0),
    (0.50, 5.0), (0.50, 45.0),
    (0.25, 30.0), (0.25, 90.0),
    (0.01, 5.0), (0.01, 45.0),
)

# Decision boundary: orientation < 45 -> Go, > 45 -> NoGo, == 45 -> 50/50.
BOUNDARY_DEG: float = 45.0

# Archetypes shown in the Basic curriculum phase.
ARCHETYPE_ORIENTATIONS_DEG: tuple[int, ...] = (0, 90)
ARCHETYPE_PAIR: tuple[float, float] = (1.00, 5.0)  # (contrast, dispersion)


@dataclass(frozen=True)
class Config:
    """Immutable parameter bundle. Use ``dataclasses.replace`` to vary one."""

    # ---- stimulus rendering ----
    img_size: int = 128
    n_components: int = 9          # mixture components (CreateGratings numAngles)
    grating_freq: float = 6.0      # cycles across the image width
    grating_phase: float = float(np.pi / 12)

    # ---- V1 layer ----
    n_v1: int = 320
    frac_complex: float = 0.5      # fraction of phase-invariant (complex) cells
    gabor_sf_mean: float = 6.0     # cycles across image, matched to grating_freq
    gabor_sf_jitter: float = 0.32  # log-normal sigma on SF
    gabor_size_min: float = 0.10   # envelope sigma, fraction of image width
    gabor_size_max: float = 0.22
    gabor_aspect: float = 1.5      # envelope elongation along the bars

    # V1 recurrence -- connectivity by tuning (orientation) similarity
    rec_exc: float = 1.05          # like-to-like excitatory gain
    rec_inh: float = 0.70          # broad inhibitory gain (normalisation)
    rec_width_deg: float = 24.0    # orientation half-width of recurrent excitation
    rec_gain: float = 1.0          # global recurrent scale (per-animal trait)
    rec_spectral_target: float = 0.82  # W_rec is scaled to this spectral radius

    # V1 dynamics
    tau: float = 0.20              # membrane time constant, s
    dt: float = 0.08               # integration step, s
    t_trial: float = 2.0           # trial duration, s
    v1_ff_gain: float = 2.5        # feed-forward drive scale (keeps V1 below sat)
    v1_sat: float = 1.0            # saturation level of the V1 nonlinearity
    # --- Orientation jitter: the dominant noise source ---
    # Trial-to-trial noise in the *encoded orientation itself*. It is
    # information-limiting (it lies along the task-relevant dimension, so
    # averaging across neurons cannot remove it) and reliability-scaled (low
    # contrast / high dispersion -> larger jitter -> genuinely harder), the
    # neural mirror of the stimulus-IO's sensory precision kappa(c, d). This
    # sets both stimulus difficulty and bundle width. ``ori_jitter_gain`` is a
    # per-animal trait.
    ori_jitter_gain: float = 4.0       # jitter sd (deg) at contrast 1, dispersion 0
    ori_jitter_c_power: float = 0.78   # jitter grows as contrast ** -this
    ori_jitter_d_power: float = 0.013  # jitter grows as exp(this * dispersion)
    # --- Secondary V1 noise (isotropic, small) ---
    #  sensory_noise:      across-trial isotropic input noise (a small extra
    #                      bundle-width component beyond orientation jitter).
    #  within_trial_noise: per-timestep noise -> the within-trial diffusion
    #                      that produces Var_t[SI].
    sensory_noise: float = 0.06        # across-trial isotropic input-noise sd
    within_trial_noise: float = 0.05   # per-timestep input-noise sd
    shared_noise_frac: float = 0.35    # fraction of across-trial noise that is shared

    # ---- decision layer ----
    w_init_scale: float = 0.05     # initial archetype-trajectory sd (-> random early choices)
    si_eps: float = 1e-6
    n_exemplars: int = 20          # K stored exemplar trajectories per class (the bundle)
    ddm_drift_gain: float = 12.0   # SI(t) -> drift-rate scaling. Large because SI
                                   # magnitudes are small in the realistic regime
                                   # (V1 prototypes share most of their structure).
    ddm_sigma: float = 0.35        # within-trial diffusion sd (per-animal trait)
    ddm_bound: float = 1.0         # symmetric absorbing bounds at +/- this

    # ---- plasticity: three-factor reward-modulated Hebbian ----
    learning_rate: float = 0.006   # per-animal trait.

    # ---- reward structure ----
    reward_mode: str = "symmetric"   # "symmetric" or "asymmetric"
    reward_correct: float = 1.0      # symmetric mode
    reward_error: float = -1.0
    reward_hit: float = 1.0            # asymmetric mode (water-task-like)
    reward_false_alarm: float = -0.25  # |FA cost| < hit reward -> a Go-incentive
    reward_correct_reject: float = 0.25  # small reward -> the NoGo channel can learn
    reward_miss: float = 0.0

    # ---- curriculum ----
    n_animals: int = 10
    phase1_cap: int = 1200             # max archetype-only trials
    phase1_acc_threshold: float = 0.80  # rolling accuracy to graduate Phase 1
    phase1_acc_window: int = 200
    phase2_trials: int = 3300
    archetype_fraction: float = 0.50   # share of Phase-2 trials that are archetypes
    train_chunk: int = 256             # trials per V1-simulation batch during training
    n_eval_per_cond: int = 60          # frozen-weight evaluation trials per condition

    # ---- ideal observers ----
    # Stimulus-IO sensory precision: kappa = (kappa_min + kappa_amp)
    #   * contrast**c_power * exp(-d_power * dispersion).
    io_kappa_amp: float = 22.0
    io_kappa_min: float = 1.0
    io_c_power: float = 1.10
    io_d_power: float = 0.035
    # Neural-IO: Gaussian class-conditional decoder of trial-mean V1 activity.
    io_samples_per_cond: int = 180
    io_cov_shrinkage: float = 0.10   # shrink pooled covariance toward its diagonal

    # ---- misc ----
    master_seed: int = 20260519

    @property
    def n_steps(self) -> int:
        """Number of V1 dynamics timesteps per trial."""
        return int(round(self.t_trial / self.dt))

    @property
    def n_conditions(self) -> int:
        """Number of unique (orientation, contrast, dispersion) conditions."""
        return len(ORIENTATIONS_DEG) * len(CONTRAST_DISPERSION_PAIRS)


def sample_animal_traits(base: Config, rng: np.random.Generator) -> Config:
    """Draw one animal's trait parameters (Decision 5, UNDERSTANDING.md).

    Four traits vary log-normally around the base config: orientation-jitter
    gain (the dominant control of bundle width / lapse rate), learning rate,
    DDM diffusion (exploration temperature) and recurrent gain. Together they
    give the cohort genuine spread in bundle width, lapse rate and learning
    speed -- the spread the framework's cross-animal predictions are tested on.
    """
    return replace(
        base,
        ori_jitter_gain=float(base.ori_jitter_gain * np.exp(rng.normal(0.0, 0.33))),
        learning_rate=float(base.learning_rate * np.exp(rng.normal(0.0, 0.40))),
        ddm_sigma=float(base.ddm_sigma * np.exp(rng.normal(0.0, 0.28))),
        rec_gain=float(base.rec_gain * np.exp(rng.normal(0.0, 0.16))),
    )


def category_of(orientation_deg: float) -> int:
    """Return 1 for Go (orientation < 45), 0 for NoGo (> 45).

    The 45-degree boundary itself is ambiguous (handled as 50/50 by callers);
    this helper returns 1 for the exact boundary only as a placeholder.
    """
    return 1 if orientation_deg < BOUNDARY_DEG else 0
