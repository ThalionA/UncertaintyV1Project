"""Tests for v1_model.py -- Gabor RFs + recurrent dynamics.

Correctness is defined on known structure: V1 neurons must be orientation
tuned (a grating drives neurons whose preferred orientation matches it),
drive must scale with contrast, the recurrent matrix must be stable, and the
population trajectory r(t) must ramp from baseline and stay bounded.
"""

from __future__ import annotations

import numpy as np

from si_network_model import stimuli
from si_network_model.config import Config
from si_network_model.v1_model import V1Model, orientation_jitter_sd


def _circular_mean_deg(values_deg: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean on the 180-deg orientation circle."""
    ang = np.deg2rad(values_deg) * 2.0
    c = np.sum(weights * np.cos(ang))
    s = np.sum(weights * np.sin(ang))
    return float((np.degrees(np.arctan2(s, c)) / 2.0) % 180.0)


def _small_cfg() -> Config:
    # Smaller network keeps the test suite fast.
    return Config(n_v1=120, img_size=64)


def test_v1_run_shapes():
    cfg = _small_cfg()
    rng = np.random.default_rng(0)
    v1 = V1Model(cfg, rng)
    bank = stimuli.StimulusBank(cfg)
    conds = stimuli.archetype_conditions()
    images = np.stack([bank.image(c) for c in conds])
    traj = v1.run(images, np.random.default_rng(1))
    assert traj.shape == (2, cfg.n_steps, cfg.n_v1)
    assert np.all(np.isfinite(traj))


def test_v1_deterministic_without_noise():
    cfg = Config(n_v1=120, img_size=64, sensory_noise=0.0, within_trial_noise=0.0)
    rng = np.random.default_rng(2)
    v1 = V1Model(cfg, rng)
    bank = stimuli.StimulusBank(cfg)
    img = bank.image(stimuli.archetype_conditions()[0])[None, :, :]
    a = v1.run(img, np.random.default_rng(10))
    b = v1.run(img, np.random.default_rng(99))
    assert np.allclose(a, b)  # no noise -> rng irrelevant


def test_v1_orientation_tuning():
    """A grating drives the neurons whose preferred orientation matches it;
    rotating the stimulus 90 deg rotates the population response 90 deg."""
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(3))
    bank = stimuli.StimulusBank(cfg)
    img0 = bank.image(stimuli.Condition(0.0, 1.0, 5.0))
    img90 = bank.image(stimuli.Condition(90.0, 1.0, 5.0))
    d0 = v1.feedforward_drive(img0[None])[0]
    d90 = v1.feedforward_drive(img90[None])[0]
    # drive must be structured, not flat
    assert d0.max() > 3.0 * d0.mean()
    pv0 = _circular_mean_deg(v1.preferred_orientation_deg, d0)
    pv90 = _circular_mean_deg(v1.preferred_orientation_deg, d90)
    diff = abs(pv0 - pv90)
    diff = min(diff, 180.0 - diff)
    assert abs(diff - 90.0) < 25.0


def test_v1_contrast_scales_drive():
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(4))
    bank = stimuli.StimulusBank(cfg)
    drives = []
    for c in (0.1, 0.5, 1.0):
        img = bank.image(stimuli.Condition(30.0, c, 5.0))
        drives.append(float(np.linalg.norm(v1.feedforward_drive(img[None])[0])))
    assert drives[0] < drives[1] < drives[2]


def test_v1_dynamics_ramp_and_bounded():
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(5))
    bank = stimuli.StimulusBank(cfg)
    img = bank.image(stimuli.Condition(0.0, 1.0, 5.0))[None]
    traj = v1.run(img, np.random.default_rng(6))[0]  # (n_steps, n_v1)
    early = traj[0].mean()
    late = traj[-3:].mean()
    assert late > early                      # population activity ramps up
    assert traj.min() >= -1e-6               # rectified -> non-negative
    assert traj.max() < 5.0 * cfg.v1_sat     # bounded by the saturating nonlinearity


def test_recurrent_spectral_radius():
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(7))
    eig = np.linalg.eigvalsh(v1.W_rec)
    assert abs(np.max(np.abs(eig)) - cfg.rec_spectral_target) < 1e-6


def test_noise_creates_trial_variability():
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(8))
    bank = stimuli.StimulusBank(cfg)
    img = bank.image(stimuli.Condition(0.0, 1.0, 5.0))[None]
    a = v1.run(img, np.random.default_rng(100))[0]
    b = v1.run(img, np.random.default_rng(200))[0]
    assert not np.allclose(a, b)              # noise -> distinct trajectories
    # but same condition -> trajectories still strongly correlated
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.5


def test_orientation_jitter_scales_with_reliability():
    """Jitter is larger for low contrast and for high dispersion."""
    cfg = _small_cfg()
    easy = orientation_jitter_sd(1.0, 5.0, cfg)
    low_contrast = orientation_jitter_sd(0.01, 5.0, cfg)
    high_dispersion = orientation_jitter_sd(1.0, 90.0, cfg)
    assert low_contrast > easy
    assert high_dispersion > easy


def test_present_shapes_and_jitter():
    """present() returns a trajectory batch; orientation jitter -> variability."""
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(9))
    v1.build_drive_grid()
    conds = [stimuli.Condition(0.0, 1.0, 5.0)] * 8
    traj = v1.present(conds, np.random.default_rng(11))
    assert traj.shape == (8, cfg.n_steps, cfg.n_v1)
    assert np.all(np.isfinite(traj))
    # identical condition, independent trials -> distinct r_bar (jitter + noise)
    r_bar = traj.mean(axis=1)
    assert not np.allclose(r_bar[0], r_bar[1])


def test_drive_grid_interpolation_matches_at_nodes():
    """drive_for() at a grid orientation equals the stored grid value."""
    cfg = _small_cfg()
    v1 = V1Model(cfg, np.random.default_rng(12))
    v1.build_drive_grid()
    node = v1.ori_grid[40]
    interp = v1.drive_for(1.0, 5.0, np.array([node]))[0]
    assert np.allclose(interp, v1.drive_grid[(1.0, 5.0)][40])
