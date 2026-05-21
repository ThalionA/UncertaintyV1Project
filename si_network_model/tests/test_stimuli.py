"""Tests for stimuli.py -- the grating recipe (matches CreateGratings.m).

Written test-first. Correctness is defined on synthetic stimuli with known
properties: a zero-contrast grating must be uniform, contrast must set the
luminance range, dispersion must set the orientation-mixture bandwidth, and
the orientation parameter must actually rotate the grating.
"""

from __future__ import annotations

import numpy as np

from si_network_model import stimuli
from si_network_model.config import Config


def _dominant_fft_angle(img: np.ndarray) -> float:
    """Angle (deg, mod 180) of the dominant non-DC Fourier component."""
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    power = np.abs(f) ** 2
    n = img.shape[0]
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    power[r < 3] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    return float(np.degrees(np.arctan2(iy - cy, ix - cx)) % 180.0)


def test_mixture_weights_sum_to_one():
    for disp in (5.0, 30.0, 90.0):
        w = stimuli.mixture_weights(disp, n_components=9)
        assert w.shape == (9,)
        assert abs(w.sum() - 1.0) < 1e-9
        assert np.all(w >= 0.0)


def test_dispersion_controls_bandwidth():
    """Low dispersion -> one component dominates; high -> spread."""
    w_tight = stimuli.mixture_weights(5.0, n_components=9)
    w_broad = stimuli.mixture_weights(90.0, n_components=9)
    centre = 9 // 2
    assert w_tight[centre] > 0.99      # essentially a single orientation
    assert w_broad[centre] < 0.30      # genuinely mixed


def test_zero_contrast_is_uniform():
    cfg = Config()
    img = stimuli.render_grating(0.0, contrast=0.0, dispersion_deg=5.0, cfg=cfg)
    assert img.shape == (cfg.img_size, cfg.img_size)
    assert img.std() < 1e-9            # flat grey field


def test_contrast_sets_luminance_range():
    cfg = Config()
    for c in (0.25, 0.5, 1.0):
        img = stimuli.render_grating(30.0, contrast=c, dispersion_deg=5.0, cfg=cfg)
        # square-wave grating spans grey +/- c*grey -> peak-to-peak == c
        assert abs((img.max() - img.min()) - c) < 0.02


def test_render_is_deterministic():
    cfg = Config()
    a = stimuli.render_grating(40.0, 1.0, 5.0, cfg)
    b = stimuli.render_grating(40.0, 1.0, 5.0, cfg)
    assert np.array_equal(a, b)


def test_orientation_actually_rotates_grating():
    """A 0-deg and a 90-deg grating differ by ~90 deg in Fourier angle."""
    cfg = Config()
    g0 = stimuli.render_grating(0.0, 1.0, 5.0, cfg)
    g90 = stimuli.render_grating(90.0, 1.0, 5.0, cfg)
    diff = abs(_dominant_fft_angle(g0) - _dominant_fft_angle(g90))
    diff = min(diff, 180.0 - diff)
    assert abs(diff - 90.0) < 15.0


def test_enumerate_conditions():
    conds = stimuli.enumerate_conditions()
    assert len(conds) == 81                       # 9 orientations x 9 pairs
    assert len({c.key for c in conds}) == 81      # all unique
    archetypes = stimuli.archetype_conditions()
    assert len(archetypes) == 2
    for a in archetypes:
        assert a in conds                         # archetypes live in the grid


def test_condition_category():
    go = stimuli.Condition(orientation_deg=0.0, contrast=1.0, dispersion_deg=5.0)
    nogo = stimuli.Condition(orientation_deg=90.0, contrast=1.0, dispersion_deg=5.0)
    assert go.category == 1       # Go  (orientation < 45)
    assert nogo.category == 0     # NoGo (orientation > 45)


def test_stimulus_bank_caches_all_conditions():
    cfg = Config()
    bank = stimuli.StimulusBank(cfg)
    conds = stimuli.enumerate_conditions()
    img_a = bank.image(conds[10])
    img_b = bank.image(conds[10])
    assert img_a is img_b                          # cached, same object
    assert img_a.shape == (cfg.img_size, cfg.img_size)
