"""Tests for analysis.py -- metric functions on synthetic data."""

from __future__ import annotations

import numpy as np

from si_network_model import analysis


def test_category():
    out = analysis.category(np.array([0.0, 30.0, 44.0, 46.0, 90.0]))
    assert list(out) == [1, 1, 1, 0, 0]


def test_rolling_accuracy_bounds_and_mean():
    correct = np.tile([1, 1, 0, 0], 60)  # 50 % accuracy
    roll = analysis.rolling_accuracy(correct, window=20)
    assert np.all(roll >= 0.0) and np.all(roll <= 1.0)
    assert abs(roll.mean() - 0.5) < 0.1


def test_psychometric_levels_and_curve():
    ori = np.array([0.0, 0.0, 45.0, 45.0, 90.0, 90.0])
    p_go = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    levels, curve = analysis.psychometric(ori, p_go)
    assert list(levels) == [0.0, 45.0, 90.0]
    assert np.allclose(curve, [1.0, 0.5, 0.0])


def _synthetic_eval(n_v1: int = 40, width: float = 0.05,
                    seed: int = 0) -> dict:
    """A fake evaluation block: two well-separated archetype bundles."""
    rng = np.random.default_rng(seed)
    per = 40
    go_mean = np.abs(rng.normal(size=n_v1)) + 0.6
    nogo_mean = np.abs(rng.normal(size=n_v1)) + 0.6
    ori, contrast, disp, r_bar, mean_si = [], [], [], [], []
    for orientation, mean in [(0.0, go_mean), (90.0, nogo_mean)]:
        for _ in range(per):
            ori.append(orientation)
            contrast.append(1.0)
            disp.append(5.0)
            r_bar.append(mean + width * rng.normal(size=n_v1))
            base = 0.3 if orientation < 45 else -0.3
            mean_si.append(base + 0.02 * rng.normal())
    ori = np.array(ori)
    return {
        "orientation": ori, "contrast": np.array(contrast),
        "dispersion": np.array(disp), "r_bar": np.array(r_bar),
        "choice": analysis.category(ori), "mean_si": np.array(mean_si),
    }


def test_bundle_concentration_tighter_bundle_higher_kappa():
    tight = analysis.bundle_concentration(_synthetic_eval(width=0.03))
    broad = analysis.bundle_concentration(_synthetic_eval(width=0.20))
    assert np.isfinite(tight) and np.isfinite(broad)
    assert tight > broad      # tighter bundle -> higher concentration


def test_lapse_rate_zero_when_choices_perfect():
    ev = _synthetic_eval()
    assert analysis.lapse_rate(ev) == 0.0


def test_trial_mean_si_spread_nonnegative():
    spread = analysis.trial_mean_si_spread(_synthetic_eval())
    assert spread >= 0.0 and np.isfinite(spread)


def test_network_accuracy_perfect_choices():
    ev = _synthetic_eval()
    assert analysis.network_accuracy(ev) == 1.0
