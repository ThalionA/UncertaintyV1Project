"""Tests for ideal_observer.py -- stimulus-IO and neural-IO."""

from __future__ import annotations

import numpy as np

from si_network_model import stimuli
from si_network_model.config import Config
from si_network_model.ideal_observer import (
    NeuralIdealObserver,
    StimulusIdealObserver,
    decision_threshold,
)
from si_network_model.v1_model import V1Model


def test_decision_threshold_symmetric_is_zero():
    assert abs(decision_threshold(Config(reward_mode="symmetric"))) < 1e-9


def test_decision_threshold_asymmetric_is_go_biased():
    # Asymmetric water-task reward -> negative threshold -> Go-bias.
    assert decision_threshold(Config(reward_mode="asymmetric")) < 0.0


def test_stimulus_io_psychometric_monotonic():
    """On an easy stimulus P(Go) falls monotonically as orientation 0 -> 90."""
    io = StimulusIdealObserver(Config())
    pgo = [io.p_go(ori, contrast=1.0, dispersion=5.0)
           for ori in (0, 15, 30, 40, 50, 60, 75, 90)]
    assert all(earlier >= later - 1e-6 for earlier, later in zip(pgo, pgo[1:]))
    assert pgo[0] > 0.9        # clear Go
    assert pgo[-1] < 0.1       # clear NoGo


def test_stimulus_io_easy_beats_hard():
    """A high-contrast/low-dispersion stimulus is more discriminable."""
    io = StimulusIdealObserver(Config())
    easy = io.p_go(0.0, contrast=1.0, dispersion=5.0)
    hard = io.p_go(0.0, contrast=0.01, dispersion=90.0)
    assert easy > hard
    assert abs(hard - 0.5) < 0.15   # hardest stimulus is near chance


def test_stimulus_io_boundary_is_chance():
    io = StimulusIdealObserver(Config())
    assert abs(io.p_go(45.0, contrast=1.0, dispersion=5.0) - 0.5) < 0.05


def test_neural_io_fit_and_decode():
    """Neural-IO log-odds is positive for Go stimuli, negative for NoGo."""
    cfg = Config(n_v1=90, img_size=48, io_samples_per_cond=40)
    v1 = V1Model(cfg, np.random.default_rng(0))
    v1.build_drive_grid()
    nio = NeuralIdealObserver.fit(v1, cfg, np.random.default_rng(1))

    go_rbar = v1.present([stimuli.Condition(0.0, 1.0, 5.0)] * 60,
                         np.random.default_rng(2)).mean(axis=1)
    nogo_rbar = v1.present([stimuli.Condition(90.0, 1.0, 5.0)] * 60,
                           np.random.default_rng(3)).mean(axis=1)

    assert nio.log_odds(go_rbar).mean() > 0.0
    assert nio.log_odds(nogo_rbar).mean() < 0.0
    pgo = nio.p_go(go_rbar)
    assert np.all(pgo >= 0.0) and np.all(pgo <= 1.0)


def test_neural_io_above_chance_on_archetypes():
    cfg = Config(n_v1=90, img_size=48, io_samples_per_cond=40)
    v1 = V1Model(cfg, np.random.default_rng(4))
    v1.build_drive_grid()
    nio = NeuralIdealObserver.fit(v1, cfg, np.random.default_rng(5))

    correct = 0
    total = 0
    for cond in stimuli.archetype_conditions():
        rbar = v1.present([cond] * 50, np.random.default_rng(6)).mean(axis=1)
        pred_go = nio.p_go(rbar) > 0.5
        correct += int(np.sum(pred_go == (cond.category == 1)))
        total += len(pred_go)
    assert correct / total > 0.8     # archetypes are easy -> near-perfect
