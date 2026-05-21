"""Tests for train.py -- trial sampling, the curriculum, and the cohort."""

from __future__ import annotations

import numpy as np

from si_network_model import train
from si_network_model.agent import Animal
from si_network_model.analysis import summarize_cohort
from si_network_model.config import Config


def test_phase1_samples_only_archetypes():
    cfg = Config()
    rng = np.random.default_rng(0)
    for _ in range(300):
        cond, rewarded = train.sample_trial(1, cfg, rng)
        assert cond.orientation_deg in (0.0, 90.0)
        assert (cond.contrast, cond.dispersion_deg) == (1.0, 5.0)
        assert rewarded == cond.category


def test_sampling_has_flat_category_prior():
    cfg = Config()
    rng = np.random.default_rng(1)
    cats = [train.sample_trial(2, cfg, rng)[1] for _ in range(4000)]
    assert 0.45 < np.mean(cats) < 0.55


def test_phase2_biased_toward_archetypes():
    cfg = Config(archetype_fraction=0.5)
    rng = np.random.default_rng(2)
    n_arch = 0
    for _ in range(4000):
        cond, _ = train.sample_trial(2, cfg, rng)
        if ((cond.contrast, cond.dispersion_deg) == (1.0, 5.0)
                and cond.orientation_deg in (0.0, 90.0)):
            n_arch += 1
    # explicit archetypes (~0.5) plus the small chance a full-grid draw lands there
    assert 0.45 < n_arch / 4000 < 0.65


def test_train_animal_learns():
    """A reward-trained animal goes from ~chance to well above chance."""
    cfg = Config(n_v1=110, img_size=64, phase1_cap=500, phase2_trials=200)
    animal = Animal.build(0, cfg, seed=7)
    log = train.train_animal(animal, np.random.default_rng(8))
    assert log["correct"][:20].mean() < 0.95            # not perfect at the start
    p1 = log["correct"][log["phase"] == 1]
    assert p1[-150:].mean() > 0.85                       # archetypes are learned
    assert log["correct"][-200:].mean() > 0.70           # above chance overall


def test_evaluate_animal_shapes():
    cfg = Config(n_v1=110, img_size=64, phase1_cap=300,
                 phase2_trials=120, n_eval_per_cond=6)
    animal = Animal.build(1, cfg, seed=9)
    train.train_animal(animal, np.random.default_rng(10))
    ev = train.evaluate_animal(animal, np.random.default_rng(11))
    n = 81 * 6
    assert ev["choice"].shape == (n,)
    assert ev["r_bar"].shape == (n, cfg.n_v1)
    assert ev["si_full"].shape == (n, cfg.n_steps)


def test_cohort_run_and_summary():
    """A small cohort runs end-to-end and the summary metrics are sane."""
    cfg = Config(n_animals=2, n_v1=110, img_size=64, phase1_cap=300,
                 phase2_trials=150, n_eval_per_cond=8, io_samples_per_cond=60)
    cohort = train.run_cohort(cfg, verbose=False)
    assert len(cohort.runs) == 2
    summary = summarize_cohort(cohort)
    assert 0.5 < summary["cohort_network_acc"] <= 1.0
    assert 0.5 < summary["cohort_neural_io_acc"] <= 1.0
    # the network cannot beat the optimal readout of its own V1
    assert summary["cohort_network_acc"] <= summary["cohort_neural_io_acc"] + 0.05
    assert 0.0 < summary["cohort_efficiency"] <= 1.05
