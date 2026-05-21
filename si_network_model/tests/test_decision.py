"""Tests for decision.py -- cosine readout, SI(t), and the two-bound DDM.

Archetypes are time-resolved trajectories ``(n_steps, n_v1)``; the cosine
readout compares r(t) to the archetype's value at the same time t.
"""

from __future__ import annotations

import numpy as np

from si_network_model import decision
from si_network_model.config import Config


def _archetype(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return np.abs(rng.normal(size=(cfg.n_steps, cfg.n_v1))) + 0.1


def test_identical_archetypes_give_zero_si():
    cfg = Config(n_v1=40)
    rng = np.random.default_rng(0)
    traj = np.abs(rng.normal(size=(cfg.n_steps, cfg.n_v1)))
    a = _archetype(cfg, rng)
    si = decision.si_trace(traj, a, a, cfg.si_eps)
    assert si.shape == (cfg.n_steps,)
    assert np.allclose(si, 0.0, atol=1e-9)


def test_si_in_bounds():
    cfg = Config(n_v1=40)
    rng = np.random.default_rng(1)
    traj = np.abs(rng.normal(size=(cfg.n_steps, cfg.n_v1)))
    si = decision.si_trace(traj, _archetype(cfg, rng), _archetype(cfg, rng),
                           cfg.si_eps)
    assert np.all(si >= -1.0 - 1e-9) and np.all(si <= 1.0 + 1e-9)


def test_si_positive_when_trajectory_matches_go_archetype():
    """r(t) aligned with the Go archetype and away from NoGo -> SI > 0."""
    cfg = Config(n_v1=60)
    rng = np.random.default_rng(2)
    p = np.abs(rng.normal(size=cfg.n_v1)) + 0.1
    q = np.abs(rng.normal(size=cfg.n_v1)) + 0.1
    traj = np.tile(p, (cfg.n_steps, 1))
    proto_go = np.tile(p, (cfg.n_steps, 1))
    proto_nogo = np.tile(q, (cfg.n_steps, 1))
    si = decision.si_trace(traj, proto_go, proto_nogo, cfg.si_eps)
    assert np.all(si > 0.0)


def test_si_batched_matches_single():
    cfg = Config(n_v1=50)
    rng = np.random.default_rng(3)
    traj = np.abs(rng.normal(size=(4, cfg.n_steps, cfg.n_v1)))
    pg, pn = _archetype(cfg, rng), _archetype(cfg, rng)
    batched = decision.si_trace(traj, pg, pn, cfg.si_eps)
    assert batched.shape == (4, cfg.n_steps)
    assert np.allclose(batched[1], decision.si_trace(traj[1], pg, pn, cfg.si_eps))


def test_accumulator_strong_positive_si_picks_go():
    cfg = Config()
    rng = np.random.default_rng(4)
    si = np.full((200, cfg.n_steps), 0.9)
    choice, rt, xtr = decision.accumulate(si, cfg, rng)
    assert choice.shape == (200,) and rt.shape == (200,)
    assert choice.mean() > 0.95
    assert np.all(rt > 0.0) and np.all(rt <= cfg.t_trial + 1e-9)
    assert xtr.shape == (200, cfg.n_steps)


def test_accumulator_strong_negative_si_picks_nogo():
    cfg = Config()
    rng = np.random.default_rng(5)
    si = np.full((200, cfg.n_steps), -0.9)
    choice, _, _ = decision.accumulate(si, cfg, rng)
    assert choice.mean() < 0.05


def test_accumulator_zero_si_is_unbiased():
    cfg = Config()
    rng = np.random.default_rng(6)
    si = np.zeros((2000, cfg.n_steps))
    choice, _, _ = decision.accumulate(si, cfg, rng)
    assert 0.40 < choice.mean() < 0.60


def test_decision_layer_random_archetypes_unbiased_on_average():
    """Fresh (random) archetypes give a per-layer bias (the drift gain
    amplifies the random-init SI), but no *systematic* bias: the grand mean
    over many random layers is roughly chance."""
    cfg = Config(n_v1=80)
    rng = np.random.default_rng(8)
    layer = decision.DecisionLayer(cfg, np.random.default_rng(0))
    assert layer.proto_go.shape == (cfg.n_steps, cfg.n_v1)
    means = []
    for seed in range(12):
        layer = decision.DecisionLayer(cfg, np.random.default_rng(seed))
        traj = np.abs(rng.normal(size=(300, cfg.n_steps, cfg.n_v1)))
        choice, _, _ = layer.decide(traj, rng)
        means.append(choice.mean())
    assert 0.30 < float(np.mean(means)) < 0.70
