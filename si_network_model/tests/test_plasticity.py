"""Tests for plasticity.py -- three-factor reward-modulated Hebbian + RPE.

The key correctness claim (framework Mechanistic Realisation): repeatedly
pairing a V1 *trajectory* with a rewarded choice drives that choice's
archetype trajectory to the pattern -- the rule recovers the prototype.
"""

from __future__ import annotations

import numpy as np

from si_network_model.config import Config
from si_network_model.decision import DecisionLayer
from si_network_model.plasticity import Plasticity, compute_reward


def _traj_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-timestep cosine between two (n_steps, n_v1) trajectories."""
    num = np.einsum("tn,tn->t", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.mean(num / den))


def _pattern(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return np.abs(rng.normal(size=(cfg.n_steps, cfg.n_v1))) + 0.2


def test_compute_reward_symmetric():
    cfg = Config(reward_mode="symmetric")
    assert compute_reward(1, 1, cfg) == 1.0
    assert compute_reward(0, 0, cfg) == 1.0
    assert compute_reward(1, 0, cfg) == -1.0
    assert compute_reward(0, 1, cfg) == -1.0


def test_compute_reward_asymmetric():
    cfg = Config(reward_mode="asymmetric")
    assert compute_reward(1, 1, cfg) == cfg.reward_hit
    assert compute_reward(1, 0, cfg) == cfg.reward_false_alarm
    assert compute_reward(0, 0, cfg) == cfg.reward_correct_reject
    assert compute_reward(0, 1, cfg) == cfg.reward_miss


def test_update_returns_reward_signal():
    cfg = Config(n_v1=40)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(0))
    traj = _pattern(cfg, np.random.default_rng(1))
    assert plast.update(layer, traj, choice=1, reward=1.0) == 1.0


def test_zero_reward_freezes_the_archetype():
    """A zero-reward outcome leaves the archetype direction unchanged
    (only renormalised) -- this is what lets the NoGo channel rest."""
    cfg = Config(n_v1=40)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(0))
    before = layer.proto_nogo / np.linalg.norm(layer.proto_nogo, axis=1,
                                               keepdims=True)
    plast.update(layer, _pattern(cfg, np.random.default_rng(1)),
                 choice=0, reward=0.0)
    assert np.allclose(layer.proto_nogo, before)


def test_plasticity_recovers_planted_prototype():
    """Pair a fixed trajectory with rewarded Go choices -> proto_go aligns to it.

    Uses an explicit learning rate so the test exercises the *mechanism*
    independently of the (deliberately slow) production default."""
    cfg = Config(n_v1=80, learning_rate=0.05)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(2))
    p = _pattern(cfg, np.random.default_rng(3))
    for _ in range(300):
        plast.update(layer, p, choice=1, reward=1.0)
    assert _traj_cos(layer.proto_go, p) > 0.99


def test_plasticity_separates_two_prototypes():
    cfg = Config(n_v1=80, learning_rate=0.05)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(4))
    rng = np.random.default_rng(5)
    p, q = _pattern(cfg, rng), _pattern(cfg, rng)
    for _ in range(300):
        plast.update(layer, p, choice=1, reward=1.0)
        plast.update(layer, q, choice=0, reward=1.0)
    assert _traj_cos(layer.proto_go, p) > 0.99
    assert _traj_cos(layer.proto_nogo, q) > 0.99


def test_error_moves_archetype_away_from_pattern():
    """A punished Go choice on pattern Q reduces proto_go's alignment with Q."""
    cfg = Config(n_v1=80)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(6))
    rng = np.random.default_rng(7)
    p, q = _pattern(cfg, rng), _pattern(cfg, rng)
    layer.proto_go = p / np.linalg.norm(p, axis=1, keepdims=True)
    before = _traj_cos(layer.proto_go, q)
    plast.update(layer, q, choice=1, reward=-1.0)
    assert _traj_cos(layer.proto_go, q) < before


def test_only_chosen_neuron_updates():
    cfg = Config(n_v1=50)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(8))
    nogo_before = layer.proto_nogo.copy()
    plast.update(layer, _pattern(cfg, np.random.default_rng(9)),
                 choice=1, reward=1.0)
    assert np.array_equal(layer.proto_nogo, nogo_before)
    assert not np.array_equal(layer.proto_go, nogo_before)


def test_archetype_timesteps_stay_unit_norm():
    cfg = Config(n_v1=60)
    plast = Plasticity(cfg)
    layer = DecisionLayer(cfg, np.random.default_rng(10))
    rng = np.random.default_rng(11)
    for _ in range(20):
        plast.update(layer, _pattern(cfg, rng), choice=int(rng.random() < 0.5),
                     reward=float(rng.choice([-1.0, 1.0])))
    assert np.allclose(np.linalg.norm(layer.proto_go, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(layer.proto_nogo, axis=1), 1.0)
