"""Training tests: the actor-critic must learn the real task (sensitivity) and
must NOT improve when the reward signal is uninformative (specificity)."""

from __future__ import annotations

import numpy as np

from si_network_model.config import Config

from rnn_rl_model import train as T
from rnn_rl_model.config import RLConfig

CFG = RLConfig(base=Config(n_v1=80), n_train_trials=2400, batch_size=64)


def test_learns_real_task():
    """Accuracy rises well above chance on the Go/NoGo task (both V1 modes)."""
    v1 = T.build_v1(CFG, seed=7)
    for train_v1 in (False, True):
        _, hist = T.train_agent(v1, CFG, seed=7, train_v1=train_v1)
        early = hist["acc"][:5].mean()
        late = hist["acc"][-10:].mean()
        assert early < 0.65, f"started too good ({early:.2f})"
        assert late > 0.70, f"failed to learn ({late:.2f}, train_v1={train_v1})"


def test_uninformative_reward_does_not_learn(monkeypatch):
    """With reward independent of the action, accuracy stays at chance
    (specificity: learning is reward-driven, not an artefact of the loop)."""
    rng = np.random.default_rng(0)

    def random_reward(action, rewarded, base):
        return rng.choice([-1.0, 1.0], size=len(action))

    monkeypatch.setattr(T, "batched_reward", random_reward)
    v1 = T.build_v1(CFG, seed=3)
    _, hist = T.train_agent(v1, CFG, seed=3, train_v1=False)
    assert 0.40 < hist["acc"][-10:].mean() < 0.60


def test_batched_reward_symmetric():
    base = Config()
    action = np.array([1, 1, 0, 0])
    rewarded = np.array([1, 0, 0, 1])
    r = T.batched_reward(action, rewarded, base)
    assert list(r) == [1.0, -1.0, 1.0, -1.0]
