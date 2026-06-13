"""Model unit tests: the torch port must match the numpy reference, and the
fixed/trained switch must control what is a Parameter."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from si_network_model.config import Config
from si_network_model.decision import si_trace as np_si_trace

from rnn_rl_model.config import RLConfig
from rnn_rl_model.model import RecurrentReadoutAgent

CFG = RLConfig(base=Config(n_v1=48))


def _agent(train_v1=False, seed=0):
    W = np.zeros((CFG.n_v1, CFG.n_v1), dtype=np.float32)
    return RecurrentReadoutAgent(W, CFG, train_v1=train_v1, seed=seed)


def test_si_matches_numpy_reference():
    """The torch SI(t) equals si_network_model.decision.si_trace bit-for-bit
    (the framework's bounded prototype SI), for a static archetype broadcast
    over time."""
    agent = _agent()
    T, n = CFG.n_steps, CFG.n_v1
    rng = np.random.default_rng(1)
    traj = rng.standard_normal((4, T, n)).astype(np.float32)
    wgo, wnogo = agent.archetype_vectors()

    _, _, si_torch = agent.similarities(torch.from_numpy(traj))
    si_torch = si_torch.detach().numpy()
    proto_go = np.tile(wgo, (T, 1))
    proto_nogo = np.tile(wnogo, (T, 1))
    si_np = np_si_trace(traj, proto_go, proto_nogo, CFG.base.si_eps)
    assert np.allclose(si_torch, si_np, atol=1e-5)


def test_fixed_vs_trained_parameter():
    """W_rec is a Parameter iff train_v1; archetypes always learn."""
    assert not isinstance(_agent(False).W_rec, nn.Parameter)
    assert isinstance(_agent(True).W_rec, nn.Parameter)
    for tv in (False, True):
        names = {n for n, _ in _agent(tv).named_parameters()}
        assert {"w_go", "w_nogo", "bias", "beta_raw"} <= names
        assert ("W_rec" in names) == tv


def test_rollout_bounded_and_shaped():
    """Saturating phi keeps the trajectory bounded; shape is (B, T, n)."""
    agent = _agent()
    gen = torch.Generator().manual_seed(0)
    drive = 5.0 * torch.ones(3, CFG.n_v1)  # large drive -> would blow up if unbounded
    traj = agent.rollout(drive, gen)
    assert traj.shape == (3, CFG.n_steps, CFG.n_v1)
    assert torch.isfinite(traj).all()
    assert traj.abs().max() <= CFG.base.v1_sat + 1e-4


def test_beta_positive():
    """The softplus-parametrised decision gain is positive and ~ the init."""
    agent = _agent()
    assert float(agent.beta) > 0
    assert abs(float(agent.beta) - CFG.policy_beta_init) < 0.5
