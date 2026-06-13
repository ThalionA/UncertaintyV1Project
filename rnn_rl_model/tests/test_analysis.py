"""Analysis tests on synthetic eval blocks with known ground truth."""

from __future__ import annotations

import numpy as np

from rnn_rl_model import analysis as A


def _synthetic_eval(n_v1=40, per=60, direction=None, noise=0.3, seed=0):
    """Eval block where Go/NoGo r_bar separate along ``direction``."""
    rng = np.random.default_rng(seed)
    oris = np.array([0.0, 30.0, 60.0, 90.0])  # 2 Go, 2 NoGo, no boundary
    if direction is None:
        direction = rng.standard_normal(n_v1)
    direction = direction / np.linalg.norm(direction)
    ori = np.repeat(oris, per)
    label = (ori < 45).astype(int)
    r_bar = (label[:, None] * 2.0 - 1.0) * direction[None, :]
    r_bar = r_bar + noise * rng.standard_normal(r_bar.shape)
    return {
        "orientation": ori,
        "contrast": np.ones_like(ori),
        "dispersion": np.full_like(ori, 5.0),
        "r_bar": r_bar.astype(float),
        "choice": label.copy(),
        "mean_si": (label * 2.0 - 1.0) + 0.1 * rng.standard_normal(len(ori)),
        "var_si": np.full(len(ori), 0.01),
    }, direction


def test_template_alignment_sensitivity():
    """Readout along the true class-mean axis -> cos ~ 1; orthogonal -> ~ 0."""
    ev, d = _synthetic_eval(seed=1)
    w_go, w_nogo = d, -d
    res = A.template_alignment(ev, w_go, w_nogo)
    assert res["cos_readout_template"] > 0.9

    rng = np.random.default_rng(5)
    rand = rng.standard_normal(len(d))
    res0 = A.template_alignment(ev, rand, -rand)
    assert abs(res0["cos_readout_template"]) < 0.5


def test_network_accuracy_perfect_and_chance():
    ev, _ = _synthetic_eval(seed=2)
    assert A.network_accuracy(ev) == 1.0  # choice == label by construction
    rng = np.random.default_rng(0)
    ev["choice"] = rng.integers(0, 2, len(ev["choice"]))
    assert 0.3 < A.network_accuracy(ev) < 0.7


def test_template_whitened_distinguishable():
    """When the noise covariance is anisotropic, template and whitened
    directions differ -- the test can tell them apart."""
    rng = np.random.default_rng(3)
    n_v1, per = 30, 200
    # signal spread equally over dims 0 and 1; dim 1 is noisy, so whitening
    # downweights it and rotates the discriminant toward dim 0.
    direction = np.zeros(n_v1)
    direction[0] = direction[1] = 1.0 / np.sqrt(2.0)
    ori = np.repeat([0.0, 90.0], per)
    label = (ori < 45).astype(int)
    noise = rng.standard_normal((len(ori), n_v1))
    noise[:, 1] *= 6.0
    r_bar = (label[:, None] * 2.0 - 1.0) * direction[None, :] + 0.3 * noise
    ev = {"orientation": ori, "contrast": np.ones_like(ori),
          "dispersion": np.full_like(ori, 5.0), "r_bar": r_bar,
          "choice": label, "mean_si": label * 2.0 - 1.0,
          "var_si": np.full(len(ori), 0.01)}
    res = A.template_alignment(ev, direction, -direction)
    assert res["cos_readout_template"] > 0.9   # readout IS the template
    assert res["cos_readout_whitened"] < 0.95  # whitened direction has rotated away
