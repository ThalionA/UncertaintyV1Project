# -*- coding: utf-8 -*-
"""Smoke tests for the IO-HMM diagnostic plots (scripts/io_hmm_diagnostics.py).

Drives the real plotting code on a small synthetic fit and asserts a non-trivial
figure is written, for both the velocity and choice-only paths.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("ideal_observer/io_hmm", "scripts"):
    p = os.path.join(REPO_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import io_core            # noqa: E402
import states as states_mod   # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod     # noqa: E402
import simulate as sim    # noqa: E402
import io_hmm_diagnostics  # noqa: E402


def _fit_small(use_velocity, seed=0):
    grids = io_core.IOGrids.default()
    stage1 = emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)
    sl = states_mod.default_v0_states(grids)
    tp = {"Perfect": {"beta": 1.5}, "Thirsty": {"alpha": 0.5, "beta": 1.0},
          "Disengaged": {"alpha": -0.5}, "Naive": {"beta": 1.0}}
    vel = {"Perfect": {"mu": 2.0, "sigma": 1.0}, "Thirsty": {"mu": 1.0, "sigma": 1.0},
           "Disengaged": {"mu": -1.0, "sigma": 1.0}, "Naive": {"mu": 0.0, "sigma": 1.0}}
    A = 0.85 * np.eye(4) + 0.05 * (np.ones((4, 4)) - np.eye(4))
    A /= A.sum(1, keepdims=True)
    pi = np.full(4, 0.25)
    conds = np.array([[s, c, 0.0] for s in [20, 40, 50, 70] for c in (0.5, 1.0)], float)
    rng = np.random.default_rng(seed)
    tl = []
    for _ in range(2):
        tr, _z = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds,
                                       T=200, rng=rng,
                                       vel_per_state=vel if use_velocity else None)
        tl.append(tr)
    params, _ = fit_mod.fit(tl, sl, stage1, grids, use_velocity=use_velocity,
                            n_restarts=1, max_iters=15, seed=1)
    paths = fit_mod.viterbi_paths(params, tl, sl, stage1, grids)
    return params, tl, paths, sl, stage1, grids


def test_plot_animal_fit_with_velocity(tmp_path):
    params, tl, paths, sl, stage1, grids = _fit_small(use_velocity=True)
    out = str(tmp_path / "fit_Cb15.png")
    ret = io_hmm_diagnostics.plot_animal_fit(
        "Cb15", params, tl, paths, sl, stage1, grids, out, use_velocity=True)
    assert ret == out
    assert os.path.exists(out) and os.path.getsize(out) > 5000


def test_plot_animal_fit_choice_only(tmp_path):
    params, tl, paths, sl, stage1, grids = _fit_small(use_velocity=False)
    out = str(tmp_path / "sub" / "fit_choice.png")   # nested dir auto-created
    io_hmm_diagnostics.plot_animal_fit(
        "Cb17", params, tl, paths, sl, stage1, grids, out, use_velocity=False)
    assert os.path.exists(out) and os.path.getsize(out) > 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
