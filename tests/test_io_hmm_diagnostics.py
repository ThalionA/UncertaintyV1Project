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
    sl = states_mod.default_v0_states(grids, with_velocity=use_velocity)
    tp = {"Perfect": {"beta": 1.5}, "Thirsty": {"alpha": 0.5, "beta": 1.0},
          "Disengaged": {"alpha": -0.5}, "Naive": {"beta": 1.0}}
    vel = {"Perfect": {"beta_vel": 1.5, "alpha_vel": 0.0, "sigma_vel": 1.0},
           "Thirsty": {"beta_vel": 0.5, "alpha_vel": 1.0, "sigma_vel": 1.0},
           "Disengaged": {"alpha_vel": -1.0, "sigma_vel": 1.0},  # beta_vel=0 fixed
           "Naive": {"beta_vel": 0.0, "alpha_vel": 0.0, "sigma_vel": 1.0}}
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


def _fake_summaries(n=3, with_velocity=True):
    names = ["Perfect", "Thirsty", "Disengaged", "Naive"]
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        occ = rng.dirichlet(np.ones(4))
        A = 0.8 * np.eye(4) + 0.2 / 4
        A = (A / A.sum(1, keepdims=True))
        s = {
            "animal": f"Cb{i}", "n_sessions": 3, "n_trials": 900,
            "final_log_lik": -1600.0 - 10 * i, "n_em_iters": 20,
            "state_names": names,
            "state_occupancy": dict(zip(names, occ.tolist())),
            "pi": [0.25] * 4, "A": A.tolist(),
            "psych_per_state": {"Thirsty": {"alpha": 0.5 + 0.1 * i, "beta": 1.0},
                                "Disengaged": {"alpha": -0.4}},
            "vel_per_state": ({n: {"beta_vel": float(k), "alpha_vel": 0.0,
                                    "sigma_vel": 1.0}
                               for k, n in enumerate(names)}
                              if with_velocity else {}),
        }
        out.append(s)
    return out


def test_plot_group_summary_with_velocity(tmp_path):
    out = str(tmp_path / "group_summary.png")
    io_hmm_diagnostics.plot_group_summary(_fake_summaries(with_velocity=True), out)
    assert os.path.exists(out) and os.path.getsize(out) > 5000


def test_plot_group_summary_choice_only(tmp_path):
    out = str(tmp_path / "g2.png")
    io_hmm_diagnostics.plot_group_summary(_fake_summaries(with_velocity=False), out)
    assert os.path.exists(out) and os.path.getsize(out) > 5000


def test_write_group_table(tmp_path):
    import csv
    out = str(tmp_path / "group_table.csv")
    io_hmm_diagnostics.write_group_table(_fake_summaries(n=3), out)
    with open(out) as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:5] == ["animal", "n_sessions", "n_trials",
                           "final_log_lik", "ll_per_trial"]
    assert len(rows) == 1 + 3 + 1                 # header + 3 animals + group mean
    assert rows[-1][0] == "GROUP_MEAN"
    occ_cols = [c for c in rows[0] if c.startswith("occ_")]
    assert len(occ_cols) == 4                     # one occupancy col per state


def test_group_aggregation_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        io_hmm_diagnostics.plot_group_summary([], str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        io_hmm_diagnostics.write_group_table([], str(tmp_path / "x.csv"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
