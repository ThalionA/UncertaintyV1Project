# -*- coding: utf-8 -*-
"""Tests for the real-data loader (io_hmm.data_io).

The real export is gitignored, so these write synthetic ``.mat`` fixtures that
mirror the structure of ``VR_Decoder_Data_Export.mat`` (``TrialTbl_Struct``) and
``IOResults.mat`` with ``scipy.io.savemat``, then exercise the same code paths
the production loader uses on the real files -- including an end-to-end fit
through the real-data path.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import scipy.io as sio

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IO_HMM = os.path.join(REPO_ROOT, "ideal_observer", "io_hmm")
if IO_HMM not in sys.path:
    sys.path.insert(0, IO_HMM)

import io_core            # noqa: E402
import states as states_mod   # noqa: E402
import emissions as emissions_mod  # noqa: E402
import fit as fit_mod     # noqa: E402
import simulate as sim    # noqa: E402
import data_io            # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_trial_table(path, *, animals, sessions_per_animal, trials_per_session,
                       rng, with_nan=False):
    """Write a synthetic TrialTbl_Struct (struct-of-arrays) export."""
    animal_col, session_col = [], []
    s_deg, contrast, dispersion, choice, velocity = [], [], [], [], []
    for a in animals:
        for s in range(sessions_per_animal):
            T = trials_per_session
            animal_col += [a] * T
            session_col += [s] * T
            s_deg += list(rng.uniform(0, 90, T))
            contrast += list(rng.choice([0.5, 1.0], T))
            dispersion += list(rng.uniform(0, 10, T))
            choice += list(rng.integers(0, 2, T).astype(float))
            velocity += list(rng.normal(2.0, 1.0, T))
    velocity = np.array(velocity)
    if with_nan:
        velocity[::13] = np.nan  # missing velocities, like aborted-kinematics trials
    tbl = {
        "animal": np.array(animal_col, dtype=object),
        "session": np.array(session_col, dtype=float),
        "abs_from_go": np.array(s_deg),
        "contrast": np.array(contrast),
        "dispersion": np.array(dispersion),
        "goChoice": np.array(choice),
        "preRZ_velocity": velocity,
    }
    sio.savemat(path, {"TrialTbl_Struct": tbl})


def _write_ioresults(path, *, group, animals=None, kappa_min=1.0):
    res = {
        "meta": {"fixed_params": {"kappa_min": kappa_min}},
        "group": {"params": group},
    }
    if animals is not None:
        res["animals"] = np.array([
            {"tag": tag, "fit": {"full_params": pars}}
            for tag, pars in animals
        ], dtype=object)
    sio.savemat(path, {"IOResults": res})


# ---------------------------------------------------------------------------
# Trial-table loading
# ---------------------------------------------------------------------------


def test_list_animals_and_field_mapping(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(0)
    _write_trial_table(p, animals=["Cb15", "Cb17"], sessions_per_animal=2,
                       trials_per_session=40, rng=rng)
    assert data_io.list_animals(p) == ["Cb15", "Cb17"]

    tl = data_io.load_animal_trials("Cb15", export_path=p, min_trials=10)
    assert len(tl) == 2                       # split by session
    for tr in tl:
        assert tr.has_velocity
        assert tr.n_trials == 40
        assert np.all((tr.s_deg >= 0) & (tr.s_deg <= 90))
        assert set(np.unique(tr.choice)).issubset({0.0, 1.0})


def test_contrast_clamp_and_velocity_standardized(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(1)
    _write_trial_table(p, animals=["Cb15"], sessions_per_animal=1,
                       trials_per_session=300, rng=rng)
    tl = data_io.load_animal_trials("Cb15", export_path=p, min_trials=10)
    v = np.concatenate([t.velocity for t in tl])
    assert abs(v.mean()) < 1e-6 and abs(v.std() - 1.0) < 1e-6   # robust z
    c = np.concatenate([t.c for t in tl])
    assert set(np.unique(c)).issubset({0.5, 1.0})               # clamped/binned


def test_with_velocity_false_gives_choice_only(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(2)
    _write_trial_table(p, animals=["Cb15"], sessions_per_animal=1,
                       trials_per_session=60, rng=rng)
    tl = data_io.load_animal_trials("Cb15", export_path=p, with_velocity=False,
                                    min_trials=10)
    assert all(not t.has_velocity for t in tl)


def test_nan_velocities_mean_filled_not_dropped(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(3)
    _write_trial_table(p, animals=["Cb15"], sessions_per_animal=1,
                       trials_per_session=130, rng=rng, with_nan=True)
    tl = data_io.load_animal_trials("Cb15", export_path=p, min_trials=10)
    v = np.concatenate([t.velocity for t in tl])
    assert np.all(np.isfinite(v))            # NaNs filled, trials kept
    assert sum(t.n_trials for t in tl) == 130


def test_min_trials_filters_short_sessions(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(4)
    _write_trial_table(p, animals=["Cb15"], sessions_per_animal=3,
                       trials_per_session=15, rng=rng)
    with pytest.raises(ValueError):          # all sessions below threshold
        data_io.load_animal_trials("Cb15", export_path=p, min_trials=50)
    tl = data_io.load_animal_trials("Cb15", export_path=p, min_trials=10)
    assert len(tl) == 3


def test_no_session_returns_single_sequence(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(5)
    _write_trial_table(p, animals=["Cb15"], sessions_per_animal=3,
                       trials_per_session=30, rng=rng)
    tl = data_io.load_animal_trials("Cb15", export_path=p, by_session=False,
                                    min_trials=10)
    assert len(tl) == 1 and tl[0].n_trials == 90


def test_load_all_trials(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(6)
    _write_trial_table(p, animals=["Cb15", "Cb21"], sessions_per_animal=2,
                       trials_per_session=40, rng=rng)
    allt = data_io.load_all_trials(p, min_trials=10)
    assert set(allt) == {"Cb15", "Cb21"}
    assert all(len(v) == 2 for v in allt.values())


def test_missing_animal_raises(tmp_path):
    p = str(tmp_path / "export.mat")
    rng = np.random.default_rng(7)
    _write_trial_table(p, animals=["Cb15"], sessions_per_animal=1,
                       trials_per_session=40, rng=rng)
    with pytest.raises(ValueError):
        data_io.load_animal_trials("Nope", export_path=p)


# ---------------------------------------------------------------------------
# Stage-1 params from IOResults
# ---------------------------------------------------------------------------


def test_load_stage1_group_level(tmp_path):
    p = str(tmp_path / "io.mat")
    _write_ioresults(p, group={"kappa_amp": 22.0, "c_power": 1.10,
                               "d_power": 0.035}, kappa_min=1.0)
    s1 = data_io.load_stage1(ioresults_path=p)
    assert (s1.kappa_amp, s1.c_power, s1.d_power, s1.kappa_min) == \
        pytest.approx((22.0, 1.10, 0.035, 1.0))


def test_load_stage1_prefers_matching_animal(tmp_path):
    p = str(tmp_path / "io.mat")
    _write_ioresults(
        p,
        group={"kappa_amp": 10.0, "c_power": 1.0, "d_power": 0.0},
        animals=[("Cb15", {"kappa_amp": 20.0, "c_power": 1.2, "d_power": 0.05}),
                 ("Cb21", {"kappa_amp": 30.0, "c_power": 1.3, "d_power": 0.07})],
    )
    s_group = data_io.load_stage1(ioresults_path=p)
    assert s_group.kappa_amp == pytest.approx(10.0)        # group fallback
    s_cb21 = data_io.load_stage1("Cb21", ioresults_path=p)
    assert (s_cb21.kappa_amp, s_cb21.c_power) == pytest.approx((30.0, 1.3))


def test_load_stage1_missing_keys_raises(tmp_path):
    p = str(tmp_path / "io.mat")
    sio.savemat(p, {"IOResults": {"meta": {"fixed_params": {"kappa_min": 1.0}}}})
    with pytest.raises(ValueError):
        data_io.load_stage1(ioresults_path=p)


# ---------------------------------------------------------------------------
# End-to-end: real-data path on a controlled fixture
# ---------------------------------------------------------------------------


def test_end_to_end_fit_through_real_data_path(tmp_path):
    """Simulate sessions with known states, write them as a TrialTbl export,
    load via data_io, and fit through the production path with velocity."""
    grids = io_core.IOGrids.default()
    stage1 = emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)
    bimodal = io_core.prior_bimodal(grids)
    sl = [states_mod.IOState("Lo", bimodal, states_mod.PsychSpec(beta=1., gamma=0., delta=0.),
                             vel=states_mod.VelocitySpec(beta_vel=0.)),
          states_mod.IOState("Hi", bimodal, states_mod.PsychSpec(beta=1., gamma=0., delta=0.),
                             vel=states_mod.VelocitySpec(beta_vel=0.))]
    tp = {"Lo": {"alpha": -0.5}, "Hi": {"alpha": 0.5}}
    vel = {"Lo": {"alpha_vel": 0.0, "sigma_vel": 1.0},
           "Hi": {"alpha_vel": 3.0, "sigma_vel": 1.0}}
    A = np.array([[0.92, 0.08], [0.08, 0.92]]); pi = np.array([0.5, 0.5])
    conds = np.array([[s, c, 0.0] for s in [30, 40, 45, 50, 60] for c in (0.5, 1.0)], float)
    rng = np.random.default_rng(0)

    # build a TrialTbl export from two simulated "sessions" of one animal
    animal_col, session_col = [], []
    s_deg, contrast, dispersion, choice, velocity = [], [], [], [], []
    for s in range(2):
        tr, _ = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds,
                                      T=500, rng=rng, vel_per_state=vel)
        n = tr.n_trials
        animal_col += ["Cb15"] * n; session_col += [s] * n
        s_deg += list(tr.s_deg); contrast += list(tr.c); dispersion += list(tr.d)
        choice += list(tr.choice); velocity += list(tr.velocity)
    p = str(tmp_path / "export.mat")
    sio.savemat(p, {"TrialTbl_Struct": {
        "animal": np.array(animal_col, dtype=object),
        "session": np.array(session_col, dtype=float),
        "abs_from_go": np.array(s_deg), "contrast": np.array(contrast),
        "dispersion": np.array(dispersion), "goChoice": np.array(choice),
        "preRZ_velocity": np.array(velocity)}})

    # load raw (no z-score) so the fitted markers recover the true [0, 3] scale
    trials_list = data_io.load_animal_trials("Cb15", export_path=p,
                                             standardize_velocity=False,
                                             min_trials=20)
    assert len(trials_list) == 2 and all(t.has_velocity for t in trials_list)

    params, history = fit_mod.fit(trials_list, sl, stage1, grids,
                                  use_velocity=True, n_restarts=2, max_iters=40,
                                  seed=1)
    assert np.all(np.diff(history) >= -1e-6)         # monotone EM on loaded data
    assert set(params.vel_per_state) == {"Lo", "Hi"}
    mus = sorted(v["alpha_vel"] for v in params.vel_per_state.values())
    assert np.allclose(mus, [0.0, 3.0], atol=0.5)    # baselines recovered
    paths = fit_mod.viterbi_paths(params, trials_list, sl, stage1, grids)
    assert all(len(p_) == t.n_trials for p_, t in zip(paths, trials_list))


def test_end_to_end_standardized_velocity_separates_states(tmp_path):
    """Default standardize_velocity=True rescales the channel; the two markers
    must still be cleanly separated (separation ~ true_d' / pooled_std)."""
    grids = io_core.IOGrids.default()
    stage1 = emissions_mod.Stage1Params(kappa_amp=8.0, c_power=1.0, d_power=1.0)
    bimodal = io_core.prior_bimodal(grids)
    sl = [states_mod.IOState("Lo", bimodal, states_mod.PsychSpec(beta=1., gamma=0., delta=0.),
                             vel=states_mod.VelocitySpec(beta_vel=0.)),
          states_mod.IOState("Hi", bimodal, states_mod.PsychSpec(beta=1., gamma=0., delta=0.),
                             vel=states_mod.VelocitySpec(beta_vel=0.))]
    tp = {"Lo": {"alpha": -0.5}, "Hi": {"alpha": 0.5}}
    vel = {"Lo": {"alpha_vel": 0.0, "sigma_vel": 1.0},
           "Hi": {"alpha_vel": 3.0, "sigma_vel": 1.0}}
    A = np.array([[0.92, 0.08], [0.08, 0.92]]); pi = np.array([0.5, 0.5])
    conds = np.array([[s, c, 0.0] for s in [30, 40, 45, 50, 60] for c in (0.5, 1.0)], float)
    rng = np.random.default_rng(0)
    animal_col, session_col, s_deg, contrast, dispersion, choice, velocity = \
        [], [], [], [], [], [], []
    for s in range(2):
        tr, _ = sim.simulate_sequence(sl, stage1, grids, pi, A, tp, conds,
                                      T=500, rng=rng, vel_per_state=vel)
        n = tr.n_trials
        animal_col += ["Cb15"] * n; session_col += [s] * n
        s_deg += list(tr.s_deg); contrast += list(tr.c); dispersion += list(tr.d)
        choice += list(tr.choice); velocity += list(tr.velocity)
    p = str(tmp_path / "export.mat")
    sio.savemat(p, {"TrialTbl_Struct": {
        "animal": np.array(animal_col, dtype=object),
        "session": np.array(session_col, dtype=float),
        "abs_from_go": np.array(s_deg), "contrast": np.array(contrast),
        "dispersion": np.array(dispersion), "goChoice": np.array(choice),
        "preRZ_velocity": np.array(velocity)}})

    trials_list = data_io.load_animal_trials("Cb15", export_path=p, min_trials=20)
    params, _ = fit_mod.fit(trials_list, sl, stage1, grids, use_velocity=True,
                            n_restarts=2, max_iters=40, seed=1)
    mus = sorted(v["alpha_vel"] for v in params.vel_per_state.values())
    assert mus[1] - mus[0] > 1.0    # states still clearly separated post-z-score


# ---------------------------------------------------------------------------
# MATLAB v7.3 (HDF5) reading -- the real export/IOResults are saved this way,
# which scipy.io.loadmat cannot read. These build MATLAB-style HDF5 fixtures
# (structs->groups, char->uint16, cell arrays->object refs) and exercise the
# h5py fallback through the public API.
# ---------------------------------------------------------------------------


def _mat_char(grp, name, s):
    import h5py
    d = grp.create_dataset(name, data=np.array([ord(c) for c in s],
                                               dtype="uint16").reshape(-1, 1))
    d.attrs["MATLAB_class"] = np.bytes_(b"char")


def _mat_dbl(grp, name, val):
    arr = np.atleast_2d(np.asarray(val, dtype="float64"))
    d = grp.create_dataset(name, data=arr)
    d.attrs["MATLAB_class"] = np.bytes_(b"double")


def _cellstr_refs(refs, base, strings):
    import h5py
    out = []
    for i, s in enumerate(strings):
        d = refs.create_dataset(f"{base}{i}",
                                data=np.array([ord(c) for c in s],
                                              dtype="uint16").reshape(-1, 1))
        d.attrs["MATLAB_class"] = np.bytes_(b"char")
        out.append(d.ref)
    return out


def _write_v73(path, *, trial_tbl=None, ioresults=None):
    import h5py
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        if trial_tbl is not None:
            T = f.create_group("TrialTbl_Struct")
            T.attrs["MATLAB_class"] = np.bytes_(b"struct")
            for k, v in trial_tbl.items():
                if k == "animal":
                    ref_arr = _cellstr_refs(refs, "an", v)
                    d = T.create_dataset("animal", data=np.array(
                        ref_arr, dtype=h5py.ref_dtype).reshape(1, -1))
                    d.attrs["MATLAB_class"] = np.bytes_(b"cell")
                else:
                    _mat_dbl(T, k, np.asarray(v).reshape(-1, 1))
        if ioresults is not None:
            IO = f.create_group("IOResults")
            IO.attrs["MATLAB_class"] = np.bytes_(b"struct")
            meta = IO.create_group("meta")
            _mat_dbl(meta.create_group("fixed_params"), "kappa_min", 1.0)
            ms = meta.create_group("model_spec")
            name_refs = _cellstr_refs(refs, "nm", ioresults["fit_params"])
            d = ms.create_dataset("fit_params", data=np.array(
                name_refs, dtype=h5py.ref_dtype).reshape(1, -1))
            d.attrs["MATLAB_class"] = np.bytes_(b"cell")
            _mat_dbl(IO.create_group("group"), "params", ioresults["group"])
            arefs = []
            for ai, (tag, pars) in enumerate(ioresults["animals"]):
                ag = refs.create_group(f"animal{ai}")
                ag.attrs["MATLAB_class"] = np.bytes_(b"struct")
                _mat_char(ag, "tag", tag)
                fit = ag.create_group("fit")
                full = fit.create_group("full_params")
                for kk, vv in pars.items():
                    _mat_dbl(full, kk, vv)
                arefs.append(ag.ref)
            d = IO.create_dataset("animals", data=np.array(
                arefs, dtype=h5py.ref_dtype).reshape(1, -1))
            d.attrs["MATLAB_class"] = np.bytes_(b"cell")


def test_load_animal_trials_from_v73_export(tmp_path):
    pytest.importorskip("h5py")
    p = str(tmp_path / "export_v73.mat")
    rng = np.random.default_rng(0)
    T = 80
    tbl = {
        "animal": ["Cb15"] * T + ["Cb21"] * T,
        "session": np.concatenate([np.zeros(T), np.zeros(T)]),
        "abs_from_go": rng.uniform(0, 90, 2 * T),
        "contrast": rng.choice([0.5, 1.0], 2 * T),
        "dispersion": rng.uniform(0, 10, 2 * T),
        "goChoice": rng.integers(0, 2, 2 * T).astype(float),
        "preRZ_velocity": rng.normal(2.0, 1.0, 2 * T),
    }
    _write_v73(p, trial_tbl=tbl)
    assert data_io.list_animals(p) == ["Cb15", "Cb21"]
    tl = data_io.load_animal_trials("Cb15", export_path=p, min_trials=10)
    assert sum(t.n_trials for t in tl) == T and all(t.has_velocity for t in tl)


def test_load_stage1_from_v73_ioresults(tmp_path):
    pytest.importorskip("h5py")
    p = str(tmp_path / "io_v73.mat")
    _write_v73(p, ioresults={
        "fit_params": ["kappa_amp", "c_power", "d_power",
                       "vel_slope", "vel_intercept", "vel_std"],
        "group": [12.0, 1.2, 0.04, -2.0, 0.0, 0.5],
        "animals": [("Animal_1", {"kappa_amp": 20.0, "c_power": 1.10, "d_power": 0.05}),
                    ("Animal_2", {"kappa_amp": 30.0, "c_power": 1.30, "d_power": 0.07})],
    })
    # tags are Animal_1/2 (won't match Cb21) -> positional index resolves it
    s_idx = data_io.load_stage1("Cb21", animal_index=1, ioresults_path=p)
    assert (s_idx.kappa_amp, s_idx.c_power, s_idx.d_power, s_idx.kappa_min) == \
        pytest.approx((30.0, 1.30, 0.07, 1.0))
    # no animal -> group-level numeric vector via fit_params names
    s_grp = data_io.load_stage1(ioresults_path=p)
    assert (s_grp.kappa_amp, s_grp.c_power, s_grp.d_power) == \
        pytest.approx((12.0, 1.2, 0.04))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
