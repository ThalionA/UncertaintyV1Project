# -*- coding: utf-8 -*-
"""Load real VR mouse behaviour into IO-HMM ``Trials`` + frozen ``Stage1Params``.

Primary source is the unified MATLAB export ``data/VR_Decoder_Data_Export.mat``
produced by ``preprocessing/VR_multi_animal_analysis.m``. It stores
``TrialTbl_Struct`` -- a struct-of-arrays (one row per trial, pooled over all
animals/sessions) with the columns we need:

    TrialTbl_Struct column   ->  Trials field   meaning
    ----------------------       ------------    -------------------------------
    abs_from_go              ->  s_deg           |angle from the go pole| (deg, already go-aligned, 0..90)
    contrast                ->  c               stimulus contrast (0..1)
    dispersion              ->  d               stimulus dispersion (the ``d`` in kappa = ... * exp(-d_power*d))
    goChoice                ->  choice          1 = go / lick (Hit or FA), 0 = nogo
    preRZ_velocity          ->  velocity        mean pre-reward-zone running speed
    animal                                       animal tag (e.g. 'Cb15')
    session                                      per-animal session index

Note ``d`` is **dispersion**, not duration -- it matches ``get_kappa_for_trial``
in ``ideal_observer_hierarchical_fitting_v2.m`` and ``io_core.kappa_for_trial``.

Frozen Stage-1 sensory params (kappa_amp, c_power, d_power, kappa_min) are read
from the MATLAB Stage-1 fit ``data/IOResults.mat`` (``ideal_observer_
hierarchical_fitting_v2.m`` output): per-animal under
``IOResults.animals{i}.fit.full_params`` or group-level under
``IOResults.group.params``; ``kappa_min`` from ``IOResults.meta.fixed_params``.

The unified export is gitignored (large + raw data), so nothing here is exercised
against it in CI; ``tests/test_io_hmm_data_io.py`` drives the same code paths on
synthetic ``.mat`` fixtures written with ``scipy.io.savemat``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import scipy.io as sio

import emissions as emissions_mod

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_EXPORT = os.path.join(_REPO_ROOT, "data", "VR_Decoder_Data_Export.mat")
DEFAULT_IORESULTS = os.path.join(_REPO_ROOT, "data", "IOResults.mat")

# Trials field  ->  TrialTbl_Struct column
_TRIAL_COLS = {
    "s_deg": "abs_from_go",
    "c": "contrast",
    "d": "dispersion",
    "choice": "goChoice",
    "velocity": "preRZ_velocity",
}
_SENSORY_KEYS = ("kappa_amp", "c_power", "d_power")


# ---------------------------------------------------------------------------
# Trial-table loading
# ---------------------------------------------------------------------------


def _load_mat_struct(path: str, key: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected MATLAB file not found: {path}. The real data is "
            f"gitignored; point the loader at your local copy.")
    try:
        mat = sio.loadmat(path, simplify_cells=True)
    except (NotImplementedError, ValueError):
        # MATLAB v7.3 saves as HDF5, which loadmat rejects (NotImplementedError,
        # or ValueError "Unknown mat file type" for a bare .h5). Fall back to
        # h5py when the file really is HDF5; otherwise re-raise the original.
        if not _is_hdf5(path):
            raise
        mat = _loadmat_v73(path)
    if key not in mat:
        raise KeyError(f"'{key}' not found in {path} (keys: "
                       f"{[k for k in mat if not k.startswith('__')]})")
    return mat[key]


def _is_hdf5(path: str) -> bool:
    """True if the file carries the HDF5 signature (incl. at a userblock).

    MATLAB v7.3 files prepend a text header then the HDF5 block at a power-of-two
    offset, so check the signature at the start and at 512/1024/... byte offsets.
    """
    sig = b"\x89HDF\r\n\x1a\n"
    try:
        with open(path, "rb") as fh:
            blob = fh.read(2048)
    except OSError:
        return False
    if blob[:8] == sig:
        return True
    return any(blob[off:off + 8] == sig for off in (512, 1024))


def _loadmat_v73(path: str) -> dict:
    """Read a MATLAB v7.3 (HDF5) file into a nested dict ~ ``simplify_cells``.

    Handles the constructs this loader needs: structs -> dict, char arrays ->
    str, numeric arrays -> squeezed ndarray/scalar, and cell / struct arrays
    (HDF5 object references) -> list. Requires ``h5py`` (optional dependency;
    only imported on the v7.3 path).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NotImplementedError(
            f"{path} is a MATLAB v7.3 file; install h5py to read it "
            f"(`pip install h5py`), or re-save it as -v7 in MATLAB.") from exc
    with h5py.File(path, "r") as fh:
        return {k: _h5_node(fh[k], fh) for k in fh.keys() if k != "#refs#"}


def _h5_node(node, fh):
    import h5py
    if isinstance(node, h5py.Group):
        return {k: _h5_node(node[k], fh) for k in node.keys() if k != "#refs#"}
    # Dataset
    mcls = node.attrs.get("MATLAB_class", b"")
    if isinstance(mcls, bytes):
        mcls = mcls.decode("ascii", "ignore")
    if h5py.check_dtype(ref=node.dtype) is not None:
        refs = np.asarray(node[()]).ravel()
        items = [_h5_node(fh[r], fh) for r in refs if r]
        if len(items) == 0:
            return []
        return items[0] if len(items) == 1 else items
    data = node[()]
    if mcls == "char":
        codes = np.asarray(data).ravel().astype(int)
        return "".join(chr(c) for c in codes if c != 0)
    arr = np.asarray(data).squeeze()
    return arr.item() if arr.ndim == 0 else arr



def load_trial_table(export_path: Optional[str] = None) -> dict:
    """Return the raw ``TrialTbl_Struct`` dict (struct-of-arrays)."""
    return _load_mat_struct(export_path or DEFAULT_EXPORT, "TrialTbl_Struct")


def list_animals(export_path: Optional[str] = None) -> list[str]:
    """Animal tags present in the export, in first-appearance order."""
    tbl = load_trial_table(export_path)
    if "animal" not in tbl:
        return []
    animals = np.asarray(tbl["animal"]).ravel()
    seen, out = set(), []
    for a in animals:
        a = str(a)
        if a not in seen:
            seen.add(a); out.append(a)
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    """NaN-safe z-score; mean-fills NaNs first and returns zeros if sd==0.

    Mirrors ``zscore_safe`` + the NaN mean-fill in the MATLAB Stage-1 loader.
    """
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x) if np.any(~np.isnan(x)) else 0.0
    x = np.where(np.isnan(x), mu, x)
    sd = np.std(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def load_animal_trials(animal: str, *,
                       export_path: Optional[str] = None,
                       by_session: bool = True,
                       standardize_velocity: bool = True,
                       clamp_contrast: bool = True,
                       with_velocity: bool = True,
                       min_trials: int = 20) -> list[emissions_mod.Trials]:
    """Per-session ``Trials`` for one animal from the unified export.

    Parameters
    ----------
    by_session :
        Split into one ``Trials`` per ``session`` (recommended -- the HMM's
        initial-state prior ``pi`` applies per sequence, so sessions are the
        natural sequence unit). If False (or no ``session`` column), the
        animal's trials are returned as a single sequence.
    standardize_velocity :
        Robust z-score velocity across the animal's pooled trials (so the
        Gaussian channel sees comparable scale across sessions), matching the
        MATLAB ``conf_vel`` stream. Applied before the session split.
    clamp_contrast :
        Round contrast to 2 dp and snap >0.9 to 1.0, matching the v2 condition
        binning (keeps the unique-condition cache small).
    with_velocity :
        Attach the velocity channel. If False, returned ``Trials`` are
        choice-only.
    min_trials :
        Drop sequences shorter than this (too short to inform the HMM).

    NaN trials (missing stimulus/contrast/dispersion/choice) are dropped;
    missing velocities are mean-filled (inside ``_robust_z``).
    """
    tbl = load_trial_table(export_path)
    n = len(np.asarray(tbl[_TRIAL_COLS["choice"]]).ravel())
    if "animal" in tbl:
        mask = np.asarray(tbl["animal"]).ravel().astype(str) == str(animal)
    else:
        mask = np.ones(n, dtype=bool)
    if not mask.any():
        raise ValueError(f"No trials for animal {animal!r}. "
                         f"Available: {list_animals(export_path)}")

    cols = {f: np.asarray(tbl[col]).ravel()[mask].astype(float)
            for f, col in _TRIAL_COLS.items()}
    if clamp_contrast:
        c = np.round(cols["c"], 2)
        c[c > 0.9] = 1.0
        cols["c"] = c
    if standardize_velocity:
        cols["velocity"] = _robust_z(cols["velocity"])
    else:
        v = cols["velocity"]
        cols["velocity"] = np.where(np.isnan(v), np.nanmean(v), v)

    # drop trials missing any required (non-velocity) field
    valid = np.ones(mask.sum(), dtype=bool)
    for f in ("s_deg", "c", "d", "choice"):
        valid &= np.isfinite(cols[f])
    for f in cols:
        cols[f] = cols[f][valid]

    # session index aligned to the same mask + validity
    if by_session and "session" in tbl:
        sess = np.asarray(tbl["session"]).ravel()[mask][valid]
        groups = [(s, np.where(sess == s)[0]) for s in _ordered_unique(sess)]
    else:
        groups = [(0, np.arange(len(cols["choice"])))]

    out: list[emissions_mod.Trials] = []
    for _, idx in groups:
        if len(idx) < min_trials:
            continue
        out.append(emissions_mod.Trials(
            s_deg=cols["s_deg"][idx], c=cols["c"][idx], d=cols["d"][idx],
            choice=cols["choice"][idx],
            velocity=cols["velocity"][idx] if with_velocity else None,
        ))
    if not out:
        raise ValueError(
            f"Animal {animal!r}: no session had >= {min_trials} valid trials.")
    return out


def _ordered_unique(x: np.ndarray) -> list:
    seen, out = set(), []
    for v in x:
        key = v.item() if hasattr(v, "item") else v
        if key not in seen:
            seen.add(key); out.append(key)
    return out


def load_all_trials(export_path: Optional[str] = None, **kwargs
                    ) -> dict[str, list[emissions_mod.Trials]]:
    """``{animal_tag: [Trials, ...]}`` for every animal in the export."""
    return {a: load_animal_trials(a, export_path=export_path, **kwargs)
            for a in list_animals(export_path)}


# ---------------------------------------------------------------------------
# Stage-1 sensory params from the MATLAB IOResults fit
# ---------------------------------------------------------------------------


def _as_param_dict(obj) -> Optional[dict]:
    """Return ``obj`` if it is a mapping holding all sensory keys, else None."""
    if isinstance(obj, dict) and all(k in obj for k in _SENSORY_KEYS):
        try:
            return {k: float(obj[k]) for k in _SENSORY_KEYS}
        except (TypeError, ValueError):
            return None
    return None


def _find_param_dict(obj) -> Optional[dict]:
    """Depth-first search for a mapping containing the three sensory keys."""
    direct = _as_param_dict(obj)
    if direct is not None:
        return direct
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_param_dict(v)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for v in obj:
            found = _find_param_dict(v)
            if found is not None:
                return found
    return None


def _animal_list(results: dict) -> list:
    animals = results.get("animals")
    if animals is None:
        return []
    return [animals] if isinstance(animals, dict) else list(animals)


def _animal_subtree(results: dict, animal: str,
                    animal_index: Optional[int] = None) -> Optional[dict]:
    """Locate the per-animal IOResults entry for ``animal``.

    Matches by tag string first; if that fails (the v2 fit may tag animals
    positionally as ``Animal_<n>`` rather than ``Cb15``), falls back to
    ``animal_index`` into the animals list.
    """
    animals = _animal_list(results)
    for entry in animals:
        if isinstance(entry, dict):
            for v in entry.values():
                if isinstance(v, str) and v == str(animal):
                    return entry
    if animal_index is not None and 0 <= animal_index < len(animals):
        entry = animals[animal_index]
        return entry if isinstance(entry, dict) else None
    return None


def _fit_param_names(results: dict) -> Optional[list]:
    """``meta.model_spec.fit_params`` (ordered names of the fitted vector)."""
    spec = results.get("meta", {})
    names = _find_scalar_list(spec, "fit_params")
    return names


def _find_scalar_list(obj, name):
    """Depth-first search for a list/array stored under ``name``."""
    if isinstance(obj, dict):
        if name in obj:
            val = obj[name]
            if isinstance(val, str):
                return [val]
            if isinstance(val, (list, tuple, np.ndarray)):
                return [str(x) for x in np.asarray(val).ravel()]
        for v in obj.values():
            r = _find_scalar_list(v, name)
            if r is not None:
                return r
    return None


def _params_from_vector(results: dict, vec) -> Optional[dict]:
    """Map a fitted-param *vector* to sensory keys via ``fit_params`` names.

    The v2 ``group.params`` / ``fit_params_vec`` are numeric vectors ordered by
    ``model_spec.fit_params`` (sensory params first), so this recovers the
    sensory triple positionally when no named struct is available.
    """
    names = _fit_param_names(results)
    vec = np.asarray(vec, dtype=float).ravel()
    if not names or len(vec) < len(_SENSORY_KEYS):
        return None
    mapping = {n: float(vec[i]) for i, n in enumerate(names) if i < len(vec)}
    if all(k in mapping for k in _SENSORY_KEYS):
        return {k: mapping[k] for k in _SENSORY_KEYS}
    return None


def _find_scalar(obj, name: str) -> Optional[float]:
    if isinstance(obj, dict):
        if name in obj:
            try:
                return float(np.asarray(obj[name]).ravel()[0])
            except (TypeError, ValueError, IndexError):
                pass
        for v in obj.values():
            r = _find_scalar(v, name)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for v in obj:
            r = _find_scalar(v, name)
            if r is not None:
                return r
    return None


def load_stage1(animal: Optional[str] = None, *,
                animal_index: Optional[int] = None,
                ioresults_path: Optional[str] = None,
                kappa_min_default: float = 1.0) -> emissions_mod.Stage1Params:
    """Frozen ``Stage1Params`` from the MATLAB Stage-1 fit (IOResults.mat).

    Resolution order:
      1. per-animal named fit ``animals{i}.fit.full_params`` (matched by tag,
         else by ``animal_index`` -- the v2 fit may tag animals positionally as
         ``Animal_<n>`` rather than ``Cb15``);
      2. per-animal fitted *vector* ``animals{i}.fit.params_vec`` mapped via
         ``meta.model_spec.fit_params`` names;
      3. group-level fit ``group.params`` (named struct or numeric vector);
      4. any subtree carrying the named sensory keys.
    ``kappa_min`` comes from ``IOResults.meta.fixed_params`` (default 1.0).
    """
    res = _load_mat_struct(ioresults_path or DEFAULT_IORESULTS, "IOResults")

    params = None
    if animal is not None or animal_index is not None:
        sub = _animal_subtree(res, animal, animal_index)
        if sub is not None:
            params = _find_param_dict(sub)              # named full_params
            if params is None and isinstance(sub.get("fit"), dict):
                params = _params_from_vector(res, sub["fit"].get("params_vec"))
    if params is None:
        group = res.get("group")
        if isinstance(group, dict):
            params = _find_param_dict(group) or _params_from_vector(
                res, group.get("params"))
        elif isinstance(group, (list, np.ndarray)):
            params = _params_from_vector(res, group)
    if params is None:
        params = _find_param_dict(res)
    if params is None:
        raise ValueError(
            f"Could not find {_SENSORY_KEYS} in IOResults at "
            f"{ioresults_path or DEFAULT_IORESULTS}.")

    kappa_min = _find_scalar(res.get("meta", {}), "kappa_min")
    if kappa_min is None:
        kappa_min = kappa_min_default
    return emissions_mod.Stage1Params(
        kappa_amp=params["kappa_amp"], c_power=params["c_power"],
        d_power=params["d_power"], kappa_min=float(kappa_min))


__all__ = [
    "DEFAULT_EXPORT", "DEFAULT_IORESULTS",
    "load_trial_table", "list_animals",
    "load_animal_trials", "load_all_trials", "load_stage1",
]
