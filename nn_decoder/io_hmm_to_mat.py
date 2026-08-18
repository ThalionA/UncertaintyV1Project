"""Convert the collaborator's IO-HMM pickle to a .mat file for MATLAB.

MATLAB cannot read Python pickles, and this one additionally needs the
jax/datastructs unpickle stubs from :mod:`io_hmm_data`. So the route into
MATLAB is: unpickle here, write a ``.mat``, open that in MATLAB.

Layout of the output ``.mat`` — one struct per mouse::

    mouse0.params      struct of scalar fit parameters (pi, lam, kappa, ...)
    mouse0.data        struct of the raw per-trial fields, as pickled
    mouse0.posteriors  (n_trials x 72) double, rows sum to 1  [convenience]
    mouse0.grid_deg    (1 x 72) double, orientation bin centres in degrees

``data.PS_stim_G_tr`` keeps its shipped **bins-first** ``(72, n_trials)``
layout; ``posteriors`` is the transposed, row-normalised copy you actually
want to plot (see :mod:`io_hmm_data` for why). Ragged per-trial fields
(``times``, ``licks``, ``velocities``, ``positions``, ``displacements``)
become MATLAB **cell arrays**, one cell per trial. Fields that are ``None``
in the pickle are dropped and listed on stdout.

Usage::

    python nn_decoder/io_hmm_to_mat.py                     # all recoverable mice
    python nn_decoder/io_hmm_to_mat.py --mice 0 --out /tmp/m0.mat
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_hmm_data import (  # noqa: E402
    GRID_DEG_IO, load_io_hmm_pkl, scavenge_params,
)

DEFAULT_PKL = "data/fitted_data_and_posteriors.pkl"
DEFAULT_OUT = "data/fitted_data_and_posteriors.mat"

# Time-resolved posteriors are (n_trials, ~202, 72) per mouse — tens to
# hundreds of MB each, and the HMM refit adds a _by_state copy on top. scipy
# writes v5 .mat, which caps at 2 GB, so these are dropped unless asked for.
HEAVY_FIELDS = (
    "PS_x_G_tr",
    "PS_x_G_xtilde_ytilde_utilde",
    "PS_x_G_xtilde_ytilde_utilde_by_state",
    "PS_tr_G_resp_choice_by_state",   # (K, 72, 101, 2, n_trials) — 165 MB/mouse
)
MAT_V5_LIMIT_MB = 2000


def _to_matlab(value):
    """Coerce one pickled field into something savemat can write.

    Returns ``None`` for fields that should be dropped (None, or an empty
    container). Ragged sequences become object arrays -> MATLAB cells.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        shapes = {np.shape(v) for v in value}
        if len(shapes) == 1:
            return np.asarray(value)          # rectangular -> plain matrix
        cells = np.empty(len(value), dtype=object)
        for i, v in enumerate(value):
            cells[i] = np.asarray(v)
        return cells                          # ragged -> MATLAB cell array
    arr = np.asarray(value)
    if arr.dtype == object:
        return None
    return arr


MATLAB_FIELD_MAX = 31


def _sanitise_keys(d, label):
    """MATLAB struct field names cap at 31 characters; truncate and report.

    Truncation keeps the first 31 characters and uniquifies on collision, so
    the mapping is deterministic across runs. Renames are printed rather than
    applied silently — a field you cannot find by its pickle name is worse
    than a slightly ugly one.
    """
    out, renamed = {}, []
    for key, val in d.items():
        name = key
        if len(name) > MATLAB_FIELD_MAX:
            name = key[:MATLAB_FIELD_MAX]
            i = 1
            while name in out:
                suffix = str(i)
                name = key[:MATLAB_FIELD_MAX - len(suffix)] + suffix
                i += 1
            renamed.append(f"{key} -> {name}")
        out[name] = val
    if renamed:
        print(f"  {label}: field names over {MATLAB_FIELD_MAX} chars renamed: "
              f"{'; '.join(renamed)}")
    return out


def _clean_params(d):
    """savemat cannot write None; an unset Params field becomes an empty array."""
    return {k: (np.array([]) if v is None else v) for k, v in d.items()}


def convert(pkl_path=DEFAULT_PKL, out_path=DEFAULT_OUT, mice=None,
            allow_partial=True, include_heavy=False):
    loaded = load_io_hmm_pkl(pkl_path, allow_partial=allow_partial)
    wanted = sorted(loaded) if mice is None else [m for m in mice if m in loaded]
    missing = [] if mice is None else [m for m in mice if m not in loaded]
    if missing:
        raise RuntimeError(
            f"Mice {missing} are not in the pickle (available: {sorted(loaded)}). "
            f"A truncated download only yields the early mice."
        )

    out = {}
    for mouse_id in wanted:
        entry = loaded[mouse_id]
        data, dropped, skipped = {}, [], []
        for key, val in entry["data"].items():
            if key in HEAVY_FIELDS and not include_heavy:
                if val is not None:
                    skipped.append(f"{key}{np.shape(val)}")
                continue
            conv = _to_matlab(val)
            if conv is None:
                dropped.append(key)
            else:
                data[key] = conv

        struct = {
            "params": _sanitise_keys(_clean_params(entry["params"]), "params"),
            "data": _sanitise_keys(data, "data"),
        }

        # HMM fits carry one Params per latent state alongside the mouse-level
        # container; ship them as a 1 x n_states struct array.
        by_state = entry.get("params_by_state") or []
        if by_state:
            keys = sorted(by_state[0])
            struct["params_by_state"] = np.array(
                [tuple(_clean_params(b)[k] for k in keys) for b in by_state],
                dtype=[(k, object) for k in keys],
            )
            struct["n_states"] = len(by_state)

        # A truncated stream can yield a mouse with behaviour but no posterior;
        # write it anyway, minus the posteriors/grid_deg convenience fields.
        # HMM layout keeps posteriors on the entry; the old layout on Data.
        post_src = entry.get("posteriors") or {}
        if post_src:
            post, post_skipped = {}, []
            for key, val in post_src.items():
                if key in HEAVY_FIELDS and not include_heavy:
                    post_skipped.append(f"{key}{np.shape(val)}")
                    continue
                conv = _to_matlab(val)
                if conv is not None:
                    post[key] = conv
            struct["hmm"] = _sanitise_keys(post, "hmm")
            struct["n_states"] = entry.get("n_states", 0)
            print(f"  hmm posteriors: {len(post)} fields kept"
                  + (f"; skipped as heavy: {', '.join(post_skipped)}" if post_skipped else ""))

        raw_ps = post_src.get("PS_stim_G_tr")
        if raw_ps is None:
            raw_ps = entry["data"].get("PS_stim_G_tr")
        if raw_ps is None:
            n_trials = int(np.asarray(entry["data"]["orientation"]).size)
            print(f"mouse {mouse_id}: {n_trials} trials, NO POSTERIOR "
                  f"(PS_stim_G_tr is None) — behavioural fields only; "
                  f"{len(data)} kept")
        else:
            ps = np.asarray(raw_ps, dtype=np.float64)
            posteriors = ps.T / ps.T.sum(axis=1, keepdims=True)
            struct["posteriors"] = posteriors
            struct["grid_deg"] = GRID_DEG_IO.reshape(1, -1)
            print(f"mouse {mouse_id}: {posteriors.shape[0]} trials, "
                  f"{len(data)} data fields kept, dropped (empty/None): "
                  f"{', '.join(dropped) if dropped else 'none'}")

        out[f"mouse{mouse_id}"] = struct
        if skipped:
            print(f"  skipped as heavy (pass --heavy to include): "
                  f"{', '.join(skipped)}")

    est_mb = sum(
        v.nbytes / 1e6
        for mouse in out.values()
        for v in mouse["data"].values()
        if isinstance(v, np.ndarray) and v.dtype != object
    )
    if est_mb > MAT_V5_LIMIT_MB:
        print(f"\nWARNING: ~{est_mb:.0f} MB of array data exceeds the {MAT_V5_LIMIT_MB} MB "
              f"v5 .mat ceiling scipy writes to. Convert fewer mice (--mice) or "
              f"drop --heavy.")

    if out and all(not m["params"] for m in out.values()):
        blocks = scavenge_params(pkl_path)
        if blocks:
            keys = sorted({k for b in blocks for k in b})
            recovered = {}
            for k in keys:
                col = np.empty(len(blocks), dtype=object)
                for i, b in enumerate(blocks):
                    val = b.get(k)
                    col[i] = np.array([]) if val is None else np.asarray(val)
                recovered[k] = col
            out["recovered_params"] = recovered
            print(f"\nparams could not be attributed to mice; wrote "
                  f"{len(blocks)} unlabelled Params block(s) as "
                  f"'recovered_params' ({len(keys)} fields, one column per block)")

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path
    savemat(str(out_path), out, do_compression=True, oned_as="column")
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB) "
          f"with variables: {', '.join(sorted(out))}")
    return out_path


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pkl", default=DEFAULT_PKL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--mice", type=int, nargs="*", default=None,
                   help="Mouse ids to convert (default: all recoverable).")
    p.add_argument("--strict", action="store_true",
                   help="Fail on a truncated pickle instead of partial recovery.")
    p.add_argument("--heavy", action="store_true",
                   help=f"Also write the time-resolved posteriors ({', '.join(HEAVY_FIELDS)}); "
                        f"hundreds of MB per mouse.")
    args = p.parse_args(argv)
    convert(args.pkl, args.out, args.mice, allow_partial=not args.strict,
            include_heavy=args.heavy)


if __name__ == "__main__":
    _main()
