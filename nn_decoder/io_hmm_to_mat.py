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
from io_hmm_data import GRID_DEG_IO, load_io_hmm_pkl  # noqa: E402

DEFAULT_PKL = "data/fitted_data_and_posteriors.pkl"
DEFAULT_OUT = "data/fitted_data_and_posteriors.mat"


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


def convert(pkl_path=DEFAULT_PKL, out_path=DEFAULT_OUT, mice=None,
            allow_partial=True):
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
        data, dropped = {}, []
        for key, val in entry["data"].items():
            conv = _to_matlab(val)
            if conv is None:
                dropped.append(key)
            else:
                data[key] = conv

        ps = np.asarray(entry["data"]["PS_stim_G_tr"], dtype=np.float64)
        posteriors = ps.T / ps.T.sum(axis=1, keepdims=True)

        out[f"mouse{mouse_id}"] = {
            "params": {k: v for k, v in entry["params"].items()},
            "data": data,
            "posteriors": posteriors,
            "grid_deg": GRID_DEG_IO.reshape(1, -1),
        }
        print(f"mouse {mouse_id}: {posteriors.shape[0]} trials, "
              f"{len(data)} data fields kept, dropped (empty/None): "
              f"{', '.join(dropped) if dropped else 'none'}")

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
    args = p.parse_args(argv)
    convert(args.pkl, args.out, args.mice, allow_partial=not args.strict)


if __name__ == "__main__":
    _main()
