"""Loader for the collaborator's ideal-observer HMM fits (Slack, 2026-08-08).

``data/fitted_data_and_posteriors.pkl`` is a pickle of
``{mouse_int: {'params': datastructs.Params, 'data': datastructs.Data}}``
produced by the collaborator's jax-based ideal-observer HMM pipeline. This
module loads it with **no jax and no datastructs installed**: both modules are
stubbed at unpickle time so jax arrays come back as plain numpy and the
custom classes come back as attribute bags (their ``__dict__`` is the state).

Key facts, all of which have burned time already:

* **Support**: the per-trial stimulus posterior lives on 72 circular
  orientation bins of 2.5 deg spanning [0, 180) — ``GRID_DEG_IO``. Ori 0
  peaks across bins {71, 0, 1} (it wraps), 45 deg -> bin 18, 90 deg -> bin 36.
  This is NOT the 91-bin linear 0..90 grid used elsewhere in nn_decoder.
* **Bins-first layout**: ``PS_stim_G_tr`` ships as ``(72, n_trials)`` with
  COLUMNS summing to 1. It must be transposed to ``(n_trials, 72)`` before
  use; this module does so and renormalises rows to sum exactly 1 in float64
  (``make_target('perception')`` does not renormalise, so this is load-bearing).
* **Truncation failure mode**: the local copy was downloaded from Slack and is
  cut off at exactly 33 MiB, so a full ``pickle.load`` raises
  ``UnpicklingError('pickle data was truncated')``. ``load_io_hmm_pkl``
  converts that into a ``RuntimeError`` telling you to re-download from Slack;
  pass ``allow_partial=True`` to memo-recover whichever mice parsed fully
  (mouse 0 is recoverable from the truncated copy).
* **Trial alignment**: the neural export ``VR_Decoder_Data_Export.mat`` has
  fewer trials than the pkl (940 vs 945 for mouse 0) but preserves trial
  order. ``align_trials_to_export`` matches the export into the pkl by a
  greedy in-order pass over the (orientation, dispersion, contrast) barcode.
  Behavioural choice agrees ~98.7% for mouse 0 — the dozen mismatches are a
  known collaborator-side coding discrepancy on easy trials, logged in the
  alignment report rather than treated as an alignment failure.

Importing this module installs stub modules named ``datastructs``, ``jax``,
``jax._src`` and ``jax._src.array`` into ``sys.modules`` on first load (they
overwrite any pre-existing entries — this repo does not use jax).
"""

import pickle
import sys
import types
import warnings
from pathlib import Path

import numpy as np

# Circular orientation support of the IO HMM posterior: 72 bins x 2.5 deg,
# spanning [0, 180). New figures should take their degree axis from here, not
# from the 91-bin linear GRID_DEG in training/targets.py.
GRID_DEG_IO = np.arange(72) * 2.5

# The HMM refit (6 mice, 2026-08-18). Relative paths resolve against the repo root.
DEFAULT_PKL = 'data/fitted_data_and_posteriors_hmm.pkl'

# Alignment sanity floor: raise if export-vs-pkl choice agreement drops below
# this (mouse 0 verified at 0.987; ~0.99 is the expected regime).
MIN_CHOICE_AGREEMENT = 0.95

# Per-path cache so a 6-mouse loop does not re-read (and re-parse) the pkl
# six times. Keyed by (resolved path, allow_partial).
_PKL_CACHE = {}


# ---------------------------------------------------------------------------
# Stub unpickling machinery — lets the pickle load without jax or datastructs
# ---------------------------------------------------------------------------

class _StubBase:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


def _make_stub_module(name):
    mod = types.ModuleType(name)
    def __getattr__(attr):
        if attr.startswith("__"): raise AttributeError(attr)
        cls = type(attr, (_StubBase,), {"__module__": name})
        setattr(mod, attr, cls); return cls
    mod.__getattr__ = __getattr__
    sys.modules[name] = mod
    return mod


def _reconstruct_array(fun, args, arr_state, *rest):
    # Replacement for jax._src.array._reconstruct_array: build the underlying
    # numpy array and skip the device_put, so no jax is needed at load time.
    v = fun(*args); v.__setstate__(arr_state); return v


def _install_stubs():
    """Install the datastructs / jax stub modules into sys.modules."""
    _make_stub_module("datastructs")
    _make_stub_module("jax")
    _make_stub_module("jax._src")
    arr_mod = _make_stub_module("jax._src.array")
    arr_mod._reconstruct_array = _reconstruct_array


# ---------------------------------------------------------------------------
# Conversion of stub objects to plain dicts
# ---------------------------------------------------------------------------

def _params_to_dict(params_obj):
    """Convert a recovered Params stub to a plain dict of floats.

    Scalar (0-d) numpy values become Python floats; anything with shape is
    kept as a numpy array.
    """
    if params_obj is None:
        return {}
    out = {}
    for key, val in vars(params_obj).items():
        arr = np.asarray(val)
        if arr.ndim == 0:
            try:
                out[key] = float(arr)
            except (TypeError, ValueError):
                out[key] = val
        else:
            out[key] = arr
    return out


def _data_to_dict(data_obj):
    """Convert a recovered Data stub to a plain dict (values left as-is)."""
    return dict(vars(data_obj))


# Keys of a per-mouse entry that are not posteriors.
_ENTRY_NON_POSTERIOR = ("params", "data", "params_by_state")


def _entry_to_dict(entry):
    """Normalise one per-mouse entry, whichever layout the pickle uses.

    The original (non-HMM) fits pickle ``{'params': Params, 'data': Data}``
    with the posteriors living *inside* the Data object. The HMM refit moved
    them up: the entry is a 17-key dict carrying ``params``, ``data``,
    ``K`` (number of latent states), ``params_by_state``, and the posteriors
    themselves — both marginal (``PS_stim_G_tr``) and per-state
    (``PS_stim_G_tr_by_state``, ``gamma``, ``hard_state``, ...). In that
    layout ``Data.PS_stim_G_tr`` is None, so reading only ``data`` silently
    loses every posterior in the file.

    Returns ``{'params', 'params_by_state', 'data', 'posteriors', 'n_states'}``;
    ``posteriors`` is empty for the old layout, where ``data`` holds them.
    """
    by_state = [
        _params_to_dict(p) for p in np.atleast_1d(entry.get("params_by_state", []))
        if p is not None
    ]
    posteriors = {
        k: v for k, v in entry.items()
        if k not in _ENTRY_NON_POSTERIOR and v is not None
    }
    return {
        "params": _params_to_dict(entry.get("params")),
        "params_by_state": by_state,
        "data": _data_to_dict(entry["data"]),
        "posteriors": posteriors,
        "n_states": int(entry["K"]) if entry.get("K") is not None else len(by_state),
    }


def _resolve_pkl_path(pkl_path):
    """Resolve a pkl path; relative paths are taken from the repo root."""
    path = Path(pkl_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / path
    return path


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def diagnose_truncation(path):
    """Describe *why* a pickle looks truncated, for the load error message.

    Every truncated copy seen so far has been an incomplete download, and the
    tell is the size: it lands on an exact 0.5 MiB boundary (33.00, 218.50,
    279.50 MiB) instead of an arbitrary byte count, and the last byte is not
    the pickle STOP opcode ``.``.
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(-1, 2)
        last = f.read(1)
    mib = size / (1 << 20)
    lines = [f"  size: {size} bytes ({mib:.2f} MiB)"]
    if size % (1 << 19) == 0:
        lines.append(
            f"  -> lands on an exact {mib:.2f} MiB block boundary: this is a "
            f"cut-off download, not a corrupt file."
        )
    lines.append(
        f"  last byte: {last!r} "
        f"({'STOP - stream looks complete' if last == b'.' else 'not the pickle STOP opcode .'})"
    )
    return "\n".join(lines)


def load_io_hmm_pkl(pkl_path, allow_partial=False):
    """Load the IO HMM pickle into plain-python / numpy structures.

    Parameters
    ----------
    pkl_path : str or Path
        Path to ``fitted_data_and_posteriors.pkl``. Relative paths resolve
        against the repo root.
    allow_partial : bool
        If the file is truncated, a full load raises RuntimeError. With
        ``allow_partial=True`` the pure-python unpickler's memo is scavenged
        instead and whichever mice parsed fully are returned (a warning lists
        what is missing).

    Returns
    -------
    dict
        ``{mouse_id: {'params': dict, 'data': dict}}``. Params values are
        floats (scalars) or numpy arrays; Data values are the raw per-trial
        fields (numpy arrays, lists, scalars) exactly as pickled — in
        particular ``PS_stim_G_tr`` keeps its bins-first ``(72, n_trials)``
        layout here (``load_io_hmm_targets`` handles the transpose).
    """
    path = _resolve_pkl_path(pkl_path)
    cache_key = (str(path), bool(allow_partial))
    if cache_key in _PKL_CACHE:
        return _PKL_CACHE[cache_key]

    if not path.exists():
        raise FileNotFoundError(f"IO HMM pickle not found at: {path}")

    _install_stubs()

    with open(path, "rb") as f:
        try:
            raw = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError) as err:
            if not allow_partial:
                raise RuntimeError(
                    f"IO HMM pickle at {path} failed to load ({err}).\n"
                    f"{diagnose_truncation(path)}\n"
                    f"Re-download the full file, or pass allow_partial=True to "
                    f"recover whichever mice parsed fully."
                ) from err
            raw = None

    if raw is not None:
        result = {
            int(mouse_id): _entry_to_dict(entry)
            for mouse_id, entry in raw.items()
        }
    else:
        result = _recover_partial(path)

    _PKL_CACHE[cache_key] = result
    return result


def _group_params(params_objs):
    """Group a flat list of Params blocks into one set per mouse.

    The HMM refit pickles, per mouse, a **container** Params holding the
    stacked per-state values plus the HMM's ``trans_logits`` / ``init_logits``,
    immediately followed by one Params per latent state with those same fields
    unstacked. A container is therefore the block with a non-empty
    ``trans_logits``, and everything up to the next container belongs to it.
    The non-HMM fits have no ``trans_logits`` at all, so each block is its own
    (container, []) group and the behaviour matches the old 1:1 pairing.

    Returns a list of ``(container_dict, [per_state_dict, ...])``.
    """
    def _is_container(obj):
        # `or ()` would be ambiguous here: trans_logits is a numpy array.
        logits = getattr(obj, "trans_logits", None)
        return logits is not None and np.size(logits) > 0

    if not any(_is_container(o) for o in params_objs):
        return [(_params_to_dict(o), []) for o in params_objs]

    groups = []
    for obj in params_objs:
        if _is_container(obj):
            groups.append((_params_to_dict(obj), []))
        elif groups:
            groups[-1][1].append(_params_to_dict(obj))
    return groups


def _recover_partial(path):
    """Memo-scavenge a truncated pickle for fully parsed mice.

    The pure-python unpickler memoises every constructed object; when the
    stream runs out mid-file, the objects built so far are still in the memo.
    Params/Data objects appear there in stream order, i.e. mouse order, so
    pairing the i-th Params with the i-th Data recovers mouse i.
    """
    with open(path, "rb") as f:
        up = pickle._Unpickler(f)
        try:
            up.load()
        except Exception:
            pass

    memo_vals = list(up.memo.values())

    # A completed per-mouse entry dict is strictly better than a bare Data
    # object: in the HMM layout it is the only thing holding the posteriors.
    entries = [
        v for v in memo_vals
        if isinstance(v, dict) and "data" in v and "params" in v
        and type(v.get("data")).__name__ == "Data"
    ]
    if entries:
        result = {i: _entry_to_dict(e) for i, e in enumerate(entries)}

        # A mouse whose entry dict never completed can still have left a fully
        # built Data object behind: keep it as a posterior-less mouse rather
        # than dropping its behaviour on the floor.
        claimed = {id(e["data"]) for e in entries}
        orphans = [
            v for v in memo_vals
            if type(v).__name__ == "Data" and id(v) not in claimed
            and getattr(v, "orientation", None) is not None
        ]
        for j, data_obj in enumerate(orphans):
            result[len(entries) + j] = {
                "params": {},
                "params_by_state": [],
                "data": _data_to_dict(data_obj),
                "posteriors": {},
                "n_states": 0,
            }

        no_post = [i for i, r in result.items()
                   if not r["posteriors"] and r["data"].get("PS_stim_G_tr") is None]
        msg = (
            f"Partial recovery from truncated pickle {path.name}: recovered "
            f"{len(entries)} complete mouse entr(y/ies) plus {len(orphans)} "
            f"orphan Data object(s); any later mice are cut off mid-stream. "
            f"Re-download for the full set."
        )
        if no_post:
            msg += f" Mice {no_post} carry no posterior at all."
        warnings.warn(msg, stacklevel=3)
        return result

    params_objs = [v for v in memo_vals if type(v).__name__ == "Params"]
    data_objs = [v for v in memo_vals if type(v).__name__ == "Data"]

    if not data_objs:
        raise RuntimeError(
            f"Partial recovery of {path} reached no Data object at all — the "
            f"truncation is too early to salvage anything.\n"
            f"{diagnose_truncation(path)}"
        )

    params_groups = _group_params(params_objs)
    pairable = len(params_groups) == len(data_objs)

    result = {}
    no_posterior = []
    for i, data_obj in enumerate(data_objs):
        if getattr(data_obj, "PS_stim_G_tr", None) is None:
            no_posterior.append(i)
        container, by_state = params_groups[i] if pairable else ({}, [])
        result[i] = {
            "params": container,
            "params_by_state": by_state,
            "data": _data_to_dict(data_obj),
            "posteriors": {},
            "n_states": len(by_state),
        }

    msg = (
        f"Partial recovery from truncated pickle {path.name}: recovered "
        f"mice {sorted(result)}; any later mice are absent from the stream "
        f"entirely. Re-download the full file for the complete set."
    )
    if no_posterior:
        msg += (
            f" Mice {no_posterior} have NO posterior (PS_stim_G_tr is None) — "
            f"behavioural fields only."
        )
    if not pairable:
        msg += (
            f" Params left empty: the stream holds {len(params_objs)} Params "
            f"blocks grouping into {len(params_groups)} mouse-level set(s) for "
            f"{len(data_objs)} Data object(s), so the pairing is ambiguous "
            f"(use scavenge_params for the raw list)."
        )
    warnings.warn(msg, stacklevel=3)
    return result


def scavenge_params(pkl_path):
    """Return every ``Params`` block found in a (possibly truncated) pickle.

    Used when :func:`_recover_partial` cannot attribute Params to mice — the
    blocks are still worth reading, just not worth labelling.
    """
    path = _resolve_pkl_path(pkl_path)
    _install_stubs()
    with open(path, "rb") as f:
        up = pickle._Unpickler(f)
        try:
            up.load()
        except Exception:
            pass
    return [
        _params_to_dict(v) for v in up.memo.values()
        if type(v).__name__ == "Params"
    ]


def align_trials_to_export(pkl_data_dict, trials_dict, mouse_id=None):
    """Align the neural export's trials into the pkl's trial sequence.

    Both sides preserve trial order; the export is a subsequence of the pkl
    (a handful of pkl trials were dropped from the export). The match is a
    greedy in-order pass over the per-trial stimulus-condition barcode
    (orientation, dispersion, round(contrast, 3)).

    Parameters
    ----------
    pkl_data_dict : dict
        The ``'data'`` dict for one mouse from :func:`load_io_hmm_pkl`.
    trials_dict : dict
        The trials dict returned by ``utils.load_vr_export`` (keys
        'orientation', 'dispersion', 'contrast', 'choice').
    mouse_id : int, optional
        Used only to make the failure message identifiable.

    Returns
    -------
    idx : np.ndarray
        Length ``n_export`` int array; ``idx[i]`` is the pkl trial index of
        export trial ``i``.
    report : dict
        ``{'n_export', 'n_pkl', 'dropped_pkl_trials', 'choice_agreement'}``.
        Choice agreement below 1.0 is expected (~0.987 for mouse 0): a dozen
        easy trials carry a known collaborator-side choice-coding discrepancy.
    """
    def _barcode(ori, disp, con):
        ori = np.asarray(ori).ravel().astype(float)
        disp = np.asarray(disp).ravel().astype(float)
        con = np.round(np.asarray(con).ravel().astype(float), 3)
        return list(zip(ori, disp, con))

    export_codes = _barcode(
        trials_dict["orientation"], trials_dict["dispersion"], trials_dict["contrast"]
    )
    pkl_codes = _barcode(
        pkl_data_dict["orientation"], pkl_data_dict["dispersion"], pkl_data_dict["contrast"]
    )
    n_export, n_pkl = len(export_codes), len(pkl_codes)

    idx = np.empty(n_export, dtype=np.int64)
    j = 0
    for i, code in enumerate(export_codes):
        while j < n_pkl and pkl_codes[j] != code:
            j += 1
        if j == n_pkl:
            raise RuntimeError(
                f"Trial alignment failed for mouse {mouse_id}: export trial "
                f"{i} with barcode {code} has no in-order match in the pkl "
                f"(pkl exhausted at {n_pkl} trials). The export is expected "
                f"to be an order-preserving subsequence of the pkl."
            )
        idx[i] = j
        j += 1

    dropped = sorted(set(range(n_pkl)) - set(idx.tolist()))
    export_choice = np.asarray(trials_dict["choice"]).ravel().astype(float)
    pkl_choice = np.asarray(pkl_data_dict["choices"]).ravel().astype(float)
    choice_agreement = float(np.mean(export_choice == pkl_choice[idx]))

    report = {
        "n_export": n_export,
        "n_pkl": n_pkl,
        "dropped_pkl_trials": dropped,
        "choice_agreement": choice_agreement,
    }
    # The barcode matches by construction, so choice agreement is the only
    # runtime tell of a plausible-but-wrong alignment (or of a different choice
    # coding in a new pkl export). Mouse 0's verified value is 0.987.
    if choice_agreement < MIN_CHOICE_AGREEMENT:
        raise RuntimeError(
            f"Trial alignment for mouse {mouse_id} is suspect: choice "
            f"agreement {choice_agreement:.3f} < {MIN_CHOICE_AGREEMENT} "
            f"(expected ~0.99 with the known dozen-trial coding discrepancy). "
            f"Inspect the alignment before training on these targets."
        )
    return idx, report


def _get_posterior_field(entry, name):
    """Fetch a posterior array from either pkl layout.

    HMM layout keeps posteriors at entry level (``entry['posteriors'][name]``)
    and leaves ``Data.<name>`` as None; the older non-HMM layout keeps them
    inside ``Data``. Reading only ``entry['data']`` on the HMM file silently
    yields None for every posterior — that cost a round of wrong conclusions
    on 2026-08-18, so this helper is the single sanctioned access path.
    """
    v = (entry.get("posteriors") or {}).get(name)
    if v is None:
        v = entry["data"].get(name)
    return v


def load_io_hmm_targets(mouse_id, pkl_path, trials_dict, allow_partial=False,
                        state=None):
    """Build export-aligned decoder targets from the IO HMM pkl.

    Parameters
    ----------
    state : None | int | 'gamma'
        ``None`` — the marginal posterior ``PS_stim_G_tr`` (default; the only
        option for the non-HMM file). ``int z`` — the state-conditional posterior
        ``PS_stim_G_tr_by_state[z]`` for every trial (a "what if the animal
        were in state z" target; use with ``gamma`` weights or a hard-state
        trial mask downstream). ``'gamma'`` — same as ``None`` (the marginal IS
        sum_z gamma_z * P(s|z), verified exact), kept as an explicit alias so a
        config can say what it means.

    Returns
    -------
    dict
        ``'targets_perc'`` : (n_export_trials, 72) float64, rows sum to 1 —
        the per-trial posterior over stimulus orientation on ``GRID_DEG_IO``;
        ``'targets_dec'`` : (n_export_trials, 2) float64 [P(Go), 1 - P(Go)];
        ``'align_report'`` : dict from :func:`align_trials_to_export`;
        ``'grid_deg'`` : ``GRID_DEG_IO``;
        ``'gamma'`` : (n_export_trials, K) state posterior, or None (non-HMM);
        ``'hard_state'`` : (n_export_trials,) int, or None;
        ``'n_states'`` : K, or 0.
    """
    mice = load_io_hmm_pkl(pkl_path, allow_partial=allow_partial)
    if mouse_id not in mice:
        raise RuntimeError(
            f"Mouse {mouse_id} not present in IO HMM pickle "
            f"(available: {sorted(mice)}). With a truncated file only the "
            f"early mice are recoverable — re-download from Slack."
        )
    entry = mice[mouse_id]
    data = entry["data"]
    n_states = int(entry.get("n_states") or 0)

    if state is None or state == "gamma":
        ps_stim = _get_posterior_field(entry, "PS_stim_G_tr")
    else:
        by_state = _get_posterior_field(entry, "PS_stim_G_tr_by_state")
        if by_state is None:
            raise RuntimeError(
                f"Mouse {mouse_id}: state-conditional targets requested "
                f"(state={state!r}) but the pickle has no PS_stim_G_tr_by_state "
                f"— this is the non-HMM export.")
        z = int(state)
        if not 0 <= z < n_states:
            raise RuntimeError(
                f"Mouse {mouse_id}: state {z} out of range for K={n_states}.")
        ps_stim = np.asarray(by_state)[z]
    if ps_stim is None:
        raise RuntimeError(
            f"Mouse {mouse_id}: no PS_stim_G_tr in either layout of the pickle.")

    # Bins-first layout: (72, n_trials), columns sum to 1 — transpose, then
    # renormalise rows in float64 (downstream make_target does not renormalise).
    ps_stim = np.asarray(ps_stim, dtype=np.float64)
    if ps_stim.shape[0] != len(GRID_DEG_IO):
        raise RuntimeError(
            f"Mouse {mouse_id}: PS_stim_G_tr has shape {ps_stim.shape}, "
            f"expected bins-first ({len(GRID_DEG_IO)}, n_trials)."
        )
    posteriors = ps_stim.T
    row_sums = posteriors.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise RuntimeError(
            f"Mouse {mouse_id}: transposed PS_stim_G_tr rows do not sum to ~1 "
            f"(max deviation {np.abs(row_sums - 1.0).max():.3g}) — layout or "
            f"content is not as expected."
        )
    posteriors = posteriors / row_sums

    idx, report = align_trials_to_export(data, trials_dict, mouse_id=mouse_id)

    # SEMANTICS (verified 2026-08-18): PS_Go_G_tr is the ideal observer's BELIEF
    # that the stimulus is in the Go category — r = 0.9999 with the Go-half mass
    # of PS_stim_G_tr — NOT the model's choice probability. PS_choice_G_tr is
    # the model psych curve (tracks the empirical Go rate; e.g. mouse 5 at 90 deg:
    # empirical 0.56, PS_choice 0.57, PS_Go 0.15). The 'd' (decision) target in
    # this repo means P(Go choice), so it is built from PS_choice_G_tr; the belief
    # is returned separately as 'p_go_belief' for analyses that want it.
    p_choice = np.asarray(_get_posterior_field(entry, "PS_choice_G_tr"), dtype=np.float64).ravel()
    p_go_belief = np.asarray(_get_posterior_field(entry, "PS_Go_G_tr"), dtype=np.float64).ravel()
    targets_dec = np.column_stack([p_choice[idx], 1.0 - p_choice[idx]])

    gamma = _get_posterior_field(entry, "gamma")
    hard = _get_posterior_field(entry, "hard_state")
    return {
        "targets_perc": posteriors[idx],
        "targets_dec": targets_dec,
        "align_report": report,
        "grid_deg": GRID_DEG_IO,
        "p_go_belief": p_go_belief[idx],
        "gamma": None if gamma is None else np.asarray(gamma, dtype=np.float64)[idx],
        "hard_state": None if hard is None else np.asarray(hard).astype(int)[idx],
        "n_states": n_states,
        "state": state,
    }


# ---------------------------------------------------------------------------
# CLI: summary report per mouse (no data dumps)
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarise the IO HMM pickle and its alignment to the neural export."
    )
    parser.add_argument(
        "--pkl", default="data/fitted_data_and_posteriors.pkl",
        help="Path to the pickle (relative paths resolve against the repo root).",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Memo-recover from a truncated file instead of failing.",
    )
    args = parser.parse_args(argv)

    # torch/scipy are fine here (CLI only) — the loader itself stays numpy-pure.
    try:
        from utils import load_vr_export
    except ImportError:
        from nn_decoder.utils import load_vr_export

    mice = load_io_hmm_pkl(args.pkl, allow_partial=args.allow_partial)
    print(f"Loaded {len(mice)} mouse/mice from {args.pkl}: {sorted(mice)}")

    for mouse_id in sorted(mice):
        data = mice[mouse_id]["data"]
        ps_raw = np.asarray(data["PS_stim_G_tr"], dtype=np.float64)
        n_trials = ps_raw.shape[1]
        raw_row_dev = np.abs(ps_raw.T.sum(axis=1) - 1.0).max()

        print(f"\n--- Mouse {mouse_id} ---")
        print(f"n_trials (pkl): {n_trials}")
        print(f"PS_stim_G_tr shape (bins-first, as shipped): {ps_raw.shape}")
        print(f"row-sum check after transpose: max |sum - 1| = {raw_row_dev:.3g}")

        # Posterior sanity: mean posterior per orientation should peak at the
        # matching circular bin (deg / 2.5, mod 72).
        oris = np.asarray(data["orientation"]).ravel()
        print("mean-posterior argmax bin per orientation:")
        for ori in np.unique(oris):
            mean_post = ps_raw[:, oris == ori].mean(axis=1)
            expected = int(round(float(ori) / 2.5)) % len(GRID_DEG_IO)
            print(f"  ori {int(ori):3d} deg -> bin {int(mean_post.argmax()):2d} "
                  f"(expected {expected:2d}, n={int((oris == ori).sum())})")

        # End-to-end: aligned targets against the neural export.
        try:
            _, _, _, _, trials_dict = load_vr_export(mouse_id)
        except Exception as err:
            print(f"export alignment skipped (load_vr_export failed: {err})")
            continue
        targets = load_io_hmm_targets(
            mouse_id, args.pkl, trials_dict, allow_partial=args.allow_partial
        )
        rep = targets["align_report"]
        perc = targets["targets_perc"]
        print(f"alignment vs export: {rep['n_export']} export trials matched into "
              f"{rep['n_pkl']} pkl trials")
        print(f"  dropped pkl trials: {rep['dropped_pkl_trials']}")
        print(f"  choice agreement: {rep['choice_agreement']:.4f}")
        print(f"  targets_perc {perc.shape}, rows sum to 1 exactly: "
              f"{bool(np.allclose(perc.sum(axis=1), 1.0, atol=1e-12))}; "
              f"targets_dec {targets['targets_dec'].shape}")


if __name__ == "__main__":
    _main()


# --------------------------------------------------------------- engaged state
ENGAGED_MIN_N = 15


def engaged_state(mouse_id, pkl_path=DEFAULT_PKL, min_n=ENGAGED_MIN_N,
                  allow_partial=False, rule='dprime'):
    """The mouse's ENGAGED latent state, by signal-detection sensitivity (d').

    d' = z(hit rate) - z(false-alarm rate) over the state's hard-assigned
    (argmax-gamma) trials, hit = Go response to a Go stimulus (ori < 45 deg),
    false alarm = Go response to a NoGo stimulus. Rates are loglinear-corrected
    ((k + 0.5) / (n + 1)) so a state with a perfect 0 or 1 rate gives a finite d'
    instead of an infinity.

    WHY d' AND NOT THE FALSE-ALARM RATE (Theo, 2026-08-28): a bare false-alarm
    rate confounds sensitivity with criterion — a state in which the animal simply
    stops licking has FA = 0 and would be crowned 'engaged' for being disengaged.
    d' separates the two, and is the standard measure for exactly this question.
    ``rule='false_alarms'`` keeps the simpler criterion for comparison.

    HMM state indices are arbitrary per fit, so any across-animal grouping needs a
    rule that names states by what they DO. This is the simplest rule that works:
    one measured behavioural quantity, no feature selection, no z-scoring, no
    linkage method and no cut level to choose.

    False alarm = a Go response to a NoGo stimulus (orientation > 45 deg), over the
    state's hard-assigned (argmax-gamma) trials. States with fewer than ``min_n``
    such trials are excluded before the comparison — mouse 5's 4-trial state is
    exactly the noise that gate exists for.

    HOW THE CRITERIA DISAGREE (measured 2026-08-28, all six mice; the earlier
    claim in this docstring that they converge 6/6 was WRONG and is corrected here):

        criterion                    m0  m1  m2  m3  m4  m5   vs d'
        d' (this default)            s0  s1  s0  s2  s0  s0    --
        lowest false-alarm rate      s0  s1  s1  s2  s1  s0   4/6
        12-feature clustering        s0  s1  s1  s2  s1  s0   4/6
        highest mean running speed   s0  s1  s1  s0  s1  s0   3/6

    The disagreements are not noise, they are two different meanings of "engaged".
    m2's s1 has FA 0.13 but a hit rate of only 0.39 — the animal has stopped
    responding, which a bare false-alarm rate rewards and d' does not. m3's s0 is
    the fastest-running state but lasts 1.3 trials on average (n=60), so speed
    picks a transient. TASK PERFORMANCE (d') and AROUSAL (speed) agree in only
    3/6 animals here; choosing between them is a scientific decision, not a
    detail. Anything resting on the engaged label must state which rule it used,
    and the engaged-vs-other CONTRAST is not robust to the choice: the deck's
    strongest contrast (h8_flat_lh3e-3, t=+4.74 p=0.005 under the clustering map)
    becomes t=+0.47 p=0.657 under d'.

    Returns
    -------
    state : int
        The engaged state index for this mouse.
    info : dict
        ``per_state`` {state: {'n', 'false_alarms'}} and ``margin`` — the runner-up's
        false-alarm rate minus the winner's. A SMALL margin means the animal's
        states are behaviourally alike and the label is arbitrary: mouse 2 sits at
        0.24 vs 0.13 here, and is a coin flip under the discriminability rule
        (0.251 vs 0.239). Report the margin wherever the label carries weight.
    """
    entry = load_io_hmm_pkl(pkl_path, allow_partial=allow_partial)[mouse_id]
    gamma = np.asarray(_get_posterior_field(entry, 'gamma'), dtype=np.float64)
    hard = gamma.argmax(axis=1)
    data = entry['data']
    ori = np.asarray(data['orientation'], dtype=np.float64).ravel()
    choice = np.asarray(data['choices'], dtype=np.float64).ravel()
    nogo = ori > 45.0                      # Go/NoGo boundary at 45 deg

    from scipy.stats import norm

    def _rate(hits, n):
        return (hits + 0.5) / (n + 1.0)          # loglinear correction

    per_state = {}
    for z in range(gamma.shape[1]):
        sel = hard == z
        n = int(sel.sum())
        if n < min_n:
            continue
        g, ng = sel & ~nogo, sel & nogo
        hit = _rate(float(choice[g].sum()), int(g.sum()))
        fa = _rate(float(choice[ng].sum()), int(ng.sum()))
        per_state[z] = {'n': n, 'n_go': int(g.sum()), 'n_nogo': int(ng.sum()),
                        'hit_rate': hit, 'false_alarms': fa,
                        'dprime': float(norm.ppf(hit) - norm.ppf(fa))}
    if not per_state:
        raise RuntimeError(f'mouse {mouse_id}: no state has >= {min_n} hard trials')

    key = {'dprime': lambda z: -per_state[z]['dprime'],
           'false_alarms': lambda z: per_state[z]['false_alarms']}[rule]
    order = sorted(per_state, key=key)
    margin = (abs(key(order[1]) - key(order[0])) if len(order) > 1 else float('nan'))
    return order[0], {'per_state': per_state, 'margin': margin, 'rule': rule}
