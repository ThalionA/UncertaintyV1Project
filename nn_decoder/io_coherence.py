# -*- coding: utf-8 -*-
"""IO–decoder coherence: take each architecture's decoded Q(theta) and push
it through the IO's saved Stage 2 choice psychometric and Stage 1 velocity
emission to predict the animal's actual behaviour.

If the V1 representation truly encodes the IO's perceptual posterior, then
the decoded Q-hat run through the IO's behavioural mapping should predict
choice and velocity at IO-comparable accuracy.

Closed-form pieces
------------------

Log posterior odds:
    g(Q) = log [(P(theta < 45) + 0.5 * Q(theta = 45))
                / (P(theta > 45) + 0.5 * Q(theta = 45))]

Stage 2 choice psychometric with lapse:
    P_hat(Go) = gamma_r + (1 - gamma_r - delta_r) * sigmoid(alpha_r * g + beta_r)

Stage 1 velocity emission (IO operates on z-scored kinematics; predictions
are in z-scored space):
    EU(Go|Q)   = sum_s Q(s) * U(Go,  s)
    EU(NoGo|Q) = sum_s Q(s) * U(NoGo, s)
    DV(Q)      = EU(Go|Q) - EU(NoGo|Q)
    vel_hat    = beta_vel * DV(Q) + alpha_vel

For the 2-bin decision target [P(Go), P(NoGo)] the decoder already produces
the integral, so g and the lapse psychometric apply directly without
boundary integration.

Trial alignment with the decoder
--------------------------------

The neural-side trial table (TrialTbl_Struct in VR_Decoder_Data_Export.mat)
is built from IOResults by ``stack_IO_trial_tables`` in
``preprocessing/refresh_IO_in_export.m``, which **drops the last trial of
each session** for every animal. Any per-trial array taken from
IOResults must therefore be filtered with the same trials-to-keep mask
before being aligned with the decoder's ``full_decoded`` predictions.

This module replicates that mask in ``get_trials_to_keep_mask`` and runs a
hard sanity assert in ``assert_choices_aligned`` to catch silent
misalignments.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.io as sio

import paths

try:
    import h5py
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False


# ----------------------------------------------------------------------
# Closed-form per-trial transforms
# ----------------------------------------------------------------------

DEFAULT_UTILITY = dict(R_hit=1.0, R_miss=0.0, R_cr=0.1, R_fa=-0.2)
BOUNDARY_DEG = 45
DEFAULT_GRID_91 = np.arange(0, 91, 1, dtype=float)
LOG_ODDS_CLIP = 50.0


def _boundary_pgo(Q, grid):
    """Boundary integral of Q to give P(Go), accounting for the ambiguous
    boundary bin theta=45 which is split 50/50 between Go and No-Go in the
    IO methods."""
    Q = np.asarray(Q, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if Q.shape[1] != grid.shape[0]:
        raise ValueError(f"Q has {Q.shape[1]} bins; grid has {grid.shape[0]}.")
    is_go = grid < BOUNDARY_DEG
    is_boundary = grid == BOUNDARY_DEG
    pgo = Q[:, is_go].sum(axis=1) + 0.5 * Q[:, is_boundary].sum(axis=1)
    pnogo = Q[:, ~is_go & ~is_boundary].sum(axis=1) + 0.5 * Q[:, is_boundary].sum(axis=1)
    # Renormalise for tiny numerical drift if Q is unnormalised.
    s = pgo + pnogo
    s = np.where(s > 0, s, 1.0)
    return pgo / s, pnogo / s


def compute_log_posterior_odds(Q, grid):
    """g(Q) = log [P(Go|Q) / P(NoGo|Q)] per trial, clipped to a finite range.

    The clip prevents Q with literal zero on one side from producing -inf
    or +inf, which would NaN the downstream sigmoid via overflow.
    """
    pgo, pnogo = _boundary_pgo(Q, grid)
    # log(pgo) - log(pnogo) avoids overflow in either branch
    log_pgo = np.log(np.clip(pgo, 1e-30, 1.0))
    log_pnogo = np.log(np.clip(pnogo, 1e-30, 1.0))
    g = log_pgo - log_pnogo
    return np.clip(g, -LOG_ODDS_CLIP, LOG_ODDS_CLIP)


def predict_choice_lapse(g, alpha_r, beta_r, gamma_r, delta_r):
    """Apply the IO's Stage 2 four-parameter psychometric to log-odds g."""
    g = np.asarray(g, dtype=float)
    z = alpha_r * g + beta_r
    sig = 1.0 / (1.0 + np.exp(-z))
    return gamma_r + (1.0 - gamma_r - delta_r) * sig


def predict_choice_naive(Q, grid):
    """P(Go) without any lapse correction: just the boundary integral."""
    pgo, _ = _boundary_pgo(Q, grid)
    return pgo


def _utility_vectors(grid, utility):
    """Per-bin utility for Go and No-Go actions on the orientation grid,
    matching the MATLAB ``get_utility_vectors`` exactly: boundary bin
    receives 0.5*(R_hit + R_fa) for Go and 0.5*(R_miss + R_cr) for NoGo."""
    grid = np.asarray(grid, dtype=float)
    is_go = grid < BOUNDARY_DEG
    is_boundary = grid == BOUNDARY_DEG
    is_nogo = grid > BOUNDARY_DEG

    u_go = np.zeros_like(grid)
    u_go[is_go] = utility['R_hit']
    u_go[is_boundary] = 0.5 * utility['R_hit'] + 0.5 * utility['R_fa']
    u_go[is_nogo] = utility['R_fa']

    u_nogo = np.zeros_like(grid)
    u_nogo[is_go] = utility['R_miss']
    u_nogo[is_boundary] = 0.5 * utility['R_miss'] + 0.5 * utility['R_cr']
    u_nogo[is_nogo] = utility['R_cr']

    return u_go, u_nogo


def predict_velocity(Q, grid, beta_vel, alpha_vel, utility=DEFAULT_UTILITY):
    """Predicted z-scored velocity from a decoded Q via the Stage 1 emission.

    DV(Q) = sum Q(s) * (U(Go,s) - U(NoGo,s)); vel = beta_vel * DV + alpha_vel.
    """
    Q = np.asarray(Q, dtype=float)
    u_go, u_nogo = _utility_vectors(grid, utility)
    eu_go = (Q * u_go[None, :]).sum(axis=1)
    eu_nogo = (Q * u_nogo[None, :]).sum(axis=1)
    dv = eu_go - eu_nogo
    return beta_vel * dv + alpha_vel


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_choice_metrics(p_pred, choice_actual):
    """NLL and AUROC of a soft-probability choice prediction against binary
    actual choices. NLL clips probabilities to [eps, 1-eps] to keep log
    finite for saturated predictions."""
    p_pred = np.asarray(p_pred, dtype=float)
    actual = np.asarray(choice_actual, dtype=float)
    eps = 1e-12
    p_safe = np.clip(p_pred, eps, 1 - eps)
    nll = -np.mean(actual * np.log(p_safe) + (1 - actual) * np.log(1 - p_safe))

    # AUROC: rank-based, undefined if only one class present.
    classes = np.unique(actual)
    if len(classes) < 2:
        auroc = np.nan
    else:
        # sklearn-equivalent AUROC via Mann–Whitney U.
        # rank ascending; for ties, average ranks.
        order = np.argsort(p_pred, kind='mergesort')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p_pred) + 1)
        # Average ranks over ties
        s = np.sort(p_pred)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                tied_idx = order[i:j + 1]
                avg = ranks[tied_idx].mean()
                ranks[tied_idx] = avg
            i = j + 1

        n_pos = (actual == 1).sum()
        n_neg = (actual == 0).sum()
        rank_pos = ranks[actual == 1].sum()
        u = rank_pos - n_pos * (n_pos + 1) / 2
        auroc = u / (n_pos * n_neg)
    return float(nll), float(auroc)


def compute_velocity_metrics(vel_pred, vel_actual):
    """R^2 (coefficient of determination) and Pearson r between predicted
    and actual velocity. Returns NaN for r when pred has zero variance."""
    p = np.asarray(vel_pred, dtype=float)
    a = np.asarray(vel_actual, dtype=float)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # ``np.std`` of a constant-valued array can return ~1e-17 due to
    # floating-point error in the mean; ``np.ptp`` is exactly zero for
    # truly-constant arrays. Use a tolerance over the ptp to robustly
    # detect zero-variance predictors and report NaN for the (undefined)
    # correlation coefficient.
    if np.ptp(p) <= 1e-12 or np.ptp(a) <= 1e-12:
        r = np.nan
    else:
        r = float(np.corrcoef(p, a)[0, 1])
    return float(r2), float(r)


# ----------------------------------------------------------------------
# Trial-mask alignment
# ----------------------------------------------------------------------

def get_trials_to_keep_mask(session_names):
    """Replicate ``stack_IO_trial_tables`` from refresh_IO_in_export.m: for
    each session, drop the last trial. Sessions are encountered in their
    original temporal order (not lexicographic).

    Returns a boolean mask of length n_trials.
    """
    s = np.asarray(session_names)
    n = len(s)
    if n == 0:
        return np.array([], dtype=bool)
    keep = np.ones(n, dtype=bool)
    # Identify sessions in order of first occurrence and find their end indices.
    seen = {}
    order = []
    for i, name in enumerate(s):
        key = str(name)
        if key not in seen:
            seen[key] = i
            order.append(key)
    # Last index of each session = index just before next session starts,
    # or n-1 for the final session.
    starts = [seen[k] for k in order]
    last_indices = [starts[i + 1] - 1 for i in range(len(starts) - 1)] + [n - 1]
    for li in last_indices:
        keep[li] = False
    return keep


def assert_choices_aligned(io_choices_masked, decoder_choices, label='animal'):
    """Hard sanity check that the masked IO choices match the decoder-side
    choice array exactly. Raises AssertionError on length or value mismatch."""
    a = np.asarray(io_choices_masked, dtype=float)
    b = np.asarray(decoder_choices, dtype=float)
    assert a.shape == b.shape, (
        f"[{label}] choice array length mismatch: "
        f"IO masked = {a.shape}, decoder = {b.shape}"
    )
    if not np.array_equal(a, b):
        n_mismatch = int(np.sum(a != b))
        raise AssertionError(
            f"[{label}] {n_mismatch} of {len(a)} choice values disagree "
            f"between IO (masked) and decoder. Trial alignment is broken."
        )


# ----------------------------------------------------------------------
# IO loading
# ----------------------------------------------------------------------

def load_io_results(path):
    """Load IOResults.mat as a list of per-animal structs.

    Tries scipy.io.loadmat first (handles v7 and earlier). On a
    NotImplementedError (raised for MATLAB v7.3 / HDF5 files saved with
    ``save(..., '-v7.3')``), falls back to ``_load_io_results_v73`` which
    walks the file via h5py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Sniff for HDF5 magic bytes. MATLAB v7.3 files are HDF5 and start with
    # \x89HDF\r\n\x1a\n. Detecting this directly lets us route to the h5py
    # loader without relying on scipy's exact exception type, which differs
    # between MATLAB-written and h5py-written files (the latter lack the
    # MATLAB header at bytes 124-127 and provoke ValueError instead of
    # NotImplementedError).
    with open(path, 'rb') as f:
        magic = f.read(8)
    if magic == b'\x89HDF\r\n\x1a\n':
        return _load_io_results_v73(path)

    try:
        mat = sio.loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        # Belt-and-braces: also fall back if scipy itself flags v7.3.
        if 'v7.3' in str(exc) or 'HDF' in str(exc).upper():
            return _load_io_results_v73(path)
        raise
    animals = mat['IOResults']['animals']
    # simplify_cells turns the cell array into a list of dicts
    if isinstance(animals, dict):
        animals = [animals]  # single-animal case
    return list(animals)


# ----------------------------------------------------------------------
# MATLAB v7.3 (HDF5) loader
# ----------------------------------------------------------------------

def _h5_to_native(node, root):
    """Convert an h5py Group or Dataset to a native Python value matching
    what ``scipy.io.loadmat(simplify_cells=True)`` would produce for the
    v7-formatted version of the same file.

    Disambiguation is keyed on the ``MATLAB_class`` HDF5 attribute that
    MATLAB writes on every node:

      - 'cell'   -> dataset of object references; dereference each entry
                    and return a Python list (length-1 unwrapped to a
                    scalar to mimic simplify_cells unwrapping).
      - 'char'   -> uint16 dataset; decode codepoints to a Python str.
      - 'logical'-> numeric array reinterpreted as bool.
      - everything else (numeric / unknown) -> numpy array, transposed if
        2+D (MATLAB col-major <-> HDF5 row-major), squeezed of singletons.

    Groups become dicts of {field_name: walked child}.
    """
    if isinstance(node, h5py.Group):
        # Could be a struct (group with named fields). MATLAB struct arrays
        # of length > 1 produce groups whose fields are themselves cell
        # arrays of refs; for our schema the IOResults animal structs are
        # all scalar structs (one group per animal), so this branch is
        # adequate.
        out = {}
        for key, child in node.items():
            out[key] = _h5_to_native(child, root)
        return out

    # Read the dataset's raw bytes
    arr = node[()]

    # Resolve MATLAB_class to a Python str if present
    matlab_class = node.attrs.get('MATLAB_class')
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode('utf-8', errors='replace')

    if matlab_class == 'char':
        # MATLAB char arrays: stored as uint16 codepoints. Flattened decode
        # is robust for the simple-string fields in our schema (session
        # names etc.). Strips the typical zero terminator.
        flat = np.asarray(arr).flatten()
        return ''.join(chr(int(c)) for c in flat if int(c) != 0)

    if matlab_class == 'cell':
        # Object-ref dataset; each entry points to another node.
        items = []
        flat = np.asarray(arr).flat
        for ref in flat:
            try:
                items.append(_h5_to_native(root[ref], root))
            except (KeyError, ValueError, TypeError):
                items.append(None)
        # Empty cell -> empty list
        if len(items) == 0:
            return []
        # 1-element cell -> unwrap to mimic simplify_cells
        if len(items) == 1:
            return items[0]
        return items

    if matlab_class == 'logical':
        arr = np.asarray(arr).astype(bool)

    # Numeric arrays — transpose to flip MATLAB col-major into row-major
    # so that sio.loadmat-equivalent shapes are preserved.
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        arr = arr.T

    # Squeeze singleton dimensions for simplify_cells parity
    arr = np.squeeze(arr)

    # 0-D array -> scalar
    if arr.ndim == 0:
        return arr.item()
    return arr


def _load_io_results_v73(path):
    """Walk a MATLAB v7.3 IOResults.mat via h5py and return the per-animal
    list, matching the layout produced by scipy.io.loadmat(simplify_cells=True).
    """
    if not _HAVE_H5PY:
        raise ImportError(
            "IOResults.mat is in MATLAB v7.3 format and h5py is not "
            "installed. `pip install h5py` (or `mamba install h5py`) and "
            "retry, or re-save IOResults with `-v7` from MATLAB."
        )
    with h5py.File(path, 'r') as f:
        if 'IOResults' not in f:
            raise KeyError(
                f"Expected top-level 'IOResults' in {path!r}; found "
                f"{list(f.keys())}."
            )
        ior = _h5_to_native(f['IOResults'], f)
    animals = ior['animals']
    # Ensure list-of-dicts even if there's a single animal
    if isinstance(animals, dict):
        animals = [animals]
    return list(animals)


# ----------------------------------------------------------------------
# Per-animal driver
# ----------------------------------------------------------------------

def compute_io_coherence_for_animal(
    animal,
    mouse_id,
    split,
    decoder_full_decoded_by_target,
    grid=DEFAULT_GRID_91,
):
    """Compute per-trial coherence predictions for one animal across all
    target types in ``decoder_full_decoded_by_target`` and return a long-form
    DataFrame.

    Parameters
    ----------
    animal : dict
        Single entry from IOResults.animals after simplify_cells unpack.
    mouse_id : int
        Decoder-side index for this animal.
    split : str
        Split label to attach to every output row.
    decoder_full_decoded_by_target : dict
        Maps target_type ('Q', 'L', 'd') to dict of arch -> array of
        shape (n_trials_kept, n_categories). The arch sub-dict typically
        has keys 'spat' and 'temp'; shuffled controls can be included
        and will be passed through too.
    grid : ndarray
        Orientation grid for Q and L (default arange(0, 91)). Ignored for
        the 2-bin decision target.

    Returns
    -------
    pd.DataFrame
        Long-form: one row per (arch, target, trial). Columns include
        the per-trial predictions, IO baselines, and actual behavioural
        readouts.

    Raises
    ------
    AssertionError
        If the masked IO choice array does not match the decoder's input
        size, or if there is a value mismatch (caught by the caller via
        ``assert_choices_aligned`` once we have the decoder choices).
    """
    sess = animal['data']['trial_keys']['session_name']
    keep_mask = get_trials_to_keep_mask(sess)
    n_kept = int(keep_mask.sum())

    # Masked per-trial IO arrays.
    io_choices = np.asarray(animal['data']['choices'])[keep_mask].astype(float)
    io_vel = np.asarray(animal['data']['conf_vel'])[keep_mask].astype(float)
    io_pred_pgo = np.asarray(animal['pred']['cv_p_choice_stage2'])[keep_mask].astype(float)
    io_pred_vel = np.asarray(animal['pred']['cv_vel'])[keep_mask].astype(float)

    # Stage 1 / Stage 2 fitted parameters
    choice = animal['fit']['choice']
    full_params = animal['fit']['full_params']
    utility = animal['fit'].get('utility', DEFAULT_UTILITY)

    alpha_r = float(choice['alpha_r'])
    beta_r = float(choice['beta_r'])
    gamma_r = float(choice['gamma_r'])
    delta_r = float(choice['delta_r'])
    beta_vel = float(full_params['vel_slope'])
    alpha_vel = float(full_params['vel_intercept'])

    rows = []

    for target_type, arch_dict in decoder_full_decoded_by_target.items():
        for arch, decoded in arch_dict.items():
            decoded = np.asarray(decoded)
            assert decoded.shape[0] == n_kept, (
                f"[mouse {mouse_id} / {target_type} / {arch}] "
                f"decoder full_decoded has {decoded.shape[0]} trials but "
                f"IO masked arrays have {n_kept}. Trial alignment is broken."
            )

            if target_type in ('Q', 'L'):
                g = compute_log_posterior_odds(decoded, grid)
                p_lapse = predict_choice_lapse(g, alpha_r, beta_r, gamma_r, delta_r)
                p_naive = predict_choice_naive(decoded, grid)
                vel_hat = predict_velocity(
                    decoded, grid, beta_vel=beta_vel, alpha_vel=alpha_vel,
                    utility=utility,
                )
            elif target_type == 'd':
                # decoded is (n_trials, 2); first col is P(Go) per the
                # MATLAB convention.
                pgo = decoded[:, 0]
                pnogo = decoded[:, 1]
                # log-odds straight from the 2-bin decoder
                g = np.clip(
                    np.log(np.clip(pgo, 1e-30, 1.0))
                    - np.log(np.clip(pnogo, 1e-30, 1.0)),
                    -LOG_ODDS_CLIP, LOG_ODDS_CLIP,
                )
                p_lapse = predict_choice_lapse(g, alpha_r, beta_r, gamma_r, delta_r)
                p_naive = pgo
                # Velocity prediction is undefined for d (no s-grid). Leave NaN.
                vel_hat = np.full(n_kept, np.nan)
            else:
                raise ValueError(f"Unknown target_type {target_type!r}.")

            for i in range(n_kept):
                rows.append(dict(
                    mouse=mouse_id, arch=arch, target=target_type, split=split,
                    trial=i,
                    pred_pgo_lapse=p_lapse[i],
                    pred_pgo_naive=p_naive[i],
                    pred_vel=vel_hat[i],
                    actual_choice=io_choices[i],
                    actual_vel=io_vel[i],
                    io_pred_pgo=io_pred_pgo[i],
                    io_pred_vel=io_pred_vel[i],
                ))

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate_metrics_per_mouse(trial_df):
    """Collapse the per-trial coherence frame into per-(mouse, arch, target,
    split) summary metrics: choice NLL, AUROC, vel R^2, vel r, plus the
    same metrics for the IO baseline."""
    rows = []
    keys = ['mouse', 'arch', 'target', 'split']
    for key, sub in trial_df.groupby(keys):
        # Choice metrics under the lapse and naive prediction paths
        nll_lapse, auroc_lapse = compute_choice_metrics(
            sub['pred_pgo_lapse'].to_numpy(), sub['actual_choice'].to_numpy(),
        )
        nll_naive, auroc_naive = compute_choice_metrics(
            sub['pred_pgo_naive'].to_numpy(), sub['actual_choice'].to_numpy(),
        )
        nll_io, auroc_io = compute_choice_metrics(
            sub['io_pred_pgo'].to_numpy(), sub['actual_choice'].to_numpy(),
        )

        # Velocity metrics: skip if pred_vel is all-NaN (e.g. for target='d')
        pred_vel = sub['pred_vel'].to_numpy()
        actual_vel = sub['actual_vel'].to_numpy()
        io_pred_vel = sub['io_pred_vel'].to_numpy()
        if np.all(np.isnan(pred_vel)):
            r2_dec, r_dec = np.nan, np.nan
        else:
            r2_dec, r_dec = compute_velocity_metrics(pred_vel, actual_vel)
        r2_io, r_io = compute_velocity_metrics(io_pred_vel, actual_vel)

        row = dict(zip(keys, key))
        row.update(dict(
            nll_lapse=nll_lapse, auroc_lapse=auroc_lapse,
            nll_naive=nll_naive, auroc_naive=auroc_naive,
            nll_io=nll_io, auroc_io=auroc_io,
            vel_r2=r2_dec, vel_r=r_dec,
            io_vel_r2=r2_io, io_vel_r=r_io,
        ))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Full driver: load + decompose all targets / splits
# ----------------------------------------------------------------------

# Iteration tuples sourced from paths.FIT_BASENAMES so the canonical
# basenames live in exactly one place (paths.py). Kept module-level here
# only for backwards-compatible re-import; downstream code should prefer
# ``paths.FIT_BASENAMES`` directly.
PRODUCTION_TARGETS = tuple(
    (t, stem) for t, stem in paths.FIT_BASENAMES['fixed'].items()
    if t in ('Q', 'L', 'd')
)
STIM_MEAN_TARGETS = tuple(
    (t, stem) for t, stem in paths.FIT_BASENAMES['stim_mean'].items()
    if t in ('Q', 'L', 'd')
)


def _merge_archs_from_file(out, path, target_type, arch_keys):
    """Helper: read a population_results-shaped .mat and merge any of
    ``arch_keys`` that have ``full_decoded`` into the nested out dict
    keyed by (mouse_id, target_type)."""
    if not os.path.exists(path):
        return False
    mat = sio.loadmat(path, simplify_cells=True)
    for mouse_key, mdata in mat['results'].items():
        try:
            mid = int(str(mouse_key).split('_')[-1])
        except ValueError:
            mid = mouse_key
        dist = mdata['Dist']
        bucket = out.setdefault(mid, {}).setdefault(target_type, {})
        for arch in arch_keys:
            if arch in dist and 'full_decoded' in dist[arch]:
                bucket[arch] = np.asarray(dist[arch]['full_decoded'])
    return True


def _load_full_decoded_for_split(split, directory):
    """Load every population_results_fixed_*_{split}.mat file present and
    the optional population_results_stim_mean_*_{split}.mat baselines.
    Returns nested dict: {mouse_id: {target_type: {arch: full_decoded}}}.

    Skips missing files with a printed warning so partial runs still work.
    """
    out = {}
    for target_type, stem in PRODUCTION_TARGETS:
        path = os.path.join(directory, f"{stem}_{split}.mat")
        if not _merge_archs_from_file(
            out, path, target_type,
            arch_keys=('spat', 'temp', 'spat_shf', 'temp_shf'),
        ):
            print(f"[io_coherence] missing {path}; skipping target {target_type}")

    # Stim-mean baselines (single-arch) merged into the same per-(mouse,
    # target) bucket so downstream iteration treats stim_mean as just
    # another arch alongside spat/temp/.shf.
    for target_type, stem in STIM_MEAN_TARGETS:
        path = os.path.join(directory, f"{stem}_{split}.mat")
        _merge_archs_from_file(out, path, target_type, arch_keys=('stim_mean',))
    return out


def compute_io_coherence(
    io_path,
    splits,
    directory='.',
    grid=DEFAULT_GRID_91,
):
    """End-to-end driver. Loads IOResults.mat, then for each split loads the
    three production results files and computes per-trial coherence
    predictions for every (mouse, arch, target_type, split). Returns a
    single concatenated long-form DataFrame.

    Aborts loudly on any animal whose masked IO choice array does not match
    the decoder's choice array of the same length (caught by an explicit
    ``assert_choices_aligned`` between the two paths to the same data).
    """
    animals = load_io_results(io_path)
    n_animals = len(animals)

    parts = []
    for split in splits:
        decoded_by_mouse = _load_full_decoded_for_split(split, directory)
        if not decoded_by_mouse:
            print(f"[io_coherence] no decoder files for split={split}; skipping.")
            continue

        for mid, target_dict in decoded_by_mouse.items():
            if mid >= n_animals:
                print(f"[io_coherence] mouse_id {mid} >= {n_animals} animals; skip.")
                continue
            df = compute_io_coherence_for_animal(
                animals[mid], mouse_id=mid, split=split,
                decoder_full_decoded_by_target=target_dict, grid=grid,
            )
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
