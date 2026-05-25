# -*- coding: utf-8 -*-
"""Quick check: what is the *current* diagonal recovery loss for each
mouse, using the saved recovery cache from the production run?

Reads ``recovery_cache_fixed_perception.npy`` (the Q-target recovery
crossover that ``run_fixed_recovery.py`` produced) and computes the
per-mouse mean PCA-weighted Euclidean loss for every combination of:

  - target_arch     : architecture that generated the synthetic target
                      ('spat' = PPC,  'temp' = SBC)
  - decoder_arch    : architecture being retrained against that target

The DIAGONAL cells (target_arch == decoder_arch) answer the supervisor's
question directly: ``recovery_results['spat'][mouse]['Dist']['spat']``
is the PPC' decoder fit to PPC-generated targets, etc.

Output: a per-mouse 2x2 table plus the grand mean across mice. No
retraining required.

If the diagonal numbers are non-zero, that's the starting point. The
follow-up question — does that floor go away with more training? — is
answered by ``recovery_convergence_probe.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402

import numpy as np

from pca_loss import pca_distance

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load_recovery_cache(path):
    """Returns the {target_arch: {mouse_X: {Dist: ...}}, base_config: ...}
    dict produced by run_fixed_recovery.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def quick_check(cache_path: str):
    rec = load_recovery_cache(cache_path)
    loss_func = rec['base_config'].get('custom_loss_func', 'PCA')
    print(f"\nRecovery cache: {cache_path}")
    print(f"  loss_func = {loss_func}")
    print(f"  target_archs in cache: {[k for k in rec if k != 'base_config']}")

    target_archs = [k for k in ('spat', 'temp') if k in rec]
    decoder_archs = ('spat', 'temp')

    # Build per-mouse 2x2 loss matrix
    mice = sorted(set().union(*[set(rec[t]) for t in target_archs]))
    print(f"\nPer-mouse mean {loss_func} loss [rows = target_arch, cols = decoder_arch]:")
    grand = {(ta, da): [] for ta in target_archs for da in decoder_archs}
    for m_key in mice:
        try:
            mid = int(m_key.split('_')[-1])
        except ValueError:
            mid = m_key
        print(f"\n  Mouse {mid}:")
        header = "    " + " " * 16 + " | ".join(f"dec={da:<10}" for da in decoder_archs)
        print(header)
        for ta in target_archs:
            parent = rec[ta][m_key]['Dist']
            pcs = parent.get('pcs', None)
            evar = parent.get('explained_var', None)
            row_vals = []
            for da in decoder_archs:
                d = parent[da]
                pcs_arr = np.asarray(pcs) if pcs is not None and len(pcs) else None
                evar_arr = np.asarray(evar) if evar is not None and len(evar) else None
                if pcs_arr is None:
                    # No PCA basis for this cell — record NaN so the
                    # np.nanmean below skips it. (pca_distance raises on a
                    # missing basis; this script formerly fell back to MSE
                    # silently, which mixed two metrics in one table.)
                    losses = np.full(np.asarray(d['decoded']).shape[0], np.nan)
                else:
                    losses = pca_distance(
                        np.asarray(d['decoded']), np.asarray(d['target']),
                        pcs_arr, evar_arr,
                    )
                mean_loss = float(np.nanmean(losses))
                row_vals.append(mean_loss)
                grand[(ta, da)].append(mean_loss)
            marker = lambda ta_, da_: ' <-- DIAGONAL' if ta_ == da_ else ''
            row_str = "    target={:>4}    | ".format(ta) + \
                " | ".join(f"{v:>10.5f}{marker(ta, da)}" for v, da in zip(row_vals, decoder_archs))
            print(row_str)

    # Grand mean across mice
    print("\n  Grand mean across mice:")
    print("    " + " " * 16 + " | ".join(f"dec={da:<10}" for da in decoder_archs))
    for ta in target_archs:
        row_vals = [np.nanmean(grand[(ta, da)]) for da in decoder_archs]
        row_str = "    target={:>4}    | ".format(ta) + " | ".join(
            f"{v:>10.5f}{'  <-- DIAGONAL' if ta == da else ''}"
            for v, da in zip(row_vals, decoder_archs)
        )
        print(row_str)

    print("\nInterpretation:")
    print("  - DIAGONAL cells are: PPC' fit to PPC-generated, SBC' fit to SBC-generated.")
    print("  - By construction loss=0 is achievable (same model that generated the target)")
    print("    — any non-zero diagonal here is training-procedure imperfection.")
    print("  - If diagonal << off-diagonal: architectural dissociation survives, but the")
    print("    training procedure is leaving slack on the table.")
    print("  - Run recovery_convergence_probe.py to test whether more epochs/inits")
    print("    drives the diagonal toward 0.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache', default=str(paths.recovery_cache('perception')))
    args, _ = parser.parse_known_args()
    quick_check(args.cache)
