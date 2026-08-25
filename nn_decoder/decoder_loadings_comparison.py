# -*- coding: utf-8 -*-
"""Functional-alignment comparison of decoder loadings.

The spatial and temporal architectures (and the Q vs stim_cat targets) all share
the same MLP backbone: a single hidden layer of H units mapping V1
activity to a 91-bin output. Asking "do the Q decoder and the stim
decoder use the same V1 directions" is meaningful only AFTER the
hidden-unit permutation symmetry is broken, since hidden-unit ordering
is arbitrary and `tanh(-x) = -tanh(x)` gives a sign symmetry on top.

Strategy: align the hidden units of two decoders by what they do at the
OUTPUT (Hungarian assignment on |cos(W_out)| with sign-correction),
then measure how similar their INPUT weights are after alignment. The
alignment criterion is independent of W_in, so the post-alignment W_in
cosines are a clean measurement of "given that two units do the same
thing at the output, do they listen to the same V1 directions?".

Interpretation of the aggregate aligned-pair W_in cosine:
  - high (>= 0.7): the Q decoder lives in the same V1 subspace as the
    stim decoder. The Q decoder is, at the V1 read-out level, basically
    a stim decoder.
  - low (<= 0.3): the Q decoder uses V1 directions the stim decoder
    isn't using. The posterior content is not just a stim-decode reskinned.
  - in between: muddy; look at the spectrum, not just the scalar.

Synthetic tests in tests/test_decoder_loadings_comparison.py cover:
  - identity (align decoder to itself -> identity perm, cos=1)
  - permutation recovery (known unit permutation -> recovered)
  - sign-flip recovery (known sign flips -> recovered)
  - independent-random null (cos distribution centred ~0)
  - planted shared subspace (cos distribution shifted above null)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from figsave import save_fig


# ----------------------------------------------------------------------
# Functional alignment primitives — pure numpy, fully testable.
# ----------------------------------------------------------------------


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine between columns of A and columns of B.

    A: (D, H_A); B: (D, H_B); returns (H_A, H_B) with entries
    cos(A[:, k], B[:, j]).
    """
    norms_A = np.linalg.norm(A, axis=0, keepdims=True)  # (1, H_A)
    norms_B = np.linalg.norm(B, axis=0, keepdims=True)  # (1, H_B)
    denom = (norms_A.T @ norms_B) + 1e-12
    return (A.T @ B) / denom


def hungarian_align_units(W_out_A: np.ndarray, W_out_B: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find permutation pi and sign flips s such that B's units are
    matched to A's units by output-weight similarity.

    Parameters
    ----------
    W_out_A : (n_output, H_A)
        Decoder A's hidden -> output weights.
    W_out_B : (n_output, H_B)
        Decoder B's hidden -> output weights.

    Returns
    -------
    row_ind : (H,) array of int
        Index into A's units. Always range(H) where H = min(H_A, H_B).
    pi : (H,) array of int
        For each row_ind[k], the index into B's units that best matches.
    signs : (H,) array of {-1, +1}
        Sign correction for B's unit pi[k] so that the cosine with A's
        unit k is positive. Apply via: B_aligned[k] = signs[k] * B[pi[k]]
        for both W_in_B and W_out_B (the network function is invariant
        under joint sign flip of W_in and W_out columns for tanh units).
    """
    # scipy is imported lazily so the module imports under environments
    # without scipy (e.g. lightweight test runners).
    from scipy.optimize import linear_sum_assignment

    cos_matrix = _cosine_matrix(W_out_A, W_out_B)  # (H_A, H_B)
    # Hungarian on -|cos| to maximise total |cos|. |cos| because the
    # tanh sign symmetry means (-W_out_B[:, j], -W_in_B[j, :]) is the
    # same unit; we want to align by functional shape, not raw sign.
    row_ind, col_ind = linear_sum_assignment(-np.abs(cos_matrix))
    pi = col_ind
    # Sign correction: pick s_k = sign(cos(A[k], B[pi[k]])) so that the
    # signed cosine is positive after alignment.
    raw_cos = cos_matrix[row_ind, col_ind]
    signs = np.where(raw_cos >= 0, 1.0, -1.0)
    return row_ind, pi, signs


def aligned_pair_input_cosines(W_in_A: np.ndarray, W_in_B: np.ndarray,
                                 pi: np.ndarray, signs: np.ndarray
                                 ) -> np.ndarray:
    """Signed cosine between W_in_A[k] and signs[k] * W_in_B[pi[k]] for
    each aligned pair.

    Parameters
    ----------
    W_in_A : (H_A, n_neurons)
    W_in_B : (H_B, n_neurons)
    pi, signs : as returned by hungarian_align_units

    Returns
    -------
    (H,) array of cosines, where H = len(pi).
    """
    H = len(pi)
    out = np.zeros(H)
    for k in range(H):
        a = W_in_A[k]
        b = signs[k] * W_in_B[pi[k]]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            out[k] = 0.0
        else:
            out[k] = float(np.dot(a, b) / (na * nb))
    return out


def aligned_pair_output_cosines(W_out_A: np.ndarray, W_out_B: np.ndarray,
                                  pi: np.ndarray, signs: np.ndarray
                                  ) -> np.ndarray:
    """Signed cosine between W_out_A[:, k] and signs[k] * W_out_B[:, pi[k]]
    for each aligned pair. These are the cosines the Hungarian
    maximised; reporting them alongside the input-side cosines lets you
    sort pairs by alignment quality and see how the input cosine
    distribution depends on it.
    """
    H = len(pi)
    out = np.zeros(H)
    for k in range(H):
        a = W_out_A[:, k]
        b = signs[k] * W_out_B[:, pi[k]]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            out[k] = 0.0
        else:
            out[k] = float(np.dot(a, b) / (na * nb))
    return out


def compare_decoders(W_in_A: np.ndarray, W_out_A: np.ndarray,
                       W_in_B: np.ndarray, W_out_B: np.ndarray
                       ) -> Dict[str, np.ndarray]:
    """Full alignment + per-pair scoring for one (A, B) decoder pair.

    Returns a dict:
      - 'pi':              alignment permutation (length H)
      - 'signs':           sign corrections (length H)
      - 'cos_out_aligned': signed output cosines after alignment
      - 'cos_in_aligned':  signed input cosines after alignment
      - 'cos_in_raw':      input cosines BEFORE alignment (unit k of A
                            vs unit k of B) — a null sanity check;
                            should be at chance unless A and B already
                            share unit ordering.
    """
    _, pi, signs = hungarian_align_units(W_out_A, W_out_B)
    cos_out_aligned = aligned_pair_output_cosines(W_out_A, W_out_B, pi, signs)
    cos_in_aligned = aligned_pair_input_cosines(W_in_A, W_in_B, pi, signs)
    # Raw "unaligned" comparison: pair B's units with A's units in the
    # order they happened to come out of training. Random ordering means
    # this is at chance even when the networks are functionally similar.
    H = min(W_in_A.shape[0], W_in_B.shape[0])
    cos_in_raw = aligned_pair_input_cosines(
        W_in_A[:H], W_in_B[:H],
        pi=np.arange(H), signs=np.ones(H),
    )
    return {
        'pi': pi,
        'signs': signs,
        'cos_out_aligned': cos_out_aligned,
        'cos_in_aligned': cos_in_aligned,
        'cos_in_raw': cos_in_raw,
        # Pre-alignment |cos(W_out)| matrix (H_A x H_B) — included so the
        # per-mouse plot can render the alignment landscape without
        # round-tripping through the original W_out arrays.
        'cos_out_matrix_pre': np.abs(_cosine_matrix(W_out_A, W_out_B)),
    }


# ----------------------------------------------------------------------
# .mat I/O — loads the per-(mouse, split) decoder weight snapshots that
# run_animal_decoder now saves under the 'Weights' key.
# ----------------------------------------------------------------------


ARCH_KEYS = ('spat', 'temp')  # 'spat' = spatial, 'temp' = temporal


@dataclass
class DecoderRun:
    """One target's results across mice and splits, loaded from
    results/<run_name>/<slug>/<split>.mat."""
    name: str
    weights_by_arch_mouse_split: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]
    # weights_by_arch_mouse_split[arch][mouse_id][split]
    #   = {'W_in': ..., 'b_in': ..., 'W_out': ..., 'b_out': ...}


def load_decoder_run(results_dir: Path, name: str,
                       splits: Sequence[str] = ('stratified_balanced',
                                                  'generalize_contrast',
                                                  'generalize_dispersion')
                       ) -> DecoderRun:
    """Load every (mouse, split) weight snapshot from one slug directory.

    Parameters
    ----------
    results_dir : Path
        Path to a slug directory (e.g. results/<run_name>/Q_PCA_half_100ms_condmean).
    name : str
        Tag used to identify this run in downstream plots/output (e.g.
        'Q_condmean', 'Q_residual', 'stim_cat').
    splits : iterable of str
        Which split files to load. Missing files are skipped with a warning.
    """
    import scipy.io as sio

    by_arch: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {
        arch: {} for arch in ARCH_KEYS
    }
    for split in splits:
        mat_path = results_dir / f'{split}.mat'
        if not mat_path.exists():
            print(f"[warn] {mat_path} not found; skipping")
            continue
        data = sio.loadmat(str(mat_path), simplify_cells=True)
        results = data['results']
        for mouse_key, mouse_dict in results.items():
            # mouse_key is "mouse_0", "mouse_1", ...
            if not mouse_key.startswith('mouse_'):
                continue
            mid = int(mouse_key.split('_', 1)[1])
            weights_block = mouse_dict.get('Weights')
            if weights_block is None:
                print(f"[warn] {mat_path}::{mouse_key} has no 'Weights' key; "
                       f"this is an old (pre-weight-saving) .mat file. Re-run "
                       f"the training pipeline to populate it.")
                continue
            for arch in ARCH_KEYS:
                if arch not in weights_block:
                    continue
                by_arch[arch].setdefault(mid, {})[split] = {
                    'W_in':  np.asarray(weights_block[arch]['W_in']),
                    'b_in':  np.asarray(weights_block[arch]['b_in']),
                    'W_out': np.asarray(weights_block[arch]['W_out']),
                    'b_out': np.asarray(weights_block[arch]['b_out']),
                }
    return DecoderRun(name=name, weights_by_arch_mouse_split=by_arch)


# ----------------------------------------------------------------------
# Driver: per (arch, mouse, split), compare two DecoderRuns and aggregate.
# ----------------------------------------------------------------------


def compare_runs(run_a: DecoderRun, run_b: DecoderRun,
                  out_dir: Optional[Path] = None,
                  ) -> Dict[Tuple[str, int, str], Dict[str, np.ndarray]]:
    """Compare two decoder runs (e.g. Q vs stim_cat) across every
    (arch, mouse, split) they both have.

    Returns a flat dict keyed by (arch, mouse_id, split) with the
    per-pair scoring dict (output + input cosines, permutation, signs).
    If out_dir is given, writes a tidy summary CSV.
    """
    results: Dict[Tuple[str, int, str], Dict[str, np.ndarray]] = {}
    for arch in ARCH_KEYS:
        mice_a = set(run_a.weights_by_arch_mouse_split[arch])
        mice_b = set(run_b.weights_by_arch_mouse_split[arch])
        for mid in sorted(mice_a & mice_b):
            splits_a = set(run_a.weights_by_arch_mouse_split[arch][mid])
            splits_b = set(run_b.weights_by_arch_mouse_split[arch][mid])
            for split in sorted(splits_a & splits_b):
                wa = run_a.weights_by_arch_mouse_split[arch][mid][split]
                wb = run_b.weights_by_arch_mouse_split[arch][mid][split]
                r = compare_decoders(
                    wa['W_in'], wa['W_out'],
                    wb['W_in'], wb['W_out'],
                )
                results[(arch, mid, split)] = r
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_summary_csv(results, run_a.name, run_b.name, out_dir)
    return results


def _write_summary_csv(results, name_a, name_b, out_dir: Path) -> None:
    import csv
    rows = []
    for (arch, mid, split), r in results.items():
        rows.append({
            'arch': arch,
            'mouse_id': mid,
            'split': split,
            'n_units': int(len(r['cos_in_aligned'])),
            'mean_cos_in_aligned': float(np.mean(r['cos_in_aligned'])),
            'median_cos_in_aligned': float(np.median(r['cos_in_aligned'])),
            'mean_cos_out_aligned': float(np.mean(np.abs(r['cos_out_aligned']))),
            'mean_cos_in_raw': float(np.mean(r['cos_in_raw'])),
        })
    csv_path = out_dir / f'loadings_summary_{name_a}_vs_{name_b}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                  ['arch', 'mouse_id', 'split', 'n_units',
                                    'mean_cos_in_aligned',
                                    'median_cos_in_aligned',
                                    'mean_cos_out_aligned',
                                    'mean_cos_in_raw'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
# Plotting — optional, behind a flag so headless test runs don't pull
# matplotlib.
# ----------------------------------------------------------------------


def plot_per_mouse(results, name_a, name_b, out_dir: Path) -> None:
    """Per (arch, mouse, split), a two-panel figure:
      - panel 1: barplot of per-unit aligned cosines, sorted by output
        alignment quality (best-aligned-first).
      - panel 2: |cos(W_out)| pre-alignment heatmap (H_A x H_B).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (arch, mid, split), r in results.items():
        cos_in = r['cos_in_aligned']
        cos_out = np.abs(r['cos_out_aligned'])
        order = np.argsort(-cos_out)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        x = np.arange(len(cos_in))
        ax.bar(x, cos_in[order], color='steelblue', alpha=0.75,
                label='cos(W_in) after alignment')
        ax.plot(x, cos_out[order], 'o-', color='darkorange', alpha=0.8,
                 label='|cos(W_out)| (alignment quality)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Hidden unit (sorted by output alignment quality)')
        ax.set_ylabel('Cosine similarity')
        ax.set_title(f'{name_a} vs {name_b} — arch={arch}, mouse={mid}, split={split}')
        ax.set_ylim(-1, 1)
        ax.legend(loc='lower left', fontsize=8)
        ax = axes[1]
        # Pre-alignment |cos(W_out)| heatmap, raw unit ordering — shows
        # whether the alignment had a clean signal to work with.
        ax.imshow(r['cos_out_matrix_pre'],
                   vmin=0, vmax=1, cmap='magma', aspect='auto')
        ax.set_xlabel(f'{name_b} hidden unit')
        ax.set_ylabel(f'{name_a} hidden unit')
        ax.set_title('|cos(W_out)| pre-alignment')
        fig.tight_layout()
        save_fig(fig, out_dir,
                 f'loadings_{name_a}_vs_{name_b}_{arch}_mouse{mid}_{split}')


def plot_aggregate(results, name_a, name_b, out_dir: Path) -> None:
    """Across all (mouse, split) for each architecture, plot the
    distribution of aligned-pair W_in cosines vs the pre-alignment raw
    cosines (which serve as a chance baseline)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arch in ARCH_KEYS:
        aligned_all = []
        raw_all = []
        for (a, mid, split), r in results.items():
            if a != arch:
                continue
            aligned_all.append(r['cos_in_aligned'])
            raw_all.append(r['cos_in_raw'])
        if not aligned_all:
            continue
        aligned_all = np.concatenate(aligned_all)
        raw_all = np.concatenate(raw_all)
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-1, 1, 41)
        ax.hist(raw_all, bins=bins, alpha=0.55, label='pre-alignment (chance)',
                  color='gray', density=True)
        ax.hist(aligned_all, bins=bins, alpha=0.7,
                  label='aligned (Hungarian on |cos(W_out)|)',
                  color='steelblue', density=True)
        ax.axvline(np.mean(aligned_all), color='steelblue', lw=2)
        ax.axvline(np.mean(raw_all), color='gray', lw=2)
        ax.set_xlabel('cosine(W_in_A, sign-corrected W_in_B[pi(k)])')
        ax.set_ylabel('density')
        ax.set_title(f'{name_a} vs {name_b} — arch={arch}  '
                       f'(mean aligned = {np.mean(aligned_all):.3f}, '
                       f'mean chance = {np.mean(raw_all):.3f})')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        save_fig(fig, out_dir, f'loadings_aggregate_{name_a}_vs_{name_b}_{arch}')


# ----------------------------------------------------------------------
# CLI / Spyder driver
# ----------------------------------------------------------------------


def main(run_name: str = 'post_fix_loadings_2026_05_17',
         results_root: str = 'results',
         out_dir: str = 'figures/decoder_loadings',
         plots: bool = True):
    """Load the three configurations from a single run_name and emit
    Q-condmean vs stim_cat, Q-residual vs stim_cat, and
    Q-condmean vs Q-residual comparisons.
    """
    root = Path(results_root) / run_name
    out = Path(out_dir)

    q_cond = load_decoder_run(root / 'Q_PCA_half_100ms_condmean', 'Q_condmean')
    q_res  = load_decoder_run(root / 'Q_PCA_half_100ms_residual', 'Q_residual')
    stim   = load_decoder_run(root / 'stim_cat_CE_half_100ms', 'stim_cat')

    for run_a, run_b in [(q_cond, stim),
                          (q_res, stim),
                          (q_cond, q_res)]:
        results = compare_runs(run_a, run_b, out_dir=out)
        if plots:
            plot_aggregate(results, run_a.name, run_b.name, out)
            plot_per_mouse(results, run_a.name, run_b.name, out)
    print(f"Done. Plots + CSVs under {out}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='post_fix_loadings_2026_05_17')
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--out-dir', default='figures/decoder_loadings')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    main(run_name=args.run_name,
         results_root=args.results_root,
         out_dir=args.out_dir,
         plots=not args.no_plots)
