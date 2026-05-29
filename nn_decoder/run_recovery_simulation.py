# -*- coding: utf-8 -*-
"""Driver for the generative model-recovery simulation.

Synthesises PPC and SBC neural populations from known latents, fits every
decoder (PPC, SBC, free mlp/gru/tcn) against the ground-truth posterior /
likelihood / decision posterior, and reports the recovery matrix plus a
double-dissociation plot.

Example
-------
    python run_recovery_simulation.py --targets Q L d \
        --n-trials 400 --n-neurons 80 --epochs 40 --out results/recovery_sim

The matrix is saved as a .json (human-readable metrics) and .npz, and one
heatmap per target showing fit-loss normalised to the shuffle control
(lower = better recovery); a clean double dissociation has the diagonal
(decoder matches generative scheme) lighter than the off-diagonal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from simulation.recovery import run_recovery_matrix, DECODER_SPEC


def _plot_matrix(matrix, out_dir, schemes, decoders):
    """One heatmap per target: rows = generative scheme, cols = decoder,
    cell = fit-loss normalised to shuffle (lower is better recovery)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[plot] skipped ({exc})")
        return

    targets = list(matrix)
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.6),
                             squeeze=False)
    for ax, target in zip(axes[0], targets):
        M = np.array([[matrix[target][s][d]['fit_loss_norm'] for d in decoders]
                      for s in schemes])
        im = ax.imshow(M, cmap='viridis_r', aspect='auto')
        ax.set_xticks(range(len(decoders)))
        ax.set_xticklabels(decoders, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(schemes)))
        ax.set_yticklabels([f"{s.upper()}-gen" for s in schemes])
        ax.set_title(f"target {target}\n(fit/shuffle, lower=better)", fontsize=9)
        for i in range(len(schemes)):
            for j in range(len(decoders)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center',
                        color='white', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / 'recovery_dissociation_matrix.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--schemes', nargs='+', default=['ppc', 'sbc'],
                   choices=['ppc', 'sbc'])
    p.add_argument('--targets', nargs='+', default=['Q', 'L', 'd'],
                   choices=['Q', 'L', 'd'])
    p.add_argument('--decoders', nargs='+', default=list(DECODER_SPEC),
                   choices=list(DECODER_SPEC))
    p.add_argument('--n-trials', type=int, default=400)
    p.add_argument('--n-neurons', type=int, default=80)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--rep', type=int, default=3)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--tuning-width', type=float, default=12.0)
    p.add_argument('--sample-gain', type=float, default=12.0)
    p.add_argument('--uncertainty-levels', nargs='+', type=float,
                   default=[4.0, 8.0, 16.0])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/recovery_sim')
    args = p.parse_args(argv)

    matrix = run_recovery_matrix(
        schemes=tuple(args.schemes), targets=tuple(args.targets),
        decoders=tuple(args.decoders), seed=args.seed,
        n_trials=args.n_trials, n_neurons=args.n_neurons, T=args.T,
        uncertainty_levels=tuple(args.uncertainty_levels),
        tuning_width=args.tuning_width, sample_gain=args.sample_gain,
        REP=args.rep, num_epochs=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'recovery_matrix.json', 'w') as f:
        json.dump(matrix, f, indent=2)
    np.savez(out_dir / 'recovery_matrix.npz', matrix=matrix)
    print(f"Saved metrics to {out_dir / 'recovery_matrix.json'}")
    _plot_matrix(matrix, out_dir, args.schemes, args.decoders)

    # Headline: the diagonal dissociation on the posterior target.
    if 'Q' in matrix and 'ppc' in args.schemes and 'sbc' in args.schemes \
            and {'PPC', 'SBC'}.issubset(args.decoders):
        q = matrix['Q']
        print("\nDouble-dissociation check (target Q, fit/shuffle, lower=better):")
        print(f"  PPC-gen: PPC={q['ppc']['PPC']['fit_loss_norm']:.3f} "
              f"SBC={q['ppc']['SBC']['fit_loss_norm']:.3f}")
        print(f"  SBC-gen: PPC={q['sbc']['PPC']['fit_loss_norm']:.3f} "
              f"SBC={q['sbc']['SBC']['fit_loss_norm']:.3f}")


if __name__ == '__main__':
    main()
