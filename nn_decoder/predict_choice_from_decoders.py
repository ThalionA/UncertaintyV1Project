# -*- coding: utf-8 -*-
"""Predict the animal's trial-by-trial Go/NoGo choice from the decoders'
fitted distributions, and compare against the animal's true choices.

For each decoder cell the decoded distribution on every trial is mapped
to a predicted P(Go), in two variants:

  * **WITH** the ideal-observer stage-2 psychometric — the decoded
    orientation posterior is integrated to a log-odds
    ``g = log P(Go|Q)/P(NoGo|Q)`` and pushed through the IO's fitted
    four-parameter lapse psychometric
    ``P(Go) = gamma + (1-gamma-delta)*sigmoid(alpha*g + beta)`` (per
    mouse, from ``IOResults.mat``).
  * **WITHOUT** — the same log-odds through a plain sigmoid
    (parameter-free: the lapse-free, unit-slope, zero-offset special
    case). Tests how much the IO's behavioural calibration adds.

The d (decision) decoder already outputs a 2-D [P(Go), P(NoGo)]
posterior, so its P(Go) is read off column 0 directly — there is no
orientation->choice mapping, hence d has the without-variant only.

Each (target, arch, variant, mouse) is scored against the animal's
actual choice with three metrics: AUROC, accuracy (at a 0.5 threshold),
and NLL. Outputs:

  * ``choice_prediction_summary.csv`` — one row per
    (target, arch, variant, mouse).
  * ``psychometric_<target>.svg`` — animal vs decoder-with vs
    decoder-without psychometric curves, spatial and temporal panels.
  * ``choice_metric_<auroc|accuracy|nll>.svg`` — spatial-vs-temporal
    bar plots per target and variant.

CLI::

    python predict_choice_from_decoders.py --run-name clean_2026_05_19

Spyder::

    from predict_choice_from_decoders import main
    main(run_name='clean_2026_05_19')
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_coherence as ioc  # noqa: E402
from io_coherence import (  # noqa: E402
    DEFAULT_GRID_91,
    compute_choice_metrics,
    compute_log_posterior_odds,
    predict_choice_lapse,
)
import decoder_plotting_utils as dpu  # noqa: E402


# ----------------------------------------------------------------------
# Which slug carries each choice-prediction target
# ----------------------------------------------------------------------
# Q / L use the condition-mean PCA basis as the canonical decoder.
# (The residual-basis cells exist too; pass a different mapping to
# ``main`` via ``target_slugs`` to use them instead.)

DEFAULT_TARGET_SLUGS = {
    'Q': 'Q_PCA_half_100ms_condmean',
    'L': 'L_PCA_half_100ms_condmean',
    'd': 'd_MSE_half_100ms',
}

# Variant availability: d has no orientation posterior to push through a
# psychometric, so it is without-only.
TARGET_VARIANTS = {
    'Q': ('with', 'without'),
    'L': ('with', 'without'),
    'd': ('without',),
}

REAL_ARCHES = ('spat', 'temp')
CANONICAL_SPLIT = 'stratified_balanced'


# ----------------------------------------------------------------------
# Core: decoded distribution -> P(Go)
# ----------------------------------------------------------------------

def predict_pgo(decoded, target_type, variant,
                grid=None, io_params=None):
    """Map a decoder's per-trial output to predicted P(Go).

    Parameters
    ----------
    decoded : ndarray
        ``(n_trials, n_cats)``. For Q/L this is the 91-D orientation
        posterior; for d it is the 2-D [P(Go), P(NoGo)].
    target_type : {'Q', 'L', 'd'}
        Which decoder. 'd' returns column 0 directly.
    variant : {'with', 'without'}
        'with' applies the IO stage-2 lapse psychometric (needs
        ``io_params``); 'without' applies a plain sigmoid of the
        log-odds. Ignored for target_type='d'.
    grid : ndarray, optional
        Orientation grid for the log-odds integral. Defaults to the
        0..90 deg, 1-deg grid.
    io_params : tuple, optional
        ``(alpha_r, beta_r, gamma_r, delta_r)`` — required for the
        'with' variant.

    Returns
    -------
    ndarray
        Per-trial P(Go), shape ``(n_trials,)``.
    """
    decoded = np.asarray(decoded, dtype=float)
    if target_type == 'd':
        # Decision decoder already outputs [P(Go), P(NoGo)].
        return decoded[:, 0]

    if grid is None:
        grid = DEFAULT_GRID_91
    g = compute_log_posterior_odds(decoded, grid)
    if variant == 'with':
        if io_params is None:
            raise ValueError("variant='with' requires io_params "
                              "(alpha_r, beta_r, gamma_r, delta_r)")
        a, b, gm, dl = io_params
        return predict_choice_lapse(g, a, b, gm, dl)
    elif variant == 'without':
        # Plain sigmoid: the lapse-free, unit-slope, zero-offset case.
        return predict_choice_lapse(g, 1.0, 0.0, 0.0, 0.0)
    raise ValueError(f"variant must be 'with' or 'without', got {variant!r}")


def choice_metrics(p_pred, choice_actual):
    """AUROC, accuracy (0.5 threshold) and NLL of a soft P(Go)
    prediction against the animal's binary choices."""
    nll, auroc = compute_choice_metrics(p_pred, choice_actual)
    p = np.asarray(p_pred, dtype=float)
    a = np.asarray(choice_actual, dtype=float)
    acc = float(np.mean((p >= 0.5).astype(float) == a))
    return {'auroc': auroc, 'accuracy': acc, 'nll': nll}


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def load_io_stage2_params(io_path) -> Dict[int, Tuple[float, float, float, float]]:
    """Return ``{mouse_idx: (alpha_r, beta_r, gamma_r, delta_r)}`` from
    IOResults.mat. Animals are assumed to be in mouse-index order."""
    animals = ioc.load_io_results(str(io_path))
    out = {}
    for i, animal in enumerate(animals):
        try:
            ch = animal['fit']['choice']
            out[i] = (float(ch['alpha_r']), float(ch['beta_r']),
                       float(ch['gamma_r']), float(ch['delta_r']))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[warn] IO stage-2 params missing for animal {i}: {exc}")
    return out


def load_animal_choice_and_orientation(mouse_idx) -> Tuple[np.ndarray, np.ndarray]:
    """Animal Go/NoGo choice and stimulus orientation for every trial of
    one mouse, in the decoder's trial order. Uses the cached raw-trials
    dict from decoder_plotting_utils."""
    raw = dpu._cached_raw_trials(mouse_idx)
    choice = np.asarray(raw['choice'], dtype=float).reshape(-1)
    orient = np.asarray(raw['orientation'], dtype=float).reshape(-1)
    return choice, orient


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _per_mouse_predictions(run_name, results_root, target, slug,
                            io_params):
    """Yield per-mouse prediction bundles for one target/slug.

    Each bundle: ``(mouse_id, arch, variant, p_go, choice, orient)``,
    using the model's full-dataset predictions (``full_decoded``) from
    the canonical split, aligned trial-for-trial with the animal's
    behaviour.
    """
    tree = dpu.load_run_tree(run_name, slug, splits=[CANONICAL_SPLIT],
                              results_root=results_root)
    if CANONICAL_SPLIT not in tree:
        print(f"[warn] {slug}: no {CANONICAL_SPLIT}.mat; skipping target {target}")
        return
    results = tree[CANONICAL_SPLIT]['results']

    for mouse_key, m_data in results.items():
        if not mouse_key.startswith('mouse_'):
            continue
        mid = int(mouse_key.split('_', 1)[1])
        dist = m_data.get('Dist', {})

        choice, orient = load_animal_choice_and_orientation(mid)

        for arch in REAL_ARCHES:
            if arch not in dist:
                continue
            decoded = np.asarray(dist[arch].get('full_decoded'), dtype=float)
            if decoded.ndim != 2 or decoded.shape[0] == 0:
                continue
            # Align decoded predictions with behaviour trial-for-trial.
            n = min(decoded.shape[0], choice.shape[0])
            if decoded.shape[0] != choice.shape[0]:
                print(f"[warn] {slug}/mouse_{mid}/{arch}: decoded "
                       f"{decoded.shape[0]} trials vs {choice.shape[0]} "
                       f"behavioural — truncating to {n}")
            dec_n = decoded[:n]
            ch_n = choice[:n]
            or_n = orient[:n]
            # Drop trials with a non-binary / missing choice.
            valid = np.isfinite(ch_n) & np.isin(ch_n, (0.0, 1.0))
            dec_n, ch_n, or_n = dec_n[valid], ch_n[valid], or_n[valid]
            if ch_n.size == 0:
                continue

            for variant in TARGET_VARIANTS[target]:
                ip = io_params.get(mid) if variant == 'with' else None
                if variant == 'with' and ip is None:
                    continue
                p_go = predict_pgo(dec_n, target, variant,
                                    grid=DEFAULT_GRID_91, io_params=ip)
                yield (mid, arch, variant, p_go, ch_n, or_n)


def run_predictions(run_name, results_root, io_path,
                     target_slugs=None) -> List[dict]:
    """Compute choice-prediction metrics for every
    (target, arch, variant, mouse). Returns a list of row dicts and the
    per-(target,arch,variant) pooled curves for plotting are recomputed
    by the plot functions from the same bundles, so this also stashes
    the raw bundles on each row under '_bundle'."""
    if target_slugs is None:
        target_slugs = DEFAULT_TARGET_SLUGS
    io_params = load_io_stage2_params(io_path)
    if not io_params:
        print("[warn] no IO stage-2 params loaded — 'with' variant will "
               "be skipped for all mice")

    rows: List[dict] = []
    for target, slug in target_slugs.items():
        for (mid, arch, variant, p_go, choice, orient) in _per_mouse_predictions(
                run_name, results_root, target, slug, io_params):
            m = choice_metrics(p_go, choice)
            rows.append({
                'target': target,
                'slug': slug,
                'arch': arch,
                'variant': variant,
                'mouse_id': mid,
                'n_trials': int(choice.size),
                'auroc': m['auroc'],
                'accuracy': m['accuracy'],
                'nll': m['nll'],
                '_p_go': p_go,
                '_choice': choice,
                '_orient': orient,
            })
            print(f"  {target:<2} {slug:<28} {arch:<5} {variant:<8} "
                   f"mouse_{mid}: AUROC={m['auroc']:.3f}  "
                   f"acc={m['accuracy']:.3f}  NLL={m['nll']:.3f}")
    return rows


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _set_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 1.2, 'font.size': 10,
    })


def _binned_curve(x, y, edges):
    """Mean of y per x-bin; returns (centers, means, sems)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    means = np.full(len(centers), np.nan)
    sems = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        v = y[(idx == b) & np.isfinite(y)]
        if v.size:
            means[b] = np.mean(v)
            sems[b] = (np.std(v, ddof=1) / np.sqrt(v.size)
                        if v.size > 1 else 0.0)
    return centers, means, sems


def plot_psychometric(rows: List[dict], target: str, out_path: Path,
                       n_orient_bins=10):
    """Psychometric curves for one target: P(Go) vs stimulus orientation,
    overlaying the animal's actual choice with the decoder-predicted
    P(Go) for each variant. Spatial and temporal in separate panels.
    Trials pooled across mice."""
    import matplotlib.pyplot as plt
    _set_style()
    sub = [r for r in rows if r['target'] == target]
    if not sub:
        return
    edges = np.linspace(0, 90, n_orient_bins + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    variant_style = {
        'with':    dict(color='seagreen', marker='s',
                          label='Decoder P(Go) — with IO stage-2'),
        'without': dict(color='steelblue', marker='o',
                          label='Decoder P(Go) — plain sigmoid'),
    }
    for ax, arch in zip(axes, REAL_ARCHES):
        arch_rows = [r for r in sub if r['arch'] == arch]
        if not arch_rows:
            ax.set_visible(False)
            continue
        # Animal psychometric — actual choice, pooled (same across variants).
        orient = np.concatenate([r['_orient'] for r in arch_rows])
        choice = np.concatenate([r['_choice'] for r in arch_rows])
        c, m, s = _binned_curve(orient, choice, edges)
        ax.errorbar(c, m, yerr=s, color='black', marker='D', lw=2,
                     capsize=3, label='Animal actual choice', zorder=5)
        # Decoder-predicted psychometric per variant.
        for variant in ('with', 'without'):
            vr = [r for r in arch_rows if r['variant'] == variant]
            if not vr:
                continue
            o = np.concatenate([r['_orient'] for r in vr])
            p = np.concatenate([r['_p_go'] for r in vr])
            c, m, s = _binned_curve(o, p, edges)
            st = variant_style[variant]
            ax.errorbar(c, m, yerr=s, color=st['color'], marker=st['marker'],
                         capsize=3, alpha=0.9, label=st['label'])
        ax.axvline(45, color='red', linestyle='--', lw=1, alpha=0.6,
                    label='Go/NoGo boundary')
        ax.set_xlabel('Stimulus orientation (deg)')
        ax.set_title(f'{arch}')
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel('P(Go)')
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    fig.suptitle(f'Choice psychometrics — target {target}: '
                  f'animal vs decoder-predicted', fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_metric_bars(rows: List[dict], metric: str, out_path: Path):
    """Spatial-vs-temporal bar plot of one choice metric, grouped by
    (target, variant). Per-mouse dots overlaid, mean ± SEM."""
    import matplotlib.pyplot as plt
    _set_style()
    # (target, variant) groups in a stable display order.
    groups = []
    for target in ('Q', 'L', 'd'):
        for variant in TARGET_VARIANTS[target]:
            if any(r['target'] == target and r['variant'] == variant
                    for r in rows):
                groups.append((target, variant))
    if not groups:
        return
    arch_color = {'spat': 'darkorange', 'temp': 'steelblue'}
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(groups)), 5))
    for i, arch in enumerate(REAL_ARCHES):
        offset = (i - 0.5) * width
        means, sems = [], []
        for (target, variant) in groups:
            vals = [r[metric] for r in rows
                     if r['target'] == target and r['variant'] == variant
                     and r['arch'] == arch and np.isfinite(r[metric])]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                              if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan); sems.append(np.nan)
        ax.bar(x + offset, means, width=width, yerr=sems, capsize=4,
                color=arch_color[arch], edgecolor='black', alpha=0.85,
                label=f'{arch}', zorder=2)
        # Per-mouse dots.
        for gi, (target, variant) in enumerate(groups):
            vals = [r[metric] for r in rows
                     if r['target'] == target and r['variant'] == variant
                     and r['arch'] == arch and np.isfinite(r[metric])]
            jit = np.random.RandomState(gi * 10 + i).uniform(-0.06, 0.06, len(vals))
            ax.scatter([x[gi] + offset] * len(vals) + jit, vals,
                        color='black', s=20, zorder=4, alpha=0.65,
                        edgecolor='white', linewidth=0.4)
    if metric == 'auroc':
        ax.axhline(0.5, color='red', linestyle='--', lw=1.3, alpha=0.8,
                    label='Chance (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{v}" for (t, v) in groups], fontsize=9)
    ylabels = {'auroc': 'AUROC', 'accuracy': 'Accuracy (0.5 threshold)',
                'nll': 'NLL (lower = better)'}
    ax.set_ylabel(ylabels.get(metric, metric))
    ax.set_title(f'Choice prediction — {ylabels.get(metric, metric)}: '
                  f'spatial vs temporal, by target and variant')
    ax.legend(loc='best', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_summary_csv(rows: List[dict], out_path: Path):
    """Tidy CSV — one row per (target, arch, variant, mouse). The bulky
    per-trial arrays ('_p_go' etc.) are dropped."""
    fields = ['target', 'slug', 'arch', 'variant', 'mouse_id', 'n_trials',
              'auroc', 'accuracy', 'nll']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"Wrote {out_path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(run_name: str = 'clean_2026_05_19',
         results_root: str = 'results',
         io_path: Optional[str] = None,
         out_dir: str = 'figures/choice_prediction',
         target_slugs: Optional[Dict[str, str]] = None):
    """End-to-end: predict choices, score, write CSV + all plots.

    Parameters
    ----------
    run_name : str
        Training RUN_NAME under ``results_root``.
    results_root : str
        Results tree root. Default ``'results'``.
    io_path : str, optional
        Path to ``IOResults.mat``. Defaults to ``<repo>/data/IOResults.mat``.
    out_dir : str
        Output directory for the CSV and SVGs.
    target_slugs : dict, optional
        Override the {target: slug} mapping (e.g. to use the residual
        basis for Q/L).
    """
    if io_path is None:
        io_path = str(Path(__file__).resolve().parent.parent
                       / 'data' / 'IOResults.mat')
    out = Path(out_dir)

    print(f"Choice prediction — run {run_name}")
    print(f"IO stage-2 params from {io_path}")
    rows = run_predictions(run_name, results_root, io_path,
                            target_slugs=target_slugs)
    if not rows:
        print("No predictions produced. Has training completed?")
        return

    write_summary_csv(rows, out / 'choice_prediction_summary.csv')

    for target in ('Q', 'L', 'd'):
        plot_psychometric(rows, target, out / f'psychometric_{target}.svg')
    for metric in ('auroc', 'accuracy', 'nll'):
        plot_metric_bars(rows, metric, out / f'choice_metric_{metric}.svg')

    # Headline.
    print()
    for variant in ('without', 'with'):
        for arch in REAL_ARCHES:
            aurocs = [r['auroc'] for r in rows
                       if r['variant'] == variant and r['arch'] == arch
                       and np.isfinite(r['auroc'])]
            if aurocs:
                print(f"  {variant:<8} {arch:<5} mean AUROC = "
                       f"{np.mean(aurocs):.3f}  (n={len(aurocs)})")
    print(f"\nDone. Outputs under {out.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default='clean_2026_05_19')
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--io-path', default=None)
    parser.add_argument('--out-dir', default='figures/choice_prediction')
    args = parser.parse_args()
    main(run_name=args.run_name, results_root=args.results_root,
         io_path=args.io_path, out_dir=args.out_dir)
