# -*- coding: utf-8 -*-
"""Round-trip sanity check for trained NN-decoder checkpoints.

For each saved checkpoint (``.pt`` under
``<run_dir>/<slug>/checkpoints/mouse_<mid>_<split>.pt``):

  1. Rebuild the model from ``model_params`` + ``state_dict``.
  2. Forward the saved ``X_test`` through it.
  3. Compare to the saved ``pred_probs`` — bit-wise max/mean |Δ| (the
     "saved weights actually reproduce the reported outputs" invariant).
  4. Compute fit-loss with the saved ``pred_probs`` as BOTH the
     prediction AND the target. By construction this lands at the fp32
     floor — the "model perfectly reproduces its own outputs" identity
     check.
  5. Compute the entropy regulariser
     ``entropy_lambda * H(pred_probs_new)`` separately, as a diagnostic
     (the contamination story from
     ``audit/AUDIT_loss_consumers.md``).

Then compares the identity-check fit-loss against:

  * **Real-data fit-loss**: the model's actual held-out loss on the IO
    targets. Read from the new ``fit_loss`` field in the .mat (or
    recomputed via ``decoder_plotting_utils.calc_fit_loss`` for older
    files that only have ``KLs`` + ``Dist``).
  * **From-scratch recovery loss**: how well a freshly-initialised model
    can recover the base model's predictions when retrained on
    ``(X, base_pred)``. Read from ``paths.recovery_cache(target)`` if
    ``run_fixed_recovery.py`` has been run for this target.

Emits one CSV and three SVG plots under
``<out_dir>/`` (default ``figures/recovery_probe/sanity_check/``):

  * ``sanity_check_summary.csv`` — one row per
    ``(cell, mouse, split, arch)``.
  * ``identity_fit_loss_strip.svg`` — per-cell strip plot; should hug 0.
  * ``compare_identity_recovery_realdata.svg`` — three-way bar
    comparison per ``(cell, arch)``.
  * ``pred_probs_diff_hist.svg`` — distribution of
    ``|pred_new − pred_saved|`` across all trials/cells.

CLI::

    python recovery_sanity_check.py --run-dir results/clean_2026_05_19

Spyder::

    from recovery_sanity_check import main
    main(run_dir='results/clean_2026_05_19')
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Local imports — keep relative-safe even when invoked from a different cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
from nn_classifier import (  # noqa: E402
    SimpleFlexibleNNClassifier,
    custom_loss_all_H,
    get_model_probabilities,
)
import decoder_plotting_utils as dpu  # noqa: E402
from figsave import save_fig  # noqa: E402


# ----------------------------------------------------------------------
# Cell → recovery-cache target name
# ----------------------------------------------------------------------
#
# `paths.recovery_cache(target)` -> recovery_cache_fixed_<target>.npy.
# The `target` name in the recovery cache is the long-form one used by
# `run_animal_decoder` (perception / likelihood / decision / stim_cat),
# not the short slug prefix (Q / L / d / stim_cat). This dict maps from
# the slug prefix to the long-form name.

SLUG_PREFIX_TO_RECOVERY_TARGET = {
    'Q':         'perception',
    'L':         'likelihood',
    'd':         'decision',
    'stim_cat':  'stim_cat',
}

REAL_ARCHES = ('spat', 'temp')


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def discover_checkpoints(run_dir: Path) -> List[Tuple[str, str, str, Path]]:
    """Walk ``<run_dir>/<slug>/checkpoints/mouse_<mid>_<split>.pt``.

    Returns a list of ``(slug, mouse_str, split, ckpt_path)`` tuples,
    sorted for stable output ordering.
    """
    out = []
    if not run_dir.exists():
        return out
    for slug_dir in sorted(run_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        ckpt_dir = slug_dir / 'checkpoints'
        if not ckpt_dir.is_dir():
            continue
        for pt in sorted(ckpt_dir.glob('mouse_*_*.pt')):
            stem = pt.stem  # mouse_<mid>_<split>
            parts = stem.split('_', 2)
            if len(parts) != 3 or parts[0] != 'mouse':
                continue
            mid = parts[1]
            split = parts[2]
            out.append((slug_dir.name, mid, split, pt))
    return out


def _slug_prefix(slug: str) -> str:
    """Pull the target prefix off a slug like 'Q_PCA_half_100ms_condmean'."""
    return slug.split('_', 1)[0]


def _mean_fit_loss_for_arch(animal_res: dict, arch: str,
                              loss_func: str) -> Optional[float]:
    """Per-mouse mean fit-loss for one arch key, from an animal-results
    dict. ``arch`` may be a real arch ('spat'/'temp') or a shuffle
    control ('spat_shf'/'temp_shf').

    Reads the new ``fit_loss`` field if present; falls back to
    recomputing from ``Dist[arch]['target']``/``['decoded']`` via
    ``calc_fit_loss`` for older outputs (pre-2026-05-19) that only have
    the contaminated ``KLs``.
    """
    fit_dict = animal_res.get('fit_loss')
    if isinstance(fit_dict, dict) and arch in fit_dict:
        arr = np.asarray(fit_dict[arch], dtype=float).reshape(-1)
        if arr.size:
            return float(np.nanmean(arr))

    dist = animal_res.get('Dist')
    if isinstance(dist, dict) and arch in dist:
        per_trial = dpu.calc_fit_loss(
            dist[arch], loss_func,
            pcs=dist.get('pcs'), evar=dist.get('explained_var'),
        )
        if per_trial is not None and len(per_trial):
            return float(np.nanmean(per_trial))
    return None


def load_real_data_fit_loss(slug_dir: Path, split: str, mid: str,
                              arch: str, loss_func: str
                              ) -> Tuple[Optional[float], Optional[float]]:
    """Per-mouse mean fit-loss on the real IO targets, plus the matching
    shuffle-control loss.

    Returns ``(fit, shuffle)`` where ``fit`` is the loss for ``arch``
    ('spat'/'temp') and ``shuffle`` is the loss for the matching
    ``<arch>_shf`` control. Either may be ``None`` if absent.
    """
    import scipy.io as sio
    mat_path = slug_dir / f'{split}.mat'
    if not mat_path.exists():
        return None, None
    try:
        mat = sio.loadmat(str(mat_path), simplify_cells=True)
    except Exception as exc:
        print(f"[warn] could not load {mat_path}: {exc}")
        return None, None
    mouse_key = f'mouse_{mid}'
    if mouse_key not in mat.get('results', {}):
        return None, None
    v = mat['results'][mouse_key]
    fit = _mean_fit_loss_for_arch(v, arch, loss_func)
    shf = _mean_fit_loss_for_arch(v, f'{arch}_shf', loss_func)
    return fit, shf


def load_recovery_loss(slug: str, mid: str, arch: str,
                        loss_func: str
                        ) -> Tuple[Optional[float], Optional[float]]:
    """Per-mouse mean from-scratch recovery fit-loss, plus its shuffle.

    Reads ``paths.recovery_cache(<long_target_name>)`` produced by
    ``run_fixed_recovery.py``. Returns ``(fit, shuffle)`` — both
    ``None`` if the cache doesn't exist (e.g. recovery hasn't been run
    yet for this target).

    The cache is structured as
    ``{'base_config': cfg, 'spat': {mouse_X: res}, 'temp': {mouse_X: res}}``
    where each ``res`` is the full animal_results dict produced by
    ``run_animal_decoder`` when trained with
    ``target_source = 'recovery_<arch>'`` — i.e. fresh model fit to
    the base model's ``full_decoded`` predictions as targets.
    """
    prefix = _slug_prefix(slug)
    long_target = SLUG_PREFIX_TO_RECOVERY_TARGET.get(prefix)
    if long_target is None:
        return None, None
    cache_path = paths.recovery_cache(long_target)
    if not cache_path.exists():
        return None, None
    try:
        rec = np.load(str(cache_path), allow_pickle=True).item()
    except Exception as exc:
        print(f"[warn] could not load recovery cache {cache_path}: {exc}")
        return None, None

    branch = rec.get(arch)
    if not isinstance(branch, dict):
        return None, None
    mouse_key = f'mouse_{mid}'
    if mouse_key not in branch:
        return None, None
    animal_res = branch[mouse_key]
    fit = _mean_fit_loss_for_arch(animal_res, arch, loss_func)
    shf = _mean_fit_loss_for_arch(animal_res, f'{arch}_shf', loss_func)
    return fit, shf


# ----------------------------------------------------------------------
# Per-checkpoint round-trip
# ----------------------------------------------------------------------

def round_trip_one_checkpoint(ckpt: dict) -> Dict[str, np.ndarray]:
    """Reload the model, replay forward pass on saved X_test, compute
    diagnostics. Returns per-trial arrays:

      * ``identity_fit_loss`` — loss(pred_new, pred_saved) using the
        cell's training loss. For MSE/PCA/JS/KL this is ~0 on a clean
        round-trip; for cross-entropy it is ~H(pred) (CE(p,p) = H(p)).
      * ``identity_baseline`` — loss(pred_saved, pred_saved). 0 for
        divergence losses, H(pred) for cross-entropy.
      * ``identity_residual`` — ``identity_fit_loss − identity_baseline``.
        Equals KL(pred_saved ‖ pred_new); 0 iff the round-trip is exact,
        for EVERY loss type. This is the headline drift metric.
      * ``entropy_penalty`` — λ·H(pred_new). Always 0 for ppc.
      * ``max_diff`` — max |pred_new − pred_saved| across all trial/bin/cat.
      * ``all_diffs`` — flat array of |pred_new − pred_saved| values
        (used for the histogram plot).
    """
    mp = ckpt['model_params']
    model = SimpleFlexibleNNClassifier(
        input_size=mp['input_size'],
        hidden_sizes=mp['hidden_sizes'],
        output_size=mp['output_size'],
        activation=mp['activation_function'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    X_test = ckpt['X_test']           # (n_trials, T, n_neurons)
    pred_saved = ckpt['pred_probs']   # (n_trials, 1 or T, n_cats)
    pcs = ckpt.get('pcs')
    evar = ckpt.get('explained_var')
    loss_func = ckpt['loss_func']
    entropy_lambda = float(ckpt['entropy_lambda'])
    model_type = ckpt['model_type']

    n_trials = X_test.shape[0]
    identity_fit_losses = np.empty(n_trials, dtype=float)
    identity_baselines = np.empty(n_trials, dtype=float)
    identity_residuals = np.empty(n_trials, dtype=float)
    entropy_penalties = np.empty(n_trials, dtype=float)
    max_diffs = np.empty(n_trials, dtype=float)
    all_diffs_list = []

    with torch.no_grad():
        for i in range(n_trials):
            x = X_test[i]
            saved = pred_saved[i]
            new = get_model_probabilities(model, x, model_type)
            diff = (new - saved).abs()
            max_diffs[i] = float(diff.max().item())
            all_diffs_list.append(diff.flatten().cpu().numpy())

            # fit  : loss of the round-tripped prediction vs the saved
            #        prediction (used as target). Same path as
            #        evaluate_model_entropy.
            # base : loss of the saved prediction vs ITSELF. For
            #        divergence losses (MSE/PCA/JS/KL) this is 0; for
            #        cross-entropy it is H(pred_saved) — CE(p,p) = H(p),
            #        NOT 0.
            # The round-trip drift is the RESIDUAL fit - base, which
            # equals KL(pred_saved || pred_new) and is therefore 0 iff
            # pred_new == pred_saved, for EVERY loss type. The raw `fit`
            # alone would falsely flag every CE-trained cell as a
            # failure even on a bit-perfect round-trip.
            _, fit, penalty = custom_loss_all_H(
                new, saved, entropy_lambda, model_type, pcs, evar, loss_func,
            )
            _, fit_base, _ = custom_loss_all_H(
                saved, saved, entropy_lambda, model_type, pcs, evar, loss_func,
            )
            identity_fit_losses[i] = float(fit.item())
            identity_baselines[i] = float(fit_base.item())
            identity_residuals[i] = float(fit.item()) - float(fit_base.item())
            entropy_penalties[i] = float(penalty.item())

    return {
        'identity_fit_loss': identity_fit_losses,
        'identity_baseline': identity_baselines,
        'identity_residual': identity_residuals,
        'entropy_penalty': entropy_penalties,
        'max_diff': max_diffs,
        'all_diffs': np.concatenate(all_diffs_list) if all_diffs_list else np.array([]),
    }


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


def plot_identity_strip(rows: List[dict], out_path: Path):
    """One strip per (cell, arch) — each dot is a (mouse, split, trial-mean)
    of the identity RESIDUAL (loss(new,saved) − loss(saved,saved)).
    Should sit tight against 0 for every loss type on a clean round-trip
    (the raw identity fit-loss would not, for cross-entropy cells)."""
    import matplotlib.pyplot as plt
    _set_style()
    cells = sorted({r['cell'] for r in rows})
    arches = REAL_ARCHES
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(cells) * len(arches)), 5))

    x_positions = []
    labels = []
    color = {'spat': 'darkorange', 'temp': 'steelblue'}
    for i, cell in enumerate(cells):
        for j, arch in enumerate(arches):
            x = i * (len(arches) + 1) + j
            x_positions.append(x)
            labels.append(f"{cell}\n{arch}")
            vals = [r['identity_residual_mean'] for r in rows
                    if r['cell'] == cell and r['arch'] == arch]
            if vals:
                jitter = np.random.RandomState(i * 10 + j).uniform(-0.15, 0.15, size=len(vals))
                ax.scatter([x] * len(vals) + jitter, vals,
                            color=color[arch], s=40, alpha=0.7, edgecolor='black',
                            linewidth=0.5)
                ax.scatter([x], [np.mean(vals)],
                            color='black', marker='_', s=400, linewidth=2.5, zorder=3)

    ax.axhline(0, color='red', linestyle='--', lw=1, alpha=0.6, label='Identity target (0)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Identity residual\nloss(new,saved) − loss(saved,saved)')
    ax.set_title('Round-trip sanity: identity residual (= KL of round-trip drift)\n'
                  '(should hug 0; dots = per (mouse, split), bar = mean)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_path.parent, out_path.stem)
    print(f"Wrote {out_path}")


def plot_three_way_compare(rows: List[dict], out_path: Path):
    """Side-by-side bars per (cell, arch): real-data fit-loss vs
    from-scratch recovery loss vs identity-check loss. The headline plot
    — proves the model can exactly reproduce its own outputs (identity ~0)
    while from-scratch recovery has some residual and real-data loss is
    the largest of the three."""
    import matplotlib.pyplot as plt
    _set_style()
    cells = sorted({r['cell'] for r in rows})
    arches = REAL_ARCHES
    n_groups = len(cells) * len(arches)

    real_means, recov_means, ident_means = [], [], []
    real_sems, recov_sems, ident_sems = [], [], []
    group_labels = []
    for cell in cells:
        for arch in arches:
            subset = [r for r in rows if r['cell'] == cell and r['arch'] == arch]
            def _agg(key):
                vs = [r[key] for r in subset if r.get(key) is not None and np.isfinite(r[key])]
                if not vs:
                    return np.nan, 0.0
                return float(np.mean(vs)), (float(np.std(vs, ddof=1) / np.sqrt(len(vs)))
                                              if len(vs) > 1 else 0.0)
            rm, rs = _agg('real_data_fit_loss')
            cm, cs = _agg('recovery_fit_loss')
            # Identity bar uses the RESIDUAL — comparable across loss
            # types and ~0 on a clean round-trip (the raw identity
            # fit-loss would tower for CE cells where it equals H(pred)).
            im, isem = _agg('identity_residual_mean')
            real_means.append(rm);  real_sems.append(rs)
            recov_means.append(cm); recov_sems.append(cs)
            ident_means.append(im); ident_sems.append(isem)
            group_labels.append(f"{cell}\n{arch}")

    x = np.arange(n_groups)
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_groups), 6))
    ax.bar(x - width, real_means, width, yerr=real_sems, capsize=3,
            color='steelblue', edgecolor='black', alpha=0.85, label='Real-data fit-loss')
    ax.bar(x,         recov_means, width, yerr=recov_sems, capsize=3,
            color='goldenrod', edgecolor='black', alpha=0.85, label='From-scratch recovery')
    ax.bar(x + width, ident_means, width, yerr=ident_sems, capsize=3,
            color='seagreen', edgecolor='black', alpha=0.85, label='Identity residual (this script)')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean fit-loss (per-mouse mean → grand mean ± SEM)')
    ax.set_title('Fit-loss comparison: real data vs from-scratch recovery vs identity residual\n'
                  '(identity residual ~0; recovery > 0 but < real-data; gaps indicate model capacity vs noise)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_path.parent, out_path.stem)
    print(f"Wrote {out_path}")


def plot_diff_hist(all_diffs: np.ndarray, out_path: Path):
    """Distribution of |pred_new − pred_saved| across every trial/bin/cat.
    Should be tight at fp32 floor (~1e-7) — anything bigger is a smoking
    gun for state_dict/data corruption."""
    import matplotlib.pyplot as plt
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    if all_diffs.size == 0:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                  ha='center', va='center')
    else:
        nonzero = all_diffs[all_diffs > 0]
        if nonzero.size:
            # Log-scale histogram to see fp32 scale clearly.
            ax.hist(np.log10(nonzero + 1e-30), bins=80,
                     color='steelblue', edgecolor='black', alpha=0.85)
            ax.set_xlabel(r'$\log_{10}|p_{\mathrm{new}} - p_{\mathrm{saved}}|$  '
                            '(nonzero entries only)')
        else:
            ax.text(0.5, 0.5,
                      'All |Δ| are exactly 0 — bit-perfect round-trip.',
                      transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.set_ylabel('Count')
    ax.set_title(f'Round-trip drift in pred_probs (n_entries = {all_diffs.size})')
    fig.tight_layout()
    save_fig(fig, out_path.parent, out_path.stem)
    print(f"Wrote {out_path}")


def plot_normalized_compare_qld(rows: List[dict], out_path: Path):
    """Shuffle-normalised real-data vs from-scratch recovery loss, for the
    divergence-loss cells (Q, L, d — i.e. loss_func in {PCA, MSE}) only.

    Cross-entropy cells (stim_cat) are excluded: CE has an irreducible
    floor of H(target) ≈ 1.5-2 nats, so its raw loss is not comparable
    to the PCA/MSE cells whose loss floors at 0. Within the PCA/MSE
    family, each bar is the actual fit-loss divided by that model's own
    shuffle control — i.e. "fraction of chance", lower = better, 1.0 =
    shuffle baseline.

    The identity check is intentionally NOT shown here — it is ~0 by
    construction (see identity_fit_loss_strip.svg) and this figure is
    about comparing the real decoder against the from-scratch recovery
    on a meaningful normalised scale.
    """
    import matplotlib.pyplot as plt
    _set_style()
    # Divergence-loss cells only.
    qld_rows = [r for r in rows if r['loss_func'] in ('PCA', 'MSE')]
    if not qld_rows:
        print("[warn] no PCA/MSE cells; skipping normalised Q/L/d plot")
        return
    cells = sorted({r['cell'] for r in qld_rows})
    arches = REAL_ARCHES
    n_groups = len(cells) * len(arches)

    real_means, recov_means = [], []
    real_sems, recov_sems = [], []
    group_labels = []
    for cell in cells:
        for arch in arches:
            subset = [r for r in qld_rows
                       if r['cell'] == cell and r['arch'] == arch]

            def _agg(key):
                vs = [r[key] for r in subset
                       if r.get(key) is not None and np.isfinite(r[key])]
                if not vs:
                    return np.nan, 0.0
                return (float(np.mean(vs)),
                         float(np.std(vs, ddof=1) / np.sqrt(len(vs)))
                         if len(vs) > 1 else 0.0)

            rm, rs = _agg('real_data_norm')
            cm, cs = _agg('recovery_norm')
            real_means.append(rm);  real_sems.append(rs)
            recov_means.append(cm); recov_sems.append(cs)
            group_labels.append(f"{cell}\n{arch}")

    x = np.arange(n_groups)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_groups), 6))
    ax.bar(x - width / 2, real_means, width, yerr=real_sems, capsize=3,
            color='steelblue', edgecolor='black', alpha=0.85,
            label='Real-data loss / shuffle')
    ax.bar(x + width / 2, recov_means, width, yerr=recov_sems, capsize=3,
            color='goldenrod', edgecolor='black', alpha=0.85,
            label='From-scratch recovery / shuffle')
    ax.axhline(1.0, color='red', linestyle='--', lw=1.5, alpha=0.8,
                label='Shuffle baseline (chance)')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of shuffle loss (lower = better)')
    ax.set_title('Q / L / d cells: real-data vs from-scratch recovery, '
                  'each normalised to its own shuffle\n'
                  '(PCA & MSE cells only — CE excluded; '
                  'identity residual ~0, see strip plot)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_path.parent, out_path.stem)
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _resolve_run_dir(run_dir: Optional[str]) -> Path:
    """If ``run_dir`` is None, pick the newest subdirectory under
    ``paths.RESULTS``. Otherwise use it as-given (absolute or relative
    to cwd)."""
    if run_dir is not None:
        return Path(run_dir)
    candidates = [p for p in paths.RESULTS.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No subdirs under {paths.RESULTS}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main(run_dir: Optional[str] = None,
         out_dir: str = 'figures/recovery_probe/sanity_check'):
    """End-to-end driver. Writes CSV + 3 SVGs to ``out_dir``.

    Parameters
    ----------
    run_dir : str, optional
        Path to ``results/<run_name>``. Defaults to the newest
        subdirectory under ``paths.RESULTS``.
    out_dir : str
        Output directory (relative to cwd or absolute).
    """
    run_path = _resolve_run_dir(run_dir)
    print(f"Sanity-check run: {run_path}")

    discovered = discover_checkpoints(run_path)
    if not discovered:
        print(f"No checkpoints found under {run_path}/*/checkpoints/. "
               "Has training run with the new checkpoint plumbing?")
        return

    rows: List[dict] = []
    all_diffs_concat: List[np.ndarray] = []
    for slug, mid, split, ckpt_path in discovered:
        try:
            bundle = torch.load(str(ckpt_path), map_location='cpu',
                                  weights_only=False)
        except Exception as exc:
            print(f"[warn] could not load {ckpt_path}: {exc}")
            continue
        # Each .pt holds the full Checkpoints bundle: {'spat': <ckpt>,
        # 'temp': <ckpt>}. Iterate the real arches in turn. (Older /
        # hand-built single-arch .pt files — those carrying a
        # 'model_type' key at the top level — are still accepted.)
        if 'model_type' in bundle:
            arch0 = 'spat' if bundle.get('model_type') == 'ppc' else 'temp'
            per_arch = {arch0: bundle}
        else:
            per_arch = {a: bundle[a] for a in REAL_ARCHES if a in bundle}
        if not per_arch:
            print(f"[warn] {ckpt_path} has no recognisable arch entries; skipping")
            continue

        for arch, ckpt in per_arch.items():
            loss_func = ckpt['loss_func']
            diags = round_trip_one_checkpoint(ckpt)
            all_diffs_concat.append(diags['all_diffs'])

            real_loss, real_shf = load_real_data_fit_loss(
                slug_dir=run_path / slug, split=split, mid=mid,
                arch=arch, loss_func=loss_func,
            )
            recov_loss, recov_shf = load_recovery_loss(
                slug=slug, mid=mid, arch=arch, loss_func=loss_func,
            )

            def _norm(num, den):
                """Shuffle-normalised loss (fraction of chance). None if
                either operand is missing or the shuffle is non-positive."""
                if num is None or den is None or not np.isfinite(den) or den <= 0:
                    return None
                return float(num) / float(den)

            rows.append({
                'cell': slug,
                'mouse_id': int(mid),
                'split': split,
                'arch': arch,
                'loss_func': loss_func,
                'entropy_lambda': float(ckpt['entropy_lambda']),
                'n_trials': int(diags['identity_residual'].size),
                # identity_residual is the headline drift metric — 0 for a
                # clean round-trip regardless of loss type. identity_fit_loss
                # and identity_baseline are kept for transparency (for CE
                # cells both ≈ H(pred), and only their difference is ~0).
                'identity_residual_mean':   float(np.nanmean(diags['identity_residual'])),
                'identity_residual_max':    float(np.nanmax(np.abs(diags['identity_residual']))),
                'identity_fit_loss_mean':   float(np.nanmean(diags['identity_fit_loss'])),
                'identity_baseline_mean':   float(np.nanmean(diags['identity_baseline'])),
                'entropy_penalty_mean':     float(np.nanmean(diags['entropy_penalty'])),
                'max_pred_diff':            float(np.nanmax(diags['max_diff'])),
                'real_data_fit_loss':       real_loss,
                'real_data_shuffle':        real_shf,
                'real_data_norm':           _norm(real_loss, real_shf),
                'recovery_fit_loss':        recov_loss,
                'recovery_shuffle':         recov_shf,
                'recovery_norm':            _norm(recov_loss, recov_shf),
            })
            print(f"  {slug}/mouse_{mid}/{split:<24}{arch:<5}"
                   f"residual={rows[-1]['identity_residual_max']:.2e}  "
                   f"(fit={rows[-1]['identity_fit_loss_mean']:.2e} "
                   f"base={rows[-1]['identity_baseline_mean']:.2e})  "
                   f"penalty={rows[-1]['entropy_penalty_mean']:.2e}  "
                   f"max|Δp|={rows[-1]['max_pred_diff']:.2e}  "
                   f"real={real_loss!r}  recov={recov_loss!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / 'sanity_check_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")

    # Plots
    plot_identity_strip(rows, out / 'identity_fit_loss_strip.svg')
    plot_three_way_compare(rows, out / 'compare_identity_recovery_realdata.svg')
    plot_normalized_compare_qld(rows, out / 'compare_QLd_normalized.svg')
    plot_diff_hist(
        np.concatenate(all_diffs_concat) if all_diffs_concat else np.array([]),
        out / 'pred_probs_diff_hist.svg',
    )

    # Headline summary. The verdict is based on identity_residual (drift
    # of the round-trip, 0 for every loss type on a clean reload) and the
    # direct |Δ pred_probs| — NOT identity_fit_loss, which for CE cells is
    # ~H(pred) even on a bit-perfect round-trip and would falsely FAIL.
    residual_vals = np.array([r['identity_residual_max'] for r in rows])
    max_diffs = np.array([r['max_pred_diff'] for r in rows])
    print()
    print(f"Headline: max |identity residual| = {np.nanmax(residual_vals):.2e} "
           f"(across {len(rows)} checkpoints)")
    print(f"          max |Δ pred_probs|      = {np.nanmax(max_diffs):.2e}")
    if np.nanmax(residual_vals) < 1e-4 and np.nanmax(max_diffs) < 1e-4:
        print("PASS: round-trip is clean to fp32 precision.")
    else:
        print("FAIL: round-trip drift exceeds fp32 floor — investigate.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-dir', default=None,
                         help='Path to results/<run_name>. Defaults to the newest '
                              'subdirectory under paths.RESULTS.')
    parser.add_argument('--out-dir',
                         default='figures/recovery_probe/sanity_check',
                         help='Where to write the CSV + SVG plots.')
    args = parser.parse_args()
    main(run_dir=args.run_dir, out_dir=args.out_dir)
