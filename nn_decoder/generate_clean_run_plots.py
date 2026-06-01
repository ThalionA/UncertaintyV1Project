# -*- coding: utf-8 -*-
"""Comprehensive decoder-performance plots for a clean run tree.

Analogous to ``generate_all_fixed_plots.py`` but reads the nested
``results/<run_name>/<slug>/<split>.mat`` tree written by
``training/run.py`` (via ``decoder_plotting_utils.load_run_tree``)
rather than the legacy flat-named files.

For each PCA-loss orientation cell (Q, L — the 91-D posterior decoders)
it produces the full performance breakdown:

  * normalised performance bars (spatial vs temporal vs temporal-bins)
  * per-mouse performance bars with within-mouse stats
  * accuracy across orientation
  * ambiguity heatmaps (orientation x dispersion), per split
  * temporal dynamics — per-bin decoding trajectory, per split
  * performance vs target certainty (loss binned by target entropy)
  * raw posterior examples and per-condition averages

The orientation-breakdown plots use ``calc_pca_dist`` and are only
meaningful for the 91-D PCA-loss cells, so d (2-D MSE) and stim_cat (CE)
are excluded from those — their headline performance is covered by
``plot_post_fix_performance.py``. All cells, however, are included in
the cross-target comparison, which self-normalises each target to its
own shuffle and falls back to MSE distance for low-dim targets.

Output tree::

    figures/clean_run_plots/
      <slug>/                 # one dir per PCA-loss cell
        1_Normalized_Performance_Bars_Paired_Stats.svg
        3_Orientation_Performance.svg
        4_Temporal_Dynamics_<split>.svg
        4B_Performance_vs_Certainty_<split>.svg
        ...
      Multi_Target_Comparison_<split>.svg

Spyder::

    from generate_clean_run_plots import generate_all
    generate_all(run_name='production_full_targets_alltrials_v1')

CLI::

    python generate_clean_run_plots.py --run-name production_full_targets_alltrials_v1
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
import decoder_plotting_utils as dpu  # noqa: E402


SPLITS = ['stratified_balanced', 'generalize_contrast', 'generalize_dispersion']

# Friendly display name per slug prefix, used as the cross-target
# comparison legend label.
PREFIX_DISPLAY = {
    'Q': 'Perception (Q)',
    'L': 'Likelihood (L)',
    'd': 'Decision (d)',
    'stim': 'Stim category',
}


def _slug_prefix(slug: str) -> str:
    return slug.split('_', 1)[0]


def _discover_slugs(run_dir: Path):
    """Sorted slug subdirectories of a run dir (skips non-dirs and any
    stray 'checkpoints' folder that lives one level deeper anyway)."""
    if not run_dir.is_dir():
        return []
    return sorted(d.name for d in run_dir.iterdir()
                   if d.is_dir() and d.name != 'checkpoints')


def _loss_func(results: dict) -> str:
    """Pull custom_loss_func from the first split's config."""
    for mat in results.values():
        cfg = mat.get('config', {})
        return cfg.get('custom_loss_func') or cfg.get('loss_func') or ''
    return ''


def generate_all(run_name: str = 'production_full_targets_alltrials_v1',
                  results_root: str | None = None,
                  out_root: str = 'figures/clean_run_plots'):
    """Build the full plot suite for one run tree.

    Parameters
    ----------
    run_name : str
        Subdirectory under ``results_root`` (the training RUN_NAME).
    results_root : str
        Root of the results tree. Default ``'results'``.
    out_root : str
        Root directory for the figure tree.
    """
    # Default to paths.RESULTS so the script works irrespective of the
    # kernel/shell cwd. The bare relative 'results' default tripped over
    # Spyder runs whose kernel cwd was the project root, not nn_decoder/.
    if results_root is None:
        results_root = str(paths.RESULTS)
    run_dir = Path(results_root) / run_name
    slugs = _discover_slugs(run_dir)
    if not slugs:
        print(f"No slug directories under {run_dir}. Has training run?")
        return

    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir}")
    print(f"Slugs: {slugs}")

    # Pass 1 — per-slug breakdown. Each slug's results dict is loaded,
    # used, then freed before the next slug so peak memory stays bounded
    # to roughly one slug (the .mat tree + the 558 MB VR export load
    # behind the per-mouse trial cache otherwise stack up and OOM).
    for slug in slugs:
        results = dpu.load_run_tree(run_name, slug, splits=SPLITS,
                                      results_root=results_root)
        if not results:
            print(f"  [skip] {slug}: no .mat files loaded")
            continue
        loss_func = _loss_func(results)

        # The orientation-grid breakdown is PCA-loss specific (Q, L).
        if loss_func != 'PCA':
            print(f"  {slug}: loss={loss_func} — cross-target comparison "
                   f"only (breakdown is PCA-loss specific)")
            del results
            gc.collect()
            continue

        out_dir = out_root_path / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {slug}: full breakdown -> {out_dir}")

        # Every loss-based plot is produced three times under different
        # normalisations:
        #   * 'shuffle'  (the historical headline, no suffix) — each
        #                mouse's loss / its own shuffle-trained
        #                decoder's loss; 1.0 = chance.
        #   * 'variance' (_varnorm suffix) — each mouse's loss / its
        #                own variance baseline (marginal-mean loss,
        #                see dpu.variance_baseline). Reduces
        #                inter-mouse scale variation so the bands
        #                across mice are comparable.
        #   * 'raw'      (_raw suffix) — un-normalised PCA loss.
        #                Surfaces absolute spatial-vs-temporal
        #                differences that normalisation can mask.
        for normalize in ('shuffle', 'variance', 'raw'):
            # Aggregate (across-mouse) and per-mouse performance.
            dpu.plot_normalized_performance_with_lines(
                results, SPLITS, out_dir=str(out_dir), normalize=normalize)
            dpu.plot_per_mouse_performance_with_stats(
                results, SPLITS, out_dir=str(out_dir), normalize=normalize)
            # Accuracy across the orientation grid.
            dpu.plot_orientation_performance(
                results, SPLITS, out_dir=str(out_dir), normalize=normalize)
            # Performance binned by target certainty (target-entropy).
            dpu.plot_performance_vs_certainty(
                results, SPLITS, out_dir=str(out_dir), normalize=normalize)
            # Per-split: ambiguity heatmaps + temporal dynamics.
            for split in SPLITS:
                if split in results:
                    dpu.plot_ambiguity_heatmaps(
                        results, split=split, out_dir=str(out_dir), normalize=normalize)
                    dpu.plot_temporal_dynamics(
                        results, split=split, out_dir=str(out_dir), normalize=normalize)

        # Example raw posteriors + per-condition averages (stratified split).
        dpu.plot_posterior_examples_and_averages(
            results, splits=['stratified_balanced'], out_dir=str(out_dir),
        )

        # Within-mouse stats — printed to console.
        dpu.calculate_within_mouse_stats(results, SPLITS)

        # Belt-and-suspenders: close any figure a plot helper left open
        # so they don't accumulate across slugs.
        import matplotlib.pyplot as _plt
        _plt.close('all')
        del results
        gc.collect()

    # Pass 2 — cross-target comparison. plot_multi_target needs every
    # cell's results at once; reload them here (after the breakdown loop
    # has freed its memory). It self-normalises each target to its own
    # shuffle and MSE-falls-back for low-dim targets, so mixing
    # PCA / MSE / CE cells is safe.
    labelled = {}
    for slug in slugs:
        results = dpu.load_run_tree(run_name, slug, splits=SPLITS,
                                      results_root=results_root)
        if not results:
            continue
        disp = PREFIX_DISPLAY.get(_slug_prefix(slug), slug)
        tail = slug.split('_', 1)[1] if '_' in slug else ''
        labelled[f"{disp}\n{tail}"] = results
    if len(labelled) >= 2:
        dpu.plot_multi_target_comparison(labelled, SPLITS, out_dir=str(out_root_path))
        print(f"  cross-target comparison -> {out_root_path}")
    del labelled
    gc.collect()

    print(f"\nDone. Figures under {out_root_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-name', default='production_full_targets_alltrials_v1')
    parser.add_argument('--results-root', default=None,
                         help='Root of the results tree. Defaults to '
                              'nn_decoder/results (via paths.RESULTS).')
    parser.add_argument('--out-root', default='figures/clean_run_plots')
    args = parser.parse_args()
    generate_all(run_name=args.run_name,
                  results_root=args.results_root,
                  out_root=args.out_root)
