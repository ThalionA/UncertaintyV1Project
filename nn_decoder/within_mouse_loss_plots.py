# -*- coding: utf-8 -*-
"""Full per-cell performance breakdown, per loss — the figures
``generate_clean_run_plots.py`` makes for PCA cells, but for EVERY loss.

#3 from the 2026-06 meeting. ``generate_clean_run_plots`` gates its full
breakdown behind ``loss_func == 'PCA'``, so CE/KL/JS/Wasserstein cells never get
it — even though every plot scores with ``calc_pca_dist`` and only needs the
saved ``pcs``/``explained_var``, which all 91-D cells have. This driver calls
the SAME extracted breakdown (``generate_clean_run_plots.breakdown_one_cell``)
once per loss, so the figure set is identical to the PCA cells — no reinvention,
one breakdown implementation shared by both drivers.

Per loss it emits the whole suite under
``figures/loss_sweep_plots/<run>/within_mouse/<loss>/`` (×3 normalisations:
shuffle / variance / raw):
    1_Normalized_Performance_Bars_Paired_Stats.svg     within-mouse paired stats
    1b_PerMouse_Performance_<split>.svg
    Orientation performance, Performance-vs-certainty,
    Ambiguity heatmaps (per split), Temporal dynamics (per split),
    Posterior examples + per-condition averages
and prints the within-mouse stats to the console.

Scoring is PCA distance (the consistent cross-loss yardstick). For each loss's
decoder scored under its OWN training metric with paired stats, see
``cross_loss_eval.py`` (12_spat_temp_paired_stats.csv).

Usage
-----
    python within_mouse_loss_plots.py --run-name loss_comparison_v1
    python within_mouse_loss_plots.py --run-name loss_comparison_v1 \
        --target L --window full --bin 50 --splits stratified_balanced \
        generalize_contrast generalize_dispersion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import decoder_plotting_utils as dpu
from generate_clean_run_plots import breakdown_one_cell


LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')


def _slug(target: str, loss: str, window: str, bin_ms: int) -> str:
    base = f"{target}_{loss}_{window}_{bin_ms}ms"
    return f"{base}_all" if loss == 'PCA' else base


def main(run_name, target='Q', window='half', bin_ms=100,
         splits=('stratified_balanced',), losses=LOSSES,
         out_root='figures/loss_sweep_plots'):
    splits = list(splits)
    # Per-cell dir (split-agnostic: the breakdown spans all requested splits).
    cell = f'{target}_{window}_{bin_ms}ms'
    out_base = Path(out_root) / run_name / cell / 'within_mouse'
    print(f"Per-loss full breakdown: {run_name} | cell {cell}")
    print(f"  cell: {target} {window} {bin_ms}ms | splits={splits}")
    for loss in losses:
        slug = _slug(target, loss, window, bin_ms)
        results = dpu.load_run_tree(run_name, slug, splits=splits)
        present = [s for s in splits if results and s in results and results[s]]
        if not present:
            print(f"  [skip] {loss}: no data at {slug} for {splits}")
            continue
        out_dir = out_base / loss
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            breakdown_one_cell(results, out_dir, splits=present)
        except Exception as exc:
            print(f"  [!] {loss}: {exc!r}")
        print(f"  {loss}: -> {out_dir}")
    print(f"Done. {out_base.resolve()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--splits', nargs='+', default=['stratified_balanced'])
    ap.add_argument('--losses', nargs='+', default=list(LOSSES))
    a = ap.parse_args()
    main(run_name=a.run_name, target=a.target, window=a.window,
         bin_ms=a.bin_ms, splits=tuple(a.splits), losses=tuple(a.losses))
