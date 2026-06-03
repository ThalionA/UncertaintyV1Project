# -*- coding: utf-8 -*-
"""Discover every completed cell in a loss-comparison run and plot all four
figure suites for each: the 9-figure loss-sweep suite, the cross-loss skill
matrices + spat/temp stats, the per-loss within-mouse breakdown, and the
per-trial spat-vs-temp scatter/density figures.

A "cell" is a (target, window, bin) triple; each is plotted for every split that
has a merged ``.mat`` on disk. Partial runs are fine — only cells whose ``.mat``
exist are touched, so this can be re-run as the grid fills in.

Usage
-----
    python plot_all_cells.py --run-name loss_comparison_v1
    python plot_all_cells.py --run-name loss_comparison_v1 --targets Q L \
        --skip scatter            # subset / skip a suite
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

# The dpu/seaborn breakdown is chatty on OOD splits (empty per-condition cells →
# nanmean-of-empty RuntimeWarnings; seaborn use_inf_as_na FutureWarnings). They
# are expected and drown the progress log, so silence them here.
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import paths
import plot_loss_sweep
import cross_loss_eval
import within_mouse_loss_plots
import loss_scatter_spat_temp
import plot_loss_spat_temp_comparison

# slug = TARGET_LOSS_WINDOW_BINms[_all]; capture the cell coordinates.
_SLUG_RE = re.compile(
    r'^(?P<target>[A-Za-z]+)_(?P<loss>PCA|CE|KL|JS|Wasserstein|MSE)_'
    r'(?P<window>full|half|last_quarter)_(?P<bin>\d+)ms(?:_all)?$')
SPLITS_ORDER = ('stratified_balanced', 'generalize_contrast',
                'generalize_dispersion')


def discover_cells(run_name, results_root=None):
    """Return ``{(target, window, bin): set(splits)}`` for every slug dir that
    has at least one merged ``<split>.mat``."""
    root = Path(results_root) if results_root else paths.RESULTS
    run_dir = root / run_name
    cells = defaultdict(set)
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        m = _SLUG_RE.match(d.name)
        if not m:
            continue
        splits = {f.stem for f in d.glob('*.mat')}
        if not splits:
            continue
        key = (m['target'], m['window'], int(m['bin']))
        cells[key] |= splits
    return cells


def main(run_name, results_root=None, out_root='figures/loss_sweep_plots',
         targets=None, skip=()):
    cells = discover_cells(run_name, results_root)
    if not cells:
        print(f"No completed cells found under {run_name}.")
        return
    if targets:
        cells = {k: v for k, v in cells.items() if k[0] in set(targets)}
    print(f"Discovered {len(cells)} cell(s) in {run_name}:")
    for (t, w, b), splits in sorted(cells.items()):
        ss = [s for s in SPLITS_ORDER if s in splits] + \
             sorted(splits - set(SPLITS_ORDER))
        print(f"  {t} {w} {b}ms -> splits: {ss}")
    print()

    for (target, window, bin_ms), splits in sorted(cells.items()):
        present = [s for s in SPLITS_ORDER if s in splits] + \
                  sorted(splits - set(SPLITS_ORDER))
        # within-mouse breakdown spans all splits of the cell — run once.
        if 'within_mouse' not in skip:
            within_mouse_loss_plots.main(
                run_name, target=target, window=window, bin_ms=bin_ms,
                splits=tuple(present), out_root=out_root)
        # per-split suites.
        for split in present:
            if 'sweep' not in skip:
                plot_loss_sweep.generate_all(
                    run_name, results_root=results_root, out_root=out_root,
                    target=target, window=window, bin_ms=bin_ms, split=split)
            if 'crossloss' not in skip:
                cross_loss_eval.main(
                    run_name, results_root=results_root, out_root=out_root,
                    target=target, window=window, bin_ms=bin_ms, split=split)
            if 'spat_temp' not in skip:
                plot_loss_spat_temp_comparison.main(
                    run_name, results_root=results_root, out_root=out_root,
                    target=target, window=window, bin_ms=bin_ms, split=split)
            if 'scatter' not in skip:
                loss_scatter_spat_temp.main(
                    run_name, target=target, window=window, bin_ms=bin_ms,
                    split=split, out_root=out_root,
                    results_root=str(results_root) if results_root else 'results')
    print(f"\nAll cells done -> {Path(out_root) / run_name}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    ap.add_argument('--targets', nargs='+', default=None,
                    help='Subset of targets (e.g. Q L). Default: all found.')
    ap.add_argument('--skip', nargs='+', default=[],
                    choices=['sweep', 'crossloss', 'within_mouse', 'scatter',
                             'spat_temp'],
                    help='Figure suites to skip.')
    a = ap.parse_args()
    main(run_name=a.run_name, results_root=a.results_root, out_root=a.out_root,
         targets=a.targets, skip=tuple(a.skip))
