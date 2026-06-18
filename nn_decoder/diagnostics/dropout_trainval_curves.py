# -*- coding: utf-8 -*-
"""Train vs validation loss curves with dropout (2026-06-17 — needs the `monitor_val`
knob, since patience-0 runs otherwise log no val curve).

Reads the `dropval` runs (PCA, patience 0, `--monitor-val`, 200 ep, dropout ∈ {0,…}).
Per arch: mean (over mice) train_total_loss and val_total_loss vs epoch, one colour per
dropout level. The train–val *gap* = overfitting. The point: does dropout shrink the gap
(it regularises capacity/overfitting) — even though we showed it does NOT reduce the
over-sharpening (peakiness)? If so, the over-sharpening is *not* overfitting.

Outputs (PNG+SVG) under figures/peakiness_scatter/:  dropout_trainval_curves.png

Usage:  python diagnostics/dropout_trainval_curves.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402

ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
SLUG = 'Q_PCA_half_100ms_all'


def _drop(name):
    m = re.search(r'drop([0-9p]+)', name)
    return float(m.group(1).replace('p', '.')) if m else 0.0


def _conds(results_root, base='dropval'):
    root = Path(results_root)
    runs = [d.name for d in root.glob(f'{base}*') if d.is_dir() and (d / SLUG).is_dir()]
    return sorted(runs, key=_drop)


def _curves(results_root, run, arch):
    """mean (over mice) train & val total-loss curves; (train, val) or (None, None)."""
    ck_dir = Path(results_root) / run / SLUG / 'checkpoints'
    tr, va = [], []
    for pt in sorted(ck_dir.glob('mouse_*_stratified_balanced.pt')):
        h = torch.load(str(pt), map_location='cpu', weights_only=False)[arch].get('history', {})
        t, v = h.get('train_total_loss'), h.get('val_total_loss')
        if t and v:
            tr.append(np.asarray(t, float)); va.append(np.asarray(v, float))
    if not tr:
        return None, None
    L = min(min(len(a) for a in tr), min(len(a) for a in va))
    return np.mean([a[:L] for a in tr], 0), np.mean([a[:L] for a in va], 0)


def main(results_root, out_root):
    ps.apply()
    conds = _conds(results_root)
    if not conds:
        raise SystemExit('no dropval* runs found (train with --monitor-val first).')
    fig, axes = plt.subplots(1, len(ARCHS), figsize=ps.figsize(len(ARCHS), 1))
    print(f"{'arch':9s} {'dropout':>8s} {'final train':>12s} {'final val':>10s} {'gap':>8s}")
    for j, (arch, alabel) in enumerate(ARCHS):
        ax = axes[j]
        for run in conds:
            p = _drop(run)
            tr, va = _curves(results_root, run, arch)
            if tr is None:
                continue
            col = plt.cm.viridis(p / max(_drop(conds[-1]), 1e-9))
            ep = np.arange(len(tr))
            ax.plot(ep, tr, color=col, lw=1.6)
            ax.plot(ep, va, color=col, lw=1.6, ls='--')
            print(f"{alabel:9s} {p:8g} {tr[-1]:12.4f} {va[-1]:10.4f} {va[-1]-tr[-1]:8.4f}")
        ax.set_xlabel('epoch'); ax.set_ylabel('Projection-based total loss')
        ax.set_title(f'{alabel} — train (solid) vs val (dashed)', fontsize=9)
        if j == 0:
            from matplotlib.lines import Line2D
            ch = [Line2D([0], [0], color=plt.cm.viridis(_drop(r) / max(_drop(conds[-1]), 1e-9)),
                         lw=2, label=f'dropout p={_drop(r):g}') for r in conds]
            sh = [Line2D([0], [0], color='0.35', lw=2, ls='-', label='train'),
                  Line2D([0], [0], color='0.35', lw=2, ls='--', label='val')]
            ax.legend(handles=ch + sh, fontsize=7, loc='upper right', frameon=True)
    ps.label_panels(axes)
    fig.suptitle('Train vs val loss with dropout (Projection-based, patience 0 + monitor_val, 6 mice) — '
                 'does dropout shrink the gap?', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'dropout_trainval_curves')
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
