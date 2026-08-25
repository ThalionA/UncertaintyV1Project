# -*- coding: utf-8 -*-
"""Overfitting and peakiness for the production cells, spat vs temp, BOTH arms.

Companion to spat_temp_best_cell (which carries the loss comparison): the same
per-mouse-paired house style, but for the two metrics that are NOT a fit loss —
the overfitting gap and the equivalent sharpening — read straight from
``figures/io_hmm_wide/cells_paired.csv`` (io_hmm_wide_extract on both arms; no
re-extraction here).

Cells: kl_h8_lh0 (production primary), kl_h4_lh0, js_h8_lh0 — h4/h8 only, the
overparameterised widths are excluded by design. Arms: NEW io_hmm_v3 (IO-HMM
marginal targets, 72 x 2.5 deg circular) and OLD exportref (export Q,
91 x 1 deg linear); identical trials/splits/seeds, only the target family
differs.

Figures (PNG+SVG) in figures/io_hmm_wide/spat_temp/:

  overfit_s_hat_cells      rows = metric (overfitting gap; equivalent
                           sharpening s-hat), cols = cell; x = arm, spat vs
                           temp side by side within each arm slot, per-mouse
                           paired points, black median bars, reference lines
                           (0 for the gap, 1 for s-hat), paired t + sign
                           counts on the figure (n = 6 mice).
  peakiness_within_arm_kl_h8   the classic raw view for kl_h8_lh0: per-mouse
                           median (+ IQR) of the per-trial decoded max-prob,
                           spat and temp, with the target max-prob as an open
                           marker — WITHIN-ARM ONLY: the supports differ
                           (72 circular vs 91 linear bins), so max-prob is
                           never compared across arms.

Both metrics in the top figure are cross-arm comparable by construction: the
gap is (val - train at best epoch) in the cell's own training-loss units
divided by that mouse's own-arm LOO predict-mean loss, and s-hat is the
equivalent sharpening of each arm's own target (1 = target width).

Usage (from nn_decoder/):
    python diagnostics/spat_temp_overfit_peakiness.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import scipy.stats as sstats                          # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from scipy.io import loadmat                          # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import peakiness_style as ps                          # noqa: E402
from figsave import save_fig                          # noqa: E402
from io_hmm_wide_paired import ARM_LABEL, MOUSE_MARKER  # noqa: E402

CSV = _HERE.parent / 'figures' / 'io_hmm_wide' / 'cells_paired.csv'
OUT_DIR = _HERE.parent / 'figures' / 'io_hmm_wide' / 'spat_temp'
RESULTS = _HERE.parent / 'results'

CELLS = ('kl_h8_lh0', 'kl_h4_lh0', 'js_h8_lh0')
CELL_LABEL = {'kl_h8_lh0': 'KL · h8 (production)', 'kl_h4_lh0': 'KL · h4',
              'js_h8_lh0': 'JS · h8'}
ARMS = ('new', 'old')                       # HMM arm first (production targets)
ARM_SHORT = {'new': 'HMM\n(72 × 2.5°)', 'old': 'export Q\n(91 × 1°)'}
RUN_BY_ARM = {'new': 'io_hmm_v3', 'old': 'io_hmm_v3_exportref'}
LOSS_DIR = {'kl': 'Q_KL_half_100ms', 'js': 'Q_JS_half_100ms'}
DECODERS = (('spat', 'spatial'), ('temp', 'temporal'))
MICE = tuple(range(6))

METRICS = (
    ('overfit_gap', 'overfitting gap\n(val − train at best epoch) / predict-mean loss', 0.0),
    ('s_hat', 'equivalent sharpening ŝ\n(1 = target width)', 1.0),
)

PEAK_CELL = 'kl_h8_lh0'
DODGE = 0.17


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_csv(csv=CSV):
    """cells_paired.csv restricted to CELLS; asserts 6 mice per block."""
    df = pd.read_csv(csv)
    d = df[df.cell.isin(CELLS)].copy()
    for cell in CELLS:
        for arm in ARMS:
            for dec, _ in DECODERS:
                b = d[(d.cell == cell) & (d.arm == arm) & (d.decoder == dec)]
                if sorted(b.mouse) != list(MICE):
                    raise SystemExit(f'ABORT: {cell}/{arm}/{dec} has mice '
                                     f'{sorted(b.mouse)}, expected {list(MICE)}')
    n_cl = int(d.s_hat_clamped.sum())
    if n_cl:
        print(f'NOTE: {n_cl} s_hat values clamped in these cells — marked ×')
    return d


def block(d, cell, arm, metric):
    """(6,) per-mouse arrays {dec: values} for one (cell, arm, metric)."""
    out = {}
    for dec, _ in DECODERS:
        b = (d[(d.cell == cell) & (d.arm == arm) & (d.decoder == dec)]
             .set_index('mouse')[metric].reindex(list(MICE)))
        out[dec] = b.to_numpy(float)
    return out


def load_peakiness(cell=PEAK_CELL):
    """{(arm, mouse): {'spat': maxP per trial, 'temp': ..., 'tgt': ..., 'nb': bins}}
    from the per-mouse shards (test half only)."""
    loss = cell.split('_')[0]
    out = {}
    for arm in ARMS:
        cdir = RESULTS / RUN_BY_ARM[arm] / cell / LOSS_DIR[loss]
        for m in MICE:
            sh = loadmat(str(cdir / 'shards' / f'stratified_balanced__mouse_{m}.mat'),
                         simplify_cells=True)
            D = sh['results'][f'mouse_{m}']['Dist']
            tg_s = np.asarray(D['spat']['target'], float)
            tg_t = np.asarray(D['temp']['target'], float)
            if not np.allclose(tg_s, tg_t, equal_nan=True):
                raise SystemExit(f'ABORT: spat/temp targets differ, {arm} mouse {m}')
            rec = {'nb': tg_s.shape[1]}
            ok = np.isfinite(tg_s).all(1)
            for dec, _ in DECODERS:
                dd = np.asarray(D[dec]['decoded'], float)
                ok &= np.isfinite(dd).all(1)
            for dec, _ in DECODERS:
                rec[dec] = np.asarray(D[dec]['decoded'], float)[ok].max(axis=1)
            rec['tgt'] = tg_s[ok].max(axis=1)
            out[(arm, m)] = rec
    return out


# ----------------------------------------------------------------------
# figure 1 — overfitting gap + s_hat, per-mouse paired, both arms
# ----------------------------------------------------------------------
def fig_overfit_s_hat(d, out_dir):
    fig, axes = plt.subplots(len(METRICS), len(CELLS),
                             figsize=ps.figsize(len(CELLS), len(METRICS)),
                             sharex=True, sharey='row')
    stats = {}
    for r, (metric, ylab, ref) in enumerate(METRICS):
        for c, cell in enumerate(CELLS):
            ax = axes[r, c]
            for xa, arm in enumerate(ARMS):
                v = block(d, cell, arm, metric)
                xs, xt = xa - DODGE, xa + DODGE
                for m in MICE:
                    ax.plot([xs, xt], [v['spat'][m], v['temp'][m]], '-',
                            lw=0.8, color='0.72', zorder=2)
                    ax.plot(xs, v['spat'][m], MOUSE_MARKER[m], ms=4.6,
                            color=ps.SPATIAL, mec='k', mew=0.3, zorder=3)
                    ax.plot(xt, v['temp'][m], MOUSE_MARKER[m], ms=4.6,
                            color=ps.TEMPORAL, mec='k', mew=0.3, zorder=3)
                for x0, dec in ((xs, 'spat'), (xt, 'temp')):
                    ax.plot([x0 - 0.11, x0 + 0.11], [np.median(v[dec])] * 2,
                            '-', lw=2.4, color='k', zorder=5)
                t, p = sstats.ttest_rel(v['temp'], v['spat'])
                k = int((v['temp'] > v['spat']).sum())
                stats[(metric, cell, arm)] = (v, t, p, k)
                ax.text(xa, 0.99, f't(5)={t:+.2f} p={p:.3f}\ntemp>spat {k}/6',
                        transform=ax.get_xaxis_transform(), fontsize=6,
                        ha='center', va='top', color='0.25')
            ax.axhline(ref, ls=':', lw=1.1, color='0.4', zorder=1)
            ax.set_xticks(range(len(ARMS)))
            ax.set_xticklabels([ARM_SHORT[a] for a in ARMS], fontsize=8)
            ax.set_xlim(-0.55, len(ARMS) - 0.45)
            ax.grid(True, axis='y', lw=0.4, alpha=0.3)
            if r == 0:
                ax.set_title(CELL_LABEL[cell], fontsize=10)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if r == len(METRICS) - 1:
                ax.set_xlabel('target arm', fontsize=8)
    for r, (_, _, ref) in enumerate(METRICS):
        # ONE expansion per y-shared row (per-axis would compound), keeping
        # headroom for the stats text and the reference line in view
        lo, hi = axes[r, 0].get_ylim()
        span = hi - lo
        axes[r, 0].set_ylim(min(lo, ref - 0.04 * span), hi + 0.24 * span)
    handles = ([Line2D([0], [0], marker=MOUSE_MARKER[m], ls='none', color='0.45',
                       ms=5, label=f'mouse {m}') for m in MICE]
               + [Line2D([0], [0], marker='o', ls='none', color=ps.SPATIAL,
                         mec='k', mew=0.3, ms=6, label='spatial'),
                  Line2D([0], [0], marker='o', ls='none', color=ps.TEMPORAL,
                         mec='k', mew=0.3, ms=6, label='temporal'),
                  Line2D([0], [0], ls='-', lw=2.4, color='k', label='median (n = 6 mice)'),
                  Line2D([0], [0], ls=':', lw=1.1, color='0.4',
                         label='reference (gap 0 / ŝ 1)')])
    fig.legend(handles=handles, loc='outside lower center', ncol=10, fontsize=7.5,
               frameon=False)
    fig.suptitle('Overfitting gap and equivalent sharpening — production cells, both arms, '
                 'n = 6 mice\n'
                 f'{ARM_LABEL["new"]} vs {ARM_LABEL["old"]}; gap in each cell\'s own '
                 'training-loss units ÷ own-arm LOO predict-mean; ŝ vs each arm\'s own target',
                 fontsize=10)
    save_fig(fig, out_dir, 'overfit_s_hat_cells')
    return stats


# ----------------------------------------------------------------------
# figure 2 — classic within-arm peakiness, kl_h8_lh0
# ----------------------------------------------------------------------
def fig_peakiness(P, out_dir):
    fig, axes = plt.subplots(1, len(ARMS), figsize=ps.figsize(len(ARMS), 1))
    counts = {}
    for ax, arm in zip(axes, ARMS):
        nb = P[(arm, 0)]['nb']
        for m in MICE:
            rec = P[(arm, m)]
            for dx, (dec, _) in zip((-DODGE, +DODGE), DECODERS):
                q1, q2, q3 = np.percentile(rec[dec], [25, 50, 75])
                ax.plot([m + dx] * 2, [q1, q3], '-', lw=1.0,
                        color=ps.ARCH[dec], alpha=0.55, zorder=2)
                ax.plot(m + dx, q2, 'o', ms=5.5, color=ps.ARCH[dec],
                        mec='k', mew=0.3, zorder=4)
            ax.plot(m, np.median(rec['tgt']), 'D', ms=6, mfc='none', mec='k',
                    mew=1.3, zorder=5)
        ax.axhline(1.0 / nb, ls=':', lw=1.1, color=ps.CHANCE_GREY, zorder=1)
        ax.text(len(MICE) - 0.55, 1.0 / nb, f' uniform 1/{nb}', fontsize=7,
                va='bottom', ha='right', color='0.35')
        for dec, dec_lab in DECODERS:
            med = np.array([np.median(P[(arm, m)][dec]) for m in MICE])
            tgt = np.array([np.median(P[(arm, m)]['tgt']) for m in MICE])
            counts[(arm, dec)] = (med, tgt, int((med > tgt).sum()))
        k_ts = int(sum(np.median(P[(arm, m)]['temp']) > np.median(P[(arm, m)]['spat'])
                       for m in MICE))
        counts[(arm, 'temp_gt_spat')] = k_ts
        ax.set_title(f'{ARM_LABEL[arm]}\n'
                     f"decoded > target: spat {counts[(arm, 'spat')][2]}/6, "
                     f"temp {counts[(arm, 'temp')][2]}/6; temp > spat {k_ts}/6",
                     fontsize=8.5)
        n_tr = [len(P[(arm, m)]['spat']) for m in MICE]
        ax.set_xticks(list(MICE))
        ax.set_xticklabels([f'm{m}\n(n={n})' for m, n in zip(MICE, n_tr)], fontsize=7)
        ax.set_xlabel('mouse (n = held-out test trials)', fontsize=8)
        ax.grid(True, axis='y', lw=0.4, alpha=0.3)
        ax.set_ylim(0, None)
    axes[0].set_ylabel('per-trial max-probability\n(per-mouse median, bar = IQR)',
                       fontsize=8)
    handles = [Line2D([0], [0], marker='o', ls='none', color=ps.SPATIAL, mec='k',
                      mew=0.3, ms=6, label='spatial decoded'),
               Line2D([0], [0], marker='o', ls='none', color=ps.TEMPORAL, mec='k',
                      mew=0.3, ms=6, label='temporal decoded'),
               Line2D([0], [0], marker='D', ls='none', mfc='none', mec='k',
                      mew=1.3, ms=6, label='target'),
               Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, label='uniform')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=8,
               frameon=False)
    fig.suptitle(f'Peakiness (max-prob), {CELL_LABEL[PEAK_CELL]} — WITHIN-ARM ONLY: '
                 'supports differ (72 circular vs 91 linear bins),\nso max-prob is '
                 'never compared across arms; targets identical across decoders (asserted)',
                 fontsize=10)
    save_fig(fig, out_dir, 'peakiness_within_arm_kl_h8')
    return counts


# ----------------------------------------------------------------------
# printed readout
# ----------------------------------------------------------------------
def report(stats, counts):
    print('\n== overfit gap + s_hat (from cells_paired.csv; medians over 6 mice) ==')
    for metric, _, ref in METRICS:
        print(f'  {metric}  (reference {ref:g})')
        for cell in CELLS:
            for arm in ARMS:
                v, t, p, k = stats[(metric, cell, arm)]
                ms, mt = np.median(v['spat']), np.median(v['temp'])
                if metric == 'overfit_gap':
                    ks = int((v['spat'] > 0).sum()); kt = int((v['temp'] > 0).sum())
                    extra = f'gap>0: spat {ks}/6 temp {kt}/6'
                else:
                    ks = int((v['spat'] < 1).sum()); kt = int((v['temp'] < 1).sum())
                    extra = f'ŝ<1 (under-sharpened): spat {ks}/6 temp {kt}/6'
                print(f'    {cell:10s} {arm}: spat {ms:6.3f}  temp {mt:6.3f}  '
                      f'Δ(temp−spat) {mt - ms:+6.3f}  temp>spat {k}/6  '
                      f't(5)={t:+.2f} p={p:.3f}  | {extra}')
    print(f'\n== classic peakiness, {PEAK_CELL} (per-mouse median max-prob; '
          'within-arm only) ==')
    for arm in ARMS:
        for dec, dec_lab in DECODERS:
            med, tgt, kk = counts[(arm, dec)]
            print(f'  {arm} {dec_lab:8s}: decoded {np.median(med):.3f} '
                  f'(per mouse {" ".join(f"{x:.3f}" for x in med)})  '
                  f'target {np.median(tgt):.3f}  decoded>target {kk}/6')
        print(f'  {arm} temp > spat (per-mouse medians): '
              f'{counts[(arm, "temp_gt_spat")]}/6')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', type=Path, default=CSV)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    ps.apply()
    d = load_csv(args.csv)
    print(f'{len(d)} rows for {len(CELLS)} cells x 2 arms x 2 decoders x 6 mice '
          f'from {args.csv}')
    stats = fig_overfit_s_hat(d, args.out_dir)
    P = load_peakiness()
    counts = fig_peakiness(P, args.out_dir)
    report(stats, counts)
    print(f'\nDone -> {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
