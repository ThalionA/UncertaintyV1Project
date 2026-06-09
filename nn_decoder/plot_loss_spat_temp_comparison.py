# -*- coding: utf-8 -*-
"""All losses on one axis: spatial vs temporal held-out performance, per mouse,
with within-mouse paired stats — the clean-run spat/temp bars, but cross-loss.

``decoder_plotting_utils`` draws spat/temp bars for ONE cell (one loss);
``within_mouse_loss_plots`` runs that per loss but in separate figures. This
puts every loss side-by-side on a single axis so the spat-vs-temp comparison is
directly readable across training objectives.

Scoring is PCA distance (the clean-run yardstick), per mouse, in two modes:
  * shuffle-normalised (each mouse's loss / its own shuffle-decoder loss; 1.0 =
    chance) — the headline.
  * raw PCA distance.

Figures (per cell), written to the cell's figure dir:
  14_loss_spat_temp_comparison_<mode>.{png,svg}
      x = losses; paired spatial/temporal bars (mean ± SEM across mice), each
      mouse's spat→temp shown as a faint connecting line, and a within-mouse
      paired-t significance star per loss.
  14b_loss_spat_temp_permouse_<mode>.{png,svg}
      one subpanel per mouse; spatial vs temporal across losses.

Usage
-----
    python plot_loss_spat_temp_comparison.py --run-name loss_comparison_v1
    python plot_loss_spat_temp_comparison.py --run-name loss_comparison_v1 \
        --target L --window full --bin 50 --split stratified_balanced
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

import torch

import plot_loss_sweep as P
from cross_loss_eval import EVAL_LOSSES
from nn_classifier import fit_loss_per_trial

ARCH_COLOR = {'spat': '#e08214', 'temp': '#2166ac'}   # spatial orange / temporal blue


def _per_trial(md, arch, metric):
    """Per-trial loss for one arch of one mouse under ``metric`` ('PCA' or the
    loss's own name). Uses fit_loss_per_trial so the numbers match the training
    objective exactly (PCA via fit_loss_per_trial == calc_pca_dist, pinned by
    tests)."""
    d = md['Dist']
    dec = torch.tensor(np.asarray(d[arch]['decoded'], dtype=np.float64))
    tgt = torch.tensor(np.asarray(d[arch]['target'], dtype=np.float64))
    pcs_t = evar_t = None
    if metric == 'PCA':
        pcs_t = torch.tensor(np.asarray(d['pcs'], dtype=np.float64))
        evar_t = torch.tensor(np.asarray(d['explained_var'], dtype=np.float64))
    return fit_loss_per_trial(dec, tgt, metric, pcs_t, evar_t).cpu().numpy()


def _metric_for(loss, own):
    """Scoring metric for a training loss: its own name if own-metric mode,
    else the common PCA-distance yardstick."""
    return loss if own else 'PCA'


def _collect(sweep, losses, own):
    """{loss: {'mice': [...], 'spat_raw':[], 'temp_raw':[], 'spat_skill':[],
    'temp_skill':[]}} aligned by mouse. Scored under each loss's own metric
    when ``own`` else under PCA distance. Per-mouse mean (raw) and
    mean/shuffle-mean (skill)."""
    data = {}
    for loss in losses:
        metric = _metric_for(loss, own)
        mat = sweep[loss]
        mice = sorted(mat['results'], key=lambda m: int(m.split('_')[1]))
        sr, tr, ss, ts = [], [], [], []
        for m in mice:
            md = mat['results'][m]
            sp = np.nanmean(_per_trial(md, 'spat', metric))
            tp = np.nanmean(_per_trial(md, 'temp', metric))
            sps = np.nanmean(_per_trial(md, 'spat_shf', metric))
            tps = np.nanmean(_per_trial(md, 'temp_shf', metric))
            sr.append(sp); tr.append(tp)
            ss.append(sp / sps if sps > 0 else np.nan)
            ts.append(tp / tps if tps > 0 else np.nan)
        data[loss] = {'mice': mice, 'spat_raw': sr, 'temp_raw': tr,
                      'spat_skill': ss, 'temp_skill': ts}
    return data


def _star(p):
    if not np.isfinite(p):
        return 'n/a'
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05
            else 'n.s.')


def plot_comparison(data, losses, mode, out_dir, cell, own=False):
    sk = mode == 'shuffle'
    sfx = '_skill' if sk else '_raw'
    fig, ax = plt.subplots(figsize=(1.5 * len(losses) + 2.5, 5))
    x = np.arange(len(losses))
    w = 0.36
    for i, loss in enumerate(losses):
        spat = np.asarray(data[loss]['spat' + sfx], float)
        temp = np.asarray(data[loss]['temp' + sfx], float)
        ax.bar(x[i] - w / 2, np.nanmean(spat), w, color=ARCH_COLOR['spat'],
               alpha=0.85, yerr=st.sem(spat, nan_policy='omit'), capsize=3,
               label='spatial' if i == 0 else None, zorder=2)
        ax.bar(x[i] + w / 2, np.nanmean(temp), w, color=ARCH_COLOR['temp'],
               alpha=0.85, yerr=st.sem(temp, nan_policy='omit'), capsize=3,
               label='temporal' if i == 0 else None, zorder=2)
        # per-mouse spat->temp connecting lines + dots
        for s, t in zip(spat, temp):
            ax.plot([x[i] - w / 2, x[i] + w / 2], [s, t], '-', color='0.5',
                    lw=0.7, alpha=0.6, zorder=3)
        ax.scatter(np.full(spat.shape, x[i] - w / 2), spat, s=14, color='0.2',
                   zorder=4)
        ax.scatter(np.full(temp.shape, x[i] + w / 2), temp, s=14, color='0.2',
                   zorder=4)
        # within-mouse paired t (spatial vs temporal)
        ok = np.isfinite(spat) & np.isfinite(temp)
        p = st.ttest_rel(spat[ok], temp[ok]).pvalue if ok.sum() > 1 else np.nan
        top = np.nanmax(np.concatenate([spat, temp]))
        ax.text(x[i], top * 1.05, _star(p), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    metric_lab = "each loss's own metric" if own else 'PCA distance'
    if sk:
        ax.axhline(1.0, color='red', ls=':', lw=1, alpha=0.7, zorder=1)
        ax.set_ylabel(f'loss / shuffle  (1.0 = chance) — {metric_lab}')
    else:
        ax.set_ylabel(f'held-out {metric_lab}')
    ax.set_xticks(x)
    ax.set_xticklabels(losses)
    ax.set_xlabel('training loss')
    ax.set_title(f'Spatial vs temporal across losses — {cell}\n'
                 f'scored under {metric_lab}; bars = mean ± SEM over mice; '
                 f'lines = per-mouse; * = across-mouse paired t')
    ax.legend(frameon=False)
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    tag = '_ownmetric' if own else ''
    stem = f'14_loss_spat_temp_comparison{tag}_{mode}'
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=140)
    plt.close(fig)
    print(f"  -> {stem}.png/.svg")


def plot_comparison_faceted(data, losses, mode, out_dir, cell, own):
    """Across-mouse spatial-vs-temporal, but one panel per loss so each can keep
    its own raw metric scale (KL-nats / Wasserstein-bins / CE-nats don't share a
    y-axis). Each panel: spat vs temp bars (mean ± SEM over mice), per-mouse dots
    + spat→temp lines, and an across-mouse paired-t star (n = mice)."""
    sk = mode == 'shuffle'
    sfx = '_skill' if sk else '_raw'
    n = len(losses)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow),
                             squeeze=False, sharey=False)
    for li, loss in enumerate(losses):
        ax = axes[li // ncol][li % ncol]
        spat = np.asarray(data[loss]['spat' + sfx], float)
        temp = np.asarray(data[loss]['temp' + sfx], float)
        ax.bar(0, np.nanmean(spat), 0.6, color=ARCH_COLOR['spat'], alpha=0.85,
               yerr=st.sem(spat, nan_policy='omit'), capsize=3,
               label='spatial' if li == 0 else None)
        ax.bar(1, np.nanmean(temp), 0.6, color=ARCH_COLOR['temp'], alpha=0.85,
               yerr=st.sem(temp, nan_policy='omit'), capsize=3,
               label='temporal' if li == 0 else None)
        for s, t in zip(spat, temp):
            ax.plot([0, 1], [s, t], '-', color='0.5', lw=0.7, alpha=0.6)
        ax.scatter(np.zeros_like(spat), spat, s=16, color='0.2', zorder=4)
        ax.scatter(np.ones_like(temp), temp, s=16, color='0.2', zorder=4)
        ok = np.isfinite(spat) & np.isfinite(temp)
        p = st.ttest_rel(spat[ok], temp[ok]).pvalue if ok.sum() > 1 else np.nan
        top = np.nanmax(np.concatenate([spat, temp]))
        ax.text(0.5, top * 1.03, _star(p), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        if sk:
            ax.axhline(1.0, color='red', ls=':', lw=1, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['spat', 'temp'])
        ax.set_title(loss, fontsize=11)
        ax.set_ylabel(f'{loss}/shuffle' if sk else f'{loss} loss')
        if li == 0:
            ax.legend(frameon=False, fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    metric_lab = "own metric" if own else 'PCA distance'
    fig.suptitle(f'Spatial vs temporal across mice, by loss — {cell}\n'
                 f'scored under {metric_lab}; bars = mean ± SEM over mice; '
                 f'lines = per-mouse; * = across-mouse paired t (n = mice)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    tag = '_ownmetric' if own else ''
    stem = f'14_loss_spat_temp_comparison{tag}_facet_{mode}'
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {stem}.png/.svg")


def plot_per_mouse(sweep, losses, mode, out_dir, cell, own=False):
    """Subpanels = losses. Within each: per-mouse spatial vs temporal bars
    (mean ± SEM over trials), with a TRIAL-LEVEL paired t-test per mouse
    (n = that mouse's test trials). Scored under each loss's own metric when
    ``own`` else PCA distance."""
    sk = mode == 'shuffle'
    n = len(losses)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    # own-metric panels have different units per loss → don't share y.
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False, sharey=(not own))
    w = 0.36
    for li, loss in enumerate(losses):
        ax = axes[li // ncol][li % ncol]
        metric = _metric_for(loss, own)
        mat = sweep[loss]
        mice = sorted(mat['results'], key=lambda m: int(m.split('_')[1]))
        xticks, xlabels = [], []
        for mi, mouse in enumerate(mice):
            md = mat['results'][mouse]
            spat = _per_trial(md, 'spat', metric)
            temp = _per_trial(md, 'temp', metric)
            if sk:
                ss = np.nanmean(_per_trial(md, 'spat_shf', metric))
                ts = np.nanmean(_per_trial(md, 'temp_shf', metric))
                spat = spat / ss if ss > 0 else spat
                temp = temp / ts if ts > 0 else temp
            ok = np.isfinite(spat) & np.isfinite(temp)
            ntr = int(ok.sum())
            ax.bar(mi - w / 2, np.nanmean(spat), w, color=ARCH_COLOR['spat'],
                   alpha=0.85, yerr=st.sem(spat[ok]) if ntr > 1 else 0,
                   capsize=2, label='spatial' if mi == 0 else None)
            ax.bar(mi + w / 2, np.nanmean(temp), w, color=ARCH_COLOR['temp'],
                   alpha=0.85, yerr=st.sem(temp[ok]) if ntr > 1 else 0,
                   capsize=2, label='temporal' if mi == 0 else None)
            # trial-level paired test for THIS mouse (n = trials)
            p = (st.ttest_rel(spat[ok], temp[ok]).pvalue if ntr > 1 else np.nan)
            top = np.nanmax([np.nanmean(spat), np.nanmean(temp)])
            ax.text(mi, top * 1.04, _star(p), ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
            xticks.append(mi)
            xlabels.append(f"m{mouse.split('_')[1]}\nn={ntr}")
        if sk:
            ax.axhline(1.0, color='red', ls=':', lw=1, alpha=0.7)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(loss, fontsize=11)
        # own-metric: each panel labels its own metric; PCA: shared label on left.
        if own:
            ax.set_ylabel(f'{loss}/shuffle' if sk else f'{loss} loss')
        elif li % ncol == 0:
            ax.set_ylabel('PCA / shuffle (1 = chance)' if sk else 'PCA distance')
        if li == 0:
            ax.legend(frameon=False, fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    metric_lab = "own metric" if own else 'PCA distance'
    fig.suptitle(f'Per-mouse spatial vs temporal, by loss — {cell}\n'
                 f'scored under {metric_lab}; bars = mean ± SEM over trials; '
                 f'* = per-mouse trial-level paired t (n = trials)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    tag = '_ownmetric' if own else ''
    stem = f'14b_loss_spat_temp_permouse{tag}_{mode}'
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {stem}.png/.svg")


def main(run_name, target='Q', window='half', bin_ms=100,
         split='stratified_balanced', results_root=None,
         out_root='figures/loss_sweep_plots'):
    P.TARGET, P.WINDOW, P.BIN_MS, P.SPLIT = target, window, bin_ms, split
    sweep = P.load_loss_sweep(run_name, results_root=results_root)
    losses = [l for l in EVAL_LOSSES
              if sweep.get(l) is not None and 'results' in sweep[l]]
    cell = P.cell_tag()
    print(f"Loss spat/temp comparison: {run_name} | {cell}")
    print(f"  losses: {losses}")
    out_dir = Path(out_root) / run_name / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    # own=False → common PCA-distance yardstick; own=True → each loss scored
    # under its own training metric at test time.
    for own in (False, True):
        data = _collect(sweep, losses, own)
        for mode in ('shuffle', 'raw'):
            # Single shared-axis across-mouse bars: fine for PCA (any mode) and
            # for own-metric SHUFFLE (skill is unitless). Own-metric RAW would
            # mix metric scales on one axis, so it's served by the faceted
            # version below instead.
            if not (own and mode == 'raw'):
                plot_comparison(data, losses, mode, out_dir, cell, own=own)
            # Faceted across-mouse (one panel per loss, own y-scale) — this is
            # what makes own-metric RAW across mice readable.
            if own:
                plot_comparison_faceted(data, losses, mode, out_dir, cell, own)
            plot_per_mouse(sweep, losses, mode, out_dir, cell, own=own)
    print(f"Done. {out_dir.resolve()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(run_name=a.run_name, target=a.target, window=a.window, bin_ms=a.bin_ms,
         split=a.split, results_root=a.results_root, out_root=a.out_root)
