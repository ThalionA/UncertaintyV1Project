# -*- coding: utf-8 -*-
"""Two summary figures telling the over-sharpening story end to end.

FIG 1 — DIAGNOSIS: what the problem is, and what it is not.
  (a) the projection-based loss trains decoders ~5-11x more confident than the ideal
      observer, while KL/JS sit on the target;
  (b) the projection metric CANNOT SEE this — the same decoders score ~flat under it,
      while KL exposes the projection decoder as worse than chance;
  (c) it is NOT capacity: removing the entire hidden layer (a linear decoder, zero hidden
      units) barely dents the over-sharpening;
  (d) bias =/= variance: the peakiness ordering across losses is the INVERSE of the
      train-val overfitting ordering.

FIG 2 — CURE: which interventions actually work. Every candidate on one axis, judged on
  peakiness (top) AND chance-normalised loss (bottom). Interventions that land peakiness
  on the IO target while never beating chance (weight_decay, smooth_lambda) are lobotomies:
  they cure the symptom by destroying the decoder.

Sources: `prodfix_v1` (H=8, restart selection on val) for everything except the dropout and
weight_decay cells, which come from `hpsweep_v2` (same architecture and schedule, but
trained under the older training-loss restart rule). That difference is <=6% on the
val/train ratio per the matched control, and the effects shown here are far larger, but the
cross-regime bars are hatched and footnoted so the distinction is never invisible.

Outputs (PNG+SVG) under figures/prodfix/.
Usage:  python diagnostics/story_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpsweep_spec as S  # noqa: E402
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
N_CATS = 91
PROD = 'prodfix_v1'


# ------------------------------------------------------------------ loading
def _slug_dir(results_root, run, cell, loss=None):
    """Slug directory for a cell. `loss` MUST be given for hpsweep-style runs: there the
    cell dir encodes only the non-loss axes, so it contains ONE SLUG PER LOSS and picking
    the first one silently reads an arbitrary loss (this bug produced a weight_decay bar
    that disagreed with the measured value). prodfix cells carry the loss in the run name,
    so they have exactly one slug and `loss=None` is safe — but it is verified below."""
    d = Path(results_root) / run / cell
    if not d.is_dir():
        raise SystemExit(f"missing {d} — rsync the run down first.")
    if loss is not None:
        sd = d / S.LOSS_SLUG[loss]
        if not sd.is_dir():
            raise SystemExit(f"missing {sd} (loss={loss}).")
        return sd
    subs = [p for p in d.iterdir() if p.is_dir()]
    if len(subs) != 1:
        raise SystemExit(
            f"{d} holds {len(subs)} slug dirs {[p.name for p in subs]} — ambiguous. Pass "
            f"an explicit loss: hpsweep cell dirs contain one slug per loss.")
    return subs[0]


def _results(results_root, run, cell, loss=None):
    return sio.loadmat(str(_slug_dir(results_root, run, cell, loss) / 'stratified_balanced.mat'),
                       simplify_cells=True)['results']


def peaky(res, arch):
    return np.array([np.asarray(res[m]['Dist'][arch]['decoded'], float).max(1).mean()
                     for m in sorted(res)])


def tgt_peak(res, arch):
    return np.array([np.asarray(res[m]['Dist'][arch]['target'], float).max(1).mean()
                     for m in sorted(res)])


def normloss(res, arch, metric='KL'):
    return np.array(_norm_by_mouse(res, arch)[(metric, 'pm')], float)


def overfit_ratio(results_root, run, cell, arch):
    ck = _slug_dir(results_root, run, cell) / 'checkpoints'
    rs = []
    for pt in sorted(ck.glob('mouse_*_stratified_balanced.pt')):
        h = (torch.load(str(pt), map_location='cpu', weights_only=False).get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if t and v and t[-1] > 0:
            rs.append(v[-1] / t[-1])
    return np.array(rs, float)


def _ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


# ------------------------------------------------------------------ figure 1
def fig_diagnosis(results_root, out_root):
    ps.apply()
    LOSSES = [('A_baseline_pca', 'PCA'), ('A_reference_kl', 'KL'), ('A_reference_js', 'JS')]
    R = {c: _results(results_root, PROD, c) for c, _ in LOSSES}
    R['C_nohidden_pca'] = _results(results_root, PROD, 'C_nohidden_pca')
    R['C_nohidden_kl'] = _results(results_root, PROD, 'C_nohidden_kl')
    R['C_nohidden_js'] = _results(results_root, PROD, 'C_nohidden_js')
    arch = 'spat'
    tgt = tgt_peak(R['A_baseline_pca'], arch).mean()

    fig, ax = plt.subplots(1, 4, figsize=ps.figsize(4, 1))

    # (a) the problem
    for j, (cell, loss) in enumerate(LOSSES):
        m, s = _ms(peaky(R[cell], arch))
        ax[0].bar(j, m, 0.62, yerr=s, capsize=3, color=ps.color(loss), edgecolor='k', lw=0.5)
    ax[0].axhline(tgt, color='k', ls=':', lw=1.4)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([ps.loss_label(l) for _, l in LOSSES],
                                                      rotation=15, fontsize=7.5)
    ax[0].set_ylabel('decoded peakiness (max-prob)')
    ax[0].set_title('The problem:\nover-confident posteriors', fontsize=9)
    ax[0].legend(handles=[Line2D([0], [0], color='k', ls=':', lw=1.4, label='IO target')],
                 fontsize=7, frameon=True)

    # (b) the training metric is blind
    w = 0.36
    for j, (cell, loss) in enumerate(LOSSES):
        for k, (metric, hatch, lab) in enumerate([('PCA', '', 'projection metric'),
                                                  ('KL', '//', 'KL metric')]):
            m, s = _ms(normloss(R[cell], arch, metric))
            ax[1].bar(j + (k - 0.5) * w, m, w * 0.95, yerr=s, capsize=3,
                      color=ps.color(loss), alpha=0.95 if k == 0 else 0.55, hatch=hatch,
                      edgecolor='k', lw=0.5, label=lab if j == 0 else None)
    ax[1].axhline(1.0, color='k', ls=':', lw=1.4)
    ax[1].set_yscale('log')
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([ps.loss_label(l) for _, l in LOSSES],
                                                      rotation=15, fontsize=7.5)
    ax[1].set_ylabel('normalised loss ÷ predict-mean')
    ax[1].set_title('Its own metric is blind:\nonly KL sees the failure', fontsize=9)
    ax[1].legend(fontsize=6.5, frameon=True)

    # (c) not capacity
    for j, (mlp, lin, loss) in enumerate([('A_baseline_pca', 'C_nohidden_pca', 'PCA'),
                                          ('A_reference_kl', 'C_nohidden_kl', 'KL'),
                                          ('A_reference_js', 'C_nohidden_js', 'JS')]):
        for k, (cell, hatch, lab) in enumerate([(mlp, '', 'H = 8'), (lin, '//', 'no hidden layer')]):
            m, s = _ms(peaky(R[cell], arch))
            ax[2].bar(j + (k - 0.5) * w, m, w * 0.95, yerr=s, capsize=3,
                      color=ps.color(loss), alpha=0.95 if k == 0 else 0.55, hatch=hatch,
                      edgecolor='k', lw=0.5, label=lab if j == 0 else None)
    ax[2].axhline(tgt, color='k', ls=':', lw=1.4)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([ps.loss_label(l) for _, l in LOSSES],
                                                      rotation=15, fontsize=7.5)
    ax[2].set_ylabel('decoded peakiness (max-prob)')
    ax[2].set_title('Not capacity:\nzero hidden units, same problem', fontsize=9)
    ax[2].legend(fontsize=6.5, frameon=True)

    # (d) bias is not variance — the two failure modes are ANTI-correlated across losses.
    # A scatter rather than twin axes: the point is the inversion, which twin y-scales
    # obscure (and whose relative bar heights are meaningless anyway).
    for cell, loss in LOSSES:
        xo = overfit_ratio(results_root, PROD, cell, arch).mean()
        yp = np.mean(peaky(R[cell], arch) / tgt_peak(R[cell], arch))
        ax[3].scatter([xo], [yp], s=150, color=ps.color(loss), edgecolor='k', lw=0.8, zorder=5)
        ax[3].annotate(ps.loss_label(loss), (xo, yp), textcoords='offset points',
                       xytext=(0, -16), ha='center', fontsize=7.5)
    ax[3].axhline(1.0, color='k', ls=':', lw=1.3)
    ax[3].set_xscale('log'); ax[3].set_yscale('log')
    ax[3].margins(x=0.30, y=0.30)          # room for the point labels at the extremes
    ax[3].set_xlabel('overfitting   val ÷ train loss')
    ax[3].set_ylabel('over-sharpening   peakiness ÷ target')
    ax[3].set_title('Bias ≠ variance:\nthe two are anti-correlated', fontsize=9)
    ax[3].legend(handles=[Line2D([0], [0], color='k', ls=':', lw=1.3, label='calibrated (=1)')],
                 fontsize=6.5, frameon=True, loc='lower left')

    ps.label_panels(ax)
    fig.suptitle('Diagnosis — the projection-based loss trains over-confident decoders, cannot detect it, '
                 'and the cause is the loss geometry, not network capacity   (spatial decoder, 6 mice, mean±sem)',
                 y=1.03, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'story_fig1_diagnosis')


# ------------------------------------------------------------------ figure 2
def fig_cure(results_root, out_root):
    ps.apply()
    v2 = S.SPECS['v2']
    # (label, run, cell, colour-key, cross-regime?)
    # (label, run, cell, loss, cross-regime?) — `loss` also selects the slug, which is
    # REQUIRED for the hpsweep_v2 cells (one slug per loss inside each cell dir).
    CANDS = [
        ('none\n(baseline)',   PROD, 'A_baseline_pca', 'PCA', False),
        ('no hidden\nlayer',   PROD, 'C_nohidden_pca', 'PCA', False),
        ('dropout\n0.9',       'hpsweep_v2', S.cell_for(v2, 'dropout', 0.9), 'PCA', True),
        ('weight decay\n0.01', 'hpsweep_v2', S.cell_for(v2, 'weight_decay', 0.01), 'PCA', True),
        ('smooth\nλ=0.3',      PROD, 'A_smooth0p3', 'PCA', False),
        ('smooth\nλ=1',        PROD, 'A_smooth1', 'PCA', False),
        ('shape\nλ=0.1',       PROD, 'A_shape10', 'PCA', False),
        ('shape\nλ=0.3',       PROD, 'A_shape30', 'PCA', False),
        ('use KL',             PROD, 'A_reference_kl', 'KL', False),
        ('use JS',             PROD, 'A_reference_js', 'JS', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), sharex=True)
    for c, (arch, alab) in enumerate(ARCHS):
        pk, nl, cols, hat = [], [], [], []
        for lab, run, cell, key, cross in CANDS:
            res = _results(results_root, run, cell, loss=key if cross else None)
            pk.append(_ms(peaky(res, arch)))
            nl.append(_ms(normloss(res, arch, 'KL')))
            cols.append(ps.color(key)); hat.append('xx' if cross else '')
        x = np.arange(len(CANDS))
        tgt = tgt_peak(_results(results_root, PROD, 'A_baseline_pca'), arch).mean()
        for j in range(len(CANDS)):
            axes[0][c].bar(j, pk[j][0], 0.68, yerr=pk[j][1], capsize=2, color=cols[j],
                           hatch=hat[j], edgecolor='k', lw=0.5)
            axes[1][c].bar(j, nl[j][0], 0.68, yerr=nl[j][1], capsize=2, color=cols[j],
                           hatch=hat[j], edgecolor='k', lw=0.5)
        axes[0][c].axhline(tgt, color='k', ls=':', lw=1.4)
        axes[0][c].axhline(1.0 / N_CATS, color='0.45', ls='--', lw=1.1)
        axes[1][c].axhline(1.0, color='k', ls=':', lw=1.4)
        axes[1][c].set_yscale('log')
        axes[0][c].set_title(alab, fontsize=10)
        axes[1][c].set_xticks(x)
        axes[1][c].set_xticklabels([c0[0] for c0 in CANDS], rotation=55, ha='right', fontsize=6.5)
    axes[0][0].set_ylabel('decoded peakiness\n(dotted = IO target)')
    axes[1][0].set_ylabel('normalised loss ÷ predict-mean\n(dotted = chance)')
    axes[0][0].legend(handles=[
        Line2D([0], [0], color='k', ls=':', lw=1.4, label='IO target'),
        Line2D([0], [0], color='0.45', ls='--', lw=1.1, label='uniform (1/91)'),
        Patch(facecolor='w', edgecolor='k', hatch='xx', label='from hpsweep_v2\n(older restart rule)')],
        fontsize=6, frameon=True)
    ps.label_panels(axes.ravel())
    fig.suptitle('Cure — every candidate judged twice. Landing peakiness on the IO target (top) is NOT enough: '
                 'strong dropout, weight decay and smoothness never beat chance (bottom) — they cure the symptom '
                 'by destroying the decoder. Only shape λ=0.3, KL and JS do both.', y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'story_fig2_cure')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/prodfix')
    a = ap.parse_args()
    fig_diagnosis(a.results_root, a.out_root)
    fig_cure(a.results_root, a.out_root)
    print(f'Done -> {Path(a.out_root).resolve()}')


if __name__ == '__main__':
    main()
