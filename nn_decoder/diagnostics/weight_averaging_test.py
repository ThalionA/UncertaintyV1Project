# -*- coding: utf-8 -*-
"""Does averaging across late training epochs tame PCA's over-sharpening?
(2026-06-16, "method 1" — weight/output averaging, from saved snapshots; no retraining.)

The over-sharpening is a one-signed *drift* during training (§4½). Two cheap,
post-hoc averaging tricks on the `noreg` run (200 epochs, no regulariser, the most
over-sharp; weights snapshotted every 10 epochs in `history['state_dicts']`):

  baseline    — the final-epoch model (deploy as-is).
  SWA         — Stochastic Weight Averaging: average the *weights* of the late
                snapshots (epoch ≥ cutoff), forward once.
  output-avg  — average the decoded *posteriors* across the late snapshots
                (temporal ensembling in output space).

Per loss × arch (6 mice): decoded peakiness (max-prob) + KL(decoded ‖ IO target).
Falsifiable prior: if the drift is monotonic, SWA lands near the late (sharp) state
(little help); output-avg can still smooth bin-to-bin spikes if they wander across
epochs. We'll see which.

Forward matches training (DATA_MAP): spatial = softmax(MLP(mean_t X)); temporal =
mean_t softmax(MLP(X_t)). All local — uses `X_test` saved in each checkpoint.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  weight_averaging_test.png        — peakiness + KL: baseline vs SWA vs output-avg
  weight_averaging_examples.png    — example posteriors (PCA), the three vs the target

Usage:  python diagnostics/weight_averaging_test.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one, _agg  # noqa: E402
from nn_classifier import SimpleFlexibleNNClassifier  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
LCOL = {'PCA': ps.PCA_EVAR, 'CE': ps.CE, 'KL': ps.KL, 'JS': ps.JS,
        'Wasserstein': ps.WASSERSTEIN}
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
VARIANTS = [('base', 'final epoch', '#525252'), ('swa', 'SWA (avg weights)', '#1b7837'),
            ('oavg', 'output-avg', '#762a83')]
THETA = np.arange(91.0)


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _model(model_params):
    return SimpleFlexibleNNClassifier(
        input_size=int(model_params['input_size']),
        hidden_sizes=list(np.atleast_1d(model_params['hidden_sizes'])),
        output_size=int(model_params['output_size']),
        activation=str(model_params.get('activation_function', 'relu')),
        dropout=float(model_params.get('dropout', 0.0)))


def _forward(state_dict, model_params, X, arch):
    """Decoded posteriors (n, 91) for one weight set, matching the arch's forward."""
    m = _model(model_params); m.load_state_dict(state_dict); m.eval()
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    with torch.no_grad():
        if arch == 'spat':
            p = F.softmax(m(Xt.mean(1)), dim=-1)                  # mean over time, then decode
        else:
            n, T, nN = Xt.shape
            logits = m(Xt.reshape(n * T, nN)).reshape(n, T, -1)   # per-bin
            p = F.softmax(logits, dim=-1).mean(1)                 # then average over time
    return p.numpy()


def _avg_sd(sds):
    return {k: torch.stack([sd[k].float() for sd in sds]).mean(0) for k in sds[0]}


def _decoded_set(ck, arch, cutoff):
    """(baseline, SWA, output-avg) decoded (n,91) for one mouse/arch."""
    h = ck[arch]['history']
    mp, X = ck[arch]['model_params'], ck[arch]['X_test']
    sds, se = h['state_dicts'], list(np.atleast_1d(h['snapshot_epochs']))
    late = [sd for sd, e in zip(sds, se) if e >= cutoff] or [sds[-1]]
    base = _forward(sds[-1], mp, X, arch)
    swa = _forward(_avg_sd(late), mp, X, arch)
    oavg = np.mean([_forward(sd, mp, X, arch) for sd in late], axis=0)
    return base, swa, oavg


def collect(results_root, run, split, cutoff):
    M = {(l, a): {v[0]: {'peak': [], 'kl': []} for v in VARIANTS} for l in LOSSES for a, _ in ARCHS}
    io_peak = []
    for loss in LOSSES:
        cell = Path(results_root) / run / _slug(loss)
        f = cell / f'{split}.mat'
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get('results')
        for mk in sorted(res):
            cpt = cell / 'checkpoints' / f'{mk}_{split}.pt'
            if not cpt.is_file():
                continue
            ck = torch.load(str(cpt), map_location='cpu', weights_only=False)
            for arch, _ in ARCHS:
                D = res[mk]['Dist'][arch]
                tgt = np.asarray(D['target'], float)
                pcs, evar = res[mk]['Dist'].get('pcs'), res[mk]['Dist'].get('explained_var')
                dec = dict(zip(('base', 'swa', 'oavg'), _decoded_set(ck, arch, cutoff)))
                for v, d in dec.items():
                    M[(loss, arch)][v]['peak'].append(float(np.nanmean(d.max(1))))
                    M[(loss, arch)][v]['kl'].append(_eval_one(d, tgt, 'KL', pcs, evar))
                io_peak.append(float(np.nanmean(tgt.max(1))))
    return M, (float(np.mean(io_peak)) if io_peak else np.nan)


def fig_quant(M, io_peak, out_root):
    ps.apply()
    x = np.arange(len(LOSSES)); w = 0.26
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), sharex=True)
    for r, (arch, alabel) in enumerate(ARCHS):
        for c, (key, mlabel) in enumerate([('peak', 'peakiness (max-prob)'),
                                           ('kl', 'KL(decoded ‖ IO target)')]):
            ax = axes[r, c]
            for k, (v, vlab, vcol) in enumerate(VARIANTS):
                ys = [_agg(M[(l, arch)][v][key])[0] for l in LOSSES]
                es = [_agg(M[(l, arch)][v][key])[1] for l in LOSSES]
                ax.bar(x + (k - 1) * w, ys, w, yerr=es, capsize=2, color=vcol, label=vlab)
            if key == 'peak' and np.isfinite(io_peak):
                ax.axhline(io_peak, color='k', ls=':', lw=1.4, label='IO target')
            ax.set_xticks(x); ax.set_xticklabels(LOSSES, rotation=20, ha='right')
            ax.set_ylabel(f'{alabel}\n{mlabel}' if c == 0 else mlabel, fontsize=8)
            if r == 0:
                ax.set_title(mlabel, fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=6.5, loc='upper right')
    ps.label_panels(axes.ravel())
    fig.suptitle('Late-epoch averaging vs the final model (noreg, 6 mice): does it tame '
                 'over-sharpening?', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'weight_averaging_test')


def fig_examples(results_root, run, split, cutoff, out_root, mouse='mouse_0', ncol=4):
    ps.apply()
    fig, axes = plt.subplots(len(ARCHS), ncol, figsize=ps.figsize(ncol, len(ARCHS), panel_w=1.9, panel_h=1.5),
                             sharex=True)
    axes = np.atleast_2d(axes)
    cell = Path(results_root) / run / _slug('PCA')
    res = sio.loadmat(str(cell / f'{split}.mat'), simplify_cells=True)['results']
    ck = torch.load(str(cell / 'checkpoints' / f'{mouse}_{split}.pt'), map_location='cpu', weights_only=False)
    for r, (arch, alabel) in enumerate(ARCHS):
        tgt = np.asarray(res[mouse]['Dist'][arch]['target'], float)
        base, swa, oavg = _decoded_set(ck, arch, cutoff)
        sel = np.argsort(tgt.max(1))[np.linspace(0, tgt.shape[0] - 1, ncol).round().astype(int)]  # broad→peaky
        for c, tr in enumerate(sel):
            ax = axes[r, c]
            ax.fill_between(THETA, tgt[tr], color='0.85', lw=0, zorder=0)
            ax.plot(THETA, base[tr], color=VARIANTS[0][2], lw=1.0, label='final')
            ax.plot(THETA, swa[tr], color=VARIANTS[1][2], lw=1.4, label='SWA')
            ax.plot(THETA, oavg[tr], color=VARIANTS[2][2], lw=1.4, label='output-avg')
            ps.cap_posterior_ylim(ax, float(tgt[tr].max()), mult=3.0)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'target mp {tgt[tr].max():.3f}', fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f'{alabel}\nPCA posterior', fontsize=8)
            if r == len(ARCHS) - 1:
                ax.set_xlabel('orientation (deg)', fontsize=8)
    axes[0, 0].legend(fontsize=6.5, loc='upper right')
    fig.suptitle(f'PCA posteriors: final epoch vs SWA vs output-avg vs IO target '
                 f'({mouse}, noreg)', y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'weight_averaging_examples')


def main(results_root, run, split, out_root, cutoff):
    M, io_peak = collect(results_root, run, split, cutoff)
    if not any(M[(l, a)]['base']['peak'] for l in LOSSES for a, _ in ARCHS):
        raise SystemExit(f'no snapshots found in {run} (need history.state_dicts).')
    fig_quant(M, io_peak, out_root)
    fig_examples(results_root, run, split, cutoff, out_root)
    print(f'late-epoch averaging on {run} (epoch ≥ {cutoff}); IO-target peakiness ≈ {io_peak:.3f}')
    print(f"{'loss':12s}{'arch':9s} | {'peak base/SWA/oavg':>26s} | {'KL base/SWA/oavg':>26s}")
    for loss in LOSSES:
        for arch, alabel in ARCHS:
            pk = [_agg(M[(loss, arch)][v]['peak'])[0] for v in ('base', 'swa', 'oavg')]
            kl = [_agg(M[(loss, arch)][v]['kl'])[0] for v in ('base', 'swa', 'oavg')]
            print(f"{loss:12s}{alabel:9s} | {pk[0]:7.3f}{pk[1]:8.3f}{pk[2]:8.3f}      | "
                  f"{kl[0]:7.3f}{kl[1]:8.3f}{kl[2]:8.3f}")
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='noreg')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    ap.add_argument('--cutoff', type=int, default=100, help='average snapshots with epoch >= cutoff')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root, a.cutoff)
